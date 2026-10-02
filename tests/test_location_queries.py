"""Tests for the code-computed answers a question about zones gets.

The bug these exist for, measured 2026-10-02: asked "猫有没有进过水池" the bot answered
"数据里没这个记录" while the database held four `sink` stays that day. Two things were
wrong - the owner's word and the stored identifier never met, and the counting was left
to the model - so there are two halves to test here: the vocabulary that ties the words
together, and the tools that count.

Nothing here needs a camera, a network or an LLM: the database is a throwaway file and
the clock is injected.
"""

import json
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.monitoring.discord_bot as discord_bot
import src.monitoring.location_queries as queries
from src.monitoring.location_report import day_bounds
from src.monitoring.location_store import LocationStore
from src.monitoring.zone_vocabulary import (
    DEFAULT_ZONE_ALIASES,
    ambiguous_words,
    known_zones,
    resolve_zones,
    vocabulary_text,
    zone_aliases,
    zone_groups,
)

TZ = ZoneInfo("Asia/Tokyo")
NOW = datetime(2026, 10, 3, 12, 0, tzinfo=TZ)


def day_start(day: date) -> float:
    return day_bounds(day, TZ)[0]


def seed(database: Path, rows: list[tuple[date, int, str, str, str, float]]) -> Path:
    """``(day, minute-of-day, cat, camera, zone, minutes)`` -> a visit row."""
    store = LocationStore(database)
    for day, minute, cat, camera, zone, minutes in rows:
        start_ts = day_start(day) + minute * 60
        visit_id = store.open_visit(start_ts, cat, camera, zone, 0.9)
        store.touch_visit(visit_id, start_ts + minutes * 60, 3, 0.9)
    store.close()
    return database


def seed_observations(
    database: Path,
    rows: list[tuple[int, str, str, float, float]],
    *,
    frame: tuple[int, int] = (1600, 900),
    day: date = date(2026, 10, 3),
) -> Path:
    """``(minute-of-day, cat, camera, x, y)`` -> an observation with a camera anchor."""
    store = LocationStore(database)
    for minute, cat, camera, x, y in rows:
        store.record_observation(
            day_start(day) + minute * 60,
            cat,
            camera,
            None,
            0.9,
            camera_point=(x, y),
            frame_size=frame,
        )
    store.close()
    return database


SINK_ROWS = [
    (date(2026, 10, 2), 2 * 60, "kurumi", "living_room", "sink", 0.7),
    (date(2026, 10, 2), 13 * 60 + 12, "kurumi", "living_room", "sink", 1.8),
    (date(2026, 10, 2), 22 * 60 + 39, "kurumi", "living_room", "sink", 1.3),
    (date(2026, 10, 2), 9 * 60, "bagel", "sofa", "floor", 12.0),
]


# --- the vocabulary --------------------------------------------------------


def test_the_owners_word_resolves_to_the_stored_identifier() -> None:
    """The exact mismatch that produced the false "no such record"."""
    assert resolve_zones("kurumi 有没有进过水池") == ["sink"]
    assert resolve_zones("有没有在水槽待过") == ["sink"]
    assert resolve_zones("进过猫砂盆吗") == ["toilet_1"]


def test_a_zone_can_also_be_named_by_its_identifier() -> None:
    assert resolve_zones("kurumi 有没有在 sink 这个区域待过") == ["sink"]


def test_the_longest_word_wins() -> None:
    """ "餐桌底下" must not be read as `on_dining_table` via the "餐桌" inside it."""
    assert resolve_zones("餐桌底下") == ["under_dining_table"]
    assert resolve_zones("沙发上") == ["on_sofa"]
    assert resolve_zones("沙发底下") == ["under_sofa"]


# --- one word, several zones ----------------------------------------------
#
# The cat has two wet food bowls, so "去过湿粮碗吗" means either and usually both. That
# is not the same thing as two zones accidentally sharing a word, which would be a typo:
# aliases stay one word to one zone (a test enforces it) and this lives in `zone_groups`.


def test_a_word_can_mean_several_zones() -> None:
    assert resolve_zones("kurumi 有没有去过湿粮碗") == ["wet_food_bowl_1", "wet_food_bowl_2"]
    assert resolve_zones("饭碗呢") == ["wet_food_bowl_1", "wet_food_bowl_2"]


def test_a_numbered_word_still_beats_the_group_it_belongs_to() -> None:
    assert resolve_zones("湿粮碗2") == ["wet_food_bowl_2"]
    assert resolve_zones("湿粮碗1") == ["wet_food_bowl_1"]


def test_a_group_and_a_specific_zone_in_one_question_are_both_found() -> None:
    assert resolve_zones("湿粮碗和猫砂盆") == [
        "wet_food_bowl_1",
        "wet_food_bowl_2",
        "toilet_1",
    ]


def test_no_group_word_is_also_an_alias_word() -> None:
    """Otherwise the two mechanisms silently merge into one union."""
    alias_words = {word.lower() for words in zone_aliases().values() for word in words}
    group_words = {word.lower() for word in zone_groups()}
    assert alias_words & group_words == set()


def test_every_group_names_a_zone_that_exists_on_some_camera() -> None:
    existing = set(known_zones())
    missing = {zone for zones in zone_groups().values() for zone in zones if zone not in existing}
    assert missing == set(), f"these groups point at zones no camera has: {sorted(missing)}"


def test_the_prompt_text_offers_a_group_as_alternatives() -> None:
    text = vocabulary_text()
    assert "wet_food_bowl_1 + wet_food_bowl_2 = 湿粮碗 (either one)" in text
    assert "wet_food_bowl_1 = 湿粮碗1" in text


def test_a_question_about_no_place_resolves_to_nothing() -> None:
    assert resolve_zones("今天两只猫都做什么了") == []
    assert resolve_zones("报告一下") == []


def test_no_word_is_claimed_by_two_zones() -> None:
    """A shared word would silently answer about the wrong zone."""
    assert ambiguous_words(zone_aliases()) == {}
    assert ambiguous_words(DEFAULT_ZONE_ALIASES) == {}


def test_every_default_alias_names_a_zone_that_exists_on_some_camera() -> None:
    """A typo here is invisible: the question just resolves to nothing again."""
    existing = set(known_zones())
    missing = {zone for zone in DEFAULT_ZONE_ALIASES if zone not in existing}
    assert missing == set(), f"these aliases point at zones no camera has: {sorted(missing)}"


def test_the_configured_block_wins_per_zone() -> None:
    aliases = zone_aliases({"sink": ["洗手台"]})
    assert aliases["sink"] == ["洗手台"]
    # Other zones keep their defaults rather than being wiped.
    assert aliases["toilet_1"] == DEFAULT_ZONE_ALIASES["toilet_1"]


def test_the_prompt_text_lists_identifiers_with_their_words() -> None:
    text = vocabulary_text()
    assert "sink = 水池/水槽" in text
    assert "toilet_1 = 猫砂盆" in text


# --- ranges ----------------------------------------------------------------


def test_the_default_range_is_today() -> None:
    since, until, label = queries.parse_range("kurumi 在水池吗", now=NOW)
    assert label == "2026-10-03"
    assert since == day_start(date(2026, 10, 3))
    assert until == day_start(date(2026, 10, 4))


def test_a_past_tense_question_without_a_time_word_covers_the_history() -> None:
    """The default is today, but "待过吗" is not asking about today."""
    since, until, _ = queries.parse_range("kurumi 在水池待过吗", now=NOW)
    assert since < day_start(date(2020, 1, 1)) < until


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("今天进过水池吗", "2026-10-03"),
        ("昨天进过水池吗", "2026-10-02"),
        ("前天进过水池吗", "2026-10-01"),
    ],
)
def test_day_words_pick_the_day(text: str, expected: str) -> None:
    assert queries.parse_range(text, now=NOW)[2] == expected


def test_the_day_words_agree_with_the_report_path() -> None:
    """Two implementations of "昨天" is one too many, so they are pinned together."""
    for text in ("今天的报告", "昨天做了什么", "前天做什么了"):
        offset = discord_bot.day_offset(text)
        expected = discord_bot.target_day(offset, now=NOW).isoformat()
        assert queries.parse_range(text, now=NOW)[2] == expected


def test_a_recent_days_phrase_counts_the_last_n_days_including_today() -> None:
    since, until, label = queries.parse_range("最近3天去过哪", now=NOW)
    assert label == "2026-10-01..2026-10-03"
    assert until == day_start(date(2026, 10, 4))


def test_this_week_starts_on_monday() -> None:
    assert queries.parse_range("这周进过水池吗", now=NOW)[2] == "2026-09-28..2026-10-03"


def test_an_explicit_date_is_a_range() -> None:
    assert queries.parse_range("10月1日进过水池吗", now=NOW)[2] == "2026-10-01"
    assert (
        queries.parse_range("2026-09-30 到 2026-10-02 呢", now=NOW)[2] == "2026-09-30..2026-10-02"
    )


def test_an_ever_question_covers_the_recorded_history_not_just_today() -> None:
    """ "有没有进过" asks about all of it - the failure was hidden behind today's range."""
    since, until, label = queries.parse_range("kurumi 有没有进过水池", now=NOW)
    assert label == "all recorded history (up to 2026-10-03)"
    assert since < day_start(date(2020, 1, 1)) < until


def test_a_time_word_beats_an_ever_marker() -> None:
    assert queries.parse_range("kurumi 今天有没有进过水池", now=NOW)[2] == "2026-10-03"


def test_an_absurd_lookback_is_capped() -> None:
    """Otherwise "最近300天" would pull the whole database into one prompt."""
    since, until, label = queries.parse_range("最近300天进过水池吗", now=NOW)
    assert label == "2026-09-03..2026-10-03"
    assert until - since == pytest.approx(31 * 86400)


def test_date_span_reads_tool_arguments_and_includes_both_ends() -> None:
    since, until, label = queries.date_span("2026-10-01", "2026-10-02", now=NOW)
    assert label == "2026-10-01..2026-10-02"
    assert until - since == pytest.approx(2 * 86400)


def test_date_span_rejects_an_unreadable_date() -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        queries.date_span("last tuesday", None, now=NOW)


# --- the tools -------------------------------------------------------------


def test_zone_stay_counts_the_whole_range_not_the_returned_sample(tmp_path: Path) -> None:
    database = seed(tmp_path / "history.db", SINK_ROWS)
    result = queries.zone_stay(
        zones=["sink"],
        cat="kurumi",
        since="2026-10-02",
        until="2026-10-02",
        database=database,
        max_stays=2,
        now=NOW,
    )
    assert result["count"] == 3
    assert result["found"] is True
    assert result["total_minutes"] == pytest.approx(3.8)
    assert len(result["stays"]) == 2 and result["truncated"] is True
    assert result["counts_by_cat"] == {"kurumi": 3}
    assert result["cameras"] == ["living_room"]
    assert result["stays"][0]["start"] == "2026-10-02 02:00:00"


def test_zone_stay_says_which_camera_because_zone_names_repeat(tmp_path: Path) -> None:
    """`sink` exists on living_room and on sofa, so a bare zone name is ambiguous."""
    rows = [
        (date(2026, 10, 2), 60, "kurumi", "living_room", "sink", 1.0),
        (date(2026, 10, 2), 120, "kurumi", "sofa", "sink", 2.0),
    ]
    database = seed(tmp_path / "history.db", rows)
    result = queries.zone_stay(zones=["sink"], since="2026-10-02", database=database, now=NOW)
    assert result["cameras"] == ["living_room", "sofa"]
    assert {stay["camera"] for stay in result["stays"]} == {"living_room", "sofa"}


def test_zone_stay_reports_a_genuine_absence_as_zero(tmp_path: Path) -> None:
    """The fix must not turn "no" into "yes" - it must make both of them true."""
    database = seed(tmp_path / "history.db", SINK_ROWS)
    result = queries.zone_stay(
        zones=["sink"], cat="bagel", since="2026-10-02", database=database, now=NOW
    )
    assert result["count"] == 0
    assert result["found"] is False
    assert result["total_minutes"] == 0.0


def test_zone_stay_needs_at_least_one_zone() -> None:
    with pytest.raises(ValueError, match="at least one zone"):
        queries.stay_rows(0, 1, [], None)


def test_zone_totals_ranks_the_places_by_time(tmp_path: Path) -> None:
    database = seed(tmp_path / "history.db", SINK_ROWS)
    result = queries.zone_totals(since="2026-10-02", database=database, now=NOW)
    assert [row["location"] for row in result["locations"]] == ["floor", "sink"]
    assert result["locations"][1]["visits"] == 3
    assert result["total_minutes"] == pytest.approx(15.8)


def test_daily_summary_is_the_report_summary(tmp_path: Path) -> None:
    database = seed(tmp_path / "history.db", SINK_ROWS)
    summary = queries.daily_summary(days_ago=1, database=database, now=NOW)
    assert summary["date"] == "2026-10-02"
    kurumi = next(entry for entry in summary["cats"] if entry["cat"] == "kurumi")
    assert {"location": "sink", "minutes": 3.8} in kurumi["locations"]


# --- the registry a harness calls -----------------------------------------


def test_every_tool_has_an_openai_compatible_schema() -> None:
    for schema in queries.tools_schema():
        assert schema["type"] == "function"
        function = schema["function"]
        assert set(function) == {"name", "description", "parameters"}
        assert function["name"] in queries.TOOLS
        assert function["parameters"]["type"] == "object"
        assert function["description"].strip()
    # A harness gets this as JSON, so it has to serialize.
    json.dumps(queries.tools_schema())


def test_a_tool_is_called_by_name_with_json_arguments(tmp_path: Path) -> None:
    database = seed(tmp_path / "history.db", SINK_ROWS)
    reply = queries.call_tool(
        "zone_stay",
        {"zones": ["sink"], "cat": "kurumi", "since": "2026-10-02", "until": "2026-10-02"},
        database=database,
        now=NOW,
    )
    assert reply["ok"] is True
    assert reply["result"]["count"] == 3


def test_an_unknown_tool_is_answered_not_raised() -> None:
    reply = queries.call_tool("does_not_exist", {})
    assert reply["ok"] is False
    assert "does_not_exist" in reply["error"]
    assert "zone_stay" in reply["tools"]


def test_bad_arguments_come_back_as_an_error() -> None:
    """A harness feeds this back to the model; an exception would just crash the call."""
    missing = queries.call_tool("zone_stay", {})
    assert missing["ok"] is False
    assert "zones" in missing["error"]
    wrong_type = queries.call_tool("zone_stay", {"zones": ["sink"], "nonsense": 1})
    assert wrong_type["ok"] is False
    assert "bad arguments" in wrong_type["error"]


# --- the question router ---------------------------------------------------


def test_a_question_naming_no_place_gets_no_computed_answer(tmp_path: Path) -> None:
    """The signal that the ordinary report path should be left alone."""
    database = seed(tmp_path / "history.db", SINK_ROWS)
    assert queries.answer_question("今天两只猫都做什么了", database=database, now=NOW) is None


def test_the_sink_question_is_answered_from_the_data(tmp_path: Path) -> None:
    """The regression, pinned: this used to be answered "no such record"."""
    database = seed(tmp_path / "history.db", SINK_ROWS)
    verdict = queries.answer_question("kurumi 有没有进过水池", database=database, now=NOW)
    assert verdict is not None
    assert verdict["tool"] == "zone_stay"
    assert verdict["zones"] == ["sink"]
    assert verdict["count"] == 3
    assert verdict["total_minutes"] == pytest.approx(3.8)
    assert verdict["range"]["label"] == "all recorded history (up to 2026-10-03)"


def test_the_question_narrows_to_the_cat_it_names(tmp_path: Path) -> None:
    database = seed(tmp_path / "history.db", SINK_ROWS)
    both = queries.answer_question("两只猫有没有在水池待过", database=database, now=NOW)
    one = queries.answer_question("kurumi 有没有在水池待过", database=database, now=NOW)
    assert both["count"] == 3 == one["count"]

    only_bagel = queries.answer_question("bagel 有没有在水池待过", database=database, now=NOW)
    assert only_bagel["count"] == 0


def test_a_group_question_counts_both_zones_and_still_says_which(tmp_path: Path) -> None:
    """ "去过湿粮碗吗" has to reach both bowls without losing which one it was."""
    rows = [
        (date(2026, 10, 2), 60, "kurumi", "feeder", "wet_food_bowl_1", 5.0),
        (date(2026, 10, 2), 300, "kurumi", "feeder", "wet_food_bowl_2", 3.0),
        (date(2026, 10, 2), 600, "kurumi", "feeder", "feeder", 1.0),
    ]
    database = seed(tmp_path / "history.db", rows)

    verdict = queries.answer_question("kurumi 有没有去过湿粮碗", database=database, now=NOW)
    assert verdict["count"] == 2
    assert verdict["total_minutes"] == pytest.approx(8.0)
    assert verdict["counts_by_zone"] == {"wet_food_bowl_1": 1, "wet_food_bowl_2": 1}

    only_two = queries.answer_question("kurumi 去过湿粮碗2吗", database=database, now=NOW)
    assert only_two["count"] == 1
    assert only_two["counts_by_zone"] == {"wet_food_bowl_2": 1}


# --- points: "who was near this spot" --------------------------------------
#
# The floor zone is far too coarse to point at a spot on it. These cover the answer to
# "哪只猫在地板 (x, y) 处待得比较久" - the coordinate the owner reads off the live
# picture, not a zone name.


def test_an_observation_keeps_the_point_in_the_camera_frame(tmp_path: Path) -> None:
    database = seed_observations(tmp_path / "history.db", [(60, "kurumi", "living_room", 0.4, 0.7)])
    store = LocationStore(database)
    try:
        row = store.observations_between(0, day_start(date(2026, 10, 4)))[0]
    finally:
        store.close()
    assert row.cam_x == pytest.approx(0.4)
    assert row.cam_y == pytest.approx(0.7)
    assert row.camera_anchor == pytest.approx((0.4, 0.7))
    assert (row.frame_width, row.frame_height) == (1600, 900)


def test_a_row_with_only_a_box_still_yields_a_camera_anchor(tmp_path: Path) -> None:
    """Old rows predate cam_x/cam_y, but the box is stored untransformed, so it recovers."""
    database = tmp_path / "history.db"
    store = LocationStore(database)
    store.record_observation(
        day_start(date(2026, 10, 3)) + 60,
        "bagel",
        "sofa",
        "floor",
        0.9,
        box=(0.1, 0.2, 0.5, 0.8),
    )
    store.close()

    store = LocationStore(database)
    try:
        row = store.observations_between(0, day_start(date(2026, 10, 4)))[0]
    finally:
        store.close()
    assert row.cam_x is None
    assert row.camera_anchor == pytest.approx((0.3, 0.8))


def test_the_store_migrates_a_database_that_lacks_the_camera_columns(tmp_path: Path) -> None:
    database = tmp_path / "old.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE observations (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,"
        " cat TEXT NOT NULL, camera TEXT NOT NULL, zone TEXT, confidence REAL NOT NULL)"
    )
    connection.commit()
    connection.close()

    store = LocationStore(database)
    try:
        columns = {row[1] for row in store._connection.execute("PRAGMA table_info(observations)")}
    finally:
        store.close()
    assert {"cam_x", "cam_y", "frame_width", "frame_height"} <= columns


def test_point_stay_ranks_the_cat_that_stayed_longest(tmp_path: Path) -> None:
    rows = [
        (0, "kurumi", "living_room", 0.50, 0.50),
        (1, "kurumi", "living_room", 0.51, 0.50),
        (2, "kurumi", "living_room", 0.50, 0.49),
        (3, "kurumi", "living_room", 0.52, 0.51),
        (4, "kurumi", "living_room", 0.50, 0.50),
        (10, "bagel", "living_room", 0.50, 0.50),
        (11, "bagel", "living_room", 0.49, 0.50),
    ]
    database = seed_observations(tmp_path / "history.db", rows)
    result = queries.point_stay(
        camera="living_room", x=0.5, y=0.5, since="2026-10-03", database=database, now=NOW
    )
    assert result["found"] is True
    assert result["minutes_by_cat"] == {"kurumi": 4.0, "bagel": 1.0}
    assert result["total_minutes"] == pytest.approx(5.0)
    assert result["stays"][0]["cat"] == "kurumi"
    assert result["stays"][0]["samples"] == 5


def test_point_stay_excludes_a_cat_elsewhere_on_the_same_camera(tmp_path: Path) -> None:
    rows = [
        (0, "kurumi", "living_room", 0.50, 0.50),
        (1, "kurumi", "living_room", 0.50, 0.50),
        (2, "bagel", "living_room", 0.10, 0.10),
    ]
    database = seed_observations(tmp_path / "history.db", rows)
    result = queries.point_stay(
        camera="living_room", x=0.5, y=0.5, since="2026-10-03", database=database, now=NOW
    )
    assert result["minutes_by_cat"] == {"kurumi": 1.0}
    assert "bagel" not in result["samples_by_cat"]


def test_a_long_gap_splits_one_cat_into_two_stays(tmp_path: Path) -> None:
    rows = [
        (0, "kurumi", "living_room", 0.50, 0.50),
        (1, "kurumi", "living_room", 0.50, 0.50),
        (30, "kurumi", "living_room", 0.50, 0.50),
        (31, "kurumi", "living_room", 0.50, 0.50),
    ]
    database = seed_observations(tmp_path / "history.db", rows)
    result = queries.point_stay(
        camera="living_room", x=0.5, y=0.5, since="2026-10-03", database=database, now=NOW
    )
    assert [stay["minutes"] for stay in result["stays"]] == [1.0, 1.0]
    assert result["samples_by_cat"] == {"kurumi": 4}


def test_point_stay_reads_pixel_coordinates_against_the_frame_size(tmp_path: Path) -> None:
    database = seed_observations(tmp_path / "history.db", [(3, "kurumi", "living_room", 0.5, 0.5)])
    result = queries.point_stay(
        camera="living_room",
        x=800,
        y=450,
        radius=160,
        unit="pixel",
        since="2026-10-03",
        database=database,
        now=NOW,
    )
    assert result["found"] is True
    assert result["point"]["x"] == pytest.approx(0.5)
    assert result["point"]["y"] == pytest.approx(0.5)
    assert result["point"]["pixels"] == [800, 450]


def test_point_stay_refuses_pixels_without_a_frame_size(tmp_path: Path) -> None:
    database = tmp_path / "history.db"
    store = LocationStore(database)
    store.record_observation(day_start(date(2026, 10, 3)) + 180, "kurumi", "living_room", None, 0.9)
    store.close()
    with pytest.raises(ValueError, match="frame size"):
        queries.point_stay(
            camera="living_room",
            x=800,
            y=450,
            unit="pixel",
            since="2026-10-03",
            database=database,
            now=NOW,
        )


def test_point_stay_needs_a_camera() -> None:
    with pytest.raises(ValueError, match="camera is required"):
        queries.point_stay_between(0, 1, camera="", x=0.5, y=0.5)


# --- reading a coordinate out of a question --------------------------------


@pytest.mark.parametrize(
    ("text", "x", "y", "unit"),
    [
        ("地板上 x=1200 y=800 谁待得久", 1200.0, 800.0, "pixel"),
        ("客厅坐标 0.6 0.5 那里", 0.6, 0.5, "normalized"),
        ("客厅地板上 (0.6, 0.5)", 0.6, 0.5, "normalized"),
        ("sofa (1200,800)", 1200.0, 800.0, "pixel"),
    ],
)
def test_a_point_can_be_read_from_a_question(text: str, x: float, y: float, unit: str) -> None:
    point = queries.parse_point(text)
    assert point is not None
    assert (point["x"], point["y"], point["unit"]) == (x, y, unit)


def test_a_question_without_a_coordinate_has_no_point() -> None:
    assert queries.parse_point("今天两只猫都做什么了") is None
    assert queries.parse_point("2026-10-03 的报告") is None


def test_a_coordinate_question_is_answered_from_the_data(tmp_path: Path) -> None:
    rows = [
        (0, "kurumi", "living_room", 0.50, 0.50),
        (1, "kurumi", "living_room", 0.50, 0.50),
        (2, "bagel", "living_room", 0.50, 0.50),
    ]
    database = seed_observations(tmp_path / "history.db", rows)
    verdict = queries.answer_question(
        "客厅地板上 (0.50, 0.50) 哪只猫待得久", database=database, now=NOW
    )
    assert verdict is not None
    assert verdict["tool"] == "point_stay"
    assert verdict["camera"] == "living_room"
    assert verdict["stays"][0]["cat"] == "kurumi"
    assert verdict["range"]["label"] == "2026-10-03"


def test_a_coordinate_question_without_a_camera_asks_which_one(tmp_path: Path) -> None:
    """The same numbers are a different place on another camera, so do not guess."""
    database = seed_observations(tmp_path / "history.db", [(0, "kurumi", "living_room", 0.5, 0.5)])
    verdict = queries.answer_question(
        "地板上 (0.50, 0.50) 哪只猫待得久", database=database, now=NOW
    )
    assert verdict["tool"] == "point_stay"
    assert verdict["needs_camera"] is True
    assert verdict["found"] is False
    assert "living_room" in verdict["camera_candidates"]


def test_the_point_tool_is_registered_and_dispatchable(tmp_path: Path) -> None:
    database = seed_observations(tmp_path / "history.db", [(0, "kurumi", "living_room", 0.5, 0.5)])
    names = {tool["function"]["name"] for tool in queries.tools_schema()}
    assert "point_stay" in names
    reply = queries.call_tool(
        "point_stay",
        {"camera": "living_room", "x": 0.5, "y": 0.5, "since": "2026-10-03"},
        database=database,
        now=NOW,
    )
    assert reply["ok"] is True
    assert reply["result"]["camera"] == "living_room"
