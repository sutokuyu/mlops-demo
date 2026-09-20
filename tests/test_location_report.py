"""Tests for the location report's delivery layer."""

import io
import json
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.monitoring.location_report as location_report
from src.monitoring.location_report import (
    DEFAULT_TEMPERATURE,
    EVENTS,
    GROUNDING_RULES,
    MAX_REPORT_CHARACTERS,
    NEUTRAL_PERSONA,
    PRESENTATION,
    TASK,
    DeliveryError,
    build_instruction,
    build_summary,
    call_llm,
    day_bounds,
    llm_temperature,
    post_json,
    toilet_events,
)
from src.monitoring.location_store import LocationStore


class FakeResponse:
    def __init__(self, body: bytes = b"") -> None:
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args) -> bool:
        return False

    def read(self) -> bytes:
        return self.body


def capture_request(monkeypatch, response=None, error=None) -> dict:
    """Intercept urlopen and record the Request that was built."""
    captured: dict = {}

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        if error is not None:
            raise error
        return response if response is not None else FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return captured


def test_post_json_sends_a_user_agent(monkeypatch) -> None:
    """Discord's Cloudflare answers urllib's default agent with 403 / error 1010.

    Omitting the header made the daily report fail with a bare
    "HTTP Error 403: Forbidden" even though the webhook was valid.
    """
    captured = capture_request(monkeypatch)
    post_json("https://discord.invalid/webhook", {"content": "hi"})

    request = captured["request"]
    agent = request.get_header("User-agent")
    assert agent, "post_json must send a User-Agent or Cloudflare rejects it"
    assert "python-urllib" not in agent.lower()


def test_post_json_sends_json_and_an_optional_token(monkeypatch) -> None:
    captured = capture_request(monkeypatch)
    post_json("https://example.invalid/hook", {"content": "喂"}, token="secret")

    request = captured["request"]
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("Authorization") == "Bearer secret"
    assert json.loads(request.data.decode("utf-8")) == {"content": "喂"}


def test_post_json_omits_the_authorization_header_without_a_token(monkeypatch) -> None:
    captured = capture_request(monkeypatch)
    post_json("https://example.invalid/hook", {"content": "hi"})
    assert captured["request"].get_header("Authorization") is None


def test_post_json_reports_the_response_body(monkeypatch) -> None:
    """The body is where Discord and LLM gateways explain the real problem."""
    error = urllib.error.HTTPError(
        "https://discord.invalid/webhook",
        403,
        "Forbidden",
        {},
        io.BytesIO(b"error code: 1010"),
    )
    capture_request(monkeypatch, error=error)

    with pytest.raises(DeliveryError) as excinfo:
        post_json("https://discord.invalid/webhook", {"content": "hi"})

    message = str(excinfo.value)
    assert "403" in message
    assert "1010" in message


def test_post_json_truncates_a_huge_error_body(monkeypatch) -> None:
    error = urllib.error.HTTPError(
        "https://example.invalid/hook",
        500,
        "Server Error",
        {},
        io.BytesIO(b"x" * 5000),
    )
    capture_request(monkeypatch, error=error)

    with pytest.raises(DeliveryError) as excinfo:
        post_json("https://example.invalid/hook", {"content": "hi"})

    assert len(str(excinfo.value)) < 400


def test_post_json_returns_nothing_on_success(monkeypatch) -> None:
    capture_request(monkeypatch, response=FakeResponse(b'{"ok":true}'))
    assert post_json("https://example.invalid/hook", {"content": "hi"}) is None


def test_build_instruction_keeps_every_requirement_with_a_custom_persona() -> None:
    instruction = build_instruction("zh", persona="你是大肥鱼，一条爱吐槽的 AI 鲸鱼。")
    assert instruction.startswith("你是大肥鱼")
    assert TASK in instruction
    assert PRESENTATION.format(language="zh") in instruction
    assert GROUNDING_RULES in instruction
    assert str(MAX_REPORT_CHARACTERS) in instruction


def test_location_names_are_asked_for_explicitly() -> None:
    """Folding this into TASK made the model print raw identifiers like on_sofa."""
    instruction = build_instruction("zh")
    assert "translate them into natural zh" in instruction
    assert "on_sofa" in instruction


def test_build_instruction_falls_back_to_a_neutral_persona() -> None:
    """An explicitly empty persona asks for the plain factual voice."""
    for persona in ("", "   "):
        instruction = build_instruction("zh", persona=persona)
        assert instruction.startswith(NEUTRAL_PERSONA)
        assert GROUNDING_RULES in instruction
        assert "大肥鱼" not in instruction


def test_clearing_the_persona_keeps_the_length_cap() -> None:
    """The cap lives in code because Discord truncates without any error.

    If it were part of the persona, trying out the neutral voice would silently
    uncap the message and cut the last sentence off in Discord.
    """
    instruction = build_instruction("zh", persona="", style="", hints="")
    assert str(MAX_REPORT_CHARACTERS) in instruction
    assert TASK in instruction
    assert PRESENTATION.format(language="zh") in instruction
    assert GROUNDING_RULES in instruction


def test_build_instruction_none_means_use_the_configured_block() -> None:
    assert build_instruction("zh", persona=None) == build_instruction("zh")
    assert build_instruction("zh", style=None) == build_instruction("zh")
    assert build_instruction("zh", hints=None) == build_instruction("zh")


def test_style_and_hints_are_optional_blocks() -> None:
    instruction = build_instruction("zh", persona="P", style="S-STYLE", hints="H-HINT")
    assert "S-STYLE" in instruction
    assert "H-HINT" in instruction

    bare = build_instruction("zh", persona="P", style="  ", hints="")
    assert "S-STYLE" not in bare
    assert "\n\n\n" not in bare


def test_grounding_rules_come_last() -> None:
    """Recency: the hard rule sits after every tunable block."""
    instruction = build_instruction("zh", persona="P", style="S", hints="H")
    assert instruction.rstrip().endswith(GROUNDING_RULES)
    assert instruction.index("P") < instruction.index("S") < instruction.index("H")


def test_style_comes_after_the_fixed_requirements() -> None:
    instruction = build_instruction("zh", persona="P", style="S-STYLE", hints="")
    assert instruction.index(TASK) < instruction.index("S-STYLE")
    assert instruction.index(PRESENTATION.format(language="zh")) < instruction.index("S-STYLE")


@pytest.mark.parametrize("field", ["persona", "style", "hints"])
def test_each_configured_block_reaches_the_prompt(field: str) -> None:
    """Whatever locations.yaml says is what the model gets told."""
    configured = (location_report.REPORT_CONFIG["llm"].get(field) or "").strip()
    if not configured:
        pytest.skip(f"locations.yaml does not define report.llm.{field}")
    assert configured in build_instruction("zh")


def test_llm_temperature_comes_from_the_config() -> None:
    """locations.yaml ships 0.7 for the playful persona."""
    assert llm_temperature() == pytest.approx(0.7)


def test_llm_temperature_override_wins() -> None:
    assert llm_temperature(0.0) == pytest.approx(0.0)
    assert llm_temperature(1.4) == pytest.approx(1.4)


def test_llm_temperature_falls_back_when_unset(monkeypatch) -> None:
    monkeypatch.setitem(location_report.REPORT_CONFIG, "llm", {})
    assert llm_temperature() == pytest.approx(DEFAULT_TEMPERATURE)


@pytest.mark.parametrize("bad", [-0.1, 2.5, 99])
def test_llm_temperature_rejects_out_of_range(bad: float) -> None:
    with pytest.raises(RuntimeError, match="between 0.0 and"):
        llm_temperature(bad)


def test_llm_temperature_rejects_a_non_numeric_config(monkeypatch) -> None:
    monkeypatch.setitem(location_report.REPORT_CONFIG, "llm", {"temperature": "very warm"})
    with pytest.raises(RuntimeError, match="must be a number"):
        llm_temperature()


def test_call_llm_sends_the_configured_temperature(monkeypatch) -> None:
    captured = {}

    def fake_post(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return json.dumps({"choices": [{"message": {"content": "喵"}}]}).encode("utf-8")

    monkeypatch.setattr(location_report, "_post", fake_post)
    monkeypatch.setitem(
        location_report.REPORT_CONFIG,
        "llm",
        {**location_report.REPORT_CONFIG["llm"], "api_key": "test-key"},
    )

    assert call_llm({"cats": []}) == "喵"
    assert captured["body"]["temperature"] == pytest.approx(0.7)
    assert captured["body"]["messages"][1]["content"] == '{"cats": []}'


def test_call_llm_honours_a_temperature_override(monkeypatch) -> None:
    captured = {}

    def fake_post(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return json.dumps({"choices": [{"message": {"content": "喵"}}]}).encode("utf-8")

    monkeypatch.setattr(location_report, "_post", fake_post)
    monkeypatch.setitem(
        location_report.REPORT_CONFIG,
        "llm",
        {**location_report.REPORT_CONFIG["llm"], "api_key": "test-key"},
    )

    call_llm({"cats": []}, temperature=1.2)
    assert captured["body"]["temperature"] == pytest.approx(1.2)


def test_compose_returns_the_llm_narrative_in_llm_mode(monkeypatch) -> None:
    monkeypatch.setitem(location_report.REPORT_CONFIG, "mode", "llm")
    monkeypatch.setattr(location_report, "call_llm", lambda summary, temperature=None: "喵喵喵")
    assert location_report.compose({"cats": []}, "plain text") == "喵喵喵"


def test_compose_returns_the_plain_text_in_discord_mode(monkeypatch) -> None:
    monkeypatch.setitem(location_report.REPORT_CONFIG, "mode", "discord")
    monkeypatch.setattr(
        location_report,
        "call_llm",
        lambda summary, temperature=None: pytest.fail("the LLM must not be called"),
    )
    assert location_report.compose({"cats": []}, "plain text") == "plain text"


def test_dry_run_posts_nothing(monkeypatch, capsys) -> None:
    """--dry-run is how the persona gets tuned without spamming Discord."""
    monkeypatch.setitem(location_report.REPORT_CONFIG, "mode", "llm")
    monkeypatch.setattr(location_report, "call_llm", lambda summary, temperature=None: "喵")
    monkeypatch.setattr(
        location_report,
        "send_to_discord",
        lambda content: pytest.fail("a dry run must not post anything"),
    )
    monkeypatch.setattr(
        location_report,
        "post_json",
        lambda *args, **kwargs: pytest.fail("a dry run must not post anything"),
    )

    location_report.main(["--date", "2026-09-19", "--dry-run"])
    output = capsys.readouterr().out
    assert "未发送" in output
    assert output.rstrip().endswith("喵")


def test_build_summary_reads_the_requested_database(tmp_path: Path) -> None:
    """--database lets a report run against an alternative history.

    Without it a test would have to touch the real tracking database.
    """
    day = date(2026, 9, 19)
    start_ts, _ = day_bounds(day, ZoneInfo("Asia/Tokyo"))

    database = tmp_path / "history.db"
    store = LocationStore(database)
    visit_id = store.open_visit(start_ts + 60, "bagel", "sofa", "on_sofa", 0.9)
    store.touch_visit(visit_id, start_ts + 660, 4, 0.91)
    store.close()

    summary = build_summary(day, database)
    assert [entry["cat"] for entry in summary["cats"]] == ["bagel"]
    assert summary["cats"][0]["locations"] == [{"location": "on_sofa", "minutes": 10.0}]

    # An empty database reports no cats rather than silently reading the default.
    empty = tmp_path / "empty.db"
    LocationStore(empty).close()
    assert build_summary(day, empty)["cats"] == []


def test_build_summary_clips_visits_that_span_midnight(tmp_path: Path) -> None:
    day = date(2026, 9, 19)
    start_ts, end_ts = day_bounds(day, ZoneInfo("Asia/Tokyo"))

    database = tmp_path / "history.db"
    store = LocationStore(database)
    # Starts 10 minutes before the day begins and ends 5 minutes into it.
    visit_id = store.open_visit(start_ts - 600, "kurumi", "sofa", "carpet", 0.8)
    store.touch_visit(visit_id, start_ts + 300, 2, 0.8)
    # Ends 10 minutes after the day finishes having started 20 minutes before it.
    late = store.open_visit(end_ts - 1200, "kurumi", "sofa", "on_sofa", 0.8)
    store.touch_visit(late, end_ts + 600, 3, 0.8)
    store.close()

    summary = build_summary(day, database)
    # Locations are ranked by dwell time, and each visit only counts the part
    # of itself that falls inside the day being summarized.
    assert summary["cats"][0]["locations"] == [
        {"location": "on_sofa", "minutes": 20.0},
        {"location": "carpet", "minutes": 5.0},
    ]


# --- toilet verdict ---------------------------------------------------------
#
# The tracker only catches a cat going in and coming out, so a real visit is a
# cluster of sightings with nothing in between. An LLM asked to work this out
# from a text timeline got it right about a third of the time, so the verdict is
# computed here instead and handed over as a fact.

TOILET_ZONES = {"toilet_1"}
BASE_TS = 1_760_000_000.0


def sighting(seconds: float, zone: str = "toilet_1", end: float | None = None) -> dict:
    """A minimal stand-in for a stored visit, in seconds after midnight."""
    start_ts = BASE_TS + seconds
    return {
        "cat": "bagel",
        "camera": "feeder",
        "zone": zone,
        "location": "feeder",
        "start_ts": start_ts,
        "end_ts": start_ts + (20.0 if end is None else end),
    }


def as_visit(entry: dict):
    class Visit:
        pass

    visit = Visit()
    visit.__dict__.update(entry)
    return visit


def test_toilet_events_merges_sightings_of_one_visit() -> None:
    """Going in and coming out is one visit, not two."""
    events = toilet_events([as_visit(sighting(0)), as_visit(sighting(130))], TOILET_ZONES, 300.0, 2)
    assert len(events) == 1
    assert events[0]["kind"] == "toilet_use"
    assert events[0]["zone"] == "toilet_1"
    # Second-level resolution, because a 20 second visit renders as one minute.
    assert events[0]["start"] != events[0]["end"]


def test_toilet_events_ignores_a_single_sighting() -> None:
    """Walking past the toilet on the way to the feeder is not a visit."""
    assert toilet_events([as_visit(sighting(0))], TOILET_ZONES, 300.0, 2) == []


def test_toilet_events_ignores_sightings_split_by_another_location() -> None:
    """Two pass-bys 46 minutes apart are not one long visit."""
    visits = [
        as_visit(sighting(0)),
        as_visit(sighting(300, zone="feeder")),
        as_visit(sighting(2760)),
    ]
    assert toilet_events(visits, TOILET_ZONES, 300.0, 2) == []


def test_toilet_events_separates_visits_further_apart_than_the_window() -> None:
    visits = [as_visit(sighting(0)), as_visit(sighting(130)), as_visit(sighting(5000))]
    visits.append(as_visit(sighting(5100)))
    assert len(toilet_events(visits, TOILET_ZONES, 300.0, 2)) == 2


def test_toilet_events_is_off_without_configured_zones() -> None:
    """An empty ``toilet_zones`` turns the whole feature off."""
    visits = [as_visit(sighting(0)), as_visit(sighting(130))]
    assert toilet_events(visits, set(), 300.0, 2) == []
    assert toilet_events(visits, TOILET_ZONES, 300.0, 1) == []


def test_build_summary_hides_raw_toilet_sightings_and_reports_the_verdict(
    tmp_path: Path,
) -> None:
    """The model must not be able to reach a toilet sighting on its own.

    Leaving them in the timeline made it credit pass-bys no matter how firmly the
    prompt forbade it, so the raw entries are dropped from the data instead.
    """
    day = date(2026, 9, 19)
    start_ts, _ = day_bounds(day, ZoneInfo("Asia/Tokyo"))

    database = tmp_path / "history.db"
    store = LocationStore(database)
    # bagel: two sightings two minutes apart, which is one visit.
    first = store.open_visit(start_ts + 600, "bagel", "feeder", "toilet_1", 0.9)
    store.touch_visit(first, start_ts + 620, 2, 0.9)
    second = store.open_visit(start_ts + 730, "bagel", "feeder", "toilet_1", 0.9)
    store.touch_visit(second, start_ts + 750, 2, 0.9)
    # bagel: an unrelated location it actually dwells in.
    sofa = store.open_visit(start_ts + 3600, "bagel", "sofa", "on_sofa", 0.9)
    store.touch_visit(sofa, start_ts + 7200, 5, 0.9)
    # kurumi: one lone sighting, which is a pass-by.
    lone = store.open_visit(start_ts + 5000, "kurumi", "feeder", "toilet_1", 0.8)
    store.touch_visit(lone, start_ts + 5020, 2, 0.8)
    store.close()

    summary = build_summary(day, database)
    bagel, kurumi = {entry["cat"]: entry for entry in summary["cats"]}.values()

    assert len(bagel["events"]) == 1
    assert bagel["events"][0]["kind"] == "toilet_use"
    assert kurumi["events"] == []

    for cat in (bagel, kurumi):
        assert "toilet_1" not in [entry["zone"] for entry in cat["timeline"]]
        assert "toilet_1" not in [entry["location"] for entry in cat["locations"]]


def test_the_events_block_says_the_toilet_is_missing_from_the_timeline() -> None:
    """The prompt has to agree with the filtered data, or it contradicts itself."""
    assert EVENTS in build_instruction("zh")
    assert "toilet_use" in EVENTS
    assert "timeline" in EVENTS
