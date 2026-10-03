"""Answer history questions in code, as tools a harness can call.

The bot's questions used to be answered by handing the model one day of JSON (measured:
50,131 characters, 189 timeline rows for one cat) and letting it count. That produced
confident false negatives - asked "猫有没有进过水池" it answered "数据里没这个记录" while
the database held four ``sink`` stays that day, and when given the identifier instead it
answered correctly from the same data. Deciding a fact from a text timeline is the same
mistake as the toilet rule (right about a third of the time) and the meal rule (the model
invented "did it really eat?" and applied it unevenly). So the counting happens here.

**Shape of everything in this module**, because a DeepSeek harness (or any function-calling
loop) is meant to call it directly:

* one question -> one dictionary in, one dictionary out, JSON-serializable both ways;
* pure and deterministic - no LLM, no network, no clock except an injectable ``now``;
* no side effects: the database is opened read-only and closed again;
* every tool is described by a JSON schema (:func:`tools_schema`) that OpenAI-compatible
  endpoints accept as-is, and dispatched through :func:`call_tool`, which never raises.

Adding a tool means writing one function with keyword arguments and adding one
:class:`Tool` entry; nothing else needs to change.
"""

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from math import hypot
from pathlib import Path
from zoneinfo import ZoneInfo

from src.monitoring.location_config import (
    IDENTITY_CLASSES,
    REPORT_CONFIG,
    configured_cameras,
    location_database,
)
from src.monitoring.location_report import build_summary, day_bounds
from src.monitoring.location_store import LocationStore
from src.monitoring.snapshot import camera_labels, snapshot_settings
from src.monitoring.zone_vocabulary import known_zones, resolve_zones

DEFAULT_MAX_DAYS = 31
DEFAULT_MAX_STAYS = 20
DEFAULT_TOP_ZONES = 10
# A "near this spot" question is asked about a spot the owner is looking at, so the
# default radius is a fraction of the frame wide enough to cover a cat's body, and a
# gap of a minute splits one visit from the next.
DEFAULT_POINT_RADIUS = 0.05
DEFAULT_POINT_GAP_SECONDS = 60.0

# See discord_bot.DAY_BEFORE_YESTERDAY_WORDS / YESTERDAY_WORDS: the same phrases, kept
# here so a question's range and its report day cannot drift apart (a test compares them).
DAY_BEFORE_YESTERDAY_WORDS = ("前天", "day before yesterday")
YESTERDAY_WORDS = ("昨天", "昨日", "yesterday")

_RECENT_DAYS = re.compile(r"(?:最近|近|过去|前)\s*(\d+)\s*(?:天|日)")
_ISO_DATE = re.compile(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?")
_MONTH_DAY = re.compile(r"(?<!\d)(\d{1,2})\s*[月/-]\s*(\d{1,2})\s*日?")
_THIS_WEEK_WORDS = ("这周", "本周", "这个星期", "这星期", "this week")
_THIS_MONTH_WORDS = ("这个月", "本月", "this month")

# "有没有进过水池" asks about all of recorded history, not about today. Without this,
# the default range (today) answers a question the owner did not ask - and on
# 2026-10-03 that would have hidden the 02:00 sink stay from 10-02 entirely. A time word
# always wins over these markers, so "今天进过水池吗" still means today.
EVER_MARKERS = ("曾经", "有没有", "进过", "待过", "去过", "过吗", "过没", "ever")
# The lookback used for an "ever" question. Nothing is capped here on purpose: the
# max_days limit exists to bound the prompt, and a count over the whole history is a
# handful of numbers however long the history is.
ALL_HISTORY_DAYS = 3650


def report_timezone() -> ZoneInfo:
    return ZoneInfo(REPORT_CONFIG.get("timezone", "Asia/Tokyo"))


def query_limits() -> dict:
    """The ``report.query`` limits, with the code defaults behind them."""
    raw = REPORT_CONFIG.get("query") or {}
    return {
        "max_days": int(raw.get("max_days", DEFAULT_MAX_DAYS)),
        "max_stays": int(raw.get("max_stays", DEFAULT_MAX_STAYS)),
        "top_zones": int(raw.get("top_zones", DEFAULT_TOP_ZONES)),
        "point_radius": float(raw.get("point_radius", DEFAULT_POINT_RADIUS)),
        "point_gap_seconds": float(raw.get("point_gap_seconds", DEFAULT_POINT_GAP_SECONDS)),
    }


def _now_local(now: datetime | None = None) -> datetime:
    tz = report_timezone()
    return now.astimezone(tz) if now is not None else datetime.now(tz)


def known_cats() -> list[str]:
    """The cat names the model can emit, i.e. the identity classes."""
    return sorted({str(name) for name in IDENTITY_CLASSES.values()})


def resolve_cats(text: str, *, cats: Sequence[str] | None = None) -> list[str]:
    """Every cat named in the text. No name at all means "not narrowed to one cat"."""
    lowered = text.lower()
    return [cat for cat in (cats or known_cats()) if cat.lower() in lowered]


# --- cameras ---------------------------------------------------------------
#
# A coordinate exists only on one camera: (0.6, 0.5) is the sink on living_room and a
# cushion on sofa. So a point question must say which camera. The words for a camera are
# deliberately NOT a second vocabulary - they are the same ``discord_bot.snapshot.aliases``
# the picture feature already uses (客厅/沙发/喂食器...), so the owner keeps one list and
# both features understand the same names.


def resolve_cameras(text: str, *, cameras: Sequence[str] | None = None) -> list[str]:
    """Every camera the text names, by identifier or by one of its words.

    Deliberately does not guess: no camera named returns an empty list, and the caller
    asks which one rather than combining coordinates from two rooms. Longest label first,
    so a longer name and a prefix of it cannot disagree about what was said.
    """
    available = list(cameras if cameras is not None else configured_cameras())
    lowered = text.lower()
    labels = camera_labels(available, snapshot_settings().aliases)
    found: list[str] = []
    for label, camera in sorted(labels.items(), key=lambda item: (-len(item[0]), item[0])):
        if label and label in lowered and camera not in found:
            found.append(camera)
    return found


# --- ranges ----------------------------------------------------------------


def date_span(
    since: str | None = None,
    until: str | None = None,
    *,
    now: datetime | None = None,
    max_days: int | None = None,
) -> tuple[float, float, str]:
    """Two inclusive ISO dates -> a half-open epoch range, capped how far back it may go.

    Dates rather than timestamps because that is what a model can produce reliably, and
    both ends are inclusive because "from the 1st to the 3rd" including the 3rd is what
    anyone means. The cap exists so a question like "最近300天" cannot pull the whole
    database into one prompt.
    """
    tz = report_timezone()
    today = _now_local(now).date()
    limit = query_limits()["max_days"] if max_days is None else max_days

    start_day = _parse_date(since, tz) or today
    end_day = _parse_date(until, tz) or start_day
    if end_day < start_day:
        start_day, end_day = end_day, start_day
    if (today - start_day).days > limit:
        start_day = today - timedelta(days=limit)
    if end_day > today:
        end_day = today

    start_ts = datetime.combine(start_day, time.min, tzinfo=tz).timestamp()
    end_ts = datetime.combine(end_day + timedelta(days=1), time.min, tzinfo=tz).timestamp()
    label = start_day.isoformat() if start_day == end_day else f"{start_day}..{end_day}"
    return start_ts, end_ts, label


def _parse_date(value: str | date | None, tz: ZoneInfo) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.astimezone(tz).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if text.isdigit() and len(text) == 8:  # YYYYMMDD
        return date(int(text[:4]), int(text[4:6]), int(text[6:]))
    match = _ISO_DATE.search(text)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = _MONTH_DAY.search(text)
    if match:
        return date(_now_local().year, int(match.group(1)), int(match.group(2)))
    raise ValueError(f"could not read a date from {value!r}; use YYYY-MM-DD")


def parse_range(
    text: str,
    *,
    now: datetime | None = None,
    max_days: int | None = None,
) -> tuple[float, float, str]:
    """The range a question means: 今天 / 昨天 / 前天 / 最近N天 / 这周 / 本月 / a date.

    Deliberately literal, and it defaults to today - the same default the report uses -
    so a question with no time word behaves exactly as it did before.
    """
    tz = report_timezone()
    today = _now_local(now).date()
    limit = query_limits()["max_days"] if max_days is None else max_days
    lowered = text.lower()

    recent = _RECENT_DAYS.search(text)
    if recent:
        days = max(1, min(int(recent.group(1)), limit))
        since = today - timedelta(days=days - 1)
        return _span(since, today, tz)

    if any(word in lowered for word in DAY_BEFORE_YESTERDAY_WORDS):
        return _span(today - timedelta(days=2), today - timedelta(days=2), tz)
    if any(word in lowered for word in YESTERDAY_WORDS):
        return _span(today - timedelta(days=1), today - timedelta(days=1), tz)
    if "今天" in text or "今日" in text:
        return _span(today, today, tz)

    if any(word in lowered for word in _THIS_WEEK_WORDS):
        monday = today - timedelta(days=today.weekday())
        return _span(monday, today, tz)
    if any(word in lowered for word in _THIS_MONTH_WORDS):
        return _span(today.replace(day=1), today, tz)

    iso_dates = _ISO_DATE.findall(text)
    if iso_dates:
        days = [date(int(y), int(m), int(d)) for y, m, d in iso_dates]
        return _span(min(days), max(days), tz)

    month_days = _MONTH_DAY.findall(text)
    if month_days:
        days = [date(today.year, int(m), int(d)) for m, d in month_days]
        return _span(min(days), max(days), tz)

    if any(marker in lowered for marker in EVER_MARKERS):
        start_ts, _ = day_bounds(today - timedelta(days=ALL_HISTORY_DAYS), tz)
        _, end_ts = day_bounds(today, tz)
        return start_ts, end_ts, f"all recorded history (up to {today.isoformat()})"

    return _span(today, today, tz)


def _span(since: date, until: date, tz: ZoneInfo) -> tuple[float, float, str]:
    start_ts, _ = day_bounds(since, tz)
    _, end_ts = day_bounds(until, tz)
    label = since.isoformat() if since == until else f"{since}..{until}"
    return start_ts, end_ts, label


# --- the tools -------------------------------------------------------------


def _as_list(value) -> list[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.replace("，", ",").split(",") if item.strip()]
    return [str(item) for item in value]


def _format_time(ts: float, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(ts, tz).strftime("%Y-%m-%d %H:%M:%S")


def _format_day(ts: float, tz: ZoneInfo) -> str:
    """The calendar day a range boundary falls in, for a payload a model reads."""
    return datetime.fromtimestamp(ts, tz).strftime("%Y-%m-%d")


def _range_payload(start_ts: float, end_ts: float, label: str) -> dict:
    tz = report_timezone()
    return {
        "since": _format_day(start_ts, tz),
        # end_ts is exclusive, so the last included day is one second before it.
        "until": _format_day(end_ts - 1, tz),
        "label": label,
        "inclusive": True,
    }


def stay_rows(
    start_ts: float,
    end_ts: float,
    zones: Iterable[str],
    cat: str | None,
    *,
    database: Path | None = None,
    max_stays: int | None = None,
) -> dict:
    """The core count, over every visit in the range - not over the returned sample.

    ``count`` and ``total_minutes`` always describe the whole range; ``stays`` may be
    truncated to ``max_stays`` and says so, because a model that is handed 20 rows and
    told "the count is 137" must not read 20 as the answer.
    """
    wanted = {str(zone) for zone in zones}
    if not wanted:
        raise ValueError("at least one zone is required")

    limit = query_limits()["max_stays"] if max_stays is None else int(max_stays)
    tz = report_timezone()
    store = LocationStore(database or location_database())
    try:
        visits = store.visits_between(start_ts, end_ts)
    finally:
        store.close()

    matched = [
        visit
        for visit in sorted(visits, key=lambda item: item.start_ts)
        if visit.zone in wanted and (cat is None or visit.cat == cat)
    ]
    stays = [
        {
            "cat": visit.cat,
            "camera": visit.camera,
            "zone": visit.zone,
            "start": _format_time(visit.start_ts, tz),
            "end": _format_time(visit.end_ts, tz),
            "minutes": round(max(0.0, visit.end_ts - visit.start_ts) / 60, 1),
            "samples": visit.samples,
        }
        for visit in matched[: max(0, limit)]
    ]
    counts_by_cat: dict[str, int] = {}
    counts_by_zone: dict[str, int] = {}
    for visit in matched:
        counts_by_cat[visit.cat] = counts_by_cat.get(visit.cat, 0) + 1
        counts_by_zone[visit.zone] = counts_by_zone.get(visit.zone, 0) + 1
    return {
        "zones": sorted(wanted),
        "cat": cat or "all",
        "count": len(matched),
        "found": bool(matched),
        "total_minutes": round(
            sum(max(0.0, visit.end_ts - visit.start_ts) for visit in matched) / 60, 1
        ),
        "counts_by_cat": counts_by_cat,
        # A word like 湿粮碗 covers several zones; this is how the answer can still say
        # which bowl the cat used.
        "counts_by_zone": counts_by_zone,
        "cameras": sorted({visit.camera for visit in matched}),
        "stays": stays,
        "truncated": len(matched) > len(stays),
        "max_stays": limit,
    }


def zone_stay(
    *,
    zones,
    cat: str | None = None,
    since: str | None = None,
    until: str | None = None,
    database: Path | None = None,
    max_stays: int | None = None,
    now: datetime | None = None,
) -> dict:
    """How often and how long a cat stayed in the given zones, with exact times."""
    start_ts, end_ts, label = date_span(since, until, now=now)
    result = stay_rows(
        start_ts, end_ts, _as_list(zones), cat or None, database=database, max_stays=max_stays
    )
    result["range"] = _range_payload(start_ts, end_ts, label)
    return result


def zone_totals(
    *,
    since: str | None = None,
    until: str | None = None,
    cat: str | None = None,
    top: int | None = None,
    database: Path | None = None,
    now: datetime | None = None,
) -> dict:
    """Total time per zone and camera over a range, so cross-day questions can be answered."""
    start_ts, end_ts, label = date_span(since, until, now=now)
    limit = query_limits()["top_zones"] if top is None else max(1, int(top))
    store = LocationStore(database or location_database())
    try:
        visits = store.visits_between(start_ts, end_ts)
    finally:
        store.close()

    totals: dict[tuple[str, str, str | None], dict] = {}
    for visit in visits:
        if cat is not None and visit.cat != cat:
            continue
        duration = max(0.0, min(visit.end_ts, end_ts) - max(visit.start_ts, start_ts))
        key = (visit.cat, visit.camera, visit.zone)
        entry = totals.setdefault(
            key,
            {
                "cat": visit.cat,
                "camera": visit.camera,
                "zone": visit.zone,
                "seconds": 0.0,
                "visits": 0,
            },
        )
        entry["seconds"] += duration
        entry["visits"] += 1

    ranked = sorted(totals.values(), key=lambda item: item["seconds"], reverse=True)
    total_seconds = sum(entry["seconds"] for entry in ranked)
    for entry in ranked:
        entry["location"] = entry["zone"] or entry["camera"]
        entry["minutes"] = round(entry.pop("seconds") / 60, 1)
    return {
        "cat": cat or "all",
        "range": _range_payload(start_ts, end_ts, label),
        "locations": ranked[:limit],
        "truncated": len(ranked) > limit,
        "total_minutes": round(total_seconds / 60, 1),
    }


def _frame_size(rows: Iterable) -> tuple[int, int] | None:
    """The most common (width, height) among rows that recorded one.

    A camera's frame size is a property of the camera, so the mode is a safe answer even
    if a resolution was changed mid-history.
    """
    counts: dict[tuple[int, int], int] = {}
    for row in rows:
        if row.frame_width and row.frame_height:
            key = (int(row.frame_width), int(row.frame_height))
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def _finish_point_stay(camera: str, cat: str, first: float, last: float, samples: int) -> dict:
    tz = report_timezone()
    return {
        "cat": cat,
        "camera": camera,
        "start": _format_time(first, tz),
        "end": _format_time(last, tz),
        "minutes": round(max(0.0, last - first) / 60, 1),
        "samples": samples,
    }


def point_stay_between(
    start_ts: float,
    end_ts: float,
    *,
    camera: str,
    x: float,
    y: float,
    radius: float | None = None,
    unit: str = "normalized",
    cat: str | None = None,
    database: Path | None = None,
    max_stays: int | None = None,
) -> dict:
    """Who stayed near a point on one camera, over an explicit range.

    The point is matched against the anchor in the camera's own frame, so the coordinate
    the owner points at on the live picture - the spot on the floor where the cat was
    sick, say - can be looked up without knowing anything about alignment. A cat counts
    only while consecutive samples stay inside ``radius`` and no more than
    ``point_gap_seconds`` apart; everything else is someone walking through, and sorting
    the result by minutes is what separates the two.
    """
    limits = query_limits()
    if not camera:
        raise ValueError("a camera is required; a coordinate only means something on one camera")
    if x is None or y is None:
        raise ValueError("both x and y are required")

    keep = float(radius) if radius is not None else limits["point_radius"]
    want_pixel = str(unit).lower().startswith("pixel")

    store = LocationStore(database or location_database())
    try:
        rows = [row for row in store.observations_between(start_ts, end_ts) if row.camera == camera]
    finally:
        store.close()

    frame = _frame_size(rows)
    if want_pixel:
        if frame is None:
            raise ValueError(
                f"no frame size recorded for {camera}; give x/y as fractions 0..1 instead"
            )
        width, height = frame
        qx, qy = float(x) / width, float(y) / height
        radius_norm = keep / width
    else:
        qx, qy = float(x), float(y)
        radius_norm = keep

    if frame is not None:
        width, height = frame
        threshold = radius_norm * width

        def distance(ax: float, ay: float) -> float:
            # Pixels, so the x and y axes are comparable on a 16:9 frame.
            return hypot((ax - qx) * width, (ay - qy) * height)

    else:
        threshold = radius_norm

        def distance(ax: float, ay: float) -> float:
            return hypot(ax - qx, ay - qy)

    matched: dict[str, list] = {}
    for row in rows:
        if cat is not None and row.cat != cat:
            continue
        anchor = row.camera_anchor
        if anchor is None:
            continue
        if distance(*anchor) <= threshold:
            matched.setdefault(row.cat, []).append(row)

    gap = limits["point_gap_seconds"]
    limit = limits["max_stays"] if max_stays is None else int(max_stays)
    stays: list[dict] = []
    samples_by_cat: dict[str, int] = {}
    for cat_name, cat_rows in matched.items():
        samples_by_cat[cat_name] = len(cat_rows)
        current: dict | None = None
        for row in cat_rows:  # observations_between orders by ts
            if current is not None and row.ts - current["last"] <= gap:
                current["last"] = row.ts
                current["samples"] += 1
            else:
                if current is not None:
                    stays.append(
                        _finish_point_stay(
                            camera, cat_name, current["first"], current["last"], current["samples"]
                        )
                    )
                current = {"first": row.ts, "last": row.ts, "samples": 1}
        if current is not None:
            stays.append(
                _finish_point_stay(
                    camera, cat_name, current["first"], current["last"], current["samples"]
                )
            )

    stays.sort(key=lambda item: (-item["minutes"], item["cat"], item["start"]))
    minutes_by_cat: dict[str, float] = {}
    for stay in stays:
        minutes_by_cat[stay["cat"]] = round(
            minutes_by_cat.get(stay["cat"], 0.0) + stay["minutes"], 1
        )

    point = {"x": round(qx, 5), "y": round(qy, 5), "unit": "normalized"}
    if frame is not None:
        point["pixels"] = [round(qx * frame[0]), round(qy * frame[1])]
    answer = {
        "camera": camera,
        "point": point,
        "radius": round(radius_norm, 5),
        "cat": cat or "all",
        "found": bool(stays),
        "stays": stays[: max(0, limit)],
        "truncated": len(stays) > limit,
        "max_stays": limit,
        "samples_by_cat": samples_by_cat,
        "minutes_by_cat": minutes_by_cat,
        "total_samples": sum(samples_by_cat.values()),
        "total_minutes": round(sum(minutes_by_cat.values()), 1),
    }
    return answer


def point_stay(
    *,
    camera: str,
    x: float,
    y: float,
    radius: float | None = None,
    unit: str = "normalized",
    cat: str | None = None,
    since: str | None = None,
    until: str | None = None,
    database: Path | None = None,
    max_stays: int | None = None,
    now: datetime | None = None,
) -> dict:
    """Which cat stayed near a point on one camera, and for how long.

    ``x``/``y`` are fractions of the frame (0..1, the default) or pixels when
    ``unit="pixel"``; ``radius`` is in the same unit and defaults to a fraction wide
    enough to cover a cat's body.
    """
    start_ts, end_ts, label = date_span(since, until, now=now)
    result = point_stay_between(
        start_ts,
        end_ts,
        camera=camera,
        x=x,
        y=y,
        radius=radius,
        unit=unit,
        cat=cat,
        database=database,
        max_stays=max_stays,
    )
    result["range"] = _range_payload(start_ts, end_ts, label)
    return result


def daily_summary(
    *,
    days_ago: int = 0,
    database: Path | None = None,
    now: datetime | None = None,
) -> dict:
    """The daily report's own summary, so a harness can ask "what happened today"."""
    day = _now_local(now).date() - timedelta(days=max(0, int(days_ago)))
    return build_summary(day, database)


@dataclass(frozen=True)
class Tool:
    """One callable, self-described, JSON-only at both ends."""

    name: str
    description: str
    parameters: dict
    function: Callable

    def schema(self) -> dict:
        """OpenAI-compatible function definition - what a harness registers."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_DATE_ARGUMENT = {"type": "string", "description": "Inclusive date, YYYY-MM-DD."}

TOOLS: dict[str, Tool] = {
    "zone_stay": Tool(
        name="zone_stay",
        description=(
            "How many times and for how long a cat was recorded in one or more zones, with "
            "exact start/end times and the camera. Use this for any 'has the cat ever been "
            "in X', 'how often', 'when was it in X' question. Zone names repeat across "
            "cameras, so always say which camera the stay was on."
        ),
        parameters={
            "type": "object",
            "properties": {
                "zones": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Zone identifiers, e.g. sink, toilet_1, on_sofa.",
                },
                "cat": {"type": "string", "description": "Limit to one cat; omit for both."},
                "since": _DATE_ARGUMENT,
                "until": _DATE_ARGUMENT,
            },
            "required": ["zones"],
        },
        function=zone_stay,
    ),
    "zone_totals": Tool(
        name="zone_totals",
        description=(
            "Total minutes per zone and camera over a date range, ranked. Use for 'where did "
            "the cat spend its time' and for questions spanning more than one day."
        ),
        parameters={
            "type": "object",
            "properties": {
                "since": _DATE_ARGUMENT,
                "until": _DATE_ARGUMENT,
                "cat": {"type": "string", "description": "Limit to one cat; omit for both."},
                "top": {"type": "integer", "description": "How many rows to return."},
            },
        },
        function=zone_totals,
    ),
    "daily_summary": Tool(
        name="daily_summary",
        description=(
            "The daily report's own summary for one day: minutes per zone per cat plus the "
            "code-computed meal, drinking and toilet verdicts."
        ),
        parameters={
            "type": "object",
            "properties": {
                "days_ago": {"type": "integer", "description": "0 = today, 1 = yesterday."},
            },
        },
        function=daily_summary,
    ),
    "point_stay": Tool(
        name="point_stay",
        description=(
            "Which cat stayed near a specific spot on one camera, and for how long. Use "
            "this when the owner names a point rather than a zone - e.g. a spot on the "
            "floor where something happened. x/y are fractions of the frame (0..1) unless "
            "unit is 'pixel'. A coordinate only means something on one camera, so camera "
            "is required and the same x/y on another camera is a different place."
        ),
        parameters={
            "type": "object",
            "properties": {
                "camera": {
                    "type": "string",
                    "description": "Camera identifier, e.g. living_room, sofa, feeder.",
                },
                "x": {"type": "number", "description": "Horizontal position."},
                "y": {"type": "number", "description": "Vertical position."},
                "unit": {
                    "type": "string",
                    "enum": ["normalized", "pixel"],
                    "description": "How to read x/y and radius. Default normalized (0..1).",
                },
                "radius": {
                    "type": "number",
                    "description": "How far around the point counts, in the same unit as x/y.",
                },
                "cat": {"type": "string", "description": "Limit to one cat; omit for both."},
                "since": _DATE_ARGUMENT,
                "until": _DATE_ARGUMENT,
            },
            "required": ["camera", "x", "y"],
        },
        function=point_stay,
    ),
}


def tools_schema() -> list[dict]:
    """Every tool as an OpenAI-compatible definition, in registry order."""
    return [tool.schema() for tool in TOOLS.values()]


def call_tool(
    name: str,
    arguments: Mapping | None = None,
    *,
    database: Path | None = None,
    now: datetime | None = None,
) -> dict:
    """Run one tool. Never raises - a harness needs an error it can feed back."""
    tool = TOOLS.get(str(name))
    if tool is None:
        return {"ok": False, "error": f"unknown tool {name!r}", "tools": sorted(TOOLS)}
    arguments = dict(arguments or {})
    arguments.setdefault("database", database)
    arguments.setdefault("now", now)
    try:
        result = tool.function(**arguments)
    except TypeError as error:
        return {"ok": False, "tool": tool.name, "error": f"bad arguments: {error}"}
    except (ValueError, LookupError, OSError) as error:
        return {"ok": False, "tool": tool.name, "error": str(error)}
    return {"ok": True, "tool": tool.name, "result": result}


# A point the owner names, in the few forms that are unambiguous enough to trust:
# "x=1200 y=800", "坐标 1200 800", "(1200, 800)". Dates ("2026-10-03") do not match any
# of these, which is why the bare two-number form needs the 坐标 marker.
_POINT_XY = re.compile(
    r"x\s*[=:：]\s*(-?\d+(?:\.\d+)?)[^\d\-]+y\s*[=:：]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE
)
_POINT_COORD = re.compile(
    r"坐标\s*(?:是|为|在|[:：=])?\s*(-?\d+(?:\.\d+)?)[,，\s]+(-?\d+(?:\.\d+)?)"
)
_POINT_PAREN = re.compile(r"[（(]\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*[)）]")
_POINT_RADIUS = re.compile(r"半径\s*[:=：]?\s*(\d+(?:\.\d+)?)")


def parse_point(text: str) -> dict | None:
    """The spot a question names, or ``None``.

    Values at or below 1 on both axes are read as fractions of the frame; anything larger
    is pixels. That is the one convention that lets "0.6, 0.5" and "1200, 800" both mean
    what the owner sees.
    """
    match = _POINT_XY.search(text) or _POINT_COORD.search(text) or _POINT_PAREN.search(text)
    if match is None:
        return None
    x, y = float(match.group(1)), float(match.group(2))
    radius = _POINT_RADIUS.search(text)
    return {
        "x": x,
        "y": y,
        "unit": "normalized" if abs(x) <= 1 and abs(y) <= 1 else "pixel",
        "radius": float(radius.group(1)) if radius else None,
    }


def answer_question(
    content: str,
    *,
    now: datetime | None = None,
    database: Path | None = None,
    aliases: Mapping | None = None,
    max_stays: int | None = None,
) -> dict | None:
    """A code-computed answer when the question names a zone or a spot, else ``None``.

    ``None`` is the signal to leave the existing behaviour alone: a question that names no
    place is answered by the daily report path exactly as before. When a place *is* named
    the counting is done here, because that is the part the model measurably got wrong.

    A named coordinate is more specific than a named zone ("地板上 (0.6, 0.5)" is a spot
    inside the floor, not the whole floor), so it wins when both appear.
    """
    start_ts, end_ts, label = parse_range(content, now=now)
    cats = resolve_cats(content)
    cat = cats[0] if len(cats) == 1 else None

    point = parse_point(content)
    if point is not None:
        result = {
            "tool": "point_stay",
            "question": content.strip()[:200],
            "range": _range_payload(start_ts, end_ts, label),
        }
        cameras = resolve_cameras(content)
        if len(cameras) != 1:
            # Coordinates are per camera, so two rooms mean two different places; ask
            # rather than silently combining them.
            result["found"] = False
            result["needs_camera"] = True
            result["point"] = {key: point[key] for key in ("x", "y", "unit")}
            result["camera_candidates"] = cameras or sorted(configured_cameras())
            result["cat"] = cat or "all"
            return result
        answer = point_stay_between(
            start_ts,
            end_ts,
            camera=cameras[0],
            x=point["x"],
            y=point["y"],
            radius=point["radius"],
            unit=point["unit"],
            cat=cat,
            database=database,
            max_stays=max_stays,
        )
        answer.update(result)
        if cat is None and len(cats) > 1:
            answer["asked_about"] = cats
        return answer

    zones = resolve_zones(content, aliases=aliases, zones=known_zones())
    if not zones:
        return None

    result = stay_rows(start_ts, end_ts, zones, cat, database=database, max_stays=max_stays)
    result["tool"] = "zone_stay"
    result["question"] = content.strip()[:200]
    result["range"] = _range_payload(start_ts, end_ts, label)
    if cat is None and len(cats) > 1:
        result["asked_about"] = cats
    return result


def tools_documentation() -> str:
    """The registry as text, for a prompt or a human: name, description, JSON schema."""
    return json.dumps(tools_schema(), ensure_ascii=False, indent=2)
