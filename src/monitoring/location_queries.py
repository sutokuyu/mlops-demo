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
    TRACKING_CONFIG,
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
# The gap that still counts as ONE stay, and therefore what "待过" means. The recorder
# samples every 5s, so a hole of four samples (20s) is a detection miss while a hole of
# 60s is twelve misses in a row - the cat left. Measured on 2026-10-08: with 60s bagel's
# six samples at 13:50:22 and two blips at 13:51:07/13:51:27 merged into one 65-second
# "stay", so the 25-second event the owner was asking about was reported as a minute-odd
# blob. At 20s it comes out as exactly 13:50:22-13:50:47 (25s) and the later sightings
# stand alone.
DEFAULT_POINT_GAP_SECONDS = 20.0
# A stay that lands just OUTSIDE the radius is still an answer to "who was around here".
# Measured 2026-10-10: the owner asked about sofa (35,75) and bagel had been 141px from
# a 128px radius for 25 seconds - the reply was a silent "no record", because a 5px
# miss is indistinguishable from nothing once the rows are filtered. So the same
# clustering runs again over this wider area and whatever fell entirely outside the
# radius is reported as `nearby`, carrying how far outside in the units the owner types.
DEFAULT_POINT_FALLBACK_RADIUS = 0.15
DEFAULT_NEARBY_MAX = 3

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
        "point_fallback_radius": float(
            raw.get("point_fallback_radius", DEFAULT_POINT_FALLBACK_RADIUS)
        ),
        "point_nearby_max": int(raw.get("point_nearby_max", DEFAULT_NEARBY_MAX)),
        "point_gap_seconds": float(raw.get("point_gap_seconds", DEFAULT_POINT_GAP_SECONDS)),
    }


def _now_local(now: datetime | None = None) -> datetime:
    tz = report_timezone()
    return now.astimezone(tz) if now is not None else datetime.now(tz)


def camera_repositioned_at(camera: str) -> float | None:
    """When this camera was last physically moved, as an epoch, or ``None``.

    A point query compares the stored camera-frame anchor against a coordinate the owner
    read off the CURRENT picture. Those are only comparable if the camera has not moved
    in between: after a move the same x/y is a different physical place, so matching old
    rows would invent a story about a spot nobody was ever near. ``tracking.repositioned_at``
    records the moment, per camera; a camera that was only redrawn or re-anchored in
    place must NOT be listed there, because nothing became incomparable in that case.
    """
    raw = (TRACKING_CONFIG.get("repositioned_at") or {}).get(camera)
    if raw in (None, ""):
        return None
    if isinstance(raw, datetime):
        moment = raw
    else:
        try:
            moment = datetime.fromisoformat(str(raw))
        except ValueError as error:
            raise ValueError(
                f"tracking.repositioned_at.{camera} is not an ISO timestamp: {raw!r}"
            ) from error
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=report_timezone())
    return moment.timestamp()


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


def _safe_date(year: int, month: int, day: int) -> date | None:
    """A calendar date, or ``None`` when the numbers are not one.

    The regexes are loose enough to catch a coordinate: ``30-70`` matches the month/day
    shape, and building ``date(year, 30, 70)`` raises. A question must not be lost - or
    worse, crash the caller - because its text happened to contain two numbers.
    """
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def explicit_day(text: str, *, now: datetime | None = None) -> date | None:
    """The one calendar day a message names outright, or ``None``.

    "10月8号", "10/8", "2026-10-08" pin a single day. "昨天" and ranges ("最近3天",
    "这周") deliberately do not count here - they are not a single date and the range
    logic already handles them. This exists because the report day used to come only
    from 昨天/前天, so "10月8号猫都干啥了" was answered with TODAY's report.
    """
    iso = _ISO_DATE.search(text)
    if iso:
        return _safe_date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
    month_day = _MONTH_DAY.search(text)
    if month_day:
        return _safe_date(_now_local(now).year, int(month_day.group(1)), int(month_day.group(2)))
    return None


def _parse_date(value: str | date | None, tz: ZoneInfo) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.astimezone(tz).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if text.isdigit() and len(text) == 8:  # YYYYMMDD
        return _safe_date(int(text[:4]), int(text[4:6]), int(text[6:]))
    match = _ISO_DATE.search(text)
    if match:
        return _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = _MONTH_DAY.search(text)
    if match:
        return _safe_date(_now_local().year, int(match.group(1)), int(match.group(2)))
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

    iso_dates = [_safe_date(int(y), int(m), int(d)) for y, m, d in _ISO_DATE.findall(text)]
    iso_dates = [value for value in iso_dates if value is not None]
    if iso_dates:
        return _span(min(iso_dates), max(iso_dates), tz)

    month_days = [_safe_date(today.year, int(m), int(d)) for m, d in _MONTH_DAY.findall(text)]
    month_days = [value for value in month_days if value is not None]
    if month_days:
        return _span(min(month_days), max(month_days), tz)

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
    min_seconds: float | None = None,
) -> dict:
    """The core count, over every visit in the range - not over the returned sample.

    ``count`` and ``total_minutes`` always describe the whole range; ``stays`` may be
    truncated to ``max_stays`` and says so, because a model that is handed 20 rows and
    told "the count is 137" must not read 20 as the answer.

    ``min_seconds`` drops stays shorter than that, for a question that asks for them
    (""停留过超过5秒""). A visit's duration is real here - it was opened and closed
    across samples - so unlike a point stay there is nothing to estimate.
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
        if visit.zone in wanted
        and (cat is None or visit.cat == cat)
        and (min_seconds is None or (visit.end_ts - visit.start_ts) >= float(min_seconds))
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
    result = {
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
    if min_seconds is not None:
        result["min_seconds"] = float(min_seconds)
    return result


def zone_stay(
    *,
    zones,
    cat: str | None = None,
    since: str | None = None,
    until: str | None = None,
    database: Path | None = None,
    max_stays: int | None = None,
    min_seconds: float | None = None,
    now: datetime | None = None,
) -> dict:
    """How often and how long a cat stayed in the given zones, with exact times."""
    start_ts, end_ts, label = date_span(since, until, now=now)
    result = stay_rows(
        start_ts,
        end_ts,
        _as_list(zones),
        cat or None,
        database=database,
        max_stays=max_stays,
        min_seconds=min_seconds,
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
    span = max(0.0, last - first)
    return {
        "cat": cat,
        "camera": camera,
        "start": _format_time(first, tz),
        "end": _format_time(last, tz),
        "seconds": round(span, 1),
        "minutes": round(span / 60, 1),
        "samples": samples,
    }


def _cluster_point_stays(camera: str, cat: str, ranked: list[tuple], gap: float) -> list[dict]:
    """``[(row, distance)]`` for one cat -> stay dicts, each carrying its closest sample.

    Consecutive samples no more than ``gap`` apart are one stay; the closest distance is
    kept because that is what decides whether the stay was inside the asked radius, and
    how far outside it was when it was not.
    """
    stays: list[dict] = []
    current: dict | None = None
    for row, distance in ranked:  # observations_between orders by ts
        if current is not None and row.ts - current["last"] <= gap:
            current["last"] = row.ts
            current["samples"] += 1
            current["closest"] = min(current["closest"], distance)
        else:
            if current is not None:
                stays.append(
                    {
                        **_finish_point_stay(
                            camera, cat, current["first"], current["last"], current["samples"]
                        ),
                        "closest": current["closest"],
                    }
                )
            current = {
                "first": row.ts,
                "last": row.ts,
                "samples": 1,
                "closest": distance,
            }
    if current is not None:
        stays.append(
            {
                **_finish_point_stay(
                    camera, cat, current["first"], current["last"], current["samples"]
                ),
                "closest": current["closest"],
            }
        )
    return stays


def point_stay_between(
    start_ts: float,
    end_ts: float,
    *,
    camera: str,
    x: float,
    y: float,
    radius: float | None = None,
    unit: str | None = None,
    cat: str | None = None,
    database: Path | None = None,
    max_stays: int | None = None,
    min_seconds: float | None = None,
) -> dict:
    """Who stayed near a point on one camera, over an explicit range.

    The point is matched against the anchor in the camera's own frame, so the coordinate
    the owner points at on the live picture - the spot on the floor where the cat was
    sick, say - can be looked up without knowing anything about alignment. A cat counts
    only while consecutive samples stay inside ``radius`` and no more than
    ``point_gap_seconds`` apart; everything else is someone walking through, and sorting
    the result by minutes is what separates the two.

    ``min_seconds`` is the ""超过5秒钟"" part of a question: it keeps only stays whose
    first and last detection are at least that far apart. A single detection has a
    0-second span - nothing between the two ends to measure - so it cannot demonstrate a
    duration, and the ones dropped are counted in ``dropped_below_min_seconds`` rather
    than vanishing.
    """
    limits = query_limits()
    if not camera:
        raise ValueError("a camera is required; a coordinate only means something on one camera")
    if x is None or y is None:
        raise ValueError("both x and y are required")

    kind = str(unit or "").strip().lower()
    if not kind:
        # No unit given: the magnitude decides, exactly as it does in a typed question.
        if abs(float(x)) <= 1 and abs(float(y)) <= 1:
            kind = "normalized"
        elif abs(float(x)) <= 100 and abs(float(y)) <= 100:
            kind = "percent"
        else:
            kind = "pixel"
    want_pixel = kind.startswith("pixel")
    want_percent = kind.startswith("percent") or kind in ("%", "pct")

    store = LocationStore(database or location_database())
    try:
        rows = [row for row in store.observations_between(start_ts, end_ts) if row.camera == camera]
    finally:
        store.close()

    # Rows from before the camera was physically moved hold coordinates of a different
    # view, so they are dropped rather than matched - and counted, so the reply can say
    # the range is not empty rather than presenting it as "nobody was there".
    repositioned = camera_repositioned_at(camera)
    excluded_reposition = 0
    if repositioned is not None:
        kept = [row for row in rows if row.ts >= repositioned]
        excluded_reposition = len(rows) - len(kept)
        rows = kept

    frame = _frame_size(rows)
    # ``radius`` is given in the caller's unit, but the default is a fraction of the frame
    # (points are stored 0..1), so the default is not divided by 100 or by the frame.
    default_radius = limits["point_radius"]
    if want_pixel:
        if frame is None:
            raise ValueError(f"no frame size recorded for {camera}; give x/y as 0-100 instead")
        width, height = frame
        qx, qy = float(x) / width, float(y) / height
        radius_norm = float(radius) / width if radius is not None else default_radius
    elif want_percent:
        qx, qy = float(x) / 100.0, float(y) / 100.0
        radius_norm = float(radius) / 100.0 if radius is not None else default_radius
    else:
        qx, qy = float(x), float(y)
        radius_norm = float(radius) if radius is not None else default_radius

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

    # Distances are pixels when the frame size is known and already normalized when it is
    # not, so dividing by the frame width (or by 1) gives the 0-100 units the owner types.
    unit_scale = float(frame[0]) if frame is not None else 1.0
    fallback_threshold = limits["point_fallback_radius"] * unit_scale

    def candidates(max_distance: float) -> dict[str, list]:
        """Rows within ``max_distance``, grouped by cat, each paired with its distance."""
        grouped: dict[str, list] = {}
        for row in rows:
            if cat is not None and row.cat != cat:
                continue
            anchor = row.camera_anchor
            if anchor is None:
                continue
            found = distance(*anchor)
            if found <= max_distance:
                grouped.setdefault(row.cat, []).append((row, found))
        return grouped

    gap = limits["point_gap_seconds"]
    limit = limits["max_stays"] if max_stays is None else int(max_stays)
    stays: list[dict] = []
    for cat_name, ranked in candidates(threshold).items():
        for stay in _cluster_point_stays(camera, cat_name, ranked, gap):
            # The distance only means something for a near-miss, so the primary answer
            # keeps exactly the shape it had.
            stay.pop("closest")
            stays.append(stay)

    stays.sort(key=lambda item: (-item["minutes"], item["cat"], item["start"]))

    # Near-misses, so a point a few units off cannot turn a real stay into "no record".
    # A cluster counts as nearby only when EVERY one of its samples was outside the
    # radius, which is also why it can never duplicate something already in `stays`.
    nearby: list[dict] = []
    for cat_name, ranked in candidates(fallback_threshold).items():
        for stay in _cluster_point_stays(camera, cat_name, ranked, gap):
            if stay["closest"] > threshold:
                stay["distance_percent"] = round(stay.pop("closest") / unit_scale * 100, 1)
                nearby.append(stay)
    nearby.sort(key=lambda item: (item["distance_percent"], item["cat"]))

    dropped = 0
    if min_seconds is not None:
        threshold_seconds = float(min_seconds)
        kept = [stay for stay in stays if stay["seconds"] >= threshold_seconds]
        dropped = len(stays) - len(kept)
        stays = kept
        nearby = [stay for stay in nearby if stay["seconds"] >= threshold_seconds]
    nearby = nearby[: limits["point_nearby_max"]]

    # Counted over the stays that are being reported, so the sample counts and the
    # answer cannot disagree once a duration threshold has dropped some.
    samples_by_cat: dict[str, int] = {}
    minutes_by_cat: dict[str, float] = {}
    for stay in stays:
        samples_by_cat[stay["cat"]] = samples_by_cat.get(stay["cat"], 0) + stay["samples"]
        minutes_by_cat[stay["cat"]] = round(
            minutes_by_cat.get(stay["cat"], 0.0) + stay["minutes"], 1
        )

    point = {
        "x": round(qx, 5),
        "y": round(qy, 5),
        "unit": "normalized",
        # The 0-100 form, so a harness can hand back the numbers the owner typed.
        "percent": [round(qx * 100, 2), round(qy * 100, 2)],
    }
    if frame is not None:
        point["pixels"] = [round(qx * frame[0]), round(qy * frame[1])]
    answer = {
        "camera": camera,
        "point": point,
        "radius": round(radius_norm, 5),
        "radius_percent": round(radius_norm * 100, 2),
        "cat": cat or "all",
        "found": bool(stays),
        "stays": stays[: max(0, limit)],
        "truncated": len(stays) > limit,
        "max_stays": limit,
        # Stays that were only just outside the radius, with how far outside. Empty
        # whenever everything near the point was already inside it.
        "nearby": nearby,
        "samples_by_cat": samples_by_cat,
        "minutes_by_cat": minutes_by_cat,
        "total_samples": sum(samples_by_cat.values()),
        "total_minutes": round(sum(minutes_by_cat.values()), 1),
    }
    if min_seconds is not None:
        answer["min_seconds"] = float(min_seconds)
        answer["dropped_below_min_seconds"] = dropped
    if repositioned is not None:
        answer["excluded_before_reposition"] = excluded_reposition
    return answer


def point_stay(
    *,
    camera: str,
    x: float,
    y: float,
    radius: float | None = None,
    unit: str | None = None,
    cat: str | None = None,
    since: str | None = None,
    until: str | None = None,
    database: Path | None = None,
    max_stays: int | None = None,
    min_seconds: float | None = None,
    now: datetime | None = None,
) -> dict:
    """Which cat stayed near a point on one camera, and for how long.

    ``x``/``y`` are read on the 0-100 scale the preview readout shows, or as 0..1
    fractions, or as pixels - the magnitudes decide when ``unit`` is omitted. ``radius``
    is in the same unit and defaults to a fraction wide enough to cover a cat's body.
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
        min_seconds=min_seconds,
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
                "min_seconds": {
                    "type": "number",
                    "description": (
                        "Only count stays at least this many seconds long. Use it when the "
                        "owner says 超过5秒 / 至少2分钟."
                    ),
                },
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
            "floor where something happened. x/y are on the 0-100 scale the preview "
            "readout shows (0-1 also accepted; unit='pixel' for raw pixels). A coordinate "
            "only means something on one camera, so camera is required and the same x/y "
            "on another camera is a different place."
        ),
        parameters={
            "type": "object",
            "properties": {
                "camera": {
                    "type": "string",
                    "description": "Camera identifier, e.g. living_room, sofa, feeder.",
                },
                "x": {"type": "number", "description": "Horizontal position, 0-100 or 0-1."},
                "y": {"type": "number", "description": "Vertical position, 0-100 or 0-1."},
                "unit": {
                    "type": "string",
                    "enum": ["percent", "normalized", "pixel"],
                    "description": (
                        "How to read x/y and radius: percent (0-100), normalized (0-1) "
                        "or pixel. Omit it and the magnitudes decide."
                    ),
                },
                "radius": {
                    "type": "number",
                    "description": "How far around the point counts, in the same unit as x/y.",
                },
                "cat": {"type": "string", "description": "Limit to one cat; omit for both."},
                "since": _DATE_ARGUMENT,
                "until": _DATE_ARGUMENT,
                "min_seconds": {
                    "type": "number",
                    "description": (
                        "Only count stays at least this many seconds long (the span between "
                        "the first and last detection). Use it when the owner says "
                        "超过5秒 / 至少2分钟; a single detection is 0 seconds long."
                    ),
                },
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
# "x=1200 y=800", "x30 y70", "x35y75", "坐标 1200 800", "(1200, 800)". Dates
# ("2026-10-03") do not match any of these, which is why the bare two-number form needs
# the 坐标 marker.
#
# The `x`/`y` label is required: "sofa 30,70" stays unparsed rather than being guessed
# at, because two bare numbers are as likely to be a date, a duration or a sentence.
# The separator after a label is optional AND the gap between the two numbers may be
# empty - the owner wrote "x35y75" - so the gap is `*` and not `+`. With `+` the gap
# swallowed the `y` label itself ("x35y75" -> the gap ate the "y", then a literal "y"
# was still required) and the whole message parsed as no coordinate at all. Lazy
# backtracking is what makes the empty gap work: greedy tries "y", fails, and gives it
# back.
_POINT_XY = re.compile(
    r"x\s*[=:：]?\s*(-?\d+(?:\.\d+)?)[^\d\-]*y\s*[=:：]?\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE
)
_POINT_COORD = re.compile(
    r"坐标\s*(?:是|为|在|[:：=])?\s*(-?\d+(?:\.\d+)?)[,，\s]+(-?\d+(?:\.\d+)?)"
)
_POINT_PAREN = re.compile(r"[（(]\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*[)）]")
_POINT_RADIUS = re.compile(r"半径\s*[:=：]?\s*(\d+(?:\.\d+)?)")

# "停留超过5秒钟" / "待够2分钟" - a floor on how long a stay has to last to count. The
# unit is required: a bare number after 超过 is as likely to be a count as a duration.
_DURATION_UNITS = {"小时": 3600.0, "分钟": 60.0, "分": 60.0, "秒钟": 1.0, "秒": 1.0}
_DURATION = re.compile(r"(?:超过|至少|大于|不少于|够)\s*(\d+(?:\.\d+)?)\s*(小时|分钟|秒钟|秒|分)")


def parse_duration(text: str) -> float | None:
    """The minimum stay the question asks for, in seconds, or ``None``.

    "超过5秒钟", "至少2分钟", "待够30秒". Kept separate from the point/zone parsers because
    it is a property of the question, not of a place - the same phrase has to be able to
    narrow a stay on a zone or near a point.
    """
    match = _DURATION.search(text)
    if match is None:
        return None
    return float(match.group(1)) * _DURATION_UNITS[match.group(2)]


def parse_point(text: str) -> dict | None:
    """The spot a question names, or ``None``.

    Three scales are read from the numbers themselves, because the owner should not have
    to say which one they mean: at or below 1 is a fraction of the frame (how the data
    is stored), up to 100 is the 0-100 scale the readout shows, and anything larger is a
    raw pixel coordinate. That keeps old questions and new ones working from one syntax.
    """
    match = _POINT_XY.search(text) or _POINT_COORD.search(text) or _POINT_PAREN.search(text)
    if match is None:
        return None
    x, y = float(match.group(1)), float(match.group(2))
    radius = _POINT_RADIUS.search(text)
    if abs(x) <= 1 and abs(y) <= 1:
        unit = "normalized"
    elif abs(x) <= 100 and abs(y) <= 100:
        unit = "percent"
    else:
        unit = "pixel"
    return {
        "x": x,
        "y": y,
        "unit": unit,
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
    # "超过5秒钟" narrows a stay wherever the question points, so it is read once here
    # and handed to whichever tool runs.
    min_seconds = parse_duration(content)

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
            min_seconds=min_seconds,
        )
        answer.update(result)
        if cat is None and len(cats) > 1:
            answer["asked_about"] = cats
        return answer

    zones = resolve_zones(content, aliases=aliases, zones=known_zones())
    if not zones:
        return None

    result = stay_rows(
        start_ts,
        end_ts,
        zones,
        cat,
        database=database,
        max_stays=max_stays,
        min_seconds=min_seconds,
    )
    result["tool"] = "zone_stay"
    result["question"] = content.strip()[:200]
    result["range"] = _range_payload(start_ts, end_ts, label)
    if cat is None and len(cats) > 1:
        result["asked_about"] = cats
    return result


def tools_documentation() -> str:
    """The registry as text, for a prompt or a human: name, description, JSON schema."""
    return json.dumps(tools_schema(), ensure_ascii=False, indent=2)
