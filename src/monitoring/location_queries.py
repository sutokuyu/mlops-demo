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
from pathlib import Path
from zoneinfo import ZoneInfo

from src.monitoring.location_config import IDENTITY_CLASSES, REPORT_CONFIG, location_database
from src.monitoring.location_report import build_summary, day_bounds
from src.monitoring.location_store import LocationStore
from src.monitoring.zone_vocabulary import known_zones, resolve_zones

DEFAULT_MAX_DAYS = 31
DEFAULT_MAX_STAYS = 20
DEFAULT_TOP_ZONES = 10

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
    for visit in matched:
        counts_by_cat[visit.cat] = counts_by_cat.get(visit.cat, 0) + 1
    return {
        "zones": sorted(wanted),
        "cat": cat or "all",
        "count": len(matched),
        "found": bool(matched),
        "total_minutes": round(
            sum(max(0.0, visit.end_ts - visit.start_ts) for visit in matched) / 60, 1
        ),
        "counts_by_cat": counts_by_cat,
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


def answer_question(
    content: str,
    *,
    now: datetime | None = None,
    database: Path | None = None,
    aliases: Mapping | None = None,
    max_stays: int | None = None,
) -> dict | None:
    """A code-computed answer when the question names a zone, else ``None``.

    ``None`` is the signal to leave the existing behaviour alone: a question that names no
    place is answered by the daily report path exactly as before. When a place *is* named
    the counting is done here, because that is the part the model measurably got wrong.
    """
    zones = resolve_zones(content, aliases=aliases, zones=known_zones())
    if not zones:
        return None
    cats = resolve_cats(content)
    cat = cats[0] if len(cats) == 1 else None
    start_ts, end_ts, label = parse_range(content, now=now)

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
