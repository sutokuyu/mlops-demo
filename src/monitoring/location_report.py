"""Summarize a day of cat locations and deliver it to an external API."""

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import date as date_type
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


def _resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not locate project root")


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config, resolve_config_path
from src.monitoring.location_store import LocationStore
from src.monitoring.location_zones import load_zones

CONFIG = load_config(PROJECT_ROOT / "configs" / "config.yaml")
LOCATION_CONFIG = load_config(PROJECT_ROOT / "configs" / "locations.yaml")
REPORT_CONFIG = LOCATION_CONFIG["report"]
TRACKING_CONFIG = LOCATION_CONFIG["tracking"]

DISCORD_MESSAGE_LIMIT = 1900

# Discord sits behind Cloudflare, which answers urllib's default
# "Python-urllib/3.x" agent with "403 error code: 1010". Any explicit agent works.
USER_AGENT = "mlops-cat-demo/1.0"
ERROR_BODY_LIMIT = 300


class DeliveryError(RuntimeError):
    """A report endpoint rejected the request, with the server's own message."""


NEUTRAL_PERSONA = "You are a home assistant for a cat owner."

DEFAULT_TEMPERATURE = 0.3
MAX_TEMPERATURE = 2.0

# The message has to survive Discord, which truncates at DISCORD_MESSAGE_LIMIT
# without any error. That cap lives here rather than in the configurable persona
# so clearing the persona cannot silently uncap the message.
MAX_REPORT_CHARACTERS = 600

DEFAULT_TOILET_WINDOW_SECONDS = 300.0
DEFAULT_TOILET_MIN_APPEARANCES = 2

# What the message must contain. Not configurable: it follows from what the
# tracker records, and a report that silently drops meals or toilet visits is
# wrong no matter how nicely it is written.
TASK = (
    "Report what each cat did today: how long it stayed in each location, which location "
    "it preferred, and any notable routine or movement. Meals and water always matter. "
    "Add the exact time of the cat's final appearance."
)

# The toilet rules used to live in `hints` and the model had to apply them by
# comparing timestamps in the timeline. Measured over eleven runs it was right
# about a third of the time, and rewording only traded false positives for
# omissions, so the verdict is now computed and handed over as a fact.
EVENTS = (
    "Toilet sightings are not part of `locations` or `timeline`: they are turned into "
    "`events` instead. A `toilet_use` entry there is a confirmed visit, with the exact "
    "time it started and ended. Report those. A cat with no `toilet_use` entry did not use "
    "the toilet, no matter where it was seen."
)

# Its own block because it needs to be unambiguous: folding "write in zh" into TASK
# was read as "write the report in Chinese", and the model left `on_sofa` untranslated.
PRESENTATION = (
    "Write the whole message in {language}. The location names in the data are English "
    "identifiers such as on_sofa or kitchen_counter; translate them into natural "
    "{language} wording rather than printing the identifier. Keep each cat's name "
    "exactly as it appears. Time should be numbers rather than words."
)

# Deliberately not configurable. Models happily invent a plausible day when a
# playful tone is requested, so this stays last in the prompt for recency.
GROUNDING_RULES = (
    "Use only the JSON data provided. Never invent events: every duration, count and time "
    "must come from the data, and anything the data does not cover must be left out."
)


def build_instruction(
    language: str,
    persona: str | None = None,
    style: str | None = None,
    hints: str | None = None,
) -> str:
    """Compose the system prompt from tunable voice plus fixed requirements.

    Three things in ``locations.yaml`` are configurable because they change for
    different reasons: ``persona`` (who is speaking), ``style`` (how it reads) and
    ``hints`` (loose notes about how to read the data). What the report must
    *contain* lives in TASK and EVENTS, so rewording the jokes can never drop a
    fact. Rules that can be computed from the data go in code instead: the toilet
    verdict used to be a hint and is now ``toilet_events()``.

    Passing ``None`` for any block means "use the configured one"; an explicit
    empty string asks for that block to be omitted.
    """
    config = REPORT_CONFIG["llm"]
    if persona is None:
        persona = config.get("persona") or ""
    if style is None:
        style = config.get("style") or ""
    if hints is None:
        hints = config.get("hints") or ""

    blocks = [
        persona.strip() or NEUTRAL_PERSONA,
        TASK,
        EVENTS,
        PRESENTATION.format(language=language),
        style.strip(),
        hints.strip(),
        f"Keep the whole message under {MAX_REPORT_CHARACTERS} characters.",
        GROUNDING_RULES,
    ]
    return "\n\n".join(block for block in blocks if block)


def day_bounds(day: date_type, tz: ZoneInfo) -> tuple[float, float]:
    start = datetime.combine(day, time.min, tzinfo=tz)
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def toilet_events(visits: list, zones: set[str], window: float, minimum: int) -> list[dict]:
    """Decide which toilet sightings were real visits, in seconds of precision.

    The tracker only catches a cat going in and coming out, so a genuine visit
    shows up as several short appearances close together with nothing else in
    between. That is a deterministic predicate over the visit list, so it is
    computed here rather than left to the model: an LLM asked to compare
    timestamps across a text timeline got it right about a third of the time.

    A group of ``minimum`` appearances spanning no more than ``window`` seconds,
    uninterrupted by any other location, becomes one ``toilet_use`` event.

    The number of separate camera sightings is deliberately not exposed: the model
    read it as "went twice" and reported a single visit as two.
    """
    if not zones or minimum < 2:
        return []

    tz = ZoneInfo(REPORT_CONFIG["timezone"])
    events: list[dict] = []
    group: list = []

    def clock(stamp: float) -> str:
        return datetime.fromtimestamp(stamp, tz).strftime("%H:%M:%S")

    def flush() -> None:
        if len(group) >= minimum:
            events.append(
                {
                    "kind": "toilet_use",
                    "zone": group[0].zone,
                    "start": clock(group[0].start_ts),
                    "end": clock(group[-1].end_ts),
                }
            )
        group.clear()

    for visit in visits:  # ordered by start_ts by the query
        if visit.zone not in zones:
            flush()
            continue
        if group and visit.start_ts - group[0].start_ts > window:
            flush()
        group.append(visit)
    flush()
    return events


def build_summary(day: date_type, database: Path | None = None) -> dict:
    tz = ZoneInfo(REPORT_CONFIG["timezone"])
    start_ts, end_ts = day_bounds(day, tz)
    store = LocationStore(database or resolve_config_path(TRACKING_CONFIG["database"]))
    try:
        visits = store.visits_between(start_ts, end_ts)
    finally:
        store.close()

    events_config = REPORT_CONFIG.get("events", {})
    toilet_zones = set(events_config.get("toilet_zones") or [])
    toilet_window = float(events_config.get("toilet_window_seconds", DEFAULT_TOILET_WINDOW_SECONDS))
    toilet_minimum = int(
        events_config.get("toilet_min_appearances", DEFAULT_TOILET_MIN_APPEARANCES)
    )

    totals: dict[str, dict[str, float]] = {}
    timeline: dict[str, list[dict]] = {}
    by_cat: dict[str, list] = {}
    for visit in visits:
        by_cat.setdefault(visit.cat, []).append(visit)
    for visit in visits:
        # Interpreted zones are reported through `events` instead. Leaving the raw
        # sightings in the timeline made the model mention pass-bys no matter how
        # firmly the prompt forbade it, so the data itself is filtered here.
        if visit.zone in toilet_zones:
            continue
        # Clip visits that span midnight so each day only counts its own share.
        clipped_start = max(visit.start_ts, start_ts)
        clipped_end = min(visit.end_ts, end_ts)
        duration = max(0.0, clipped_end - clipped_start)
        if duration <= 0:
            continue
        location = visit.location
        cat_totals = totals.setdefault(visit.cat, {})
        cat_totals[location] = cat_totals.get(location, 0.0) + duration
        timeline.setdefault(visit.cat, []).append(
            {
                "location": location,
                "camera": visit.camera,
                "zone": visit.zone,
                "start": datetime.fromtimestamp(clipped_start, tz).strftime("%H:%M:%S"),
                "end": datetime.fromtimestamp(clipped_end, tz).strftime("%H:%M:%S"),
                "minutes": round(duration / 60, 1),
            }
        )

    return {
        "date": day.isoformat(),
        "timezone": REPORT_CONFIG["timezone"],
        "cats": [
            {
                "cat": cat,
                "events": toilet_events(
                    by_cat.get(cat, []), toilet_zones, toilet_window, toilet_minimum
                ),
                "locations": [
                    {"location": location, "minutes": round(seconds / 60, 1)}
                    for location, seconds in sorted(
                        totals.get(cat, {}).items(), key=lambda item: item[1], reverse=True
                    )
                ],
                "timeline": timeline.get(cat, []),
            }
            for cat in sorted(set(totals) | set(by_cat))
        ],
    }


def _format_minutes(minutes: float) -> str:
    return "<1" if minutes < 1 else f"{minutes:.0f}"


def render_text(summary: dict) -> str:
    if not summary["cats"]:
        return f"{summary['date']}: no cat sightings recorded."
    lines = [f"Cat location summary for {summary['date']}:"]
    for entry in summary["cats"]:
        total = sum(location["minutes"] for location in entry["locations"])
        lines.append(f"\n{entry['cat']} — {_format_minutes(total)} min observed")
        for location in entry["locations"]:
            share = location["minutes"] / total * 100 if total else 0
            lines.append(
                f"  {location['location']}: {_format_minutes(location['minutes'])} min"
                f" ({share:.0f}%)"
            )
    return "\n".join(lines)


def _post(request: urllib.request.Request, timeout: int) -> bytes:
    """POST and turn an HTTP error into a message that includes the body.

    Discord and LLM gateways explain the real problem in the response body, so
    dropping it leaves the operator with only "HTTP Error 403: Forbidden".
    """
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:ERROR_BODY_LIMIT].strip()
        raise DeliveryError(f"{error.code} {error.reason}: {detail}") from error


def llm_temperature(override: float | None = None) -> float:
    """Sampling temperature from locations.yaml, unless overridden for one run.

    A livelier persona wants a higher value, but the same knob also makes the
    model likelier to drift away from the data. That is why the grounding rules
    stay in the system prompt no matter what persona or temperature is set.
    """
    if override is not None:
        value = float(override)
    else:
        raw = REPORT_CONFIG["llm"].get("temperature", DEFAULT_TEMPERATURE)
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"report.llm.temperature must be a number, got {raw!r}") from error
    if not 0.0 <= value <= MAX_TEMPERATURE:
        raise RuntimeError(
            f"report.llm.temperature must be between 0.0 and {MAX_TEMPERATURE}, got {value}"
        )
    return value


def call_llm(summary: dict, temperature: float | None = None) -> str:
    llm = REPORT_CONFIG["llm"]
    if not llm["api_key"]:
        raise RuntimeError("report.llm.api_key is not configured")
    language = REPORT_CONFIG.get("language", "zh")
    instruction = build_instruction(language)
    payload = {
        "model": llm["model"],
        "temperature": llm_temperature(temperature),
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(summary, ensure_ascii=False)},
        ],
    }
    body = _post(
        urllib.request.Request(
            llm["endpoint"],
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {llm['api_key']}",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        ),
        timeout=90,
    )
    return json.loads(body.decode("utf-8"))["choices"][0]["message"]["content"].strip()


def post_json(url: str, payload: dict, token: str = "") -> None:
    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    _post(
        urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        ),
        timeout=60,
    )


def send_to_discord(content: str) -> None:
    webhook_url = REPORT_CONFIG["discord"]["webhook_url"]
    if not webhook_url:
        raise RuntimeError("report.discord.webhook_url is not configured")
    post_json(
        webhook_url,
        {
            "content": content[:DISCORD_MESSAGE_LIMIT],
            "username": REPORT_CONFIG["discord"].get("username", "Cat Location Bot"),
        },
    )


def compose(summary: dict, text: str, temperature: float | None = None) -> str:
    """The outbound text, after any LLM rewriting for the configured mode.

    Split out of deliver() so a dry run can generate and show exactly what would
    have been posted without touching the network beyond the LLM call itself.
    """
    if REPORT_CONFIG.get("mode", "discord").lower() == "llm":
        return call_llm(summary, temperature)
    return text


def deliver(summary: dict, text: str, temperature: float | None = None) -> str:
    mode = REPORT_CONFIG.get("mode", "discord").lower()
    if mode == "http":
        http = REPORT_CONFIG["http"]
        if not http["url"]:
            raise RuntimeError("report.http.url is not configured")
        post_json(http["url"], {"text": text, "summary": summary}, http.get("token", ""))
        return "posted JSON to the configured HTTP endpoint"
    if mode == "llm":
        narrative = compose(summary, text, temperature)
        if REPORT_CONFIG["discord"]["webhook_url"]:
            send_to_discord(narrative)
            return "sent the LLM summary to Discord"
        return narrative
    send_to_discord(text)
    return "sent the summary to Discord"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate the day's cat locations and send the summary to an external API.",
    )
    parser.add_argument(
        "--date",
        type=date_type.fromisoformat,
        default=None,
        help="Day to summarize (YYYY-MM-DD); defaults to today in the configured timezone.",
    )
    parser.add_argument(
        "--days-ago",
        type=int,
        default=0,
        help="Summarize N days before today; convenient for a midnight cron job.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Location database to read; defaults to tracking.database in locations.yaml.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override report.llm.temperature for one run; handy for tuning the tone.",
    )
    parser.add_argument(
        "--print-only", action="store_true", help="Print the summary, send nothing."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the message (including the LLM summary) but post nothing; "
        "use this to iterate on the persona without spamming Discord.",
    )
    parser.add_argument("--json", action="store_true", help="Print the raw summary JSON.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    tz = ZoneInfo(REPORT_CONFIG["timezone"])
    if args.date is not None:
        target_day = args.date
    else:
        target_day = (datetime.now(tz) - timedelta(days=args.days_ago)).date()

    zones_configured = sum(len(zones) for zones in load_zones().values())
    if not zones_configured:
        print("Warning: no zones defined yet; locations fall back to camera names.")

    summary = build_summary(target_day, args.database)
    text = render_text(summary)
    print(text)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.print_only:
        return
    try:
        if args.dry_run:
            preview = compose(summary, text, args.temperature)
            print(f"\n--- 以下内容原本会发送（--dry-run，未发送）---\n\n{preview}")
            return
        print(f"Report: {deliver(summary, text, args.temperature)}")
    except (urllib.error.URLError, RuntimeError, KeyError, IndexError) as error:
        raise SystemExit(f"Failed to deliver the report: {error}") from error


if __name__ == "__main__":
    main()
