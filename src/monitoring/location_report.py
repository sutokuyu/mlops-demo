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

# Feeding is judged by how long the cat stayed, not by how many times it appeared:
# one 3-minute stay at the feeder is a meal, and a cat walking past is not.
# ``merge`` glues a meal back together when the cat steps between the feeder and a
# food bowl, or stops for water in the middle.
DEFAULT_FEEDING_MIN_SECONDS = 30.0
DEFAULT_FEEDING_MERGE_SECONDS = 120.0

# The litter box hides the cat, so its zone label is useless: either the cat is not
# detected at all inside the box, or the detection box's bottom edge lands on the
# floor in front of it and the sample is recorded as `floor`. Measured on
# 2026-09-21 against the smart litter box's own log (six uses): the zone-based rule
# caught none of them, and `visits` has not held a single `toilet_1` row in its
# entire history.
#
# So membership is geometric instead: a rectangle covering the box's visible part
# and the floor in front of it. Two appearances inside it, within the window, is a
# visit - which is also the only setting that caught all six measured uses (three
# appearances caught four of six).
DEFAULT_TOILET_REGION_GAP_SECONDS = 300.0
DEFAULT_TOILET_REGION_MIN_SAMPLES = 2
# What turns a sighting into a confident one. Both are needed: a cat parked at the
# box for a minute may only be detected twice while it is hidden, and a cat dashing
# past can produce four samples in ten seconds.
DEFAULT_TOILET_REGION_CONFIRM_SECONDS = 30.0
DEFAULT_TOILET_REGION_CONFIRM_SAMPLES = 4

# What the message must contain. Not configurable: it follows from what the
# tracker records, and a report that silently drops meals or toilet visits is
# wrong no matter how nicely it is written.
TASK = (
    "Report what each cat did today: how long it stayed in each location, which location "
    "it preferred, and any notable routine or movement. Meals and water always matter: "
    "say whether the cat ate and drank, using the verdicts in `events`. Add the exact "
    "time of the cat's final appearance."
)

# The toilet rules used to live in `hints` and the model had to apply them by
# comparing timestamps in the timeline. Measured over eleven runs it was right
# about a third of the time, and rewording only traded false positives for
# omissions, so the verdict is now computed and handed over as a fact.
#
# Feeding went the same way later, for a different reason: asked "did bagel eat
# today?", the model invented a criterion of its own ("did it really eat, or just
# sit by the bowl?") and then failed to apply it evenly - the same two-to-five
# minute feeder visit was a meal for one cat and not a meal for the other. What
# counts as eating is a definition, and a definition belongs in code.
EVENTS = (
    "Some facts are decided in code and handed over as verdicts rather than left to you, "
    "because deciding them from a text timeline is unreliable and comes out inconsistent "
    "between the two cats. `meal` is a confirmed meal and `drinking` is a confirmed "
    "drink: the cat was at a feeder, food bowl or water bowl long enough to be using "
    "it. `toilet_use` is a visit to the litter box and carries a `certainty`: "
    '"confirmed" means the cat stayed there, "brief" means it was only glimpsed at the '
    "box - the box hides the cat, so a brief sighting was either a visit or the cat "
    "walking past, and the camera cannot tell which. Report all of them with their "
    "times, stating the confirmed ones plainly and the brief ones as uncertain. A cat "
    "with no `meal` entry did not eat today. Raw `toilet_1` sightings are excluded "
    "from `locations` and `timeline`."
)

# Its own block because it needs to be unambiguous: folding "write in zh" into TASK
# was read as "write the report in Chinese", and the model left `on_sofa` untranslated.
PRESENTATION = (
    "Write the whole message in {language}. The location names in the data are English "
    "identifiers such as on_sofa or kitchen_counter; translate them into natural "
    "{language} wording rather than printing the identifier. Keep each cat's name "
    "exactly as it appears. Time should be numbers rather than words."
)

# Appended to the prompt when someone asks for something specific instead of the
# plain daily report (the Discord bot passes the message through). It sits right
# after TASK so the EVENTS verdicts, the presentation rules and the grounding
# rules all still apply to the answer; only the "what to cover" part is widened.
QUESTION = (
    'The owner asked this by message: "{question}"\n'
    "Answer it directly and first. The daily requirements above still apply: meals, water "
    "and toilet verdicts are never dropped, and a question the data cannot answer is "
    "answered by saying the data does not cover it."
)

# User text ends up inside the system prompt, so it is capped: a long message
# would otherwise be able to push GROUNDING_RULES out of the model's attention.
MAX_QUESTION_CHARACTERS = 200

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
    question: str | None = None,
) -> str:
    """Compose the system prompt from tunable voice plus fixed requirements.

    Three things in ``locations.yaml`` are configurable because they change for
    different reasons: ``persona`` (who is speaking), ``style`` (how it reads) and
    ``hints`` (loose notes about how to read the data). What the report must
    *contain* lives in TASK and EVENTS, so rewording the jokes can never drop a
    fact. Rules that can be computed from the data go in code instead: the toilet
    verdict used to be a hint and is now ``toilet_events()``.

    ``question`` is an optional message from the owner (the Discord bot forwards
    it). It adds a block and changes nothing else, so every fixed requirement is
    still in the prompt.

    Passing ``None`` for any block means "use the configured one"; an explicit
    empty string asks for that block to be omitted. A blank ``question`` is
    omitted like any other empty block.
    """
    config = REPORT_CONFIG["llm"]
    if persona is None:
        persona = config.get("persona") or ""
    if style is None:
        style = config.get("style") or ""
    if hints is None:
        hints = config.get("hints") or ""

    question_block = ""
    if question and question.strip():
        question_block = QUESTION.format(
            question=question.strip()[:MAX_QUESTION_CHARACTERS],
        )

    blocks = [
        persona.strip() or NEUTRAL_PERSONA,
        TASK,
        question_block,
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


def _feeding_groups(
    visits: list,
    zones: set[str],
    minimum_seconds: float,
    merge_seconds: float,
) -> list[list]:
    """The visits that make up each meal/drink, in start_ts order.

    A visit that is too short is skipped without ending the group; only time does.
    """
    if not zones:
        return []

    groups: list[list] = []
    group: list = []
    for visit in visits:  # ordered by start_ts by the query
        if visit.zone not in zones:
            continue
        if visit.end_ts - visit.start_ts < minimum_seconds:
            continue
        if group and visit.start_ts - group[-1].end_ts > merge_seconds:
            groups.append(group)
            group = []
        group.append(visit)
    if group:
        groups.append(group)
    return groups


def feeding_events(
    visits: list,
    zones: set[str],
    minimum_seconds: float,
    merge_seconds: float,
    kind: str,
) -> list[dict]:
    """The meals (or drinks) a cat definitely had, decided here rather than by the model.

    The toilet verdict can lean on "it went in and came out", which is two
    appearances. Feeding cannot: a single three-minute stay at the feeder is a
    meal, and a cat trotting past is not, so the discriminator is the dwell time.

    Groups are merged by the gap between one visit's end and the next one's start,
    and a group may span zones - stepping from `feeder_1` to `wet_food_bowl_1`, or
    pausing at the water server, is one meal. That is deliberately looser than the
    toilet rule, which is cut by any other location: leaving the feeder for two
    minutes and coming back is the same sitting, whereas leaving the litter box and
    coming back is a second visit.
    """
    tz = ZoneInfo(REPORT_CONFIG["timezone"])
    events: list[dict] = []
    for group in _feeding_groups(visits, zones, minimum_seconds, merge_seconds):
        events.append(
            {
                "kind": kind,
                "zones": sorted({visit.zone for visit in group}),
                "start": datetime.fromtimestamp(group[0].start_ts, tz).strftime("%H:%M:%S"),
                "end": datetime.fromtimestamp(group[-1].end_ts, tz).strftime("%H:%M:%S"),
                # Summed dwell, not wall clock: a cat that napped halfway through
                # did not spend that time at the bowl.
                "minutes": round(sum(visit.end_ts - visit.start_ts for visit in group) / 60, 1),
            }
        )
    return events


def feeding_spans(
    visits: list,
    zones: set[str],
    minimum_seconds: float,
    merge_seconds: float,
) -> list[tuple[float, float]]:
    """When the cat was at a bowl, for rules that care what else it was doing."""
    return [
        (group[0].start_ts, group[-1].end_ts)
        for group in _feeding_groups(visits, zones, minimum_seconds, merge_seconds)
    ]


def toilet_region_events(
    observations: list,
    region: dict | None,
    gap_seconds: float,
    minimum_samples: int,
    confirm_seconds: float,
    confirm_samples: int,
    busy_spans: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """Litter box visits, decided from where the cat was seen rather than its zone label.

    The box hides the cat. Inside it the tracker either loses it entirely or sees a
    box whose bottom edge sits on the floor in front, so the zone comes back as
    `floor` or empty - and the visit gate (two consecutive agreeing samples) then
    never opens a visit at all. Measured against the smart litter box's own log on
    2026-09-21: six real uses, none of which the zone rule found.

    So membership is geometric, and the evidence is appearances rather than dwell
    time: a cat that steps in and out of a box is hidden for most of the visit.
    Two appearances within ``gap_seconds`` is the threshold that caught all six
    measured uses; three caught four of them.

    ``certainty`` is the honest part. A brief sighting at the box is genuinely
    ambiguous - the measured set contains a five-second pair of samples that really
    was a visit and another five-second pair that was a cat walking past - so those
    are marked ``brief`` and the report is asked to hedge them rather than pretend.

    ``busy_spans`` are the cat's own meals and drinks. The bowls sit right beside the
    box in this room, so the cat's path to dinner crosses the region, and an
    appearance during a meal is the cat at the bowls rather than at the box. Those
    appearances are skipped AND they end the group, which matters: a cat that ate in
    the middle of a box visit would otherwise have the whole visit timed from its
    pre-dinner walk-through (measured: that turned a confirmed visit at 03:51 into a
    7.6 minute "brief" one).
    """
    if not region or minimum_samples < 2:
        return []

    tz = ZoneInfo(REPORT_CONFIG["timezone"])
    camera = region.get("camera")
    x_min, x_max = float(region["x_min"]), float(region["x_max"])
    y_min, y_max = float(region["y_min"]), float(region["y_max"])
    busy_spans = busy_spans or []

    events: list[dict] = []
    group: list = []

    def flush() -> None:
        if len(group) < minimum_samples:
            group.clear()
            return
        start, end = group[0].ts, group[-1].ts
        dwell = end - start
        confirmed = dwell >= confirm_seconds or len(group) >= confirm_samples
        events.append(
            {
                "kind": "toilet_use",
                "start": datetime.fromtimestamp(start, tz).strftime("%H:%M:%S"),
                "end": datetime.fromtimestamp(end, tz).strftime("%H:%M:%S"),
                "minutes": round(dwell / 60, 1),
                "certainty": "confirmed" if confirmed else "brief",
            }
        )
        group.clear()

    for observation in observations:  # ordered by ts by the query
        if camera is not None and observation.camera != camera:
            continue
        if observation.norm_x is None or observation.norm_y is None:
            continue
        if not (x_min <= observation.norm_x <= x_max and y_min <= observation.norm_y <= y_max):
            continue
        if any(start <= observation.ts <= end for start, end in busy_spans):
            flush()
            continue
        if group and observation.ts - group[-1].ts > gap_seconds:
            flush()
        group.append(observation)
    flush()
    return events


def build_summary(day: date_type, database: Path | None = None) -> dict:
    tz = ZoneInfo(REPORT_CONFIG["timezone"])
    start_ts, end_ts = day_bounds(day, tz)
    store = LocationStore(database or resolve_config_path(TRACKING_CONFIG["database"]))
    try:
        visits = store.visits_between(start_ts, end_ts)
        # The toilet region works on raw appearances rather than visits: an
        # enclosed box hides the cat badly enough that no visit is ever opened.
        observations = store.observations_between(start_ts, end_ts)
    finally:
        store.close()

    events_config = REPORT_CONFIG.get("events", {})
    toilet_zones = set(events_config.get("toilet_zones") or [])
    toilet_window = float(events_config.get("toilet_window_seconds", DEFAULT_TOILET_WINDOW_SECONDS))
    toilet_minimum = int(
        events_config.get("toilet_min_appearances", DEFAULT_TOILET_MIN_APPEARANCES)
    )
    toilet_region = events_config.get("toilet_region")
    toilet_region_gap = float(
        events_config.get("toilet_region_gap_seconds", DEFAULT_TOILET_REGION_GAP_SECONDS)
    )
    toilet_region_minimum = int(
        events_config.get("toilet_region_min_samples", DEFAULT_TOILET_REGION_MIN_SAMPLES)
    )
    toilet_region_confirm = float(
        events_config.get("toilet_region_confirm_seconds", DEFAULT_TOILET_REGION_CONFIRM_SECONDS)
    )
    toilet_region_confirm_samples = int(
        events_config.get("toilet_region_confirm_samples", DEFAULT_TOILET_REGION_CONFIRM_SAMPLES)
    )
    feeding_zones = set(events_config.get("feeding_zones") or [])
    water_zones = set(events_config.get("water_zones") or [])
    feeding_minimum = float(events_config.get("feeding_min_seconds", DEFAULT_FEEDING_MIN_SECONDS))
    feeding_merge = float(events_config.get("feeding_merge_seconds", DEFAULT_FEEDING_MERGE_SECONDS))

    totals: dict[str, dict[str, float]] = {}
    timeline: dict[str, list[dict]] = {}
    by_cat: dict[str, list] = {}
    observations_by_cat: dict[str, list] = {}
    for visit in visits:
        by_cat.setdefault(visit.cat, []).append(visit)
    for observation in observations:
        observations_by_cat.setdefault(observation.cat, []).append(observation)
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

    summaries = []
    for cat in sorted(set(totals) | set(by_cat) | set(observations_by_cat)):
        cat_visits = by_cat.get(cat, [])
        meals = feeding_events(cat_visits, feeding_zones, feeding_minimum, feeding_merge, "meal")
        drinks = feeding_events(cat_visits, water_zones, feeding_minimum, feeding_merge, "drinking")
        # Both rules stay: the region one is the only thing that works when the box
        # hides the cat, and the zone one is right when the cat is actually visible
        # inside it (an open box). Whichever is configured for this room wins.
        if toilet_region:
            # The bowls sit beside the box, so eating and a box appearance can
            # overlap; the appearances are still reported, just less confidently.
            busy_spans = feeding_spans(
                cat_visits, feeding_zones, feeding_minimum, feeding_merge
            ) + feeding_spans(cat_visits, water_zones, feeding_minimum, feeding_merge)
            toilet = toilet_region_events(
                observations_by_cat.get(cat, []),
                toilet_region,
                toilet_region_gap,
                toilet_region_minimum,
                toilet_region_confirm,
                toilet_region_confirm_samples,
                busy_spans,
            )
        else:
            toilet = toilet_events(cat_visits, toilet_zones, toilet_window, toilet_minimum)

        summaries.append(
            {
                "cat": cat,
                "events": [*toilet, *meals, *drinks],
                "locations": [
                    {"location": location, "minutes": round(seconds / 60, 1)}
                    for location, seconds in sorted(
                        totals.get(cat, {}).items(), key=lambda item: item[1], reverse=True
                    )
                ],
                "timeline": timeline.get(cat, []),
            }
        )

    return {
        "date": day.isoformat(),
        "timezone": REPORT_CONFIG["timezone"],
        "cats": summaries,
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


def call_llm(summary: dict, temperature: float | None = None, question: str | None = None) -> str:
    llm = REPORT_CONFIG["llm"]
    if not llm["api_key"]:
        raise RuntimeError("report.llm.api_key is not configured")
    language = REPORT_CONFIG.get("language", "zh")
    instruction = build_instruction(language, question=question)
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


def compose(
    summary: dict,
    text: str,
    temperature: float | None = None,
    question: str | None = None,
) -> str:
    """The outbound text, after any LLM rewriting for the configured mode.

    Split out of deliver() so a dry run can generate and show exactly what would
    have been posted without touching the network beyond the LLM call itself.

    ``question`` is only used by the Discord bot (and by any other caller that
    wants the narrative to answer something specific); the daily report leaves it
    as ``None``.
    """
    if REPORT_CONFIG.get("mode", "discord").lower() == "llm":
        return call_llm(summary, temperature, question)
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
