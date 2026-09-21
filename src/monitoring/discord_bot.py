"""Answer questions posted in a Discord channel, e.g. "报告一下今天两只猫都做什么了".

The project already talks to Discord through a webhook, but a webhook is one-way:
it can send, never read. To hear the owner, the bot has to be a real Discord
application holding a Gateway connection. That is also the option this machine can
actually run: the bot dials *out* over a WebSocket, so no inbound port, no public
URL and no tunnel is needed behind NAT/WSL. The alternative (slash commands over
the Interactions HTTP endpoint) needs a public HTTPS URL and an answer within three
seconds, neither of which this box offers.

Message Content is a privileged intent and has to be enabled in the Discord
Developer Portal. Without it the Gateway refuses the connection outright
(``PrivilegedIntentsRequired``), which is why the preflight checks the
application's own flags before systemd is allowed to start anything.

Everything except ``build_client``/``run`` is deliberately free of any ``discord``
import. The decision rules (which messages to answer, which day they mean, what to
reply) are the part worth testing, and they must stay testable - and
preflightable - without a token, a connection, or the library installed. The
Gateway shell is duck-typed for the same reason: :func:`handle_message` only needs
an object that quacks like ``discord.Message``.
"""

import argparse
import asyncio
import json
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from src.config_loader import resolve_config_path
from src.monitoring.location_config import DISCORD_BOT_CONFIG, REPORT_CONFIG
from src.monitoring.location_report import (
    DISCORD_MESSAGE_LIMIT,
    MAX_QUESTION_CHARACTERS,
    build_summary,
    compose,
    render_text,
)

DEFAULT_TRIGGERS = ("报告", "日报", "report")
DEFAULT_DAYS_AGO = 0

# Asking about another day is a phrase, not a setting, so it is matched rather
# than parsed: "昨天两只猫在干嘛" should not need a syntax.
DAY_BEFORE_YESTERDAY_WORDS = ("前天", "day before yesterday")
YESTERDAY_WORDS = ("昨天", "昨日", "yesterday")

# Mentions are noise inside a prompt, and <@123> is not something an LLM reads
# naturally.
MENTION_PATTERN = re.compile(r"<@[!&]?\d+>|@everyone|@here")

# The same set location_report.main() catches. DeliveryError is a RuntimeError, so
# both the report builder and every HTTP path are covered.
REPORT_ERRORS = (urllib.error.URLError, RuntimeError, KeyError, IndexError, OSError)

MISSING_TOKEN_MESSAGE = (
    "discord_bot.token is empty, so the bot cannot log in.\n"
    "Put the bot token in .env as DISCORD_BOT_TOKEN=... and check that Message\n"
    "Content Intent is enabled for the application in the Discord Developer Portal\n"
    "(Bot -> Privileged Gateway Intents). Without that intent the bot connects but\n"
    "sees every message as empty text."
)

# Discord sits behind Cloudflare, which answers urllib's default agent with
# "403 error code: 1010". Any explicit agent works.
USER_AGENT = "mlops-cat-demo/1.0"
DISCORD_API = "https://discord.com/api/v10"
REQUEST_TIMEOUT_SECONDS = 30

# Application flags that answer "is the Message Content intent switched on".
# The *_LIMITED variant means it is on but the application has reached 100
# servers, where content stops being delivered - irrelevant for one home server,
# and worth telling apart from "off" so the advice is not misleading.
MESSAGE_CONTENT_FLAG = 1 << 18
MESSAGE_CONTENT_LIMITED_FLAG = 1 << 19

INTENT_OFF_MESSAGE = (
    "Message Content Intent is NOT enabled for this application.\n"
    "Discord will refuse the Gateway connection (PrivilegedIntentsRequired), so\n"
    "the service would only restart forever. To fix it:\n"
    "  Developer Portal -> your application -> Bot -> Privileged Gateway Intents\n"
    "  -> enable 'Message Content Intent' -> Save Changes."
)


class DiscordCheckError(RuntimeError):
    """The online preflight found something Discord would reject at runtime."""


@dataclass(frozen=True)
class BotSettings:
    """Everything the bot needs, resolved from locations.yaml + the environment."""

    token: str
    allowed_channel_ids: frozenset[int]
    allowed_user_ids: frozenset[int]
    triggers: tuple[str, ...]
    days_ago: int
    database: Path | None
    temperature: float | None


def _numeric_ids(raw, field: str) -> frozenset[int]:
    """Read an allowlist, accepting a list, one id, or a comma-separated string.

    The string form exists so the list can come from the environment
    (``${DISCORD_BOT_CHANNEL_IDS:}``), which is where the rest of this project's
    deployment secrets live.
    """
    if raw in (None, ""):
        return frozenset()
    if isinstance(raw, (int, float, str)):
        raw = [raw]
    ids: set[int] = set()
    for item in raw:
        for part in str(item).replace("，", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                ids.add(int(part))
            except ValueError as error:
                raise ValueError(
                    f"discord_bot.{field} must contain numeric Discord IDs, got {item!r}"
                ) from error
    return frozenset(ids)


def _triggers(raw) -> tuple[str, ...]:
    if raw in (None, "", []):
        return tuple(trigger.lower() for trigger in DEFAULT_TRIGGERS)
    if isinstance(raw, str):
        raw = [raw]
    return tuple(str(item).strip().lower() for item in raw if str(item).strip())


def bot_settings() -> BotSettings:
    """Read the ``discord_bot`` block. Raises on a malformed allowlist."""
    database = DISCORD_BOT_CONFIG.get("database")
    temperature = DISCORD_BOT_CONFIG.get("temperature")
    return BotSettings(
        token=str(DISCORD_BOT_CONFIG.get("token") or "").strip(),
        allowed_channel_ids=_numeric_ids(
            DISCORD_BOT_CONFIG.get("allowed_channel_ids"), "allowed_channel_ids"
        ),
        allowed_user_ids=_numeric_ids(
            DISCORD_BOT_CONFIG.get("allowed_user_ids"), "allowed_user_ids"
        ),
        triggers=_triggers(DISCORD_BOT_CONFIG.get("triggers")),
        days_ago=int(DISCORD_BOT_CONFIG.get("days_ago", DEFAULT_DAYS_AGO)),
        database=resolve_config_path(database) if database else None,
        temperature=float(temperature) if temperature is not None else None,
    )


def matches_trigger(content: str, triggers: Sequence[str]) -> bool:
    """A trigger is a substring, matched case-insensitively ("Report" == "report")."""
    lowered = content.lower()
    return any(trigger.lower() in lowered for trigger in triggers)


def day_offset(content: str, default: int = DEFAULT_DAYS_AGO) -> int:
    """Which day the message means; "昨天" asks for the previous one."""
    lowered = content.lower()
    if any(word in lowered for word in DAY_BEFORE_YESTERDAY_WORDS):
        return 2
    if any(word in lowered for word in YESTERDAY_WORDS):
        return 1
    return default


def target_day(days_ago: int, now: datetime | None = None) -> date:
    """The calendar day N days back in the configured timezone.

    ``now`` is injectable so a test does not have to depend on when it runs.
    """
    tz = ZoneInfo(REPORT_CONFIG.get("timezone", "Asia/Tokyo"))
    moment = now.astimezone(tz) if now is not None else datetime.now(tz)
    return (moment - timedelta(days=days_ago)).date()


def message_question(content: str) -> str:
    """The owner's own words, cleaned up for the prompt.

    Mentions are dropped and the length is capped: this text goes into the system
    prompt, and a long message would push the grounding rules out of the model's
    attention (see MAX_QUESTION_CHARACTERS).
    """
    without_mentions = MENTION_PATTERN.sub(" ", content)
    return " ".join(without_mentions.split())[:MAX_QUESTION_CHARACTERS]


def should_respond(
    content: str,
    *,
    channel_id: int,
    author_id: int,
    is_bot: bool,
    settings: BotSettings,
) -> bool:
    """Whether this message is a report request the bot should answer.

    A bot author is always ignored, which includes the bot's own replies - it does
    quote the question back, and answering that would be an infinite loop. An empty
    allowlist does not restrict anything; it is the trigger word that decides.
    """
    if is_bot:
        return False
    if settings.allowed_channel_ids and channel_id not in settings.allowed_channel_ids:
        return False
    if settings.allowed_user_ids and author_id not in settings.allowed_user_ids:
        return False
    return matches_trigger(content, settings.triggers)


def format_failure(day: date, error: Exception) -> str:
    """What to say when the data itself could not be read."""
    return (
        f"{day.isoformat()} 的记录本鱼没翻出来：{error}\n"
        "数据没丢，细节在 journalctl --user -u cat-discord 里。"
    )


def build_reply(
    content: str,
    *,
    settings: BotSettings | None = None,
    days_ago: int | None = None,
    database: Path | None = None,
    temperature: float | None = None,
) -> str:
    """The message to send back. Always returns a string, never raises.

    Two separate fallbacks, because the two failures lose different things:

    * the day's data cannot be read -> an apology with the day in it. There is
      nothing truthful to send instead.
    * only the LLM rewrite fails -> the plain ``render_text`` summary. Losing the
      mechanical numbers because the phrasing failed would be a bad trade, the
      same rule ``alert_voice`` follows for its alerts.
    """
    settings = settings or bot_settings()
    offset = day_offset(content, settings.days_ago) if days_ago is None else days_ago
    day = target_day(offset)
    question = message_question(content)

    try:
        summary = build_summary(day, database if database is not None else settings.database)
    except REPORT_ERRORS as error:
        print(f"discord_bot: could not read {day.isoformat()}: {error}", file=sys.stderr)
        return format_failure(day, error)

    text = render_text(summary)
    try:
        narrative = compose(
            summary,
            text,
            temperature if temperature is not None else settings.temperature,
            question,
        )
    except REPORT_ERRORS as error:
        print(
            f"discord_bot: the LLM rewrite failed ({error}); sending the plain summary",
            file=sys.stderr,
        )
        return text[:DISCORD_MESSAGE_LIMIT]
    return narrative[:DISCORD_MESSAGE_LIMIT]


def describe_settings(settings: BotSettings) -> str:
    """A one-line summary for the log, with no secret in it."""
    channels = ", ".join(str(item) for item in sorted(settings.allowed_channel_ids)) or "any"
    users = ", ".join(str(item) for item in sorted(settings.allowed_user_ids)) or "any"
    return (
        f"triggers={settings.triggers} default_day=days_ago:{settings.days_ago} "
        f"channels={channels} users={users}"
    )


async def handle_message(message, settings: BotSettings) -> None:
    """Answer one Gateway message. Deliberately free of any ``discord`` import.

    ``message`` only has to quack: ``content``, ``channel``, ``author`` and an
    async ``reply()``. Keeping it duck-typed means the I/O shell - the part that
    decides whether the blocking report runs off the event loop - is testable
    without a token or a connection.
    """
    if not should_respond(
        message.content,
        channel_id=message.channel.id,
        author_id=message.author.id,
        is_bot=bool(message.author.bot),
        settings=settings,
    ):
        return
    print(
        f"discord_bot: report requested by {message.author} in #{message.channel}",
        flush=True,
    )
    # build_summary and the LLM call are blocking (sqlite + urllib). Running them
    # on the event loop would stall the Gateway heartbeat and every other channel
    # while the model thinks, so they go to a thread.
    async with message.channel.typing():
        reply = await asyncio.to_thread(build_reply, message.content, settings=settings)
    await message.reply(reply)


def build_client(settings: BotSettings):
    """Create the Discord client and register its handlers; connects nothing.

    The ``discord`` import lives here rather than at module level so the pure
    helpers above stay usable on a fresh clone whose dependencies are not
    installed yet - ``--check-only`` has to work before the library is there.
    """
    import discord

    # Message Content is privileged: it must also be enabled in the Developer
    # Portal, or every message arrives with empty content.
    intents = discord.Intents.default()
    intents.message_content = True

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready() -> None:
        guilds = ", ".join(guild.name for guild in client.guilds) or "no server yet"
        print(f"discord_bot: logged in as {client.user} ({guilds})", flush=True)
        print(f"discord_bot: {describe_settings(settings)}", flush=True)

    @client.event
    async def on_message(message) -> None:
        await handle_message(message, settings)

    return client


def run(settings: BotSettings) -> None:
    """Hold the Gateway connection and answer messages until interrupted."""
    if not settings.token:
        raise SystemExit(MISSING_TOKEN_MESSAGE)
    build_client(settings).run(settings.token)


def _api_get(path: str, token: str) -> dict:
    """GET a Discord REST endpoint with a bot token.

    The Authorization scheme for bots is ``Bot <token>``, not ``Bearer`` - a
    bearer token is rejected with 401 and looks like an invalid token.
    """
    request = urllib.request.Request(
        DISCORD_API + path,
        headers={"User-Agent": USER_AGENT, "Authorization": f"Bot {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:200].strip()
        if error.code == 401:
            raise DiscordCheckError(
                "Discord rejected the token (401 Unauthorized). It was probably\n"
                "reset in the Developer Portal after it was copied into .env;\n"
                f"copy the current one again. Discord said: {detail}"
            ) from error
        raise DiscordCheckError(f"Discord returned {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise DiscordCheckError(f"could not reach Discord: {error.reason}") from error


def verify_discord_access(settings: BotSettings) -> list[str]:
    """Check the things only Discord can answer. Raises DiscordCheckError.

    Both of these are invisible offline and both were seen for real: a token that
    was reset after it was copied into .env, and a Message Content intent that was
    never switched on. Either one turns into a restart loop under systemd, so the
    preflight asks before the unit is allowed to start.

    The intent is readable without connecting: the application object carries the
    ``GATEWAY_MESSAGE_CONTENT`` flag, which is set by the portal switch itself.
    """
    application = _api_get("/applications/@me", settings.token)
    flags = int(application.get("flags") or 0)
    name = application.get("name", "?")

    if flags & MESSAGE_CONTENT_LIMITED_FLAG:
        intent_note = "Message Content Intent: enabled (limited to 100 servers)"
    elif flags & MESSAGE_CONTENT_FLAG:
        intent_note = "Message Content Intent: enabled"
    else:
        raise DiscordCheckError(INTENT_OFF_MESSAGE)

    return [f"Discord accepted the token (application: {name})", intent_note]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Answer cat-report questions posted in a Discord channel.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the configuration and exit; used before letting systemd "
        "start the service, so a broken setup is not installed as a restart loop.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="With --check-only, skip the two checks that need Discord (is the "
        "token still valid, is Message Content Intent on).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    settings = bot_settings()
    if args.check_only:
        if not settings.token:
            raise SystemExit(MISSING_TOKEN_MESSAGE)
        if not settings.allowed_channel_ids:
            print(
                "Warning: discord_bot.allowed_channel_ids is empty, so the bot will "
                "answer in every channel it can read."
            )
        # Offline checks are done; the remaining two answers live on Discord's
        # side and are exactly the ones that produce a restart loop.
        if not args.offline:
            try:
                for note in verify_discord_access(settings):
                    print(note)
            except DiscordCheckError as error:
                raise SystemExit(str(error)) from error
        print(f"discord_bot: configuration looks usable ({describe_settings(settings)}).")
        return
    if not settings.allowed_channel_ids:
        print(
            "discord_bot: allowed_channel_ids is empty; answering in every channel "
            "the bot can read."
        )
    run(settings)


if __name__ == "__main__":  # pragma: no cover - thin wrapper
    main()
