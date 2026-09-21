"""Tests for the Discord bot that answers report requests from a channel.

The Gateway handler itself is a thin shell; what is worth testing is the part that
decides. Every test here runs without a token, a network connection, or discord.py
having logged in anywhere.
"""

import asyncio
import inspect
import sys
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.monitoring.discord_bot as discord_bot
import src.monitoring.location_report as location_report
from src.monitoring.discord_bot import (
    DEFAULT_TRIGGERS,
    MISSING_TOKEN_MESSAGE,
    BotSettings,
    build_reply,
    day_offset,
    describe_settings,
    format_failure,
    matches_trigger,
    message_question,
    should_respond,
    target_day,
)
from src.monitoring.location_store import LocationStore

DAY = date(2026, 9, 21)


def settings(**overrides) -> BotSettings:
    defaults = {
        "token": "test-token",
        "allowed_channel_ids": frozenset(),
        "allowed_user_ids": frozenset(),
        "triggers": tuple(trigger.lower() for trigger in DEFAULT_TRIGGERS),
        "days_ago": 0,
        "database": None,
        "temperature": None,
    }
    return BotSettings(**{**defaults, **overrides})


def database_with_a_visit(tmp_path: Path, days_ago: int = 0, cat: str = "bagel") -> Path:
    """A throwaway database holding one 10 minute visit on the day being asked for.

    Seeded relative to ``target_day`` rather than to a fixed date, so a test that
    asks for "today" keeps working tomorrow.
    """
    tz = ZoneInfo(location_report.REPORT_CONFIG["timezone"])
    start_ts, _ = location_report.day_bounds(discord_bot.target_day(days_ago), tz)
    database = tmp_path / "history.db"
    store = LocationStore(database)
    visit_id = store.open_visit(start_ts + 60, cat, "sofa", "on_sofa", 0.9)
    store.touch_visit(visit_id, start_ts + 660, 4, 0.91)
    store.close()
    return database


class FakeTyping:
    def __init__(self) -> None:
        self.entered = False

    async def __aenter__(self) -> "FakeTyping":
        self.entered = True
        return self

    async def __aexit__(self, *exception) -> bool:
        return False


class FakeChannel:
    def __init__(self, channel_id: int = 1) -> None:
        self.id = channel_id
        self.typing_context = FakeTyping()

    def typing(self) -> FakeTyping:
        return self.typing_context

    def __str__(self) -> str:
        return "fake-channel"


class FakeMessage:
    """Just enough of ``discord.Message`` for ``handle_message`` to answer it."""

    def __init__(
        self,
        content: str,
        channel_id: int = 1,
        author_id: int = 2,
        is_bot: bool = False,
    ) -> None:
        self.content = content
        self.channel = FakeChannel(channel_id)
        self.author = SimpleNamespace(id=author_id, bot=is_bot)
        self.replies: list[str] = []

    async def reply(self, content: str) -> None:
        self.replies.append(content)


# --- which messages get answered -------------------------------------------


def test_a_configured_trigger_word_asks_for_a_report() -> None:
    for message in ("报告一下今天两只猫都做什么了", "来个日报", "report please", "Report!"):
        assert matches_trigger(message, DEFAULT_TRIGGERS), message


def test_chatter_without_a_trigger_is_ignored() -> None:
    for message in ("", "喵", "今天天气不错", "报"):
        assert not matches_trigger(message, DEFAULT_TRIGGERS), message


def test_a_trigger_inside_a_longer_word_still_counts() -> None:
    """Matching is a substring test, which is loose on purpose.

    "reported" ending in "report" is the price of "report please" working without
    a command syntax. The failure mode is one extra answer to a curious message,
    not a missed report.
    """
    assert matches_trigger("reported", DEFAULT_TRIGGERS)


def test_triggers_from_the_config_win_over_the_defaults() -> None:
    assert matches_trigger("喵喵喵", ["喵"])
    assert not matches_trigger("报告", ["喵"])


def test_bots_are_never_answered() -> None:
    """The reply quotes the question back, so answering a bot would loop forever."""
    assert not should_respond("报告", channel_id=1, author_id=2, is_bot=True, settings=settings())


def test_an_empty_allowlist_does_not_restrict_anything() -> None:
    assert should_respond("报告", channel_id=1, author_id=2, is_bot=False, settings=settings())


def test_the_channel_allowlist_is_enforced() -> None:
    configured = settings(allowed_channel_ids=frozenset({42}))
    assert should_respond("报告", channel_id=42, author_id=2, is_bot=False, settings=configured)
    assert not should_respond("报告", channel_id=43, author_id=2, is_bot=False, settings=configured)


def test_the_user_allowlist_is_enforced() -> None:
    configured = settings(allowed_user_ids=frozenset({7}))
    assert should_respond("报告", channel_id=1, author_id=7, is_bot=False, settings=configured)
    assert not should_respond("报告", channel_id=1, author_id=8, is_bot=False, settings=configured)


# --- which day is being asked about ----------------------------------------


def test_a_bare_request_means_today() -> None:
    assert day_offset("报告一下", default=0) == 0


def test_yesterday_and_the_day_before_can_be_asked_for_in_words() -> None:
    assert day_offset("昨天两只猫干嘛了") == 1
    assert day_offset("yesterday please") == 1
    assert day_offset("前天的报告") == 2
    assert day_offset("day before yesterday") == 2


def test_an_explicit_word_beats_the_configured_default() -> None:
    """A daily-report style default of 1 must not turn "昨天" into two days back."""
    assert day_offset("昨天", default=1) == 1


def test_target_day_uses_the_configured_timezone() -> None:
    # 2026-09-21 00:30 in Tokyo is still 2026-09-20 in UTC. The report is the
    # Tokyo day, so the same instant has to land on the same date whichever
    # timezone it is handed over in.
    moment = datetime(2026, 9, 21, 0, 30, tzinfo=ZoneInfo("Asia/Tokyo"))
    as_utc = moment.astimezone(ZoneInfo("UTC"))
    assert as_utc.date() == date(2026, 9, 20), "the fixture must straddle the day"

    assert target_day(0, moment) == date(2026, 9, 21)
    assert target_day(1, moment) == date(2026, 9, 20)
    # Same instant, other timezone, same answer.
    assert target_day(0, as_utc) == date(2026, 9, 21)


# --- the question that goes into the prompt --------------------------------


def test_mentions_are_stripped_from_the_question() -> None:
    """ "<@123> 报告" is not something a model reads naturally."""
    assert message_question("<@12345> 报告一下") == "报告一下"
    assert message_question("报告 <@!678> 今天") == "报告 今天"
    assert message_question("@everyone 报告") == "报告"


def test_a_long_message_is_capped_before_it_reaches_the_prompt() -> None:
    cleaned = message_question("报" * (location_report.MAX_QUESTION_CHARACTERS + 100))
    assert len(cleaned) == location_report.MAX_QUESTION_CHARACTERS


# --- building the reply ----------------------------------------------------


def test_the_reply_is_the_days_summary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(location_report.REPORT_CONFIG, "mode", "llm")
    monkeypatch.setattr(
        location_report,
        "call_llm",
        lambda summary, temperature=None, question=None: f"喵：{summary['date']}",
    )

    reply = build_reply("报告", settings=settings(), database=database_with_a_visit(tmp_path))
    assert reply == f"喵：{discord_bot.target_day(0).isoformat()}"


def test_the_question_reaches_the_model(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_call_llm(summary, temperature=None, question=None):
        captured["question"] = question
        return "喵"

    monkeypatch.setitem(location_report.REPORT_CONFIG, "mode", "llm")
    monkeypatch.setattr(location_report, "call_llm", fake_call_llm)

    build_reply(
        "<@1> kurumi 今天在哪待得最久？",
        settings=settings(),
        database=database_with_a_visit(tmp_path),
    )
    assert captured["question"] == "kurumi 今天在哪待得最久？"


def test_asking_for_yesterday_reads_yesterday(monkeypatch, tmp_path: Path) -> None:
    """The whole point of the word "昨天": the reply must not be today's numbers."""
    monkeypatch.setitem(location_report.REPORT_CONFIG, "mode", "llm")
    monkeypatch.setattr(
        location_report,
        "call_llm",
        lambda summary, temperature=None, question=None: f"喵：{summary['date']}",
    )

    reply = build_reply(
        "昨天的报告", settings=settings(), database=database_with_a_visit(tmp_path, days_ago=1)
    )
    assert reply == f"喵：{discord_bot.target_day(1).isoformat()}"
    assert reply != f"喵：{discord_bot.target_day(0).isoformat()}"


def test_a_failing_llm_still_sends_the_numbers(monkeypatch, tmp_path: Path) -> None:
    """Losing the data because the phrasing failed would be a bad trade.

    Same rule alert_voice follows: the mechanical summary is already written, so
    it is a better answer than an apology.
    """

    def explode(summary, temperature=None, question=None):
        raise location_report.DeliveryError("502 Bad Gateway: upstream is sad")

    monkeypatch.setitem(location_report.REPORT_CONFIG, "mode", "llm")
    monkeypatch.setattr(location_report, "call_llm", explode)

    reply = build_reply("报告", settings=settings(), database=database_with_a_visit(tmp_path))
    assert "on_sofa" in reply
    assert "502" not in reply


def test_an_unreadable_day_gets_an_apology_not_a_crash(monkeypatch) -> None:
    def explode(day, database=None):
        raise RuntimeError("unable to open database file")

    monkeypatch.setattr(discord_bot, "build_summary", explode)
    reply = build_reply("报告", settings=settings(), days_ago=0)

    assert DAY.isoformat() in reply
    assert "unable to open database file" in reply


def test_a_reply_is_never_longer_than_discord_accepts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(location_report.REPORT_CONFIG, "mode", "llm")
    monkeypatch.setattr(
        location_report,
        "call_llm",
        lambda summary, temperature=None, question=None: "长" * 5000,
    )

    reply = build_reply("报告", settings=settings(), database=database_with_a_visit(tmp_path))
    assert len(reply) == location_report.DISCORD_MESSAGE_LIMIT


def test_format_failure_names_the_day_and_the_reason() -> None:
    text = format_failure(DAY, RuntimeError("boom"))
    assert DAY.isoformat() in text
    assert "boom" in text


# --- the Gateway handler ---------------------------------------------------
#
# handle_message takes anything that quacks like a discord.Message, which is what
# keeps this I/O shell testable without a token or a connection.


def test_handle_message_replies_with_the_report(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(location_report.REPORT_CONFIG, "mode", "discord")
    message = FakeMessage("报告一下今天两只猫都做什么了")

    asyncio.run(
        discord_bot.handle_message(
            message, settings=settings(database=database_with_a_visit(tmp_path))
        )
    )

    assert len(message.replies) == 1
    assert "on_sofa" in message.replies[0]
    assert message.channel.typing_context.entered, "the owner should see it typing"


def test_handle_message_stays_silent_for_other_chatter(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(location_report.REPORT_CONFIG, "mode", "discord")
    monkeypatch.setattr(
        discord_bot,
        "build_reply",
        lambda *args, **kwargs: pytest.fail("chatter must not reach the report builder"),
    )
    message = FakeMessage("今天天气不错")

    asyncio.run(discord_bot.handle_message(message, settings=settings()))
    assert message.replies == []


def test_handle_message_ignores_other_bots(monkeypatch) -> None:
    monkeypatch.setattr(
        discord_bot,
        "build_reply",
        lambda *args, **kwargs: pytest.fail("a bot must not be answered"),
    )
    message = FakeMessage("报告", is_bot=True)

    asyncio.run(discord_bot.handle_message(message, settings=settings()))
    assert message.replies == []


def test_handle_message_honours_the_channel_allowlist(monkeypatch) -> None:
    monkeypatch.setattr(
        discord_bot,
        "build_reply",
        lambda *args, **kwargs: pytest.fail("another channel must not trigger a report"),
    )
    message = FakeMessage("报告", channel_id=99)

    asyncio.run(
        discord_bot.handle_message(message, settings=settings(allowed_channel_ids=frozenset({42})))
    )
    assert message.replies == []


def test_the_client_asks_for_message_content_and_registers_the_handler(
    monkeypatch, tmp_path: Path
) -> None:
    """The wiring itself, not just the rules.

    Message Content is a privileged intent: if it is not requested the Gateway
    still connects and every ``message.content`` is empty, which looks exactly
    like a bot whose trigger words are wrong.
    """
    discord = pytest.importorskip("discord")
    monkeypatch.setitem(location_report.REPORT_CONFIG, "mode", "discord")

    client = discord_bot.build_client(settings(database=database_with_a_visit(tmp_path)))
    assert client.intents.message_content is True
    # discord.py 2.x exposes @client.event handlers as attributes.
    assert inspect.iscoroutinefunction(client.on_message)
    assert inspect.iscoroutinefunction(client.on_ready)

    message = FakeMessage("报告")
    asyncio.run(client.on_message(message))
    assert len(message.replies) == 1
    # Never actually log in: a fake token would hit Discord's API.
    assert isinstance(client, discord.Client)


# --- configuration ---------------------------------------------------------


def test_the_allowlists_accept_a_list_one_id_or_a_comma_string() -> None:
    """The string form is what lets the ids come from the environment."""
    assert discord_bot._numeric_ids([1, "2"], "x") == frozenset({1, 2})
    assert discord_bot._numeric_ids(3, "x") == frozenset({3})
    assert discord_bot._numeric_ids("4, 5，6", "x") == frozenset({4, 5, 6})
    assert discord_bot._numeric_ids(None, "x") == frozenset()


def test_a_non_numeric_allowlist_entry_is_reported() -> None:
    with pytest.raises(ValueError, match="allowed_channel_ids"):
        discord_bot._numeric_ids(["not-an-id"], "allowed_channel_ids")


def test_describe_settings_leaks_no_secret() -> None:
    text = describe_settings(settings(token="super-secret-token"))
    assert "super-secret-token" not in text
    assert "报告" in text


def test_check_only_refuses_to_start_without_a_token(monkeypatch) -> None:
    """install_services.sh relies on this to not install a failing unit."""
    monkeypatch.setattr(discord_bot, "bot_settings", lambda: settings(token=""))

    with pytest.raises(SystemExit) as error:
        discord_bot.main(["--check-only"])
    assert MISSING_TOKEN_MESSAGE in str(error.value)


def test_check_only_passes_with_a_token(monkeypatch, capsys) -> None:
    monkeypatch.setattr(discord_bot, "bot_settings", lambda: settings())
    monkeypatch.setattr(
        discord_bot, "run", lambda settings: pytest.fail("--check-only must not connect")
    )

    discord_bot.main(["--check-only"])
    assert "usable" in capsys.readouterr().out


def test_check_only_warns_when_every_channel_is_allowed(monkeypatch, capsys) -> None:
    monkeypatch.setattr(discord_bot, "bot_settings", lambda: settings())
    discord_bot.main(["--check-only"])
    assert "allowed_channel_ids" in capsys.readouterr().out


def test_the_bot_refuses_to_start_without_a_token(monkeypatch) -> None:
    monkeypatch.setattr(discord_bot, "bot_settings", lambda: settings(token=""))
    with pytest.raises(SystemExit) as error:
        discord_bot.main([])
    assert MISSING_TOKEN_MESSAGE in str(error.value)


def test_bot_settings_reads_the_configured_block(monkeypatch) -> None:
    monkeypatch.setitem(
        discord_bot.DISCORD_BOT_CONFIG,
        "triggers",
        ["喵"],
    )
    monkeypatch.setitem(discord_bot.DISCORD_BOT_CONFIG, "allowed_channel_ids", "11, 12")
    monkeypatch.setitem(discord_bot.DISCORD_BOT_CONFIG, "days_ago", 1)

    configured = discord_bot.bot_settings()
    assert configured.triggers == ("喵",)
    assert configured.allowed_channel_ids == frozenset({11, 12})
    assert configured.days_ago == 1


def test_bot_settings_defaults_to_the_project_triggers(monkeypatch) -> None:
    """A missing block must not leave the bot deaf with no explanation."""
    monkeypatch.setattr(discord_bot, "DISCORD_BOT_CONFIG", {})
    configured = discord_bot.bot_settings()
    assert configured.triggers == tuple(trigger.lower() for trigger in DEFAULT_TRIGGERS)
    assert configured.days_ago == 0
