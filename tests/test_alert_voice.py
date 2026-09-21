"""Tests for the layer that turns a mechanical alert into something the bot says."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring import alert_voice, recalibration

PLAIN = "⚠️ `sofa` 自动重新锚定没成功。"


def voice_settings(enabled: bool = True, **overrides) -> dict:
    settings = {
        "enabled": enabled,
        "endpoint": "https://example.invalid/chat/completions",
        "model": "test-model",
        "api_key": "test-key" if enabled else "",
        "temperature": 0.7,
        "persona": "你是测试机器人。",
        "style": "只说一句话。",
    }
    settings.update(overrides)
    return settings


def refuse(*args, **kwargs):
    raise AssertionError("the model must not be called")


def test_a_disabled_voice_keeps_the_plain_wording(monkeypatch) -> None:
    monkeypatch.setattr(alert_voice, "alert_voice_settings", lambda: voice_settings(False))
    monkeypatch.setattr(alert_voice, "_chat", refuse)

    assert alert_voice.phrase_alert("reanchor_failed", {"a": 1}, PLAIN) == PLAIN


def test_the_model_wording_is_used_when_the_model_answers(monkeypatch) -> None:
    monkeypatch.setattr(alert_voice, "alert_voice_settings", lambda: voice_settings())
    seen = {}

    def fake_chat(settings, system, facts):
        seen["system"] = system
        seen["facts"] = facts
        return "客厅那台歪啦，本鱼正在想办法。"

    monkeypatch.setattr(alert_voice, "_chat", fake_chat)

    phrased = alert_voice.phrase_alert("reanchor_failed", {"摄像头": "living_room"}, PLAIN)

    assert phrased == "客厅那台歪啦，本鱼正在想办法。"
    assert seen["facts"] == {"事件": "reanchor_failed", "事实": {"摄像头": "living_room"}}
    # The rules and the persona both have to reach the model, or the alert stops
    # sounding like the bot (and starts inventing facts).
    assert "一到两句" in seen["system"]
    assert "测试机器人" in seen["system"]
    assert "只说一句话" in seen["system"]


@pytest.mark.parametrize("mode", ["unreachable", "garbage", "empty", "rambling"])
def test_nothing_may_lose_the_warning(monkeypatch, mode: str) -> None:
    """Every way the nicer wording can fail has to fall back, not disappear."""
    monkeypatch.setattr(alert_voice, "alert_voice_settings", lambda: voice_settings())

    def fake_chat(settings, system, facts):
        if mode == "unreachable":
            raise OSError("connection refused")
        if mode == "garbage":
            raise ValueError("no choices in the response")
        if mode == "empty":
            return "   "
        return "本鱼" * 5000

    monkeypatch.setattr(alert_voice, "_chat", fake_chat)

    assert alert_voice.phrase_alert("reanchor_ok", {}, PLAIN) == PLAIN


def test_a_wrapping_code_fence_is_stripped(monkeypatch) -> None:
    monkeypatch.setattr(alert_voice, "alert_voice_settings", lambda: voice_settings())
    monkeypatch.setattr(alert_voice, "_chat", lambda *args: "```\n沙发那台歪啦\n```")

    assert alert_voice.phrase_alert("reanchor_failed", {}, PLAIN) == "沙发那台歪啦"


def test_an_unusable_temperature_falls_back_to_the_default() -> None:
    assert alert_voice._temperature("0.7") == pytest.approx(0.7)
    assert alert_voice._temperature(9.0) == alert_voice.MAX_TEMPERATURE
    assert alert_voice._temperature("hot") == alert_voice.DEFAULT_TEMPERATURE


def test_a_machine_reason_is_glossed_with_its_number_kept() -> None:
    assert "10" in recalibration.describe_reason("only 10 feature matches")
    assert "特征" in recalibration.describe_reason("current frame has too few features")
    # An unknown reason is passed through rather than guessed at.
    assert recalibration.describe_reason("nobody planned for this") == "nobody planned for this"
