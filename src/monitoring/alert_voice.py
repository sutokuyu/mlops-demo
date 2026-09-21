"""Say an operational alert the way the bot would say it, not the way a log would.

The tracker's Discord messages used to be raw status lines ("Camera `sofa` could not
be re-anchored automatically. Reason: only 10 feature matches"). They are correct
and completely charmless. This module hands the facts to the LLM configured under
``report.llm`` and asks for one or two sentences in the bot's persona.

Everything here is best-effort. ``phrase_alert`` returns the caller's own plain
sentence whenever the model is disabled, unreachable, slow, or answers with
something unusable - losing a warning because a nicer wording failed would be a bad
trade every time. The technical facts stay in the console log as well, so nothing is
dropped by phrasing it warmly.
"""

import json
import re
import urllib.error
import urllib.request

from src.monitoring.location_config import LOCATION_CONFIG

# Cloudflare sits in front of Discord, and some LLM gateways block the default
# urllib agent outright, so always send a real one.
USER_AGENT = "mlops-cat-demo/1.0"

# Deliberately shorter than the daily report's timeout: this call happens inside the
# sampling loop, so a slow gateway delays every camera, not just this alert.
REQUEST_TIMEOUT_SECONDS = 15
MAX_ALERT_CHARACTERS = 240
DEFAULT_TEMPERATURE = 0.7
MAX_TEMPERATURE = 2.0
# A reply longer than this is not a Discord alert any more, it is a report. Fall back
# rather than post a wall of text.
RIDICULOUS_LENGTH = MAX_ALERT_CHARACTERS * 2

INSTRUCTION = """\
你是猫位置监控机器人的拟人化形象「大肥鱼」（鲸鱼娘），自称「本鱼」。
现在要把一条机械的系统告警，改写成对主人说的话，发到 Discord。

必须遵守：
- 一到两句中文，最多 {limit} 字。不要标题、列表、代码块、Markdown 标记。
- 说清三件事：哪台摄像头、发生了什么、主人需不需要动手。
- 只许使用「事实」里给出的内容。不要补充时间、地点、原因，不要编造任何数字。
- 摄像头标识（sofa / living_room / feeder）自然地译成中文（沙发 / 客厅 / 喂食器），
  不要照抄英文。
- 语气傲娇、慵懒、爱吐槽，但正事要先说完再撒娇，不许因为卖萌漏掉信息。

下面是风格示范，事实以实际输入为准：
- 重锚失败：「客厅那台摄像头歪了，本鱼想自己摆正，可是画面里对得上的点太少……
  你有空看一眼呗？」
- 重锚成功：「沙发那台不知道被谁碰歪了，本鱼已经帮你重新摆正，区域也一起挪好了。」
"""


def alert_voice_settings() -> dict:
    """The LLM block plus its ``alerts`` sub-block, read on every call.

    Read at call time rather than at import: locations.yaml substitutes values from
    the environment, and a service that runs for weeks should not have to be
    restarted to notice a new key - nor silently keep using one that was missing
    when it started.
    """
    llm = LOCATION_CONFIG.get("report", {}).get("llm", {})
    alerts = llm.get("alerts", {})
    return {
        "enabled": bool(alerts.get("enabled", True)) and bool(llm.get("api_key")),
        "endpoint": llm.get("endpoint", ""),
        "model": llm.get("model", ""),
        "api_key": llm.get("api_key", ""),
        "temperature": _temperature(llm.get("temperature", DEFAULT_TEMPERATURE)),
        # The alert shares the report's persona - it is the same bot talking - but
        # may override it, and always gets its own style rules.
        "persona": alerts.get("persona") or llm.get("persona", ""),
        "style": alerts.get("style", ""),
    }


def _temperature(raw) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_TEMPERATURE
    return min(max(value, 0.0), MAX_TEMPERATURE)


def _clean(text: str) -> str:
    """Strip whitespace and a wrapping code fence, which models like to add."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _chat(settings: dict, system: str, facts: dict) -> str:
    payload = {
        "model": settings["model"],
        "temperature": settings["temperature"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(facts, ensure_ascii=False)},
        ],
    }
    request = urllib.request.Request(
        settings["endpoint"],
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings['api_key']}",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def phrase_alert(event: str, facts: dict, fallback: str) -> str:
    """Phrase ``facts`` for Discord, or return ``fallback`` if that is not possible.

    ``fallback`` is the caller's own mechanical sentence. The caller can therefore
    ignore the difference between the two: something sensible is always returned.
    """
    settings = alert_voice_settings()
    if not settings["enabled"]:
        return fallback
    system = "\n\n".join(
        part
        for part in (
            INSTRUCTION.format(limit=MAX_ALERT_CHARACTERS),
            settings["persona"],
            settings["style"],
        )
        if part
    )
    try:
        text = _clean(_chat(settings, system, {"事件": event, "事实": facts}))
    except (
        OSError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        json.JSONDecodeError,
    ):
        # OSError covers HTTPError, the timeout and a DNS failure. Whatever it was,
        # the warning still has to go out.
        return fallback
    if not text or len(text) > RIDICULOUS_LENGTH:
        return fallback
    return text
