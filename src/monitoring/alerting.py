"""The one way an operational alert reaches Discord.

Extracted from ``recalibration`` when the camera-discovery alerts needed the same
plumbing. The rules are not specific to re-anchoring:

* ``notifications_enabled`` is checked *before* phrasing, because asking the LLM to
  word a message that is going to be dropped costs a network round trip inside the
  tracker's sampling loop, where it delays every camera.
* ``alert_content`` is best-effort: an unavailable model returns the caller's own
  Chinese sentence rather than losing the warning.
* ``notify`` reports whether the message actually went out, so the caller can stop
  keeping a local copy of an image Discord now holds.

The security rules live here too: Discord sits behind Cloudflare, which rejects
requests without a User-Agent (``error code: 1010``), and the payload needs an
explicit ``username`` or the webhook posts under whatever name it was created with.
"""

from pathlib import Path

from src.monitoring.alert_voice import phrase_alert
from src.notification.notification_controller import post_discord_message


def notifications_enabled(settings: dict) -> bool:
    """False when there is nowhere to send to, or the caller asked for silence."""
    return bool(settings.get("discord_webhook")) and settings.get("notify") is not False


def alert_content(event: str, settings: dict, facts: dict, fallback: str) -> str:
    """Phrase an alert, but only when it is actually going to be sent."""
    if not notifications_enabled(settings):
        return fallback
    return phrase_alert(event, facts, fallback)


def notify(settings: dict, camera: str, content: str, attachment: Path | None) -> tuple[bool, str]:
    """Send a Discord message and report whether it actually went out.

    The caller uses the flag to decide whether an attached image is still needed: a
    copy that Discord already holds is duplicated storage that piles up.
    """
    webhook_url = settings.get("discord_webhook") or ""
    if not notifications_enabled(settings):
        return False, "notification skipped (no webhook configured)"
    try:
        post_discord_message(
            {"content": content, "username": settings.get("discord_username") or camera},
            [attachment] if attachment is not None else None,
            webhook_url=webhook_url,
        )
    except (OSError, RuntimeError, ValueError) as error:
        return False, f"notification failed: {error}"
    return True, "sent to Discord"
