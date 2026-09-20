"""Post messages to a Discord webhook, optionally with file attachments.

Only ``post_discord_message`` survived the move from the toilet monitor to the
location tracker. The mail path, the canned "{cat_name} used the toilet"
templates and the ``send_notification`` dispatcher were reachable only from the
deleted ``src/monitoring/toilet_monitor.py``; callers now build their own
payload, which is what ``recalibration`` does.
"""

import json
import mimetypes
import sys
import urllib.request
import uuid
from pathlib import Path


def _resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not locate project root")


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config

CONFIG = load_config(PROJECT_ROOT / "configs" / "config.yaml")
# Fallback used when a caller does not pass its own webhook.
DISCORD_WEBHOOK_URL = CONFIG.get("notification", {}).get("discord", {}).get("webhook_url", "")


def post_discord_message(
    payload: dict,
    attachment_paths: list[Path] | None = None,
    webhook_url: str = "",
) -> None:
    """Send a Discord webhook message, optionally with file attachments."""
    endpoint = webhook_url or DISCORD_WEBHOOK_URL
    if not endpoint:
        raise RuntimeError("Discord webhook URL is not configured.")

    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"

    # Discord expects payload_json plus distinct file fields (files[0], files[1], ...).
    body = (
        f"--{boundary}"
        "\r\n"
        'Content-Disposition: form-data; name="payload_json"'
        "\r\n\r\n"
        f"{json.dumps(payload)}"
        "\r\n"
    ).encode()

    for index, attachment_path in enumerate(attachment_paths or []):
        with open(attachment_path, "rb") as handle:
            file_data = handle.read()

        mimetype, _ = mimetypes.guess_type(attachment_path.name)
        if mimetype is None:
            mimetype = "application/octet-stream"

        body += (
            f"--{boundary}"
            "\r\n"
            f'Content-Disposition: form-data; name="files[{index}]"; filename="{attachment_path.name}"'
            "\r\n"
            f"Content-Type: {mimetype}"
            "\r\n\r\n"
        ).encode()
        body += file_data + b"\r\n"

    body += f"--{boundary}--\r\n".encode()

    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "mlops-cat-demo/1.0",
        },
    )

    with urllib.request.urlopen(request) as response:
        status = response.getcode()
        if status >= 400:
            raise RuntimeError(f"Discord webhook failed with status {status}")
