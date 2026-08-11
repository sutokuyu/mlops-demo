import json
import mimetypes
import smtplib
import sys
import uuid
import urllib.request
from email.message import EmailMessage
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

from src.config_loader import load_config, resolve_config_path

CONFIG = load_config(PROJECT_ROOT / "configs" / "config.yaml")
NOTIFICATION_CONFIG = CONFIG.get("notification", {})

MODE = NOTIFICATION_CONFIG.get("mode", "mail").lower()

MAIL_CONFIG = NOTIFICATION_CONFIG.get("mail", {})
RECIPIENTS = MAIL_CONFIG.get("recipients", [])
SMTP_SERVER = MAIL_CONFIG.get("smtp_server", "")
SMTP_PORT = MAIL_CONFIG.get("smtp_port", 587)
SMTP_USERNAME = MAIL_CONFIG.get("smtp_username", "")
SMTP_PASSWORD = MAIL_CONFIG.get("smtp_password", "")
FROM_EMAIL = MAIL_CONFIG.get("from_email", SMTP_USERNAME)
USE_TLS = MAIL_CONFIG.get("use_tls", True)
USE_SSL = MAIL_CONFIG.get("use_ssl", False)
SUBJECT_TEMPLATE = MAIL_CONFIG.get(
    "subject_template",
    "{cat_name} used the toilet"
)
BODY_TEMPLATE = MAIL_CONFIG.get(
    "body_template",
    "A cat used the toilet. See attached image."
)

DISCORD_CONFIG = NOTIFICATION_CONFIG.get("discord", {})
DISCORD_WEBHOOK_URL = DISCORD_CONFIG.get("webhook_url", "")
DISCORD_USERNAME = DISCORD_CONFIG.get("username", "Toilet Monitor")
DISCORD_AVATAR_URL = DISCORD_CONFIG.get("avatar_url", "")
DISCORD_CONTENT_TEMPLATE = DISCORD_CONFIG.get(
    "content_template",
    "{cat_name} used the toilet"
)


def _build_message(cat_name: str, attachment_paths: list[Path]) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = SUBJECT_TEMPLATE.format(cat_name=cat_name)
    message["From"] = FROM_EMAIL or SMTP_USERNAME or "no-reply@example.com"
    message["To"] = ", ".join(RECIPIENTS)
    message.set_content(BODY_TEMPLATE.format(cat_name=cat_name))

    for attachment_path in attachment_paths:
        with open(attachment_path, "rb") as handle:
            data = handle.read()
            maintype = "image"
            subtype = attachment_path.suffix.lstrip(".").lower() or "jpeg"
            message.add_attachment(
                data,
                maintype=maintype,
                subtype=subtype,
                filename=attachment_path.name,
            )

    return message


def _build_discord_payload(cat_name: str) -> dict[str, str]:
    return {
        "content": DISCORD_CONTENT_TEMPLATE.format(cat_name=cat_name),
        "username": DISCORD_USERNAME,
        "avatar_url": DISCORD_AVATAR_URL,
    }


def _send_discord_notification(cat_name: str, attachment_paths: list[Path]) -> None:
    if not DISCORD_WEBHOOK_URL:
        raise RuntimeError("Discord webhook URL is not configured for notifications.")

    content = _build_discord_payload(cat_name)
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"

    # Discord expects payload_json plus distinct file fields (files[0], files[1], ...).
    payload_part = (
        f"--{boundary}"
        "\r\n"
        "Content-Disposition: form-data; name=\"payload_json\""
        "\r\n\r\n"
        f"{json.dumps(content)}"
        "\r\n"
    ).encode("utf-8")

    body = payload_part

    for index, attachment_path in enumerate(attachment_paths):
        with open(attachment_path, "rb") as handle:
            file_data = handle.read()

        mimetype, _ = mimetypes.guess_type(attachment_path.name)
        if mimetype is None:
            mimetype = "application/octet-stream"

        file_part = (
            f"--{boundary}"
            "\r\n"
            f"Content-Disposition: form-data; name=\"files[{index}]\"; filename=\"{attachment_path.name}\""
            "\r\n"
            f"Content-Type: {mimetype}"
            "\r\n\r\n"
        ).encode("utf-8")

        body += file_part + file_data + b"\r\n"

    body += f"--{boundary}--\r\n".encode("utf-8")

    request = urllib.request.Request(
        DISCORD_WEBHOOK_URL,
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

    print("✅ Discord notification sent")


def send_notification(cat_name: str, attachment_paths: list[str] | str) -> None:
    if isinstance(attachment_paths, (str, Path)):
        attachment_paths = [attachment_paths]

    attachments = [Path(path) for path in attachment_paths]
    for attachment in attachments:
        if not attachment.exists():
            raise FileNotFoundError(f"Notification attachment not found: {attachment}")

    if MODE == "discord":
        print(f"📧 Sending Discord notification via webhook: {DISCORD_WEBHOOK_URL}")
        _send_discord_notification(cat_name, attachments)
        return

    if not RECIPIENTS:
        print("📧 Notification skipped: no recipients configured.")
        return

    if not SMTP_SERVER:
        raise RuntimeError("SMTP server is not configured for notifications.")

    message = _build_message(cat_name, attachments)

    print(f"📧 Sending mail notification to: {RECIPIENTS}")

    if USE_SSL:
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
    else:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)

    with server:
        if not USE_SSL and USE_TLS:
            server.starttls()

        if SMTP_USERNAME and SMTP_PASSWORD:
            server.login(SMTP_USERNAME, SMTP_PASSWORD)

        server.send_message(message)

    print("✅ Mail notification sent")
