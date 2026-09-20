"""Shared configuration for the location tracking scripts."""

import sys
from pathlib import Path


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

CONFIG = load_config(PROJECT_ROOT / "configs" / "config.yaml")
LOCATION_CONFIG = load_config(PROJECT_ROOT / "configs" / "locations.yaml")
TRACKING_CONFIG = LOCATION_CONFIG["tracking"]
ALIGNMENT_CONFIG = LOCATION_CONFIG.get("alignment", {})
CALIBRATION_CONFIG = LOCATION_CONFIG.get("calibration", {})
REPORT_CONFIG = LOCATION_CONFIG.get("report", {})
ALERT_CONFIG = LOCATION_CONFIG.get("alerts", {}).get("discord", {})
PREVIEW_CONFIG = LOCATION_CONFIG.get("preview", {})

IDENTITY_CLASSES = {index: name for index, name in enumerate(CONFIG["cats"]["identity_classes"])}
DEFAULT_IDENTITY_MODEL = resolve_config_path(CONFIG["models"]["identity_detection_model_path"])


def configured_cameras() -> list[str]:
    return list(LOCATION_CONFIG["cameras"])


def camera_rtsp_url(camera_name: str) -> str:
    for camera_config in CONFIG["identity_collection"]["cameras"]:
        if camera_config["name"] == camera_name:
            return camera_config["rtsp_url"]
    raise RuntimeError(f"Camera '{camera_name}' is not defined in identity_collection.cameras")


def alert_webhook() -> str:
    """Dedicated alert webhook, falling back to the daily report webhook."""
    return ALERT_CONFIG.get("webhook_url") or REPORT_CONFIG.get("discord", {}).get(
        "webhook_url", ""
    )


def alignment_settings() -> dict:
    """Everything the alignment and re-anchor code paths need, in one place."""
    directory = CALIBRATION_CONFIG.get("directory", "data/calibrations")
    return {
        "work_width": ALIGNMENT_CONFIG.get("work_width", 640),
        "min_inliers": ALIGNMENT_CONFIG.get("min_inliers", 15),
        "good_inlier_ratio": ALIGNMENT_CONFIG.get("good_inlier_ratio", 0.5),
        "max_residual": ALIGNMENT_CONFIG.get("max_residual", 0.01),
        "calibration_dir": str(resolve_config_path(directory)),
        "discord_webhook": alert_webhook(),
        "discord_username": ALERT_CONFIG.get("username", "Cat Location Bot"),
        "notify": ALIGNMENT_CONFIG.get("notify_on_reanchor", True),
    }


def location_database() -> Path:
    return resolve_config_path(TRACKING_CONFIG["database"])
