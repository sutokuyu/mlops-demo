"""Re-anchor a camera's zones onto a fresh reference frame and report the result.

When the camera drifts far enough that alignment degrades, the zones are
projected from the old reference frame onto a freshly captured frame. The new
calibration takes effect immediately; an overlay image showing the projected
zones is saved and sent to Discord so the owner can spot a bad projection and
redraw the zones if needed.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def _resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not locate project root")


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.alignment import (
    ReferenceFeatures,
    build_reference_features,
    describe_transform,
    estimate_alignment,
    invert,
    transform_points,
)
from src.monitoring.location_config import (
    alignment_settings,
    camera_rtsp_url,
    configured_cameras,
)
from src.monitoring.location_zones import (
    ZONES_PATH,
    Calibration,
    Zone,
    load_calibrations,
    save_calibrations,
)
from src.monitoring.rtsp_stream import StreamReader
from src.notification.notification_controller import post_discord_message

ZONE_COLOR = (0, 255, 0)
OFF_FRAME_COLOR = (0, 0, 255)
HEADER_COLOR = (0, 200, 255)


@dataclass
class ReanchorOutcome:
    ok: bool
    calibration: Calibration | None
    message: str
    overlay_path: Path | None = None
    transform: str = ""
    off_frame_zones: tuple[str, ...] = ()


def overlay_zone(frame, zone: Zone, off_frame: bool) -> None:
    height, width = frame.shape[:2]
    color = OFF_FRAME_COLOR if off_frame else ZONE_COLOR
    polygon = np.array([[int(x * width), int(y * height)] for x, y in zone.points], dtype=np.int32)
    cv2.polylines(frame, [polygon], True, color, 2)
    center_x, center_y = zone.centroid
    label = f"{zone.name} (off-frame)" if off_frame else zone.name
    text_x = int(np.clip(center_x * width, 4, max(4, width - 260)))
    text_y = int(np.clip(center_y * height, 22, max(22, height - 10)))
    cv2.putText(
        frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
    )


def render_overlay(frame: np.ndarray, zones: list[Zone], header: str, subheader: str = ""):
    canvas = frame.copy()
    width = canvas.shape[1]
    cv2.rectangle(canvas, (0, 0), (width, 34), (0, 0, 0), -1)
    cv2.putText(
        canvas, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HEADER_COLOR, 2, cv2.LINE_AA
    )
    if subheader:
        cv2.putText(
            canvas,
            subheader,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    for zone in zones:
        center_x, center_y = zone.centroid
        off_frame = not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0)
        overlay_zone(canvas, zone, off_frame)
    return canvas


def reference_features_for(
    calibration: Calibration, work_width: int = 640
) -> ReferenceFeatures | None:
    path = calibration.reference_path
    if path is None or not path.is_file():
        return None
    frame = cv2.imread(str(path))
    if frame is None:
        return None
    return build_reference_features(frame, work_width)


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _notify(settings: dict, camera: str, content: str, attachment: Path | None) -> str:
    webhook_url = settings.get("discord_webhook") or ""
    if not webhook_url or settings.get("notify") is False:
        return "notification skipped (no webhook configured)"
    try:
        post_discord_message(
            {"content": content, "username": settings.get("discord_username") or camera},
            [attachment] if attachment is not None else None,
            webhook_url=webhook_url,
        )
    except (OSError, RuntimeError, ValueError) as error:
        return f"notification failed: {error}"
    return "overlay sent to Discord"


def reanchor(
    calibration: Calibration,
    frame: np.ndarray,
    settings: dict,
    now: datetime | None = None,
) -> ReanchorOutcome:
    """Project the existing zones onto the current frame and store a new calibration."""
    now = now or datetime.now()
    calibration_dir = Path(settings["calibration_dir"])
    camera_dir = calibration_dir / calibration.camera
    camera_dir.mkdir(parents=True, exist_ok=True)
    zones_path = Path(settings["zones_path"]) if settings.get("zones_path") else ZONES_PATH

    # Milliseconds keep two re-anchors in the same second distinct: the id names
    # the reference file and is what observations use to identify a calibration.
    calibration_id = (
        f"{calibration.camera}-{now.strftime('%Y%m%dT%H%M%S')}{now.microsecond // 1000:03d}"
    )
    reference_path = camera_dir / f"reference_{calibration_id}.jpg"
    reference = (
        reference_features_for(calibration, settings.get("work_width", 640))
        if calibration.reference_frame
        else None
    )

    if not calibration.zones:
        return ReanchorOutcome(False, None, "no zones defined for this camera yet")

    transform_note = ""
    if reference is None:
        # Nothing to project from: the zones were drawn on the live view, so keep
        # them and simply adopt the current frame as the new reference.
        projected = list(calibration.zones)
        mode = "initial reference frame"
    else:
        result = estimate_alignment(
            reference,
            frame,
            work_width=settings.get("work_width", 640),
            min_inliers=settings.get("min_inliers", 15),
            good_inlier_ratio=settings.get("good_inlier_ratio", 0.5),
            max_residual=settings.get("max_residual", 0.01),
        )
        if not result.usable:
            overlay = render_overlay(
                frame,
                [],
                f"{calibration.camera}: automatic re-anchor failed",
                f"{result.reason} - please redraw the zones manually",
            )
            overlay_path = camera_dir / f"reanchor_failed_{calibration_id}.jpg"
            cv2.imwrite(str(overlay_path), overlay)
            content = (
                f"⚠️ Camera `{calibration.camera}` could not be re-anchored automatically.\n"
                f"Reason: {result.reason}\n"
                "The previous zones are kept and locations fall back to the camera name. "
                "Please run the zone editor to redraw them."
            )
            note = _notify(settings, calibration.camera, content, overlay_path)
            return ReanchorOutcome(
                False, None, f"automatic re-anchor failed: {result.reason}; {note}", overlay_path
            )

        # Zones live in reference coordinates, so invert to draw them on this frame.
        reference_to_current = invert(result.matrix)
        projected = [
            Zone(name=zone.name, points=transform_points(reference_to_current, zone.points))
            for zone in calibration.zones
        ]
        transform_note = describe_transform(result.matrix)
        mode = "zones projected from the previous reference frame"

    off_frame = tuple(
        zone.name
        for zone in projected
        if not (0.0 <= zone.centroid[0] <= 1.0 and 0.0 <= zone.centroid[1] <= 1.0)
    )
    subheader = transform_note or "no transform available"
    overlay = render_overlay(frame, projected, f"{calibration.camera}: {mode}", subheader)
    overlay_path = camera_dir / f"reanchor_{calibration_id}.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    if not cv2.imwrite(str(reference_path), frame):
        return ReanchorOutcome(False, None, f"could not save reference frame: {reference_path}")

    updated = Calibration(
        camera=calibration.camera,
        zones=projected,
        calibration_id=calibration_id,
        created_at=now.isoformat(timespec="seconds"),
        reference_frame=_relative(reference_path),
    )
    calibrations = load_calibrations(zones_path)
    calibrations[calibration.camera] = updated
    save_calibrations(calibrations, zones_path)

    lines = [
        f"🔄 Camera `{calibration.camera}` re-anchored to a new reference frame.",
        f"Calibration: `{calibration_id}`",
        f"{mode}",
    ]
    if transform_note:
        lines.append(f"Transform: {transform_note}")
    lines.append(f"Zones: {len(projected)}")
    if off_frame:
        lines.append(f"⚠️ These zones are off-frame now: {', '.join(off_frame)}")
    lines.append("Check the overlay image; redraw the zones if the alignment looks wrong.")
    note = _notify(settings, calibration.camera, "\n".join(lines), overlay_path)

    return ReanchorOutcome(
        True, updated, f"{mode}; {note}", overlay_path, transform_note, off_frame
    )


def wait_for_frame(reader, timeout_seconds: float):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        frame = reader.get_latest_frame()
        if frame is not None:
            return frame
        time.sleep(0.2)
    return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Re-anchor a camera's zones onto a fresh frame, save the overlay image, "
            "and send it to Discord for review."
        ),
    )
    parser.add_argument(
        "--camera",
        action="append",
        dest="camera_names",
        help="Camera to re-anchor; repeat for several. Defaults to all configured cameras.",
    )
    parser.add_argument("--capture-timeout", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    settings = alignment_settings()
    calibrations = load_calibrations()

    for camera_name in args.camera_names or configured_cameras():
        calibration = calibrations.get(camera_name)
        if calibration is None or not calibration.zones:
            print(f"[{camera_name}] skipped: no zones defined yet")
            continue
        try:
            rtsp_url = camera_rtsp_url(camera_name)
        except RuntimeError as error:
            print(f"[{camera_name}] skipped: {error}")
            continue

        reader = StreamReader(source=rtsp_url, name=camera_name)
        reader.start()
        try:
            frame = wait_for_frame(reader, args.capture_timeout)
        finally:
            reader.close()
        if frame is None:
            print(f"[{camera_name}] skipped: no frame captured")
            continue

        outcome = reanchor(calibration, frame, settings)
        print(f"[{camera_name}] {outcome.message}")
        if outcome.overlay_path is not None:
            print(f"[{camera_name}] overlay: {outcome.overlay_path}")


if __name__ == "__main__":
    main()
