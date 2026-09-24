"""Re-anchor a camera's zones onto a fresh reference frame and report the result.

When the camera drifts far enough that alignment degrades, the zones are
projected from the old reference frame onto a freshly captured frame. The new
calibration takes effect immediately; an overlay image showing the projected
zones is saved and sent to Discord so the owner can spot a bad projection and
redraw the zones if needed.
"""

import argparse
import re
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

from src.config_loader import load_env_file

# Before the monitoring imports below: they read the config, and the config substitutes
# ${VAR} from the environment at import time.
load_env_file()

from src.monitoring import alerting
from src.monitoring.alert_voice import phrase_alert
from src.monitoring.alignment import (
    USE_FRAME,
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


def describe_reason(reason: str) -> str:
    """Chinese gloss for a machine reason, so the alert reads like a sentence.

    ``estimate_alignment``'s reasons are precise and machine-shaped ("only 10
    feature matches"). They stay in the console log as they are; this is what the
    owner and the LLM are told the failure means.
    """
    for pattern, gloss in (
        (
            r"reference frame has too few features",
            "参考帧本身几乎找不到可匹配的特征（当初这张参考帧可能就是糊的）",
        ),
        (
            r"current frame has too few features",
            "当前画面几乎找不到可匹配的特征（可能太暗、太糊或者被挡住了）",
        ),
        (r"only (\d+) feature matches", r"画面里能跟参考帧对上号的点只剩 \1 个，不够定位"),
        (r"only (\d+) inliers after RANSAC", r"对上号的点里只有 \1 个能拼出一致的变换"),
        (r"transform could not be estimated", "点够多了，但拼不出一个可信的变换"),
    ):
        if re.fullmatch(pattern, reason):
            return re.sub(pattern, gloss, reason)
    return reason


# The drift sentences ``location_tracker.reanchor_reason`` builds. Each one names a
# different piece of evidence, and only the last carries a magnitude, so the numbers
# are the part worth preserving through the gloss.
DRIFT_REASONS = (
    (r"(\d+) consecutive failed matches", r"有连续 \1 次画面跟参考帧完全对不上"),
    (r"(\d+) consecutive samples below the good tier", r"有连续 \1 次匹配质量偏低"),
    (
        r"the transform has stayed large for (\d+) samples "
        r"\(shift ([\d.]+), rotation ([\d.-]+)deg, scale ([\d.]+)\)",
        r"连续 \1 次画面相对参考帧偏得很大（平移 \2、旋转 \3 度、缩放 \4）",
    ),
)


def describe_drift(reason: str) -> str:
    """Chinese gloss for ``reanchor_reason``'s own sentences.

    Same job as ``describe_reason``: the machine sentence stays in the console log
    while the owner and the LLM get something readable. The numbers are carried over
    unchanged, because the alert voice is told not to invent any of its own.
    """
    for pattern, gloss in DRIFT_REASONS:
        if re.fullmatch(pattern, reason):
            return re.sub(pattern, gloss, reason)
    return reason


def _alert_content(event: str, settings: dict, facts: dict, fallback: str) -> str:
    """Phrase an alert, but only when it is actually going to be sent.

    Skipping the call when notifications are off keeps the tracker from spending a
    network round trip - inside its sampling loop - on a message that will be
    dropped anyway.

    Deliberately calls this module's own ``phrase_alert`` rather than the shared
    ``alerting.alert_content``, because that name is the seam the tests use to watch
    what would have been said.
    """
    if not alerting.notifications_enabled(settings):
        return fallback
    return phrase_alert(event, facts, fallback)


def _notify(settings: dict, camera: str, content: str, attachment: Path | None) -> tuple[bool, str]:
    """Send a Discord message and report whether it actually went out.

    The caller uses the flag to decide whether the attachment is still needed: once
    Discord holds the image, a local copy is duplicated storage that piles up.

    The plumbing lives in ``alerting`` so the camera-discovery alerts obey the same
    rules - no webhook, no send; no User-Agent, Cloudflare rejects it.
    """
    return alerting.notify(settings, camera, content, attachment)


def reanchor(
    calibration: Calibration,
    frame: np.ndarray,
    settings: dict,
    now: datetime | None = None,
    notify_on_failure: bool = True,
) -> ReanchorOutcome:
    """Project the existing zones onto the current frame and store a new calibration.

    ``notify_on_failure=False`` skips the Discord message and the failure overlay
    for a failure. The tracker uses it to report only the first failure of a streak:
    retrying every cooldown is still worth it because the lighting may swing back,
    but repeating the same warning every ten minutes is not. A success always
    notifies, whatever this is set to, which is what re-arms the alert.
    """
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
        mode_zh = "区域是在这个画面上直接画的，原样保留"
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
            if not notify_on_failure:
                return ReanchorOutcome(
                    False,
                    None,
                    f"automatic re-anchor failed: {result.reason}; "
                    "notification suppressed (already reported for this camera)",
                )
            # What the failure means depends on the configured fallback.
            if settings.get("on_alignment_failure") == USE_FRAME:
                advice = (
                    "区域仍然按当前画面解析，所以位置还能用，这些样本事后也能筛掉。"
                    "如果看着不对，请在标注页面重新保存一次区域，把现在的光线当成新的参考帧。"
                )
            else:
                advice = "旧区域还会继续用，但位置会退化成相机名字。请打开标注页面重画区域。"
            overlay = render_overlay(
                frame,
                [],
                f"{calibration.camera}: automatic re-anchor failed",
                result.reason,
            )
            overlay_path = camera_dir / f"reanchor_failed_{calibration_id}.jpg"
            cv2.imwrite(str(overlay_path), overlay)
            plain = (
                f"⚠️ `{calibration.camera}` 自动重新锚定没成功。\n"
                f"原因：{describe_reason(result.reason)}\n"
                f"{advice}"
            )
            content = _alert_content(
                "reanchor_failed",
                settings,
                {
                    "摄像头": calibration.camera,
                    "结果": "想自动重新锚定，但失败了",
                    "失败原因": describe_reason(result.reason),
                    "位置现在还能不能用": settings.get("on_alignment_failure") == USE_FRAME,
                    "需要主人做什么": advice,
                },
                plain,
            )
            delivered, note = _notify(settings, calibration.camera, content, overlay_path)
            if delivered:
                # Discord already holds the image, so the local copy is duplicated
                # storage that otherwise grows by one file per failed attempt.
                overlay_path.unlink(missing_ok=True)
                overlay_path = None
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
        mode_zh = "把旧参考帧上的区域投影到新的参考帧"

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
        f"🔄 `{calibration.camera}` 已经重新锚定到新的参考帧。",
        f"区域处理方式：{mode_zh}",
    ]
    if transform_note:
        lines.append(f"变换：{transform_note}")
    lines.append(f"区域数量：{len(projected)}")
    if off_frame:
        lines.append(f"⚠️ 这些区域现在跑到画面外了：{', '.join(off_frame)}")
    lines.append("看一眼下面的图确认区域没歪；歪了就手动重画。")
    needs_hand = bool(off_frame)
    plain = "\n".join(lines)
    content = _alert_content(
        "reanchor_ok",
        settings,
        {
            "摄像头": calibration.camera,
            "结果": "已经自动重新锚定到新的参考帧",
            "区域处理方式": mode_zh,
            "区域数量": len(projected),
            "摄像头位置变化的量级": transform_note or "没有需要补偿的变化",
            "跑到画面外的区域": list(off_frame) or "没有",
            "需要主人做什么": (
                "有区域跑到画面外了，需要主人手动重画"
                if needs_hand
                else "不用动手，看一眼下面的图确认一下就行"
            ),
        },
        plain,
    )
    _, note = _notify(settings, calibration.camera, content, overlay_path)

    return ReanchorOutcome(
        True, updated, f"{mode}; {note}", overlay_path, transform_note, off_frame
    )


def announce_drift(calibration: Calibration, settings: dict, reason: str, notify=None) -> str:
    """Report that the camera drifted, and leave the zones alone.

    This is the whole of alert-only mode, and it exists because a wrong automatic
    repair is worse than no repair. Projecting the zones onto a new frame is only as
    trustworthy as the transform behind it, and when that transform goes bad it does
    not fail loudly - it writes nonsense into every polygon. Measured once for real:
    a collapsed match reported ``scale=0.000`` and stored all eight of feeder's
    hand-drawn zones as a single point at (0, 0). Detecting the drift loses nothing,
    so the detection is kept and the repair waits for the owner.
    """
    notify = notify or _notify
    readable = describe_drift(reason)
    plain = (
        f"\u26a0\ufe0f `{calibration.camera}` 的画面跟参考帧对不上了，可能是被碰歪了。\n"
        f"依据：{readable}\n"
        "区域没有被动过，还是上次手动保存的那一份。\n"
        "看一眼画面：区域确实歪了，就在标注页面重画一次。"
    )
    content = _alert_content(
        "reanchor_drift",
        settings,
        {
            "摄像头": calibration.camera,
            "发生了什么": "画面跟参考帧对不上了，可能是摄像头被碰歪了",
            "对不上的程度": readable,
            "区域有没有被自动改动": "没有。自动重锚已经关掉，区域还是上次手动保存的那一份",
            "需要主人做什么": "看一眼画面；区域确实歪了就重新画一次，没歪就不用管",
        },
        plain,
    )
    _, note = notify(settings, calibration.camera, content, None)
    return note


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
