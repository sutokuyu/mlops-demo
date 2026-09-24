"""Run the browser preview and zone editor.

Reads every configured camera, draws the identity detections and the zone anchor
(the bottom-centre of the box, which is the point zones are matched against), and
publishes the result to the web UI in ``web_preview``. Nothing is written to the
database here: this is the operator tool for drawing and verifying zones, while
``location_tracker`` does the recording.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


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

from src.monitoring.detections import bottom_center, boxes_from_result, select_best_per_class
from src.monitoring.location_config import (
    DEFAULT_IDENTITY_MODEL,
    IDENTITY_CLASSES,
    PREVIEW_CONFIG,
    TRACKING_CONFIG,
    alignment_settings,
    camera_rtsp_url,
    configured_cameras,
)
from src.monitoring.rtsp_stream import StreamReader
from src.monitoring.web_preview import FrameHub, PreviewContext, start_server

BOX_COLOR = (100, 220, 130)
ANCHOR_COLOR = (0, 210, 255)
LABEL_COLOR = (16, 24, 16)


def annotate(frame, boxes) -> object:
    """Draw detection boxes plus the zone anchor that zone lookup actually uses."""
    canvas = frame.copy()
    for class_id, confidence, box in boxes:
        x1, y1, x2, y2 = (int(value) for value in box)
        label = f"{IDENTITY_CLASSES.get(class_id, class_id)} {confidence:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR, 2)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            canvas, (x1, max(0, y1 - text_height - 10)), (x1 + text_width + 8, y1), BOX_COLOR, -1
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            LABEL_COLOR,
            2,
            cv2.LINE_AA,
        )
        anchor_x, anchor_y = (int(value) for value in bottom_center(box))
        cv2.drawMarker(canvas, (anchor_x, anchor_y), ANCHOR_COLOR, cv2.MARKER_CROSS, 26, 2)
    return canvas


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a browser preview plus zone editor for the configured cameras.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        dest="camera_names",
        help="Camera to serve; repeat for several. Defaults to all configured cameras.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_IDENTITY_MODEL)
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold.")
    parser.add_argument(
        "--raw-conf",
        type=float,
        default=0.05,
        help="Internal detector threshold, kept low so the UI can report how close a "
        "sub-threshold detection came.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=TRACKING_CONFIG.get("device"))
    parser.add_argument(
        "--fps",
        type=float,
        default=float(PREVIEW_CONFIG.get("fps", 2.0)),
        help="Detection and repaint rate.",
    )
    parser.add_argument("--host", default=PREVIEW_CONFIG.get("host", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(PREVIEW_CONFIG.get("port", 8765)))
    parser.add_argument(
        "--display-width",
        type=int,
        default=int(PREVIEW_CONFIG.get("display_width", 1280)),
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=int(PREVIEW_CONFIG.get("jpeg_quality", 80))
    )
    parser.add_argument(
        "--cross-class-iou",
        type=float,
        default=0.90,
        help="Drop the weaker box when bagel and kurumi boxes overlap this much.",
    )
    parser.add_argument(
        "--stale-after",
        type=float,
        default=15.0,
        help="Treat frames older than this as unavailable.",
    )
    parser.add_argument(
        "--no-detections",
        action="store_true",
        help="Publish frames without running the model; useful for a quick look.",
    )
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=15.0,
        help="How long to keep a camera's stream open after the last browser stops "
        "watching it, so a reload or a tab switch does not re-dial the camera.",
    )
    return parser.parse_args(argv)


def stream_schedule(
    camera_names: list[str],
    running: set[str],
    watched: set[str],
    idle_since: dict[str, float],
    now: float,
    grace_seconds: float,
) -> tuple[list[str], list[str], dict[str, float]]:
    """Which camera streams to open and which to release at this moment.

    Open what a browser is watching; release what stopped being watched a grace period
    ago - long enough that a reload or a tab switch does not drop and re-dial the
    camera, short enough that an abandoned tab does not hold a session all night.

    Kept pure and separate from the loop, so the rule "the tab you are looking at is
    the only stream open" can be tested without a camera, a socket or a browser.
    """
    idle = dict(idle_since)
    start: list[str] = []
    release: list[str] = []
    for name in camera_names:
        if name in watched:
            idle.pop(name, None)
            if name not in running:
                start.append(name)
            continue
        if name not in running:
            idle.pop(name, None)
            continue
        first_idle = idle.setdefault(name, now)
        if now - first_idle >= grace_seconds:
            release.append(name)
            idle.pop(name, None)
    return start, release, idle


def open_reader(camera_name: str) -> StreamReader | None:
    """Start one camera's reader, or explain why it cannot be started.

    The URL is resolved here rather than at startup, so the address the tracker last
    wrote into .env is picked up on the next tab switch instead of needing this
    process restarted.
    """
    try:
        rtsp_url = camera_rtsp_url(camera_name)
    except RuntimeError as error:
        print(f"[{camera_name}] cannot stream: {error}")
        return None
    if not rtsp_url:
        print(f"[{camera_name}] cannot stream: no RTSP URL configured")
        return None
    reader = StreamReader(source=rtsp_url, name=camera_name)
    reader.start()
    return reader


def cameras_with_urls(camera_names: list[str]) -> list[str]:
    """The cameras this preview can stream, checked without opening any of them."""
    usable: list[str] = []
    for camera_name in camera_names:
        try:
            if camera_rtsp_url(camera_name):
                usable.append(camera_name)
                continue
        except RuntimeError:
            pass
        print(f"[{camera_name}] skipped: no RTSP URL configured")
    return usable


def main(argv=None) -> None:
    args = parse_args(argv)
    camera_names = cameras_with_urls(args.camera_names or configured_cameras())
    hub = FrameHub(display_width=args.display_width, quality=args.jpeg_quality)
    if not camera_names:
        print(
            "No camera has an RTSP URL.\n"
            "Export the URLs first, for example:\n"
            "  export LIVING_ROOM_RTSP_URL='rtsp://...'\n"
            "  export SOFA_RTSP_URL='rtsp://...'\n"
            "  export FEEDER_RTSP_URL='rtsp://...'"
        )
        raise SystemExit(2)
    for camera_name in camera_names:
        hub.register(camera_name)
        hub.set_status(camera_name, "idle (not watched)")

    context = PreviewContext(hub=hub, settings=alignment_settings())
    server = start_server(context, host=args.host, port=args.port)
    print(f"Preview ready — open http://localhost:{args.port} in your browser")
    print(f"Serving cameras: {', '.join(camera_names)}")
    print(
        "Streams are opened on demand: the tab you are looking at is the only camera "
        f"this process holds open, and it is released {args.idle_seconds:.0f}s after "
        "the last browser looks away."
    )

    model = None
    if not args.no_detections:
        model = YOLO(str(args.model))

    interval = 1.0 / args.fps if args.fps > 0 else 0.0
    readers: dict[str, StreamReader] = {}
    idle_since: dict[str, float] = {}
    # A camera that cannot be opened (no URL, or a name the config does not know) will
    # not become openable a microsecond later, so a failed attempt is not retried on
    # every pass of the loop - that would be one log line per frame.
    retry_after: dict[str, float] = {}
    try:
        while True:
            loop_start = time.monotonic()
            start, release, idle_since = stream_schedule(
                camera_names,
                set(readers),
                hub.watched(),
                idle_since,
                loop_start,
                args.idle_seconds,
            )
            for camera_name in start:
                if loop_start < retry_after.get(camera_name, 0.0):
                    continue
                reader = open_reader(camera_name)
                if reader is None:
                    retry_after[camera_name] = loop_start + 30.0
                    hub.set_status(camera_name, "no stream (check its *_RTSP_URL)")
                    continue
                retry_after.pop(camera_name, None)
                readers[camera_name] = reader
                print(f"[{camera_name}] opening the stream (a browser is watching)")
            for camera_name in release:
                readers.pop(camera_name).close()
                hub.set_status(camera_name, "idle (not watched)")
                print(f"[{camera_name}] nobody is watching; released the stream")

            for camera_name, reader in list(readers.items()):
                frame = reader.get_latest_frame(stale_after_seconds=args.stale_after)
                if frame is None:
                    hub.set_status(
                        camera_name,
                        "connected, waiting for frame" if reader.connected else "connecting",
                    )
                    continue
                if model is None:
                    hub.publish(camera_name, frame, frame, detections=0)
                    continue
                result = model.predict(
                    frame, conf=args.raw_conf, imgsz=args.imgsz, device=args.device, verbose=False
                )[0]
                raw_boxes = boxes_from_result(result, IDENTITY_CLASSES)
                boxes = select_best_per_class(raw_boxes, args.conf, args.cross_class_iou)
                best_seen = max((conf for _, conf, _ in raw_boxes), default=0.0)
                hub.publish(
                    camera_name,
                    frame,
                    annotate(frame, boxes),
                    detections=len(boxes),
                    best_confidence=best_seen,
                )
            remaining = interval - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        for reader in readers.values():
            reader.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
