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
    return parser.parse_args(argv)


def build_readers(camera_names: list[str]) -> list[tuple[str, StreamReader]]:
    readers: list[tuple[str, StreamReader]] = []
    for camera_name in camera_names:
        try:
            rtsp_url = camera_rtsp_url(camera_name)
        except RuntimeError as error:
            print(f"[{camera_name}] skipped: {error}")
            continue
        if not rtsp_url:
            print(f"[{camera_name}] skipped: no RTSP URL configured")
            continue
        reader = StreamReader(source=rtsp_url, name=camera_name)
        reader.start()
        readers.append((camera_name, reader))
    return readers


def main(argv=None) -> None:
    args = parse_args(argv)
    camera_names = args.camera_names or configured_cameras()
    hub = FrameHub(display_width=args.display_width, quality=args.jpeg_quality)

    readers = build_readers(camera_names)
    if not readers:
        print(
            "No camera has an RTSP URL.\n"
            "Export the URLs first, for example:\n"
            "  export LIVING_ROOM_RTSP_URL='rtsp://...'\n"
            "  export SOFA_RTSP_URL='rtsp://...'\n"
            "  export FEEDER_RTSP_URL='rtsp://...'"
        )
        raise SystemExit(2)
    for camera_name, _ in readers:
        hub.register(camera_name)

    context = PreviewContext(hub=hub, settings=alignment_settings())
    server = start_server(context, host=args.host, port=args.port)
    print(f"Preview ready — open http://localhost:{args.port} in your browser")
    print(f"Serving cameras: {', '.join(name for name, _ in readers)}")

    model = None
    if not args.no_detections:
        model = YOLO(str(args.model))

    interval = 1.0 / args.fps if args.fps > 0 else 0.0
    try:
        while True:
            loop_start = time.monotonic()
            for camera_name, reader in readers:
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
        for _, reader in readers:
            reader.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
