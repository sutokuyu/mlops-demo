"""Show a live OpenCV window with YOLO detections drawn on top, sampled at a low FPS."""

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

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

from src.config_loader import load_config

CONFIG = load_config(PROJECT_ROOT / "configs" / "config.yaml")
COLLECTION_CONFIG = CONFIG["identity_collection"]

BOX_COLOR = (0, 255, 0)


@dataclass
class StreamReader:
    """Continuously reads an RTSP/webcam source in the background and keeps only the newest frame."""

    source: str | int
    capture: cv2.VideoCapture | None = None
    latest_frame = None
    frame_lock: threading.Lock = dataclass_field(default_factory=threading.Lock)
    thread: threading.Thread | None = None
    stop_event: threading.Event = dataclass_field(default_factory=threading.Event)

    def connect(self) -> bool:
        api_preference = cv2.CAP_FFMPEG if isinstance(self.source, str) else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(self.source, api_preference)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            return False

        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True

    def _read_loop(self) -> None:
        while not self.stop_event.is_set():
            capture = self.capture
            if capture is None:
                break
            ok, frame = capture.read()
            if not ok:
                if not self.stop_event.is_set():
                    print("[realtime-view] lost stream; reader stopped")
                break
            with self.frame_lock:
                self.latest_frame = frame

    def get_latest_frame(self):
        with self.frame_lock:
            return self.latest_frame

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def resolve_source(args: argparse.Namespace) -> str | int:
    if args.source is not None:
        # Allow selecting a local webcam by index (e.g. "0").
        return int(args.source) if args.source.isdigit() else args.source

    if args.camera is not None:
        for camera_config in COLLECTION_CONFIG["cameras"]:
            if camera_config["name"] == args.camera:
                rtsp_url = camera_config["rtsp_url"]
                if not rtsp_url:
                    raise RuntimeError(f"No RTSP URL configured for camera '{args.camera}'")
                return rtsp_url
        raise RuntimeError(f"Unknown camera '{args.camera}'; check configs/config.yaml")

    rtsp_url = CONFIG["camera"]["rtsp_url"]
    if not rtsp_url:
        raise RuntimeError(
            "No --source/--camera given and configs/config.yaml has no camera.rtsp_url"
        )
    return rtsp_url


def draw_detections(frame, result) -> None:
    for box in result.boxes or []:
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = f"{result.names.get(class_id, class_id)} {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_height - 6), (x1 + text_width + 4, y1), BOX_COLOR, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a live window showing YOLO detections on a camera feed, sampled at a low FPS.",
    )
    parser.add_argument(
        "--camera",
        help="Named camera from configs/config.yaml identity_collection.cameras.",
    )
    parser.add_argument(
        "--source",
        help="RTSP URL, video file path, or webcam index; overrides --camera.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="YOLO weights to run for detection.",
    )
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=CONFIG["training"]["device"])
    parser.add_argument(
        "--fps", type=float, default=1.0, help="Detection/display rate in frames per second."
    )
    parser.add_argument("--window-name", default="Realtime Monitor")
    parser.add_argument(
        "--reconnect-interval",
        type=float,
        default=5.0,
        help="Seconds to wait between reconnect attempts when the stream is unavailable.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    source = resolve_source(args)
    interval = 1.0 / args.fps if args.fps > 0 else 0.0

    model = YOLO(str(args.model))

    reader = StreamReader(source=source)
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            loop_start = time.monotonic()

            if reader.capture is None:
                if not reader.connect():
                    print(f"[realtime-view] unable to connect to {source}; retrying...")
                    time.sleep(args.reconnect_interval)
                    continue
                print(f"[realtime-view] connected to {source}")

            frame = reader.get_latest_frame()
            if frame is None:
                if reader.thread is not None and not reader.thread.is_alive():
                    reader.close()
                time.sleep(0.1)
                continue

            results = model.predict(
                frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False
            )
            annotated = frame.copy()
            draw_detections(annotated, results[0])

            cv2.imshow(args.window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(args.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            elapsed = time.monotonic() - loop_start
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
