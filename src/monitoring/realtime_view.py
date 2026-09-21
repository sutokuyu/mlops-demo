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
from src.monitoring.alignment import AlignmentTracker, transform_point
from src.monitoring.detections import (
    ANCHOR_STRATEGIES,
    BOTTOM_CENTER,
    DEFAULT_GRID,
    anchor_points,
    bottom_center,
    boxes_from_result,
    select_best_per_class,
)
from src.monitoring.location_config import (
    ALIGNMENT_CONFIG,
    DEFAULT_IDENTITY_MODEL,
    IDENTITY_CLASSES,
    PREVIEW_CONFIG,
    TRACKING_CONFIG,
    alignment_settings,
    camera_rtsp_url,
    configured_cameras,
)
from src.monitoring.location_zones import Calibration, load_calibrations, vote_zone
from src.monitoring.recalibration import reference_features_for
from src.monitoring.web_preview import FrameHub, PreviewContext, start_server

CONFIG = load_config(PROJECT_ROOT / "configs" / "config.yaml")

BOX_COLOR = (0, 255, 0)
ANCHOR_COLOR = (0, 210, 255)
LABEL_COLOR = (16, 24, 16)

# Frames to process before explaining why nothing has been detected yet.
WARN_AFTER_FRAMES = 8


@dataclass
class StreamReader:
    """Continuously reads an RTSP/webcam source in the background and keeps only the newest frame."""

    source: str | int
    name: str = "stream"
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
                    print(f"[{self.name}] lost stream; reader stopped")
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
        # Drop the last frame so a finished stream cannot be reported as live.
        with self.frame_lock:
            self.latest_frame = None


def _coerce_source(raw: str) -> str | int:
    """Allow a bare number to mean a local webcam index."""
    return int(raw) if raw.isdigit() else raw


def draw_detection(frame, cat: str, confidence: float, box, zone: str | None) -> None:
    """Draw one deduplicated box, its label and the zone anchor used for lookup."""
    x1, y1, x2, y2 = (int(value) for value in box)
    label = f"{cat} {confidence:.2f}" if zone is None else f"{cat} {confidence:.2f} @ {zone}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(
        frame, (x1, max(0, y1 - text_height - 10)), (x1 + text_width + 8, y1), BOX_COLOR, -1
    )
    cv2.putText(
        frame, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LABEL_COLOR, 2, cv2.LINE_AA
    )

    # The anchor is what zone lookup actually tests, so it is worth showing.
    anchor_x, anchor_y = (int(value) for value in bottom_center(box))
    cv2.drawMarker(frame, (anchor_x, anchor_y), ANCHOR_COLOR, cv2.MARKER_CROSS, 26, 2)
    if zone is None:
        cv2.putText(
            frame,
            "no zone",
            (anchor_x + 14, anchor_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            ANCHOR_COLOR,
            2,
            cv2.LINE_AA,
        )


def window_closed(name: str) -> bool:
    """True only when OpenCV reports the window as gone.

    ``getWindowProperty`` returns -1 for properties the active backend does not
    implement, so only an exact 0 means the operator closed the window. Testing
    ``< 1`` made this loop exit on its very first frame on backends that return -1.
    """
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) == 0
    except cv2.error:
        return False


@dataclass
class Location:
    """Where a cat is, in the reference frame the zones were drawn on."""

    zone: str | None = None
    quality: str = "unknown"
    norm_x: float = 0.0
    norm_y: float = 0.0
    # How lopsided the zone vote was, e.g. "9/9". Empty when no vote ran.
    share: str = ""


class ZoneLocator:
    """Resolve detection boxes to zone names, compensating for camera drift.

    Mirrors location_tracker so the printed location matches what the recorder
    stores: the same anchor strategy picks the candidate points and the same vote
    turns them into a zone, after being transformed back into the reference frame.
    """

    def __init__(
        self,
        camera: str,
        calibration: Calibration | None = None,
        settings: dict | None = None,
        strategy: str | None = None,
        grid: int | None = None,
    ) -> None:
        settings = settings or alignment_settings()
        self.camera = camera
        self.strategy = strategy or TRACKING_CONFIG.get("anchor_strategy", BOTTOM_CENTER)
        self.grid = int(grid or TRACKING_CONFIG.get("anchor_grid", DEFAULT_GRID))
        if self.strategy not in ANCHOR_STRATEGIES:
            raise ValueError(f"unknown anchor strategy: {self.strategy}")
        self.calibration = (
            calibration or load_calibrations().get(camera) or Calibration(camera=camera)
        )
        self.alignment: AlignmentTracker | None = None
        if ALIGNMENT_CONFIG.get("enabled", True) and self.calibration.reference_path:
            self.alignment = AlignmentTracker(
                camera=camera,
                work_width=settings["work_width"],
                min_inliers=settings["min_inliers"],
                good_inlier_ratio=settings["good_inlier_ratio"],
                max_residual=settings["max_residual"],
                trust_last_good_seconds=ALIGNMENT_CONFIG.get("trust_last_good_seconds", 60.0),
            )
            self.alignment.set_reference(
                reference_features_for(self.calibration, settings["work_width"])
            )

    @property
    def zone_count(self) -> int:
        return len(self.calibration.zones)

    def resolve(self, frame):
        """Align this frame to the reference frame; None when there is no reference."""
        if self.alignment is None:
            return None
        return self.alignment.resolve(frame)

    def locate(self, state, frame_shape, box) -> Location:
        height, width = frame_shape[:2]
        anchor_x, anchor_y = bottom_center(box)
        norm_x, norm_y = anchor_x / width, anchor_y / height
        quality = state.quality if state is not None else "no-reference"
        candidates = [
            (x / width, y / height) for x, y in anchor_points(box, self.strategy, self.grid)
        ]
        if state is not None:
            if not state.zone_lookup_allowed:
                # Alignment is too far off to trust; naming a zone here would be a lie.
                return Location(None, quality, norm_x, norm_y)
            if state.matrix is not None:
                norm_x, norm_y = transform_point(state.matrix, (norm_x, norm_y))
                candidates = [transform_point(state.matrix, point) for point in candidates]
        # A missing reference frame is not the same as an untrustworthy one: the
        # zones are used as captured, which is what an uncalibrated camera gets.
        vote = vote_zone(self.calibration.zones, candidates)
        return Location(vote.zone, quality, norm_x, norm_y, vote.share)


def format_location(camera: str, cat: str, location: Location, confidence: float) -> str:
    votes = f" vote={location.share}" if location.share else ""
    return (
        f"[{camera}] {cat} -> {location.zone or 'unknown'}"
        f"  conf={confidence:.2f}"
        f" anchor=({location.norm_x:.3f}, {location.norm_y:.3f})"
        f"{votes}"
        f" align={location.quality}"
    )


@dataclass
class Channel:
    """One camera stream: its reader, its own zone lookup and its own state.

    Everything that used to be a local variable in the run loop lives here, so
    several cameras can be sampled without their bookkeeping interfering.
    """

    name: str
    source: str | int
    locator: ZoneLocator | None = None
    reader: StreamReader | None = None
    last_reported: dict[str, tuple[str | None, str]] = dataclass_field(default_factory=dict)
    frames_processed: int = 0
    frames_without_detection: int = 0
    best_seen: float = 0.0
    warned_band: int = -1
    next_attempt_at: float = 0.0

    def open_reader(self) -> None:
        """Create the background reader; the run loop connects on its first pass."""
        self.reader = StreamReader(source=self.source, name=self.name)


def window_title(channel: Channel, args: argparse.Namespace) -> str:
    return f"{args.window_name}: {channel.name}"


def build_channels(args: argparse.Namespace) -> list[Channel]:
    """One channel per selected camera, or a single channel for an explicit --source."""
    if args.source is not None:
        return [Channel(name=args.zones_camera or "camera", source=_coerce_source(args.source))]

    channels = []
    for camera_name in args.camera_names or configured_cameras():
        try:
            rtsp_url = camera_rtsp_url(camera_name)
        except RuntimeError as error:
            print(f"[{camera_name}] skipped: {error}")
            continue
        if not rtsp_url:
            print(f"[{camera_name}] skipped: no RTSP URL configured")
            continue
        channels.append(Channel(name=camera_name, source=rtsp_url))
    return channels


def attach_locators(channels: list[Channel], args: argparse.Namespace) -> None:
    if args.no_zones:
        return
    for channel in channels:
        if args.source is not None and args.zones_camera is None:
            print("[zones] pass --zones-camera with --source to look up zones")
            continue
        locator = ZoneLocator(channel.name)
        if not locator.zone_count:
            print(f"[zones] no zones stored for '{channel.name}'; draw them first")
            continue
        channel.locator = locator
        print(f"[zones] {locator.zone_count} zone(s) loaded for '{channel.name}'")


def report_missing_detections(channel: Channel, args: argparse.Namespace) -> None:
    """Explain an empty browser panel instead of leaving it silent."""
    if channel.frames_without_detection == WARN_AFTER_FRAMES:
        if channel.best_seen > 0:
            print(
                f"[{channel.name}] {channel.frames_processed} frames in, nothing above "
                f"conf={args.conf:.2f} (best sub-threshold {channel.best_seen:.2f}). "
                f"If the cat is on screen, try --conf {max(0.05, channel.best_seen - 0.05):.2f}"
            )
            channel.warned_band = int(channel.best_seen * 10)
        else:
            print(
                f"[{channel.name}] {channel.frames_processed} frames in, no cat found at all "
                f"(tried down to conf={args.raw_conf:.2f}). Check the model and --imgsz "
                f"({args.imgsz})."
            )
        return

    if (
        channel.frames_without_detection > WARN_AFTER_FRAMES
        and channel.best_seen > 0
        and int(channel.best_seen * 10) > channel.warned_band
    ):
        channel.warned_band = int(channel.best_seen * 10)
        print(
            f"[{channel.name}] still below conf={args.conf:.2f}; "
            f"best sub-threshold is now {channel.best_seen:.2f}"
        )


def process_channel(
    channel: Channel,
    model,
    args: argparse.Namespace,
    context: PreviewContext | None,
    use_window: bool,
) -> bool:
    """Sample one camera. Returns True when the operator asked to stop."""
    reader = channel.reader
    if reader.capture is None:
        now = time.monotonic()
        if now < channel.next_attempt_at:
            return False
        if not reader.connect():
            channel.next_attempt_at = now + args.reconnect_interval
            print(
                f"[{channel.name}] cannot open the stream; "
                f"retrying in {args.reconnect_interval:.0f}s"
            )
            return False
        print(f"[{channel.name}] connected")

    frame = reader.get_latest_frame()
    if frame is None:
        if reader.thread is not None and not reader.thread.is_alive():
            reader.close()
        return False

    result = model.predict(
        frame, conf=args.raw_conf, imgsz=args.imgsz, device=args.device, verbose=False
    )[0]
    raw_boxes = boxes_from_result(result, IDENTITY_CLASSES)
    boxes = select_best_per_class(raw_boxes, args.conf, args.cross_class_iou)
    channel.frames_processed += 1
    channel.best_seen = max([channel.best_seen, *(conf for _, conf, _ in raw_boxes)])

    if boxes:
        if channel.frames_without_detection:
            print(
                f"[{channel.name}] detections started after "
                f"{channel.frames_without_detection} frame(s) without a cat"
            )
        channel.frames_without_detection = 0
    else:
        channel.frames_without_detection += 1
        report_missing_detections(channel, args)

    # Align once per frame: the transform is the same for every box in it.
    state = channel.locator.resolve(frame) if channel.locator is not None else None
    annotated = frame.copy()
    reports = []
    for class_id, confidence, box in boxes:
        cat = IDENTITY_CLASSES.get(class_id, str(class_id))
        location = (
            channel.locator.locate(state, frame.shape, box)
            if channel.locator is not None
            else Location(None, "disabled", 0.0, 0.0)
        )
        draw_detection(annotated, cat, confidence, box, location.zone)
        if channel.locator is not None:
            reports.append((cat, location, confidence))

    for cat, location, confidence in reports:
        # Only the place matters for "where is the cat", so a confidence wobble
        # on the same zone must not reprint the same line.
        key = (location.zone, location.quality)
        if args.print_every_sample or key != channel.last_reported.get(cat):
            line = format_location(channel.name, cat, location, confidence)
            print(line)
            if context is not None:
                # Same text the console prints, so the browser panel and the
                # terminal never disagree.
                context.log.append(
                    camera=channel.name,
                    cat=cat,
                    zone=location.zone,
                    quality=location.quality,
                    confidence=confidence,
                    norm_x=location.norm_x,
                    norm_y=location.norm_y,
                    text=line,
                )
        channel.last_reported[cat] = key

    if context is not None:
        context.hub.publish(
            channel.name,
            frame,
            annotated,
            detections=len(boxes),
            best_confidence=channel.best_seen,
        )

    if not use_window:
        return False
    title = window_title(channel, args)
    cv2.imshow(title, annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        return True
    return window_closed(title)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a live window showing YOLO detections on a camera feed, sampled at a low FPS.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        dest="camera_names",
        help="Camera to monitor; repeat for several. Defaults to every configured camera "
        "that has an RTSP URL.",
    )
    parser.add_argument(
        "--source",
        help="RTSP URL, video file path, or webcam index; monitoring a single source "
        "instead of the configured cameras.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_IDENTITY_MODEL,
        help="YOLO weights to run for detection.",
    )
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold.")
    parser.add_argument(
        "--raw-conf",
        type=float,
        default=0.05,
        help="Internal detector threshold. Kept low so sub-threshold boxes can be "
        "reported as a hint when nothing passes --conf.",
    )
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
    parser.add_argument(
        "--zones-camera",
        help="Camera whose zones to use with --source; also names the browser tab.",
    )
    parser.add_argument(
        "--no-zones",
        action="store_true",
        help="Only draw detections; do not resolve or print locations.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Console only: skip the OpenCV window, which WSLg cannot show reliably.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Show the stream and a live location panel in a browser instead of a window.",
    )
    parser.add_argument("--host", default=PREVIEW_CONFIG.get("host", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(PREVIEW_CONFIG.get("port", 8765)))
    parser.add_argument(
        "--display-width",
        type=int,
        default=int(PREVIEW_CONFIG.get("display_width", 1280)),
        help="Streams are downscaled to this width.",
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=int(PREVIEW_CONFIG.get("jpeg_quality", 80))
    )
    parser.add_argument(
        "--print-every-sample",
        action="store_true",
        help="Print the location on every sample instead of only when it changes.",
    )
    parser.add_argument(
        "--cross-class-iou",
        type=float,
        default=0.90,
        help="Drop the weaker box when bagel and kurumi boxes overlap this much.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    channels = build_channels(args)
    if not channels:
        print(
            "No camera to monitor.\n"
            "Set the RTSP URLs and try again, for example:\n"
            "  export LIVING_ROOM_RTSP_URL='rtsp://...'\n"
            "  export SOFA_RTSP_URL='rtsp://...'\n"
            "  export FEEDER_RTSP_URL='rtsp://...'\n"
            "or pass --source <url|file|index> to monitor one explicit stream."
        )
        raise SystemExit(2)
    attach_locators(channels, args)

    interval = 1.0 / args.fps if args.fps > 0 else 0.0
    model = YOLO(str(args.model))

    hub = None
    context = None
    server = None
    use_window = not args.no_window and not args.serve
    if args.serve:
        hub = FrameHub(display_width=args.display_width, quality=args.jpeg_quality)
        for channel in channels:
            hub.register(channel.name)
        context = PreviewContext(hub=hub, settings=alignment_settings())
        server = start_server(context, host=args.host, port=args.port)
        print(f"[realtime-view] browser console: http://localhost:{args.port}")
        print(
            f"[realtime-view] switch cameras with the tabs: {', '.join(c.name for c in channels)}"
        )
    elif args.no_window:
        print("[realtime-view] console mode: locations are printed, no window is opened")
    else:
        for channel in channels:
            cv2.namedWindow(window_title(channel, args), cv2.WINDOW_NORMAL)

    for channel in channels:
        channel.open_reader()

    try:
        while True:
            loop_start = time.monotonic()
            stopped = False
            for channel in channels:
                if process_channel(channel, model, args, context, use_window):
                    stopped = True
            if stopped:
                break
            remaining = interval - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        for channel in channels:
            if channel.reader is not None:
                channel.reader.close()
        if use_window:
            cv2.destroyAllWindows()
        if server is not None:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
