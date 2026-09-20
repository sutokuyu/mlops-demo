"""Sample the cameras periodically and record where each cat is staying."""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

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

from src.monitoring.alignment import GOOD, AlignmentTracker, transform_point
from src.monitoring.detections import bottom_center, boxes_from_result, select_best_per_class
from src.monitoring.location_config import (
    ALIGNMENT_CONFIG,
    DEFAULT_IDENTITY_MODEL,
    IDENTITY_CLASSES,
    LOCATION_CONFIG,
    TRACKING_CONFIG,
    alignment_settings,
    camera_rtsp_url,
    configured_cameras,
    location_database,
)
from src.monitoring.location_store import LocationStore
from src.monitoring.location_zones import Calibration, find_zone, load_calibrations
from src.monitoring.recalibration import reanchor, reference_features_for
from src.monitoring.rtsp_stream import StreamReader

EXPECTED_CLASSES = IDENTITY_CLASSES


@dataclass
class Observation:
    cat: str
    camera: str
    zone: str | None
    confidence: float
    norm_x: float | None = None
    norm_y: float | None = None
    calibration_id: str | None = None
    alignment_quality: str | None = None


@dataclass
class ActiveVisit:
    cat: str
    camera: str
    zone: str | None
    start_ts: float
    last_seen_ts: float
    samples: int
    max_confidence: float
    visit_id: int
    pending_key: tuple[str, str | None] | None = None
    pending_since_ts: float = 0.0
    pending_samples: int = 0


@dataclass
class CameraTracker:
    name: str
    rtsp_url: str
    calibration: Calibration
    alignment: AlignmentTracker | None = None
    reader: StreamReader | None = None
    last_sampled_at: float = 0.0
    degraded_streak: int = 0
    last_reanchor_at: float = 0.0

    @property
    def zones(self):
        return self.calibration.zones


def build_trackers() -> list[CameraTracker]:
    calibrations = load_calibrations()
    settings = alignment_settings()
    trackers = []
    for camera_name in configured_cameras():
        calibration = calibrations.get(camera_name) or Calibration(camera=camera_name)
        alignment = None
        if ALIGNMENT_CONFIG.get("enabled", True):
            alignment = AlignmentTracker(
                camera=camera_name,
                work_width=settings["work_width"],
                min_inliers=settings["min_inliers"],
                good_inlier_ratio=settings["good_inlier_ratio"],
                max_residual=settings["max_residual"],
                trust_last_good_seconds=ALIGNMENT_CONFIG.get("trust_last_good_seconds", 60.0),
            )
            alignment.set_reference(reference_features_for(calibration, settings["work_width"]))
        if not calibration.zones:
            print(f"[{camera_name}] no zones defined; locations fall back to the camera name")
        elif calibration.reference_path is None:
            print(f"[{camera_name}] zones have no reference frame; drift cannot be compensated")
        trackers.append(
            CameraTracker(
                name=camera_name,
                rtsp_url=camera_rtsp_url(camera_name),
                calibration=calibration,
                alignment=alignment,
            )
        )
    if not trackers:
        raise RuntimeError("No cameras configured under locations.yaml")
    return trackers


def sample_cameras(trackers: list[CameraTracker], model, args) -> list[Observation]:
    """Run detection on the newest frame of each camera and resolve zones."""
    observations: list[Observation] = []
    for tracker in trackers:
        frame = tracker.reader.get_latest_frame(TRACKING_CONFIG["frame_stale_after_seconds"])
        if frame is None:
            continue

        state = None
        if tracker.alignment is not None:
            state = tracker.alignment.resolve(frame)
            if state.quality == GOOD:
                tracker.degraded_streak = 0
            else:
                if tracker.degraded_streak == 0:
                    print(f"[{tracker.name}] alignment {state.quality}: {state.note}")
                tracker.degraded_streak += 1

        result = model.predict(
            frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False
        )[0]
        image_height, image_width = frame.shape[:2]
        boxes = select_best_per_class(
            boxes_from_result(result, EXPECTED_CLASSES), args.conf, args.cross_class_iou
        )
        for class_id, confidence, box in boxes:
            anchor_x, anchor_y = bottom_center(box)
            norm_x, norm_y = anchor_x / image_width, anchor_y / image_height
            zone = None
            if state is None or state.zone_lookup_allowed:
                if state is not None and state.matrix is not None:
                    norm_x, norm_y = transform_point(state.matrix, (norm_x, norm_y))
                found = find_zone(tracker.calibration.zones, norm_x, norm_y)
                zone = found.name if found else None
            observations.append(
                Observation(
                    cat=EXPECTED_CLASSES[class_id],
                    camera=tracker.name,
                    zone=zone,
                    confidence=confidence,
                    norm_x=norm_x,
                    norm_y=norm_y,
                    calibration_id=tracker.calibration.calibration_id or None,
                    alignment_quality=state.quality if state is not None else None,
                )
            )
    return observations


def best_observation_per_cat(observations: list[Observation]) -> dict[str, Observation]:
    """One cat can be seen by several cameras; keep the most confident sighting."""
    best: dict[str, Observation] = {}
    for observation in observations:
        current = best.get(observation.cat)
        if current is None or observation.confidence > current.confidence:
            best[observation.cat] = observation
    return best


def close_visit(store: LocationStore, visit: ActiveVisit, end_ts: float, reason: str) -> None:
    if end_ts <= visit.start_ts:
        end_ts = visit.start_ts
    store.touch_visit(visit.visit_id, end_ts, visit.samples, visit.max_confidence)
    duration = end_ts - visit.start_ts
    print(
        f"[{visit.cat}] {visit.zone or visit.camera} for {duration:.0f}s "
        f"({visit.samples} samples) -> {reason}"
    )


def apply_observations(
    store: LocationStore,
    active_visits: dict[str, ActiveVisit],
    observations: dict[str, Observation],
    now: float,
) -> None:
    for cat in set(active_visits) | set(observations):
        observation = observations.get(cat)
        visit = active_visits.get(cat)

        if observation is None:
            if visit and now - visit.last_seen_ts >= TRACKING_CONFIG["missing_timeout_seconds"]:
                close_visit(store, visit, visit.last_seen_ts, "no longer visible")
                del active_visits[cat]
            continue

        store.record_observation(
            now,
            cat,
            observation.camera,
            observation.zone,
            observation.confidence,
            observation.norm_x,
            observation.norm_y,
            observation.calibration_id,
            observation.alignment_quality,
        )
        key = (observation.camera, observation.zone)

        if visit is None:
            visit_id = store.open_visit(
                now,
                cat,
                observation.camera,
                observation.zone,
                observation.confidence,
                observation.calibration_id,
            )
            active_visits[cat] = ActiveVisit(
                cat=cat,
                camera=observation.camera,
                zone=observation.zone,
                start_ts=now,
                last_seen_ts=now,
                samples=1,
                max_confidence=observation.confidence,
                visit_id=visit_id,
            )
            print(f"[{cat}] now at {observation.zone or observation.camera}")
            continue

        if (visit.camera, visit.zone) == key:
            visit.last_seen_ts = now
            visit.samples += 1
            visit.max_confidence = max(visit.max_confidence, observation.confidence)
            visit.pending_key = None
            visit.pending_samples = 0
            store.touch_visit(visit.visit_id, now, visit.samples, visit.max_confidence)
            continue

        # Require repeated agreement before moving, otherwise a single noisy
        # sample would split one stay into several short visits.
        if visit.pending_key != key:
            visit.pending_key = key
            visit.pending_since_ts = now
            visit.pending_samples = 1
            store.touch_visit(visit.visit_id, now, visit.samples, visit.max_confidence)
            continue

        visit.pending_samples += 1
        if visit.pending_samples < TRACKING_CONFIG["switch_min_samples"]:
            store.touch_visit(visit.visit_id, now, visit.samples, visit.max_confidence)
            continue

        close_visit(store, visit, visit.pending_since_ts, f"moved to {key[1] or key[0]}")
        visit_id = store.open_visit(
            visit.pending_since_ts,
            cat,
            key[0],
            key[1],
            observation.confidence,
            observation.calibration_id,
        )
        active_visits[cat] = ActiveVisit(
            cat=cat,
            camera=key[0],
            zone=key[1],
            start_ts=visit.pending_since_ts,
            last_seen_ts=now,
            samples=1,
            max_confidence=observation.confidence,
            visit_id=visit_id,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Track where each cat stays by sampling fixed cameras and zone polygons.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_IDENTITY_MODEL,
    )
    parser.add_argument("--conf", type=float, default=TRACKING_CONFIG["confidence_threshold"])
    parser.add_argument("--imgsz", type=int, default=TRACKING_CONFIG["imgsz"])
    parser.add_argument("--device", default=TRACKING_CONFIG["device"])
    parser.add_argument(
        "--interval", type=float, default=TRACKING_CONFIG["sample_interval_seconds"]
    )
    parser.add_argument(
        "--cross-class-iou",
        type=float,
        default=0.90,
        help="Drop the weaker box when bagel and kurumi boxes overlap this much.",
    )
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument(
        "--camera",
        action="append",
        dest="camera_names",
        help="Restrict tracking to the given camera(s); repeat for multiple cameras.",
    )
    return parser.parse_args(argv)


def maybe_reanchor(tracker: CameraTracker, now: float, settings: dict) -> None:
    """Re-anchor a drifted camera, then report the projected zones to Discord."""
    if tracker.alignment is None:
        return
    cooldown = ALIGNMENT_CONFIG.get("reanchor_min_interval_seconds", 600)
    needs_reanchor = tracker.alignment.consecutive_failures >= ALIGNMENT_CONFIG.get(
        "reanchor_after_failures", 6
    ) or tracker.degraded_streak >= ALIGNMENT_CONFIG.get("reanchor_after_degraded_samples", 12)
    if not needs_reanchor or now - tracker.last_reanchor_at < cooldown:
        return

    tracker.last_reanchor_at = now
    tracker.degraded_streak = 0
    if not tracker.calibration.zones:
        return
    frame = (
        tracker.reader.get_latest_frame(TRACKING_CONFIG["frame_stale_after_seconds"])
        if tracker.reader is not None
        else None
    )
    if frame is None:
        print(f"[{tracker.name}] re-anchor skipped: no fresh frame")
        return

    outcome = reanchor(tracker.calibration, frame, settings)
    print(f"[{tracker.name}] re-anchor: {outcome.message}")
    if outcome.ok and outcome.calibration is not None:
        tracker.calibration = outcome.calibration
        tracker.alignment.set_reference(
            reference_features_for(outcome.calibration, settings["work_width"])
        )


def run(args) -> None:
    if not args.model.is_file():
        raise FileNotFoundError(f"Identity detection model not found: {args.model}")
    if args.interval <= 0:
        raise ValueError("--interval must be greater than 0")

    if args.camera_names:
        selected = [name for name in configured_cameras() if name in args.camera_names]
        LOCATION_CONFIG["cameras"] = selected

    database = args.database or location_database()
    store = LocationStore(database)
    model = YOLO(str(args.model))
    if model.names != EXPECTED_CLASSES:
        raise RuntimeError(f"Model classes must be {EXPECTED_CLASSES}, but found {model.names}")

    settings = alignment_settings()
    trackers = build_trackers()
    for tracker in trackers:
        tracker.reader = StreamReader(source=tracker.rtsp_url, name=tracker.name)
        tracker.reader.start()

    print(f"Recording locations to {database}. Press Ctrl+C to stop.")
    active_visits: dict[str, ActiveVisit] = {}
    try:
        while True:
            loop_start = time.monotonic()
            now = time.time()
            observations = sample_cameras(trackers, model, args)
            apply_observations(store, active_visits, best_observation_per_cat(observations), now)
            for tracker in trackers:
                maybe_reanchor(tracker, now, settings)
            remaining = args.interval - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        for visit in active_visits.values():
            close_visit(store, visit, visit.last_seen_ts, "tracker stopped")
        for tracker in trackers:
            if tracker.reader is not None:
                tracker.reader.close()
        store.close()


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
