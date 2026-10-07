"""Sample the cameras periodically and record where each cat is staying."""

import argparse
import signal
import sys
import threading
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

from src.config_loader import load_env_file

# Before the monitoring imports below: they read the config, and the config substitutes
# ${VAR} from the environment at import time.
load_env_file()

from src.monitoring import alerting
from src.monitoring.alignment import GOOD, AlignmentTracker, transform_point
from src.monitoring.camera_discovery import (
    DiscoveryResult,
    DiscoverySettings,
    apply_hosts_to_env,
    camera_env_vars,
    discover,
    discovery_settings,
    host_of,
)
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
    DEFAULT_REANCHOR_MODE,
    IDENTITY_CLASSES,
    LOCATION_CONFIG,
    PROJECT_ROOT,
    REANCHOR_ALERT,
    TRACKING_CONFIG,
    alignment_settings,
    camera_rtsp_url,
    configured_cameras,
    location_database,
)
from src.monitoring.location_store import LocationStore
from src.monitoring.location_zones import (
    Calibration,
    ZoneVote,
    load_calibrations,
    vote_zone,
)
from src.monitoring.recalibration import announce_drift, reanchor, reference_features_for
from src.monitoring.rtsp_stream import StreamReader

EXPECTED_CLASSES = IDENTITY_CLASSES

# How long the alignment has to stay good before a reported drift is re-armed. Long
# enough to cover the night-time flicker described in note_alignment_restored(), and
# short enough that a camera someone actually put back is not mute for a day.
DEFAULT_ALERT_CLEAR_SECONDS = 600.0


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
    # Normalized detection box, kept so a future anchor strategy can be applied to
    # this history instead of only to new samples.
    box: tuple[float, float, float, float] | None = None
    # (points that agreed, points sampled) - 9/9 is confident, 5/9 is not.
    vote: tuple[int, int] | None = None
    # The anchor in the camera's own frame, before alignment, so "who was near this
    # spot" can be asked with the coordinates the owner reads off the screen.
    camera_anchor: tuple[float, float] | None = None
    # (width, height) in pixels, to turn those fractions into pixels and back.
    frame_size: tuple[int, int] | None = None


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
    # Set after a failed re-anchor so the same failure is reported only once. A
    # successful re-anchor clears it and re-arms the alert.
    reanchor_failure_reported: bool = False
    # One message per drift while re-anchoring only reports. Cleared once the camera
    # has lined up with its reference frame for a stretch, which re-arms the alert.
    drift_reported: bool = False
    # When that clean stretch started, 0 while it is not running. A wall-clock stamp,
    # like the other cooldowns here, because "has it been ten minutes" is what the
    # owner cares about rather than how many samples that was.
    drift_cleared_since: float = 0.0
    # Monotonic timestamp of the first failed sample of a run, 0 while the camera is
    # answering. The clock is monotonic so a system clock jump cannot fake an outage.
    unreachable_since: float = 0.0
    last_discovery_at: float = 0.0
    discovery_thread: threading.Thread | None = None
    # One message per outage, not one per retry, and re-armed by the recovery.
    outage_reported: bool = False
    # Where the camera was found when it had to be relocated, for the recovery note.
    relocated_to: str = ""
    relocated_from: str = ""

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
                on_failure=settings["on_alignment_failure"],
                min_shift=settings["min_shift"],
                min_rotation_deg=settings["min_rotation_deg"],
                min_scale_delta=settings["min_scale_delta"],
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
        strategy = getattr(args, "anchor_strategy", None) or TRACKING_CONFIG.get(
            "anchor_strategy", BOTTOM_CENTER
        )
        grid = int(
            getattr(args, "anchor_grid", None) or TRACKING_CONFIG.get("anchor_grid", DEFAULT_GRID)
        )
        for class_id, confidence, box in boxes:
            # norm_x/norm_y keep their historical meaning - the transformed
            # bottom-centre anchor - so old and new rows stay comparable.
            anchor_x, anchor_y = bottom_center(box)
            normal_box = (
                box[0] / image_width,
                box[1] / image_height,
                box[2] / image_width,
                box[3] / image_height,
            )
            norm_x, norm_y = anchor_x / image_width, anchor_y / image_height
            # Kept before the alignment transform below overwrites norm_x/norm_y.
            camera_point = (norm_x, norm_y)
            candidates = [
                (x / image_width, y / image_height) for x, y in anchor_points(box, strategy, grid)
            ]
            vote = ZoneVote(None, 0, len(candidates))
            if state is None or state.zone_lookup_allowed:
                if state is not None and state.matrix is not None:
                    norm_x, norm_y = transform_point(state.matrix, (norm_x, norm_y))
                    candidates = [transform_point(state.matrix, point) for point in candidates]
                vote = vote_zone(tracker.calibration.zones, candidates)
            observations.append(
                Observation(
                    cat=EXPECTED_CLASSES[class_id],
                    camera=tracker.name,
                    zone=vote.zone,
                    confidence=confidence,
                    norm_x=norm_x,
                    norm_y=norm_y,
                    calibration_id=tracker.calibration.calibration_id or None,
                    alignment_quality=state.quality if state is not None else None,
                    box=normal_box,
                    vote=(vote.matches, vote.samples),
                    camera_anchor=camera_point,
                    frame_size=(image_width, image_height),
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


def describe_location(camera: str, zone: str | None) -> str:
    """``sofa/carpet``, or ``feeder/unknown`` when no zone matched.

    The camera belongs in the string. A cat crossing feeder -> sofa -> living room
    reports the same zone name (``floor``) three times, so a zone-only label made a
    real move read as "floor for 35s -> moved to floor", which looks like nothing
    happened.
    """
    return f"{camera}/{zone or 'unknown'}"


def close_visit(store: LocationStore, visit: ActiveVisit, end_ts: float, reason: str) -> None:
    if end_ts <= visit.start_ts:
        end_ts = visit.start_ts
    store.touch_visit(visit.visit_id, end_ts, visit.samples, visit.max_confidence)
    duration = end_ts - visit.start_ts
    print(
        f"[{visit.cat}] {describe_location(visit.camera, visit.zone)} for {duration:.0f}s "
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
            observation.box,
            observation.vote,
            observation.camera_anchor,
            observation.frame_size,
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
            print(f"[{cat}] now at {describe_location(observation.camera, observation.zone)}")
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

        close_visit(
            store,
            visit,
            visit.pending_since_ts,
            f"moved to {describe_location(key[0], key[1])}",
        )
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
        "--anchor-strategy",
        choices=ANCHOR_STRATEGIES,
        default=TRACKING_CONFIG.get("anchor_strategy", BOTTOM_CENTER),
        help="How a detection box becomes the point(s) zone lookup matches against.",
    )
    parser.add_argument(
        "--anchor-grid",
        type=int,
        default=int(TRACKING_CONFIG.get("anchor_grid", DEFAULT_GRID)),
        help="Grid size per side for --anchor-strategy lower_grid.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        dest="camera_names",
        help="Restrict tracking to the given camera(s); repeat for multiple cameras.",
    )
    return parser.parse_args(argv)


def reanchor_reason(tracker: CameraTracker) -> str | None:
    """Why this camera should adopt a fresh reference frame, or None.

    The trigger has to be evidence that the *camera* moved, because that is the only
    thing re-anchoring can fix. ``displacement`` is the default and the only mode
    that fires while it can still work: projecting the zones onto a new frame needs
    the very match that alignment produces, so a trigger that waits for matching to
    collapse guarantees every attempt fails. That is exactly what the old
    ``failures`` trigger did - living_room and feeder logged "automatic re-anchor
    failed: only N feature matches" every cooldown for a whole day without a single
    success, because an illumination change is what produced those failures and an
    illumination change is not something a re-anchor can repair.

    The other two modes are kept for a room where they fit better:

    ``degraded``  a long run of marginal-but-usable matches, whatever their size.
    ``failures``  the old consecutive-FAILED counter. Counted either way, so the
                  number stays available for logs and for this mode.
    """
    alignment = tracker.alignment
    if alignment is None:
        return None
    trigger = ALIGNMENT_CONFIG.get("reanchor_trigger", "displacement")
    if trigger == "failures":
        threshold = ALIGNMENT_CONFIG.get("reanchor_after_failures", 6)
        if alignment.consecutive_failures >= threshold:
            return f"{alignment.consecutive_failures} consecutive failed matches"
        return None
    if trigger == "degraded":
        threshold = ALIGNMENT_CONFIG.get("reanchor_after_degraded_samples", 12)
        if tracker.degraded_streak >= threshold:
            return f"{tracker.degraded_streak} consecutive samples below the good tier"
        return None
    threshold = ALIGNMENT_CONFIG.get("reanchor_after_displacement_samples", 3)
    if alignment.displacement_streak >= threshold:
        shift, rotation, scale_delta = alignment.last_magnitude or (0.0, 0.0, 0.0)
        return (
            f"the transform has stayed large for {alignment.displacement_streak} samples "
            f"(shift {shift:.3f}, rotation {rotation:.2f}deg, scale {scale_delta:.3f})"
        )
    return None


def reanchor_mode() -> str:
    """``apply`` re-anchors a drifted camera; ``alert`` only reports the drift.

    Read from the config rather than decided here, because which one is right depends
    on how much the projection can be trusted in that room. On these cameras it was
    the repair, not the drift, that cost the most work.
    """
    return ALIGNMENT_CONFIG.get("reanchor_mode", DEFAULT_REANCHOR_MODE)


def report_drift(tracker: CameraTracker, settings: dict, reason: str) -> None:
    """Warn that a camera drifted, once, and change nothing else.

    The flag is set before sending, so a webhook that is briefly unreachable cannot
    turn one drift into a message per sampling interval. What re-arms it is a lasting
    recovery, not a quiet sample; see ``note_alignment_restored``.
    """
    if tracker.drift_reported:
        return
    tracker.drift_reported = True
    tracker.drift_cleared_since = 0.0
    note = announce_drift(tracker.calibration, settings, reason)
    print(f"[{tracker.name}] drift reported: {reason}; {note}")


def note_alignment_restored(tracker: CameraTracker, now: float) -> None:
    """Re-arm the drift alert, but only once the alignment has really recovered.

    One clean sample is not a recovery. At night the match flickers between the two
    states - three samples read as large, one lines up, three read as large again -
    so clearing the flag on that one sample re-armed the alert every few samples.
    Measured on 2026-09-26: sofa sent thirty-two messages between 01:09 and 01:47,
    at intervals as short as thirty seconds, every one of them describing the same
    drift. Requiring the clean stretch to last is what makes "one message per drift"
    true; ``alignment.reanchor_alert_clear_seconds`` is how long it has to last.

    A failed or marginal sample is not a recovery either: the streak reads zero
    because there was nothing to measure, not because the camera is back in place.
    """
    if tracker.alignment is None or tracker.alignment.quality != GOOD:
        tracker.drift_cleared_since = 0.0
        return
    if not tracker.drift_reported:
        tracker.drift_cleared_since = 0.0
        return
    if not tracker.drift_cleared_since:
        tracker.drift_cleared_since = now
        return
    if now - tracker.drift_cleared_since >= ALIGNMENT_CONFIG.get(
        "reanchor_alert_clear_seconds", DEFAULT_ALERT_CLEAR_SECONDS
    ):
        tracker.drift_reported = False
        tracker.drift_cleared_since = 0.0


def maybe_reanchor(tracker: CameraTracker, now: float, settings: dict) -> None:
    """Re-anchor a drifted camera, or - in alert mode - just say that it drifted."""
    reason = reanchor_reason(tracker)
    if reason is None:
        note_alignment_restored(tracker, now)
        return
    # Drift evidence is back, so any recovery in progress was not one.
    tracker.drift_cleared_since = 0.0
    if reanchor_mode() == REANCHOR_ALERT:
        report_drift(tracker, settings, reason)
        return
    cooldown = ALIGNMENT_CONFIG.get("reanchor_min_interval_seconds", 600)
    if now - tracker.last_reanchor_at < cooldown:
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

    print(f"[{tracker.name}] re-anchor triggered: {reason}")

    outcome = reanchor(
        tracker.calibration,
        frame,
        settings,
        notify_on_failure=not tracker.reanchor_failure_reported,
    )
    print(f"[{tracker.name}] re-anchor: {outcome.message}")
    if outcome.ok and outcome.calibration is not None:
        tracker.reanchor_failure_reported = False
        tracker.calibration = outcome.calibration
        tracker.alignment.set_reference(
            reference_features_for(outcome.calibration, settings["work_width"])
        )
    else:
        tracker.reanchor_failure_reported = True


def recording_hosts(
    trackers: list[CameraTracker], now: float, discovery: DiscoverySettings
) -> set[str]:
    """Addresses the search must leave alone, because a camera is using them.

    A camera that merely flapped still counts: measured on this network, the relay
    drops all three streams for about thirty seconds every few minutes. A search that
    raced those flaps would open streams on cameras that are in the middle of
    reconnecting, on the same link that is already struggling. Only a camera that has
    been gone long enough to be searched for at all gives up its protection - and that
    is the camera being looked for, whose own address is excluded by the caller.
    """
    protected: set[str] = set()
    for tracker in trackers:
        if tracker.reader is None:
            continue
        down_for = now - tracker.unreachable_since if tracker.unreachable_since else 0.0
        if tracker.reader.connected or down_for < discovery.missing_after_seconds:
            protected.add(host_of(tracker.rtsp_url))
    return protected


def note_connection_state(tracker: CameraTracker, now: float, settings: dict, notify=None) -> None:
    """Track how long a camera has been unreachable, and report the turn-around once.

    A camera that is offline is a different failure from a camera that has drifted:
    alignment has nothing to say about it, and the reader would retry the same dead
    address forever. The owner hears about it the way they hear about a knocked
    camera - one LLM-worded Discord message - but only once per outage, because a
    camera behind a flaky relay would otherwise repeat itself every few minutes, and
    a warning that repeats stops being read.
    """
    reader = tracker.reader
    if reader is None:
        return
    if reader.connected:
        if tracker.outage_reported:
            report_camera_recovered(tracker, settings, notify)
            tracker.outage_reported = False
        tracker.unreachable_since = 0.0
        return
    # 0 means "never seen this camera down before", which is also how the retry
    # cooldown below reads its own timestamp. Treating it as a real timestamp would
    # suppress the first search for the first `retry_seconds` of uptime.
    if not tracker.unreachable_since:
        tracker.unreachable_since = now


def report_camera_missing(tracker: CameraTracker, settings: dict, subnets, notify=None) -> bool:
    """Tell the owner a camera is unreachable and could not be found again."""
    notify = notify or alerting.notify
    searched = ", ".join(subnets)
    plain = (
        f"\u26a0\ufe0f `{tracker.name}` 连不上了。\n"
        f"本鱼把 {searched} 扫了一遍，没有找到它。\n"
        f"可能是断电了，或者被中继换到了别的网段。请看一眼它的电源和中继。"
    )
    content = alerting.alert_content(
        "camera_missing",
        settings,
        {
            "摄像头": tracker.name,
            "发生了什么": "画面断了，而且在局域网里也找不到它",
            "已经试过": f"扫描了 {searched}",
            "可能的原因": "摄像头断电、被中继换到了别的网段，或者中继本身掉线",
            "需要主人做什么": "看一眼摄像头的电源和中继",
        },
        plain,
    )
    delivered, note = notify(settings, tracker.name, content, None)
    print(f"[{tracker.name}] camera missing: {note}")
    return delivered


def report_camera_recovered(tracker: CameraTracker, settings: dict, notify=None) -> None:
    """Tell the owner the camera is back. This is also what re-arms the alert."""
    notify = notify or alerting.notify
    moved = bool(tracker.relocated_to) and tracker.relocated_from != tracker.relocated_to
    facts = {"摄像头": tracker.name, "发生了什么": "画面恢复了", "需要主人做什么": "不用管"}
    if moved:
        plain = (
            f"\u2705 `{tracker.name}` 又连上了，"
            f"它从 {tracker.relocated_from} 换到了 {tracker.relocated_to}，本鱼已经自己接上。"
        )
        facts["新的地址"] = tracker.relocated_to
        facts["原来的地址"] = tracker.relocated_from
    else:
        plain = f"\u2705 `{tracker.name}` 又连上了，地址没变。"
    content = alerting.alert_content("camera_recovered", settings, facts, plain)
    delivered, note = notify(settings, tracker.name, content, None)
    print(f"[{tracker.name}] camera recovered: {note}")


def remember_address(
    camera: str,
    host: str,
    env_path: Path | None = None,
    variables: dict[str, str] | None = None,
) -> None:
    """Write a camera's new address into .env, so a restart uses it too.

    Best-effort on purpose: a failed write deserves a log line, never the samples the
    relocation just restored. The sync script fixes the file on its next run.
    """
    variable = (camera_env_vars() if variables is None else variables).get(camera)
    if variable is None:
        print(f"[{camera}] names no ${{VAR}} in config.yaml, so .env was not updated")
        return
    path = env_path if env_path is not None else PROJECT_ROOT / ".env"
    try:
        changes = apply_hosts_to_env(path, {variable: host})
    except OSError as error:
        print(f"[{camera}] could not update .env: {error}")
        return
    for key, (old_host, new_host) in changes.items():
        print(f"[{camera}] .env updated: {key} {old_host} -> {new_host}")


def relocate_camera(tracker: CameraTracker, host: str, url: str) -> None:
    """Point a camera at a new address and reconnect, without restarting the tracker.

    A fresh StreamReader rather than an assignment on the old one: the reader owns a
    thread that is retrying the dead address, and it has no other way to be told to
    stop or to change target.

    .env is rewritten here as well, and this is the ordinary path for it: the tracker
    is the only component that learns a camera moved without opening a second stream
    to find out, and the lookup on the timer deliberately stands down while the
    tracker is running so that it cannot compete with these recordings.
    """
    previous = tracker.reader
    tracker.relocated_from = host_of(tracker.rtsp_url)
    tracker.relocated_to = host
    tracker.rtsp_url = url
    tracker.reader = StreamReader(source=url, name=tracker.name)
    tracker.reader.start()
    if previous is not None:
        previous.close()
    tracker.unreachable_since = 0.0
    print(f"[{tracker.name}] answering again at {host} (was {tracker.relocated_from})")
    remember_address(tracker.name, host)


def recover_camera(
    tracker: CameraTracker,
    settings: dict,
    discovery: DiscoverySettings,
    notify=None,
    live_hosts: tuple[str, ...] = (),
) -> DiscoveryResult:
    """Search the network for this camera. Blocking; called from its own thread.

    Only this camera is looked for, so a healthy one can never be reported as missing
    merely because the search was told to leave its address alone. The picture check
    still compares against every camera's reference frame - that is local work on a
    frame which has already been grabbed - and ``live_hosts`` keeps the search off the
    addresses a camera is *already* being recorded from.
    """
    result = discover(cameras=[tracker.name], settings=discovery, skip_hosts=tuple(live_hosts))
    found = result.found.get(tracker.name)
    if found is None:
        if not tracker.outage_reported:
            report_camera_missing(tracker, settings, discovery.subnets, notify)
            tracker.outage_reported = True
        return result
    if found.host != host_of(tracker.rtsp_url):
        relocate_camera(tracker, found.host, found.url)
    # A recovery is announced when the stream actually answers again (see
    # note_connection_state), not here: an address that was found but still refuses to
    # open is not a recovery.
    return result


def maybe_recover_camera(
    tracker: CameraTracker,
    now: float,
    settings: dict,
    discovery: DiscoverySettings,
    notify=None,
    live_hosts: tuple[str, ...] = (),
) -> None:
    """Go looking for a camera that has been unreachable for long enough.

    In a thread, because this sweeps a /24 and opens a stream per candidate - seconds
    to tens of seconds, which inside the sampling loop would stall every other camera.
    """
    if not discovery.enabled or tracker.reader is None or not tracker.unreachable_since:
        return
    if now - tracker.unreachable_since < discovery.missing_after_seconds:
        return
    if tracker.discovery_thread is not None and tracker.discovery_thread.is_alive():
        return
    if tracker.last_discovery_at and now - tracker.last_discovery_at < discovery.retry_seconds:
        return
    tracker.last_discovery_at = now
    tracker.discovery_thread = threading.Thread(
        target=recover_camera,
        args=(tracker, settings, discovery, notify, live_hosts),
        name=f"discover-{tracker.name}",
        daemon=True,
    )
    tracker.discovery_thread.start()


def _raise_keyboard_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def run(args) -> None:
    if not args.model.is_file():
        raise FileNotFoundError(f"Identity detection model not found: {args.model}")
    if args.interval <= 0:
        raise ValueError("--interval must be greater than 0")

    # systemd's unit sends SIGINT (KillSignal=SIGINT), which Python already turns into
    # KeyboardInterrupt, so the except/finally below closes any open visit. launchd has
    # no equivalent per-job setting and stops a job with SIGTERM instead, so the same
    # path is wired up here too - otherwise a visit left open at shutdown never closes.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    if args.camera_names:
        selected = [name for name in configured_cameras() if name in args.camera_names]
        LOCATION_CONFIG["cameras"] = selected

    database = args.database or location_database()
    store = LocationStore(database)
    model = YOLO(str(args.model))
    if model.names != EXPECTED_CLASSES:
        raise RuntimeError(f"Model classes must be {EXPECTED_CLASSES}, but found {model.names}")

    settings = alignment_settings()
    discovery = discovery_settings()
    trackers = build_trackers()
    for tracker in trackers:
        tracker.reader = StreamReader(source=tracker.rtsp_url, name=tracker.name)
        tracker.reader.start()

    print(f"Recording locations to {database}. Press Ctrl+C to stop.")
    if reanchor_mode() == REANCHOR_ALERT:
        print(
            "Automatic re-anchoring is off (alignment.reanchor_mode = alert): a camera"
            " that drifts is reported to Discord once, and its zones are left as drawn."
        )
    else:
        print("Automatic re-anchoring is on (alignment.reanchor_mode = apply).")
    if discovery.enabled:
        print(
            "Camera discovery is on: a camera that stays unreachable for "
            f"{discovery.missing_after_seconds:.0f}s is searched for on "
            f"{', '.join(discovery.subnets)}, then every "
            f"{discovery.retry_seconds:.0f}s until it turns up."
        )
    active_visits: dict[str, ActiveVisit] = {}
    try:
        while True:
            loop_start = time.monotonic()
            now = time.time()
            for tracker in trackers:
                note_connection_state(tracker, loop_start, settings)
            # The search must not open a second stream on a camera that is being
            # recorded, or on one that is merely reconnecting: two concurrent lookups
            # alongside the tracker were measured to stall a 2560x1440 handshake past
            # 30 seconds and drop the tracker's own feeder stream.
            protected = recording_hosts(trackers, loop_start, discovery)
            for tracker in trackers:
                maybe_recover_camera(
                    tracker,
                    loop_start,
                    settings,
                    discovery,
                    live_hosts=tuple(protected - {host_of(tracker.rtsp_url)}),
                )
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
