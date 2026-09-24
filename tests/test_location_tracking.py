"""Tests for zone lookup, box filtering, dwell segments, and daily aggregation."""

import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring import location_config, location_report, location_tracker
from src.monitoring.alignment import AlignmentTracker
from src.monitoring.camera_discovery import (
    DiscoveredCamera,
    DiscoveryResult,
    DiscoverySettings,
    HostEvidence,
)
from src.monitoring.detections import bottom_center, select_best_per_class
from src.monitoring.location_store import LocationStore
from src.monitoring.location_zones import Calibration, Zone, find_zone
from src.monitoring.recalibration import ReanchorOutcome

ROOM = Zone("living_room", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
SOFA = Zone("sofa", [(0.1, 0.6), (0.4, 0.6), (0.4, 0.95), (0.1, 0.95)])
DAY_START = datetime(2026, 9, 20, 9, 0, tzinfo=ZoneInfo("Asia/Tokyo")).timestamp()


def test_find_zone_prefers_the_smallest_containing_polygon() -> None:
    assert find_zone([ROOM, SOFA], 0.25, 0.8).name == "sofa"
    assert find_zone([ROOM, SOFA], 0.8, 0.3).name == "living_room"
    assert find_zone([SOFA], 0.9, 0.1) is None


class FakeReader:
    """Enough of StreamReader for maybe_reanchor, which only wants a fresh frame."""

    def __init__(self, frame) -> None:
        self.frame = frame

    def get_latest_frame(self, *args, **kwargs):
        return self.frame


@pytest.fixture
def trigger_config(monkeypatch):
    """A controlled re-anchor config, so these tests do not read locations.yaml."""
    config = {
        "reanchor_trigger": "displacement",
        "reanchor_after_displacement_samples": 3,
        "reanchor_after_failures": 6,
        "reanchor_after_degraded_samples": 12,
        "reanchor_min_interval_seconds": 600.0,
    }
    monkeypatch.setattr(location_tracker, "ALIGNMENT_CONFIG", config)
    return config


def a_tracker(alignment) -> location_tracker.CameraTracker:
    return location_tracker.CameraTracker(
        name="sofa",
        rtsp_url="rtsp://example/stream",
        calibration=Calibration(camera="sofa", zones=[SOFA], calibration_id="sofa-1"),
        alignment=alignment,
    )


def test_a_collapsed_match_does_not_ask_for_a_reanchor(trigger_config) -> None:
    """The old trigger waited for matching to fail, which guaranteed failure.

    Re-anchoring puts the zones onto a new frame by matching against the old
    reference, so it needs the very match that just collapsed. On these cameras a
    collapse means the light changed - not that anything moved - so those retries
    could never succeed. The log bears it out: a whole day of "automatic re-anchor
    failed: only N feature matches" every ten minutes, and not one success.
    """
    tracker = a_tracker(AlignmentTracker(camera="sofa", consecutive_failures=99))

    assert location_tracker.reanchor_reason(tracker) is None


def test_a_large_transform_asks_for_a_reanchor(trigger_config) -> None:
    alignment = AlignmentTracker(camera="sofa", displacement_streak=2)
    tracker = a_tracker(alignment)

    assert location_tracker.reanchor_reason(tracker) is None, "two samples is not a move yet"

    alignment.displacement_streak = 3
    alignment.last_magnitude = (0.081, 2.4, 0.011)
    reason = location_tracker.reanchor_reason(tracker)

    assert reason is not None
    assert "shift 0.081" in reason, "the log line has to say how big the move was"


def test_the_two_older_trigger_modes_are_still_available(trigger_config) -> None:
    tracker = a_tracker(AlignmentTracker(camera="sofa", consecutive_failures=99))
    tracker.degraded_streak = 99

    assert location_tracker.reanchor_reason(tracker) is None, "not in displacement mode"

    trigger_config["reanchor_trigger"] = "failures"
    assert "99 consecutive failed" in location_tracker.reanchor_reason(tracker)

    trigger_config["reanchor_trigger"] = "degraded"
    assert "99 consecutive samples" in location_tracker.reanchor_reason(tracker)


def test_a_camera_without_alignment_is_never_reanchored(trigger_config) -> None:
    assert location_tracker.reanchor_reason(a_tracker(None)) is None


def test_reanchor_failures_are_reported_once_until_a_success(trigger_config, monkeypatch) -> None:
    """A camera that cannot be re-anchored must not alert on every retry.

    Retrying is still worth it - the camera may be nudged back, or the match may
    recover once the light settles - so only the first failure of a streak notifies,
    and a success re-arms the alert.
    """
    tracker = a_tracker(AlignmentTracker(camera="sofa", displacement_streak=3))
    tracker.reader = FakeReader(object())

    asked_to_notify_on_failure: list[bool] = []

    def failing_reanchor(calibration, frame, settings, now=None, notify_on_failure=True):
        asked_to_notify_on_failure.append(notify_on_failure)
        return ReanchorOutcome(False, None, "automatic re-anchor failed", None)

    def succeed(calibration, frame, settings, now=None, notify_on_failure=True):
        asked_to_notify_on_failure.append(notify_on_failure)
        return ReanchorOutcome(True, calibration, "projected", None)

    monkeypatch.setattr(location_tracker, "reanchor", failing_reanchor)
    monkeypatch.setattr(location_tracker, "reference_features_for", lambda *args: None)
    settings = {"work_width": 640}

    tracker.alignment.displacement_streak = 3
    location_tracker.maybe_reanchor(tracker, 1000.0, settings)
    assert asked_to_notify_on_failure == [True], "the first failure is news"
    assert tracker.reanchor_failure_reported

    tracker.alignment.displacement_streak = 3
    location_tracker.maybe_reanchor(tracker, 5000.0, settings)
    assert asked_to_notify_on_failure == [True, False], "the next retry is not"

    monkeypatch.setattr(location_tracker, "reanchor", succeed)
    tracker.alignment.displacement_streak = 3
    location_tracker.maybe_reanchor(tracker, 9000.0, settings)
    # The success still notifies inside reanchor() - the flag only ever gates
    # failures - and it re-arms the alert for the next genuine failure.
    assert not tracker.reanchor_failure_reported, "a success re-arms the alert"

    monkeypatch.setattr(location_tracker, "reanchor", failing_reanchor)
    tracker.alignment.displacement_streak = 3
    location_tracker.maybe_reanchor(tracker, 20000.0, settings)
    assert asked_to_notify_on_failure[-1] is True, "so the next failure is news again"


def test_the_reanchor_mode_defaults_to_applying_the_repair(trigger_config) -> None:
    """A config with no mode key keeps the old behaviour, so nothing changes quietly."""
    assert location_tracker.reanchor_mode() == "apply"


def test_alert_mode_reports_a_drift_without_touching_the_zones(trigger_config, monkeypatch) -> None:
    """Alert mode exists because a bad projection destroys hand-drawn work.

    Measured once for real: a collapsed match reporting ``scale=0.000`` rewrote all
    eight of feeder's zones as a single point at (0, 0), and every polygon had to be
    redrawn by hand. Reporting the drift loses none of the evidence and leaves the
    repair to the owner.
    """
    trigger_config["reanchor_mode"] = "alert"
    tracker = a_tracker(AlignmentTracker(camera="sofa", displacement_streak=3))
    tracker.alignment.last_magnitude = (0.081, 2.4, 0.011)
    tracker.reader = FakeReader(object())
    drawn = list(tracker.calibration.zones)
    reported: list[str] = []

    def must_not_run(*args, **kwargs):
        raise AssertionError("alert mode must not re-anchor")

    monkeypatch.setattr(location_tracker, "reanchor", must_not_run)
    monkeypatch.setattr(
        location_tracker,
        "announce_drift",
        lambda calibration, settings, reason: reported.append(reason) or "sent",
    )

    location_tracker.maybe_reanchor(tracker, 1000.0, {"work_width": 640})

    assert len(reported) == 1, "the drift is news, so it is reported"
    assert "shift 0.081" in reported[0], "with the magnitude that produced it"
    assert tracker.calibration.zones == drawn, "the drawing is the one that was drawn"


def test_a_drift_is_reported_once_until_the_camera_lines_up_again(
    trigger_config, monkeypatch
) -> None:
    """One message per drift, not one per sampling interval.

    A camera left where it is would otherwise repeat itself every few seconds, and a
    warning that repeats stops being read. What re-arms it is the drift going away.
    """
    trigger_config["reanchor_mode"] = "alert"
    tracker = a_tracker(AlignmentTracker(camera="sofa", displacement_streak=3))
    tracker.reader = FakeReader(object())
    reported: list[str] = []
    monkeypatch.setattr(location_tracker, "reanchor", lambda *args, **kwargs: pytest.fail())
    monkeypatch.setattr(
        location_tracker,
        "announce_drift",
        lambda calibration, settings, reason: reported.append(reason) or "sent",
    )
    settings = {"work_width": 640}

    location_tracker.maybe_reanchor(tracker, 1000.0, settings)
    # Well past the re-anchor cooldown, and still quiet: the flag suppresses this,
    # not the clock.
    location_tracker.maybe_reanchor(tracker, 9000.0, settings)
    assert len(reported) == 1

    tracker.alignment.displacement_streak = 0
    location_tracker.maybe_reanchor(tracker, 17000.0, settings)

    tracker.alignment.displacement_streak = 3
    location_tracker.maybe_reanchor(tracker, 18000.0, settings)
    assert len(reported) == 2, "the next drift is news again"


def test_an_unknown_reanchor_mode_is_rejected_before_the_tracker_starts(monkeypatch) -> None:
    """A typo must not quietly pick a side: one of these modes rewrites your zones."""
    monkeypatch.setattr(location_config, "ALIGNMENT_CONFIG", {"reanchor_mode": "notice"})

    with pytest.raises(ValueError, match="reanchor_mode"):
        location_config.alignment_settings()


def test_zone_area_is_used_for_specificity() -> None:
    assert SOFA.area < ROOM.area


def test_select_best_per_class_keeps_one_box_per_class() -> None:
    boxes = [
        (0, 0.40, (0, 0, 10, 10)),
        (0, 0.88, (50, 50, 60, 60)),
        (1, 0.60, (100, 100, 110, 110)),
        (1, 0.20, (0, 0, 5, 5)),
    ]
    kept = select_best_per_class(boxes, confidence_threshold=0.5)
    assert sorted(class_id for class_id, _, _ in kept) == [0, 1]
    assert dict((class_id, confidence) for class_id, confidence, _ in kept) == {0: 0.88, 1: 0.60}


def test_select_best_per_class_drops_the_weaker_cross_class_duplicate() -> None:
    overlapping = [(0, 0.55, (0, 0, 100, 100)), (1, 0.70, (0, 0, 100, 100))]
    kept = select_best_per_class(overlapping, confidence_threshold=0.5)
    assert len(kept) == 1
    assert kept[0][:2] == (1, 0.70)

    separate = [(0, 0.55, (0, 0, 100, 100)), (1, 0.70, (90, 90, 200, 200))]
    assert len(select_best_per_class(separate, confidence_threshold=0.5)) == 2


def test_bottom_center_is_used_as_the_zone_anchor() -> None:
    assert bottom_center((0.0, 0.0, 10.0, 20.0)) == (5.0, 20.0)


@pytest.fixture
def store(tmp_path):
    location_store = LocationStore(tmp_path / "history.db")
    yield location_store
    location_store.close()


def _observe(store, active, offset, camera, zone, confidence=0.8) -> None:
    location_tracker.apply_observations(
        store,
        active,
        {"bagel": location_tracker.Observation("bagel", camera, zone, confidence)},
        DAY_START + offset,
    )


def test_visit_holds_through_one_noisy_sample(store) -> None:
    active: dict = {}
    _observe(store, active, 0, "living_room", "sofa")
    _observe(store, active, 10, "living_room", "sofa")
    _observe(store, active, 20, "living_room", None)
    assert active["bagel"].zone == "sofa"


def test_visit_switches_after_repeated_agreement(store) -> None:
    active: dict = {}
    _observe(store, active, 0, "living_room", "sofa")
    _observe(store, active, 10, "living_room", "sofa")
    _observe(store, active, 20, "living_room", "carpet")
    _observe(store, active, 30, "living_room", "carpet")
    assert active["bagel"].zone == "carpet"

    visits = store.visits_between(DAY_START - 1, DAY_START + 100)
    assert len(visits) == 2
    assert [visit.zone for visit in visits] == ["sofa", "carpet"]
    # The new visit starts when the switch was first seen, so no time is lost.
    assert visits[1].start_ts == DAY_START + 20


def test_visit_closes_after_the_missing_timeout(store) -> None:
    active: dict = {}
    _observe(store, active, 0, "living_room", "sofa")
    location_tracker.apply_observations(store, active, {}, DAY_START + 200)
    assert not active

    visits = store.visits_between(DAY_START - 1, DAY_START + 400)
    assert len(visits) == 1
    assert visits[0].end_ts == DAY_START


def test_zone_is_optional_and_falls_back_to_the_camera_name(store) -> None:
    active: dict = {}
    _observe(store, active, 0, "living_room", None)
    visit = store.visits_between(DAY_START - 1, DAY_START + 10)[0]
    assert visit.zone is None
    assert visit.location == "living_room"


def test_daily_summary_aggregates_and_sorts_dwell_time(store, monkeypatch) -> None:
    location_report.TRACKING_CONFIG["database"] = str(store.path)
    monkeypatch.setattr(location_report, "resolve_config_path", lambda value: Path(value))

    active: dict = {}
    for offset in range(0, 120, 10):
        _observe(store, active, offset, "living_room", "sofa")
    for offset in range(140, 180, 10):
        _observe(store, active, offset, "living_room", "sofa", confidence=0.9)
    for offset in range(200, 240, 10):
        _observe(store, active, offset, "living_room", "carpet")
    location_tracker.apply_observations(store, active, {}, DAY_START + 600)

    summary = location_report.build_summary(date(2026, 9, 20))
    assert summary["date"] == "2026-09-20"
    bagel = summary["cats"][0]
    assert bagel["cat"] == "bagel"
    assert bagel["locations"][0]["location"] == "sofa"

    text = location_report.render_text(summary)
    assert "sofa" in text
    assert "bagel" in text


# --- a camera that moved on the network -------------------------------------------
#
# A camera that is offline fails differently from a camera that drifted: alignment has
# nothing to say about it, and the reader would retry the dead address forever. These
# tests cover the rules that keep the recovery quiet enough to be worth reading - one
# message per outage, a search that only starts after a delay, and a new address
# adopted without restarting the tracker.


class FakeReaderState:
    """Only the part of StreamReader the health check looks at."""

    def __init__(self, connected: bool) -> None:
        self.connected = connected


class FakeStreamReader:
    """A StreamReader that connects to nothing."""

    instances: list = []

    def __init__(self, source, name="stream") -> None:
        self.source = source
        self.name = name
        self.started = False
        self.closed = False
        FakeStreamReader.instances.append(self)

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True


class FakeThread:
    """A thread that records that it was created and never runs."""

    started: list = []

    def __init__(self, target=None, args=(), name="", daemon=False) -> None:
        self.target = target
        self.args = args
        self.name = name
        FakeThread.started.append(self)

    def start(self) -> None:
        pass

    def is_alive(self) -> bool:
        return False


def recording_notify(sent: list):
    def notify(settings, camera, content, attachment):
        sent.append(content)
        return True, "sent"

    return notify


def a_movable_tracker() -> location_tracker.CameraTracker:
    return location_tracker.CameraTracker(
        name="feeder",
        rtsp_url="rtsp://admin:pw@192.168.3.13:554/h264/ch1/main/av_stream",
        calibration=Calibration(camera="feeder", zones=[SOFA], calibration_id="feeder-1"),
    )


def test_an_unreachable_camera_is_timed_from_the_first_failed_sample() -> None:
    tracker = a_movable_tracker()
    tracker.reader = FakeReaderState(connected=False)
    location_tracker.note_connection_state(tracker, 100.0, {})
    assert tracker.unreachable_since == 100.0
    location_tracker.note_connection_state(tracker, 200.0, {})
    assert tracker.unreachable_since == 100.0, "the clock starts once, not every sample"


def test_a_camera_that_answers_again_is_announced_once_and_re_arms_the_alert() -> None:
    sent: list[str] = []
    notify = recording_notify(sent)
    tracker = a_movable_tracker()
    tracker.reader = FakeReaderState(connected=False)
    location_tracker.note_connection_state(tracker, 100.0, {}, notify)
    assert sent == [], "still connected, so there is nothing to say"

    tracker.outage_reported = True
    tracker.relocated_from, tracker.relocated_to = "192.168.3.13", "192.168.3.59"
    tracker.reader = FakeReaderState(connected=True)
    location_tracker.note_connection_state(tracker, 200.0, {}, notify)
    assert len(sent) == 1
    assert "192.168.3.59" in sent[0], "the new address is the useful part of the news"
    assert tracker.outage_reported is False, "so the next outage is news again"

    location_tracker.note_connection_state(tracker, 300.0, {}, notify)
    assert len(sent) == 1, "a camera that is up says nothing more"


def test_the_search_waits_for_the_delay_and_then_respects_its_cooldown(monkeypatch) -> None:
    monkeypatch.setattr(location_tracker.threading, "Thread", FakeThread)
    FakeThread.started.clear()
    tracker = a_movable_tracker()
    tracker.reader = FakeReaderState(connected=False)
    discovery = DiscoverySettings(missing_after_seconds=120.0, retry_seconds=300.0)

    location_tracker.maybe_recover_camera(tracker, 1000.0, {}, discovery)
    assert FakeThread.started == [], "no failed sample yet, so no clock to wait out"

    location_tracker.note_connection_state(tracker, 1000.0, {})
    location_tracker.maybe_recover_camera(tracker, 1100.0, {}, discovery)
    assert FakeThread.started == [], "100s down is under the 120s delay"

    location_tracker.maybe_recover_camera(tracker, 1200.0, {}, discovery)
    assert len(FakeThread.started) == 1

    tracker.discovery_thread = None  # as if the search had finished
    location_tracker.maybe_recover_camera(tracker, 1300.0, {}, discovery)
    assert len(FakeThread.started) == 1, "and not again inside the cooldown"
    location_tracker.maybe_recover_camera(tracker, 1600.0, {}, discovery)
    assert len(FakeThread.started) == 2


def test_a_camera_that_just_flapped_keeps_its_address_protected() -> None:
    """A camera on a link that is dropping frames is still a camera in use.

    Measured on this network: the relay stalls all three streams for about thirty
    seconds every few minutes. A search that raced those flaps would open streams on
    cameras that are mid-reconnect, on the link that is already struggling.
    """
    discovery = DiscoverySettings(missing_after_seconds=120.0)
    recording = a_movable_tracker()
    recording.reader = FakeReaderState(connected=True)
    flapping = a_movable_tracker()
    flapping.name = "sofa"
    flapping.rtsp_url = "rtsp://admin:pw@192.168.3.48:554/h264/ch1/main/av_stream"
    flapping.reader = FakeReaderState(connected=False)
    gone = a_movable_tracker()
    gone.name = "living_room"
    gone.rtsp_url = "rtsp://admin:pw@192.168.3.57:554/h264/ch1/main/av_stream"
    gone.reader = FakeReaderState(connected=False)

    location_tracker.note_connection_state(flapping, 1000.0, {})
    location_tracker.note_connection_state(gone, 1000.0, {})

    assert location_tracker.recording_hosts([recording, flapping], 1040.0, discovery) == {
        "192.168.3.13",
        "192.168.3.48",
    }, "40 seconds down is a flap, not a camera that moved"
    assert location_tracker.recording_hosts([recording, gone], 1400.0, discovery) == {
        "192.168.3.13"
    }, "past the delay the address is fair game, which is how a moved camera is found"


def test_a_disabled_search_never_runs(monkeypatch) -> None:
    monkeypatch.setattr(location_tracker.threading, "Thread", FakeThread)
    FakeThread.started.clear()
    tracker = a_movable_tracker()
    tracker.reader = FakeReaderState(connected=False)
    location_tracker.note_connection_state(tracker, 1000.0, {})
    location_tracker.maybe_recover_camera(
        tracker, 99999.0, {}, DiscoverySettings(enabled=False, missing_after_seconds=0.0)
    )
    assert FakeThread.started == []


def test_a_camera_found_elsewhere_is_relocated_without_a_restart(monkeypatch) -> None:
    FakeStreamReader.instances.clear()
    monkeypatch.setattr(location_tracker, "StreamReader", FakeStreamReader)
    # The real one would edit this repo's .env, which is the point of it - but not
    # from a test. Its own behaviour is covered below.
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        location_tracker,
        "remember_address",
        lambda camera, host: recorded.append((camera, host)),
    )
    tracker = a_movable_tracker()
    dead_reader = FakeStreamReader("rtsp://admin:pw@192.168.3.13:554/x", "feeder")
    tracker.reader = dead_reader

    found = DiscoveredCamera(
        camera="feeder",
        host="192.168.3.59",
        url="rtsp://admin:pw@192.168.3.59:554/h264/ch1/main/av_stream",
        evidence=HostEvidence("192.168.3.59", "feeder", credential_ok=True, matches={"feeder": 79}),
        previous_host="192.168.3.13",
    )
    monkeypatch.setattr(
        location_tracker, "discover", lambda **kwargs: DiscoveryResult(found={"feeder": found})
    )

    location_tracker.recover_camera(tracker, {}, DiscoverySettings())

    assert tracker.rtsp_url.endswith("192.168.3.59:554/h264/ch1/main/av_stream")
    assert tracker.reader is not dead_reader
    assert tracker.reader.started and not tracker.reader.closed
    assert dead_reader.closed, "the thread retrying the dead address has to be stopped"
    assert tracker.relocated_to == "192.168.3.59"
    assert recorded == [("feeder", "192.168.3.59")], "so a restart comes up here too"


def test_the_new_address_is_written_back_to_env(tmp_path) -> None:
    """A restart has to come up on the address that was just adopted."""
    env = tmp_path / ".env"
    env.write_text(
        "FEEDER_RTSP_URL='rtsp://admin:pw@192.168.3.13:554/h264/ch1/main/av_stream'\n",
        encoding="utf-8",
    )

    location_tracker.remember_address(
        "feeder", "192.168.3.59", env_path=env, variables={"feeder": "FEEDER_RTSP_URL"}
    )

    assert "192.168.3.59" in env.read_text(encoding="utf-8")
    assert (tmp_path / ".env.bak").exists()


def test_a_camera_with_no_placeholder_leaves_env_alone(tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_text("FEEDER_RTSP_URL='rtsp://admin:pw@192.168.3.13:554/x'\n", encoding="utf-8")

    location_tracker.remember_address("feeder", "192.168.3.59", env_path=env, variables={})

    assert "192.168.3.13" in env.read_text(encoding="utf-8")


def test_an_unwritable_env_does_not_raise(tmp_path) -> None:
    """Losing the samples a relocation just restored would be the worse failure."""
    location_tracker.remember_address(
        "feeder",
        "192.168.3.59",
        env_path=tmp_path / "missing" / ".env",
        variables={"feeder": "FEEDER_RTSP_URL"},
    )


def test_a_camera_that_cannot_be_found_is_reported_once_per_outage(monkeypatch) -> None:
    sent: list[str] = []
    notify = recording_notify(sent)
    tracker = a_movable_tracker()
    monkeypatch.setattr(
        location_tracker, "discover", lambda **kwargs: DiscoveryResult(missing=["feeder"])
    )

    location_tracker.recover_camera(tracker, {}, DiscoverySettings(), notify)
    assert len(sent) == 1
    assert "192.168.3.0/24" in sent[0], "the owner is told what was searched"
    assert tracker.outage_reported is True

    location_tracker.recover_camera(tracker, {}, DiscoverySettings(), notify)
    assert len(sent) == 1, "a camera that stays missing is not worth repeating"

    tracker.reader = FakeReaderState(connected=True)
    location_tracker.note_connection_state(tracker, 5000.0, {}, notify)
    assert len(sent) == 2, "it came back, which is news"
    assert tracker.outage_reported is False

    tracker.reader = FakeReaderState(connected=False)
    location_tracker.note_connection_state(tracker, 6000.0, {})
    location_tracker.recover_camera(tracker, {}, DiscoverySettings(), notify)
    assert len(sent) == 3, "a new outage gets its own warning"
