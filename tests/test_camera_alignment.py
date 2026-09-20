"""Tests for camera drift alignment, re-anchoring, and store migration."""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring import alignment as alignment_module
from src.monitoring import recalibration
from src.monitoring.alignment import (
    DEGRADED,
    FAILED,
    GOOD,
    AlignmentResult,
    AlignmentTracker,
    build_reference_features,
    estimate_alignment,
    invert,
    transform_point,
    transform_points,
)
from src.monitoring.location_store import LocationStore
from src.monitoring.location_zones import (
    Calibration,
    Zone,
    load_calibrations,
    save_calibrations,
)

WIDTH, HEIGHT = 640, 480


def make_texture(seed: int = 0, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """A deterministic, feature-rich scene that ORB can match."""
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width), 255, np.uint8)
    for _ in range(160):
        x = int(rng.integers(0, width - 60))
        y = int(rng.integers(0, height - 60))
        w = int(rng.integers(10, 60))
        h = int(rng.integers(10, 60))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), int(rng.integers(0, 200)), -1)
    return cv2.cvtColor(cv2.GaussianBlur(canvas, (3, 3), 0), cv2.COLOR_GRAY2BGR)


def shifted(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]), borderValue=(255,))


def test_alignment_recovers_a_known_camera_shift() -> None:
    reference = build_reference_features(make_texture())
    frame = shifted(make_texture(), 12, -8)

    result = estimate_alignment(reference, frame)

    assert result.quality in (GOOD, DEGRADED)
    assert result.usable
    # The transform maps the current frame onto the reference, so the shift flips sign.
    assert result.matrix[0][2] == pytest.approx(-12 / WIDTH, abs=0.006)
    assert result.matrix[1][2] == pytest.approx(8 / HEIGHT, abs=0.006)


def test_alignment_is_near_identity_for_an_unchanged_frame() -> None:
    frame = make_texture()
    result = estimate_alignment(build_reference_features(frame), frame.copy())

    assert result.quality == GOOD
    assert abs(result.matrix[0][2]) < 0.002
    assert abs(result.matrix[1][2]) < 0.002


def test_alignment_fails_without_features_to_match() -> None:
    blank = np.full((HEIGHT, WIDTH, 3), 127, np.uint8)
    result = estimate_alignment(build_reference_features(blank), blank.copy())

    assert result.quality == FAILED
    assert not result.usable
    assert result.reason


def test_transform_points_inverts_cleanly() -> None:
    reference = build_reference_features(make_texture())
    result = estimate_alignment(reference, shifted(make_texture(), 15, 10))

    original = [(0.25, 0.5), (0.75, 0.8)]
    round_tripped = transform_points(
        result.matrix, transform_points(invert(result.matrix), original)
    )
    for (expected_x, expected_y), (actual_x, actual_y) in zip(original, round_tripped, strict=True):
        assert actual_x == pytest.approx(expected_x, abs=1e-4)
        assert actual_y == pytest.approx(expected_y, abs=1e-4)


def test_transform_point_matches_point_list_helper() -> None:
    matrix = np.float32([[1.0, 0.0, 0.1], [0.0, 1.0, -0.2]])
    assert transform_point(matrix, (0.4, 0.4)) == pytest.approx((0.5, 0.2))


def test_alignment_tracker_reuses_the_last_good_transform(monkeypatch) -> None:
    frame = make_texture()
    tracker = AlignmentTracker(camera="living_room", trust_last_good_seconds=60.0)
    tracker.set_reference(build_reference_features(frame))

    now = 1000.0
    state = tracker.resolve(shifted(frame, 10, 0), now)
    assert state.quality == GOOD
    assert tracker.last_good_matrix is not None

    monkeypatch.setattr(
        alignment_module,
        "estimate_alignment",
        lambda *args, **kwargs: AlignmentResult(FAILED, None, 0, 0, 0.0, "blurred"),
    )

    # Within the trust window the previous transform is reused.
    state = tracker.resolve(frame, now + 10)
    assert state.quality == DEGRADED
    assert state.zone_lookup_allowed
    assert "reusing the last good alignment" in state.note

    # Past the trust window the camera is treated as uncalibrated.
    state = tracker.resolve(frame, now + 120)
    assert state.quality == FAILED
    assert not state.zone_lookup_allowed
    assert state.matrix is None


def test_alignment_tracker_without_reference_uses_zones_as_captured() -> None:
    tracker = AlignmentTracker(camera="sofa")
    tracker.set_reference(None)

    state = tracker.resolve(make_texture(), 0.0)

    assert state.matrix is None
    assert state.zone_lookup_allowed
    assert state.quality == DEGRADED


@pytest.fixture
def reanchor_settings(tmp_path):
    return {
        "work_width": WIDTH,
        "calibration_dir": str(tmp_path / "calibrations"),
        "zones_path": str(tmp_path / "zones.yaml"),
        "notify": False,
        "discord_webhook": "",
        "discord_username": "test",
    }


def calibrated_camera(tmp_path, settings, zones, reference_frame):
    calibration_dir = Path(settings["calibration_dir"]) / "living_room"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    reference_path = calibration_dir / "reference_initial.jpg"
    cv2.imwrite(str(reference_path), reference_frame)
    calibration = Calibration(
        camera="living_room",
        zones=zones,
        calibration_id="living_room-initial",
        created_at=datetime(2026, 9, 20, 10, 0).isoformat(),
        reference_frame=str(reference_path),
    )
    save_calibrations({"living_room": calibration}, Path(settings["zones_path"]))
    return calibration


def test_reanchor_projects_zones_and_writes_a_new_calibration(reanchor_settings) -> None:
    zones = [Zone("sofa", [(0.2, 0.6), (0.5, 0.6), (0.5, 0.95), (0.2, 0.95)])]
    reference_frame = make_texture()
    calibration = calibrated_camera(reanchor_settings, reanchor_settings, zones, reference_frame)

    # The camera drifted: the scene content moved 20px right and 10px up.
    drifted_frame = shifted(reference_frame, 20, -10)
    outcome = recalibration.reanchor(
        calibration,
        drifted_frame,
        reanchor_settings,
        now=datetime(2026, 9, 20, 12, 0),
    )

    assert outcome.ok, outcome.message
    assert outcome.calibration is not None
    assert outcome.overlay_path is not None and outcome.overlay_path.is_file()
    assert outcome.transform

    # Zones follow the scene, so they move with the drift rather than staying put.
    projected = outcome.calibration.zones[0]
    moved_x = projected.centroid[0] - zones[0].centroid[0]
    moved_y = projected.centroid[1] - zones[0].centroid[1]
    assert moved_x == pytest.approx(20 / WIDTH, abs=0.01)
    assert moved_y == pytest.approx(-10 / HEIGHT, abs=0.01)

    stored = load_calibrations(Path(reanchor_settings["zones_path"]))["living_room"]
    assert stored.calibration_id.startswith("living_room-20260920T120000")
    assert stored.reference_frame
    assert outcome.calibration.reference_path.is_file()


def test_reanchor_keeps_previous_zones_when_alignment_fails(reanchor_settings) -> None:
    zones = [Zone("sofa", [(0.2, 0.6), (0.5, 0.6), (0.5, 0.95), (0.2, 0.95)])]
    calibration = calibrated_camera(reanchor_settings, reanchor_settings, zones, make_texture())
    blank = np.full((HEIGHT, WIDTH, 3), 127, np.uint8)

    outcome = recalibration.reanchor(
        calibration, blank, reanchor_settings, now=datetime(2026, 9, 20, 12, 0)
    )

    assert not outcome.ok
    assert "failed" in outcome.message
    # The old calibration must survive so the failure stays recoverable.
    stored = load_calibrations(Path(reanchor_settings["zones_path"]))["living_room"]
    assert stored.calibration_id == "living_room-initial"
    assert [zone.name for zone in stored.zones] == ["sofa"]
    assert outcome.overlay_path is not None and outcome.overlay_path.is_file()


def test_reanchor_without_previous_reference_adopts_the_current_frame(reanchor_settings) -> None:
    zones = [Zone("carpet", [(0.1, 0.7), (0.6, 0.7), (0.6, 0.99), (0.1, 0.99)])]
    draft = Calibration(camera="living_room", zones=zones)
    frame = make_texture()

    outcome = recalibration.reanchor(
        draft, frame, reanchor_settings, now=datetime(2026, 9, 20, 9, 0)
    )

    assert outcome.ok
    assert outcome.calibration is not None
    # Zones were drawn on this frame, so they are stored unchanged.
    assert outcome.calibration.zones[0].points == zones[0].points
    assert outcome.calibration.calibration_id.startswith("living_room-20260920T090000")
    assert outcome.calibration.reference_path.is_file()


def test_store_migration_adds_columns_to_an_existing_database(tmp_path) -> None:
    database = tmp_path / "legacy.db"
    connection = sqlite3.connect(str(database))
    connection.executescript(
        """
        CREATE TABLE observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, cat TEXT NOT NULL,
            camera TEXT NOT NULL, zone TEXT, confidence REAL NOT NULL
        );
        CREATE TABLE visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT, cat TEXT NOT NULL, camera TEXT NOT NULL,
            zone TEXT, start_ts REAL NOT NULL, end_ts REAL NOT NULL,
            samples INTEGER NOT NULL DEFAULT 0, max_confidence REAL NOT NULL DEFAULT 0
        );
        INSERT INTO observations (ts, cat, camera, zone, confidence)
            VALUES (1.0, 'bagel', 'sofa', 'cushion', 0.9);
        """
    )
    connection.commit()
    connection.close()

    store = LocationStore(database)
    try:
        store.record_observation(2.0, "bagel", "sofa", "cushion", 0.8, 0.4, 0.7, "cal-1", GOOD)
        observations = store.observations_between(0.0, 10.0)
    finally:
        store.close()

    assert len(observations) == 2
    assert observations[0].norm_x is None  # pre-migration row keeps working
    assert observations[1].norm_x == 0.4
    assert observations[1].calibration_id == "cal-1"
    assert observations[1].alignment_quality == GOOD


def test_visit_records_the_calibration_id(tmp_path) -> None:
    store = LocationStore(tmp_path / "history.db")
    try:
        visit_id = store.open_visit(1.0, "bagel", "sofa", "cushion", 0.9, "cal-7")
        assert visit_id > 0
        visit = store.visits_between(0.0, 10.0)[0]
    finally:
        store.close()

    assert visit.calibration_id == "cal-7"
    assert visit.location == "cushion"


def test_alignment_history_is_bounded() -> None:
    frame = make_texture()
    tracker = AlignmentTracker(camera="sofa")
    tracker.set_reference(build_reference_features(frame))

    for index in range(80):
        tracker.resolve(shifted(frame, index % 5, 0), float(index))

    assert len(tracker.history) <= 50
