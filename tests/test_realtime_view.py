"""Tests for the zone lookup and printing added to the realtime viewer."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.location_zones import Calibration, Zone, find_zone
from src.monitoring.realtime_view import (
    Channel,
    Location,
    ZoneLocator,
    attach_locators,
    build_channels,
    format_location,
    parse_args,
)

WIDTH, HEIGHT = 640, 480

ZONES = [
    Zone(name="floor", points=[(0.02, 0.55), (0.98, 0.55), (0.98, 0.98), (0.02, 0.98)]),
    Zone(name="table_top", points=[(0.20, 0.20), (0.40, 0.20), (0.40, 0.35), (0.20, 0.35)]),
    Zone(name="under_table", points=[(0.18, 0.56), (0.44, 0.56), (0.44, 0.98), (0.18, 0.98)]),
]


def make_texture(seed: int = 0) -> np.ndarray:
    """Deterministic, feature-rich scene that ORB can match."""
    rng = np.random.default_rng(seed)
    canvas = np.full((HEIGHT, WIDTH), 255, np.uint8)
    for _ in range(160):
        x = int(rng.integers(0, WIDTH - 60))
        y = int(rng.integers(0, HEIGHT - 60))
        w = int(rng.integers(10, 60))
        h = int(rng.integers(10, 60))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), int(rng.integers(0, 200)), -1)
    return cv2.cvtColor(cv2.GaussianBlur(canvas, (3, 3), 0), cv2.COLOR_GRAY2BGR)


def shift_frame(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, matrix, (WIDTH, HEIGHT), borderValue=(255, 255, 255))


def box_above(norm_x: float, norm_y: float, box_width: float = 0.10, box_height: float = 0.20):
    """A detection box whose bottom-centre anchor lands on (norm_x, norm_y)."""
    return (
        (norm_x - box_width / 2) * WIDTH,
        (norm_y - box_height) * HEIGHT,
        (norm_x + box_width / 2) * WIDTH,
        norm_y * HEIGHT,
    )


@pytest.fixture
def locator(tmp_path: Path) -> ZoneLocator:
    reference = make_texture()
    reference_path = tmp_path / "reference.jpg"
    assert cv2.imwrite(str(reference_path), reference)
    calibration = Calibration(
        camera="sofa",
        zones=ZONES,
        calibration_id="sofa-test",
        reference_frame=str(reference_path),
    )
    return ZoneLocator(
        "sofa",
        calibration=calibration,
        settings={
            "work_width": 640,
            "min_inliers": 15,
            "good_inlier_ratio": 0.5,
            "max_residual": 0.01,
        },
    )


def test_locator_loads_the_zone_count(locator: ZoneLocator) -> None:
    assert locator.zone_count == 3


def test_anchor_is_the_box_bottom_centre(locator: ZoneLocator) -> None:
    frame = make_texture()
    state = locator.resolve(frame)

    # Anchored at (0.75, 0.80): inside floor, well away from under_table.
    wide = locator.locate(state, frame.shape, box_above(0.75, 0.80, 0.40, 0.40))
    narrow = locator.locate(state, frame.shape, box_above(0.75, 0.80, 0.05, 0.05))
    assert wide.zone == narrow.zone == "floor"
    assert wide.norm_x == pytest.approx(0.75, abs=0.01)
    assert wide.norm_y == pytest.approx(0.80, abs=0.01)


def test_smallest_containing_zone_wins(locator: ZoneLocator) -> None:
    frame = make_texture()
    state = locator.resolve(frame)

    # (0.30, 0.70) is inside both floor and the smaller under_table.
    location = locator.locate(state, frame.shape, box_above(0.30, 0.70))
    assert location.zone == "under_table"
    assert find_zone(ZONES, 0.30, 0.70).name == "under_table"


def test_anchor_outside_every_zone_is_reported_as_unknown(locator: ZoneLocator) -> None:
    frame = make_texture()
    state = locator.resolve(frame)

    location = locator.locate(state, frame.shape, box_above(0.50, 0.10))
    assert location.zone is None


def test_zones_still_resolve_after_the_camera_drifts(locator: ZoneLocator) -> None:
    # The floor zone covers (0.75, 0.80); drift must not move the reported zone.
    for dx, dy in [(0, 0), (24, 0), (-24, 0), (0, 18), (0, -18), (30, -14)]:
        frame = shift_frame(make_texture(), dx, dy)
        state = locator.resolve(frame)
        location = locator.locate(state, frame.shape, box_above(0.75, 0.80))
        assert location.zone == "floor", f"drift ({dx}, {dy}) misresolved: {location}"
        assert location.quality in ("good", "degraded")


def test_locate_returns_reference_coordinates(locator: ZoneLocator) -> None:
    frame = make_texture()
    state = locator.resolve(frame)

    # With no drift the reference coordinates equal the normalised anchor.
    location = locator.locate(state, frame.shape, box_above(0.60, 0.70))
    assert location.norm_x == pytest.approx(0.60, abs=0.01)
    assert location.norm_y == pytest.approx(0.70, abs=0.01)

    # The frame content moved 32px right, so whatever now appears at x=0.60 was
    # at x=0.55 in the reference frame: the reported coordinate drops by 32px.
    drifted = shift_frame(frame, 32, 0)
    drifted_state = locator.resolve(drifted)
    fixed_anchor = locator.locate(drifted_state, drifted.shape, box_above(0.60, 0.70))
    assert fixed_anchor.norm_x == pytest.approx(0.60 - 32 / 640, abs=0.02)

    # Pointing at the same physical spot before and after the drift must report
    # the same reference coordinate. That invariance is the whole point of aligning.
    same_spot = locator.locate(drifted_state, drifted.shape, box_above(0.60 + 32 / 640, 0.70))
    assert same_spot.norm_x == pytest.approx(location.norm_x, abs=0.01)


def test_no_reference_means_no_alignment_and_no_zone(tmp_path: Path) -> None:
    calibration = Calibration(camera="ghost", zones=ZONES)  # no reference_frame
    locator = ZoneLocator(
        "ghost",
        calibration=calibration,
        settings={
            "work_width": 640,
            "min_inliers": 15,
            "good_inlier_ratio": 0.5,
            "max_residual": 0.01,
        },
    )
    assert locator.alignment is None

    frame = make_texture()
    state = locator.resolve(frame)
    assert state is None

    location = locator.locate(state, frame.shape, box_above(0.75, 0.80))
    assert location.zone == "floor"
    assert location.quality == "no-reference"


def test_empty_calibration_reports_no_zone(locator: ZoneLocator) -> None:
    locator.calibration = Calibration(camera="sofa", zones=ZONES[:0])
    frame = make_texture()
    state = locator.resolve(frame)
    assert locator.locate(state, frame.shape, box_above(0.75, 0.80)).zone is None


def test_format_location_reads_like_a_report() -> None:
    text = format_location("sofa", "bagel", Location("on_sofa", "good", 0.512, 0.744), 0.87)
    assert text == "[sofa] bagel -> on_sofa  conf=0.87 anchor=(0.512, 0.744) align=good"

    unknown = format_location("sofa", "kurumi", Location(None, "degraded", 0.1, 0.2), 0.55)
    assert "-> unknown" in unknown
    assert "align=degraded" in unknown


def test_channel_open_reader_accepts_the_channel_name(tmp_path: Path) -> None:
    """Guards the reader constructor: a mismatched kwarg breaks every camera."""
    channel = Channel(name="sofa", source=str(tmp_path / "missing.avi"))
    channel.open_reader()
    assert channel.reader is not None
    assert channel.reader.name == "sofa"


def test_build_channels_selects_every_requested_camera(monkeypatch) -> None:
    args = parse_args([])
    monkeypatch.setattr(
        "src.monitoring.realtime_view.configured_cameras", lambda: ["living_room", "sofa", "feeder"]
    )
    monkeypatch.setattr(
        "src.monitoring.realtime_view.camera_rtsp_url",
        lambda name: "" if name == "feeder" else f"rtsp://example/{name}",
    )
    channels = build_channels(args)
    assert [channel.name for channel in channels] == ["living_room", "sofa"]


def test_build_channels_honours_repeated_camera_flags(monkeypatch) -> None:
    args = parse_args(["--camera", "sofa", "--camera", "feeder"])
    monkeypatch.setattr(
        "src.monitoring.realtime_view.camera_rtsp_url", lambda name: f"rtsp://example/{name}"
    )
    channels = build_channels(args)
    assert [channel.name for channel in channels] == ["sofa", "feeder"]


def test_build_channels_uses_zones_camera_for_a_single_source() -> None:
    args = parse_args(["--source", "/tmp/clip.avi", "--zones-camera", "sofa"])
    channels = build_channels(args)
    assert len(channels) == 1
    assert channels[0].name == "sofa"
    assert channels[0].source == "/tmp/clip.avi"


def test_build_channels_coerces_a_numeric_source_to_a_webcam_index() -> None:
    channels = build_channels(parse_args(["--source", "0"]))
    assert channels[0].source == 0


def test_attach_locators_loads_zones_per_camera(monkeypatch, tmp_path: Path) -> None:
    reference = make_texture()
    reference_path = tmp_path / "reference.jpg"
    assert cv2.imwrite(str(reference_path), reference)
    calibration = Calibration(
        camera="sofa",
        zones=ZONES,
        calibration_id="sofa-test",
        reference_frame=str(reference_path),
    )
    monkeypatch.setattr(
        "src.monitoring.realtime_view.load_calibrations",
        lambda *args, **kwargs: {"sofa": calibration},
    )

    channels = [Channel(name="sofa", source="a"), Channel(name="feeder", source="b")]
    attach_locators(channels, parse_args([]))
    assert channels[0].locator is not None
    assert channels[0].locator.zone_count == 3
    # No calibration for feeder, so it keeps working without location lookup.
    assert channels[1].locator is None


def test_attach_locators_can_be_disabled(monkeypatch) -> None:
    channels = [Channel(name="sofa", source="a")]
    attach_locators(channels, parse_args(["--no-zones"]))
    assert channels[0].locator is None
