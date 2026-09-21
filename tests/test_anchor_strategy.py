"""Tests for how a detection box becomes the point zone lookup matches against."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.detections import BOTTOM_CENTER, LOWER_GRID, anchor_points, bottom_center
from src.monitoring.location_zones import Zone, vote_zone

BOX = (100.0, 200.0, 300.0, 400.0)  # x1, y1, x2, y2


def test_bottom_center_is_the_midpoint_of_the_bottom_edge() -> None:
    assert bottom_center(BOX) == (200.0, 400.0)


def test_bottom_center_strategy_yields_that_one_point() -> None:
    assert anchor_points(BOX, BOTTOM_CENTER) == [(200.0, 400.0)]


def test_lower_grid_stays_inside_the_bottom_half() -> None:
    points = anchor_points(BOX, LOWER_GRID, 3)
    assert len(points) == 9
    x1, y1, x2, y2 = BOX
    middle = (y1 + y2) / 2
    assert all(x1 <= x <= x2 for x, _ in points)
    assert all(middle <= y <= y2 for _, y in points), "a dangling tail lives below the middle"


def test_lower_grid_is_deterministic() -> None:
    """Random sampling would give the same frame two different answers."""
    assert anchor_points(BOX, LOWER_GRID, 3) == anchor_points(BOX, LOWER_GRID, 3)


def test_unknown_anchor_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown anchor strategy"):
        anchor_points(BOX, "middle_something")


BULK = Zone("room", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
SILL = Zone("sill", [(0.1, 0.1), (0.4, 0.1), (0.4, 0.3), (0.1, 0.3)])
FLOOR = Zone("floor", [(0.0, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0)])
ZONES = [BULK, SILL, FLOOR]


def test_vote_zone_takes_the_majority_and_reports_the_score() -> None:
    vote = vote_zone(ZONES, [(0.2, 0.2), (0.2, 0.25), (0.2, 0.8)])
    assert vote.zone == "sill", "the smallest containing polygon wins, as in find_zone"
    assert (vote.matches, vote.samples) == (2, 3)
    assert vote.share == "2/3"


def test_vote_zone_lets_the_body_outvote_a_dangling_tail() -> None:
    """The whole point of the strategy: one stray point must not decide."""
    body = [(0.2, 0.2), (0.2, 0.25), (0.25, 0.22)]
    tail = [(0.2, 0.8)]
    vote = vote_zone(ZONES, body + tail)
    assert vote.zone == "sill"
    assert vote.share == "3/4", "the tail is outvoted but still counted"

    # The same box scored on the bottom edge alone follows the tail instead.
    assert vote_zone(ZONES, tail).zone == "floor"


def test_vote_zone_returns_no_zone_when_every_point_misses() -> None:
    vote = vote_zone([SILL], [(0.9, 0.9), (0.8, 0.95)])
    assert vote.zone is None
    assert vote.share == "0/2"


def test_vote_zone_is_deterministic_on_a_tie() -> None:
    """Ties resolve by point order, not by dict iteration luck."""
    points = [(0.2, 0.2), (0.2, 0.8)]
    assert vote_zone([SILL, FLOOR], points).zone == "sill"
    assert vote_zone([SILL, FLOOR], list(reversed(points))).zone == "floor"


DATA_DIR = PROJECT_ROOT / "dataset"


def _labelled_boxes():
    """Every annotated box, grouped by the camera its filename encodes.

    Camera names contain underscores (``living_room``), so the prefix has to be
    matched against the configured list rather than split on the first ``_``.
    """
    from src.monitoring.location_config import configured_cameras

    cameras = configured_cameras()
    boxes: dict[str, list[tuple[float, float, float, float]]] = {}
    for path in sorted(DATA_DIR.glob("*/labels/*/*.txt")):
        camera = next((name for name in cameras if path.stem.startswith(name + "_")), None)
        if camera is None:
            continue
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                boxes.setdefault(camera, []).append(tuple(float(v) for v in parts[1:5]))
    return boxes


@pytest.mark.skipif(
    not DATA_DIR.is_dir(),
    reason="the annotated dataset is not checked in (DVC), so this only runs locally",
)
def test_lower_grid_covers_at_least_what_the_bottom_edge_does() -> None:
    """Regression guard on the change, measured against real postures.

    The absolute numbers move whenever zones are edited, so the invariant asserted
    here is the relative one: switching to the grid must never resolve fewer boxes
    than the bottom edge did. The printed figures are the useful part.
    """
    from src.monitoring.location_zones import load_calibrations

    calibrations = load_calibrations()
    boxes = _labelled_boxes()
    assert boxes, "expected some annotated boxes"

    totals = {"boxes": 0, BOTTOM_CENTER: 0, LOWER_GRID: 0}
    per_camera: dict[str, list[int]] = {}
    for camera, samples in boxes.items():
        calibration = calibrations.get(camera)
        if calibration is None:
            continue
        counts = {BOTTOM_CENTER: 0, LOWER_GRID: 0}
        for cx, cy, w, h in samples:
            box = (
                (cx - w / 2) * 1000,
                (cy - h / 2) * 1000,
                (cx + w / 2) * 1000,
                (cy + h / 2) * 1000,
            )
            for strategy in (BOTTOM_CENTER, LOWER_GRID):
                points = [(x / 1000, y / 1000) for x, y in anchor_points(box, strategy)]
                if vote_zone(calibration.zones, points).zone is not None:
                    counts[strategy] += 1
        totals["boxes"] += len(samples)
        totals[BOTTOM_CENTER] += counts[BOTTOM_CENTER]
        totals[LOWER_GRID] += counts[LOWER_GRID]
        per_camera[camera] = [len(samples), counts[BOTTOM_CENTER], counts[LOWER_GRID]]

    total = totals["boxes"]
    print(f"\n{'camera':14}{'boxes':>7}{'bottom_center':>15}{'lower_grid':>13}")
    for camera, (n, bottom, grid) in sorted(per_camera.items()):
        print(f"{camera:14}{n:7}{bottom / n * 100:14.1f}%{grid / n * 100:12.1f}%")
    print(
        f"{'ALL':14}{total:7}"
        f"{totals[BOTTOM_CENTER] / total * 100:14.1f}%{totals[LOWER_GRID] / total * 100:12.1f}%"
    )

    assert totals[LOWER_GRID] >= totals[BOTTOM_CENTER], "the grid must not lose coverage"
