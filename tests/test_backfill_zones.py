"""Tests for the zone backfill that repairs samples recorded during failed alignment."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backfill_zones import majority_zone, recoverable_zones
from src.monitoring.location_zones import Calibration, Zone

SOFA = Zone("sofa", [(0.1, 0.6), (0.4, 0.6), (0.4, 0.95), (0.1, 0.95)])
CALIBRATIONS = {
    "sofa": Calibration(camera="sofa", zones=[SOFA], calibration_id="sofa-current"),
}


def observation(row_id: int, x=0.25, y=0.8, camera="sofa", calibration_id="sofa-current") -> dict:
    return {
        "id": row_id,
        "camera": camera,
        "calibration_id": calibration_id,
        "norm_x": x,
        "norm_y": y,
        "zone": None,
    }


def test_majority_zone_picks_the_most_common_name() -> None:
    assert majority_zone(["floor", "floor", "carpet"]) == "floor"


def test_majority_zone_ignores_samples_that_never_landed() -> None:
    assert majority_zone([None, "floor", None]) == "floor"
    assert majority_zone([None, None]) is None
    assert majority_zone([]) is None


def test_recoverable_zones_uses_the_stored_anchor() -> None:
    recovered = recoverable_zones([observation(1)], CALIBRATIONS)
    assert recovered == {1: "sofa"}


def test_recoverable_zones_skips_anchors_outside_every_polygon() -> None:
    assert recoverable_zones([observation(1, x=0.9, y=0.1)], CALIBRATIONS) == {}


def test_recoverable_zones_skips_an_unknown_camera() -> None:
    assert recoverable_zones([observation(1, camera="attic")], CALIBRATIONS) == {}


def test_recoverable_zones_skips_rows_without_an_anchor() -> None:
    assert recoverable_zones([observation(1, x=None)], CALIBRATIONS) == {}


def test_recoverable_zones_is_strict_about_the_calibration_by_default() -> None:
    """Redrawn zones mean the old polygons are gone, so the name would be a guess."""
    stale = [observation(1, calibration_id="sofa-old")]
    assert recoverable_zones(stale, CALIBRATIONS) == {}
    assert recoverable_zones(stale, CALIBRATIONS, require_matching_calibration=False) == {1: "sofa"}
