"""Tests for zone lookup, box filtering, dwell segments, and daily aggregation."""

import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring import location_report, location_tracker
from src.monitoring.detections import bottom_center, select_best_per_class
from src.monitoring.location_store import LocationStore
from src.monitoring.location_zones import Zone, find_zone

ROOM = Zone("living_room", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
SOFA = Zone("sofa", [(0.1, 0.6), (0.4, 0.6), (0.4, 0.95), (0.1, 0.95)])
DAY_START = datetime(2026, 9, 20, 9, 0, tzinfo=ZoneInfo("Asia/Tokyo")).timestamp()


def test_find_zone_prefers_the_smallest_containing_polygon() -> None:
    assert find_zone([ROOM, SOFA], 0.25, 0.8).name == "sofa"
    assert find_zone([ROOM, SOFA], 0.8, 0.3).name == "living_room"
    assert find_zone([SOFA], 0.9, 0.1) is None


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
