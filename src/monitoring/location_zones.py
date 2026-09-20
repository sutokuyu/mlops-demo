"""Zone geometry and the calibration records that anchor zones to a reference frame.

A calibration binds a set of zone polygons to the camera frame they were drawn
on. Every sample is aligned back to that reference frame, so the zones keep
working when the camera drifts or gets re-aimed. Re-anchoring writes a new
calibration instead of overwriting the previous one, which keeps historical
records interpretable.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def _resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not locate project root")


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ZONES_PATH = PROJECT_ROOT / "configs" / "zones.yaml"


@dataclass
class Zone:
    name: str
    points: list[tuple[float, float]]

    @property
    def area(self) -> float:
        """Shoelace area in normalized units, used to prefer the most specific zone."""
        total = 0.0
        for index, (x1, y1) in enumerate(self.points):
            x2, y2 = self.points[(index + 1) % len(self.points)]
            total += x1 * y2 - x2 * y1
        return abs(total) / 2

    @property
    def centroid(self) -> tuple[float, float]:
        count = len(self.points) or 1
        return (
            sum(x for x, _ in self.points) / count,
            sum(y for _, y in self.points) / count,
        )


@dataclass
class Calibration:
    """Zones belong to the coordinate system of their reference frame."""

    camera: str
    zones: list[Zone] = field(default_factory=list)
    calibration_id: str = ""
    created_at: str = ""
    reference_frame: str | None = None

    @property
    def reference_path(self) -> Path | None:
        if not self.reference_frame:
            return None
        path = Path(self.reference_frame)
        return path if path.is_absolute() else PROJECT_ROOT / path


def contains(zone: Zone, x: float, y: float) -> bool:
    """Ray-casting point-in-polygon test."""
    inside = False
    points = zone.points
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        if (y1 > y) != (y2 > y):
            slope_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < slope_x:
                inside = not inside
    return inside


def find_zone(zones: list[Zone], x: float, y: float) -> Zone | None:
    """Return the smallest zone containing the point, so nested zones win."""
    matches = [zone for zone in zones if contains(zone, x, y)]
    if not matches:
        return None
    return min(matches, key=lambda zone: zone.area)


def _parse_zone(entry: dict) -> Zone:
    return Zone(name=entry["name"], points=[tuple(point) for point in entry["points"]])


def load_calibrations(path: Path = ZONES_PATH) -> dict[str, Calibration]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    calibrations: dict[str, Calibration] = {}
    for camera_name, entry in data.items():
        if isinstance(entry, list):
            # Older format that stored a bare zone list; still loadable.
            calibrations[camera_name] = Calibration(
                camera=camera_name, zones=[_parse_zone(item) for item in entry]
            )
            continue
        calibrations[camera_name] = Calibration(
            camera=camera_name,
            zones=[_parse_zone(item) for item in entry.get("zones", [])],
            calibration_id=entry.get("calibration_id", ""),
            created_at=entry.get("created_at", ""),
            reference_frame=entry.get("reference_frame"),
        )
    return calibrations


def save_calibrations(calibrations: dict[str, Calibration], path: Path = ZONES_PATH) -> None:
    payload = {
        camera_name: {
            "calibration_id": calibration.calibration_id,
            "created_at": calibration.created_at,
            "reference_frame": calibration.reference_frame,
            "zones": [
                {"name": zone.name, "points": [[round(x, 4), round(y, 4)] for x, y in zone.points]}
                for zone in calibration.zones
            ],
        }
        for camera_name, calibration in sorted(calibrations.items())
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Written by src/monitoring/web_preview.py and src/monitoring/recalibration.py.\n"
        "# Zone points are normalized (0-1) and belong to the reference frame above.\n"
        + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_zones(path: Path = ZONES_PATH) -> dict[str, list[Zone]]:
    """Zone lookup by camera, for callers that do not need calibration metadata."""
    return {
        camera_name: calibration.zones
        for camera_name, calibration in load_calibrations(path).items()
    }
