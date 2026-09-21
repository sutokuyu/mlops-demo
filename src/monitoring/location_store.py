"""SQLite persistence for cat location observations and dwell visits.

Observations keep the cat's anchor point in reference-frame coordinates, so the
history can be re-interpreted after re-drawing zones or re-anchoring a camera
without re-running detection.

They also keep the detection box and how lopsided the zone vote was. The anchor
point alone is not enough to re-derive a location under a different anchor
strategy, which is exactly the situation the first zone backfill ran into: the
stored point was all there was to go on.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    cat TEXT NOT NULL,
    camera TEXT NOT NULL,
    zone TEXT,
    confidence REAL NOT NULL,
    norm_x REAL,
    norm_y REAL,
    calibration_id TEXT,
    alignment_quality TEXT,
    box_x1 REAL,
    box_y1 REAL,
    box_x2 REAL,
    box_y2 REAL,
    zone_matches INTEGER,
    zone_samples INTEGER
);
CREATE INDEX IF NOT EXISTS idx_observations_ts ON observations (ts);

CREATE TABLE IF NOT EXISTS visits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cat TEXT NOT NULL,
    camera TEXT NOT NULL,
    zone TEXT,
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL,
    samples INTEGER NOT NULL DEFAULT 0,
    max_confidence REAL NOT NULL DEFAULT 0,
    calibration_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_visits_start ON visits (start_ts, end_ts);
"""

MIGRATIONS = {
    "observations": {
        "norm_x": "ALTER TABLE observations ADD COLUMN norm_x REAL",
        "norm_y": "ALTER TABLE observations ADD COLUMN norm_y REAL",
        "calibration_id": "ALTER TABLE observations ADD COLUMN calibration_id TEXT",
        "alignment_quality": "ALTER TABLE observations ADD COLUMN alignment_quality TEXT",
        # Normalized detection box, so any anchor strategy can be re-applied to
        # history without re-running detection.
        "box_x1": "ALTER TABLE observations ADD COLUMN box_x1 REAL",
        "box_y1": "ALTER TABLE observations ADD COLUMN box_y1 REAL",
        "box_x2": "ALTER TABLE observations ADD COLUMN box_x2 REAL",
        "box_y2": "ALTER TABLE observations ADD COLUMN box_y2 REAL",
        # How lopsided the zone vote was: 9/9 is confident, 5/9 is not.
        "zone_matches": "ALTER TABLE observations ADD COLUMN zone_matches INTEGER",
        "zone_samples": "ALTER TABLE observations ADD COLUMN zone_samples INTEGER",
    },
    "visits": {
        "calibration_id": "ALTER TABLE visits ADD COLUMN calibration_id TEXT",
    },
}


@dataclass
class VisitRow:
    cat: str
    camera: str
    zone: str | None
    start_ts: float
    end_ts: float
    samples: int
    max_confidence: float
    calibration_id: str | None = None

    @property
    def location(self) -> str:
        """Human-readable location; falls back to the camera when no zone matched."""
        return self.zone or self.camera


@dataclass
class ObservationRow:
    ts: float
    cat: str
    camera: str
    zone: str | None
    confidence: float
    norm_x: float | None
    norm_y: float | None
    calibration_id: str | None
    alignment_quality: str | None
    box_x1: float | None = None
    box_y1: float | None = None
    box_x2: float | None = None
    box_y2: float | None = None
    zone_matches: int | None = None
    zone_samples: int | None = None

    @property
    def box(self) -> tuple[float, float, float, float] | None:
        """The detection box in normalized frame coordinates, if it was recorded."""
        if self.box_x1 is None or self.box_y1 is None:
            return None
        if self.box_x2 is None or self.box_y2 is None:
            return None
        return (self.box_x1, self.box_y1, self.box_x2, self.box_y2)


class LocationStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._connection = sqlite3.connect(str(path))
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(SCHEMA)
        self._migrate()
        self._connection.commit()

    def _migrate(self) -> None:
        """Add columns that were introduced after a database was first created."""
        for table, columns in MIGRATIONS.items():
            existing = {row[1] for row in self._connection.execute(f"PRAGMA table_info({table})")}
            for column, statement in columns.items():
                if column not in existing:
                    self._connection.execute(statement)

    def record_observation(
        self,
        ts: float,
        cat: str,
        camera: str,
        zone: str | None,
        confidence: float,
        norm_x: float | None = None,
        norm_y: float | None = None,
        calibration_id: str | None = None,
        alignment_quality: str | None = None,
        box: tuple[float, float, float, float] | None = None,
        vote: tuple[int, int] | None = None,
    ) -> None:
        """box`` and ``vote`` are normalized box coords and (matches, samples)."""
        box_x1, box_y1, box_x2, box_y2 = box if box is not None else (None, None, None, None)
        zone_matches, zone_samples = vote if vote is not None else (None, None)
        self._connection.execute(
            "INSERT INTO observations"
            " (ts, cat, camera, zone, confidence, norm_x, norm_y, calibration_id,"
            "  alignment_quality, box_x1, box_y1, box_x2, box_y2, zone_matches, zone_samples)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ts,
                cat,
                camera,
                zone,
                confidence,
                norm_x,
                norm_y,
                calibration_id,
                alignment_quality,
                box_x1,
                box_y1,
                box_x2,
                box_y2,
                zone_matches,
                zone_samples,
            ),
        )
        self._connection.commit()

    def open_visit(
        self,
        ts: float,
        cat: str,
        camera: str,
        zone: str | None,
        confidence: float,
        calibration_id: str | None = None,
    ) -> int:
        cursor = self._connection.execute(
            "INSERT INTO visits"
            " (cat, camera, zone, start_ts, end_ts, samples, max_confidence, calibration_id)"
            " VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
            (cat, camera, zone, ts, ts, confidence, calibration_id),
        )
        self._connection.commit()
        return int(cursor.lastrowid)

    def touch_visit(
        self, visit_id: int, end_ts: float, samples: int, max_confidence: float
    ) -> None:
        self._connection.execute(
            "UPDATE visits SET end_ts = ?, samples = ?, max_confidence = ? WHERE id = ?",
            (end_ts, samples, max_confidence, visit_id),
        )
        self._connection.commit()

    def visits_between(self, start_ts: float, end_ts: float) -> list[VisitRow]:
        rows = self._connection.execute(
            "SELECT cat, camera, zone, start_ts, end_ts, samples, max_confidence, calibration_id"
            " FROM visits WHERE end_ts > ? AND start_ts < ? ORDER BY start_ts",
            (start_ts, end_ts),
        ).fetchall()
        return [VisitRow(*row) for row in rows]

    def observations_between(self, start_ts: float, end_ts: float) -> list[ObservationRow]:
        """Raw anchors, kept so past days can be re-interpreted with newer zones."""
        rows = self._connection.execute(
            "SELECT ts, cat, camera, zone, confidence, norm_x, norm_y, calibration_id,"
            " alignment_quality FROM observations WHERE ts >= ? AND ts < ? ORDER BY ts",
            (start_ts, end_ts),
        ).fetchall()
        return [ObservationRow(*row) for row in rows]

    def close(self) -> None:
        self._connection.close()
