#!/usr/bin/env python
"""Recompute zone names for samples that were recorded while alignment was failing.

Before ``alignment.on_alignment_failure`` existed, a hard alignment failure left
``zone`` NULL and the location fell back to the camera name, so a cat sitting on
the cat wall was recorded as plain ``living_room``. The raw anchor point was still
stored: failed samples are never transformed, so ``observations.norm_x`` and
``norm_y`` hold the anchor exactly as measured. That makes the zone recomputable
after the fact, as long as the camera's zones have not been redrawn since - the
row's ``calibration_id`` has to still match ``configs/zones.yaml``.

Observations are recomputed directly. Visits only store one zone for the whole
span, so theirs comes from a majority vote over the observations inside the visit
window. A visit that silently merged two zones while the lookup was disabled still
comes out as a single zone; re-segmenting it would mean rewriting the timeline and
is deliberately not attempted here.

Usage:
    .venv/bin/python scripts/backfill_zones.py --dry-run
    .venv/bin/python scripts/backfill_zones.py --apply
"""

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path


def _resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not locate project root")


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.location_config import location_database
from src.monitoring.location_zones import find_zone, load_calibrations


def recoverable_zones(
    observations: list[sqlite3.Row],
    calibrations: dict,
    require_matching_calibration: bool = True,
) -> dict[int, str]:
    """Map observation id to the zone its stored anchor falls in.

    By default only rows whose ``calibration_id`` still matches the loaded zones are
    usable: if the polygons were redrawn since, the recomputed name is a guess
    rather than a recovery, because the old polygons are gone.

    ``require_matching_calibration=False`` skips that check. It is right when the
    zones were merely re-saved - the editor keeps the polygons and only swaps the
    reference frame and the id - and wrong when they were actually redrawn.
    """
    recovered: dict[int, str] = {}
    for row in observations:
        if row["norm_x"] is None or row["norm_y"] is None:
            continue
        calibration = calibrations.get(row["camera"])
        if calibration is None:
            continue
        if require_matching_calibration and calibration.calibration_id != row["calibration_id"]:
            continue
        zone = find_zone(calibration.zones, row["norm_x"], row["norm_y"])
        if zone is not None:
            recovered[row["id"]] = zone.name
    return recovered


def majority_zone(zones: list[str | None]) -> str | None:
    """The zone a visit spent most of its samples in, or None if it never landed."""
    known = [zone for zone in zones if zone]
    if not known:
        return None
    return Counter(known).most_common(1)[0][0]


def plan_backfill(
    connection: sqlite3.Connection, calibrations: dict, require_matching_calibration: bool = True
) -> dict:
    """Work out every change without writing anything."""
    observations = connection.execute(
        "SELECT id, camera, calibration_id, norm_x, norm_y, zone "
        "FROM observations WHERE zone IS NULL AND norm_x IS NOT NULL AND norm_y IS NOT NULL"
    ).fetchall()
    recovered = recoverable_zones(observations, calibrations, require_matching_calibration)
    unresolved_observations = len(observations) - len(recovered)

    visits = connection.execute(
        "SELECT id, cat, camera, start_ts, end_ts FROM visits WHERE zone IS NULL"
    ).fetchall()
    visit_updates: list[tuple[str, int]] = []
    visits_without_data = 0
    visit_zones: dict[int, str] = {}
    for visit in visits:
        rows = connection.execute(
            "SELECT id FROM observations WHERE cat = ? AND camera = ? AND ts >= ? AND ts <= ?",
            (visit["cat"], visit["camera"], visit["start_ts"], visit["end_ts"]),
        ).fetchall()
        zone = majority_zone([recovered.get(row["id"]) for row in rows])
        if zone is None:
            visits_without_data += 1
            continue
        visit_updates.append((zone, visit["id"]))
        visit_zones[visit["id"]] = zone

    return {
        "observations_considered": len(observations),
        "observations_recovered": recovered,
        "observations_unresolved": unresolved_observations,
        "visits_considered": len(visits),
        "visits_recovered": visit_updates,
        "visits_unresolved": visits_without_data,
        "visit_zone_by_id": visit_zones,
    }


def summarise(plan: dict) -> None:
    print(f"observations with zone NULL : {plan['observations_considered']}")
    print(f"  recoverable               : {len(plan['observations_recovered'])}")
    print(f"  not recoverable           : {plan['observations_unresolved']}")
    print(f"visits with zone NULL       : {plan['visits_considered']}")
    print(f"  recoverable               : {len(plan['visits_recovered'])}")
    print(f"  not recoverable           : {plan['visits_unresolved']}")

    counts = Counter(plan["observations_recovered"].values())
    if counts:
        print()
        print("recovered zone names (observations):")
        for name, count in counts.most_common():
            print(f"  {count:6} x {name}")


def backup(database: Path) -> Path:
    """Copy the database through SQLite so WAL contents are included.

    The tracker holds this file open and keeps writing, so copying the bytes would
    risk a torn snapshot.
    """
    target = database.with_suffix(database.suffix + ".pre-backfill.bak")
    source = sqlite3.connect(database)
    try:
        destination = sqlite3.connect(target)
        try:
            source.backup(destination)
        finally:
            destination.close()
    finally:
        source.close()
    return target


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would change. This is the default unless --apply is given.",
    )
    parser.add_argument("--apply", action="store_true", help="Write the changes.")
    parser.add_argument(
        "--any-calibration",
        action="store_true",
        help="Also recover rows whose calibration_id differs from zones.yaml. Correct "
        "only if the zones were re-saved rather than redrawn.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    database = args.database or location_database()
    if not database.is_file():
        print(f"database not found: {database}", file=sys.stderr)
        return 1

    calibrations = load_calibrations()
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        plan = plan_backfill(connection, calibrations, not args.any_calibration)
        print(f"database: {database}")
        print()
        summarise(plan)

        if not args.apply:
            # Show what relaxing the calibration check would add, so the choice is
            # visible rather than something you have to guess at.
            relaxed = plan_backfill(connection, calibrations, require_matching_calibration=False)
            extra = len(relaxed["observations_recovered"]) - len(plan["observations_recovered"])
            extra_visits = len(relaxed["visits_recovered"]) - len(plan["visits_recovered"])
            if extra or extra_visits:
                print()
                print(f"--any-calibration would recover {extra} more observations and ")
                print(f"{extra_visits} more visits, from rows whose calibration_id changed.")
                print("Use it only if those zones were re-saved, not redrawn.")
            print()
            print("dry run - nothing written. Re-run with --apply to commit.")
            return 0

        if not plan["observations_recovered"] and not plan["visits_recovered"]:
            print()
            print("nothing to do.")
            return 0

        backup_path = backup(database)
        with connection:
            connection.executemany(
                "UPDATE observations SET zone = ? WHERE id = ?",
                [(zone, row_id) for row_id, zone in plan["observations_recovered"].items()],
            )
            connection.executemany(
                "UPDATE visits SET zone = ? WHERE id = ?", plan["visits_recovered"]
            )
        print()
        print(
            f"applied: {len(plan['observations_recovered'])} observations, "
            f"{len(plan['visits_recovered'])} visits"
        )
        print(f"backup:  {backup_path}")
        print()
        print("The report reads visits, so it will pick the new zones up on its next run.")
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
