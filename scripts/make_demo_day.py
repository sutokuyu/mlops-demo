"""Build a fabricated day of cat locations for testing the report pipeline.

Writes to a throwaway database so the real history is never touched.
"""

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.location_store import LocationStore

TZ = ZoneInfo("Asia/Tokyo")
TARGET_DAY = "2026-09-19"

# (start, end, camera, zone, samples). Times accept "HH:MM" or "HH:MM:SS".
#
# toilet_1 sits on the feeder camera and a cat walks through it on the way to
# feeder / water_server. Per report.llm.hints, two or more appearances within 5
# minutes with nothing else in between means a real toilet visit.
BAGEL = [
    ("00:12", "05:40", "sofa", "carpet", 65),
    # PASS-BY (1/2): on the way to breakfast. Paired with the return trip below,
    # 45 minutes later and separated by `feeder`, so this must NOT count as a
    # toilet visit on either count.
    ("05:44:30", "05:44:45", "feeder", "toilet_1", 1),
    ("05:45", "06:30", "feeder", "feeder", 9),
    ("06:30:05", "06:30:20", "feeder", "toilet_1", 1),
    ("06:35", "08:10", "living_room", "kitchen_counter", 19),
    ("08:15", "11:30", "sofa", "on_sofa", 39),
    # REAL USE: two glimpses 2 minutes apart, nothing else in between.
    ("11:32:00", "11:32:20", "feeder", "toilet_1", 1),
    ("11:34:10", "11:34:35", "feeder", "toilet_1", 1),
    ("11:35", "12:05", "living_room", "floor", 6),
    ("12:10", "13:00", "feeder", "water_server", 10),
    ("13:05", "16:20", "sofa", "on_kangaroo_chair", 39),
    ("16:25", "17:10", "living_room", "on_tv", 9),
    ("17:15", "18:40", "living_room", "table_top", 17),
    ("18:45", "19:30", "feeder", "feeder", 9),
    ("19:35", "22:50", "sofa", "on_sofa", 39),
    ("22:55", "23:55", "sofa", "under_sofa", 12),
]

KURUMI = [
    ("00:05", "06:10", "living_room", "cat_wall", 73),
    ("06:15", "07:00", "feeder", "feeder", 9),
    ("07:05", "09:30", "living_room", "curtain", 29),
    ("09:35", "11:00", "sofa", "near_side_cabinet", 17),
    ("11:05", "12:30", "sofa", "on_side_cabinet", 17),
    # PASS-BY on the way to the water server: a single appearance, so never a use.
    ("12:33:40", "12:33:55", "feeder", "toilet_1", 1),
    ("12:35", "13:20", "feeder", "water_server", 9),
    ("13:25", "15:00", "living_room", "under_table", 19),
    ("15:05", "17:40", "sofa", "on_sofa", 31),
    ("17:45", "18:20", "living_room", "floor", 7),
    ("18:25", "19:10", "feeder", "feeder", 9),
    ("19:15", "21:00", "sofa", "carpet", 21),
    ("21:05", "22:30", "living_room", "on_white_chair", 17),
    ("22:35", "23:50", "sofa", "under_kangaroo_chair", 15),
]

# What the report should conclude from the data above.
EXPECTED = [
    "bagel 用过厕所（11:32 与 11:34 两次，相距 2 分钟，中间没有其他位置）",
    "bagel 05:44 / 06:30 的两次只是路过（相隔 45 分钟，中间夹着 feeder）",
    "kurumi 只是路过（只出现 1 次）",
]


def stamp(day: str, clock: str) -> float:
    """Accept "HH:MM" or "HH:MM:SS" so toilet glimpses can be sub-minute."""
    if clock.count(":") == 1:
        clock = f"{clock}:00"
    return datetime.fromisoformat(f"{day}T{clock}").replace(tzinfo=TZ).timestamp()


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/cat_demo_day.db")
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)

    store = LocationStore(path)
    total = 0
    try:
        for cat, plan in (("bagel", BAGEL), ("kurumi", KURUMI)):
            for start, end, camera, zone, samples in plan:
                start_ts = stamp(TARGET_DAY, start)
                end_ts = stamp(TARGET_DAY, end)
                confidence = 0.82 + (samples % 15) / 100
                visit_id = store.open_visit(
                    start_ts,
                    cat,
                    camera,
                    zone,
                    confidence,
                    calibration_id=f"{camera}-20260919T102233000",
                )
                store.touch_visit(visit_id, end_ts, samples, confidence)
                # A few raw anchors per visit, so the archive stays realistic.
                for index in range(min(samples, 5)):
                    store.record_observation(
                        start_ts + index * (end_ts - start_ts) / max(1, min(samples, 5)),
                        cat,
                        camera,
                        zone,
                        confidence,
                        norm_x=0.2 + 0.05 * index,
                        norm_y=0.5 + 0.05 * index,
                        calibration_id=f"{camera}-20260919T102233000",
                        alignment_quality="good",
                    )
                total += 1
    finally:
        store.close()

    print(f"wrote {path}: {total} visits across 2 cats")
    for cat, plan in (("bagel", BAGEL), ("kurumi", KURUMI)):
        minutes = sum(
            (
                (int(end.split(":")[0]) * 60 + int(end.split(":")[1]))
                - (int(start.split(":")[0]) * 60 + int(start.split(":")[1]))
            )
            for start, end, *_ in plan
        )
        print(f"  {cat}: {len(plan)} visits, {minutes} min ({minutes // 60}h{minutes % 60:02d}m)")
    print("\n报告应该得出的结论：")
    for line in EXPECTED:
        print(f"  - {line}")


if __name__ == "__main__":
    main()
