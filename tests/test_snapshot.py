"""Tests for the on-demand live frame ("@bot 沙发") the bot answers with.

Everything here runs without a camera, a socket or a Discord connection: the frame
comes from an injected provider and the readers are fakes. What is worth testing is
the part that has bitten this project before - which camera a message means, that a
held stream is released, and that the zones are drawn where they are STORED rather
than projected onto the live frame (projecting them would hide the very
misalignment the owner is looking for).
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.monitoring.snapshot as snapshot
from src.monitoring.alignment import GOOD, AlignmentResult
from src.monitoring.location_zones import Calibration, Zone
from src.monitoring.snapshot import (
    DEFAULT_ALIASES,
    SnapshotCache,
    SnapshotSettings,
    build_snapshot,
    camera_only_reference,
    current_source,
    describe_cameras,
    resolve_camera,
    snapshot_settings,
)

THREE_CAMERAS = ["living_room", "sofa", "feeder"]


def settings(**overrides) -> SnapshotSettings:
    defaults = {
        "triggers": ("画面", "截图"),
        "aliases": dict(DEFAULT_ALIASES),
        "max_width": 1280,
        "jpeg_quality": 80,
        "idle_timeout_seconds": 120.0,
        "frame_timeout_seconds": 5.0,
    }
    return SnapshotSettings(**{**defaults, **overrides})


def synthetic_frame(width: int = 200, height: int = 100) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (40, 40, 40)
    return frame


def square_zone(name: str = "on_sofa") -> Zone:
    return Zone(name=name, points=[(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])


# --- which camera a message means ------------------------------------------


def test_a_camera_can_be_named_by_key_or_by_chinese_alias() -> None:
    assert resolve_camera("看一下 sofa", THREE_CAMERAS, DEFAULT_ALIASES) == "sofa"
    assert resolve_camera("沙发画面", THREE_CAMERAS, DEFAULT_ALIASES) == "sofa"
    assert resolve_camera("feeder 截图", THREE_CAMERAS, DEFAULT_ALIASES) == "feeder"
    assert resolve_camera("来一张客厅的照片", THREE_CAMERAS, DEFAULT_ALIASES) == "living_room"
    # The config key with its underscore written as a space, which is what a person
    # typing an English message produces.
    assert resolve_camera("living room please", THREE_CAMERAS, DEFAULT_ALIASES) == "living_room"


def test_a_message_about_no_camera_resolves_to_none() -> None:
    assert resolve_camera("来一张照片", THREE_CAMERAS, DEFAULT_ALIASES) is None


def test_the_longest_name_wins_so_the_answer_is_not_dict_order() -> None:
    """A shorter alias that is a prefix of another one must not win by accident."""
    aliases = {"\u732b": "feeder", "\u732b\u7897": "living_room"}
    assert resolve_camera("猫碗", THREE_CAMERAS, aliases) == "living_room"


def test_a_bare_mention_of_a_camera_is_a_camera_reference() -> None:
    assert camera_only_reference("<@7> sofa", THREE_CAMERAS, DEFAULT_ALIASES)
    assert camera_only_reference("<@7> 沙发", THREE_CAMERAS, DEFAULT_ALIASES)
    assert camera_only_reference("<@7> 看一下沙发", THREE_CAMERAS, DEFAULT_ALIASES)
    assert camera_only_reference("<@7> show me the sofa", THREE_CAMERAS, DEFAULT_ALIASES)


def test_a_question_that_merely_mentions_a_room_is_not_a_camera_reference() -> None:
    """Why this matters: "沙发" is both a camera and a place the cat sits."""
    assert not camera_only_reference(
        "<@7> 今天两只猫在沙发上待了多久", THREE_CAMERAS, DEFAULT_ALIASES
    )
    assert not camera_only_reference("<@7> 报告一下", THREE_CAMERAS, DEFAULT_ALIASES)
    assert not camera_only_reference(
        "<@7> some cats were on the sofa", THREE_CAMERAS, DEFAULT_ALIASES
    )


def test_describe_cameras_lists_every_way_to_ask_for_one() -> None:
    text = describe_cameras(THREE_CAMERAS, {"沙发": "sofa"})
    assert "sofa（沙发）" in text
    assert "living_room" in text and "feeder" in text


# --- configuration ---------------------------------------------------------


def test_snapshot_defaults_are_usable_without_a_config_block() -> None:
    configured = snapshot_settings({})
    assert configured.triggers == snapshot.DEFAULT_SNAPSHOT_TRIGGERS
    assert configured.aliases["沙发"] == "sofa"
    assert configured.idle_timeout_seconds > 0


def test_a_configured_block_overrides_triggers_aliases_and_sizes() -> None:
    configured = snapshot_settings(
        {
            "triggers": ["喵一张"],
            "aliases": {"喵": "feeder"},
            "max_width": 640,
            "jpeg_quality": 50,
            "idle_timeout_seconds": 30,
        }
    )
    assert configured.triggers == ("喵一张",)
    assert configured.aliases["喵"] == "feeder"
    # The built-in aliases survive an override instead of being replaced wholesale.
    assert configured.aliases["沙发"] == "sofa"
    assert (configured.max_width, configured.jpeg_quality) == (640, 50)
    assert configured.idle_timeout_seconds == 30


def test_an_empty_trigger_list_falls_back_to_the_defaults() -> None:
    """An empty list reads as "no way to ask", which is never what is meant."""
    assert snapshot_settings({"triggers": []}).triggers == snapshot.DEFAULT_SNAPSHOT_TRIGGERS


# --- the address is re-read, because the tracker rewrites .env -------------


def test_the_fresh_env_value_beats_the_one_loaded_at_startup(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / "test.env"
    env_file.write_text(
        "SNAPSHOT_TEST_RTSP_URL=rtsp://admin:pw@10.0.0.9:554/fresh\n", encoding="utf-8"
    )
    monkeypatch.setenv("MLOPS_ENV_FILE", str(env_file))
    monkeypatch.setattr(snapshot, "camera_env_vars", lambda: {"sofa": "SNAPSHOT_TEST_RTSP_URL"})
    monkeypatch.setattr(
        snapshot, "camera_rtsp_url", lambda camera: "rtsp://admin:pw@10.0.0.1:554/stale"
    )
    monkeypatch.setenv("SNAPSHOT_TEST_RTSP_URL", "rtsp://admin:pw@10.0.0.1:554/stale")
    try:
        assert current_source("sofa").endswith("/fresh")
    finally:
        # load_env_file() writes os.environ directly, so monkeypatch cannot undo it.
        import os

        os.environ.pop("SNAPSHOT_TEST_RTSP_URL", None)


def test_a_camera_with_no_placeholder_falls_back_to_the_config(monkeypatch) -> None:
    monkeypatch.setattr(snapshot, "camera_env_vars", lambda: {})
    monkeypatch.setattr(snapshot, "camera_rtsp_url", lambda camera: "rtsp://from/config")
    assert current_source("sofa") == "rtsp://from/config"


# --- holding the stream only while it is being used ------------------------


class ScriptedReader:
    """A reader that hands out an old frame first and a live one after that.

    That is exactly what a held session looks like from the outside: the frame
    already in it predates the request, so returning it as "the live view" would be
    a quiet lie.
    """

    def __init__(self, source: str, name: str, delay_polls: int = 1) -> None:
        self.source = source
        self.name = name
        self.delay_polls = delay_polls
        self.calls = 0
        self.closed = False
        self.started = False
        self.latest_frame = None
        self.latest_frame_at = 0.0

    def start(self) -> None:
        self.started = True
        self.latest_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.latest_frame_at = 0.0

    def get_latest_frame(self):
        self.calls += 1
        if self.calls > self.delay_polls:
            self.latest_frame = np.full((4, 4, 3), 7, dtype=np.uint8)
            self.latest_frame_at = time.monotonic()
        return self.latest_frame

    def close(self) -> None:
        self.closed = True


def cache_with(readers: list) -> SnapshotCache:
    def factory(source: str, name: str) -> ScriptedReader:
        return readers.pop(0)

    return SnapshotCache(
        idle_timeout_seconds=60.0, frame_timeout_seconds=2.0, reader_factory=factory
    )


def test_the_frame_is_one_captured_after_the_request() -> None:
    reader = ScriptedReader("rtsp://x", "sofa/snapshot")
    cache = cache_with([reader])
    frame = cache.frame("sofa", "rtsp://x")
    assert frame is not None and int(frame[0, 0, 0]) == 7
    assert reader.calls >= 2, "the frame already in the session must not be sent"


def test_a_stalled_stream_times_out_instead_of_hanging() -> None:
    reader = ScriptedReader("rtsp://x", "sofa/snapshot", delay_polls=10_000)
    cache = cache_with([reader])
    cache.frame_timeout_seconds = 0.2
    assert cache.frame("sofa", "rtsp://x") is None


def test_the_same_camera_reuses_its_stream_instead_of_redialling() -> None:
    first, second = (
        ScriptedReader("rtsp://x", "sofa/snapshot"),
        ScriptedReader("rtsp://x", "sofa/snapshot"),
    )
    cache = cache_with([first, second])
    cache.frame("sofa", "rtsp://x")
    cache.frame("sofa", "rtsp://x")
    assert second.started is False, "a second request must not open a second stream"
    assert not first.closed


def test_a_camera_that_moved_gets_a_new_stream() -> None:
    """The tracker rewrites .env when a camera relocates; the old stream is dead."""
    stale, fresh = (
        ScriptedReader("rtsp://old", "sofa/snapshot"),
        ScriptedReader("rtsp://new", "sofa/snapshot"),
    )
    cache = cache_with([stale, fresh])
    cache.frame("sofa", "rtsp://old")
    cache.frame("sofa", "rtsp://new")
    assert stale.closed
    assert fresh.started


def test_an_idle_stream_is_released() -> None:
    reader = ScriptedReader("rtsp://x", "sofa/snapshot")
    cache = cache_with([reader])
    cache.frame("sofa", "rtsp://x")
    assert cache.close_idle(now=time.monotonic() + 1_000) == ["sofa"]
    assert reader.closed
    assert cache.sessions == {}


def test_a_stream_that_was_just_used_is_kept() -> None:
    reader = ScriptedReader("rtsp://x", "sofa/snapshot")
    cache = cache_with([reader])
    now = time.monotonic()
    cache.frame("sofa", "rtsp://x")
    assert cache.close_idle(now=now) == []
    assert not reader.closed


# --- the picture -----------------------------------------------------------


def build_for(monkeypatch, tmp_path: Path, content: str, *, cameras=None, frame=None, **extra):
    monkeypatch.setattr(snapshot, "configured_cameras", lambda: list(cameras or THREE_CAMERAS))
    monkeypatch.setattr(snapshot, "current_source", lambda camera: "rtsp://x")
    monkeypatch.setattr(
        snapshot,
        "load_calibrations",
        lambda: {"sofa": Calibration(camera="sofa", zones=[square_zone()])},
    )
    monkeypatch.setattr(snapshot, "alignment_verdict", lambda *a, **k: ("align=good", "一致。"))
    return build_snapshot(
        content,
        settings=settings(),
        frame_provider=lambda camera, source: synthetic_frame() if frame is None else frame,
        now=datetime(2026, 9, 30, 14, 32, 5),
        **extra,
    )


def test_a_named_camera_gets_an_image(monkeypatch, tmp_path: Path) -> None:
    reply = build_for(monkeypatch, tmp_path, "沙发画面")
    assert reply.image_path is not None
    assert reply.image_path.is_file()
    assert "sofa" in reply.caption
    assert "14:32:05" in reply.caption
    assert "区域 1 个" in reply.caption
    assert reply.caption.endswith("一致。")
    reply.image_path.unlink()


def test_the_zones_are_drawn_where_they_are_stored_not_projected(
    monkeypatch, tmp_path: Path
) -> None:
    """The documented decision, pinned.

    A big transform is reported by the alignment verdict, but it must NOT move the
    polygons: if it did, the picture would follow the camera and hide the drift.
    """
    reference = Calibration(
        camera="sofa",
        zones=[square_zone()],
        reference_frame="data/calibrations/sofa/reference_x.jpg",
    )
    monkeypatch.setattr(snapshot, "configured_cameras", lambda: THREE_CAMERAS)
    monkeypatch.setattr(snapshot, "current_source", lambda camera: "rtsp://x")
    monkeypatch.setattr(snapshot, "load_calibrations", lambda: {"sofa": reference})
    monkeypatch.setattr(snapshot, "reference_features_for", lambda *a, **k: object())
    monkeypatch.setattr(
        snapshot,
        "estimate_alignment",
        lambda *a, **k: AlignmentResult(
            GOOD, np.array([[1.0, 0.0, 0.4], [0.0, 1.0, 0.4]]), 50, 60, 0.001
        ),
    )
    reply = build_snapshot(
        "沙发画面",
        settings=settings(),
        frame_provider=lambda camera, source: synthetic_frame(),
        now=datetime(2026, 9, 30, 14, 32, 5),
    )
    assert reply.image_path is not None
    image = cv2.imread(str(reply.image_path))
    reply.image_path.unlink()
    green = (image[:, :, 1] > 200) & (image[:, :, 0] < 100) & (image[:, :, 2] < 100)
    # The polygon's top edge is the stored y=0.25 on a 100px frame, i.e. row 25 from
    # x=50 to x=150. A 0.4 shift would have moved it to row 65.
    assert green[25, 50:151].sum() >= 95
    assert green[65, 50:151].sum() < 95


def test_a_message_naming_no_camera_asks_which_one(monkeypatch, tmp_path: Path) -> None:
    reply = build_for(monkeypatch, tmp_path, "来一张照片")
    assert reply.image_path is None
    assert "sofa" in reply.caption and "feeder" in reply.caption


def test_a_single_camera_is_used_without_being_named(monkeypatch, tmp_path: Path) -> None:
    reply = build_for(monkeypatch, tmp_path, "来一张照片", cameras=["sofa"])
    assert reply.image_path is not None
    assert "sofa" in reply.caption
    reply.image_path.unlink()


def test_an_unreachable_camera_reads_as_a_sentence_not_silence(monkeypatch) -> None:
    monkeypatch.setattr(snapshot, "configured_cameras", lambda: THREE_CAMERAS)
    monkeypatch.setattr(snapshot, "current_source", lambda camera: "rtsp://x")
    monkeypatch.setattr(snapshot, "load_calibrations", lambda: {"sofa": Calibration(camera="sofa")})
    reply = build_snapshot(
        "沙发画面", settings=settings(), frame_provider=lambda camera, source: None
    )
    assert reply.image_path is None
    assert "sofa" in reply.caption
    assert "没抓到" in reply.caption


def test_a_camera_with_no_configured_url_is_explained(monkeypatch) -> None:
    monkeypatch.setattr(snapshot, "configured_cameras", lambda: THREE_CAMERAS)
    monkeypatch.setattr(snapshot, "current_source", lambda camera: "")
    reply = build_snapshot("沙发画面", settings=settings())
    assert reply.image_path is None
    assert "RTSP" in reply.caption


def test_a_camera_missing_from_the_config_is_explained_rather_than_raised(monkeypatch) -> None:
    def explode(camera: str) -> str:
        raise RuntimeError(f"Camera '{camera}' is not defined")

    monkeypatch.setattr(snapshot, "configured_cameras", lambda: THREE_CAMERAS)
    monkeypatch.setattr(snapshot, "current_source", explode)
    reply = build_snapshot("沙发画面", settings=settings())
    assert reply.image_path is None
    assert "拿不到 RTSP 地址" in reply.caption


def test_a_camera_with_no_zones_still_gets_a_picture(monkeypatch) -> None:
    monkeypatch.setattr(snapshot, "configured_cameras", lambda: ["sofa"])
    monkeypatch.setattr(snapshot, "current_source", lambda camera: "rtsp://x")
    monkeypatch.setattr(snapshot, "load_calibrations", lambda: {})
    reply = build_snapshot(
        "画面", settings=settings(), frame_provider=lambda camera, source: synthetic_frame()
    )
    assert reply.image_path is not None
    assert "还没有画过区域" in reply.caption
    reply.image_path.unlink()


# --- the alignment verdict -------------------------------------------------


def test_no_reference_frame_is_said_plainly() -> None:
    subheader, sentence = snapshot.alignment_verdict(Calibration(camera="sofa"), synthetic_frame())
    assert subheader == ""
    assert "还没有参考帧" in sentence


def test_alignment_turned_off_is_reported_instead_of_running_orb(monkeypatch) -> None:
    """Otherwise a deployment that disabled alignment pays for it on every picture."""
    monkeypatch.setitem(snapshot.ALIGNMENT_CONFIG, "enabled", False)
    monkeypatch.setattr(
        snapshot, "estimate_alignment", lambda *a, **k: pytest.fail("alignment is off")
    )
    calibration = Calibration(
        camera="sofa", reference_frame="data/calibrations/sofa/reference_x.jpg"
    )
    subheader, sentence = snapshot.alignment_verdict(calibration, synthetic_frame())
    assert subheader == ""
    assert "已经关了" in sentence


def test_a_bad_match_is_explained_as_usually_being_the_light(monkeypatch) -> None:
    """The re-anchor work measured this: a lighting change collapses the match."""
    calibration = Calibration(
        camera="sofa", reference_frame="data/calibrations/sofa/reference_x.jpg"
    )
    monkeypatch.setattr(snapshot, "reference_features_for", lambda *a, **k: object())
    monkeypatch.setattr(
        snapshot,
        "estimate_alignment",
        lambda *a, **k: AlignmentResult("failed", None, 0, 4, 0.0, "only 4 feature matches"),
    )
    subheader, sentence = snapshot.alignment_verdict(calibration, synthetic_frame())
    assert subheader == "align=failed"
    assert "only 4 feature matches" in sentence
    assert "光线" in sentence


def test_a_good_match_reports_the_transform(monkeypatch) -> None:
    calibration = Calibration(
        camera="sofa", reference_frame="data/calibrations/sofa/reference_x.jpg"
    )
    monkeypatch.setattr(snapshot, "reference_features_for", lambda *a, **k: object())
    monkeypatch.setattr(
        snapshot,
        "estimate_alignment",
        lambda *a, **k: AlignmentResult(GOOD, np.eye(2, 3, dtype=float), 40, 40, 0.0),
    )
    subheader, sentence = snapshot.alignment_verdict(calibration, synthetic_frame())
    assert subheader.startswith("align=good")
    assert "shift=" in subheader
    assert "基本一致" in sentence


def test_an_unwritable_temp_file_is_reported(monkeypatch) -> None:
    monkeypatch.setattr(snapshot, "configured_cameras", lambda: ["sofa"])
    monkeypatch.setattr(snapshot, "current_source", lambda camera: "rtsp://x")
    monkeypatch.setattr(snapshot, "load_calibrations", lambda: {})
    monkeypatch.setattr(snapshot, "_write_jpeg", lambda image, quality: None)
    reply = build_snapshot(
        "画面", settings=settings(), frame_provider=lambda camera, source: synthetic_frame()
    )
    assert reply.image_path is None
    assert "失败" in reply.caption


def test_the_image_is_downscaled_to_the_configured_width(monkeypatch) -> None:
    monkeypatch.setattr(snapshot, "configured_cameras", lambda: ["sofa"])
    monkeypatch.setattr(snapshot, "current_source", lambda camera: "rtsp://x")
    monkeypatch.setattr(snapshot, "load_calibrations", lambda: {})
    reply = build_snapshot(
        "画面",
        settings=settings(max_width=100),
        frame_provider=lambda camera, source: synthetic_frame(width=400, height=200),
    )
    assert reply.image_path is not None
    image = cv2.imread(str(reply.image_path))
    reply.image_path.unlink()
    assert image.shape[1] == 100
    assert image.shape[0] == 50


# --- the spot a coordinate question asked about ----------------------------
#
# The answer carries a frame with the searched point circled, so an owner who meant a
# different spot can see that and ask again. The circle must land on the coordinate the
# question used, in the camera's own frame - the same one the lookup ran against.


def point_build(monkeypatch, camera="sofa", point=(0.5, 0.5), radius=0.1, frame=None):
    monkeypatch.setattr(snapshot, "current_source", lambda name: "rtsp://x")
    monkeypatch.setattr(
        snapshot,
        "load_calibrations",
        lambda: {"sofa": Calibration(camera="sofa", zones=[square_zone()])},
    )
    monkeypatch.setattr(snapshot, "alignment_verdict", lambda *a, **k: ("align=good", "一致。"))
    return snapshot.build_point_snapshot(
        camera,
        point,
        radius,
        settings=settings(),
        frame_provider=lambda name, source: (
            synthetic_frame(width=400, height=200) if frame is None else frame
        ),
    )


def test_a_coordinate_answer_can_come_with_the_spot_marked(monkeypatch) -> None:
    reply = point_build(monkeypatch)
    assert reply.image_path is not None
    assert reply.image_path.is_file()
    assert "50.0" in reply.caption and "sofa" in reply.caption
    image = cv2.imread(str(reply.image_path))
    marker = np.array(snapshot.POINT_COLOR)
    # JPEG is lossy, so the colour is close rather than exact.
    assert (np.abs(image.astype(int) - marker).sum(axis=2) < 90).any()
    reply.image_path.unlink()


def test_the_marker_lands_on_the_normalized_point(monkeypatch) -> None:
    """Normalized, because that is what a coordinate question carries."""
    reply = point_build(monkeypatch, point=(0.25, 0.75), radius=0.05)
    image = cv2.imread(str(reply.image_path))
    height, width = image.shape[:2]
    pixel = image[int(0.75 * height), int(0.25 * width)].astype(int)
    assert np.abs(pixel - np.array(snapshot.POINT_COLOR)).sum() < 90
    reply.image_path.unlink()


def test_a_frame_that_never_arrives_leaves_no_image(monkeypatch) -> None:
    monkeypatch.setattr(snapshot, "current_source", lambda name: "rtsp://x")
    reply = snapshot.build_point_snapshot(
        "sofa",
        (0.5, 0.5),
        0.1,
        settings=settings(),
        frame_provider=lambda name, source: None,
    )
    assert reply.image_path is None
    assert "没抓到" in reply.caption


# --- the coordinate reference picture --------------------------------------
#
# "沙发参考图": the live frame plus a 0-100 tick grid. The scale is the one the
# questions speak, so the number read off a line is the number that can be typed back
# into "sofa 的 x30 y70 附近谁待过".


def test_the_grid_is_drawn_on_the_0_100_scale() -> None:
    frame = synthetic_frame(width=200, height=100)
    before = frame.copy()
    snapshot.draw_grid(frame, 10)

    # Value 50 lands on the middle of the frame; a cell interior is left alone.
    assert not np.array_equal(frame[5, 100], before[5, 100])
    assert np.array_equal(frame[5, 90], before[5, 90])
    assert not np.array_equal(frame[50, 150], before[50, 150])
    assert np.array_equal(frame[45, 150], before[45, 150])


def test_the_grid_step_is_configurable() -> None:
    frame = synthetic_frame(width=200, height=100)
    before = frame.copy()
    snapshot.draw_grid(frame, 40)

    # 40 and 80 are lines at step 40; 50 is not, so a 10-grid is not being drawn.
    assert not np.array_equal(frame[5, 160], before[5, 160])
    assert np.array_equal(frame[5, 100], before[5, 100])


def test_the_grid_defaults_are_usable_without_a_config_block() -> None:
    configured = snapshot_settings({})
    assert configured.grid_triggers == snapshot.DEFAULT_GRID_TRIGGERS
    assert configured.grid_step == snapshot.DEFAULT_GRID_STEP


def test_the_grid_block_is_read_from_the_config(monkeypatch) -> None:
    monkeypatch.setattr(snapshot, "SNAPSHOT_CONFIG", {"grid_triggers": ["喵刻度"], "grid_step": 25})
    configured = snapshot.snapshot_settings()
    assert configured.grid_triggers == ("喵刻度",)
    assert configured.grid_step == 25


def test_a_reference_request_returns_a_frame_with_a_caption(monkeypatch) -> None:
    monkeypatch.setattr(snapshot, "current_source", lambda name: "rtsp://x")
    reply = snapshot.build_grid_snapshot(
        "沙发参考图",
        settings=settings(),
        frame_provider=lambda name, source: synthetic_frame(320, 180),
    )
    assert reply.image_path is not None
    assert "sofa" in reply.caption
    # The owner has to be told what the numbers mean and how to use them.
    assert "刻度" in reply.caption and "x30 y70" in reply.caption
    assert cv2.imread(str(reply.image_path)) is not None
    reply.image_path.unlink()


def test_a_reference_request_without_a_camera_asks_which_one(monkeypatch) -> None:
    monkeypatch.setattr(snapshot, "current_source", lambda name: "rtsp://x")
    reply = snapshot.build_grid_snapshot(
        "参考图", settings=settings(), frame_provider=lambda name, source: synthetic_frame()
    )
    assert reply.image_path is None
    assert "哪个相机" in reply.caption
