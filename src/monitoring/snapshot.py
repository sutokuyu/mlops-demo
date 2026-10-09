"""Send one live camera frame with the zone polygons drawn on it, on request.

The owner wanted to aim a camera without opening the editor: ask the Discord bot for
a picture, see where the zones land, nudge the camera, ask again.

Two decisions shape the picture.

**The polygons are drawn where they are stored** - in the coordinate system of the
camera's reference frame - and not projected onto the live frame through an alignment
transform. Projecting them would drag them along with whatever the camera is doing
and hide the very misalignment the owner is looking for. Drawn as stored, a polygon
that no longer sits on its furniture is the answer; the alignment verdict in the
caption says whether the difference came from the camera moving or from the light.

**The stream is held open only while the owner keeps asking.** This machine already
records all three cameras through one Wi-Fi relay, so a stream per request would add
an RTSP session per message. Instead one session per camera is opened on demand,
refreshed by each request and released after ``idle_timeout_seconds``. An aiming
session therefore costs one extra session and answers instantly - the same bargain
the browser preview makes, and the relay has dropped all three streams before when
it was pushed harder than that.

Nothing here imports ``discord``: the message text goes in, a caption and a local
JPEG come out, and the bot's Gateway shell does the uploading.
"""

import os
import re
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def _resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not locate project root")


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_env_file
from src.monitoring.alignment import (
    GOOD,
    describe_transform,
    estimate_alignment,
)
from src.monitoring.camera_discovery import camera_env_vars
from src.monitoring.location_config import (
    ALIGNMENT_CONFIG,
    SNAPSHOT_CONFIG,
    alignment_settings,
    camera_rtsp_url,
    configured_cameras,
)
from src.monitoring.location_zones import Calibration, load_calibrations
from src.monitoring.recalibration import HEADER_HEIGHT, reference_features_for, render_overlay
from src.monitoring.rtsp_stream import StreamReader

# Words that ask for a picture. Deliberately NOT the report triggers ("报告/日报/report"):
# the two answers are entirely different, so a message must not be read as both. The
# list also deliberately leaves out the vague "看看/看一下", because "看看今天猫干啥了"
# is a report question; a message with nothing but a camera name in it is covered
# separately by camera_only_reference().
DEFAULT_SNAPSHOT_TRIGGERS = (
    "画面",
    "截图",
    "照片",
    "拍一张",
    "来一张",
    "snapshot",
    "frame",
    "photo",
    "picture",
)

# Words that ask for the coordinate reference picture: the live frame plus a 0-100
# tick grid, so the owner can read a coordinate off it and type it back. A separate
# list from the plain picture words because the two pictures answer different
# questions - "沙发画面" is "is the camera aimed right", "沙发参考图" is "where is
# x30 y70". "参考图" is also the word the owner already uses for it.
DEFAULT_GRID_TRIGGERS = (
    "参考图",
    "刻度图",
    "坐标图",
    "坐标网格",
    "grid",
)

# How a camera can be called in a message. The Chinese names are here because the
# owner talks about rooms, not about config keys. Overridable per deployment under
# ``discord_bot.snapshot.aliases``.
DEFAULT_ALIASES = {
    "客厅": "living_room",
    "起居室": "living_room",
    "沙发": "sofa",
    "喂食器": "feeder",
    "猫碗": "feeder",
    "饭盆": "feeder",
}

DEFAULT_MAX_WIDTH = 1280
DEFAULT_JPEG_QUALITY = 80
# Ticks every 10 units of the 0-100 scale the questions use (see
# ``location_queries.parse_point``), so the numbers on the picture and the numbers the
# owner types are the same numbers.
DEFAULT_GRID_STEP = 10
# Long enough that a slow re-ask does not re-dial the camera, short enough that an
# abandoned aiming session is not one more stream held all night.
DEFAULT_IDLE_TIMEOUT_SECONDS = 120.0
# Opening an RTSP stream costs a DESCRIBE plus a keyframe wait; the tracker's own
# readers report frames within a few seconds, so this only has to outlast a stall.
DEFAULT_FRAME_TIMEOUT_SECONDS = 20.0

# The marker for the spot a point question asked about, in BGR. Deliberately not the
# zone green: this is the answer's target, not a stored polygon.
POINT_COLOR = (0, 140, 255)

# The coordinate grid. Amber, and drawn faintly: it is there to be read off, and a
# grid heavy enough to hide the cat would defeat the purpose of a live picture.
GRID_COLOR = (0, 220, 255)
GRID_TEXT_COLOR = (0, 220, 255)
GRID_ALPHA = 0.35
GRID_FONT_SCALE = 0.5

MENTION_PATTERN = re.compile(r"<@[!&]?\d+>|@everyone|@here")

# Words that add nothing to "show me this camera", stripped along with the camera
# names when deciding whether a message is ONLY about a camera. The test is "is
# anything left?" and it only runs for a real @mention, so a generous list cannot turn
# a question into a picture request - a question always leaves content words behind.
SOFT_WORDS = (
    "看一下",
    "看一眼",
    "看看",
    "瞅瞅",
    "来一张",
    "给我",
    "帮我",
    "一张",
    "一下",
    "画面",
    "截图",
    "照片",
    "please",
    "show me",
    "the",
    "me",
    "看",
    "瞅",
    "来",
    "的",
    "个",
    "张",
)


@dataclass(frozen=True)
class SnapshotSettings:
    """Everything the picture needs, resolved from ``discord_bot.snapshot``."""

    triggers: tuple[str, ...] = DEFAULT_SNAPSHOT_TRIGGERS
    aliases: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_ALIASES))
    max_width: int = DEFAULT_MAX_WIDTH
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS
    frame_timeout_seconds: float = DEFAULT_FRAME_TIMEOUT_SECONDS
    grid_triggers: tuple[str, ...] = DEFAULT_GRID_TRIGGERS
    grid_step: int = DEFAULT_GRID_STEP


def snapshot_settings(config: dict | None = None) -> SnapshotSettings:
    """Read the ``discord_bot.snapshot`` block, falling back to the defaults above."""
    raw = (config if config is not None else SNAPSHOT_CONFIG) or {}
    triggers = raw.get("triggers")
    grid_triggers = raw.get("grid_triggers")
    aliases = dict(DEFAULT_ALIASES)
    configured = raw.get("aliases")
    if isinstance(configured, Mapping):
        aliases.update({str(alias): str(camera) for alias, camera in configured.items()})
    return SnapshotSettings(
        triggers=(
            DEFAULT_SNAPSHOT_TRIGGERS
            if triggers in (None, "", [])
            else tuple(str(item).strip() for item in triggers if str(item).strip())
        ),
        aliases=aliases,
        max_width=int(raw.get("max_width", DEFAULT_MAX_WIDTH)),
        jpeg_quality=int(raw.get("jpeg_quality", DEFAULT_JPEG_QUALITY)),
        idle_timeout_seconds=float(raw.get("idle_timeout_seconds", DEFAULT_IDLE_TIMEOUT_SECONDS)),
        frame_timeout_seconds=float(
            raw.get("frame_timeout_seconds", DEFAULT_FRAME_TIMEOUT_SECONDS)
        ),
        grid_triggers=(
            DEFAULT_GRID_TRIGGERS
            if grid_triggers in (None, "", [])
            else tuple(str(item).strip() for item in grid_triggers if str(item).strip())
        ),
        grid_step=int(raw.get("grid_step", DEFAULT_GRID_STEP)),
    )


def camera_labels(
    cameras: Sequence[str], aliases: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Every string that can name a camera, lower-cased, mapped to the camera.

    The config key ("living_room") and its spaced form ("living room") both work, so
    an English message does not need the underscores right.
    """
    labels: dict[str, str] = {}
    for camera in cameras:
        labels[camera.lower()] = camera
        labels[camera.replace("_", " ").lower()] = camera
    for alias, camera in (aliases or {}).items():
        labels[str(alias).lower()] = str(camera)
    return labels


def resolve_camera(
    content: str, cameras: Sequence[str], aliases: Mapping[str, str] | None = None
) -> str | None:
    """Which camera the message names, or ``None`` when it names none.

    The longest label wins, so a message containing both "客厅" and "沙发" resolves
    deterministically instead of by dict order.
    """
    lowered = content.lower()
    hits = [
        (len(label), camera)
        for label, camera in camera_labels(cameras, aliases).items()
        if label and label in lowered
    ]
    if not hits:
        return None
    hits.sort(key=lambda item: (-item[0], item[1]))
    return hits[0][1]


def _strip(text: str, word: str) -> str:
    """Remove a label, with word boundaries around an ASCII one.

    The boundary matters for the short English soft words: a bare ``me`` must not eat
    the ``me`` in "some". A Chinese word needs no boundary - the script has no spaces
    to be fooled by.
    """
    if word.isascii():
        pattern = rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])"
    else:
        pattern = re.escape(word)
    return re.sub(pattern, " ", text, flags=re.IGNORECASE)


def camera_only_reference(
    content: str, cameras: Sequence[str], aliases: Mapping[str, str] | None = None
) -> bool:
    """True when the message is nothing but a camera's name, e.g. "@bot sofa".

    This is the shortest way to ask for a picture, and it is why the camera names are
    stripped out before deciding - along with the handful of filler words in
    SOFT_WORDS, so "@bot 看一下沙发" works too. A question that merely mentions a room
    ("今天两只猫在沙发上待了多久") leaves plenty of words behind, so it stays a report
    request, which matters because "沙发" is both a camera and a place the cat sits.
    """
    residue = MENTION_PATTERN.sub(" ", content)
    for label in camera_labels(cameras, aliases):
        if label:
            residue = _strip(residue, label)
    for word in SOFT_WORDS:
        residue = _strip(residue, word)
    return not residue.strip(" \t\r\n，。、,.!！?？~～:：;；")


def describe_cameras(cameras: Sequence[str], aliases: Mapping[str, str] | None = None) -> str:
    """The cameras and the names that select them, for a "which one?" reply."""
    hints = []
    for camera in cameras:
        names = [alias for alias, target in (aliases or {}).items() if target == camera]
        hints.append(f"{camera}（{'/'.join(names)}）" if names else camera)
    return "、".join(hints)


def current_source(camera: str) -> str:
    """The camera's RTSP URL, re-read from .env so a relocated camera is still found.

    The tracker rewrites .env the moment it finds a camera at a new address, and this
    bot outlives that by weeks. ``load_env_file()`` is a no-op the second time, so the
    refresh has to be asked for explicitly; without it a snapshot would keep dialling
    an address the relay abandoned and look like a broken camera.
    """
    key = camera_env_vars().get(camera)
    if key:
        load_env_file(force=True)
        fresh = os.environ.get(key, "").strip()
        if fresh:
            return fresh
    return camera_rtsp_url(camera)


@dataclass
class CameraSession:
    """One camera's held stream, with the moment it was last asked for."""

    camera: str
    source: str
    reader: StreamReader
    last_used: float


class SnapshotCache:
    """One live reader per camera, released once the owner stops asking."""

    def __init__(
        self,
        idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
        frame_timeout_seconds: float = DEFAULT_FRAME_TIMEOUT_SECONDS,
        reader_factory=StreamReader,
    ) -> None:
        self.idle_timeout_seconds = idle_timeout_seconds
        self.frame_timeout_seconds = frame_timeout_seconds
        self.reader_factory = reader_factory
        self.sessions: dict[str, CameraSession] = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.janitor: threading.Thread | None = None

    def start_janitor(self, interval_seconds: float = 15.0) -> None:
        """Release idle streams in the background, so the last request cannot leak one.

        A janitor thread rather than a check on the next request, because the next
        request may never come - and then the stream would be held all night.
        """
        if self.janitor is not None and self.janitor.is_alive():
            return
        self.stop_event = threading.Event()
        self.janitor = threading.Thread(
            target=self._janitor_loop,
            args=(interval_seconds,),
            name="snapshot-janitor",
            daemon=True,
        )
        self.janitor.start()

    def _janitor_loop(self, interval_seconds: float) -> None:
        while not self.stop_event.wait(interval_seconds):
            self.close_idle()

    def frame(self, camera: str, source: str) -> np.ndarray | None:
        """A frame captured after this call started, or ``None`` if none arrived.

        Freshness matters: a held session has an old frame sitting in it, and sending
        that as "the live view" would be a quiet lie.
        """
        now = time.monotonic()
        with self.lock:
            session = self.sessions.get(camera)
            if session is not None and session.source != source:
                # The camera moved; the address it was opened at is worthless.
                self.sessions.pop(camera)
                session.reader.close()
                session = None
            if session is None:
                reader = self.reader_factory(source=source, name=f"{camera}/snapshot")
                reader.start()
                session = CameraSession(camera, source, reader, now)
                self.sessions[camera] = session
            session.last_used = now

        deadline = now + self.frame_timeout_seconds
        while time.monotonic() < deadline:
            frame = session.reader.get_latest_frame()
            if frame is not None and session.reader.latest_frame_at > now:
                return frame
            time.sleep(0.05)
        return None

    def close_idle(self, now: float | None = None) -> list[str]:
        """Close every stream untouched for longer than the idle timeout."""
        moment = time.monotonic() if now is None else now
        expired: list[CameraSession] = []
        with self.lock:
            for camera, session in list(self.sessions.items()):
                if moment - session.last_used >= self.idle_timeout_seconds:
                    expired.append(self.sessions.pop(camera))
        for session in expired:
            session.reader.close()
        return [session.camera for session in expired]

    def close_all(self) -> None:
        self.stop_event.set()
        with self.lock:
            sessions = list(self.sessions.values())
            self.sessions.clear()
        for session in sessions:
            session.reader.close()


_DEFAULT_CACHE: SnapshotCache | None = None
_DEFAULT_CACHE_LOCK = threading.Lock()


def default_cache() -> SnapshotCache:
    """The process-wide cache, with its janitor running."""
    global _DEFAULT_CACHE
    with _DEFAULT_CACHE_LOCK:
        if _DEFAULT_CACHE is None:
            settings = snapshot_settings()
            _DEFAULT_CACHE = SnapshotCache(
                idle_timeout_seconds=settings.idle_timeout_seconds,
                frame_timeout_seconds=settings.frame_timeout_seconds,
            )
            _DEFAULT_CACHE.start_janitor()
        return _DEFAULT_CACHE


def alignment_verdict(
    calibration: Calibration | None, frame: np.ndarray, settings: dict | None = None
) -> tuple[str, str]:
    """``(ascii subheader, Chinese sentence)`` comparing the frame to the reference.

    The two come back together because they say the same thing to two audiences: the
    subheader is burned into the image (OpenCV cannot draw Chinese) and the sentence
    goes into the caption, where it can explain what a poor match does and does not
    mean. A weak match is usually a lighting change, not a moved camera - that
    distinction is the whole reason ``reanchor_mode`` is ``alert`` here.
    """
    settings = settings or alignment_settings()
    if not ALIGNMENT_CONFIG.get("enabled", True):
        return "", "对齐功能已经关了，区域按参考帧的原始坐标显示。"
    reference = (
        reference_features_for(calibration, settings["work_width"])
        if calibration is not None and calibration.reference_path
        else None
    )
    if reference is None:
        return "", "这个相机还没有参考帧，区域位置没法核对。"

    result = estimate_alignment(
        reference,
        frame,
        work_width=settings["work_width"],
        min_inliers=settings["min_inliers"],
        good_inlier_ratio=settings["good_inlier_ratio"],
        max_residual=settings["max_residual"],
    )
    if result.matrix is None:
        return (
            f"align={result.quality}",
            f"画面和参考帧对不上号（{result.reason}）。多数时候是光线变了，区域位置照旧；"
            "但要是区域明显没落在该在的地方，就是这个相机要调了。",
        )

    transform = describe_transform(result.matrix)
    if result.quality == GOOD:
        return f"align={result.quality} {transform}", f"画面和参考帧基本一致（{transform}）。"
    return (
        f"align={result.quality} {transform}",
        f"画面和参考帧有差距（{transform}），可能是光线也可能是相机歪了。",
    )


def _resize_for_display(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if max_width <= 0 or width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(frame, (max_width, max(1, int(round(height * scale)))))


def _write_jpeg(image: np.ndarray, quality: int) -> Path | None:
    handle, name = tempfile.mkstemp(prefix="cat-snapshot-", suffix=".jpg")
    os.close(handle)
    path = Path(name)
    if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]):
        path.unlink(missing_ok=True)
        return None
    return path


@dataclass(frozen=True)
class SnapshotReply:
    """What to say, and the local image to upload with it (None when there is none)."""

    caption: str
    image_path: Path | None = None


def build_snapshot(
    content: str,
    *,
    settings: SnapshotSettings | None = None,
    cache: SnapshotCache | None = None,
    now: datetime | None = None,
    frame_provider=None,
) -> SnapshotReply:
    """Resolve the camera, grab one frame, draw the zones, write a JPEG.

    Every camera problem - unknown name, no URL, a stream that never delivers, an
    unwritable temp file - comes back as a sentence, because a picture that cannot be
    taken has to be explainable in the channel. Unexpected failures still raise; the
    bot's own wrapper turns those into a reply too, so nothing is ever silent.
    ``frame_provider`` is the seam the tests use to skip RTSP entirely.
    """
    settings = settings or snapshot_settings()
    cameras = configured_cameras()
    camera = resolve_camera(content, cameras, settings.aliases)
    if camera is None:
        if len(cameras) == 1:
            camera = cameras[0]
        else:
            return SnapshotReply(
                "要哪个相机的画面？消息里带上名字就行："
                + describe_cameras(cameras, settings.aliases)
                + "。"
            )

    try:
        source = current_source(camera)
    except RuntimeError as error:
        return SnapshotReply(f"`{camera}` 拿不到 RTSP 地址：{error}")
    if not source:
        return SnapshotReply(f"`{camera}` 还没有配置 RTSP 地址，抓不了画面。")

    provider = frame_provider or (cache or default_cache()).frame
    frame = provider(camera, source)
    if frame is None:
        return SnapshotReply(
            f"`{camera}` 的画面没抓到（等超时了）。相机可能正忙或者不在线，过一会儿再要一次。"
        )

    calibration = load_calibrations().get(camera)
    zones = list(calibration.zones) if calibration is not None else []
    subheader, verdict = alignment_verdict(calibration, frame)

    moment = now or datetime.now()
    header = f"{camera} live {moment.strftime('%Y-%m-%d %H:%M:%S')}  zones={len(zones)}"
    image = render_overlay(_resize_for_display(frame, settings.max_width), zones, header, subheader)
    image_path = _write_jpeg(image, settings.jpeg_quality)
    if image_path is None:
        return SnapshotReply(f"`{camera}` 的画面抓到了，但存成图片失败，再看一眼日志。")

    lines = [f"📷 `{camera}` 的实时画面（{moment.strftime('%H:%M:%S')}）"]
    if zones:
        lines.append(
            f"区域 {len(zones)} 个，按参考帧的原始坐标画在图上："
            "区域没落在该在的家具上，就说明这个相机需要调角度了。"
        )
    else:
        lines.append("这个相机还没有画过区域，图上只有画面本身。")
    lines.append(verdict)
    return SnapshotReply("\n".join(lines), image_path)


def _grid_label(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    """A tick number with a dark halo, so it reads on white furniture and black alike.

    Measured on a real sofa frame: a bare amber digit disappears against the fridge and
    the cabinets. Drawing it twice - thick black, then the colour - costs one call and
    makes the number readable wherever the line lands.
    """
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, GRID_FONT_SCALE, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        GRID_FONT_SCALE,
        GRID_TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def draw_grid(image: np.ndarray, step: int = DEFAULT_GRID_STEP, top: int = 0) -> None:
    """Draw the 0-100 read-off grid the owner uses to type coordinates.

    The scale is the one the questions speak (:func:`location_queries.parse_point` reads
    x/y as 0-100), so the number printed on a line is the number the owner types back.
    The lines are faint on purpose: a grid heavy enough to hide the cat would defeat the
    point of a live picture.

    Labels run along the bottom and the left edge, because the top of the frame is the
    header bar and the alignment line. ``top`` is where the vertical lines may start, so
    they do not stripe the header.
    """
    if step <= 0:
        return
    height, width = image.shape[:2]
    limits = range(step, 100, step)
    tint = image.copy()
    for value in limits:
        x = int(round(value / 100 * width))
        y = int(round(value / 100 * height))
        if 0 < x < width:
            cv2.line(tint, (x, top), (x, height), GRID_COLOR, 1)
        if 0 < y < height:
            cv2.line(tint, (0, y), (width, y), GRID_COLOR, 1)
    image[:] = cv2.addWeighted(tint, GRID_ALPHA, image, 1 - GRID_ALPHA, 0)

    for value in limits:
        x = int(round(value / 100 * width))
        y = int(round(value / 100 * height))
        if 0 < x < width:
            _grid_label(image, str(value), (max(2, x - 10), max(14, height - 8)))
        if 0 < y < height:
            _grid_label(image, str(value), (4, y + 14))


def build_grid_snapshot(
    content: str,
    *,
    settings: SnapshotSettings | None = None,
    cache: SnapshotCache | None = None,
    now: datetime | None = None,
    frame_provider=None,
) -> SnapshotReply:
    """The live frame plus a 0-100 tick grid, so a coordinate can be read off it.

    The same picture as :func:`build_snapshot` with the grid added, because the two
    requests are neighbours: one checks where the zones land, this one produces a
    coordinate to type into a question like "sofa 的 x30 y70 附近谁待过". Camera problems
    come back as a sentence, exactly as they do there.
    """
    settings = settings or snapshot_settings()
    cameras = configured_cameras()
    camera = resolve_camera(content, cameras, settings.aliases)
    if camera is None:
        if len(cameras) == 1:
            camera = cameras[0]
        else:
            return SnapshotReply(
                "要哪个相机的参考图？消息里带上名字就行："
                + describe_cameras(cameras, settings.aliases)
                + "（例如 `沙发参考图`）。"
            )

    try:
        source = current_source(camera)
    except RuntimeError as error:
        return SnapshotReply(f"`{camera}` 拿不到 RTSP 地址：{error}")
    if not source:
        return SnapshotReply(f"`{camera}` 还没有配置 RTSP 地址，画不了参考图。")

    provider = frame_provider or (cache or default_cache()).frame
    frame = provider(camera, source)
    if frame is None:
        return SnapshotReply(
            f"`{camera}` 的画面没抓到（等超时了）。相机可能正忙或者不在线，过一会儿再要一次。"
        )

    calibration = load_calibrations().get(camera)
    zones = list(calibration.zones) if calibration is not None else []
    subheader, verdict = alignment_verdict(calibration, frame)

    moment = now or datetime.now()
    header = (
        f"{camera} reference {moment.strftime('%Y-%m-%d %H:%M:%S')}"
        f"  grid={settings.grid_step} zones={len(zones)}"
    )
    image = render_overlay(_resize_for_display(frame, settings.max_width), zones, header, subheader)
    draw_grid(image, settings.grid_step, top=HEADER_HEIGHT)
    image_path = _write_jpeg(image, settings.jpeg_quality)
    if image_path is None:
        return SnapshotReply(f"`{camera}` 的画面抓到了，但存成图片失败，再看一眼日志。")

    lines = [
        f"📐 `{camera}` 的坐标参考图（{moment.strftime('%H:%M:%S')}）",
        (
            f"刻度每 {settings.grid_step} 一格；横轴 x 向右、纵轴 y 向下，都是 0-100"
            "（图的四条边就是 0 和 100）。"
        ),
    ]
    if zones:
        lines.append(
            f"区域 {len(zones)} 个，按参考帧坐标画在图上：网格和区域对不上，就是这台相机要调了。"
        )
    else:
        lines.append("这个相机还没有画过区域，图上只有画面和网格。")
    lines.append(
        "读到坐标后这样问本鱼：`x30 y70 附近谁待过`——别忘了带相机名，"
        "同一组数字在别的相机上是另一个地方。"
    )
    lines.append(verdict)
    return SnapshotReply("\n".join(lines), image_path)


def draw_point(image: np.ndarray, point: tuple[float, float], radius: float) -> None:
    """Circle + crosshair at a normalized point, with the radius it was matched with.

    Drawn in the camera's own coordinates - the same ones the point question matched
    against - so a circle that is not on the thing the owner meant is exactly the
    signal that the coordinates were wrong.
    """
    height, width = image.shape[:2]
    cx = int(round(float(point[0]) * width))
    cy = int(round(float(point[1]) * height))
    pixels = max(4, int(round(float(radius) * width)))
    cv2.circle(image, (cx, cy), pixels, POINT_COLOR, 2)
    cv2.drawMarker(image, (cx, cy), POINT_COLOR, cv2.MARKER_CROSS, 22, 2)
    # ASCII only: cv2.putText cannot draw the Chinese that would read better here. The
    # 0-100 form is shown because that is what the owner reads off the preview and types.
    label = (
        f"point {float(point[0]) * 100:.1f},{float(point[1]) * 100:.1f} (0-100)"
        f"  r={float(radius) * 100:.1f}"
    )
    # Keep the label on the frame when the point sits near an edge.
    tx = min(max(cx + 12, 4), max(4, width - 320))
    ty = min(max(cy - 12, 20), max(20, height - 8))
    cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, POINT_COLOR, 2)


def build_point_snapshot(
    camera: str,
    point: tuple[float, float],
    radius: float,
    *,
    settings: SnapshotSettings | None = None,
    cache: SnapshotCache | None = None,
    frame_provider=None,
) -> SnapshotReply:
    """One live frame with the asked-about spot marked, so the owner can check it.

    Sent with a coordinate answer so a circle that landed on the wrong place can be
    corrected by asking again - the whole reason the picture is attached. Camera
    problems come back as a sentence, exactly like :func:`build_snapshot`.
    """
    settings = settings or snapshot_settings()
    try:
        source = current_source(camera)
    except RuntimeError as error:
        return SnapshotReply(f"`{camera}` 拿不到 RTSP 地址：{error}")
    if not source:
        return SnapshotReply(f"`{camera}` 还没有配置 RTSP 地址，标不了点。")

    provider = frame_provider or (cache or default_cache()).frame
    frame = provider(camera, source)
    if frame is None:
        return SnapshotReply(f"`{camera}` 的画面没抓到，标不了点（相机可能正忙或不在线）。")

    calibration = load_calibrations().get(camera)
    zones = list(calibration.zones) if calibration is not None else []
    subheader, verdict = alignment_verdict(calibration, frame)
    moment = datetime.now()
    header = (
        f"{camera} live {moment.strftime('%Y-%m-%d %H:%M:%S')}"
        f"  point=({point[0] * 100:.1f},{point[1] * 100:.1f}) r={radius * 100:.1f} (0-100)"
    )
    image = render_overlay(_resize_for_display(frame, settings.max_width), zones, header, subheader)
    draw_point(image, point, radius)
    image_path = _write_jpeg(image, settings.jpeg_quality)
    if image_path is None:
        return SnapshotReply(f"`{camera}` 的画面抓到了，但存成图片失败，再看一眼日志。")

    caption = (
        f"📍 图上橙色的圈就是你问的那个点：`{camera}` x={point[0] * 100:.1f} y={point[1] * 100:.1f}"
        f"（0-100 体系），半径 {radius * 100:.1f}。圈的不是你想的地方就换个坐标再问一次。"
    )
    return SnapshotReply(caption, image_path)
