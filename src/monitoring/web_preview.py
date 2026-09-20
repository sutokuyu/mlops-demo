"""Browser preview and zone editor for the location tracker.

WSLg cannot keep an OpenCV window visible when the monitor layout changes, so the
operator UI is served over HTTP instead of using ``cv2.imshow``: an MJPEG stream
for the live view plus a canvas overlay that posts zone polygons back to the
server. Saving goes through the same re-anchor path the desktop editor uses, so
the calibration format, the reference frame and the Discord overlay are identical.

Endpoints:

* ``GET  /``                     the editor page
* ``GET  /stream/<camera>.mjpg`` live view with detection boxes and anchors
* ``GET  /stream/<camera>/clean.mjpg`` live view without any drawing
* ``GET  /api/status``           per-camera capture status, polled by the UI
* ``GET  /api/zones/<camera>``   zones currently stored for a camera
* ``POST /api/zones/<camera>``   replace a camera's zones and re-anchor
* ``POST /api/reanchor/<camera>`` project the stored zones onto a fresh frame
"""

import json
import re
import sys
import threading
import time
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

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

from src.monitoring.location_zones import ZONES_PATH, Calibration, Zone, load_calibrations
from src.monitoring.recalibration import reanchor

STREAM_BOUNDARY = "frame"
POLL_INTERVAL_SECONDS = 0.05
DEFAULT_DISPLAY_WIDTH = 1280
DEFAULT_JPEG_QUALITY = 80
DEFAULT_MESSAGE_LIMIT = 300
ZONES_ROUTE = re.compile(r"^/api/zones/([^/]+)$")
REANCHOR_ROUTE = re.compile(r"^/api/reanchor/([^/]+)$")
STREAM_ROUTE = re.compile(r"^/stream/([^/]+?)(/clean)?\.mjpg$")
SNAPSHOT_ROUTE = re.compile(r"^/api/snapshot/([^/]+)\.jpg$")


def _resize_for_display(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if max_width <= 0 or width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(frame, (max_width, max(1, int(round(height * scale)))))


def _encode_jpeg(frame: np.ndarray, quality: int) -> bytes | None:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buffer.tobytes() if ok else None


@dataclass
class CameraFeed:
    """Newest frames for one camera plus the status the UI displays."""

    name: str
    status: str = "starting"
    width: int = 0
    height: int = 0
    annotated_jpeg: bytes | None = None
    clean_jpeg: bytes | None = None
    clean_frame: np.ndarray | None = None
    updated_at: float = 0.0
    frame_count: int = 0
    detections: int = 0
    best_confidence: float = 0.0

    @property
    def age_seconds(self) -> float | None:
        return time.monotonic() - self.updated_at if self.updated_at else None


class FrameHub:
    """Thread-safe registry of the newest frame of every camera."""

    def __init__(self, display_width: int = DEFAULT_DISPLAY_WIDTH, quality: int = 80) -> None:
        self.lock = threading.Lock()
        self.display_width = display_width
        self.quality = quality
        self.feeds: dict[str, CameraFeed] = {}

    def register(self, name: str) -> None:
        with self.lock:
            self.feeds.setdefault(name, CameraFeed(name=name))

    def names(self) -> list[str]:
        with self.lock:
            return list(self.feeds)

    def get(self, name: str) -> CameraFeed | None:
        """Return a shallow copy so readers never observe a half-written feed."""
        with self.lock:
            feed = self.feeds.get(name)
            if feed is None:
                return None
            return CameraFeed(
                name=feed.name,
                status=feed.status,
                width=feed.width,
                height=feed.height,
                annotated_jpeg=feed.annotated_jpeg,
                clean_jpeg=feed.clean_jpeg,
                clean_frame=feed.clean_frame,
                updated_at=feed.updated_at,
                frame_count=feed.frame_count,
                detections=feed.detections,
                best_confidence=feed.best_confidence,
            )

    def set_status(self, name: str, status: str) -> None:
        with self.lock:
            feed = self.feeds.get(name)
            if feed is not None:
                feed.status = status

    def publish(
        self,
        name: str,
        clean_frame: np.ndarray,
        annotated_frame: np.ndarray,
        detections: int = 0,
        best_confidence: float = 0.0,
    ) -> None:
        height, width = clean_frame.shape[:2]
        annotated = _encode_jpeg(
            _resize_for_display(annotated_frame, self.display_width), self.quality
        )
        clean = _encode_jpeg(_resize_for_display(clean_frame, self.display_width), self.quality)
        with self.lock:
            feed = self.feeds.get(name)
            if feed is None:
                return
            if annotated is not None:
                feed.annotated_jpeg = annotated
            if clean is not None:
                feed.clean_jpeg = clean
            feed.clean_frame = clean_frame
            feed.width = width
            feed.height = height
            feed.status = "live"
            feed.updated_at = time.monotonic()
            feed.frame_count += 1
            feed.detections = detections
            feed.best_confidence = best_confidence


@dataclass
class LocationEvent:
    """One "the cat is here" report, shown in the browser location panel."""

    seq: int
    camera: str
    cat: str
    zone: str | None
    quality: str
    confidence: float
    norm_x: float
    norm_y: float
    text: str
    ts: float


class MessageLog:
    """Bounded, thread-safe log of location reports for the browser panel.

    Reports are appended only when a place changes, so this reads like a
    movement timeline rather than one line per sample.
    """

    def __init__(self, limit: int = DEFAULT_MESSAGE_LIMIT) -> None:
        self.lock = threading.Lock()
        self.limit = limit
        self.events: list[LocationEvent] = []
        self.seq = 0

    def append(
        self,
        camera: str,
        cat: str,
        zone: str | None,
        quality: str = "",
        confidence: float = 0.0,
        norm_x: float = 0.0,
        norm_y: float = 0.0,
        text: str = "",
    ) -> LocationEvent:
        with self.lock:
            self.seq += 1
            event = LocationEvent(
                seq=self.seq,
                camera=camera,
                cat=cat,
                zone=zone,
                quality=quality,
                confidence=confidence,
                norm_x=norm_x,
                norm_y=norm_y,
                text=text,
                ts=time.time(),
            )
            self.events.append(event)
            if len(self.events) > self.limit:
                del self.events[: len(self.events) - self.limit]
            return event

    def since(self, seq: int, camera: str | None = None, limit: int = 50) -> list[LocationEvent]:
        with self.lock:
            found = [
                event
                for event in self.events
                if event.seq > seq and (camera is None or event.camera == camera)
            ]
            return found[-limit:]

    def latest_seq(self) -> int:
        with self.lock:
            return self.seq

    def clear(self) -> None:
        with self.lock:
            self.events.clear()


@dataclass
class PreviewContext:
    """Everything the HTTP handlers need."""

    hub: FrameHub
    settings: dict
    log: MessageLog = dataclass_field(default_factory=MessageLog)
    lock: threading.Lock = dataclass_field(default_factory=threading.Lock)


def _zone_points(raw_points) -> list[tuple[float, float]]:
    points = []
    for point in raw_points or []:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        points.append((float(point[0]), float(point[1])))
    return points


def zones_path_for(context: PreviewContext) -> Path:
    configured = context.settings.get("zones_path")
    return Path(configured) if configured else ZONES_PATH


def zones_for(context: PreviewContext, camera: str) -> list[dict]:
    """Zones are read from disk so external re-anchoring shows up immediately."""
    calibration = load_calibrations(zones_path_for(context)).get(camera)
    if calibration is None:
        return []
    return [
        {
            "name": zone.name,
            "points": [[round(x, 5), round(y, 5)] for x, y in zone.points],
            "area": round(zone.area, 5),
        }
        for zone in calibration.zones
    ]


def locations_payload(context: PreviewContext, since: int, camera: str | None) -> dict:
    events = context.log.since(since, camera)
    return {
        "seq": context.log.latest_seq(),
        "events": [
            {
                "seq": event.seq,
                "camera": event.camera,
                "cat": event.cat,
                "zone": event.zone,
                "quality": event.quality,
                "confidence": round(event.confidence, 3),
                "norm_x": round(event.norm_x, 4),
                "norm_y": round(event.norm_y, 4),
                "text": event.text,
                "ts": round(event.ts, 3),
            }
            for event in events
        ],
    }


def status_payload(context: PreviewContext) -> dict:
    cameras = []
    for name in context.hub.names():
        feed = context.hub.get(name)
        if feed is None:
            continue
        age = feed.age_seconds
        cameras.append(
            {
                "name": name,
                "status": feed.status,
                "age_seconds": None if age is None else round(age, 2),
                "width": feed.width,
                "height": feed.height,
                "frames": feed.frame_count,
                "detections": feed.detections,
                "best_confidence": round(feed.best_confidence, 3),
                "has_zones": bool(zones_for(context, name)),
            }
        )
    return {"cameras": cameras}


def save_zones(context: PreviewContext, camera: str, body: dict) -> dict:
    """Replace a camera's zones using the frame the operator just drew on."""
    zones = []
    for entry in body.get("zones") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        points = _zone_points(entry.get("points"))
        if not name or len(points) < 3:
            continue
        zones.append(Zone(name=name, points=points))
    if not zones:
        return {"ok": False, "message": "没有有效的区域：至少需要 3 个点", "zones": []}

    feed = context.hub.get(camera)
    frame = feed.clean_frame if feed is not None else None
    if frame is None:
        return {
            "ok": False,
            "message": "还没有采集到画面，等预览出来再保存",
            "zones": zones_for(context, camera),
        }

    with context.lock:
        # No reference frame on the draft, so reanchor adopts the current frame
        # as the new reference and keeps the polygons exactly as drawn.
        draft = Calibration(camera=camera, zones=zones)
        outcome = reanchor(draft, frame, context.settings)
    return {
        "ok": outcome.ok,
        "message": outcome.message,
        "calibration_id": outcome.calibration.calibration_id if outcome.calibration else "",
        "off_frame_zones": list(outcome.off_frame_zones),
        "zones": zones_for(context, camera),
    }


def reanchor_camera(context: PreviewContext, camera: str) -> dict:
    """Project the stored zones onto a fresh frame without redrawing them."""
    calibration = load_calibrations(zones_path_for(context)).get(camera)
    if calibration is None or not calibration.zones:
        return {"ok": False, "message": "该摄像头还没有区域，请先标注", "zones": []}
    feed = context.hub.get(camera)
    frame = feed.clean_frame if feed is not None else None
    if frame is None:
        return {"ok": False, "message": "还没有采集到画面", "zones": zones_for(context, camera)}
    with context.lock:
        outcome = reanchor(calibration, frame, context.settings)
    return {
        "ok": outcome.ok,
        "message": outcome.message,
        "calibration_id": outcome.calibration.calibration_id if outcome.calibration else "",
        "off_frame_zones": list(outcome.off_frame_zones),
        "zones": zones_for(context, camera),
    }


INDEX_HTML = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>猫的位置 · 区域标注</title>
<style>
  :root {
    --bg:#0e1015; --panel:#161a22; --line:#252b36; --text:#e7eaf0;
    --muted:#8b93a3; --accent:#4ade80; --warn:#fbbf24; --bad:#f87171;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.55 system-ui,"Segoe UI","Noto Sans CJK SC",sans-serif; }
  header { display:flex; align-items:center; gap:18px; padding:11px 18px;
           background:var(--panel); border-bottom:1px solid var(--line);
           position:sticky; top:0; z-index:5; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:650; letter-spacing:.2px; }
  .tabs { display:flex; gap:6px; }
  button { background:#212838; color:var(--text); border:1px solid var(--line);
           border-radius:9px; padding:7px 13px; cursor:pointer; font:inherit;
           transition:background .12s ease; }
  button:hover { background:#2c3446; }
  button.active { background:var(--accent); color:#06210f; border-color:var(--accent);
                  font-weight:650; }
  button.primary { background:var(--accent); color:#06210f; border-color:var(--accent);
                   font-weight:650; }
  button.ghost { background:transparent; }
  main { display:grid; grid-template-columns:minmax(0,1fr) 310px; gap:16px; padding:16px; }
  @media (max-width:900px) { main { grid-template-columns:1fr; } }
  .stage { position:relative; background:#05070a; border:1px solid var(--line);
           border-radius:12px; overflow:hidden; }
  .stage img { display:block; width:100%; }
  .stage canvas { position:absolute; inset:0; cursor:crosshair; }
  .placeholder { padding:64px 24px; text-align:center; color:var(--muted); }
  aside { background:var(--panel); border:1px solid var(--line); border-radius:12px;
          padding:14px; }
  aside h2 { font-size:12px; text-transform:uppercase; letter-spacing:.09em;
             color:var(--muted); margin:16px 0 8px; }
  aside h2:first-child { margin-top:0; }
  .toolbar { display:flex; flex-wrap:wrap; gap:7px; }
  .toolbar button { flex:1 1 auto; }
  #zone-name { width:100%; padding:8px 11px; margin-bottom:8px; border-radius:9px;
               background:#0f1218; border:1px solid var(--line); color:var(--text);
               font:inherit; }
  #zone-name:focus { outline:none; border-color:var(--accent); }
  #zone-name::placeholder { color:#5d6577; }
  .toggle { display:flex; align-items:center; gap:7px; width:100%;
            color:var(--muted); font-size:13px; margin-top:4px; }
  ul { list-style:none; margin:0; padding:0; }
  li { display:flex; align-items:center; gap:8px; padding:6px 8px; border-radius:8px;
       background:#1d2230; margin-bottom:5px; }
  li .swatch { width:11px; height:11px; border-radius:3px; flex:0 0 auto; }
  li .name { flex:1 1 auto; overflow:hidden; text-overflow:ellipsis;
             white-space:nowrap; }
  li .area { color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }
  li button { padding:2px 8px; border-radius:6px; font-size:12px; }
  .hint { color:var(--muted); font-size:12.5px; margin:14px 0 0;
          border-top:1px solid var(--line); padding-top:12px; }
  .hint b { color:var(--text); }
  .now { background:#101722; border:1px solid var(--line); border-radius:10px;
         padding:10px 12px; font-size:13.5px; line-height:1.75; }
  .now .cat { color:var(--accent); font-weight:650; }
  .now .zone { color:#fff; font-weight:650; }
  .now .dim { color:var(--muted); font-size:12px; }
  .log { list-style:none; margin:8px 0 0; padding:0; max-height:180px; overflow-y:auto;
         font-size:12.5px; font-variant-numeric:tabular-nums; }
  .log li { display:block; padding:3px 0; border-bottom:1px solid #1b202b;
            color:var(--muted); }
  .log li .t { color:#5d6577; margin-right:7px; }
  .log li .z { color:var(--text); font-weight:600; }
  .status { margin-left:auto; color:var(--muted); font-size:12.5px;
            font-variant-numeric:tabular-nums; }
  .msg { margin-top:10px; padding:8px 10px; border-radius:8px; font-size:12.5px;
         background:#1d2230; color:var(--muted); word-break:break-word; }
  .msg.ok { color:#86efac; } .msg.bad { color:#fca5a5; }
</style>
</head>
<body>
<header>
  <h1>🐾 猫的位置 · 区域标注</h1>
  <div class="tabs" id="tabs"></div>
  <span class="status" id="status">正在连接…</span>
</header>
<main>
  <section class="stage" id="stage">
    <img id="stream" alt="camera stream">
    <canvas id="overlay"></canvas>
    <div class="placeholder" id="placeholder">等待摄像头画面…</div>
  </section>
  <aside>
    <h2>📍 实时位置</h2>
    <div class="now" id="now">等待检测…</div>
    <ul class="log" id="location-log"></ul>

    <h2>区域编辑</h2>
    <input id="zone-name" list="zone-names" placeholder="区域名称，例如 floor" autocomplete="off">
    <datalist id="zone-names">
      <option value="floor"></option>
      <option value="table_top"></option>
      <option value="under_table"></option>
      <option value="sofa_cushion"></option>
      <option value="under_sofa"></option>
      <option value="counter_top"></option>
      <option value="carpet"></option>
    </datalist>
    <div class="toolbar">
      <button id="close-poly" class="primary">闭合多边形</button>
      <button id="undo">撤销点</button>
      <button id="clear-draft">清除绘制</button>
      <button id="save" class="primary">保存区域</button>
      <button id="reanchor">重锚定</button>
      <button id="reload" class="ghost">重新载入</button>
    </div>
    <label class="toggle">
      <input type="checkbox" id="detections" checked> 显示检测框和锚点
    </label>
    <label class="toggle">
      <input type="checkbox" id="show-zones" checked> 显示区域描点
    </label>
    <div class="msg" id="msg">左键点击开始绘制区域。</div>
    <h2>已保存区域</h2>
    <ul id="zone-list"></ul>
    <p class="hint">
      画的是<b>猫站立的表面</b>（桌面、桌下地面、沙发坐垫、地板），<br>
      不是家具轮廓——判定用的锚点是检测框的<b>底边中点</b>。<br>
      留一个覆盖整片可见地面的 <b>floor</b> 兜底。<br>
      同一个名字可以重复使用，等于给同一位置追加第二块多边形。
    </p>
  </aside>
</main>
<script>
const PALETTE = ["#4ade80","#60a5fa","#f472b6","#fbbf24","#a78bfa","#22d3ee","#fb7185","#34d399"];
const state = { cameras: [], camera: null, zones: {}, draft: [] };

const stream = document.getElementById("stream");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const el = (id) => document.getElementById(id);

function colorFor(index) { return PALETTE[index % PALETTE.length]; }
function zoneEntries() { return state.zones[state.camera] || []; }

function centroid(points) {
  let x = 0, y = 0;
  for (const [px, py] of points) { x += px; y += py; }
  return [x / points.length, y / points.length];
}

function draw() {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!w || !h) return;
  ctx.lineJoin = "round";

  const showSaved = el("show-zones").checked;
  (showSaved ? zoneEntries() : []).forEach((zone, index) => {
    const color = colorFor(index);
    ctx.beginPath();
    zone.points.forEach(([x, y], i) => {
      const px = x * w, py = y * h;
      if (i === 0) { ctx.moveTo(px, py); } else { ctx.lineTo(px, py); }
    });
    ctx.closePath();
    ctx.fillStyle = color + "26";
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();

    const [cx, cy] = centroid(zone.points);
    ctx.fillStyle = color;
    ctx.font = "600 13px system-ui, sans-serif";
    ctx.fillText(zone.name, cx * w - 18, cy * h + 4);
  });

  if (state.draft.length) {
    ctx.beginPath();
    state.draft.forEach(([x, y], i) => {
      const px = x * w, py = y * h;
      if (i === 0) { ctx.moveTo(px, py); } else { ctx.lineTo(px, py); }
    });
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();
    for (const [x, y] of state.draft) {
      ctx.beginPath();
      ctx.arc(x * w, y * h, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
    }
  }
}

function syncCanvas() {
  const rect = stream.getBoundingClientRect();
  if (!rect.width) return;
  canvas.width = Math.round(rect.width);
  canvas.height = Math.round(rect.height);
  draw();
}
new ResizeObserver(syncCanvas).observe(stream);
window.addEventListener("resize", syncCanvas);

canvas.addEventListener("click", (event) => {
  const rect = canvas.getBoundingClientRect();
  state.draft.push([
    (event.clientX - rect.left) / rect.width,
    (event.clientY - rect.top) / rect.height,
  ]);
  draw();
});

function message(text, kind) {
  const box = el("msg");
  box.textContent = text;
  box.className = "msg" + (kind ? " " + kind : "");
}

function streamUrl(name) {
  const suffix = el("detections").checked ? ".mjpg" : "/clean.mjpg";
  return "/stream/" + encodeURIComponent(name) + suffix;
}

// Rebuilding the tabs on every status poll is not just wasteful: it replaces
// the buttons under the pointer, so a click can be swallowed mid-render.
let tabsSignature = "";

function renderTabs() {
  const signature =
    state.cameras.map((camera) => camera.name + (camera.has_zones ? "*" : "")).join(",") +
    "|" +
    state.camera;
  if (signature === tabsSignature && el("tabs").children.length === state.cameras.length) {
    return;
  }
  tabsSignature = signature;

  const tabs = el("tabs");
  tabs.textContent = "";
  for (const camera of state.cameras) {
    const button = document.createElement("button");
    button.textContent = camera.name + (camera.has_zones ? " ●" : "");
    button.title = camera.status;
    if (camera.name === state.camera) button.className = "active";
    button.onclick = () => selectCamera(camera.name);
    tabs.appendChild(button);
  }
}

function renderZoneList() {
  const list = el("zone-list");
  list.textContent = "";
  const entries = zoneEntries();
  if (!entries.length) {
    const empty = document.createElement("li");
    empty.textContent = "（还没有区域）";
    empty.style.color = "var(--muted)";
    list.appendChild(empty);
    return;
  }
  entries.forEach((zone, index) => {
    const item = document.createElement("li");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = colorFor(index);
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = zone.name;
    const area = document.createElement("span");
    area.className = "area";
    const count = entries.filter((other) => other.name === zone.name).length;
    area.textContent = count > 1 ? "×" + count : "";
    const remove = document.createElement("button");
    remove.textContent = "✕";
    remove.onclick = () => {
      zoneEntries().splice(index, 1);
      renderZoneList();
      draw();
    };
    item.append(swatch, name, area, remove);
    list.appendChild(item);
  });
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    state.cameras = data.cameras || [];
  } catch (error) {
    el("status").textContent = "服务不可用";
    return;
  }
  if (!state.camera && state.cameras.length) {
    state.camera = state.cameras[0].name;
    stream.src = streamUrl(state.camera);
    await loadZones();
  }
  const current = state.cameras.find((camera) => camera.name === state.camera);
  el("status").textContent = current
    ? current.status + " · " + current.width + "×" + current.height +
      (current.age_seconds !== null ? " · " + current.age_seconds.toFixed(1) + "s" : "") +
      " · " + (current.has_zones ? "已标注" : "未标注")
    : "没有摄像头";
  el("placeholder").style.display = current && current.frames ? "none" : "block";
  renderTabs();
  renderCurrentLocation();
}

async function loadZones() {
  if (!state.camera) return;
  const response = await fetch("/api/zones/" + encodeURIComponent(state.camera));
  const data = await response.json();
  state.zones[state.camera] = data.zones || [];
  renderZoneList();
  draw();
}

async function selectCamera(name) {
  state.camera = name;
  state.draft = [];
  stream.src = streamUrl(name);
  await loadZones();
  renderTabs();
  renderCurrentLocation();
  syncCanvas();
  message("已切换到 " + name);
}

async function post(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return response.json();
}

el("close-poly").onclick = () => {
  if (state.draft.length < 3) { message("一个区域至少需要 3 个点", "bad"); return; }
  const name = el("zone-name").value.trim();
  if (!name) {
    message("先在上面填写区域名称，例如 floor", "bad");
    el("zone-name").focus();
    return;
  }
  if (!state.zones[state.camera]) state.zones[state.camera] = [];
  state.zones[state.camera].push({ name: name, points: state.draft });
  state.draft = [];
  renderZoneList();
  draw();
  message("已添加区域「" + name + "」，记得点保存。");
};

el("undo").onclick = () => { state.draft.pop(); draw(); };
el("clear-draft").onclick = () => { state.draft = []; draw(); };

el("save").onclick = async () => {
  if (!state.camera) return;
  message("保存中…");
  try {
    const data = await post("/api/zones/" + encodeURIComponent(state.camera), {
      zones: zoneEntries(),
    });
    state.zones[state.camera] = data.zones || [];
    renderZoneList();
    draw();
    await loadStatus();
    message((data.ok ? "✅ " : "⚠️ ") + data.message, data.ok ? "ok" : "bad");
  } catch (error) {
    message("保存失败：" + error, "bad");
  }
};

el("reanchor").onclick = async () => {
  if (!state.camera) return;
  message("重锚定中…");
  try {
    const data = await post("/api/reanchor/" + encodeURIComponent(state.camera), {});
    state.zones[state.camera] = data.zones || [];
    renderZoneList();
    draw();
    message((data.ok ? "✅ " : "⚠️ ") + data.message, data.ok ? "ok" : "bad");
  } catch (error) {
    message("重锚定失败：" + error, "bad");
  }
};

el("reload").onclick = async () => { await loadZones(); message("已从 zones.yaml 重新载入"); };

el("detections").onchange = () => {
  if (state.camera) stream.src = streamUrl(state.camera);
};

el("show-zones").onchange = draw;

stream.addEventListener("load", syncCanvas);

// --- live location panel ---------------------------------------------------
let locationSeq = 0;
const latestByKey = {};
const MAX_LOG_ROWS = 40;

async function pollLocations() {
  let data;
  try {
    const response = await fetch("/api/locations?since=" + locationSeq);
    data = await response.json();
  } catch (error) {
    return; // the server may be restarting; keep polling
  }
  if (typeof data.seq === "number") locationSeq = data.seq;
  for (const event of data.events || []) {
    // Keyed by camera too: two cameras can legitimately report the same cat.
    latestByKey[event.camera + "|" + event.cat] = event;
    appendLocation(event);
  }
  renderCurrentLocation();
}

// Only the tab the operator is looking at, so the readout matches the picture.
function visibleEvents() {
  return Object.values(latestByKey)
    .filter((event) => event.camera === state.camera)
    .sort((a, b) => (a.cat < b.cat ? -1 : 1));
}

function appendLocation(event) {
  const list = el("location-log");
  const item = document.createElement("li");
  const time = document.createElement("span");
  time.className = "t";
  time.textContent = new Date(event.ts * 1000).toLocaleTimeString();
  const place = document.createElement("span");
  place.className = "z";
  place.textContent = event.cat + " → " + (event.zone || "未知位置");
  const meta = document.createElement("span");
  meta.textContent = "  " + event.camera + " · " + event.quality;
  item.append(time, place, meta);
  list.prepend(item);
  while (list.children.length > MAX_LOG_ROWS) list.lastElementChild.remove();
}

function renderCurrentLocation() {
  const box = el("now");
  const events = visibleEvents();
  box.textContent = "";
  if (!events.length) {
    // Nothing reported for this camera, so say why instead of a bare placeholder.
    for (const line of idleReason()) {
      const row = document.createElement("div");
      row.textContent = line;
      box.appendChild(row);
    }
    return;
  }
  for (const event of events) {
    const row = document.createElement("div");
    const name = document.createElement("span");
    name.className = "cat";
    name.textContent = event.cat;
    const zone = document.createElement("span");
    zone.className = "zone";
    zone.textContent = event.zone || "未知位置";
    const detail = document.createElement("span");
    detail.className = "dim";
    detail.textContent = "  " + event.quality + " · conf " + event.confidence.toFixed(2);
    row.append(name, document.createTextNode(" 在 "), zone, detail);
    box.appendChild(row);
  }
}

// Returns one string per line; never embed a newline in a JS string here.
function idleReason() {
  const camera = state.cameras.find((entry) => entry.name === state.camera) || state.cameras[0];
  if (!camera) return ["没有摄像头"];
  if (!camera.frames) return ["等待画面…"];
  if (camera.detections > 0) return ["检测到猫，正在判定位置…"];
  if (camera.best_confidence > 0) {
    return [
      "画面正常，但置信度没过阈值",
      "最高 " + camera.best_confidence.toFixed(2) + "，试试点降低 --conf",
    ];
  }
  return ["画面正常，但没有检测到猫", "已处理 " + camera.frames + " 帧画面"];
}

pollLocations();
setInterval(pollLocations, 1000);

loadStatus();
setInterval(loadStatus, 3000);
setInterval(() => { if (!state.camera) loadStatus(); }, 6000);
</script>
</body>
</html>
"""


def _make_handler(context: PreviewContext) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "CatPreview/1.0"

        def log_message(self, fmt, *args) -> None:  # keep the console for the detector
            return

        def _send_bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, payload: dict, status: int = 200) -> None:
            self._send_bytes(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def _send_html(self, markup: str) -> None:
            self._send_bytes(markup.encode("utf-8"), "text/html; charset=utf-8")

        def _read_json(self) -> dict | None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return None
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
            return payload if isinstance(payload, dict) else None

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._send_html(INDEX_HTML)
            if path == "/api/status":
                return self._send_json(status_payload(context))

            if path == "/api/locations":
                query = parse_qs(urlparse(self.path).query)
                try:
                    since = int((query.get("since") or ["0"])[0] or 0)
                except ValueError:
                    since = 0
                camera = (query.get("camera") or [None])[0]
                return self._send_json(locations_payload(context, since, camera))

            match = STREAM_ROUTE.match(path)
            if match:
                return self._stream(unquote(match.group(1)), clean=bool(match.group(2)))

            match = SNAPSHOT_ROUTE.match(path)
            if match:
                feed = context.hub.get(unquote(match.group(1)))
                if feed is None or feed.clean_jpeg is None:
                    return self.send_error(404, "no frame yet")
                return self._send_bytes(feed.clean_jpeg, "image/jpeg")

            match = ZONES_ROUTE.match(path)
            if match:
                camera = unquote(match.group(1))
                return self._send_json({"camera": camera, "zones": zones_for(context, camera)})

            return self.send_error(404, "not found")

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            body = self._read_json()
            if body is None:
                return self._send_json({"ok": False, "message": "请求体不是合法 JSON"}, 400)

            match = ZONES_ROUTE.match(path)
            if match:
                result = save_zones(context, unquote(match.group(1)), body)
                return self._send_json(result, 200 if result["ok"] else 400)

            match = REANCHOR_ROUTE.match(path)
            if match:
                result = reanchor_camera(context, unquote(match.group(1)))
                return self._send_json(result, 200 if result["ok"] else 400)

            return self.send_error(404, "not found")

        def _stream(self, camera: str, clean: bool) -> None:
            if context.hub.get(camera) is None:
                return self.send_error(404, "unknown camera")
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY}"
            )
            self.end_headers()
            last_seen = -1
            try:
                while True:
                    feed = context.hub.get(camera)
                    if feed is None:
                        break
                    if feed.frame_count != last_seen:
                        last_seen = feed.frame_count
                        payload = feed.clean_jpeg if clean else feed.annotated_jpeg
                        if payload:
                            self.wfile.write(f"--{STREAM_BOUNDARY}\r\n".encode("ascii"))
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(payload)))
                            self.end_headers()
                            self.wfile.write(payload)
                            self.wfile.write(b"\r\n")
                    time.sleep(POLL_INTERVAL_SECONDS)
            except (BrokenPipeError, ConnectionResetError, OSError):
                # The browser closed the tab or navigated away.
                pass

    return Handler


def start_server(
    context: PreviewContext, host: str = "127.0.0.1", port: int = 8765
) -> ThreadingHTTPServer:
    """Serve the editor on a daemon thread and return the running server."""
    server = ThreadingHTTPServer((host, port), _make_handler(context))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="preview-http", daemon=True).start()
    return server
