"""Tests for the browser preview and zone editor endpoints."""

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.location_zones import load_calibrations
from src.monitoring.web_preview import (
    INDEX_HTML,
    FrameHub,
    MessageLog,
    PreviewContext,
    locations_payload,
    save_zones,
    start_server,
    status_payload,
    zones_for,
)

GOOD_ZONES = [
    {
        "name": "floor",
        "points": [[0.05, 0.55], [0.45, 0.55], [0.45, 0.95], [0.05, 0.95]],
    },
    {
        "name": "table_top",
        "points": [[0.50, 0.40], [0.70, 0.40], [0.70, 0.55], [0.50, 0.55]],
    },
]


def make_frame(width: int = 640, height: int = 480, seed: int = 0) -> np.ndarray:
    """Deterministic, feature-rich scene so ORB alignment can match on it."""
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width), 255, np.uint8)
    for _ in range(160):
        x = int(rng.integers(0, max(1, width - 60)))
        y = int(rng.integers(0, max(1, height - 60)))
        w = int(rng.integers(10, 60))
        h = int(rng.integers(10, 60))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), int(rng.integers(0, 200)), -1)
    return cv2.cvtColor(cv2.GaussianBlur(canvas, (3, 3), 0), cv2.COLOR_GRAY2BGR)


def shift_frame(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        frame, matrix, (frame.shape[1], frame.shape[0]), borderValue=(255, 255, 255)
    )


def read_some(response, minimum: int = 256, deadline: float = 5.0) -> bytes:
    """An MJPEG response never ends, so read only what has already arrived."""
    collected = bytearray()
    end = time.monotonic() + deadline
    while len(collected) < minimum and time.monotonic() < end:
        piece = response.read1(4096)
        if piece:
            collected.extend(piece)
    return bytes(collected)


@pytest.fixture
def context(tmp_path: Path) -> PreviewContext:
    """Context wired to temporary paths so the real config is never touched."""
    hub = FrameHub(display_width=320, quality=70)
    hub.register("sofa")
    hub.publish("sofa", make_frame(), make_frame(), detections=1)
    return PreviewContext(
        hub=hub,
        settings={
            "work_width": 640,
            "min_inliers": 15,
            "good_inlier_ratio": 0.5,
            "max_residual": 0.01,
            "calibration_dir": str(tmp_path / "calibrations"),
            "zones_path": str(tmp_path / "zones.yaml"),
            "discord_webhook": "",
            "discord_username": "test",
            "notify": False,
        },
    )


@pytest.fixture
def server(context: PreviewContext):
    """A running server, exposed together with the context that drives it."""
    httpd = start_server(context, host="127.0.0.1", port=0)
    host, port = httpd.server_address[:2]
    yield f"http://{host}:{port}", context
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def base_url(server) -> str:
    return server[0]


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def test_index_page_is_served(base_url: str) -> None:
    with urllib.request.urlopen(base_url + "/", timeout=10) as response:
        body = response.read().decode("utf-8")
    assert response.status == 200
    assert "text/html" in response.headers["Content-Type"]
    assert "canvas" in body
    assert "/api/status" in body


def test_status_reports_published_frames(base_url: str) -> None:
    payload = get_json(base_url + "/api/status")
    assert [camera["name"] for camera in payload["cameras"]] == ["sofa"]
    camera = payload["cameras"][0]
    assert camera["status"] == "live"
    assert camera["width"] == 640
    assert camera["height"] == 480
    assert camera["frames"] == 1
    assert camera["detections"] == 1
    assert camera["has_zones"] is False


def test_status_reports_camera_without_frames(context: PreviewContext) -> None:
    context.hub.register("feeder")
    payload = status_payload(context)
    feeder = next(camera for camera in payload["cameras"] if camera["name"] == "feeder")
    assert feeder["status"] == "starting"
    assert feeder["frames"] == 0
    assert feeder["age_seconds"] is None


def test_stream_sends_multipart_jpeg(base_url: str) -> None:
    with urllib.request.urlopen(base_url + "/stream/sofa.mjpg", timeout=10) as response:
        assert "multipart/x-mixed-replace" in response.headers["Content-Type"]
        chunk = read_some(response)
    assert b"--frame" in chunk
    # Every MJPEG part carries a JPEG payload, so a SOI marker must be present.
    assert b"\xff\xd8\xff" in chunk
    assert b"Content-Type: image/jpeg" in chunk


def test_clean_stream_is_also_available(base_url: str) -> None:
    with urllib.request.urlopen(base_url + "/stream/sofa/clean.mjpg", timeout=10) as response:
        chunk = read_some(response)
    assert b"--frame" in chunk


def test_snapshot_endpoint_returns_jpeg(base_url: str) -> None:
    with urllib.request.urlopen(base_url + "/api/snapshot/sofa.jpg", timeout=10) as response:
        assert response.headers["Content-Type"] == "image/jpeg"
        assert response.read(3) == b"\xff\xd8\xff"


def test_unknown_camera_stream_returns_404(base_url: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(base_url + "/stream/ghost.mjpg", timeout=10)
    assert error.value.code == 404


def test_zones_start_empty(base_url: str) -> None:
    payload = get_json(base_url + "/api/zones/sofa")
    assert payload == {"camera": "sofa", "zones": []}


def test_save_zones_writes_calibration(context: PreviewContext, base_url: str) -> None:
    status, payload = post_json(base_url + "/api/zones/sofa", {"zones": GOOD_ZONES})
    assert status == 200, payload
    assert payload["ok"] is True
    assert payload["calibration_id"].startswith("sofa-")
    assert [zone["name"] for zone in payload["zones"]] == ["floor", "table_top"]

    stored = load_calibrations(Path(context.settings["zones_path"]))["sofa"]
    assert [zone.name for zone in stored.zones] == ["floor", "table_top"]
    # reanchor must have adopted the live frame as the new reference.
    assert stored.reference_path is not None
    assert stored.reference_path.is_file()
    assert stored.calibration_id == payload["calibration_id"]

    # Duplicate names are legitimate: two polygons for one location.
    reloaded = get_json(base_url + "/api/zones/sofa")
    assert len(reloaded["zones"]) == 2


def test_save_zones_accepts_repeated_names(context: PreviewContext, base_url: str) -> None:
    split_floor = [
        {"name": "floor", "points": [[0.02, 0.60], [0.30, 0.60], [0.30, 0.98], [0.02, 0.98]]},
        {"name": "floor", "points": [[0.70, 0.60], [0.98, 0.60], [0.98, 0.98], [0.70, 0.98]]},
    ]
    status, payload = post_json(base_url + "/api/zones/sofa", {"zones": split_floor})
    assert status == 200, payload
    assert [zone["name"] for zone in payload["zones"]] == ["floor", "floor"]


def test_save_zones_rejects_degenerate_polygons(base_url: str) -> None:
    status, payload = post_json(
        base_url + "/api/zones/sofa", {"zones": [{"name": "floor", "points": [[0.1, 0.1]]}]}
    )
    assert status == 400
    assert payload["ok"] is False
    assert payload["zones"] == []


def test_save_zones_rejects_garbage_body(base_url: str) -> None:
    request = urllib.request.Request(
        base_url + "/api/zones/sofa",
        data=b"not json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=10)
    assert error.value.code == 400


def test_save_zones_without_a_frame_is_rejected(context: PreviewContext) -> None:
    context.hub.register("feeder")
    saved = save_zones(context, "feeder", {"zones": GOOD_ZONES})
    assert saved["ok"] is False
    assert "画面" in saved["message"]


def test_reanchor_camera_requires_existing_zones(base_url: str) -> None:
    status, payload = post_json(base_url + "/api/reanchor/sofa", {})
    assert status == 400
    assert payload["ok"] is False


def test_reanchor_camera_projects_stored_zones(context: PreviewContext, base_url: str) -> None:
    post_json(base_url + "/api/zones/sofa", {"zones": GOOD_ZONES})
    first = load_calibrations(Path(context.settings["zones_path"]))["sofa"]

    # Drift the camera by 20px right and 10px down before the next re-anchor.
    drifted = shift_frame(make_frame(), 20, 10)
    context.hub.publish("sofa", drifted, drifted, detections=0)

    status, payload = post_json(base_url + "/api/reanchor/sofa", {})
    assert status == 200, payload
    assert payload["ok"] is True
    second = load_calibrations(Path(context.settings["zones_path"]))["sofa"]
    assert second.calibration_id != first.calibration_id
    assert second.reference_path is not None and second.reference_path.is_file()
    assert len(second.zones) == 2

    # Zones live in reference coordinates, so storing the drifted frame as the
    # new reference shifts every polygon by exactly the drift in normalised units.
    assert [zone.name for zone in second.zones] == [zone.name for zone in first.zones]
    for before, after in zip(first.zones, second.zones, strict=True):
        for (bx, by), (ax, ay) in zip(before.points, after.points, strict=True):
            assert ax == pytest.approx(bx + 20 / 640, abs=0.01)
            assert ay == pytest.approx(by + 10 / 480, abs=0.01)


def test_zones_for_reads_the_configured_path(tmp_path: Path) -> None:
    hub = FrameHub()
    hub.register("sofa")
    missing = PreviewContext(
        hub=hub,
        settings={"zones_path": str(tmp_path / "absent.yaml")},
    )
    assert zones_for(missing, "sofa") == []


def test_embedded_javascript_has_no_unescaped_newlines() -> None:
    """A raw newline inside a JS string silently kills the whole page.

    ``INDEX_HTML`` is a normal Python string, so a ``\\n`` written in the JS
    becomes a real newline once the page is served: the string literal ends
    early, the browser drops the entire script, and every panel freezes on its
    static placeholder text. Quote balance per line catches it.
    """
    script = INDEX_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    for number, line in enumerate(script.splitlines(), 1):
        code = line.split("//", 1)[0]
        assert code.count('"') % 2 == 0, f"unbalanced double quote on line {number}: {line!r}"
        assert code.count("'") % 2 == 0, f"unbalanced single quote on line {number}: {line!r}"


# Browser globals and JS keywords that legitimately appear as a bare call.
JS_ALLOWED_CALLS = {
    "fetch",
    "setInterval",
    "setTimeout",
    "clearInterval",
    "encodeURIComponent",
    "decodeURIComponent",
    "encodeURI",
    "decodeURI",
    "requestAnimationFrame",
    "Date",
    "Number",
    "String",
    "Boolean",
    "Array",
    "Error",
    "RegExp",
    "ResizeObserver",
    "Event",
    "URL",
    "isNaN",
    "isFinite",
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "typeof",
    "new",
    "function",
    "do",
    "else",
    "in",
    "of",
    "await",
    "async",
    "delete",
    "void",
    "throw",
    "case",
}


def test_embedded_javascript_defines_every_function_it_calls() -> None:
    """A renamed or dropped helper only surfaces as a page error in the browser.

    Editing a large embedded script through string replacements can silently
    leave a call behind, which freezes the panel with a ReferenceError that no
    Python test would otherwise notice.
    """
    script = INDEX_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    code = "\n".join(line.split("//", 1)[0] for line in script.splitlines())
    # Blank out string literals so CSS like "var(--muted)" is not read as a call.
    code = re.sub(r'"[^"\n]*"', '""', code)
    code = re.sub(r"'[^'\n]*'", "''", code)

    declared = set(re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)", code))
    declared |= set(re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)", code))

    # Drop declarations so `function foo(` is not mistaken for a call.
    calls_only = re.sub(r"\bfunction\s+[A-Za-z_$][\w$]*\s*\(", " ", code)
    called = set(re.findall(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*\(", calls_only))

    undefined = called - declared - JS_ALLOWED_CALLS
    assert undefined == set(), f"called but never defined: {sorted(undefined)}"


def test_index_page_has_a_live_location_panel(base_url: str) -> None:
    with urllib.request.urlopen(base_url + "/", timeout=10) as response:
        body = response.read().decode("utf-8")
    assert 'id="now"' in body
    assert 'id="location-log"' in body
    assert "/api/locations" in body


def test_locations_are_empty_until_something_is_reported(base_url: str) -> None:
    payload = get_json(base_url + "/api/locations?since=0")
    assert payload == {"seq": 0, "events": []}


def test_reported_location_is_returned_with_its_details(server) -> None:
    base_url, context = server
    context.log.append(
        camera="sofa",
        cat="bagel",
        zone="on_sofa",
        quality="good",
        confidence=0.87,
        norm_x=0.512,
        norm_y=0.744,
        text="[sofa] bagel -> on_sofa",
    )

    body = get_json(base_url + "/api/locations?since=0")
    assert body["seq"] == 1
    assert len(body["events"]) == 1
    event = body["events"][0]
    assert event["cat"] == "bagel"
    assert event["zone"] == "on_sofa"
    assert event["quality"] == "good"
    assert event["confidence"] == pytest.approx(0.87)
    assert event["norm_x"] == pytest.approx(0.512)
    assert event["text"] == "[sofa] bagel -> on_sofa"
    assert event["ts"] > 0

    # A later poll asking only for newer events must not repeat the first one.
    assert get_json(f"{base_url}/api/locations?since={body['seq']}")["events"] == []


def test_locations_can_be_filtered_by_camera(server) -> None:
    base_url, context = server
    context.log.append(camera="sofa", cat="bagel", zone="on_sofa")
    context.log.append(camera="feeder", cat="kurumi", zone="feeder")

    everything = get_json(base_url + "/api/locations?since=0")
    assert len(everything["events"]) == 2

    only_sofa = get_json(base_url + "/api/locations?since=0&camera=sofa")
    assert [event["cat"] for event in only_sofa["events"]] == ["bagel"]
    # seq still tracks the whole log so a filtered poll never re-reads old events.
    assert only_sofa["seq"] == everything["seq"]


def test_locations_tolerates_a_bad_since_parameter(server) -> None:
    base_url, context = server
    context.log.append(camera="sofa", cat="bagel", zone="on_sofa")
    payload = get_json(base_url + "/api/locations?since=not-a-number")
    assert len(payload["events"]) == 1


def test_message_log_sequence_only_advances(server) -> None:
    _, context = server
    seqs = [
        context.log.append(camera="sofa", cat="bagel", zone=zone).seq
        for zone in ("floor", "on_sofa", "under_table")
    ]
    assert seqs == [1, 2, 3]

    # Asking for everything after the first two leaves only the newest event.
    body = locations_payload(context, 2, None)
    assert body["seq"] == 3
    assert [event["zone"] for event in body["events"]] == ["under_table"]


def test_message_log_is_bounded() -> None:
    log = MessageLog(limit=5)
    for index in range(20):
        log.append(camera="sofa", cat="bagel", zone=f"zone{index}")

    assert log.latest_seq() == 20
    kept = log.since(0, None, limit=100)
    assert len(kept) == 5
    assert [event.zone for event in kept] == [f"zone{index}" for index in range(15, 20)]


def test_message_log_limit_caps_a_single_poll() -> None:
    log = MessageLog(limit=100)
    for index in range(30):
        log.append(camera="sofa", cat="bagel", zone=f"zone{index}")
    assert len(log.since(0, None, limit=10)) == 10


def test_status_reports_the_best_sub_threshold_confidence(server) -> None:
    """The UI needs this to explain an empty location panel."""
    _, context = server
    context.hub.publish(
        "sofa",
        make_frame(),
        make_frame(),
        detections=0,
        best_confidence=0.31,
    )
    camera = status_payload(context)["cameras"][0]
    assert camera["detections"] == 0
    assert camera["best_confidence"] == pytest.approx(0.31)
