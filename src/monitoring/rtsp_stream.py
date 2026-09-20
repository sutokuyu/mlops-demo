"""Self-healing RTSP reader that keeps only the newest frame.

An RTSP stream must be drained continuously. If the client stops reading, the
decoder falls behind the sender and starts emitting errors such as
``[hevc] Could not find ref with POC ...`` because reference frames were dropped.
"""

import os
import threading
import time
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import cv2

RECONNECT_INTERVAL_SECONDS = 5.0
READ_TIMEOUT_SECONDS = 10.0


def _configure_ffmpeg_environment() -> None:
    """Prefer TCP transport so packet loss does not corrupt HEVC/H.264 refs."""
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


@dataclass
class StreamReader:
    """Read a source in a background thread and expose the latest frame."""

    source: str | int
    name: str = "stream"
    reconnect_interval_seconds: float = RECONNECT_INTERVAL_SECONDS
    frame_lock: threading.Lock = dataclass_field(default_factory=threading.Lock)
    stop_event: threading.Event = dataclass_field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    latest_frame = None
    latest_frame_at: float = 0.0
    connected: bool = False
    last_error: str = ""

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"reader-{self.name}", daemon=True)
        self.thread.start()

    def _open_capture(self) -> cv2.VideoCapture | None:
        _configure_ffmpeg_environment()
        api_preference = cv2.CAP_FFMPEG if isinstance(self.source, str) else cv2.CAP_ANY
        capture = cv2.VideoCapture(self.source, api_preference)
        if not capture.isOpened():
            capture.release()
            return None
        return capture

    def _run(self) -> None:
        capture: cv2.VideoCapture | None = None
        while not self.stop_event.is_set():
            if capture is None:
                capture = self._open_capture()
                if capture is None:
                    self.connected = False
                    self.last_error = "unable to open stream"
                    self.stop_event.wait(self.reconnect_interval_seconds)
                    continue
                self.connected = True
                self.last_error = ""
                print(f"[{self.name}] connected")

            ok, frame = capture.read()
            if not ok:
                self.connected = False
                self.last_error = "stream ended"
                print(f"[{self.name}] lost stream; reconnecting")
                capture.release()
                capture = None
                self.stop_event.wait(self.reconnect_interval_seconds)
                continue

            with self.frame_lock:
                self.latest_frame = frame
                self.latest_frame_at = time.monotonic()

        if capture is not None:
            capture.release()
        self.connected = False

    def get_latest_frame(self, stale_after_seconds: float = 0.0):
        """Return the newest frame, or None when missing/stale."""
        with self.frame_lock:
            frame = self.latest_frame
            frame_at = self.latest_frame_at
        if frame is None:
            return None
        if stale_after_seconds and time.monotonic() - frame_at > stale_after_seconds:
            return None
        return frame

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.connected = False
