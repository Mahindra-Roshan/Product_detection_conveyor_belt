"""
Background workers for the multi-line bag counting dashboard.

Each LineWorker owns a daemon thread that loops its assigned video file
forever, runs every frame through BagCounterPipeline, and publishes the
latest annotated frame + running counts into thread-safe state that the
Streamlit UI polls on each rerun. This decouples continuous "camera feed"
simulation from Streamlit's script-rerun execution model.

Line 3 additionally simulates a one-time conveyor jam: after a random
5-10s of normal playback it freezes on the current frame and raises a
jam flag permanently — it does not clear or resume on its own. The line
stays jammed until the worker is stopped (and restarted fresh on the
next Start).
"""

import random
import threading
import time

import cv2

from bag_counter import BagCounterPipeline

JAM_MIN_INTERVAL_SEC = 5
JAM_MAX_INTERVAL_SEC = 10

# The UI only polls every ~0.5s, so encoding every detected frame at full
# source fps (e.g. 24fps) wastes ~90%+ of the JPEG encode work. Throttling
# publishing to this interval cuts CPU load with no visible UI difference.
PUBLISH_INTERVAL_SEC = 0.4


class LineWorker:
    def __init__(self, name, video_path, direction, product_hint=None, simulate_jam=False):
        self.name = name
        self.video_path = video_path
        self.direction = direction
        self.product_hint = product_hint  # e.g. force a line's dominant product label
        self.simulate_jam = simulate_jam

        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self.frame_jpeg = None
        self.total_count = 0
        self.product_counts = {"Flour": 0, "Semolina": 0}
        self.bucket_counts = {}
        self.events = []  # recent events, capped
        self.jam_active = False
        self.status = "idle"  # idle | running | stopped

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.status = "running"

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.status = "stopped"
        self.jam_active = False

    def snapshot(self):
        """Thread-safe read of current state for the UI to render."""
        with self._lock:
            return {
                "name": self.name,
                "frame_jpeg": self.frame_jpeg,
                "total_count": self.total_count,
                "product_counts": dict(self.product_counts),
                "bucket_counts": dict(self.bucket_counts),
                "events": list(self.events[-10:]),
                "jam_active": self.jam_active,
                "status": self.status,
            }

    def _publish_frame(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self.frame_jpeg = buf.tobytes()

    def _next_jam_at(self, now):
        return now + random.uniform(JAM_MIN_INTERVAL_SEC, JAM_MAX_INTERVAL_SEC)

    def _run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            with self._lock:
                self.status = "error"
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        frame_period = 1.0 / fps

        pipeline = BagCounterPipeline(w, h, direction=self.direction)

        now = time.time()
        jam_at = self._next_jam_at(now) if self.simulate_jam else None
        jammed = False

        last_frame = None
        last_publish_at = 0.0

        while not self._stop_event.is_set():
            loop_start = time.time()

            if jammed:
                # Permanently frozen on the last frame until stopped.
                if last_frame is not None:
                    self._publish_frame(last_frame)
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            pipeline.process_frame(frame)
            last_frame = frame

            if self.simulate_jam and jam_at is not None and loop_start >= jam_at:
                jammed = True
                with self._lock:
                    self.jam_active = True

            with self._lock:
                self.total_count = pipeline.total_count
                self.product_counts = dict(pipeline.product_counts)
                self.bucket_counts = dict(pipeline.bucket_counts)
                self.events = list(pipeline.events[-10:])

            if loop_start - last_publish_at >= PUBLISH_INTERVAL_SEC:
                self._publish_frame(frame)
                last_publish_at = loop_start

            elapsed = time.time() - loop_start
            sleep_for = frame_period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        cap.release()
        with self._lock:
            self.status = "stopped"
