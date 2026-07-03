from __future__ import annotations

import os
import queue
import time
from pathlib import Path
from typing import Any

import cv2
from PySide6.QtCore import QObject, Signal, Slot

from runtime_config import FRAME_QUEUE_SIZE


class CaptureWorker(QObject):
    frame_captured = Signal(int, int, float)
    error = Signal(str)
    status = Signal(str)
    finished = Signal(int)

    def __init__(self, frame_queue: queue.Queue, source: Any, source_name: str, processing_width: int, run_id: int) -> None:
        super().__init__()
        self.frame_queue = frame_queue
        self.source = source
        self.source_name = source_name
        self.processing_width = processing_width
        self.run_id = run_id
        self._running = False
        self._dropped_frames = 0
        self._capture: cv2.VideoCapture | None = None

    @Slot()
    def run(self) -> None:
        self._running = True
        self._dropped_frames = 0
        capture_source = self.source
        if isinstance(capture_source, Path):
            capture_source = str(capture_source)

        self._capture = self._open_capture(capture_source)
        if not self._capture.isOpened():
            self.error.emit(f"Could not open source: {self.source_name}")
            self._cleanup()
            self.finished.emit(self.run_id)
            return

        self.status.emit(f"Started {self.source_name}")
        frame_number = 0
        consecutive_failures = 0
        source_fps = self._capture.get(cv2.CAP_PROP_FPS) if self._is_file_source(self.source) else 0.0
        min_frame_interval = 1.0 / source_fps if source_fps and source_fps > 1.0 else 0.0
        last_emit_time = 0.0
        try:
            while self._running:
                if min_frame_interval > 0 and last_emit_time:
                    elapsed = time.perf_counter() - last_emit_time
                    sleep_time = min_frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(min(sleep_time, 0.05))

                ok, frame = self._capture.read()
                if not ok or frame is None:
                    if self._is_file_source(self.source):
                        self.status.emit("End of selected video")
                        break
                    consecutive_failures += 1
                    if consecutive_failures <= 30:
                        time.sleep(0.03)
                        continue
                    self.error.emit(f"Could not read frame from {self.source_name}")
                    break

                consecutive_failures = 0
                frame_number += 1
                frame = self._resize_once(frame)
                item = {
                    "frame": frame,
                    "frame_number": frame_number,
                    "source": self.source_name,
                    "dropped_frames": self._dropped_frames,
                    "captured_at": time.perf_counter(),
                }
                self._put_latest(item)
                last_emit_time = item["captured_at"]
                self.frame_captured.emit(frame_number, self._dropped_frames, item["captured_at"])
        except Exception as exc:
            self.error.emit(f"Capture error: {exc}")
        finally:
            self._cleanup()
            self.finished.emit(self.run_id)

    @Slot()
    def stop(self) -> None:
        self._running = False

    def _open_capture(self, capture_source):
        if isinstance(capture_source, int):
            capture = cv2.VideoCapture(capture_source, cv2.CAP_DSHOW)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            capture.set(cv2.CAP_PROP_FPS, 30)
            return capture
        if isinstance(capture_source, str) and capture_source.lower().startswith("rtsp://"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        capture = cv2.VideoCapture(capture_source, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _is_file_source(self, source: Any) -> bool:
        if isinstance(source, Path):
            return True
        if not isinstance(source, str):
            return False
        lowered = source.lower()
        if "://" in lowered:
            return False
        return Path(source).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

    def _resize_once(self, frame):
        if self.processing_width <= 0 or frame.shape[1] <= self.processing_width:
            return frame
        scale = self.processing_width / float(frame.shape[1])
        height = max(1, int(frame.shape[0] * scale))
        return cv2.resize(frame, (self.processing_width, height), interpolation=cv2.INTER_AREA)

    def _put_latest(self, item: dict) -> None:
        while self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
                self._dropped_frames += 1
            except queue.Empty:
                break
        try:
            self.frame_queue.put_nowait(item)
        except queue.Full:
            self._dropped_frames += 1

    def _cleanup(self) -> None:
        self._running = False
        if self._capture is not None:
            self._capture.release()
            self._capture = None
