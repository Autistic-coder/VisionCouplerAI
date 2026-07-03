from __future__ import annotations

import queue
from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, Signal, Slot

from inspect_coupler import (
    STATUS_BOX_TOO_LARGE,
    STATUS_DISCONNECTED,
    STATUS_POSSIBLE_DISCONNECTED,
    STATUS_UNCLEAR,
)
from runtime_config import LOG_QUEUE_SIZE, PROCESSED_VIDEO_DIR, SCREENSHOT_DIR


ALERT_STATUSES = {STATUS_DISCONNECTED, STATUS_UNCLEAR, STATUS_POSSIBLE_DISCONNECTED, STATUS_BOX_TOO_LARGE}


class LoggingWorker(QObject):
    error = Signal(str)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.queue: queue.Queue = queue.Queue(maxsize=LOG_QUEUE_SIZE)
        self._running = False
        self._writer: cv2.VideoWriter | None = None
        self._writer_path: Path | None = None

    @Slot()
    def run(self) -> None:
        self._running = True
        while self._running or not self.queue.empty():
            try:
                frame, data = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._write_event(frame, data)
            except Exception as exc:
                self.error.emit(f"Logging error: {exc}")
        self._release_writer()
        self.finished.emit()

    @Slot(object, dict)
    def enqueue(self, frame, data: dict) -> None:
        settings = data.get("settings", {})
        if not settings.get("save_screenshots", False) and not settings.get("save_processed_video", False):
            self._release_writer()
            return

        try:
            self.queue.put_nowait((frame.copy(), dict(data)))
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait((frame.copy(), dict(data)))
            except Exception:
                pass

    @Slot()
    def stop(self) -> None:
        self._running = False

    def _write_event(self, frame, data: dict) -> None:
        settings = data.get("settings", {})
        frame_number = int(data.get("frame_number", 0))
        status = data.get("status", "")

        if settings.get("save_screenshots", False):
            every_n = max(1, int(settings.get("save_every_n_alert_frames", 20)))
            if status in ALERT_STATUSES and frame_number % every_n == 0:
                self._save_screenshot(frame, data)

        if settings.get("save_processed_video", False):
            self._write_video(frame, data)
        else:
            self._release_writer()

    def _save_screenshot(self, frame, data: dict) -> None:
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        source = Path(str(data.get("source", "source"))).stem or "webcam"
        safe_status = str(data.get("status", "status")).lower().replace(" ", "_").replace("/", "_")
        path = SCREENSHOT_DIR / f"{source}_frame_{data.get('frame_number', 0):06d}_{safe_status}_{timestamp}.jpg"
        cv2.imwrite(str(path), frame)

    def _write_video(self, frame, data: dict) -> None:
        if self._writer is None:
            PROCESSED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            source = Path(str(data.get("source", "source"))).stem or "webcam"
            self._writer_path = PROCESSED_VIDEO_DIR / f"{source}_{timestamp}_desktop_processed.mp4"
            height, width = frame.shape[:2]
            fps = max(1.0, min(30.0, float(data.get("display_fps", 20.0) or 20.0)))
            self._writer = cv2.VideoWriter(str(self._writer_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if self._writer is not None:
            self._writer.write(frame)

    def _release_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self._writer_path = None
