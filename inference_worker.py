from __future__ import annotations

import queue
import time
import threading
import gc
from collections import deque
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
from io import StringIO

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from inspect_coupler import (
    STATUS_BOX_TOO_LARGE,
    STATUS_CONNECTED,
    STATUS_DISCONNECTED,
    STATUS_NO_DETECTION,
    STATUS_POSSIBLE_DISCONNECTED,
    STATUS_UNCLEAR,
    choose_best_detection,
    classify_decision_crop,
    draw_results,
    majority_status,
)
from runtime_config import RuntimeSettings


_MODEL_CACHE_LOCK = threading.RLock()
_MODEL_CACHE = {
    "path": None,
    "model": None,
    "torch": None,
    "device": "CPU",
    "half": False,
    "classifier_path": None,
    "classifier_model": None,
}


class InferenceWorker(QObject):
    result_ready = Signal(object, dict)
    error = Signal(str)
    status = Signal(str)
    model_status = Signal(str)
    final_summary_ready = Signal(dict)
    finished = Signal()

    def __init__(self, frame_queue: queue.Queue, settings: RuntimeSettings) -> None:
        super().__init__()
        self.frame_queue = frame_queue
        self.settings = settings
        self._settings_lock = threading.RLock()
        self._running = False
        self._model = None
        self._classifier_model = None
        self._device = "CPU"
        self._half = False
        self._last_result = None
        self._last_decision: dict | None = None
        self._statuses: deque[str] = deque(maxlen=settings.majority_vote_window)
        self._processed_frames = 0
        self._skipped_frames = 0
        self._error_count = 0
        self._start_time = 0.0
        self._last_emit_time = 0.0
        self._last_emitted_status = ""
        self._torch = None
        self._cuda_cleanup_interval = 30.0
        self._last_cuda_cleanup = 0.0
        self._cuda_error_count = 0
        self._final_counts = {
            "connected": 0,
            "disconnected": 0,
            "unclear": 0,
            "no_detection": 0,
            "total": 0,
        }
        self._last_source = ""

    @Slot()
    def run(self) -> None:
        self._running = True
        self._start_time = time.perf_counter()
        if not self._load_model():
            self.finished.emit()
            return

        while self._running:
            try:
                item = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            frame = item["frame"]
            frame_number = int(item["frame_number"])
            source = item["source"]
            self._last_source = str(source)
            with self._settings_lock:
                local_settings = self.settings
            should_process = frame_number % max(1, local_settings.process_every_nth_frame) == 0

            try:
                if should_process or self._last_result is None or self._last_decision is None:
                    inference_start = time.perf_counter()
                    result = self._infer(frame)
                    if self._device == "GPU":
                        self._cuda_error_count = 0
                    inference_time = max(time.perf_counter() - inference_start, 1e-6)
                    decision = choose_best_detection(
                        result,
                        local_settings.confidence_threshold,
                        frame.shape,
                        local_settings.disconnected_override_confidence,
                        local_settings.large_disconnected_area_ratio,
                    )
                    decision = self._classify_decision(frame, result, decision, local_settings)
                    self._last_result = result
                    self._last_decision = decision
                    self._processed_frames += 1
                    inference_fps = 1.0 / inference_time
                else:
                    result = self._last_result
                    decision = dict(self._last_decision)
                    self._skipped_frames += 1
                    inference_fps = 0.0

                self._record_final_status(decision["status"])
                self._statuses.append(decision["status"])
                display_status = majority_status(self._statuses)
                now = time.perf_counter()
                min_emit_interval = 1.0 / max(1, int(local_settings.target_display_fps))
                status_changed = display_status != self._last_emitted_status
                if self._last_emit_time and not status_changed and now - self._last_emit_time < min_emit_interval:
                    continue

                annotated = draw_results(
                    frame,
                    result,
                    decision,
                    local_settings.confidence_threshold,
                    local_settings.show_confidence,
                    display_status,
                    local_settings.show_all_boxes,
                    local_settings.show_decision_box_only,
                )
                display_fps = 1.0 / max(now - self._last_emit_time, 1e-6) if self._last_emit_time else 0.0
                self._last_emit_time = now
                self._last_emitted_status = display_status
                average_fps = (self._processed_frames + self._skipped_frames) / max(now - self._start_time, 1e-6)
                payload = {
                    **decision,
                    "status": display_status,
                    "raw_status": decision["status"],
                    "frame_number": frame_number,
                    "source": source,
                    "display_fps": display_fps,
                    "inference_fps": inference_fps,
                    "average_fps": average_fps,
                    "processed_frames": self._processed_frames,
                    "skipped_frames": self._skipped_frames,
                    "dropped_frames": int(item.get("dropped_frames", 0)),
                    "error_count": self._error_count,
                    "device": self._device,
                    "settings": asdict(self.settings),
                }
                self.result_ready.emit(annotated, payload)
            except RuntimeError as exc:
                message = str(exc)
                if "CUDA" in message.upper() and self._device == "GPU":
                    self._cuda_error_count += 1
                    self._cleanup_cuda(force=True)
                    if self._cuda_error_count <= 3:
                        self._half = False
                        self.error.emit("CUDA inference hiccup cleared. Retrying on GPU with safer precision.")
                        continue
                    self.error.emit("CUDA inference failed repeatedly. Falling back to CPU to keep the app running.")
                    self._device = "CPU"
                    self._half = False
                    self._move_model_to_cpu()
                    continue
                self._handle_frame_error(message)
            except Exception as exc:
                self._handle_frame_error(str(exc))

        self.final_summary_ready.emit(self._build_final_summary())
        self._release_model()
        self.finished.emit()

    @Slot(dict)
    def update_settings(self, values: dict) -> None:
        with self._settings_lock:
            current = asdict(self.settings)
            current.update(values)
            self.settings = RuntimeSettings(**current)
            self._statuses = deque(self._statuses, maxlen=self.settings.majority_vote_window)

    @Slot()
    def stop(self) -> None:
        self._running = False

    def _load_model(self) -> bool:
        if not self.settings.model_path.exists():
            self.error.emit(f"Model missing: {self.settings.model_path}")
            self.model_status.emit("Model missing")
            self._running = False
            return False

        try:
            from ultralytics import YOLO
            import torch

            model_path = str(self.settings.model_path.resolve())
            classifier_path = (
                str(self.settings.classifier_model_path.resolve())
                if self.settings.use_classifier and self.settings.classifier_model_path.exists()
                else None
            )
            with _MODEL_CACHE_LOCK:
                if _MODEL_CACHE["model"] is not None and _MODEL_CACHE["path"] == model_path:
                    self._model = _MODEL_CACHE["model"]
                    self._torch = _MODEL_CACHE["torch"]
                    self._device = str(_MODEL_CACHE["device"])
                    self._half = bool(_MODEL_CACHE["half"])
                else:
                    with self._quiet_console():
                        self._model = YOLO(str(self.settings.model_path))
                    self._torch = torch
                    if torch.cuda.is_available():
                        self._device = "GPU"
                        self._half = True
                        self._model.to("cuda:0")
                        self._warmup_gpu()
                    else:
                        self._device = "CPU"
                        self._half = False
                    _MODEL_CACHE.update(
                        {
                            "path": model_path,
                            "model": self._model,
                            "torch": self._torch,
                            "device": self._device,
                            "half": self._half,
                        }
                    )

                if classifier_path is None:
                    self._classifier_model = None
                    _MODEL_CACHE["classifier_path"] = None
                    _MODEL_CACHE["classifier_model"] = None
                elif (
                    _MODEL_CACHE["classifier_model"] is not None
                    and _MODEL_CACHE["classifier_path"] == classifier_path
                ):
                    self._classifier_model = _MODEL_CACHE["classifier_model"]
                else:
                    with self._quiet_console():
                        self._classifier_model = YOLO(str(self.settings.classifier_model_path))
                    if self._device == "GPU":
                        self._classifier_model.to("cuda:0")
                    _MODEL_CACHE["classifier_path"] = classifier_path
                    _MODEL_CACHE["classifier_model"] = self._classifier_model

            classifier_status = " + classifier" if self._classifier_model is not None else " (classifier optional)"
            self.model_status.emit(f"YOLO loaded on {self._device}{classifier_status}")
            return True
        except Exception as exc:
            self.error.emit(f"Could not load YOLO model: {exc}")
            self.model_status.emit("Model load failed")
            self._running = False
            return False

    def _infer(self, frame):
        device_arg = 0 if self._device == "GPU" else "cpu"
        torch_module = self._torch
        inference_context = torch_module.inference_mode() if torch_module is not None else self._null_context()
        with inference_context:
            with self._quiet_console():
                result = self._model.predict(
                    frame,
                    imgsz=self.settings.inference_size,
                    conf=0.05,
                    iou=self.settings.iou_threshold,
                    device=device_arg,
                    half=self._half,
                    verbose=False,
                )[0]
        if self._device == "GPU":
            result = result.cpu()
            self._cleanup_cuda_if_needed()
        return result

    def _classify_decision(self, frame, result, decision: dict, settings: RuntimeSettings) -> dict:
        if not settings.use_classifier:
            return decision
        device_arg = 0 if self._device == "GPU" else "cpu"
        torch_module = self._torch
        inference_context = torch_module.inference_mode() if torch_module is not None else self._null_context()
        with inference_context:
            with self._quiet_console():
                refined = classify_decision_crop(
                    self._classifier_model,
                    frame,
                    result,
                    decision,
                    settings.classifier_confidence_threshold,
                    settings.classifier_inference_size,
                    device_arg,
                    self._half and self._device == "GPU",
                )
        if self._device == "GPU":
            self._cleanup_cuda_if_needed()
        return refined

    def _warmup_gpu(self) -> None:
        if self._model is None:
            return
        try:
            warmup_frame = np.zeros((self.settings.inference_size, self.settings.inference_size, 3), dtype=np.uint8)
            with self._quiet_console():
                self._model.predict(
                    warmup_frame,
                    imgsz=self.settings.inference_size,
                    conf=0.05,
                    iou=self.settings.iou_threshold,
                    device=0,
                    half=self._half,
                    verbose=False,
                )
            self._cleanup_cuda(force=True)
        except Exception as exc:
            self.error.emit(f"GPU warmup failed: {exc}")

    def _cleanup_cuda_if_needed(self) -> None:
        now = time.perf_counter()
        if now - self._last_cuda_cleanup >= self._cuda_cleanup_interval:
            self._cleanup_cuda(force=True)

    def _cleanup_cuda(self, force: bool = False) -> None:
        if not force or self._torch is None or self._device != "GPU":
            return
        try:
            self._torch.cuda.empty_cache()
            self._last_cuda_cleanup = time.perf_counter()
        except Exception:
            pass

    @contextmanager
    def _quiet_console(self):
        sink = StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            yield

    @contextmanager
    def _null_context(self):
        yield

    def _move_model_to_cpu(self) -> None:
        try:
            self._model.to("cpu")
        except Exception:
            pass
        try:
            if self._classifier_model is not None:
                self._classifier_model.to("cpu")
        except Exception:
            pass
        self._cleanup_cuda(force=True)
        self.model_status.emit("Model running on CPU")

    def _record_final_status(self, status: str) -> None:
        self._final_counts["total"] += 1
        if status == STATUS_CONNECTED:
            self._final_counts["connected"] += 1
        elif status in {STATUS_DISCONNECTED, STATUS_POSSIBLE_DISCONNECTED}:
            self._final_counts["disconnected"] += 1
        elif status == STATUS_NO_DETECTION:
            self._final_counts["no_detection"] += 1
        elif status in {STATUS_UNCLEAR, STATUS_BOX_TOO_LARGE}:
            self._final_counts["unclear"] += 1
        else:
            self._final_counts["unclear"] += 1

    def _build_final_summary(self) -> dict:
        connected = self._final_counts["connected"]
        disconnected = self._final_counts["disconnected"]
        unclear = self._final_counts["unclear"]
        no_detection = self._final_counts["no_detection"]
        total = self._final_counts["total"]
        clear_total = connected + disconnected

        if clear_total == 0:
            outcome = "UNCLEAR - MANUAL CHECK REQUIRED"
            reason = "No connected/disconnected detections were confident enough."
        elif connected == disconnected:
            outcome = "UNCLEAR - MANUAL CHECK REQUIRED"
            reason = "Connected and disconnected counts are tied."
        elif disconnected > connected:
            outcome = STATUS_DISCONNECTED
            reason = "Disconnected appeared more often than connected."
        else:
            outcome = STATUS_CONNECTED
            reason = "Connected appeared more often than disconnected."

        confidence = (max(connected, disconnected) / clear_total) if clear_total else 0.0
        return {
            "source": self._last_source,
            "outcome": outcome,
            "reason": reason,
            "connected": connected,
            "disconnected": disconnected,
            "unclear": unclear,
            "no_detection": no_detection,
            "total": total,
            "clear_total": clear_total,
            "confidence": confidence,
        }

    def _release_model(self) -> None:
        self._last_result = None
        self._last_decision = None
        if self._torch is not None and self._device == "GPU":
            try:
                self._torch.cuda.synchronize()
            except Exception:
                pass
        self._cleanup_cuda(force=True)
        self._model = None
        self._classifier_model = None
        gc.collect()

    def _handle_frame_error(self, message: str) -> None:
        self._error_count += 1
        self.error.emit(f"Inference frame error: {message}")
        fallback = {
            "status": STATUS_NO_DETECTION,
            "raw_status": STATUS_NO_DETECTION,
            "detected_class": "",
            "confidence": -1.0,
            "frame_number": 0,
            "display_fps": 0.0,
            "inference_fps": 0.0,
            "average_fps": 0.0,
            "processed_frames": self._processed_frames,
            "skipped_frames": self._skipped_frames,
            "dropped_frames": 0,
            "error_count": self._error_count,
            "device": self._device,
        }
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        self.result_ready.emit(blank, fallback)
