from __future__ import annotations

import queue
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from runtime_config import (
    APP_NAME,
    APP_DIR,
    CAMERA_CONFIG_PATH,
    DEFAULT_CAMERA_INDEX,
    FRAME_QUEUE_SIZE,
    RuntimeSettings,
    ensure_runtime_dirs,
    load_ethernet_camera_sources,
)


STATUS_COLORS = {
    "COUPLER CONNECTED": "#178f3b",
    "COUPLER DISCONNECTED": "#c92a2a",
    "UNCLEAR - MANUAL CHECK REQUIRED": "#c99700",
    "POSSIBLE DISCONNECTED - CHECK REQUIRED": "#c99700",
    "UNCLEAR - BOX TOO LARGE": "#c99700",
    "NO COUPLER DETECTED": "#d97000",
}


class MainWindow(QMainWindow):
    stop_capture_requested = Signal()
    stop_inference_requested = Signal()
    stop_logging_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        ensure_runtime_dirs()
        self.setWindowTitle(APP_NAME)
        self.resize(1320, 820)

        self.selected_video: Path | None = None
        self.frame_queue: queue.Queue | None = None
        self.capture_thread: QThread | None = None
        self.capture_worker: Any | None = None
        self.inference_thread: QThread | None = None
        self.inference_worker: Any | None = None
        self.logging_thread: QThread | None = None
        self.logging_worker: Any | None = None
        self.last_status = ""
        self._stopping = False
        self._run_id = 0
        self._pending_frame = None
        self._pending_data: dict | None = None
        self._current_source_is_video = False
        self._source_finished = False
        self._desired_pixmap: QPixmap | None = None

        self._build_ui()
        self._load_desired_condition_image()
        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(max(1, int(1000 / RuntimeSettings().target_display_fps)))
        self._ui_timer.timeout.connect(self._flush_pending_result)
        self._ui_timer.start()
        self._set_running(False)
        self._update_model_status()
        self.refresh_cameras()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        main_layout = QHBoxLayout()
        layout.addLayout(main_layout, stretch=1)

        desired_panel = self._build_display_panel("DESIRED CONDITION")
        self.desired_label = desired_panel["display"]
        main_layout.addWidget(desired_panel["widget"], stretch=4)

        actual_panel = self._build_display_panel("ACTUAL CONDITION")
        self.video_label = actual_panel["display"]
        self.video_label.setText("Start camera or select a video")
        self.video_label.setAlignment(Qt.AlignCenter)
        actual_column = QVBoxLayout()
        actual_column.addWidget(actual_panel["widget"], stretch=1)

        controls = QGridLayout()
        self.camera_combo = QComboBox()
        self.refresh_camera_button = QPushButton("Refresh Cameras")
        self.start_webcam_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop")
        self.select_video_button = QPushButton("Select Video")
        self.run_video_button = QPushButton("Run Selected Video")
        controls.addWidget(self.camera_combo, 0, 0)
        controls.addWidget(self.refresh_camera_button, 0, 1)
        controls.addWidget(self.start_webcam_button, 0, 2)
        controls.addWidget(self.stop_button, 1, 0)
        controls.addWidget(self.select_video_button, 1, 1)
        controls.addWidget(self.run_video_button, 1, 2)
        actual_column.addLayout(controls)
        main_layout.addLayout(actual_column, stretch=5)

        right = QVBoxLayout()
        main_layout.addLayout(right, stretch=2)

        telemetry = QGroupBox("STATUS")
        grid = QGridLayout(telemetry)
        self.model_status = QLabel("Checking model...")
        self.fps_label = QLabel("Display FPS: 0.0")
        self.inference_fps_label = QLabel("Inference FPS: 0.0")
        self.average_fps_label = QLabel("Average FPS: 0.0")
        self.frame_label = QLabel("Frame: 0")
        self.dropped_label = QLabel("Dropped: 0")
        self.processed_label = QLabel("Processed: 0")
        self.skipped_label = QLabel("Skipped: 0")
        self.class_label = QLabel("Class: -")
        self.conf_label = QLabel("Confidence: -")
        self.classifier_label = QLabel("Classifier: -")
        self.device_label = QLabel("Device: CPU")
        self.error_label = QLabel("Errors: 0")
        labels = [
            self.model_status,
            self.fps_label,
            self.inference_fps_label,
            self.average_fps_label,
            self.frame_label,
            self.dropped_label,
            self.processed_label,
            self.skipped_label,
            self.class_label,
            self.conf_label,
            self.classifier_label,
            self.device_label,
            self.error_label,
        ]
        for index, label in enumerate(labels):
            grid.addWidget(label, index // 2, index % 2)
        right.addWidget(telemetry)

        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setFixedHeight(120)
        self.event_log.setStyleSheet(
            "background:#101418;color:#9db3c8;font-size:12px;border:1px solid #303840;"
        )
        right.addWidget(self.event_log)
        self.final_outcome_label = QLabel("")
        self.final_outcome_label.setWordWrap(True)
        self.final_outcome_label.setMinimumHeight(76)
        self.final_outcome_label.setAlignment(Qt.AlignCenter)
        self.final_outcome_label.setStyleSheet(
            "background:#101418;color:#dce3ea;font-size:18px;font-weight:700;border:1px solid #303840;padding:8px;"
        )
        self.final_outcome_label.hide()
        right.addWidget(self.final_outcome_label)
        right.addStretch(1)

        bottom_layout = QHBoxLayout()
        layout.addLayout(bottom_layout)

        self.status_banner = QLabel("NO COUPLER DETECTED")
        self.status_banner.setAlignment(Qt.AlignCenter)
        self.status_banner.setMinimumHeight(58)
        self.status_banner.setStyleSheet(self._banner_style("#d97000"))
        bottom_layout.addWidget(self.status_banner, stretch=1)

        self.footer_label = QLabel("FY 26-27\ndesigned by: Vaibhav Bedi")
        self.footer_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.footer_label.setStyleSheet("color:#7b858f;font-size:10px;padding:0 4px 4px 0;")
        bottom_layout.addWidget(self.footer_label)

        self.refresh_camera_button.clicked.connect(self.refresh_cameras)
        self.start_webcam_button.clicked.connect(self.start_webcam)
        self.stop_button.clicked.connect(self.stop_all)
        self.select_video_button.clicked.connect(self.select_video)
        self.run_video_button.clicked.connect(self.run_selected_video)

    def _build_display_panel(self, title: str) -> dict[str, QWidget | QLabel]:
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        heading = QLabel(title)
        heading.setAlignment(Qt.AlignCenter)
        heading.setMinimumHeight(34)
        heading.setStyleSheet("background:#101418;color:#dce3ea;font-size:17px;font-weight:700;border:1px solid #303840;")
        display = QLabel()
        display.setAlignment(Qt.AlignCenter)
        display.setMinimumSize(360, 500)
        display.setStyleSheet("background:#101418;color:#dce3ea;font-size:18px;border:1px solid #303840;")
        panel_layout.addWidget(heading)
        panel_layout.addWidget(display, stretch=1)
        return {"widget": panel, "display": display}

    def current_settings(self) -> RuntimeSettings:
        return RuntimeSettings().with_performance_mode()

    def _load_desired_condition_image(self) -> None:
        image_path = self._find_desired_condition_image()
        if image_path is None:
            self._desired_pixmap = None
            self.desired_label.setText("Desired condition image not found")
            return

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self._desired_pixmap = None
            self.desired_label.setText("Desired condition image not found")
            return

        self._desired_pixmap = pixmap
        self.desired_label.setToolTip(str(image_path.relative_to(APP_DIR)) if image_path.is_relative_to(APP_DIR) else image_path.name)
        self._show_desired_condition_image()

    def _find_desired_condition_image(self) -> Path | None:
        image_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
        search_roots = [
            APP_DIR / "reference_images",
            APP_DIR / "raw_photos",
            APP_DIR / "dataset",
            APP_DIR / "datasets",
            APP_DIR / "data",
            APP_DIR / "images",
            APP_DIR / "train" / "images",
            APP_DIR / "valid" / "images",
            APP_DIR / "annotated_images" / "images" / "train",
            APP_DIR / "annotated_images" / "images" / "val",
            APP_DIR / "annotated_images" / "images" / "test",
            APP_DIR / "final annotated images",
            APP_DIR / "annotated_mixed",
            APP_DIR / "annotated_images",
        ]
        preferred_words = ("connected", "engaged", "positive", "coupler")
        candidates: list[tuple[int, str, Path]] = []
        seen: set[Path] = set()
        for root in search_roots:
            if not root.exists():
                continue
            for image_path in root.rglob("*"):
                if not image_path.is_file() or image_path.suffix.lower() not in image_suffixes:
                    continue
                resolved = image_path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                lowered = str(image_path).lower()
                score = sum(10 for word in preferred_words if word in lowered)
                if image_path.parent.name.lower() == "reference_images":
                    score += 100
                if self._has_connected_label(image_path):
                    score += 50
                candidates.append((score, lowered, image_path))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][2]

    def _has_connected_label(self, image_path: Path) -> bool:
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            return False
        try:
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts and parts[0] == "0":
                    return True
        except OSError:
            return False
        return False

    def _show_desired_condition_image(self) -> None:
        if self._desired_pixmap is None:
            return
        self.desired_label.setPixmap(
            self._desired_pixmap.scaled(
                self.desired_label.size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            )
        )

    @Slot()
    def start_webcam(self) -> None:
        selected_camera = self.camera_combo.currentData()
        if not isinstance(selected_camera, dict):
            selected_camera = {"source": DEFAULT_CAMERA_INDEX, "name": f"Camera {DEFAULT_CAMERA_INDEX}"}
        source = selected_camera.get("source", DEFAULT_CAMERA_INDEX)
        source_name = str(selected_camera.get("name") or f"Camera {source}")
        self._start_source(source, source_name)

    @Slot()
    def refresh_cameras(self) -> None:
        import cv2

        selected = self.camera_combo.currentData() if hasattr(self, "camera_combo") else None
        selected_source = selected.get("source") if isinstance(selected, dict) else selected
        self.camera_combo.clear()
        found_any = False
        configured_cameras = load_ethernet_camera_sources()
        for camera in configured_cameras:
            found_any = True
            self.camera_combo.addItem(camera.name, {"source": camera.source, "name": camera.name})

        for index in range(6):
            capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            available = capture.isOpened()
            if available:
                found_any = True
                self.camera_combo.addItem(f"Local Camera {index}", {"source": index, "name": f"Local Camera {index}"})
            capture.release()

        if not found_any:
            self.camera_combo.addItem(
                f"Local Camera {DEFAULT_CAMERA_INDEX}",
                {"source": DEFAULT_CAMERA_INDEX, "name": f"Local Camera {DEFAULT_CAMERA_INDEX}"},
            )
            self._add_event(
                f"Ethernet camera is not auto-detected like USB. Enter its RTSP/HTTP URL in {CAMERA_CONFIG_PATH.name}."
            )

        for item_index in range(self.camera_combo.count()):
            item = self.camera_combo.itemData(item_index)
            item_source = item.get("source") if isinstance(item, dict) else item
            if item_source == selected_source:
                self.camera_combo.setCurrentIndex(item_index)
                break

    @Slot()
    def select_video(self) -> None:
        if self.stop_button.isEnabled():
            QMessageBox.information(self, APP_NAME, "Stop the current inspection before selecting another video.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select inspection video",
            str((APP_DIR / "raw_videos").resolve()),
            "Videos (*.mp4 *.avi *.mov *.mkv *.wmv);;All files (*.*)",
        )
        if path:
            self.selected_video = Path(path)
            self._add_event(f"Selected video: {self.selected_video.name}")
            self.run_video_button.setEnabled(True)

    @Slot()
    def run_selected_video(self) -> None:
        if self.selected_video is None:
            QMessageBox.warning(self, APP_NAME, "Select a video first.")
            return
        self._start_source(self.selected_video, str(self.selected_video))

    def _start_source(self, source, source_name: str) -> None:
        from camera_worker import CaptureWorker
        from inference_worker import InferenceWorker

        if self._object_alive(self.capture_thread) or self._object_alive(self.inference_thread):
            self.stop_all()
            self._add_event("Still stopping previous source. Start again when Stop is complete.")
            return
        self._ensure_logging_worker()
        self._run_id += 1
        run_id = self._run_id
        settings = self.current_settings()
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self._current_source_is_video = isinstance(source, Path)
        self._source_finished = False
        self._reset_final_outcome()

        self.capture_worker = CaptureWorker(self.frame_queue, source, source_name, settings.processing_width, run_id)
        self.capture_thread = QThread()
        capture_thread = self.capture_thread
        capture_worker = self.capture_worker
        capture_worker.moveToThread(capture_thread)
        capture_thread.started.connect(capture_worker.run)
        capture_worker.error.connect(self._show_error)
        capture_worker.status.connect(self._add_event)
        capture_worker.finished.connect(self._handle_capture_finished)
        capture_worker.finished.connect(lambda _run_id, thread=capture_thread: thread.quit())
        capture_worker.finished.connect(lambda _run_id, worker=capture_worker: worker.deleteLater())
        capture_thread.finished.connect(lambda run_id=run_id: self._clear_capture_refs(run_id))
        capture_thread.finished.connect(capture_thread.deleteLater)

        self.inference_worker = InferenceWorker(self.frame_queue, settings)
        self.inference_thread = QThread()
        inference_thread = self.inference_thread
        inference_worker = self.inference_worker
        inference_worker.moveToThread(inference_thread)
        inference_thread.started.connect(inference_worker.run)
        inference_worker.result_ready.connect(self._handle_result)
        inference_worker.error.connect(self._show_error)
        inference_worker.status.connect(self._add_event)
        inference_worker.model_status.connect(self._set_model_status)
        inference_worker.final_summary_ready.connect(self._handle_final_summary)
        inference_worker.finished.connect(inference_thread.quit)
        inference_worker.finished.connect(inference_worker.deleteLater)
        inference_thread.finished.connect(lambda run_id=run_id: self._clear_inference_refs(run_id))
        inference_thread.finished.connect(inference_thread.deleteLater)

        self._set_running(True)
        inference_thread.start()
        capture_thread.start()
        self._add_event(f"Running {source_name}")

    @Slot()
    def stop_all(self) -> None:
        if not (
            self._object_alive(self.capture_worker)
            or self._object_alive(self.inference_worker)
            or self._object_alive(self.capture_thread)
            or self._object_alive(self.inference_thread)
        ):
            self.frame_queue = None
            self._current_source_is_video = False
            self._source_finished = False
            self._set_running(False)
            self._stopping = False
            return

        self._stopping = True
        self._pending_frame = None
        self._pending_data = None
        self._request_stop(self.capture_worker)
        self._request_stop(self.inference_worker)
        self._set_running(True)
        self._add_event("Stopping current source...")

    @Slot(int)
    def _handle_capture_finished(self, run_id: int) -> None:
        if run_id != self._run_id:
            return
        self._source_finished = True
        self._request_stop(self.inference_worker)
        if not self._stopping:
            self._add_event("Source finished. Releasing model before next video...")

    def _clear_capture_refs(self, run_id: int) -> None:
        if run_id == self._run_id:
            self.capture_thread = None
            self.capture_worker = None
            self._maybe_finish_stopping()
            self._maybe_finish_video_run()

    def _clear_inference_refs(self, run_id: int) -> None:
        if run_id == self._run_id:
            self.inference_thread = None
            self.inference_worker = None
            self._maybe_finish_stopping()
            self._maybe_finish_video_run()

    def _maybe_finish_stopping(self) -> None:
        if not self._stopping:
            return
        if self.capture_thread is not None or self.inference_thread is not None:
            return
        self.frame_queue = None
        self._current_source_is_video = False
        self._source_finished = False
        self._stopping = False
        self._set_running(False)
        self._add_event("Stopped. You can select another video.")

    def _maybe_finish_video_run(self) -> None:
        if self._stopping or not self._source_finished:
            return
        if self.capture_thread is not None or self.inference_thread is not None:
            return
        self.frame_queue = None
        self._current_source_is_video = False
        self._source_finished = False
        self._set_running(False)
        self._add_event("Source finished. You can select another video.")

    def _wait_thread(self, thread: QThread | None) -> None:
        if not self._object_alive(thread):
            return
        if thread is not None and thread.isRunning():
            thread.quit()
            if not thread.wait(60000):
                self._add_event("Worker is still stopping. Please wait before starting another source.")

    def _object_alive(self, obj) -> bool:
        if obj is None:
            return False
        try:
            obj.objectName()
            return True
        except RuntimeError:
            return False

    def _request_stop(self, worker) -> None:
        if not self._object_alive(worker):
            return
        try:
            worker.stop()
        except RuntimeError:
            pass

    def _ensure_logging_worker(self) -> None:
        if self._object_alive(self.logging_thread) and self.logging_worker is not None:
            return

        from logging_worker import LoggingWorker

        self.logging_worker = LoggingWorker()
        self.logging_thread = QThread()
        self.logging_worker.moveToThread(self.logging_thread)
        self.logging_thread.started.connect(self.logging_worker.run)
        self.logging_worker.error.connect(self._show_error)
        self.logging_worker.finished.connect(self.logging_thread.quit)
        self.logging_worker.finished.connect(self.logging_worker.deleteLater)
        self.logging_thread.finished.connect(self.logging_thread.deleteLater)
        self.stop_logging_requested.connect(self.logging_worker.stop, Qt.DirectConnection)
        self.logging_thread.start()

    @Slot(object, dict)
    def _handle_result(self, frame, data: dict) -> None:
        if self.logging_worker is not None:
            self.logging_worker.enqueue(frame, data)
        self._pending_frame = frame
        self._pending_data = data

    @Slot()
    def _flush_pending_result(self) -> None:
        if self._pending_frame is None or self._pending_data is None:
            return
        frame = self._pending_frame
        data = self._pending_data
        self._pending_frame = None
        self._pending_data = None

        self._show_frame(frame)
        status = data.get("status", "NO COUPLER DETECTED")
        self.status_banner.setText(status)
        self.status_banner.setStyleSheet(self._banner_style(STATUS_COLORS.get(status, "#d97000")))
        self.fps_label.setText(f"Display FPS: {data.get('display_fps', 0):.1f}")
        self.inference_fps_label.setText(f"Inference FPS: {data.get('inference_fps', 0):.1f}")
        self.average_fps_label.setText(f"Average FPS: {data.get('average_fps', 0):.1f}")
        self.frame_label.setText(f"Frame: {data.get('frame_number', 0)}")
        self.dropped_label.setText(f"Dropped: {data.get('dropped_frames', 0)}")
        self.processed_label.setText(f"Processed: {data.get('processed_frames', 0)}")
        self.skipped_label.setText(f"Skipped: {data.get('skipped_frames', 0)}")
        self.class_label.setText(f"Class: {data.get('detected_class') or '-'}")
        conf = data.get("confidence", -1)
        self.conf_label.setText(f"Confidence: {conf:.3f}" if conf >= 0 else "Confidence: -")
        classifier_conf = data.get("classifier_confidence", -1)
        classifier_class = data.get("classifier_class") or "-"
        classifier_note = "used" if data.get("classifier_used") else "available" if data.get("classifier_available") else "missing"
        self.classifier_label.setText(
            f"Classifier: {classifier_class} {classifier_conf:.3f} ({classifier_note})"
            if classifier_conf >= 0
            else f"Classifier: {classifier_note}"
        )
        self.device_label.setText(f"Device: {data.get('device', 'CPU')}")
        self.error_label.setText(f"Errors: {data.get('error_count', 0)}")
        if status != self.last_status:
            self._add_event(f"{data.get('frame_number', 0)}: {status}")
            self.last_status = status

    def _show_frame(self, frame) -> None:
        height, width, channels = frame.shape
        if hasattr(QImage, "Format_BGR888"):
            image = QImage(frame.data, width, height, channels * width, QImage.Format_BGR888).copy()
        else:
            import cv2

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )
        self.video_label.setPixmap(pixmap)

    @Slot(str)
    def _show_error(self, message: str) -> None:
        self.error_label.setText(f"Errors: {message}")

    @Slot(str)
    def _add_event(self, message: str) -> None:
        existing = self.event_log.toPlainText()
        lines = existing.splitlines() if existing.strip() else []
        lines.insert(0, message)
        self.event_log.setPlainText("\n".join(lines[:30]))

    @Slot(str)
    def _set_model_status(self, message: str) -> None:
        self.model_status.setText(f"Model: {message}")

    def _update_model_status(self) -> None:
        path = RuntimeSettings().model_path
        self._set_model_status("Ready" if path.exists() else f"Missing {path}")

    def _reset_final_outcome(self) -> None:
        self.final_outcome_label.clear()
        self.final_outcome_label.hide()

    @Slot(dict)
    def _handle_final_summary(self, summary: dict) -> None:
        if not self._current_source_is_video:
            self.final_outcome_label.hide()
            return

        outcome = str(summary.get("outcome", "UNCLEAR - MANUAL CHECK REQUIRED"))
        connected = int(summary.get("connected", 0))
        disconnected = int(summary.get("disconnected", 0))
        unclear = int(summary.get("unclear", 0))
        no_detection = int(summary.get("no_detection", 0))
        clear_total = int(summary.get("clear_total", 0))
        confidence = float(summary.get("confidence", 0.0))

        color = STATUS_COLORS.get(outcome, "#c99700")
        self.final_outcome_label.setStyleSheet(
            f"background:{color};color:white;font-size:18px;font-weight:800;border:1px solid #303840;padding:8px;"
        )
        if clear_total:
            confidence_text = f"{confidence * 100:.1f}% of clear frames"
        else:
            confidence_text = "no clear connected/disconnected frames"
        self.final_outcome_label.setText(
            f"FINAL VIDEO RESULT: {outcome}\n"
            f"Connected: {connected} | Disconnected: {disconnected} | Unclear: {unclear} | No detection: {no_detection}\n"
            f"Decision basis: {confidence_text}"
        )
        self.final_outcome_label.show()

    def _disconnect_stop_signal(self, signal, slot) -> None:
        try:
            signal.disconnect(slot)
        except (RuntimeError, TypeError):
            pass

    def _set_running(self, running: bool) -> None:
        self.start_webcam_button.setEnabled(not running)
        self.select_video_button.setEnabled(not running)
        self.run_video_button.setEnabled((not running) and self.selected_video is not None)
        self.stop_button.setEnabled(running)

    def _banner_style(self, color: str) -> str:
        return f"background:{color};color:white;font-size:24px;font-weight:700;border-radius:4px;"

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._show_desired_condition_image()

    def closeEvent(self, event) -> None:
        self._stopping = True
        self._ui_timer.stop()
        self._pending_frame = None
        self._pending_data = None
        self._request_stop(self.capture_worker)
        self._request_stop(self.inference_worker)
        if self._object_alive(self.logging_worker):
            self.stop_logging_requested.emit()
        self._wait_thread(self.capture_thread)
        self._wait_thread(self.inference_thread)
        self._wait_thread(self.logging_thread)
        self.capture_thread = None
        self.capture_worker = None
        self.inference_thread = None
        self.inference_worker = None
        self.logging_thread = None
        self.logging_worker = None
        self.frame_queue = None
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
