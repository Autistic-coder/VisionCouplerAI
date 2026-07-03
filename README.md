# VisionCouplerAI

Computer vision system for real-time coupler engagement inspection.

## Overview

VisionCouplerAI is a Python-based computer vision project for inspecting coupler engagement from live camera feeds or selected video sources. The project combines YOLO-based object detection, optional classifier verification, PySide6 desktop controls, frame processing utilities, logging, and training/evaluation scripts used during model development.

The public repository is intentionally code-focused. Private datasets, CCTV/video footage, raw frames, annotations, trained weights, and generated outputs are excluded for privacy and size reasons.

## Features

- Real-time coupler engagement monitoring from webcam, RTSP, or video file sources
- YOLO object detection pipeline with configurable confidence and IOU thresholds
- Optional classifier stage for engagement/disengagement verification
- PySide6 desktop application for operator-facing inspection workflows
- Runtime settings for inference size, processing width, frame skipping, overlays, screenshots, and processed video output
- Training and dataset-preparation utilities for local model iteration
- Logging and health-check helpers for runtime diagnostics
- Privacy-first repository setup that excludes datasets, videos, annotations, model weights, and output artifacts

## Tech Stack

- Python 3.11+
- Ultralytics YOLO
- OpenCV
- PyTorch and TorchVision
- PySide6
- NumPy, pandas, and PyYAML
- PyInstaller for optional desktop packaging

## Folder Structure

```text
VisionCouplerAI/
|-- desktop_app.py                         # Main desktop UI
|-- inspect_coupler.py                     # Inspection logic and video processing
|-- inference_worker.py                    # Model inference worker
|-- camera_worker.py                       # Camera/video capture worker
|-- runtime_config.py                      # Runtime defaults and config loading
|-- logging_worker.py                      # Logging utilities
|-- health_check.py                        # Environment/runtime checks
|-- train_*.py                             # Local training scripts
|-- build_*.py                             # Local dataset/build helpers
|-- evaluate_*.py                          # Local evaluation scripts
|-- *.example.json                         # Sanitized config templates
|-- requirements.txt                       # Python dependencies
|-- CouplerGuardAI.spec                    # PyInstaller spec
|-- RUN_COUPLER_GUARD_AI.bat               # Windows launcher
`-- README.md
```

Excluded local-only folders include `datasets/`, `data/`, `images/`, `videos/`, `frames/`, `annotations/`, `runs/`, `outputs/`, `models/`, `weights/`, `checkpoints/`, `.venv/`, `build/`, and `dist/`.

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Autistic-coder/VisionCouplerAI.git
   cd VisionCouplerAI
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create local runtime configuration files from the templates:

   ```bash
   copy camera_config.example.json camera_config.json
   copy dataset_paths.example.json dataset_paths.json
   ```

5. Place your private model weights locally, for example:

   ```text
   models/best.pt
   models/classifier.pt
   ```

   Model files are ignored by git and are not included in this repository.

## How to Run

Run the desktop application:

```bash
python desktop_app.py
```

Or use the Windows launcher:

```bash
RUN_COUPLER_GUARD_AI.bat
```

Run a runtime health check:

```bash
python health_check.py
```

Training and evaluation scripts are included for local development. They expect private datasets and model files to be present on your machine and should not be run until the local paths in `dataset_paths.json` are configured.

## Dataset Privacy

No datasets, videos, training images, raw frames, annotation exports, CCTV footage, model weights, prediction outputs, or company/internal files are included in this repository. These files are intentionally ignored through `.gitignore` to protect privacy and avoid uploading large binary artifacts.

If you train or evaluate the project locally, keep all datasets and generated files in ignored folders such as `datasets/`, `data/`, `runs/`, `outputs/`, `models/`, or `weights/`.

## Future Improvements

- Add a small synthetic/demo input generator for public testing
- Add unit tests for configuration loading and inference decision logic
- Add structured application logging with rotation
- Add model version metadata and local calibration reports
- Add a deployment guide for production camera environments

## Disclaimer

This repository contains code and configuration templates only. Datasets, private videos, raw/annotated frames, trained model weights, and generated outputs are not included for privacy, security, and repository-size reasons.
