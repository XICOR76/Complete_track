# Multi-Object Tracking (MOT) Project

Three approaches to Multi-Object Tracking using deep learning.

## Approaches

| Approach | Detector | Tracker | ReID |
|---|---|---|---|
| 1 | YOLOv11 | ByteTrack | — |
| 2 | Faster R-CNN | DeepSORT | Strong ReID (ResNet50) |
| 3 | YOLO / Faster R-CNN | Kalman / Particle Filter | Custom ReID |

## Setup
```bash
pip install -r requirements_v2.txt
```

## Run
```bash
# Approach 1
python Complete_track1/main.py --source train_video.mp4 --output outputs/approach1_result.mp4

# Approach 2
python Complete_track2/main.py --source train_video.mp4 --output outputs/approach2_result.mp4

# Approach 3 — Kalman
python Complete_track3/main.py --source train_video.mp4 --motion kalman --output outputs/approach3_result.mp4

# Approach 3 — Particle Filter
python Complete_track3/main.py --source train_video.mp4 --motion particle --output outputs/approach3_result.mp4
```

## Project Structure
```
Complete_track/
├── Complete_track1/       # YOLO + ByteTrack
├── Complete_track2/       # Faster R-CNN + DeepSORT + ReID
├── Complete_track3/       # Custom Research Tracker
├── dataset/               # Training data
├── weights/               # Model weights (not tracked by git)
├── outputs/               # Result videos (not tracked by git)
└── requirements_v2.txt
```

## Dataset

Dataset is included in this repository under `complete_track.v1i.yolov11/`

Structure:
complete_track.v1i.yolov11/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml