"""
Approach 3 — Main Pipeline
Custom Detector + Kalman/Particle Filter + Custom ReID

Fully configurable from CLI. Designed for research experiments.

Usage examples:
# YOLO + Kalman + cosine ReID
python Complete_track3/main.py --source train_video.mp4 --detector yolo --motion kalman --metric cosine --output outputs/a3_kalman_video.mp4
# Laptop camera (built-in)
python Complete_track3/main.py --source 0 --detector yolo --motion kalman --metric cosine --output outputs/a3_kalman_laptop.mp4
# External USB camera
python Complete_track3/main.py --source 1 --detector yolo --motion kalman --metric cosine --output outputs/a3_kalman_external.mp4
    
# YOLO + Particle Filter (experimental)
python Complete_track3/main.py --source train_video.mp4 --detector yolo --motion particle --n_particles 500 --output outputs/a3_particle_video.mp4
# Laptop camera
python Complete_track3/main.py --source 0 --detector yolo --motion particle --n_particles 500 --output outputs/a3_particle_laptop.mp4
# External USB camera
python Complete_track3/main.py --source 1 --detector yolo --motion particle --n_particles 500 --output outputs/a3_particle_external.mp4

# Faster R-CNN + Kalman, no ReID (pure motion baseline)
python Complete_track3/main.py --source train_video.mp4 --detector fasterrcnn --motion kalman --no_reid --output outputs/a3_rcnn_video.mp4
# Laptop camera
python Complete_track3/main.py --source 0 --detector fasterrcnn --motion kalman --no_reid --output outputs/a3_rcnn_laptop.mp4
# External USB camera
python Complete_track3/main.py --source 1 --detector fasterrcnn --motion kalman --no_reid --output outputs/a3_rcnn_external.mp4

# Full research mode with trajectory visualisation
python Complete_track3/main.py --source train_video.mp4 --motion particle --show_trajectory --show_uncertainty --output outputs/a3_research_video.mp4
# Laptop camera
python Complete_track3/main.py --source 0 --motion particle --show_trajectory --show_uncertainty --output outputs/a3_research_laptop.mp4
# External USB camera
python Complete_track3/main.py --source 1 --motion particle --show_trajectory --show_uncertainty --output outputs/a3_research_external.mp4"""

import argparse
import time
import sys
import cv2
import numpy as np
from pathlib import Path

from motion_models import (
    KalmanMotionModel, ParticleFilterModel, ParticleFilterConfig
)
from reid_matcher import ReIDMatcher, DistanceMetric, GalleryStrategy
from tracker import ResearchTracker


# ─────────────────────────────────────────────────────────────────────────────
# Detector factory (supports YOLO or Faster R-CNN)
# ─────────────────────────────────────────────────────────────────────────────

def build_detector(args):
    if args.detector == 'yolo':
        try:
            from ultralytics import YOLO
            model = YOLO(args.yolo_model)
            print(f"[Detector] YOLO loaded: {args.yolo_model}")

            class YOLODetector:
                def __init__(self, m, conf, cls): self.m = m; self.conf = conf; self.cls = cls
                def detect(self, frame):
                    results = self.m(frame, conf=self.conf, classes=self.cls, verbose=False)[0]
                    boxes = results.boxes
                    if boxes is None or len(boxes) == 0:
                        return np.zeros((0, 4)), np.zeros(0)
                    bboxes = boxes.xyxy.cpu().numpy()
                    confs  = boxes.conf.cpu().numpy()
                    return bboxes.astype(np.float32), confs.astype(np.float32)

            return YOLODetector(model, args.conf_thresh, [0])  # class 0 = person

        except ImportError:
            print("[Detector] ultralytics not installed. Falling back to Faster R-CNN.")
            args.detector = 'fasterrcnn'

    if args.detector == 'fasterrcnn':
        sys.path.insert(0, str(Path(__file__).parent.parent / 'Complete_track2'))
        from detector import FasterRCNNDetector

        rcnn = FasterRCNNDetector(confidence_threshold=args.conf_thresh)

        class RCNNWrapper:
            def __init__(self, d): self.d = d
            def detect(self, frame):
                dets = self.d.detect(frame)
                if not dets:
                    return np.zeros((0, 4)), np.zeros(0)
                bboxes = np.array([d.bbox for d in dets], dtype=np.float32)
                confs  = np.array([d.confidence for d in dets], dtype=np.float32)
                return bboxes, confs

        return RCNNWrapper(rcnn)

    raise ValueError(f"Unknown detector: {args.detector}")


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = np.random.default_rng(0).integers(60, 240, (1000, 3))


def draw_tracks(
    frame: np.ndarray,
    tracks: list,
    raw_tracks,
    motion_model,
    show_trajectory: bool = False,
    show_uncertainty: bool = False
) -> np.ndarray:
    vis = frame.copy()

    # ── Trajectories (behind boxes) ───────────────────────────────────────────
    if show_trajectory:
        track_by_id = {t.track_id: t for t in raw_tracks}
        for info in tracks:
            tid = info['track_id']
            t = track_by_id.get(tid)
            if t and len(t.trajectory) > 1:
                colour = tuple(int(c) for c in PALETTE[tid % len(PALETTE)])
                pts = []
                for bbox in t.trajectory[-30:]:  # last 30 frames
                    cx = int((bbox[0] + bbox[2]) / 2)
                    cy = int((bbox[1] + bbox[3]) / 2)
                    pts.append((cx, cy))
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    c = tuple(int(x * alpha) for x in colour)
                    cv2.line(vis, pts[i-1], pts[i], c, 2)

    # ── Bounding boxes ────────────────────────────────────────────────────────
    for info in tracks:
        tid = info['track_id']
        x1, y1, x2, y2 = map(int, info['bbox'])
        colour = tuple(int(c) for c in PALETTE[tid % len(PALETTE)])

        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

        label = f"ID:{tid}"
        if info.get('occlusion_count', 0) > 0:
            label += f" occ:{info['occlusion_count']}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # ── Uncertainty ellipse (Particle filter only) ────────────────────────
        if show_uncertainty and info.get('state') != 'deleted':
            track_by_id = {t.track_id: t for t in raw_tracks}
            t = track_by_id.get(tid)
            if t and t.motion_state.get('type') == 'particle':
                var = motion_model.get_uncertainty(t.motion_state) if hasattr(motion_model, 'get_uncertainty') else None
                if var is not None:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    axes = (max(5, int(np.sqrt(var[0]) * 2)), max(5, int(np.sqrt(var[1]) * 2)))
                    cv2.ellipse(vis, (cx, cy), axes, 0, 0, 360, colour, 1)

    return vis


def draw_hud(
    frame: np.ndarray,
    fps: float,
    n_tracks: int,
    frame_id: int,
    motion_type: str,
    reid_type: str
) -> np.ndarray:
    lines = [
        f"FPS: {fps:.1f}  Tracks: {n_tracks}  Frame: {frame_id}",
        f"Motion: {motion_type}  ReID: {reid_type}"
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 28 + 28 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class Approach3Pipeline:
    """Research-oriented tracking pipeline — fully configurable."""

    def __init__(self, args):
        self.args = args

        # ── Motion model ──────────────────────────────────────────────────────
        if args.motion == 'kalman':
            self.motion_model = KalmanMotionModel(
                process_noise_std=args.process_noise,
                measurement_noise_std=args.meas_noise
            )
            motion_label = 'Kalman'
        else:
            cfg = ParticleFilterConfig(
                n_particles=args.n_particles,
                process_std=args.process_noise * 10,
                measurement_std=args.meas_noise * 20
            )
            self.motion_model = ParticleFilterModel(config=cfg)
            motion_label = f'Particle(N={args.n_particles})'

        # ── ReID matcher ──────────────────────────────────────────────────────
        reid_matcher = None
        reid_label   = 'None'
        if not args.no_reid:
            metric   = DistanceMetric(args.metric)
            strategy = GalleryStrategy(args.gallery)
            reid_matcher = ReIDMatcher(
                distance_metric=metric,
                gallery_strategy=strategy,
                base_threshold=args.reid_thresh,
                gallery_max_size=args.gallery_size
            )
            reid_label = f'{args.metric}/{args.gallery}'

        # ── Tracker ───────────────────────────────────────────────────────────
        self.tracker = ResearchTracker(
            motion_model=self.motion_model,
            reid_matcher=reid_matcher,
            max_age=args.max_age,
            n_init=args.n_init,
            iou_threshold=args.iou_thresh,
            fusion_threshold=args.fusion_thresh,
            motion_weight=args.motion_weight,
            appearance_weight=args.app_weight,
            long_absence_threshold=args.long_absence
        )

        # ── Detector ──────────────────────────────────────────────────────────
        self.detector = build_detector(args)

        self.motion_label = motion_label
        self.reid_label   = reid_label

    def run(self):
        source = int(self.args.source) if self.args.source.isdigit() else self.args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {source}")

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Pipeline] {W}×{H} @ {fps_src:.1f}fps  Motion={self.motion_label}  ReID={self.reid_label}")

        writer = None
        if self.args.output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(self.args.output, fourcc, fps_src, (W, H))

        frame_id = 0
        fps_ema  = 0.0
        alpha    = 0.1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            t0 = time.perf_counter()

            # Detect
            bboxes, confs = self.detector.detect(frame)

            # Track
            tracks = self.tracker.update(frame, bboxes, confs)

            # Visualise
            elapsed = time.perf_counter() - t0
            fps_ema = alpha / elapsed + (1 - alpha) * fps_ema

            vis = draw_tracks(
                frame, tracks, self.tracker.get_all_tracks(), self.motion_model,
                show_trajectory=self.args.show_trajectory,
                show_uncertainty=self.args.show_uncertainty
            )
            vis = draw_hud(vis, fps_ema, len(tracks), frame_id,
                           self.motion_label, self.reid_label)

            if writer:
                writer.write(vis)

            if not self.args.headless:
                cv2.imshow(f"Approach 3: {self.motion_label} + {self.reid_label}", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_id % 100 == 0:
                print(f"  Frame {frame_id:5d} | FPS {fps_ema:.1f} | Tracks: {len(tracks)}")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"[Pipeline] Finished. {frame_id} frames processed.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Approach 3: Custom Detector + Motion + ReID")

    # Source / output
    p.add_argument('--source',     default='0')
    p.add_argument('--output',     default='')
    p.add_argument('--headless',   action='store_true')

    # Detector
    p.add_argument('--detector',   default='yolo',        choices=['yolo', 'fasterrcnn'])
    p.add_argument('--yolo_model', default='yolo11s.pt')
    p.add_argument('--conf_thresh',default=0.4, type=float)

    # Motion model
    p.add_argument('--motion',     default='kalman',      choices=['kalman', 'particle'])
    p.add_argument('--n_particles',default=300, type=int, help='Particle filter N')
    p.add_argument('--process_noise', default=1.0, type=float)
    p.add_argument('--meas_noise',    default=1.0, type=float)

    # ReID
    p.add_argument('--no_reid',    action='store_true')
    p.add_argument('--metric',     default='cosine',
                   choices=['cosine', 'euclidean', 'mahalanobis'])
    p.add_argument('--gallery',    default='fifo',
                   choices=['fifo', 'mean', 'ema', 'cluster'])
    p.add_argument('--reid_thresh',  default=0.4,  type=float)
    p.add_argument('--gallery_size', default=50,   type=int)

    # Tracker
    p.add_argument('--max_age',      default=60,   type=int)
    p.add_argument('--n_init',       default=3,    type=int)
    p.add_argument('--iou_thresh',   default=0.7,  type=float)
    p.add_argument('--fusion_thresh',default=0.6,  type=float)
    p.add_argument('--motion_weight',default=0.4,  type=float)
    p.add_argument('--app_weight',   default=0.6,  type=float)
    p.add_argument('--long_absence', default=10,   type=int)

    # Visualisation
    p.add_argument('--show_trajectory',  action='store_true')
    p.add_argument('--show_uncertainty', action='store_true',
                   help='Show particle uncertainty ellipses')

    return p.parse_args()


if __name__ == '__main__':
    pipeline = Approach3Pipeline(parse_args())
    pipeline.run()
