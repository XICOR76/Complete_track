"""
Faster R-CNN  +  DeepSORT  +  Strong ReID

Usage:# Video file
python Complete_track2/main.py --source train_video.mp4 --output outputs/approach2_result.mp4

# With ReID weights
python Complete_track2/main.py --source train_video.mp4 --reid_weights weights/reid_market1501.pth --output outputs/approach2_result.mp4

# Laptop camera
python Complete_track2/main.py --source 0 --output outputs/approach2_laptop.mp4

# External camera
python Complete_track2/main.py --source 1 --output outputs/approach2_external.mp4
    
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path

from detector        import FasterRCNNDetector
from reid_model      import StrongReIDModel, ReIDExtractor
from deepsort_tracker import DeepSORTTracker


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = np.random.default_rng(42).integers(50, 255, size=(1000, 3))


def draw_tracks(frame: np.ndarray, tracks: list) -> np.ndarray:
    vis = frame.copy()
    for t in tracks:
        tid  = t['track_id']
        x1, y1, x2, y2 = map(int, t['bbox'])
        colour = tuple(int(c) for c in PALETTE[tid % len(PALETTE)])

        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
        label = f"ID:{tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def draw_info(frame: np.ndarray, fps: float, n_tracks: int) -> np.ndarray:
    cv2.putText(frame, f"FPS: {fps:.1f}  Tracks: {n_tracks}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class Approach2Pipeline:
    """
    End-to-end Faster R-CNN + DeepSORT + Strong ReID tracking pipeline.
    """

    def __init__(self, args):
        # ── Detector ──────────────────────────────────────────────────────────
        self.detector = FasterRCNNDetector(
            confidence_threshold=args.conf_thresh,
            target_classes=['person']
        )

        # ── ReID Extractor ────────────────────────────────────────────────────
        reid_extractor = None
        if args.reid_weights and Path(args.reid_weights).exists():
            print(f"[Pipeline] Loading ReID weights: {args.reid_weights}")
            reid_extractor = ReIDExtractor.from_pretrained(args.reid_weights)
        else:
            print("[Pipeline] No ReID weights → using ImageNet-pretrained features (weaker ReID).")
            reid_model = StrongReIDModel(num_classes=751)
            reid_extractor = ReIDExtractor(model=reid_model)

        # ── DeepSORT Tracker ──────────────────────────────────────────────────
        self.tracker = DeepSORTTracker(
            reid_extractor          = reid_extractor,
            max_age                 = args.max_age,
            n_init                  = args.n_init,
            max_iou_distance        = args.max_iou_dist,
            max_appearance_distance = args.max_app_dist,
            appearance_weight       = args.app_weight,
            color_weight            = args.color_weight,
            deep_weight             = args.deep_weight,
            zone_weight             = args.zone_weight,
            ambiguity_margin        = args.ambiguity_margin,
            long_absence_threshold  = args.long_absence
        )

        self.args = args

    def run(self):
        # ── Open source ───────────────────────────────────────────────────────
        source = int(self.args.source) if self.args.source.isdigit() else self.args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Pipeline] Source: {source}  {W}×{H} @ {fps_src:.1f} fps")

        # ── Output writer ─────────────────────────────────────────────────────
        writer = None
        if self.args.output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(self.args.output, fourcc, fps_src, (W, H))

        # ── Main loop ─────────────────────────────────────────────────────────
        frame_id = 0
        fps_ema  = 0.0
        alpha    = 0.1     # EMA smoothing

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            t0 = time.perf_counter()

            # 1. Detect
            detections = self.detector.detect(frame)

            if detections:
                bboxes = np.array([d.bbox for d in detections], dtype=np.float32)
                confs  = np.array([d.confidence for d in detections], dtype=np.float32)
            else:
                bboxes = np.zeros((0, 4), dtype=np.float32)
                confs  = np.zeros(0, dtype=np.float32)

            # 2. Track (DeepSORT + ReID)
            tracks = self.tracker.update(frame, bboxes, confs)

            # 3. Visualise
            vis = draw_tracks(frame, tracks)
            elapsed = time.perf_counter() - t0
            fps_ema = alpha / elapsed + (1 - alpha) * fps_ema
            vis = draw_info(vis, fps_ema, len(tracks))

            if writer:
                writer.write(vis)

            if not self.args.headless:
                cv2.imshow("Approach 2: Faster R-CNN + DeepSORT + ReID", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_id % 100 == 0:
                print(f"  Frame {frame_id:5d} | FPS: {fps_ema:.1f} | Active tracks: {len(tracks)}")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"[Pipeline] Done. Processed {frame_id} frames.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Approach 2: Faster R-CNN + DeepSORT + Strong ReID")
    p.add_argument('--source',       default='0',          help='video path or webcam index')
    p.add_argument('--reid_weights', default='',           help='path to ReID .pth weights')
    p.add_argument('--output',       default='',           help='output video path')
    p.add_argument('--conf_thresh',  default=0.5,  type=float)
    p.add_argument('--max_age',          default=60,   type=int)
    p.add_argument('--n_init',           default=3,    type=int)
    p.add_argument('--max_iou_dist',     default=0.7,  type=float)
    p.add_argument('--max_app_dist',     default=0.45, type=float)
    p.add_argument('--app_weight',       default=0.5,  type=float)
    p.add_argument('--color_weight',     default=0.35, type=float,
                   help='Weight for color histogram matching')
    p.add_argument('--deep_weight',      default=0.40, type=float,
                   help='Weight for deep embedding matching')
    p.add_argument('--zone_weight',      default=0.15, type=float,
                   help='Weight for entry zone penalty')
    p.add_argument('--ambiguity_margin', default=0.08, type=float,
                   help='Margin below which assignments are rejected as ambiguous')
    p.add_argument('--long_absence',     default=10,   type=int,
                   help='Frames before switching to ReID-only matching')
    p.add_argument('--headless',         action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    pipeline = Approach2Pipeline(parse_args())
    pipeline.run()
