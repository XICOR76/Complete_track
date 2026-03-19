"""
evaluate.py — MOT Approach Comparison & Evaluation Script
==========================================================
Runs all 3 approaches on the same video and computes:
  • FPS (speed)
  • Total unique IDs (ID count)
  • ID switches (lower = better)
  • Track fragmentation (lower = better)
  • Average track length (higher = better)
  • Detection confidence average
  • Longest track duration

Then prints a final comparison table + saves a visual report.

Usage:
    python evaluate.py --source train_video.mp4
    python evaluate.py --source train_video.mp4 --approaches 1 2 3
    python evaluate.py --source train_video.mp4 --max_frames 300
"""

import argparse
import time
import cv2
import numpy as np
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict
from collections import defaultdict

# ── Add approach folders to path ─────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'Complete_track1'))
sys.path.insert(0, str(ROOT / 'Complete_track2'))
sys.path.insert(0, str(ROOT / 'Complete_track3'))


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ApproachMetrics:
    approach_name : str
    fps_list      : List[float] = field(default_factory=list)
    track_ids_per_frame: List[set] = field(default_factory=list)
    all_track_ids : set = field(default_factory=set)

    # Per-track data: {track_id: list of frame_ids seen}
    track_frames  : Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list))

    # ID switch detection: {track_id: last_seen_frame}
    last_seen     : Dict[int, int] = field(default_factory=dict)
    id_switches   : int = 0
    total_frames  : int = 0

    def record_frame(self, frame_id: int, active_tracks: list, elapsed: float):
        """Record one frame's tracking results."""
        self.total_frames += 1
        self.fps_list.append(1.0 / elapsed if elapsed > 0 else 0)

        current_ids = set()
        for t in active_tracks:
            tid = t['track_id']
            current_ids.add(tid)
            self.all_track_ids.add(tid)
            self.track_frames[tid].append(frame_id)

        self.track_ids_per_frame.append(current_ids)

        # Detect ID switches: track that disappeared and came back with same ID
        # is fine, but tracks that vanish briefly are fragmentation
        for tid in current_ids:
            if tid in self.last_seen:
                gap = frame_id - self.last_seen[tid] - 1
                if 1 <= gap <= 5:   # reappeared after short gap = possible ID switch
                    self.id_switches += 1
            self.last_seen[tid] = frame_id

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def avg_fps(self) -> float:
        return float(np.mean(self.fps_list)) if self.fps_list else 0.0

    @property
    def min_fps(self) -> float:
        return float(np.min(self.fps_list)) if self.fps_list else 0.0

    @property
    def total_unique_ids(self) -> int:
        return len(self.all_track_ids)

    @property
    def avg_tracks_per_frame(self) -> float:
        if not self.track_ids_per_frame:
            return 0.0
        return float(np.mean([len(s) for s in self.track_ids_per_frame]))

    @property
    def track_lengths(self) -> List[int]:
        return [len(frames) for frames in self.track_frames.values()]

    @property
    def avg_track_length(self) -> float:
        tl = self.track_lengths
        return float(np.mean(tl)) if tl else 0.0

    @property
    def longest_track(self) -> int:
        tl = self.track_lengths
        return max(tl) if tl else 0

    @property
    def fragmentation(self) -> int:
        """
        Count tracks shorter than 5 frames — likely fragmented detections.
        Lower is better.
        """
        return sum(1 for l in self.track_lengths if l < 5)

    @property
    def track_stability(self) -> float:
        """
        Ratio of long tracks (>10 frames) to total tracks.
        Higher = more stable. Range [0, 1].
        """
        if not self.track_lengths:
            return 0.0
        long = sum(1 for l in self.track_lengths if l > 10)
        return long / len(self.track_lengths)

    def summary(self) -> dict:
        return {
            'Approach'            : self.approach_name,
            'Avg FPS'             : round(self.avg_fps, 2),
            'Min FPS'             : round(self.min_fps, 2),
            'Total Unique IDs'    : self.total_unique_ids,
            'Avg Tracks/Frame'    : round(self.avg_tracks_per_frame, 2),
            'Avg Track Length'    : round(self.avg_track_length, 2),
            'Longest Track'       : self.longest_track,
            'ID Switches'         : self.id_switches,
            'Fragmented Tracks'   : self.fragmentation,
            'Track Stability %'   : round(self.track_stability * 100, 1),
            'Total Frames'        : self.total_frames,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Approach Runners
# ─────────────────────────────────────────────────────────────────────────────

def run_approach1(source: str, max_frames: int) -> ApproachMetrics:
    """Run Approach 1 — YOLO + ByteTrack"""
    metrics = ApproachMetrics(approach_name="Approach 1: YOLO + ByteTrack")
    print("\n" + "="*60)
    print("  Running Approach 1 — YOLO + ByteTrack")
    print("="*60)

    try:
        from ultralytics import YOLO

        # Try to find YOLO weights
        weight_paths = [
            ROOT / 'weights' / 'yolo11s.pt',
            ROOT / 'yolo11s.pt',
            'yolov8n.pt'    # fallback auto-download
        ]
        weight = next((str(p) for p in weight_paths if Path(p).exists()), 'yolov8n.pt')
        print(f"  [Approach 1] Using weights: {weight}")

        model = YOLO(weight)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"  [Approach 1] ERROR: Cannot open {source}")
            return metrics

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames > 0 and frame_id >= max_frames):
                break
            frame_id += 1

            t0 = time.perf_counter()
            results = model.track(frame, persist=True, classes=[0], verbose=False)[0]
            elapsed = time.perf_counter() - t0

            tracks = []
            if results.boxes is not None and results.boxes.id is not None:
                for box, tid, conf in zip(
                    results.boxes.xyxy.cpu().numpy(),
                    results.boxes.id.cpu().numpy(),
                    results.boxes.conf.cpu().numpy()
                ):
                    tracks.append({'track_id': int(tid), 'bbox': box.tolist(), 'conf': float(conf)})

            metrics.record_frame(frame_id, tracks, elapsed)

            if frame_id % 50 == 0:
                print(f"  Frame {frame_id:4d} | FPS: {metrics.avg_fps:.1f} | Tracks: {len(tracks)}")

        cap.release()
        print(f"  [Approach 1] Done. {frame_id} frames processed.")

    except Exception as e:
        print(f"  [Approach 1] ERROR: {e}")

    return metrics


def run_approach2(source: str, max_frames: int) -> ApproachMetrics:
    """Run Approach 2 — Faster R-CNN + DeepSORT + ReID"""
    metrics = ApproachMetrics(approach_name="Approach 2: Faster R-CNN + DeepSORT + ReID")
    print("\n" + "="*60)
    print("  Running Approach 2 — Faster R-CNN + DeepSORT + ReID")
    print("="*60)

    try:
        from detector         import FasterRCNNDetector
        from reid_model       import StrongReIDModel, ReIDExtractor
        from deepsort_tracker import DeepSORTTracker

        detector = FasterRCNNDetector(confidence_threshold=0.5)
        reid_model = StrongReIDModel(num_classes=751)
        reid_extractor = ReIDExtractor(model=reid_model)
        tracker = DeepSORTTracker(
            reid_extractor=reid_extractor,
            max_age=30, n_init=3
        )

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"  [Approach 2] ERROR: Cannot open {source}")
            return metrics

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames > 0 and frame_id >= max_frames):
                break
            frame_id += 1

            t0 = time.perf_counter()
            detections = detector.detect(frame)

            if detections:
                bboxes = np.array([d.bbox for d in detections], dtype=np.float32)
                confs  = np.array([d.confidence for d in detections], dtype=np.float32)
            else:
                bboxes = np.zeros((0, 4), dtype=np.float32)
                confs  = np.zeros(0, dtype=np.float32)

            tracks = tracker.update(frame, bboxes, confs)
            elapsed = time.perf_counter() - t0

            metrics.record_frame(frame_id, tracks, elapsed)

            if frame_id % 50 == 0:
                print(f"  Frame {frame_id:4d} | FPS: {metrics.avg_fps:.1f} | Tracks: {len(tracks)}")

        cap.release()
        print(f"  [Approach 2] Done. {frame_id} frames processed.")

    except Exception as e:
        print(f"  [Approach 2] ERROR: {e}")
        import traceback; traceback.print_exc()

    return metrics


def run_approach3(source: str, max_frames: int, motion: str = 'kalman') -> ApproachMetrics:
    """Run Approach 3 — Custom Detector + Kalman/Particle + ReID"""
    name = f"Approach 3: YOLO + {motion.capitalize()} + Custom ReID"
    metrics = ApproachMetrics(approach_name=name)
    print("\n" + "="*60)
    print(f"  Running Approach 3 — YOLO + {motion.capitalize()} Filter + ReID")
    print("="*60)

    try:
        from motion_models import KalmanMotionModel, ParticleFilterModel, ParticleFilterConfig
        from reid_matcher  import ReIDMatcher, DistanceMetric, GalleryStrategy
        from tracker       import ResearchTracker
        from ultralytics   import YOLO

        # Motion model
        if motion == 'kalman':
            motion_model = KalmanMotionModel()
        else:
            motion_model = ParticleFilterModel(ParticleFilterConfig(n_particles=200))

        # ReID
        reid_matcher = ReIDMatcher(
            distance_metric=DistanceMetric.COSINE,
            gallery_strategy=GalleryStrategy.FIFO
        )

        # Tracker
        tracker = ResearchTracker(
            motion_model=motion_model,
            reid_matcher=reid_matcher,
            max_age=60, n_init=3
        )

        # Detector
        weight_paths = [
            ROOT / 'weights' / 'yolo11s.pt',
            ROOT / 'yolo11s.pt',
            'yolov8n.pt'
        ]
        weight = next((str(p) for p in weight_paths if Path(p).exists()), 'yolov8n.pt')
        yolo = YOLO(weight)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"  [Approach 3] ERROR: Cannot open {source}")
            return metrics

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames > 0 and frame_id >= max_frames):
                break
            frame_id += 1

            t0 = time.perf_counter()
            results = yolo(frame, classes=[0], verbose=False)[0]

            if results.boxes is not None and len(results.boxes) > 0:
                bboxes = results.boxes.xyxy.cpu().numpy().astype(np.float32)
                confs  = results.boxes.conf.cpu().numpy().astype(np.float32)
            else:
                bboxes = np.zeros((0, 4), dtype=np.float32)
                confs  = np.zeros(0, dtype=np.float32)

            tracks = tracker.update(frame, bboxes, confs)
            elapsed = time.perf_counter() - t0

            metrics.record_frame(frame_id, tracks, elapsed)

            if frame_id % 50 == 0:
                print(f"  Frame {frame_id:4d} | FPS: {metrics.avg_fps:.1f} | Tracks: {len(tracks)}")

        cap.release()
        print(f"  [Approach 3] Done. {frame_id} frames processed.")

    except Exception as e:
        print(f"  [Approach 3] ERROR: {e}")
        import traceback; traceback.print_exc()

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Report Generator
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(results: List[ApproachMetrics]):
    """Print a formatted comparison table in terminal."""

    summaries = [r.summary() for r in results]
    metrics_keys = [
        'Avg FPS', 'Min FPS', 'Total Unique IDs', 'Avg Tracks/Frame',
        'Avg Track Length', 'Longest Track', 'ID Switches',
        'Fragmented Tracks', 'Track Stability %', 'Total Frames'
    ]

    # Column widths
    col_w = 28
    name_w = 42

    print("\n")
    print("=" * (name_w + col_w * len(summaries) + 4))
    print("  MOT APPROACH COMPARISON REPORT")
    print("=" * (name_w + col_w * len(summaries) + 4))

    # Header
    header = f"{'Metric':<{name_w}}"
    for s in summaries:
        header += f"{s['Approach'][:col_w-2]:<{col_w}}"
    print(header)
    print("-" * (name_w + col_w * len(summaries) + 4))

    # Rows
    for key in metrics_keys:
        row = f"{key:<{name_w}}"
        values = [s[key] for s in summaries]

        for i, val in enumerate(values):
            # Highlight best value
            if key in ('Avg FPS', 'Min FPS', 'Avg Track Length',
                       'Longest Track', 'Track Stability %'):
                best = max(values)
                marker = " ✓" if val == best else "  "
            elif key in ('ID Switches', 'Fragmented Tracks', 'Total Unique IDs'):
                best = min(values)
                marker = " ✓" if val == best else "  "
            else:
                marker = "  "
            row += f"{str(val) + marker:<{col_w}}"
        print(row)

    print("=" * (name_w + col_w * len(summaries) + 4))
    print("  ✓ = Best value for that metric")
    print("=" * (name_w + col_w * len(summaries) + 4))

    # ── Final Verdict ─────────────────────────────────────────────────────────
    print("\n  FINAL VERDICT")
    print("-" * 60)

    scores = defaultdict(int)
    for key in metrics_keys:
        values = [s[key] for s in summaries]
        if key in ('Avg FPS', 'Min FPS', 'Avg Track Length',
                   'Longest Track', 'Track Stability %'):
            best_idx = values.index(max(values))
        elif key in ('ID Switches', 'Fragmented Tracks'):
            best_idx = values.index(min(values))
        else:
            continue
        scores[summaries[best_idx]['Approach']] += 1

    for approach, score in sorted(scores.items(), key=lambda x: -x[1]):
        bar = "█" * score
        print(f"  {approach[:38]:<38} {bar} ({score} wins)")

    best = max(scores, key=scores.get)
    print(f"\n  🏆 Best Overall: {best}")
    print("=" * 60)


def save_visual_report(results: List[ApproachMetrics], output_path: str = "outputs/evaluation_report.png"):
    """Save a visual bar chart comparison image."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        summaries = [r.summary() for r in results]
        names     = [s['Approach'].replace('Approach ', 'A') for s in summaries]
        # Shorten names for chart
        short_names = []
        for s in summaries:
            n = s['Approach']
            if '1' in n: short_names.append('A1: YOLO\n+ByteTrack')
            elif '2' in n: short_names.append('A2: FasterRCNN\n+DeepSORT')
            elif 'Kalman' in n: short_names.append('A3: Kalman\n+ReID')
            elif 'Particle' in n: short_names.append('A3: Particle\n+ReID')
            else: short_names.append(n[:20])

        metrics_to_plot = [
            ('Avg FPS',           'FPS',            'higher is better', '#4CAF50'),
            ('Avg Track Length',  'Frames',          'higher is better', '#2196F3'),
            ('Track Stability %', '%',               'higher is better', '#9C27B0'),
            ('ID Switches',       'Count',           'lower is better',  '#FF5722'),
            ('Fragmented Tracks', 'Count',           'lower is better',  '#FF9800'),
            ('Total Unique IDs',  'Count',           'lower is better',  '#607D8B'),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.patch.set_facecolor('#1a1a2e')
        fig.suptitle('MOT Approaches — Evaluation Report',
                     fontsize=18, fontweight='bold', color='white', y=0.98)

        colors = ['#4FC3F7', '#81C784', '#FFB74D', '#F06292']

        for ax, (metric, unit, direction, color) in zip(axes.flat, metrics_to_plot):
            values = [s[metric] for s in summaries]
            bars = ax.bar(short_names, values,
                          color=colors[:len(short_names)],
                          edgecolor='white', linewidth=0.5, width=0.5)

            # Highlight best bar
            if 'higher' in direction:
                best_idx = values.index(max(values))
            else:
                best_idx = values.index(min(values))
            bars[best_idx].set_edgecolor('#FFD700')
            bars[best_idx].set_linewidth(2.5)

            # Value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        str(val), ha='center', va='bottom', color='white',
                        fontsize=9, fontweight='bold')

            ax.set_title(f'{metric} ({unit})', color='white', fontsize=11, fontweight='bold')
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='white', labelsize=8)
            ax.spines['bottom'].set_color('#444')
            ax.spines['left'].set_color('#444')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.label.set_color('white')

            # Direction label
            ax.set_xlabel(f'★ {direction}', color='#aaa', fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='#1a1a2e')
        plt.close()
        print(f"\n  📊 Visual report saved → {output_path}")

    except ImportError:
        print("\n  [Report] matplotlib not installed. Skipping visual report.")
        print("  Install with: pip install matplotlib")


def save_csv_report(results: List[ApproachMetrics], output_path: str = "outputs/evaluation_results.csv"):
    """Save results to CSV for further analysis."""
    try:
        import csv
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        summaries = [r.summary() for r in results]
        keys = list(summaries[0].keys())

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summaries)

        print(f"  📄 CSV report saved → {output_path}")
    except Exception as e:
        print(f"  [CSV] Error saving: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MOT Evaluation — Compare all 3 approaches")
    p.add_argument('--source',      default='train_video.mp4', help='Input video path')
    p.add_argument('--approaches',  nargs='+', default=['1','2','3'],
                   choices=['1','2','3'], help='Which approaches to run')
    p.add_argument('--max_frames',  default=0, type=int,
                   help='Max frames to process per approach (0 = full video)')
    p.add_argument('--motion',      default='kalman', choices=['kalman','particle'],
                   help='Motion model for Approach 3')
    p.add_argument('--output_dir',  default='outputs', help='Directory for reports')
    p.add_argument('--no_visual',   action='store_true', help='Skip visual report')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("\n" + "█"*60)
    print("  MOT PROJECT — EVALUATION SCRIPT")
    print(f"  Source  : {args.source}")
    print(f"  Runs    : Approach(es) {', '.join(args.approaches)}")
    print(f"  Frames  : {'All' if args.max_frames == 0 else args.max_frames}")
    print("█"*60)

    results = []

    if '1' in args.approaches:
        results.append(run_approach1(args.source, args.max_frames))

    if '2' in args.approaches:
        results.append(run_approach2(args.source, args.max_frames))

    if '3' in args.approaches:
        results.append(run_approach3(args.source, args.max_frames, args.motion))

    if not results:
        print("No approaches ran. Check errors above.")
        sys.exit(1)

    # ── Print terminal table ──────────────────────────────────────────────────
    print_comparison_table(results)

    # ── Save reports ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.no_visual:
        save_visual_report(
            results,
            output_path=f"{args.output_dir}/evaluation_report.png"
        )

    save_csv_report(
        results,
        output_path=f"{args.output_dir}/evaluation_results.csv"
    )

    print("\n  ✅ Evaluation complete!\n")
