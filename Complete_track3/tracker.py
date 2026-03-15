"""
Approach 3 — Custom Research Tracker
Most flexible and experimental architecture.
Plug in ANY combination of:
  • Detector     : YOLO / Faster R-CNN
  • Motion model : Kalman / Particle Filter
  • ReID         : Custom matcher (cosine / euclidean / Mahalanobis)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from motion_models import MotionModel, KalmanMotionModel, ParticleFilterModel
from reid_matcher  import ReIDMatcher


# ─────────────────────────────────────────────────────────────────────────────
# Track state
# ─────────────────────────────────────────────────────────────────────────────

class TrackState(Enum):
    TENTATIVE = 'tentative'
    CONFIRMED = 'confirmed'
    DELETED   = 'deleted'
    OCCLUDED  = 'occluded'   # extra state for research: tracked but not visible


@dataclass
class ResearchTrack:
    """
    Research track with richer metadata than standard DeepSORT track.
    """
    track_id   : int
    motion_state: dict                          # managed by motion model
    n_init     : int = 3
    max_age    : int = 60

    state      : TrackState = field(init=False, default=TrackState.TENTATIVE)
    hits       : int        = field(init=False, default=1)
    age        : int        = field(init=False, default=1)
    time_since_update: int  = field(init=False, default=0)

    # Research metadata
    trajectory : List[np.ndarray] = field(init=False, default_factory=list)
    confidence_history: List[float] = field(init=False, default_factory=list)
    occlusion_count: int = field(init=False, default=0)

    def predict(self, motion_model: MotionModel):
        self.motion_state = motion_model.predict(self.motion_state)
        self.age += 1
        self.time_since_update += 1

    def update(self, motion_model: MotionModel, measurement: np.ndarray, confidence: float = 1.0):
        self.motion_state = motion_model.update(self.motion_state, measurement)
        bbox = motion_model.get_bbox_estimate(self.motion_state)
        self.trajectory.append(bbox.copy())
        self.confidence_history.append(confidence)
        self.hits += 1
        self.time_since_update = 0

        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.OCCLUDED:
            self.state = TrackState.CONFIRMED

    def mark_missed(self):
        self.occlusion_count += 1
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.time_since_update > self.max_age:
            self.state = TrackState.DELETED
        elif self.state == TrackState.CONFIRMED and self.time_since_update > 3:
            self.state = TrackState.OCCLUDED

    def get_bbox(self, motion_model: MotionModel) -> np.ndarray:
        return motion_model.get_bbox_estimate(self.motion_state)

    def is_confirmed(self): return self.state == TrackState.CONFIRMED
    def is_occluded(self):  return self.state == TrackState.OCCLUDED
    def is_tentative(self): return self.state == TrackState.TENTATIVE
    def is_deleted(self):   return self.state == TrackState.DELETED
    def is_active(self):    return self.state in (TrackState.CONFIRMED, TrackState.OCCLUDED)


# ─────────────────────────────────────────────────────────────────────────────
# Assignment strategies
# ─────────────────────────────────────────────────────────────────────────────

def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-8)


def build_cost_matrix(
    tracks: List[ResearchTrack],
    track_indices: List[int],
    det_bboxes: np.ndarray,
    det_embeddings: Optional[np.ndarray],
    reid_matcher: Optional[ReIDMatcher],
    motion_model: MotionModel,
    motion_weight: float = 0.4,
    appearance_weight: float = 0.6
) -> np.ndarray:
    """
    Fused cost matrix: weighted combination of appearance + IoU.

    The weights are tunable for research experiments:
      - motion_weight=1.0, appearance_weight=0.0 → pure IoU (SORT)
      - motion_weight=0.0, appearance_weight=1.0 → pure ReID
      - motion_weight=0.4, appearance_weight=0.6 → balanced (default)
    """
    n_t = len(track_indices)
    n_d = len(det_bboxes)
    cost = np.zeros((n_t, n_d), dtype=np.float32)

    # ── IoU cost ──────────────────────────────────────────────────────────────
    iou_cost = np.zeros((n_t, n_d))
    for i, t_idx in enumerate(track_indices):
        pred_box = tracks[t_idx].get_bbox(motion_model)
        for j, det_box in enumerate(det_bboxes):
            iou_cost[i, j] = 1.0 - iou(pred_box, det_box)

    cost += motion_weight * iou_cost

    # ── Appearance cost ───────────────────────────────────────────────────────
    if reid_matcher is not None and det_embeddings is not None:
        time_since = {tracks[t_idx].track_id: tracks[t_idx].time_since_update
                      for t_idx in track_indices}
        app_cost = reid_matcher.compute_distance_matrix(
            track_ids=[tracks[t_idx].track_id for t_idx in track_indices],
            det_embeddings=det_embeddings,
            time_since_update=time_since
        )
        # Cap appearance cost to [0,1] range for invalid pairs
        app_cost_clipped = np.clip(app_cost, 0, 1)
        cost += appearance_weight * app_cost_clipped

    return cost


def hungarian_assign(
    cost: np.ndarray,
    threshold: float
) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    row, col = linear_sum_assignment(cost)
    matches, u_tracks, u_dets = [], [], []
    matched_rows, matched_cols = set(), set()

    for r, c in zip(row, col):
        if cost[r, c] > threshold:
            continue
        matches.append((r, c))
        matched_rows.add(r); matched_cols.add(c)

    u_tracks = [i for i in range(cost.shape[0]) if i not in matched_rows]
    u_dets   = [i for i in range(cost.shape[1]) if i not in matched_cols]
    return matches, u_tracks, u_dets


# ─────────────────────────────────────────────────────────────────────────────
# Research Tracker
# ─────────────────────────────────────────────────────────────────────────────

class ResearchTracker:
    """
    Modular research tracker.

    All components are swappable:
      motion_model   → KalmanMotionModel() or ParticleFilterModel()
      reid_matcher   → ReIDMatcher(metric=..., strategy=...) or None
      detector       → pass in bboxes from any upstream detector

    Three-stage matching cascade:
      Stage 1: Appearance + IoU for confirmed tracks (recent)
      Stage 2: IoU-only for confirmed tracks (long-absent)
      Stage 3: IoU-only for tentative tracks
    """

    def __init__(
        self,
        motion_model: MotionModel,
        reid_matcher: Optional[ReIDMatcher] = None,
        max_age: int = 60,
        n_init: int = 3,
        iou_threshold: float = 0.7,
        fusion_threshold: float = 0.6,
        motion_weight: float = 0.4,
        appearance_weight: float = 0.6,
        long_absence_threshold: int = 10   # frames before switching to IoU-only
    ):
        self.motion_model     = motion_model
        self.reid_matcher     = reid_matcher
        self.max_age          = max_age
        self.n_init           = n_init
        self.iou_threshold    = iou_threshold
        self.fusion_threshold = fusion_threshold
        self.motion_weight    = motion_weight
        self.appearance_weight = appearance_weight
        self.long_absence     = long_absence_threshold

        self.tracks: List[ResearchTrack] = []
        self._next_id = 1

        model_name = type(motion_model).__name__
        reid_name  = type(reid_matcher).__name__ if reid_matcher else 'None'
        print(f"[ResearchTracker] Motion={model_name}  ReID={reid_name}  "
              f"max_age={max_age}  n_init={n_init}")

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
        bboxes: np.ndarray,        # (N, 4) [x1,y1,x2,y2]
        confidences: np.ndarray    # (N,)
    ) -> List[dict]:
        """
        Full update cycle per frame.

        Returns active track dicts with rich metadata.
        """
        # ── 1. Extract ReID features ──────────────────────────────────────────
        det_embeddings = None
        if self.reid_matcher is not None and len(bboxes) > 0:
            det_embeddings = self.reid_matcher.extract_embeddings(frame, bboxes)

        # ── 2. Measurements: [cx, cy, w, h] ──────────────────────────────────
        measurements = self._to_measurements(bboxes)

        # ── 3. Predict all tracks ─────────────────────────────────────────────
        for track in self.tracks:
            track.predict(self.motion_model)

        # ── 4. Three-stage cascade matching ───────────────────────────────────
        matches, unmatched_tracks, unmatched_dets = self._three_stage_match(
            bboxes, measurements, det_embeddings, confidences
        )

        # ── 5. Update matched ─────────────────────────────────────────────────
        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(
                self.motion_model,
                measurements[d_idx],
                float(confidences[d_idx])
            )
            if det_embeddings is not None and self.reid_matcher is not None:
                self.reid_matcher.update_gallery(
                    self.tracks[t_idx].track_id,
                    det_embeddings[d_idx]
                )

        # ── 6. Handle unmatched tracks ────────────────────────────────────────
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].mark_missed()

        # ── 7. Create new tracks ──────────────────────────────────────────────
        for d_idx in unmatched_dets:
            self._create_track(
                measurements[d_idx],
                det_embeddings[d_idx] if det_embeddings is not None else None,
                float(confidences[d_idx])
            )

        # ── 8. Remove deleted, clean galleries ───────────────────────────────
        deleted = [t for t in self.tracks if t.is_deleted()]
        if self.reid_matcher:
            for t in deleted:
                self.reid_matcher.remove_track(t.track_id)
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # ── 9. Return results ─────────────────────────────────────────────────
        results = []
        for t in self.tracks:
            if t.is_confirmed() and t.time_since_update <= 1:
                bbox = t.get_bbox(self.motion_model)
                results.append({
                    'track_id'       : t.track_id,
                    'bbox'           : bbox.tolist(),
                    'state'          : t.state.value,
                    'age'            : t.age,
                    'hits'           : t.hits,
                    'occlusion_count': t.occlusion_count,
                    'trajectory_len' : len(t.trajectory)
                })
        return results

    def get_all_tracks(self) -> List[ResearchTrack]:
        """Return raw track objects (for analysis/debugging)."""
        return self.tracks

    # ── Matching logic ────────────────────────────────────────────────────────

    def _three_stage_match(
        self,
        bboxes, measurements, det_embeddings, confidences
    ) -> Tuple[List, List, List]:

        confirmed_recent = [
            i for i, t in enumerate(self.tracks)
            if t.is_confirmed() and t.time_since_update <= self.long_absence
        ]
        confirmed_old = [
            i for i, t in enumerate(self.tracks)
            if (t.is_confirmed() or t.is_occluded())
            and t.time_since_update > self.long_absence
        ]
        tentative = [i for i, t in enumerate(self.tracks) if t.is_tentative()]

        all_matches: List[Tuple[int, int]] = []
        unmatched_dets = list(range(len(bboxes)))
        all_unmatched_tracks: List[int] = []

        # Stage 1 — appearance + IoU, confirmed recent
        if confirmed_recent and unmatched_dets:
            det_subset = bboxes[unmatched_dets]
            emb_subset = det_embeddings[unmatched_dets] if det_embeddings is not None else None
            cost = build_cost_matrix(
                self.tracks, confirmed_recent, det_subset, emb_subset,
                self.reid_matcher, self.motion_model,
                self.motion_weight, self.appearance_weight
            )
            m, u_t, u_d = hungarian_assign(cost, self.fusion_threshold)
            for lt, ld in m:
                all_matches.append((confirmed_recent[lt], unmatched_dets[ld]))
            all_unmatched_tracks += [confirmed_recent[i] for i in u_t]
            unmatched_dets = [unmatched_dets[i] for i in u_d]

        # Stage 2 — IoU only, confirmed but long-absent
        if confirmed_old and unmatched_dets:
            det_subset = bboxes[unmatched_dets]
            cost = build_cost_matrix(
                self.tracks, confirmed_old, det_subset, None,
                None, self.motion_model, 1.0, 0.0
            )
            m, u_t, u_d = hungarian_assign(cost, self.iou_threshold)
            for lt, ld in m:
                all_matches.append((confirmed_old[lt], unmatched_dets[ld]))
            all_unmatched_tracks += [confirmed_old[i] for i in u_t]
            unmatched_dets = [unmatched_dets[i] for i in u_d]

        # Stage 3 — IoU only, tentative tracks
        if tentative and unmatched_dets:
            det_subset = bboxes[unmatched_dets]
            cost = build_cost_matrix(
                self.tracks, tentative, det_subset, None,
                None, self.motion_model, 1.0, 0.0
            )
            m, u_t, u_d = hungarian_assign(cost, self.iou_threshold)
            for lt, ld in m:
                all_matches.append((tentative[lt], unmatched_dets[ld]))
            all_unmatched_tracks += [tentative[i] for i in u_t]
            unmatched_dets = [unmatched_dets[i] for i in u_d]

        return all_matches, all_unmatched_tracks, unmatched_dets

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_measurements(self, bboxes: np.ndarray) -> np.ndarray:
        if len(bboxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        w  = bboxes[:, 2] - bboxes[:, 0]
        h  = bboxes[:, 3] - bboxes[:, 1]
        cx = bboxes[:, 0] + w / 2
        cy = bboxes[:, 1] + h / 2
        return np.stack([cx, cy, w, h], axis=1).astype(np.float32)

    def _create_track(
        self,
        measurement: np.ndarray,
        embedding: Optional[np.ndarray],
        confidence: float
    ):
        state = self.motion_model.initiate(measurement)
        track = ResearchTrack(
            track_id    = self._next_id,
            motion_state = state,
            n_init      = self.n_init,
            max_age     = self.max_age
        )
        track.confidence_history.append(confidence)
        if embedding is not None and self.reid_matcher is not None:
            self.reid_matcher.update_gallery(self._next_id, embedding)
        self.tracks.append(track)
        self._next_id += 1
