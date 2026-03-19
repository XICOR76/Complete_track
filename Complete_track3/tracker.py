"""
Approach 3 — Custom Research Tracker (Enhanced)
Plug in ANY combination of:
  • Detector     : YOLO / Faster R-CNN
  • Motion model : Kalman / Particle Filter
  • ReID         : Enhanced matcher (color + deep + proportion + zone)
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
    OCCLUDED  = 'occluded'


@dataclass
class ResearchTrack:
    track_id    : int
    motion_state: dict
    n_init      : int = 3
    max_age     : int = 60

    state             : TrackState = field(init=False, default=TrackState.TENTATIVE)
    hits              : int        = field(init=False, default=1)
    age               : int        = field(init=False, default=1)
    time_since_update : int        = field(init=False, default=0)
    trajectory        : List[np.ndarray] = field(init=False, default_factory=list)
    confidence_history: List[float]      = field(init=False, default_factory=list)
    occlusion_count   : int        = field(init=False, default=0)

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
        elif self.state == TrackState.CONFIRMED and self.time_since_update > 30:
            self.state = TrackState.OCCLUDED

    def get_bbox(self, motion_model: MotionModel) -> np.ndarray:
        return motion_model.get_bbox_estimate(self.motion_state)

    def is_confirmed(self): return self.state == TrackState.CONFIRMED
    def is_occluded(self):  return self.state == TrackState.OCCLUDED
    def is_tentative(self): return self.state == TrackState.TENTATIVE
    def is_deleted(self):   return self.state == TrackState.DELETED
    def is_active(self):    return self.state in (TrackState.CONFIRMED, TrackState.OCCLUDED)


# ─────────────────────────────────────────────────────────────────────────────
# Assignment helpers
# ─────────────────────────────────────────────────────────────────────────────

def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-8)


def build_iou_cost(
    tracks: list,
    track_indices: List[int],
    det_bboxes: np.ndarray,
    motion_model: MotionModel
) -> np.ndarray:
    """Pure IoU cost matrix."""
    n_t = len(track_indices)
    n_d = len(det_bboxes)
    cost = np.zeros((n_t, n_d), dtype=np.float32)
    for i, t_idx in enumerate(track_indices):
        pred_box = tracks[t_idx].get_bbox(motion_model)
        for j, det_box in enumerate(det_bboxes):
            cost[i, j] = 1.0 - iou(pred_box, det_box)
    return cost


def build_enhanced_cost(
    tracks: list,
    track_indices: List[int],
    det_bboxes: np.ndarray,
    det_embeddings: Optional[np.ndarray],
    reid_matcher: Optional[ReIDMatcher],
    motion_model: MotionModel,
    iou_weight: float,
    reid_weight: float,
    # Enhanced feature subsets
    det_color_hists: Optional[np.ndarray] = None,
    det_proportions: Optional[np.ndarray] = None,
    frame_shape: Optional[Tuple[int,int]] = None,
    time_since: Optional[Dict[int,int]] = None,
) -> np.ndarray:
    """
    Fused cost = iou_weight * IoU_cost + reid_weight * ReID_cost.
    ReID cost now includes color histogram + proportions + entry zone.
    """
    n_t = len(track_indices)
    n_d = len(det_bboxes)
    cost = np.zeros((n_t, n_d), dtype=np.float32)

    # ── IoU component ─────────────────────────────────────────────────────────
    if iou_weight > 0:
        iou_cost = build_iou_cost(tracks, track_indices, det_bboxes, motion_model)
        cost += iou_weight * iou_cost

    # ── ReID component (enhanced) ─────────────────────────────────────────────
    if reid_weight > 0 and reid_matcher is not None and det_embeddings is not None:
        track_ids = [tracks[t_idx].track_id for t_idx in track_indices]

        if hasattr(reid_matcher, 'compute_distance_matrix'):
            app_cost = reid_matcher.compute_distance_matrix(
                track_ids=track_ids,
                det_embeddings=det_embeddings,
                time_since_update=time_since,
                det_color_hists=det_color_hists,
                det_proportions=det_proportions,
                det_bboxes=det_bboxes,
                frame_shape=frame_shape,
            )
        else:
            app_cost = np.ones((n_t, n_d), dtype=np.float32)

        app_cost_clipped = np.clip(app_cost, 0, 1)
        cost += reid_weight * app_cost_clipped

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
        matched_rows.add(r)
        matched_cols.add(c)

    u_tracks = [i for i in range(cost.shape[0]) if i not in matched_rows]
    u_dets   = [i for i in range(cost.shape[1]) if i not in matched_cols]
    return matches, u_tracks, u_dets


# ─────────────────────────────────────────────────────────────────────────────
# Research Tracker
# ─────────────────────────────────────────────────────────────────────────────

class ResearchTracker:
    """
    Enhanced modular research tracker.
    Now uses color histogram + body proportions + entry zone + mutual exclusivity
    to prevent ID swaps between 2 people going in/out of frame.
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
        long_absence_threshold: int = 10
    ):
        self.motion_model      = motion_model
        self.reid_matcher      = reid_matcher
        self.max_age           = max_age
        self.n_init            = n_init
        self.iou_threshold     = iou_threshold
        self.fusion_threshold  = fusion_threshold
        self.motion_weight     = motion_weight
        self.appearance_weight = appearance_weight
        self.long_absence      = long_absence_threshold

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
        bboxes: np.ndarray,
        confidences: np.ndarray
    ) -> List[dict]:

        fh, fw = frame.shape[:2]

        # ── 1. Extract ALL features ───────────────────────────────────────────
        det_embeddings  = None
        det_color_hists = None
        det_proportions = None

        if self.reid_matcher is not None and len(bboxes) > 0:
            if hasattr(self.reid_matcher, 'extract_all_features'):
                all_feats       = self.reid_matcher.extract_all_features(frame, bboxes)
                det_embeddings  = all_feats['embeddings']
                det_color_hists = all_feats['color_hists']
                det_proportions = all_feats['proportions']
            else:
                det_embeddings = self.reid_matcher.extract_embeddings(frame, bboxes)

        # ── 2. Measurements ───────────────────────────────────────────────────
        measurements = self._to_measurements(bboxes)

        # ── 3. Predict all tracks ─────────────────────────────────────────────
        for track in self.tracks:
            track.predict(self.motion_model)

        # ── 4. Four-stage cascade matching ────────────────────────────────────
        matches, unmatched_tracks, unmatched_dets = self._four_stage_match(
            bboxes, measurements, det_embeddings, confidences,
            det_color_hists=det_color_hists,
            det_proportions=det_proportions,
            frame_shape=(fh, fw)
        )

        # ── 5. Update matched tracks ──────────────────────────────────────────
        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(
                self.motion_model,
                measurements[d_idx],
                float(confidences[d_idx])
            )
            if det_embeddings is not None and self.reid_matcher is not None:
                self.reid_matcher.update_gallery(
                    self.tracks[t_idx].track_id,
                    det_embeddings[d_idx],
                    frame=frame,
                    bbox=bboxes[d_idx]
                )

        # ── 6. Mark missed tracks ─────────────────────────────────────────────
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].mark_missed()

        # ── 7. Create new tracks ──────────────────────────────────────────────
        for d_idx in unmatched_dets:
            self._create_track(
                measurements[d_idx],
                det_embeddings[d_idx] if det_embeddings is not None else None,
                float(confidences[d_idx]),
                frame=frame,
                bbox=bboxes[d_idx]
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
            if (t.is_confirmed() or t.is_occluded()) and t.time_since_update <= 1:
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
        return self.tracks

    # ── Matching logic ────────────────────────────────────────────────────────

    def _four_stage_match(
        self,
        bboxes, measurements, det_embeddings, confidences,
        det_color_hists=None,
        det_proportions=None,
        frame_shape=None,
    ) -> Tuple[List, List, List]:

        # ── Bucket 1: CONFIRMED recently seen ────────────────────────────────
        confirmed_recent = [
            i for i, t in enumerate(self.tracks)
            if t.is_confirmed() and t.time_since_update <= self.long_absence
        ]
        # ── Bucket 2: CONFIRMED long-absent ──────────────────────────────────
        confirmed_old = [
            i for i, t in enumerate(self.tracks)
            if t.is_confirmed() and t.time_since_update > self.long_absence
        ]
        # ── Bucket 3: OCCLUDED (out of frame) ────────────────────────────────
        occluded = [
            i for i, t in enumerate(self.tracks)
            if t.is_occluded()
        ]
        # ── Bucket 4: TENTATIVE (new, no gallery) ────────────────────────────
        tentative = [
            i for i, t in enumerate(self.tracks)
            if t.is_tentative()
        ]

        all_matches: List[Tuple[int, int]] = []
        unmatched_dets = list(range(len(bboxes)))
        all_unmatched_tracks: List[int] = []

        time_since = {
            t.track_id: t.time_since_update for t in self.tracks
        }

        # ── Stage 1: Confirmed recent → ReID(0.6) + IoU(0.4) ─────────────────
        if confirmed_recent and unmatched_dets:
            sub_bboxes = bboxes[unmatched_dets]
            sub_emb    = det_embeddings[unmatched_dets] if det_embeddings is not None else None
            sub_col    = det_color_hists[unmatched_dets] if det_color_hists is not None else None
            sub_prop   = det_proportions[unmatched_dets] if det_proportions is not None else None

            cost = build_enhanced_cost(
                self.tracks, confirmed_recent, sub_bboxes, sub_emb,
                self.reid_matcher, self.motion_model,
                iou_weight=self.motion_weight,
                reid_weight=self.appearance_weight,
                det_color_hists=sub_col,
                det_proportions=sub_prop,
                frame_shape=frame_shape,
                time_since=time_since,
            )
            m, u_t, u_d = hungarian_assign(cost, self.fusion_threshold)

            # Mutual exclusivity check for stage 1
            m, rejected = self._mutual_exclusivity(cost, m)
            u_d = list(set(u_d) | set(rejected))

            for lt, ld in m:
                all_matches.append((confirmed_recent[lt], unmatched_dets[ld]))
            all_unmatched_tracks += [confirmed_recent[i] for i in u_t]
            unmatched_dets = [unmatched_dets[i] for i in u_d]

        # ── Stage 2: Confirmed long-absent → ReID(0.8) + IoU(0.2) ───────────
        if confirmed_old and unmatched_dets:
            sub_bboxes = bboxes[unmatched_dets]
            sub_emb    = det_embeddings[unmatched_dets] if det_embeddings is not None else None
            sub_col    = det_color_hists[unmatched_dets] if det_color_hists is not None else None
            sub_prop   = det_proportions[unmatched_dets] if det_proportions is not None else None

            cost = build_enhanced_cost(
                self.tracks, confirmed_old, sub_bboxes, sub_emb,
                self.reid_matcher, self.motion_model,
                iou_weight=0.2,
                reid_weight=0.8,
                det_color_hists=sub_col,
                det_proportions=sub_prop,
                frame_shape=frame_shape,
                time_since=time_since,
            )
            m, u_t, u_d = hungarian_assign(cost, self.fusion_threshold)
            m, rejected = self._mutual_exclusivity(cost, m)
            u_d = list(set(u_d) | set(rejected))

            for lt, ld in m:
                all_matches.append((confirmed_old[lt], unmatched_dets[ld]))
            all_unmatched_tracks += [confirmed_old[i] for i in u_t]
            unmatched_dets = [unmatched_dets[i] for i in u_d]

        # ── Stage 3: OCCLUDED → pure ReID (1.0), zero IoU ────────────────────
        if occluded and unmatched_dets:
            sub_bboxes = bboxes[unmatched_dets]
            sub_emb    = det_embeddings[unmatched_dets] if det_embeddings is not None else None
            sub_col    = det_color_hists[unmatched_dets] if det_color_hists is not None else None
            sub_prop   = det_proportions[unmatched_dets] if det_proportions is not None else None

            cost = build_enhanced_cost(
                self.tracks, occluded, sub_bboxes, sub_emb,
                self.reid_matcher, self.motion_model,
                iou_weight=0.0,
                reid_weight=1.0,
                det_color_hists=sub_col,
                det_proportions=sub_prop,
                frame_shape=frame_shape,
                time_since=time_since,
            )
            m, u_t, u_d = hungarian_assign(cost, self.fusion_threshold)
            m, rejected = self._mutual_exclusivity(cost, m)
            u_d = list(set(u_d) | set(rejected))

            for lt, ld in m:
                all_matches.append((occluded[lt], unmatched_dets[ld]))
            all_unmatched_tracks += [occluded[i] for i in u_t]
            unmatched_dets = [unmatched_dets[i] for i in u_d]

        # ── Stage 4: Tentative → pure IoU ────────────────────────────────────
        if tentative and unmatched_dets:
            sub_bboxes = bboxes[unmatched_dets]
            cost = build_iou_cost(
                self.tracks, tentative, sub_bboxes, self.motion_model
            )
            m, u_t, u_d = hungarian_assign(cost, self.iou_threshold)
            for lt, ld in m:
                all_matches.append((tentative[lt], unmatched_dets[ld]))
            all_unmatched_tracks += [tentative[i] for i in u_t]
            unmatched_dets = [unmatched_dets[i] for i in u_d]

        return all_matches, all_unmatched_tracks, unmatched_dets

    def _mutual_exclusivity(
        self,
        cost: np.ndarray,
        matches: List[Tuple[int, int]],
        margin: float = 0.08
    ) -> Tuple[List[Tuple[int,int]], List[int]]:
        """
        Reject ambiguous assignments where 2 tracks competed closely.
        Returns (clean_matches, rejected_det_local_indices).
        """
        if cost.size == 0 or len(matches) == 0:
            return matches, []

        n_dets = cost.shape[1]
        ambiguous = set()

        for d in range(n_dets):
            col = cost[:, d]
            valid_costs = sorted([c for c in col if c < 1e4])
            if len(valid_costs) >= 2:
                if (valid_costs[1] - valid_costs[0]) < margin and valid_costs[0] < self.fusion_threshold:
                    ambiguous.add(d)

        clean   = [(t, d) for t, d in matches if d not in ambiguous]
        rejected = list(ambiguous)
        return clean, rejected

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
        confidence: float,
        frame: np.ndarray = None,
        bbox: np.ndarray = None,
    ):
        state = self.motion_model.initiate(measurement)
        track = ResearchTrack(
            track_id     = self._next_id,
            motion_state = state,
            n_init       = self.n_init,
            max_age      = self.max_age
        )
        track.confidence_history.append(confidence)
        if embedding is not None and self.reid_matcher is not None:
            self.reid_matcher.update_gallery(
                self._next_id,
                embedding,
                frame=frame,
                bbox=bbox
            )
        self.tracks.append(track)
        self._next_id += 1
