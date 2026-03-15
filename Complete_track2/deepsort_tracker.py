"""
Approach 2 — DeepSORT Tracker
Integrates:
  • Kalman Filter     → motion prediction
  • Hungarian Algorithm → optimal assignment
  • Appearance Embeddings (from StrongReID) → re-identification after occlusion
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Kalman Filter  (constant-velocity motion model)
# ─────────────────────────────────────────────────────────────────────────────

class KalmanFilter:
    """
    8-dim state: [cx, cy, a, h, vcx, vcy, va, vh]
      cx, cy  = centre x, y
      a       = aspect ratio (w/h)
      h       = height
      v*      = velocities

    4-dim measurement: [cx, cy, a, h]
    """

    def __init__(self):
        dt = 1.0  # time step

        # State transition matrix (constant velocity)
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = dt

        # Measurement matrix
        self.H = np.eye(4, 8)

        # Process noise covariance
        self.Q = np.eye(8)
        self.Q[4:, 4:] *= 0.01   # small velocity uncertainty

        # Measurement noise covariance
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0   # higher uncertainty on aspect & height

        # Initial covariance
        self.P0_diag = np.array([
            2.0, 2.0, 1e-2, 2.0,
            10.0, 10.0, 1e-5, 10.0
        ])

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a new track from measurement [cx, cy, a, h].
        Returns (mean, covariance).
        """
        mean = np.zeros(8)
        mean[:4] = measurement
        covariance = np.diag(self.P0_diag ** 2)
        return mean, covariance

    def predict(
        self,
        mean: np.ndarray,
        covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state."""
        # Scale process noise by current state magnitude
        std_pos = [
            self._std_weight_pos * mean[3],
            self._std_weight_pos * mean[3],
            1e-2,
            self._std_weight_pos * mean[3]
        ]
        std_vel = [
            self._std_weight_vel * mean[3],
            self._std_weight_vel * mean[3],
            1e-5,
            self._std_weight_vel * mean[3]
        ]
        Q = np.diag(np.array(std_pos + std_vel) ** 2)

        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + Q
        return mean, covariance

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Correct prediction with new measurement (Kalman update step)."""
        # Measurement noise scaled to object size
        std = [
            self._std_weight_pos * mean[3],
            self._std_weight_pos * mean[3],
            1e-1,
            self._std_weight_pos * mean[3]
        ]
        R = np.diag(np.array(std) ** 2)

        S = self.H @ covariance @ self.H.T + R       # innovation covariance
        K = covariance @ self.H.T @ np.linalg.inv(S) # Kalman gain

        innovation = measurement - self.H @ mean
        mean = mean + K @ innovation
        covariance = (np.eye(8) - K @ self.H) @ covariance
        return mean, covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray
    ) -> np.ndarray:
        """Mahalanobis distance for gating (chi-squared threshold)."""
        projected_mean = self.H @ mean
        projected_cov  = self.H @ covariance @ self.H.T + np.diag(
            [self._std_weight_pos * mean[3]] * 2 + [1e-1, self._std_weight_pos * mean[3]]
        ) ** 2

        diff = measurements - projected_mean
        chol = np.linalg.cholesky(projected_cov)
        z = np.linalg.solve(chol, diff.T)
        return np.sum(z ** 2, axis=0)   # (N,) Mahalanobis^2

    _std_weight_pos = 1.0 / 20
    _std_weight_vel = 1.0 / 160


# ─────────────────────────────────────────────────────────────────────────────
# Track
# ─────────────────────────────────────────────────────────────────────────────

class TrackState:
    TENTATIVE   = 1   # not yet confirmed (< n_init hits)
    CONFIRMED   = 2   # stable track
    DELETED     = 3   # lost > max_age frames


@dataclass
class Track:
    track_id    : int
    mean        : np.ndarray
    covariance  : np.ndarray
    n_init      : int = 3        # frames required to confirm
    max_age     : int = 30       # frames before deletion
    max_gallery : int = 100      # max stored embeddings for ReID

    state       : int     = field(init=False, default=TrackState.TENTATIVE)
    hits        : int     = field(init=False, default=1)
    age         : int     = field(init=False, default=1)
    time_since_update : int = field(init=False, default=0)

    # Embedding gallery
    features    : deque   = field(init=False)

    def __post_init__(self):
        self.features = deque(maxlen=self.max_gallery)

    # ── State helpers ─────────────────────────────────────────────────────────

    def predict(self, kf: KalmanFilter):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf: KalmanFilter, detection_measure: np.ndarray, feature: Optional[np.ndarray] = None):
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection_measure)
        if feature is not None:
            self.features.append(feature)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self):
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.time_since_update > self.max_age:
            self.state = TrackState.DELETED

    def is_confirmed(self): return self.state == TrackState.CONFIRMED
    def is_deleted(self):   return self.state == TrackState.DELETED
    def is_tentative(self): return self.state == TrackState.TENTATIVE

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def to_tlwh(self) -> np.ndarray:
        """Convert [cx,cy,a,h] → [x1,y1,w,h]"""
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, w, h])

    def to_tlbr(self) -> np.ndarray:
        """Convert to [x1,y1,x2,y2]"""
        x1, y1, w, h = self.to_tlwh()
        return np.array([x1, y1, x1 + w, y1 + h])

    # ── Gallery embedding ─────────────────────────────────────────────────────

    def mean_feature(self) -> Optional[np.ndarray]:
        if not self.features:
            return None
        mat = np.stack(self.features)
        avg = mat.mean(axis=0)
        norm = np.linalg.norm(avg)
        return avg / (norm + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Hungarian Matching
# ─────────────────────────────────────────────────────────────────────────────

def iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - inter + 1e-8
    return inter / union


def hungarian_matching(
    cost_matrix: np.ndarray,
    threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Hungarian algorithm assignment.

    Returns:
        matches          : list of (track_idx, det_idx)
        unmatched_tracks : list of track indices
        unmatched_dets   : list of detection indices
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_dets = [], [], []
    matched_rows, matched_cols = set(), set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > threshold:
            continue
        matches.append((r, c))
        matched_rows.add(r)
        matched_cols.add(c)

    unmatched_tracks = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
    unmatched_dets   = [i for i in range(cost_matrix.shape[1]) if i not in matched_cols]

    return matches, unmatched_tracks, unmatched_dets


# ─────────────────────────────────────────────────────────────────────────────
# DeepSORT Tracker
# ─────────────────────────────────────────────────────────────────────────────

class DeepSORTTracker:
    """
    Full DeepSORT pipeline:
      1. Predict all track positions (Kalman)
      2. Cascade matching (appearance + IoU)
      3. Update matched tracks
      4. Create new tracks for unmatched detections
      5. Delete stale tracks
    """

    # Chi-squared gating threshold (95th percentile, 4-DOF)
    CHI2_THRESHOLD = 9.4877

    def __init__(
        self,
        reid_extractor=None,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_appearance_distance: float = 0.4,
        appearance_weight: float = 0.5
    ):
        self.reid_extractor       = reid_extractor
        self.max_age              = max_age
        self.n_init               = n_init
        self.max_iou_distance     = max_iou_distance
        self.max_appearance_dist  = max_appearance_distance
        self.appearance_weight    = appearance_weight

        self.kf = KalmanFilter()
        self.tracks: List[Track] = []
        self._next_id = 1

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
        bboxes: np.ndarray,         # (N, 4) [x1,y1,x2,y2]
        confidences: np.ndarray     # (N,)
    ) -> List[dict]:
        """
        Main update loop called every frame.

        Returns list of active track dicts:
          {'track_id', 'bbox': [x1,y1,x2,y2], 'confidence'}
        """
        # ── 1. Extract ReID embeddings ────────────────────────────────────────
        features = None
        if self.reid_extractor is not None and len(bboxes) > 0:
            features = self.reid_extractor.extract(frame, bboxes)  # (N, D)

        # ── 2. Predict all tracks ─────────────────────────────────────────────
        for track in self.tracks:
            track.predict(self.kf)

        # ── 3. Convert detections to [cx, cy, a, h] ───────────────────────────
        det_measurements = self._bboxes_to_measurements(bboxes)

        # ── 4. Cascade matching ───────────────────────────────────────────────
        matches, unmatched_tracks, unmatched_dets = self._cascade_match(
            bboxes, det_measurements, features
        )

        # ── 5. Update matched tracks ──────────────────────────────────────────
        for t_idx, d_idx in matches:
            feat = features[d_idx] if features is not None else None
            self.tracks[t_idx].update(self.kf, det_measurements[d_idx], feat)

        # ── 6. Mark unmatched tracks ──────────────────────────────────────────
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].mark_missed()

        # ── 7. Create new tracks ──────────────────────────────────────────────
        for d_idx in unmatched_dets:
            self._initiate_track(
                det_measurements[d_idx],
                features[d_idx] if features is not None else None
            )

        # ── 8. Remove deleted tracks ──────────────────────────────────────────
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # ── 9. Return confirmed tracks ────────────────────────────────────────
        results = []
        for t in self.tracks:
            if t.is_confirmed() and t.time_since_update <= 1:
                results.append({
                    'track_id'  : t.track_id,
                    'bbox'      : t.to_tlbr().tolist(),
                    'age'       : t.age,
                    'hits'      : t.hits
                })
        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cascade_match(self, bboxes, measurements, features):
        """
        DeepSORT cascade matching strategy:
          Pass 1 — appearance + Mahalanobis (confirmed tracks, age 1..max_age)
          Pass 2 — IoU only (remaining unmatched, includes tentative)
        """
        confirmed_idx = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        tentative_idx = [i for i, t in enumerate(self.tracks) if t.is_tentative()]

        unmatched_dets = list(range(len(bboxes)))
        all_matches: List[Tuple[int, int]] = []
        unmatched_tracks: List[int] = []

        # ── Pass 1: Cascade over age levels (appearance + motion) ─────────────
        for age_level in range(1, self.max_age + 1):
            if not unmatched_dets:
                break
            age_tracks = [i for i in confirmed_idx if self.tracks[i].time_since_update == age_level]
            if not age_tracks:
                continue

            # Build cost only over currently unmatched detections (subset)
            sub_bboxes       = bboxes[unmatched_dets]
            sub_measurements = measurements[unmatched_dets]
            sub_features     = features[unmatched_dets] if features is not None else None

            cost = self._cost_appearance_motion(
                age_tracks, sub_bboxes, sub_measurements, sub_features
            )
            matches, u_trk, u_det_local = hungarian_matching(cost, self.max_appearance_dist)

            # u_det_local indexes into sub_bboxes → map back to global det indices
            for local_t, local_d in matches:
                all_matches.append((age_tracks[local_t], unmatched_dets[local_d]))

            unmatched_tracks += [age_tracks[i] for i in u_trk]
            unmatched_dets    = [unmatched_dets[i] for i in u_det_local]

        # ── Pass 2: IoU matching (leftover + tentative tracks) ────────────────
        iou_track_idx = list(set(unmatched_tracks) | set(tentative_idx))
        if iou_track_idx and unmatched_dets:
            # Build cost only over currently unmatched detections (subset)
            sub_bboxes = bboxes[unmatched_dets]
            cost_iou = self._cost_iou(iou_track_idx, sub_bboxes)
            matches2, u_trk2, u_det2_local = hungarian_matching(cost_iou, self.max_iou_distance)

            for local_t, local_d in matches2:
                all_matches.append((iou_track_idx[local_t], unmatched_dets[local_d]))

            final_unmatched_tracks = [iou_track_idx[i] for i in u_trk2]
            final_unmatched_dets   = [unmatched_dets[i] for i in u_det2_local]
        else:
            final_unmatched_tracks = iou_track_idx
            final_unmatched_dets   = unmatched_dets

        return all_matches, final_unmatched_tracks, final_unmatched_dets

    def _cost_appearance_motion(self, track_indices, bboxes, measurements, features):
        """
        Combined cost = appearance_weight * cosine_dist
                      + (1 - appearance_weight) * mahalanobis_gating
        """
        n_tracks = len(track_indices)
        n_dets   = len(bboxes)
        cost = np.full((n_tracks, n_dets), fill_value=1e5, dtype=np.float32)

        for i, t_idx in enumerate(track_indices):
            track = self.tracks[t_idx]

            # Mahalanobis gating
            mah_sq = self.kf.gating_distance(track.mean, track.covariance, measurements)
            valid = mah_sq < self.CHI2_THRESHOLD

            # Appearance cost
            if features is not None and track.mean_feature() is not None:
                gallery = track.mean_feature()[None, :]       # (1, D)
                app_dist = cdist(gallery, features, metric='cosine')[0]  # (N,)
            else:
                app_dist = np.zeros(n_dets)

            # IoU cost (complement)
            track_box = track.to_tlbr()
            iou_dist = np.array([1 - iou(track_box, bboxes[j]) for j in range(n_dets)])

            combined = (self.appearance_weight * app_dist
                       + (1 - self.appearance_weight) * iou_dist)
            combined[~valid] = 1e5
            cost[i] = combined

        return cost

    def _cost_iou(self, track_indices, bboxes):
        """Pure IoU cost matrix."""
        cost = np.zeros((len(track_indices), len(bboxes)), dtype=np.float32)
        for i, t_idx in enumerate(track_indices):
            track_box = self.tracks[t_idx].to_tlbr()
            for j, box in enumerate(bboxes):
                cost[i, j] = 1 - iou(track_box, box)
        return cost

    def _bboxes_to_measurements(self, bboxes: np.ndarray) -> np.ndarray:
        """[x1,y1,x2,y2] → [cx, cy, aspect, height]"""
        if len(bboxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        cx = bboxes[:, 0] + w / 2
        cy = bboxes[:, 1] + h / 2
        a  = w / (h + 1e-8)
        return np.stack([cx, cy, a, h], axis=1).astype(np.float32)

    def _initiate_track(self, measurement: np.ndarray, feature: Optional[np.ndarray]):
        mean, cov = self.kf.initiate(measurement)
        track = Track(
            track_id   = self._next_id,
            mean       = mean,
            covariance = cov,
            n_init     = self.n_init,
            max_age    = self.max_age
        )
        if feature is not None:
            track.features.append(feature)
        self.tracks.append(track)
        self._next_id += 1