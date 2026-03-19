"""
Approach 2 — DeepSORT Tracker (Enhanced)
Integrates:
  • Kalman Filter          → motion prediction
  • Hungarian Algorithm    → optimal assignment
  • Strong ReID embedding  → appearance matching
  • Color Histogram        → clothing color matching (NEW)
  • OCCLUDED state         → keeps track alive after person leaves frame (NEW)
  • Mutual Exclusivity     → prevents wrong ID when 2 people look similar (NEW)
  • 3-pass cascade         → recent / long-absent+occluded / tentative (NEW)
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
        dt = 1.0

        # State transition matrix (constant velocity)
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = dt

        # Measurement matrix
        self.H = np.eye(4, 8)

        # Process noise covariance
        self.Q = np.eye(8)
        self.Q[4:, 4:] *= 0.01

        # Measurement noise covariance
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0

        # Initial covariance
        self.P0_diag = np.array([
            2.0, 2.0, 1e-2, 2.0,
            10.0, 10.0, 1e-5, 10.0
        ])

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.zeros(8)
        mean[:4] = measurement
        covariance = np.diag(self.P0_diag ** 2)
        return mean, covariance

    def predict(self, mean, covariance):
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

    def update(self, mean, covariance, measurement):
        std = [
            self._std_weight_pos * mean[3],
            self._std_weight_pos * mean[3],
            1e-1,
            self._std_weight_pos * mean[3]
        ]
        R = np.diag(np.array(std) ** 2)
        S = self.H @ covariance @ self.H.T + R
        K = covariance @ self.H.T @ np.linalg.inv(S)
        innovation = measurement - self.H @ mean
        mean = mean + K @ innovation
        covariance = (np.eye(8) - K @ self.H) @ covariance
        return mean, covariance

    def gating_distance(self, mean, covariance, measurements):
        projected_mean = self.H @ mean
        projected_cov  = self.H @ covariance @ self.H.T + np.diag(
            [self._std_weight_pos * mean[3]] * 2 + [1e-1, self._std_weight_pos * mean[3]]
        ) ** 2
        diff = measurements - projected_mean
        chol = np.linalg.cholesky(projected_cov)
        z = np.linalg.solve(chol, diff.T)
        return np.sum(z ** 2, axis=0)

    _std_weight_pos = 1.0 / 20
    _std_weight_vel = 1.0 / 160


# ─────────────────────────────────────────────────────────────────────────────
# Color Histogram Extractor
# ─────────────────────────────────────────────────────────────────────────────

class ColorHistogramExtractor:
    """
    Extracts HSV color histograms from upper and lower body separately.
    Used to distinguish people with different clothing colors.

    Upper body (shirt) + lower body (pants/skirt) histograms
    are concatenated into one 512-dim feature vector.

    Why HSV?  Separates color (Hue) from brightness (Value),
    making it robust to lighting changes.
    """

    def __init__(self, h_bins: int = 16, s_bins: int = 4, v_bins: int = 4):
        self.h_bins   = h_bins
        self.s_bins   = s_bins
        self.v_bins   = v_bins
        self.hist_size = h_bins * s_bins * v_bins   # 256 per zone
        self.feat_dim  = self.hist_size * 2          # 512 total

    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract combined upper+lower color histogram from one bbox."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10:
            return np.zeros(self.feat_dim, dtype=np.float32)

        mid   = crop.shape[0] // 2
        upper = self._hsv_hist(crop[:mid])
        lower = self._hsv_hist(crop[mid:])

        combined = np.concatenate([upper, lower])
        norm = np.linalg.norm(combined)
        return (combined / (norm + 1e-8)).astype(np.float32)

    def extract_batch(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        if len(bboxes) == 0:
            return np.zeros((0, self.feat_dim), dtype=np.float32)
        return np.stack([self.extract(frame, box) for box in bboxes])

    def distance(self, hist_a: np.ndarray, hist_b: np.ndarray) -> float:
        """Bhattacharyya distance — robust for histogram comparison."""
        similarity = np.sum(np.sqrt(np.abs(hist_a * hist_b)))
        return float(1.0 - np.clip(similarity, 0, 1))

    def _hsv_hist(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr.size == 0:
            return np.zeros(self.hist_size, dtype=np.float32)
        hsv  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            [self.h_bins, self.s_bins, self.v_bins],
            [0, 180, 0, 256, 0, 256]
        )
        hist = hist.flatten().astype(np.float32)
        return hist / (hist.sum() + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Zone
# ─────────────────────────────────────────────────────────────────────────────

def get_exit_zone(bbox: np.ndarray, frame_w: int, frame_h: int, margin: float = 0.15) -> str:
    """Remember which edge a person was last seen near."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    if cx < frame_w * margin:          return 'left'
    elif cx > frame_w * (1 - margin):  return 'right'
    elif cy < frame_h * margin:        return 'top'
    elif cy > frame_h * (1 - margin):  return 'bottom'
    else:                               return 'center'


def zone_penalty(zone_a: str, zone_b: str) -> float:
    """Penalty when re-entry zone doesn't match exit zone."""
    if zone_a == 'unknown' or zone_b == 'unknown':
        return 0.0
    if zone_a == zone_b:
        return 0.0
    opposites = {'left':'right','right':'left','top':'bottom','bottom':'top'}
    if opposites.get(zone_a) == zone_b:
        return 1.0
    return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Track State
# ─────────────────────────────────────────────────────────────────────────────

class TrackState:
    TENTATIVE = 1   # new, not confirmed yet
    CONFIRMED = 2   # stable track
    DELETED   = 3   # lost too long → remove
    OCCLUDED  = 4   # out of frame → keep alive for ReID re-entry


# ─────────────────────────────────────────────────────────────────────────────
# Track
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Track:
    track_id    : int
    mean        : np.ndarray
    covariance  : np.ndarray
    n_init      : int = 3
    max_age     : int = 60       # increased from 30 → 60
    max_gallery : int = 100

    state             : int  = field(init=False, default=TrackState.TENTATIVE)
    hits              : int  = field(init=False, default=1)
    age               : int  = field(init=False, default=1)
    time_since_update : int  = field(init=False, default=0)

    # Deep embedding gallery
    features     : deque = field(init=False)
    # Color histogram gallery (NEW)
    color_hists  : deque = field(init=False)
    # Exit zone memory (NEW)
    exit_zone    : str   = field(init=False, default='unknown')
    frame_w      : int   = field(init=False, default=0)
    frame_h      : int   = field(init=False, default=0)

    def __post_init__(self):
        self.features    = deque(maxlen=self.max_gallery)
        self.color_hists = deque(maxlen=self.max_gallery)

    # ── State helpers ─────────────────────────────────────────────────────────

    def predict(self, kf: KalmanFilter):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(
        self,
        kf: KalmanFilter,
        detection_measure: np.ndarray,
        feature: Optional[np.ndarray] = None,
        color_hist: Optional[np.ndarray] = None,
        bbox: Optional[np.ndarray] = None,
        frame_w: int = 0,
        frame_h: int = 0
    ):
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection_measure)

        if feature is not None:
            self.features.append(feature)
        if color_hist is not None:
            self.color_hists.append(color_hist)

        # Update exit zone from last known position
        if bbox is not None and frame_w > 0:
            self.exit_zone = get_exit_zone(bbox, frame_w, frame_h)
            self.frame_w = frame_w
            self.frame_h = frame_h

        self.hits += 1
        self.time_since_update = 0

        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.OCCLUDED:
            self.state = TrackState.CONFIRMED   # restored on re-entry

    def mark_missed(self):
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.time_since_update > self.max_age:
            self.state = TrackState.DELETED
        # Switch to OCCLUDED after 30 frames missing
        # keeps track alive so ReID can re-match on re-entry
        elif self.state == TrackState.CONFIRMED and self.time_since_update > 30:
            self.state = TrackState.OCCLUDED

    def is_confirmed(self): return self.state == TrackState.CONFIRMED
    def is_deleted(self):   return self.state == TrackState.DELETED
    def is_tentative(self): return self.state == TrackState.TENTATIVE
    def is_occluded(self):  return self.state == TrackState.OCCLUDED

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def to_tlwh(self) -> np.ndarray:
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, w, h])

    def to_tlbr(self) -> np.ndarray:
        x1, y1, w, h = self.to_tlwh()
        return np.array([x1, y1, x1 + w, y1 + h])

    # ── Gallery helpers ───────────────────────────────────────────────────────

    def mean_feature(self) -> Optional[np.ndarray]:
        """Mean of all stored deep embeddings."""
        if not self.features:
            return None
        mat = np.stack(self.features)
        avg = mat.mean(axis=0)
        return avg / (np.linalg.norm(avg) + 1e-8)

    def mean_color_hist(self) -> Optional[np.ndarray]:
        """Mean of all stored color histograms."""
        if not self.color_hists:
            return None
        mat = np.stack(self.color_hists)
        avg = mat.mean(axis=0)
        return avg / (np.linalg.norm(avg) + 1e-8)

    def vote_embedding(self, query_emb: np.ndarray) -> float:
        """
        Gallery voting: fraction of stored embeddings that agree
        with the query (cosine distance < 0.5).
        Higher score = stronger match confidence.
        """
        if not self.features:
            return 0.0
        mat   = np.stack(self.features)
        dists = 1.0 - mat @ query_emb
        return float(np.sum(dists < 0.5)) / len(self.features)


# ─────────────────────────────────────────────────────────────────────────────
# IoU helper
# ─────────────────────────────────────────────────────────────────────────────

def iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    return inter / (area_a + area_b - inter + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Hungarian Matching
# ─────────────────────────────────────────────────────────────────────────────

def hungarian_matching(
    cost_matrix: np.ndarray,
    threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
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
# DeepSORT Tracker (Enhanced)
# ─────────────────────────────────────────────────────────────────────────────

class DeepSORTTracker:
    """
    Enhanced DeepSORT with:
      • 3-pass cascade matching
      • Color histogram matching
      • OCCLUDED state for long-absent tracks
      • Gallery voting for confident matching
      • Entry zone memory + penalty
      • Mutual exclusivity check to prevent wrong assignment
    """

    CHI2_THRESHOLD = 9.4877   # 95th percentile, 4-DOF chi-squared

    def __init__(
        self,
        reid_extractor=None,
        max_age: int = 60,          # increased from 30
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_appearance_distance: float = 0.45,
        appearance_weight: float = 0.5,
        # Enhanced params
        color_weight: float = 0.35,
        deep_weight: float = 0.40,
        zone_weight: float = 0.15,
        prop_weight: float = 0.10,
        ambiguity_margin: float = 0.08,
        long_absence_threshold: int = 10,
    ):
        self.reid_extractor      = reid_extractor
        self.max_age             = max_age
        self.n_init              = n_init
        self.max_iou_distance    = max_iou_distance
        self.max_appearance_dist = max_appearance_distance
        self.appearance_weight   = appearance_weight
        self.color_weight        = color_weight
        self.deep_weight         = deep_weight
        self.zone_weight         = zone_weight
        self.ambiguity_margin    = ambiguity_margin
        self.long_absence        = long_absence_threshold

        self.kf              = KalmanFilter()
        self.color_extractor = ColorHistogramExtractor()
        self.tracks: List[Track] = []
        self._next_id = 1

        print(f"[DeepSORT Enhanced] max_age={max_age}  "
              f"deep:{deep_weight} color:{color_weight} zone:{zone_weight}")

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
        bboxes: np.ndarray,
        confidences: np.ndarray
    ) -> List[dict]:

        fh, fw = frame.shape[:2]

        # ── 1. Extract deep embeddings + color histograms ─────────────────────
        features     = None
        color_hists  = None

        if self.reid_extractor is not None and len(bboxes) > 0:
            features    = self.reid_extractor.extract(frame, bboxes)
            color_hists = self.color_extractor.extract_batch(frame, bboxes)

        # ── 2. Predict all tracks ─────────────────────────────────────────────
        for track in self.tracks:
            track.predict(self.kf)

        # ── 3. Convert detections → [cx, cy, a, h] ───────────────────────────
        det_measurements = self._bboxes_to_measurements(bboxes)

        # ── 4. Three-pass cascade matching ────────────────────────────────────
        matches, unmatched_tracks, unmatched_dets = self._cascade_match(
            bboxes, det_measurements, features, color_hists, fw, fh
        )

        # ── 5. Update matched tracks ──────────────────────────────────────────
        for t_idx, d_idx in matches:
            feat  = features[d_idx]    if features    is not None else None
            chist = color_hists[d_idx] if color_hists is not None else None
            self.tracks[t_idx].update(
                self.kf,
                det_measurements[d_idx],
                feature=feat,
                color_hist=chist,
                bbox=bboxes[d_idx],
                frame_w=fw,
                frame_h=fh
            )

        # ── 6. Mark unmatched tracks ──────────────────────────────────────────
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].mark_missed()

        # ── 7. Create new tracks ──────────────────────────────────────────────
        for d_idx in unmatched_dets:
            feat  = features[d_idx]    if features    is not None else None
            chist = color_hists[d_idx] if color_hists is not None else None
            self._initiate_track(
                det_measurements[d_idx],
                feat,
                chist,
                bbox=bboxes[d_idx],
                frame_w=fw,
                frame_h=fh
            )

        # ── 8. Remove deleted tracks ──────────────────────────────────────────
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # ── 9. Return confirmed + just-re-entered occluded tracks ─────────────
        results = []
        for t in self.tracks:
            if (t.is_confirmed() or t.is_occluded()) and t.time_since_update <= 1:
                results.append({
                    'track_id' : t.track_id,
                    'bbox'     : t.to_tlbr().tolist(),
                    'age'      : t.age,
                    'hits'     : t.hits,
                    'state'    : 'confirmed' if t.is_confirmed() else 'occluded'
                })
        return results

    # ── Cascade matching ──────────────────────────────────────────────────────

    def _cascade_match(self, bboxes, measurements, features, color_hists, fw, fh):
        """
        3-pass cascade:
          Pass 1 — CONFIRMED recent (age ≤ long_absence):
                   deep embedding + color histogram + Mahalanobis gating
          Pass 2 — CONFIRMED old + OCCLUDED (age > long_absence):
                   pure ReID (deep + color + zone) — no IoU, position drifted
          Pass 3 — TENTATIVE:
                   pure IoU only (no gallery built yet)
        """
        confirmed_recent = [
            i for i, t in enumerate(self.tracks)
            if t.is_confirmed() and t.time_since_update <= self.long_absence
        ]
        confirmed_old = [
            i for i, t in enumerate(self.tracks)
            if t.is_confirmed() and t.time_since_update > self.long_absence
        ]
        occluded_idx = [
            i for i, t in enumerate(self.tracks)
            if t.is_occluded()
        ]
        tentative_idx = [
            i for i, t in enumerate(self.tracks)
            if t.is_tentative()
        ]

        unmatched_dets   = list(range(len(bboxes)))
        all_matches      = []
        unmatched_tracks = []

        # ── Pass 1: Confirmed recent — full enhanced cost ─────────────────────
        for age_level in range(1, self.long_absence + 1):
            if not unmatched_dets:
                break
            age_tracks = [
                i for i in confirmed_recent
                if self.tracks[i].time_since_update == age_level
            ]
            if not age_tracks:
                continue

            sub_bboxes  = bboxes[unmatched_dets]
            sub_meas    = measurements[unmatched_dets]
            sub_feat    = features[unmatched_dets]    if features    is not None else None
            sub_col     = color_hists[unmatched_dets] if color_hists is not None else None

            cost = self._cost_enhanced(
                age_tracks, sub_bboxes, sub_meas, sub_feat, sub_col, fw, fh
            )
            matches, u_trk, u_det_local = hungarian_matching(cost, self.max_appearance_dist)

            # Mutual exclusivity check
            matches, rejected = self._mutual_exclusivity(cost, matches)
            u_det_local = list(set(u_det_local) | set(rejected))

            for lt, ld in matches:
                all_matches.append((age_tracks[lt], unmatched_dets[ld]))
            unmatched_tracks += [age_tracks[i] for i in u_trk]
            unmatched_dets    = [unmatched_dets[i] for i in u_det_local]

        # ── Pass 2: Confirmed old + OCCLUDED — pure ReID (no IoU) ────────────
        reid_tracks = list(set(confirmed_old) | set(occluded_idx))
        if reid_tracks and unmatched_dets:
            sub_feat = features[unmatched_dets]    if features    is not None else None
            sub_col  = color_hists[unmatched_dets] if color_hists is not None else None
            sub_bbox = bboxes[unmatched_dets]

            cost = self._cost_reid_only(reid_tracks, sub_feat, sub_col, sub_bbox, fw, fh)
            matches2, u_trk2, u_det2_local = hungarian_matching(cost, self.max_appearance_dist)

            matches2, rejected2 = self._mutual_exclusivity(cost, matches2)
            u_det2_local = list(set(u_det2_local) | set(rejected2))

            for lt, ld in matches2:
                all_matches.append((reid_tracks[lt], unmatched_dets[ld]))
            unmatched_tracks += [reid_tracks[i] for i in u_trk2]
            unmatched_dets    = [unmatched_dets[i] for i in u_det2_local]

        # ── Pass 3: Tentative — pure IoU ─────────────────────────────────────
        if tentative_idx and unmatched_dets:
            sub_bboxes = bboxes[unmatched_dets]
            cost_iou   = self._cost_iou(tentative_idx, sub_bboxes)
            matches3, u_trk3, u_det3_local = hungarian_matching(cost_iou, self.max_iou_distance)

            for lt, ld in matches3:
                all_matches.append((tentative_idx[lt], unmatched_dets[ld]))
            unmatched_tracks += [tentative_idx[i] for i in u_trk3]
            unmatched_dets    = [unmatched_dets[i] for i in u_det3_local]

        return all_matches, unmatched_tracks, unmatched_dets

    # ── Cost functions ────────────────────────────────────────────────────────

    def _cost_enhanced(self, track_indices, bboxes, measurements, features, color_hists, fw, fh):
        """
        Full enhanced cost for recently-seen confirmed tracks.
        Combines: deep embedding + color histogram + Mahalanobis gating + zone.
        """
        n_tracks = len(track_indices)
        n_dets   = len(bboxes)
        cost = np.full((n_tracks, n_dets), fill_value=1e5, dtype=np.float32)

        for i, t_idx in enumerate(track_indices):
            track = self.tracks[t_idx]

            # ── Mahalanobis gating (gates out impossible pairs) ───────────────
            mah_sq = self.kf.gating_distance(track.mean, track.covariance, measurements)
            valid  = mah_sq < self.CHI2_THRESHOLD

            # ── Deep embedding cost ───────────────────────────────────────────
            if features is not None and track.mean_feature() is not None:
                gallery  = track.mean_feature()[None, :]
                emb_dist = cdist(gallery, features, metric='cosine')[0]

                # Gallery voting bonus
                vote_bonus = np.array([
                    track.vote_embedding(features[j]) for j in range(n_dets)
                ])
                emb_dist = emb_dist * (1.0 - 0.2 * vote_bonus)
            else:
                emb_dist = np.zeros(n_dets)

            # ── Color histogram cost ──────────────────────────────────────────
            if color_hists is not None and track.mean_color_hist() is not None:
                mean_hist = track.mean_color_hist()
                col_dist  = np.array([
                    self.color_extractor.distance(mean_hist, color_hists[j])
                    for j in range(n_dets)
                ], dtype=np.float32)
            else:
                col_dist = np.zeros(n_dets)

            # ── IoU cost ─────────────────────────────────────────────────────
            track_box = track.to_tlbr()
            iou_dist  = np.array([1 - iou(track_box, bboxes[j]) for j in range(n_dets)])

            # ── Zone penalty ──────────────────────────────────────────────────
            zone_pen = np.zeros(n_dets, dtype=np.float32)
            for j in range(n_dets):
                det_zone   = get_exit_zone(bboxes[j], fw, fh)
                zone_pen[j] = zone_penalty(track.exit_zone, det_zone)

            # ── Weighted combination ──────────────────────────────────────────
            combined = (
                self.deep_weight  * emb_dist  +
                self.color_weight * col_dist   +
                (1 - self.deep_weight - self.color_weight - self.zone_weight) * iou_dist +
                self.zone_weight  * zone_pen
            )
            combined[~valid] = 1e5
            cost[i] = combined

        return cost

    def _cost_reid_only(self, track_indices, features, color_hists, det_bboxes, fw, fh):
        """
        Pure ReID cost for long-absent and OCCLUDED tracks.
        No IoU — predicted position has drifted too far.
        Uses deep embedding + color histogram + entry zone penalty.
        """
        n_tracks = len(track_indices)
        n_dets   = len(features) if features is not None else 0
        cost = np.full((n_tracks, n_dets), fill_value=1e5, dtype=np.float32)

        if n_dets == 0:
            return cost

        for i, t_idx in enumerate(track_indices):
            track = self.tracks[t_idx]

            # ── Deep embedding ────────────────────────────────────────────────
            if features is not None and track.mean_feature() is not None:
                gallery  = track.mean_feature()[None, :]
                emb_dist = cdist(gallery, features, metric='cosine')[0]
                vote_bonus = np.array([
                    track.vote_embedding(features[j]) for j in range(n_dets)
                ])
                emb_dist = emb_dist * (1.0 - 0.2 * vote_bonus)
            else:
                continue   # no gallery = skip

            # ── Color histogram ───────────────────────────────────────────────
            if color_hists is not None and track.mean_color_hist() is not None:
                mean_hist = track.mean_color_hist()
                col_dist  = np.array([
                    self.color_extractor.distance(mean_hist, color_hists[j])
                    for j in range(n_dets)
                ], dtype=np.float32)
            else:
                col_dist = np.zeros(n_dets)

            # ── Entry zone penalty ────────────────────────────────────────────
            zone_pen = np.zeros(n_dets, dtype=np.float32)
            for j in range(n_dets):
                det_zone    = get_exit_zone(det_bboxes[j], fw, fh)
                zone_pen[j] = zone_penalty(track.exit_zone, det_zone)

            # ── Pure ReID weighted combination (no IoU) ───────────────────────
            deep_w  = self.deep_weight / (self.deep_weight + self.color_weight)
            color_w = self.color_weight / (self.deep_weight + self.color_weight)

            combined = (
                deep_w  * emb_dist +
                color_w * col_dist +
                self.zone_weight * zone_pen
            )
            cost[i] = combined

        return cost

    def _cost_iou(self, track_indices, bboxes):
        """Pure IoU cost matrix for tentative tracks."""
        cost = np.zeros((len(track_indices), len(bboxes)), dtype=np.float32)
        for i, t_idx in enumerate(track_indices):
            track_box = self.tracks[t_idx].to_tlbr()
            for j, box in enumerate(bboxes):
                cost[i, j] = 1 - iou(track_box, box)
        return cost

    def _mutual_exclusivity(
        self,
        cost: np.ndarray,
        matches: List[Tuple[int, int]]
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Reject assignments where 2 tracks competed for the same
        detection with very similar scores (within ambiguity_margin).
        Prevents wrong ID when 2 people look similar.
        """
        if cost.size == 0 or not matches:
            return matches, []

        n_dets    = cost.shape[1]
        ambiguous = set()

        for d in range(n_dets):
            col   = cost[:, d]
            valid = sorted([c for c in col if c < 1e4])
            if len(valid) >= 2:
                gap = valid[1] - valid[0]
                if gap < self.ambiguity_margin and valid[0] < self.max_appearance_dist:
                    ambiguous.add(d)

        clean    = [(t, d) for t, d in matches if d not in ambiguous]
        rejected = list(ambiguous)

        if rejected:
            print(f"  [DeepSORT] Rejected {len(rejected)} ambiguous assignment(s)")

        return clean, rejected

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _bboxes_to_measurements(self, bboxes: np.ndarray) -> np.ndarray:
        if len(bboxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        w  = bboxes[:, 2] - bboxes[:, 0]
        h  = bboxes[:, 3] - bboxes[:, 1]
        cx = bboxes[:, 0] + w / 2
        cy = bboxes[:, 1] + h / 2
        a  = w / (h + 1e-8)
        return np.stack([cx, cy, a, h], axis=1).astype(np.float32)

    def _initiate_track(
        self,
        measurement: np.ndarray,
        feature: Optional[np.ndarray],
        color_hist: Optional[np.ndarray] = None,
        bbox: Optional[np.ndarray] = None,
        frame_w: int = 0,
        frame_h: int = 0
    ):
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
        if color_hist is not None:
            track.color_hists.append(color_hist)
        if bbox is not None and frame_w > 0:
            track.exit_zone = get_exit_zone(bbox, frame_w, frame_h)
            track.frame_w   = frame_w
            track.frame_h   = frame_h

        self.tracks.append(track)
        self._next_id += 1
