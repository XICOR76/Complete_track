"""
Approach 3 — Enhanced ReID Matching Module
Solves the 2-person ID swap problem by combining:

  1. Deep embedding (MobileNetV3)     — overall appearance
  2. Color histogram (upper + lower)  — clothing color, very discriminative
  3. Body proportion features         — height/width ratio, torso shape
  4. Entry zone memory                — which frame edge person exited from
  5. Mutual exclusivity check         — rejects ambiguous close-call assignments
  6. Gallery voting                   — majority vote across all stored embeddings

When 2 people leave and re-enter:
  → Color histograms are almost always distinct (different clothing)
  → Entry zone narrows down which person enters from which side
  → Mutual exclusivity prevents wrong assignment when scores are close
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Dict, Optional, Tuple
from enum import Enum
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Distance metrics
# ─────────────────────────────────────────────────────────────────────────────

class DistanceMetric(Enum):
    COSINE      = 'cosine'
    EUCLIDEAN   = 'euclidean'
    MAHALANOBIS = 'mahalanobis'


def pairwise_distance(
    query: np.ndarray,
    gallery: np.ndarray,
    metric: DistanceMetric = DistanceMetric.COSINE,
    covariance_inv: Optional[np.ndarray] = None
) -> np.ndarray:
    if metric == DistanceMetric.COSINE:
        return 1.0 - query @ gallery.T
    elif metric == DistanceMetric.EUCLIDEAN:
        return cdist(query, gallery, metric='euclidean')
    elif metric == DistanceMetric.MAHALANOBIS:
        if covariance_inv is None:
            raise ValueError("Mahalanobis requires covariance_inv")
        return cdist(query, gallery, metric='mahalanobis', VI=covariance_inv)
    raise ValueError(f"Unknown metric: {metric}")


# ─────────────────────────────────────────────────────────────────────────────
# Color Histogram Extractor
# ─────────────────────────────────────────────────────────────────────────────

class ColorHistogramExtractor:
    """
    Extracts HSV color histograms from upper and lower body separately.

    Why upper + lower separately?
      Person with red shirt / blue jeans:
        upper hist → heavily red
        lower hist → heavily blue
      This is far more discriminative than a single full-body histogram.

    Why HSV not RGB?
      HSV separates color (Hue) from brightness (Value) and saturation.
      More robust to lighting changes — same shirt looks same under
      different lighting conditions.
    """

    def __init__(
        self,
        h_bins: int = 16,    # Hue bins   (color)
        s_bins: int = 4,     # Saturation bins
        v_bins: int = 4,     # Value bins (brightness)
    ):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        # Total histogram size per zone = h_bins * s_bins * v_bins
        self.hist_size = h_bins * s_bins * v_bins   # 256 dims per zone
        self.feat_dim  = self.hist_size * 2          # upper + lower = 512 dims

    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Extract combined upper+lower body color histogram from one bbox.

        Returns: (feat_dim,) L2-normalised float32 vector
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 4:
            return np.zeros(self.feat_dim, dtype=np.float32)

        # Split into upper body (top 50%) and lower body (bottom 50%)
        mid = crop.shape[0] // 2
        upper = crop[:mid, :]
        lower = crop[mid:, :]

        upper_hist = self._compute_hsv_histogram(upper)
        lower_hist = self._compute_hsv_histogram(lower)

        combined = np.concatenate([upper_hist, lower_hist])
        norm = np.linalg.norm(combined)
        return (combined / (norm + 1e-8)).astype(np.float32)

    def extract_batch(
        self, frame: np.ndarray, bboxes: np.ndarray
    ) -> np.ndarray:
        """Extract histograms for all bboxes. Returns (N, feat_dim)."""
        if len(bboxes) == 0:
            return np.zeros((0, self.feat_dim), dtype=np.float32)
        return np.stack([self.extract(frame, box) for box in bboxes])

    def distance(self, hist_a: np.ndarray, hist_b: np.ndarray) -> float:
        """
        Histogram distance using Bhattacharyya coefficient.
        Better than L2 for histograms — handles different lighting.
        Range [0, 1]: 0 = identical, 1 = completely different.
        """
        # Bhattacharyya: -ln(sum(sqrt(a*b)))
        # Both already L2-normalised, treat as probability distributions
        similarity = np.sum(np.sqrt(np.abs(hist_a * hist_b)))
        similarity = np.clip(similarity, 0, 1)
        return float(1.0 - similarity)

    def distance_matrix(
        self, query_hists: np.ndarray, gallery_hists: np.ndarray
    ) -> np.ndarray:
        """Pairwise histogram distance matrix. Returns (M, N)."""
        M, N = len(query_hists), len(gallery_hists)
        dist = np.zeros((M, N), dtype=np.float32)
        for i in range(M):
            for j in range(N):
                dist[i, j] = self.distance(query_hists[i], gallery_hists[j])
        return dist

    def _compute_hsv_histogram(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Compute 3D HSV histogram for a crop, returned as flat vector."""
        if crop_bgr.size == 0:
            return np.zeros(self.hist_size, dtype=np.float32)

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [self.h_bins, self.s_bins, self.v_bins],
            [0, 180, 0, 256, 0, 256]
        )
        hist = hist.flatten().astype(np.float32)
        norm = np.sum(hist)
        return hist / (norm + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Body Proportion Extractor
# ─────────────────────────────────────────────────────────────────────────────

class BodyProportionExtractor:
    """
    Extracts physical proportion features from bounding box.

    Features:
      - aspect_ratio      : height / width  (tall vs wide person)
      - relative_height   : bbox height / frame height
      - relative_width    : bbox width / frame width
      - upper_body_ratio  : upper body brightness vs lower
      - edge_density      : how much detail/texture in the crop

    These are stable across frames and help distinguish people of
    different heights/builds even when color is similar.
    """

    FEAT_DIM = 5

    def extract(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> np.ndarray:
        """Returns (5,) normalised proportion feature vector."""
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)

        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)

        aspect_ratio     = bh / (bw + 1e-8)
        relative_height  = bh / (fh + 1e-8)
        relative_width   = bw / (fw + 1e-8)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(self.FEAT_DIM, dtype=np.float32)

        # Upper vs lower brightness ratio
        mid = crop.shape[0] // 2
        upper_bright = np.mean(cv2.cvtColor(crop[:mid], cv2.COLOR_BGR2GRAY)) / 255.0 \
                       if mid > 0 else 0.5
        lower_bright = np.mean(cv2.cvtColor(crop[mid:], cv2.COLOR_BGR2GRAY)) / 255.0 \
                       if crop.shape[0] - mid > 0 else 0.5
        brightness_ratio = upper_bright / (lower_bright + 1e-8)

        # Edge density — how much texture/detail
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        edge_density = np.mean(np.abs(edges)) / 255.0

        feat = np.array([
            np.clip(aspect_ratio / 5.0, 0, 1),       # normalise to [0,1]
            relative_height,
            relative_width,
            np.clip(brightness_ratio / 3.0, 0, 1),
            np.clip(edge_density, 0, 1)
        ], dtype=np.float32)

        return feat

    def extract_batch(
        self, frame: np.ndarray, bboxes: np.ndarray
    ) -> np.ndarray:
        if len(bboxes) == 0:
            return np.zeros((0, self.FEAT_DIM), dtype=np.float32)
        return np.stack([self.extract(frame, box) for box in bboxes])

    def distance_matrix(
        self, query_props: np.ndarray, gallery_props: np.ndarray
    ) -> np.ndarray:
        """L2 distance between proportion vectors. Returns (M, N)."""
        return cdist(query_props, gallery_props, metric='euclidean').astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Zone Memory
# ─────────────────────────────────────────────────────────────────────────────

class EntryZone(Enum):
    LEFT   = 'left'
    RIGHT  = 'right'
    TOP    = 'top'
    BOTTOM = 'bottom'
    CENTER = 'center'   # appeared in the middle (e.g. from behind object)
    UNKNOWN = 'unknown'


def get_entry_zone(
    bbox: np.ndarray,
    frame_w: int,
    frame_h: int,
    edge_margin: float = 0.15   # 15% of frame width/height = "near edge"
) -> EntryZone:
    """
    Determine which zone of the frame a bbox is in.
    Used to remember where a person exited and expect re-entry from same zone.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    left_thresh   = frame_w * edge_margin
    right_thresh  = frame_w * (1 - edge_margin)
    top_thresh    = frame_h * edge_margin
    bottom_thresh = frame_h * (1 - edge_margin)

    if cx < left_thresh:
        return EntryZone.LEFT
    elif cx > right_thresh:
        return EntryZone.RIGHT
    elif cy < top_thresh:
        return EntryZone.TOP
    elif cy > bottom_thresh:
        return EntryZone.BOTTOM
    else:
        return EntryZone.CENTER


def zone_distance(zone_a: EntryZone, zone_b: EntryZone) -> float:
    """
    Distance between two zones.
    Same zone = 0.0, opposite zone = 1.0, adjacent zone = 0.5.
    """
    if zone_a == EntryZone.UNKNOWN or zone_b == EntryZone.UNKNOWN:
        return 0.0   # unknown zone → no penalty

    if zone_a == zone_b:
        return 0.0

    opposites = {
        EntryZone.LEFT: EntryZone.RIGHT,
        EntryZone.RIGHT: EntryZone.LEFT,
        EntryZone.TOP: EntryZone.BOTTOM,
        EntryZone.BOTTOM: EntryZone.TOP,
    }
    if opposites.get(zone_a) == zone_b:
        return 1.0

    return 0.5   # adjacent zones


# ─────────────────────────────────────────────────────────────────────────────
# Per-Track Feature Store
# ─────────────────────────────────────────────────────────────────────────────

class TrackFeatureStore:
    """
    Stores ALL feature types for one track:
      - Deep embeddings (gallery)
      - Color histograms (gallery)
      - Body proportions (running mean)
      - Exit zone (last known frame edge)
      - Frame dimensions (for zone computation)
    """

    def __init__(self, max_gallery: int = 50):
        self.max_gallery    = max_gallery
        self.embeddings     : deque = deque(maxlen=max_gallery)
        self.color_hists    : deque = deque(maxlen=max_gallery)
        self.proportions    : deque = deque(maxlen=20)
        self.exit_zone      : EntryZone = EntryZone.UNKNOWN
        self.frame_w        : int = 0
        self.frame_h        : int = 0

    def update(
        self,
        embedding:   np.ndarray,
        color_hist:  np.ndarray,
        proportion:  np.ndarray,
        bbox:        np.ndarray,
        frame_w:     int,
        frame_h:     int
    ):
        self.embeddings.append(embedding / (np.linalg.norm(embedding) + 1e-8))
        self.color_hists.append(color_hist)
        self.proportions.append(proportion)
        self.frame_w = frame_w
        self.frame_h = frame_h
        # Always update exit zone — last seen position
        self.exit_zone = get_entry_zone(bbox, frame_w, frame_h)

    def mean_embedding(self) -> Optional[np.ndarray]:
        if not self.embeddings:
            return None
        mat = np.stack(self.embeddings)
        avg = mat.mean(axis=0)
        return avg / (np.linalg.norm(avg) + 1e-8)

    def mean_color_hist(self) -> Optional[np.ndarray]:
        if not self.color_hists:
            return None
        mat = np.stack(self.color_hists)
        avg = mat.mean(axis=0)
        return avg / (np.linalg.norm(avg) + 1e-8)

    def mean_proportion(self) -> Optional[np.ndarray]:
        if not self.proportions:
            return None
        return np.stack(self.proportions).mean(axis=0)

    def vote_embedding(self, query_emb: np.ndarray) -> float:
        """
        Gallery voting: fraction of stored embeddings that are
        close to the query (cosine distance < 0.5).
        Returns score in [0, 1]. Higher = more votes = better match.
        """
        if not self.embeddings:
            return 0.0
        mat = np.stack(self.embeddings)
        dists = 1.0 - mat @ query_emb
        votes = np.sum(dists < 0.5)
        return float(votes) / len(self.embeddings)

    def __len__(self):
        return len(self.embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight ReID Backbone
# ─────────────────────────────────────────────────────────────────────────────

class LightReIDBackbone(nn.Module):
    def __init__(self, feat_dim: int = 512, num_classes: int = 0):
        super().__init__()
        self.feat_dim = feat_dim

        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        self.backbone = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        in_features = 576
        self.embedding = nn.Sequential(
            nn.Linear(in_features, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim)
        )
        self.classifier = nn.Linear(feat_dim, num_classes) if num_classes > 0 else None
        self._init_weights()

    def _init_weights(self):
        for m in self.embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x: torch.Tensor, return_feat: bool = True):
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)
        feat = self.embedding(feat)
        feat = F.normalize(feat, p=2, dim=1)
        if return_feat or self.classifier is None:
            return feat
        return feat, self.classifier(feat)


# ─────────────────────────────────────────────────────────────────────────────
# Gallery strategy (kept for compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class GalleryStrategy(Enum):
    FIFO           = 'fifo'
    MEAN           = 'mean'
    EXPONENTIAL_MA = 'ema'
    CLUSTER        = 'cluster'


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced ReID Matcher
# ─────────────────────────────────────────────────────────────────────────────

class ReIDMatcher:
    """
    Enhanced ReID Matcher — solves 2-person ID swap problem.

    Scoring pipeline per (track, detection) pair:
    ─────────────────────────────────────────────
      deep_score    = cosine distance of embeddings        (weight: 0.40)
      color_score   = Bhattacharyya histogram distance     (weight: 0.35)
      prop_score    = body proportion L2 distance          (weight: 0.10)
      zone_penalty  = entry zone mismatch penalty          (weight: 0.15)
      ─────────────────────────────────────────────────────────────────
      combined_cost = weighted sum of all four

    Mutual exclusivity check:
    ─────────────────────────
      After Hungarian assignment, if two tracks competed for same detection
      and their scores are within `ambiguity_margin` of each other →
      REJECT both → treat detection as new track.
      This prevents wrong assignment when two people look similar.
    """

    TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(
        self,
        backbone: nn.Module = None,
        device: str = None,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        gallery_strategy: GalleryStrategy = GalleryStrategy.FIFO,
        base_threshold: float = 0.45,
        gallery_max_size: int = 50,
        batch_size: int = 32,
        # Feature weights
        deep_weight:  float = 0.40,
        color_weight: float = 0.35,
        prop_weight:  float = 0.10,
        zone_weight:  float = 0.15,
        # Mutual exclusivity
        ambiguity_margin: float = 0.08,
        # Entry zone
        use_entry_zone: bool = True,
    ):
        self.device           = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.metric           = distance_metric
        self.gallery_strategy = gallery_strategy
        self.base_threshold   = base_threshold
        self.gallery_max      = gallery_max_size
        self.batch_size       = batch_size

        # Feature weights
        self.deep_weight      = deep_weight
        self.color_weight     = color_weight
        self.prop_weight      = prop_weight
        self.zone_weight      = zone_weight
        self.ambiguity_margin = ambiguity_margin
        self.use_entry_zone   = use_entry_zone

        # Backbone
        if backbone is None:
            backbone = LightReIDBackbone(feat_dim=512)
        self.backbone = backbone.to(self.device)
        self.backbone.eval()

        # Feature extractors
        self.color_extractor = ColorHistogramExtractor()
        self.prop_extractor  = BodyProportionExtractor()

        # Per-track feature stores: {track_id: TrackFeatureStore}
        self.stores: Dict[int, TrackFeatureStore] = {}

        print(f"[ReIDMatcher] Enhanced mode | device={self.device}")
        print(f"  Weights → deep:{deep_weight} color:{color_weight} "
              f"prop:{prop_weight} zone:{zone_weight}")

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def extract_embeddings(
        self,
        frame: np.ndarray,
        bboxes: np.ndarray
    ) -> np.ndarray:
        """Extract deep embeddings. Returns (N, feat_dim)."""
        if len(bboxes) == 0:
            return np.zeros((0, self.backbone.feat_dim), dtype=np.float32)

        crops = [self._crop(frame, box) for box in bboxes]
        result = []
        for i in range(0, len(crops), self.batch_size):
            batch = torch.stack(crops[i:i + self.batch_size]).to(self.device)
            emb   = self.backbone(batch, return_feat=True)
            result.append(emb.cpu().numpy())
        return np.vstack(result).astype(np.float32)

    def extract_all_features(
        self,
        frame: np.ndarray,
        bboxes: np.ndarray
    ) -> dict:
        """
        Extract ALL feature types for all detections in one call.
        Returns dict with keys: embeddings, color_hists, proportions
        """
        embeddings  = self.extract_embeddings(frame, bboxes)
        color_hists = self.color_extractor.extract_batch(frame, bboxes)
        proportions = self.prop_extractor.extract_batch(frame, bboxes)
        return {
            'embeddings'  : embeddings,
            'color_hists' : color_hists,
            'proportions' : proportions,
        }

    def update_gallery(
        self,
        track_id:   int,
        embedding:  np.ndarray,
        frame:      np.ndarray = None,
        bbox:       np.ndarray = None,
    ):
        """Update track's feature store with new observation."""
        if track_id not in self.stores:
            self.stores[track_id] = TrackFeatureStore(max_gallery=self.gallery_max)

        store = self.stores[track_id]

        if frame is not None and bbox is not None:
            fh, fw = frame.shape[:2]
            color_hist = self.color_extractor.extract(frame, bbox)
            proportion = self.prop_extractor.extract(frame, bbox)
            store.update(embedding, color_hist, proportion, bbox, fw, fh)
        else:
            # Fallback: update only embedding
            store.embeddings.append(
                embedding / (np.linalg.norm(embedding) + 1e-8)
            )

    def remove_track(self, track_id: int):
        self.stores.pop(track_id, None)

    def compute_distance_matrix(
        self,
        track_ids:          List[int],
        det_embeddings:     np.ndarray,
        time_since_update:  Optional[Dict[int, int]] = None,
        # Enhanced features
        det_color_hists:    Optional[np.ndarray] = None,
        det_proportions:    Optional[np.ndarray] = None,
        det_bboxes:         Optional[np.ndarray] = None,
        frame_shape:        Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Compute enhanced (N_tracks, N_dets) cost matrix combining
        deep embedding + color histogram + proportions + entry zone.
        """
        n_tracks = len(track_ids)
        n_dets   = len(det_embeddings)
        cost = np.full((n_tracks, n_dets), fill_value=1e5, dtype=np.float32)

        if n_dets == 0:
            return cost

        for i, tid in enumerate(track_ids):
            store = self.stores.get(tid)
            if store is None or len(store) == 0:
                continue

            # ── 1. Deep embedding cost ────────────────────────────────────────
            mean_emb = store.mean_embedding()
            if mean_emb is None:
                continue

            emb_dist = pairwise_distance(
                mean_emb[None, :],
                det_embeddings,
                metric=self.metric
            )[0]   # (N_dets,)

            # Gallery voting bonus — detections that match many gallery items
            vote_bonus = np.array([
                store.vote_embedding(det_embeddings[j])
                for j in range(n_dets)
            ])
            # Reduce embedding distance by vote bonus (max 20% reduction)
            emb_dist = emb_dist * (1.0 - 0.2 * vote_bonus)

            combined = self.deep_weight * emb_dist

            # ── 2. Color histogram cost ───────────────────────────────────────
            if det_color_hists is not None:
                mean_hist = store.mean_color_hist()
                if mean_hist is not None:
                    color_dist = np.array([
                        self.color_extractor.distance(mean_hist, det_color_hists[j])
                        for j in range(n_dets)
                    ], dtype=np.float32)
                    combined += self.color_weight * color_dist

            # ── 3. Body proportion cost ───────────────────────────────────────
            if det_proportions is not None:
                mean_prop = store.mean_proportion()
                if mean_prop is not None:
                    prop_dist = np.linalg.norm(
                        det_proportions - mean_prop[None, :], axis=1
                    ).astype(np.float32)
                    # Normalise prop distance to [0, 1]
                    prop_dist = np.clip(prop_dist / 2.0, 0, 1)
                    combined += self.prop_weight * prop_dist

            # ── 4. Entry zone penalty ─────────────────────────────────────────
            if self.use_entry_zone and det_bboxes is not None and frame_shape is not None:
                fh, fw = frame_shape
                for j in range(n_dets):
                    det_zone = get_entry_zone(det_bboxes[j], fw, fh)
                    z_dist   = zone_distance(store.exit_zone, det_zone)
                    combined[j] += self.zone_weight * z_dist

            # ── Adaptive threshold ────────────────────────────────────────────
            threshold = self._adaptive_threshold(tid, time_since_update)
            combined[combined > threshold] = 1e5
            cost[i] = combined

        return cost

    def mutual_exclusivity_check(
        self,
        cost_matrix: np.ndarray,
        matches: List[Tuple[int, int]],
        ambiguity_margin: float = None
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Remove ambiguous assignments where two tracks competed for
        the same detection with similar scores.

        Example:
          Track 1 → Det A: cost = 0.31
          Track 2 → Det A: cost = 0.33   (margin = 0.02, very close)
          → REJECT both → Det A becomes unmatched → new track created

        This prevents wrong ID assignment when 2 people look similar.

        Returns:
          clean_matches    : matches that passed exclusivity check
          rejected_dets    : detection indices that were rejected
        """
        margin = ambiguity_margin or self.ambiguity_margin

        # For each detection, find the two best competing tracks
        n_dets = cost_matrix.shape[1]
        ambiguous_dets = set()

        for d_idx in range(n_dets):
            track_costs = cost_matrix[:, d_idx]
            valid = track_costs < 1e4
            valid_costs = track_costs[valid]

            if np.sum(valid) >= 2:
                sorted_costs = np.sort(valid_costs)
                best  = sorted_costs[0]
                second = sorted_costs[1]
                # If top 2 scores are very close → ambiguous
                if (second - best) < margin and best < self.base_threshold:
                    ambiguous_dets.add(d_idx)

        # Filter out ambiguous matches
        clean_matches = [
            (t, d) for t, d in matches
            if d not in ambiguous_dets
        ]
        rejected_dets = list(ambiguous_dets)

        if rejected_dets:
            print(f"  [ReIDMatcher] Rejected {len(rejected_dets)} ambiguous assignment(s)")

        return clean_matches, rejected_dets

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _crop(self, frame: np.ndarray, box: np.ndarray) -> torch.Tensor:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((128, 64, 3), dtype=np.uint8)
        return self.TRANSFORM(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    def _adaptive_threshold(
        self,
        track_id: int,
        time_since_update: Optional[Dict[int, int]]
    ) -> float:
        if time_since_update is None:
            return self.base_threshold
        age = time_since_update.get(track_id, 0)
        # Relax threshold slightly for long-absent tracks
        relaxation = min(age / 20.0, 1.0) * 0.2
        return self.base_threshold + relaxation
