"""
Approach 3 — Custom ReID Matching Module
Flexible appearance-based re-identification with multiple distance metrics,
gallery management strategies, and adaptive matching thresholds.
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
    COSINE     = 'cosine'
    EUCLIDEAN  = 'euclidean'
    MAHALANOBIS = 'mahalanobis'


def pairwise_distance(
    query: np.ndarray,         # (M, D)
    gallery: np.ndarray,       # (N, D)
    metric: DistanceMetric = DistanceMetric.COSINE,
    covariance_inv: Optional[np.ndarray] = None   # for Mahalanobis
) -> np.ndarray:
    """
    Compute (M, N) pairwise distance matrix.

    Args:
        query:          Query embeddings (M, D)
        gallery:        Gallery embeddings (N, D)
        metric:         Distance metric to use
        covariance_inv: Required for Mahalanobis; (D, D) inverse covariance

    Returns: (M, N) distance matrix
    """
    if metric == DistanceMetric.COSINE:
        # Already L2-normalised → dot product gives cosine similarity
        return 1.0 - query @ gallery.T

    elif metric == DistanceMetric.EUCLIDEAN:
        return cdist(query, gallery, metric='euclidean')

    elif metric == DistanceMetric.MAHALANOBIS:
        if covariance_inv is None:
            raise ValueError("Mahalanobis requires covariance_inv")
        return cdist(query, gallery, metric='mahalanobis', VI=covariance_inv)

    raise ValueError(f"Unknown metric: {metric}")


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight ReID backbone (custom, smaller than BoT)
# ─────────────────────────────────────────────────────────────────────────────

class LightReIDBackbone(nn.Module):
    """
    Lightweight ReID encoder for research experiments.
    MobileNetV3-Small backbone → 512-dim L2-normalised embedding.
    ~3M params vs BoT's ~25M — faster, easier to fine-tune.
    """

    def __init__(self, feat_dim: int = 512, num_classes: int = 0):
        super().__init__()
        self.feat_dim = feat_dim

        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        # Keep everything except final classifier
        self.backbone = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        in_features = 576  # MobileNetV3-Small output channels
        self.embedding = nn.Sequential(
            nn.Linear(in_features, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim)
        )

        # Optional ID head for training
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

        logits = self.classifier(feat)
        return feat, logits


# ─────────────────────────────────────────────────────────────────────────────
# Gallery management strategies
# ─────────────────────────────────────────────────────────────────────────────

class GalleryStrategy(Enum):
    FIFO           = 'fifo'          # rolling window of last N embeddings
    MEAN           = 'mean'          # single running mean
    EXPONENTIAL_MA = 'ema'           # exponential moving average
    CLUSTER        = 'cluster'       # K representative embeddings (k-means lite)


class AppearanceGallery:
    """
    Per-track embedding gallery with pluggable update strategy.
    """

    def __init__(
        self,
        strategy: GalleryStrategy = GalleryStrategy.FIFO,
        max_size: int = 50,
        ema_alpha: float = 0.9,
        n_clusters: int = 5
    ):
        self.strategy   = strategy
        self.max_size   = max_size
        self.ema_alpha  = ema_alpha
        self.n_clusters = n_clusters

        self._buffer: deque = deque(maxlen=max_size)
        self._mean:   Optional[np.ndarray] = None
        self._ema:    Optional[np.ndarray] = None

    def add(self, embedding: np.ndarray):
        """Add a new embedding to the gallery."""
        e = embedding / (np.linalg.norm(embedding) + 1e-8)

        if self.strategy == GalleryStrategy.FIFO:
            self._buffer.append(e)

        elif self.strategy == GalleryStrategy.MEAN:
            if self._mean is None:
                self._mean = e.copy()
            else:
                n = len(self._buffer) + 1
                self._mean = (self._mean * (n - 1) + e) / n
            self._buffer.append(e)

        elif self.strategy == GalleryStrategy.EXPONENTIAL_MA:
            if self._ema is None:
                self._ema = e.copy()
            else:
                self._ema = self.ema_alpha * self._ema + (1 - self.ema_alpha) * e
                self._ema /= np.linalg.norm(self._ema) + 1e-8
            self._buffer.append(e)

        elif self.strategy == GalleryStrategy.CLUSTER:
            self._buffer.append(e)

    def get_representative(self) -> Optional[np.ndarray]:
        """Return representative embedding for distance computation."""
        if not self._buffer:
            return None

        if self.strategy == GalleryStrategy.FIFO:
            mat = np.stack(self._buffer)
            avg = mat.mean(axis=0)
            return avg / (np.linalg.norm(avg) + 1e-8)

        elif self.strategy == GalleryStrategy.MEAN:
            return self._mean

        elif self.strategy == GalleryStrategy.EXPONENTIAL_MA:
            return self._ema

        elif self.strategy == GalleryStrategy.CLUSTER:
            return self._cluster_representative()

        return None

    def get_all(self) -> Optional[np.ndarray]:
        """Return all stored embeddings (N, D)."""
        if not self._buffer:
            return None
        return np.stack(self._buffer)

    def _cluster_representative(self) -> np.ndarray:
        """
        Lightweight clustering: use the embedding closest to the mean
        as the representative (avoids noisy outliers).
        """
        mat = np.stack(self._buffer)
        center = mat.mean(axis=0)
        dists = np.linalg.norm(mat - center, axis=1)
        return mat[np.argmin(dists)]

    def __len__(self):
        return len(self._buffer)


# ─────────────────────────────────────────────────────────────────────────────
# Custom ReID Matching Module
# ─────────────────────────────────────────────────────────────────────────────

class ReIDMatcher:
    """
    Research-oriented ReID matching module.

    Features:
      • Pluggable distance metrics (cosine / euclidean / Mahalanobis)
      • Pluggable gallery strategies (FIFO / mean / EMA / cluster)
      • Adaptive matching threshold based on occlusion duration
      • Query-gallery distance to all stored embeddings (soft scoring)
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
        base_threshold: float = 0.4,
        gallery_max_size: int = 50,
        batch_size: int = 32
    ):
        self.device          = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.metric          = distance_metric
        self.gallery_strategy = gallery_strategy
        self.base_threshold  = base_threshold
        self.gallery_max     = gallery_max_size
        self.batch_size      = batch_size

        # Load backbone
        if backbone is None:
            backbone = LightReIDBackbone(feat_dim=512)
        self.backbone = backbone.to(self.device)
        self.backbone.eval()

        # Per-track galleries: {track_id: AppearanceGallery}
        self.galleries: Dict[int, AppearanceGallery] = {}

        print(f"[ReIDMatcher] metric={distance_metric.value}  "
              f"strategy={gallery_strategy.value}  device={self.device}")

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def extract_embeddings(
        self,
        frame: np.ndarray,
        bboxes: np.ndarray   # (N, 4)  [x1,y1,x2,y2]
    ) -> np.ndarray:
        """Extract (N, D) embeddings for all bounding boxes."""
        if len(bboxes) == 0:
            return np.zeros((0, self.backbone.feat_dim), dtype=np.float32)

        crops = [self._crop(frame, box) for box in bboxes]
        result = []
        for i in range(0, len(crops), self.batch_size):
            batch = torch.stack(crops[i:i + self.batch_size]).to(self.device)
            emb   = self.backbone(batch, return_feat=True)
            result.append(emb.cpu().numpy())

        return np.vstack(result).astype(np.float32)

    def update_gallery(self, track_id: int, embedding: np.ndarray):
        """Add a new embedding to a track's gallery."""
        if track_id not in self.galleries:
            self.galleries[track_id] = AppearanceGallery(
                strategy=self.gallery_strategy,
                max_size=self.gallery_max
            )
        self.galleries[track_id].add(embedding)

    def remove_track(self, track_id: int):
        """Clean up gallery when track is deleted."""
        self.galleries.pop(track_id, None)

    def compute_distance_matrix(
        self,
        track_ids: List[int],
        det_embeddings: np.ndarray,    # (N_det, D)
        time_since_update: Optional[Dict[int, int]] = None
    ) -> np.ndarray:
        """
        Compute (N_tracks, N_dets) appearance distance matrix.

        Uses adaptive thresholding: longer absence → higher allowed distance
        (accounts for appearance drift during occlusion).

        Returns: cost matrix with invalid pairs set to large value.
        """
        n_tracks = len(track_ids)
        n_dets   = len(det_embeddings)
        cost = np.full((n_tracks, n_dets), fill_value=1e5, dtype=np.float32)

        if n_dets == 0:
            return cost

        for i, tid in enumerate(track_ids):
            gallery = self.galleries.get(tid)
            if gallery is None or len(gallery) == 0:
                continue

            rep = gallery.get_representative()
            if rep is None:
                continue

            # Distance from this track's representative to all detections
            dist = pairwise_distance(
                rep[None, :],               # (1, D)
                det_embeddings,             # (N, D)
                metric=self.metric
            )[0]                            # (N,)

            # Adaptive threshold (relax after long absence)
            threshold = self._adaptive_threshold(tid, time_since_update)
            dist[dist > threshold] = 1e5
            cost[i] = dist

        return cost

    def compute_soft_scores(
        self,
        track_id: int,
        det_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute soft similarity by comparing against ALL gallery embeddings
        (not just representative). Returns (N_dets,) score in [0, 1].
        Useful for research metrics and analysis.
        """
        gallery = self.galleries.get(track_id)
        if gallery is None:
            return np.zeros(len(det_embeddings))

        all_gallery = gallery.get_all()
        if all_gallery is None:
            return np.zeros(len(det_embeddings))

        # (N_gallery, N_dets) distance matrix
        dist = pairwise_distance(all_gallery, det_embeddings, metric=self.metric)

        # Min-distance aggregation (most generous matching)
        min_dist = dist.min(axis=0)      # (N_dets,)
        return np.clip(1.0 - min_dist, 0, 1)

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
        """
        Relax matching threshold for longer-absent tracks.
        Linear ramp: base_threshold at age=0, up to 0.7 at age≥10.
        """
        if time_since_update is None:
            return self.base_threshold
        age = time_since_update.get(track_id, 0)
        relaxation = min(age / 10.0, 1.0) * 0.3
        return self.base_threshold + relaxation
