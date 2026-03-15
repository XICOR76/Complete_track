"""
Approach 2 — Strong ReID Model
Dedicated Re-Identification network for robust appearance embeddings.
Architecture: ResNet50 backbone + BatchNorm neck + ID classification head.

Used to enhance DeepSORT's re-assignment after long occlusions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import cv2
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Backbone + neck
# ─────────────────────────────────────────────────────────────────────────────

class BatchNormNeck(nn.Module):
    """BNNeck: trains with BN features, infers with pre-BN features."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)

    def forward(self, x):
        return self.bn(x)


class StrongReIDModel(nn.Module):
    """
    Strong ReID baseline.
    Paper reference: Bag of Tricks (BoT) for Person Re-ID (Luo et al., 2019).

    Architecture:
        ResNet50 (pretrained) → Global Avg Pool → BNNeck → L2-norm embedding
        + FC classifier head (training only)
    """

    def __init__(self, num_classes: int = 751, feat_dim: int = 2048):
        super().__init__()
        self.feat_dim = feat_dim

        # ── Backbone ──────────────────────────────────────────────────────────
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove final FC + avg pool
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # Last stride trick: change stride of layer4 from 2→1
        # Preserves spatial resolution → richer features
        self.backbone[-1][0].conv2.stride = (1, 1)
        self.backbone[-1][0].downsample[0].stride = (1, 1)

        # ── Pooling ───────────────────────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)          # Global Average Pool
        self.gem = GeM()                              # GeM pool (optional)

        # ── Neck ──────────────────────────────────────────────────────────────
        self.bnneck = BatchNormNeck(feat_dim)

        # ── ID Classifier (training only) ─────────────────────────────────────
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out')

    def forward(self, x, return_feat=True):
        """
        Args:
            x: (B, 3, H, W)
            return_feat: if True return BNNeck features (inference)
                         if False return (feat, logits) (training)
        Returns:
            inference → (B, feat_dim) L2-normalised embedding
            training  → (feat_bn, logits)
        """
        feat_map = self.backbone(x)        # (B, 2048, H', W')
        feat = self.gap(feat_map)          # (B, 2048, 1, 1)
        feat = feat.flatten(1)             # (B, 2048)  ← before BN

        feat_bn = self.bnneck(feat)        # (B, 2048)  ← after BN

        if return_feat:
            return F.normalize(feat_bn, p=2, dim=1)

        logits = self.classifier(feat_bn)
        return feat_bn, logits


class GeM(nn.Module):
    """Generalised Mean Pooling — stronger than GAP for ReID."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p)\
                .mean(dim=(-2, -1), keepdim=True)\
                .pow(1.0 / self.p)


# ─────────────────────────────────────────────────────────────────────────────
# ReID Feature Extractor (inference wrapper)
# ─────────────────────────────────────────────────────────────────────────────

class ReIDExtractor:
    """
    Wraps StrongReIDModel for easy per-crop embedding extraction.
    Designed to plug directly into DeepSORT's appearance module.
    """

    TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),     # standard ReID input size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    def __init__(
        self,
        model: StrongReIDModel,
        device: str = None,
        batch_size: int = 32
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        print(f"[ReIDExtractor] Ready on {self.device}")

    @classmethod
    def from_pretrained(cls, weights_path: str, num_classes: int = 751, **kwargs):
        """Load from checkpoint file."""
        model = StrongReIDModel(num_classes=num_classes)
        state = torch.load(weights_path, map_location='cpu')
        # Handle both raw state_dicts and checkpoint dicts
        if 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state, strict=False)
        return cls(model=model, **kwargs)

    @torch.no_grad()
    def extract(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """
        Extract appearance embeddings for all bboxes in a frame.

        Args:
            frame:  BGR frame (H, W, 3)
            bboxes: (N, 4) array of [x1, y1, x2, y2]

        Returns:
            embeddings: (N, feat_dim) float32 numpy array
        """
        if len(bboxes) == 0:
            return np.zeros((0, self.model.feat_dim), dtype=np.float32)

        crops = self._crop_and_transform(frame, bboxes)

        embeddings = []
        for i in range(0, len(crops), self.batch_size):
            batch = torch.stack(crops[i:i + self.batch_size]).to(self.device)
            emb = self.model(batch, return_feat=True)     # (B, feat_dim)
            embeddings.append(emb.cpu().numpy())

        return np.vstack(embeddings).astype(np.float32)

    def _crop_and_transform(self, frame: np.ndarray, bboxes: np.ndarray):
        """Crop each bbox from frame and apply ReID transforms."""
        h, w = frame.shape[:2]
        tensors = []
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((256, 128, 3), dtype=np.uint8)

            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.TRANSFORM(rgb_crop))
        return tensors

    def cosine_distance_matrix(
        self,
        query_embs: np.ndarray,
        gallery_embs: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise cosine distance matrix.
        Both inputs already L2-normalised → dot product = cosine similarity.

        Returns: (N_query, N_gallery) distance matrix in [0, 2]
        """
        similarity = query_embs @ gallery_embs.T      # cosine similarity
        return 1.0 - similarity                         # convert to distance
