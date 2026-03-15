"""
Approach 2 — Faster R-CNN Detector
Two-stage object detector using torchvision's Faster R-CNN.
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Detection:
    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class FasterRCNNDetector:
    """
    Two-stage Faster R-CNN detector.
    More stable detections at cost of speed vs YOLO.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        target_classes: List[str] = None,
        device: str = None
    ):
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes or ['person']
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[FasterRCNN] Loading model on {self.device}...")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        self.model.to(self.device)
        self.model.eval()

        # Build target class index set
        self.target_ids = {
            i for i, name in enumerate(COCO_CLASSES)
            if name in self.target_classes
        }
        self.transform = weights.transforms()
        print(f"[FasterRCNN] Ready. Targeting: {self.target_classes}")

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run Faster R-CNN on a single BGR frame.
        Returns list of Detection objects.
        """
        # BGR → RGB → tensor
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        outputs = self.model([img_tensor])
        result = outputs[0]

        boxes   = result['boxes'].cpu().numpy()
        scores  = result['scores'].cpu().numpy()
        labels  = result['labels'].cpu().numpy()

        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if score < self.confidence_threshold:
                continue
            if label not in self.target_ids:
                continue

            detections.append(Detection(
                bbox       = box.astype(np.float32),      # [x1,y1,x2,y2]
                confidence = float(score),
                class_id   = int(label),
                class_name = COCO_CLASSES[int(label)]
            ))

        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Batch inference for multiple frames."""
        tensors = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensors.append(t.to(self.device))

        with torch.no_grad():
            outputs = self.model(tensors)

        all_detections = []
        for result in outputs:
            boxes  = result['boxes'].cpu().numpy()
            scores = result['scores'].cpu().numpy()
            labels = result['labels'].cpu().numpy()

            dets = []
            for box, score, label in zip(boxes, scores, labels):
                if score < self.confidence_threshold:
                    continue
                if label not in self.target_ids:
                    continue
                dets.append(Detection(
                    bbox       = box.astype(np.float32),
                    confidence = float(score),
                    class_id   = int(label),
                    class_name = COCO_CLASSES[int(label)]
                ))
            all_detections.append(dets)

        return all_detections
