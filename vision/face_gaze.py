"""Face detection + full-face appearance-based gaze estimation (Phase 1).

Uses `uniface` (MIT-licensed, ONNX Runtime): RetinaFace for face detection and
MobileGaze (ResNet/MobileNet full-face CNN, pretrained on gaze estimation
datasets — iTracker / full-face-CNN lineage, see MASTER_PROJECT_DOCUMENT.md
Section 1.2/14) for pitch/yaw gaze direction. Weights auto-download on first
use and are SHA-256 verified; no training from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from uniface.constants import GazeWeights
from uniface.detection import RetinaFace
from uniface.gaze import MobileGaze


@dataclass
class FaceGazeResult:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixels
    detection_confidence: float
    pitch: float  # radians; + = looking up
    yaw: float  # radians; + = looking right


class FaceGazeEstimator:
    """Detects the primary (most central) face in a frame and its gaze direction."""

    def __init__(self, gaze_model: GazeWeights = GazeWeights.RESNET34) -> None:
        self._detector = RetinaFace()
        self._gaze = MobileGaze(model_name=gaze_model)

    def estimate(self, frame: np.ndarray) -> FaceGazeResult | None:
        faces = self._detector.detect(frame, max_num=1, metric="max")
        if not faces:
            return None
        face = faces[0]

        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        if x2 <= x1 or y2 <= y1:
            return None

        face_crop = frame[y1:y2, x1:x2]
        gaze = self._gaze.estimate(face_crop)

        return FaceGazeResult(
            bbox=(x1, y1, x2, y2),
            detection_confidence=float(face.confidence),
            pitch=float(gaze.pitch),
            yaw=float(gaze.yaw),
        )
