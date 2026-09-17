"""End-to-end Phase 1 pipeline: webcam -> face/gaze -> normalized (x, y) point.

Yields dicts matching the WebSocket schema in MASTER_PROJECT_DOCUMENT.md
Section 16 (gaze_x, gaze_y, confidence, selected_key), plus debug-only extras
(pitch, yaw, bbox, fps, frame). Phase 2's backend should forward only the
JSON-safe subset over the WebSocket and drop the debug extras.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .face_gaze import FaceGazeEstimator
from .gaze_to_screen import GazeMappingConfig, GazeToScreenMapper
from .webcam import Webcam

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "gaze_mapping.yaml"


class GazePipeline:
    def __init__(self, camera_index: int = 0, target_fps: int = 30) -> None:
        self.webcam = Webcam(camera_index=camera_index, target_fps=target_fps)
        self.estimator = FaceGazeEstimator()
        mapping_config = (
            GazeMappingConfig.from_yaml(str(CONFIG_PATH))
            if CONFIG_PATH.exists()
            else GazeMappingConfig()
        )
        self.mapper = GazeToScreenMapper(mapping_config)

    def stream(self) -> Iterator[dict[str, Any]]:
        last_time = time.monotonic()
        for frame in self.webcam.frames():
            now = time.monotonic()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now

            result = self.estimator.estimate(frame)
            if result is None:
                self.mapper.reset()
                yield {
                    "gaze_x": None,
                    "gaze_y": None,
                    "confidence": 0.0,
                    "selected_key": None,
                    "face_found": False,
                    "fps": fps,
                    "frame": frame,
                }
                continue

            gaze_x, gaze_y = self.mapper.map(result.pitch, result.yaw)
            yield {
                "gaze_x": gaze_x,
                "gaze_y": gaze_y,
                "confidence": result.detection_confidence,
                "selected_key": None,
                "face_found": True,
                "pitch": result.pitch,
                "yaw": result.yaw,
                "bbox": result.bbox,
                "fps": fps,
                "frame": frame,
            }

    def close(self) -> None:
        self.webcam.release()

    def __enter__(self) -> "GazePipeline":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
