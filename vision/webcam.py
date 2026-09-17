"""Webcam capture at a target frame rate (Phase 1)."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator

import cv2
import numpy as np


class Webcam:
    """Thin OpenCV VideoCapture wrapper that paces reads to a target FPS."""

    def __init__(
        self,
        camera_index: int = 0,
        target_fps: int = 30,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.target_fps = target_fps
        # DirectShow backend starts up faster and more reliably on Windows.
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(camera_index, backend)
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, target_fps)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {camera_index}. Check that a "
                "camera is connected and not in use by another application."
            )

    def frames(self) -> Iterator[np.ndarray]:
        """Yields frames paced to roughly self.target_fps."""
        min_interval = 1.0 / self.target_fps
        last = 0.0
        while True:
            elapsed = time.monotonic() - last
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            ok, frame = self._cap.read()
            last = time.monotonic()
            if not ok:
                break
            yield frame

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "Webcam":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
