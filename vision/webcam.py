"""Webcam capture at a target frame rate (Phase 1)."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator

import cv2
import numpy as np


class Webcam:
    """OpenCV VideoCapture wrapper that always hands out the NEWEST frame.

    A background thread keeps reading the camera as fast as it delivers, and
    `frames()` yields only the most recent one. Without this, a slow consumer
    (face detection + gaze CNN take tens to hundreds of ms per frame) leaves
    the driver's internal queue full of stale frames, so the gaze point
    lags the eye by a second or more -- which on a dwell keyboard means
    selecting whatever key you looked at *before* the one you're on.
    """

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
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # honoured by some backends; the thread below covers the rest
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {camera_index}. Check that a "
                "camera is connected and not in use by another application."
            )

        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._latest: np.ndarray | None = None
        self._seq = 0  # increments per captured frame, so frames() never yields the same one twice
        self._failed = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._grab_loop, name="webcam-grab", daemon=True)
        self._thread.start()

    def _grab_loop(self, max_consecutive_failures: int = 60) -> None:
        # A single failed read (ok=False) is common and transient with
        # DirectShow on Windows -- USB scheduling hiccups, brief driver stalls
        # -- and does not mean the camera is gone. Only give up after
        # max_consecutive_failures reads in a row fail (default ~2s at 30fps).
        failures = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                failures += 1
                if failures >= max_consecutive_failures:
                    break
                time.sleep(0.01)
                continue
            failures = 0
            with self._new_frame:
                self._latest = frame
                self._seq += 1
                self._new_frame.notify_all()
        with self._new_frame:
            self._failed = True
            self._new_frame.notify_all()

    def frames(self) -> Iterator[np.ndarray]:
        """Yields the newest frame each time one is available, skipping any
        that were captured while the consumer was still busy with the last."""
        last_seq = 0
        while True:
            with self._new_frame:
                while self._seq == last_seq and not self._failed and not self._stop.is_set():
                    self._new_frame.wait(timeout=0.5)
                if self._seq == last_seq:  # camera gave up, or released
                    return
                frame, last_seq = self._latest, self._seq
            assert frame is not None
            yield frame

    def release(self) -> None:
        self._stop.set()
        with self._new_frame:
            self._new_frame.notify_all()
        self._thread.join(timeout=2.0)
        self._cap.release()

    def __enter__(self) -> "Webcam":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
