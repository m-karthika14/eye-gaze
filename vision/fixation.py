"""Fixation/dwell temporal aggregation (Phase 3, Stage 2/4): groups raw
30fps gaze samples into a fixation window and aggregates them into one
(mu, sigma) selection-event distribution, instead of trusting a single
per-frame point.

The key idea: during a genuine fixation, sample-to-sample scatter around the
mean IS the gaze noise for that moment -- lighting, face angle, camera
distance, and how still the person is all vary fixation to fixation, so this
gives a per-event uncertainty estimate rather than one fixed global number.
This is deliberately fed the RAW (pre-One-Euro-Filter) mapped point from
GazeToScreenMapper: the filter's smoothing would artificially shrink the
observed spread, underestimating true noise.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass


@dataclass
class FixationEvent:
    mu_x: float
    mu_y: float
    sigma_x: float
    sigma_y: float
    n_samples: int
    duration_ms: float


class FixationAggregator:
    def __init__(
        self,
        min_window_ms: float = 200.0,
        max_window_ms: float = 800.0,
        fixation_radius: float = 0.06,
    ) -> None:
        """min/max_window_ms: see config/decoder_params.yaml `fixation`.
        fixation_radius: normalized-screen distance from the current buffer's
        running mean beyond which a new sample is treated as the start of a
        new fixation rather than continued scatter within the same one --
        roughly half a key's width by default.
        """
        self.min_window_ms = min_window_ms
        self.max_window_ms = max_window_ms
        self.fixation_radius = fixation_radius
        self._buffer: deque[tuple[float, float, float]] = deque()  # (x, y, t)

    def reset(self) -> None:
        self._buffer.clear()

    def update(self, x: float | None, y: float | None, timestamp: float) -> FixationEvent | None:
        """Feed one raw (x, y, timestamp) sample. Returns a FixationEvent
        summarizing the current stable window once it spans >= min_window_ms,
        recomputed (and refined) on every subsequent sample until the gaze
        moves away or is lost -- else None.
        """
        if x is None or y is None:
            self._buffer.clear()
            return None

        if self._buffer:
            mean_x = statistics.fmean(s[0] for s in self._buffer)
            mean_y = statistics.fmean(s[1] for s in self._buffer)
            dist = math.hypot(x - mean_x, y - mean_y)
            if dist > self.fixation_radius:
                # Gaze jumped to a new location -- this is a new fixation.
                self._buffer.clear()

        self._buffer.append((x, y, timestamp))

        # Slide the window: drop samples older than max_window_ms.
        newest_t = self._buffer[-1][2]
        while self._buffer and (newest_t - self._buffer[0][2]) * 1000.0 > self.max_window_ms:
            self._buffer.popleft()

        if len(self._buffer) < 2:
            return None

        window_ms = (self._buffer[-1][2] - self._buffer[0][2]) * 1000.0
        if window_ms < self.min_window_ms:
            return None

        xs = [s[0] for s in self._buffer]
        ys = [s[1] for s in self._buffer]
        return FixationEvent(
            mu_x=statistics.fmean(xs),
            mu_y=statistics.fmean(ys),
            sigma_x=statistics.pstdev(xs),
            sigma_y=statistics.pstdev(ys),
            n_samples=len(self._buffer),
            duration_ms=window_ms,
        )
