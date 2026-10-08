"""Deliberate-blink detection for blink-to-select.

Eye openness comes from the eye-contour points of a face-mesh landmark model
(uniface.landmark.FaceMesh, MediaPipe-compatible 468-point indices) as the
standard eye aspect ratio:

    EAR = (|upper1 - lower1| + |upper2 - lower2|) / (2 * |outer corner - inner corner|)

~0.25-0.35 with the eye open, falling towards 0 as the lid closes. The two eyes
are averaged. "Closed" is judged against a per-user, self-adapting open-eye
baseline rather than a fixed threshold, since EAR varies with face shape,
camera angle and glasses.

Natural blinks last ~100-300ms; a select needs the eyes held closed for at
least `min_ms` (default 450ms), so ordinary blinking never types anything.
It fires the moment that duration is reached (not on re-opening, which would
add another ~200ms of lag), once per closure.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# MediaPipe face-mesh indices: (outer corner, inner corner, upper1, upper2, lower1, lower2)
LEFT_EYE = (33, 133, 160, 158, 144, 153)
RIGHT_EYE = (362, 263, 385, 387, 380, 373)


def _eye_ear(pts: np.ndarray, idx: tuple[int, int, int, int, int, int]) -> float:
    p_out, p_in, u1, u2, l1, l2 = (pts[i] for i in idx)
    width = float(np.linalg.norm(p_out - p_in))
    if width < 1e-6:
        return 0.0
    return float((np.linalg.norm(u1 - l1) + np.linalg.norm(u2 - l2)) / (2.0 * width))


def eye_aspect_ratio(points_2d: np.ndarray) -> float:
    """Mean EAR of both eyes from a (468+, 2) landmark array."""
    return 0.5 * (_eye_ear(points_2d, LEFT_EYE) + _eye_ear(points_2d, RIGHT_EYE))


@dataclass
class BlinkState:
    eyes_closed: bool  # currently below the closed threshold -- gaze is unreliable, hold the cursor
    progress: float  # 0..1 of min_ms elapsed in the current closure (for UI feedback)
    fired: bool  # True on exactly the sample where a deliberate blink completes its hold time
    closed_since: float | None  # timestamp the current closure began


class BlinkDetector:
    def __init__(
        self,
        min_ms: float = 450.0,
        close_ratio: float = 0.62,
        open_ratio: float = 0.75,
        baseline_window: int = 90,
        min_baseline: float = 0.12,
    ) -> None:
        """close_ratio/open_ratio: EAR as a fraction of the open-eye baseline
        below which the eye counts as closed / above which it counts as open
        again (hysteresis, so flicker around one threshold can't split a
        closure in two)."""
        self.min_ms = min_ms
        self.close_ratio = close_ratio
        self.open_ratio = open_ratio
        self.min_baseline = min_baseline
        self._open_ears: deque[float] = deque(maxlen=baseline_window)
        self._closed = False
        self._closed_since: float | None = None
        self._fired = False

    def reset(self) -> None:
        self._closed = False
        self._closed_since = None
        self._fired = False

    @property
    def baseline(self) -> float | None:
        if len(self._open_ears) < 10:
            return None
        return float(np.percentile(self._open_ears, 80))

    def update(self, ear: float | None, t: float) -> BlinkState:
        """Feed one frame's EAR (None = no face/landmarks) at time t (seconds)."""
        if ear is None:
            self.reset()
            return BlinkState(False, 0.0, False, None)

        base = self.baseline
        if base is None:
            self._open_ears.append(ear)  # still learning what "open" looks like
            return BlinkState(False, 0.0, False, None)

        threshold = base * (self.open_ratio if self._closed else self.close_ratio)
        is_closed = base >= self.min_baseline and ear < threshold

        if not is_closed:
            self._open_ears.append(ear)  # only open-eye frames update the baseline
            self.reset()
            return BlinkState(False, 0.0, False, None)

        if not self._closed:
            self._closed = True
            self._closed_since = t
            self._fired = False
        assert self._closed_since is not None
        held_ms = (t - self._closed_since) * 1000.0
        fired = False
        if held_ms >= self.min_ms and not self._fired:
            self._fired = True
            fired = True
        return BlinkState(True, min(held_ms / self.min_ms, 1.0), fired, self._closed_since)


# ---------------------------------------------------------------------------
# Mouth-open gesture: a second hands-free "click". Unlike a blink it doesn't
# interfere with the eyes, so gaze stays usable and the gesture is large and easy
# to see on a webcam. Mouth aspect ratio from the inner-lip points:
#     MAR = |upper inner lip - lower inner lip| / |left corner - right corner|
# ~0.0-0.05 closed, >0.3 clearly open.
# ---------------------------------------------------------------------------
MOUTH = (13, 14, 78, 308)  # upper inner lip, lower inner lip, left corner, right corner


def mouth_aspect_ratio(points_2d: np.ndarray) -> float:
    up, lo, left, right = (points_2d[i] for i in MOUTH)
    width = float(np.linalg.norm(left - right))
    if width < 1e-6:
        return 0.0
    return float(np.linalg.norm(up - lo) / width)


@dataclass
class GestureState:
    active: bool  # gesture currently held
    progress: float  # 0..1 of min_ms elapsed
    fired: bool  # True on exactly the sample where the hold time is reached
    since: float | None  # timestamp the current hold began


class MouthOpenDetector:
    def __init__(self, min_ms: float = 400.0, open_delta: float = 0.14, release_delta: float = 0.08, min_open: float = 0.18, baseline_window: int = 90) -> None:
        """Mouth counts as open when MAR exceeds the resting (closed-mouth) baseline by
        open_delta (and is at least min_open); it stays "open" until it drops below
        baseline + release_delta (hysteresis). Holding it open for min_ms fires once."""
        self.min_ms = min_ms
        self.open_delta = open_delta
        self.release_delta = release_delta
        self.min_open = min_open
        self._rest: deque[float] = deque(maxlen=baseline_window)
        self._open = False
        self._since: float | None = None
        self._fired = False

    def reset(self) -> None:
        self._open = False
        self._since = None
        self._fired = False

    @property
    def baseline(self) -> float | None:
        if len(self._rest) < 10:
            return None
        return float(np.percentile(self._rest, 30))

    def update(self, mar: float | None, t: float) -> GestureState:
        if mar is None:
            self.reset()
            return GestureState(False, 0.0, False, None)
        base = self.baseline
        if base is None:
            self._rest.append(mar)
            return GestureState(False, 0.0, False, None)
        threshold = base + (self.release_delta if self._open else self.open_delta)
        is_open = mar > max(threshold, self.min_open if not self._open else 0.0)
        if not is_open:
            self._rest.append(mar)  # only closed-mouth frames define the resting level
            self.reset()
            return GestureState(False, 0.0, False, None)
        if not self._open:
            self._open = True
            self._since = t
            self._fired = False
        assert self._since is not None
        held_ms = (t - self._since) * 1000.0
        fired = held_ms >= self.min_ms and not self._fired
        if fired:
            self._fired = True
        return GestureState(True, min(held_ms / self.min_ms, 1.0), fired, self._since)
