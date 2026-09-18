"""One Euro Filter (Casiez, Roussel & Vogel, CHI 2012) -- adaptive low-pass
filter for noisy interactive signals: heavy smoothing when the signal is
near-stationary (stable during a fixation/dwell), fast response when it's
moving quickly (no lag during a saccade to a new target). Used in place of a
flat exponential moving average, which forces a single lag/jitter trade-off
regardless of whether the user is fixating or actively moving their gaze.

Reference: https://cristal.univ-lille.fr/~casiez/1euro/
"""

from __future__ import annotations

import math


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def _exponential_smoothing(a: float, x: float, x_prev: float) -> float:
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    """Filters a single scalar signal sampled at irregular timestamps.

    min_cutoff: cutoff frequency when the signal is still -- lower means more
        smoothing (less jitter) at rest.
    beta: how much the cutoff increases with speed -- higher means faster
        response (less lag) during quick movement.
    d_cutoff: cutoff frequency for the derivative estimate itself.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev = 0.0
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def filter(self, t: float, x: float) -> float:
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x

        t_e = t - self._t_prev
        if t_e <= 0:
            return self._x_prev

        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self._x_prev) / t_e
        dx_hat = _exponential_smoothing(a_d, dx, self._dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exponential_smoothing(a, x, self._x_prev)

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat
