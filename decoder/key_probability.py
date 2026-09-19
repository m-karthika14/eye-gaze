"""Key probability mapping (Phase 3): integrates a 2D Gaussian gaze estimate
over each keyboard key's region to produce a full probability distribution,
instead of S0's hard nearest-key choice. Implements Stage 2/3 of
MASTER_PROJECT_DOCUMENT.md Section 8:

    G ~ N(mu, Sigma)
    P(K_i | G) = integral over K_i of p(g | mu, Sigma) dg

Sigma is modelled as diagonal (x/y treated independently) -- there's no
principled reason to expect correlated x/y gaze noise from an appearance
model, and it keeps the integral a simple separable product of two 1D
Gaussian interval masses (closed-form via erf, no numerical integration
needed).
"""

from __future__ import annotations

import math

from .nearest_key import KeyboardLayout

# Floor on sigma so a near-degenerate fixation (very few samples, near-zero
# observed spread) doesn't collapse the distribution to a single key with
# false certainty. See config/decoder_params.yaml gaze_uncertainty.min_sigma.
DEFAULT_MIN_SIGMA = 0.01


def _gaussian_interval_mass(a: float, b: float, mu: float, sigma: float) -> float:
    """P(a <= X <= b) for X ~ N(mu, sigma^2), via the error function."""
    z_a = (a - mu) / (sigma * math.sqrt(2))
    z_b = (b - mu) / (sigma * math.sqrt(2))
    return 0.5 * (math.erf(z_b) - math.erf(z_a))


def key_probabilities(
    mu_x: float,
    mu_y: float,
    sigma_x: float,
    sigma_y: float,
    layout: KeyboardLayout,
    min_sigma: float = DEFAULT_MIN_SIGMA,
) -> dict[str, float]:
    """Returns {key_id: probability}, normalized to sum to 1 over all keys.

    mu_x/mu_y and sigma_x/sigma_y are in the same normalized [0,1] screen
    space as config/keyboard_layout.json.
    """
    sigma_x = max(sigma_x, min_sigma)
    sigma_y = max(sigma_y, min_sigma)

    raw: dict[str, float] = {}
    for key in layout.keys:
        px = _gaussian_interval_mass(key.x, key.x + key.width, mu_x, sigma_x)
        py = _gaussian_interval_mass(key.y, key.y + key.height, mu_y, sigma_y)
        raw[key.id] = px * py

    total = sum(raw.values())
    if total <= 0.0:
        # Degenerate case (e.g. fixation far off the keyboard entirely):
        # fall back to uniform rather than dividing by zero.
        n = len(layout.keys)
        return {key_id: 1.0 / n for key_id in raw}

    return {key_id: mass / total for key_id, mass in raw.items()}
