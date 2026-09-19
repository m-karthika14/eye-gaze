"""Shared gaze-sequence format for the S0-S4 comparison (Phase 5): one
GazeObservation per intended character, so every system can be run against
identical recorded or simulated data (MASTER_PROJECT_DOCUMENT.md Section 10).

S0-S4 model a fixed-length decoding problem -- one gaze observation per
intended character, no insert/delete -- distinct from S0's own live
interactive dwell-and-backspace loop (experiments/s0.py), which is a
separate, already-validated mechanic. Phase 9 replaces simulate_gaze_sequence
with real recorded multi-character sequences; this is the mechanical/
regression-testing tool until then.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from decoder.nearest_key import KeyboardLayout

CHAR_TO_KEY = {chr(c): chr(c).upper() for c in range(ord("a"), ord("z") + 1)}
CHAR_TO_KEY[" "] = "SPACE"
KEY_TO_CHAR = {key_id: char for char, key_id in CHAR_TO_KEY.items()}


def char_to_key_id(char: str) -> str:
    return CHAR_TO_KEY[char]


def key_id_to_char(key_id: str) -> str:
    return KEY_TO_CHAR[key_id]


@dataclass
class GazeObservation:
    mu_x: float
    mu_y: float
    sigma_x: float
    sigma_y: float
    true_char: str | None = None  # ground truth, for evaluation only -- never fed to a decoder


def simulate_gaze_sequence(
    text: str,
    layout: KeyboardLayout,
    noise_sigma: float,
    rng: random.Random | None = None,
) -> list[GazeObservation]:
    """One noisy Gaussian gaze observation per character of `text`
    (lowercase a-z and space only), centered on that character's true key
    with isotropic noise `noise_sigma`."""
    rng = rng or random.Random()
    key_by_id = {k.id: k for k in layout.keys}
    observations = []
    for char in text:
        key_id = char_to_key_id(char)
        cx, cy = key_by_id[key_id].center()
        mu_x = cx + rng.gauss(0, noise_sigma)
        mu_y = cy + rng.gauss(0, noise_sigma)
        observations.append(
            GazeObservation(mu_x=mu_x, mu_y=mu_y, sigma_x=noise_sigma, sigma_y=noise_sigma, true_char=char)
        )
    return observations
