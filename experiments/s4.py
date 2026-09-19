"""S4: S3 + joint gaze-probability + language decoding -- the core research
method (MASTER_PROJECT_DOCUMENT.md Section 6 C2, Section 10).

P(T|G) ~ P(G|T) * P(T): unlike S1-S3, gaze is NEVER collapsed to a hard
choice before language sees it. Each observation's REAL per-key Gaussian
probability distribution (decoder/key_probability.py, from that
observation's ACTUAL mu/sigma) is what beam search weighs against the
character LM at every step -- not a generic confusion matrix. This is the
one comparison that isolates whether preserving gaze uncertainty helps
(H1), not just whether "using an LM" helps (S1 already uses one too).
"""

from __future__ import annotations

from decoder.key_probability import key_probabilities
from decoder.nearest_key import KeyboardLayout
from experiments.gaze_sequence import GazeObservation, key_id_to_char
from experiments.s3 import S3System


def real_gaze_likelihoods(obs: GazeObservation, layout: KeyboardLayout) -> dict[str, float]:
    """Converts the observation's actual Gaussian into a per-CHARACTER
    likelihood. key_probability.py works in key-id space; BACKSPACE has no
    character equivalent in this fixed-length decoding formulation, so it's
    dropped and the remaining mass renormalized."""
    key_probs = key_probabilities(obs.mu_x, obs.mu_y, obs.sigma_x, obs.sigma_y, layout)
    key_probs.pop("BACKSPACE", None)
    total = sum(key_probs.values())
    return {key_id_to_char(key_id): p / total for key_id, p in key_probs.items()}


class S4System(S3System):
    def _char_likelihoods(self, observations: list[GazeObservation]) -> list[dict[str, float]]:
        return [real_gaze_likelihoods(obs, self.layout) for obs in observations]
