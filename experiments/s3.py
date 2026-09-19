"""S3: S2 + online user adaptation, no explicit calibration
(MASTER_PROJECT_DOCUMENT.md Section 10).

Online bias adaptation is Phase 7's job (learns from typed-text read-back
moments during live use). Until then, this is the explicitly-sanctioned
stub (Phase 5 spec: "stub is acceptable until Phase 7 is complete") -- a
bias-correction hook defaulting to (0, 0), i.e. a no-op. Phase 7 replaces
the defaults with a real per-user learned bias.
"""

from __future__ import annotations

from decoder.beam_search import beam_search_decode
from decoder.char_lm import CharLM
from decoder.nearest_key import KeyboardLayout
from experiments.gaze_sequence import GazeObservation
from experiments.s2 import S2System


class S3System(S2System):
    def __init__(
        self,
        layout: KeyboardLayout | None = None,
        char_lm: CharLM | None = None,
        beam_width: int = 8,
        bias_x: float = 0.0,
        bias_y: float = 0.0,
    ) -> None:
        super().__init__(layout=layout, char_lm=char_lm, beam_width=beam_width)
        self.bias_x = bias_x
        self.bias_y = bias_y

    def _apply_bias(self, observations: list[GazeObservation]) -> list[GazeObservation]:
        return [
            GazeObservation(
                mu_x=obs.mu_x - self.bias_x,
                mu_y=obs.mu_y - self.bias_y,
                sigma_x=obs.sigma_x,
                sigma_y=obs.sigma_y,
                true_char=obs.true_char,
            )
            for obs in observations
        ]

    def run_sequence(self, observations: list[GazeObservation]) -> str:
        corrected = self._apply_bias(observations)
        likelihoods = self._char_likelihoods(corrected)
        return beam_search_decode(likelihoods, self.char_lm, self.beam_width)
