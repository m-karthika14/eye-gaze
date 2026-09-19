"""S1: hard-selected key -> post-hoc local character-LM correction
(MASTER_PROJECT_DOCUMENT.md Section 10).

Gaze is collapsed to a single hard-selected character BEFORE language ever
sees it, so the "correction" step can only use a GENERIC, position-blind
confusion model (any non-selected character equally plausible) -- the real
per-observation gaze likelihood has already been discarded by the time
language runs. That's exactly the discarded information S4 keeps instead;
the difference between them is what isolates H1.
"""

from __future__ import annotations

from decoder.beam_search import beam_search_decode
from decoder.char_lm import ALPHABET, CharLM
from decoder.nearest_key import KeyboardLayout, nearest_key
from experiments.gaze_sequence import GazeObservation, key_id_to_char

DEFAULT_MATCH_PROB = 0.85  # generic confidence in the hard-selected character


def hard_select_char(obs: GazeObservation, layout: KeyboardLayout) -> str:
    key_id = nearest_key(obs.mu_x, obs.mu_y, layout)
    return key_id_to_char(key_id) if key_id is not None and key_id != "BACKSPACE" else " "


def generic_confusion_likelihoods(hard_char: str, match_prob: float = DEFAULT_MATCH_PROB) -> dict[str, float]:
    """A FIXED confusion matrix, the same regardless of how close other keys
    actually were in this specific observation -- unlike S4's real
    per-observation likelihood, this only knows "this char was selected,
    typos happen sometimes," nothing about which other chars were plausible."""
    other_prob = (1.0 - match_prob) / (len(ALPHABET) - 1)
    return {c: (match_prob if c == hard_char else other_prob) for c in ALPHABET}


class S1System:
    def __init__(
        self,
        layout: KeyboardLayout | None = None,
        char_lm: CharLM | None = None,
        beam_width: int = 8,
    ) -> None:
        self.layout = layout or KeyboardLayout.from_json()
        self.char_lm = char_lm or CharLM.load_default()
        self.beam_width = beam_width

    def _char_likelihoods(self, observations: list[GazeObservation]) -> list[dict[str, float]]:
        raw_chars = [hard_select_char(obs, self.layout) for obs in observations]
        return [generic_confusion_likelihoods(c) for c in raw_chars]

    def run_sequence(self, observations: list[GazeObservation]) -> str:
        likelihoods = self._char_likelihoods(observations)
        return beam_search_decode(likelihoods, self.char_lm, self.beam_width)
