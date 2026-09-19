"""Shared noisy-channel beam search: P(T|G) ~ P(G|T) * P(T)
(MASTER_PROJECT_DOCUMENT.md Section 8, Stage 5; Section 6, C2).

Used by both S1 (a generic, position-blind confusion likelihood) and S4 (the
real per-observation Gaussian key-probability likelihood) -- same search
algorithm, different P(G|T) source. That's deliberate: it isolates whether
preserving real gaze uncertainty helps, rather than conflating it with
"using a language model helps" (S1 already uses the LM too).
"""

from __future__ import annotations

import math

from .char_lm import CharLM


def beam_search_decode(
    char_likelihoods: list[dict[str, float]],
    char_lm: CharLM,
    beam_width: int = 8,
) -> str:
    """char_likelihoods[i] maps a candidate character to P(G_i | that
    character) for the i-th gaze observation. Returns the text with the
    highest Sum_i [log P(G_i|T_i) + log P(T_i | T_<i)]."""
    beams: list[tuple[str, float]] = [("", 0.0)]

    for likelihoods in char_likelihoods:
        candidates: list[tuple[str, float]] = []
        for text, score in beams:
            next_probs = char_lm.next_char_probs(text)
            for char, gaze_p in likelihoods.items():
                if gaze_p <= 0.0:
                    continue
                new_score = score + math.log(gaze_p) + math.log(next_probs[char])
                candidates.append((text + char, new_score))
        candidates.sort(key=lambda item: -item[1])
        beams = candidates[:beam_width]

    if not beams:
        return ""
    return max(beams, key=lambda item: item[1])[0]
