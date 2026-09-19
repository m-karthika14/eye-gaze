"""Phase 6 (product): live key resizing driven by the local character LM's
predicted next-character likelihoods -- a simplified SliceType-style merging
keyboard (MASTER_PROJECT_DOCUMENT.md Section 3.2, Section 9).

Reflows each row's key WIDTHS proportionally to P(next char | typed text so
far); row y-positions and each row's overall left/right bounds are left
untouched, so this never invalidates Phase 3's calibration (derived against
the existing row geometry) and never affects /decoder's research code.

Product-layer only. S1-S4 (experiments/s1.py-s4.py) always decode against
the static config/keyboard_layout.json, per Section 10's requirement that
both conditions in the S1-vs-S4 comparison see an identical static keyboard
-- they never import this module.
"""

from __future__ import annotations

from decoder.char_lm import CharLM
from decoder.nearest_key import KeyboardLayout, KeyRegion
from decoder.text_buffer import TextBuffer

# Floor on a key's LM weight (well below the ~1/27 uniform average) so a
# genuinely unlikely key still shrinks noticeably without vanishing entirely.
MIN_WEIGHT = 0.01
# BACKSPACE has no character/LM equivalent -- treated as a neutral,
# average-likelihood key so it neither dominates nor disappears.
BACKSPACE_WEIGHT = 1.0 / 27
# Dampens raw probability -> size (weight**TEMPERATURE before normalizing).
# Pure proportional sizing (temperature=1.0) lets one strongly-dominant
# prediction balloon to >50% of a row, squeezing the other keys too small to
# comfortably target if that prediction turns out wrong. 0.5 (sqrt) keeps
# likely keys visibly bigger without that extreme.
TEMPERATURE = 0.5


def compute_adaptive_layout(base_layout: KeyboardLayout, char_lm: CharLM, typed_text: str) -> KeyboardLayout:
    """Returns a new KeyboardLayout with each row's key widths reflowed by
    next-character likelihood given `typed_text`. Row y/height unchanged."""
    next_probs = char_lm.next_char_probs(typed_text)

    rows: dict[float, list[KeyRegion]] = {}
    for key in base_layout.keys:
        rows.setdefault(key.y, []).append(key)

    new_keys: list[KeyRegion] = []
    for row_keys in rows.values():
        row_keys_sorted = sorted(row_keys, key=lambda k: k.x)
        row_left = min(k.x for k in row_keys_sorted)
        row_right = max(k.x + k.width for k in row_keys_sorted)
        row_width = row_right - row_left

        weights = []
        for key in row_keys_sorted:
            if key.id == "BACKSPACE":
                weight = BACKSPACE_WEIGHT
            elif key.id == "SPACE":
                weight = max(next_probs.get(" ", MIN_WEIGHT), MIN_WEIGHT)
            else:
                weight = max(next_probs.get(key.id.lower(), MIN_WEIGHT), MIN_WEIGHT)
            weights.append(weight**TEMPERATURE)
        total_weight = sum(weights)

        x = row_left
        for key, weight in zip(row_keys_sorted, weights):
            width = row_width * (weight / total_weight)
            new_keys.append(KeyRegion(id=key.id, x=x, y=key.y, width=width, height=key.height))
            x += width

    return KeyboardLayout(keys=new_keys)


def accept_suggestion(buffer: TextBuffer, pending_prefix: str, suggestion: str) -> None:
    """Replaces the word-in-progress (the last `pending_prefix` characters of
    `buffer`, which SPACE has just followed) with `suggestion` -- the
    gaze-driven acceptance mechanism for Phase 6's word suggestions (see
    module docstring: reuses the existing, already-calibrated SPACE key
    rather than adding a new gaze-interactive screen region).

    Call this AFTER SPACE has already been applied to the buffer via the
    normal S0 dwell flow (so buffer currently ends with `pending_prefix + " "`);
    it undoes that prefix and space, then types the suggestion followed by a
    space, using the same TextBuffer.apply_key() primitives S0 already uses.
    """
    for _ in range(len(pending_prefix) + 1):  # + 1 for the space just applied
        buffer.apply_key("BACKSPACE")
    for char in suggestion:
        buffer.apply_key("SPACE" if char == " " else char.upper())
    buffer.apply_key("SPACE")
