"""Synthetic regression check for the S0 baseline: replays a fabricated gaze
path (dwelling at each key's center in turn) through S0System and asserts the
decoded text matches. No webcam needed -- exercises nearest_key + dwell +
text_buffer together.

Run from the repo root:

    .venv/Scripts/python.exe -m experiments.test_s0_synthetic
"""

from __future__ import annotations

import sys

from decoder.nearest_key import KeyboardLayout
from experiments.s0 import S0System

SAMPLE_HZ = 30.0
DWELL_MS = 500.0


def synthetic_samples_for_word(layout: KeyboardLayout, keys: list[str]):
    """Yields (x, y, timestamp) dwelling at each key's center for comfortably
    longer than the dwell threshold, with a short off-keyboard gap between
    keys so consecutive identical letters/re-selections behave correctly."""
    key_centers = {k.id: k.center() for k in layout.keys}
    t = 0.0
    dt = 1.0 / SAMPLE_HZ
    dwell_frames = int((DWELL_MS / 1000.0) * SAMPLE_HZ) + 4
    gap_frames = 3

    for key_id in keys:
        cx, cy = key_centers[key_id]
        for _ in range(dwell_frames):
            yield (cx, cy, t)
            t += dt
        for _ in range(gap_frames):
            yield (None, None, t)  # brief gaze-away so the next dwell starts fresh
            t += dt


def main() -> int:
    layout = KeyboardLayout.from_json()
    system = S0System(layout=layout, dwell_ms=DWELL_MS)

    word_keys = ["H", "E", "L", "L", "O"]
    samples = list(synthetic_samples_for_word(layout, word_keys))
    result = system.run_sequence(samples)

    expected = "hello"
    if result != expected:
        print(f"[FAIL] expected {expected!r}, got {result!r}")
        return 1
    print(f"[OK] S0 decoded {word_keys} -> {result!r}")

    # Also check BACKSPACE correction: type "HELP", backspace, then "LO" -> "hello"
    system2 = S0System(layout=layout, dwell_ms=DWELL_MS)
    keys2 = ["H", "E", "L", "P", "BACKSPACE", "L", "O"]
    result2 = system2.run_sequence(synthetic_samples_for_word(layout, keys2))
    if result2 != expected:
        print(f"[FAIL] backspace case: expected {expected!r}, got {result2!r}")
        return 1
    print(f"[OK] S0 decoded {keys2} -> {result2!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
