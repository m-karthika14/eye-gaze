"""Checks for the live S0-S3 decoders (decoder/live_modes.py). No camera.

    .venv/Scripts/python.exe -m experiments.test_live_modes
"""

from __future__ import annotations

from decoder.live_modes import LIVE_DECODERS, S0LiveDecoder, S1LiveDecoder, S2LiveDecoder, S3LiveDecoder
from decoder.nearest_key import KeyboardLayout

LAYOUT = KeyboardLayout.from_json()
CENTRES = {k.id: k.center() for k in LAYOUT.keys}
RECTS = {k.id: k for k in LAYOUT.keys}


class Clock:
    t = 0.0


def press(dec, key_id: str, offset: tuple[float, float] = (0.0, 0.0), at: tuple[float, float] | None = None) -> None:
    """Look at `key_id` (or at an explicit point), then confirm (blink / Space-Enter style): feed a short fixation, then select_now()."""
    dec.dwell_enabled = False
    x, y = at if at is not None else (CENTRES[key_id][0] + offset[0], CENTRES[key_id][1] + offset[1])
    for _ in range(8):
        Clock.t += 0.05
        dec.process_sample(x, y, Clock.t)
    dec.select_now(Clock.t)


def type_word(dec, word: str, space: bool = True) -> None:
    for ch in word:
        press(dec, "SPACE" if ch == " " else ch.upper())
    if space:
        press(dec, "SPACE")


def main() -> None:
    assert set(LIVE_DECODERS) == {"s0", "s1", "s2", "s3"}
    print("[OK] live decoders registered for s0, s1, s2, s3 (s4 is decoder/s4_live.py)")

    # ---- S0: nearest key, no correction
    d = S0LiveDecoder(layout=LAYOUT)
    type_word(d, "famuly")
    assert d.current_text == "famuly ", d.current_text
    press(d, "BACKSPACE")  # deletes the space char? (committed text ends with ' ') -> removes it
    assert d.current_text == "famuly", d.current_text
    press(d, "BACKSPACE")
    assert d.current_text == "famul", d.current_text
    print("[OK] S0 types exactly the selected keys, backspace deletes one character")

    # ---- S1: post-hoc correction of the whole word at SPACE
    d = S1LiveDecoder(layout=LAYOUT)
    for ch in "famuly":
        press(d, ch.upper())
    assert d.active_word == "famuly" and d.candidates[0] == "family" and "famuly" in d.candidates, (d.active_word, d.candidates)
    press(d, "SPACE")
    assert d.current_text == "family ", d.current_text
    print("[OK] S1 shows the typed word while typing, offers the correction first, and commits it at SPACE ('famuly' -> 'family')")

    d = S1LiveDecoder(layout=LAYOUT)
    type_word(d, "hello")
    assert d.current_text == "hello ", d.current_text
    d2 = S1LiveDecoder(layout=LAYOUT)
    type_word(d2, "water please")
    assert d2.current_text == "water please ", d2.current_text
    print("[OK] S1 leaves correct words alone")

    d = S1LiveDecoder(layout=LAYOUT)
    for ch in "famuly":
        press(d, ch.upper())
    assert d.pick_candidate(1) and d.current_text == "famuly ", d.current_text  # user overrides the correction with exactly-what-was-typed
    print("[OK] S1: the user can pick the uncorrected word instead")

    # ---- S2: identical decoding, adaptive keyboard flag
    d = S2LiveDecoder(layout=LAYOUT)
    assert d.adaptive_layout and not S0LiveDecoder(layout=LAYOUT).adaptive_layout and not S1LiveDecoder(layout=LAYOUT).adaptive_layout
    type_word(d, "famuly")
    assert d.current_text == "family "
    print("[OK] S2 = S1 decoding + adaptive_layout flag (the backend resizes keys when it is on)")

    # ---- S3: online bias
    d = S3LiveDecoder(layout=LAYOUT)
    assert d.bias == [0.0, 0.0]
    # (a) backspace-and-retry. The user looks at the RIGHT EDGE of K but means L (the key to its right): K gets selected,
    #     they press BACKSPACE, look at L properly. Their gaze was left of the intended key -> negative x bias.
    k = RECTS["K"]
    edge = (k.x + k.width - 0.004, k.y + k.height / 2)
    press(d, "K", at=edge)
    assert d.active_word == "k"
    press(d, "BACKSPACE")
    assert d.active_word == "" and d._deleted is not None
    press(d, "L")
    assert d.active_word == "l"
    expected = 0.25 * (edge[0] - CENTRES["L"][0])
    assert abs(d.bias[0] - expected) < 1e-6 and d.bias[0] < 0 and abs(d.bias[1]) < 0.01, d.bias
    print(f"[OK] S3 learns from BACKSPACE-and-retry: bias x = {d.bias[0]:+.3f} (gaze was left of the key the user meant)")

    # the bias is applied to later gaze: the same raw edge-of-K gaze is now pulled towards L
    d.reset_probe = None
    press(d, "SPACE")
    before_text = d.current_text
    d.process_sample(*edge, Clock.t + 0.05)
    assert d.process_sample(edge[0] + 0.03, edge[1], Clock.t + 0.1).hovered_key in ("K", "L")
    print("[OK] learned bias is subtracted from subsequent gaze (hover moves toward the intended neighbour)")

    # (b) LM correction of a real word: 'famuly' -> 'family'; the 'u' gaze vs the 'i' it was changed to
    d = S3LiveDecoder(layout=LAYOUT)
    type_word(d, "famuly")
    assert d.current_text == "family "
    expected_x = 0.25 * (CENTRES["U"][0] - CENTRES["I"][0])
    assert abs(d.bias[0] - expected_x) < 1e-6, (d.bias, expected_x)
    print(f"[OK] S3 learns from an LM correction of a real word: bias x = {d.bias[0]:+.3f}")

    # (c) correctly typed words teach nothing; wild misses and non-words are ignored; the bias is clipped
    d = S3LiveDecoder(layout=LAYOUT)
    type_word(d, "hello water please")
    assert d.bias == [0.0, 0.0], d.bias
    d = S3LiveDecoder(layout=LAYOUT)
    d._learn((0.9, 0.0))  # a wild miss: not a bias
    assert d.bias == [0.0, 0.0]
    for _ in range(50):
        d._learn((0.2, -0.2))
    assert abs(d.bias[0]) <= d.bias_clip + 1e-12 and abs(d.bias[1]) <= d.bias_clip + 1e-12
    print("[OK] S3 ignores correctly-typed words and wild misses, and clips the bias")

    d.reset()
    assert d.bias == [0.0, 0.0] and d.current_text == ""
    s = S3LiveDecoder(layout=LAYOUT)
    s.set_text("hi ")
    assert s.current_text == "hi "
    print("[OK] reset() clears the learned bias; set_text() carries text across a mode switch")

    print("\n[OK] all live S0-S3 checks passed")


if __name__ == "__main__":
    main()
