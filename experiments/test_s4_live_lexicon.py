"""Synthetic checks for the live S4 decoder's lexicon path (decoder/s4_live.py,
decoder/lexicon.py). No camera, no network.

    .venv/Scripts/python.exe -m experiments.test_s4_live_lexicon
"""

from __future__ import annotations

import random

from decoder.lexicon import Lexicon
from decoder.nearest_key import KeyboardLayout
from decoder.s4_live import S4LiveDecoder
from experiments.gaze_sequence import GazeObservation

LAYOUT = KeyboardLayout.from_json()
CENTERS = {k.id: k.center() for k in LAYOUT.keys}
WORDS = ["hello", "water", "please", "help", "thank", "family", "hungry", "tired", "today", "yes"]


def dwell_word(decoder: S4LiveDecoder, word: str, err: float, rng: random.Random, fps: float = 20.0) -> None:
    """Drive process_sample like the live app: ~1s of steady gaze on each key
    (offset from its centre by a per-dwell error), then a short gap with no
    gaze (a blink / look-away) so the next key starts a fresh fixation instead
    of the eye "dwelling" on whatever it passed through."""
    t = 0.0
    dt = 1.0 / fps
    for ch in word:
        cx, cy = CENTERS[ch.upper()]
        ox, oy = rng.gauss(0, err), rng.gauss(0, err)
        for _ in range(int(1.0 * fps)):
            t += dt
            decoder.process_sample(cx + ox + rng.gauss(0, 0.004), cy + oy + rng.gauss(0, 0.004), t)
        for _ in range(int(0.8 * fps)):
            t += dt
            decoder.process_sample(None, None, t)


def word_accuracy(make_decoder, err: float, trials: int = 12) -> float:
    rng = random.Random(11)
    hits = total = 0
    for _ in range(trials):
        for w in WORDS:
            d = make_decoder()
            dwell_word(d, w, err, rng)
            hits += d.active_word == w
            total += 1
    return hits / total


def main() -> None:
    lexicon = Lexicon.load_default()
    assert lexicon is not None, "models/lexicon.json missing -- run `python -m decoder.lexicon`"
    assert lexicon.logp("hello") > lexicon.oov_logp
    print(f"[OK] lexicon loaded: {len(lexicon.word_logp)} words")

    # 1. Exact gaze -> exact word, through the full dwell machinery.
    d = S4LiveDecoder(lexicon=lexicon)
    dwell_word(d, "hello", 0.0, random.Random(1))
    assert d.active_word == "hello", d.active_word
    assert d.candidates[0] == "hello"
    print("[OK] exact gaze on h-e-l-l-o decodes to 'hello' via process_sample")

    # 2. The headline: at ~one-key gaze error the lexicon decoder must beat
    #    the char-LM-only decoder (an empty Lexicon degrades to beam search only).
    err = 0.08
    with_lex = word_accuracy(lambda: S4LiveDecoder(lexicon=lexicon), err)
    beam_only = word_accuracy(lambda: S4LiveDecoder(lexicon=Lexicon({})), err)
    assert with_lex > beam_only + 0.15, (with_lex, beam_only)
    assert with_lex >= 0.5, with_lex
    print(f"[OK] at per-dwell error {err}: lexicon word accuracy {with_lex:.2f} vs char-beam-only {beam_only:.2f}")

    # 3. Out-of-vocabulary strings with sharp gaze are not hijacked into dictionary words.
    d = S4LiveDecoder(lexicon=lexicon)
    dwell_word(d, "zxcvb", 0.0, random.Random(2))
    assert d.active_word == "zxcvb", d.active_word
    print("[OK] sharp gaze on a non-word ('zxcvb') is kept as typed, not replaced by a dictionary word")

    # 4. Alternatives are exposed and can be picked.
    d = S4LiveDecoder(lexicon=lexicon)
    dwell_word(d, "hello", 0.07, random.Random(5))
    assert 1 <= len(d.candidates) <= 3 and len(set(d.candidates)) == len(d.candidates)
    chosen = d.candidates[-1]
    assert d.pick_candidate(len(d.candidates) - 1)
    assert d.current_text == chosen + " " and d.active_word == "" and d.candidates == []
    assert not d.pick_candidate(0)
    print(f"[OK] candidates exposed and pick_candidate commits the chosen one ({chosen!r})")

    # 5. Backspace removes the last dwell and re-decodes.
    d = S4LiveDecoder(lexicon=lexicon)
    dwell_word(d, "helpx", 0.0, random.Random(3))
    n = len(d.active_word_obs)
    t0 = 1000.0
    bx, by = CENTERS["BACKSPACE"]
    for i in range(30):
        d.process_sample(bx, by, t0 + i * 0.05)
    assert len(d.active_word_obs) == n - 1 and d.active_word == "help", (d.active_word, len(d.active_word_obs))
    print("[OK] BACKSPACE drops the last dwell observation and re-decodes ('helpx' -> 'help')")

    print("\n[OK] all 5 live-decoder checks passed")


if __name__ == "__main__":
    main()
