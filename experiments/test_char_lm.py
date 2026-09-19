"""Regression check for Phase 4's local character-level LM. Requires the
trained model (run decoder/train_char_lm.py first if models/char_lm.json is
missing). No webcam needed.

Run from the repo root:

    .venv/Scripts/python.exe -m experiments.test_char_lm
"""

from __future__ import annotations

import sys
from pathlib import Path

from decoder.char_lm import ALPHABET, CharLM, clean_text

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "char_lm.json"


def check_clean_text() -> bool:
    cases = [
        ("Hello, World!", "hello world"),
        ("  multiple   spaces  ", "multiple spaces"),
        ("Don't", "dont"),
        ("123 numbers 456", "numbers"),
    ]
    for raw, expected in cases:
        got = clean_text(raw)
        if got != expected:
            print(f"[FAIL] clean_text({raw!r}) = {got!r}, expected {expected!r}")
            return False
    print("[OK] clean_text strips to a-z + space as expected")
    return True


def check_distribution_valid(lm: CharLM) -> bool:
    for prefix in ["", "t", "th", "the ", "xyz", "a very long prefix indeed"]:
        probs = lm.next_char_probs(prefix)
        if set(probs.keys()) != set(ALPHABET):
            print(f"[FAIL] next_char_probs({prefix!r}) keys don't match ALPHABET")
            return False
        total = sum(probs.values())
        if abs(total - 1.0) > 1e-9:
            print(f"[FAIL] next_char_probs({prefix!r}) sums to {total}, not 1.0")
            return False
        if any(p < 0 for p in probs.values()):
            print(f"[FAIL] next_char_probs({prefix!r}) has a negative probability")
            return False
    print("[OK] next_char_probs always returns a valid distribution over ALPHABET")
    return True


def check_sensible_predictions(lm: CharLM) -> bool:
    # Each entry: prefix, character that should rank in the top few.
    cases = [
        ("th", "e"),  # "the" is overwhelmingly the most common th- word
        ("qu", "i"),  # qu- is almost always followed by a vowel (i/e/a)
    ]
    for prefix, expected_top in cases:
        probs = lm.next_char_probs(prefix)
        top = max(probs, key=probs.get)
        if top != expected_top:
            print(f"[FAIL] next_char_probs({prefix!r}): top char {top!r}, expected {expected_top!r}")
            return False
    print("[OK] common English prefixes produce the expected top next-character")
    return True


def check_log_prob_prefers_real_words(lm: CharLM) -> bool:
    real = lm.log_prob("the quick brown fox")
    scrambled = lm.log_prob("xqj zwvk brplm fxo")
    if not (real > scrambled):
        print(f"[FAIL] log_prob(real text)={real:.2f} not > log_prob(scrambled)={scrambled:.2f}")
        return False
    print(f"[OK] log_prob ranks real English above scrambled text ({real:.2f} vs {scrambled:.2f})")
    return True


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"[FAIL] {MODEL_PATH} not found -- run: .venv/Scripts/python.exe -m decoder.train_char_lm")
        return 1

    lm = CharLM.load(MODEL_PATH)

    checks = [
        check_clean_text,
        lambda: check_distribution_valid(lm),
        lambda: check_sensible_predictions(lm),
        lambda: check_log_prob_prefers_real_words(lm),
    ]
    results = [check() for check in checks]
    if all(results):
        print(f"\n[OK] all {len(results)} char LM checks passed")
        return 0
    print(f"\n[FAIL] {results.count(False)}/{len(results)} checks failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
