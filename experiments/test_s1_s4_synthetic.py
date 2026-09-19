"""Synthetic regression check for Phase 5: S1-S4 all decode correctly, run
on identical gaze sequences, and show H1's predicted qualitative pattern
(S4's advantage over S1 grows as gaze noise increases -- Section 10,
Experiment 2). This is a mechanical sanity check, not the real experiment --
Phase 9 runs the actual statistically-powered H1/H2 experiments.

Run from the repo root:

    .venv/Scripts/python.exe -m experiments.test_s1_s4_synthetic
"""

from __future__ import annotations

import random
import sys

from decoder.char_lm import CharLM
from decoder.nearest_key import KeyboardLayout
from experiments.gaze_sequence import simulate_gaze_sequence
from experiments.s1 import S1System
from experiments.s2 import S2System
from experiments.s3 import S3System
from experiments.s4 import S4System

TEXTS = ["the cat sat on the mat", "i need water", "please call the doctor", "how are you today"]


def char_accuracy(true: str, predicted: str) -> float:
    return sum(a == b for a, b in zip(true, predicted)) / len(true)


def check_near_zero_noise_decodes_correctly(layout, lm) -> bool:
    s1, s4 = S1System(layout, lm), S4System(layout, lm)
    for text in TEXTS:
        obs = simulate_gaze_sequence(text, layout, noise_sigma=0.001, rng=random.Random(0))
        if s1.run_sequence(obs) != text:
            print(f"[FAIL] S1 failed to decode near-noiseless input: {text!r}")
            return False
        if s4.run_sequence(obs) != text:
            print(f"[FAIL] S4 failed to decode near-noiseless input: {text!r}")
            return False
    print("[OK] S1 and S4 both decode near-noiseless input exactly")
    return True


def check_s2_identical_to_s1_for_replay(layout, lm) -> bool:
    s1, s2 = S1System(layout, lm), S2System(layout, lm)
    obs = simulate_gaze_sequence(TEXTS[0], layout, noise_sigma=0.1, rng=random.Random(1))
    if s1.run_sequence(obs) != s2.run_sequence(obs):
        print("[FAIL] S2 should decode identically to S1 for batch/replay (see experiments/s2.py)")
        return False
    print("[OK] S2 decodes identically to S1 for batch/replay, as designed")
    return True


def check_s3_stub_is_noop_by_default(layout, lm) -> bool:
    s2, s3 = S2System(layout, lm), S3System(layout, lm)  # S3 defaults bias to (0, 0)
    obs = simulate_gaze_sequence(TEXTS[0], layout, noise_sigma=0.1, rng=random.Random(2))
    if s2.run_sequence(obs) != s3.run_sequence(obs):
        print("[FAIL] S3 with default (0,0) bias should decode identically to S2")
        return False
    print("[OK] S3's Phase-7 stub is a true no-op by default, as designed")
    return True


def check_s3_bias_correction_actually_shifts_decoding(layout, lm) -> bool:
    """Sanity check that the bias hook is wired to something real, not dead
    code -- a nonzero bias should change the decoded output."""
    text = "the cat sat"
    obs = simulate_gaze_sequence(text, layout, noise_sigma=0.02, rng=random.Random(3))
    s3_unbiased = S3System(layout, lm, bias_x=0.0, bias_y=0.0)
    s3_biased = S3System(layout, lm, bias_x=0.3, bias_y=0.0)
    if s3_unbiased.run_sequence(obs) == s3_biased.run_sequence(obs):
        print("[FAIL] a large nonzero bias should change the decoded output")
        return False
    print("[OK] S3's bias-correction hook measurably affects decoding when nonzero")
    return True


def check_s4_advantage_grows_with_noise(layout, lm) -> bool:
    s1, s4 = S1System(layout, lm), S4System(layout, lm)
    noise_levels = [0.02, 0.08, 0.15]
    gaps = []
    for noise in noise_levels:
        s1_accs, s4_accs = [], []
        for seed in range(3):
            for text in TEXTS:
                obs = simulate_gaze_sequence(text, layout, noise_sigma=noise, rng=random.Random(seed * 100 + len(text)))
                s1_accs.append(char_accuracy(text, s1.run_sequence(obs)))
                s4_accs.append(char_accuracy(text, s4.run_sequence(obs)))
        gap = (sum(s4_accs) - sum(s1_accs)) / len(s1_accs)
        gaps.append(gap)
        print(f"[info] noise={noise:.2f}: S1={sum(s1_accs)/len(s1_accs):.3f} "
              f"S4={sum(s4_accs)/len(s4_accs):.3f} gap={gap:+.3f}")

    if not all(g >= -0.02 for g in gaps):
        print(f"[FAIL] S4 should never meaningfully lose to S1: gaps={gaps}")
        return False
    if not (gaps[-1] > gaps[0]):
        print(f"[FAIL] S4's advantage should grow from low to higher noise: gaps={gaps}")
        return False
    print(f"[OK] S4 >= S1 at every noise level tested, and the advantage grows with noise: {gaps}")
    return True


def main() -> int:
    layout = KeyboardLayout.from_json()
    lm = CharLM.load_default()

    checks = [
        lambda: check_near_zero_noise_decodes_correctly(layout, lm),
        lambda: check_s2_identical_to_s1_for_replay(layout, lm),
        lambda: check_s3_stub_is_noop_by_default(layout, lm),
        lambda: check_s3_bias_correction_actually_shifts_decoding(layout, lm),
        lambda: check_s4_advantage_grows_with_noise(layout, lm),
    ]
    results = [check() for check in checks]
    if all(results):
        print(f"\n[OK] all {len(results)} S1-S4 checks passed")
        return 0
    print(f"\n[FAIL] {results.count(False)}/{len(results)} checks failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
