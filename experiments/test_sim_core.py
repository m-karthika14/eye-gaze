"""Checks for the trace-driven simulation core (experiments/sim_core.py) and the stats helpers in run_results.py. No camera.

    .venv/Scripts/python.exe -m experiments.test_sim_core
"""

from __future__ import annotations

import random

import numpy as np

from experiments.run_results import bootstrap_ci, make_tuning_phrases, paired_stats
from experiments.sim_core import (
    OnlineBias,
    Params,
    cer,
    ctx,
    decode,
    gaussian_pool,
    levenshtein,
    make_observations,
    phrase_metrics,
    run_session,
    wer,
)


def main() -> None:
    # ---- metrics
    assert levenshtein("kitten", "sitting") == 3 and levenshtein("", "abc") == 3 and levenshtein("abc", "abc") == 0
    assert cer("hello", "hallo") == 0.2 and wer("i am hungry", "i am hunger") == 1 / 3
    m = phrase_metrics("i am hungry", "i am hungry")
    assert m == {"cer": 0.0, "wer": 0.0, "char_acc": 1.0, "exact": 1.0}
    print("[OK] CER / WER / char-accuracy / exact-match metrics")

    # ---- paired design: identical seed => identical observations, for every system
    pool = gaussian_pool(0.08)
    a = make_observations("i need water", pool, random.Random(5))
    b = make_observations("i need water", pool, random.Random(5))
    assert [(o.mu_x, o.mu_y) for o in a] == [(o.mu_x, o.mu_y) for o in b]
    assert [(o.mu_x, o.mu_y) for o in a] != [(o.mu_x, o.mu_y) for o in make_observations("i need water", pool, random.Random(6))]
    print("[OK] same seed -> identical observation sequence (what makes the system comparison paired)")

    # ---- noiseless gaze: every system must recover the text exactly
    quiet = gaussian_pool(1e-4)
    obs = make_observations("can you help me", quiet, random.Random(0))
    for system in ("S0", "S1", "S1P", "S2", "S3", "S4"):
        assert decode(system, obs, Params(s1_match=0.7, s1_lm=1.0)) == "can you help me", system
    print("[OK] with (almost) no noise, S0-S4 (and the S1+ baseline) all decode exactly")

    # ---- S2 decodes identically to S1; S3 equals S1 when its bias is zero
    noisy = make_observations("i am hungry", gaussian_pool(0.1), random.Random(2))
    p = Params(s1_match=0.7, s1_lm=1.0)
    assert decode("S1", noisy, p) == decode("S2", noisy, p) == decode("S3", noisy, p, bias=(0.0, 0.0))
    print("[OK] S2 == S1 decoding; S3 == S1 until a bias is learned")

    # ---- the shipped S1 defaults never override a selected key (S1 == S0): the reason S1 has to be tuned
    default = Params()
    same = 0
    total = 0
    for s in range(40):
        o = make_observations("i need water", gaussian_pool(0.1), random.Random(s))
        same += decode("S1", o, default) == decode("S0", o, default)
        total += 1
    assert same == total, (same, total)
    print("[OK] with the shipped defaults S1 is identical to S0 on every draw (hence the tuning protocol)")

    # ---- S4 beats nearest-key on moderately noisy data (the mechanism behind H1)
    rng_cers = {"S0": [], "S4": []}
    for s in range(60):
        o = make_observations("i need water", gaussian_pool(0.08), random.Random(1000 + s))
        for system in rng_cers:
            rng_cers[system].append(cer("i need water", decode(system, o, Params(s4_lm=1.0, s4_sigma_mult=0.8))))
    assert np.mean(rng_cers["S4"]) < 0.8 * np.mean(rng_cers["S0"]), {k: np.mean(v) for k, v in rng_cers.items()}
    print(f"[OK] S4 mean CER {np.mean(rng_cers['S4']):.3f} < S0 {np.mean(rng_cers['S0']):.3f} at sigma 0.08")

    # S1+ (hard key + adjacency-aware noise model) is a genuine improvement over S0, and the strongest hard-selection baseline
    s1p = [cer("i need water", decode("S1P", make_observations("i need water", gaussian_pool(0.08), random.Random(1000 + k)), Params(s1p_lm=1.0, s1p_sigma_mult=1.0))) for k in range(60)]
    assert np.mean(s1p) < np.mean(rng_cers["S0"]) and np.mean(rng_cers["S4"]) < np.mean(s1p), (np.mean(s1p), np.mean(rng_cers["S4"]))
    print(f"[OK] ordering at sigma 0.08: S4 {np.mean(rng_cers['S4']):.3f} < S1+ {np.mean(s1p):.3f} < S0 {np.mean(rng_cers['S0']):.3f}")

    # ---- online bias: learns the right sign from LM corrections of real words, never from correct words
    layout, _, lex, centres = ctx()
    assert lex is not None
    ob = OnlineBias()
    obs = make_observations("family", gaussian_pool(1e-4), random.Random(0))
    # make the gaze for the 'i' land on 'u' (one key left): decoded text says family, hard keys say famuly
    ux, uy = centres["U"]
    obs[3] = type(obs[3])(ux, uy, 1e-4, 1e-4, "i")
    ob.learn(obs, "family")
    assert ob.value[0] < 0 and abs(ob.value[1]) < 0.01, ob.value  # gaze was LEFT of the intended key
    ob2 = OnlineBias()
    ob2.learn(make_observations("family", gaussian_pool(1e-4), random.Random(0)), "family")  # nothing changed -> nothing learned
    assert ob2.value == [0.0, 0.0]
    print(f"[OK] OnlineBias: learns bias x={ob.value[0]:+.3f} from an LM-corrected word, learns nothing from correct text")

    # ---- sessions: deterministic, and drift hurts a static system
    phrases = ["i am hungry", "i need water", "can you help me"] * 4
    r1 = run_session("S2", phrases, gaussian_pool(0.05), 11, Params(s1_match=0.7, s1_lm=1.0), adapt=False)
    r2 = run_session("S2", phrases, gaussian_pool(0.05), 11, Params(s1_match=0.7, s1_lm=1.0), adapt=False)
    assert r1 == r2
    still = np.mean([np.mean([x["cer"] for x in run_session("S2", phrases, gaussian_pool(0.05), s, Params(s1_match=0.7, s1_lm=1.0), False)]) for s in range(20)])
    drifting = np.mean([np.mean([x["cer"] for x in run_session("S2", phrases, gaussian_pool(0.05), s, Params(s1_match=0.7, s1_lm=1.0), False, drift_total=(-0.14, -0.10))]) for s in range(20)])
    assert drifting > still + 0.05, (still, drifting)
    print(f"[OK] sessions are deterministic per seed; drift raises a static system's CER ({still:.3f} -> {drifting:.3f})")

    # ---- stats helpers
    x = np.array([0.30, 0.28, 0.35, 0.40, 0.33, 0.31, 0.29, 0.36])
    y = x - 0.1
    st = paired_stats(x, y)
    assert abs(st["mean_diff"] - 0.1) < 1e-9 and st["ci95"][0] > 0.09 and st["wilcoxon_p"] < 0.05 and abs(st["rel_reduction_pct"] - 100 * 0.1 / x.mean()) < 1e-9
    mean, lo, hi = bootstrap_ci(x)
    assert lo < mean < hi
    assert paired_stats(x, x)["wilcoxon_p"] == 1.0
    tp = make_tuning_phrases(25, {"i am hungry"})
    assert len(tp) == 25 and "i am hungry" not in tp and all(c.isalpha() or c == " " for ph in tp for c in ph)
    print("[OK] paired stats (mean diff, bootstrap CI, Wilcoxon, relative reduction) and disjoint tuning phrases")

    print("\n[OK] all simulation-core checks passed")


if __name__ == "__main__":
    main()
