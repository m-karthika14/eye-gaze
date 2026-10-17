"""Synthetic users for TESTING the analysis pipeline only (leave-one-person-out, summaries). Not participants, never evidence.

    .venv/Scripts/python.exe -m experiments.make_synthetic_users          # writes data/synthetic_test/{calibration_history,sessions}
    .venv/Scripts/python.exe -m experiments.make_synthetic_users --analyse  # also runs the leave-one-person-out table on them

Everything is written under data/synthetic_test/, which no analysis script reads by default, and every row carries "synthetic": true, so
it cannot leak into results/ built from real runs. Each user gets their own eye-to-feature mixing, offset and noise level.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from decoder.nearest_key import KeyboardLayout
from experiments.sim_core import ROOT
from vision.eye_features import FEATURE_SUBSETS, N_FEATURES

OUT = ROOT / "data" / "synthetic_test"
N_USERS, TRIALS = 10, 32


def make_user(idx: int, layout: KeyboardLayout, shared_mix: np.ndarray) -> list[dict]:
    rng = np.random.default_rng(1000 + idx)
    cols = FEATURE_SUBSETS["eye_region"]
    mix = shared_mix + rng.normal(0, 0.25, shared_mix.shape)  # each user's eyes map position to features a bit differently
    offset = rng.normal(0, 1.0, len(cols)) * 0.03  # per-user shift (face shape, camera angle)
    noise = rng.uniform(0.003, 0.008)  # per-user feature noise
    keys = [layout.keys[i] for i in rng.choice(len(layout.keys), TRIALS, replace=True)]
    rows = []
    for k in keys:
        tx, ty = k.center()
        base = np.zeros(N_FEATURES)
        base[cols] = (np.array([tx, ty]) - 0.5) @ mix * 0.05 + offset
        frames = [base + rng.normal(0, noise, N_FEATURES) for _ in range(9)]
        rows.append({
            "synthetic": True, "participant": f"SYN{idx + 1:02d}", "target_key": k.id, "true_x": float(tx), "true_y": float(ty),
            "mu_x": 0.5, "mu_y": 0.5, "rel_yaw": 0.0, "rel_pitch": 0.0,
            "abs_yaw": float((tx - 0.5) / 2.5 + rng.normal(0, 0.01)), "abs_pitch": float(-(ty - 0.35) / 1.6 + rng.normal(0, 0.01)),
            "eye_feat": [round(float(v), 5) for v in np.mean(frames, axis=0)],
            "eye_frames": [[round(float(v), 4) for v in f] for f in frames],
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    layout = KeyboardLayout.from_json()
    shared = np.zeros((2, len(FEATURE_SUBSETS["eye_region"])))
    shared[:] = np.random.default_rng(7).normal(0, 1, shared.shape)
    hist = OUT / "calibration_history"
    hist.mkdir(parents=True, exist_ok=True)
    (OUT / "README.txt").write_text("SYNTHETIC test data made by experiments/make_synthetic_users.py. Not participants. Do not use as evidence.\n", encoding="utf-8")
    for i in range(N_USERS):
        rows = make_user(i, layout, shared)
        (hist / f"20990101_0000{i:02d}_SYN{i + 1:02d}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"wrote {N_USERS} synthetic users to {hist}")
    if a.analyse:
        import experiments.loo_person as lp
        import experiments.sim_core as sim

        by_person = sim.load_runs_by_person(hist)
        layout = KeyboardLayout.from_json()
        print("leave-one-person-out on SYNTHETIC users (pipeline test only):")
        for kind, recentre, label in lp.CONFIGS[:2]:
            res = sim.lopo_predictions(by_person, kind, recentre)
            m = lp.metrics(res, layout)
            hit = np.mean([v[0] for v in m.values()])
            rmse = np.mean([v[1] for v in m.values()])
            print(f"  {label}: mean key hit {100 * hit:.0f}% , mean RMSE {rmse:.3f}  ({len(m)} synthetic users)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
