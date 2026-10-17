"""Checks for the study tooling: leave-one-person-out, study summary, mapping switch. Synthetic data, no camera.

    .venv/Scripts/python.exe -m experiments.test_study_tools
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from experiments import study_summary
from experiments.sim_core import ErrorPool, load_error_pools, load_runs_by_person, lopo_predictions, person_of
from experiments.use_mapping import GENERIC, read_feature_model, use
from vision.eye_features import FEATURE_SUBSETS, N_FEATURES

KEYS = ["Q", "W", "P", "O", "A", "S", "L", "Z", "X", "M", "T", "D", "H", "B", "SPACE", "BACKSPACE"]
RNG = np.random.default_rng(0)
MIX = np.zeros((2, N_FEATURES))
MIX[:, FEATURE_SUBSETS["eye_region"][:40]] = RNG.normal(0, 1, (2, 40))
DIRS = RNG.normal(0, 1, 40) * 0.05


def fake_run(seed: int, person_offset: float, participant: str | None = None) -> list[dict]:
    r = np.random.default_rng(seed)
    rows = []
    for i in range(32):
        tx, ty = r.uniform(0.05, 0.95), r.uniform(0.12, 0.88)
        feat = (np.array([tx, ty]) - 0.5) @ MIX * 0.05 + r.normal(0, 0.004, N_FEATURES)
        feat[FEATURE_SUBSETS["eye_region"][:40]] += person_offset * DIRS  # this person's eyes sit differently: a per-person shift across the informative features
        row = {"target_key": KEYS[i % len(KEYS)], "true_x": float(tx), "true_y": float(ty), "mu_x": 0.5, "mu_y": 0.5, "rel_yaw": 0.0, "rel_pitch": 0.0,
               "abs_yaw": float((tx - 0.5) / 2.5), "abs_pitch": float(-(ty - 0.35) / 1.6), "eye_feat": [float(v) for v in feat]}
        if participant:
            row["participant"] = participant
        rows.append(row)
    return rows


def main() -> None:
    # ---- who a run belongs to
    assert person_of(Path("20261010_144635.jsonl"), [{"x": 1}]) == "legacy"
    assert person_of(Path("20261010_144100_run.jsonl"), [{"x": 1}]) == "legacy"
    assert person_of(Path("20261017_101500_P03.jsonl"), [{"x": 1}]) == "P03"
    assert person_of(Path("whatever.jsonl"), [{"participant": "P07"}]) == "P07"
    print("[OK] person_of: participant field, file-name id, and legacy runs (one person)")

    # ---- load_runs_by_person + pools from a temp history dir
    with tempfile.TemporaryDirectory() as d:
        hist = Path(d)
        for k, (pid, off) in enumerate((("P01", 0.0), ("P02", 1.0), ("P03", -1.0), ("P04", 2.0), ("P05", -2.0))):
            rows = fake_run(k, off, pid)
            (hist / f"20261017_10{k:02d}00_{pid}.jsonl").write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
        (hist / "20261010_000000.jsonl").write_text("\n".join(json.dumps({k: v for k, v in x.items() if k != "eye_feat"}) for x in fake_run(99, 0)), encoding="utf-8")  # old run: no features
        by_person = load_runs_by_person(hist)
        assert sorted(by_person) == ["P01", "P02", "P03", "P04", "P05"], sorted(by_person)  # the feature-less legacy run is skipped, not crashed on
        print("[OK] load_runs_by_person: groups runs per participant, skips runs recorded before eye features existed")

        zero = lopo_predictions(by_person, "lm_eye", recentre=False)
        recentred = lopo_predictions(by_person, "lm_eye", recentre=True)
        assert set(zero) == set(by_person) and all(r["n"] == 32 for r in zero.values())
        assert all(r["n"] == 31 for r in recentred.values())  # the anchor trial is excluded from scoring
        rm = lambda res: float(np.mean([np.sqrt(((r["pred"] - r["truth"]) ** 2).sum(1).mean()) for r in res.values()]))
        print(f"[OK] leave-one-person-out: zero-shot RMSE {rm(zero):.3f}; with a one-point re-centre {rm(recentred):.3f}")
        # Pooling several people teaches the model to ignore per-person differences, so the re-centre need not help here (one noisy anchor trial
        # shifts everything) -- loo_person.py reports both for that reason. What it MUST do is remove a constant offset pooling cannot remove
        # (e.g. a different camera angle): test that directly.
        import experiments.sim_core as sim

        original = sim._fit_predict
        try:
            sim._fit_predict = lambda kind, train, test: np.array([[x["true_x"], x["true_y"]] for x in test]) + np.array([0.10, -0.06])
            shifted = lopo_predictions(by_person, "lm_eye", recentre=False)
            fixed = lopo_predictions(by_person, "lm_eye", recentre=True)
        finally:
            sim._fit_predict = original
        assert rm(shifted) > 0.11 and rm(fixed) < 1e-9, (rm(shifted), rm(fixed))
        print("[OK] one-point re-centre removes a constant offset (RMSE 0.117 -> 0) and excludes the anchor trial from scoring")

        # the held-out person's own data never reaches the fit: changing it must not change their predictions
        altered = {p: [list(run) for run in runs] for p, runs in by_person.items()}
        for row in altered["P03"][0]:
            row["true_x"], row["true_y"] = 0.9, 0.9
        again = lopo_predictions(altered, "lm_eye", recentre=False)
        assert np.allclose(again["P03"]["pred"], zero["P03"]["pred"])
        print("[OK] no leakage: a person's own trials do not influence the mapping used for them")

        pools = load_error_pools(hist)
        assert {"landmarks_lopo", "landmarks_lopo_recentre", "cnn_lopo"} <= set(pools), sorted(pools)
        assert isinstance(pools["landmarks_lopo"], ErrorPool) and len(pools["landmarks_lopo"].vectors) == 160
        print("[OK] load_error_pools adds the calibration-free (across-people) pools once 3+ people exist")

    # only 2 people -> no across-people pools (the 'generic' mapping would be one person's)
    with tempfile.TemporaryDirectory() as d:
        for k, pid in enumerate(("A", "B")):
            (Path(d) / f"20261017_10{k:02d}00_{pid}.jsonl").write_text("\n".join(json.dumps(x) for x in fake_run(k, 0.0, pid)), encoding="utf-8")
        assert not any(n.startswith(("landmarks_lopo", "cnn_lopo")) for n in load_error_pools(Path(d)))
    print("[OK] fewer than 3 people -> no across-people pools (it would not be a generic mapping)")

    # ---- study summary
    def rec(pid, mode, target, typed, elapsed, sel, bs=0, picks=0):
        return {"participant": pid, "decoder_mode": mode, "target": target, "typed": typed, "elapsed_s": elapsed, "selections": sel, "backspaces": bs, "candidate_picks": picks}

    recs = []
    for pid in ("P1", "P2", "P3"):
        recs += [rec(pid, "s4", "i am hungry", "i am hungry", 12.0, 11), rec(pid, "s4", "i need water", "i need water", 13.0, 12),
                 rec(pid, "s0", "i am hungry", "i am hungru", 12.0, 12, bs=1), rec(pid, "s0", "i need water", "i nees water", 13.0, 13, bs=1)]
    s = study_summary.summarise(recs)
    assert s["participants"] == ["P1", "P2", "P3"] and s["modes"]["s4"]["n_participants"] == 3 and s["modes"]["s4"]["n_phrases"] == 6
    assert s["modes"]["s4"]["cer"]["mean"] == 0.0 and s["modes"]["s4"]["exact"]["mean"] == 1.0
    assert abs(s["modes"]["s0"]["cer"]["mean"] - (1 / 11 + 1 / 12) / 2) < 1e-9
    assert abs(study_summary.phrase_stats(recs[0])["wpm"] - 11.0) < 1e-9  # 11 characters = 2.2 words in 0.2 minutes
    assert "s0" in s["paired_vs_S4"] and s["paired_vs_S4"]["s0"]["cer"]["mean_diff_vs_S4"] > 0
    assert abs(study_summary.phrase_stats(recs[2])["bs_rate"] - 1 / 12) < 1e-9
    print("[OK] study_summary: CER / WER / exact / WPM / KSPC / backspace rate per participant and system, paired differences vs S4")

    # ---- mapping switch preserves everything else in the config
    with tempfile.TemporaryDirectory() as d:
        cfg, choice = Path(d) / "gaze_mapping.yaml", Path(d) / "mapping_choice.json"
        cfg.write_text("# a comment that must survive\nort_threads: 2\nfeature_model: models/eye_feature_model.npz\nstability_radius: 0.03  # trailing note\n", encoding="utf-8")
        assert use("generic", cfg, choice) == GENERIC and read_feature_model(cfg) == GENERIC
        text = cfg.read_text(encoding="utf-8")
        assert "# a comment that must survive" in text and "ort_threads: 2" in text and "stability_radius: 0.03" in text
        assert (Path(d) / "gaze_mapping.backup_before_generic.yaml").exists()
        assert use("generic", cfg, choice) == GENERIC  # idempotent
        assert use("personal", cfg, choice) == "models/eye_feature_model.npz" and read_feature_model(cfg) == "models/eye_feature_model.npz"
    print("[OK] use_mapping switches generic <-> personal, keeps comments and other settings, backs up once, is idempotent")

    print("\n[OK] all study-tooling checks passed")


if __name__ == "__main__":
    main()
