"""Perception-side results table for the paper: held-out accuracy of the two gaze back-ends on your own calibration runs.

    .venv/Scripts/python.exe -m experiments.perception_summary

Reads data/calibration_history/*.jsonl (one file per Calibrate run; the app archives every run) and writes results/perception.md and
results/perception.json. For each run: the eye-landmark mapping and the gaze-CNN mapping are each evaluated LEAVE-ONE-TRIAL-OUT
(every trial predicted by a model that never saw it), reporting key-hit rate and error. "As recorded" is how the mapping that was LIVE
during that run did, which is the only place between-session drift shows up (a stale mapping vs a fresh one).

Caveat to state in the paper: the trials of one run are one person, one sitting, one camera -- held-out here means held-out trials,
not held-out people or held-out sessions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from decoder.nearest_key import KeyboardLayout, nearest_key
from experiments.calibrate_uncertainty import fit_affine, fit_feature_mapping
from experiments.sim_core import HISTORY_DIR, ROOT


def key_hit(pred: np.ndarray, rows: list[dict], layout: KeyboardLayout) -> float:
    return float(np.mean([nearest_key(*np.clip(p, 0, 1), layout) == r["target_key"] for p, r in zip(pred, rows)]))


def main() -> int:
    layout = KeyboardLayout.from_json()
    out_rows, md = [], ["# Perception: held-out accuracy per calibration run", "",
                        "| run | trials | eye-landmark: key hit / RMSE (subset) | gaze CNN: key hit / RMSE | live mapping during run: key hit / RMSE |", "|---|---|---|---|---|"]
    for path in sorted(HISTORY_DIR.glob("*.jsonl")):
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(rows) < 12:
            continue
        truth = np.array([[r["true_x"], r["true_y"]] for r in rows])
        rec = np.array([[r["mu_x"], r["mu_y"]] for r in rows])
        ff, cf = fit_feature_mapping(rows), fit_affine(rows)
        rec_rmse = float(np.sqrt(((rec - truth) ** 2).sum(1).mean()))
        entry = {"run": path.stem, "trials": len(rows), "as_recorded": {"key_hit": key_hit(rec, rows, layout), "rmse": rec_rmse}}
        cells = []
        for name, fit in (("landmarks", ff), ("cnn", cf)):
            if fit is None:
                cells.append("n/a")
                continue
            pred = np.array(fit["loo_mu"])
            e = pred - truth
            entry[name] = {"key_hit": key_hit(pred, rows, layout), "rmse": float(fit["loo_rmse"]), "err_std_xy": [float(e[:, 0].std()), float(e[:, 1].std())], **({"subset": fit["subset"], "alpha": fit["alpha"]} if name == "landmarks" else {"method": fit["method"]})}
            cells.append(f"{100 * entry[name]['key_hit']:.0f}% / {fit['loo_rmse']:.3f}" + (f" ({fit['subset']})" if name == "landmarks" else ""))
        md.append(f"| {path.stem} | {len(rows)} | {cells[0]} | {cells[1]} | {100 * entry['as_recorded']['key_hit']:.0f}% / {rec_rmse:.3f} |")
        out_rows.append(entry)
    # Cross-run generalisation: fit on all OTHER runs, test on this one (a later sitting).
    from experiments.sim_core import cross_run_errors

    runs = [[json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] for p in sorted(HISTORY_DIR.glob("*.jsonl"))]
    runs = [r for r in runs if len(r) >= 12]
    cross = {}
    if len(runs) >= 2:
        md += ["", "## Across sittings: fit on the other run(s), predict this one", "", "| configuration | key hit (per held-out run) | RMSE (per held-out run) |", "|---|---|---|"]
        for kind, label in (("lm_eye", "eye-landmark, eye-region features, alpha 10"), ("lm_iris", "eye-landmark, iris-only features, alpha 10"), ("cnn", "gaze-CNN absolute-angle affine")):
            errs = cross_run_errors(runs, kind)
            hits, rmses = [], []
            for run, e in zip(runs, errs):
                truth = np.array([[r["true_x"], r["true_y"]] for r in run])
                hits.append(key_hit(truth + e, run, layout))
                rmses.append(float(np.sqrt((e**2).sum(1).mean())))
            cross[kind] = {"key_hit": hits, "rmse": rmses}
            md.append(f"| {label} | " + ", ".join(f"{100 * h:.0f}%" for h in hits) + " | " + ", ".join(f"{r:.3f}" for r in rmses) + " |")
        md += ["", "Compare with the within-run numbers above: a mapping fitted in one sitting loses a large part of its accuracy in the next (head position, "
               "lighting and seating differ). The iris-only configuration generalised worst in this data."]
    md += ["", "Key hit = nearest key to the predicted point equals the target key (chance with 40 keys is about 2.5%). RMSE is in normalised canvas units "
           "(a key is 0.10 wide and 0.25 tall). Held-out = leave-one-trial-out within the run."]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "perception.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (ROOT / "results" / "perception.json").write_text(json.dumps({"per_run": out_rows, "cross_run": cross}, indent=2), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
