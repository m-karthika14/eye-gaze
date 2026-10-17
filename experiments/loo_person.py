"""Calibration-free test across PEOPLE: leave-one-person-out accuracy of the gaze mapping.

    .venv/Scripts/python.exe -m experiments.loo_person

Each person does the app's accuracy check (Calibrate) once WITH a participant ID; every run is archived in data/calibration_history/.
Here, for each person, a mapping is fitted on everybody ELSE's trials only and applied to them with no calibration of their own. That is
what "calibration-free" means operationally: the user's first minute, with a mapping learned from other people.

Reports per person: key hit (nearest key to the predicted point == target; chance with 40 keys ~2.5%) and RMSE (normalised canvas
units: a key is 0.10 wide, 0.25 tall), for the eye-landmark and gaze-CNN back-ends, with and without a one-point re-centre (pressing C
while looking at the middle of the keyboard). Then mean [95% bootstrap CI over people], median and the worst person.

Needs >= 3 people to be meaningful (with 2, the "generic" mapping is one person's). Writes results/lopo.md and results/lopo.json.
"""

from __future__ import annotations

import json

import numpy as np

from decoder.nearest_key import KeyboardLayout, nearest_key
from experiments.sim_core import HISTORY_DIR, ROOT, load_runs_by_person, lopo_predictions

CONFIGS = (
    ("lm_eye", False, "eye-landmark, zero-shot"),
    ("lm_eye", True, "eye-landmark, + one-point re-centre"),
    ("lm_full", False, "eye-landmark (all features), zero-shot"),
    ("cnn", False, "gaze CNN, zero-shot"),
    ("cnn", True, "gaze CNN, + one-point re-centre"),
)


def metrics(res: dict, layout: KeyboardLayout) -> dict[str, tuple[float, float]]:
    out = {}
    for person, r in res.items():
        hit = float(np.mean([nearest_key(*np.clip(p, 0, 1), layout) == k for p, k in zip(r["pred"], r["keys"])]))
        rmse = float(np.sqrt(((r["pred"] - r["truth"]) ** 2).sum(axis=1).mean()))
        out[person] = (hit, rmse)
    return out


def boot_ci(v: np.ndarray, n: int = 10_000) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    by_person = load_runs_by_person(HISTORY_DIR)
    people = sorted(by_person)
    print(f"{len(people)} people with usable calibration runs: " + ", ".join(f"{p} ({len(by_person[p])} run(s))" for p in people))
    if len(people) < 3:
        print("Need at least 3 people (each with the participant-ID box filled in before running the accuracy check). Nothing to report yet.")
        return 1
    layout = KeyboardLayout.from_json()
    table = {}
    for kind, recentre, label in CONFIGS:
        table[label] = metrics(lopo_predictions(by_person, kind, recentre), layout)

    md = ["# Calibration-free test: leave-one-person-out", "",
          f"{len(people)} people. Each person is scored with a mapping fitted only on the other people (no calibration of their own). "
          "Key hit chance with 40 keys is about 2.5%. RMSE in normalised canvas units (key = 0.10 wide, 0.25 tall).", "",
          "| person | " + " | ".join(label for _, _, label in CONFIGS) + " |", "|---|" + "---|" * len(CONFIGS)]
    for p in people:
        md.append(f"| {p} | " + " | ".join(f"{100 * table[l][p][0]:.0f}% / {table[l][p][1]:.3f}" for _, _, l in CONFIGS) + " |")
    summary = {}
    md += ["", "| summary over people (key hit / RMSE) | mean [95% CI] key hit | median | worst person | mean RMSE |", "|---|---|---|---|---|"]
    for _, _, label in CONFIGS:
        hits = np.array([table[label][p][0] for p in people])
        rm = np.array([table[label][p][1] for p in people])
        m, lo, hi = boot_ci(hits)
        summary[label] = {"mean_key_hit": m, "ci95": [lo, hi], "median": float(np.median(hits)), "worst": float(hits.min()), "worst_person": people[int(hits.argmin())], "mean_rmse": float(rm.mean())}
        md.append(f"| {label} | {100 * m:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}] | {100 * np.median(hits):.0f}% | {100 * hits.min():.0f}% ({people[int(hits.argmin())]}) | {rm.mean():.3f} |")
    md += ["", "Report the worst person as well as the mean: a calibration-free claim is only as good as its worst user. With few people the "
           "intervals are wide; say how many people there were."]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "lopo.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (ROOT / "results" / "lopo.json").write_text(json.dumps({"people": people, "per_person": {l: {p: {"key_hit": v[0], "rmse": v[1]} for p, v in t.items()} for l, t in table.items()}, "summary": summary}, indent=2), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
