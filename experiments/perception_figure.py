"""Figure + summary numbers for the Perception results paragraph (reads results/perception.json, run perception_summary first).

    .venv/Scripts/python.exe -m experiments.perception_figure
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.sim_core import ROOT

RESULTS = ROOT / "results"


def main() -> int:
    d = json.loads((RESULTS / "perception.json").read_text(encoding="utf-8"))
    runs = d["per_run"]
    n = len(runs)
    labels = [f"R{i + 1}" for i in range(n)]
    within = {k: np.array([r[k]["key_hit"] for r in runs]) for k in ("landmarks", "cnn")}
    within_rmse = {k: np.array([r[k]["rmse"] for r in runs]) for k in ("landmarks", "cnn")}
    stale = np.array([r["as_recorded"]["key_hit"] for r in runs])
    across = {k: np.array(d["cross_run"][k]["key_hit"]) for k in ("lm_eye", "cnn")}
    across_rmse = {k: np.array(d["cross_run"][k]["rmse"]) for k in ("lm_eye", "cnn")}

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(n)
    w = 0.2
    ax[0].bar(x - 1.5 * w, within["landmarks"], w, label="Eye-landmark, within sitting", color="#1f77b4")
    ax[0].bar(x - 0.5 * w, across["lm_eye"], w, label="Eye-landmark, across sittings", color="#9ecae1")
    ax[0].bar(x + 0.5 * w, within["cnn"], w, label="Gaze CNN, within sitting", color="#ff7f0e")
    ax[0].bar(x + 1.5 * w, across["cnn"], w, label="Gaze CNN, across sittings", color="#fdd0a2")
    ax[0].axhline(1 / 28, color="grey", ls=":", lw=1)
    ax[0].text(n - 0.5, 0.05, "chance", ha="right", fontsize=7, color="grey")
    ax[0].set_xticks(x, labels)
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("Key-hit rate (28 keys)")
    ax[0].set_xlabel("Calibration run (held out)")
    ax[0].legend(fontsize=7, loc="upper left")
    ax[0].set_title("(a) Key-hit rate per run")

    groups = [("Eye-landmark\nwithin", within_rmse["landmarks"]), ("Eye-landmark\nacross", across_rmse["lm_eye"]),
              ("Gaze CNN\nwithin", within_rmse["cnn"]), ("Gaze CNN\nacross", across_rmse["cnn"])]
    ax[1].boxplot([g for _, g in groups], tick_labels=[t for t, _ in groups], widths=0.5, showfliers=False)
    for i, (_, g) in enumerate(groups, 1):
        ax[1].scatter(np.full(len(g), i) + np.linspace(-0.08, 0.08, len(g)), g, s=14, color="k", zorder=3)
    ax[1].set_ylabel("RMSE (canvas units; key = 0.10 wide)")
    ax[1].set_title("(b) Position error")
    fig.tight_layout()
    out = RESULTS / "figures" / "fig_perception.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)

    f = lambda a: f"{a.mean():.2f} (range {a.min():.2f}-{a.max():.2f})"
    print(f"runs {n}; trials {[r['trials'] for r in runs]}")
    print("within  eye-landmark hit", f(within["landmarks"]), "rmse", f(within_rmse["landmarks"]))
    print("across  eye-landmark hit", f(across["lm_eye"]), "rmse", f(across_rmse["lm_eye"]))
    print("within  CNN          hit", f(within["cnn"]), "rmse", f(within_rmse["cnn"]))
    print("across  CNN          hit", f(across["cnn"]), "rmse", f(across_rmse["cnn"]))
    print("live mapping as recorded hit", f(stale))
    print("frame-level runs (last 3) within hit", f(within["landmarks"][-3:]), "rmse", f(within_rmse["landmarks"][-3:]))
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
