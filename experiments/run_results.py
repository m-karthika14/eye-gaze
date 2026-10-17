"""Phase 9 results runner: S0-S4 on trace-driven simulated typing (see experiments/sim_core.py for what that means and does not mean).

    .venv/Scripts/python.exe -m experiments.run_results            # full run (a few minutes on 8 cores)
    .venv/Scripts/python.exe -m experiments.run_results --quick    # smoke test with tiny sample sizes

Protocol (state all of it in the paper):
  1. TUNING   S1 (confusion match probability, LM weight) and S4 (LM weight, sigma multiplier) are each tuned by grid search on a
              TUNING split: phrases drawn from the AAC word list, disjoint from the evaluation phrases, disjoint seeds.
              Same budget for both -- the baseline gets no handicap. (The shipped S1 defaults never override a selected key, i.e.
              S1 == S0; tuning is what makes it a real baseline.)
  2. TEST     the 20 FIXED phrases in config/evaluation_phrases.txt x N simulated noise draws each, seeds disjoint from tuning.
              Every system decodes the identical observation sequences (paired).
  3. CONDITIONS
        real_landmarks / real_cnn   noise bootstrapped from measured held-out errors of the two perception back-ends
        sweep                       iid Gaussian noise, sigma 0.04 ... 0.20 per axis (H1: advantage vs noise)
        drift                       40-phrase sessions with a slowly growing gaze bias (posture drift) (H2: online adaptation)
  4. STATS    per-phrase paired differences, Wilcoxon signed-rank + percentile bootstrap CI of the mean difference; for drift, per-session.
Outputs go to results/: tables.md, summary.json, tuning CSVs, per_phrase.csv, figures (png).
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

from experiments.sim_core import (
    ErrorPool,
    Params,
    ROOT,
    decode,
    gaussian_pool,
    load_error_pools,
    make_observations,
    phrase_metrics,
    run_session,
)

OUT = ROOT / "results"
PHRASES_PATH = ROOT / "config" / "evaluation_phrases.txt"
VOCAB_PATH = ROOT / "config" / "vocabulary.txt"
METRICS = ("cer", "wer", "char_acc", "exact")

S1_GRID = {"s1_match": (0.2, 0.35, 0.5, 0.7, 0.85, 0.95), "s1_lm": (0.25, 0.5, 1.0, 2.0)}
S4_GRID = {"s4_lm": (0.25, 0.5, 1.0, 1.5, 2.0), "s4_sigma_mult": (0.6, 0.8, 1.0, 1.25, 1.5)}
S1P_GRID = {"s1p_lm": (0.25, 0.5, 1.0, 1.5, 2.0), "s1p_sigma_mult": (0.6, 0.8, 1.0, 1.25, 1.5, 2.0)}
SWEEP_SIGMAS = (0.04, 0.08, 0.12, 0.16, 0.20)
DRIFT_LEVELS = {"drift_half": (-0.07, -0.05), "drift_measured": (-0.14, -0.10), "drift_double": (-0.28, -0.20)}
TUNE_SEED, TEST_SEED, SESSION_SEED = 1_000_000, 50_000_000, 90_000_000


def load_eval_phrases() -> list[str]:
    return [l.strip() for l in PHRASES_PATH.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def make_tuning_phrases(n: int, exclude: set[str]) -> list[str]:
    words = [w for w in (l.strip().lower() for l in VOCAB_PATH.read_text(encoding="utf-8").splitlines() if not l.startswith("#")) if w.isalpha()]
    rng = random.Random(123)
    out: list[str] = []
    while len(out) < n:
        phrase = " ".join(rng.choice(words) for _ in range(rng.randint(3, 5)))
        if phrase not in exclude and len(phrase) <= 34:
            out.append(phrase)
    return out


# ---------------------------------------------------------------------------------------------- worker tasks (process pool)
def _mean_cer_task(args) -> tuple[str, dict, float]:
    system, params, pool, phrases, sims, seed0 = args
    total, n = 0.0, 0
    for pi, ph in enumerate(phrases):
        for s in range(sims):
            obs = make_observations(ph, pool, random.Random(seed0 + pi * 7919 + s))
            total += phrase_metrics(ph, decode(system, obs, params))["cer"]
            n += 1
    return system, params.__dict__, total / n


def _eval_task(args) -> tuple[str, int, list[dict]]:
    system, params, pool, phrase, pi, sims, seed0, scale = args
    rows = []
    for s in range(sims):
        obs = make_observations(phrase, pool, random.Random(seed0 + pi * 7919 + s), scale)
        rows.append(phrase_metrics(phrase, decode(system, obs, params)))
    return system, pi, rows


def _session_task(args) -> tuple[str, bool, int, list[dict]]:
    system, adapt, params, pool, phrases, seed, drift = args
    return system, adapt, seed, run_session(system, phrases, pool, seed, params, adapt, drift)


# ---------------------------------------------------------------------------------------------- statistics
def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), (n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: matched per-unit scores (lower = better). Reports mean(a - b), 95% bootstrap CI, Wilcoxon p, and the relative reduction."""
    from scipy.stats import wilcoxon

    d = a - b
    mean, lo, hi = bootstrap_ci(d)
    try:
        p = 1.0 if np.allclose(d, 0) else float(wilcoxon(a, b).pvalue)
    except ValueError:
        p = 1.0
    return {"mean_diff": mean, "ci95": [lo, hi], "wilcoxon_p": p, "n": int(len(d)), "rel_reduction_pct": float(100 * (a.mean() - b.mean()) / a.mean()) if a.mean() > 0 else 0.0}


# ---------------------------------------------------------------------------------------------- tuning
def tune(ex: ProcessPoolExecutor, pool: ErrorPool, phrases: list[str], sims: int, label: str, out_dir: Path) -> Params:
    jobs = []
    for m, lm in product(*S1_GRID.values()):
        jobs.append(("S1", Params(s1_match=m, s1_lm=lm), pool, phrases, sims, TUNE_SEED))
    for lm, mult in product(*S4_GRID.values()):
        jobs.append(("S4", Params(s4_lm=lm, s4_sigma_mult=mult), pool, phrases, sims, TUNE_SEED))
    for lm, mult in product(*S1P_GRID.values()):
        jobs.append(("S1P", Params(s1p_lm=lm, s1p_sigma_mult=mult), pool, phrases, sims, TUNE_SEED))
    results = list(ex.map(_mean_cer_task, jobs, chunksize=1))
    best = {}
    with open(out_dir / f"tuning_{label}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "s1_match", "s1_lm", "s4_lm", "s4_sigma_mult", "s1p_lm", "s1p_sigma_mult", "tuning_mean_cer"])
        for system, prm, cer in results:
            w.writerow([system, prm["s1_match"], prm["s1_lm"], prm["s4_lm"], prm["s4_sigma_mult"], prm["s1p_lm"], prm["s1p_sigma_mult"], f"{cer:.5f}"])
            if system not in best or cer < best[system][1]:
                best[system] = (prm, cer)
    chosen = Params(
        s1_match=best["S1"][0]["s1_match"], s1_lm=best["S1"][0]["s1_lm"],
        s4_lm=best["S4"][0]["s4_lm"], s4_sigma_mult=best["S4"][0]["s4_sigma_mult"],
        s1p_lm=best["S1P"][0]["s1p_lm"], s1p_sigma_mult=best["S1P"][0]["s1p_sigma_mult"],
    )
    print(f"  tuned [{label}]  S1: match={chosen.s1_match} lm={chosen.s1_lm} ({best['S1'][1]:.3f})   S1+: lm={chosen.s1p_lm} sigma x{chosen.s1p_sigma_mult} ({best['S1P'][1]:.3f})   S4: lm={chosen.s4_lm} sigma x{chosen.s4_sigma_mult} ({best['S4'][1]:.3f})")
    return chosen


# ---------------------------------------------------------------------------------------------- evaluation of one condition
def evaluate(ex, pool, params, phrases, sims, systems, scale=1.0) -> dict[str, dict[str, np.ndarray]]:
    """-> {system: {metric: array (n_phrases, sims)}} -- identical observations for every system."""
    jobs = [(s, params, pool, ph, pi, sims, TEST_SEED, scale) for s in systems for pi, ph in enumerate(phrases)]
    data = {s: {m: np.zeros((len(phrases), sims)) for m in METRICS} for s in systems}
    for system, pi, rows in ex.map(_eval_task, jobs, chunksize=2):
        for si, row in enumerate(rows):
            for m in METRICS:
                data[system][m][pi, si] = row[m]
    return data


def summarise(data) -> dict:
    out = {}
    for s, per in data.items():
        out[s] = {}
        for m, arr in per.items():
            phrase_means = arr.mean(axis=1)  # cluster by phrase: the unit of analysis is the phrase, not the simulation draw
            mean, lo, hi = bootstrap_ci(phrase_means)
            out[s][m] = {"mean": mean, "ci95": [lo, hi]}
    return out


def md_table(title: str, summary: dict, order: list[str], note: str = "") -> str:
    lines = [f"### {title}", "", "| System | CER ↓ | WER ↓ | Char. acc. ↑ | Sentence exact ↑ |", "|---|---|---|---|---|"]
    for s in order:
        r = summary[s]
        cell = lambda m, pct=False: (f"{100 * r[m]['mean']:.1f}% [{100 * r[m]['ci95'][0]:.1f}, {100 * r[m]['ci95'][1]:.1f}]" if pct else f"{r[m]['mean']:.3f} [{r[m]['ci95'][0]:.3f}, {r[m]['ci95'][1]:.3f}]")
        lines.append(f"| {s} | {cell('cer')} | {cell('wer')} | {cell('char_acc', True)} | {cell('exact', True)} |")
    if note:
        lines += ["", note]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny sample sizes, for checking the pipeline only")
    ap.add_argument("--sims", type=int, default=None, help="noise draws per phrase on the test split (default 100, quick 8)")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    quick = args.quick
    sims = args.sims or (8 if quick else 100)
    tune_phrases_n, tune_sims = (30, 1) if quick else (120, 2)
    sessions = 10 if quick else 100
    OUT.mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    phrases = load_eval_phrases()
    tune_phrases = make_tuning_phrases(tune_phrases_n, set(phrases))
    pools = load_error_pools()
    if "landmarks_loo" not in pools or "cnn_loo" not in pools:
        print("need at least one calibration run with eye-landmark features in data/calibration_history/ (run Calibrate in the app)")
        return 1
    print(f"{len(phrases)} test phrases x {sims} draws | {len(tune_phrases)} tuning phrases x {tune_sims} | {sessions} drift sessions | pools: " + ", ".join(f"{k}(n={len(p.vectors)})" for k, p in pools.items()))
    t0 = time.time()
    summary: dict = {"settings": {"sims_per_phrase": sims, "test_phrases": len(phrases), "tuning_phrases": len(tune_phrases), "tuning_sims": tune_sims, "drift_sessions": sessions, "seeds": {"tune": TUNE_SEED, "test": TEST_SEED, "session": SESSION_SEED}}, "pools": {k: {"n": len(p.vectors), "sigma_xy": list(p.sigma), "mean_err_xy": p.vectors.mean(axis=0).tolist()} for k, p in pools.items()}}
    md: list[str] = []
    per_phrase_rows: list[list] = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        # ---------------- real measured noise -------------------------------------------------------------------------
        print("[1/3] real measured noise")
        real_pools = [
            ("landmarks_loo", "eye-landmark perception, freshly calibrated (within-run held-out errors)"),
            ("cnn_loo", "gaze-CNN perception, freshly calibrated (within-run held-out errors)"),
            ("landmarks_xrun_eye", "eye-landmark perception, eye-region features, across sittings (fit on other runs)"),
            ("landmarks_xrun_iris", "eye-landmark perception, iris-only features, across sittings (fit on other runs)"),
            ("cnn_xrun", "gaze-CNN perception, across sittings (fit on other runs)"),
            # Across PEOPLE: only present once calibration runs from >= 3 participants are in data/calibration_history/
            ("landmarks_lopo", "eye-landmark perception, NO calibration of the user (mapping fitted on other people)"),
            ("landmarks_lopo_recentre", "eye-landmark perception, NO calibration of the user, plus one-point re-centre"),
            ("cnn_lopo", "gaze-CNN perception, NO calibration of the user (mapping fitted on other people)"),
        ]
        for pname, label in real_pools:
            if pname not in pools:
                continue
            pool = pools[pname]
            params = tune(ex, pool, tune_phrases, tune_sims, pname, OUT)
            systems = ["S0", "S1", "S1P", "S4"]
            data = evaluate(ex, pool, params, phrases, sims, systems)
            summ = summarise(data)
            h1 = {m: paired_stats(data["S1"][m].mean(axis=1), data["S4"][m].mean(axis=1)) for m in ("cer", "wer")}
            h1p = {m: paired_stats(data["S1P"][m].mean(axis=1), data["S4"][m].mean(axis=1)) for m in ("cer", "wer")}
            h0 = {m: paired_stats(data["S0"][m].mean(axis=1), data["S4"][m].mean(axis=1)) for m in ("cer", "wer")}
            summary[pname] = {"tuned_params": params.__dict__, "systems": summ, "H1_S1_vs_S4": h1, "H1_S1plus_vs_S4": h1p, "S0_vs_S4": h0}
            md.append(md_table(f"Real noise: {label}", {**summ, "S2": summ["S1"]}, ["S0", "S1", "S1P", "S2", "S4"],
                               f"S2 decodes identically to S1 on recorded sequences (experiments/s2.py), so its row equals S1. S1 tuned: match={params.s1_match}, LM weight={params.s1_lm}; S4 tuned: LM weight={params.s4_lm}, sigma x{params.s4_sigma_mult}. "
                               f"S1+ (adjacency-aware, hard-selected key) tuned: LM weight={params.s1p_lm}, sigma x{params.s1p_sigma_mult}. "
                               f"Mean [95% bootstrap CI over the {len(phrases)} phrases]. **H1 (S1 vs S4)**: CER reduction {h1['cer']['rel_reduction_pct']:.1f}% (mean diff {h1['cer']['mean_diff']:+.3f}, 95% CI [{h1['cer']['ci95'][0]:+.3f}, {h1['cer']['ci95'][1]:+.3f}], Wilcoxon p={h1['cer']['wilcoxon_p']:.2g}, n={h1['cer']['n']} phrases). "
                               f"**H1 vs the strongest hard-selection baseline (S1+ vs S4)**: CER reduction {h1p['cer']['rel_reduction_pct']:.1f}% (mean diff {h1p['cer']['mean_diff']:+.3f}, 95% CI [{h1p['cer']['ci95'][0]:+.3f}, {h1p['cer']['ci95'][1]:+.3f}], Wilcoxon p={h1p['cer']['wilcoxon_p']:.2g})."))
            for pi, ph in enumerate(phrases):
                for s in systems:
                    per_phrase_rows.append([pname, ph, s] + [f"{data[s][m][pi].mean():.5f}" for m in METRICS])

        # ---------------- noise sweep -----------------------------------------------------------------------------------
        print("[2/3] noise sweep")
        sweep = {}
        for sg in SWEEP_SIGMAS:
            pool = gaussian_pool(sg)
            params = tune(ex, pool, tune_phrases, tune_sims, f"sweep_{sg:g}", OUT)
            data = evaluate(ex, pool, params, phrases, sims, ["S0", "S1", "S1P", "S4"])
            summ = summarise(data)
            h1 = paired_stats(data["S1"]["cer"].mean(axis=1), data["S4"]["cer"].mean(axis=1))
            h1p = paired_stats(data["S1P"]["cer"].mean(axis=1), data["S4"]["cer"].mean(axis=1))
            sweep[f"{sg:g}"] = {"tuned_params": params.__dict__, "systems": summ, "H1_S1_vs_S4_cer": h1, "H1_S1plus_vs_S4_cer": h1p}
        summary["sweep"] = sweep
        lines = ["### Noise sweep (iid Gaussian gaze noise, per-axis sigma in normalised canvas units; a key is 0.10 wide, 0.25 tall)", "",
                 "| sigma | S0 CER | S1 CER | S1+ CER | S4 CER | S1→S4 reduction | S1+→S4 reduction | Wilcoxon p (S1+ vs S4) |", "|---|---|---|---|---|---|---|---|"]
        for sg, r in sweep.items():
            s = r["systems"]
            lines.append(f"| {sg} | {s['S0']['cer']['mean']:.3f} | {s['S1']['cer']['mean']:.3f} | {s['S1P']['cer']['mean']:.3f} | {s['S4']['cer']['mean']:.3f} | {r['H1_S1_vs_S4_cer']['rel_reduction_pct']:.1f}% | {r['H1_S1plus_vs_S4_cer']['rel_reduction_pct']:.1f}% | {r['H1_S1plus_vs_S4_cer']['wilcoxon_p']:.2g} |")
        md.append("\n".join(lines) + "\n")

        # ---------------- drift / online adaptation ----------------------------------------------------------------------
        print("[3/3] drift (online adaptation)")
        pool = pools["landmarks_loo"]
        params = Params(**summary["landmarks_loo"]["tuned_params"])
        session_phrases_base = phrases + phrases  # 40 phrases per session
        drift_out = {}
        for dname, drift in DRIFT_LEVELS.items():
            jobs = []
            for si in range(sessions):
                order = session_phrases_base[:]
                random.Random(SESSION_SEED + si).shuffle(order)
                for system, adapt in (("S2", False), ("S3", True), ("S4", False), ("S4", True)):
                    jobs.append((system, adapt, params, pool, order, SESSION_SEED + si, drift))
            results = list(ex.map(_session_task, jobs, chunksize=4))
            cols = {}
            for system, adapt, seed, rows in results:
                cols.setdefault((system, adapt), {})[seed] = np.array([r["cer"] for r in rows])
            names = {("S2", False): "S2 (no adaptation)", ("S3", True): "S3 (S2 + online bias)", ("S4", False): "S4 (static)", ("S4", True): "S4 + online bias"}
            curves = {names[k]: np.vstack([v[s] for s in sorted(v)]) for k, v in cols.items()}  # (sessions, 40)
            sess_mean = {n: c.mean(axis=1) for n, c in curves.items()}
            late = {n: c[:, -10:].mean(axis=1) for n, c in curves.items()}
            drift_out[dname] = {
                "drift_total_xy": list(drift),
                "session_mean_cer": {n: bootstrap_ci(v) for n, v in sess_mean.items()},
                "last10_cer": {n: bootstrap_ci(v) for n, v in late.items()},
                "H2_S2_vs_S3_session": paired_stats(sess_mean["S2 (no adaptation)"], sess_mean["S3 (S2 + online bias)"]),
                "H2_S2_vs_S3_last10": paired_stats(late["S2 (no adaptation)"], late["S3 (S2 + online bias)"]),
                "H2_S4_static_vs_adaptive_last10": paired_stats(late["S4 (static)"], late["S4 + online bias"]),
                "curves_mean": {n: c.mean(axis=0).tolist() for n, c in curves.items()},
            }
        summary["drift"] = drift_out
        lines = ["### Drift: 40-phrase sessions with the gaze bias growing linearly from 0 to the stated total (posture drift)", "",
                 "| drift (x, y) | System | session CER | last-10-phrase CER |", "|---|---|---|---|"]
        for dname, r in drift_out.items():
            for n in r["session_mean_cer"]:
                a, b = r["session_mean_cer"][n], r["last10_cer"][n]
                lines.append(f"| {dname} {tuple(r['drift_total_xy'])} | {n} | {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] | {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}] |")
        lines += ["", "| drift | H2: S2→S3, session CER | H2: S2→S3, last-10 CER | S4 static→adaptive, last-10 CER |", "|---|---|---|---|"]
        for dname, r in drift_out.items():
            f = lambda t: f"{t['rel_reduction_pct']:.1f}% (diff {t['mean_diff']:+.3f}, CI [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}], p={t['wilcoxon_p']:.2g}, n={t['n']} sessions)"
            lines.append(f"| {dname} | {f(r['H2_S2_vs_S3_session'])} | {f(r['H2_S2_vs_S3_last10'])} | {f(r['H2_S4_static_vs_adaptive_last10'])} |")
        md.append("\n".join(lines) + "\n")

    # ---------------- write outputs --------------------------------------------------------------------------------------
    with open(OUT / "per_phrase.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "phrase", "system", *METRICS])
        w.writerows(per_phrase_rows)
    summary["run_info"] = {"seconds": round(time.time() - t0, 1), "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(), "quick": quick}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    (OUT / "tables.md").write_text("# Results (trace-driven simulation)\n\nGenerated by `python -m experiments.run_results`" + (" --quick (SMOKE TEST, do not report)" if quick else "") + "\n\n" + "\n".join(md), encoding="utf-8")
    try:
        make_figures(summary)
    except Exception as exc:  # noqa: BLE001 -- figures are a convenience; the tables/JSON are the result
        print("figures skipped:", exc)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")
    return 0


def make_figures(summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"S0": "#888888", "S1": "#d95f02", "S1P": "#7570b3", "S4": "#1b9e77"}
    labels = {"S0": "S0", "S1": "S1", "S1P": "S1+", "S4": "S4"}
    # Fig 1: noise sweep
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    xs = [float(k) for k in summary["sweep"]]
    for s in ("S0", "S1", "S1P", "S4"):
        ys = [summary["sweep"][k]["systems"][s]["cer"]["mean"] for k in summary["sweep"]]
        lo = [summary["sweep"][k]["systems"][s]["cer"]["ci95"][0] for k in summary["sweep"]]
        hi = [summary["sweep"][k]["systems"][s]["cer"]["ci95"][1] for k in summary["sweep"]]
        ax.plot(xs, ys, "-o", color=colors[s], label=labels[s], ms=4)
        ax.fill_between(xs, lo, hi, color=colors[s], alpha=0.15)
    ax.set_xlabel("gaze noise σ (normalised canvas units, per axis)")
    ax.set_ylabel("character error rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_noise_sweep.png", dpi=200)
    plt.close(fig)
    # Fig 2: drift curves (measured drift)
    r = summary["drift"]["drift_measured"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for name, curve in r["curves_mean"].items():
        ax.plot(range(1, len(curve) + 1), curve, label=name, lw=1.4)
    ax.set_xlabel("phrase index within session")
    ax.set_ylabel("character error rate (mean over sessions)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_drift.png", dpi=200)
    plt.close(fig)
    # Fig 3: real conditions
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    width = 0.2
    for i, s in enumerate(("S0", "S1", "S1P", "S4")):
        vals = [summary[c]["systems"][s]["cer"]["mean"] for c in ("landmarks_loo", "cnn_loo")]
        err = [[v - summary[c]["systems"][s]["cer"]["ci95"][0] for v, c in zip(vals, ("landmarks_loo", "cnn_loo"))], [summary[c]["systems"][s]["cer"]["ci95"][1] - v for v, c in zip(vals, ("landmarks_loo", "cnn_loo"))]]
        ax.bar([x + i * width for x in range(2)], vals, width, yerr=err, color=colors[s], label=labels[s], capsize=2)
    ax.set_xticks([x + 1.5 * width for x in range(2)], ["eye-landmark\nperception", "gaze-CNN\nperception"])
    ax.set_ylabel("character error rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_real_conditions.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
