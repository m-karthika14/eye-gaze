"""Typing-study results: what people actually typed, per participant and per system S0-S4.

    .venv/Scripts/python.exe -m experiments.study_summary

Reads data/sessions/<participant>.jsonl (one line per finished phrase, written by the app's Study bar: target phrase, typed text, elapsed
seconds, selections, backspaces, candidate picks, system, selection method) and writes results/study.md and results/study.json.

Per phrase: CER, WER, exact match, WPM = (characters/5)/minutes, selections per character (KSPC), backspaces per selection.
Unit of analysis = the PARTICIPANT: phrases are averaged within a participant and system, then summarised across participants with a
bootstrap 95% CI. Paired comparisons (Wilcoxon signed-rank over participants who did both systems) are against S4, descriptive unless you
pre-specified them; correct for the number of comparisons you report (Holm).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.sim_core import ROOT, cer, wer

SESSIONS_DIR = ROOT / "data" / "sessions"
MODES = ("s0", "s1", "s2", "s3", "s4")
LABELS = {"s0": "S0", "s1": "S1", "s2": "S2", "s3": "S3", "s4": "S4"}


def norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def phrase_stats(rec: dict) -> dict[str, float]:
    target, typed = norm(rec["target"]), norm(rec.get("typed", ""))
    minutes = max(float(rec.get("elapsed_s", 0.0)), 1e-6) / 60.0
    sel = int(rec.get("selections", 0))
    return {
        "cer": cer(target, typed),
        "wer": wer(target, typed),
        "exact": float(target == typed),
        "wpm": (len(target) / 5.0) / minutes,
        "kspc": sel / max(len(target), 1),
        "bs_rate": int(rec.get("backspaces", 0)) / max(sel, 1),
        "picks": float(rec.get("candidate_picks", 0)),
    }


def load_sessions(directory: Path = SESSIONS_DIR) -> list[dict]:
    recs = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs


def boot_ci(v: np.ndarray, n: int = 10_000) -> tuple[float, float, float]:
    if len(v) < 2:
        return float(v.mean()), float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(recs: list[dict]) -> dict:
    per: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in recs:
        if r.get("decoder_mode") in MODES and r.get("target"):
            per[(r["participant"], r["decoder_mode"])].append(phrase_stats(r))
    cells = {k: {m: float(np.mean([p[m] for p in v])) for m in v[0]} | {"n_phrases": len(v)} for k, v in per.items()}
    participants = sorted({p for p, _ in cells})
    out: dict = {"participants": participants, "modes": {}, "paired_vs_S4": {}}
    for mode in MODES:
        have = [p for p in participants if (p, mode) in cells]
        if not have:
            continue
        out["modes"][mode] = {"n_participants": len(have), "n_phrases": int(sum(cells[(p, mode)]["n_phrases"] for p in have))}
        for metric in ("cer", "wer", "exact", "wpm", "kspc", "bs_rate", "picks"):
            vals = np.array([cells[(p, mode)][metric] for p in have])
            m, lo, hi = boot_ci(vals)
            out["modes"][mode][metric] = {"mean": m, "ci95": [lo, hi]}
    if "s4" in out["modes"]:
        from scipy.stats import wilcoxon

        for mode in MODES:
            if mode == "s4" or mode not in out["modes"]:
                continue
            both = [p for p in participants if (p, mode) in cells and (p, "s4") in cells]
            if len(both) < 2:
                continue
            res = {"n_participants": len(both)}
            for metric in ("cer", "wer", "wpm"):
                a = np.array([cells[(p, mode)][metric] for p in both])
                b = np.array([cells[(p, "s4")][metric] for p in both])
                d = a - b
                try:
                    pv = 1.0 if np.allclose(d, 0) else float(wilcoxon(a, b).pvalue)
                except ValueError:
                    pv = 1.0
                m, lo, hi = boot_ci(d)
                res[metric] = {"mean_diff_vs_S4": m, "ci95": [lo, hi], "wilcoxon_p": pv}
            out["paired_vs_S4"][mode] = res
    return out


def main() -> int:
    recs_all = load_sessions()
    if any("mapping" in r for r in recs_all):  # one table per mapping: calibration-free (generic) vs personally calibrated
        parts = []
        for mapping in ("generic", "personal"):
            sub = [r for r in recs_all if r.get("mapping") == mapping]
            if sub:
                parts.append((mapping, sub))
    else:
        parts = [("all", recs_all)]
    if not recs_all:
        print(f"No study data yet in {SESSIONS_DIR}. Turn on the Study button in the app, enter a participant ID, and type phrases.")
        return 1
    fmt = lambda r, k, p=3: f"{r[k]['mean']:.{p}f}" + ("" if np.isnan(r[k]["ci95"][0]) else f" [{r[k]['ci95'][0]:.{p}f}, {r[k]['ci95'][1]:.{p}f}]")
    md = ["# Typing study", ""]
    out: dict = {}
    for mapping, recs in parts:
        s = summarise(recs)
        out[mapping] = s
        title = {"generic": "Calibration-free (generic mapping built from other people)", "personal": "After the person's own calibration", "all": "All phrases"}[mapping]
        md += [f"## {title}", "", f"{len(s['participants'])} participant(s): {', '.join(s['participants'])}. Unit of analysis: the participant "
               "(phrases averaged within participant and system); intervals are bootstrap 95% CIs over participants (undefined with one participant).", "",
               "| System | participants | phrases | CER | WER | exact | WPM | selections/char | backspaces/selection | candidate picks/phrase |", "|---|---|---|---|---|---|---|---|---|---|"]
        for mode, r in s["modes"].items():
            md.append(f"| {LABELS[mode]} | {r['n_participants']} | {r['n_phrases']} | {fmt(r,'cer')} | {fmt(r,'wer')} | {fmt(r,'exact',2)} | {fmt(r,'wpm',2)} | {fmt(r,'kspc',2)} | {fmt(r,'bs_rate',2)} | {fmt(r,'picks',2)} |")
        if s["paired_vs_S4"]:
            md += ["", "Paired differences against S4 (positive CER/WER difference = that system is worse; positive WPM difference = faster than S4):", "",
                   "| System | participants | CER diff [95% CI], p | WER diff, p | WPM diff, p |", "|---|---|---|---|---|"]
            for mode, r in s["paired_vs_S4"].items():
                c = lambda m: f"{r[m]['mean_diff_vs_S4']:+.3f} [{r[m]['ci95'][0]:+.3f}, {r[m]['ci95'][1]:+.3f}], p={r[m]['wilcoxon_p']:.2g}"
                md.append(f"| {LABELS[mode]} | {r['n_participants']} | {c('cer')} | {r['wer']['mean_diff_vs_S4']:+.3f}, p={r['wer']['wilcoxon_p']:.2g} | {r['wpm']['mean_diff_vs_S4']:+.2f}, p={r['wpm']['wilcoxon_p']:.2g} |")
            md += ["", "With few participants these tests have little power: report the CIs and say it is a pilot."]
        md.append("")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "study.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (ROOT / "results" / "study.json").write_text(json.dumps(out if len(out) > 1 or "all" not in out else out["all"], indent=2, default=float), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
