"""Trace-driven simulation core for the S0-S4 comparison (Phase 9).

"Trace-driven": the gaze noise is not an assumed Gaussian but is drawn from error vectors MEASURED on a real webcam, in your
calibration runs (data/calibration_history/*.jsonl): for every calibration trial we know the true key and where the mapping put the
gaze. Each simulated character is the true key's centre plus one such measured error vector. All five systems decode the SAME
simulated observation sequences (paired design), which is what makes the S1-vs-S4 comparison an isolation of H1.

What this is NOT: a human study. Real typing adds dwell/blink timing, fatigue, error-correction behaviour and temporal / spatial
correlation in the errors that bootstrapping independent vectors does not reproduce. Report it as a simulation, say so.

All functions are top-level and take plain data so they can run in a process pool.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from decoder.beam_search import beam_search_decode
from decoder.char_lm import ALPHABET, CharLM
from decoder.lexicon import Lexicon
from decoder.nearest_key import KeyboardLayout
from experiments.gaze_sequence import GazeObservation, char_to_key_id
from experiments.s1 import generic_confusion_likelihoods, hard_select_char
from experiments.s4 import real_gaze_likelihoods

ROOT = Path(__file__).resolve().parent.parent
HISTORY_DIR = ROOT / "data" / "calibration_history"

_CTX: dict = {}


def ctx() -> tuple[KeyboardLayout, CharLM, Lexicon | None, dict]:
    """Layout, character LM, lexicon and key-centre table (loaded once per process)."""
    if not _CTX:
        layout = KeyboardLayout.from_json()
        _CTX["layout"] = layout
        _CTX["lm"] = CharLM.load_default()
        _CTX["lex"] = Lexicon.load_default()
        _CTX["centres"] = {k.id: k.center() for k in layout.keys}
    return _CTX["layout"], _CTX["lm"], _CTX["lex"], _CTX["centres"]


# ------------------------------------------------------------------------------------------------ metrics
def levenshtein(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    return levenshtein(ref, hyp) / max(len(ref), 1)


def wer(ref: str, hyp: str) -> float:
    r = ref.split()
    return levenshtein(r, hyp.split()) / max(len(r), 1)


def phrase_metrics(ref: str, hyp: str) -> dict[str, float]:
    return {
        "cer": cer(ref, hyp),
        "wer": wer(ref, hyp),
        "char_acc": sum(a == b for a, b in zip(ref, hyp)) / len(ref),  # fixed-length: one observation per character
        "exact": float(ref == hyp),
    }


# ------------------------------------------------------------------------------------------------ noise sources
@dataclass
class ErrorPool:
    """A source of gaze error vectors (normalised canvas units) plus the sigma a calibrated system would report for it."""

    name: str
    vectors: np.ndarray | None  # (n, 2) measured errors; None => iid Gaussian
    sigma: tuple[float, float]
    description: str = ""

    def sample(self, rng: random.Random, scale: float = 1.0) -> tuple[float, float]:
        if self.vectors is None:
            return rng.gauss(0, self.sigma[0]) * scale, rng.gauss(0, self.sigma[1]) * scale
        v = self.vectors[rng.randrange(len(self.vectors))]
        return float(v[0]) * scale, float(v[1]) * scale


def gaussian_pool(sigma: float) -> ErrorPool:
    return ErrorPool(f"gauss_{sigma:g}", None, (sigma, sigma), f"iid Gaussian, sigma={sigma:g} per axis")


def cross_run_errors(runs: list[list[dict]], kind: str) -> list[np.ndarray]:
    """Cross-SITTING error: for each calibration run, fit on ALL OTHER runs and predict this one. This is what using a mapping later
    looks like (new seating, new lighting, new head position), as opposed to the within-run leave-one-trial-out error, which is
    measured inside one sitting and flatters the system (measured: RMSE ~0.12 within-run vs ~0.15-0.29 across runs, and about twice the key-hit rate).

      kind = "lm_eye"   eye-landmark ridge on the eye-region features, alpha = 10
             "lm_iris"  eye-landmark ridge on the iris-only features (the config Calibrate currently deploys), alpha = 10
             "cnn"      gaze-CNN absolute-angle affine fit
    The kind is fixed in advance of looking at any held-out run, not selected on it (with two runs there is no data left to select on).
    """
    from vision.eye_features import FEATURE_SUBSETS
    from vision.feature_model import fit_ridge

    usable = [r for r in runs if len(r) >= 12 and all("eye_feat" in x and "abs_yaw" in x for x in r)]
    if len(usable) < 2:
        return []
    out = []
    for i, held in enumerate(usable):
        train = [x for j, r in enumerate(usable) if j != i for x in r]
        y_tr = np.array([[x["true_x"], x["true_y"]] for x in train])
        y_te = np.array([[x["true_x"], x["true_y"]] for x in held])
        if kind == "cnn":
            a_tr = np.array([[x["abs_yaw"], x["abs_pitch"]] for x in train])
            a_te = np.array([[x["abs_yaw"], x["abs_pitch"]] for x in held])
            f_tr, f_te = np.c_[np.ones(len(a_tr)), a_tr], np.c_[np.ones(len(a_te)), a_te]
            pred = f_te @ np.linalg.solve(f_tr.T @ f_tr + 1e-4 * np.eye(3), f_tr.T @ y_tr)
        else:
            cols = FEATURE_SUBSETS["eye_region" if kind == "lm_eye" else "iris_core"]
            model = fit_ridge(np.array([x["eye_feat"] for x in train]), y_tr, 10.0, columns=cols)
            pred = model.predict(np.array([x["eye_feat"] for x in held]))
        out.append(pred - y_te)
    return out


# ------------------------------------------------------------------------------------------------ across PEOPLE (calibration-free test)
def person_of(path: Path, rows: list[dict]) -> str:
    """Who a calibration run belongs to: the `participant` field (set by the app's participant-ID box), else an id in the file name
    (YYYYMMDD_HHMMSS_<id>.jsonl), else "legacy" -- all runs recorded before participant ids existed are treated as ONE person."""
    pid = rows[0].get("participant") if rows else None
    if pid:
        return str(pid)
    m = re.match(r"\d{8}_\d{6}_(.+)\.jsonl$", path.name)
    if m and m.group(1) != "run":
        return m.group(1)
    return "legacy"


def load_runs_by_person(history_dir: Path = HISTORY_DIR) -> dict[str, list[list[dict]]]:
    """{person: [run, run, ...]} for runs that carry eye features and absolute angles (older runs cannot be used here)."""
    out: dict[str, list[list[dict]]] = {}
    for path in sorted(history_dir.glob("*.jsonl")):
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(rows) >= 12 and all("eye_feat" in r and "abs_yaw" in r for r in rows):
            out.setdefault(person_of(path, rows), []).append(rows)
    return out


def _fit_predict(kind: str, train: list[dict], test: list[dict]) -> np.ndarray:
    from vision.eye_features import FEATURE_SUBSETS
    from vision.feature_model import fit_ridge

    y_tr = np.array([[x["true_x"], x["true_y"]] for x in train])
    if kind == "cnn":
        a_tr = np.array([[x["abs_yaw"], x["abs_pitch"]] for x in train])
        a_te = np.array([[x["abs_yaw"], x["abs_pitch"]] for x in test])
        f_tr, f_te = np.c_[np.ones(len(a_tr)), a_tr], np.c_[np.ones(len(a_te)), a_te]
        return f_te @ np.linalg.solve(f_tr.T @ f_tr + 1e-4 * np.eye(3), f_tr.T @ y_tr)
    cols = FEATURE_SUBSETS["eye_region" if kind == "lm_eye" else "iris_core" if kind == "lm_iris" else "full"]
    model = fit_ridge(np.array([x["eye_feat"] for x in train]), y_tr, 10.0, columns=cols)
    return model.predict(np.array([x["eye_feat"] for x in test]))


def lopo_predictions(by_person: dict[str, list[list[dict]]], kind: str, recentre: bool = False) -> dict[str, dict]:
    """LEAVE-ONE-PERSON-OUT: for each person, a mapping fitted ONLY on everybody else's calibration trials is applied to that person
    with no calibration of their own -- what a calibration-free deployment would do on day one.

      recentre=True additionally applies a one-point correction (what pressing C while looking at the middle of the keyboard does): the
      held-out person's trial closest to the screen centre gives an offset that is added to all their predictions (that trial is then
      excluded from scoring so it cannot flatter the result).
    Returns {person: {"pred", "truth", "keys", "n"}} (arrays over that person's scored trials).
    """
    persons = sorted(by_person)
    out = {}
    for held in persons:
        train = [x for p in persons if p != held for run in by_person[p] for x in run]
        test = [x for run in by_person[held] for x in run]
        if not train or not test:
            continue
        pred = _fit_predict(kind, train, test)
        truth = np.array([[x["true_x"], x["true_y"]] for x in test])
        keys = [x["target_key"] for x in test]
        if recentre:
            anchor = int(np.argmin(((truth - 0.5) ** 2).sum(axis=1)))
            pred = pred + (truth[anchor] - pred[anchor])
            keep = np.arange(len(test)) != anchor
            pred, truth, keys = pred[keep], truth[keep], [k for k, m in zip(keys, keep) if m]
        out[held] = {"pred": pred, "truth": truth, "keys": keys, "n": len(keys)}
    return out


def load_error_pools(history_dir: Path = HISTORY_DIR) -> dict[str, ErrorPool]:
    """Real measured error pools from every calibration run on disk.

      landmarks_loo        held-out error of the eye-landmark mapping, leave-one-trial-out INSIDE a run (a freshly calibrated system)
      cnn_loo              same for the gaze-CNN mapping
      landmarks_xrun_eye   cross-sitting error (fit on the other runs, predict this one): eye-region features
      landmarks_xrun_iris  cross-sitting error, iris-only features (what Calibrate currently deploys)
      cnn_xrun             cross-sitting error, gaze CNN
      as_recorded          error of whichever mapping was LIVE during the run (includes between-session drift when it was stale)
    """
    from experiments.calibrate_uncertainty import fit_affine, fit_feature_mapping

    feat, cnn, rec = [], [], []
    all_runs: list[list[dict]] = []
    runs = 0
    for path in sorted(history_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) < 12:
            continue
        runs += 1
        all_runs.append(rows)
        truth = np.array([[r["true_x"], r["true_y"]] for r in rows])
        rec.append(np.array([[r["mu_x"], r["mu_y"]] for r in rows]) - truth)
        ff = fit_feature_mapping(rows)
        if ff is not None:
            feat.append(np.array(ff["loo_mu"]) - truth)
        cf = fit_affine(rows)
        if cf is not None:
            cnn.append(np.array(cf["loo_mu"]) - truth)

    def pool(name: str, parts: list[np.ndarray], desc: str) -> ErrorPool | None:
        if not parts:
            return None
        v = np.vstack(parts)
        return ErrorPool(name, v, (float(v[:, 0].std()), float(v[:, 1].std())), f"{desc} (n={len(v)} measured trials from {len(parts)} run(s))")

    pools = {
        p.name: p
        for p in (
            pool("landmarks_loo", feat, "eye-landmark mapping, within-run held-out error"),
            pool("cnn_loo", cnn, "gaze-CNN mapping, within-run held-out error"),
            pool("landmarks_xrun_eye", cross_run_errors(all_runs, "lm_eye"), "eye-landmark (eye-region) mapping, cross-sitting error"),
            pool("landmarks_xrun_iris", cross_run_errors(all_runs, "lm_iris"), "eye-landmark (iris-only) mapping, cross-sitting error"),
            pool("cnn_xrun", cross_run_errors(all_runs, "cnn"), "gaze-CNN mapping, cross-sitting error"),
            pool("as_recorded", rec, "live mapping at calibration time, incl. drift"),
        )
        if p is not None
    }
    # Across people (needs >= 3 people with participant ids): the calibration-free condition.
    by_person = load_runs_by_person(history_dir)
    if len(by_person) >= 3:
        for name, kind, recentre, desc in (
            ("landmarks_lopo", "lm_eye", False, "eye-landmark mapping fitted on OTHER people, no calibration of their own"),
            ("landmarks_lopo_recentre", "lm_eye", True, "same, plus a one-point re-centre (press C looking at the middle)"),
            ("cnn_lopo", "cnn", False, "gaze-CNN mapping fitted on OTHER people, no calibration of their own"),
        ):
            res = lopo_predictions(by_person, kind, recentre)
            vecs = np.vstack([r["pred"] - r["truth"] for r in res.values()]) if res else None
            if vecs is not None and len(vecs) >= 12:
                pools[name] = ErrorPool(name, vecs, (float(vecs[:, 0].std()), float(vecs[:, 1].std())), f"{desc} (n={len(vecs)} trials, {len(res)} people)")
    return pools


# ------------------------------------------------------------------------------------------------ observations
def make_observations(
    phrase: str,
    pool: ErrorPool,
    rng: random.Random,
    scale: float = 1.0,
    bias: tuple[float, float] = (0.0, 0.0),
    sigma_report: tuple[float, float] | None = None,
) -> list[GazeObservation]:
    """One observation per character: true key centre + a measured error vector (+ an optional slow bias, i.e. drift)."""
    _, _, _, centres = ctx()
    sx, sy = sigma_report or (pool.sigma[0] * scale, pool.sigma[1] * scale)
    out = []
    for ch in phrase:
        cx, cy = centres[char_to_key_id(ch)]
        ex, ey = pool.sample(rng, scale)
        out.append(GazeObservation(mu_x=cx + ex + bias[0], mu_y=cy + ey + bias[1], sigma_x=sx, sigma_y=sy, true_char=ch))
    return out


# ------------------------------------------------------------------------------------------------ decoders
@dataclass(frozen=True)
class Params:
    """Per-system tunables. Every system gets the same tuning budget (see run_results.tune)."""

    s1_match: float = 0.85  # experiments/s1.py default
    s1_lm: float = 0.25  # beam_search_decode default
    s4_lm: float = 0.25
    s4_sigma_mult: float = 1.0
    s1p_sigma_mult: float = 1.0  # S1+: width of its (hard-key-centred) noise model, as a multiple of the calibrated per-axis sigma
    s1p_lm: float = 0.25
    beam: int = 8


def _shift(obs: list[GazeObservation], bias: tuple[float, float]) -> list[GazeObservation]:
    if bias == (0.0, 0.0):
        return obs
    return [GazeObservation(o.mu_x - bias[0], o.mu_y - bias[1], o.sigma_x, o.sigma_y, o.true_char) for o in obs]


def decode(system: str, obs: list[GazeObservation], p: Params, bias: tuple[float, float] = (0.0, 0.0)) -> str:
    """S0 nearest key | S1/S2/S3 hard key + generic-confusion LM correction | S4 real gaze distribution x LM.
    `bias` (S3/S4 online adaptation) is subtracted from the gaze first."""
    layout, lm, _, _ = ctx()
    obs = _shift(obs, bias)
    if system == "S0":
        return "".join(hard_select_char(o, layout) for o in obs)
    if system in ("S1", "S2", "S3"):  # S2 decodes identically to S1 (experiments/s2.py); S3 differs only through `bias`
        lik = [generic_confusion_likelihoods(hard_select_char(o, layout), p.s1_match) for o in obs]
        return beam_search_decode(lik, lm, p.beam, lm_weight=p.s1_lm)
    if system == "S1P":
        # S1+ : the strongest HARD-selection baseline. Gaze is still collapsed to the selected key first (so the continuous position
        # inside / around the key is discarded), but the post-hoc corrector knows the keyboard: the chance that the user meant key c
        # given that key h was selected falls off with the distance between their centres, using the calibrated per-axis sigma.
        # Compared with S4 it isolates exactly one thing -- keeping the continuous gaze distribution vs keeping only the key.
        _, _, _, centres = ctx()
        lik = []
        for o in obs:
            hx, hy = centres[char_to_key_id(hard_select_char(o, layout))]
            sx, sy = max(o.sigma_x * p.s1p_sigma_mult, 1e-3), max(o.sigma_y * p.s1p_sigma_mult, 1e-3)
            w = {}
            for c in ALPHABET:
                cx, cy = centres[char_to_key_id(c)]
                w[c] = math.exp(-0.5 * (((cx - hx) / sx) ** 2 + ((cy - hy) / sy) ** 2))
            z = sum(w.values())
            lik.append({c: v / z for c, v in w.items()})
        return beam_search_decode(lik, lm, p.beam, lm_weight=p.s1p_lm)
    if system == "S4":
        scaled = [GazeObservation(o.mu_x, o.mu_y, o.sigma_x * p.s4_sigma_mult, o.sigma_y * p.s4_sigma_mult, o.true_char) for o in obs]
        lik = [real_gaze_likelihoods(o, layout) for o in scaled]
        return beam_search_decode(lik, lm, p.beam, lm_weight=p.s4_lm)
    raise ValueError(system)


# ------------------------------------------------------------------------------------------------ online adaptation (S3 / S4)
@dataclass
class OnlineBias:
    """Same rule as decoder/live_modes.S3LiveDecoder: learn a gaze bias from LM corrections of REAL words only.
    Where the system's decoded word differs from the hard-selected keys, the gaze at the changed letters vs the letters they were
    changed to is evidence of bias. Letters left unchanged teach nothing. No oracle: it never sees the true text."""

    rate: float = 0.25
    clip: float = 0.15
    max_residual: float = 0.25
    min_word_len: int = 3
    value: list[float] = field(default_factory=lambda: [0.0, 0.0])

    def as_tuple(self) -> tuple[float, float]:
        return (self.value[0], self.value[1])

    def learn(self, shifted_obs: list[GazeObservation], decoded: str) -> None:
        layout, _, lex, centres = ctx()
        if lex is None:
            return
        hard = "".join(hard_select_char(o, layout) for o in shifted_obs)
        for m in re.finditer(r"[a-z]+", decoded):
            a, b = m.span()
            word, raw = decoded[a:b], hard[a:b]
            if word == raw or len(word) < self.min_word_len or word not in lex.word_logp:
                continue
            for i, (c_raw, c_new) in enumerate(zip(raw, word)):
                if c_raw == c_new:
                    continue
                cx, cy = centres[char_to_key_id(c_new)]
                rx, ry = shifted_obs[a + i].mu_x - cx, shifted_obs[a + i].mu_y - cy
                if abs(rx) > self.max_residual or abs(ry) > self.max_residual:
                    continue
                self.value[0] = max(-self.clip, min(self.clip, self.value[0] + self.rate * rx))
                self.value[1] = max(-self.clip, min(self.clip, self.value[1] + self.rate * ry))


def run_session(
    system: str,
    phrases: list[str],
    pool: ErrorPool,
    seed: int,
    p: Params,
    adapt: bool,
    drift_total: tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
) -> list[dict[str, float]]:
    """One typing session: `phrases` in order, a slowly growing bias `drift_total * (i / n)` added to the gaze (posture drift),
    and -- if `adapt` (S3 / S4 online adaptation) -- a bias estimate that persists and is updated after every phrase.
    The observation stream depends only on `seed`, so every system sees identical gaze."""
    rng = random.Random(seed)
    bias = OnlineBias()
    out = []
    n = len(phrases)
    for i, phrase in enumerate(phrases):
        frac = i / max(n - 1, 1)
        obs = make_observations(phrase, pool, rng, scale, (drift_total[0] * frac, drift_total[1] * frac))
        decoded = decode(system, obs, p, bias.as_tuple() if adapt else (0.0, 0.0))
        out.append(phrase_metrics(phrase, decoded))
        if adapt:
            bias.learn(_shift(obs, bias.as_tuple()), decoded)
    return out
