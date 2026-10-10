"""Phase 3 deliverable: empirically calibrate the fixation uncertainty (sigma)
against known targets, instead of trusting the raw temporal scatter blindly.

Shows a sequence of keys on screen, one at a time, and asks you to look at
each. Once your gaze settles into a stable fixation, records the predicted
(mu, sigma) against the key's known true center. After all trials, computes
how well sigma matches your ACTUAL error using a robust (median-based)
statistic -- see analyze()'s docstring for why mean(z^2) alone is the wrong
tool here. Writes the resulting correction factor back into
config/decoder_params.yaml's gaze_uncertainty.sigma_scale.

Waits for the pipeline's startup bias-lock to finish before recording any
trial (vision/gaze_to_screen.py reports a placeholder center point with
exactly-zero scatter during that ~1s window -- recording through it produces
fake "trials" that are actually measuring nothing).

Run from the repo root (only one process can hold the webcam -- make sure
the backend isn't running):

    .venv/Scripts/python.exe -m experiments.calibrate_uncertainty
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ruamel.yaml import YAML

from decoder.nearest_key import KeyboardLayout
from vision.pipeline import GazePipeline

# Round-trip loader/dumper: preserves comments and formatting on write-back,
# unlike plain PyYAML's safe_load/safe_dump which silently discards them --
# these config files are hand-documented and get re-written by this script.
yaml = YAML()
yaml.preserve_quotes = True

DECODER_PARAMS_PATH = Path(__file__).resolve().parent.parent / "config" / "decoder_params.yaml"
GAZE_MAPPING_PATH = Path(__file__).resolve().parent.parent / "config" / "gaze_mapping.yaml"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLES_PATH = DATA_DIR / "uncertainty_calibration_samples.jsonl"

CANVAS_SIZE = (720, 1280)  # h, w -- stand-in for "the screen", matches vision/run_demo.py

# A spread across the keyboard, weighted towards the EDGES and corners: those are the
# trials that pin down the gain of the gaze-to-screen mapping, and the keys on the far
# left/right/top (Q W P O A L Z M ...) were the ones that "weren't reachable" when the
# fit was dominated by central keys. 16 keys x 2 repeats = 32 trials.
TARGET_KEYS = ["Q", "W", "P", "O", "A", "S", "L", "Z", "X", "M", "T", "D", "H", "B", "SPACE", "BACKSPACE"]

# Each target is visited this many times (in randomized order, not back to
# back) so the estimate isn't skewed by one or two rough trials -- a handful
# of trials per key is not enough to trust a single global scale factor from.
# 3 repeats proved too few in practice: real per-key data showed single
# trials swinging 2-3x from their siblings (webcam appearance-based gaze
# noise, not a bug), so the median/robust estimators below didn't have
# enough samples per key to average that out. 3 x 12 keys = 36 trials, ~2.5 min.
REPEATS = 2

# A recorded trial must be a steady fixation of at least this long AND contain
# at least MIN_FIXATION_SAMPLES gaze samples. The sample floor is what actually
# protects against a slow pipeline: the previous 400ms / no-floor setting
# recorded "fixations" of 2 frames each when the pipeline ran at ~3 fps, so
# every trial was effectively a single noisy reading.
TARGET_FIXATION_MS = 500.0
MIN_FIXATION_SAMPLES = 4  # the aggregator keeps <=800ms, so this still needs >= ~5 fps
SETTLE_TIMEOUT_S = 7.0  # give up on a trial if no stable fixation forms in 7s
# The target appears, then the eye needs time to travel there (saccade +
# camera/inference latency). Anything the fixation aggregator saw before this
# is the *previous* target (or the transit) -- discard it. Without this a trial
# could record the old target's fixation the instant the new one appeared,
# which is exactly what the pre-existing calibration file shows (gaze-vs-target
# correlation ~0.2-0.3, with the strongest correlation at a one-trial lag).
SETTLE_DELAY_S = 0.9


def draw_prompt(layout: KeyboardLayout, target_id: str, status: str) -> np.ndarray:
    canvas = np.zeros((*CANVAS_SIZE, 3), dtype=np.uint8)
    for key in layout.keys:
        x1 = int(key.x * CANVAS_SIZE[1])
        y1 = int(key.y * CANVAS_SIZE[0])
        x2 = int((key.x + key.width) * CANVAS_SIZE[1])
        y2 = int((key.y + key.height) * CANVAS_SIZE[0])
        color = (0, 200, 0) if key.id == target_id else (60, 60, 60)
        thickness = -1 if key.id == target_id else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            canvas, key.id, (x1 + 8, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
    cv2.putText(canvas, f"LOOK AT: {target_id}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(canvas, status, (20, CANVAS_SIZE[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    return canvas


def warm_up(pipeline: GazePipeline) -> bool:
    """Consume frames until the mapper's startup bias-lock completes. Before
    that, map() reports a placeholder center point with zero scatter --
    recording through it produces fake trials. Returns False if 'q' was
    pressed or the stream ended before warm-up finished.
    """
    for sample in pipeline.stream():
        canvas = np.zeros((*CANVAS_SIZE, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            "Warming up -- look at the screen...",
            (20, CANVAS_SIZE[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        cv2.imshow("Uncertainty calibration", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
        if pipeline.mapper.bias_locked:
            return True
    return False


def run_trials(layout: KeyboardLayout) -> list[dict]:
    pipeline = GazePipeline(camera_index=0, target_fps=30, gestures=False)
    samples: list[dict] = []

    trial_order = TARGET_KEYS * REPEATS
    random.shuffle(trial_order)

    with pipeline:
        if not warm_up(pipeline):
            cv2.destroyAllWindows()
            return samples

        for target_id in trial_order:
            target_key = next(k for k in layout.keys if k.id == target_id)
            true_cx, true_cy = target_key.center()

            trial_start = time.monotonic()
            recorded = False
            for sample in pipeline.stream():
                elapsed = time.monotonic() - trial_start
                fixation = sample.get("fixation")
                status = f"trial {len(samples) + 1}/{len(trial_order)}  target={target_id}  elapsed={elapsed:.1f}s"
                if fixation is not None:
                    status += f"  fixation={fixation.duration_ms:.0f}ms n={fixation.n_samples}"

                canvas = draw_prompt(layout, target_id, status)
                cv2.imshow("Uncertainty calibration", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    return samples

                if elapsed < SETTLE_DELAY_S:
                    pipeline.fixation.reset()  # drop transit / previous-target samples
                    continue

                if (
                    fixation is not None
                    and fixation.duration_ms >= TARGET_FIXATION_MS
                    and fixation.n_samples >= MIN_FIXATION_SAMPLES
                ):
                    samples.append(
                        {
                            "target_key": target_id,
                            "true_x": true_cx,
                            "true_y": true_cy,
                            "mu_x": fixation.mu_x,
                            "mu_y": fixation.mu_y,
                            "sigma_x": fixation.sigma_x,
                            "sigma_y": fixation.sigma_y,
                            "n_samples": fixation.n_samples,
                            # Raw angle + the range in use at trial time --
                            # lets range_from_samples() reconstruct what range
                            # WOULD have been needed to reach the true target,
                            # without the information loss mu_x/mu_y suffer
                            # once clamping kicks in near screen edges.
                            "rel_yaw": pipeline.mapper.last_rel_yaw,
                            "rel_pitch": pipeline.mapper.last_rel_pitch,
                            "range_yaw": pipeline.mapper.range_yaw,
                            "range_pitch": pipeline.mapper.range_pitch,
                            **face_fields(sample),
                        }
                    )
                    recorded = True
                    break

                if elapsed > SETTLE_TIMEOUT_S:
                    print(f"[skip] no stable fixation on {target_id} within {SETTLE_TIMEOUT_S:.0f}s")
                    break

            if not recorded and elapsed <= SETTLE_TIMEOUT_S:
                break  # user pressed 'q'

    cv2.destroyAllWindows()
    return samples


def compute_accuracy_dashboard(samples: list[dict], layout: KeyboardLayout) -> dict:
    """Pure computation behind the accuracy dashboard -- per-trial target vs.
    predicted position, error, and hit/miss, plus the aggregate key-hit rate,
    mean signed error (directional bias vs. plain noise), and a left/right,
    top/bottom split (surfaces axis-specific problems like a yaw-inversion
    bug at a glance). Returns plain data so both the CLI script
    (print_accuracy_dashboard, below) and the in-app calibration flow
    (backend/calibration.py) can present the same numbers in their own UI.
    """
    from decoder.nearest_key import nearest_key

    rows = []
    for s in samples:
        err_x, err_y = s["mu_x"] - s["true_x"], s["mu_y"] - s["true_y"]
        euclidean = (err_x**2 + err_y**2) ** 0.5
        predicted_key = nearest_key(s["mu_x"], s["mu_y"], layout)
        hit = predicted_key == s["target_key"]
        rows.append(
            {
                "target": s["target_key"],
                "predicted": predicted_key,
                "err_x": err_x,
                "err_y": err_y,
                "euclidean": euclidean,
                "hit": hit,
            }
        )
    rows.sort(key=lambda r: -r["euclidean"])

    hit_rate = sum(r["hit"] for r in rows) / len(rows)
    mean_err_x = statistics.fmean(r["err_x"] for r in rows)
    mean_err_y = statistics.fmean(r["err_y"] for r in rows)

    splits = {}
    for axis, label in [("x", "left_right"), ("y", "top_bottom")]:
        true_key = f"true_{axis}"
        mu_key = f"mu_{axis}"
        low = [s for s in samples if s[true_key] < 0.5]
        high = [s for s in samples if s[true_key] >= 0.5]
        if low and high:
            splits[label] = {
                "low_mean_abs_error": statistics.fmean(abs(s[mu_key] - s[true_key]) for s in low),
                "high_mean_abs_error": statistics.fmean(abs(s[mu_key] - s[true_key]) for s in high),
            }

    return {
        "rows": rows,
        "hit_rate": hit_rate,
        "hits": sum(r["hit"] for r in rows),
        "total": len(rows),
        "mean_err_x": mean_err_x,
        "mean_err_y": mean_err_y,
        "splits": splits,
    }


def print_accuracy_dashboard(samples: list[dict], layout: KeyboardLayout) -> None:
    """Terminal presentation of compute_accuracy_dashboard() -- see there for
    what each number means and why (Section 18's "key hit accuracy" is the
    single most meaningful number for this application, not an arbitrary
    aggregate score)."""
    d = compute_accuracy_dashboard(samples, layout)
    print()
    print("=== Accuracy dashboard (worst first) ===")
    print(f"{'target':<10}{'predicted':<10}{'err_x':>8}{'err_y':>8}{'err_px(1920x1080)':>20}{'hit?':>6}")
    for r in d["rows"]:
        px = f"({r['err_x'] * 1920:+.0f}, {r['err_y'] * 1080:+.0f})"
        print(f"{r['target']:<10}{r['predicted'] or '?':<10}{r['err_x']:>+8.3f}{r['err_y']:>+8.3f}{px:>20}{'YES' if r['hit'] else 'no':>6}")

    print()
    print(f"[dashboard] key-hit accuracy: {d['hit_rate'] * 100:.0f}% ({d['hits']}/{d['total']} trials landed on the intended key)")
    print(f"[dashboard] mean signed error: x={d['mean_err_x']:+.3f} y={d['mean_err_y']:+.3f}  "
          f"(consistently nonzero = directional bias; near-zero with high scatter = noise, not bias)")

    if "left_right" in d["splits"]:
        s = d["splits"]["left_right"]
        print(f"[dashboard] LEFT vs RIGHT: LEFT mean abs error={s['low_mean_abs_error']:.3f}  RIGHT mean abs error={s['high_mean_abs_error']:.3f}")
    if "top_bottom" in d["splits"]:
        s = d["splits"]["top_bottom"]
        print(f"[dashboard] TOP vs BOTTOM: TOP mean abs error={s['low_mean_abs_error']:.3f}  BOTTOM mean abs error={s['high_mean_abs_error']:.3f}")


# For Z ~ N(0, 1), median(|Z|) = 0.6745 (the 75th-percentile point of the
# standard normal). This is the standard median-absolute-deviation robust
# scale estimator: unlike mean(z^2), which squares every error and lets one
# or two bad trials (a blink, a momentary tracking glitch, a wrong key
# looked at) dominate the whole result, the median only cares about the
# typical trial and barely moves if a handful are way off.
MEDIAN_ABS_Z_STANDARD_NORMAL = 0.6745


def compute_sigma_scale(samples: list[dict], min_sigma: float) -> dict:
    """Pure computation behind analyze() -- see there for what sigma_scale
    means. Returns both the robust (median-based) and naive (mean(z^2)-based)
    estimates so a caller (CLI or in-app) can show how much outliers would
    have skewed the naive one."""
    z_abs: list[float] = []
    for s in samples:
        sigma_x = max(s["sigma_x"], min_sigma)
        sigma_y = max(s["sigma_y"], min_sigma)
        z_abs.append(abs((s["mu_x"] - s["true_x"]) / sigma_x))
        z_abs.append(abs((s["mu_y"] - s["true_y"]) / sigma_y))

    median_abs_z = statistics.median(z_abs)
    sigma_scale = median_abs_z / MEDIAN_ABS_Z_STANDARD_NORMAL
    mean_z_sq = sum(z * z for z in z_abs) / len(z_abs)
    naive_scale = mean_z_sq**0.5

    return {"sigma_scale": sigma_scale, "naive_scale": naive_scale, "n_trials": len(samples)}


def analyze(samples: list[dict], min_sigma: float) -> float:
    """Returns sigma_scale such that (mu - true) / (sigma * scale) behaves
    like a standard normal across trials, estimated robustly via the median
    of |z| rather than the mean of z^2."""
    result = compute_sigma_scale(samples, min_sigma)
    sigma_scale, naive_scale = result["sigma_scale"], result["naive_scale"]

    print()
    print(f"[analysis] {len(samples)} trials")
    print(f"[analysis] robust (median-based) sigma_scale = {sigma_scale:.3f}")
    print(f"[analysis] naive (mean(z^2)-based) sigma_scale = {naive_scale:.3f}  (for comparison)")
    if naive_scale > sigma_scale * 1.5:
        print("[analysis] naive stat is much larger than the robust one -- a few outlier trials are dominating it")
    if sigma_scale > 1.3:
        print("[analysis] raw sigma UNDERESTIMATES real error -- scaling it up")
    elif sigma_scale < 0.7:
        print("[analysis] raw sigma OVERESTIMATES real error -- scaling it down")
    else:
        print("[analysis] raw sigma is reasonably well-calibrated already")
    return sigma_scale


def analyze_range(samples: list[dict]) -> tuple[float, float] | None:
    """Reconstructs the range (half_fov_yaw_rad/half_fov_pitch_rad) that
    WOULD have mapped this person's actual observed gaze angle exactly onto
    each trial's true target -- i.e. what the auto-expanding range in
    vision/gaze_to_screen.py should have converged to, computed directly
    instead of hoping the initial guess was already close enough for it to
    trigger expansion from.

    Uses rel_yaw/rel_pitch (the raw bias-corrected angle) rather than
    mu_x/mu_y, since mu_x/mu_y are clamped to [0,1] and lose information
    exactly on the corner cases we most need this to be accurate for.
    Returns None if there isn't enough off-center data.
    """
    yaw_estimates, pitch_estimates = [], []
    for s in samples:
        if "rel_yaw" not in s:
            continue  # samples recorded before this field existed
        dx, dy = s["true_x"] - 0.5, s["true_y"] - 0.5
        # Skip near-center targets -- dividing by a small dx/dy amplifies
        # ordinary measurement noise into wildly unstable per-trial estimates
        # (observed empirically: the home row's dy=0.125 trials produced
        # implied half_fov values spanning 0.003 to 1.43 -- a ~500x spread --
        # which then dominates/pollutes the median below). 0.25 keeps only
        # the outermost row/column on each axis, where the denominator is
        # large enough that per-trial noise stays a minor fraction of the
        # estimate instead of the whole story.
        if abs(dx) > 0.25:
            yaw_estimates.append(abs(s["rel_yaw"]) / (2 * abs(dx)))
        if abs(dy) > 0.25:
            pitch_estimates.append(abs(s["rel_pitch"]) / (2 * abs(dy)))

    if len(yaw_estimates) < 4 or len(pitch_estimates) < 4:
        return None
    return statistics.median(yaw_estimates), statistics.median(pitch_estimates)


def face_fields(sample: dict) -> dict:
    """Head-position fields to store with a calibration trial (see fit_affine)."""
    out: dict = {}
    f = sample.get("face_feat")
    if f is not None:
        out.update({"face_cx": f[0], "face_cy": f[1], "face_w": f[2]})
    a = sample.get("raw_angle_mean")  # (pitch, yaw) averaged over the last 0.5s, absolute (no session zero point)
    if a is not None:
        out.update({"abs_pitch": a[0], "abs_yaw": a[1]})
    e = sample.get("eye_feat_mean")  # EyeTrax-style eye/iris features averaged over the last 0.5s (vision/eye_features.py)
    if e is not None:
        out["eye_feat"] = [round(float(v), 5) for v in e]
    return out


def _inverse_fit(ang: np.ndarray, pos: np.ndarray, feats: np.ndarray | None, ridge: float = 1e-4, ridge_feat: float = 2.0):
    """angle = b0 + B1 @ pos + G @ head  (regress angles on positions: noise is only in the dependent variable,
    so no regression dilution), then invert for position. Returns (W, ref) with
    pos = [1, yaw, pitch, head - ref] @ W, or None if the relation is near-singular."""
    n = len(pos)
    if feats is None:
        ref, sd, fs = None, None, None
        design = np.c_[np.ones(n), pos]
    else:
        ref = feats.mean(axis=0)
        sd = np.maximum(feats.std(axis=0), 1e-3)
        fs = (feats - ref) / sd
        design = np.c_[np.ones(n), pos, fs]
    reg = np.diag([ridge] * 3 + [ridge_feat] * (design.shape[1] - 3))
    c = np.linalg.solve(design.T @ design + reg, design.T @ ang)  # (cols x 2): [yaw, pitch]
    b0, b1 = c[0], c[1:3].T
    if abs(np.linalg.det(b1)) < 1e-6 or np.linalg.cond(b1) > 50:
        return None
    inv = np.linalg.inv(b1)
    rows = [-(inv @ b0), inv.T[0], inv.T[1]]
    w = np.vstack([rows[0], rows[1], rows[2]])
    if feats is not None:
        g = c[3:].T  # (2 x k) in standardized units
        w = np.vstack([w, (-(inv @ g)).T / sd[:, None]])
    return w, ref


def _forward_fit(ang: np.ndarray, pos: np.ndarray, ridge: float = 1e-4) -> tuple[np.ndarray, None]:
    """Direct least squares, pos = [1, yaw, pitch] @ W. Lowest error on noisy data, but shrinks towards
    the centre (regression dilution), so the extremes are under-reached."""
    design = np.c_[np.ones(len(ang)), ang]
    return np.linalg.solve(design.T @ design + ridge * np.eye(3), design.T @ pos), None


MIN_HEAD_SPREAD = 0.008  # head position features must vary at least this much (fraction of frame) during calibration


def fit_affine(samples: list[dict], ridge: float = 1e-4, min_trials: int = 12) -> dict | None:
    """Learn the screen mapping from the calibration trials:

        x = c0 + c1*yaw + c2*pitch [+ c3*dcx + c4*dcy + c5*dw],   y likewise

    yaw/pitch are the bias-corrected gaze angles recorded at each trial; dcx/dcy/dw (optional) are the
    face's position and width in the camera frame relative to their calibration average.

    Three candidate fits are compared leave-one-out and the most accurate (lowest RMSE) is kept:
      - forward_angles:  position regressed directly on the angles (lowest error, but shrinks to the centre)
      - inverse_angles:  angles regressed on position, then inverted (unbiased gain / reaches the edges, but
                         amplifies noise -- on noisy data it can be much worse; measured 0.385 vs 0.256)
      - inverse_head:    inverse_angles plus head position -- only considered if the head actually moved
                         during calibration (>= MIN_HEAD_SPREAD in every feature); otherwise its
                         coefficients are fit to noise and would swing the cursor on any later head shift.
    An inverse fit is preferred when within 8% of the best RMSE, since it keeps the edges reachable.

    Returns None (caller keeps the existing mapping) with too few trials, or a fit that is clearly worse than
    the mapping the trials were recorded with.

    Returns {"affine_x", "affine_y", "face_ref" (or None), "method", "uses_head", "loo_rmse", "loo_by_method",
             "raw_rmse", "loo_mu": [(x, y)...]}.
    """
    usable = [s for s in samples if "rel_yaw" in s and "rel_pitch" in s]
    if len(usable) < min_trials:
        return None
    # Prefer ABSOLUTE, 0.5s-averaged angles (no per-session zero point; far less per-frame noise in the inputs).
    absolute = all("abs_yaw" in s and "abs_pitch" in s for s in usable)
    angles = np.array([[s["abs_yaw"], s["abs_pitch"]] for s in usable]) if absolute else np.array([[s["rel_yaw"], s["rel_pitch"]] for s in usable])
    truth = np.array([[s["true_x"], s["true_y"]] for s in usable])
    raw = np.array([[s["mu_x"], s["mu_y"]] for s in usable])
    have_head = all(k in s for s in usable for k in ("face_cx", "face_cy", "face_w"))
    head = np.array([[s["face_cx"], s["face_cy"], s["face_w"]] for s in usable]) if have_head else None
    head_moved = head is not None and bool((head.std(axis=0) >= MIN_HEAD_SPREAD).all())

    def fit(method: str, idx: np.ndarray):
        if method == "forward_angles":
            return _forward_fit(angles[idx], truth[idx], ridge)
        return _inverse_fit(angles[idx], truth[idx], head[idx] if method == "inverse_head" else None, ridge)

    def predict(w: np.ndarray, ref: np.ndarray | None, i: int) -> np.ndarray:
        x = np.r_[1.0, angles[i]]
        if ref is not None:
            x = np.r_[x, head[i] - ref]
        return x @ w

    methods = ["inverse_angles", "forward_angles"] + (["inverse_head"] if head_moved else [])
    loo: dict[str, np.ndarray] = {}
    for method in methods:
        out = np.zeros_like(truth)
        ok = True
        for i in range(len(usable)):
            f = fit(method, np.arange(len(usable)) != i)
            if f is None:
                ok = False
                break
            out[i] = predict(f[0], f[1], i)
        if ok:
            loo[method] = out
    if not loo:
        return None

    def rmse(pred: np.ndarray) -> float:
        return float(np.sqrt(((pred - truth) ** 2).sum(axis=1).mean()))

    scores = {m: rmse(p) for m, p in loo.items()}
    best = min(scores, key=scores.get)
    chosen = next((m for m in ("inverse_angles", "inverse_head") if m in scores and scores[m] <= 1.08 * scores[best]), best)
    loo_rmse, raw_rmse = scores[chosen], float(np.sqrt(((raw - truth) ** 2).sum(axis=1).mean()))
    if loo_rmse > 0.35 or loo_rmse > 1.2 * raw_rmse:
        return None
    f = fit(chosen, np.ones(len(usable), dtype=bool))
    if f is None:
        return None
    w, ref = f
    return {
        "affine_x": [round(float(v), 4) for v in w[:, 0]],
        "affine_y": [round(float(v), 4) for v in w[:, 1]],
        "face_ref": None if ref is None else [round(float(v), 5) for v in ref],
        "method": chosen,
        "absolute": absolute,
        "uses_head": chosen == "inverse_head",
        "loo_rmse": loo_rmse,
        "loo_by_method": {m: round(v, 4) for m, v in scores.items()},
        "raw_rmse": raw_rmse,
        "loo_mu": [(float(a), float(b)) for a, b in loo[chosen]],
    }


FEATURE_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "eye_feature_model.npz"


def fit_feature_mapping(samples: list[dict], min_trials: int = 12) -> dict | None:
    """EyeTrax-style mapping: standardise the head-normalised eye/iris features recorded at each trial and ridge-regress the
    target position on them (vision/feature_model.py). The ridge strength is chosen by leave-one-trial-out RMSE.

    Returns {"model": FeatureModel, "alpha", "loo_rmse", "loo_mu", "alpha_scores"} or None when the trials don't all
    carry features. `loo_mu` lines up with fit_affine's (same `usable` subset and order), so either can be handed to
    with_corrected_mu().
    """
    from vision.eye_features import FEATURE_SUBSETS
    from vision.feature_model import fit_ridge, select_alpha

    usable = [s for s in samples if "rel_yaw" in s and "rel_pitch" in s]
    if len(usable) < min_trials or not all("eye_feat" in s for s in usable):
        return None
    x = np.array([s["eye_feat"] for s in usable], dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        return None
    y = np.array([[s["true_x"], s["true_y"]] for s in usable], dtype=np.float64)
    # Try each feature subset x ridge strength; keep the best by leave-one-trial-out RMSE.
    best = None
    by_subset: dict[str, float] = {}
    for name, cols in FEATURE_SUBSETS.items():
        if max(cols) >= x.shape[1]:
            continue
        alpha, loo, scores = select_alpha(x[:, cols], y)
        rmse = float(np.sqrt(((loo - y) ** 2).sum(axis=1).mean()))
        by_subset[name] = round(rmse, 4)
        if best is None or rmse < best[0]:
            best = (rmse, name, cols, alpha, loo)
    if best is None:
        return None
    rmse, name, cols, alpha, loo = best
    return {
        "model": fit_ridge(x, y, alpha, columns=cols),
        "alpha": alpha,
        "subset": name,
        "loo_rmse": rmse,
        "loo_by_subset": by_subset,
        "loo_mu": [(float(a), float(b)) for a, b in loo],
    }


def write_back_feature_model(path: str | None) -> None:
    """Point config/gaze_mapping.yaml at the saved feature model, or remove the key (CNN mapping in use)."""
    with open(GAZE_MAPPING_PATH, encoding="utf-8") as f:
        data = yaml.load(f)
    if path:
        data["feature_model"] = path
    elif "feature_model" in data:
        del data["feature_model"]
    with open(GAZE_MAPPING_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    print(f"[OK] feature_model -> {path}")


def with_corrected_mu(samples: list[dict], fit: dict) -> list[dict]:
    """Samples with mu_x/mu_y replaced by the leave-one-out affine prediction --
    what the learned mapping would have produced -- so the dashboard and
    sigma_scale describe the mapping that will actually be used."""
    usable = [s for s in samples if "rel_yaw" in s and "rel_pitch" in s]
    return [dict(s, mu_x=x, mu_y=y) for s, (x, y) in zip(usable, fit["loo_mu"])]


def write_back_affine(affine_x: list[float], affine_y: list[float], face_ref: list[float] | None = None, absolute: bool = False) -> None:
    with open(GAZE_MAPPING_PATH, encoding="utf-8") as f:
        data = yaml.load(f)
    data["affine_x"] = affine_x
    data["affine_y"] = affine_y
    data["affine_absolute"] = bool(absolute)
    if face_ref is not None:
        data["face_ref"] = face_ref
    elif "face_ref" in data:
        del data["face_ref"]  # angles-only fit: stale head-position reference must not linger
    with open(GAZE_MAPPING_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    print(f"[OK] wrote affine_x={affine_x}, affine_y={affine_y} to {GAZE_MAPPING_PATH}")


def write_back_canvas(width_px: float, height_px: float) -> None:
    """Record the keyboard canvas size (pixels) the calibration was done at. The mapping is learned against
    that exact on-screen rectangle; the typing UI compares it with the live canvas and warns if they differ."""
    with open(GAZE_MAPPING_PATH, encoding="utf-8") as f:
        data = yaml.load(f)
    data["calibrated_canvas"] = [int(round(width_px)), int(round(height_px))]
    with open(GAZE_MAPPING_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    print(f"[OK] wrote calibrated_canvas={data['calibrated_canvas']} to {GAZE_MAPPING_PATH}")


def write_back(sigma_scale: float) -> None:
    with open(DECODER_PARAMS_PATH, encoding="utf-8") as f:
        data = yaml.load(f)
    data.setdefault("gaze_uncertainty", {})
    data["gaze_uncertainty"]["sigma_scale"] = round(sigma_scale, 4)
    data["gaze_uncertainty"]["calibrated"] = True
    with open(DECODER_PARAMS_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    print(f"[OK] wrote sigma_scale={sigma_scale:.4f} to {DECODER_PARAMS_PATH}")


def write_back_range(half_fov_yaw_rad: float, half_fov_pitch_rad: float) -> None:
    with open(GAZE_MAPPING_PATH, encoding="utf-8") as f:
        data = yaml.load(f)
    data["half_fov_yaw_rad"] = round(half_fov_yaw_rad, 4)
    data["half_fov_pitch_rad"] = round(half_fov_pitch_rad, 4)
    with open(GAZE_MAPPING_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    print(
        f"[OK] wrote half_fov_yaw_rad={half_fov_yaw_rad:.4f}, "
        f"half_fov_pitch_rad={half_fov_pitch_rad:.4f} to {GAZE_MAPPING_PATH}"
    )


def main() -> int:
    layout = KeyboardLayout.from_json()

    total_trials = len(TARGET_KEYS) * REPEATS
    print(f"Calibration: {len(TARGET_KEYS)} keys x {REPEATS} repeats = {total_trials} trials, randomized order.")
    print("Look at each highlighted key as it appears.")
    print("Press 'q' at any point to stop early (uses whatever trials were collected so far).")

    try:
        samples = run_trials(layout)
    except RuntimeError as exc:
        print(f"[FAIL] {exc}")
        return 1

    if len(samples) < 8:
        print(f"[FAIL] only {len(samples)} usable trials -- need at least 8 for a meaningful estimate.")
        return 1

    DATA_DIR.mkdir(exist_ok=True)
    with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"[OK] saved {len(samples)} trials to {SAMPLES_PATH}")

    print_accuracy_dashboard(samples, layout)

    range_result = analyze_range(samples)
    if range_result is not None:
        with open(GAZE_MAPPING_PATH, encoding="utf-8") as f:
            old_mapping = yaml.load(f)
        old_yaw = old_mapping.get("half_fov_yaw_rad")
        old_pitch = old_mapping.get("half_fov_pitch_rad")
        new_yaw, new_pitch = range_result
        print()
        print(f"[analysis] derived half_fov_yaw_rad={new_yaw:.4f} (was {old_yaw}), "
              f"half_fov_pitch_rad={new_pitch:.4f} (was {old_pitch})")
        write_back_range(new_yaw, new_pitch)
    else:
        print("[analysis] not enough off-center trials to derive a better range -- skipping")

    with open(DECODER_PARAMS_PATH, encoding="utf-8") as f:
        decoder_params = yaml.load(f)
    min_sigma = float(decoder_params.get("gaze_uncertainty", {}).get("min_sigma", 0.01))

    sigma_scale = analyze(samples, min_sigma)
    write_back(sigma_scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
