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
import yaml

from decoder.nearest_key import KeyboardLayout
from vision.pipeline import GazePipeline

DECODER_PARAMS_PATH = Path(__file__).resolve().parent.parent / "config" / "decoder_params.yaml"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLES_PATH = DATA_DIR / "uncertainty_calibration_samples.jsonl"

CANVAS_SIZE = (720, 1280)  # h, w -- stand-in for "the screen", matches vision/run_demo.py

# A spread across the keyboard: corners, edges, center -- not all 28 keys,
# just enough variety to characterize the error across the whole surface.
TARGET_KEYS = ["Q", "P", "Z", "M", "D", "H", "A", "L", "T", "B", "SPACE", "BACKSPACE"]

# Each target is visited this many times (in randomized order, not back to
# back) so the estimate isn't skewed by one or two rough trials -- a handful
# of trials per key is not enough to trust a single global scale factor from.
REPEATS = 3

TARGET_FIXATION_MS = 600.0  # how long we wait for a fixation to accumulate before recording it
SETTLE_TIMEOUT_S = 8.0  # give up on a trial if no stable fixation forms in this long


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
    pipeline = GazePipeline(camera_index=0, target_fps=30)
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

                if fixation is not None and fixation.duration_ms >= TARGET_FIXATION_MS:
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


# For Z ~ N(0, 1), median(|Z|) = 0.6745 (the 75th-percentile point of the
# standard normal). This is the standard median-absolute-deviation robust
# scale estimator: unlike mean(z^2), which squares every error and lets one
# or two bad trials (a blink, a momentary tracking glitch, a wrong key
# looked at) dominate the whole result, the median only cares about the
# typical trial and barely moves if a handful are way off.
MEDIAN_ABS_Z_STANDARD_NORMAL = 0.6745


def analyze(samples: list[dict], min_sigma: float) -> float:
    """Returns sigma_scale such that (mu - true) / (sigma * scale) behaves
    like a standard normal across trials, estimated robustly via the median
    of |z| rather than the mean of z^2."""
    z_abs: list[float] = []
    for s in samples:
        sigma_x = max(s["sigma_x"], min_sigma)
        sigma_y = max(s["sigma_y"], min_sigma)
        z_abs.append(abs((s["mu_x"] - s["true_x"]) / sigma_x))
        z_abs.append(abs((s["mu_y"] - s["true_y"]) / sigma_y))

    median_abs_z = statistics.median(z_abs)
    sigma_scale = median_abs_z / MEDIAN_ABS_Z_STANDARD_NORMAL

    # Still surface the mean(z^2) view for comparison -- a big gap between
    # the two numbers itself tells you outliers are driving the naive stat.
    mean_z_sq = sum(z * z for z in z_abs) / len(z_abs)
    naive_scale = mean_z_sq**0.5

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


def write_back(sigma_scale: float) -> None:
    data = yaml.safe_load(DECODER_PARAMS_PATH.read_text(encoding="utf-8")) or {}
    data.setdefault("gaze_uncertainty", {})
    data["gaze_uncertainty"]["sigma_scale"] = round(sigma_scale, 4)
    data["gaze_uncertainty"]["calibrated"] = True
    DECODER_PARAMS_PATH.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"[OK] wrote sigma_scale={sigma_scale:.4f} to {DECODER_PARAMS_PATH}")


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

    decoder_params = yaml.safe_load(DECODER_PARAMS_PATH.read_text(encoding="utf-8")) or {}
    min_sigma = float(decoder_params.get("gaze_uncertainty", {}).get("min_sigma", 0.01))

    sigma_scale = analyze(samples, min_sigma)
    write_back(sigma_scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
