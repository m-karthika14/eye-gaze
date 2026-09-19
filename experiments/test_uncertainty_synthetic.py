"""Synthetic regression check for Phase 3: fixation aggregation, Gaussian
key-probability integration, and the calibration analysis math. No webcam
needed.

Run from the repo root:

    .venv/Scripts/python.exe -m experiments.test_uncertainty_synthetic
"""

from __future__ import annotations

import random
import sys

from decoder.key_probability import key_probabilities
from decoder.nearest_key import KeyboardLayout
from experiments.calibrate_uncertainty import analyze, analyze_range
from vision.fixation import FixationAggregator


def check_fixation_recovers_noise() -> bool:
    random.seed(1)
    agg = FixationAggregator(min_window_ms=200, max_window_ms=800, fixation_radius=0.06)
    true_mu, true_sigma = (0.42, 0.58), 0.015
    t, dt = 0.0, 1 / 30
    event = None
    for _ in range(30):
        x = true_mu[0] + random.gauss(0, true_sigma)
        y = true_mu[1] + random.gauss(0, true_sigma)
        event = agg.update(x, y, t)
        t += dt

    if event is None:
        print("[FAIL] fixation: expected a resolved event after 30 stable samples")
        return False
    if abs(event.mu_x - true_mu[0]) > 0.01 or abs(event.mu_y - true_mu[1]) > 0.01:
        print(f"[FAIL] fixation: mu {event.mu_x:.3f},{event.mu_y:.3f} too far from true {true_mu}")
        return False
    if not (0.008 < event.sigma_x < 0.025) or not (0.008 < event.sigma_y < 0.025):
        print(f"[FAIL] fixation: sigma {event.sigma_x:.4f},{event.sigma_y:.4f} not close to true {true_sigma}")
        return False
    print(f"[OK] fixation recovers injected noise: mu=({event.mu_x:.3f},{event.mu_y:.3f}) "
          f"sigma=({event.sigma_x:.4f},{event.sigma_y:.4f}) vs true mu={true_mu} sigma={true_sigma}")
    return True


def check_fixation_resets_on_saccade_and_loss() -> bool:
    agg = FixationAggregator(min_window_ms=200, max_window_ms=800)
    t, dt = 0.0, 1 / 30
    for _ in range(20):
        agg.update(0.3, 0.3, t)
        t += dt
    ev = agg.update(0.9, 0.9, t)  # saccade to a far point
    if ev is not None:
        print("[FAIL] fixation: saccade should reset the window, not extend it")
        return False

    agg2 = FixationAggregator(min_window_ms=200, max_window_ms=800)
    t2 = 0.0
    for _ in range(20):
        agg2.update(0.5, 0.5, t2)
        t2 += dt
    agg2.update(None, None, t2)  # gaze lost
    ev2 = agg2.update(0.5, 0.5, t2 + dt)
    if ev2 is not None:
        print("[FAIL] fixation: gaze loss should clear the buffer, not preserve it")
        return False

    print("[OK] fixation correctly resets on saccade and on gaze loss")
    return True


def check_key_probabilities_shape() -> bool:
    layout = KeyboardLayout.from_json()
    d_key = next(k for k in layout.keys if k.id == "D")
    cx, cy = d_key.center()
    probs = key_probabilities(cx, cy, sigma_x=0.08, sigma_y=0.03, layout=layout)

    total = sum(probs.values())
    if abs(total - 1.0) > 1e-9:
        print(f"[FAIL] key_probabilities: does not sum to 1.0 (got {total})")
        return False
    top = max(probs, key=probs.get)
    if top != "D":
        print(f"[FAIL] key_probabilities: expected top key D, got {top}")
        return False
    if not (probs["S"] == probs["F"] and probs["A"] == probs["G"]):
        print("[FAIL] key_probabilities: expected left/right symmetry around D's center")
        return False
    print(f"[OK] key_probabilities: sums to 1.0, peaked at D ({probs['D']:.3f}), symmetric neighbors")
    return True


def check_calibration_analysis_recovers_scale() -> bool:
    """If the raw sigma systematically underestimates real error by a known
    factor, analyze() should recover ~that factor as sigma_scale."""
    random.seed(7)
    true_extra_scale = 1.8  # raw sigma is 1.8x too small vs actual error
    samples = []
    for _ in range(200):
        true_x, true_y = random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)
        raw_sigma = 0.03
        actual_error_sigma = raw_sigma * true_extra_scale
        mu_x = true_x + random.gauss(0, actual_error_sigma)
        mu_y = true_y + random.gauss(0, actual_error_sigma)
        samples.append(
            {"true_x": true_x, "true_y": true_y, "mu_x": mu_x, "mu_y": mu_y, "sigma_x": raw_sigma, "sigma_y": raw_sigma}
        )

    scale = analyze(samples, min_sigma=0.01)
    if abs(scale - true_extra_scale) > 0.3:
        print(f"[FAIL] calibration: recovered scale {scale:.3f} too far from true {true_extra_scale}")
        return False
    print(f"[OK] calibration analysis recovers injected scale error: {scale:.3f} vs true {true_extra_scale}")
    return True


def check_calibration_robust_to_outliers() -> bool:
    """A couple of wild outlier trials (blink, wrong key looked at) should
    barely move the robust (median-based) sigma_scale, unlike a naive
    mean(z^2) statistic which would be dominated by them."""
    random.seed(3)
    true_extra_scale = 1.5
    samples = []
    for _ in range(30):
        true_x, true_y = random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)
        raw_sigma = 0.02
        actual_error_sigma = raw_sigma * true_extra_scale
        mu_x = true_x + random.gauss(0, actual_error_sigma)
        mu_y = true_y + random.gauss(0, actual_error_sigma)
        samples.append(
            {"true_x": true_x, "true_y": true_y, "mu_x": mu_x, "mu_y": mu_y, "sigma_x": raw_sigma, "sigma_y": raw_sigma}
        )

    scale_clean = analyze(samples, min_sigma=0.01)

    # Inject 2 wild outliers among the 30 clean trials (~7%) -- e.g. the
    # tracker briefly locked onto a blink, or the wrong key was looked at.
    outliers = [
        {"true_x": 0.1, "true_y": 0.1, "mu_x": 0.9, "mu_y": 0.9, "sigma_x": 0.02, "sigma_y": 0.02},
        {"true_x": 0.9, "true_y": 0.1, "mu_x": 0.1, "mu_y": 0.9, "sigma_x": 0.02, "sigma_y": 0.02},
    ]
    scale_with_outliers = analyze(samples + outliers, min_sigma=0.01)

    drift = abs(scale_with_outliers - scale_clean)
    if drift > 0.5:
        print(f"[FAIL] robust scale moved too much from outliers: {scale_clean:.3f} -> {scale_with_outliers:.3f}")
        return False
    print(f"[OK] robust sigma_scale barely moves with 2 wild outliers among 30 trials: "
          f"{scale_clean:.3f} -> {scale_with_outliers:.3f}")
    return True


def check_analyze_range_recovers_true_range() -> bool:
    """If a person's real yaw/pitch range differs from whatever the mapper's
    current guess happens to be, analyze_range() should recover the REAL
    range from (rel_yaw/rel_pitch, true_x/true_y) pairs, independent of that
    guess -- this is what lets calibration fix a stuck-too-wide initial range
    instead of only ever being able to widen it further."""
    true_range_yaw, true_range_pitch = 0.15, 0.09
    targets = [(0.05, 0.125), (0.95, 0.125), (0.05, 0.875), (0.95, 0.875), (0.5, 0.875), (0.05, 0.5)]
    samples = []
    for true_x, true_y in targets:
        rel_yaw = (true_x - 0.5) * 2 * true_range_yaw
        rel_pitch = -(true_y - 0.5) * 2 * true_range_pitch
        samples.append({"true_x": true_x, "true_y": true_y, "rel_yaw": rel_yaw, "rel_pitch": rel_pitch})

    result = analyze_range(samples)
    if result is None:
        print("[FAIL] analyze_range: expected a result, got None")
        return False
    recovered_yaw, recovered_pitch = result
    if abs(recovered_yaw - true_range_yaw) > 1e-6 or abs(recovered_pitch - true_range_pitch) > 1e-6:
        print(f"[FAIL] analyze_range: recovered {result} != true ({true_range_yaw}, {true_range_pitch})")
        return False
    print(f"[OK] analyze_range recovers the true range exactly: {result}")
    return True


def main() -> int:
    checks = [
        check_fixation_recovers_noise,
        check_fixation_resets_on_saccade_and_loss,
        check_key_probabilities_shape,
        check_calibration_analysis_recovers_scale,
        check_calibration_robust_to_outliers,
        check_analyze_range_recovers_true_range,
    ]
    results = [check() for check in checks]
    if all(results):
        print(f"\n[OK] all {len(results)} Phase 3 checks passed")
        return 0
    print(f"\n[FAIL] {results.count(False)}/{len(results)} checks failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
