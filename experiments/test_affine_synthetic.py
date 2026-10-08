"""Synthetic checks for the learned affine gaze mapping (experiments/calibrate_uncertainty.fit_affine
+ vision/gaze_to_screen.py). No camera.

    .venv/Scripts/python.exe -m experiments.test_affine_synthetic
"""

from __future__ import annotations

import random

from experiments.calibrate_uncertainty import fit_affine, with_corrected_mu
from vision.gaze_to_screen import GazeMappingConfig, GazeToScreenMapper


def make_samples(rng: random.Random, n: int, sign: float, noise: float) -> list[dict]:
    rows = []
    for _ in range(n):
        tx, ty = rng.uniform(0.05, 0.95), rng.uniform(0.1, 0.9)
        yaw = sign * (tx - 0.52) / 2.5 + rng.gauss(0, noise)
        pitch = -(ty - 0.35) / 1.6 + rng.gauss(0, noise)
        # mu_* as the (wrongly-signed, biased) heuristic mapping would have produced
        rows.append({"true_x": tx, "true_y": ty, "rel_yaw": yaw, "rel_pitch": pitch, "mu_x": 1 - tx, "mu_y": ty + 0.17})
    return rows


def main() -> None:
    rng = random.Random(0)
    for sign in (+1.0, -1.0):
        fit = fit_affine(make_samples(rng, 36, sign, 0.01))
        assert fit is not None and fit["loo_rmse"] < 0.1 and fit["loo_rmse"] < 0.5 * fit["raw_rmse"], fit
        print(f"[OK] fit recovers mapping for yaw sign {sign:+.0f}: loo_rmse={fit['loo_rmse']:.3f} vs heuristic {fit['raw_rmse']:.3f}")

    mapper = GazeToScreenMapper(GazeMappingConfig(startup_bias_frames=1, affine_x=[0.5, 2.5, 0.0], affine_y=[0.35, 0.0, -1.6],
                                                  drift_correction_enabled=False, filter_min_cutoff=1000.0, filter_beta=0.0))
    mapper.map(0.0, 0.0, 0.0)  # lock bias at zero
    out = mapper.map(0.1, -0.1, 0.1)  # (pitch, yaw)
    assert abs(out.raw_x - 0.25) < 1e-9 and abs(out.raw_y - 0.19) < 1e-9, out
    print("[OK] mapper applies affine coefficients (and ignores invert flags)")

    above = mapper.map(0.35, 0.0, 0.2)  # (pitch, yaw): looking well above the keyboard -> raw_y = 0.35 - 1.6*0.35 < -0.06
    assert not above.on_screen and above.raw_y == 0.0, above
    assert mapper.map(0.0, 0.0, 0.3).on_screen
    print("[OK] gaze past the keyboard edge is flagged off-screen (cursor still clamps for display)")

    assert fit_affine(make_samples(rng, 5, 1.0, 0.01)) is None
    pure_noise = [dict(s, rel_yaw=rng.gauss(0, 0.1), rel_pitch=rng.gauss(0, 0.1), mu_x=s["true_x"], mu_y=s["true_y"]) for s in make_samples(rng, 36, 1.0, 0)]
    assert fit_affine(pure_noise) is None
    print("[OK] too few trials, or no better than the existing mapping -> returns None (heuristic kept)")

    good = make_samples(rng, 20, 1.0, 0.01)
    corrected = with_corrected_mu(good, fit_affine(good))
    assert len(corrected) == len(good) and all("mu_x" in c for c in corrected)
    print("\n[OK] all affine checks passed")


if __name__ == "__main__":
    main()
