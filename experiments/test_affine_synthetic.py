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

    # --- cursor stability: staring at one spot with ~0.05 per-frame scatter (as measured on real data)
    def stare(min_cutoff: float, beta: float, leash: float) -> float:
        c = GazeMappingConfig(startup_bias_frames=1, affine_x=[0.5, 2.5, 0.0], affine_y=[0.35, 0.0, -1.6], drift_correction_enabled=False,
                              filter_min_cutoff=min_cutoff, filter_beta=beta, stability_radius=leash)
        m = GazeToScreenMapper(c)
        m.map(0.0, 0.0, 0.0)
        jr, pts, t = random.Random(4), [], 0.0
        for _ in range(200):  # 20 s at 10 fps
            t += 0.1
            o = m.map(jr.gauss(0, 0.05 / 1.6), jr.gauss(0, 0.05 / 2.5), t)
            pts.append((o.smoothed_x, o.smoothed_y))
        pts = pts[30:]
        return sum(((pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2) ** 0.5 for i in range(1, len(pts))) / (len(pts) - 1)

    old_step, new_step = stare(1.0, 3.0, 0.0), stare(0.5, 1.5, 0.05)
    assert new_step < 0.2 * old_step, (old_step, new_step)
    print(f"[OK] staring at one spot: mean cursor movement per frame {new_step:.4f} vs {old_step:.4f} before ({old_step / new_step:.0f}x steadier)")

    mapper = GazeToScreenMapper(GazeMappingConfig(startup_bias_frames=1, affine_x=[0.5, 2.5, 0.0], affine_y=[0.35, 0.0, -1.6], drift_correction_enabled=False,
                                                  filter_min_cutoff=0.5, filter_beta=1.5, stability_radius=0.05))
    mapper.map(0.0, 0.0, 0.0)
    t = 0.0
    for _ in range(30):
        t += 0.1
        mapper.map(0.0, 0.0, t)
    # ...but it still follows a genuine move to another key promptly.
    for _ in range(6):  # ~0.6 s at 10 fps
        t += 0.1
        out = mapper.map(0.0, -0.12, t)  # yaw -0.12 -> x = 0.5 - 0.30 = 0.20
    assert abs(out.smoothed_x - 0.20) < 0.08, out
    print("[OK] cursor still follows a real move to another key")

    # --- top row: a single noisy frame past the top edge must not drop the highlight; sustained gaze above must.
    m2 = GazeToScreenMapper(GazeMappingConfig(startup_bias_frames=1, affine_x=[0.5, 2.5, 0.0], affine_y=[0.125, 0.0, -1.6], drift_correction_enabled=False))
    m2.map(0.0, 0.0, 0.0)
    t, flags = 0.0, []
    for i in range(30):
        t += 0.1
        pitch = 0.0 if i % 6 else 0.16  # looking at the top-row centre (y=0.125); every 6th frame jumps 0.26 above it
        flags.append(m2.map(pitch, 0.0, t).on_screen)
    assert all(flags), flags
    for _ in range(15):
        t += 0.1
        sustained = m2.map(0.35, 0.0, t)  # y = 0.125 - 0.56 = -0.43: reading the text box above
    assert not sustained.on_screen
    print("[OK] top row: isolated noisy frames stay on the keyboard; sustained gaze at the text above does not")

    # --- edge reach: with noisy angles, forward regression (position ~ angle) shrinks towards the
    # centre and stops short at the screen edges; the inverse fit used by fit_affine does not.
    import numpy as np

    big = make_samples(random.Random(9), 600, 1.0, 0.04)  # 0.04 rad noise ~ the measured per-frame scatter
    fit = fit_affine(big)
    assert fit is not None
    ang = np.array([[1.0, s["rel_yaw"], s["rel_pitch"]] for s in big])
    tru = np.array([[s["true_x"], s["true_y"]] for s in big])
    fwd = np.linalg.solve(ang.T @ ang, ang.T @ tru)
    inv_pred = ang @ np.array([fit["affine_x"], fit["affine_y"]]).T
    fwd_pred = ang @ fwd
    far = (tru[:, 0] > 0.85) | (tru[:, 0] < 0.15)
    reach_fwd = np.abs((fwd_pred[far, 0] - 0.5) * np.sign(tru[far, 0] - 0.5)).mean()  # how far from centre it lands
    reach_inv = np.abs((inv_pred[far, 0] - 0.5) * np.sign(tru[far, 0] - 0.5)).mean()
    reach_true = np.abs(tru[far, 0] - 0.5).mean()
    assert abs(reach_inv - reach_true) < 0.5 * abs(reach_fwd - reach_true), (reach_fwd, reach_inv, reach_true)
    print(f"[OK] edge keys: true distance from centre {reach_true:.2f}; forward fit reaches {reach_fwd:.2f}, inverse fit reaches {reach_inv:.2f}")

    # --- head position: the gaze CNN's angle depends on where the head is. When the head wanders during
    # calibration, modelling it must clearly cut the error; when it doesn't matter, it must NOT be used.
    def head_samples(seed: int, head_gain: float, n: int = 36) -> list[dict]:
        r = random.Random(seed)
        rows, hx, hy, hw = [], 0.5, 0.45, 0.30
        for _ in range(n):
            hx += r.gauss(0, 0.025); hy += r.gauss(0, 0.025); hw += r.gauss(0, 0.012)  # posture random walk
            tx, ty = r.uniform(0.05, 0.95), r.uniform(0.1, 0.9)
            yaw = (tx - 0.5) / 2.5 + head_gain * (hx - 0.5) * 0.9 + r.gauss(0, 0.008)
            pitch = -(ty - 0.35) / 1.6 + head_gain * ((hy - 0.45) * 0.8 - (hw - 0.30) * 0.6) + r.gauss(0, 0.008)
            rows.append({"true_x": tx, "true_y": ty, "rel_yaw": yaw, "rel_pitch": pitch, "mu_x": 0.5, "mu_y": 0.5,
                         "face_cx": hx, "face_cy": hy, "face_w": hw})
        return rows

    moving = fit_affine(head_samples(1, 1.0))
    assert moving is not None and moving["uses_head"] and moving["face_ref"] is not None and len(moving["affine_x"]) == 6
    assert moving["loo_rmse"] < 0.7 * moving["loo_by_method"]["inverse_angles"], moving
    print(f"[OK] head moving during calibration: head-aware fit loo_rmse {moving['loo_rmse']:.3f} vs angles-only {moving['loo_by_method']['inverse_angles']:.3f}")
    still = fit_affine(head_samples(2, 0.0))
    assert still is not None and not still["uses_head"] and len(still["affine_x"]) == 3 and still["face_ref"] is None, still
    print("[OK] head irrelevant -> angles-only fit kept (no overfitting to head position)")

    m6 = GazeToScreenMapper(GazeMappingConfig(startup_bias_frames=1, affine_x=[0.5, 2.5, 0.0, 1.0, 0.0, 0.0], affine_y=[0.35, 0.0, -1.6, 0.0, 2.0, 0.0],
                                              face_ref=[0.5, 0.45, 0.3], drift_correction_enabled=False, filter_min_cutoff=1000.0, filter_beta=0.0, stability_radius=0.0))
    m6.map(0.0, 0.0, 0.0)
    o6 = m6.map(0.0, 0.0, 0.1, face=(0.56, 0.5, 0.3))  # head moved right 0.06 and down 0.05
    assert abs(o6.raw_x - 0.56) < 1e-9 and abs(o6.raw_y - 0.45) < 1e-9, o6
    far = m6.map(0.0, 0.0, 0.2, face=(0.9, 0.45, 0.3))  # absurd head offset 0.4: clamped to 0.08, never extrapolated
    assert abs(far.raw_x - 0.58) < 1e-9, far
    print("[OK] mapper applies the head-position terms (and still works when no face features are given)")
    assert 0.0 <= m6.map(0.0, 0.0, 0.2).raw_x <= 1.0

    # --- absolute mode: no per-session zero point. The same raw angle must map to the same screen position
    # whatever the user happened to be looking at when the session started.
    abs_cfg = dict(affine_x=[0.5, 2.5, 0.0], affine_y=[0.35, 0.0, -1.6], affine_absolute=True, filter_min_cutoff=1000.0,
                   filter_beta=0.0, stability_radius=0.0)
    first_look_centre = GazeToScreenMapper(GazeMappingConfig(**abs_cfg))
    first_look_corner = GazeToScreenMapper(GazeMappingConfig(**abs_cfg))
    for i in range(30):  # 30 frames while looking at different things (e.g. the calibration "Done" button)
        first_look_centre.map(0.0, 0.0, i * 0.1)
        first_look_corner.map(-0.2, 0.2, i * 0.1)
    a1 = first_look_centre.map(0.05, -0.1, 3.1)
    a2 = first_look_corner.map(0.05, -0.1, 3.1)
    assert abs(a1.raw_x - a2.raw_x) < 1e-9 and abs(a1.raw_y - a2.raw_y) < 1e-9 and abs(a1.raw_x - 0.25) < 1e-9, (a1, a2)
    assert first_look_centre.bias_locked  # callers waiting for the old startup auto-centre must not hang
    rel_cfg = {**abs_cfg, "affine_absolute": False, "startup_bias_frames": 5, "drift_correction_enabled": False}
    r1, r2 = GazeToScreenMapper(GazeMappingConfig(**rel_cfg)), GazeToScreenMapper(GazeMappingConfig(**rel_cfg))
    for i in range(30):
        r1.map(0.0, 0.0, i * 0.1)
        r2.map(-0.2, 0.2, i * 0.1)
    assert abs(r1.map(0.05, -0.1, 3.1).raw_x - r2.map(0.05, -0.1, 3.1).raw_x) > 0.2  # the old mode really did depend on it
    print("[OK] absolute mode: same angle -> same position regardless of what you looked at at start-up (relative mode: offset > 0.2)")

    first_look_centre.recenter()  # user presses C while looking at the centre of the keyboard
    assert abs(first_look_centre.map(0.05, -0.1, 3.2).raw_x - 0.5) < 1e-9
    print("[OK] recenter (C) shifts the absolute mapping so the current gaze maps to the centre")
    assert first_look_centre.shift != (0.0, 0.0)
    first_look_centre.reset_shift()  # the footer's "shift ... reset" button: back to exactly what calibration learned
    assert first_look_centre.shift == (0.0, 0.0) and abs(first_look_centre.map(0.05, -0.1, 3.3).raw_x - 0.25) < 1e-9
    print("[OK] reset_shift removes the recenter offset again")

    # --- fit on absolute, 0.5s-averaged angles when the trials carry them
    def abs_samples(seed: int, n: int = 36) -> list[dict]:
        r = random.Random(seed)
        rows = []
        for _ in range(n):
            tx, ty = r.uniform(0.05, 0.95), r.uniform(0.1, 0.9)
            yaw, pitch = (tx - 0.5) / 2.5 + 0.03, -(ty - 0.35) / 1.6 - 0.12  # absolute angles include the model's own bias
            rows.append({"true_x": tx, "true_y": ty, "mu_x": 0.5, "mu_y": 0.5,
                         "rel_yaw": yaw + r.gauss(0, 0.05), "rel_pitch": pitch + r.gauss(0, 0.05),   # single noisy frame
                         "abs_yaw": yaw + r.gauss(0, 0.012), "abs_pitch": pitch + r.gauss(0, 0.012)})  # 0.5s average
        return rows

    samples = abs_samples(3)
    fit_abs = fit_affine(samples)
    fit_rel = fit_affine([{k: v for k, v in s.items() if not k.startswith("abs_")} for s in samples])
    assert fit_abs is not None and fit_abs["absolute"] and fit_rel is not None and not fit_rel["absolute"]
    assert fit_abs["loo_rmse"] < 0.6 * fit_rel["loo_rmse"], (fit_abs["loo_rmse"], fit_rel["loo_rmse"])
    print(f"[OK] fit uses absolute averaged angles when present: loo_rmse {fit_abs['loo_rmse']:.3f} vs {fit_rel['loo_rmse']:.3f} from single-frame relative angles")

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
