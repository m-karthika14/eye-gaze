"""Synthetic checks for the EyeTrax-style eye-landmark mapping (vision/eye_features.py, vision/feature_model.py,
calibrate_uncertainty.fit_feature_mapping, GazeToScreenMapper feature mode). No camera.

    .venv/Scripts/python.exe -m experiments.test_feature_mapping_synthetic
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import numpy as np

from experiments.calibrate_uncertainty import fit_feature_mapping, with_corrected_mu
from vision.eye_features import FEATURE_SUBSETS, N_FEATURES, head_aligned_features
from vision.feature_model import FeatureModel, fit_ridge, select_alpha
from vision.gaze_to_screen import GazeMappingConfig, GazeToScreenMapper


def fake_face(rng: np.random.Generator) -> np.ndarray:
    """A (478, 3) 'face' in MediaPipe-style normalised coordinates with the landmarks the features need in plausible places."""
    pts = rng.normal(0, 0.01, (478, 3)) + np.array([0.5, 0.5, 0.0])
    pts[33] = (0.44, 0.45, 0); pts[263] = (0.56, 0.45, 0); pts[10] = (0.50, 0.20, 0)  # eye corners, top of head
    pts[152] = (0.50, 0.75, 0.0)
    return pts


def main() -> None:
    rng = np.random.default_rng(0)
    face = fake_face(rng)
    f0 = head_aligned_features(face)
    assert f0 is not None and f0.shape == (N_FEATURES,), (None if f0 is None else f0.shape)
    print(f"[OK] feature vector has {N_FEATURES} dimensions (163 landmarks x 3 + yaw/pitch/roll), as in EyeTrax")

    # Head translation and distance (scale) must not change the eye geometry part of the features.
    moved = (face - np.array([0.5, 0.5, 0.0])) * 1.3 + np.array([0.42, 0.55, 0.0])
    f1 = head_aligned_features(moved)
    assert np.abs(f0[:-3] - f1[:-3]).max() < 1e-6, np.abs(f0[:-3] - f1[:-3]).max()
    print("[OK] head translation + scale leave the eye features unchanged (head-normalised)")

    # The iris moving inside the eye socket DOES change them (that's the signal).
    gazing = face.copy()
    gazing[468:473, 0] += 0.004  # left iris shifted right inside the socket
    assert np.abs(head_aligned_features(gazing)[:-3] - f0[:-3]).max() > 0.01
    print("[OK] iris movement inside the eye changes the features")

    # Mapping: features that depend linearly on screen position (a few informative dims among ~500) + noise.
    r = np.random.default_rng(1)
    mix = np.zeros((2, N_FEATURES))  # position -> features: only 12 of the ~500 dimensions carry the iris signal
    informative = r.choice(FEATURE_SUBSETS['iris_core'][:-3], 12, replace=False)  # the iris moving inside the eye
    mix[:, informative] = r.normal(0, 1, (2, 12))

    def trials(n: int, noise: float):
        pos = np.c_[r.uniform(0.05, 0.95, n), r.uniform(0.1, 0.9, n)]
        feats = (pos - 0.5) @ mix * 0.05 + r.normal(0, noise, (n, N_FEATURES)) * 0.01
        return pos, feats

    pos, feats = trials(32, 0.3)  # ~ the number of calibration trials we actually get
    baseline = float(np.sqrt(((pos - pos.mean(0)) ** 2).sum(axis=1).mean()))
    rows = [{"true_x": float(p_[0]), "true_y": float(p_[1]), "mu_x": 0.5, "mu_y": 0.5, "rel_yaw": 0.0, "rel_pitch": 0.0,
             "eye_feat": [float(v) for v in f_]} for p_, f_ in zip(pos, feats)]
    probe = fit_feature_mapping(rows)
    assert probe is not None and probe["subset"] == "iris_core", probe["loo_by_subset"]
    assert probe["loo_by_subset"]["iris_core"] < 0.5 * baseline
    # ...and the all-features ridge (EyeTrax's setting) is much worse here -- the reason subsets are tried at all
    assert probe["loo_by_subset"]["full"] > 1.5 * probe["loo_by_subset"]["iris_core"], probe["loo_by_subset"]
    alpha = probe["alpha"]
    print(f"[OK] with only 32 trials, the narrow iris subset wins: held-out RMSE {probe['loo_by_subset']['iris_core']:.3f} "
          f"vs {probe['loo_by_subset']['full']:.3f} for all 486 features (guessing the mean: {baseline:.3f})")

    model = probe["model"]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.npz"
        model.save(p)
        again = FeatureModel.load(p)
    assert np.allclose(model.predict(feats), again.predict(feats)) and again.n_features == N_FEATURES
    print("[OK] model save/load round-trips")

    # fit_feature_mapping on trial dicts, handing its LOO predictions to with_corrected_mu like the CNN fit
    samples = [{"true_x": float(p_[0]), "true_y": float(p_[1]), "mu_x": 0.5, "mu_y": 0.5, "rel_yaw": 0.0, "rel_pitch": 0.0,
                "eye_feat": [float(v) for v in f_]} for p_, f_ in zip(pos, feats)]
    fit = probe
    corrected = with_corrected_mu(samples, fit)
    assert len(corrected) == len(samples) and abs(corrected[0]["mu_x"] - fit["loo_mu"][0][0]) < 1e-12
    assert fit_feature_mapping(samples[:5]) is None and fit_feature_mapping([{k: v for k, v in s.items() if k != "eye_feat"} for s in samples]) is None
    print("[OK] fit_feature_mapping: usable on real-shaped trial dicts; refuses too few trials or missing features")

    # Mapper in feature mode: uses the model, ignores CNN angles, honours recenter
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.npz"
        model.save(p)
        cfg = GazeMappingConfig(feature_model=str(p), filter_min_cutoff=1000.0, filter_beta=0.0, stability_radius=0.0)
        mapper = GazeToScreenMapper(cfg)
        a = mapper.map(0.3, -0.4, 0.0, features=feats[3])   # wild angles on purpose: they must not matter
        b = mapper.map(-0.3, 0.4, 0.1, features=feats[3])
        expected = np.clip(model.predict(feats[3]), 0, 1)
        assert abs(a.raw_x - expected[0]) < 1e-9 and abs(a.raw_y - expected[1]) < 1e-9 and abs(a.raw_x - b.raw_x) < 1e-12
        mapper.recenter()
        c = mapper.map(0.0, 0.0, 0.2, features=feats[3])
        assert abs(c.raw_x - 0.5) < 1e-9 and abs(c.raw_y - 0.5) < 1e-9
        no_feat = mapper.map(0.0, 0.0, 0.3)  # feature missing for one frame -> falls back, never crashes
        assert 0.0 <= no_feat.raw_x <= 1.0
        assert GazeToScreenMapper(GazeMappingConfig(feature_model="models/does_not_exist.npz")).map(0, 0, 0).raw_x == 0.5  # missing file -> CNN path
    print("[OK] mapper feature mode: predicts from features, ignores angles, recenter works, missing file/feature falls back")

    print("\n[OK] all feature-mapping checks passed")


if __name__ == "__main__":
    main()
