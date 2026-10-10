"""Ridge regression from head-normalised eye features (vision/eye_features.py) to screen position.

The EyeTrax recipe (https://github.com/ck-zhang/EyeTrax, MIT): standardise the feature vector, fit a ridge regression to the
calibration targets, predict the normalised canvas position. Implemented in plain numpy (dual form, so it stays fast
with ~500 features and a few dozen calibration trials) -- no scikit-learn needed at runtime.

Stored as an .npz next to the other models and referenced from config/gaze_mapping.yaml (`feature_model`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MIN_SCALE = 5e-4  # floor on a feature's std: near-constant features must not be blown up by standardisation
DEFAULT_ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)


class FeatureModel:
    def __init__(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
        coef: np.ndarray,
        intercept: np.ndarray,
        alpha: float,
        columns: np.ndarray | None = None,
        n_input: int | None = None,
    ) -> None:
        """`columns`: which entries of the full feature vector the model uses (None = all). `mean`/`scale`/`coef` are over
        just those columns; `n_input` is the length of the full vector it expects."""
        self.mean, self.scale, self.coef, self.intercept, self.alpha = mean, scale, coef, intercept, float(alpha)
        self.columns = None if columns is None else np.asarray(columns, dtype=np.int64)
        self.n_input = int(n_input) if n_input is not None else int(mean.shape[0])

    @property
    def n_features(self) -> int:
        """Length of the FULL feature vector this model takes as input."""
        return self.n_input

    def predict(self, features: np.ndarray) -> np.ndarray:
        """features: (d,) or (n, d). Returns (2,) or (n, 2): the normalised canvas (x, y)."""
        x = np.asarray(features, dtype=np.float64)
        if self.columns is not None:
            x = x[..., self.columns]
        z = (x - self.mean) / self.scale
        return z @ self.coef + self.intercept

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, mean=self.mean, scale=self.scale, coef=self.coef, intercept=self.intercept, alpha=self.alpha,
            columns=np.array([-1]) if self.columns is None else self.columns, n_input=self.n_input,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureModel":
        d = np.load(path)
        cols = d["columns"] if "columns" in d else None
        if cols is not None and cols.shape == (1,) and cols[0] == -1:
            cols = None
        n_input = int(d["n_input"]) if "n_input" in d else None
        return cls(d["mean"], d["scale"], d["coef"], d["intercept"], float(d["alpha"]), cols, n_input)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float, columns=None) -> FeatureModel:
    """Standardise x (optionally just `columns` of it), ridge-regress y (n x 2) on it. Dual form: solves an n x n system,
    not d x d."""
    x = np.asarray(x, dtype=np.float64)
    n_input = x.shape[1]
    if columns is not None:
        x = x[:, columns]
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), MIN_SCALE)
    z = (x - mean) / scale
    y_mean = y.mean(axis=0)
    yc = y - y_mean
    gram = z @ z.T
    dual = np.linalg.solve(gram + alpha * np.eye(len(z)), yc)
    coef = z.T @ dual  # (d x 2)
    return FeatureModel(mean, scale, coef, y_mean, alpha, columns, n_input)


def loo_predictions(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Leave-one-trial-out predictions (re-standardised inside every fold, so nothing leaks)."""
    out = np.zeros_like(y, dtype=np.float64)
    idx = np.arange(len(x))
    for i in range(len(x)):
        keep = idx != i
        out[i] = fit_ridge(x[keep], y[keep], alpha).predict(x[i])
    return out


def select_alpha(x: np.ndarray, y: np.ndarray, alphas=DEFAULT_ALPHAS) -> tuple[float, np.ndarray, dict[float, float]]:
    """Pick the ridge strength by leave-one-out RMSE. Returns (alpha, its LOO predictions, {alpha: rmse})."""
    scores: dict[float, float] = {}
    best: tuple[float, np.ndarray] | None = None
    for a in alphas:
        pred = loo_predictions(x, y, a)
        rmse = float(np.sqrt(((pred - y) ** 2).sum(axis=1).mean()))
        scores[a] = rmse
        if best is None or rmse < scores[best[0]]:
            best = (a, pred)
    assert best is not None
    return best[0], best[1], scores
