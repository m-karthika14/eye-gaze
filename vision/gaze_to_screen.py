"""Heuristic, calibration-free mapping from raw gaze angle to a normalized
on-screen point. See config/gaze_mapping.yaml for the (placeholder) constants
and MASTER_PROJECT_DOCUMENT.md Section 8 (Stages 1-2) for where this sits in
the pipeline. Phase 3 replaces this with a calibrated probabilistic model.

Every appearance-based gaze model has a per-person, per-camera-setup bias --
its raw pitch/yaw "zero" rarely lines up with this particular webcam's screen
center (this is the same subject-dependent bias Chen & Shi 2020 and EyeO
model explicitly; see MASTER_PROJECT_DOCUMENT.md Section 6, C3). Without
correcting for it, the mapped point can sit permanently clamped at one edge
regardless of real eye movement. Phase 2 corrects this with a one-time
startup auto-center: the first `startup_bias_frames` samples are averaged
into a bias estimate (assuming the user looks roughly at the screen while the
app starts up), then locked for the rest of the session -- no explicit
calibration screen, just an implicit zero-point.

That one-time lock has a real failure mode on longer sessions: head/posture
position drifts gradually over minutes (confirmed on real calibration data --
per-key raw-angle readings crept monotonically across repeats within a single
multi-minute session), silently invalidating the locked bias partway through.
A scoped slice of Phase 7 is pulled forward here to address it: after the
initial lock, the bias continues to slowly chase a long-time-constant (tens
of seconds) exponential moving average of the raw signal -- the same "gaze
averages out near center over time" assumption the startup lock uses, just
renewed continuously instead of frozen after the first second. The time
constant is long enough that no single key's dwell (under ~2s) meaningfully
moves it; only sustained drift across many seconds does. Full Phase 7 online
adaptation from natural read-back behaviour (e.g. anchoring on the text
display specifically, rather than an undifferentiated running average) is
still future work.

The `half_fov_*_rad` constants are a *guess* at how many radians of pitch/yaw
correspond to the full screen half-width/height -- calibrated for an assumed
50cm laptop setup. Real setups (screen size, distance, this particular
person's actual achievable gaze range) vary, so a fixed guess systematically
under- or over-shoots: e.g. looking at an edge key never reaches the edge of
the mapped [0,1] range. To stay calibration-free while still correcting for
this, the mapper also auto-expands its effective range the first time it
observes a larger swing than the configured guess -- so once you've looked at
a screen edge once, that becomes the new reference and later glances there
reach it correctly. The configured value is only the *initial* (and minimum)
range; it grows, never shrinks, and is capped by `max_range_rad` against
noise/outliers.

Final-stage smoothing uses a One Euro Filter (Casiez et al. 2012) instead of
a flat exponential moving average: a flat EMA forces one lag/jitter
trade-off for both cases, but dwell selection reads this same output, so
that's actually two different, conflicting needs -- stay put (low jitter)
while fixating on a key, but arrive quickly (low lag) when saccading to a
new one. One Euro Filter adapts its own smoothing to the signal's speed,
giving both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from .one_euro_filter import OneEuroFilter


@dataclass
class MappedGaze:
    raw_x: float  # pre-filter -- feed this to FixationAggregator for uncertainty estimation
    raw_y: float
    smoothed_x: float  # post-One-Euro-Filter -- use for cursor display and S0's dwell mechanic
    smoothed_y: float
    # False when the UNCLAMPED mapped point lies beyond the keyboard by more than
    # offscreen_margin (e.g. glancing at the text box above it). The x/y values are
    # still clamped to [0,1] for display, which alone would make such a glance look
    # like a fixation on the nearest edge key and select stray letters.
    on_screen: bool = True


@dataclass
class GazeMappingConfig:
    half_fov_yaw_rad: float = 0.33
    half_fov_pitch_rad: float = 0.19
    invert_x: bool = False
    invert_y: bool = False
    startup_bias_frames: int = 20  # ~1s at 20fps; averaged to auto-center the resting gaze bias
    max_range_rad: float = 0.9  # sanity cap on auto-expanded range, so one noisy outlier can't blow it up
    filter_min_cutoff: float = 1.0  # One Euro Filter: lower = more smoothing at rest (less jitter)
    filter_beta: float = 3.0  # One Euro Filter: higher = faster response during movement (less lag)
    filter_d_cutoff: float = 1.0  # One Euro Filter: cutoff for the internal velocity estimate
    drift_correction_enabled: bool = True  # continuous post-lock bias adaptation (Phase 7 slice); see module docstring
    drift_tau_s: float = 30.0  # time constant of the chase -- long enough that no single key's dwell moves it
    drift_max_alpha: float = 0.2  # per-update cap, so one long pause/gap can't snap the bias to a single sample
    # Learned screen mapping from Calibrate (experiments/calibrate_uncertainty.py fit_affine):
    #   x = c0 + c1*rel_yaw + c2*rel_pitch ; y likewise, [c0, c1, c2] each.
    # When both are set they REPLACE the heuristic range/invert mapping below (the fit already
    # contains the sign, scale, intercept and yaw/pitch cross-talk for this model + setup).
    offscreen_margin: float = 0.08  # how far past the left/right/bottom keyboard edge (normalized) gaze may stray and still count as on a key
    offscreen_margin_top: float = 0.12  # larger on top: the top row needs looking up, where the estimate is noisiest, and the text box above is ~0.2 away
    # Cursor leash: the displayed/selecting point only moves once the filtered gaze strays more
    # than this far from it, then trails at this distance. Hides the ~0.05 per-frame scatter of
    # webcam gaze while you look at one key; 0 disables.
    stability_radius: float = 0.04
    # affine_x / affine_y are either 3 coefficients [c0, c_yaw, c_pitch] or 6 with head position:
    #   [c0, c_yaw, c_pitch, c_cx, c_cy, c_w]  applied to  (face_feat - face_ref)
    # where face_feat = (face centre x, face centre y, face width), each as a fraction of the frame. The gaze CNN
    # reports a direction in CAMERA coordinates, so where it lands on the screen depends on where the head is.
    affine_x: list[float] | None = None
    affine_y: list[float] | None = None
    face_ref: list[float] | None = None
    # True: affine_* were fit on ABSOLUTE gaze angles (yaw/pitch straight from the CNN), so there is no per-session
    # zero point at all. The older bias-relative mode measured angles against wherever the user happened to be
    # looking during the first second of each session -- calibrate while looking at one thing, start typing while
    # looking at the "Done" button, and the whole mapping is offset. Press C (recenter) to nudge it.
    affine_absolute: bool = False
    # Path (relative to the repo root) of a vision/feature_model.py .npz: EyeTrax-style ridge regression from
    # head-normalised eye/iris landmark features to screen position. When present (and features are supplied) it
    # replaces the gaze-CNN affine mapping entirely; written by Calibrate only when it beats the CNN mapping on
    # held-out trials.
    feature_model: str | None = None
    # Average the eye features over the last N frames before the feature model sees them (calibration trains on 0.5 s means,
    # a single live frame is noisier). 1 = off.
    feature_avg_frames: int = 1

    @classmethod
    def from_yaml(cls, path: str) -> "GazeMappingConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class GazeToScreenMapper:
    """Converts (pitch, yaw) into a smoothed, normalized screen point (x, y),
    after subtracting a one-time auto-centered bias and auto-expanding the
    mapped range to whatever extremes have actually been observed (see
    module docstring)."""

    def __init__(self, config: GazeMappingConfig | None = None) -> None:
        self.config = config or GazeMappingConfig()
        self._bias_pitch = 0.0
        self._bias_yaw = 0.0
        self._bias_samples: list[tuple[float, float]] = []
        self._bias_locked = False
        self._range_yaw = self.config.half_fov_yaw_rad
        self._range_pitch = self.config.half_fov_pitch_rad
        # Last computed bias-corrected angle, pre-clamp -- exposed for
        # diagnostics (e.g. experiments/calibrate_uncertainty.py reconstructs
        # what range would have been needed to reach a known true target,
        # which mu_x/mu_y alone can't give you once clamping has kicked in).
        self._last_rel_yaw = 0.0
        self._last_rel_pitch = 0.0
        self._last_drift_ts: float | None = None
        self._filter_x = self._make_filter()
        self._filter_y = self._make_filter()
        self._stable: tuple[float, float] | None = None
        self._shift = (0.0, 0.0)  # absolute mode: output offset set by recenter()
        self._feature_model = self._load_feature_model()
        self._last_unshifted: tuple[float, float] | None = None
        self._feat_window: list = []
        self._unshifted_hist: list[tuple[float, float, float]] = []  # (t, x, y) before the recenter shift, last ~1 s

    def _load_feature_model(self):
        path = self.config.feature_model
        if not path:
            return None
        from pathlib import Path

        from .feature_model import FeatureModel

        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent / p
        try:
            return FeatureModel.load(p) if p.exists() else None
        except Exception:  # noqa: BLE001 -- a corrupt model file must not take the app down; fall back to the CNN mapping
            return None

    def _remember_unshifted(self, t: float, x: float, y: float) -> None:
        self._unshifted_hist.append((t, x, y))
        while self._unshifted_hist and t - self._unshifted_hist[0][0] > 1.0:
            self._unshifted_hist.pop(0)

    @property
    def shift(self) -> tuple[float, float]:
        """Current recenter() offset (absolute / feature-model mode). (0, 0) = none."""
        return self._shift

    def reset_shift(self) -> None:
        """Drop any recenter() offset: back to exactly what calibration learned."""
        self._shift = (0.0, 0.0)
        self._filter_x.reset()
        self._filter_y.reset()
        self._stable = None

    def _make_filter(self) -> OneEuroFilter:
        cfg = self.config
        return OneEuroFilter(min_cutoff=cfg.filter_min_cutoff, beta=cfg.filter_beta, d_cutoff=cfg.filter_d_cutoff)

    @property
    def bias_locked(self) -> bool:
        """True once the startup auto-center has finished (see module
        docstring). Before this, map() reports a placeholder center point --
        callers that need real measurements (e.g. calibration) should wait
        for this to flip True before recording anything."""
        return self._bias_locked

    @property
    def range_yaw(self) -> float:
        """Current auto-expanded horizontal range (radians)."""
        return self._range_yaw

    @property
    def range_pitch(self) -> float:
        """Current auto-expanded vertical range (radians)."""
        return self._range_pitch

    @property
    def last_rel_yaw(self) -> float:
        """Most recent bias-corrected yaw (radians), pre-clamp."""
        return self._last_rel_yaw

    @property
    def last_rel_pitch(self) -> float:
        """Most recent bias-corrected pitch (radians), pre-clamp."""
        return self._last_rel_pitch

    def reset(self) -> None:
        """Call when the face is lost, so the next detection doesn't lerp from a stale point."""
        self._feat_window = []
        self._filter_x.reset()
        self._filter_y.reset()
        self._stable = None

    def recenter(self, pitch: float | None = None, yaw: float | None = None) -> None:
        """Immediately locks bias to the current gaze angle so (0.5, 0.5) centers on screen."""
        cfg = self.config
        if (cfg.affine_absolute and cfg.affine_x and cfg.affine_y) or self._feature_model is not None:
            # Absolute mode: shift the output so wherever the user is looking right now maps to the centre.
            if self._last_unshifted is not None:
                # Median of the last second, not the last frame: one noisy frame would shift everything by its error.
                hist = self._unshifted_hist
                ref = (float(np.median([h[1] for h in hist])), float(np.median([h[2] for h in hist]))) if len(hist) >= 3 else self._last_unshifted
                self._shift = (0.5 - ref[0], 0.5 - ref[1])
                self._unshifted_hist = []
            self._filter_x.reset()
            self._filter_y.reset()
            self._stable = None
            return
        p = pitch if pitch is not None else getattr(self, "_last_raw_pitch", None)
        y = yaw if yaw is not None else getattr(self, "_last_raw_yaw", None)
        if p is not None and y is not None:
            self._bias_pitch = p
            self._bias_yaw = y
            self._bias_locked = True
            self._range_yaw = self.config.half_fov_yaw_rad
            self._range_pitch = self.config.half_fov_pitch_rad
            self._last_drift_ts = None
            self._filter_x.reset()
            self._filter_y.reset()
            self._stable = None
        else:
            self._bias_pitch = 0.0
            self._bias_yaw = 0.0
            self._bias_samples = []
            self._bias_locked = False
            self._range_yaw = self.config.half_fov_yaw_rad
            self._range_pitch = self.config.half_fov_pitch_rad
            self._last_drift_ts = None

    def map(
        self,
        pitch: float,
        yaw: float,
        timestamp: float,
        face: tuple[float, float, float] | None = None,
        features=None,
    ) -> MappedGaze:
        cfg = self.config
        self._last_raw_pitch = pitch
        self._last_raw_yaw = yaw

        use_features = self._feature_model is not None and features is not None and len(features) == self._feature_model.n_features
        absolute = bool((cfg.affine_absolute and cfg.affine_x and cfg.affine_y) or use_features)
        if absolute:  # no startup auto-centre, no drift chase: angles are used as-is
            self._bias_pitch = 0.0
            self._bias_yaw = 0.0
            self._bias_locked = True
        on_screen = True
        mapped_now = False  # True once this frame went through the real (bias-corrected) mapping
        unclamped_x = unclamped_y = 0.5
        if not self._bias_locked:
            self._bias_samples.append((pitch, yaw))
            if len(self._bias_samples) >= cfg.startup_bias_frames:
                self._bias_pitch = sum(p for p, _ in self._bias_samples) / len(self._bias_samples)
                self._bias_yaw = sum(y for _, y in self._bias_samples) / len(self._bias_samples)
                self._bias_locked = True
                self._last_drift_ts = timestamp
            # Report center while still collecting the bias estimate, rather
            # than showing wild swings computed against a not-yet-valid zero.
            raw_x, raw_y = 0.5, 0.5
        else:
            if cfg.drift_correction_enabled and cfg.drift_tau_s > 0 and not absolute:
                dt = max(timestamp - self._last_drift_ts, 0.0) if self._last_drift_ts is not None else 0.0
                alpha = min(dt / cfg.drift_tau_s, cfg.drift_max_alpha)
                self._bias_pitch += alpha * (pitch - self._bias_pitch)
                self._bias_yaw += alpha * (yaw - self._bias_yaw)
            self._last_drift_ts = timestamp

            rel_pitch = pitch - self._bias_pitch
            rel_yaw = yaw - self._bias_yaw
            self._last_rel_yaw = rel_yaw
            self._last_rel_pitch = rel_pitch

            # Auto-expand the effective range toward whatever extremes we've
            # actually seen, capped against a single noisy outlier.
            self._range_yaw = min(max(self._range_yaw, abs(rel_yaw)), cfg.max_range_rad)
            self._range_pitch = min(max(self._range_pitch, abs(rel_pitch)), cfg.max_range_rad)

            if use_features:
                if cfg.feature_avg_frames > 1:
                    self._feat_window.append(np.asarray(features, dtype=np.float64))
                    del self._feat_window[: -cfg.feature_avg_frames]
                    features = np.mean(self._feat_window, axis=0)
                raw_x, raw_y = (float(v) for v in self._feature_model.predict(features))
                self._last_unshifted = (raw_x, raw_y)
                self._remember_unshifted(timestamp, raw_x, raw_y)
                raw_x += self._shift[0]
                raw_y += self._shift[1]
            elif cfg.affine_x and cfg.affine_y and len(cfg.affine_x) in (3, 6) and len(cfg.affine_y) == len(cfg.affine_x):
                ax, ay = cfg.affine_x, cfg.affine_y
                raw_x = ax[0] + ax[1] * rel_yaw + ax[2] * rel_pitch
                raw_y = ay[0] + ay[1] * rel_yaw + ay[2] * rel_pitch
                if face is not None and cfg.face_ref and len(cfg.face_ref) == 3:
                    # Clamped: the fit only saw a small range of head positions; never extrapolate wildly.
                    head = [min(max(face[i] - cfg.face_ref[i], -0.08), 0.08) for i in range(3)]
                    raw_x += sum(c * h for c, h in zip(ax[3:], head))
                    raw_y += sum(c * h for c, h in zip(ay[3:], head))
                if absolute:
                    self._last_unshifted = (raw_x, raw_y)
                    self._remember_unshifted(timestamp, raw_x, raw_y)
                    raw_x += self._shift[0]
                    raw_y += self._shift[1]
            else:
                raw_x = 0.5 + 0.5 * (rel_yaw / self._range_yaw)
                raw_y = 0.5 - 0.5 * (rel_pitch / self._range_pitch)
                if cfg.invert_x:
                    raw_x = 1.0 - raw_x
                if cfg.invert_y:
                    raw_y = 1.0 - raw_y
            mapped_now = True
            unclamped_x, unclamped_y = raw_x, raw_y
            raw_x = min(max(raw_x, 0.0), 1.0)
            raw_y = min(max(raw_y, 0.0), 1.0)

        if mapped_now:
            # Filter the UNCLAMPED point and judge on/off-keyboard from the filtered value:
            # a single noisy frame past the edge (common on the top row, where the estimate is
            # noisiest) must not drop the highlight; sustained gaze beyond the margin must.
            fx = self._filter_x.filter(timestamp, unclamped_x)
            fy = self._filter_y.filter(timestamp, unclamped_y)
            on_screen = (
                -cfg.offscreen_margin <= fx <= 1.0 + cfg.offscreen_margin
                and -cfg.offscreen_margin_top <= fy <= 1.0 + cfg.offscreen_margin
            )
            filtered_x, filtered_y = min(max(fx, 0.0), 1.0), min(max(fy, 0.0), 1.0)
        else:
            filtered_x = self._filter_x.filter(timestamp, raw_x)
            filtered_y = self._filter_y.filter(timestamp, raw_y)

        smoothed_x, smoothed_y = self._leash(filtered_x, filtered_y)
        return MappedGaze(raw_x=raw_x, raw_y=raw_y, smoothed_x=smoothed_x, smoothed_y=smoothed_y, on_screen=on_screen)

    def _leash(self, x: float, y: float) -> tuple[float, float]:
        r = self.config.stability_radius
        if r <= 0 or self._stable is None:
            self._stable = (x, y)
            return x, y
        sx, sy = self._stable
        d = math.hypot(x - sx, y - sy)
        if d > r:
            k = (d - r) / d  # follow, but stay r behind the filtered point
            sx += (x - sx) * k
            sy += (y - sy) * k
            self._stable = (sx, sy)
        return self._stable
