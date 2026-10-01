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

from dataclasses import dataclass, fields

from .one_euro_filter import OneEuroFilter


@dataclass
class MappedGaze:
    raw_x: float  # pre-filter -- feed this to FixationAggregator for uncertainty estimation
    raw_y: float
    smoothed_x: float  # post-One-Euro-Filter -- use for cursor display and S0's dwell mechanic
    smoothed_y: float


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
        self._filter_x.reset()
        self._filter_y.reset()

    def recenter(self) -> None:
        """Re-run startup auto-centering and range expansion (e.g. if the user's setup changed)."""
        self._bias_pitch = 0.0
        self._bias_yaw = 0.0
        self._bias_samples = []
        self._bias_locked = False
        self._range_yaw = self.config.half_fov_yaw_rad
        self._range_pitch = self.config.half_fov_pitch_rad
        self._last_drift_ts = None

    def map(self, pitch: float, yaw: float, timestamp: float) -> MappedGaze:
        cfg = self.config

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
            if cfg.drift_correction_enabled and cfg.drift_tau_s > 0:
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

            raw_x = 0.5 + 0.5 * (rel_yaw / self._range_yaw)
            raw_y = 0.5 - 0.5 * (rel_pitch / self._range_pitch)
            if cfg.invert_x:
                raw_x = 1.0 - raw_x
            if cfg.invert_y:
                raw_y = 1.0 - raw_y
            raw_x = min(max(raw_x, 0.0), 1.0)
            raw_y = min(max(raw_y, 0.0), 1.0)

        smoothed_x = self._filter_x.filter(timestamp, raw_x)
        smoothed_y = self._filter_y.filter(timestamp, raw_y)
        return MappedGaze(raw_x=raw_x, raw_y=raw_y, smoothed_x=smoothed_x, smoothed_y=smoothed_y)
