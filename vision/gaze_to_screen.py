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
calibration screen, just an implicit zero-point. Phase 7 replaces this with
continuous online bias adaptation from natural read-back behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class GazeMappingConfig:
    half_fov_yaw_rad: float = 0.33
    half_fov_pitch_rad: float = 0.19
    invert_x: bool = False
    invert_y: bool = False
    smoothing_alpha: float = 0.35
    startup_bias_frames: int = 20  # ~1s at 20fps; averaged to auto-center the resting gaze bias

    @classmethod
    def from_yaml(cls, path: str) -> "GazeMappingConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class GazeToScreenMapper:
    """Converts (pitch, yaw) into a smoothed, normalized screen point (x, y),
    after subtracting a one-time auto-centered bias (see module docstring)."""

    def __init__(self, config: GazeMappingConfig | None = None) -> None:
        self.config = config or GazeMappingConfig()
        self._smoothed_x: float | None = None
        self._smoothed_y: float | None = None
        self._bias_pitch = 0.0
        self._bias_yaw = 0.0
        self._bias_samples: list[tuple[float, float]] = []
        self._bias_locked = False

    def reset(self) -> None:
        """Call when the face is lost, so the next detection doesn't lerp from a stale point."""
        self._smoothed_x = None
        self._smoothed_y = None

    def recenter(self) -> None:
        """Re-run startup auto-centering (e.g. if the user's position/setup changed mid-session)."""
        self._bias_pitch = 0.0
        self._bias_yaw = 0.0
        self._bias_samples = []
        self._bias_locked = False

    def map(self, pitch: float, yaw: float) -> tuple[float, float]:
        cfg = self.config

        if not self._bias_locked:
            self._bias_samples.append((pitch, yaw))
            if len(self._bias_samples) >= cfg.startup_bias_frames:
                self._bias_pitch = sum(p for p, _ in self._bias_samples) / len(self._bias_samples)
                self._bias_yaw = sum(y for _, y in self._bias_samples) / len(self._bias_samples)
                self._bias_locked = True
            # Report center while still collecting the bias estimate, rather
            # than showing wild swings computed against a not-yet-valid zero.
            raw_x, raw_y = 0.5, 0.5
        else:
            rel_pitch = pitch - self._bias_pitch
            rel_yaw = yaw - self._bias_yaw
            raw_x = 0.5 + 0.5 * (rel_yaw / cfg.half_fov_yaw_rad)
            raw_y = 0.5 - 0.5 * (rel_pitch / cfg.half_fov_pitch_rad)
            if cfg.invert_x:
                raw_x = 1.0 - raw_x
            if cfg.invert_y:
                raw_y = 1.0 - raw_y
            raw_x = min(max(raw_x, 0.0), 1.0)
            raw_y = min(max(raw_y, 0.0), 1.0)

        if self._smoothed_x is None or self._smoothed_y is None:
            self._smoothed_x, self._smoothed_y = raw_x, raw_y
        else:
            a = cfg.smoothing_alpha
            self._smoothed_x = a * raw_x + (1 - a) * self._smoothed_x
            self._smoothed_y = a * raw_y + (1 - a) * self._smoothed_y
        return self._smoothed_x, self._smoothed_y
