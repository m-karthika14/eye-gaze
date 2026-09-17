"""Heuristic, calibration-free mapping from raw gaze angle to a normalized
on-screen point. See config/gaze_mapping.yaml for the (placeholder) constants
and MASTER_PROJECT_DOCUMENT.md Section 8 (Stages 1-2) for where this sits in
the pipeline. Phase 3 replaces this with a calibrated probabilistic model.
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

    @classmethod
    def from_yaml(cls, path: str) -> "GazeMappingConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class GazeToScreenMapper:
    """Converts (pitch, yaw) into a smoothed, normalized screen point (x, y)."""

    def __init__(self, config: GazeMappingConfig | None = None) -> None:
        self.config = config or GazeMappingConfig()
        self._smoothed_x: float | None = None
        self._smoothed_y: float | None = None

    def reset(self) -> None:
        """Call when the face is lost, so the next detection doesn't lerp from a stale point."""
        self._smoothed_x = None
        self._smoothed_y = None

    def map(self, pitch: float, yaw: float) -> tuple[float, float]:
        cfg = self.config
        raw_x = 0.5 + 0.5 * (yaw / cfg.half_fov_yaw_rad)
        raw_y = 0.5 - 0.5 * (pitch / cfg.half_fov_pitch_rad)
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
