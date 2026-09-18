"""Simple fixation/dwell detector: a key must be continuously gazed at for
dwell_ms before it fires a selection. This is the first, simple version
called for in Phase 2; Phase 3 replaces per-frame key hits with proper
temporal aggregation of the gaze probability distribution over a fixation
window (see MASTER_PROJECT_DOCUMENT.md Section 8, Stage 2).
"""

from __future__ import annotations

# Typical dwell-typing thresholds in the gaze-typing literature sit in the
# 400-800ms range (e.g. EyeSwipe, SliceType); 500ms is a reasonable default.
DEFAULT_DWELL_MS = 500.0


class DwellDetector:
    def __init__(self, dwell_ms: float = DEFAULT_DWELL_MS) -> None:
        self.dwell_ms = dwell_ms
        self._current_key: str | None = None
        self._since: float | None = None  # timestamp (s) the current key hover started
        self._fired_for: str | None = None  # avoid re-firing repeatedly while still hovering
        self._last_timestamp: float | None = None

    def reset(self) -> None:
        self._current_key = None
        self._since = None
        self._fired_for = None
        self._last_timestamp = None

    def update(self, key: str | None, timestamp: float) -> str | None:
        """Feed one (key, timestamp) sample; timestamp in seconds (monotonic
        or wall-clock, just needs to be consistent within a run). Returns the
        key id the instant its dwell threshold is crossed, else None. Use
        `.progress` after calling this for how far along the current hover is
        (for a dwell-progress UI indicator).
        """
        self._last_timestamp = timestamp

        if key != self._current_key:
            self._current_key = key
            self._since = timestamp
            self._fired_for = None
            return None

        if key is None or self._since is None:
            return None

        dwell_elapsed_ms = (timestamp - self._since) * 1000.0
        if dwell_elapsed_ms >= self.dwell_ms and self._fired_for != key:
            self._fired_for = key
            return key
        return None

    @property
    def progress(self) -> float:
        """Fraction (0..1) of dwell_ms elapsed on the currently hovered key,
        as of the last update() call. 0 if not currently hovering a key."""
        if self._current_key is None or self._since is None or self._last_timestamp is None:
            return 0.0
        elapsed_ms = (self._last_timestamp - self._since) * 1000.0
        return min(max(elapsed_ms / self.dwell_ms, 0.0), 1.0)
