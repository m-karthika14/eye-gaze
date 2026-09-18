"""S0 baseline: gaze -> nearest key -> text (Phase 2 deliverable).

No language intelligence, no adaptive keyboard, no online adaptation -- pure
gaze point -> nearest key (hard selection, dwell-gated) -> text buffer. This
is the baseline every other system (S1-S4) is measured against
(MASTER_PROJECT_DOCUMENT.md Section 10).

Used two ways:
  - live, one sample at a time, by /backend for the interactive demo
  - in batch, via run_sequence(), by Phase 9 experiment scripts replaying a
    recorded or simulated gaze sequence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from decoder.dwell import DEFAULT_DWELL_MS, DwellDetector
from decoder.nearest_key import KeyboardLayout, nearest_key
from decoder.text_buffer import TextBuffer


@dataclass
class S0Event:
    text: str
    selected_key: str | None
    hovered_key: str | None
    dwell_progress: float  # 0..1, how far along the current hover is toward selecting


class S0System:
    def __init__(self, layout: KeyboardLayout | None = None, dwell_ms: float = DEFAULT_DWELL_MS) -> None:
        self.layout = layout or KeyboardLayout.from_json()
        self.dwell = DwellDetector(dwell_ms=dwell_ms)
        self.buffer = TextBuffer()

    def reset(self) -> None:
        self.dwell.reset()
        self.buffer.reset()

    def process_sample(self, gaze_x: float | None, gaze_y: float | None, timestamp: float) -> S0Event:
        key = nearest_key(gaze_x, gaze_y, self.layout) if gaze_x is not None and gaze_y is not None else None
        selected = self.dwell.update(key, timestamp)
        if selected is not None:
            self.buffer.apply_key(selected)
        return S0Event(
            text=self.buffer.text,
            selected_key=selected,
            hovered_key=key,
            dwell_progress=self.dwell.progress,
        )

    def run_sequence(self, samples: Iterable[tuple[float | None, float | None, float]]) -> str:
        """samples: iterable of (gaze_x, gaze_y, timestamp_seconds). Returns
        the final decoded text. Used by Phase 9 experiment scripts."""
        self.reset()
        for gaze_x, gaze_y, ts in samples:
            self.process_sample(gaze_x, gaze_y, ts)
        return self.buffer.text
