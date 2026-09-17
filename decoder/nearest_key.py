"""Nearest-key hard selection (S0). Loads config/keyboard_layout.json.

This is the simplest possible "key-probability mapping" -- a single hard
choice instead of a distribution. Phase 3 replaces/extends this with a full
Gaussian probability integrated over each key's region (see
MASTER_PROJECT_DOCUMENT.md Section 8, Stages 2-3); S1-S4 all build on top of
that distribution, but S0 uses only this hard nearest-key lookup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LAYOUT_PATH = Path(__file__).resolve().parent.parent / "config" / "keyboard_layout.json"


@dataclass(frozen=True)
class KeyRegion:
    id: str
    x: float
    y: float
    width: float
    height: float

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height

    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2


class KeyboardLayout:
    def __init__(self, keys: list[KeyRegion]) -> None:
        self.keys = keys

    @classmethod
    def from_json(cls, path: Path | str = DEFAULT_LAYOUT_PATH) -> "KeyboardLayout":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        keys = [
            KeyRegion(id=k["id"], x=k["x"], y=k["y"], width=k["width"], height=k["height"])
            for k in data["keys"]
        ]
        return cls(keys)


def nearest_key(gaze_x: float, gaze_y: float, layout: KeyboardLayout) -> str | None:
    """Returns the id of the key containing (gaze_x, gaze_y). If the point
    falls in a gap between keys but is still within the [0,1]x[0,1] keyboard
    canvas, falls back to the closest key by center distance. Returns None
    only when gaze is outside the canvas entirely (off-screen / no gaze).
    """
    for key in layout.keys:
        if key.contains(gaze_x, gaze_y):
            return key.id

    if not (0.0 <= gaze_x <= 1.0 and 0.0 <= gaze_y <= 1.0):
        return None

    best_id: str | None = None
    best_dist = float("inf")
    for key in layout.keys:
        cx, cy = key.center()
        dist = (cx - gaze_x) ** 2 + (cy - gaze_y) ** 2
        if dist < best_dist:
            best_dist = dist
            best_id = key.id
    return best_id
