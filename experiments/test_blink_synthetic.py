"""Synthetic checks for blink-to-select (vision/blink.py, S4LiveDecoder.hold/select_now at_time).
No camera.

    .venv/Scripts/python.exe -m experiments.test_blink_synthetic
"""

from __future__ import annotations

import numpy as np

from decoder.lexicon import Lexicon
from decoder.nearest_key import KeyboardLayout
from decoder.s4_live import S4LiveDecoder
from vision.blink import BlinkDetector, MouthOpenDetector, eye_aspect_ratio, mouth_aspect_ratio

FPS = 13.0  # roughly what the live pipeline delivers


def run(det: BlinkDetector, closed_ms: float, open_ear: float = 0.30, closed_ear: float = 0.05, warm_s: float = 3.0):
    """Open eyes for warm_s, close for closed_ms, reopen. Returns list of BlinkState."""
    dt = 1.0 / FPS
    t, states = 0.0, []
    for _ in range(int(warm_s * FPS)):
        states.append(det.update(open_ear + np.random.default_rng(0).normal(0, 0.005), t)); t += dt
    for _ in range(max(1, round(closed_ms / 1000.0 * FPS))):
        states.append(det.update(closed_ear, t)); t += dt
    for _ in range(int(1.0 * FPS)):
        states.append(det.update(open_ear, t)); t += dt
    return states


def main() -> None:
    # EAR from landmarks: open eye ~0.3, closed ~0
    pts = np.zeros((468, 2))
    for (o, i, u1, u2, l1, l2), cx in (((33, 133, 160, 158, 144, 153), 0.0), ((362, 263, 385, 387, 380, 373), 100.0)):
        pts[o] = (cx, 0); pts[i] = (cx + 30, 0)
        pts[u1] = (cx + 10, -4.5); pts[u2] = (cx + 20, -4.5); pts[l1] = (cx + 10, 4.5); pts[l2] = (cx + 20, 4.5)
    assert abs(eye_aspect_ratio(pts) - 0.30) < 1e-6
    print("[OK] eye_aspect_ratio from face-mesh points: open eye = 0.30")

    for ms in (120, 250):
        assert not any(s.fired for s in run(BlinkDetector(), ms)), ms
    print("[OK] natural blinks (120ms, 250ms) never fire")

    for ms in (600, 1000):
        states = run(BlinkDetector(), ms)
        fired = [s for s in states if s.fired]
        assert len(fired) == 1 and fired[0].closed_since is not None, (ms, len(fired))
    print("[OK] a deliberate 0.6s / 1s closure fires exactly once")

    states = run(BlinkDetector(), 800)
    assert any(s.eyes_closed for s in states) and 0.0 < max(s.progress for s in states) <= 1.0
    assert not states[-1].eyes_closed
    print("[OK] eyes_closed / progress reported while closed, cleared on reopening")

    assert not any(s.fired for s in run(BlinkDetector(), 900, open_ear=0.30, closed_ear=0.29))
    print("[OK] no fire when EAR barely changes (e.g. squinting / noise)")

    # Decoder: the key picked is the one highlighted BEFORE the lids dropped, not the garbage gaze during closure.
    layout = KeyboardLayout.from_json()
    centers = {k.id: k.center() for k in layout.keys}
    d = S4LiveDecoder(lexicon=Lexicon.load_default(), layout=layout)
    d.dwell_enabled = False
    t = 0.0
    hx, hy = centers["H"]
    for _ in range(20):
        t += 0.077
        d.process_sample(hx, hy, t)
    t_closed = t + 0.077
    gx, gy = centers["P"]
    for _ in range(3):  # lids dropping: gaze drifts onto other keys before EAR crosses the threshold
        t += 0.077
        d.process_sample(gx, gy, t)
    ev = d.select_now(t + 0.5, at_time=t_closed - 0.25)
    assert ev.selected_key == "H", ev.selected_key
    print("[OK] blink-select uses the key highlighted before the eyes closed (not the lid-drop drift)")

    d.hold()
    assert d._hover is not None
    print("[OK] hold() keeps the highlighted key through a closure")

    # --- mouth-open gesture
    mp = np.zeros((468, 2))
    mp[78] = (0, 0); mp[308] = (40, 0); mp[13] = (20, -1); mp[14] = (20, 1)
    assert abs(mouth_aspect_ratio(mp) - 0.05) < 1e-6
    print("[OK] mouth_aspect_ratio: closed mouth ~0.05")

    def run_mouth(open_ms, closed_mar=0.05, open_mar=0.45):
        det, t, out = MouthOpenDetector(), 0.0, []
        for _ in range(int(3 * FPS)):
            out.append(det.update(closed_mar, t)); t += 1 / FPS
        for _ in range(max(1, round(open_ms / 1000 * FPS))):
            out.append(det.update(open_mar, t)); t += 1 / FPS
        for _ in range(int(FPS)):
            out.append(det.update(closed_mar, t)); t += 1 / FPS
        return out

    assert not any(x.fired for x in run_mouth(150)), "brief mouth movement must not fire"
    assert sum(x.fired for x in run_mouth(700)) == 1
    assert not any(x.fired for x in run_mouth(900, open_mar=0.12)), "small movements (talking/smiling) must not fire"
    print("[OK] mouth: brief movement ignored, small movement ignored, a held wide-open mouth fires exactly once")

    print("\n[OK] all 9 blink/mouth checks passed")


if __name__ == "__main__":
    main()
