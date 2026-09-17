"""Headless Phase 1 smoke test: captures a fixed number of frames from the
webcam, runs the full pipeline, and reports measured FPS and detection rate.
No GUI window/event loop -- safe to run in a terminal without hanging.

Run from the repo root:

    .venv/Scripts/python.exe -m vision.smoke_test [num_frames]
"""

from __future__ import annotations

import sys
import time

from .pipeline import GazePipeline


def main() -> int:
    num_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    try:
        pipeline = GazePipeline(camera_index=0, target_fps=30)
    except RuntimeError as exc:
        print(f"[FAIL] {exc}")
        return 1

    detected = 0
    count = 0
    start = time.monotonic()
    with pipeline:
        for sample in pipeline.stream():
            count += 1
            if sample["face_found"]:
                detected += 1
                print(
                    f"frame {count:03d}: gaze=({sample['gaze_x']:.3f}, {sample['gaze_y']:.3f}) "
                    f"pitch={sample['pitch']:.3f} yaw={sample['yaw']:.3f} fps={sample['fps']:.1f}"
                )
            else:
                print(f"frame {count:03d}: no face detected  fps={sample['fps']:.1f}")
            if count >= num_frames:
                break
    elapsed = time.monotonic() - start

    print()
    print(f"[OK] captured {count} frames in {elapsed:.1f}s ({count / elapsed:.1f} avg fps)")
    print(f"[OK] face detected in {detected}/{count} frames ({100 * detected / max(count, 1):.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
