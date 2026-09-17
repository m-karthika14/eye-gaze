"""Phase 1 deliverable: live webcam -> gaze pipeline debug visualization.

Opens two windows: your webcam feed with the detected face box, and a black
"screen" canvas with a dot at the estimated normalized gaze point. Press 'q'
in either window to quit.

Run from the repo root (module form, so the package-relative imports work):

    .venv/Scripts/python.exe -m vision.run_demo
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from .pipeline import GazePipeline

CANVAS_SIZE = (720, 1280)  # h, w -- stand-in for "the screen"


def main() -> int:
    print("Starting webcam gaze pipeline... press 'q' in a video window to quit.")
    try:
        pipeline = GazePipeline(camera_index=0, target_fps=30)
    except RuntimeError as exc:
        print(f"[FAIL] {exc}")
        return 1

    with pipeline:
        for sample in pipeline.stream():
            frame = sample.pop("frame")
            display = frame.copy()
            canvas = np.zeros((*CANVAS_SIZE, 3), dtype=np.uint8)

            if sample["face_found"]:
                x1, y1, x2, y2 = sample["bbox"]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cx = int(sample["gaze_x"] * CANVAS_SIZE[1])
                cy = int(sample["gaze_y"] * CANVAS_SIZE[0])
                cv2.circle(canvas, (cx, cy), 18, (0, 0, 255), -1)
                cv2.putText(
                    canvas,
                    f"pitch={np.degrees(sample['pitch']):.1f}deg  yaw={np.degrees(sample['yaw']):.1f}deg",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    canvas,
                    "no face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                display,
                f"{sample['fps']:.1f} fps",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Webcam (Phase 1)", display)
            cv2.imshow("Estimated screen gaze point (debug)", canvas)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
