"""In-app calibration flow: the same live-fixation data collection as
experiments/calibrate_uncertainty.py's terminal script, driven over a
WebSocket instead of an OpenCV window -- so calibration happens in the real
product UI (the actual virtual keyboard the user types on) instead of a
separate command-line tool.

Reuses that script's constants and analysis functions directly (TARGET_KEYS,
REPEATS, compute_accuracy_dashboard, compute_sigma_scale, analyze_range,
write_back, write_back_range) rather than re-deriving them, so the two entry
points can never silently drift apart -- only the per-trial I/O (how the
target is shown and how a trial's progress is reported) differs, since that
part is fundamentally tied to cv2 vs. a WebSocket + React UI.

Only one process can hold the webcam -- the frontend/user must not have
/ws/gaze (the typing socket) open at the same time as this.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any

import yaml
from fastapi import WebSocket, WebSocketDisconnect

from decoder.nearest_key import KeyboardLayout
from experiments.calibrate_uncertainty import (
    DATA_DIR,
    DECODER_PARAMS_PATH,
    GAZE_MAPPING_PATH,
    REPEATS,
    SAMPLES_PATH,
    SETTLE_TIMEOUT_S,
    TARGET_FIXATION_MS,
    TARGET_KEYS,
    analyze_range,
    compute_accuracy_dashboard,
    compute_sigma_scale,
    write_back,
    write_back_range,
)
from vision.pipeline import GazePipeline

MIN_TRIALS = 8  # below this, the robust estimators below don't have enough data to trust (matches the CLI script)


async def run_calibration_session(websocket: WebSocket) -> None:
    await websocket.accept()
    loop = asyncio.get_event_loop()

    try:
        pipeline = GazePipeline(camera_index=0, target_fps=30)
    except RuntimeError as exc:
        # Most likely the webcam is already held by /ws/gaze (the typing
        # socket) -- only one of the two can be open at a time.
        await websocket.send_json({"phase": "error", "error": str(exc)})
        await websocket.close()
        return

    stop_requested = False

    async def receive_control_messages() -> None:
        nonlocal stop_requested
        try:
            while True:
                msg = await websocket.receive_json()
                if msg.get("stop"):
                    stop_requested = True
        except Exception:  # noqa: BLE001 -- connection closing; main loop exits too
            pass

    receiver_task = asyncio.create_task(receive_control_messages())

    def next_sample() -> dict[str, Any] | None:
        return next(stream, None)

    try:
        layout = KeyboardLayout.from_json()
        stream = pipeline.stream()

        await websocket.send_json({"phase": "warming_up"})
        while not pipeline.mapper.bias_locked and not stop_requested:
            sample = await loop.run_in_executor(None, next_sample)
            if sample is None:
                await websocket.send_json({"phase": "error", "error": "Webcam stream ended during warm-up."})
                return

        if stop_requested:
            await websocket.send_json({"phase": "error", "error": "Stopped before warm-up finished."})
            return

        trial_order = TARGET_KEYS * REPEATS
        random.shuffle(trial_order)
        samples: list[dict] = []

        for trial_index, target_id in enumerate(trial_order):
            if stop_requested:
                break
            target_key = next(k for k in layout.keys if k.id == target_id)
            true_cx, true_cy = target_key.center()
            trial_start = time.monotonic()
            recorded = False

            while not stop_requested:
                sample = await loop.run_in_executor(None, next_sample)
                if sample is None:
                    await websocket.send_json({"phase": "error", "error": "Webcam stream ended mid-trial."})
                    return

                elapsed = time.monotonic() - trial_start
                fixation = sample.get("fixation")
                fixation_progress = min(fixation.duration_ms / TARGET_FIXATION_MS, 1.0) if fixation else 0.0

                await websocket.send_json(
                    {
                        "phase": "trial",
                        "target_key": target_id,
                        "trial_index": trial_index + 1,
                        "total_trials": len(trial_order),
                        "trials_recorded": len(samples),
                        "gaze_x": sample.get("gaze_x"),
                        "gaze_y": sample.get("gaze_y"),
                        "fixation_progress": fixation_progress,
                        "elapsed_s": elapsed,
                    }
                )

                if fixation is not None and fixation.duration_ms >= TARGET_FIXATION_MS:
                    samples.append(
                        {
                            "target_key": target_id,
                            "true_x": true_cx,
                            "true_y": true_cy,
                            "mu_x": fixation.mu_x,
                            "mu_y": fixation.mu_y,
                            "sigma_x": fixation.sigma_x,
                            "sigma_y": fixation.sigma_y,
                            "n_samples": fixation.n_samples,
                            "rel_yaw": pipeline.mapper.last_rel_yaw,
                            "rel_pitch": pipeline.mapper.last_rel_pitch,
                            "range_yaw": pipeline.mapper.range_yaw,
                            "range_pitch": pipeline.mapper.range_pitch,
                        }
                    )
                    recorded = True
                    await websocket.send_json({"phase": "trial_recorded", "target_key": target_id})
                    break

                if elapsed > SETTLE_TIMEOUT_S:
                    await websocket.send_json({"phase": "trial_skipped", "target_key": target_id})
                    break

            if stop_requested and not recorded:
                break

        if len(samples) < MIN_TRIALS:
            await websocket.send_json(
                {"phase": "error", "error": f"Only {len(samples)} usable trials -- need at least {MIN_TRIALS}."}
            )
            return

        DATA_DIR.mkdir(exist_ok=True)
        with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        dashboard = compute_accuracy_dashboard(samples, layout)

        range_payload = None
        range_result = analyze_range(samples)
        if range_result is not None:
            old_mapping = yaml.safe_load(GAZE_MAPPING_PATH.read_text(encoding="utf-8")) or {}
            old_yaw = old_mapping.get("half_fov_yaw_rad")
            old_pitch = old_mapping.get("half_fov_pitch_rad")
            new_yaw, new_pitch = range_result
            write_back_range(new_yaw, new_pitch)
            range_payload = {
                "old_half_fov_yaw_rad": old_yaw,
                "old_half_fov_pitch_rad": old_pitch,
                "new_half_fov_yaw_rad": new_yaw,
                "new_half_fov_pitch_rad": new_pitch,
            }

        decoder_params = yaml.safe_load(DECODER_PARAMS_PATH.read_text(encoding="utf-8")) or {}
        min_sigma = float(decoder_params.get("gaze_uncertainty", {}).get("min_sigma", 0.01))
        sigma_result = compute_sigma_scale(samples, min_sigma)
        write_back(sigma_result["sigma_scale"])

        await websocket.send_json(
            {
                "phase": "done",
                "trials_collected": len(samples),
                "dashboard": dashboard,
                "sigma": sigma_result,
                "range": range_payload,
            }
        )

    except WebSocketDisconnect:
        pass
    finally:
        receiver_task.cancel()
        pipeline.close()
