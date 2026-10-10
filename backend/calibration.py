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
    MIN_FIXATION_SAMPLES,
    REPEATS,
    SAMPLES_PATH,
    SETTLE_DELAY_S,
    SETTLE_TIMEOUT_S,
    TARGET_FIXATION_MS,
    TARGET_KEYS,
    analyze_range,
    compute_accuracy_dashboard,
    FEATURE_MODEL_PATH,
    compute_sigma_scale,
    face_fields,
    fit_affine,
    fit_feature_mapping,
    with_corrected_mu,
    write_back,
    write_back_affine,
    write_back_canvas,
    write_back_feature_model,
    write_back_range,
)
from vision.pipeline import GazePipeline

MIN_TRIALS = 6  # 6 trials is enough for robust median-based uncertainty estimate


async def run_calibration_session(websocket: WebSocket) -> None:
    await websocket.accept()
    loop = asyncio.get_event_loop()

    try:
        pipeline = GazePipeline(camera_index=0, target_fps=30, gestures=False)
    except RuntimeError as exc:
        # Most likely the webcam is already held by /ws/gaze (the typing
        # socket) -- only one of the two can be open at a time.
        await websocket.send_json({"phase": "error", "error": str(exc)})
        await websocket.close()
        return

    stop_requested = False
    canvas: list[float] | None = None  # keyboard canvas size in px, sent by the UI when the session opens

    async def receive_control_messages() -> None:
        nonlocal stop_requested, canvas
        try:
            while True:
                msg = await websocket.receive_json()
                if msg.get("stop"):
                    stop_requested = True
                if isinstance(msg.get("canvas"), list) and len(msg["canvas"]) == 2:
                    canvas = [float(msg["canvas"][0]), float(msg["canvas"][1])]
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
            best_fixation = None
            best_progress = 0.0

            while not stop_requested:
                sample = await loop.run_in_executor(None, next_sample)
                if sample is None:
                    await websocket.send_json({"phase": "error", "error": "Webcam stream ended mid-trial."})
                    return

                elapsed = time.monotonic() - trial_start
                if elapsed < SETTLE_DELAY_S:
                    # Eye is still travelling to the new target; anything the
                    # aggregator holds now is the previous target or transit.
                    pipeline.fixation.reset()
                    fixation = None
                else:
                    fixation = sample.get("fixation")
                if fixation is not None and fixation.n_samples >= MIN_FIXATION_SAMPLES:
                    if best_fixation is None or fixation.duration_ms > best_fixation.duration_ms:
                        best_fixation = fixation
                fixation_progress = min(fixation.duration_ms / TARGET_FIXATION_MS, 1.0) if fixation else 0.0
                best_progress = max(best_progress, fixation_progress)

                await websocket.send_json(
                    {
                        "phase": "trial",
                        "target_key": target_id,
                        "trial_index": trial_index + 1,
                        "total_trials": len(trial_order),
                        "trials_recorded": len(samples),
                        "gaze_x": sample.get("gaze_x"),
                        "gaze_y": sample.get("gaze_y"),
                        "fixation_progress": best_progress,
                        "elapsed_s": elapsed,
                    }
                )

                if (
                    fixation is not None
                    and fixation.duration_ms >= TARGET_FIXATION_MS
                    and fixation.n_samples >= MIN_FIXATION_SAMPLES
                ):
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
                            **face_fields(sample),
                        }
                    )
                    recorded = True
                    await websocket.send_json({"phase": "trial_recorded", "target_key": target_id})
                    break

                if elapsed > SETTLE_TIMEOUT_S:
                    # If user maintained a solid fixation (>=250ms), record it rather than discarding the trial
                    if best_fixation is not None and best_fixation.duration_ms >= 300.0:
                        samples.append(
                            {
                                "target_key": target_id,
                                "true_x": true_cx,
                                "true_y": true_cy,
                                "mu_x": best_fixation.mu_x,
                                "mu_y": best_fixation.mu_y,
                                "sigma_x": best_fixation.sigma_x,
                                "sigma_y": best_fixation.sigma_y,
                                "n_samples": best_fixation.n_samples,
                                "rel_yaw": pipeline.mapper.last_rel_yaw,
                                "rel_pitch": pipeline.mapper.last_rel_pitch,
                                "range_yaw": pipeline.mapper.range_yaw,
                                "range_pitch": pipeline.mapper.range_pitch,
                                **face_fields(sample),
                            }
                        )
                        recorded = True
                        await websocket.send_json({"phase": "trial_recorded", "target_key": target_id})
                    else:
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

        # Learned mapping first: if it validates (leave-one-out), the dashboard and
        # sigma describe IT, not the heuristic mapping the trials were recorded with.
        affine = fit_affine(samples)
        feature_fit = fit_feature_mapping(samples)
        # Two candidate mappings, compared on the SAME held-out trials: the gaze-CNN angle fit, and the EyeTrax-style
        # eye/iris-landmark ridge regression. Keep the feature model only if it is clearly more accurate.
        use_features = feature_fit is not None and (affine is None or feature_fit["loo_rmse"] < 0.92 * affine["loo_rmse"])
        mapping_payload = {
            "method": "features" if use_features else ("cnn" if affine is not None else "none"),
            "loo_rmse": {
                "cnn": None if affine is None else round(affine["loo_rmse"], 4),
                "features": None if feature_fit is None else round(feature_fit["loo_rmse"], 4),
            },
            "alpha": None if feature_fit is None else feature_fit["alpha"],
            "subset": None if feature_fit is None else feature_fit["subset"],
        }
        eval_samples = samples
        if affine is not None:  # always kept as the fallback if the feature model is ever unavailable
            write_back_affine(affine["affine_x"], affine["affine_y"], affine["face_ref"], affine["absolute"])
        if use_features:
            feature_fit["model"].save(FEATURE_MODEL_PATH)
            write_back_feature_model("models/eye_feature_model.npz")
            eval_samples = with_corrected_mu(samples, feature_fit)
        else:
            write_back_feature_model(None)
            if affine is not None:
                eval_samples = with_corrected_mu(samples, affine)

        dashboard = compute_accuracy_dashboard(eval_samples, layout)

        range_payload = None
        range_result = analyze_range(samples) if affine is None else None
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
        sigma_result = compute_sigma_scale(eval_samples, min_sigma)
        write_back(sigma_result["sigma_scale"])
        if canvas is not None:
            write_back_canvas(*canvas)

        await websocket.send_json(
            {
                "phase": "done",
                "trials_collected": len(samples),
                "dashboard": dashboard,
                "sigma": sigma_result,
                "range": range_payload,
                "mapping": mapping_payload,
            }
        )

    except WebSocketDisconnect:
        pass
    finally:
        receiver_task.cancel()
        pipeline.close()
