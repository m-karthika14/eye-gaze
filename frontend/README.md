# /frontend

React + Vite + TypeScript + Tailwind CSS (v4). Renders the S0 virtual keyboard, live gaze cursor, text display, and Phase 6's adaptive keyboard + word suggestions.

## Run

Requires `/backend` running on `localhost:8000` (see `backend/README.md`).

```
cd frontend
npm install
npm run dev
```

Opens on `http://localhost:5173`.

## Structure

| File | Role |
|---|---|
| `src/hooks/useGazeSocket.ts` | Connects to `ws://localhost:8000/ws/gaze`, exposes the latest message, connection status, and `sendMessage()` for control messages (e.g. `{adaptive_keyboard: false}`). Takes an `enabled` flag so `CalibrationScreen` can pause it -- only one connection can hold the webcam at a time |
| `src/hooks/useCalibrationSocket.ts` | Connects to `ws://localhost:8000/ws/calibrate` on demand (`start()`/`stop()`), exposes the latest calibration message and a coarse `status` (`idle`/`connecting`/`running`/`done`/`error`) |
| `src/hooks/useKeyboardLayout.ts` | Fetches `GET /config/keyboard-layout` once on load (labels come from here even when positions are overridden live) |
| `src/components/VirtualKeyboard.tsx` | Renders keys from the layout, highlights `hovered_key`/`selected_key`, tints each key by its Phase 3 `key_probabilities` share (purple heatmap), and -- when `keyLayoutOverride` is provided -- uses Phase 6's live-resized geometry instead of the static positions. Also highlights `calibrationTarget` with its own progress ring, for calibration mode |
| `src/components/CalibrationScreen.tsx` | In-app calibration: the same live-fixation flow as `experiments/calibrate_uncertainty.py`'s terminal script, shown on the real keyboard. Starts on demand (never auto-connects), shows live trial progress, and renders the final accuracy dashboard/sigma/range results sent by `/ws/calibrate` |
| `src/components/SuggestionStrip.tsx` | Shows ranked word suggestions; the top one is accepted by dwelling on SPACE (see backend's `accept_suggestion`), not by gazing at the strip itself -- avoids needing new calibrated geometry |
| `src/components/GazeCursor.tsx` | Overlay dot at the live normalized `(gaze_x, gaze_y)`; reused during calibration trials too |
| `src/components/DwellRing.tsx` | Circular progress ring on the hovered key, filling as `dwell_progress` approaches selection; reused for calibration's `fixation_progress` |
| `src/components/TextDisplay.tsx` | Shows the backend's live S0 text buffer |
| `src/types.ts` | Mirrors the backend's keyboard-layout and WebSocket message shapes, including `/ws/calibrate`'s |

The "adaptive keyboard" checkbox in the header toggles Phase 6's live resizing/suggestions on and off (sends `{adaptive_keyboard: bool}` over the open socket); off reproduces the pure Phase 2 S0 baseline with a static keyboard. The "Calibrate" button switches to `CalibrationScreen`, pausing `/ws/gaze` so `/ws/calibrate` can hold the webcam; "Done"/"Cancel"/"Back" switch back, resuming the typing connection.

**Status:** Phase 2 (S0 baseline UI), Phase 3 (key-probability heatmap), Phase 6 (adaptive resizing + suggestions), and an in-app calibration flow done -- `npm run build` and `oxlint` pass cleanly. The dev server was verified serving against a live backend connection. Phase 11 folds in the rest of the product.
