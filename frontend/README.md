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
| `src/hooks/useGazeSocket.ts` | Connects to `ws://localhost:8000/ws/gaze`, exposes the latest message, connection status, and `sendMessage()` for control messages (e.g. `{adaptive_keyboard: false}`) |
| `src/hooks/useKeyboardLayout.ts` | Fetches `GET /config/keyboard-layout` once on load (labels come from here even when positions are overridden live) |
| `src/components/VirtualKeyboard.tsx` | Renders keys from the layout, highlights `hovered_key`/`selected_key`, tints each key by its Phase 3 `key_probabilities` share (purple heatmap), and -- when `keyLayoutOverride` is provided -- uses Phase 6's live-resized geometry instead of the static positions |
| `src/components/SuggestionStrip.tsx` | Shows ranked word suggestions; the top one is accepted by dwelling on SPACE (see backend's `accept_suggestion`), not by gazing at the strip itself -- avoids needing new calibrated geometry |
| `src/components/GazeCursor.tsx` | Overlay dot at the live normalized `(gaze_x, gaze_y)` |
| `src/components/DwellRing.tsx` | Circular progress ring on the hovered key, filling as `dwell_progress` approaches selection |
| `src/components/TextDisplay.tsx` | Shows the backend's live S0 text buffer |
| `src/types.ts` | Mirrors the backend's keyboard-layout and WebSocket message shapes |

The "adaptive keyboard" checkbox in the header toggles Phase 6's live resizing/suggestions on and off (sends `{adaptive_keyboard: bool}` over the open socket); off reproduces the pure Phase 2 S0 baseline with a static keyboard.

**Status:** Phase 2 (S0 baseline UI), Phase 3 (key-probability heatmap), and Phase 6 (adaptive resizing + suggestions) done -- `npm run build` passes cleanly, and the dev server was verified serving against a live backend connection. Phase 11 folds in the rest of the product.
