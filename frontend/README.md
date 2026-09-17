# /frontend

React + Vite + TypeScript + Tailwind CSS (v4). Renders the S0 virtual keyboard, live gaze cursor, and text display for the Phase 2 baseline.

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
| `src/hooks/useGazeSocket.ts` | Connects to `ws://localhost:8000/ws/gaze`, exposes the latest message + connection status |
| `src/hooks/useKeyboardLayout.ts` | Fetches `GET /config/keyboard-layout` once on load |
| `src/components/VirtualKeyboard.tsx` | Renders keys from the layout, highlights `hovered_key` / `selected_key` |
| `src/components/GazeCursor.tsx` | Overlay dot at the live normalized `(gaze_x, gaze_y)` |
| `src/components/TextDisplay.tsx` | Shows the backend's live S0 text buffer |
| `src/types.ts` | Mirrors the backend's keyboard-layout and WebSocket message shapes |

**Status:** Phase 2 (S0 baseline UI) done — `npm run build` passes cleanly, and the dev server was verified serving against a live backend connection. Phase 6 adds adaptive key resizing and Groq-powered suggestions on top of this; Phase 11 folds in the rest of the product.