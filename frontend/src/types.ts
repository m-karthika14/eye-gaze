// Mirrors config/keyboard_layout.json and the WebSocket schema in
// MASTER_PROJECT_DOCUMENT.md Section 16 (backend/main.py extends it with
// "text" and "hovered_key").

export interface KeyDef {
  id: string
  label: string
  row: number
  x: number
  y: number
  width: number
  height: number
}

export interface KeyboardLayoutData {
  description: string
  grid: { columns: number; rows: number; col_width: number; row_height: number }
  keys: KeyDef[]
}

export interface GazeMessage {
  gaze_x: number | null
  gaze_y: number | null
  confidence: number
  selected_key: string | null
  hovered_key: string | null
  text: string
  error?: string
}
