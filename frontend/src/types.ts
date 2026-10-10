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

export interface FixationInfo {
  mu_x: number
  mu_y: number
  sigma_x: number
  sigma_y: number
  n_samples: number
  duration_ms: number
}

export interface KeyGeometry {
  x: number
  y: number
  width: number
  height: number
}

export interface GazeMessage {
  gaze_x: number | null
  gaze_y: number | null
  confidence: number
  selected_key: string | null
  hovered_key: string | null
  dwell_progress: number
  text: string
  fixation: FixationInfo | null
  key_probabilities: Record<string, number> | null
  key_layout: Record<string, KeyGeometry> | null
  suggestions: string[]
  low_confidence_veto?: boolean
  invert_x?: boolean
  invert_y?: boolean
  face_found?: boolean
  pitch_deg?: number | null
  yaw_deg?: number | null
  typing_paused?: boolean
  decoder_mode?: 's4' | 's0'
  select_mode?: 'dwell' | 'switch' | 'blink' | 'mouth'
  calibration_stale?: boolean | null
  mouth_progress?: number
  ear?: number | null
  mar?: number | null
  blink_progress?: number
  eyes_closed?: boolean
  active_word?: string
  candidates?: string[]
  fps?: number
  infer_ms?: number
  error?: string
}

// Mirrors backend/calibration.py's WebSocket payloads (/ws/calibrate), the
// in-app equivalent of experiments/calibrate_uncertainty.py's terminal flow.
export interface CalibrationDashboardRow {
  target: string
  predicted: string | null
  err_x: number
  err_y: number
  euclidean: number
  hit: boolean
}

export interface CalibrationDashboard {
  rows: CalibrationDashboardRow[]
  hit_rate: number
  hits: number
  total: number
  mean_err_x: number
  mean_err_y: number
  splits: {
    left_right?: { low_mean_abs_error: number; high_mean_abs_error: number }
    top_bottom?: { low_mean_abs_error: number; high_mean_abs_error: number }
  }
}

export interface CalibrationSigmaResult {
  sigma_scale: number
  naive_scale: number
  n_trials: number
}

export interface CalibrationRangeResult {
  old_half_fov_yaw_rad: number | null
  old_half_fov_pitch_rad: number | null
  new_half_fov_yaw_rad: number
  new_half_fov_pitch_rad: number
}

export type CalibrationMessage =
  | { phase: 'warming_up' }
  | {
      phase: 'trial'
      target_key: string
      trial_index: number
      total_trials: number
      trials_recorded: number
      gaze_x: number | null
      gaze_y: number | null
      fixation_progress: number
      elapsed_s: number
    }
  | { phase: 'trial_recorded'; target_key: string }
  | { phase: 'trial_skipped'; target_key: string }
  | {
      phase: 'done'
      trials_collected: number
      dashboard: CalibrationDashboard
      sigma: CalibrationSigmaResult
      range: CalibrationRangeResult | null
      mapping?: { method: 'features' | 'cnn' | 'none'; loo_rmse: { cnn: number | null; features: number | null }; alpha: number | null }
    }
  | { phase: 'error'; error: string }
