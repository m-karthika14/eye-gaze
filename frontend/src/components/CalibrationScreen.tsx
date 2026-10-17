import { useCalibrationSocket } from '../hooks/useCalibrationSocket'
import type { KeyboardLayoutData } from '../types'
import { GazeCursor } from './GazeCursor'
import { Stage } from './Stage'
import { VirtualKeyboard } from './VirtualKeyboard'

interface Props {
  layout: KeyboardLayoutData
  onClose: () => void
}

/**
 * In-app calibration: the same live-fixation data collection as
 * experiments/calibrate_uncertainty.py's terminal script (highlight a key,
 * wait for a stable fixation, record it, repeat), surfaced on the real
 * virtual keyboard instead of a separate OpenCV window. Talks to
 * /ws/calibrate (backend/calibration.py), which reuses that script's exact
 * constants and analysis functions.
 */
export function CalibrationScreen({ layout, onClose }: Props) {
  const { message, status, start, stop } = useCalibrationSocket()

  const trial = message?.phase === 'trial' ? message : null
  const done = message?.phase === 'done' ? message : null
  const errored = message?.phase === 'error' ? message : null

  const top = (
    <>
      <div className="flex w-full items-center justify-between">
        <h2 className="text-base font-semibold">Accuracy Check</h2>
        {status === 'running' && (
          <button onClick={stop} className="rounded bg-slate-800 px-3 py-1 text-xs text-slate-300 hover:bg-slate-700">
            Stop early
          </button>
        )}
      </div>
      <p className="text-sm text-slate-400">
        {trial
          ? `Trial ${trial.trial_index}/${trial.total_trials} -- ${trial.trials_recorded} recorded -- look at the highlighted key and hold your gaze steady`
          : status === 'connecting'
            ? 'Connecting to webcam...'
            : message?.phase === 'warming_up'
              ? 'Warming up -- look at the screen for a moment...'
              : done
                ? 'Done.'
                : 'Look at each highlighted key. Same full-screen keyboard as typing, so keep the window as it is.'}
      </p>
    </>
  )

  return (
    <Stage top={top}>
      {trial ? (
        // The SAME canvas as live typing (see Stage): the calibration is only valid for this exact rectangle.
        <div className="relative h-full w-full">
          <VirtualKeyboard
            layout={layout}
            hoveredKey={null}
            selectedKey={null}
            dwellProgress={0}
            keyProbabilities={null}
            keyLayoutOverride={null}
            spaceLabel={null}
            calibrationTarget={trial.target_key}
            calibrationProgress={trial.fixation_progress}
          />
          <GazeCursor x={trial.gaze_x} y={trial.gaze_y} />
        </div>
      ) : (
        <div className="flex h-full w-full justify-center overflow-y-auto p-6">
          <div className="flex w-full max-w-4xl flex-col items-center gap-4">
            {status === 'idle' && (
              <div className="w-full rounded-lg border border-slate-800 bg-slate-950 p-4 text-sm text-slate-300">
                <p>
                  Shows one key at a time -- look at it and hold your gaze steady until it fills in. Keep your head still
                  (move only your eyes), sit at a consistent distance, and avoid backlighting. The keys include the
                  far left and right edges, so look all the way out to them. Takes a couple of minutes.
                </p>
                <button
                  onClick={start}
                  className="mt-3 rounded bg-emerald-700 px-4 py-2 text-sm font-medium hover:bg-emerald-600"
                >
                  Start accuracy check
                </button>
                <button onClick={onClose} className="ml-2 mt-3 rounded bg-slate-800 px-4 py-2 text-sm hover:bg-slate-700">
                  Cancel
                </button>
              </div>
            )}

            {errored && (
              <div className="w-full rounded border border-red-800 bg-red-950/50 p-4 text-sm text-red-300">
                <p className="font-medium">{errored.error}</p>
                {errored.error.toLowerCase().includes('webcam') && (
                  <p className="mt-1 text-red-400">
                    Only one connection can hold the webcam at a time -- make sure no other accuracy check or typing session is running.
                  </p>
                )}
                <button onClick={onClose} className="mt-3 rounded bg-slate-800 px-3 py-1 text-xs hover:bg-slate-700">
                  Back
                </button>
              </div>
            )}

            {done && <CalibrationResults result={done} onClose={onClose} />}
          </div>
        </div>
      )}
    </Stage>
  )
}

function CalibrationResults({
  result,
  onClose,
}: {
  result: Extract<NonNullable<ReturnType<typeof useCalibrationSocket>['message']>, { phase: 'done' }>
  onClose: () => void
}) {
  const { dashboard, sigma, range, trials_collected, mapping } = result
  const lr = dashboard.splits.left_right
  const tb = dashboard.splits.top_bottom

  return (
    <div className="w-full space-y-4">
      <div className="rounded-lg border border-slate-800 bg-slate-900 p-4">
        <p className="text-sm text-slate-300">
          {trials_collected} trials collected -- key-hit accuracy{' '}
          <span className="font-semibold text-slate-100">
            {(dashboard.hit_rate * 100).toFixed(0)}% ({dashboard.hits}/{dashboard.total})
          </span>
        </p>
        <p className="mt-1 text-xs text-slate-500">
          mean signed error: x={dashboard.mean_err_x >= 0 ? '+' : ''}
          {dashboard.mean_err_x.toFixed(3)} y={dashboard.mean_err_y >= 0 ? '+' : ''}
          {dashboard.mean_err_y.toFixed(3)} (near zero = no directional bias, just noise)
        </p>
        {mapping && mapping.method !== 'none' && (
          <p className="mt-1 text-xs text-slate-400">
            mapping used: <span className="font-semibold text-slate-200">{mapping.method === 'features' ? 'eye/iris landmarks (EyeTrax-style)' : 'gaze CNN angles'}</span>{' '}
            -- held-out error: gaze CNN {mapping.loo_rmse.cnn?.toFixed(3) ?? 'n/a'}, eye landmarks {mapping.loo_rmse.features?.toFixed(3) ?? 'n/a'} (lower is better)
          </p>
        )}
        {lr && (
          <p className="text-xs text-slate-500">
            LEFT mean abs error={lr.low_mean_abs_error.toFixed(3)} RIGHT mean abs error={lr.high_mean_abs_error.toFixed(3)}
          </p>
        )}
        {tb && (
          <p className="text-xs text-slate-500">
            TOP mean abs error={tb.low_mean_abs_error.toFixed(3)} BOTTOM mean abs error={tb.high_mean_abs_error.toFixed(3)}
          </p>
        )}
        {range && (
          <p className="mt-2 text-xs text-slate-500">
            gaze range: yaw {range.old_half_fov_yaw_rad?.toFixed(4) ?? '?'} to {range.new_half_fov_yaw_rad.toFixed(4)} rad, pitch{' '}
            {range.old_half_fov_pitch_rad?.toFixed(4) ?? '?'} to {range.new_half_fov_pitch_rad.toFixed(4)} rad -- written to config/gaze_mapping.yaml
          </p>
        )}
        <p className="text-xs text-slate-500">
          sigma_scale: {sigma.sigma_scale.toFixed(3)} (robust) vs {sigma.naive_scale.toFixed(3)} (naive) -- written to config/decoder_params.yaml
        </p>
      </div>

      <div className="max-h-64 overflow-y-auto rounded-lg border border-slate-800">
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 bg-slate-900 text-slate-400">
            <tr>
              <th className="px-3 py-2">target</th>
              <th className="px-3 py-2">predicted</th>
              <th className="px-3 py-2">err_x</th>
              <th className="px-3 py-2">err_y</th>
              <th className="px-3 py-2">hit?</th>
            </tr>
          </thead>
          <tbody>
            {dashboard.rows.map((r, i) => (
              <tr key={i} className="border-t border-slate-800 text-slate-300">
                <td className="px-3 py-1">{r.target}</td>
                <td className="px-3 py-1">{r.predicted ?? '?'}</td>
                <td className="px-3 py-1">{r.err_x >= 0 ? '+' : ''}{r.err_x.toFixed(3)}</td>
                <td className="px-3 py-1">{r.err_y >= 0 ? '+' : ''}{r.err_y.toFixed(3)}</td>
                <td className={`px-3 py-1 ${r.hit ? 'text-emerald-400' : 'text-slate-600'}`}>{r.hit ? 'YES' : 'no'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <button onClick={onClose} className="rounded bg-emerald-700 px-4 py-2 text-sm font-medium hover:bg-emerald-600">
        Done -- back to typing
      </button>
    </div>
  )
}
