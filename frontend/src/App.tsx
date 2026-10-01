import { useState } from 'react'
import { CalibrationScreen } from './components/CalibrationScreen'
import { GazeCursor } from './components/GazeCursor'
import { SuggestionStrip } from './components/SuggestionStrip'
import { TextDisplay } from './components/TextDisplay'
import { VirtualKeyboard } from './components/VirtualKeyboard'
import { useGazeSocket } from './hooks/useGazeSocket'
import { useKeyboardLayout } from './hooks/useKeyboardLayout'

function App() {
  const { layout, error: layoutError } = useKeyboardLayout()
  const [mode, setMode] = useState<'typing' | 'calibrating'>('typing')
  // Paused while calibrating -- only one connection can hold the webcam at a
  // time, and /ws/calibrate needs it (see useGazeSocket's `enabled` doc).
  const { message, status, error: socketError, sendMessage } = useGazeSocket(mode === 'typing')
  const [adaptiveEnabled, setAdaptiveEnabled] = useState(true)

  function toggleAdaptive() {
    const next = !adaptiveEnabled
    setAdaptiveEnabled(next)
    sendMessage({ adaptive_keyboard: next })
  }

  const suggestions = adaptiveEnabled ? (message?.suggestions ?? []) : []
  const keyLayoutOverride = adaptiveEnabled ? (message?.key_layout ?? null) : null
  const spaceLabel = suggestions.length > 0 ? `SPACE (${suggestions[0]})` : null

  return (
    <div className="flex min-h-screen flex-col items-center gap-6 bg-slate-950 p-6 text-slate-100">
      <header className="flex w-full max-w-4xl items-center justify-between">
        <h1 className="text-lg font-semibold">Calibration-Free Gaze Typing -- S0 Baseline + Phase 6 Adaptive Keyboard</h1>
        <div className="flex items-center gap-3">
          {mode === 'typing' && (
            <>
              <label className="flex items-center gap-2 text-xs text-slate-400">
                <input type="checkbox" checked={adaptiveEnabled} onChange={toggleAdaptive} />
                adaptive keyboard
              </label>
              <button
                onClick={() => setMode('calibrating')}
                className="rounded bg-slate-800 px-3 py-1 text-xs text-slate-300 hover:bg-slate-700"
              >
                Calibrate
              </button>
            </>
          )}
          <span
            className={`rounded-full px-2 py-1 text-xs ${
              status === 'open' ? 'bg-emerald-700 text-emerald-100' : 'bg-amber-700 text-amber-100'
            }`}
          >
            {status}
          </span>
        </div>
      </header>

      {mode === 'calibrating' ? (
        layout && <CalibrationScreen layout={layout} onClose={() => setMode('typing')} />
      ) : (
        <>
          <div className="w-full max-w-4xl space-y-2">
            <TextDisplay text={message?.text ?? ''} />
            <SuggestionStrip suggestions={suggestions} />
          </div>

          {socketError && <p className="text-sm text-red-400">{socketError}</p>}
          {layoutError && <p className="text-sm text-red-400">{layoutError}</p>}

          <div className="relative aspect-[16/10] w-full max-w-4xl overflow-hidden rounded-lg border border-slate-800 bg-slate-900">
            {layout ? (
              <>
                <VirtualKeyboard
                  layout={layout}
                  hoveredKey={message?.hovered_key ?? null}
                  selectedKey={message?.selected_key ?? null}
                  dwellProgress={message?.dwell_progress ?? 0}
                  keyProbabilities={message?.key_probabilities ?? null}
                  keyLayoutOverride={keyLayoutOverride}
                  spaceLabel={spaceLabel}
                />
                <GazeCursor x={message?.gaze_x ?? null} y={message?.gaze_y ?? null} />
              </>
            ) : (
              <p className="p-6 text-slate-400">Loading keyboard layout...</p>
            )}
          </div>

          <p className="max-w-4xl text-center text-sm text-slate-500">
            Dwell on a key for ~500ms to select it. {adaptiveEnabled
              ? 'Adaptive keyboard on: keys resize by predicted likelihood, and SPACE accepts the top word suggestion.'
              : 'Adaptive keyboard off: pure S0 baseline, static keyboard -- matches the S1-vs-S4 controlled comparison.'}
            {' '}The purple glow is Phase 3's per-key probability distribution (not yet used for selection).
          </p>

          {message?.fixation && (
            <p className="text-xs text-slate-600">
              fixation: sigma_x={message.fixation.sigma_x.toFixed(3)} sigma_y={message.fixation.sigma_y.toFixed(3)}{' '}
              n={message.fixation.n_samples} duration={message.fixation.duration_ms.toFixed(0)}ms
            </p>
          )}
        </>
      )}
    </div>
  )
}

export default App
