import { useEffect, useState } from 'react'
import { CalibrationScreen } from './components/CalibrationScreen'
import { EyeTrackerFeed } from './components/EyeTrackerFeed'
import { CandidateStrip } from './components/CandidateStrip'
import { GazeCursor } from './components/GazeCursor'
import { SuggestionStrip } from './components/SuggestionStrip'
import { TextDisplay } from './components/TextDisplay'
import { VirtualKeyboard } from './components/VirtualKeyboard'
import { useGazeSocket } from './hooks/useGazeSocket'
import { useKeyboardLayout } from './hooks/useKeyboardLayout'

function App() {
  const { layout, error: layoutError } = useKeyboardLayout()
  const [mode, setMode] = useState<'typing' | 'calibrating'>('typing')
  const { message, status, error: socketError, sendMessage } = useGazeSocket(mode === 'typing')

  // Merge keys (adaptive layout) is disabled per user request
  const adaptiveEnabled = false
  const [invertX, setInvertX] = useState<boolean>(true)
  const [invertY, setInvertY] = useState<boolean>(false)
  const [typingPaused, setTypingPaused] = useState<boolean>(false)
  const [showTrackerFeed, setShowTrackerFeed] = useState<boolean>(true)
  const [decoderMode, setDecoderMode] = useState<'s4' | 's0'>('s4')
  const [selectMode, setSelectMode] = useState<'dwell' | 'switch' | 'blink' | 'mouth'>('blink')

  // Sync state from backend if received
  useEffect(() => {
    if (message?.invert_x !== undefined) setInvertX(message.invert_x)
    if (message?.invert_y !== undefined) setInvertY(message.invert_y)
    if (message?.typing_paused !== undefined) setTypingPaused(message.typing_paused)
    if (message?.decoder_mode !== undefined) setDecoderMode(message.decoder_mode)
    if (message?.select_mode !== undefined) setSelectMode(message.select_mode)
  }, [message?.invert_x, message?.invert_y, message?.typing_paused, message?.decoder_mode, message?.select_mode])

  // Recenter helper
  function handleRecenter() {
    sendMessage({ recenter: true })
  }

  // Global keyboard shortcut: Press 'C' to recenter gaze to screen center
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'c' || e.key === 'C') {
        handleRecenter()
      } else if (e.key === ' ' || e.key === 'Enter') {
        // Confirm the key being looked at (select mode "switch"). preventDefault so a focused button isn't also clicked.
        e.preventDefault()
        sendMessage({ select: true })
      } else if (e.key === '1' || e.key === '2' || e.key === '3') {
        sendMessage({ pick_candidate: Number(e.key) - 1 })
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [sendMessage])

  function toggleInvertX() {
    const next = !invertX
    setInvertX(next)
    sendMessage({ invert_x: next })
  }

  function toggleInvertY() {
    const next = !invertY
    setInvertY(next)
    sendMessage({ invert_y: next })
  }

  function togglePauseTyping() {
    const next = !typingPaused
    setTypingPaused(next)
    sendMessage({ pause_typing: next })
  }

  function toggleSelectMode() {
    const order = ['blink', 'mouth', 'switch', 'dwell'] as const
    const next = order[(order.indexOf(selectMode) + 1) % order.length]
    setSelectMode(next)
    sendMessage({ select_mode: next })
  }

  function toggleDecoderMode() {
    const next = decoderMode === 's4' ? 's0' : 's4'
    setDecoderMode(next)
    sendMessage({ decoder_mode: next })
  }

  function handleClearText() {
    sendMessage({ clear_text: true })
  }

  function handlePickCandidate(index: number) {
    sendMessage({ pick_candidate: index })
  }

  const suggestions = adaptiveEnabled ? (message?.suggestions ?? []) : []
  // When adaptive is disabled, do NOT merge or resize keys
  const keyLayoutOverride = adaptiveEnabled ? (message?.key_layout ?? null) : null
  const spaceLabel = suggestions.length > 0 ? `SPACE (${suggestions[0]})` : null

  return (
    <div className="relative flex h-screen w-screen flex-col overflow-hidden bg-slate-950 p-2.5 text-slate-100 select-none gap-2">
      {/* Top Header */}
      <header className="flex w-full items-center justify-between border-b border-slate-800/80 pb-2 px-2">
        <div className="flex items-center gap-3">
          <h1 className="text-base font-bold tracking-tight text-slate-100 flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-cyan-400 animate-pulse" />
            Eye Gaze Keyboard
          </h1>
          <span
            className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
              status === 'open' ? 'bg-emerald-900/80 text-emerald-300 border border-emerald-700' : 'bg-amber-900/80 text-amber-300 border border-amber-700'
            }`}
          >
            {status === 'open' ? 'Live' : 'Connecting'}
          </span>
          {message?.face_found && (
            <span className="rounded-full bg-cyan-950 px-2 py-0.5 text-[10px] text-cyan-300 border border-cyan-800">
              Face Detected
            </span>
          )}
        </div>

        {mode === 'typing' && (
          <div className="flex items-center gap-2">
            {/* Decoder Toggle: S4 (Language Model) vs S0 (Raw Nearest Key) */}
            <button
              onClick={toggleDecoderMode}
              title="Toggle between S4 (character language model beam search) and S0 (raw nearest key dwell)"
              className={`rounded-md px-2.5 py-1 text-xs font-semibold border transition-all ${
                decoderMode === 's4'
                  ? 'bg-emerald-800/90 border-emerald-500 text-emerald-100 shadow-[0_0_10px_rgba(16,185,129,0.3)] hover:bg-emerald-700'
                  : 'bg-amber-900/85 border-amber-600 text-amber-100 hover:bg-amber-800'
              }`}
            >
              {decoderMode === 's4' ? '🧠 Decoder: S4 (LM Beam)' : '🔤 Decoder: S0 (Raw Key)'}
            </button>

            {/* Selection method: confirm with a key press vs. dwell */}
            <button
              onClick={toggleSelectMode}
              title="Blink: look at a key, then hold your eyes shut ~0.5s. Space/Enter: look at a key, then press Space or Enter. Dwell: hold your gaze ~0.8s. (Space/Enter always works too.)"
              className={`rounded-md px-2.5 py-1 text-xs font-semibold border transition-all ${
                selectMode !== 'dwell'
                  ? 'bg-fuchsia-900/80 border-fuchsia-500 text-fuchsia-100 hover:bg-fuchsia-800'
                  : 'bg-slate-800 border-slate-600 text-slate-200 hover:bg-slate-700'
              }`}
            >
              {{ blink: '👁️ Select: Blink', mouth: '👄 Select: Mouth open', switch: '⌨️ Select: Space/Enter', dwell: '⏱️ Select: Dwell' }[selectMode]}
            </button>

            {/* Recenter Gaze */}
            <button
              onClick={handleRecenter}
              title="Look at the center of the keyboard and click this button (or press 'C') to re-zero gaze"
              className="flex items-center gap-1.5 rounded-md bg-cyan-700/90 hover:bg-cyan-600 px-3 py-1 text-xs font-semibold text-white shadow transition-all active:scale-95"
            >
              <span>🎯 Recenter Gaze</span>
              <kbd className="rounded bg-cyan-900 px-1 py-0.5 text-[10px] text-cyan-200">C</kbd>
            </button>

            {/* Pause / Resume Typing (for testing tracking safely) */}
            <button
              onClick={togglePauseTyping}
              title="Pause key selection so you can inspect eyeball tracking without typing random letters"
              className={`rounded-md px-2.5 py-1 text-xs font-semibold border transition-colors ${
                typingPaused
                  ? 'bg-amber-700 text-white border-amber-500 shadow-[0_0_10px_rgba(245,158,11,0.5)]'
                  : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
              }`}
            >
              {typingPaused ? '⏸️ Typing Paused' : '▶️ Typing Active'}
            </button>

            {/* Invert X */}
            <button
              onClick={toggleInvertX}
              className={`rounded-md px-2.5 py-1 text-xs font-medium border transition-colors ${
                invertX
                  ? 'bg-indigo-900/80 border-indigo-700 text-indigo-200 hover:bg-indigo-800'
                  : 'bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700'
              }`}
            >
              Invert X: {invertX ? 'ON' : 'OFF'}
            </button>

            {/* Invert Y */}
            <button
              onClick={toggleInvertY}
              className={`rounded-md px-2.5 py-1 text-xs font-medium border transition-colors ${
                invertY
                  ? 'bg-indigo-900/80 border-indigo-700 text-indigo-200 hover:bg-indigo-800'
                  : 'bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700'
              }`}
            >
              Invert Y: {invertY ? 'ON' : 'OFF'}
            </button>

            {/* Camera Screen Toggle */}
            <button
              onClick={() => setShowTrackerFeed(!showTrackerFeed)}
              className={`rounded-md px-2.5 py-1 text-xs font-medium border transition-colors ${
                showTrackerFeed
                  ? 'bg-sky-900/80 border-sky-600 text-sky-200'
                  : 'bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700'
              }`}
            >
              📷 Camera Screen
            </button>

            {/* Calibrate */}
            <button
              onClick={() => setMode('calibrating')}
              className="rounded-md bg-slate-800 hover:bg-slate-700 border border-slate-700 px-3 py-1 text-xs font-medium text-slate-200 transition-colors"
            >
              Calibrate
            </button>
          </div>
        )}
      </header>

      {/* Floating Small Screen: Live Eye Tracker with Mesh & Blue Eyeball Dots */}
      {mode === 'typing' && showTrackerFeed && (
        <EyeTrackerFeed
          faceFound={message?.face_found}
          pitchDeg={message?.pitch_deg}
          yawDeg={message?.yaw_deg}
          onRecenter={handleRecenter}
        />
      )}

      {/* Main Content Area */}
      {mode === 'calibrating' ? (
        <div className="flex-1 min-h-0 w-full max-w-5xl mx-auto overflow-y-auto p-2">
          {layout && <CalibrationScreen layout={layout} onClose={() => setMode('typing')} />}
        </div>
      ) : (
        <div className="flex flex-1 min-h-0 flex-col w-full max-w-5xl mx-auto gap-2">
          {/* Text Display */}
          <div className="shrink-0 space-y-1">
            <TextDisplay
              text={message?.text ?? ''}
              activeWord={message?.active_word}
              decoderMode={decoderMode}
              onClear={handleClearText}
            />
            {suggestions.length > 0 && <SuggestionStrip suggestions={suggestions} />}
            {decoderMode === 's4' && (
              <CandidateStrip candidates={message?.candidates ?? []} onPick={handlePickCandidate} />
            )}
          </div>

          {socketError && <p className="shrink-0 text-xs text-red-400">{socketError}</p>}
          {layoutError && <p className="shrink-0 text-xs text-red-400">{layoutError}</p>}

          {/* Virtual Keyboard Box with Mouse Pointer Cursor (exact same aspect-[16/10] layout as calibration) */}
          <div className="relative aspect-[16/10] w-full max-h-[calc(100vh-165px)] mx-auto overflow-hidden rounded-lg border border-slate-800 bg-slate-900 shadow-2xl">
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
              <div className="flex h-full items-center justify-center text-sm text-slate-400">
                Loading keyboard layout...
              </div>
            )}
          </div>

          {/* Bottom Minimal Readout Bar */}
          <footer className="shrink-0 flex items-center justify-between text-xs text-slate-400 px-2 py-0.5 border-t border-slate-900">
            <div className="flex items-center gap-4">
              <span>
                Pointer: {message?.gaze_x !== null && message?.gaze_x !== undefined
                  ? `X: ${message.gaze_x.toFixed(2)} | Y: ${message.gaze_y?.toFixed(2)}`
                  : 'Searching for face...'}
              </span>
              {message?.hovered_key && (
                <span className="font-semibold text-cyan-400">
                  Target: [{message.hovered_key}] ({Math.round((message.dwell_progress || 0) * 100)}%)
                </span>
              )}
              {typingPaused && (
                <span className="text-amber-400 font-semibold animate-pulse">⏸️ Typing Paused (Gaze Tracking Only)</span>
              )}
              {message?.low_confidence_veto && !typingPaused && (
                <span className="text-amber-400 font-medium">⚠️ Low confidence selection vetoed</span>
              )}
            </div>

            <div className="text-slate-500 text-[11px]">
              <span className={(message?.fps ?? 0) < 10 ? 'text-amber-400 mr-3' : 'mr-3'}>
                {message?.fps ? `${message.fps.toFixed(0)} fps (${message.infer_ms ?? 0} ms inference)` : ''}
              </span>
              {(message?.mouth_progress ?? 0) > 0 && selectMode === 'mouth' && (
                <span className="mr-3 font-semibold text-fuchsia-300">
                  👄 hold… {Math.round((message?.mouth_progress ?? 0) * 100)}%
                </span>
              )}
              {(message?.blink_progress ?? 0) > 0 && selectMode === 'blink' && (
                <span className="mr-3 font-semibold text-fuchsia-300">
                  👁️ hold… {Math.round((message?.blink_progress ?? 0) * 100)}%
                </span>
              )}
              <span className="mr-3 text-slate-500">
                eye {message?.ear?.toFixed(2) ?? '–'} · mouth {message?.mar?.toFixed(2) ?? '–'}
              </span>
              {selectMode === 'blink' ? 'Look at a key, hold eyes shut ~0.5s to select · ' : selectMode === 'mouth' ? 'Look at a key, open your mouth wide ~0.4s to select · ' : selectMode === 'switch' ? 'Look at a key, press Space/Enter to select · ' : ''}
              <kbd className="px-1 py-0.5 rounded bg-slate-800 text-slate-300">C</kbd> re-centres.
            </div>
          </footer>
        </div>
      )}
    </div>
  )
}

export default App
