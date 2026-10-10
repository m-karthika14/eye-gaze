import { useEffect, useState } from 'react'
import { CalibrationScreen } from './components/CalibrationScreen'
import { EyeTrackerFeed } from './components/EyeTrackerFeed'
import { CandidateStrip } from './components/CandidateStrip'
import { GazeCursor } from './components/GazeCursor'
import { Stage, canvasSize } from './components/Stage'
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
  const [typingPaused, setTypingPaused] = useState<boolean>(false)
  const [showTrackerFeed, setShowTrackerFeed] = useState<boolean>(false)
  const [decoderMode, setDecoderMode] = useState<'s4' | 's0'>('s4')
  const [selectMode, setSelectMode] = useState<'dwell' | 'switch' | 'blink' | 'mouth'>('blink')

  // Sync state from backend if received
  useEffect(() => {
    if (message?.typing_paused !== undefined) setTypingPaused(message.typing_paused)
    if (message?.decoder_mode !== undefined) setDecoderMode(message.decoder_mode)
    if (message?.select_mode !== undefined) setSelectMode(message.select_mode)
  }, [message?.typing_paused, message?.decoder_mode, message?.select_mode])

  // Tell the backend how big the keyboard canvas is (on connect and on every resize), so it can warn when the
  // calibration was recorded at a different size.
  useEffect(() => {
    if (status !== 'open') return
    const send = () => sendMessage({ canvas: canvasSize() })
    send()
    window.addEventListener('resize', send)
    return () => window.removeEventListener('resize', send)
  }, [status, sendMessage])

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

  function toggleFullscreen() {
    if (document.fullscreenElement) void document.exitFullscreen()
    else void document.documentElement.requestFullscreen()
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

  if (mode === 'calibrating') {
    return layout ? <CalibrationScreen layout={layout} onClose={() => setMode('typing')} /> : null
  }

  const btn = 'whitespace-nowrap rounded-md border px-2 py-1 text-[11px] font-semibold transition-colors'

  return (
    <Stage
      top={
        <>
          {/* Row 1: title, status, controls */}
          <header className="flex w-full items-center justify-between gap-3 overflow-x-auto">
            <div className="flex shrink-0 items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-cyan-400 animate-pulse" />
              <h1 className="whitespace-nowrap text-sm font-bold tracking-tight">Eye Gaze Keyboard</h1>
              <span
                className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${
                  status === 'open'
                    ? 'border border-emerald-700 bg-emerald-900/80 text-emerald-300'
                    : 'border border-amber-700 bg-amber-900/80 text-amber-300'
                }`}
              >
                {status === 'open' ? 'Live' : 'Connecting'}
              </span>
              {message?.face_found && (
                <span className="rounded-full border border-cyan-800 bg-cyan-950 px-2 py-0.5 text-[10px] text-cyan-300">Face</span>
              )}
            </div>

            <div className="flex shrink-0 items-center gap-1.5">
              <button
                onClick={toggleDecoderMode}
                title="Toggle between S4 (language-model word decoder) and S0 (raw nearest key)"
                className={`${btn} ${
                  decoderMode === 's4'
                    ? 'border-emerald-500 bg-emerald-800/90 text-emerald-100 hover:bg-emerald-700'
                    : 'border-amber-600 bg-amber-900/85 text-amber-100 hover:bg-amber-800'
                }`}
              >
                {decoderMode === 's4' ? '🧠 S4 decoder' : '🔤 S0 raw key'}
              </button>

              <button
                onClick={toggleSelectMode}
                title="Blink: look at a key, then hold your eyes shut ~0.5s. Mouth: open your mouth wide ~0.4s. Space/Enter: press Space or Enter. Dwell: hold your gaze ~0.8s. (Space/Enter always works too.)"
                className={`${btn} ${
                  selectMode !== 'dwell'
                    ? 'border-fuchsia-500 bg-fuchsia-900/80 text-fuchsia-100 hover:bg-fuchsia-800'
                    : 'border-slate-600 bg-slate-800 text-slate-200 hover:bg-slate-700'
                }`}
              >
                {{ blink: '👁️ Blink', mouth: '👄 Mouth', switch: '⌨️ Space/Enter', dwell: '⏱️ Dwell' }[selectMode]}
              </button>

              <button
                onClick={handleRecenter}
                title="Look at the centre of the screen and click (or press C) to re-zero gaze"
                className={`${btn} border-cyan-700 bg-cyan-700/90 text-white hover:bg-cyan-600`}
              >
                🎯 Recenter <kbd className="ml-1 rounded bg-cyan-900 px-1 text-[10px] text-cyan-200">C</kbd>
              </button>

              <button
                onClick={togglePauseTyping}
                title="Pause key selection so you can inspect tracking without typing"
                className={`${btn} ${
                  typingPaused
                    ? 'border-amber-500 bg-amber-700 text-white'
                    : 'border-slate-700 bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`}
              >
                {typingPaused ? '⏸️ Paused' : '▶️ Active'}
              </button>

              <button
                onClick={() => setShowTrackerFeed(!showTrackerFeed)}
                className={`${btn} ${
                  showTrackerFeed ? 'border-sky-600 bg-sky-900/80 text-sky-200' : 'border-slate-700 bg-slate-800 text-slate-400 hover:bg-slate-700'
                }`}
              >
                📷 Camera
              </button>

              <button
                onClick={toggleFullscreen}
                title="Browser full screen (F11): the keyboard then covers the whole display. Re-run Calibrate after changing window size."
                className={`${btn} border-slate-700 bg-slate-800 text-slate-200 hover:bg-slate-700`}
              >
                ⛶ Full screen
              </button>

              <button onClick={() => setMode('calibrating')} className={`${btn} border-slate-700 bg-slate-800 text-slate-200 hover:bg-slate-700`}>
                Calibrate
              </button>
            </div>
          </header>

          {/* Row 2: typed text */}
          <TextDisplay text={message?.text ?? ''} activeWord={message?.active_word} decoderMode={decoderMode} onClear={handleClearText} />

          {/* Row 3: alternatives (left) and live readouts (right) */}
          <div className="flex min-h-6 w-full items-center justify-between gap-3 overflow-hidden text-[11px] text-slate-400">
            <div className="flex min-w-0 items-center gap-3">
              {suggestions.length > 0 && <SuggestionStrip suggestions={suggestions} />}
              {decoderMode === 's4' && <CandidateStrip candidates={message?.candidates ?? []} onPick={handlePickCandidate} />}
              {socketError && <span className="text-red-400">{socketError}</span>}
              {layoutError && <span className="text-red-400">{layoutError}</span>}
              {typingPaused && <span className="animate-pulse font-semibold text-amber-400">⏸️ Typing paused</span>}
              {message?.calibration_stale && (
                <button
                  onClick={() => setMode('calibrating')}
                  className="animate-pulse whitespace-nowrap rounded border border-amber-500 bg-amber-900/70 px-2 py-0.5 font-semibold text-amber-200 hover:bg-amber-800"
                >
                  ⚠ Not calibrated for this window size — what you look at won&apos;t match the keys. Click to calibrate
                </button>
              )}
            </div>
            <div className="flex shrink-0 items-center gap-3 whitespace-nowrap">
              {message?.hovered_key && (
                <span className="font-semibold text-cyan-400">
                  [{message.hovered_key}] {Math.round((message.dwell_progress || 0) * 100)}%
                </span>
              )}
              {(message?.mouth_progress ?? 0) > 0 && selectMode === 'mouth' && (
                <span className="font-semibold text-fuchsia-300">👄 hold… {Math.round((message?.mouth_progress ?? 0) * 100)}%</span>
              )}
              {(message?.blink_progress ?? 0) > 0 && selectMode === 'blink' && (
                <span className="font-semibold text-fuchsia-300">👁️ hold… {Math.round((message?.blink_progress ?? 0) * 100)}%</span>
              )}
              {message?.eyes_closed && <span className="font-semibold text-amber-300">⏸ pointer held</span>}
              {message?.gaze_shift && (Math.abs(message.gaze_shift[0]) > 0.02 || Math.abs(message.gaze_shift[1]) > 0.02) && (
                <button
                  onClick={() => sendMessage({ reset_shift: true })}
                  title="The Recenter (C) button has shifted the whole pointer by this much. Click to remove the shift and go back to the calibrated mapping."
                  className="rounded border border-amber-500 bg-amber-900/70 px-1.5 py-0.5 font-semibold text-amber-200 hover:bg-amber-800"
                >
                  shift {message.gaze_shift[0] >= 0 ? '+' : ''}{message.gaze_shift[0].toFixed(2)} / {message.gaze_shift[1] >= 0 ? '+' : ''}{message.gaze_shift[1].toFixed(2)} ✕ reset
                </button>
              )}
              <span className="text-slate-500">
                eye {message?.ear?.toFixed(2) ?? '–'} · mouth {message?.mar?.toFixed(2) ?? '–'}
              </span>
              <span className={(message?.fps ?? 0) < 10 ? 'text-amber-400' : ''}>
                {message?.fps ? `${message.fps.toFixed(0)} fps (${message.infer_ms ?? 0} ms)` : ''}
              </span>
              <span className="text-slate-500">
                {selectMode === 'blink'
                  ? 'hold eyes shut ~0.5s to select'
                  : selectMode === 'mouth'
                    ? 'open mouth wide ~0.4s to select'
                    : selectMode === 'switch'
                      ? 'press Space/Enter to select'
                      : 'hold gaze to select'}
              </span>
            </div>
          </div>
        </>
      }
    >
      {mode === 'typing' && showTrackerFeed && (
        <EyeTrackerFeed
          faceFound={message?.face_found}
          pitchDeg={message?.pitch_deg}
          yawDeg={message?.yaw_deg}
          onRecenter={handleRecenter}
        />
      )}

      {/* The keyboard canvas: everything below the control band, edge to edge. */}
      <div className="relative h-full w-full">
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
          <div className="flex h-full items-center justify-center text-sm text-slate-400">Loading keyboard layout...</div>
        )}
      </div>
    </Stage>
  )
}

export default App
