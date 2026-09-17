import { GazeCursor } from './components/GazeCursor'
import { TextDisplay } from './components/TextDisplay'
import { VirtualKeyboard } from './components/VirtualKeyboard'
import { useGazeSocket } from './hooks/useGazeSocket'
import { useKeyboardLayout } from './hooks/useKeyboardLayout'

function App() {
  const { layout, error: layoutError } = useKeyboardLayout()
  const { message, status, error: socketError } = useGazeSocket()

  return (
    <div className="flex min-h-screen flex-col items-center gap-6 bg-slate-950 p-6 text-slate-100">
      <header className="flex w-full max-w-4xl items-center justify-between">
        <h1 className="text-lg font-semibold">Calibration-Free Gaze Typing -- S0 Baseline</h1>
        <span
          className={`rounded-full px-2 py-1 text-xs ${
            status === 'open' ? 'bg-emerald-700 text-emerald-100' : 'bg-amber-700 text-amber-100'
          }`}
        >
          {status}
        </span>
      </header>

      <div className="w-full max-w-4xl">
        <TextDisplay text={message?.text ?? ''} />
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
            />
            <GazeCursor x={message?.gaze_x ?? null} y={message?.gaze_y ?? null} />
          </>
        ) : (
          <p className="p-6 text-slate-400">Loading keyboard layout...</p>
        )}
      </div>

      <p className="max-w-4xl text-center text-sm text-slate-500">
        Dwell on a key for ~500ms to select it. S0 baseline: no language correction, no adaptive resizing --
        gaze straight to nearest key.
      </p>
    </div>
  )
}

export default App
