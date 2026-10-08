import { useState } from 'react'

interface Props {
  faceFound?: boolean
  pitchDeg?: number | null
  yawDeg?: number | null
  onRecenter?: () => void
}

/**
 * Small screen camera preview displaying live facial mesh, eye sockets,
 * bright blue dots on the center of each eyeball, and real-time gaze vectors.
 */
export function EyeTrackerFeed({ faceFound = false, pitchDeg, yawDeg, onRecenter }: Props) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div className="absolute top-14 right-3 z-40 flex flex-col overflow-hidden rounded-xl border border-slate-700/80 bg-slate-900/95 shadow-2xl backdrop-blur-md transition-all duration-200">
      {/* Header bar */}
      <div className="flex items-center justify-between bg-slate-800/90 px-2.5 py-1.5 text-xs font-semibold text-slate-200 border-b border-slate-700/60">
        <div className="flex items-center gap-2">
          <span
            className={`h-2.5 w-2.5 rounded-full ${
              faceFound ? 'bg-emerald-400 shadow-[0_0_8px_#34d399]' : 'bg-rose-500 shadow-[0_0_8px_#f43f5e]'
            }`}
          />
          <span className="tracking-wide">Eye & Face Tracker</span>
        </div>
        <div className="flex items-center gap-1.5">
          {onRecenter && (
            <button
              onClick={onRecenter}
              title="Recenter gaze (press C)"
              className="rounded bg-cyan-700 hover:bg-cyan-600 px-1.5 py-0.5 text-[10px] text-white"
            >
              Zero
            </button>
          )}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="rounded bg-slate-700 hover:bg-slate-600 px-1.5 py-0.5 text-[10px] text-slate-300"
          >
            {collapsed ? 'Show' : 'Hide'}
          </button>
        </div>
      </div>

      {/* Video Feed Screen */}
      {!collapsed && (
        <div className="relative">
          <img
            src="http://localhost:8000/video_feed"
            alt="Live Eye & Eyeball Tracker"
            className="w-56 h-40 object-cover bg-slate-950 block"
            onError={(e) => {
              // Graceful fallback if video stream connection is restarting
              const target = e.currentTarget
              target.style.display = 'none'
              const fallback = target.nextElementSibling as HTMLElement | null
              if (fallback) fallback.style.display = 'flex'
            }}
            onLoad={(e) => {
              const target = e.currentTarget
              target.style.display = 'block'
              const fallback = target.nextElementSibling as HTMLElement | null
              if (fallback) fallback.style.display = 'none'
            }}
          />

          {/* Connection fallback box */}
          <div className="hidden w-56 h-40 flex-col items-center justify-center bg-slate-950 p-2 text-center text-xs text-slate-400">
            <span className="text-amber-400 font-medium">Connecting Camera...</span>
            <span className="text-[10px] text-slate-500 mt-1">Make sure backend is running</span>
          </div>

          {/* Overlay info badges */}
          <div className="absolute bottom-1 left-1.5 right-1.5 flex items-center justify-between text-[10px] font-mono bg-slate-950/80 rounded px-1.5 py-0.5 text-slate-300 backdrop-blur-sm border border-slate-800/80">
            <span>
              Pitch: {pitchDeg !== null && pitchDeg !== undefined ? `${pitchDeg > 0 ? '+' : ''}${pitchDeg}°` : '--'}
            </span>
            <span>
              Yaw: {yawDeg !== null && yawDeg !== undefined ? `${yawDeg > 0 ? '+' : ''}${yawDeg}°` : '--'}
            </span>
            <span className={faceFound ? 'text-emerald-400 font-semibold' : 'text-rose-400 font-semibold'}>
              {faceFound ? 'LOCKED' : 'NO FACE'}
            </span>
          </div>
        </div>
      )}

      {/* Eyeball legend */}
      {!collapsed && (
        <div className="flex items-center justify-around bg-slate-950/90 py-1 text-[10px] text-slate-400 border-t border-slate-800">
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-blue-500 inline-block" /> Blue Dot: Eyeball
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-amber-400 inline-block" /> Mesh: Face
          </span>
        </div>
      )}
    </div>
  )
}
