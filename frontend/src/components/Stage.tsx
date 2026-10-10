import type { ReactNode } from 'react'

/** Height of the control band above the keyboard. The keyboard canvas is everything below it,
 * edge to edge. Live typing and calibration BOTH render through <Stage>, so the canvas is
 * pixel-identical in the two -- the calibrated gaze-to-screen mapping is learned against this
 * exact rectangle, so they must never differ. */
export const TOP_BAND_PX = 128

/** Pixel size of the keyboard canvas right now. Sent to the backend: a calibration is only valid for
 * the canvas size it was recorded at, so the two are compared to warn when they differ. */
export function canvasSize(): [number, number] {
  return [window.innerWidth, Math.max(window.innerHeight - TOP_BAND_PX, 0)]
}

interface Props {
  top: ReactNode
  children: ReactNode
}

export function Stage({ top, children }: Props) {
  return (
    <div className="fixed inset-0 overflow-hidden bg-slate-950 text-slate-100 select-none">
      <div
        className="absolute inset-x-0 top-0 flex flex-col justify-center gap-1.5 overflow-hidden border-b border-slate-800/80 px-3"
        style={{ height: TOP_BAND_PX }}
      >
        {top}
      </div>
      <div className="absolute inset-x-0 bottom-0 overflow-hidden bg-slate-900" style={{ top: TOP_BAND_PX }}>
        {children}
      </div>
    </div>
  )
}
