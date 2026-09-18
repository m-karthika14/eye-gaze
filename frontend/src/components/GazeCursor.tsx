interface Props {
  x: number | null
  y: number | null
}

export function GazeCursor({ x, y }: Props) {
  if (x === null || y === null) return null
  return (
    <div
      className="pointer-events-none absolute z-20 -translate-x-1/2 -translate-y-1/2"
      style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
    >
      {/* Outer pulsing ring for visibility, inner solid dot for precision */}
      <div className="absolute -inset-3 animate-ping rounded-full bg-red-500 opacity-75" />
      <div className="relative h-8 w-8 rounded-full border-4 border-white bg-red-500 shadow-[0_0_12px_4px_rgba(239,68,68,0.8)]" />
    </div>
  )
}
