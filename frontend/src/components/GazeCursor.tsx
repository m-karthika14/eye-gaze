interface Props {
  x: number | null
  y: number | null
}

export function GazeCursor({ x, y }: Props) {
  if (x === null || y === null) return null
  return (
    <div
      className="pointer-events-none absolute z-10 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-red-500/80 shadow-lg"
      style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
    />
  )
}
