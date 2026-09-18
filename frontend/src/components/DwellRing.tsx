interface Props {
  progress: number // 0..1
}

const SIZE = 36
const STROKE = 4
const RADIUS = (SIZE - STROKE) / 2
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

/** Circular progress ring shown over the currently-hovered key, filling up as
 * the dwell timer approaches selection -- lets the user see the system is
 * about to select something and self-correct before it fires. */
export function DwellRing({ progress }: Props) {
  const clamped = Math.min(Math.max(progress, 0), 1)
  const offset = CIRCUMFERENCE * (1 - clamped)

  return (
    <svg
      width={SIZE}
      height={SIZE}
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
    >
      <circle
        cx={SIZE / 2}
        cy={SIZE / 2}
        r={RADIUS}
        fill="none"
        stroke="rgba(255,255,255,0.25)"
        strokeWidth={STROKE}
      />
      <circle
        cx={SIZE / 2}
        cy={SIZE / 2}
        r={RADIUS}
        fill="none"
        stroke="white"
        strokeWidth={STROKE}
        strokeLinecap="round"
        strokeDasharray={CIRCUMFERENCE}
        strokeDashoffset={offset}
        transform={`rotate(-90 ${SIZE / 2} ${SIZE / 2})`}
      />
    </svg>
  )
}
