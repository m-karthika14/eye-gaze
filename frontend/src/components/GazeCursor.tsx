interface Props {
  x: number | null
  y: number | null
}

/**
 * GazeCursor rendered as a classic mouse pointer arrow, driven by eye-gaze.
 * The active tip of the arrow cursor is pinned directly at (x, y).
 */
export function GazeCursor({ x, y }: Props) {
  if (x === null || y === null) return null

  return (
    <div
      className="pointer-events-none absolute z-50 select-none will-change-[left,top]"
      style={{
        left: `${x * 100}%`,
        top: `${y * 100}%`,
        transform: 'translate(-3px, -2px)',
      }}
    >
      {/* Precision target dot at the exact tip */}
      <div className="absolute top-[2px] left-[3px] -translate-x-1/2 -translate-y-1/2 h-2 w-2 rounded-full bg-cyan-300 border border-white shadow-[0_0_6px_#38bdf8]" />

      {/* Mouse Pointer Arrow */}
      <svg
        className="w-7 h-7 drop-shadow-[0_3px_10px_rgba(0,0,0,0.95)]"
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* High-contrast border and solid body */}
        <path
          d="M3 2L10.5 21L14 13.5L21.5 10L3 2Z"
          fill="#0284c7"
          stroke="#ffffff"
          strokeWidth="1.8"
          strokeLinejoin="round"
        />
        {/* Bright inner face so it stands out against dark and light keys */}
        <path
          d="M5.5 4.5L10 17.5L12.5 12.2L17.8 9.7L5.5 4.5Z"
          fill="#38bdf8"
        />
      </svg>
    </div>
  )
}
