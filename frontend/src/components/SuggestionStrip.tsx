interface Props {
  suggestions: string[]
}

/** Ranked word-completion suggestions (Phase 6). Purely a display -- the top
 * suggestion is accepted by dwelling on the existing SPACE key (see
 * VirtualKeyboard's spaceLabel prop and backend/adaptive_layout.py's
 * accept_suggestion), not by gazing at these directly; adding a new
 * gaze-interactive region here would need new calibration for it. */
export function SuggestionStrip({ suggestions }: Props) {
  if (suggestions.length === 0) return null

  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="text-slate-500">suggestions:</span>
      {suggestions.map((word, i) => (
        <span
          key={word}
          className={`rounded px-2 py-1 ${
            i === 0 ? 'bg-emerald-700 text-emerald-100' : 'bg-slate-800 text-slate-300'
          }`}
        >
          {word}
          {i === 0 && <span className="ml-1 text-emerald-300">(SPACE)</span>}
        </span>
      ))}
    </div>
  )
}
