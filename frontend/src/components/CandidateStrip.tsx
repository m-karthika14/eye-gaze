interface Props {
  candidates: string[]
  onPick: (index: number) => void
}

/** Ranked decodings of the word being typed (S4 live decoder). The first is
 * what SPACE will commit; clicking (or pressing 1/2/3) a lower-ranked one
 * commits that instead -- the escape hatch for when the decoder's top guess
 * is wrong but the intended word is among the alternatives. Mouse/keyboard
 * only, by design: gaze can't yet resolve a target this small outside the
 * calibrated keyboard area. */
export function CandidateStrip({ candidates, onPick }: Props) {
  if (candidates.length === 0) return null

  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="text-slate-500">did you mean:</span>
      {candidates.map((word, i) => (
        <button
          key={word}
          onClick={() => onPick(i)}
          title={i === 0 ? 'Top guess -- dwell SPACE to accept, or click' : 'Click to use this word instead'}
          className={`rounded px-2 py-1 transition-colors ${
            i === 0
              ? 'bg-emerald-700 text-emerald-100 hover:bg-emerald-600'
              : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
          }`}
        >
          <kbd className="mr-1 text-[10px] text-slate-400">{i + 1}</kbd>
          {word}
        </button>
      ))}
    </div>
  )
}
