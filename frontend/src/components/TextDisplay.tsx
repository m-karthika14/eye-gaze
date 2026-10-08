interface Props {
  text: string
  activeWord?: string
  decoderMode?: 's4' | 's0'
  onClear?: () => void
}

export function TextDisplay({ text, activeWord, decoderMode = 's4', onClear }: Props) {
  return (
    <div className="flex items-center justify-between min-h-12 w-full rounded-md border border-slate-700 bg-slate-900 px-4 py-2 text-xl tracking-wide text-slate-100 shadow-inner">
      <div className="flex items-center gap-1.5 flex-1 overflow-x-auto whitespace-nowrap">
        {text.length > 0 ? (
          <span>{text}</span>
        ) : (
          <span className="text-slate-500 text-base">Look at a key and dwell to type...</span>
        )}
        <span className="animate-pulse text-cyan-400">|</span>

        {decoderMode === 's4' && activeWord && (
          <span className="ml-2 inline-flex items-center gap-1 rounded bg-emerald-950/80 border border-emerald-500/50 px-2 py-0.5 text-xs font-semibold text-emerald-300">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-ping" />
            S4: &quot;{activeWord}&quot;
          </span>
        )}
      </div>

      {text.length > 0 && onClear && (
        <button
          onClick={onClear}
          title="Clear text buffer"
          className="ml-3 shrink-0 rounded px-2 py-1 text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-slate-700 transition-colors"
        >
          🗑️ Clear
        </button>
      )}
    </div>
  )
}
