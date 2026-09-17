interface Props {
  text: string
}

export function TextDisplay({ text }: Props) {
  return (
    <div className="min-h-12 w-full rounded-md border border-slate-700 bg-slate-900 px-4 py-3 text-xl tracking-wide text-slate-100">
      {text.length > 0 ? text : <span className="text-slate-500">Look at a key and dwell to type...</span>}
      <span className="animate-pulse">|</span>
    </div>
  )
}
