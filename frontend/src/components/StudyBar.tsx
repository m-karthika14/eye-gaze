import { useEffect, useState } from 'react'
import { BACKEND_HTTP } from '../config'
import type { GazeMessage } from '../types'

interface Props {
  message: GazeMessage | null
  sendMessage: (data: Record<string, unknown>) => void
}

/** Typing-study helper: shows the target phrase, and logs each finished phrase (typed text, time, selections, backspaces, system
 * S0-S4, selection method) to data/sessions/<participant>.jsonl on the backend. Counting and timing happen on the server.
 * Entirely passive until "Start" is pressed; nothing else in the app depends on it. */
export function StudyBar({ message, sendMessage }: Props) {
  const [phrases, setPhrases] = useState<string[]>([])
  const [idx, setIdx] = useState(0)
  const [running, setRunning] = useState(false)
  const [participant, setParticipant] = useState<string>(() => {
    try {
      return window.localStorage.getItem('participantId') ?? ''
    } catch {
      return ''
    }
  })

  useEffect(() => {
    fetch(`${BACKEND_HTTP}/config/evaluation-phrases`)
      .then((r) => r.json())
      .then((d: { phrases: string[] }) => setPhrases(d.phrases))
      .catch(() => setPhrases([]))
  }, [])

  const target = phrases[idx] ?? ''
  const mode = (message?.decoder_mode ?? 's4').toUpperCase()

  function start() {
    sendMessage({ study_start: true })
    setRunning(true)
  }

  function done() {
    sendMessage({ study_done: { participant: participant.trim(), phrase_index: idx, target } })
    setRunning(false)
    setIdx((i) => Math.min(i + 1, Math.max(phrases.length - 1, 0)))
  }

  const small = 'rounded border px-1.5 py-0.5 text-[11px] font-semibold'
  return (
    <div className="flex items-center gap-2 whitespace-nowrap rounded border border-sky-700 bg-sky-950/60 px-2 py-0.5 text-[11px] text-sky-100">
      <input
        value={participant}
        onChange={(e) => {
          setParticipant(e.target.value)
          try {
            window.localStorage.setItem('participantId', e.target.value)
          } catch {
            /* not remembered */
          }
        }}
        placeholder="participant"
        className="w-20 rounded border border-slate-700 bg-slate-900 px-1 py-0.5 text-[11px] text-slate-100"
      />
      <button onClick={() => setIdx((i) => Math.max(i - 1, 0))} disabled={running} className={`${small} border-slate-600 bg-slate-800 disabled:opacity-40`}>
        ◀
      </button>
      <span className="text-slate-400">
        {phrases.length ? `${idx + 1}/${phrases.length}` : '–'} · {mode}
      </span>
      <span className="font-semibold text-sky-200">“{target}”</span>
      <button onClick={() => setIdx((i) => Math.min(i + 1, phrases.length - 1))} disabled={running} className={`${small} border-slate-600 bg-slate-800 disabled:opacity-40`}>
        ▶
      </button>
      {running ? (
        <button onClick={done} className={`${small} border-emerald-500 bg-emerald-700 text-white`}>
          Done ✓
        </button>
      ) : (
        <button onClick={start} className={`${small} border-sky-500 bg-sky-700 text-white`}>
          Start
        </button>
      )}
    </div>
  )
}
