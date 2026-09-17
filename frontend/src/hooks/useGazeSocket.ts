import { useEffect, useRef, useState } from 'react'
import { BACKEND_WS } from '../config'
import type { GazeMessage } from '../types'

export type ConnectionStatus = 'connecting' | 'open' | 'closed' | 'error'

export function useGazeSocket() {
  const [message, setMessage] = useState<GazeMessage | null>(null)
  const [status, setStatus] = useState<ConnectionStatus>('connecting')
  const [error, setError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let cancelled = false
    const ws = new WebSocket(`${BACKEND_WS}/ws/gaze`)
    wsRef.current = ws

    ws.onopen = () => {
      if (!cancelled) setStatus('open')
    }
    ws.onclose = () => {
      if (!cancelled) setStatus('closed')
    }
    ws.onerror = () => {
      if (cancelled) return
      setStatus('error')
      setError('WebSocket connection error -- is the backend running on :8000?')
    }
    ws.onmessage = (event) => {
      if (cancelled) return
      const data = JSON.parse(event.data) as GazeMessage
      if (data.error) {
        setError(data.error)
        return
      }
      setMessage(data)
    }

    return () => {
      cancelled = true
      ws.close()
    }
  }, [])

  return { message, status, error }
}
