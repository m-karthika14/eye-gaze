import { useEffect, useState } from 'react'
import { BACKEND_WS } from '../config'
import type { GazeMessage } from '../types'

export type ConnectionStatus = 'connecting' | 'open' | 'reconnecting'

const BASE_RECONNECT_DELAY_MS = 1500
const MAX_RECONNECT_DELAY_MS = 8000

/**
 * Connects to /ws/gaze and auto-reconnects with backoff on any drop --
 * webcam reads can hiccup transiently (see vision/webcam.py), and the user
 * shouldn't have to manually reload the page to recover.
 */
export function useGazeSocket() {
  const [message, setMessage] = useState<GazeMessage | null>(null)
  const [status, setStatus] = useState<ConnectionStatus>('connecting')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    let ws: WebSocket | null = null
    let reconnectDelay = BASE_RECONNECT_DELAY_MS
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null

    function connect() {
      if (cancelled) return
      ws = new WebSocket(`${BACKEND_WS}/ws/gaze`)

      ws.onopen = () => {
        if (cancelled) return
        setStatus('open')
        setError(null)
        reconnectDelay = BASE_RECONNECT_DELAY_MS
      }
      ws.onclose = () => {
        if (cancelled) return
        setStatus('reconnecting')
        reconnectTimer = setTimeout(() => {
          reconnectDelay = Math.min(reconnectDelay * 1.5, MAX_RECONNECT_DELAY_MS)
          connect()
        }, reconnectDelay)
      }
      ws.onerror = () => {
        if (cancelled) return
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
    }

    connect()

    return () => {
      cancelled = true
      if (reconnectTimer) clearTimeout(reconnectTimer)
      ws?.close()
    }
  }, [])

  return { message, status, error }
}
