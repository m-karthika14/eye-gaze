import { useCallback, useEffect, useRef, useState } from 'react'
import { BACKEND_WS } from '../config'
import type { GazeMessage } from '../types'

export type ConnectionStatus = 'connecting' | 'open' | 'reconnecting' | 'paused'

const BASE_RECONNECT_DELAY_MS = 1500
const MAX_RECONNECT_DELAY_MS = 8000

/**
 * Connects to /ws/gaze and auto-reconnects with backoff on any drop --
 * webcam reads can hiccup transiently (see vision/webcam.py), and the user
 * shouldn't have to manually reload the page to recover.
 *
 * `enabled` (default true) lets a caller close this connection on demand --
 * needed because only one process can hold the webcam at a time, so
 * CalibrationScreen's /ws/calibrate session can't open while this is
 * connected.
 */
export function useGazeSocket(enabled = true) {
  const [message, setMessage] = useState<GazeMessage | null>(null)
  const [status, setStatus] = useState<ConnectionStatus>('connecting')
  const [error, setError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!enabled) {
      wsRef.current?.close()
      wsRef.current = null
      return
    }

    let cancelled = false
    let reconnectDelay = BASE_RECONNECT_DELAY_MS
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null

    function connect() {
      if (cancelled) return
      const ws = new WebSocket(`${BACKEND_WS}/ws/gaze`)
      wsRef.current = ws

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
      wsRef.current?.close()
    }
  }, [enabled])

  const sendMessage = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  return { message, status: enabled ? status : 'paused', error, sendMessage }
}
