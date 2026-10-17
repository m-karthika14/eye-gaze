import { useCallback, useEffect, useRef, useState } from 'react'
import { BACKEND_WS } from '../config'
import { canvasSize } from '../components/Stage'
import type { CalibrationMessage } from '../types'

export type CalibrationStatus = 'idle' | 'connecting' | 'running' | 'done' | 'error'

/**
 * Connects to /ws/calibrate on demand (not auto-connected like /ws/gaze --
 * calibration is a one-shot session the user opts into, and only one of the
 * two sockets can hold the webcam at a time). Call start() to begin, stop()
 * to end early (the backend still analyzes whatever trials were collected,
 * same as pressing 'q' in the terminal script).
 */
export function useCalibrationSocket() {
  const [message, setMessage] = useState<CalibrationMessage | null>(null)
  const [status, setStatus] = useState<CalibrationStatus>('idle')
  const wsRef = useRef<WebSocket | null>(null)

  const start = useCallback((participant?: string) => {
    if (wsRef.current) return
    setMessage(null)
    setStatus('connecting')
    const ws = new WebSocket(`${BACKEND_WS}/ws/calibrate`)
    wsRef.current = ws

    ws.onopen = () => {
      setStatus('running')
      ws.send(JSON.stringify({ canvas: canvasSize(), ...(participant ? { participant } : {}) })) // canvas: stamped into the config so typing can tell if the window changed; participant: tags this run in data/calibration_history/
    }
    ws.onclose = () => {
      wsRef.current = null
      setStatus((prev) => (prev === 'done' || prev === 'error' ? prev : 'idle'))
    }
    ws.onerror = () => setStatus('error')
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data) as CalibrationMessage
      setMessage(data)
      if (data.phase === 'done') setStatus('done')
      if (data.phase === 'error') setStatus('error')
    }
  }, [])

  const stop = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ stop: true }))
    }
  }, [])

  useEffect(() => {
    return () => {
      wsRef.current?.close()
    }
  }, [])

  return { message, status, start, stop }
}
