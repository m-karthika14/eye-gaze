import { useEffect, useState } from 'react'
import { BACKEND_HTTP } from '../config'
import type { KeyboardLayoutData } from '../types'

export function useKeyboardLayout() {
  const [layout, setLayout] = useState<KeyboardLayoutData | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch(`${BACKEND_HTTP}/config/keyboard-layout`)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load keyboard layout: ${res.status}`)
        return res.json()
      })
      .then((data: KeyboardLayoutData) => setLayout(data))
      .catch((err: unknown) => setError(String(err)))
  }, [])

  return { layout, error }
}
