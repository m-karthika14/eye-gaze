import type { CSSProperties } from 'react'
import type { KeyboardLayoutData } from '../types'
import { DwellRing } from './DwellRing'

interface Props {
  layout: KeyboardLayoutData
  hoveredKey: string | null
  selectedKey: string | null
  dwellProgress: number
  keyProbabilities: Record<string, number> | null
}

export function VirtualKeyboard({ layout, hoveredKey, selectedKey, dwellProgress, keyProbabilities }: Props) {
  return (
    <div className="relative h-full w-full">
      {layout.keys.map((key) => {
        const isSelected = key.id === selectedKey
        const isHovered = key.id === hoveredKey
        const prob = keyProbabilities?.[key.id] ?? 0
        // Scaled up since mass is usually concentrated on a handful of keys
        // out of 28 -- a "fair" 1/28 share would otherwise look invisible.
        const heatAlpha = Math.min(prob * 6, 0.85)

        const style: CSSProperties = {
          left: `${key.x * 100}%`,
          top: `${key.y * 100}%`,
          width: `${key.width * 100}%`,
          height: `${key.height * 100}%`,
        }
        if (!isSelected && !isHovered && heatAlpha > 0.02) {
          style.backgroundColor = `rgba(168, 85, 247, ${heatAlpha})`
        }

        return (
          <div
            key={key.id}
            className={`absolute flex select-none items-center justify-center border border-slate-700 text-sm font-medium transition-colors duration-150 ${
              isSelected
                ? 'bg-emerald-400 text-slate-900'
                : isHovered
                  ? 'bg-sky-600 text-white'
                  : 'bg-slate-800 text-slate-200'
            }`}
            style={style}
          >
            {key.label}
            {isHovered && <DwellRing progress={dwellProgress} />}
          </div>
        )
      })}
    </div>
  )
}
