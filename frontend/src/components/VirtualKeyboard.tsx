import type { CSSProperties } from 'react'
import type { KeyboardLayoutData, KeyGeometry } from '../types'
import { DwellRing } from './DwellRing'

interface Props {
  layout: KeyboardLayoutData
  hoveredKey: string | null
  selectedKey: string | null
  dwellProgress: number
  keyProbabilities: Record<string, number> | null
  keyLayoutOverride: Record<string, KeyGeometry> | null
  spaceLabel: string | null
  // Calibration mode (see CalibrationScreen): highlights one key as the
  // current "look at this" target, with its own progress ring driven by
  // fixation_progress from /ws/calibrate rather than S0's dwell_progress.
  calibrationTarget?: string | null
  calibrationProgress?: number
}

export function VirtualKeyboard({
  layout,
  hoveredKey,
  selectedKey,
  dwellProgress,
  keyProbabilities,
  keyLayoutOverride,
  spaceLabel,
  calibrationTarget = null,
  calibrationProgress = 0,
}: Props) {
  return (
    <div className="relative h-full w-full">
      {layout.keys.map((key) => {
        const isSelected = key.id === selectedKey
        const isHovered = key.id === hoveredKey
        const isCalibrationTarget = key.id === calibrationTarget
        const prob = keyProbabilities?.[key.id] ?? 0
        // Scaled up since mass is usually concentrated on a handful of keys
        // out of 28 -- a "fair" 1/28 share would otherwise look invisible.
        const heatAlpha = Math.min(prob * 6, 0.85)
        const geometry = keyLayoutOverride?.[key.id] ?? key

        const style: CSSProperties = {
          left: `${geometry.x * 100}%`,
          top: `${geometry.y * 100}%`,
          width: `${geometry.width * 100}%`,
          height: `${geometry.height * 100}%`,
        }
        if (!isSelected && !isHovered && !isCalibrationTarget && heatAlpha > 0.02) {
          style.backgroundColor = `rgba(168, 85, 247, ${heatAlpha})`
        }

        const label = key.id === "SPACE" && spaceLabel ? spaceLabel : key.label

        return (
          <div
            key={key.id}
            className={`absolute flex select-none items-center justify-center overflow-hidden border border-slate-700 text-sm font-medium transition-all duration-200 ${
              isCalibrationTarget
                ? 'bg-amber-400 text-slate-900 ring-4 ring-amber-300'
                : isSelected
                  ? 'bg-emerald-400 text-slate-900'
                  : isHovered
                    ? 'bg-sky-600 text-white'
                    : 'bg-slate-800 text-slate-200'
            }`}
            style={style}
          >
            {label}
            {isHovered && <DwellRing progress={dwellProgress} />}
            {isCalibrationTarget && <DwellRing progress={calibrationProgress} />}
          </div>
        )
      })}
    </div>
  )
}
