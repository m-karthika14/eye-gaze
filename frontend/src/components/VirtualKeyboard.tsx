import type { KeyboardLayoutData } from '../types'
import { DwellRing } from './DwellRing'

interface Props {
  layout: KeyboardLayoutData
  hoveredKey: string | null
  selectedKey: string | null
  dwellProgress: number
}

export function VirtualKeyboard({ layout, hoveredKey, selectedKey, dwellProgress }: Props) {
  return (
    <div className="relative h-full w-full">
      {layout.keys.map((key) => {
        const isSelected = key.id === selectedKey
        const isHovered = key.id === hoveredKey
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
            style={{
              left: `${key.x * 100}%`,
              top: `${key.y * 100}%`,
              width: `${key.width * 100}%`,
              height: `${key.height * 100}%`,
            }}
          >
            {key.label}
            {isHovered && <DwellRing progress={dwellProgress} />}
          </div>
        )
      })}
    </div>
  )
}
