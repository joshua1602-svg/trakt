import { useMemo, useState } from "react";
import type { ItlAtlas } from "@/domain/geo";
import { cn } from "@/lib/utils";

/** Sequential heat ramp (cool low → warm high), interpolated in RGB. */
const STOPS = [
  [26, 35, 64], [47, 74, 148], [79, 116, 214], [55, 185, 166], [232, 177, 60],
] as const;

function heat(t: number): string {
  const c = Math.max(0, Math.min(1, Math.pow(t, 0.7)));
  const s = c * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(s));
  const f = s - i;
  const [a, b] = [STOPS[i], STOPS[i + 1]];
  const v = (k: number) => Math.round(a[k] + (b[k] - a[k]) * f);
  return `rgb(${v(0)},${v(1)},${v(2)})`;
}

export interface ChoroHover {
  code: string;
  name: string;
  value: number;
  sharePct: number | null;
}

/**
 * UK ITL3 choropleth. Renders the bundled boundary atlas as one SVG path per
 * area, shaded by `valueByCode` (heat scale). Areas with no value render inert.
 * Presentational only — the caller supplies the atlas, values and formatting.
 */
export function UkChoropleth({
  atlas, valueByCode, shareByCode, formatValue, onHoverChange, className,
}: {
  atlas: ItlAtlas;
  valueByCode: Record<string, number>;
  shareByCode?: Record<string, number | null>;
  formatValue: (v: number) => string;
  onHoverChange?: (h: ChoroHover | null) => void;
  className?: string;
}) {
  const [hover, setHover] = useState<ChoroHover | null>(null);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);

  const max = useMemo(
    () => Math.max(0, ...Object.values(valueByCode)),
    [valueByCode],
  );
  const codes = useMemo(() => Object.keys(atlas.areas), [atlas]);

  function enter(code: string) {
    const value = valueByCode[code] ?? 0;
    const h: ChoroHover = {
      code, name: atlas.areas[code]?.name ?? code, value,
      sharePct: shareByCode?.[code] ?? null,
    };
    setHover(h);
    onHoverChange?.(h);
  }
  function leave() {
    setHover(null);
    onHoverChange?.(null);
  }

  return (
    <div className={cn("relative mx-auto w-full", className)} data-testid="uk-choropleth">
      <svg viewBox={atlas.viewBox} preserveAspectRatio="xMidYMid meet"
        className="block h-auto w-full" role="img" aria-label="UK exposure by ITL3 area"
        onMouseLeave={leave}>
        {codes.map((code) => {
          const value = valueByCode[code] ?? 0;
          return (
            <path
              key={code}
              d={atlas.areas[code].d}
              data-code={code}
              fill={value > 0 && max > 0 ? heat(value / max) : "var(--color-navy-800, #141b2b)"}
              stroke="#0b0f18"
              strokeWidth={0.6}
              className="cursor-pointer transition-[filter] duration-100 hover:brightness-125 [stroke:#0b0f18] hover:[stroke:#fff] hover:[stroke-width:1.1]"
              onMouseEnter={() => enter(code)}
              onMouseMove={(e) => setPos({ x: e.clientX, y: e.clientY })}
            />
          );
        })}
      </svg>
      {hover && pos && (
        <div
          className="pointer-events-none fixed z-20 rounded-lg border border-[var(--color-line)] bg-navy-950/95 px-2.5 py-1.5 text-[11px] shadow-xl"
          style={{ left: pos.x + 14, top: pos.y + 14 }}
          data-testid="choropleth-tip"
        >
          <div className="font-semibold text-ink-100">{hover.name}</div>
          <div className="text-ink-300">
            <span className="font-mono text-amber-300">{formatValue(hover.value)}</span>
            {hover.sharePct != null && <span> · {hover.sharePct.toFixed(1)}%</span>}
            <span className="text-ink-500"> · {hover.code}</span>
          </div>
        </div>
      )}
    </div>
  );
}
