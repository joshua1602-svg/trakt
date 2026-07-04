import { useMemo } from "react";
import type { ChartArtifact } from "@/domain";
import { cn, formatUiTitle, formatValue } from "@/lib/utils";

/**
 * Native heatmap — a lightweight CSS-grid component (no charting dependency),
 * themed to the Trakt dashboard. Renders a two-dimension intensity matrix from
 * the result rows (`xKey` × `yKey` → `valueKey`).
 *
 * Recharts has no heatmap primitive; a custom grid renders faithfully, matches
 * the dark theme exactly, and avoids pulling in Plotly for a common MI chart.
 */

// Dark sequential ramp: faint navy (low) → periwinkle → mint (high).
const STOPS: Array<[number, [number, number, number]]> = [
  [0, [27, 34, 64]], // navy-800
  [0.55, [145, 157, 209]], // periwinkle
  [1, [54, 194, 168]], // mint
];

function rgbAt(t: number): string {
  const x = Math.max(0, Math.min(1, t));
  for (let i = 1; i < STOPS.length; i++) {
    const [p0, c0] = STOPS[i - 1];
    const [p1, c1] = STOPS[i];
    if (x <= p1) {
      const f = (x - p0) / (p1 - p0 || 1);
      const c = [0, 1, 2].map((k) => Math.round(c0[k] + (c1[k] - c0[k]) * f));
      return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
    }
  }
  const last = STOPS[STOPS.length - 1][1];
  return `rgb(${last[0]}, ${last[1]}, ${last[2]})`;
}

/**
 * Numeric sort key for a bucket-band label ("20-30%", "<40", "80+", "£100k-200k")
 * so a bucket axis reads in natural band order, not the order rows happened to
 * arrive in (the executor ranks rows by value). Returns null for a non-numeric
 * category (broker names, borrower type, …), which keeps insertion order.
 */
export function bucketSortKey(label: string): number | null {
  const s = label.replace(/[£$,%\s]/g, "");
  const m = s.match(/-?\d+(?:\.\d+)?/);
  if (!m) return null;
  let key = parseFloat(m[0]);
  const mult = /\dk\b/i.test(s) || /k[-+>]?/i.test(s) ? 1e3 : /\dm\b/i.test(s) ? 1e6 : 1;
  key *= mult;
  if (/^</.test(s)) key -= 0.5; // "<40" sits just below the "40-…" band
  return key;
}

/** Order axis values by bucket band when numeric; else preserve insertion order.
 * Non-numeric buckets (e.g. "Unknown / Missing") sort last, stably. */
export function orderAxis(vals: string[]): string[] {
  const keyed = vals.map((v, i) => ({ v, i, k: bucketSortKey(v) }));
  if (!keyed.some((x) => x.k !== null)) return vals;
  return [...keyed]
    .sort((a, b) => {
      const ka = a.k ?? Infinity;
      const kb = b.k ?? Infinity;
      return ka !== kb ? ka - kb : a.i - b.i;
    })
    .map((x) => x.v);
}

export function HeatmapArtifactView({ artifact }: { artifact: ChartArtifact }) {
  const { xKey, yKey, valueKey, rows, valueFormat } = artifact;

  const model = useMemo(() => {
    if (!xKey || !yKey || !valueKey) return null;
    const xsSeen: string[] = [];
    const ysSeen: string[] = [];
    const cells = new Map<string, number>();
    let min = Infinity;
    let max = -Infinity;
    for (const r of rows) {
      const x = String(r[xKey]);
      const y = String(r[yKey]);
      const v = Number(r[valueKey]) || 0;
      if (!xsSeen.includes(x)) xsSeen.push(x);
      if (!ysSeen.includes(y)) ysSeen.push(y);
      cells.set(`${x}||${y}`, v);
      if (v < min) min = v;
      if (v > max) max = v;
    }
    // Order both axes in natural band order (a bucket axis reads 20-30, 30-40,
    // 40-50, …, not by cell count); non-bucket axes keep insertion order.
    return { xs: orderAxis(xsSeen), ys: orderAxis(ysSeen), cells, min, max };
  }, [xKey, yKey, valueKey, rows]);

  if (!model) {
    return (
      <div className="rounded-lg border border-[var(--color-line-soft)] bg-navy-850/40 p-4 text-sm text-ink-400">
        Heatmap data is incomplete (needs two dimensions and a measure).
      </div>
    );
  }

  const { xs, ys, cells, min, max } = model;
  const span = max - min || 1;

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="border-separate" style={{ borderSpacing: 2 }}>
          <thead>
            <tr>
              <th className="px-2 py-1 text-left text-[10px] font-semibold uppercase tracking-wider text-ink-500">
                {formatUiTitle(yKey)} \ {formatUiTitle(xKey)}
              </th>
              {xs.map((x) => (
                <th key={x} className="px-1.5 py-1 text-center text-[10px] font-medium text-ink-400" style={{ maxWidth: 90 }}>
                  <span className="block truncate" title={x}>{x}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ys.map((y) => (
              <tr key={y}>
                <td className="whitespace-nowrap px-2 py-1 text-right text-[11px] font-medium text-ink-300" title={y}>
                  {y}
                </td>
                {xs.map((x) => {
                  const v = cells.get(`${x}||${y}`);
                  const has = v !== undefined;
                  const t = has ? (v - min) / span : 0;
                  const bg = has ? rgbAt(t) : "rgba(255,255,255,0.02)";
                  return (
                    <td key={x} className="p-0">
                      <div
                        className={cn(
                          "flex h-10 min-w-[56px] items-center justify-center rounded font-mono text-[10px]",
                          has ? (t > 0.45 ? "text-navy-950" : "text-ink-100") : "text-ink-600",
                        )}
                        style={{ background: bg }}
                        title={has ? `${y} · ${x}: ${formatValue(v, valueFormat)}` : undefined}
                      >
                        {has ? formatValue(v, valueFormat) : "·"}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Colour legend */}
      <div className="mt-3 flex items-center gap-2 text-[10px] text-ink-500">
        <span>{formatValue(min, valueFormat)}</span>
        <span
          className="h-2 w-32 rounded-full"
          style={{ background: `linear-gradient(90deg, ${rgbAt(0)}, ${rgbAt(0.55)}, ${rgbAt(1)})` }}
        />
        <span>{formatValue(max, valueFormat)}</span>
        <span className="ml-1">{formatUiTitle(valueKey)}</span>
      </div>
    </div>
  );
}
