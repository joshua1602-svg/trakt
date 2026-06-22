/**
 * Population-aware numeric axis domains for loan-level (scatter / bubble) charts.
 *
 * Bar charts stay zero-based, but a `balance by ltv by age` bubble chart should
 * frame the *population*: if borrowers start at 55, the age axis should not waste
 * half its width showing 0–55. We compute a tight domain from the data plus a
 * little padding, and snap percent axes to readable 10-point ticks (20%, 30%, …).
 */

import { toPercentPoints, type PercentScale } from "./utils";

export interface AxisDomain {
  /** [min, max] in DATA units (recharts evaluates the domain against raw values). */
  domain: [number, number];
  /** Optional explicit tick positions in data units. */
  ticks?: number[];
}

function numericValues(rows: Array<Record<string, unknown>>, key: string): number[] {
  const out: number[] = [];
  for (const r of rows) {
    const v = Number(r[key]);
    if (Number.isFinite(v)) out.push(v);
  }
  return out;
}

/** "Nice" rounded step for a given span and target tick count. */
function niceStep(span: number, target = 5): number {
  if (span <= 0) return 1;
  const raw = span / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const snapped = norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1;
  return snapped * mag;
}

/**
 * A padded, population-framed domain for a continuous axis.
 *
 * - Does NOT force zero — the domain hugs the data with `padFraction` headroom.
 * - For percent axes (`isPercent`), values are read in display points (honouring
 *   the storage scale) and the domain is snapped to whole 10-point ticks so the
 *   labels read 20%, 30%, 40%, … Returns `null` when there is nothing to frame.
 */
export function paddedDomain(
  rows: Array<Record<string, unknown>>,
  key: string | undefined,
  opts: { isPercent?: boolean; scale?: PercentScale; padFraction?: number } = {},
): AxisDomain | null {
  if (!key) return null;
  const values = numericValues(rows, key);
  if (values.length === 0) return null;

  const { isPercent = false, scale, padFraction = 0.08 } = opts;
  // Work in display units so percent snapping is intuitive, then convert back.
  const toDisplay = (v: number) => (isPercent ? toPercentPoints(v, scale) : v);
  const fromDisplay = (v: number) => (isPercent && scale === "percent_fraction" ? v / 100 : v);

  let lo = Math.min(...values.map(toDisplay));
  let hi = Math.max(...values.map(toDisplay));

  if (lo === hi) {
    // Degenerate single-value population: open a small symmetric window.
    const bump = isPercent ? 5 : Math.max(Math.abs(lo) * 0.1, 1);
    lo -= bump;
    hi += bump;
  }

  if (isPercent) {
    const step = 10; // readable 10-point ticks (20%, 30%, …)
    let dLo = Math.floor((lo - 1) / step) * step;
    let dHi = Math.ceil((hi + 1) / step) * step;
    dLo = Math.max(0, dLo); // a percentage axis never goes below zero
    const ticks: number[] = [];
    for (let t = dLo; t <= dHi + 1e-9; t += step) ticks.push(fromDisplay(t));
    return { domain: [fromDisplay(dLo), fromDisplay(dHi)], ticks };
  }

  const span = hi - lo;
  const pad = Math.max(span * padFraction, 1);
  const step = niceStep(span + 2 * pad);
  const dLo = Math.floor((lo - pad) / step) * step;
  const dHi = Math.ceil((hi + pad) / step) * step;
  return { domain: [fromDisplay(dLo), fromDisplay(dHi)] };
}
