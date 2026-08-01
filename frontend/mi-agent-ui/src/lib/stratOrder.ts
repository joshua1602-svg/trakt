/**
 * Ordering + label hygiene for stratification bar lists.
 *
 * Backend stratifications arrive ranked by balance. For ordinal dimensions
 * (LTV band, age band, rate band, vintage year, term buckets…) that ranking is
 * unintuitive — the buckets should read in their natural order. This module
 * sorts buckets by the numeric bound parsed from their labels when the
 * dimension is ordinal, and alphabetically otherwise; "Unknown"-style buckets
 * always sink to the end. Values are never altered — display order and label
 * tidy-ups only.
 */

/** Buckets that mean "no data" — always sorted last, never parsed. */
const UNKNOWN_RE = /^(unknown|not supplied|not available|none|n\/?a|other|-)$/i;

/** Tidy a bucket label for display: "2008.0" → "2008" (float-cast years). */
export function cleanBucketLabel(label: string): string {
  const s = String(label).trim();
  // A pure number with a redundant ".0" fraction (e.g. vintage years).
  const m = s.match(/^(\d{1,4})\.0$/);
  return m ? m[1] : s;
}

/**
 * Parse the ordinal sort bound from a bucket label, or null when the label has
 * no leading numeric structure. Handles the common shapes:
 *   "40-50%" / "40–50%"   → 40
 *   "<20%" / "<=20"       → 19.5   (just below its boundary)
 *   ">=100%" / "100%+"    → 100
 *   "85+"                 → 85
 *   "2008" / "2008.0"     → 2008
 *   "£100K-£150K"         → 100
 */
export function bucketBound(label: string): number | null {
  const s = cleanBucketLabel(label);
  if (!s || UNKNOWN_RE.test(s)) return null;
  const belowBoundary = /^\s*[<≤]/.test(s);
  const m = s.replace(/[£$€,\s]/g, "").match(/-?\d+(\.\d+)?/);
  if (!m) return null;
  const n = Number(m[0]);
  if (!Number.isFinite(n)) return null;
  return belowBoundary ? n - 0.5 : n;
}

/**
 * Order stratification buckets for display: ordinal dimensions by their
 * parsed bound ascending, otherwise alphabetically; unknown buckets last.
 * A dimension counts as ordinal when at least half its real (non-unknown)
 * labels parse to a numeric bound. Input is not mutated.
 */
export function sortStratBars<T extends { label: string }>(bars: T[]): T[] {
  const entries = bars.map((bar) => {
    const label = cleanBucketLabel(bar.label);
    const unknown = UNKNOWN_RE.test(label) || !label;
    return { bar, label, unknown, bound: unknown ? null : bucketBound(label) };
  });
  const real = entries.filter((e) => !e.unknown);
  const numeric = real.filter((e) => e.bound !== null).length;
  const ordinal = real.length > 0 && numeric * 2 >= real.length;

  return [...entries]
    .sort((a, b) => {
      // Unknown-style buckets sink to the end in either mode.
      if (a.unknown !== b.unknown) return a.unknown ? 1 : -1;
      if (ordinal) {
        const ab = a.bound ?? Number.POSITIVE_INFINITY;
        const bb = b.bound ?? Number.POSITIVE_INFINITY;
        if (ab !== bb) return ab - bb;
      }
      return a.label.localeCompare(b.label, "en-GB", { sensitivity: "base", numeric: true });
    })
    .map((e) => e.bar);
}
