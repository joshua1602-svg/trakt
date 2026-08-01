# 06 — Narrow-width layout, page states and breach archetypes

## Mobile / narrow width (< md)

The three-state table folds into stacked cards; the state comparison becomes
a vertical list inside each card, funded first.

```
┌─ Regional exposure — South West ──────────────┐
│ Geography · ≤ 20.0%          [rose] EXPECTED  │
│                                     BREACH    │
│ Funded (contractual)   18.4%  [✓ Pass] 1.6pp  │
│ Expected Forecast      20.7%  [✕ Breach]      │
│                        over by 0.7pp          │
│ Full Pipeline (stress) 23.6%  [✕] over 3.6pp  │
│ Move F→E +2.3pp · Horizon 2026-02             │
│ [Detail]                                      │
└───────────────────────────────────────────────┘
```

* Summary tiles wrap 2-per-row; Emerging risks list is unchanged (text-first
  already).
* Controls collapse to a single row with wrapping; quick filters become
  toggle chips.
* The detail panel becomes full-width; state cards stack vertically.

## Page states

| State | Rendering |
|---|---|
| Loading | existing "Loading concentration tests…" card |
| Error (service unreachable) | existing amber note — "nothing is estimated in the meantime" |
| No active tests | existing empty card + open-proposal count |
| Legacy source | existing amber unapproved banner; three-state columns are hidden entirely (legacy limits have no governed forecast semantics) |
| Funded data unavailable | all three states unavailable; tiles show – |
| **Pipeline/forecast unavailable** | Funded column populated as today; Expected + Full Pipeline columns show "–" with one page-level note: "Expected Forecast is unavailable: <service reason> — funded results are unaffected." Tiles for expected/stress show –. |
| **Forecast weak (insufficient history)** | values render, plus a per-page neutral note and a LIMITED chip on affected rows: "Stage rates fall back to configured assumptions (observed sample below the sufficiency floor)." |
| Scope excludes pipeline | same as pipeline-unavailable with the governed scope reason |

## Breach archetypes (what each must look like)

1. **Funded breach** — Funded cell `[✕ Breach]` rose, risk chip `BREACH`,
   emerging-risk rank 1. Expected/full still shown. Wording: "in breach",
   never "expected".
2. **Expected breach** (funded pass) — Funded `[✓ Pass]`, Expected
   `[✕ Breach]`, risk chip `EXPECTED BREACH`, horizon shown when known.
   Wording: "expected to breach", never "in breach".
3. **Full-pipeline-only breach** — Funded and Expected pass/warn, Full
   Pipeline `[✕]` with muted styling, risk chip `STRESS ONLY`. Wording:
   "breaches only if all pipeline converts — a maximum-exposure scenario,
   not an expected outcome."

The three archetypes must be distinguishable with colour vision removed:
chip text (BREACH / EXPECTED BREACH / STRESS ONLY), badge glyphs, and the
column position of the ✕ carry the distinction.
