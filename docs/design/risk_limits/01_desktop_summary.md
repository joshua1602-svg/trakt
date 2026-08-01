# 01 — Desktop summary view (header, tiles, emerging risks)

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ Risk Limits                                      Reporting date 2025-11-30           │
│ Portfolio: Total (all portfolios)                Prior 2025-10-31 · Evaluated 08:12  │
│                                                                                      │
│ [note] Approved configuration v2 · activated by A. Operator on 2026-01-10            │
│ [note] Expected Forecast: completion-trend model · window 2025-10-27 → 2025-11-24    │
│        (5 weekly extracts, 214 cases, 41 completions) · [View methodology]           │
└──────────────────────────────────────────────────────────────────────────────────────┘

┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐
│ FUNDED     │ EXPECTED   │ FULL       │ EXPECTED   │ DETERIOR-  │ UNAVAIL-   │
│ BREACHES   │ BREACHES   │ PIPELINE   │ WARNINGS   │ ATING      │ ABLE       │
│            │            │ BREACHES   │            │            │            │
│    0 ✓     │    1 !     │    3 ⚠     │    2       │    3       │    1       │
│  (mint)    │  (rose)    │  (muted)   │  (amber)   │  (amber)   │ (neutral)  │
│ contractual│ prediction │ stress —   │            │ vs prior   │ never      │
│ position   │            │ max exposure           │ period     │ shown pass │
└────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘

┌─ Emerging risks ────────────────────────────────────────────────────── ranked ──┐
│ 1 [rose !] EXPECTED BREACH   South West is expected to breach: funded 18.4%     │
│            vs 20.0% limit; expected 20.7% (+0.7pp over). 4 pipeline loans       │
│            drive 82% of the increase.                    [Open test] [Drivers]  │
│ 2 [amber ] LOW EXPECTED HEADROOM  London + South East has 0.8pp of expected     │
│            headroom (47.9% → 49.2% vs ≤ 50%).                       [Open test] │
│ 3 [muted ] STRESS-ONLY BREACH  High-value property breaches only if ALL         │
│            pipeline converts (9.1% → 11.4% vs ≤ 10%). Not an expected outcome.  │
│ 4 [amber ] DETERIORATION  Net WAC headroom fell 0.21pp since 2025-10-31.        │
│ 5 [neutral] FORECAST LIMITED  Arrears concentration: expected state uses the    │
│            funded value only (no arrears data in pipeline).                     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

Notes

* The header carries **configuration** provenance (version, activator) and
  **forecast** provenance (model basis, observation window, sample) as two
  separate `role="note"` lines — never merged, so contractual and statistical
  provenance stay distinct.
* Tiles are grouped left→right in state order Funded → Expected → Full
  Pipeline; the Full Pipeline tile header carries the permanent sublabel
  "stress — max exposure" and uses the muted tone even when non-zero, so red
  is reserved for funded/expected problems.
* Emerging risks are rendered in the deterministic rank order supplied by the
  service (current breach → expected breach → low expected headroom → material
  deterioration → stress-only breach → data/methodology limitation). The UI
  never re-orders or invents entries.
* Each risk line: severity badge with glyph, category label in caps, one
  plain-English sentence with the numbers inline, action links that open the
  detail panel (and pre-select the Drivers tab for driver links).
```
