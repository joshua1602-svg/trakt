# 03 — Test detail panel

Opens below the table on row selection (existing pattern), extended with a
state-comparison strip and two accordions (Drivers, Methodology).

```
┌─ Regional exposure — South West                     [✕ Breach expected]  [× close] ─┐
│ Geography · high severity · effective 2026-01-01                                    │
│                                                                                     │
│ ┌─ FUNDED (contractual) ─┬─ EXPECTED FORECAST ──────┬─ FULL PIPELINE (stress) ────┐ │
│ │ 18.4%      [✓ Pass]    │ 20.7%       [✕ Breach]   │ 23.6%        [✕ Breach]     │ │
│ │ headroom 1.6pp         │ over by 0.7pp            │ over by 3.6pp               │ │
│ │ util. 92.0%            │ util. 103.5%             │ util. 118.0%                │ │
│ │ N 33.1m / D 180.0m     │ N 38.6m / D 186.6m       │ N 44.9m / D 190.4m          │ │
│ └────────────────────────┴──────────────────────────┴─────────────────────────────┘ │
│  Expected breach horizon: 2026-02 (cumulative expected completions cross the        │
│  20.0% limit in February). Status transition: Pass → expected Breach.               │
│                                                                                     │
│ Approved definition   Balance-weighted share of the portfolio secured on            │
│                       properties in [South West]. Denominator: current balance.     │
│ Forecast semantics    Expected state weights each pipeline case by its stage        │
│                       completion probability; numerator and denominator move        │
│                       together. Full Pipeline includes all active cases at 100%.    │
│ Limit                 ≤ 20.0% · warning at 90% of limit                             │
│ Funded trend          [existing line chart: funded history + limit + warning line]  │
│                       (trend is funded-only — forecasts are never charted as        │
│                        history)                                                     │
│ Source wording        "…located in the South West must not exceed 20% of the        │
│                       Portfolio." (clause 1.4, facility_schedule_2026.txt)          │
│ Approval              A. Operator on 2026-01-10 · configuration v2                  │
│                                                                                     │
│ ▸ Pipeline drivers (4 loans drive 82% of the expected increase)      [04]           │
│ ▸ Forecast methodology (window 2025-10-27 → 2025-11-24 · mixed basis) [05]          │
│ [Show contributing loans]  (existing funded drill-through, unchanged)               │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

Notes

* The three state cards share one fixed internal layout (value, badge,
  headroom/over, utilization, N/D) so the eye compares vertically.
* The panel's header badge shows the *most severe governed state*, prefixed to
  keep prediction distinct: `Breach` (funded), `Breach expected`, or
  `Stress-only breach`.
* Horizon line renders only when the service supplies one; otherwise
  "Expected breach horizon: not determinable (no completion-timing data)".
* Indicative-only families replace the Expected card body with the treatment
  note ("Expected state is indicative only for maximum-loan tests; the
  pipeline-at-risk maximum appears under Full Pipeline").
* Numerators/denominators are the service's exact figures — the panel never
  derives one from the other.
```
