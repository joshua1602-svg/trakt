# 04 — Pipeline-driver drill-through

Expanded from the detail panel ("Pipeline drivers" accordion). Shows the
cases contributing most to the expected movement of the selected test.

```
┌─ Pipeline drivers — Regional exposure — South West ─────────────────────────────────┐
│ Expected numerator movement +5.5m from 9 pipeline case(s); the top 4 drive 82%.     │
│ Contributions reconcile to the Expected Forecast numerator exactly.                 │
│                                                                                     │
│ CASE      BALANCE   STAGE       REGION      PROB.  EXPECTED   FULL      EXP.  IMPACT│
│                                                    CONTRIB.   CONTRIB.  MONTH       │
│──────────────────────────────────────────────────────────────────────────────────── │
│ P-1042    640,000   OFFER       South West  0.86   550,400    640,000   2026-01  ✕ │
│                                                                       tips breach   │
│ P-0991    510,000   OFFER       South West  0.86   438,600    510,000   2026-02     │
│ P-1077    455,000   APPLICATION South West  0.61   277,550    455,000   2026-02     │
│ P-1103    390,000   APPLICATION South West  0.61   237,900    390,000   2026-03     │
│ P-0968    120,000   KFI         South West  0.24    28,800    120,000   2026-03     │
│ … 4 more (smaller contributions)                                   [show all]       │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

Columns

| Column | Source | Notes |
|---|---|---|
| Case | pipeline case identifier | existing access rules; no borrower PII beyond the pipeline extract |
| Balance | advance amount | full, unweighted |
| Stage | canonical stage | KFI / Application / Offer |
| Region (or the test's dimension) | the dimension column the test filters on | header adapts per test |
| Prob. | completion probability | tooltip: source (historical stage rate / configured / row-level) |
| Expected contrib. | balance × probability | Σ over numerator cases = expected numerator − funded numerator |
| Full contrib. | balance | the stress contribution |
| Exp. month | expected completion period | "—" when unavailable |
| Impact | "tips breach" / "tips warning" marker | deterministic: the first case (in expected-contribution order) whose cumulative addition crosses the line |

Interaction & rules

* Sorted by expected contribution descending; ranking is the service's, the
  UI never re-ranks.
* Excluded pipeline states (withdrawn / cancelled / unknown / completed) can
  never appear — excluded upstream, not filtered in the browser.
* The reconciliation sentence is rendered from the service's own
  reconciliation flag; if it does not reconcile the panel says so in amber
  rather than hiding the discrepancy.
* Empty state: "No pipeline case contributes to this test's numerator — the
  expected movement is denominator-only (pipeline exposure outside this
  segment dilutes the share)." — this genuinely occurs for share metrics.
```
