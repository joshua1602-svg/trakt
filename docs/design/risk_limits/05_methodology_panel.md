# 05 — Forecast methodology disclosure

One accordion in the detail panel plus the same content behind the header's
[View methodology] link (page-level dialog). Content is rendered verbatim
from the service's methodology block — the UI adds no interpretation.

```
┌─ How the Expected Forecast is calculated ───────────────────────────────────────────┐
│ Model          Deterministic completion-trend model over the client's weekly        │
│                pipeline extracts. No machine learning; no invented probabilities.   │
│ Observation    2025-10-27 → 2025-11-24 · 5 weekly extracts · 214 cases tracked ·    │
│ window         41 observed completions                                              │
│ Stage rates    KFI 0.24 (observed 88 · sufficient) · Application 0.61 (observed     │
│                64 · sufficient) · Offer 0.86 (observed 41 · sufficient)             │
│                A stage needs ≥ 12 observed cases before its empirical rate is       │
│                trusted; otherwise the configured stage assumption applies.          │
│ Basis          mixed: historical stage rates + configured fallback (2 cases)        │
│ Timing         Median observed days to completion: Offer 18d · Application 41d ·    │
│                KFI 66d. Expected completion months derive from explicit dates       │
│                where supplied, else stage timing offsets.                           │
│ Exclusions     Withdrawn / cancelled / declined / lapsed: never counted or          │
│                weighted (7 cases). Unknown stage: no probability (2 cases,          │
│                disclosed, excluded from weighting).                                 │
│ Full Pipeline  Ignores probabilities entirely: every active in-scope case at        │
│                100%. A maximum-exposure stress, not a prediction.                   │
│ Limitations    Recent cohorts may not yet have had time to complete (rates are      │
│                conservative). Rates are portfolio-wide per stage, not segmented     │
│                by product or channel. Historical expected-state comparisons are     │
│                unavailable until point-in-time snapshots are persisted.             │
│ Sources        pipeline_20251124.csv (current snapshot) · 5 weekly extracts         │
│                (deduplicated)                                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

Notes

* Every line maps 1:1 to a field of the governed methodology payload
  (`historicalModelEvidence` + stage-rate table + exclusions + window).
* "Why five weeks?" is answered concretely: the window line shows the actual
  extract dates and counts rather than asserting a policy; the sufficiency
  floor sentence explains when the window is *not* enough.
* MI Query's "How was the expected forecast calculated?" answers with the
  same payload — one source of truth, two renderings.
```
