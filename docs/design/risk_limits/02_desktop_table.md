# 02 — Desktop three-state table

```
┌─ Controls ──────────────────────────────────────────────────────────────────────────┐
│ [🔎 Search tests] [Category ▾] [Status ▾] [Sort: expected risk ▾]                   │
│ [◻ Expected breaches only] [◻ Stress breaches only]                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─ Concentration tests (approved configuration v2) ───────────────────────────────────┐
│ TEST              LIMIT    FUNDED        EXPECTED       FULL          MOVE   RISK   │
│                            (contractual) FORECAST       PIPELINE      F→E           │
│                                          (prediction)   (stress)                    │
│─────────────────────────────────────────────────────────────────────────────────────│
│ Regional exposure  ≤20.0%  18.4% [✓Pass] 20.7% [✕Br.]  23.6% [✕]    +2.3pp  [rose] │
│ — South West                hd 1.6pp      over 0.7pp     over 3.6pp          EXPECTED│
│                                                                              BREACH │
│─────────────────────────────────────────────────────────────────────────────────────│
│ Regional exposure  ≤50.0%  47.9% [!Warn] 49.2% [!Warn] 52.1% [✕]    +1.3pp  [amber]│
│ — London + S.East           hd 2.1pp      hd 0.8pp       over 2.1pp          LOW HD │
│─────────────────────────────────────────────────────────────────────────────────────│
│ High-value prop.   ≤10.0%   9.1% [!Warn]  9.6% [!Warn] 11.4% [✕]    +0.5pp  [muted]│
│ — above 1,500,000           hd 0.9pp      hd 0.4pp       over 1.4pp          STRESS │
│                                                                              ONLY   │
│─────────────────────────────────────────────────────────────────────────────────────│
│ Net WAC            ≥3.75%   4.12% [✓]     4.08% [✓]     4.01% [✓]   −0.04pp [mint] │
│                             hd 0.37pp     hd 0.33pp      hd 0.26pp           OK     │
│─────────────────────────────────────────────────────────────────────────────────────│
│ Borrower max loans ≤5       3 [✓Pass]    — indicative   — see note   —      [neut.]│
│                                           (funded only)                      LIMITED│
│─────────────────────────────────────────────────────────────────────────────────────│
│ HPI ratio          ≥90%     — [–Unav.]   state-independent            —     [neut.]│
│                             no index source configured                       UNAVAIL│
└─────────────────────────────────────────────────────────────────────────────────────┘
```

Column semantics

| Column | Content | Notes |
|---|---|---|
| Test | display name + category sublabel | truncates with title tooltip |
| Limit | operator glyph + threshold | tooltip: warning fraction |
| Funded | value + status badge + headroom/over | the contractual anchor; always populated first |
| Expected Forecast | value + badge + expected headroom or "over X" | sublabel "(prediction)" in the header only |
| Full Pipeline | value + badge + headroom/over | muted header, permanent "(stress)" sublabel; badge uses outline style so a stress ✕ reads weaker than a funded/expected ✕ |
| Move F→E | funded→expected movement, signed, pp for percent units | amber text when adverse |
| Risk | the service's risk classification chip (EXPECTED BREACH / LOW HD / STRESS ONLY / DETERIORATION / LIMITED / OK / BREACH) | drives default sort |

Interaction

* Row click → detail panel (03). Keyboard: rows are buttons, Enter opens.
* Sort options: expected risk (default — rank order then expected headroom
  ascending), expected headroom, deterioration, funded headroom, name.
* "Expected breaches only" / "Stress breaches only" are additive quick
  filters over the status filter.
* Metric families where the expected state is *indicative only* render an em
  dash plus the treatment note instead of a number — a fractional loan count
  is never displayed.
* Every value cell keeps the non-colour cue inside the badge (✓ ! ✕ – …) and
  the status word; colour is never the only signal.
```
