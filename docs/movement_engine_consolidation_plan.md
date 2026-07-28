# Consolidating the three movement engines

A design note. **No code changes are proposed by this document** — it sets out
what would change, what it would cost, and in what order, so the decision can be
taken deliberately rather than discovered later through inconsistent answers.

Status: **proposed, not scheduled.**

---

## 1. The problem

Three routes answer "what changed?", and they do not agree.

| | `period_movement` (priority 70) | `temporal_compare` (90) | `period_change_analysis` (85) |
|---|---|---|---|
| Metric set | hard-coded 5 | exactly 1, via a hard-coded key map | N, from BSR metadata |
| Periods | latest vs prior month only | two named periods | 6 governed resolution modes |
| Weighted average | `snapshots._weighted_average` | `evolution._weighted_avg` | `calculations._aggregate_weighted` |
| Zero total weight | **falls back to a simple mean** | returns `None` | `zero_denominator`, no value |
| Thresholds | `LTV_STABILITY_POINTS = 0.6`, `AGE_MATERIALITY_YEARS = 0.05` | none | none, by design |
| Causation | asserts "primarily driven by completions" behind an evidence test | none | none, by design |
| Audit | none | per-period reconciliation | full selection + calculation audit |

This is not merely duplication. Three concrete divergences can produce different
numbers or different claims for the same book:

1. **Weighted averages disagree on a zero-weight population.** A book whose
   balances are all zero at a reporting date yields a simple mean from the
   landing-page KPI, `None` from the Evolution tab, and an explicit
   `zero_denominator` from period change. Two of those are answers; one is a
   refusal.
2. **`period_movement` embeds materiality.** A 0.5-point LTV move is described
   as "broadly stable" because of a constant in `movement_summary.py`. The same
   move is reported as a ranked movement with an explicit "no governed
   materiality threshold is configured" caveat by period change. A reader
   comparing the two surfaces sees a contradiction.
3. **`period_movement` asserts causation.** It says balances moved "primarily
   driven by completions" when a completions test passes. Period change never
   attributes a cause outside the balance bridge. Both may be defensible; having
   both is not.

## 2. Target state

`period_change_analysis` becomes the single movement engine. The other two become
**request shapes** over it — different selections and presentations of the same
governed calculation, not separate arithmetic.

```
  "what changed vs the prior month"   →  requested-metric-set request
  "compare October and November X"    →  single-metric request, named periods
  "what changed this month"           →  overview request
                    │
                    └──────────►  period_change_analysis  ──►  one result contract
                                                               ──► three presenters
```

Each keeps its own presenter, so the answers users already recognise still look
the way they look. What changes is that the numbers behind them come from one
place.

## 3. Migration in four steps

Each step is independently shippable and independently revertible.

### Step 1 — a named metric-set mode (no user-visible change)

Add a fourth workflow mode, `metric_set`, taking an explicit ordered field list.
`period_movement`'s five metrics become a governed set in
`config/period_change_selection.yaml`:

```yaml
metric_sets:
  monthly_movement_headline:
    - current_outstanding_balance
    - current_loan_to_value
    - current_interest_rate
    - youngest_borrower_age
```

Nothing routes to it yet. Ships with unit tests only.

### Step 2 — re-point `temporal_compare`

Its recogniser and presenter stay exactly as they are; `_route_compare` calls
`analyse_period_change` in requested-metric mode with the two resolved periods,
then renders the existing compare payload from the result.

**Expected number changes:** none for a populated book. On a zero-weight
population the route changes from "no value" to an explicit
`zero_denominator` status — a refusal replacing a silent blank, which is an
improvement but is still a change, and the existing tests must be updated to
assert it deliberately.

### Step 3 — re-point `period_movement`

The harder one, because it changes user-visible wording.

* the five metrics come from the `metric_set` mode;
* the regional attribution comes from the distribution analysis;
* the balance movement comes from the balance bridge;
* **the two embedded thresholds are removed.** "Broadly stable" and "unchanged"
  are replaced by the controlled significance vocabulary, or the thresholds are
  promoted to governed configuration and applied by the workflow. Either is
  defensible; silently keeping them in one engine and not the other is not.
* **the completions causality claim is removed or promoted.** Either it becomes
  a governed attribution calculation available to every caller, or the route
  stops asserting it. It cannot remain a claim only one surface makes.

**Expected number changes:** the LTV and age wording changes on every answer.
The figures themselves do not change (the same weighted average over the same
frames), except on a zero-weight population as in step 2.

This step needs product sign-off, not just engineering review.

### Step 4 — retire the duplicate aggregation helpers

Once nothing else calls them, `snapshots._weighted_average` and
`evolution._weighted_avg` collapse into the governed aggregation, removing the
zero-weight divergence at its source. This is the step that actually eliminates
the inconsistency; steps 1–3 only stop new callers appearing.

## 4. What this does not change

* Recogniser precedence. Each route keeps its priority and its question shape.
* The React and Copilot payloads. Presenters are unchanged; only the source of
  the numbers moves.
* The BSR, canonical transformations, onboarding, or the source field registry.

## 5. Risks

| Risk | Mitigation |
|---|---|
| A published figure changes | Step 2 and step 3 each ship with a comparison harness running both engines over the same fixtures and diffing every figure, so any change is enumerated before release, not discovered after |
| Existing route tests are rewritten wholesale | Each step updates only assertions whose behaviour genuinely changed, and the reason is recorded in the test docstring |
| The overview policy starts driving a narrow route | `metric_set` mode bypasses the overview caps entirely — a named set is returned in full |
| Wording regressions on a familiar answer | Step 3 is gated on product sign-off of the new wording, with before/after examples |

## 6. Recommendation

Steps 1, 2 and 4 are engineering work with a bounded, enumerable blast radius,
and step 4 is the one that removes the real inconsistency. Step 3 is a product
decision about materiality wording and causal claims, and should be taken
separately.

Doing nothing is also a defensible position for now — but it should be a
decision, recorded here, rather than the default. The inconsistency is live: the
zero-weight weighted-average divergence exists today, in three implementations,
independently of this workflow.
