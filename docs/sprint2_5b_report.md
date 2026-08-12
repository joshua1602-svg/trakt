# Sprint 2.5B — historical and time-series readiness

*Baseline `130f3b0` → `3463a1e`. The agent is **not built**; no forecasting, no
scenarios, no A2A.*

---

## 1. Executive summary

> **Can Trakt now understand how a portfolio has behaved through time?**

**Yes.** Trakt can measure, from governed snapshots and with no model in the
loop: how arrears moved period by period; how loans migrated between states and
which cured; the observed prepayment and redemption rate with a stated
methodology; realised losses and recoveries on every defensible denominator;
how concentration drifted; how LTV moved; and how any cohort compares with the
rest of the book.

The important part is *how* it does it. Roughly **two thirds of the capability
already existed** for React MI and is reused unchanged — transitions, cohorts,
period change, balance bridge, snapshot discovery, and a semantics registry that
already distinguishes flows from stocks. Three calculations genuinely did not
exist and were added to `analytics_lib`, beside `stratify` and `concentration`,
where React and MI Query can call them too.

**What Trakt still does not do, deliberately:** it does not forecast, project,
simulate or stress anything. Every number here is observed.

---

## 2. Existing MI capability inventory

Full trace: `docs/historical_mi_capability_inventory.md`. Filenames were not
trusted; every entry was traced to its calculation function and consumers.

| Historical capability | Existing MI calculation | React use | Agent accessible before | Action taken |
|---|---|---|---|---|
| Status / DPD transitions | `risk_monitor.migration.migration_matrix` | Risk Monitor | ❌ | **Reused** — wrapped by `transition_analysis` |
| Per-loan movement | `risk_monitor.migration.per_loan_movement` | Risk Monitor | ❌ | Reused (available, not exposed — per-loan output is unbounded) |
| Cohort / vintage tables | `analytics_lib.cohort.cohort_table` | MI workflows | ❌ | **Reused** via `cohort_comparison` |
| Two-period metric change | `period_change.calculations.metric_change` | MI Query | ✅ | Reused unchanged |
| Distribution change | `period_change.distribution` | MI Query | ✅ | Reused unchanged |
| Balance bridge | `period_change.bridge.balance_bridge` | MI Query | ✅ | Reused unchanged |
| Snapshot discovery | `mi_agent_api.snapshots` | MI | indirect | **Reused** as the history resolver |
| Metric semantics | `config/business_semantics_registry.yaml` | MI Query | indirect | **Reused** — 13 `period_flow`, 5 `cumulative` fields |
| Concentration / stratification | `analytics_lib.{stratify,concentration}` | MI, agent | ✅ | Reused |
| **Multi-period series** | — | — | ❌ | **NEW** `analytics_lib.history.portfolio_series` |
| **Prepayment / redemption rate** | — | — | ❌ | **NEW** `analytics_lib.history.prepayment_rate` |
| **Loss / recovery rates** | — | — | ❌ | **NEW** `analytics_lib.history.loss_and_recovery` |
| **Cohort vs remainder** | — | — | ❌ | **NEW** `analytics_lib.history.compare_cohorts` |

**Reused: 9 capabilities. Newly built: 4 calculations.**

Three things that look like capabilities and are not — each would have produced
plausible, wrong work:

1. **`analytics_lib/migration.py` is an explicit stub.** `transition_matrix`
   returns `None`. The real one is in `risk_monitor`. Two modules, one name.
2. **"Prepayment Speed (CPR)" in the static-pools config is a chart title.** The
   spec beneath it is `metric: prepayment_amount, agg: sum` — cumulative
   redemptions in £, plotted. **There is no CPR or SMM calculation anywhere in
   the repository.** Treating that as an existing methodology was the easiest
   available mistake in this sprint.
3. **`mi_agent_pptx/cohorts.py` computes nothing** — it adapts a payload.

Two defects found while wiring the reuse: `migration_matrix` defaults to key
`loan_id` while the canonical identifier is `loan_identifier` (silently
returning an empty matrix), and its result attribute is `frame`, not `table`.

---

## 3. Synthetic historical portfolio

`tests/history_portfolio.py` — **18 monthly snapshots**, 2025-02-28 → 2026-07-31,
**400 loans declining to 356**, with the *same loans persisting*. Twelve
unrelated portfolios would prove nothing; behaviour is only observable when a
loan can be followed.

| Cohort | Loans | Planted behaviour |
|---|---|---|
| stable × 3 | 200 | control — no arrears, normal amortisation |
| deteriorating | 40 | 30 → 60 → 90 → 240 DPD, never recovers |
| curing | 30 | delinquent to 90 DPD, cures at period 8, relapses, cures again |
| redeeming | 50 | 44 exit in seven waves |
| prepaying | 30 | four partial unscheduled prepayments of 5% |
| defaulting | 10 | defaults at period 9, recovery at period 13 |
| ltv_worsening | 20 | valuation falls 1.5%/month |
| ltv_improving | 20 | balance amortises against a flat valuation |
| refreshed valuations | 20 | re-valued mid-window, so age resets |

**Independently defined truth**, stated as literals: total losses £300,000,
recoveries £120,000, recovery rate 40.0%, redemptions £8,590,699, arrears 0% →
10.85%, London 31.69% → 36.18%, loans 400 → 356.

Two properties worth naming:

- **Concentration drift is produced by runoff**, not by editing a field. No
  loan's region ever changes; London's share rises because South East loans
  leave. That makes it an *observed* behaviour rather than an authored one.
- **The weak vintage is not labelled.** 2025 shows 44.9% 30+ DPD against 0.0%
  for 2024, and the analytics discover it.

---

## 4. Historical Readiness Framework

Current-state metrics are unchanged from Sprint 2.5 (48 metrics, 93.6%
deterministic). Historical behaviour is the new lens.

| Category | Metric | Current/Historical | Existing/New | Tool | Status |
|---|---|---|---|---|---|
| performance | Arrears trend (30/60/90) | Historical | New series over existing measures | `portfolio_history` | READY |
| performance | DPD transitions, cures | Historical | **Existing** (`migration_matrix`) | `transition_analysis` | READY |
| performance | Default trend | Historical | New series | `portfolio_history` | READY |
| performance | Cumulative losses | Historical | **New** | `loss_analysis` | READY |
| performance | Period loss rate | Historical | **New** | `loss_analysis` | READY |
| performance | Recoveries / recovery rate | Historical | **New** | `loss_analysis` | READY |
| performance | Prepayment / redemption rate | Historical | **New** | `prepayment_analysis` | READY |
| performance | Balance runoff | Historical | New series | `portfolio_history` | READY |
| collateral | LTV movement | Historical | New series | `portfolio_history` | READY |
| collateral | Valuation age trend | Historical | New series over existing profile | `portfolio_history` | READY |
| concentration | Concentration drift | Historical | New series | `portfolio_history` | READY |
| composition | Vintage performance | Historical | **Existing** (`analytics_lib.cohort`) | `cohort_comparison` | READY |
| composition | Cohort comparison | Historical | **New** | `cohort_comparison` | READY |
| data_quality | Validation / completeness trend | Historical | — | two calls, compared | JUDGEMENT_ONLY |
| trend | Period-on-period movement | Historical | **Existing** | `period_change` | READY |
| trend | Covenant deterioration | Historical | **Existing** | `evaluate_covenants` | READY |

**13 READY, 1 JUDGEMENT_ONLY.** Data-quality drift stays judgement-only for the
same reason as in Sprint 2.5: what counts as *material* drift depends on which
fields moved and why.

No `FORECAST` or `SCENARIO` category exists, and none should.

---

## 5. Prepayment methodology

**Newly implemented.** No prior methodology existed in the repository.

**What Trakt measures:** the observed rate at which principal left the book
*early* — that is, other than by scheduled amortisation.

**Numerator:** `unscheduled_principal_collections` from the closing snapshot,
plus full redemptions.

**Denominator:** the **opening** balance of the period, from the prior snapshot.

**Redemptions** are derived by set difference: a redeemed loan is *absent* from
the closing tape, so its exit cannot be read from a frame that no longer
contains it. Exits are found against the opening snapshot and valued at the
opening balance. This makes the rate work on a real servicer tape where a
per-period redemption column is often unpopulated — and it counts **every**
exit, so contractual maturity and repossession are included alongside genuine
early redemption. That caveat is in the returned notes, not hidden.

**Excluded, by name:** scheduled amortisation, contractual maturity as a
separate concept, and default-related balance reduction. None is prepayment.

**Annualisation:** `CPR = 1 − (1 − mean SMM)^12`, reported *alongside* the SMM
rather than instead of it, and labelled as an annualisation of an observed rate
— not a forecast.

**Refusal:** a tape without `unscheduled_principal_collections` returns
`available: false` with a reason. It does **not** fall back to inferring a rate
from the change in total balance.

**Result on the synthetic portfolio:**

```
OBSERVED_CPR@v1     SMM 0.7740%    CPR 8.9025%    over 17 periods
unscheduled principal   £982,499
redemptions             £8,590,699   (matches the fixture's independent figure)
```

The assertion that matters: on a constant-population window with unscheduled
principal zeroed, the balance still falls through amortisation — and the
reported SMM is **0.0000%**. A rate inferred from balance movement would have
reported a healthy-looking figure for a book where nobody prepaid anything.

---

## 6. Loss and recovery methodology

**Newly implemented.** Fields existed (`allocated_losses`,
`cumulative_recoveries`, `recoveries_in_period`); no calculation did.

**Every defensible denominator is reported and none is chosen**, because they
answer different questions:

| Rate | Question it answers | Synthetic result |
|---|---|---|
| cumulative loss / original balance | lifetime experience of the pool | **0.428%** |
| cumulative loss / opening balance | loss relative to where the window began | **0.380%** |
| cumulative loss / current balance | loss relative to what remains | **0.453%** |
| recoveries / losses | how much came back | **40.0%** |

Three answers, same portfolio, 19% apart. Silently choosing one is how a book
gets described as materially better or worse than it is.

`allocated_losses` is a **cumulative** field, so the closing snapshot carries
the running total and a period figure is the *difference* between two snapshots.
Summing across 18 snapshots would count the same loss eighteen times — asserted
directly in the tests.

**Result:** cumulative losses £300,000, recoveries £120,000, net £180,000 —
matching the planted truth exactly.

---

## 7. Arrears migration

Not just a level. The series, from `portfolio_history`:

```
30+ DPD   0.00% ................. 15.07% ..... 9.98% ..... 16.50% ..... 10.85%
90+ DPD   0.00% ......... 5.41% ......... 9.98% ................... 10.85%
```

The dips are **cures** — the curing cohort returning to performing at period 8
and again at period 15. A level alone would hide them entirely.

And the transitions, from `transition_analysis` (2025-02-28 → 2025-10-31), via
the Risk Monitor's own matrix:

| from | to | loans | balance | type |
|---|---|---|---|---|
| performing | arrears | 50 | £7,351,306 | changed |
| performing | — | 16 | £3,200,000 | exited |
| performing | performing | 334 | £66,331,814 | unchanged |

50 = the 40 deteriorating plus 10 defaulting loans. 16 = the redemption waves at
periods 4, 6 and 8. Both match the fixture.

This is the distinction that matters operationally: a rising arrears figure
caused by *new* delinquency and one caused by *existing* cases worsening call
for different responses, and only the transition view separates them.

---

## 8. Cohort and vintage

`analytics_lib.cohort` was reused rather than rebuilt; `compare_cohorts` adds
the comparison shape an investigation needs.

**The weak vintage, discovered rather than told:**

```
2025 vintage (100 loans):  44.91% 30+ DPD    WA LTV 62.54%
2024 vintage (256 loans):   0.00% 30+ DPD    WA LTV 56.60%
                          ---------------
difference                +44.91pp          +5.94pp
```

The fixture never labels a vintage as weak. `arrears["remainder"] == 0.0` is the
sharpest available evidence that the comparison is reading the right population.

**A cohort figure is never returned without its comparator.** "The high-LTV
cohort has 4% arrears" is a number; "4.0% against 1.2% for the rest" is a
finding. An empty cohort is explicitly *not* reported as good performance.

---

## 9. Concentration and LTV movement

Both observed across the window, neither authored:

| | first | last | change |
|---|---|---|---|
| largest region share | 32.95% | **36.18%** | +3.23pp, rising |
| WA LTV | 60.00% | **58.04%** | −1.96pp, falling |
| loan count | 400 | **356** | −44 |

No loan's region ever changes. The concentration rises because other loans
redeem — which is what makes it a genuine runoff effect rather than a planted
number. WA LTV *improves* overall despite a deliberately worsening cohort,
because amortisation across the rest of the book outweighs it; that tension is
in the fixture on purpose.

---

## 10. Investigation capability

The 17 questions from the brief.

| # | Question | Supported? | Tool | Evidence |
|---|---|---|---|---|
| 1 | Current composition | ✅ | `portfolio_summary`, `stratify` | 356 loans, £66.2m, WA LTV 58.04% |
| 2 | What changed over 6/12 months | ✅ | `portfolio_history` | five series above |
| 3 | Observed prepayment rate | ✅ | `prepayment_analysis` | SMM 0.774%, CPR 8.90% |
| 4 | Is prepayment accelerating? | ✅ | `prepayment_analysis` | `per_period` SMM series |
| 5 | How have arrears changed | ✅ | `portfolio_history` | 0% → 10.85% |
| 6 | DPD bucket migration | ✅ | `transition_analysis` | 50 changed, 16 exited, 334 unchanged |
| 7 | Cure behaviour | ✅ | `transition_analysis` + series | dips at periods 8 and 15 |
| 8 | Best/worst vintage | ✅ | `cohort_comparison` | 2025 44.91% vs 2024 0.00% |
| 9 | Are newer vintages deteriorating? | ✅ | `cohort_comparison` per period | comparison at any snapshot |
| 10 | Has LTV distribution changed? | ◐ **partial** | `portfolio_history` (`wa_ltv`, `high_ltv_share_pct`) | the *distribution* needs `distribution_change`; the aggregate series is available |
| 11 | Geographic concentration change | ✅ | `portfolio_history` | 32.95% → 36.18% |
| 12 | Stale valuations increasing? | ✅ | `portfolio_history` (`stale_valuation_pct`) | delegates to the shared profile |
| 13 | Cumulative loss performance | ✅ | `loss_analysis` | £300,000 on four denominators |
| 14 | What recoveries occurred | ✅ | `loss_analysis` | £120,000, 40.0% of losses |
| 15 | Which cohort drives deterioration | ✅ | `cohort_comparison` | 2025 vintage, +44.91pp |
| 16 | Flagged cohort vs the rest | ✅ | `cohort_comparison` | always returns both |
| 17 | Evidence for a historical metric | ◐ **partial** | method id + inputs + per-period detail | see §11 |

**15 supported, 2 partial, 0 unsupported.**

The two partials are honest: LTV *distribution* movement needs the existing
`distribution_change` wired into the history tool (small), and historical
explainability is method-level rather than reaching `explain_values` per input
(see §11).

---

## 11. Explainability

**"Why is observed prepayment 8.90% CPR?"** — answered deterministically, with
no model involved:

> Method `OBSERVED_CPR@v1`. Numerator: `unscheduled_principal_collections`
> (£982,499 across the window) plus full redemptions (£8,590,699, derived from
> 44 loans present in an opening snapshot and absent from the next, valued at
> their opening balance). Denominator: the opening balance of each of 17
> periods. Excluded: scheduled amortisation, contractual maturity,
> default-related reduction. SMM 0.7740% is the mean of 17 monthly rates, each
> returned in `per_period` with its own opening balance and numerator.
> CPR = 1 − (1 − 0.007740)^12 = 8.9025%, an annualisation of the observed rate.

**"Why do we say the 2025 vintage is deteriorating?"**

> `cohort_comparison` on `portfolio_cohort = 2025` at 2026-07-31: 100 loans,
> 44.91% of cohort balance 30+ DPD, against 0.00% for the 256 loans outside it.
> Both figures use the same balance-weighted measure over the same snapshot.
> `portfolio_history` shows the portfolio-level 30+ DPD series rising from 0.00%
> to 10.85% across 18 periods, and `transition_analysis` attributes 50 of those
> loans to performing→arrears movements rather than to new entrants.

**The gap, stated plainly:** these explanations are *method-level* — the
methodology, the inputs, the periods and the population. They do not yet reach
`explain_values` for each contributing field, so a reviewer can reproduce the
calculation but cannot trace one input to its source column and mapping in the
same call. That is the partial in question 17.

---

## 12. Performance

| Workload | Result |
|---|---|
| 18 periods × 7 measures, 6,876 rows | **81 ms** |
| Snapshots read | **18** — one pass per snapshot, not per measure |
| 18 periods × 8 measures, if naive | would be 144 passes |
| Prepayment, 18 periods | included in the same sweep |
| Transition analysis, 2 snapshots | ~1 ms |

The structural property matters more than the number: `portfolio_series`
evaluates **every measure in one pass per snapshot**, so adding a measure is
nearly free and adding a period costs one frame walk. Telemetry publishes
`snapshots_read` so a caller can see it.

No N × periods × loans explosion is possible through these tools: none returns a
per-loan series, the period window is capped at 60, and `cohort_comparison` is
bounded to two groups.

**No bottleneck was found requiring optimisation**, and no agent-specific fast
path was created. Larger-scale benchmarking (100k loans × 12 periods) was not
run — the shared implementation is the vectorised `analytics_lib` code already
profiled in Sprint 2, and any future problem should be profiled there rather
than worked around in the tool layer.

---

## 13. Architecture check

> **Did this sprint extend the shared MI/analytics layer, or accidentally create
> a Securitisation-Agent-specific analytics layer?**

**It extended the shared layer.** The evidence:

- All four new calculations live in **`analytics_lib/history.py`**, beside
  `stratify`, `concentration`, `cohort` and `valuation_age` — importable by
  React, MI Query and the agent tools alike. None is in agent code, a prompt,
  the MCP adapter or a React component.
- `transition_analysis` **wraps `risk_monitor.migration.migration_matrix`**
  rather than restating it, and reports `calculation_source` so the shared
  origin is visible in the response.
- `stale_valuation_pct` in the history tools **delegates to
  `analytics_lib.valuation_age`**, so "stale" means one thing across the
  current-state and historical views.
- A structural test (`test_no_history_tool_holds_its_own_calculation`) asserts
  the handler module imports from `analytics_lib.history`, references
  `migration_matrix`, and contains no local rate arithmetic.
- Nothing added is named for securitisation, and nothing checks whether the
  caller is a readiness agent.

---

## 14. Regression evidence

Full suite, both trees, `-p no:randomly` so ordering is fixed.

| | baseline `130f3b0` | current |
|---|---|---|
| passed | 5,137 | **5,260** (+123) |
| failed | 64 | **64** |
| errors | 13 | **13** |
| skipped | 33 | 33 |
| wall clock | 2,454 s | 2,429 s |

**Both complete ID sets are identical** — extracted, sorted, deduplicated and
diffed in both directions:

```
failure ids: base=64  current=64
error ids:   base=13  current=13

=== NEW FAILURES ===      (empty)
=== FIXED ===             (empty)
=== NEW ERRORS ===        (empty)

failure sets IDENTICAL
error sets IDENTICAL
```

**One caveat on how this was run, stated because it affects how much weight the
result carries.** The candidate run was started at `3463a1e` and Sprint 2.5C
edits landed in the working tree while it was still executing. pytest imports
test modules at collection, but this repository imports handler modules lazily
*inside* handler functions, so a late-running test could in principle have
picked up an edited module. The totals and both ID sets match the baseline
exactly, which is strong evidence nothing was disturbed — but it is not a clean
snapshot of `3463a1e`, and it should not be read as one.

A clean comparison (`cdefc25` → Sprint 2.5C head, tree untouched throughout) was
run subsequently and is reported in `docs/mi_metric_methodology_review.md`. That
one supersedes this as the trustworthy regression evidence for everything from
Sprint 2.5B onward.

**Sprint 2.5B added 38 tests** in `tests/test_historical_analysis.py`. No
existing test was changed in Sprint 2.5B itself; one was updated in Sprint 2.5C
and is explained there.

---

## 15. Remaining gaps

**Genuine missing deterministic capability** (small, and none blocking):

- LTV **distribution** movement across periods — the aggregate series exists;
  wiring the existing `distribution_change` into `portfolio_history` is the gap.
- Historical explainability at input level — `explain_values` for the fields
  feeding a historical metric.
- Scheduled-vs-unscheduled split reported separately — both fields are read; only
  the combined rate is published.

**Data unavailable:**

- Nothing new. The canonical model supported everything this sprint needed,
  which was the significant finding of Part 1.

**Requires human or agent judgement:**

- Whether a deterioration is *material*.
- What counts as material data-quality drift (`JUDGEMENT_ONLY`).
- Whether a trend's *shape* is accelerating or reverting — `direction` describes
  the endpoints only, deliberately.

**Future optional capability, explicitly not a blocker:**

- Forecasting, scenarios and stress testing. Out of scope by instruction, and
  nothing in this sprint depends on them. **Trakt calculates what happened.**

---

## 16. Sprint 3 recommendation

> **Can an independent Securitisation Readiness Agent now assess BOTH where the
> portfolio stands today AND how it has behaved over time, using governed Trakt
> tools?**

**Yes. Proceed to the agent build.**

The demonstration in §1–§11 is entirely deterministic — every figure produced by
a governed tool call, no model in the loop. An agent can now receive "assess this
portfolio" and obtain, through 23 governed tools:

- the current position (Sprint 2.5, 93.6% of the readiness framework);
- the behavioural history (this sprint — arrears, transitions, cures,
  prepayment, losses, recoveries, concentration drift, LTV movement, vintage
  performance);
- both classified into **external breach**, **Trakt screening flag**, **measured
  fact**, and **agent judgement**, with the authority carried on every result.

**No genuine blockers remain.** The three gaps in §15 are improvements: an agent
can complete a credible first-pass historical review without any of them.

**One caution carried forward, now sharper.** With a history available, the
temptation shifts from "compute a metric" to "characterise a trend" — to say a
book is *stabilising* or *accelerating* when Trakt has reported only two
endpoints and a direction. That characterisation is a judgement, it must be
attributed to the agent, and it must never be presented as something Trakt
measured.
