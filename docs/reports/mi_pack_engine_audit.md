# Trakt MI Pack — calculation provenance audit and pipeline capability assessment

**Branch** `claude/mi-pack-engine-audit`
**Base** `9e00606` (head of `claude/pptx-funder-pack-enhancement`)
**Phases 1 and 2 — report only. No code changed.**

---

# PHASE 1 — CALCULATION PROVENANCE AUDIT

## 1.1 Verdict

**The pack's methodology page overstates its own provenance.**

It states, on every deck:

> Figures are produced by the governed MI calculations and are identical to the
> management dashboard for the same portfolio and reporting date.

That is true of the **KPI tiles, stratifications, cross-tabs, cohort series,
concentration tests and the balance bridge** — those are engine output rendered
verbatim. It is **not** true of a further **31 values** the presentation layer
derives for itself, and it is not true in the specific sense a reader assumes:
three of the Key Measures tiles **cannot be reconciled with one another**, and a
funder who divides one by another lands **15.2 percentage points** from the tile
that claims to be that ratio.

Nothing here is a wrong number against a source. There is no external validation
source. The defect is that the pack asserts a single governed origin for figures
that have several, and asserts internal consistency it does not have.

## 1.2 The inventory — values computed outside the engine

`P` = defect (business figure derived in the presentation layer).
`F` = formatting only (unit/precision; belongs in the presentation layer).

| # | Site | Value derived | Kind | Proposed home |
|---|---|---|---|---|
| 1 | `deck.slide_pipeline:2011` | `avg_case = amount / cases` | **P** | `pipeline_contract.build_pipeline_snapshot` as `averageCaseAmount` |
| 2 | `deck.slide_pipeline:2003` | week-on-week Δ for amount and cases | **P** | `pipeline_contract` — emit `priorWeek.delta{...}` beside the levels |
| 3 | `deck._movement_finding:1366` | `share = leg / Σlegs` | **P** | `evolution.funded_balance_movement` as `legShares` |
| 4 | `deck._stock_strap:1450` | `pct = (closing − opening)/|opening|` | **P** | `evolution.funded_evolution` as `periodChangePct` |
| 5 | `deck._stock_takeaway:1488` | `delta = closing − opening` | **P** | same as #4 |
| 6 | `deck._stock_takeaway:1494` | per-book `vals[-1] − vals[0]` | **P** | `evolution.funded_evolution` breakdown as `movement` |
| 7 | `deck._stock_takeaway:1501` | `share = largest / closing` | **P** | same as #6 |
| 8 | `deck.slide_portfolio_composition:916` | `share = slice.balance / total` | **P** | `portfolio_context` as `balanceShare` |
| 9 | `deck.slide_portfolio_composition:945` | share as a % label | **P** | same as #8 |
| 10 | `deck.slide_portfolio_composition:880` | `Share` table column | **P** | same as #8 |
| 11 | `deck.slide_portfolio_comparison:1023` | `opening = total − Σ movers` | **P** | `portfolio_context` as `openingBalance` |
| 12 | `deck.slide_movement_drivers:1093` | same as #11 | **P** | same as #11 |
| 13 | `deck._has_spread:335` | `max(values)/total` vs 99.5% | **P** | `snapshots` — emit `concentrationShare` per stratification |
| 14 | `deck._projection_disclosure` | (none — reads engine fields) | — | — |
| 15 | `deck.slide_cohort_progression:1983` | `len(x.live) − 1` periods | **P** | `evolution.funded_cohort_progression` as `periodsObserved` |
| 16 | `deck.slide_portfolio_projections:1558` | `retention × 100` | **F** | — |
| 17 | `cohorts.CohortRow.average_balance:95` | `balance / loan_count` | **P** | `cohorts` service (it already emits balance and count) |
| 18 | `cohorts.retention:287` | `last / first × 100` | **P** | already emitted by the service as `balanceRetention`; this is the FALLBACK path and should be removed, not moved |
| 19 | `cohorts._to_points:129` | fraction→points heuristic | **F** | — |
| 20 | `movement.share_change_pp:101` | `end/closing − start/opening` | **P** | `evolution.funded_bridge` as `shareChangePp` |
| 21 | `movement.movers:105` | materiality floor `|opening| × 0.5%` | **P** | threshold belongs in governed config, not a module constant |
| 22 | `movement.reconciles:95` | Σ contributors vs total | **P** | already asserted by the engine; this is a second opinion |
| 23 | `insights._pct_change:89,92` | `change / |current − change|` | **P** | `snapshots.monthly_change` as `balanceChangePct` |
| 24 | `insights.portfolio_mix:328` | `slice.balance / total × 100` | **P** | same as #8 |
| 25 | `insights.movement_attribution:286` | `lead / total_move × 100` | **P** | `evolution.funded_bridge` as `contributorShare` |
| 26 | `insights.weighted_ltv_trend:396` | `current − prior` (pp) | **P** | `evolution.funded_evolution` as `waLtvChangePp` |
| 27 | `watchlist:185` | `change / |balance − change| × 100` | **P** | same as #23 |
| 28 | `watchlist:255` | `current − prior` LTV delta | **P** | same as #26 |
| 29 | `forecast_accuracy:80` | `(actual − prior)/|prior| × 100` | **P** | `evolution.forecast_evolution` — it already emits `forecast_variance` in currency; add `forecastErrorPct` |
| 30 | `forecast_accuracy:94,95` | mean signed / mean absolute error | **P** | same as #29 |
| 31 | `concentration.travel:181` | `current − prior` vs 2% of limit | **P** | `concentration_tests_api.compute_history` as `direction` |
| 32 | `concentration.stress_note:205` | `stressed − current` vs 0.5% | **P** | `concentration_tests_api` as `stressDirection` |
| 33 | `composition.build_facts:222,260` | `pipeline_share` | **P** | composition fact, but its inputs are wrong — see §1.4 |
| 34 | `metric_resolver.compact_currency:64-68` | `/1e9`, `/1e6`, `/1e3` | **F** | — |
| 35 | `metric_resolver:89` | `value × 100` for percent | **F** | — |
| 36 | `render._currency_tick_formatter` | axis scaling | **F** | — |
| 37 | `render.draw_lines:461` | fraction→points heuristic | **F** | — |
| 38 | `chart_resolver:449` | waterfall residual check | **P** | a second opinion on an engine identity; should read the engine's residual |

**Count: 31 P, 7 F.** Every `P` is a business figure the pack derives and then
presents under a claim of single governed origin.

### 1.2.1 A dead aggregation engine inside the presentation layer

`mi_agent_pptx/metric_resolver.py` contains a **complete aggregation engine** —
`sum`, `mean`, `median`, `weighted_avg` computed directly from the dataframe
(`_compute`, lines 210-248), plus a forecast bridge (`_forecast_funded`).

**It is not reachable from the production deck.** `MetricResolver` is
instantiated only in `tests/mi_agent_pptx/test_data_and_metrics.py`; the live
deck imports only the formatters (`compact_currency`, `compact_number`,
`format_percent`) from the same module. `pptx_builder.py` and
`insight_resolver.py`, its only production callers, are v1 dead code.

It is reported here because it is a loaded gun in the presentation layer, not
because it is firing. **Recommendation: delete the aggregation half of the
module and keep the formatters**, so no future slide can pick it up.

## 1.3 THE KEY MEASURES TILES — measured, not asserted

The brief's suspicion is correct, and the gap is larger than "rounding".

Measured on the three-book, five-period QA fixture (318 loans):

```
avg loan balance   (unweighted mean)         329,438.49
WA property value  (balance-weighted mean)   934,923.18
WA current LTV     (balance-weighted mean)      50.4124 %   <- the tile

avg_balance / WA_property                       35.2370 %   <- what a reader computes
                                                -15.1754 pp  from the tile

ratio of aggregates  ΣB / ΣV                    41.2146 %
                                                 -9.1978 pp  from the tile
unweighted mean property value               799,324.57
unweighted avg_bal / unweighted mean value      41.2146 %
```

### 1.3.1 Stated definitions

| Tile | Numerator | Denominator | Weighting |
|---|---|---|---|
| **Average loan balance** | Σ `current_outstanding_balance` | count of loans | **unweighted** (equal weight per loan) |
| **WA property value** | Σ (`current_valuation_amount` × balance) | Σ balance | **balance-weighted** |
| **WA current LTV** | Σ (`current_loan_to_value` × balance) | Σ balance | **balance-weighted mean OF RATIOS** |
| **WA original LTV** | Σ (`original_loan_to_value` × balance) | Σ balance | balance-weighted mean of ratios |
| **WA interest rate** | Σ (`current_interest_rate` × balance) | Σ balance | balance-weighted |
| **WA months on book** | Σ (`months_on_book` × balance) | Σ balance | balance-weighted |
| **WA youngest age** | Σ (`youngest_borrower_age` × balance) | Σ balance | balance-weighted |
| **Single borrowers** | count where `borrower_type == single` | count where type known | **unweighted, and a different denominator** (excludes unknown) |

Source: `mi_agent_api/snapshots.compute_funded_snapshot`, lines 774-845;
`_weighted_average` lines 56-71.

### 1.3.2 Can they tie?

**No. They cannot tie, for two independent reasons, and the pack must disclose
the basis on each tile.**

1. **Weighting mismatch.** Average loan balance is unweighted; WA property value
   is balance-weighted. Putting the two side by side invites a division whose
   operands are averages over different populations. Cost on the fixture:
   **≈6.0pp** (35.24% → 41.21% when both are put on the same unweighted basis).

2. **Average of ratios ≠ ratio of averages.** Even with weights aligned, the LTV
   tile is `E[B/V]`, and dividing the money tiles gives `E[B]/E[V]`. These differ
   by Jensen's inequality for any book with dispersion in LTV. Cost on the
   fixture: **≈9.2pp** (41.21% vs 50.41%).

Both are legitimate measures. Neither is wrong. But **no set of definitions
makes all three tie**, because tying requires the LTV tile to be a ratio of
aggregates and the money tiles to share a weighting basis — and a
ratio-of-aggregates LTV is a *different economic statement* (it is the book's
gearing) from a balance-weighted mean LTV (the typical pound's gearing).

**Therefore the deliverable is disclosure, not correction.** Each tile must
carry its weighting basis, and the three must not be presented as though one
divides into another. The recommended labels:

- Average loan balance — *per loan, unweighted*
- WA property value — *balance-weighted*
- WA current LTV — *balance-weighted average of loan-level LTVs*

and, where all three appear together, one line: *these are averages on different
bases and do not divide into one another.*

### 1.3.3 The same treatment, applied to every other derived measure

| Measure | Basis today | Reconciles with its neighbours? |
|---|---|---|
| Average case amount (pipeline) | unweighted, `amount / cases` | Yes — but see §1.4: the numerator includes completed cases |
| Cohort `average_balance` | unweighted | Yes |
| Cohort `balance_vs_formation` | ratio of period aggregates | Yes (and correctly renamed last sprint) |
| Cohort `loan_survival` | count ratio | Yes |
| `wa_ltv` in cohort series | balance-weighted mean of ratios | **Same Jensen gap as §1.3.2** where read against cohort balance |
| NNEG `headroom_pct` | `1 − ΣB/ΣV` — **ratio of aggregates** | **Inconsistent with `wa_ltv` in the same payload**, which is a mean of ratios. Two LTV-shaped figures on one cohort, two bases. |
| Concentration `utilisation` | `value / threshold` | Yes, engine-computed |
| Forecast `error_pct` | mean of per-period `(A−F)/F` | unweighted across periods; a large period counts the same as a small one |

The **NNEG vs `wa_ltv`** inconsistency inside a single governed cohort payload
is the one other case in the pack where two figures describing the same economic
quantity are computed on different bases. It should be disclosed the same way.

## 1.4 A material engine defect found while tracing

**The headline pipeline includes completed and withdrawn cases.**

`pipeline_prep` line 759: `total_pipeline_amount` sums
`current_outstanding_balance` over the **whole extract** with no stage filter.
`row_count` is `len(df)`. Neither excludes `COMPLETED` or `WITHDRAWN`.

Measured on the QA fixture:

```
   APPLICATION     10 cases      2,350,000.00
   COMPLETED       10 cases      2,930,000.00     <- funded; not pipeline
   KFI             10 cases      2,780,000.00
   OFFER           10 cases      2,640,000.00
   ALL   (headline) 40 cases    10,700,000.00
   LIVE  (correct)  30 cases     7,770,000.00
   OVERSTATED BY    10 cases     2,930,000.00  =  27.4% of the headline
```

This is not a presentation defect — it is in the engine, and it propagates:

- the Pipeline Overview tiles and the average case amount;
- `forecast_bridge.pipeline_amount`, hence the forecast bridge;
- `composition.pipeline_share`, hence **four slide-inclusion decisions** in the
  conditional pack;
- the executive landing page's pipeline tiles.

A completed case has funded. It is in the funded book **and** in the pipeline
stock — double-counted across the two lenses on the same page.

---

# PHASE 2 — PIPELINE CAPABILITY ASSESSMENT

## 2.1 Per-stage movement reconciliation — **COMPUTABLE**

The target form, per stage, for KFI / Application / Offer, on counts and amounts:

```
opening live + additions − moved/dropped/completed ± amount change = closing live
```

**It is computable today, and it reconciles exactly.** Proven against the real
preparation path:

```
comparing 2026-06-12 -> 2026-06-26
  KFI
    opening live           9    2,277,000.00
    + additions           10    2,780,000.00
    - moved/dropped        9    2,277,000.00
    +/- amount change      0           +0.00
    = closing live        10    2,780,000.00
    identity check              2,780,000.00   residual +0.000000
  APPLICATION  ... residual +0.000000
  OFFER        ... residual +0.000000
```

### What it needs, and what already exists

| Requirement | Status |
|---|---|
| Persisted weekly history | **EXISTS.** `pipeline_contract.weekly_extract_inventory` enumerates dated extracts; the files are the register. |
| Case-level state per extract | **EXISTS.** `load_prepared_pipeline` emits `pipeline_case_identifier` and `pipeline_stage` per row. |
| A stable case identifier | **CONDITIONAL — see 2.2.** |
| A persisted snapshot register | **NOT NEEDED**, and the one that exists is unusable — see 2.3. |

**No new analytical primitive is required.** The reconciliation is a join.

## 2.2 Is the case identifier stable? — **CONDITIONALLY, and there is a config gap**

`pipeline_case_identifier` is a **natural key carried from the source**, not a
hash. It is therefore stable across amendments by construction: amending a loan
amount does not change the case reference.

It survives preparation **only when the source column matches a declared alias**.
`config/mi/pipeline_field_contract.yaml` line 169:

```yaml
source_aliases: [account number, case id, pipeline id, application no, app id]
```

Verified: a source column named `case id` maps through to
`pipeline_case_identifier` in the prepared frame. A source column named
`unique_identifier` **does not** — the prepared frame is then **case-anonymous**,
and every pipeline figure is an aggregate over an identity-less frame.

**This is the same divergence closed in the funded bridge last sprint.**
`unique_identifier` is the ESMA Annex 2 RREL1 identifier, it is what the funded
tape carries, it is what `engine.platform_assembler.LOAN_KEY_FIELDS` accepts —
and it is absent from the pipeline contract's alias list. The QA fixture writes
exactly that column, which is why the fixture's pipeline is anonymous.

**Blocker status: NOT blocking. One line of config.** But until it is added, any
client whose pipeline extract names its key `unique_identifier` gets no
reconciliation, and the pack must say so rather than approximate.

`pipeline_prep` already raises `missing_case_identifier` as a **blocker** and
`duplicate_case_identifiers` as a **warning** (lines 880-889), so the data
quality signal for suppression already exists.

## 2.3 The snapshot register — exists, not wired, and its key is unusable

A `SnapshotStore` exists (`snapshot/store.py`) with a pipeline namespace
(`OPP_` ids via `snapshot.keys.make_pipeline_opportunity_id`).

**It is not wired into the pipeline path.** No import of `snapshot.store` or
`SnapshotStore` appears anywhere in `mi_agent_api/` or `mi_agent_pptx/`. Its
consumers are `mi_agent.states`, `mi_agent.mi_runtime`, `risk_monitor` and
`regulatory_watch`.

**And its opportunity key could not support this reconciliation anyway.**
`DEFAULT_OPPORTUNITY_FIELDS` hashes **mutable business attributes** including
`loan_amount`, `amount`, `product` and `broker`:

```
  week 1, amount 250,000        -> opp_3525d44b325c29de
  amount amended to 262,500     -> opp_8613a174de40a170   DIFFERENT — identity lost
  product corrected             -> opp_27b47de47469ced4   DIFFERENT — identity lost
  broker re-keyed 'ACME'        -> opp_3525d44b325c29de   SAME (normaliser lowercases)
```

The key breaks on precisely the event the reconciliation exists to measure — a
loan amount amendment. A case whose amount moves would read as one case exiting
and a different case arriving, inflating both the additions and the
moved/dropped legs and reporting **zero** amount change.

**Recommendation: do not wire it in.** Use `pipeline_case_identifier` from the
extracts. Report the register's key as a defect to its owners separately.

`config/mi/state_library.yaml` — which declares a `total_pipeline` state — is
explicitly marked *"CONFIG SKELETON — declaration only… nothing reads this file
yet."* The state assembler for pipeline does not exist.

## 2.4 Stage-to-stage conversion rates — **COMPUTABLE**

Proven on a fixture with cases that persist and progress:

```
transitions 2026-06-12 -> 2026-06-26:
   OFFER -> COMPLETED   25
   left the extract entirely: {'OFFER': 5}

stage-to-stage conversion (count basis):
   OFFER -> COMPLETED   25 of 30 = 83.3%
```

Computable on **both count and amount bases**, over **all-time** (join first to
last extract) and **since-date** windows (filter the extract list). No new
primitive.

### The current figure is not a conversion rate, and cannot become one

`evolution.pipeline_funnel_evolution` tracks **stock and flow aggregates only** —
`series[stage] = [{week, value, count}]`. Its "conversion" is

> average weekly FLOW into a stage over the last 5 weeks ÷ KFI STOCK as it stood
> `lag_weeks` earlier

That is a **stock ratio**. It has no case-level numerator: it cannot say how many
of the cases that were at KFI reached Application, because it never knew which
cases they were. Lagging the denominator makes it less misleading; it does not
make it a conversion rate. The engine's own docstring is honest about this — the
presentation was not.

## 2.5 Cohort engine — carries **seven** measures, not one

`evolution.funded_cohort_progression` already emits, per cohort per period:

| Measure | Field | Phase 3 item it unblocks |
|---|---|---|
| Balance | `funded_balance` | (current) |
| Loan count | `loan_count` | survival |
| **Weighted average LTV** | `wa_ltv` | **LTV migration** ✓ |
| WA interest rate | `wa_interest_rate` | — |
| Average borrower age | `avg_borrower_age` | — |
| **NNEG exposure / headroom / headroom %** | `nneg_exposure`, `nneg_headroom`, `nneg_headroom_pct` | **NNEG headroom** ✓ |
| **Survival and exits** | `loanRetention`, `balanceRetention`, `exitsCount` | **exit rate, survival** ✓ |

**All four Phase 3 cohort items are unblocked.** They are computed today and
rendered nowhere. Caveat from §1.3.3: `wa_ltv` (mean of ratios) and
`nneg_headroom_pct` (ratio of aggregates) are two LTV-shaped figures on different
bases in the same payload, and must be labelled accordingly.

## 2.6 Paired-dimension matrix — **fully generic**

`snapshots.cross_tab(df, x_dimension, y_dimension, scope)` accepts **any pair**
of governed stratification dimensions. `MULTIDIM_PAIRS` is a module-level tuple
of three, not a hardcoded renderer path.

Eleven dimensions are available: `ltv, age, region, rate, product, vintage,
status, equity, broker, borrower_type, ticket`.

So **LTV × vintage** and **borrower age × ticket size** are available now.
**Product mix** is available where the tape carries `product_type` / `product` /
`loan_product`.

Two gaps against the brief's "everything through configuration" constraint:

1. `MULTIDIM_PAIRS` is Python, not YAML. It should move to
   `config/mi/stratification_catalogue.yaml`.
2. The measure is hardcoded to `current_outstanding_balance`. The **region bubble
   chart** (balance × LTV, sized by loan count) is *not* a cross-tab — it needs a
   per-region aggregate of three measures. `cross_tab` already emits `count` per
   cell, so a one-dimensional variant carrying `(balance, wa_ltv, count)` per
   category is a small engine addition, not a new analytic.

## 2.7 Concentration engine — most of Phase 3 already exists

| Phase 3 item | Status |
|---|---|
| Minimum-type tests | **EXISTS.** `models.OPERATOR_MIN` / `OPERATOR_MAX`; `ActiveTest.operator` validates both. |
| Warning band distinct from breach | **EXISTS.** `ActiveTest.warning_fraction`, default 0.9. Already rendered in the summary strip. |
| Property value band family | **EXISTS.** `property_value_below_share`, `property_value_above_share`. |
| Balance band family | **EXISTS.** `balance_above_share`, `balance_average`, `balance_maximum`. |
| Joint lives share | **EXISTS.** `borrower_joint_share`. |
| Direction of travel | **EXISTS** (built last sprint; `concentration.travel`) — but computed in the presentation layer, see inventory #31. |
| Current and forward on one bar | **EXISTS.** `render.draw_utilisation_tests` already accepts `expectedUtilisation` and `stressUtilisation`. |
| **Two limits on the same exposure** | **DOES NOT EXIST.** `ActiveTest` carries a single `threshold`. A hard cap plus a lower converted-pipeline threshold needs either a second threshold field or two tests bound to one exposure with a declared relationship. **This is the one genuine build.** |

The library declares **42 metrics**; a typical approved client configuration uses
four. The gap between "the engine can test this" and "this client's pack shows
it" is almost entirely **approval configuration**, not capability.

---

# WHAT PHASE 3 CAN AND CANNOT DO

**Unblocked** — no new primitive needed:

- move all 31 presentation-layer calculations into the engine;
- exclude completed/withdrawn from pipeline stock (engine fix, §1.4);
- per-stage pipeline movement reconciliation (§2.1), subject to §2.2;
- stage-to-stage conversion rates (§2.4), subject to §2.2;
- all four cohort measures — LTV migration, NNEG headroom, exit rate, survival (§2.5);
- LTV × vintage, age × ticket, product mix (§2.6);
- concentration min tests, warning band, three non-geographic families,
  direction of travel, current-and-forward on one bar (§2.7);
- every slide-level layout item.

**Conditional** — reachable, but must suppress and disclose where the condition
fails:

- pipeline reconciliation and conversion rates require `pipeline_case_identifier`
  in the prepared frame. Where absent, the section must be suppressed with the
  reason on the methodology page. `pipeline_prep` already raises the blocker.

**Genuine build, small**:

- two limits on one exposure (§2.7);
- a one-dimensional multi-measure aggregate for the region bubble chart (§2.6).

**Recommended but outside the brief's list**:

- add `unique_identifier` to the pipeline contract's case-identifier aliases (§2.2);
- delete the dead aggregation engine in `metric_resolver` (§1.2.1);
- disclose the `wa_ltv` / `nneg_headroom_pct` basis split in the cohort payload (§1.3.3).

**Nothing in Phase 3 is blocked outright.**
