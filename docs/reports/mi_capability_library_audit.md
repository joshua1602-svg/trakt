# MI Capability Library Audit

**Status:** audit only. No MI calculation, React component, PPTX module, OCC
module, configuration file, presentation rule or committed test was changed.
Two temporary read-only probes were run and discarded (their output is quoted
below).
**Base:** `main` @ `e7678c81100f562c25cb39cf1cbf69798e13a5ed`.
**Branch:** `claude/mi-capability-library-audit`.

---

## 1. Executive conclusion

**How much analytical capability is already present but unused?**
A great deal. Three findings carry the audit:

1. **The economic bridge already reconciles exactly.** `mi_agent.period_change.bridge.balance_bridge`
   returns opening + new − exited + continuing-movement = closing with a checked
   residual. Probed on a two-period book: `1,080,000 + 220,000 − 380,000 + 10,500
   = 930,500`, residual `−1.16e-10`. Nothing in React or the deck renders it.
2. **The exit leg already decomposes.** `analytics_lib.history.classify_exits`
   splits disappearances into redemption / default / maturity / unknown on
   evidence, refusing to call an unexplained disappearance a prepayment. Probed:
   `redemption 200,000 + default_exit 180,000 = 380,000`, which is exactly the
   bridge's exit leg. The two compose without an adapter.
3. **Period × group balance series already exist and already reconcile.**
   `evolution.assemble_funded_evolution` emits a `breakdowns` block for broker,
   region and LTV band on every call; each reconciles to its period total. It is
   typed in React's domain model (`evolution.ts:44`) and **rendered by nothing**.

Beyond those, a mechanical sweep found **65 payload keys computed by the governed
services and rendered by neither surface**, including an entire per-portfolio
forward projection with governed run-off curves.

**Can we materially improve the funded/funder pack without building new MI?**
Yes — substantially. Of the twelve enhancements assessed in §13, **nine are
"already computed, just surface it" or "composition of existing outputs"**. The
two most valuable — a reconciled economic movement bridge, and funded balance
stacked by constituent portfolio — are both in that group.

**Is there already an asset-class applicability / concept architecture?**
**Yes, and more of one than expected.** Three complementary layers exist, each
production-reachable:

| Layer | Owner | What it holds | Size |
|---|---|---|---|
| Concept + asset applicability | `config/business_semantics_registry.yaml` | `analytical_concept`, `asset_applicability`, `temporality`, `directionality`, `default_aggregation`, `weight_field` per field | 243 fields · 24 concepts · 6 asset classes |
| Capability availability | `config/system/mi_capability_registry.yaml` + `trakt_core/capability.py` | whether a capability can be produced for a portfolio, **and why not** | 28 capabilities · 6 states · 5 condition types |
| Composition | `mi_workflows/analytical/registry.py` | which capabilities compose into a plan, and which route already owns each | 10 capabilities |

The capability registry is explicit that it is not a fourth metadata store and
names the other three by id. Its header states the design principle the brief
asks for, in its own words: *"There is no `if asset_class == ...` anywhere in the
resolver. A capability declares the ECONOMIC CONDITIONS it needs… Onboarding a
new asset class requires no code here."*

**Recommended next move.** Do not build a concept library. **Route the one that
exists into the two surfaces that ignore it**, and surface the reconciled bridge
and the per-portfolio stock series. That is a presentation-and-routing sprint
with zero new MI. Detail in §14.

---

## 2. Current production capability map

### 2.1 Three distinct reachability tiers

Reachability is not binary here, and conflating the tiers is the main way this
estate gets misread.

| Tier | What it means | Evidence |
|---|---|---|
| **MOUNTED** | Served by the React app's API today | route registered on `mi_agent_api.app` |
| **SHIPPED, GATED** | In the App Service artefact, mounted only when an operator sets a flag | `app.py:1942` `if _agent_api.enabled():`; `agent_api.enabled()` reads `TRAKT_AGENT_API_ENABLED`, **off by default** |
| **ONBOARDING-ONLY** | Runs in the ingestion/onboarding engine, not the MI serving path | `engine/onboarding_agent/*` |

`trakt_tools` — the 27-tool governed surface carrying most of the credit,
prepayment, loss and cashflow analytics — is **SHIPPED, GATED**. It is listed in
`deploy/trakt-mi-api/package_contents.txt` ("governed tool registry (agent API
surface)") and the code is deployed; the HTTP surface is a deployment decision.
Turning it on is an environment variable, not a code change. Every capability
below marked *gated* is in that position.

### 2.2 The registered tool surface (27 tools)

```
analysis      portfolio_summary · stratify · concentration · rank_loans
              data_completeness · list_validation_exceptions
history       portfolio_history · prepayment_analysis · default_analysis
              cure_analysis · loss_analysis · transition_analysis · cohort_comparison
contractual   contractual_analytics
movement      period_change · covenant_drillthrough
covenants     evaluate_covenants
capability    portfolio_capabilities
readiness     readiness_framework · readiness_metrics · regulatory_readiness
              evaluate_rule_packs · valuation_age_profile
loans         get_loan · get_loans
provenance    explain_value · explain_values
```

### 2.3 Mounted MI services (the React estate)

| Module | Produces | React | PPTX | MI Query | Determinism | Tested |
|---|---|:--:|:--:|:--:|:--:|:--:|
| `snapshots.compute_funded_snapshot` | 14 KPIs incl. NNEG/arrears risk tile; 8 stratifications w/ availability | ✓ | ✓ | ✓ | yes | ✓ |
| `snapshots.multidimensional`* | — | — | — | — | — | — |
| `evolution.funded_evolution` | period metrics + **`breakdowns`** | partial | partial | ✓ | yes | ✓ |
| `evolution.funded_bridge` | dimensional attribution, reconciling | drawer | ✓ | ✓ | yes | ✓ |
| `evolution.funded_cohort_progression` | static pool: survival, exits, retention, NNEG series | ✓ | ✓ | ✓ | yes | ✓ |
| `evolution.pipeline_evolution` / `pipeline_funnel_evolution` | stock, weekly flow, **conversion + lag + sufficiency** | ✓ | partial | ✓ | yes | ✓ |
| `evolution.forecast_evolution` | funded / weighted pipeline / forecast per run | ✓ | handler only | ✓ | yes | ✓ |
| `cohorts.cohort_analysis` | vintage composition (Y/Q/M) | ✓ | ✓ | ✓ | yes | ✓ |
| `pipeline_contract.compute_pipeline_snapshot` | stage, completion month, broker, region breakdowns | ✓ | partial | ✓ | yes | ✓ |
| `forecast_bridge.compute_forecast_bridge` | funded + weighted pipeline → forecast | ✓ | ✓ | ✓ | yes | ✓ |
| `forecast_bridge.portfolio_projections` | **per-portfolio run-off projection** | ✗ | ✗ | ✗ | yes | ✓ |
| `forecast_extrapolation.build_extrapolation` | run-rate + KFI models, milestone ladder | ✓ | ✓ | ✓ | yes | ✓ |
| `concentration_tests_api.compute_concentration_tests` | current / expected / stress, headroom, RAG, breach horizon, emerging risks | ✓ | ✓ | ✓ | yes | ✓ |
| `concentration_tests_api.compute_history` | utilisation per test across snapshots | ✓ | ✗ | ✓ | yes | ✓ |
| `risk_limits.compute_risk_limits` | legacy extracted monitor | ✓ | ✓ | ✓ | yes | ✓ |
| `geo.exposure_by_itl3` | ITL3 exposure + coverage | ✓ | ✓ | ✓ | yes | ✓ |
| `workspace.forecast_breakdowns` | forecast by region / LTV / month | ✓ | ✗ | ✓ | yes | ✓ |
| `movement_detail` | 6-component pipeline lifecycle | drawer | ✗ | ✓ | yes | ✓ |
| `period_change` (workflow) | metric change, distribution shift, **balance bridge** | ✗ | ✗ | ✓ | yes | ✓ |
| `temporal_compare` | two-period metric comparison | ✗ | ✗ | ✓ | yes | ✓ |
| `scenario.apply_scenario` | run-rate what-if (**not** NNEG/HPI) | ✗ | ✗ | ✓ | yes | ✓ |

\* `snapshots.multidimensional` and `/mi/multidim` exist only on the parity
branch, not on `main`. Not counted.

### 2.4 Shared analytic primitives (`analytics_lib`)

| Module | Functions | Reachability |
|---|---|---|
| `history` | `portfolio_series`, `classify_exits`, `prepayment_rate`, `default_rate`, `cure_rate`, `loss_and_recovery`, `compare_cohorts` | gated tools |
| `contractual` | `schedule_frame`, `contractual_wal`, `portfolio_wal`, `contractual_ytm` | gated tools + `trakt_core.capability` (mounted, for determinism assessment) |
| `valuation_age` | `valuation_age_profile` | gated tools |
| `buckets` / `stratify` / `cohort` / `concentration` / `numeric` / `dates` | banding, stratification, point-in-time cohorts, top-N | **mounted** (used throughout the MI services) |
| `migration` | `transition_matrix`, `deterioration_flags` | **stub — raises `NotImplementedError`** |

`mi_agent.risk_monitor.migration.migration_matrix` / `per_loan_movement` are the
real implementations (the `analytics_lib.migration` stub documents that), reached
by `risk_monitor.monitor` and the gated `transition_analysis` tool.

---

## 3. Business concept taxonomy

Classified against the brief's families. Statuses: **SUPPORTED NOW** ·
**PARTIAL** · **DATA PRESENT, ANALYSIS NOT COMPOSED** · **LEGACY ONLY** ·
**NOT SUPPORTED**.

### A. Scale / stock
| Concept | Status | Where |
|---|---|---|
| Balance, count, average balance | SUPPORTED NOW | `compute_funded_snapshot`, capability registry `total_balance` / `loan_count` / `average_loan_size` |
| Composition by dimension | SUPPORTED NOW | 8 governed stratifications with per-dimension availability + reason |
| Composition by constituent portfolio | SUPPORTED NOW | `deck_context` type slices; `portfolio_scope` |
| **Stock by portfolio over time** | **DATA PRESENT, ANALYSIS NOT COMPOSED** | see §5 |

### B. Flow / movement
| Concept | Status | Where |
|---|---|---|
| Net change | SUPPORTED NOW | `funded_bridge.netChange`, `monthly_change` |
| New loans (count + balance) | SUPPORTED NOW | `balance_bridge.new_loan_closing_balance` |
| Exits (count + balance) | SUPPORTED NOW | `balance_bridge.exited_loan_opening_balance`; `exited_loans` KPI |
| **Exit *reason* split** | SUPPORTED NOW (gated) | `classify_exits` → redemption / default / maturity / unknown |
| Movement on continuing loans | SUPPORTED NOW | `balance_bridge.movement_on_continuing_loans` |
| Interest accrual / roll-up **as its own leg** | **NOT SUPPORTED** | inseparable from continuing movement without transactions — see §4 |
| Scheduled amortisation | PARTIAL | `contractual.schedule_frame` derives contractual principal; not reconciled to observed movement |
| Acquisitions as a distinct leg | DATA PRESENT, ANALYSIS NOT COMPOSED | provenance columns are on the bridge key; the bridge is not run per type |
| Prepayment rate (SMM/CPR) | SUPPORTED NOW (gated) | `prepayment_rate` |
| Maturities | SUPPORTED NOW (gated) | `classify_exits` maturity bucket |
| Restatements / adjustments | NOT SUPPORTED | no restatement concept anywhere |
| Distribution shift between periods | SUPPORTED NOW | `period_change.distribution` + `ranking` |

### C. Credit / performance
| Concept | Status | Where |
|---|---|---|
| Arrears stock | SUPPORTED NOW | `_risk_tile` (mounted); `arrears_stock` capability |
| Arrears 30+/90+ bands | SUPPORTED NOW (gated) | capability registry `arrears_30_plus` / `arrears_90_plus` |
| Arrears transition / roll rates | SUPPORTED NOW (gated) | `arrears_transition`; `risk_monitor.migration` |
| Default rate | SUPPORTED NOW (gated) | `default_rate` |
| Cure rate | SUPPORTED NOW (gated) | `cure_rate` |
| Loss / recovery / severity | SUPPORTED NOW (gated) | `loss_and_recovery` |
| **NNEG current exposure** | SUPPORTED NOW | `_risk_tile`; `evolution._nneg_metrics` |
| **NNEG projected / HPI stress** | **NOT SUPPORTED** | see §7 |
| Migration matrices | SUPPORTED NOW (gated) | `risk_monitor.migration` |

### D. Return / economics
| Concept | Status | Where |
|---|---|---|
| WA coupon / rate | SUPPORTED NOW | snapshot KPI; `wa_coupon` capability |
| Contractual WAL | SUPPORTED NOW (gated), **conditional on determinism** | `contractual_wal` |
| Contractual YTM | SUPPORTED NOW (gated), conditional | `contractual_ytm` |
| Expected WAL | MODEL_REQUIRED by declaration | `expected_wal` capability |
| Accrued interest as a measure | NOT SUPPORTED | field exists; no analytic |
| Cash yield / effective yield | NOT SUPPORTED | — |
| Expected cashflows | PARTIAL | `schedule_frame` builds contractual schedules; no portfolio cashflow view |
| Run-off | SUPPORTED NOW, **config-dependent** | `portfolio_projections` applies a *client-supplied* governed run-off curve; Trakt models none |

### E. Collateral / security
| Concept | Status | Where |
|---|---|---|
| LTV, WA LTV, high-LTV exposure | SUPPORTED NOW | snapshot, capability registry, `trakt_core.valuation` |
| LTV banding + governed ladder | SUPPORTED NOW | `config/mi/buckets.yaml` |
| Valuation age / method quality | SUPPORTED NOW (gated) | `valuation_age_profile` |
| LTV migration | SUPPORTED NOW (gated) | `risk_monitor.migration` over `ltv_bucket` |
| HPI index / property revaluation | **NOT SUPPORTED** | no index anywhere; the only "hpi" token is a covenant keyword class |

### F. Origination
| Concept | Status | Where |
|---|---|---|
| Pipeline stock, stage, completion month | SUPPORTED NOW | `compute_pipeline_snapshot` |
| Stage conversion + lag + sufficiency | SUPPORTED NOW | `pipeline_funnel_evolution` |
| Channel / broker | SUPPORTED NOW | breakdowns |
| Forecast bridge | SUPPORTED NOW | `compute_forecast_bridge` |
| Time to scale | SUPPORTED NOW | `build_extrapolation` milestone ladder |

### G. Seasoning / cohort
| Concept | Status | Where |
|---|---|---|
| Vintage composition | SUPPORTED NOW | `cohort_analysis` |
| Static pools, survival, retention, exits | SUPPORTED NOW | `funded_cohort_progression` |
| Cohort performance comparison | SUPPORTED NOW (gated) | `compare_cohorts` |
| Governed seasoning axis (front/back book, lending windows) | SUPPORTED NOW | `mi_agent/seasoning.py` + `config/mi/buckets.yaml` `seasoning:` |
| Cohort migration | PARTIAL | `risk_monitor.migration` can take a cohort mask; not wired |

### H. Concentration / limits
| Concept | Status | Where |
|---|---|---|
| Current / expected / stress, headroom, RAG | SUPPORTED NOW | `compute_concentration_tests` |
| Breach horizon, emerging risks, pipeline drivers | SUPPORTED NOW | `concentration_tests.forward` |
| Movement across snapshots | SUPPORTED NOW | `compute_history` (React only) |

### I. Data / change control
| Concept | Status | Where |
|---|---|---|
| Canonical completeness | SUPPORTED NOW (gated) | `data_completeness` |
| Regulatory readiness | SUPPORTED NOW (gated) | `regulatory_readiness` |
| Validation exceptions | SUPPORTED NOW (gated) | `list_validation_exceptions` |
| Per-dimension chart availability + reason | SUPPORTED NOW | stratification `availability` / `reason` |
| **Restatement / prior-period correction** | **NOT SUPPORTED** | no concept |

---

## 4. Asset-agnostic vs asset-specific

**Asset-agnostic primitives** (work on any book meeting the data conditions):
`stratify`, `buckets`, `cohort`, `concentration`, `numeric`, `dates`,
`portfolio_series`, `classify_exits`, `balance_bridge`, `distribution_change`,
`funded_bridge`, `migration_matrix`, `cohort_analysis`,
`funded_cohort_progression`, `compute_concentration_tests`, the whole
`forecast_extrapolation` ladder, and `presentation`-layer banding.

**Asset-specific implementations** — and there are only three, all narrow:

| Adapter | Mechanism | Where |
|---|---|---|
| Risk tile | `portfolio_risk_type(df) == "erm"` → NNEG exposure, else arrears balance, else controlled "unavailable" naming the missing fields | `snapshots._risk_tile` |
| NNEG series | emitted only when balance and valuation both present | `evolution._nneg_metrics` |
| Contractual determinism | RREL35 amortisation type: BLLT/FIXE rate-independent → AVAILABLE; FRXX/DEXX on a floating rate → ASSUMPTION_REQUIRED; `OTHR` (what a lifetime mortgage reports) → NOT_APPLICABLE with an economic explanation | `trakt_core.capability._principal_determinism` |

**This is the pattern the brief asks for, already implemented.** None of the
three tests an asset-class label. Each tests an economic condition — is there a
valuation, is there an arrears field, what does the contract say about principal
— and the asset class falls out of the data. The lifetime-mortgage case is
handled by saying *"repayment is contingent on death, sale or long-term care, so
no contractual repayment date exists"*, not by branching on a product name.

**Asset applicability is declared at field level** in the BSR:

| `asset_applicability` | Fields |
|---|---:|
| `cross_asset` | 190 |
| `sme` | 18 |
| `equity_release` | 14 |
| `commercial_real_estate` | 10 |
| `equipment_leasing` | 7 |
| `residential_mortgage` | 4 |

**Caveat, stated plainly.** `config/asset/product_profiles.yaml` contains exactly
**one** profile (`equity_release_lifetime_mortgage`), and it governs *onboarding*
field relaxation, not MI read-time capability. So the asset-class **model** is
general; the asset-class **population** is one product deep. Bridge, auto and
equipment finance are represented in the BSR field vocabulary and in the ESMA
canonical model, and nowhere else. Nothing should claim Trakt "supports" them
analytically today.

---

## 5. Funded balance stock and movement

### 5.1 The economic identity — what reconciles today

Probed against a synthetic two-period book (5 opening loans across two
portfolios; one redeemed with a `loan_redemption_flag`, one exited with a
`default_date`, one new loan, 1.5 % roll-up on the three continuing loans):

```
opening_balance                1,080,000.00
new_loan_closing_balance         220,000.00
exited_loan_opening_balance      380,000.00
movement_on_continuing_loans      10,500.00
closing_balance                  930,500.00
identifier_field   source_portfolio_id + loan_identifier
reconciles         True
residual           −1.16e-10

IDENTITY  1,080,000 + 220,000 − 380,000 + 10,500 = 930,500  ✓
```

And the exit leg decomposes, composing exactly with the bridge:

```
classify_exits   evidence = [default_date, loan_redemption_flag]
   redemption     {balance: 200,000, loan_count: 1}
   default_exit   {balance: 180,000, loan_count: 1}
   maturity       {balance:       0, loan_count: 0}
   unknown_exit   {balance:       0, loan_count: 0}
   sum = 380,000  ==  bridge exited leg 380,000  ✓
```

### 5.2 Component table

| Net-change component | Existing capability | Source | Exact / inferred | Asset applicability |
|---|---|---|---|---|
| Opening balance | `balance_bridge.opening_balance` | `period_change/bridge.py` | **exact** | any |
| New funded loans | `new_loan_closing_balance` + `new_loan_count` | same | **exact** (loan identity) | any |
| Acquired balances *as a separate leg* | not separated | — | derivable by running the bridge per `source_portfolio_type` | any multi-book |
| Net accrued / rolled-up interest | inside `movement_on_continuing_loans` | same | **inferred**; equals accretion only where nothing else moves a continuing balance | roll-up books only |
| Redemptions | `classify_exits[redemption]` | `analytics_lib/history.py:191` | **exact where evidence exists**, else `unknown_exit` | any with `loan_redemption_flag` |
| Repayments / prepayments (rate) | `prepayment_rate` (SMM/CPR) | `analytics_lib/history.py:353` | exact rate, from classified exits | amortising books |
| Scheduled amortisation | `contractual.schedule_frame` | `analytics_lib/contractual.py` | contractual, **not** reconciled to observed | deterministic-principal books |
| Maturities | `classify_exits[maturity]` | same | exact where `maturity_date` present | dated books |
| Write-offs / defaults out | `classify_exits[default_exit]` | same | exact | any with default evidence |
| Disposals / transfers | not distinguished | — | falls to `unknown_exit` | — |
| Adjustments / restatements | **none** | — | — | — |
| Closing balance | `balance_bridge.closing_balance` | same | **exact** | any |

### 5.3 Answers to the brief's eight questions

1. **Which components can the engine identify?** Opening, new, exited,
   continuing-movement, closing — exactly. Exits then split four ways.
2. **Which are explicit data fields?** `loan_redemption_flag`, `default_date`,
   `maturity_date`, `account_status`, plus **`redemptions_received_in_period`**,
   `cumulative_prepayments`, `cumulative_recoveries`, `allocated_losses` — 64 of
   the 499 canonical fields are credit/exit related. Note the flow fields exist in
   the registry but **no analytic consumes them**: `redemptions_received_in_period`
   would give a *reported* redemption leg rather than an inferred one.
3. **Which come from comparing snapshots?** All of the bridge, and every
   `classify_exits` bucket.
4. **Which need transaction-level cashflow data?** Splitting
   `movement_on_continuing_loans` into accrual vs repayment vs further advance.
   Nothing in the canonical model carries per-loan period movement.
5. **Does `funded_bridge` provide this?** No — different tool. `funded_bridge` is a
   *dimensional* attribution (which regions/brokers moved), reconciling to net
   change. `balance_bridge` is the *economic* one. Both reconcile; they answer
   different questions and should both appear.
6. **Does cohort logic identify exits?** Yes. `funded_cohort_progression` emits
   `exitsCount`, `loanRetention`, `balanceRetention`, `survivingLoanIds` per
   period per cohort — a static-pool exit view independent of the bridge.
7. **Is accrual measurable?** Only as the residual `movement_on_continuing_loans`.
   For a roll-up book with no scheduled repayment and no further advances that
   *is* accretion, but the engine cannot prove that from a snapshot pair.
   Reporting it as "interest" would be an asset-class assumption dressed as a
   measurement.
8. **Does it reconcile exactly?** Yes — proven above, residual 1e-10 against a
   documented tolerance, with the bridge refusing to report at all on duplicate
   or missing identifiers, mixed currency, or a missing balance field.

### 5.4 Stacked evolution by constituent portfolio

- **Does `funded_evolution` retain the portfolio dimension?** The *frames* do —
  `funded_frames` returns prepared frames carrying `source_portfolio_id` /
  `_type` / `_label`. The *emitted* breakdowns do not: `_FUNDED_BREAKDOWN_DIMS`
  is `{broker, region, ltv_bucket}`.
- **Does another primitive produce period × portfolio balances?** Yes — the same
  one. Probed by calling `evolution._breakdown(frame, "source_portfolio_id")`
  directly on the frames the service already builds:

```
2026-05-31  [{acquired_001: 430,000}, {direct_001: 650,000}]  sum = 1,080,000
2026-06-30  [{acquired_001: 253,750}, {direct_001: 676,750}]  sum =   930,500
```

- **Is the dimension discarded downstream?** Yes, twice over: it is never
  requested as a breakdown, and the `breakdowns` block that *is* produced is
  rendered by nothing.
- **Can it reconcile every period?** Proven — all three emitted dimensions
  reconciled to their period total on both periods, and `_breakdown` routes
  blanks to an explicit `Unknown / Missing` bucket precisely so they do.
- **Can the bridge be decomposed by portfolio?** Yes, without a new primitive:
  the bridge key is already composite (`source_portfolio_id + loan_identifier`),
  so running it per scope gives per-book legs that sum to the total.

**Verdict:** stacked-stock-by-portfolio and its reconciling movement bridge are
both **composition of existing outputs**. No new primitive.

---

## 6. Seasoning / availability model

The engine already distinguishes availability states properly. `trakt_core.capability`
publishes six: `AVAILABLE`, `UNAVAILABLE`, `NOT_APPLICABLE`, `ASSUMPTION_REQUIRED`,
`MODEL_REQUIRED`, `METHODOLOGY_NOT_APPROVED`. Its header says why the distinction
is load-bearing: *"'This portfolio has no WAL', 'Trakt lacks a field', 'WAL is
conceptually inappropriate' and 'WAL would require assumptions' are four different
findings."*

### Mapping to the brief's availability classes

| Class | Mechanism today | Capabilities |
|---|---|---|
| IMMEDIATE (1 snapshot) | `fields_present` | 15 of 28 — balance, count, LTV, arrears stock, coupon, valuation age, WAL/YTM, completeness |
| TWO-PERIOD | `history_periods: 2` | 8 — arrears transition, cure, SMM, CPR, default rate, loss/recovery/severity, `portfolio_history` |
| MULTI-PERIOD | same condition, higher minimum | supported by the mechanism; no capability currently sets >2 |
| SEASONED | **no dedicated condition** | approximated by history count + the governed seasoning axis (`front_book_max_months: 12`, lending windows 1m/3m, bands 0-12m/13-24m/25-60m/60m+) |
| **EVENT-DEPENDENT** | **not expressible** | see gap below |
| CONFIG-DEPENDENT | not a condition type; handled inside services | concentration (approved config), run-off (client curve), scale targets |

### The one real gap

`_evaluate_condition` implements five condition types: `fields_present`,
`history_periods`, `collateral_present`, `principal_series_deterministic`,
`interest_series_deterministic`. There is **no event-count condition** — nothing
can say "this needs at least one observed default", or "at least N exits", or
"at least one cohort with two periods". An unknown condition type fails closed
rather than passing silently, so adding one is a resolver change, not a config
change.

### NNEG, distinguished as the brief asks

| State | Is it expressible today? |
|---|---|
| Capability unavailable (no valuation) | **Yes** — `_risk_tile` returns a controlled "Unavailable" naming the missing fields; `collateral_present` returns `NOT_APPLICABLE` |
| Available but genuinely zero | **Yes** — the measure computes and returns 0 |
| Zero because the book is young | **No** — nothing distinguishes "no loan is yet above valuation" from "this book has not existed long enough for one to be" |
| Meaningful stress exposure despite zero today | **No** — no stress capability exists at all (§7) |

The third and fourth are the ones a funder actually asks about on a young
equity-release book, and neither is answerable today.

---

## 7. Equity-release capability

| Concept | Status | Reachability | Evidence |
|---|---|---|---|
| **Current NNEG exposure** | SUPPORTED NOW | React tile + deck KPI | `snapshots._risk_tile` — Σ(balance − valuation) where balance > valuation |
| **NNEG as a time series** | SUPPORTED NOW | React cohort progression + deck cohort slide | `evolution._nneg_metrics` → `nneg_exposure`, `nneg_headroom`, `nneg_headroom_pct` per cohort period |
| Interest roll-up | PARTIAL | — | observable as `movement_on_continuing_loans`; not labelled as accretion |
| Static-pool exits / survival | SUPPORTED NOW | React + deck | `funded_cohort_progression` |
| Redemption classification | SUPPORTED NOW | gated tools | `classify_exits`; `loan_redemption_flag` is BSR-tagged `equity_release`, concept `cashflow` |
| No contractual maturity handled correctly | SUPPORTED NOW | mounted | `_principal_determinism` → `NOT_APPLICABLE` with the contingent-repayment explanation |
| **Projected NNEG** | **NOT SUPPORTED** | — | nothing anywhere |
| **HPI / property-value projection** | **NOT SUPPORTED** | — | no index; the only `hpi` token in live code is a covenant keyword class in `concentration_query.py:48` |
| **Mortality / move-to-care** | **NOT SUPPORTED** | — | declared `MODEL_REQUIRED` by the capability registry, implemented nowhere |
| Voluntary prepayment | SUPPORTED NOW (gated) | tools | `prepayment_rate` |
| Sale costs | NOT SUPPORTED | — | no field, no analytic |
| Scenario projection | **NOT for NNEG** | mounted (chat) | `scenario.apply_scenario` is a *run-rate* what-if for time-to-scale. It has no collateral, valuation or mortality concept. |

**Answering the brief's six questions:**

1. **Is the scenario/NNEG capability production-reachable?** The current-NNEG
   measure, yes (mounted). A *scenario* NNEG capability does not exist to be
   reachable.
2. **Current architecture or legacy?** Current. `_risk_tile` and `_nneg_metrics`
   are in the live MI services. The legacy Streamlit estate has its own NNEG
   distribution chart (`analytics/generate_pptx_client.py:save_nneg_distribution`)
   which is a separate, unreachable implementation and should not be migrated.
3. **Canonical inputs needed?** For current NNEG:
   `current_outstanding_balance` + `current_valuation_amount`, both canonical.
   For projected NNEG: an HPI path, a mortality/exit model and a roll-up rate —
   none present.
4. **Can it produce a useful seasoned-book analysis today?** Partly. NNEG
   headroom per vintage over time is genuinely useful on a seasoned book and is
   already computed. What it cannot do is say what happens if house prices fall.
5. **Currently exposed?** Yes — React cohort progression metric picker (gated on
   `hasNneg`), and the deck's cohort progression metric list.
6. **Routing or new MI?** Surfacing NNEG *headroom trend* more prominently is
   routing. Projected NNEG or HPI stress is **new MI** — and specifically the kind
   the capability registry classifies `MODEL_REQUIRED`. Out of scope.

---

## 8. Conventional credit capability

**FIELD EXISTS ≠ ANALYTIC EXISTS**, and the split is stark here.

**Fields:** 64 of 499 canonical fields cover credit performance, including
`arrears_balance`, `number_of_days_in_arrears` (plus principal/interest
variants), `date_last_in_arrears`, `loan_entered_arrears`, `account_status`,
`default_date`, `default_amount`, `exposure_at_default`, `loss_given_default`,
`probability_of_default`, `allocated_losses`, `recoveries_in_period`,
`cumulative_recoveries`, `date_of_breach_cure`, `cure_payments_possible`,
`non_recoverability_determined`, `special_servicing_status`.

**Analytics:**

| Analytic | Exists | Reachability |
|---|---|---|
| Arrears stock | ✓ | **mounted** (`_risk_tile`) |
| Arrears 30+ / 90+ bands | ✓ | gated (`arrears_30_plus` / `arrears_90_plus`) |
| Arrears transition / roll rate | ✓ | gated (`arrears_transition`, `transition_analysis`) |
| Default rate | ✓ | gated (`default_analysis`) |
| Cure rate | ✓ | gated (`cure_analysis`) |
| Loss, recovery, severity | ✓ | gated (`loss_analysis`) |
| Migration matrix / per-loan movement | ✓ | gated (`transition_analysis`) + `risk_monitor` |
| Delinquency *curves* by vintage | ✗ | `compare_cohorts` could carry the measures; nothing composes the curve |
| Roll-rate *matrix* presentation | ✗ | matrix computes; nothing renders it |

So the conventional-credit analytic library is **essentially complete and
essentially unsurfaced** — one measure (arrears stock) is mounted, the rest sit
behind a deployment flag, and none reaches the deck.

---

## 9. Cashflow / WAL / return capability

| Capability | Production path | Deterministic | Asset classes | Sufficient for deck/dashboard | Surfaced |
|---|---|---|---|---|---|
| Contractual schedules (`schedule_frame`) | gated tool | yes | any with RREL35 + dates | yes | ✗ |
| Contractual WAL | gated tool; **determinism assessed on the mounted path** via `trakt_core.capability` | yes | BLLT/FIXE always; FRXX/DEXX only if rate fixed for life; `OTHR` → NOT_APPLICABLE | yes | ✗ |
| Contractual YTM | gated tool | yes | needs principal **and** interest determinism | yes | ✗ |
| Expected WAL | declared `MODEL_REQUIRED` | — | — | — | ✗ |
| WA coupon | mounted | yes | any with a rate | yes | ✓ (snapshot KPI) |
| Maturity profile | ✗ | — | — | — | ✗ |
| Accrued interest | ✗ (field only) | — | — | — | ✗ |
| Cash yield / effective yield | ✗ | — | — | — | ✗ |
| Redemption proceeds | ✗ as a measure | `redemptions_received_in_period` exists as a field | — | — | ✗ |
| Run-off projection | **mounted** (`portfolio_projections`) | yes | any book with a client-supplied curve | yes | **✗ — computed and served, rendered by nothing** |

`analytics_lib.contractual` explicitly refuses where the contract stops
determining the answer, with a four-state vocabulary (OBSERVED / CONTRACTUAL /
ASSUMPTION_REQUIRED / MODEL_REQUIRED). That refusal discipline is the reason a
lifetime-mortgage book gets "no contractual life" rather than a fabricated one.

---

## 10. Existing capabilities currently not surfaced — top 10

Ranked by value to lender management, funder, credit-committee and surveillance
reporting. A mechanical sweep found **65 emitted payload keys rendered by neither
React nor the deck**; these are the ones that matter.

| # | Capability | Where it is computed | Surfaced? | Why it matters | Work class |
|---|---|---|---|---|---|
| 1 | **Reconciled economic bridge** — opening / new / exited / continuing / closing with residual | `period_change.bridge.balance_bridge` | chat only | The single question every funder asks: *why did the balance change?* Reconciles exactly. | B |
| 2 | **Funded balance by constituent portfolio over time** | `evolution._breakdown` on frames that already carry provenance | ✗ | Where the balance sits, per book, reconciling every period | B |
| 3 | **`funded_evolution.breakdowns`** (broker / region / LTV × period) | emitted on every call; typed in React | **✗ rendered by nothing** | Composition drift over time — already reconciles | A |
| 4 | **`portfolioProjections`** — per-book current, expected originations, retention factor, projected balance, run-off disclosures | `forecast_bridge.portfolio_projections`, served at `/mi/forecast/snapshot` | **✗ rendered by nothing** | Forward view per book, with an explicit statement of where run-off is *not* modelled | A |
| 5 | **Exit reason split** — redemption / default / maturity / unknown | `classify_exits` | gated | Turns "balance fell" into "redeemed vs defaulted", and flags unexplained exits as data quality | C→B |
| 6 | **Concentration movement** — utilisation per test across snapshots | `compute_history`, `/mi/concentration-tests/history` | React only | Direction of travel on covenants, not just the level | A |
| 7 | **Arrears / default / cure / loss suite** | `analytics_lib.history` via gated tools | gated | The entire conventional-credit story for non-ERM books | C |
| 8 | **Contractual WAL + determinism verdict** | `analytics_lib.contractual` + `trakt_core.capability` | gated | Funders ask for WAL; the honest "not applicable, and here is why" is itself valuable | C |
| 9 | **Pipeline lifecycle components** — new / removed / progressed_out / increased / decreased / unchanged, exhaustive to the headline | `movement_detail` | React drawer only | Explains weekly pipeline movement; absent from the deck | A |
| 10 | **Distribution shift + ranked movers between periods** | `period_change.distribution` + `ranking` | chat only | "Which segments moved most" — a surveillance staple | B |

Runners-up worth noting: `valuation_age_profile` (collateral evidence quality),
`weeklyConversionRate` / `kfiStockNow` / `monthlyKfiInflow` (conversion detail
computed and dropped), `survivingBalance` / `monthsOnBook` / `formationPeriod`
(cohort detail), `fiveWeekAverage`.

---

## 11. Concept-library architecture — does it already exist?

**Verdict: A, with a little of B. Largely documentation and configuration of an
architecture that already exists; a modest consolidation of two fragments. Not
new architecture.**

Mapping the brief's proposed model onto what is there:

| Brief's concept | Already exists as | Owner |
|---|---|---|
| `portfolio_stock` | BSR `exposure` (19 fields) + capability `exposure` category (5) | BSR + capability registry |
| `portfolio_flow` | BSR `cashflow` (22) + capability `prepayment` (2) | BSR + capability registry |
| `credit_performance` | BSR `credit_quality` (49) + `payment_performance` (14) + `loss` (21); capability `delinquency` (5) + `loss` (4) | both |
| `collateral_risk` | BSR `collateral` (29) + `valuation` (17) + `leverage` (13); capability `collateral` (4) | both |
| `return_economics` | BSR `pricing` (13) + `maturity` (12); capability `pricing` (1) + `cashflow` (3) | both |
| `seasoning` | `mi_agent/seasoning.py` + `config/mi/buckets.yaml` `seasoning:`; capability `history` (2) | seasoning module |
| `origination` | BSR `origination` (4); `mi_workflows` `pipeline_stock`, `pipeline_completion_forecast` | BSR + analytical registry |
| `forecast` | BSR `forecast` (1); `mi_workflows` `funded_balance_forecast`, `completion_run_rate`, `threshold_projection` | analytical registry |
| `concentration` | `config/risk/concentration_test_library.yaml`; `mi_workflows.concentration_limits` | covenant library |

And the per-concept attributes the brief wants:

| Attribute | Exists? | Where |
|---|---|---|
| Common definition | ✓ | BSR `display_name` + `rationale`; capability `description` |
| Asset applicability | ✓ | BSR `asset_applicability` (6 classes) |
| Required canonical fields | ✓ | capability `conditions.fields_present` |
| Minimum history | ✓ | capability `conditions.history_periods` |
| Implementation | ✓ | capability `calculation_source`; analytical registry `engine` |
| Presentation availability | **✗** | capability `consumers` is a **declaration**, not enforced or consumed |

### The smallest consolidation

Three gaps, none of them architectural:

1. **`consumers` is fiction.** The registry declares
   `consumers: [react, mi_query, copilot, agent_tools, readiness]` on nearly every
   capability. React panels and the deck consume the registry **not at all** —
   React's `PortfolioCapability` is a different concept entirely (which portfolios
   contribute to a view). Either enforce the field or stop asserting it.
2. **No event-count condition.** Add one condition type and SEASONED /
   EVENT-DEPENDENT become expressible; today they are not.
3. **Two applicability models that do not know about each other.** The onboarding
   product profile (one profile, field relaxation) and the MI capability registry
   (28 capabilities, read-time conditions) never meet. They need not merge — they
   answer different questions at different times — but nothing currently maps
   between them.

**Do not build a new registry.** Three exist and each is authoritative for its own
thing, by explicit design.

---

## 12. PPTX / dashboard implications

Existing capability that could enrich each generic module, with no new MI:

| Module | Capability | Recommendation |
|---|---|---|
| **EXECUTIVE** | headline stock, pipeline, forecast, concentration status | already assembled on the parity branch — CORE |
| | net movement headline from `balance_bridge` | **CORE** where two periods exist |
| **FUNDED STOCK** | 8 governed stratifications with availability + reason | CORE |
| | **stock by constituent portfolio over time** | **CORE where >1 portfolio** |
| | `funded_evolution.breakdowns` (composition drift) | CONDITIONAL — ≥3 periods |
| **FUNDED MOVEMENT** | `balance_bridge` economic bridge | **CORE where the bridge reconciles**; omit with its own reason where it does not |
| | `classify_exits` exit-reason split | CONDITIONAL — needs exit evidence fields |
| | `funded_bridge` dimensional attribution | CORE (already present) |
| | `period_change.distribution` ranked movers | DEEP DIVE |
| **PERFORMANCE / HEALTH** | arrears stock | ASSET-SPECIFIC (conventional credit) + CONDITIONAL |
| | default / cure / loss / transition | ASSET-SPECIFIC + SEASONED ONLY + currently gated |
| | `valuation_age_profile` | DEEP DIVE |
| **SEASONING** | vintage composition | CONDITIONAL — ≥2 vintages |
| | static-pool progression, retention, exits | SEASONED ONLY — ≥2 periods per cohort |
| | NNEG headroom by vintage | ASSET-SPECIFIC (ERM) + SEASONED ONLY |
| **ASSET-SPECIFIC RISK** | NNEG current + headroom trend | ASSET-SPECIFIC, CONDITIONAL on valuation |
| | arrears / delinquency | ASSET-SPECIFIC, CONDITIONAL on arrears fields |
| **FORECAST** | bridge, run-rate, milestone ladder | CORE (present) |
| | **`portfolioProjections`** per-book forward view | **CONDITIONAL — CORE where >1 portfolio** |
| | actual vs prior forecast | CONDITIONAL — ≥3 forecast runs |
| **CONCENTRATION** | current / expected / stress / headroom / RAG | CORE (present) |
| | `compute_history` movement | CONDITIONAL — ≥2 snapshots |
| | emerging risks, breach horizon | CONDITIONAL |

**The shared object is the business question, not the measure.** The
`_risk_tile` adapter already demonstrates this: *"what is the credit risk on this
book?"* resolves to NNEG exposure on an ERM book, arrears balance on an
amortising one, and a controlled "unavailable" naming the missing fields
otherwise. A performance module built on that pattern needs no asset-class branch
in the deck.

---

## 13. Capability matrix

**A** already computed, just surface it · **B** composition of existing outputs ·
**C** exists but not production-reachable (deployment flag) · **D** requires new
analytical capability.

| Candidate enhancement | Class | Note |
|---|:--:|---|
| Render `funded_evolution.breakdowns` | **A** | computed every call, typed in React, drawn nowhere |
| Render `portfolioProjections` | **A** | served at `/mi/forecast/snapshot`, drawn nowhere |
| Concentration movement in the deck | **A** | `compute_history`, already in React |
| Pipeline lifecycle components in the deck | **A** | already in React's drawer |
| Conversion detail (`weeklyConversionRate`, KFI stock/inflow) | **A** | computed and dropped |
| Economic bridge (`balance_bridge`) on both surfaces | **B** | reconciles today; needs a route + a renderer |
| Funded stock stacked by portfolio | **B** | register the dimension the frames already carry |
| Bridge decomposed per constituent portfolio | **B** | composite key already supports it |
| Exit-reason split beside the bridge | **B/C** | `classify_exits` is reachable via a gated tool |
| Arrears / default / cure / loss suite | **C** | `TRAKT_AGENT_API_ENABLED` is off by default |
| Contractual WAL + determinism verdict | **C** | same gate |
| Reported redemption leg from `redemptions_received_in_period` | **D** | field exists, no analytic reads it |
| Accrual split out of continuing movement | **D** | needs transaction-level data |
| Projected NNEG / HPI stress | **D** | `MODEL_REQUIRED`; no index, no mortality model |
| Restatement / prior-period correction | **D** | no concept anywhere |
| Delinquency curves by vintage | **D** | measures exist; the curve composition does not |

**Nine of sixteen are A or B.** Four are D and should stay out.

---

## 14. Recommendation

**The smallest sprint that materially upgrades the product, reusing existing
capability.** Roughly a week, no new MI primitive.

**1. Surface what is already served (class A) — 2 days.**
Render `funded_evolution.breakdowns` as composition-over-time; render
`portfolioProjections` as the per-book forward view with its run-off
disclosures; put concentration movement and the pipeline lifecycle components in
the deck. Every one of these is a payload the API already returns.

**2. The economic bridge, on both surfaces (class B) — 2 days.**
Expose `balance_bridge` through a governed route and render it as a waterfall:
opening → new → exits → movement on continuing → closing, with the residual
stated. It reconciles today. Sit it beside the existing dimensional bridge, which
answers a different question, and label both.

**3. Funded stock by constituent portfolio (class B) — 1 day.**
Register the portfolio dimension in `_FUNDED_BREAKDOWN_DIMS` and render the
stacked series. The frames already carry provenance; the breakdown already
reconciles.

**4. One decision, not code: the tool-surface gate.**
The entire conventional-credit analytic library — arrears bands, transitions,
default, cure, loss, severity, prepayment, WAL — is written, tested and deployed,
behind `TRAKT_AGENT_API_ENABLED`. Whether to mount it is a deployment and
governance decision, not an engineering one. It should be taken deliberately
rather than left at its default.

**Explicitly not in this sprint:** projected NNEG, HPI stress, mortality,
accrual separation, restatements, delinquency curves. All are class D.

---

## Final questions, answered

**1. Can funded balance evolution already be shown as a stacked series by
portfolio?** Not as shipped — `_FUNDED_BREAKDOWN_DIMS` carries broker, region and
LTV band only. But the primitive that would produce it is the one already
running: calling `evolution._breakdown(frame, "source_portfolio_id")` on the
frames the service already builds returned a reconciling per-book series on both
probe periods. **Composition, not new MI.**

**2. Can existing capabilities explain net funded balance change as new +
acquisitions + interest − redemptions ± adjustments?** Partially, and precisely
where you would expect. New loans, exits and closing: exactly. Acquisitions:
derivable by scoping the bridge per portfolio type. Interest/accretion: only as
the residual `movement_on_continuing_loans`, which equals accretion **only** on a
book where nothing else moves a continuing balance — an asset-class assumption,
not a measurement. Adjustments/restatements: no concept exists.

**3. How much of that bridge reconciles exactly today?** The four-component
identity, exactly, with a checked residual: `1,080,000 + 220,000 − 380,000 +
10,500 = 930,500`, residual `−1.16e-10`. The exit leg then splits four ways and
sums back to `380,000` exactly. The bridge refuses to report at all on duplicate
identifiers, missing identifiers, mixed currency or a missing balance field —
which is why the reconciliation can be trusted.

**4. What exists for redemptions / exits / prepayment?** `classify_exits`
(redemption / default / maturity / unknown, on evidence, refusing to call an
unexplained disappearance a prepayment); `prepayment_rate` (SMM/CPR);
`funded_cohort_progression` (`exitsCount`, `loanRetention`, `balanceRetention`);
the `exited_loans` KPI on the mounted snapshot; and `portfolio_projections`
applying a **client-supplied** governed run-off curve. Trakt models no run-off of
its own and says so.

**5. What exists for arrears / default / delinquency?** Arrears stock is mounted.
Arrears 30+/90+, arrears transitions, default rate, cure rate, loss, recovery,
severity and migration matrices all exist, are tested and are deployed — behind
the agent-API flag. 64 canonical fields back them.

**6. What exists for NNEG / HPI / roll-up stress?** Current NNEG exposure
(mounted, both surfaces) and NNEG exposure/headroom as a per-cohort time series
(mounted, both surfaces). **Nothing** for projected NNEG, HPI, property-value
projection, mortality or move-to-care. The `scenario` module is a run-rate
what-if for time-to-scale and has no collateral concept.

**7. What exists for WAL / cashflows / yield?** `analytics_lib.contractual`:
contractual schedules, contractual WAL, portfolio WAL, contractual YTM — with a
four-state refusal vocabulary. Reachable via a gated tool. The *determinism
verdict* is assessed on the mounted path by `trakt_core.capability`, which is how
a lifetime mortgage correctly gets "no contractual life" rather than a fabricated
one. No accrued interest, cash yield or maturity-profile analytic.

**8. Which are current production MI versus legacy?** Everything cited in this
audit is current. The legacy Streamlit estate (`analytics/`) has its own NNEG
distribution, treemaps and bubble charts; it is a separate deployment
(`Dockerfile.streamlit` copies `analytics/` and `config/` only), shares no import
with the MI estate in either direction, and nothing here should be migrated from
it. `analytics_lib.migration` is a documented stub that raises; the working
implementation is `mi_agent.risk_monitor.migration`.

**9. What is computed today but discarded?** 65 payload keys reach neither
surface. The material ones: `funded_evolution.breakdowns`; the whole
`portfolioProjections` block (`currentBalance`, `expectedNewOriginations`,
`balanceRetentionFactor`, `projectedBalance`, `runoffModelled`,
`runoffNotModelled`, `runoffProfileId`, `totalProjectedBalance`,
`unattributedExpectedOriginations`); `weeklyConversionRate`, `kfiStockNow`,
`monthlyKfiInflow`, `weeklyKfiInflow`; `survivingBalance`, `monthsOnBook`,
`formationPeriod`; `fiveWeekAverage`; `weeksObservedByMetric`.

**10. Can one concept architecture support equity release, mortgages, bridge and
equipment finance without hard-coding PPTX by asset class?** Yes, and the
mechanism is already proven three times over — `_risk_tile`, `_nneg_metrics` and
`_principal_determinism` each select an asset-appropriate answer from an
*economic condition*, never an asset label. The capability registry states this as
its design principle. The constraint is not architectural: it is that only
equity release is populated as a product profile today, and only one asset class
has been onboarded.

**11. Is such a concept architecture mostly already present?** **Yes.** 243 BSR
fields carrying `analytical_concept` (24 concepts), `asset_applicability` (6 asset
classes) and `temporality`; 28 capabilities with six availability states and five
condition types; 10 composable analytical capabilities with route-ownership
deference. The gap is not the model — it is that React and the deck consume none
of it, while the registry declares that they do.

**12. Five highest-value improvements with no new MI primitive:**
1. the reconciled economic bridge, on both surfaces;
2. funded stock stacked by constituent portfolio, reconciling every period;
3. `portfolioProjections` — the per-book forward view already being served;
4. `funded_evolution.breakdowns` — composition drift over time, already computed;
5. concentration movement and the pipeline lifecycle components in the deck.

**13. What should appear only once a book has enough history/events?** Cohort
progression and retention (≥2 periods per cohort); arrears transitions, cure,
default, SMM/CPR, loss and severity (≥2 periods, and ≥1 observed event —
which the resolver **cannot currently express**); forecast-vs-actual (≥3 forecast
runs); composition drift (≥3 periods); the exit-reason split (≥1 classified
exit). The history conditions exist; the event conditions do not.

**14. If we refuse to build any new MI next sprint, how much better can the pack
become?** Materially. It gains the question a funder asks first — *why did the
balance change?* — answered with an exactly reconciling bridge; it gains *where
the balance sits, per book, over time*; it gains a forward view per constituent
portfolio with an honest statement of where run-off is not modelled; and it gains
the direction of travel on covenants. That is four new analytical stories, none of
which requires a line of new calculation. What it will still not do is stress
house prices, project NNEG, or separate interest accretion from repayment — and
those should stay out until someone decides they are worth a model.
