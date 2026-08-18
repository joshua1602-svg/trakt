# MI Query Agent — P1J: Breadth Gap Decomposition & Prioritisation

**Status:** analysis only. No production code changed, no semantics/aliases/routes
added, the 40-bank untouched. This report is the deliverable.

**Method:** every non-correct question was re-traced from the canonical fixture,
the governed registries, and the analytics modules — prior classifications were
not carried forward. Two independent repository audits (existing-analytics reuse;
governed catalogue/alias) backed the trace, and every "answerable from present
data" claim below is backed by a figure recomputed directly from the tape.

**Three business rulings were taken during this phase** (§0), because the evidence
left genuine product-semantic ambiguities that materially change the ceiling.

---

## 0. Rulings taken (so the prioritisation is not a guess)

| # | Ambiguity | Ruling |
|---|---|---|
| R1 | "Credit quality" on a fully-performing book (arrears/default/impairment all zero) | **Weighted-average current LTV** is the governed quality proxy (lower = better). |
| R2 | "New origination vs the back book" (B04) | **Recent vs older vintage** (an origination-date cutoff), *not* the direct/acquired split. B04 is therefore a **vintage-family** question. |
| R3 | "Product" / "by product" when governed `erm_product_type` is not populated on this book | **Safe refusal.** Governance holds: `product_type = erm_product_type`; `purpose` is loan-purpose, not product, and must not be substituted. Product questions stay blocked until the field is onboarded. |

---

## 1. Executive answer — why the 40-bank has barely moved

**The engineering of the last several phases was almost entirely safety and
semantic-identity work, not breadth.** P0–P1I hardened the boundary (fail-closed
guard, execution receipts, cohort/measure identity, scope resolution). Each phase
*removed a way to be wrong*, not *added a way to answer*. The 40-bank measures
breadth, so it barely moved — by design.

**The deeper finding, and the one that reframes the roadmap: the calculations the
remaining questions need overwhelmingly already exist in the repository — they are
simply not reachable through the governed MI query entrypoint.** The
existing-analytics audit found deterministic implementations of HPI stress, forward
projection-to-threshold (including a £100m milestone), ranked period movement,
per-limit concentration headroom, vintage/static-pool metrics, threshold-share
("eligibility"), largest-single-loan, weighted-average contribution, NNEG/underwater
exposure, mix-shift, and arrears share. Of the twelve analytic operations the bank
needs, **ten already exist**; only **HHI/diversification index** and
**convergence-over-time** are genuinely absent.

So the breadth gap is **not** "Trakt cannot compute these". It is three separable
things, in rough order of size:

1. **Exposure / routing** — an existing analytic is not wired to the MI query path,
   or a phrasing routes to the wrong path. This is most of the gap and the cheapest
   to close.
2. **A governed derivation not materialised on this book** — the concept is governed
   and its source column is present, but the derived field (e.g. `vintage_year` from
   `origination_date`) is not produced. One derivation unlocks several questions.
3. **Genuinely absent source data** — broker identity, `erm_product_type`, and the
   borrower-structure source (`number_of_borrowers` / `borrower_2_dob`) are simply
   not in this book. These must stay safe refusals.

The 40-bank was also written to *stress* the boundary — a large minority of its
questions deliberately name concepts this synthetic book does not carry (broker
identity, product type, borrower type). Those are not failures; a correct system
refuses them. The honest ceiling on *this* book is therefore well below 40, but far
above 11.

---

## 2. Full remaining-question inventory

Current outcome is the **deterministic governed path** (the canonical "current"
figure). `ok` in the harness is not the same as "answers as asked": two questions
return `ok=True` while answering a disclosed *narrower* question.

**Correct as asked (9):** A2a, A6, A8, B01, B06, B08, B11, B21, B25.
**`ok=True` but partial/under-answers (2):** A4, B22 — these make up the "11".

| ID | Question (abbrev) | Outcome | Current blocker (verified) | Gap class | Family |
|---|---|---|---|---|---|
| A1 | avg LTV, age, **borrower type** in London | partial refusal | borrower-type source (`number_of_borrowers`/`borrower_2_dob`) absent; LTV+age answerable | MISSING_DATA (one leg) | BORROWER (blocked) |
| A2a | which region grew most last month | **correct** | — | — | (movement, works) |
| A2b | which **broker** grew most last qtr | refusal | broker **identity** absent (only channel) | MISSING_DATA | BROKER (blocked) |
| A2c | which **product** grew most last qtr | refusal | `erm_product_type` column absent (R3: refuse) | MISSING_DATA | PRODUCT (blocked) |
| A2d | which **borrower type** grew most | refusal | borrower-type source absent | MISSING_DATA | BORROWER (blocked) |
| A3 | balance by LTV by **borrower type** | refusal | borrower-type source absent | MISSING_DATA | BORROWER (blocked) |
| A4 | balance by region by **borrower type** | partial (`ok`) | answers by region, discloses "borrower type unavailable" | MISSING_DATA (one leg) | BORROWER (blocked) |
| A5 | balance by **borrower type** by **product** | refusal | both legs' source absent | MISSING_DATA | BORROWER+PRODUCT (blocked) |
| A6 | close to breaching concentration limits? | **correct** | — | — | (concentration, works) |
| A7 | when will funded loans be £100MM | refusal | projection engine exists, not exposed | EXISTING_ANALYTIC_NOT_EXPOSED | **F1 PROJECTION** |
| A8 | avg LTV direct vs acquired | **correct** | — | — | (comparison, works) |
| A9 | avg collateral value since inception | refusal | valuation not tagged as a period-change field | REACHABILITY (tag) | F10 |
| B01 | most concentrated + headroom | **correct** | — | — | (concentration, works) |
| B02 | which **segments** driving growth this qtr | refusal | ranked-movement engine exists; "segment" undefined + routing | EXISTING_ANALYTIC_NOT_EXPOSED (+semantic) | **F2 MOVEMENT** |
| B03 | over-reliant on any single **broker** | refusal | broker identity absent | MISSING_DATA | BROKER (blocked) |
| B04 | credit quality of new origination vs back book | refusal | vintage_year not materialised (R1/R2: current-LTV by recent-vs-older vintage) | DERIVED_CONCEPT | **F3 VINTAGE** |
| B05 | share breaching 75% LTV if HPI −10% | refusal | scenario engine exists, not exposed | EXISTING_ANALYTIC_NOT_EXPOSED | **F4 STRESS** |
| B06 | exposure to borrowers over 85 | **correct** | — | — | (threshold filter, works) |
| B07 | headroom before **London** limit binds | refusal | headroom engine exists, but **no London limit configured** | CONFIG (client limit) | F6 (conditional) |
| B08 | run rate of new lending, accelerating? | **correct** | — | — | (run-rate, works) |
| B09 | which vintages have highest LTV | refusal | vintage_year not materialised | DERIVED_CONCEPT | **F3 VINTAGE** |
| B10 | share in **arrears** and where concentrated | refusal | arrears present (all zero); share-by-dim not exposed | EXISTING_ANALYTIC_NOT_EXPOSED | **F7 ARREARS** |
| B11 | which region contributes most to WA LTV | **correct** | — | — | (contribution, works) |
| B12 | how diversified vs last quarter | refusal | HHI absent + period routing | NEW_MATHS (HHI) | **F8 HHI** |
| B13 | **product** with highest avg ticket | refusal | `erm_product_type` absent (R3: refuse) | MISSING_DATA | PRODUCT (blocked) |
| B14 | is acquired converging with direct on LTV | refusal | convergence-over-time absent | NEW_MATHS | F10 |
| B15 | share eligible for 75% LTV securitisation | refusal | threshold-share engine exists, not exposed | EXISTING_ANALYTIC_NOT_EXPOSED | **F5 ELIGIBILITY** |
| B16 | which **brokers** bring highest-LTV business | refusal | broker identity absent | MISSING_DATA | BROKER (blocked) |
| B17 | what is driving change in WA LTV since inception | refusal | period × contribution not composed | NEW_MATHS (compose) | F10 |
| B18 | how has regional mix shifted over the qtr | refusal | mix-shift/movement exists, not routed | EXISTING_ANALYTIC_NOT_EXPOSED | **F2 MOVEMENT** |
| B19 | balance at year end at current rate | refusal | projection engine exists, not exposed | EXISTING_ANALYTIC_NOT_EXPOSED | **F1 PROJECTION** |
| B20 | cohorts closest to NNEG becoming a risk | refusal | scenario/underwater-exposure exists, not exposed | EXISTING_ANALYTIC_NOT_EXPOSED | **F4 STRESS** |
| B21 | largest single-loan exposure + share | **correct** | — | — | (works) |
| B22 | top-10 **postcode** concentration | partial (`ok`) | postcode present; route used ITL3 region, disclosed | EXISTING_ANALYTIC_NOT_EXPOSED | **F6 POSTCODE** |
| B23 | are older borrowers taking bigger loans (rel. value) | refusal (LLM: bad scatter) | correlation/relationship absent | NEW_MATHS | F9 |
| B24 | which part growing fastest by count not balance | refusal | movement-by-count exists; "part" dimension unresolved | EXISTING_ANALYTIC_NOT_EXPOSED (+semantic) | **F2 MOVEMENT** |
| B25 | direct vs acquired on borrower age | **correct** | — | — | (comparison, works) |
| B26 | 10% HPI fall → WA LTV | refusal | scenario engine exists, not exposed | EXISTING_ANALYTIC_NOT_EXPOSED | **F4 STRESS** |
| B27 | regions most exposed relative to last month | refusal | ranked-movement exists; phrasing routes elsewhere | EXISTING_ANALYTIC_NOT_EXPOSED | **F2 MOVEMENT** |
| B28 | quality of book by origination vintage | refusal | vintage_year not materialised (R1: current LTV) | DERIVED_CONCEPT | **F3 VINTAGE** |

---

## 3. Corrected classifications (prior labels that were wrong or imprecise)

| Question(s) | Prior/naive label | Corrected label | Evidence |
|---|---|---|---|
| B09, B28, B04 | "MISSING_DATA: vintage not available" | **DERIVED_CONCEPT** — `vintage_year` is governed (`mi_semantics_field_registry.yaml`, `derived_from: origination_date`) and `origination_date` is **present** (11,035 non-null, 2014–2026). | derivation source present; concept governed. |
| B10 | "MISSING_DATA: arrears" | **EXISTING_ANALYTIC_NOT_EXPOSED** — `arrears_balance` / `number_of_days_in_arrears` present (all zero); `_eval_arrears_share` exists. Truthful answer ≈ 0%. | fields present; calc exists. |
| B22 | "correct" (`ok=True`) | **PARTIAL** — answered top ITL3 region, not top-10 **postcode**; `postcode` is present (7,125 non-null). | route substituted ITL3; postcode present. |
| A4 | "correct" (`ok=True`) | **PARTIAL** — answered balance by region, dropped borrower type (honestly disclosed; source genuinely absent). | one leg genuinely blocked. |
| A7, B19, B05, B20, B26, B15, B18, B27, B24 | "MISSING_ANALYTIC_CAPABILITY" | **EXISTING_ANALYTIC_NOT_EXPOSED** — projection, scenario stress, threshold-share, ranked movement all exist (see §5). | audit cites the modules. |
| A2b, B03, B16 (broker) | "broker data exists" (prior claim) | **MISSING_DATA (identity)** — `origination_channel` (Broker/Direct) exists and is *collinear with the acquired/direct split*; there is **no broker-name/id** field or concept anywhere. | §4; channel ≠ identity. |
| A1, A3, A4, A5, A2d (borrower type) | "MISSING_GOVERNED_CONCEPT" | **MISSING_DATA (derivation source)** — `borrower_type`/`borrower_structure` ARE governed concepts; their source (`number_of_borrowers` / `borrower_2_dob`) is absent from this book. | concept exists; source absent. |
| A2c, B13, A5 (product) | "MISSING_GOVERNED_CONCEPT" | **MISSING_DATA (canonical column)** — `product_type` is governed = `erm_product_type`; that column is not populated here. R3: refuse (do not substitute `purpose`). | concept exists; column absent. |

---

## 4. Data availability audit (proven against the tape — 11,035 loans)

| Concept | Source data present? | Canonical field | Derived field possible? | Registry entry? | MI-reachable? | Verdict |
|---|---|---|---|---|---|---|
| Region | ✅ `collateral_geography` (named) + ITL3 codes | yes | — | yes | yes | present |
| Postcode | ✅ `postcode` 7,125 / `property_post_code` 3,908 | yes | — | yes | yes (unrouted) | **present, unexposed** |
| Vintage / origination year | ✅ `origination_date` 11,035 (2014–2026) | via derivation | ✅ `vintage_year` | ✅ governed (derived_from origination_date) | not materialised | **derivable now** |
| Seasoning / months-on-book | ✅ `origination_date` | via derivation | ✅ `months_on_book` | ✅ `time_on_book` | not materialised | derivable now |
| Arrears / delinquency | ✅ `arrears_balance`, `number_of_days_in_arrears` (all 0) | yes | share derivable | ✅ measures + `arrears_bucket` | partial | **present (zero)** |
| LTV (current) | ✅ `current_loan_to_value` (pp) | yes | — | ✅ governed core | yes | present |
| Valuation | ✅ `current_valuation_amount` 11,033 | yes | HPI-stress derivable | ✅ | partial | present |
| Rate | ✅ `current_interest_rate`, `interest_rate_bucket` | yes | — | ✅ | yes | present |
| Loan purpose | ✅ `purpose` (RENV/RMRT/EQRE) | yes | — | ✅ `purpose` | yes | present (≠ product) |
| **Product type** | ❌ `erm_product_type` not populated | no | no (distinct from purpose) | ✅ concept governed | resolves→absent col | **MISSING (R3 refuse)** |
| **Broker identity** | ❌ only `origination_channel` (Broker/Direct) | channel only | no | channel only | channel only | **MISSING (identity)** |
| **Borrower structure/type** | ❌ `number_of_borrowers`/`borrower_2_dob` absent | no | no | ✅ concept governed | resolves→absent col | **MISSING (source)** |
| Borrower age | ✅ `youngest_borrower_age`, `age_bucket` | yes | — | ✅ | yes | present |
| Ticket size | ✅ `current_outstanding_balance`, `ticket_bucket` | yes | — | ✅ `balance_band` | yes | present |
| NNEG / negative equity | ✅ derivable from LTV→100 (max LTV 104.6; 1 loan >100) | via calc | ✅ underwater exposure | ✅ scenario engine | not exposed | **derivable now** |
| Origination channel | ✅ `origination_channel` (collinear w/ direct/acquired) | yes | — | ✅ | yes | present |
| Property type | ✅ `property_type` (RHOS/RBGL/RFLT) | yes | — | ✅ | yes | present |
| Concentration limits | ⚠️ only **illustrative** (region 0.30, grade 0.25); **no London/single-name limit** | config | — | risk_monitor.yaml | engine exists | **needs client limits** |

**Independent truths computed from present data** (used below as the acceptance
anchors, recomputed directly, not via the agent):

- HPI −10% → WA current LTV **43.16 → 47.95**; share of balance breaching 75% LTV after the fall **1.75%**.
- Eligible ≤75% LTV: **99.67%** of balance (11,007 loans).
- In arrears: **0.00%** of balance (fully performing book).
- Top-10 postcode concentration: **0.38%** of the book (7,118 distinct postcodes — highly granular).
- WA current LTV by vintage: monotone **54.5% (2014) → 34.6% (2026)** — older equity-release roll-up loans carry higher LTV as interest accretes.
- B04 under R1/R2 (recent vs older vintage, 24-month cutoff): new **35.5%** vs back book **44.8%** → newer origination is **better** (lower LTV).
- NNEG headroom: 1 loan > 100% LTV, 1 loan > 90% — negligible but derivable.

---

## 5. Existing-analytics reuse audit (the "missing" calculations that already exist)

| Operation | Already exists? | Module::symbol | Bank questions it would serve |
|---|---|---|---|
| HPI / house-price stress on LTV | ✅ | `analytics/scenario_engine.py::project_portfolio` (presets incl. "House Price Stress" −10%); `mi_agent_api/risk_limits.py::_share_above` | B05, B26 |
| Forward projection to a threshold / year-end | ✅ | `mi_agent_api/forecast_extrapolation.py::build_extrapolation` (run-rate model, **milestone dates incl. £100m**) | A7, B19 |
| Ranked period-over-period movement | ✅ | `mi_agent/period_change/ranking.py::rank_movement` | B02, B18, B24, B27 |
| Mix-shift (share change pp) | ✅ | `mi_agent_api/insight_metrics.py::band_mix`; `risk_monitor/concentration.py::concentration_movement` | B18, B12(part) |
| Per-limit concentration headroom | ✅ | `mi_agent_api/risk_limits.py::compute_risk_limits` (`_headroom`, `closestHeadroom`) | B07 (needs a client limit) |
| Top-N / postcode concentration | ✅ | `analytics_lib/concentration.py::top_n_concentration`; `concentration_tests/metrics.py::_eval_postcode_area_share` | B22 |
| Largest single-loan exposure & share | ✅ | `risk_limits.py::_largest_single_loan_pct`; `concentration_tests/metrics.py::_eval_field_extremum` | B21 (works) |
| Weighted-average contribution decomposition | ✅ | `mi_query_executor.py::_execute_contribution` | B11 (works), B17(part) |
| Vintage / seasoning metrics | ✅ | `analytics/static_pools_core.py::build_vintage_metric_series`; `analytics_lib/cohort.py::cohort_period` | B09, B28, B04 |
| Threshold-share / eligibility | ✅ | `risk_limits.py::_share_above`; executor ratio path | B15, B05 |
| NNEG / underwater exposure | ✅ | `analytics/scenario_engine.py` (`underwater_exposure`, `expected_nneg_loss`) | B20 |
| Arrears / delinquency share | ✅ | `concentration_tests/metrics.py::_eval_arrears_share` | B10 |
| Portfolio comparison (direct vs acquired) | ✅ | `mi_workflows/portfolio_risk_comparison.py::run_portfolio_risk_comparison` | A8, B25 (work) |
| **Diversification index (HHI)** | ❌ **NOT FOUND** | — (explicitly "No HHI" in `concentration_analysis.py`) | B12 |
| **Convergence over time** | ❌ **NOT FOUND** | comparison is single-date only | B14 |
| **Correlation / relationship** | ❌ **NOT FOUND** | — | B23 |

**Only three genuinely new calculations are needed across the entire bank: HHI,
convergence-over-time, and a correlation/relationship metric.** Everything else is
exposure or a governed derivation.

---

## 6. Capability-family decomposition (smallest reusable families)

| Family | Questions | # | Real-world value | Data? | Concept? | Calc exists? | New maths? | Impl. complexity | Semantic risk | Best next? |
|---|---|---|---|---|---|---|---|---|---|---|
| **F3 VINTAGE & SEASONING** | B09, B28, B04 | 3 | High (credit/portfolio) | ✅ | ✅ | ✅ | derivation + WA-by-group | Low | Low | **YES (1st)** |
| **F1 PROJECTION** | A7, B19 | 2 | High (treasury) | ✅ | ✅ | ✅ | none (recombine milestones) | Low | Med (assumption disclosure) | YES |
| **F2 MOVEMENT / MIX-SHIFT ROUTING** | B02, B18, B24, B27 | 4 | High (portfolio mgmt) | ✅ | ✅ | ✅ | none (routing; define "segment") | Low-Med | Med ("segment"/"part") | YES |
| **F4 SCENARIO / HPI STRESS** | B05, B26, B20 | 3 | High (risk/treasury) | ✅ | ✅ | ✅ | one-shot revalue recombination | Med | Med | YES |
| **F5 ELIGIBILITY / THRESHOLD-SHARE** | B15 | 1 | Med-High (IR/securitisation) | ✅ | ✅ | ✅ | none | Low | Low | YES |
| **F6 POSTCODE + PER-LIMIT HEADROOM** | B22, (B07*) | 1(+1) | Med-High (risk) | ✅ (postcode); ⚠️ (limits) | ✅ | ✅ | none | Low | Low | YES (postcode) |
| **F7 ARREARS / CREDIT-STATE SHARE** | B10 | 1 | Med (credit) | ✅ (zero) | ✅ | ✅ | none | Low | Low-Med (state "0%") | YES |
| **F8 DIVERSIFICATION (HHI)** | B12 | 1 | Med (risk/IR) | ✅ | partial | ❌ | HHI + period delta | Med | Med | Later |
| **F9 RELATIONSHIP / CORRELATION** | B23 | 1 | Med (credit) | ✅ | ❌ | ❌ | correlation / bucketed cross-tab | Med | Med-High | Later |
| **F10 PERIOD-CONTRIBUTION / CONVERGENCE** | A9, B14, B17 | 3 | Med-High (CFO/credit) | ✅ | partial | partial | compose period×contribution; convergence | Med-High | Med-High | Later |
| **BLOCKED — BROKER identity** | A2b, B03, B16 | 3 | High (would-be) | ❌ | ❌ | n/a | needs source data | — | — | No (safe refusal) |
| **BLOCKED — PRODUCT type** | A2c, B13, (A5) | 2(+1) | High (would-be) | ❌ | ✅ | n/a | needs source data | — | — | No (R3 refuse) |
| **BLOCKED — BORROWER structure** | A1*, A3, A4*, A5, A2d | 3–5 | High (would-be) | ❌ | ✅ | n/a | needs source data | — | — | No (safe refusal) |

\* A1/A4 partially answer their LTV/age/region legs; B07 needs a client concentration limit (config), not new maths.

---

## 7. Prioritisation score

Scored 1–5 (5 = best/cheapest). `unlock_score = questions × commercial × data ×
reuse × semantic_simplicity ÷ implementation_risk` (illustrative, to make the
ranking explicit).

| Family | Q unlocked | Commercial | Data ready | Calc reuse | Semantic simplicity | Impl. (5=easy) | Regression risk (5=low) | Score | Rank |
|---|---|---|---|---|---|---|---|---|---|
| **F3 VINTAGE** | 3 | 5 | 5 | 5 | 4 | 4 | 4 | **very high** | **1** |
| **F1 PROJECTION** | 2 | 5 | 5 | 5 | 4 | 4 | 4 | high | 2 |
| **F2 MOVEMENT ROUTING** | 4 | 4 | 5 | 5 | 3 | 4 | 3 | high | 3 |
| **F4 HPI STRESS** | 3 | 5 | 5 | 4 | 3 | 3 | 3 | high | 4 |
| **F5 ELIGIBILITY** | 1 | 4 | 5 | 5 | 5 | 5 | 5 | med-high | 5 |
| **F6 POSTCODE/HEADROOM** | 1(+1) | 4 | 5/⚠ | 5 | 5 | 5 | 5 | med-high | 6 |
| **F7 ARREARS** | 1 | 3 | 5 | 5 | 4 | 5 | 4 | med | 7 |
| **F8 HHI** | 1 | 3 | 5 | 1 | 3 | 3 | 3 | low-med | 8 |
| **F10 CONTRIB/CONVERGE** | 3 | 4 | 4 | 2 | 2 | 2 | 2 | low-med | 9 |
| **F9 CORRELATION** | 1 | 3 | 5 | 1 | 2 | 3 | 3 | low | 10 |

F5, F6, F7 are individually tiny but nearly free — they should ride along with an
adjacent phase rather than each take a phase.

---

## 8. Near-term answer-rate ceiling (using only present data)

Only numbers supported by §4–§5 are used.

```
Current (as asked)            :  9 / 40   (A2a A6 A8 B01 B06 B08 B11 B21 B25)
Current (ok, incl. 2 partial) : 11 / 40   (+ A4, B22)

After easy reuse (F1,F2,F3,F5,F6-postcode,F7):
    + A7 B19            (F1 projection)
    + B02 B18 B24 B27   (F2 movement routing)
    + B04 B09 B28       (F3 vintage)
    + B15               (F5 eligibility)
    + B22(full)         (F6 postcode; upgrade partial→full)
    + B10               (F7 arrears, truthfully ~0%)
  = ~22 / 40

After moderate additions (F4 stress, F8 HHI, F9 correlation, F10 contrib/convergence, + a client London limit for B07):
    + B05 B20 B26       (F4 stress)
    + B12               (F8 HHI)
    + B23               (F9 correlation)
    + A9 B14 B17        (F10)
    + B07               (client limit config)
  = ~31 / 40

Blocked by genuinely-missing SOURCE DATA (must remain safe refusals):
    Broker identity     : A2b B03 B16            (3)
    Product type        : A2c B13                 (2)   [+ A5 product leg]
    Borrower structure  : A1* A3 A4* A5 A2d       (~4, A1/A4 partial)
  = ~9 / 40 blocked
```

**Realistic near-term ceiling on this book ≈ 31/40**, reached without any new source
data and without weakening a single refusal. The remaining ~9 are correctly blocked
by absent source columns (broker identity, product type, borrower structure). **Easy
reuse alone clears 20** — the target — and does so with low-risk exposure of analytics
that already exist and are already tested.

---

## 9. Recommended next 3–5 increments (ranked)

### P1J-1 — VINTAGE & SEASONING  *(recommended first)*
- **Objective:** materialise the governed `vintage_year` (and seasoning) from the
  present `origination_date`, and expose weighted-average-by-vintage so the book can
  be sliced by origination cohort.
- **Questions unlocked:** B09, B28, B04 (B04 via R1/R2: current-LTV, recent-vs-older
  vintage).
- **Real-world questions also unlocked:** "WA LTV by vintage — which cohorts are
  richest?"; "How is new-origination quality trending vs the back book?"; "What is the
  balance and count by origination year?"; "Which vintages dominate the book?"
- **Existing components reused:** `analytics/static_pools_core.py::build_vintage_metric_series`,
  `analytics_lib/cohort.py::cohort_period`, the P1E multi-measure-by-group machinery,
  `semantic_resolver.resolve_dimension` (vintage synonyms already governed).
- **New semantics required:** materialise `vintage_year` derivation on the MI frame;
  a governed "new origination" cutoff (per R2) as configuration.
- **New maths required:** none beyond bucketing origination_date and reusing WA-by-group.
- **Acceptance:** independent WA-LTV-by-year recompute (already done: 54.5%→34.6%);
  B04 new 35.5% vs back 44.8%.
- **Risk:** the "new origination" cutoff is a config value — must be governed and
  disclosed, not hard-coded.

### P1J-2 — PROJECTION
- **Objective:** expose the existing run-rate extrapolation to answer "when will X
  reach £N" and "what will the balance be at year end".
- **Questions unlocked:** A7, B19.
- **Also:** "when does the funded book reach £2bn?"; "projected originations for the
  half-year"; "at this run-rate, year-end WA LTV?"
- **Reused:** `mi_agent_api/forecast_extrapolation.py::build_extrapolation` (already
  computes milestone dates incl. £100m).
- **New semantics:** route projection intent ("when will", "by year end", "at current
  rate") to the extrapolation engine.
- **New maths:** none.
- **Acceptance:** milestone date reconciled against the governed funded-evolution series
  with base/downside/upside bands; refuse if the run-rate basis is too short.
- **Risk:** projections must carry the assumption + confidence band prominently and
  refuse when the history is too short to extrapolate.

### P1J-3 — MOVEMENT / MIX-SHIFT ROUTING
- **Objective:** route "relative to last month", "mix shifted", "which segments are
  driving growth", "fastest growing by count" to the existing ranked-movement engine.
- **Questions unlocked:** B27, B18, B24, B02.
- **Also:** "which regions gained/lost the most share this quarter?"; "is growth
  balance-led or count-led?"
- **Reused:** `mi_agent/period_change/ranking.py::rank_movement`,
  `insight_metrics.band_mix`.
- **New semantics:** recognise these phrasings as ranked-movement; resolve "segment"/
  "part" to a governed default dimension (region) or refuse if ambiguous.
- **New maths:** none.
- **Acceptance:** reconcile against A2a's own movement figures (May→June).
- **Risk:** "segment"/"part" are under-specified — must resolve to a governed dimension
  or refuse, never guess silently.

### P1J-4 — SCENARIO / HPI STRESS  (+ ELIGIBILITY ride-along)
- **Objective:** expose one-shot HPI revaluation of LTV and the threshold-share it
  produces (breach share, eligibility share, NNEG headroom).
- **Questions unlocked:** B26, B05, B20, and F5's B15.
- **Also:** "if HPI −15%, what breaches an 80% cap?"; "what share is eligible for a 70%
  LTV securitisation?"; "how close is the book to negative equity?"
- **Reused:** `analytics/scenario_engine.py::project_portfolio` (House Price Stress
  presets), `risk_limits.py::_share_above`.
- **New semantics:** route stress ("if house prices fell N%") and eligibility ("share
  eligible for an N% LTV …") intents.
- **New maths:** one-shot revalue recombination (revalue = valuation×(1−shock); restress
  LTV; share above cap) — light, reuses `_share_above`.
- **Acceptance:** independent recompute (done: −10% → WA LTV 47.95, breach share 1.75%,
  eligible 99.67%).
- **Risk:** stress semantics must be explicit (which cap, which shock) and never conflate
  a stressed figure with the actual one.

### P1J-5 — CONCENTRATION FINISH (postcode + arrears + per-limit headroom)
- **Objective:** upgrade B22 to true top-10 **postcode**; expose arrears share-by-dim;
  expose per-limit headroom (needs a client limit set for B07).
- **Questions unlocked:** B22 (full), B10, B07 (with a configured London limit).
- **Reused:** `_eval_postcode_area_share`, `_eval_arrears_share`,
  `risk_limits.compute_risk_limits`.
- **New maths:** none.
- **Acceptance:** top-10 postcode share (0.38%), arrears share (0.00%) reconciled.
- **Risk:** arrears must *answer* "0%", not refuse; B07 needs a real client limit, not
  an illustrative one.

*(Later, lower priority: F8 HHI, F9 correlation, F10 period-contribution/convergence —
these need genuinely new maths and carry the most semantic risk.)*

---

## 10. Five new CFO / Treasury / Credit acceptance questions

| # | Question | Business purpose | Required semantics | Answerable today? | Unlocked by |
|---|---|---|---|---|---|
| 1 | "What is the weighted-average LTV by origination vintage, and which cohorts are richest?" | Credit — vintage quality trend | vintage_year (derived), WA LTV by group | No | **P1J-1** |
| 2 | "At the current origination run-rate, when does the funded book reach £2bn?" | Treasury — capital planning | run-rate → milestone date + bands | No | **P1J-2** |
| 3 | "Which regions gained and lost the most share of the book this quarter?" | Portfolio mgmt — mix drift | ranked share-movement by region | No (phrasing routes wrong) | **P1J-3** |
| 4 | "If house prices fell 10%, what share of the book breaches an 80% LTV cap and what is the stressed WA LTV?" | Risk — resilience | HPI shock + threshold-share | No | **P1J-4** |
| 5 | "What share of the book is eligible for a 70% LTV securitisation, and how much sits within 5 points of the cap?" | IR/Treasury — issuance readiness | threshold-share + proximity band | No | **P1J-4/5** |

All five test reusable breadth (a governed dimension, a projection, a movement, a
stress, a threshold-share), not edge cases, and each is anchored to an existing engine.

---

## 11. Specific recommendation

The whole roadmap turns on one corrected finding: **the calculations already exist;
the breadth gap is exposure and one governed derivation, not missing maths.** That
makes the next several phases cheap and low-risk, and it is why easy reuse alone lifts
the book from 11 to ~22/40 and moderate additions to ~31/40 — the ceiling this book's
data allows — without weakening any refusal.

The first phase should give the **most breadth per unit of risk**. Vintage does:
its source (`origination_date`) is present, its concept (`vintage_year`) is already
governed, its calculation (`static_pools_core` / `cohort`) already exists, so the work
is exposing an existing analytic over a present, governed-derivable dimension — the
lowest-risk kind of change. It unlocks three bank questions (B09, B28, B04) plus a
distinct, highly commercial CFO/credit narrative (how origination quality is trending),
and it is the flagship correction of a "false missing data" case — the book was
refusing "vintage" while carrying every origination date. Its only real risk is the
"new origination" cutoff, which must be governed configuration and disclosed, not
hard-coded.

> **NEXT RECOMMENDED BREADTH PHASE: P1J-1 — VINTAGE & SEASONING**

It reuses existing tested analytics, needs no new source data and no new core maths,
carries low semantic risk, unlocks three bank questions and a commercially central
class of real-world questions, and corrects a demonstrable false-missing-data gap —
the highest breadth-per-unit-risk increment available.

---

## Appendix — safety observation (flagged, not fixed)

Per the brief, one item to flag without fixing: **B23 on the genuine-LLM path returns
`ok=True` with a degenerate scatter** ("Count of · 11,035 loans · 5,000 groups") for a
relationship question it cannot actually answer. This is a candidate silent-semantic
issue — a relationship/correlation question should refuse (F9) rather than emit a
meaningless scatter marked successful. It does not affect the deterministic path and is
not required for this analysis, so it is recorded here for a future safety pass rather
than fixed in P1J.

P1J BREADTH GAP DECOMPOSITION: COMPLETE
