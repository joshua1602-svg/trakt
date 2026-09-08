# Borrowing base — governed facility monitoring

A prototype that answers one question end to end: **how much can this book
borrow under its facility, and how much of that is left?**

It does not introduce a second source of portfolio truth. It reads the same
canonical frame the Risk Limits workspace, MI Query and the PPTX pack already
read, reuses the existing governed concentration engine for Schedule 8, and
adds exactly one new thing to the canonical model — whether a loan is an
Eligible Mortgage Loan for a facility.

---

## The flow

```
OCC onboarding
  Warehouse / Funding Facility Documentation  (asked of the client)
  structured facility terms                   (recorded by an operator)
        |
        v
config/client/config_client_<CLIENT>.yaml → funding_facility:
config/risk/funding_facilities.yaml        (platform register / prototype)
        |
        v
mi_agent.borrowing_base.config.load_facility
        |
        v
canonical portfolio  (mi_agent_api.funded_prep.prepare_funded_mi_dataset)
        |
        +-- mi_agent.borrowing_base.eligibility.derive_eligibility
        |     borrowing_base_eligible
        |     borrowing_base_eligibility_status   ELIGIBLE|INELIGIBLE|UNDETERMINED
        |     borrowing_base_eligibility_reason
        |     borrowing_base_facility_id
        v
eligible financing population
        |
        +-- Schedule 8 concentrations
        |     mi_agent.concentration_tests.evaluation.evaluate_active_tests
        |     populations={"eligible_mortgage_loans": eligible_df}
        |
        +-- borrowing base engine
              mi_agent.borrowing_base.calculator.calculate
                        |
                        v
              Eligibility & Concentrations  (the React workspace)
              GET /mi/concentration-tests   → tests + borrowingBase
              GET /mi/borrowing-base        → the facility position alone
              GET /mi/borrowing-base/loans  → the loans behind one status
```

---

## The equations

Implemented in `mi_agent/borrowing_base/calculator.py`, and nothing else:

```
concentration_limit_denominator = MAX(concentration_denominator_floor,
                                      eligible_current_balance)
gross_borrowing_base            = eligible_current_balance x advance_rate
available_borrowing_base        = MIN(gross_borrowing_base, facility_commitment)
borrowing_base_headroom         = available_borrowing_base - current_drawn_amount
borrowing_base_deficiency       = ABS(MIN(borrowing_base_headroom, 0))
borrowing_base_utilisation_pct  = current_drawn_amount
                                  / available_borrowing_base x 100
facility_utilisation_pct        = current_drawn_amount / facility_commitment x 100
```

There is **no haircut, no reserve, no overcollateralisation, and no deduction
for a concentration breach.** Schedule 8 as supplied establishes the
concentration tests; it does not state the contractual consequence of breaching
one, so v1 monitors breaches and deducts nothing.
`concentration_adjustment()` is the seam for a future approved treatment — it
REFUSES any treatment other than `monitor_only` rather than guessing at one.

The governed headroom stays **negative** when the facility is over-drawn.
Showing £0 headroom beside a deficiency is a presentation choice the UI makes;
the calculation keeps the real number.

## NOT_CALCULABLE

Every measure is a number **or** the string `NOT_CALCULABLE`. There is no third
state, and no measure is ever zero because an input was missing. When the
current drawn amount has not been supplied — which is the prototype's actual
situation — the borrowing base is still calculated and headroom, deficiency and
both utilisations report `NOT_CALCULABLE` with `current_drawn_amount` named in
`missingInputs`.

---

## Eligibility

**Schedule 8 does not define an Eligible Mortgage Loan.** It states limits that
apply to them. Passing a portfolio-level concentration test says nothing about
whether an individual loan qualifies, and nothing in this package infers one
from the other.

So the eligibility rules are configuration:

```yaml
eligibility:
  rule_version: "..."
  rules:
    - rule_id: max_original_ltv
      description: Original LTV must not exceed 50%
      field_role: ltv_original        # a governed field role, or `field:`
      operator: max                   # max|min|equals|not_equals|in|not_in|present
      value: 50
      reason_code: original_ltv_above_facility_limit
```

Rules are conjunctive. A loan meeting all of them is `ELIGIBLE`; one breaching
any is `INELIGIBLE` with that rule's reason; one whose input is missing is
`UNDETERMINED` — never `INELIGIBLE`, because "we cannot tell" is not "no".

With **no** approved rules a production facility reports `UNDETERMINED` for
every loan. That is the fail-closed default and it is why an eligible collateral
balance of zero can be a correct answer.

### The prototype assumption

```yaml
environment: prototype
eligibility:
  prototype_assume_financing_portfolio_eligible: true
```

Treats every loan in the configured Financing Portfolio as eligible so the
borrowing base can be exercised before the definition arrives. It is honoured
**only** under `environment: prototype`; configuring it on a production facility
is reported as a configuration problem and ignored. Every figure it produces is
stamped in `prototype_assumptions_used` on the receipt and labelled on the tab.
OCC writes no `environment`, so a facility onboarded through the wizard is
production and cannot inherit it.

---

## Schedule 8 and the eligible population

A Schedule 8 clause that says "Eligible Mortgage Loans" is measured over exactly
those loans. The extractor reads the population out of the clause's own wording
(`matching.detect_population`), it survives approval into the immutable
configuration (`ActiveTest.population`), and the evaluator honours it
(`evaluate_active_tests(populations=...)`).

A test declaring a population the caller did not supply reports **UNAVAILABLE**
naming the population. It is never quietly measured over the whole book, which
would answer a different question under the contractual test's name. Where no
governed eligibility exists at all — the client has no facility — the whole
funded book stands in and the substitution is disclosed on every affected test
(`populationBasis`) and in the envelope (`eligiblePopulation.note`).

### Portfolio Net WAC — NOT CALCULABLE

Schedule 8 defines it as the weighted average fixed rate of all Eligible
Mortgage Loans **weighted by Current Balance and expected duration**, minus the
**Series Fixed Rate from the Final Maturity Date onwards**. Neither the expected
duration nor the Series Fixed Rate exists in the canonical data or the facility
configuration. The test extracts with its threshold and direction, is held at
`pending_confirmation` with `net_wac_definition_uncertain`, and never activates.
A balance-only weighted average is not reported in its place.

---

## Reconciliation

Fail closed. `calculate()` raises `ReconciliationError` unless:

* `financing_portfolio_balance == eligible + ineligible + undetermined`
* `eligible + ineligible + undetermined loan count == financing_portfolio_loan_count`
* no loan holds two statuses

Financing Portfolio membership is read from the **facility attribution**
(`borrowing_base_facility_id`), not from the status. That is what lets the
invariant fail: a loan attributed to the facility whose status is missing or
unrecognised stays in the portfolio and lands in no bucket, so nothing can
disappear quietly.

---

## The MI seam — prepared, not wired

`mi_agent.borrowing_base.service` exposes named, unit-carrying, definition-
carrying measures that can be invoked with no Streamlit, FastAPI or React in the
process:

| measure_id | unit |
| --- | --- |
| `financing_portfolio_balance` | currency |
| `eligible_balance` / `ineligible_balance` / `undetermined_balance` | currency |
| `eligible_loan_count` | count |
| `concentration_limit_denominator` | currency |
| `advance_rate` | percent |
| `gross_borrowing_base` / `borrowing_base` | currency |
| `facility_commitment` / `facility_drawn` | currency |
| `borrowing_base_headroom` / `borrowing_base_deficiency` | currency |
| `borrowing_base_utilisation` / `facility_utilisation` | percent |
| `nearest_concentration_limit` | label |
| `nearest_concentration_headroom` | percent |
| `nearest_concentration_headroom_amount` | currency |
| `breached_concentration_count` | count |

**The MI Query Agent does not answer borrowing-base questions.** Its parsing,
semantic composition and routing are untouched, and a test asserts no MI module
references this package. Registering these measures is a later sprint.

---

## Receipts

`mi_agent.borrowing_base.receipt.build_receipt` produces the audit document: the
facility terms and their version and hash, the population, the eligibility rule
version and every prototype assumption, each equation's inputs and output, the
concentration results with their source wording, the invariants, what could not
be calculated and why. Two receipts for the same inputs carry the same
`content_hash`.

---

## Where things live

| Concern | Module |
| --- | --- |
| Facility terms | `config/risk/funding_facilities.yaml`, `config/client/config_client_*.yaml` |
| Configuration loader | `mi_agent/borrowing_base/config.py` |
| Typed contracts | `mi_agent/borrowing_base/models.py` |
| Loan-level eligibility | `mi_agent/borrowing_base/eligibility.py` |
| The calculation | `mi_agent/borrowing_base/calculator.py` |
| The receipt | `mi_agent/borrowing_base/receipt.py` |
| The measure interface | `mi_agent/borrowing_base/service.py` |
| Canonical derivation | `mi_agent_api/funded_prep.py` |
| HTTP service | `mi_agent_api/borrowing_base_api.py`, routes in `mi_agent_api/app.py` |
| Schedule 8 population | `mi_agent/concentration_tests/{matching,models,evaluation}.py` |
| OCC onboarding | `config/onboarding/field_catalogue.yaml` → `funding_facility` |
| UI | `frontend/mi-agent-ui/src/components/risk/BorrowingBasePanel.tsx` |

---

## MI Query — straightforward borrowing-base questions

The MI Query Agent answers borrowing-base questions from the **same governed
`borrowingBase` envelope** the Eligibility & Concentrations tab renders and
`GET /mi/borrowing-base` returns. It calculates nothing:

```
/mi/query
  → mi_agent_api.borrowing_base_query      one registered recogniser, one handler
      current position   → borrowing_base_api.compute_borrowing_base   (the dashboard's block)
      change / trend /   → evolution.funded_frames                      (governed periods)
      bridge               + borrowing_base_api.compute_from_frames    (one evaluation per period)
      reasons            → the prepared frame's governed eligibility columns
  → mi_agent.borrowing_base.analysis        reason summary · period validity · bridge composition
  → the existing Artifact Workspace contract (table · line · waterfall)
```

Recognition is a closed vocabulary — *borrowing base*, *facility utilisation /
drawn / commitment*, *eligible collateral*, *ineligible*, *ineligibility reason*.
A bare *headroom* or *utilisation* stays with its existing owner. The parser
claims the owner's nouns (`_BORROWING_BASE_NOUNS`) so "eligible collateral" is
never recorded as a category the book lacks; the receipt guard lists the route
among the temporal and share-bearing routes and reads its declared measure set.

### What v1 answers

| Family | Examples | Owner of the number |
|---|---|---|
| Current position | borrowing base · headroom · facility / borrowing-base utilisation · drawn · commitment · eligible collateral | `compute_borrowing_base` → `measures` |
| Ineligibility | count · balance · **count share** · **balance share** (never synonyms) | four additive measures in `service.MEASURES` |
| Reasons | "why are loans ineligible?", by-reason breakdown | `analysis.summarise_ineligibility_reasons` |
| Change / trend | "how has the borrowing base changed?", "over time" | `analysis.change` / `analysis.series` over per-period envelopes |
| Bridge | "why did the borrowing base change?", "as a waterfall", "as a table" | `analysis.bridge` |

### The reason table is a PRIMARY-reason table

The derivation stamps the **first failing approved rule** on an INELIGIBLE loan.
A loan failing two rules is counted once, under the first configured one. The
table does not enumerate every rule a loan failed, says so in its `basis`, and
refuses unless Σ counts and Σ balances reconcile to the calculator's own
partition. UNDETERMINED never enters it.

### What history can and cannot demonstrate

The platform holds one facility configuration and one operator-supplied
drawing. Nothing is inferred from `effective_date` (metadata no engine reads):

* a historical period is evaluated **only from the register's governed
  `governance.approved_at` date**; with none recorded — the prototype's
  situation — change, trend and bridge questions refuse and say why;
* the drawing is valid for a snapshot **only when `current_drawn_amount_as_of`
  is that snapshot's date**; otherwise drawn, headroom, deficiency and both
  utilisations are `NOT_CALCULABLE` for that period (the calculator itself
  reports it — the drawing is withheld, never overwritten). The current-position
  answer keeps the dashboard's figure and discloses the as-of date.

### The bridge

Exact by construction, in a fixed order, from two governed envelopes:

```
opening borrowing base
+ eligible collateral effect  = (E_closing − E_opening) × advance_rate_opening
+ advance-rate effect         = E_closing × (advance_rate_closing − advance_rate_opening)
+ facility cap effect         = (B − G)_closing − (B − G)_opening
= closing borrowing base                       (tolerance 5p — penny rounding only)
```

No haircut, reserve, concentration or overcollateralisation line exists because
the v1 calculator applies none. A headroom bridge (`opening headroom + Δ base −
Δ drawn = closing headroom`) needs a period-valid drawing at both ends.

### Refusals are governed, never silent

Once borrowing-base intent is claimed, a facet v1 cannot honour — "by region",
"for Scotland", a portfolio lens, a loan-level listing, a forecast, a haircut,
a concentration deduction — is refused by name. No facility → refusal. An
unreconciled eligibility population → refusal. A missing drawing →
`NOT_CALCULABLE` with `current_drawn_amount` named, never a zero.

The question bank lives in `migration_phase0/BORROWING_BASE_MI_BANK.yaml` and
runs in `mi_agent_api/tests/test_borrowing_base_query.py`.
