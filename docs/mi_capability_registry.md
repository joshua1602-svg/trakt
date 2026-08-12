# Unified MI capability registry

*Sprint 2.5E. Baseline `670c404` → `ccbcf56`.*

> **Superseded in part by the close-out pass** —
> `docs/mi_query_architecture_and_default_cure.md`. Three claims below have
> since changed and are left in place as the record of what this sprint
> delivered rather than silently corrected:
>
> * **"only 9 of the 50 readiness metrics carry a capability link"** was a
>   mis-framed statistic. Classifying all of them showed the correct
>   denominator is the 11 named derived KPIs, not 50; the other 41 are
>   covenant-library metrics and shared analytical operations for which a
>   capability link would create a second identity. §2 of the close-out report.
> * **`default_rate` and `cure_rate` are no longer `METHODOLOGY_NOT_APPROVED`.**
>   Both methodologies are now owned, versioned and registered.
> * **The MI Query integration described in §6 did not work on the real code
>   path.** The capability explanation was wired to the `unresolved_metric`
>   branch; CPR, WAL, YTM and default rate all arrive on the `unmapped` branch,
>   so it never fired for any of them. Fixed in the close-out pass.

## 1. Executive answer

> **Does Trakt now have one asset-agnostic catalogue of portfolio intelligence
> that can be consumed consistently by humans, Copilot and autonomous agents?**

**PARTIALLY — and the part that is done is the part that was missing.**

There is now **one** catalogue: 28 capabilities in
`config/system/mi_capability_registry.yaml`, resolved by `trakt_core.capability`
against a portfolio's governed data, with **no asset class anywhere in the
resolver** — asserted by a test that parses the module, strips docstrings and
comments, and fails if any identifier or comparison tests a product label.

What is genuinely complete: the catalogue, the six-state availability model
with machine-readable reason codes, a discovery tool
(`portfolio_capabilities`), MI Query explaining refusals instead of denying the
measure exists, and the Readiness framework consuming capabilities rather than
owning alternative calculations.

What is honestly **not** complete: React consumes nothing from this registry
yet — the analysis in §8 is a review, not an integration — and only 9 of the 50
readiness metrics carry a capability link, because the rest resolve through the
concentration library and linking them is mechanical work this sprint did not
do. Calling it YES would overstate both.

---

## 2. Before and after

**Before.** Four registries, each authoritative for its own thing, and no
answer to "what can Trakt tell me about this portfolio?":

```
fields_registry.yaml            499 canonical fields
business_semantics_registry     242 fields with analytical metadata
concentration_test_library       42 covenant metrics
mi_semantics_field_registry     116 MI Query fields + 1 derived metric
```

That last number is the finding. MI Query's `metric_definitions` block held
**one** entry — average loan balance, expressed as a field plus an aggregation.
It structurally cannot express a capability like contractual WAL, which is a
cash-flow enumeration rather than a column with a `sum` on it. Everything built
in 2.5B–2.5D was reachable only through whichever tool happened to wrap it.

**After.** A fifth layer that holds what none of the others could — *whether a
capability can be produced here, and why not* — and references the rest by id
rather than restating them:

```
CORE + REGIME INPUTS -> CANONICAL DATA -> SHARED ANALYTICS
                                              |
                                    MI CAPABILITY REGISTRY
                                              |
          +---------------+-----------+-------+--------+
        React         MI Query     Copilot   Agent   Readiness
      (reviewed)     (integrated) (via tool) (tool)  (9 linked)
```

---

## 3. Registered capability catalogue

28 capabilities: 25 observed, 2 contractual, 1 modelled. By category —
exposure 5, delinquency 5, collateral 4, loss 4, cashflow 3, prepayment 2,
history 2, data quality 2, pricing 1.

| Capability | Category | Methodology | Requirements | Availability logic |
|---|---|---|---|---|
| `total_balance` | exposure | — | balance | fields |
| `loan_count` | exposure | — | identifier | fields |
| `average_loan_size` | exposure | `balance_average` | balance | fields |
| `geographic_concentration` | exposure | — | balance + a geography | NOT_APPLICABLE without geography |
| `top_n_exposure` | exposure | `top_n_loans_share` | balance | fields |
| `current_ltv` | collateral | `CURRENT_LTV@v1` | balance + collateral value | NOT_APPLICABLE without collateral |
| `wa_current_ltv` | collateral | `ltv_weighted_average` | balance + collateral value | as above |
| `high_ltv_exposure` | collateral | `ltv_above_share` | balance + collateral value | as above |
| `valuation_age` | collateral | — | collateral value | as above |
| `arrears_stock` | delinquency | — | `arrears_balance` (RREL67) | fields |
| `arrears_30_plus` / `arrears_90_plus` | delinquency | `perf_arrears_share` | RREL68 + balance | fields |
| `arrears_transition` | delinquency | migration matrix | status/DPD + **2 snapshots** | history |
| `default_rate` | loss | — | RREL71/72/69 present | **METHODOLOGY_NOT_APPROVED, always** |
| `cure_rate` | delinquency | — | fields present | **METHODOLOGY_NOT_APPROVED, always** |
| `observed_smm` / `observed_cpr` | prepayment | `OBSERVED_SMM@v2` / `OBSERVED_CPR@v2` | unscheduled principal or redemption evidence + 2 snapshots | fields + history |
| `observed_loss_rate` | loss | `OBSERVED_LOSS@v2` | RREL73 + 2 snapshots | fields + history |
| `observed_recovery_rate` | loss | `OBSERVED_RECOVERY@v2` | RREL74 + RREL71 + 2 snapshots | fields + history |
| `observed_loss_severity` | loss | `OBSERVED_LOSS_SEVERITY@v1` | RREL73 + RREL71 + 2 snapshots | fields + history |
| `wa_coupon` | pricing | `rate_gross_wac` | RREL43 + balance | fields |
| `contractual_wal` | cashflow | `CONTRACTUAL_WAL@v1` | balance, RREL24, RREL35, frequency | **+ principal determinism** |
| `contractual_ytm` | cashflow | `CONTRACTUAL_YTM_PERIODIC@v1` | the above + RREL43, RREL34 | **+ principal AND interest determinism** |
| `expected_wal` | cashflow | — | — | **MODEL_REQUIRED, always** |
| `cohort_comparison` | history | `OBSERVED_SERIES@v1` | `origination_date` | fields |
| `portfolio_history` | history | `OBSERVED_SERIES@v1` | 2 snapshots | history |
| `canonical_completeness` | data_quality | — | — | always |
| `regulatory_readiness` | data_quality | — | — | always |

**Registration decisions.** REGISTERED: the 28 above. **METHODOLOGY NOT
OWNED**: default rate, cure rate — registered as capabilities so a consumer
learns the state, never as available metrics. **NOT REGISTERED**: WA seasoning,
WA remaining term and WA borrower age — 2.5C found the weighted-average
evaluator would serve them unchanged but no library metric exists, so
registering them would publish a capability nothing implements. Internal
helpers (`classify_exits`, `stratify`, `group_shares`) are deliberately absent:
they are inputs to capabilities, not capabilities.

---

## 4. Availability-state model

| State | Meaning | Reason codes |
|---|---|---|
| `AVAILABLE` | can be calculated correctly from governed data | — |
| `UNAVAILABLE` | applicable, but an input or history is missing | `MISSING_REQUIRED_INPUT`, `INSUFFICIENT_HISTORY` |
| `NOT_APPLICABLE` | economically meaningless for this portfolio | `NO_COLLATERAL`, `NO_GEOGRAPHY`, `CONTINGENT_REPAYMENT` |
| `ASSUMPTION_REQUIRED` | needs an unknown future variable | `FUTURE_RATE_PATH_REQUIRED` |
| `MODEL_REQUIRED` | needs behavioural modelling Trakt does not do | `BEHAVIOURAL_MODEL_REQUIRED` |
| `METHODOLOGY_NOT_APPROVED` | Trakt holds the data and has not settled the definition | `METHODOLOGY_NOT_APPROVED` |

**The sixth state is a deliberate addition, and it earns its place.** The brief
listed five and asked that any more be justified. Folding default rate into
`UNAVAILABLE` would send someone to the client to request `default_amount` —
which is already on the tape. "We lack a field" and "we have not decided what
this means" produce opposite next actions, and 2.5C found that conflating them
is precisely how a plausible formula gets published as though it were owned.

Every non-`AVAILABLE` result carries a reason code and an explanation, and a
test fails any explanation under 40 characters — a refusal a consumer cannot
act on is not a refusal, it is a null with extra steps.

---

## 5. Multi-asset proof

Four portfolios, one registry, no bespoke code. Measured behaviour, run at
three snapshots:

| Capability | A: equity release | B: fixed amortising | C: floating French | D: unsecured |
|---|---|---|---|---|
| `total_balance`, `loan_count`, `average_loan_size`, `top_n_exposure` | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE |
| `geographic_concentration` | AVAILABLE | AVAILABLE | AVAILABLE | **NOT_APPLICABLE** |
| `current_ltv`, `wa_current_ltv`, `high_ltv_exposure`, `valuation_age` | AVAILABLE | AVAILABLE | AVAILABLE | **NOT_APPLICABLE** |
| `arrears_stock`, `arrears_30_plus`, `arrears_90_plus`, `arrears_transition` | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE |
| `observed_smm`, `observed_cpr` | UNAVAILABLE | **AVAILABLE** | UNAVAILABLE | UNAVAILABLE |
| `observed_loss_rate`, `observed_recovery_rate`, `observed_loss_severity` | UNAVAILABLE | **AVAILABLE** | UNAVAILABLE | UNAVAILABLE |
| `wa_coupon` | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE |
| `contractual_wal` | **NOT_APPLICABLE** | AVAILABLE | **ASSUMPTION_REQUIRED** | AVAILABLE |
| `contractual_ytm` | **NOT_APPLICABLE** | AVAILABLE | **ASSUMPTION_REQUIRED** | AVAILABLE |
| `expected_wal` | MODEL_REQUIRED | MODEL_REQUIRED | MODEL_REQUIRED | MODEL_REQUIRED |
| `default_rate`, `cure_rate` | METHODOLOGY_NOT_APPROVED | ← | ← | ← |
| `cohort_comparison`, `portfolio_history`, completeness, regulatory | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE |

**Totals:** A — 18 available, 2 not applicable, 5 unavailable, 1 model, 2
methodology. B — 20 available. C — 18 available, 2 assumption-required.
D — 15 available, **5 not applicable**.

**What each portfolio proves.**

- **D is the asset-agnosticism proof.** Not a mortgage, no valuation, no
  geography, never named in any code path — and it keeps every credit and
  cash-flow capability, losing exactly the five that need a collateral *value*.
  They drop out as `NOT_APPLICABLE`, not `UNAVAILABLE`, because no field could
  be supplied to fix them.
- **C proves the determinism model is per-structure, not per-product.** Its
  yield is unknowable *and* so is its principal, because FRXX fixes the total
  instalment. A test flips only `amortisation_type` to `BLLT` on the same
  floating book and the WAL becomes `AVAILABLE` — the distinction that would
  have been lost if determinism were one flag instead of two.
- **B proves the states resolve rather than stick.** It is the only portfolio
  carrying loss and prepayment evidence and the only one that gets those
  capabilities.
- **All six states are reached** across the four, asserted by a test — a state
  nothing can produce is a state nobody has thought through.

---

## 6. MI Query

Previously an unresolved metric produced: *"that measure does not exist here.
Ask for a governed measure — e.g. balance, LTV, interest rate…"* — which is
wrong twice when someone asks an equity-release book for a WAL. The measure
exists; the portfolio is why it does not apply.

Now, before refusing, MI Query asks the capability registry. Actual responses:

> **"What is the contractual WAL of this portfolio?"** *(equity release)*
> Contractual weighted average life is NOT_APPLICABLE for this portfolio. No
> exposure carries a contractual principal profile — the book reports RREL35 =
> OTHR. For a lifetime mortgage that is correct rather than missing: repayment
> is contingent on death, sale or long-term care, so no contractual repayment
> date exists. A legal long-stop maturity is not a contractual life and is not
> used as one.

> **"What is the YTM?"** *(floating French)*
> Contractual yield to maturity (periodic) is ASSUMPTION_REQUIRED… the
> principal share of each payment depends on an unknown future reference rate.
> No rate path is assumed.

> **"What was CPR over the last 12 months?"** *(one snapshot)*
> Constant prepayment rate (observed) is UNAVAILABLE… needs at least 2 governed
> snapshots; 1 is available.

> **"What is WA LTV?"** *(unsecured)*
> Weighted-average current LTV is NOT_APPLICABLE… no collateral valuation is
> carried, so there is no value to weight a loan against.

> **"What is the default rate?"**
> METHODOLOGY_NOT_APPROVED… Use arrears_90_plus for delinquency, or read
> default_amount directly, and say which you used.

Two behaviours are asserted as firmly as the answers. **A capability that IS
available falls through to the ordinary query path** — answering here would be
a second route to the same number, which is what this sprint exists to prevent.
And **"what is the unicorn ratio?" still gets the generic refusal** — the
registry must not become a catch-all that dresses every unparsed question up as
a known metric. Every response ends "no other measure has been substituted for
the one you asked about."

Longest-phrase matching means "expected weighted average life" resolves to
`expected_wal`, not to `contractual_wal` — answering the latter would tell a
user their modelled question was a contractual one.

---

## 7. Copilot and agent exposure

One tool, `portfolio_capabilities`, taking the registered tool count to 25. It
answers the question an agent should ask first and previously could not:

```json
{"metric": "contractual_wal", "status": "NOT_APPLICABLE",
 "reason_code": "CONTINGENT_REPAYMENT",
 "explanation": "No exposure carries a contractual principal profile ..."}
{"metric": "observed_cpr", "status": "UNAVAILABLE",
 "reason_code": "MISSING_REQUIRED_INPUT",
 "missing_inputs": ["unscheduled_principal_collections", ...]}
```

The five-step loop the brief asks for is available: **what exists**
(`portfolio_capabilities`), **whether it is available here** (the status),
**the value** (the existing metric tools), **methodology and provenance**
(`include_definition: true`, plus `explain_values`), and **why not**
(reason code, explanation, `missing_inputs`).

No `get_wal`, `get_ytm`, `get_cpr` tools were created. Filtering by category or
status keeps a survey to one call.

---

## 8. React — reviewed, not integrated

This is a review, and the honest statement is that **React consumes nothing
from the capability registry today**. The frontend renders from its own
component set and the static-pools chart configuration; nothing in
`frontend/mi-agent-ui` reads the registry.

What the registry changes is that it *could*, without new backend methodology:
each capability publishes a display name, unit, category, methodology id and
calculation source, which is what a registry-driven component needs. Whether a
screen shows contractual WAL remains configuration, and should — forcing every
capability onto every portfolio screen is exactly the ERM-shaped product the
sprint is trying to avoid, in reverse.

**Not done, and not claimed:** no React component was changed, and no UX was
redesigned.

---

## 9. Readiness

`FrameworkMetric` gains a `capability` field, and 9 of 50 framework metrics now
name one — total balance, loan count, average loan size, WA LTV, geographic
concentration, prepayment, losses, contractual WAL and contractual YTM.

Two tests enforce the direction of dependency: a named capability must exist,
and where both the framework metric and the capability declare a calculation
source they must be **the same module**. That second test earned itself
immediately by catching `CONC_GEOGRAPHIC` declaring
`analytics_lib.concentration.top_n_concentration` while the capability declared
`concentration_tests.metrics.largest_group_share`. On inspection these are two
entry points at different granularity rather than a duplicated formula — a
top-N table of which the largest region is the n=1 case, versus a single share
evaluated against a covenant threshold with denominator options. The
declaration was aligned to the tool route and the overlap recorded in §11
rather than merged, since merging two concentration engines is not this
sprint's work.

**41 metrics remain unlinked**, resolving through the concentration library as
before. That is mechanical work, not a design gap.

---

## 10. Methodology and provenance

Every capability publishes `methodology`, `calculation_source`, `unit`,
`basis`, `time_basis`, `weighting` and `aggregation` — reachable through
`portfolio_capabilities` with `include_definition: true`. The methodology
identifiers are the ones the calculations already carry, referenced rather than
restated:

```
OBSERVED_SMM@v2   OBSERVED_CPR@v2   OBSERVED_LOSS@v2   OBSERVED_RECOVERY@v2
OBSERVED_LOSS_SEVERITY@v1   OBSERVED_SERIES@v1   CURRENT_LTV@v1
CONTRACTUAL_SCHEDULE@v1   CONTRACTUAL_WAL@v1   CONTRACTUAL_YTM_PERIODIC@v1
```

The `basis` field — `observed` / `contractual` / `modelled` — is the vocabulary
boundary §10 of the brief asks for, and a test rejects any other value. It is
what stops a measured CPR being read as a projected one, and it is why
`expected_wal` is registered at all: not because it is planned, but so a
consumer asking for it learns it is `modelled` and gets pointed at
`contractual_wal` and `observed_cpr` instead.

---

## 11. Remaining methodology gaps

| Gap | State | Note |
|---|---|---|
| **Default rate** | `METHODOLOGY_NOT_APPROVED` | absorbing vs reversible, count vs balance, period vs cumulative, CRR materiality. Unowned since 2.5C |
| **Cure rate** | `METHODOLOGY_NOT_APPROVED` | what counts as a cure, over what window, and whether re-default reverses it |
| WA seasoning / remaining term / borrower age | not registered | evaluator would serve them; no library metric exists |
| Day-count-exact YTM on Annex 2 | field gap | `day_count_convention` is CREL122, Annex 3 only |
| YTW | exposure gap | no represented option set, no liability-side entity |
| `outstanding_balance_period_*` | UNKNOWN | undocumented; two prior reviews disagree. Not consumed |

**Ten weighted-average helpers** exist across the PowerPoint resolver, three API
routes, three analytics modules, the history handler, the simulation oracle and
the governed evaluator. Tested on a common input **they agree** — all use
pairwise deletion, differing only in rounding — so this is not a live
correctness defect and is not reported as one. It is ten places one convention
can drift invisibly. A test now pins the agreement so a divergence fails loudly;
routing them to the shared evaluator is real work and deliberately out of scope,
and `simulation.reference_truth` must stay independent by design, since its
purpose is to be an oracle that shares no code with what it checks.

---

## 12. Performance

| Workload | Before | After |
|---|---|---|
| Capability discovery, 10k rows | 839 ms | **6.9 ms** |
| Capability discovery, 100k rows | 2,364 ms | **51.6 ms (46×)** |
| — of which availability resolution | — | **0.22 ms** |
| Computing one capability it reports available (contractual WAL, 10k) | — | 6,725 ms |

**Discovery is 94× cheaper than computing a single metric it describes**, which
is the property that makes surveying viable: an agent must not pay for a
cash-flow enumeration to be told one is possible.

The 46× came from profiling, not guessing. The portfolio scan was calling a
Python normaliser **per row** to build the amortisation and rate-type mixes —
for columns with five distinct values in them. Grouping on the raw column and
normalising the distinct values instead is the same fix as the Sprint 2
`stratify` vectorisation, and it is the third time in this programme that a
per-row Python call in a pandas path has been the bottleneck.

---

## 13. Regression

Baseline `670c404`, candidate `ccbcf56`, each from a pinned, clean worktree,
verified by `rev-parse` before and after.

| | Baseline `670c404` | Candidate `ccbcf56` | Δ |
|---|---|---|---|
| passed | 5,245 | 5,277 | **+32** |
| failed | 67 | **64** | **−3** |
| errors | 13 | 13 | 0 |
| skipped | 33 | 33 | 0 |
| collected | 5,345 | 5,374 | +29 |

**Full ID comparison, both directions:**

```
FAILED only at baseline (REPAIRED by the candidate):
  tests/test_agent_openapi_document.py::test_the_document_is_not_stale
  tests/test_agent_openapi_document.py::test_every_registered_tool_has_a_route
  tests/test_agent_openapi_document.py::test_each_route_publishes_the_registrys_own_schema

FAILED only at candidate : none
ERROR  only at baseline  : none
ERROR  only at candidate : none
```

The three repaired failures are the ones Sprint 2.5D introduced and this
sprint's `0149e20` fixed — the stale OpenAPI document. The baseline carries
them because `670c404` predates the fix; the candidate is back to the 64
long-standing failures.

**The +32 is fully accounted for:**

| Source | Tests |
|---|---|
| `tests/test_mi_capability_registry.py` (new file) | 27 |
| readiness framework — contractual metrics and ERM exclusion | 2 |
| OpenAPI tests moving from failed to passed | 3 |
| **Total** | **32** |

Confirmed by collection in both worktrees: 5,345 → 5,374 collected (+29 new
tests), and 3 previously-failing tests now pass.

**Regression neutrality is claimed, and better than neutral:** no test passing
at `670c404` fails at `ccbcf56`, no new error appears, and the candidate
repairs three failures it inherited.

---

## 14. Product assessment

> **If Trakt onboards a new asset class tomorrow and its canonical tape
> contains the required economic concepts, how much new code is required
> before existing generic metrics become usable?**

**None.**

That is not an aspiration — portfolio D is the demonstration. It is unsecured
credit, structurally unlike a mortgage, and it was added as a data fixture with
no change to the resolver, the registry conditions, or any analytics module. It
resolved to 15 available capabilities and 5 correctly-not-applicable ones on
first run.

What a new asset class needs is a canonical tape carrying the concepts — the
mapping work that already exists — and nothing else. What it would need code
for is a *new economic condition* nobody has expressed yet: if some future
product's capability depended on, say, a lease residual or a guarantor
structure, that condition would be a new named predicate in `capability.py`,
about ten lines, reusable by every capability thereafter.

The registry itself never needs an asset-class branch, and a test enforces that
it does not acquire one.

---

## 15. Sprint 3 readiness

> **Can the Securitisation Readiness Agent now discover what Trakt knows about
> a portfolio, request the appropriate metrics, understand
> unavailable/not-applicable capabilities, and investigate without being
> hard-coded to ERM?**

**Yes for discovery, retrieval and refusal handling. One genuine blocker
remains, and it is not architectural.**

The agent can call `portfolio_capabilities` to learn what is worth asking for,
call the existing tools for values, read methodology and provenance from the
same catalogue, and — the part that was missing entirely before this sprint —
distinguish *"this portfolio has no WAL"* from *"Trakt lacks a field"* from
*"that would need an assumption"* from *"we have not settled the definition"*.
Those four produce four different investigative next steps, and an agent that
cannot tell them apart either invents numbers or abandons open lines.

**Nothing is hard-coded to ERM**, and the four-portfolio matrix is the evidence
rather than the claim.

**The blocker is default rate and cure rate.** They are the two metrics a
securitisation review quotes that Trakt still cannot produce, and the reason is
a decision rather than data or code — both are one owned definition away from
being registered. Everything else the agent needs is in place.

The narrower caveat worth carrying: React does not yet consume the registry, so
anything the agent surfaces will not automatically appear on a screen. That is
a product-integration task, not a Sprint 3 dependency.
