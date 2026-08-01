# Concentration tests — client covenants as governed portfolio controls

End-to-end capability: a lender's concentration tests, eligibility limits and
portfolio-level covenants are collected once at onboarding, extracted into
structured proposals, matched to a shared governed metric library, approved by
a human operator, activated as immutable versioned configuration, evaluated
deterministically against the funded canonical dataset, and served — from one
evaluation service — to the React Risk Limits workspace, MI Query and
Microsoft 365 Copilot.

## Architecture

```
client response / facility schedule / covenant workbook
        │  (one mandatory onboarding question: config/onboarding/field_catalogue.yaml → risk_limits)
        ▼
extraction + library matching (deterministic; model may PROPOSE via the assist seam)
        │  mi_agent/concentration_tests/matching.py
        ▼
TestProposal  (matched | composed | ambiguous | unsupported, concern codes,
        │      confirmation questions)         mi_agent/concentration_tests/models.py
        ▼
operator review  →  answer confirmations · edit parameters · approve · reject ·
        │           unsupported · not applicable · clarify · supersede
        │           operations_control/concentration.py  +  /ops/concentration/* routes
        ▼
ActiveConfiguration vN  (immutable, content-hashed, tenant-scoped, audited)
        │  mi_agent/concentration_tests/store.py → blob://operations-control/{client}/concentration-tests/
        ▼
deterministic evaluation  (current + governed prior snapshot, statuses,
        │  headroom, movement, drill-through)   mi_agent/concentration_tests/evaluation.py
        ▼
one governed service:  mi_agent_api/concentration_tests_api.py
        ├── GET /mi/concentration-tests            → React Funded → Risk Limits
        ├── GET /mi/concentration-tests/drillthrough
        ├── GET /mi/concentration-tests/history
        └── chat route `risk_limits` (mi_agent_api/chat_routing.py)
              → MI Query and Copilot (`askTraktMi` delegates to the same service)
```

Design rules, enforced by tests:

* **Deterministic calculations.** Every activated test resolves to a registered
  evaluator in `metrics.py` with explicit approved parameters. There is no
  formula language; client text can never become executable code.
* **Human approval before activation.** `ConcentrationGovernanceService.approve`
  refuses while any concern or confirmation question is unresolved;
  `activate` only reads proposals in status `approved`.
* **Evidence and provenance.** Every proposal and active test carries the
  source reference, wording, locator, extraction and mapping confidence, the
  operator decision with identity and timestamp, effective date and version.
* **No silent assumptions.** Unresolved definitions (Net WAC, joint basis,
  denominators, HPI series…) become concern codes plus short confirmation
  questions; they block approval until answered or explicitly cleared.
* **Fail closed.** Unknown / unsupported / data-missing tests report
  `unavailable` / `insufficient_data` — never PASS, never zero.
* **One governed truth.** React, MI Query and Copilot consume the same
  evaluation output; drill-through reuses the evaluator's own row mask, so the
  population reconciles to the numerator by construction.

## Lifecycle states

Proposal: `proposed → pending_confirmation → pending_approval → approved |
rejected | unsupported | not_applicable | clarification_requested`, then
`superseded` once a later approval replaces it.

Evaluation: `pass | warning | breach | unavailable | insufficient_data |
pending_effective_date | expired`, with `data_status ∈ ok | data_missing |
data_partial | external_reference_unconfigured`.

Onboarding question: `pending_client_response | supplied | not_applicable |
deferred_with_reason` — operator-recorded; approval is structurally blocked
while pending, and a blank answer cannot be recorded as supplied
(`operations_control/onboarding/validation.py`,
`OccAgentService.record_concentration_outcome`).

## The library

`config/risk/concentration_test_library.yaml` — shared, versioned; client
thresholds never live here. Each entry declares: `metric_id`, names, category,
description, evaluator, unit/precision, supported operators, parameter schema,
required/optional **field roles** (resolved to canonical columns at evaluation
time, readable labels preferred over regulatory codes), aliases,
composability, implementation status and version.

Categories: geography, property_value, loan_balance, borrower, rate_product,
ltv, performance, composition, external_index, primitive. External-index
metrics are `interface_only`: they evaluate only when an approved
`ExternalIndexProvider` is configured, and are never simulated.

### Adding a metric

1. Add the entry to the YAML (schema is validated at load; unique id, known
   evaluator, closed vocabularies).
2. If a new evaluator is needed, register it in `metrics.py::EVALUATORS` and
   add it to `library.py::KNOWN_EVALUATORS`.
3. Add hand-checkable cases to `tests/concentration_tests/test_metrics.py`
   (the drill-mask reconciliation test is parametrised — add the metric).
4. Add aliases so extraction can match contractual phrasings; extend
   `matching.py` parameterisation if the metric needs bespoke parameter
   population.

### Matching outcomes

* `matched` — direct metric, parameters resolved.
* `composed` — represented through approved primitives/list parameters
  (London + South East = `geo_region_share` with two regions; balances above
  £1m = `balance_above_share`; borrowers with ≥3 loans =
  `borrower_multi_loan_share`).
* `ambiguous` — plausible metric, unresolved contractual definition → concern
  codes + confirmation questions.
* `unsupported` — a structured implementation request, never an improvised
  formula.

### Document conventions the extractor reads

Facility schedules are drafted, not written for machines. Three conventions
are handled deterministically because misreading them produces a schedule of
empty thresholds rather than an honest question:

* **Bracketed figures.** A negotiated number stays in square brackets until
  the document is agreed — `[50]%`, `£[1,500,000]`, `[3.75]% per annum`.
  `normalise_source_text` unwraps digit-bearing brackets only; `[Reserved]`
  and cross-references are left exactly as drafted.
* **Limits tables.** The comparison lives in the lead-in sentence and the
  numbers live in the rows beneath it. Each row inherits the lead-in's
  operator and denominator, and the reviewer is always shown the lead-in
  together with the row — a row never appears without the sentence that gives
  it meaning. The lead-in is only consumed when rows actually follow it.
  Regions are recognised by ITL1 code as well as by label, so
  `UKI + UKJ | London and South East` collapses to two canonical regions.
* **Defined denominators.** A schedule that defines its own denominator —
  *“Concentration Limit Denominator” means the greater of £33,000,000 and the
  Current Balance of all the Eligible Mortgage Loans* — has that floor read
  once from the definitions and applied to every clause citing the term (see
  below). The definition itself is not proposed as a test.

Wording that states a *concept* still asks. “Loans which have two Borrowers”
resolves to `borrower_joint_share` with `joint_basis: borrower_count` because
the covenant names the count; bare “joint borrowers” raises
`joint_definition_uncertain`, because it says nothing about whether the book
encodes jointness as a classification flag or a borrower count, and the two
can disagree on the same loan.

### Floored denominators

`denominator_floor` measures a share against the **greater** of a contractual
amount and the resolved denominator balance. It is declared only on metrics
whose evaluator actually measures against a balance denominator — a parameter
that could not change a result would be worse than no parameter at all — so
weighted averages and count metrics reject it rather than accept and ignore
it.

The floor is disclosed in `denominator_basis` whether or not it binds
(`current_balance (floored at 33,000,000)` vs
`current_balance (floor 33,000,000 not binding)`), so a reader never has to
infer from the number that a floor was in play. Being an absolute contractual
amount, it applies unchanged in the forward-looking states: a
probability-weighted expected balance is compared against the same floor as
the funded balance. Commercially this is what makes limits looser at ramp-up —
a single loan in a £50k book is 2.5% of a £2m floor, not 100% of the book.

## Persistence

`blob://{TRAKT_OPS_CONTAINER}/{client}/concentration-tests/`:

```
proposals/{proposal_id}.json     mutable working documents (versioned edits)
proposals/index.json             rebuildable index
versions/{0001}.json             immutable activated configurations
current.json                     pointer to the active version
```

Activation supersedes the prior version in place, never overwrites history,
and is idempotent by content hash. Decisions append to the client's
hash-chained OCC audit trail (`OpsStore.append_audit`).

## Evaluation semantics

* Denominator = the approved basis balance (current/original) summed over the
  full governed frame; numerator = the same basis over the evaluator's mask.
* Warning = actual ≥ threshold × `warning_fraction` (max) or ≤ threshold ÷
  fraction (min); `warning_fraction` is per-test configuration (default 0.9,
  the existing amber convention).
* Period change compares against the governed prior snapshot resolved by the
  same loaders the evolution services use — never an arbitrary file. Absent
  prior → explicit `priorAvailable: false`.
* History (`/mi/concentration-tests/history`) evaluates the current approved
  configuration across the real snapshot series; nothing is fabricated.

## Compatibility and migration

The legacy stores are inputs to the governed lifecycle, not competitors:

* `mi_agent_api/risk_limits.py` (Schedule 8 extracted) still computes when no
  approved configuration exists, and its output is presented in the same
  envelope explicitly marked `legacy_extracted` / *not operator-approved*.
* `mi_agent/concentration_tests/compat.py` converts
  `config/client/risk_limits_config.py` dictionaries and extracted YAML
  contracts into **proposals** (idempotently) so operators can approve them
  into governed configuration. `analytics/risk_monitor.py` and its Streamlit /
  PPTX consumers are unchanged.

## Worked example

Client response to the onboarding question:

> East Anglia ≤ 25%. East Midlands ≤ 15%. London plus South East ≤ 50%.
> Original valuation above £1.5 million ≤ 10%; below £100,000 ≤ 10%. Initial
> principal above £1 million ≤ 10%. Average initial principal balance ≤
> £300,000. Youngest borrower below 55 = 0%. Net WAC ≥ 3.75%. Joint borrowers
> ≤ 90%. House-price-index change ≥ 90%.

Extraction + matching produce:

| Extracted wording | Proposed metric | Parameters | Outcome | Concerns / follow-up |
|---|---|---|---|---|
| East Anglia ≤ 25% | `geo_region_share` | `regions=[East Anglia]`, max 25% | matched | — |
| East Midlands ≤ 15% | `geo_region_share` | `regions=[East Midlands]`, max 15% | matched | — |
| London plus South East ≤ 50% | `geo_region_share` | `regions=[London, South East]`, max 50% | **composed** | — |
| Original valuation above £1.5m ≤ 10% | `property_value_above_share` | `amount=1500000, value_basis=original`, max 10% | composed | — |
| Original valuation below £100k ≤ 10% | `property_value_below_share` | `amount=100000, value_basis=original`, max 10% | composed | — |
| Initial principal above £1m ≤ 10% | `balance_above_share` | `amount=1000000, balance_basis=original`, max 10% | composed | — |
| Average initial principal ≤ £300k | `balance_average` | `balance_basis=original`, max £300,000 | matched | — |
| Youngest borrower below 55 = 0% | `borrower_age_share` | `age=55, comparison=below, age_basis=youngest`, max 0% | composed | — |
| Net WAC ≥ 3.75% | `rate_net_wac` | *(deduction unresolved)*, min 3.75% | **ambiguous** | `net_wac_definition_uncertain` → “Does 'Net WAC' mean customer coupon net of servicing fee only, or net of servicing, hedging and funding costs?” |
| Joint borrowers ≤ 90% | `borrower_joint_share` | *(basis unresolved)*, max 90% | **ambiguous** | `joint_definition_uncertain` → “Joint/single classification or number of borrowers? Number of loans or current balance?” |
| HPI change ≥ 90% | `index_hpi_ratio` | *(series + base date unresolved)*, min 90% | **ambiguous** | `index_series_missing`, `index_base_date_missing` |

Operator sign-off: the client confirms Net WAC = coupon net of a 0.50pp
servicing fee, joint = the loan-level joint classification measured on current
balance. The operator records the answers, sets
`deduction_percent=0.5, deduction_basis=servicing_fee_only` and
`joint_basis=structure_flag`, clears the concerns, and approves each proposal
with an effective date. The HPI test stays pending until an approved index
source is configured — approving it activates a test that reports
`unavailable (external_reference_unconfigured)`, never a simulated value.
`POST /ops/concentration/{client}/activate` (administrator) mints version 1.

Dashboard evaluation (Funded → Risk Limits): each approved test shows current
value, threshold, headroom, utilization, period movement and status from the
governed evaluation — e.g. *Regional exposure — East Anglia: 18.4% (≤ 25%),
headroom 6.6pp, PASS, +1.3pp vs prior* — with the source wording, approver and
configuration version in the detail panel, and a drill-through that reconciles
exactly to the evaluated numerator.

## Three portfolio states (phase 2)

Every supported active test is evaluated across three clearly distinguished
states by ONE shared service (`mi_agent/concentration_tests/forward.py`,
served through the same `/mi/concentration-tests` envelope):

| State | Population | Meaning |
|---|---|---|
| **Funded** | governed funded portfolio only | The contractual compliance position. Never diluted with pipeline data; byte-identical to the phase-1 evaluation. |
| **Expected Forecast** | funded at 100% + each ACTIVE pipeline case × its completion probability | The statistically expected position, using the existing forecast engine's probabilities — never a second model. |
| **Full Pipeline** | funded + 100% of ALL active in-scope pipeline | A deliberately unrealistic **maximum-exposure stress**. Not a prediction, and always labelled as such. |

Excluded from both forward states, with counts disclosed: withdrawn/cancelled
cases, cases already completed (they live in the funded book), unknown-stage
cases; and — expected state only — active cases with no governed probability
(disclosed, never guessed).

### Forecast methodology (as found, reviewed, kept)

The Expected Forecast reuses the deterministic completion-trend model
unchanged (`mi_agent_api/pipeline_history.py` + `pipeline_prep.py`):

* per-case tracking across the client's deduplicated weekly extracts;
* a stage's empirical completion rate = ever-completed ÷ observed at that
  stage, trusted only at ≥ 12 observations (`MIN_OBSERVATIONS`);
* probability hierarchy per case: row-level explicit → empirical stage rate →
  configured stage probability → withdrawn excluded → unknown no probability;
* expected contribution = advance amount × probability
  (`weighted_expected_funded_amount`);
* median observed stage→completion lag; expected completion months from
  explicit dates or stage timing offsets;
* the trailing five-week window governs the funnel FLOW conversion analytics
  (`evolution.py`, `_CONVERSION_WINDOW = 5`, minimum 3 observed weeks), while
  completion probabilities use the full retained weekly history gated by the
  sufficiency floor — a strictly five-week completion window would routinely
  fail the floor and force config fallback.

Methodology review verdict: sound, deterministic, transparent — kept
unchanged. Known limitations, disclosed rather than modelled away:
right-censoring of recent cohorts (rates conservative), portfolio-wide stage
rates (no product/channel segmentation at current volume), and **no
point-in-time reproducibility** — the model reads today's retained extracts,
so historical expected-state comparisons are marked unavailable
(`pointInTimeSafe: false`) rather than fabricated with hindsight bias.

### Metric-family forecast support

Declared in `forward.FORECAST_TREATMENT`, keyed by evaluator:

| Treatment | Evaluators | Expected state | Full Pipeline state |
|---|---|---|---|
| `supported_with_exposure_weighting` | share_of_balance, share_of_balance_numeric, postcode_area_share, dimension_share, vintage_share, arrears_share, filtered_share, joint_borrower_share, weighted_average, field_average, largest_group_share | balance × probability on BOTH numerator and denominator (filters always read real columns); averages use expected weights (Σv·p / Σp) | probability 1.0 |
| `supported_with_scenario_inclusion` | top_n_share, field_maximum, field_minimum, largest_by_field_share, distinct_group_count | **indicative only** — a fractional loan is never a maximum, a top-N member or a count | whole pipeline loans included at 100% |
| `unsupported` | max_count_per_group, multi_loan_borrower_share | not evaluated (no governed borrower-identity linkage for pipeline applicants; absolute count covenants have no meaningful probability-weighted status) | not evaluated |
| `state_independent` | external_index_ratio | same value in every state | same value |

Pipeline basis approximations, disclosed per state:
`original_principal_balance := current_outstanding_balance` and
`original_valuation_amount := current_valuation_amount` (a not-yet-funded
case's advance amount IS its initial balance), and the pipeline region field
feeds the readable-label precedence.

### Pipeline drivers and breach horizon

Drivers (`/mi/concentration-tests/drivers?testId=…`) list the pipeline cases
inside a test's expected numerator, ranked by expected contribution
(balance × probability), with stage, probability source, expected completion
month and a deterministic impact marker (the first case whose cumulative
addition, against the fixed expected denominator, crosses the warning/breach
line). The listed contributions reconcile exactly to expected numerator −
funded numerator. The expected-breach horizon replays the expected state
month by month in expected-completion order and reports the first crossing
period — presentation of existing timing data, not a new methodology.

### Emerging-risk intelligence

`forward.identify_emerging_risks` ranks issues by fixed rules (one risk per
test, most severe wins): 1 current breach · 2 expected breach · 3 expected
warning with headroom below the configurable buffer (default 1.0pp) ·
4 material deterioration (default 0.5pp adverse) · 5 full-pipeline-only
breach · 6 data/methodology limitation (including a standing entry when any
stage runs on configured fallback). The ranking ships to React, MI Query and
Copilot; the language model never chooses or reorders it.

### Worked example — South West ≤ 20%

Funded book: £18.4m South West of £100.0m → **18.4%**, headroom 1.6pp.
Active pipeline (five cases, none excluded):

| Case | Balance | Stage | Probability | Expected contribution | Region |
|---|---|---|---|---|---|
| P-1042 | 640,000 | Offer | 0.86 | 550,400 | South West |
| P-0991 | 510,000 | Offer | 0.86 | 438,600 | South West |
| P-1077 | 455,000 | Application | 0.61 | 277,550 | South West |
| P-1103 | 390,000 | Application | 0.61 | 237,900 | London |
| P-0968 | 120,000 | KFI | 0.24 | 28,800 | London |

Expected numerator = 18.4m + 1,266,550 = **19.67m**; expected denominator =
100.0m + 1,533,250 = **101.53m** → Expected Forecast **19.37%**… with a 19.0%
limit that is an expected breach; against the 20.0% limit here it is a
warning with 0.63pp expected headroom. Full Pipeline numerator = 18.4m +
1,605,000; denominator = 102.115m → **19.59%**. MI Query then answers:
*"South West concentration is compliant on the funded book at 18.4% against a
20.0% limit. Based on the completion-trend model it is expected to rise to
19.4% (0.6pp of expected headroom; full pipeline exposure would reach 19.6%).
Three pipeline loans drive the expected increase, led by P-1042 at £640,000
currently at Offer."* — every figure from the governed envelope, the caveat
from the methodology payload.

### OCC review UI

`frontend/operations-control-ui` gains a **Concentration tests** area
(`/concentration` → client picker, `/concentration/{client}` → review):
paste-the-response extraction, filterable proposal list with match outcomes,
side-by-side wording/mapping/parameters/confidence detail, confirmation
questions with recorded answers, permitted-field edits (versioned, audited),
approve (admin, disabled-with-reasons while blockers remain and refused
server-side regardless), reject / unsupported / not-applicable / clarify /
supersede, an activation panel that states exactly which tests become active,
and version history. `MockConcentration.ts` mirrors the backend's refusals so
mock mode and tests exercise real governance behaviour.

## Known gaps

* `perf_cumulative_loss_share` and `perf_prepayment_share` are catalogued
  `not_implemented` (they need loss/flow data the canonical dataset does not
  yet carry) — proposals map to `unsupported` with an implementation request.
* Grouped top-N (e.g. “top 3 brokers”) has no dedicated metric yet; the
  largest-group primitive covers N = 1.
* No `ExternalIndexProvider` implementation ships; the HPI interface is
  governed but unconfigured by design.
* Delinquency migration across snapshots remains with the existing
  `mi_agent/risk_monitor` migration engine.
* Forward-looking states are evaluated for the total book only: pipeline
  exposure carries no portfolio provenance, so a narrowed portfolio context
  reports the forward states unavailable with the reason (funded results are
  unaffected).
* The completion-trend model is not point-in-time reproducible for historical
  dates; historical expected-state comparisons are unavailable until
  persisted model snapshots exist (disclosed in the methodology payload).

## Test suites

`tests/concentration_tests/` (library, metrics, matching, governance,
evaluation, compat) · `tests/concentration_tests/test_schedule_8.py` (a real
warehouse-facility schedule end to end: all seventeen limits must extract with
a threshold and an operator) ·
`tests/operations_control/test_concentration_routes.py`
(review-surface API) · `mi_agent_api/tests/test_concentration_tests_api.py`
(service, routes, MI Query/Copilot delegation) ·
`frontend/mi-agent-ui/src/components/risk/RiskLimitsWorkspace.test.tsx`
(React workspace).
