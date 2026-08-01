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

## Test suites

`tests/concentration_tests/` (library, metrics, matching, governance,
evaluation, compat) · `tests/operations_control/test_concentration_routes.py`
(review-surface API) · `mi_agent_api/tests/test_concentration_tests_api.py`
(service, routes, MI Query/Copilot delegation) ·
`frontend/mi-agent-ui/src/components/risk/RiskLimitsWorkspace.test.tsx`
(React workspace).
