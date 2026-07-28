# Period Change Analysis — the first governed workflow over the Business Semantics Registry

How Trakt answers "what changed?" deterministically, and why every number it
publishes can be reproduced from the record it leaves behind.

Audience: engineers extending the workflow, and reviewers checking that the
analysis is governed rather than asserted.

---

## 1. Purpose

Period Change Analysis answers questions of the form *"what moved between two
reporting dates, and by how much?"* — for example:

* What changed in the portfolio this month?
* How has the portfolio changed since the previous reporting date?
* What moved between March and June?
* What improved and deteriorated quarter-on-quarter?
* What are the main drivers of the balance movement?
* How has portfolio composition changed?

It is a **deterministic analytical capability**, not a narrative feature. No LLM
is involved in any calculation, ranking, interpretation or summary. The workflow
is the numerical source of truth; a presenter renders what it produced and adds
nothing.

The workflow is the first consumer of the governed **Business Semantics Registry**
(`config/business_semantics_registry.yaml`, schema version 2, registry version
0.2.0). Field meaning — what a field measures, how it aggregates, whether a rise
is good or bad — is read from that registry, never inferred from a field name.

---

## 2. Architecture and execution path

```
  question
     │
     ▼
  mi_agent/parsed_question.py           ── the EXISTING single parse. No second
     ParsedQuestion → MIQuerySpec          parser is introduced anywhere below.
     │
     ▼
  mi_agent_api/recogniser_registry.py   ── governed capability routing
     Recogniser("period_change_analysis", priority=85)
     │
     ▼
  mi_agent_api/period_change_route.py   ── the ONLY platform-aware module
     ├─ evolution.funded_frames()          existing governed per-period frames
     ├─ chat_routing._apply_lens_filter()  existing governed portfolio narrowing
     └─ renders the existing chat envelope
     │
     ▼
  mi_agent/period_change/               ── the governed workflow (pure)
     ├─ recognition.py    intent, mode, period request
     ├─ periods.py        two snapshots, or an explicit failure
     ├─ selection.py      the governed field-selection policy
     ├─ calculations.py   aggregation + temporality + directionality
     ├─ distribution.py   composition shift
     ├─ bridge.py         balance reconciliation
     └─ workflow.py       orchestration, ranking, summary, audit
     │
     ▼
  mi_agent/business_semantics.py        ── the BSR v2 loader
     │
     ▼
  PeriodChangeResult                    ── the governed result contract
     │
     ▼
  existing presenters / API adapters
```

### Package boundaries

| Module | Rule |
|---|---|
| `mi_agent/business_semantics.py` | Reads YAML. No pandas, no API, no workflow knowledge. |
| `mi_agent/period_change/` | Pure domain. Takes `SnapshotFrame`s, returns a typed result. **Never imports `mi_agent_api`.** |
| `mi_agent_api/period_change_route.py` | The only module that knows about datasets, lenses and envelopes. **Calculates nothing.** |

### What was reused rather than rebuilt

| Need | Existing component used |
|---|---|
| Parse | `mi_agent.parsed_question.ParsedQuestion` (the single parse) |
| Routing | `mi_agent_api.recogniser_registry` |
| Snapshots / frames | `mi_agent_api.evolution.funded_frames` |
| Portfolio narrowing | `chat_routing._apply_lens_filter`, `mi_agent.portfolio_scope` |
| Numeric coercion | `analytics_lib.numeric.coerce_numeric` |
| Percent storage scale | `mi_agent.mi_dataset_profile.percent_storage_scale` |
| Missing-category label | `analytics_lib.stratify.UNKNOWN_LABEL` |
| Error taxonomy | `trakt_core.errors.ErrorCode` |
| Result envelope | `chat_routing._envelope`, `trakt_core.envelope.GovernedResult` |
| Cross-portfolio detection | `mi_agent.portfolio_lens.resolve_comparison_lenses` |

### What this workflow explicitly does not do

No Portfolio Risk Comparison. No covenant or limit logic. No materiality
thresholds. No forecasting. No LLM in any calculation. No new parser, no new
canonical-field logic, no change to canonical transformations, onboarding, or the
source field registry.

---

## 3. Recognition and precedence

The recogniser (`mi_agent/period_change/recognition.py`) reads the structured
parse — `temporal_mode`, `metric`, `compare_periods`, `start_date`/`end_date` —
plus a controlled change vocabulary. It never matches on the literal words
"period change".

**Matches** on explicit change language (*changed, movement, moved, increase,
decrease, improved, deteriorated, drivers of, portfolio evolution, shift*), on
period tokens that only occur in change questions (*month-on-month,
quarter-on-quarter, year-on-year, year-to-date, current versus previous, since
the previous*), or on an explicit two-period construction (*between March and
June*, *compare 2025-12-31 with 2026-03-31*).

**Declines**, each with its own recorded reason:

| Decline reason | Why |
|---|---|
| `not_the_funded_book_view` | pipeline / forecast views are other capabilities |
| `forecast_question` | forecasting is out of scope |
| `trend_series_question` | a time series belongs to the `evolution` route |
| `raw_record_request` | loan-level tape requests |
| `transaction_reconciliation_request` | ledger / remittance reconciliation |
| `static_cross_portfolio_comparison` | two portfolios, not two periods |
| `single_metric_named_period_comparison` | the incumbent `temporal_compare` route |
| `no_change_language` | a single-date position, not a movement |

### Registry position

`period_change_analysis` is registered at **priority 85**: after `period_movement`
(70) and `portfolio_summary` (80), before `temporal_compare` (90). Every
historical route keeps its relative position and every question it already
answered.

Two consequences are deliberate and worth stating plainly:

* **`period_movement` keeps its questions.** "What has changed versus the prior
  month?" and "how has the portfolio changed month on month" still route to the
  existing composite answer. Promoting them to this workflow is a governed
  product decision, not a side effect of adding it (see §12, deferred items).
* **`temporal_compare` keeps single-metric named-period comparisons.** "Compare
  October and November funded balance" is deferred explicitly. A named-period
  question with no single metric focus ("what moved between March and June?") is
  a portfolio-wide change question and is answered here.

---

## 4. Period resolution

`mi_agent/period_change/periods.py`. The platform's reporting dates are
irregular, so nothing assumes a calendar month end.

| Request | Behaviour |
|---|---|
| explicit dates | the latest available snapshot **at or before** each requested date |
| `current_vs_previous` | the previous **governed snapshot** — no calendar arithmetic |
| `month_on_month` / `quarter_on_quarter` / `year_on_year` | end date shifted by 1 / 3 / 12 months, then nearest snapshot |
| `year_to_date` | the latest snapshot on or before 31 December of the prior year |
| nothing named | the latest available pair |

A month end shifted by whole months stays a month end (31 March − 1 month =
28 February), so a month-on-month request never lands on the wrong snapshot in a
long month.

### On-or-before, and the gap ceiling

An **explicitly requested** date resolves backwards only. Absolute-nearest
matching steps over period boundaries: a request for 15 January is one day nearer
31 December than 31 January, so it would answer about December while appearing to
answer about January. The rule is therefore *the last governed position at or
before the date asked for*. A date preceding every snapshot is refused; a later
snapshot is never substituted for a period that had not yet occurred.

**Relative** modes keep absolute-nearest matching, because their target date is
derived arithmetically — on a book whose month end is the 30th, a month-on-month
target of the 30th must be allowed to match a 31st. A day-precise relative target
equidistant from two snapshots is refused as ambiguous.

Either way the gap is bounded. `max_snapshot_gap_days` (default **45**, and
overridable per portfolio for a different reporting cadence) is the furthest a
request may resolve. Beyond it the comparison fails as `ambiguous_period_range`
rather than answering about a distant period:

```yaml
period_resolution:
  max_snapshot_gap_days: 45
  max_snapshot_gap_days_by_portfolio:
    quarterly_book_id: 100
```

Every resolution records: requested start, requested end, resolved start
snapshot, resolved end snapshot, **the gap in days at each end**, the ceiling
applied, the interval between the two snapshots, the flow basis, resolution
method, whether either end was adjusted, the adjustment wording, the full list of
available snapshots, and the portfolio scope.

### Explicit failures

| Reason | Trigger |
|---|---|
| `insufficient_snapshots` | fewer than two governed snapshots |
| `portfolio_absent_at_period` | a requested portfolio is missing at one date |
| `ambiguous_period_range` | an unparseable token, or a day-precise date equidistant from two snapshots |
| `reversed_period_range` | start later than end |
| `identical_snapshots` | both dates resolve to the same snapshot |
| `cross_tenant_access` | the requested scope is not inside the authorised scope |

The current snapshot is never silently compared with itself: a same-date request
fails rather than sliding onto the previous period.

---

## 5. Role handling

| `analytical_role` | Treatment |
|---|---|
| `measure` | eligible for direct portfolio-level period-change analysis |
| `dimension` | eligible for distribution / composition-shift analysis |
| `derived_input` | never a standalone overview metric. It feeds a governed migration calculation; no new migration formula is invented here |
| `supporting_attribute` | never an ordinary portfolio metric |

A caller who *explicitly requests* a derived input by name gets it analysed, with
its role carried on the result so a presenter can caveat it. What §6 of the
specification forbids — and what the policy enforces — is such a field appearing
unbidden as an overview metric.

---

## 6. Temporality handling

| `temporality` | Calculation | `movement_basis` |
|---|---|---|
| `point_in_time` | aggregate at each date; movement = end − start | `point_in_time_difference` |
| `period_flow` | each date's aggregate is that period's own flow; the two are compared as independent period totals | `flow_level_comparison` |
| `cumulative` | movement = end cumulative − start cumulative | `cumulative_difference` |
| `static_baseline` | reported as a reference value at each date; **excluded from change ranking**, no movement calculated | `static_baseline_reference` |

`period_flow` movements carry an explicit note on every result:

> Period flows are compared as independent reporting-period totals. The movement
> is the change in the periodic flow level between the two reporting periods, not
> an amount arising between the two dates.

That distinction is the point: a differenced cumulative *is* the interval amount;
a differenced pair of flows is a change in run-rate, and must never be read as
the first.

### Flow basis — are the two periods the same length?

The registry says a field is a `period_flow`; it does not say what period. Two
flows are comparable period totals only when each covers a reporting period of
similar length, so the workflow derives that length from the gap to each
snapshot's immediate predecessor in the governed series:

| Case | Behaviour |
|---|---|
| lengths within `flow_basis_tolerance_days` (default 5) | compared normally |
| lengths differ by more | `not_comparable_period_basis`: both values reported, **no movement calculated**, mismatch noted |
| a length cannot be established (the opening snapshot is the earliest held) | compared, with an explicit note that equal periods could not be confirmed |

Unknown is not treated as mismatched. With exactly two snapshots the opening
period's length is unknowable, and disqualifying every flow on that basis would
report uncertainty as a defect.

A cumulative value that **falls** between snapshots keeps its negative movement —
it is never floored at zero — and raises a data-quality warning
(`source_convention_uncertain`) noting a possible restatement or source
convention difference.

---

## 7. Aggregation rules

`default_aggregation` is followed exactly.

| Aggregation | Behaviour |
|---|---|
| `sum` | sum of numerically valid values; invalid rows counted and excluded |
| `average` | mean over the valid population; the valid count is the recorded denominator |
| `weighted_average` | `Σ(value × weight) / Σ(weight)` using the BSR `weight_field` |
| `share` | numerator and denominator calculated explicitly from `share_basis` |
| `distribution` | handled by the distribution analysis, not as a single aggregate |

**A weighted average never falls back to an unweighted mean.** An absent weight
field, an absent weight column, an empty weighted population or a zero total
weight each produce a controlled status (`invalid_weight_population` /
`zero_denominator`) and no value. An unweighted answer to a weighted question is
a different number wearing the same label.

Each weighted average records: metric field, weight field, valid weighted
population, excluded rows, total eligible weight, numerator and denominator.

**A count share is never silently turned into a balance share.** `share_basis`
decides; `count` is the v2 default for flag shares, `balance` uses the governed
balance field, and any other basis is refused rather than downgraded. The basis
is **displayed**, not merely recorded: the rendered table carries a **Basis**
column reading "share of loan count", "weighted average by
current_outstanding_balance", "sum" and so on, so a reader does not have to know
the registry to tell an arrears share of *loans* from one of *balance*. A flag
value outside the governed yes/no vocabulary is excluded from *both* the
numerator and the denominator and reported as `source_convention_uncertain` —
counting an unknown code as "no" would understate every flag share on a tape with
a local convention.

### Units

The BSR carries meaning, not units. The unit is resolved from governed registries
in a fixed precedence: the curated MI layer's `format`
(`mi_agent/mi_semantics_field_registry.yaml`), then the canonical registry's
`format` (`config/system/fields_registry.yaml`), then the BSR
`default_aggregation`. No field name is ever parsed.

| Unit | Movement expressed as |
|---|---|
| `currency` | absolute amount **and** percentage of the start value |
| `count` | absolute count and percentage |
| `percentage_point` | percentage points **and** basis points; no percentage-of-start |
| `ratio` | absolute movement and percentage |
| `not_applicable` | no numerical percentage movement (dates, categories) |

A rate moving 4% → 5% has risen by one point. Reporting that as "25% higher"
invites exactly the misreading this rule prevents, so it is not reported at all
for point units.

The percent storage scale (fraction `0.51` vs points `51`) is decided **once
across both snapshots together**. Deciding it per snapshot could read the start
as a fraction and the end as points, manufacturing a 5,000-point movement.

A zero start value yields no percentage change, with an explicit note. The
absolute movement is unaffected.

---

## 8. Field selection

`config/period_change_selection.yaml` is the governed policy;
`mi_agent/period_change/selection.py` applies it. The policy is configuration so
it can be reviewed and changed without touching calculation code, and every
exclusion it makes is recorded with a reason in the audit block.

**Requested-metric mode** — the caller named a field. Only that field is
analysed, or the request fails; no other metric is substituted.

**Concept mode** — the caller named an analytical concept ("what changed in
credit quality?"). Every eligible entry under that concept, up to a readability
cap.

**Portfolio-overview mode** — the caller asked broadly. The 106 `period_change`
entries are reduced to a governed subset by, in order: workflow tag → role →
asset applicability → availability → confidence floor → MI core/extended tier →
concept coverage → non-duplication (at most one measure per
`(concept, aggregation, temporality)` signature).

Selection runs in **two passes**. The first reserves one measure for every
eligible concept; the second spends the remaining budget on second and third
measures in concept order. Without the reservation the total cap was consumed by
the concepts listed first and the last ones — coverage, liquidity — vanished from
the overview entirely, which a reader takes to mean "nothing changed there"
rather than "not reported".

The overview covers exposure, payment performance, credit quality, leverage,
collateral, valuation, pricing, maturity, cashflow, loss, coverage and liquidity,
plus up to four composition dimensions.

### Asset applicability

`cross_asset` entries apply generally. An asset-specific entry applies only to a
matching book. **An asset class is used only when a caller states it** — the MI
layer has no governed asset-class signal at query time
(`snapshots.portfolio_risk_type` exists but *defaults* to `erm` with no
evidence), so with no stated class only `cross_asset` entries are eligible. That
is 92 of the 106 period-change entries. See §12 for the extension point.

### Availability

A field absent from both snapshots is omitted. A field present in only one is
retained, marked `not_comparable_due_to_availability`, and never replaced with a
loosely related field. A column of nothing but nulls is not "available" — a
metric with no population at either date reads as a real zero.

---

## 9. Distribution analysis

For `analytical_role: dimension` + `default_aggregation: distribution`, each
category reports start count, end count, start and end count share, count-share
movement, and — where the governed balance field exists in both snapshots —
balance, balance share and balance-share movement. The largest positive and
negative shifts are ranked.

Shares are computed against **each snapshot's own denominator**, so a portfolio
that grew does not manufacture a share movement, and shares always sum to 1
within a snapshot.

Categories present in only one snapshot are reported with a `presence` of
`start_only` / `end_only`. Missing, blank and placeholder values collapse into
the single `Unknown` bucket `analytics_lib.stratify` already uses, so a value
that is blank at one date and null at the other is not two categories.

No concentration index is computed — that belongs to the existing risk-monitor
service, and a second implementation would produce a second answer. No category
increase is labelled a deterioration: only a governed ordering can say that, and
this workflow has none.

---

## 10. Balance bridge

Where a stable loan identifier and the governed balance field exist in both
snapshots:

```
opening balance
  + closing balances of loans new in the end snapshot
  − opening balances of loans that exited
  + balance movement on loans present at both dates
  = closing balance
```

The reconciliation is **checked**, not asserted, with a documented rounding
tolerance of 0.01. Identifier fields are tried in canonical order
(`loan_identifier`, `original_loan_identifier`,
`underlying_exposure_identifier`).

**Loan identity is composite where provenance allows it.** Originators reuse
simple sequences, so on a consolidated book `loan_identifier` alone is not an
identity: originator A's loan `0001` exiting and originator B's `0001` arriving
would read as one continuing loan whose balance moved. Where
`source_portfolio_id` exists in both snapshots the key becomes
`source_portfolio_id + loan_identifier`; `identifier_fields` records which
columns were used, so an audit can see whether provenance was available to
disambiguate.

The bridge is **omitted entirely**, with an explicit limitation, when the balance
field is absent, the book reports more than one currency, no canonical identifier
is present in both snapshots, identifiers are duplicated, or any key component is
missing. An estimated bridge looks like a reconciliation and is not one — and a
bridge that adds GBP to EUR reconciles arithmetically while meaning nothing,
which is the worst failure mode a reconciliation has.

Deliberately out of scope: cashflow waterfalls, attribution models, inferred
transaction ledgers.

---

## 11. Materiality boundary and directionality

### Ranking is within a unit, never across units

A currency movement and a percentage-point movement have no common scale.
Ranking them in one sequence made a +0.1% balance drift outrank a +15-point
arrears rise and be labelled the largest observed increase — a plausible-looking
statement that was simply wrong.

Each unit therefore has its **own** rank sequence, its own largest increase and
its own largest decrease. `movement_rank` is a rank *within* `movement_unit`;
`rank_population` states how many metrics share that unit, so "rank 1" is never
read as "the largest movement in the portfolio". The summary reports
`top_movements_by_unit` keyed by unit, and the rendered table carries a
**Ranked within** column. Within one unit, ordering is by |relative change| where
the unit supports one and by |movement| where it does not.

### Materiality

No universal threshold is invented. The workflow computes objective ranking
features only — absolute change, percentage change, basis-point change,
percentage-point change, share shift, contribution to the total balance change,
and movement rank — and describes them in a controlled vocabulary:

`largest_observed_increase` · `largest_observed_decrease` · `notable_by_rank` ·
`main_observed_movement` · `relatively_stable` ·
`insufficient_basis_for_materiality_assessment`

Every result carries:

> No governed materiality threshold is configured for this portfolio, so
> movements are ranked by observed size only. No movement is described as
> material, significant, a breach or high risk.

> Movements are ranked within their unit of measurement. A currency movement and
> a percentage-point movement are not ranked against each other, because they
> have no common scale.

### Directionality

| BSR `directionality` | Rise | Fall |
|---|---|---|
| `higher_is_better` | improvement | deterioration |
| `lower_is_worse` | improvement | deterioration |
| `higher_is_worse` | deterioration | improvement |
| `lower_is_better` | deterioration | improvement |
| `neutral` | not assessed | not assessed |
| `context_dependent` | not assessed | not assessed |

A zero movement is `no_movement`; an uncalculated movement is `not_assessed`.
A rise in an interest rate, a prepayment or a redemption is not called good or
bad where the registry marks it context-dependent. Directionality never overrides
domain logic or a configured risk-monitor ordering — it produces an
interpretation label alongside the number, and nothing else.

---

## 12. Missing data and data quality

Controlled statuses: `available`, `partially_available`, `not_available`,
`not_comparable`, `not_comparable_due_to_availability`, `insufficient_history`,
`invalid_weight_population`, `zero_denominator`, `source_convention_uncertain`.

| Case | Behaviour |
|---|---|
| cumulative value falls | negative movement retained + warning; never floored |
| all weights missing or zero | `invalid_weight_population` / `zero_denominator`; no unweighted fallback |
| metric at only one date | `not_comparable_due_to_availability`; retained, not substituted |
| portfolio population changed | shares use each snapshot's own denominator; the change is noted |
| different schemas across dates | per-field availability; absent-from-both omitted |
| new / disappearing enum values | reported with `presence`; enum labels normalised |
| duplicate loan identifiers | bridge omitted with an explicit limitation |
| missing loan identifiers | bridge omitted with an explicit limitation |
| mixed currencies | **every** monetary output suppressed: metric aggregates (`not_comparable_mixed_currency`), distribution balance shares, and the balance bridge (`unavailable_mixed_currency`). Counts, rates and count shares remain valid and are still answered |

---

## 13. Result contract

```
workflow_id · result_schema_version · request_interpretation · portfolio_scope
period_resolution · dataset_provenance · summary · metric_changes
distribution_changes · balance_bridge · field_selection · warnings
limitations · evidence · audit
```

Each metric change carries: canonical field, display name, analytical concept,
analytical role, temporality, aggregation, weight field / share basis, start
value, end value, movement value, movement unit, movement basis, relative change
where valid, basis-point change where applicable, directionality, controlled
interpretation, status, valid and excluded population at each date, evidence
references, confidence and the registry's rationale.

The `summary` block is generated **from** those tables. It cannot introduce a
fact that is not already in `metric_changes`, `distribution_changes` or
`balance_bridge`. A presenter turns the structure into natural language; the
workflow remains the numerical source of truth.

### Channel integration

The chat route returns the existing envelope shape with the complete governed
result attached additively under `periodChange`. No existing key is renamed or
removed, and no public API contract changed. The route is also directly callable:

```python
from mi_agent_api.period_change_route import analyse_period_change

result = analyse_period_change(
    client_id="client_001", output_root=root, tenant_id=context.tenant_id,
    authorised_portfolio_ids=authorised.portfolio_ids)
payload = result.to_dict()
```

---

## 14. Audit contract

`result.audit` records everything needed to reproduce a result:

* registry version and schema version (and any governed source overrides applied);
* the selection policy name and version, and its caps;
* every selected entry with its role, temporality, aggregation, weight field,
  share basis, movement basis, movement unit, numerator, denominator, valid and
  invalid row counts, status and confidence;
* every excluded candidate with its exclusion reason;
* the resolved snapshots, filters and portfolio scope;
* the period-resolution method;
* the balance-bridge status and rounding tolerance;
* the data-source identifiers;
* the calculation version.

**No loan-level data appears in the audit, the evidence or any log.** Evidence
references a snapshot and a field, never a row.

---

## 15. Worked examples

Two governed monthly runs, 100 → 106 loans, produced by the code in this
repository (`mi_agent_api/tests/test_period_change_route.py` exercises the same
path against on-disk runs).

Resolved period: `latest_available_pair`, 2026-05-31 → 2026-06-30. Ranking below
is within each unit, so the currency and percentage-point movements each have
their own rank-1.

### 1. Current balance — a point-in-time stock

| | |
|---|---|
| temporality | `point_in_time` · basis `point_in_time_difference` |
| aggregation | `sum` |
| movement | £48,200,000 → £50,600,000 = **+£2,400,000** (+4.98%) |
| directionality | `neutral` → interpretation `not_assessed` |

### 2. Recoveries in period — a flow

| | |
|---|---|
| temporality | `period_flow` · basis `flow_level_comparison` |
| movement | £12,000 (May) versus £25,000 (June) |
| note | the two periods' own totals; not an amount arising between the dates |

### 3. Cumulative recoveries — a differenced cumulative

| | |
|---|---|
| temporality | `cumulative` · basis `cumulative_difference` |
| movement | £180,000 → £205,000 = **+£25,000** arising in the interval |

### 4. Original LTV — a static baseline

Tagged `static_baseline`. Excluded from change ranking with reason
`static_baseline_excluded_from_ranking`; no "changed between periods" figure is
produced for it.

### 5. Current LTV — a balance-weighted average

| | |
|---|---|
| aggregation | `weighted_average`, weight `current_outstanding_balance` |
| movement | 40.20% → 41.80% = **+1.60 pp** (+160 bp) |
| relative change | not reported — a point unit |
| directionality | `higher_is_worse` → **deterioration** |

### 6. Interest in arrears — a count share

| | |
|---|---|
| aggregation | `share`, basis `count` |
| opening | 4 of 100 = 4.00% |
| closing | 7 of 106 = 6.60% |
| movement | **+2.60 pp** |
| directionality | `higher_is_worse` → **deterioration** |

### 7. Geography — a distribution shift

| Category | Opening | Closing | Count share | Movement |
|---|---|---|---|---|
| South East | 50 | 58 | 50.00% → 54.72% | **+4.72 pp** |
| London | 30 | 30 | 30.00% → 28.30% | −1.70 pp |
| Wales | 20 | 18 | 20.00% → 16.98% | −3.02 pp |

Balance-share movements are reported alongside, because count share and balance
share can move in opposite directions.

### 8. Balance bridge

Key: `source_portfolio_id + loan_identifier` (provenance present in both
snapshots).

| Component | Amount | Loans |
|---|---|---|
| Opening balance | £48,200,000 | 100 |
| New loans (closing balance) | +£2,864,151 | 6 |
| Exited loans (opening balance) | −£0 | 0 |
| Movement on continuing loans | −£464,151 | 100 |
| **Closing balance** | **£50,600,000** | 106 |

Reconciles: residual £0.00, inside the 0.01 tolerance.

---

## 16. Known limitations

1. **Asset classification.** With no governed asset-class signal at MI query
   time, only `cross_asset` entries are eligible unless a caller states the class
   (§8). 14 of the 106 period-change entries are therefore withheld by default.
2. **Funded book only.** The workflow reads the funded per-period frames.
   Pipeline and forecast views are declined rather than answered from funded data.
3. **Three movement engines still exist.** `period_movement` and
   `temporal_compare` retain the questions they already answer (§3), and they do
   NOT share this workflow's arithmetic — notably they fall back to a simple mean
   on a zero-weight population where this workflow refuses. See
   [`movement_engine_consolidation_plan.md`](movement_engine_consolidation_plan.md)
   for the divergences and the proposed consolidation.
4. **Flow basis is inferred, not declared.** The reporting-period length behind a
   `period_flow` value is derived from the snapshot series, not from registry
   metadata. A field that is genuinely month-to-date while the snapshots are
   quarterly cannot be distinguished from a quarterly flow. A per-field period
   basis in the BSR would close this.
5. **No migration matrix.** `derived_input` fields (previous/current PD, LGD,
   IFRS 9 stage, risk grade) are excluded from overview metrics. No new migration
   formula is invented merely because previous/current inputs exist; the existing
   `mi_agent.risk_monitor.migration` service remains the governed migration path.
6. **Balance bridge is identity-based only.** It distinguishes new, continuing and
   exited loans. It is not a cashflow waterfall and does not attribute causes.
7. **No materiality.** Movements are ranked, never classified. Materiality
   requires a configured governed rule, and none exists.
8. **Currency.** Mixed-currency snapshots produce no monetary totals; there is no
   governed currency-normalisation capability to call.

---

## 17. Extension points

| Extension | Where |
|---|---|
| Source-specific temporality overrides | `config/business_semantics_source_overrides.yaml` → `BusinessSemanticsRegistry.for_source()`. A source that reports `allocated_losses` as a period figure is corrected in **configuration**; no workflow code changes, and the override is recorded in the audit block. Only `temporality`, `default_aggregation`, `weight_field`, `share_basis`, `confidence` and `rationale` may be overridden — re-pointing a field's concept or role is curation and belongs in the registry build. |
| Field-selection policy | `config/period_change_selection.yaml` — concepts, caps, confidence floor, MI-tier preference, non-duplication, per-concept reservation. |
| Snapshot gap ceiling | `period_resolution.max_snapshot_gap_days` plus a per-portfolio override map, for books on a different reporting cadence. |
| Flow-basis tolerance | `period_resolution.flow_basis_tolerance_days`. |
| Asset classification | `period_change_route.resolve_asset_classes()` — supply a governed asset class and the asset-specific registry entries become eligible. |
| Governed semantic hints from the parser | `ParsedQuestion.semantics_context` → `recognition.recognise(semantics_context=…)` accepts `period_change_fields` / `period_change_concepts` with no signature change. |
| New share bases | `calculations.SUPPORTED_SHARE_BASES`. An unsupported basis is refused, never downgraded. |

### Source-override file format

```yaml
# config/business_semantics_source_overrides.yaml  (optional; absent by default)
sources:
  client_001:
    allocated_losses:
      temporality: period_flow
      rationale: >
        This source reports allocated losses for the period, not losses to date.
```

---

## 18. Tests

| File | Covers |
|---|---|
| `mi_agent/tests/test_period_change_registry.py` | the BSR v2 loader, taxonomy, caveats, source overrides |
| `mi_agent/tests/test_period_change_periods.py` | every period-resolution mode and failure |
| `mi_agent/tests/test_period_change_calculations.py` | aggregation, temporality, units, all six directionality values |
| `mi_agent/tests/test_period_change_selection.py` | roles, applicability, caps, exclusion reasons |
| `mi_agent/tests/test_period_change_distribution.py` | shares, churn, unknown values, changing denominators |
| `mi_agent/tests/test_period_change_bridge.py` | reconciliation, tolerance, every unavailable case |
| `mi_agent/tests/test_period_change_recognition.py` | positives, negative controls, incumbent deference |
| `mi_agent/tests/test_period_change_workflow.py` | modes, worked examples, ranking, summary, governance, audit |
| `mi_agent_api/tests/test_period_change_route.py` | snapshot supply, rendering, basis and rank-scope columns, registry precedence, on-disk end-to-end |

Behaviours added after the first review, each with dedicated tests: on-or-before
explicit date resolution, the snapshot-gap ceiling, flow-basis mismatch,
per-unit ranking, composite bridge identity, mixed-currency suppression across
every monetary output, the per-concept reservation, and a guard asserting the
summary never asserts causation.
