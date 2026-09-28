# Slice 2 — Phase 1 architecture trace

Written at `53a7b8f6` (the accepted Slice 1 production build) **before any code
was changed**. Every claim below is a file/function reference, not a
recollection.

---

## 1. SnapshotStore

`snapshot/store.py:50` — `class SnapshotStore(abc.ABC)`.

Four **storage primitives** an adapter implements:

| method | line |
|---|---|
| `register_snapshot(header, frame)` | `snapshot/store.py:55` |
| `list_snapshots(client_id, route, cadence, since, until)` | `snapshot/store.py:60` |
| `get_snapshot(snapshot_id)` | `snapshot/store.py:68` |
| `load_loans(snapshot_id)` | `snapshot/store.py:72` |

Four **temporal resolvers**, implemented ONCE on the base class in terms of
`list_snapshots`, so every adapter resolves identically
(`snapshot/store.py:76-118`):

* `resolve_latest(client_id, route)` → the newest header by reporting date
* `resolve_as_of(client_id, reporting_date, route)` → newest header **on or
  before** a date; raises `SnapshotNotFoundError` when none exists
* `resolve_range(client_id, start_date, end_date, route)` → every header inside
  the window, ascending; `None` bounds mean unbounded
* `resolve_compare(client_id, baseline_date, current_date, route)` →
  `(baseline, current)`, each via `resolve_as_of`

Ordering is `_sorted_by_reporting_date` (`snapshot/store.py:41`), keyed on
`parse_date(reporting_date)` with `upload_timestamp` as tie-break.

The only adapter at HEAD is `snapshot/adapters/local_fs.py:50`
`LocalFsSnapshotStore` (manifest.json index + per-snapshot `loans.csv`).

## 2. SnapshotSelector

`mi_agent/states/selectors.py:27` — a frozen declarative selection carrying
`client_id`, `mode`, `route` and the per-mode date fields. Modes today:
`latest`, `as_of`, `range`, `compare` (`selectors.py:18-21`). Factories
`latest/as_of/range/compare` at `selectors.py:45-65`; `resolve(store)` at
`selectors.py:83` dispatches straight into the four store resolvers and returns
`header | list[header] | (baseline, current)`.

Its own docstring already records the intent: *"`range` / `compare` are provided
to prepare for Phase 4 temporal trends"*. Slice 2 is that consumer.

## 3. How available snapshots are discovered

Two catalogues exist in the estate, and they are **not** the same thing:

* **Governed snapshot catalogue** — `SnapshotStore.list_snapshots`, backed by
  `LocalFsSnapshotStore`'s `manifest.json`. Carries `SnapshotHeader`
  (`snapshot/model.py:176`) with `client_id`, `route`, `reporting_date`,
  `cadence`, `cut_off_date`, `snapshot_id`, `row_count`, `content_hash`.
  This is the one `mi_agent.states`, `mi_agent.mi_runtime` and
  `mi_agent.risk_monitor` consume.
* **Onboarding run catalogue** — `mi_agent_api/snapshots.py:172`
  `discover_snapshots(output_root)`, a filesystem walk over central tapes
  returning `{portfolios: [{client_id, runs: [{run_id, reporting_date, …}]}]}`.
  This is what the production API and the dashboard consume
  (`mi_agent_api/datasets.py:743` `_resolve_query_frame`,
  `mi_agent_api/evolution.py:136` `_runs_up_to`).

Slice 2 binds to the **governed** catalogue, because the architectural rule
names `SnapshotStore`/`SnapshotSelector` as the sole temporal authority. Wiring
the production run catalogue behind a `SnapshotStore` adapter is a deployment
concern and is deliberately **not** done in this offline slice.

## 4. How "current", explicit dates and ranges are resolved today

* **current** — `SnapshotStore.resolve_latest`; on the production API path the
  equivalent is "the active dataset" (`datasets.get_dataframe()`), and
  `mi_service` resolves **exactly one** frame per request
  (`mi_agent_api/mi_service.py:1801` `_resolve_frame` →
  `datasets.py:743 _resolve_query_frame`).
* **an explicit run** — a `portfolio_id` of the form `client/run` loads that
  run's tape (`datasets.py:760-772`). That is a *caller-supplied* selection, not
  a question-derived one.
* **a range** — only `SnapshotStore.resolve_range`, consumed by
  `mi_agent/states/temporal.py:310` and `mi_agent/risk_monitor/monitor.py`.
* **raw-text spans** — `mi_agent/period_request.py:138 requested_span(question)`
  reads the QUESTION with regexes. It is a legacy-path owner and must not appear
  on the governed path; its governed twin `span_from_claim(time_claim)`
  (`period_request.py:114`) reads a contract object instead, and its
  `clarification(span, available_periods)` (`period_request.py:199`) is the
  estate's existing wording for "the book does not reach that far".

## 5. Existing temporal / evolution capability owners

| owner | file | measure vocabulary | verdict for Slice 2 |
|---|---|---|---|
| state trend/compare | `mi_agent/states/temporal.py:291` / `:165` | **fixed**: `loan_count` + `balance_sum` (via `analytics_lib.stratify`), or `forecast_contribution` | cannot express average / weighted-average LTV; its `_apply_filters` (`temporal.py:66`) silently skips a filter whose column is absent. Not reusable as the Slice 2 measure owner. |
| period change | `mi_agent/period_change/*` | movement, bridge, distribution shift | **movement attribution** — explicitly out of Slice 2 scope |
| dashboard evolution | `mi_agent_api/evolution.py:148` | a fixed KPI list and a fixed `_FUNDED_BREAKDOWN_DIMS` map (`evolution.py:40`) | dashboard-only surface; the prompt forbids importing dashboard geography semantics into MI querying |
| generic executor | `mi_agent/mi_query_executor.execute_mi_query` | `{sum, avg, weighted_avg, count, min, max, median}` over any governed field, N grouping axes, governed predicates with a receipt | **this is the Slice 1 accepted owner and the Slice 2 per-snapshot owner** |

`states/temporal.trend` is nonetheless the proven *shape*: resolve headers →
`load_loans` per header → run one measure owner per frame → assemble ascending.
Slice 2 reuses that shape with the Slice 1 execution owner in place of the state
assembler, so no second snapshot-resolution owner appears.

## 6. Existing time × dimension execution path

`mi_agent/states/temporal.py:332-348` — `trend(..., stratify_by=<dim>)` emits one
row per `(reporting_date, dimension value)` with `count` and `balance`. **One**
non-time dimension only, and only those two measures.

`mi_agent_api/evolution.py:120 _breakdown` does the same for the dashboard over
`broker`/`region`/`ltv_bucket`.

Neither carries a governed predicate receipt, so neither can satisfy the Slice 2
acceptance assertion that filters survive on every snapshot. The Slice 1 owner
does: `execute_mi_query` records `applied_predicates` and `group_field_keys`, and
`plan_serving_canary.reconcile` (`plan_serving_canary.py:156`) already checks
both against the bound spec.

## 7. Existing deterministic period-comparison capability

`mi_agent/states/temporal.py:231 _compare_total` computes
`count_change`, `count_pct_change`, `balance_change`, `balance_pct_change` with
`_pct_change` (`temporal.py:126`) guarding a zero baseline. Same two-measure
limitation. `mi_agent/period_change/calculations.py` computes movement per field
but is the attribution owner Slice 2 excludes.

## 8. Current query-plan temporal fields

`mi_agent/interpretation_v2/plan.py:136` — `PeriodBinding(form, labels, grain,
periods_back, contract, resolved, owned_by_capability)`. Its docstring is
explicit: *"Deliberately NOT a snapshot id … pinning a date here would be this
package deciding something it has no data to decide."*

The governed temporal vocabulary already exists and needs **no extension**:

* `vocabulary.TIME_FORMS` = `current, previous_reporting_period, relative_pair,
  explicit_period, range, series, forward_looking` (`vocabulary.py:120`)
* `vocabulary.TIME_GRAINS` = `daily, weekly, monthly, quarterly, annual`
* `CAPABILITY_OPERATIONS["generic_analysis"]` already includes `series` and
  `compare` (`vocabulary.py:75`)
* `compiler._PERIOD_CONTRACT` (`compiler.py:126`) already names the governed
  contract per form, and `_bind_period` (`compiler.py:513`) already raises
  `AMBIGUOUS_PERIOD` for a span with no grain, label or count, and
  `UNSUPPORTED_COMPOSITION` for a forward-looking period on a capability that
  owns no forward methodology.
* `intent._DATE_LIKE` (`intent.py:105`) already **refuses** a model payload
  containing an ISO date, a `dd/mm/yyyy`, a compact date or a `snapshot_*`
  identifier in any binding slot, including `time.labels`
  (`intent.py:586 _parse_time` → `_guard_string`). So
  `MODEL_AUTHORED_PHYSICAL_SNAPSHOT_BINDINGS = 0` is a structural property of
  the accepted intent parser, not a new Slice 2 check.

Measured at HEAD (`/tmp` probe, no model calls): every Slice 2 example shape —
`series`+`periods_back`, `series` with no count, `range`+"since March",
`explicit_period`, `relative_pair`+`compare`, series×1 dim, series×2 dims,
filtered series, weighted-average series — **already compiles to a
GovernedQueryPlan today**, with the temporal intent intact. A bare `series` with
no span already CLARIFIES (`AMBIGUOUS_PERIOD`); a forward-looking generic series
already REFUSES.

**The compiler therefore needs no Slice 2 change at all.**

## 9. Where the Slice 1 eligibility gate rejects temporal plans

`mi_agent/plan_runtime_adapter.py:170 check_eligibility`:

* `ELIGIBLE_OPERATIONS = {point_in_time, breakdown}` (`:73`) →
  `OPERATION_NOT_GENERIC` for `series` and `compare`
* `ELIGIBLE_PERIOD_FORMS = {current}` (`:79`) → `PERIOD_NOT_CURRENT` for every
  other form, with the comment *"a stated historical period belongs to the
  temporal owner, which this slice does not touch"*

Both were measured, not chosen. `plan_serving_canary`'s docstring depends on
them: it justifies not re-running four legacy guards precisely because *"a
stated historical period is PERIOD_NOT_CURRENT"*.

Downstream of the gate, `plan_serving_canary.serve()` (`:287`) is handed a
**single** `frame` by the caller and never selects one.

## 10. The smallest seam into Slice 2 serving

Four edits, no new engine:

1. `SnapshotStore.resolve_last_n(client_id, periods, route)` — the one resolver
   the four existing ones cannot express ("the latest N available"). Lives on
   the store beside its siblings, so snapshot selection stays in one place.
2. `SnapshotSelector` gains the matching `last_n` mode and factory.
3. `plan_runtime_adapter.check_eligibility` is split into the shared structural
   core it already is and the Slice-1 temporal perimeter, so Slice 2 can reuse
   the core **byte-for-byte** with a different perimeter. Slice 1's public
   behaviour is unchanged and the corpus replay proves it.
4. A new `mi_agent/plan_temporal_runtime.py`: `PeriodBinding` → `SnapshotSelector`
   (the one new semantic owner), then per resolved header
   `store.load_loans` → the **existing** `adapter.spec_for_plan` +
   `execute_mi_query` + `plan_serving_canary.reconcile`, then deterministic
   series/comparison assembly.

## Stop conditions — none triggered

| condition | verdict |
|---|---|
| needs a new analytics engine | **No** — `execute_mi_query` runs once per snapshot |
| `SnapshotSelector` unusable as temporal authority | **No** — it needs one new mode |
| rebuilds the deterministic temporal runtime | **No** — `states/temporal.py` untouched |
| compiler needs a broad rewrite | **No** — compiler unchanged |
| more than one substantial new semantic owner | **No** — one: the plan→selector binder |
| raw text re-read after the plan | **No** — the resolver's only inputs are `PeriodBinding` and the catalogue |

## Budget declared before editing

```
EXPECTED_PRODUCT_FILES_CHANGED = 4
EXPECTED_NET_EXECUTABLE_LOC    = ~350
```
