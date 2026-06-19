# MIQuerySpec v2 — LLM Interpretation Contract

**Purpose:** define exactly what a (future) natural-language interpreter/LLM is
allowed to produce when turning an MI question into a governed query. The
interpreter's **only** output is a **MIQuerySpec-v2-compatible JSON object**. The
spec is then validated (`mi_agent.mi_spec_validation.validate_query_spec`) and
executed only through the governed runtime (`mi_agent.mi_runtime.run_mi_query`).

**The interpreter MUST NOT produce:** Python, pandas, SQL, arbitrary chart specs,
direct calculations/aggregated numbers, or unvalidated/free-text field names. It
chooses from the controlled vocabularies below and emits field **keys**, never
values it invented.

> This contract is enforced by code: every "valid example" below is in
> `mi_agent/mi_query_spec_v2_examples.py` and is asserted to pass validation;
> every "invalid example" is asserted to fail. See
> `tests/test_phase7_mi_query_spec_v2.py`.

---

## 1. Allowed fields (all optional unless a mode requires them)

**Core:** `query_id`, `route_id` (default `mi`), `execution_mode`, `intent`,
`chart_type`, `metric`, `dimension`, `dimensions`, `hierarchy`, `aggregation`,
`weight_field`, `filters`, `top_n`, `title`, `explanation`, `output_format`.

**State:** `state`, `segment`, `state_filters`.

**Snapshot / segmentation:** `snapshot_client_id`, `snapshot_store_root`,
`as_of_date`, `reporting_date`, `cut_off_date`, `portfolio_id`, `spv_id`,
`acquired_portfolio_id`.

**Temporal:** `temporal_mode`, `baseline_date`, `current_date`, `start_date`,
`end_date`, `comparison_basis`, `trend_grain`.

**Forecast:** `forecast_mode`, `forecast_probability_source`,
`allow_config_probability`.

**Risk:** `risk_monitor`, `risk_monitor_mode`, `migration_dimension`,
`concentration_dimension`, `risk_dimension`, `baseline_risk_field`,
`current_risk_field`.

**Buckets:** `bucket_strategy`, `bucket_count` (default 4), `bucket_field`,
`bucket_config_key`.

**Chart / output:** `output_type`, `chart_preference`, `allow_chart_fallback`.

**Governance:** `require_structured_issues` (default true), `allow_partial_result`,
`strict_mode`.

Any key not on this list is dropped by `MIQuerySpec.from_dict` (the interpreter
must not rely on smuggling extra keys).

## 2. Allowed enum values

| Field | Allowed values |
|---|---|
| `route_id` | `mi`, `mna`, `regulatory_annex2`, `regulatory_and_mi` |
| `execution_mode` | `flat`, `snapshot`, `state`, `temporal`, `risk` (usually omit — it is derived) |
| `state` | `total_funded`, `total_pipeline`, `total_forecast_funded`, `cohort_by_date`, `cohort_by_portfolio`, `cohort_by_spv`, `cohort_by_acquired_portfolio` (+ aliases `cohort_by_origination_date`/`_funding_date`/`_acquisition_date`) |
| `temporal_mode` | `latest`, `as_of`, `compare`, `trend` |
| `risk_monitor_mode` | `migration`, `concentration`, `trajectory`, `flags` |
| `bucket_strategy` | `configured`, `quantile`, `none` |
| `trend_grain` | `daily`, `weekly`, `monthly`, `quarterly` |
| `forecast_probability_source` | `row`, `config`, `explicit_balance` |
| `output_type` | `table`, `chart`, `both` |
| `chart_type` / `chart_preference` | `bar`, `line`, `scatter`, `bubble`, `heatmap`, `treemap`, `none` (no new chart types) |
| `segment` | `portfolio`, `spv`, `acquired_portfolio` |

## 3. Natural-language term mapping (mandatory)

The interpreter resolves bare concepts to **concrete field keys** — it never
emits a bare ambiguous term as a `dimension`:

| User says | Emit | Condition |
|---|---|---|
| "portfolio" | `portfolio_id` (the **Trakt portfolio reference**) | requires a client portfolio reference config; if none, raise a clarification — do **not** fall back to `acquired_portfolio_id` |
| "acquired portfolio" | `acquired_portfolio_id` | — |
| "SPV" | `spv_id` | — |
| "stage" / "pipeline stage" | `pipeline_stage` | **only** in a pipeline context (`state = total_pipeline`/`total_forecast_funded`); otherwise raise `invalid_stage_context` |
| "IFRS stage" / "IFRS 9 stage" | `ifrs9_stage` | — |
| "risk stage" / "internal risk stage" | `internal_risk_stage` | — |
| "balance band" / "by balance" | `balance_band` + `bucket_strategy: quantile` | unless a client/asset config defines fixed bands (`bucket_strategy: configured`) |
| "interest rate band" / "by rate" | `interest_rate_bucket` + `bucket_strategy: quantile` | as above |
| "time on book" | `time_on_book_bucket` + `bucket_strategy: quantile` | as above |
| "region" | `geographic_region_obligor` (or `collateral_geography` for property) | pick the obligor region by default |

Bare `stage`, `portfolio`, `region`, `rate`, `balance` as a `dimension` are
**rejected** (`ambiguous_dimension`) — they must be resolved first.

## 4. Valid spec examples

(Names map to `mi_agent/mi_query_spec_v2_examples.py::EXAMPLES`.)

- **Flat — LTV by region** (`flat_ltv_by_region`):
  ```json
  {"intent":"chart","chart_type":"bar","metric":"current_loan_to_value",
   "dimension":"collateral_geography","aggregation":"weighted_avg"}
  ```
- **Total funded latest** (`total_funded_latest`):
  ```json
  {"route_id":"mi","execution_mode":"state","state":"total_funded",
   "temporal_mode":"latest","snapshot_client_id":"clientA"}
  ```
- **Funded trend** (`funded_trend`):
  ```json
  {"route_id":"mi","state":"total_funded","temporal_mode":"trend",
   "start_date":"2024-01-01","end_date":"2024-12-31","trend_grain":"monthly",
   "snapshot_client_id":"clientA","output_type":"chart","chart_preference":"line"}
  ```
- **Funded compare** (`funded_compare`):
  ```json
  {"route_id":"mi","state":"total_funded","temporal_mode":"compare",
   "baseline_date":"2024-01-31","current_date":"2024-03-31",
   "comparison_basis":"balance","snapshot_client_id":"clientA"}
  ```
- **Risk grade migration** (`risk_grade_migration`):
  ```json
  {"route_id":"mi","risk_monitor_mode":"migration",
   "migration_dimension":"internal_risk_grade","baseline_date":"2024-01-31",
   "current_date":"2024-03-31","snapshot_client_id":"clientA"}
  ```
- **Funded by portfolio** (`funded_by_portfolio`) and **balance quantile**
  (`funded_by_balance_quantile`) — see the examples module for the full set
  (also: total pipeline/forecast latest, pipeline by stage, forecast-funded by
  region, concentration warning, IFRS9 & PD migration).

## 5. Invalid / ambiguous spec examples (must be rejected or clarified)

(Names map to `INVALID_EXAMPLES`.)

- **`ambiguous_stage_dimension`** — `concentration_dimension: "stage"` →
  `ambiguous_dimension`.
- **`compare_missing_dates`** — `temporal_mode: compare` without both dates →
  `temporal_selector_incomplete`.
- **`state_missing_client`** — state mode without `snapshot_client_id` →
  `missing_snapshot_client_id`.
- **`mna_pipeline_state`** — `route_id: mna` + `state: total_pipeline` →
  `invalid_route_for_state`.
- **`regulatory_state`** — `route_id: regulatory_annex2` + any MI state →
  `invalid_route_for_state`.
- **`risk_without_mode`** — `risk_monitor: true` without a `risk_monitor_mode` →
  `invalid_risk_monitor_spec`.
- **`invalid_state_enum`** — `state: total_unknown` → `invalid_enum_value`.

## 6. Clarification behaviour

When the request is ambiguous or under-specified, the interpreter must **ask a
clarifying question rather than guess**, in particular when:

- "portfolio" is used but no client portfolio reference config is available
  (`missing_portfolio_reference_config`);
- "stage" is used outside a pipeline context (`invalid_stage_context`);
- a temporal comparison/trend is requested without enough dates;
- a risk question does not say *which* risk dimension (grade / IFRS 9 / PD);
- a bare ambiguous term (`stage`, `portfolio`, `region`, `rate`, `balance`) would
  otherwise be emitted as a dimension.

The interpreter never invents probabilities, balances, dates, field names, or
client identifiers. It emits only keys/enums from this contract.

## 7. Hard boundaries (recap)

- Output is **MIQuerySpec-v2 JSON only**.
- No code (Python/pandas/SQL), no arbitrary chart definitions, no computed
  numbers, no unlisted fields, no chart types outside the governed library.
- Analytics are computed **only** by the governed runtime/engines, never by the
  interpreter.
