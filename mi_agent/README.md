# MI Agent — v1 Foundation

An **isolated, additive** foundation for a Management Information (MI) querying
agent. It turns natural-language MI questions into a small, validated
`MIQuerySpec`, checked against a curated *semantic* view of the canonical field
registry.

> This package does **not** touch the existing ESMA / onboarding / validation /
> reporting pipeline. It does not modify the canonical field registry, is not
> wired into `trakt_run.py`, and does not change any Streamlit dashboard.

```
mi_agent/
  __init__.py
  build_mi_semantics_registry.py     # generates the semantic registry
  mi_semantics_field_registry.yaml   # GENERATED curated semantic layer
  mi_query_spec.py                   # MIQuerySpec dataclass (v1)
  mi_query_validator.py              # validates a spec + CLI
  llm_query_parser.py                # NL -> MIQuerySpec (deterministic + optional LLM)
  README.md
  tests/
    test_mi_query_validator.py
```

## What it does

1. **Curated MI semantic registry** — a thin, analytics-oriented layer
   *generated from* `config/system/fields_registry.yaml`. It does not duplicate
   the canonical registry; each entry references a canonical field by name and
   adds MI metadata (role, format, allowed aggregations, chart roles, weighting
   and bucketing hints).
2. **`MIQuerySpec` (v1)** — a serialisable description of a chart/table/summary
   request that names semantic field keys only (never data).
3. **Validator** — checks a spec against the semantic registry (and optionally
   against a dataset's columns).
4. **LLM query parser skeleton** — translates a natural-language question into
   an `MIQuerySpec`. Works fully offline (deterministic) for tests; an optional,
   mockable Claude path is provided.

## How `mi_semantics_field_registry.yaml` is generated

`build_mi_semantics_registry.py` reads the canonical registry and projects a
**curated allowlist** (the `CURATION` dict in that script) of MI-relevant
canonical fields — currently **~61 fields** (≈37 `core`, ≈24 `extended`). This
deliberately replaces the v0.1 broad rule
(`core_canonical OR layer in {performance,product} OR category == analytics`),
which produced ~235 fields, 24 unclassifiable, and many duplicate concepts
(see `reports/mi_semantics_review.md`).

Excluded by design: identifiers, LEIs, industry/tax codes, rating-agency
equivalents, waterfall/swap fields, balance-period buckets, and duplicate
borrower-2/guarantor fields.

Each curated field carries **MI business metadata**:

- `mi_tier` — `core` (standard portfolio MI) or `extended` (less frequent)
- `business_name` — short analyst-facing label (e.g. `Balance`, `Current LTV`)
- `business_description` — one-line plain-English meaning
- `synonyms` — NL phrases for future natural-language resolution

…plus heuristically inferred (and, where needed, curation-`overrides`-pinned)
analytics metadata:

- **role**: `metric | dimension | date | identifier | flag | unknown`
- **format**: `currency | percent | integer | decimal | date | string | boolean`
- **chartable**, **allowed_aggregations** / **default_aggregation**
- **allowed_chart_roles** / **default_chart_role**
- **weight_field** (balance field used for weighted averages of rates/LTVs)
- **bucket_field** (e.g. `age_bucket`, `ltv_bucket`, `ticket_bucket`,
  `vintage_year`, `arrears_bucket`)
- **source_criteria** — canonical attributes (`core_canonical`,
  `layer:performance`, …) that justify relevance, or `curated`.

A curated field missing from the canonical registry is skipped with a warning
(and recorded under `metadata.missing_curated_fields`), so generation is robust
across registry versions. Top-level `metadata` also carries the generation
timestamp, source path, tier counts, field count, version and default weight
field.

> The metadata is heuristic + hand-curated and **requires human review before
> production use**.

## How `MIQuerySpec` works

`MIQuerySpec` (in `mi_query_spec.py`) is a stdlib `dataclass` (no pydantic
dependency). Key fields: `intent`, `chart_type`, `metric`, `dimension`, `x`,
`y`, `size`, `color`, `aggregation`, `weight_field`, `filters`, `top_n`,
`dimensions`, `hierarchy`, `title`, `explanation`, `output_format`.

Helpers: `to_dict()`, `to_json()`, `from_dict()`, `from_json()`, and
`referenced_fields()` (all semantic field keys the spec uses).

## How the validator works

`validate_mi_query(spec, semantics, available_columns=None) -> ValidationResult`
returns `ok`, `errors`, `warnings`, and `resolved_fields`. It checks:

1. referenced semantic fields exist in the registry;
2. their canonical columns exist in `available_columns` (when provided);
3. fields used in chart roles are `chartable`;
4. chart-role compatibility (x/y/size/color/dimension vs `allowed_chart_roles`);
5. the chosen aggregation is allowed for the metric;
6. chart-type structural requirements (bar/line/scatter/bubble/heatmap/treemap);
7. `weighted_avg` has a weight field (on the spec or the metric);
8. `top_n` is only used with grouped outputs (bar/table/treemap);
9. unknown `intent` / `chart_type` fail;
10. `chart_type: none` is allowed for `summary`/`table`.

## How the query executor works

`mi_query_executor.execute_mi_query(spec, data, semantics, ...)` takes a
*validated* `MIQuerySpec`, runs it against canonical portfolio data (a pandas
`DataFrame` **or** a local CSV path), and returns an `MIQueryResult` ready for
later chart rendering / Streamlit / HTML / PPTX export. It is fully
deterministic: **no LLM, no chart rendering, no Streamlit, no Azure, no
mutation of the input dataframe, never executes arbitrary code.**

```python
from mi_agent import execute_mi_query, MIQuerySpec
spec = MIQuerySpec(intent="chart", chart_type="bar",
                   metric="current_outstanding_balance",
                   dimension="geographic_region_obligor",
                   aggregation="sum", top_n=5)
result = execute_mi_query(spec, "canonical_typed.csv",
                          "mi_agent/mi_semantics_field_registry.yaml")
print(result.result_type, result.row_count)
print(result.preview())
```

`MIQueryResult` fields: `spec`, `result_type` (`table` | `summary` |
`loan_level`), `data` (DataFrame), `resolved_fields`, `row_count`, `warnings`,
`metadata`; methods `to_dict()`, `to_json()`, `to_csv(path)`, `preview(n)`
(the DataFrame serialises as records for `to_dict()`/`to_json()`).

### Confirmed canonical-output assumptions (from repo inspection)

- **Format / location:** Gate 2 (`engine/gate_2_transform/canonical_transform.py`)
  writes the active, dashboard-ready `<stem>_canonical_typed.csv` locally via
  `df.to_csv(..., index=False)`; the pipeline (`function_app.py`) also uploads
  it to Azure Blob. **This executor reads the local CSV (or a DataFrame) only —
  no Azure integration in v1.**
- **Columns** are canonical field-registry names
  (`current_outstanding_balance`, `current_loan_to_value`, `origination_date`…).
  ESMA "no-data" markers (e.g. `ND5`) coerce safely to `NaN`.
- **Bucket columns** (`age_bucket`, `ltv_bucket`, `ticket_bucket`,
  `vintage_year`, `arrears_bucket`, `term_bucket`) are **not** canonical truth —
  they are derived at the analytics layer (`analytics/mi_prep.py::add_buckets`).
  The executor **reuses a bucket column if it is already present** in the
  dataframe (heatmap/treemap grouping) and otherwise falls back to the raw
  field with a warning. It does **not** build a bucketing engine and does not
  import `analytics/` code.

### Percentage scale

The repo is **inconsistent**: `canonical_transform.py` computes LTV as
`(balance/valuation)*100` (whole-number percent) and business rules validate
LTV in 0–500, yet some sample CSVs store LTV as fractions (`0.36`). The
executor therefore **does not rescale percentages**. It heuristically detects
the apparent scale (`fraction` vs `whole_number_percent`) and records it — with
a warning — in `result.metadata` (`percent_scale_detected`,
`percent_scale_median`) so downstream renderers decide formatting.

### Supported query types

| intent / chart_type | result_type | behaviour |
|---|---|---|
| `summary` (or chart `none`) | `summary` | one-row aggregate; `loan_count` (+ `total_balance`) when no metric |
| `table` | `table` | group by dimension + aggregate, or counts by dimension |
| `bar` | `table` | group by dimension, aggregate, sort desc (asc for dates), `top_n`, concentration |
| `line` | `table` | derive monthly `YYYY-MM` period (or reuse a `vintage_year` column), group, sort ascending |
| `scatter` | `loan_level` | loan-level `x, y` (+ optional `color`); capped/sampled; **no identifiers** |
| `bubble` | `loan_level` | loan-level `x, y, size` (+ optional `color`); capped/sampled; **no identifiers** |
| `heatmap` | `table` | group by two dimensions, aggregate metric (long form) |
| `treemap` | `table` | group by hierarchy/dimensions, aggregate, `top_n`, concentration |

### Behaviour notes

- **Balance/exposure hierarchy:** `current_outstanding_balance` →
  `current_principal_balance`. Used for `balance_sum`, default `weighted_avg`
  weight, top-N ranking and concentration share.
- **Missing values:** rows with missing/blank **grouping** values are
  **excluded** from grouped results by default (a warning reports how many).
  Missing values are *not* grouped as `"Unknown"`.
- **Top-N** (bar/table/treemap only) ranks by **balance, then count, then
  concentration**. `concentration_pct = group_share / total * 100` (additive
  aggregations use the metric itself; non-additive ones like `weighted_avg` use
  the exposure share when a balance field is available).
- **Loan-level privacy:** scatter/bubble return only the requested analytical
  columns (never identifiers); output is capped at `max_loan_level_rows`
  (default **5,000**) with deterministic sampling (`sample_seed`, default 42)
  and a warning; original/returned counts and the seed are recorded in metadata.

### CLI

```bash
python -m mi_agent.mi_query_executor \
  --semantics mi_agent/mi_semantics_field_registry.yaml \
  --spec path/to/spec.json \
  --data path/to/canonical.csv \
  --out path/to/result.csv          # omit --out to print a preview
```

### Example specs

```json
{"intent":"chart","chart_type":"bar","metric":"current_outstanding_balance",
 "dimension":"geographic_region_obligor","aggregation":"sum","top_n":5}
```
```json
{"intent":"chart","chart_type":"bubble","x":"youngest_borrower_age",
 "y":"current_loan_to_value","size":"current_outstanding_balance",
 "aggregation":"loan_level"}
```

## How the chart factory works

`mi_chart_factory.create_mi_chart(result, semantics, ...)` turns an
`MIQueryResult` into an enterprise-ready Plotly figure. It is deterministic:
**no LLM, no Streamlit/Azure, no arbitrary Plotly, no re-execution, no mutation
of `result.data`.**

```python
from mi_agent import execute_mi_query, create_mi_chart, MIQuerySpec
spec = MIQuerySpec(intent="chart", chart_type="bar",
                   metric="current_outstanding_balance",
                   dimension="geographic_region_obligor",
                   aggregation="sum", top_n=5)
result = execute_mi_query(spec, "canonical_typed.csv", SEMANTICS)
chart = create_mi_chart(result, SEMANTICS)
chart.write_html("balance_by_region.html")
```

`MIChartResult` fields: `fig` (a `plotly.graph_objects.Figure`), `chart_type`,
`title`, `subtitle`, `warnings`, `metadata`; methods `to_html(path=None,
include_plotlyjs="cdn")`, `write_html(path)`, `to_json()`, and `write_image(path)`
(raises a clear error if the optional `kaleido` package is absent).

### Visual defaults (from repo inspection)

The dashboard styling in `analytics/charts_plotly.py` / `streamlit_app_erm.py`
(`apply_chart_theme`) defines: `PRIMARY_COLOR #232D55` (navy), `SECONDARY_COLOR
#919DD1` (muted blue), `ACCENT_COLOR #BFBFBF` (grey), `TEXT_DARK #2D2D2D`, font
**Calibri**, white plot/paper background, `#F0F0F0` gridlines, left-aligned
title (size 18, weight 600), horizontal legend, `hovermode="closest"`, and
`mi_prep.format_currency` (£1.2M / £25K).

The chart factory keeps an **isolated copy** of this look-and-feel (it does not
import `analytics/`, which would couple the MI agent to the pipeline via
`mi_prep`). It adds Financial-Services-grade extras: a restrained categorical
palette derived from the brand navy/blue/slate, a muted light→navy sequential
scale for heatmaps, positive/negative/neutral accents, explicit margins, subtle
gridlines, and consistent hover-label styling. A `theme` dict can override any
default; `template="none"` ensures no raw Plotly default styling leaks in.

### Supported chart types

`bar` (horizontal automatically when > 6 categories), `line` (monthly period or
a reused `vintage_year`/`maturity_year` column), `scatter` and `bubble`
(loan-level, opacity < 1, capped bubble sizing, **no identifiers in hover**),
`heatmap` (long-form pivoted internally, numeric bucket ordering), and `treemap`
(hierarchy sized by metric). `chart_type: "none"` (table-only) and
`intent: "summary"` raise a clear error — summary KPI cards are out of scope for
v1.

### Formatting rules

- **Currency:** `£1.2m` / `£450k` / `£25,000` (k from 100k, m from 1m); the
  numeric axis uses a `£` prefix with SI tick formatting.
- **Percent:** respects the executor's `percent_scale_detected` — `fraction`
  shows `0.36 → 36.0%`, `whole_number_percent` shows `37.9 → 37.9%`; ambiguous
  scale is surfaced in the subtitle. Data is never rescaled in place.
- **Count:** thousands separators in hover, compact (`1.2m`/`450k`) on axes.
- **Ratios** (DSCR/DTI): two decimals with an `x` suffix.
- **Dates:** months as `Jan-26`, years as `2026`.

### Export options

- **HTML** — `to_html()` / `write_html(path)` (Plotly.js via CDN by default).
- **JSON** — `to_json()` (chart metadata + the full Plotly figure spec).
- **Image** (PNG, etc.) — `write_image(path)` *if the optional `kaleido`
  package is installed*, otherwise a clear optional-dependency error.

## How to run

Build the semantic registry:

```bash
python -m mi_agent.build_mi_semantics_registry \
  --source config/system/fields_registry.yaml \
  --output mi_agent/mi_semantics_field_registry.yaml
```

Validate a spec:

```bash
python -m mi_agent.mi_query_validator \
  --semantics mi_agent/mi_semantics_field_registry.yaml \
  --spec path/to/spec.json \
  --data optional/path/to/canonical.csv
```

Parse a question (offline, deterministic):

```python
from mi_agent.llm_query_parser import parse_user_question
spec = parse_user_question("balance by region",
                           "mi_agent/mi_semantics_field_registry.yaml",
                           llm_enabled=False)
print(spec.to_json())
```

Run the tests:

```bash
pytest mi_agent/tests -q
```

## Known limitations / next steps

- **No Streamlit MI chat UI yet.**
- **No direct PPTX export yet** — charts export to HTML / JSON / image (image via
  optional `kaleido`); PPTX embedding is a later task.
- **No chart-level user customisation yet** beyond the `theme` dict override —
  no per-series styling, annotations, or interactivity configuration.
- **No arbitrary Plotly** — only the chart types `MIQuerySpec` supports.
- **No summary KPI cards** — `intent="summary"` raises a clear error in chart v1.
- **No full design-system / theme YAML yet** — the theme lives as a dict in
  `mi_chart_factory.py`.
- **No full bucketing engine yet** — bucket columns are reused if already present,
  otherwise the raw field is used (with a warning).
- **No complex filter expressions yet** — `filters` supports equality (scalar) and
  `isin` (list) only.
- **No scenario execution yet.**
- **The executor assumes canonical data has already been transformed and
  validated**; it performs only minimal safe coercion (numeric/date), never
  business transformations, and never rescales percentages.
- **Local CSV / DataFrame only** — no direct Azure Blob integration in v1.
- **No MI pipeline integration** — not wired into `trakt_run.py` or any gate.
- **LLM parser is a skeleton** — the live Claude path is optional and mockable;
  the deterministic parser handles only a handful of example phrasings and does
  not yet consult `synonyms` for field resolution (a v2 item).
- **Semantics are heuristic + curated** and must be reviewed by a human before
  production use.
- The LLM is only ever shown the data-free semantic catalogue; it never sees raw
  dataset values, and generated content is parsed as data only — never executed.

### MI Agent v2 recommendations

- **Synonym-driven resolution.** Use each field's `business_name` + `synonyms`
  to resolve NL terms deterministically (and to ground the LLM), instead of the
  current first-keyword-match in `find_field`. This fixes cases like "balance"
  resolving to `arrears_balance` rather than `current_outstanding_balance`.
- **Derived buckets.** Materialise the `bucket_field` hints (`age_bucket`,
  `ltv_bucket`, `ticket_bucket`, `vintage_year`, `arrears_bucket`) as real
  groupable dimensions so heatmaps/treemaps can group by banded measures.
- **Curation governance.** Track `CURATION` in review with sign-off; add a unit
  test asserting zero `role: unknown` and that every entry has a `business_name`
  and ≥1 synonym.
- **Concept de-duplication map.** Keep an explicit "preferred field per concept"
  table (one balance, one current LTV, one valuation) for NL disambiguation.
- **Tier-aware prompting.** Default the LLM/NL surface to `core` fields and only
  expose `extended` on request, to keep prompts small and answers focused.
