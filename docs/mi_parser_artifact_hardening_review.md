# MI Parser & Artifact Hardening — Architecture Review (Part 0)

Reviewed before patching, to extend (not duplicate) the existing
parse → spec → validate → execute → chart → adapter → React pipeline.

## 1. What chart types already exist?
`bar`, `line`, `scatter`, `bubble`, `heatmap`, `treemap`, `none`
(`mi_query_spec.CHART_TYPES`). Frontend renders `bar/line/area/scatter/bubble/
waterfall` via Recharts, `heatmap` via a **native CSS-grid** `HeatmapArtifactView`,
`treemap` via Recharts, with a themed Plotly fallback (`ArtifactRenderer.tsx`).
**A heatmap renderer already exists** and renders rows=`yKey`, cols=`xKey`,
cells=`valueKey`, formatted via `valueFormat`/`formatValue`.

## 2. What parser routes already exist?
`llm_query_parser._deterministic_parse` (the active zero-cost parser used by
`parse_with_repair`) routes, in order: filtered count/balance → literal `heatmap`
→ literal `treemap` → bubble (**triggered by `len(by_parts) >= 3`** — the bug) →
scatter (`vs`) → line (trend) → bar (one dim). `interpreter/deterministic.py` is a
**separate** risk/state interpreter, not this path.

## 3. How does it decide bar vs bubble vs table/KPI?
- summary/KPI: `intent=summary` or `chart_type=none` (e.g. filtered counts).
- bar: one resolved dimension + metric.
- bubble: literal "bubble"/"sized by" **or three `by`-parts** → this wrongly
  catches `balance by ltv by region` and sets x=size=balance (duplicate-column).
- table: `intent=table` / loan-level.

## 4. heatmap / matrix / pivot / treemap / stacked / line / scatter already present?
Yes: heatmap (executor `_execute_grouped` with two dims → `_build_heatmap` pivots a
matrix; validator requires ≥2 dims + metric/count; native React grid renderer),
treemap, line, scatter, bubble all exist. No pivot/stacked-bar concept beyond these.

## 5. Where to add two-dimensional grouped queries WITHOUT a parallel path?
In `_deterministic_parse`: replace the `len(by_parts) >= 3 → bubble` heuristic with
a segment classifier. Two grouping segments where **≥1 is categorical** →
`chart_type="heatmap"` with `dimensions=[...]` (numeric segments bucketed). Reuses
the existing heatmap executor/validator/chart-factory/renderer unchanged.

## 6. Where to add top-N / ranked queries WITHOUT a parallel path?
In `_deterministic_parse`: a ranking detector sets, for grouped ranking, the
existing `bar` route (executor already sorts grouped output descending, `top_n`
already honoured); for loan-level ranking, `intent="table"` + new
`ranking_mode="loan_level"` consumed by a new executor branch `_execute_ranked_loans`
(sibling of `_execute_loan_level`, not a parallel executor).

## 7. How to reuse the dataset contract & displayHints?
Unchanged: the adapter already attaches `displayHints`/`valueFormat` from the single
dataset contract; the heatmap/table artifacts reuse them for currency/percent cell
formatting. No new formatting path.

## 8. How does validation stay data-aware and avoid raw 500s?
Unchanged two-layer validation (`validate_mi_query` + workflow `validate_query_data`)
plus the executor raising controlled `MIQueryExecutionError`/`MIDuplicateColumnError`
that the workflow converts into validation failures. New fields (`sort_by` etc.) are
validated through `referenced_fields()`; the ranked-loans path raises controlled
errors, never a 500.

## Spec fields added (additive, documented)
`sort_by` (semantic key to rank by), `sort_direction` (`asc`|`desc`, default `desc`),
`limit` (loan-level row cap; falls back to `top_n`/10), `ranking_mode`
(`loan_level`|`grouped`|None). All default to no-op so existing specs are unchanged.
