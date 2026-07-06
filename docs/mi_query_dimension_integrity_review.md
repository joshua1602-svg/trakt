# MI `/query` Dimension-Integrity Review & Fix

Correctness-first review of the MI query path, addressing the reported failure:

> "Show balance by borrower type by region" detected `geographic_region` in the
> trace but returned output grouped only by `borrower_type`.

The goal was a **fail-closed** query path with an end-to-end trace and a
deterministic regression harness — not cosmetic prompt changes.

---

## 1. The full MI query path

```
question
  └─ parse_with_repair            mi_agent/llm_query_parser.py      (deterministic grammar; LLM optional)
      └─ MIQuerySpec              mi_agent/mi_query_spec.py         (dimension, dimensions[], filters, metric, chart_type…)
        └─ validate_mi_query      mi_agent/mi_query_validator.py    (fields exist in dataset)
          └─ execute_mi_query     mi_agent/mi_query_executor.py     (group-by, aggregate → MIQueryResult)
            └─ create_mi_chart    mi_agent/mi_chart_factory.py      (chart_type → MIChartResult)
              └─ adapt_workflow_result  mi_agent_api/adapters.py    (result → chart/table artifacts + trace)
                └─ responsePresenter / ChartArtifactView   frontend (render)
```

Orchestrated by `run_mi_agent_query` in `mi_agent/mi_agent_workflow.py`.

## 2. The parsed-query contract & the invariant

`mi_agent/mi_query_contract.py` documents every field a parsed MI query carries
(intent, metric(s), dimensions/group-bys, filters, date/period, portfolio scope,
chart/table preference, sort/rank/top-N, aggregation, weighted-average, output
shape, reconciliation) and defines **THE INVARIANT**:

> Every grouping dimension the parser attaches is either **(a) applied** in
> execution (present in the executor group columns / result columns) **or
> (b) explicitly rejected** with a reason. A parsed dimension that is neither is
> a **silent drop**, and the query is refused rather than answered misleadingly.

## 3. Root cause (layered)

The reported symptom was reachable at **three independent layers**:

1. **Executor** (`mi_query_executor.py`) — the `bar` and `table` paths grouped by
   `spec.dimension` (singular) / `spec.x` only, ignoring `spec.dimensions[]`. A
   two-dimension spec routed through those paths grouped by the first dimension
   alone. **Fixed**: a shared `_all_group_dims(spec)` collects
   `dimensions[] + dimension + hierarchy` (de-duped) and every group path
   (`bar`, `table`, `heatmap`, `treemap`) now groups by that full set and records
   `metadata["group_field_keys"]`; the heatmap path records any dimension beyond
   the two-axis grid into `metadata["rejected_dimensions"]` **with a reason**
   (rejected, not dropped).

2. **Adapter/renderer** (`mi_agent_api/adapters.py`) — the `bar`/`line` branch
   used the singular `_dimension_column`, mapping one axis + one series, so a
   result carrying two dimension columns lost the second **on the chart** (the
   table kept both). **Fixed**: when a `bar`/`line` result carries ≥2 dimension
   columns it is **promoted to a heatmap/matrix** (both dimensions become the
   row/column axes) — safe chart selection, never drop a dimension to fit a chart.
   The original single-dimension figure is discarded so it cannot re-introduce
   the dropped dimension.

3. **Grounded sentence** (`frontend/.../responsePresenter.ts`) — the chat lead
   named only `dimensions[0]` ("…shown by borrower type"), reinforcing the
   perception of a drop even when both were charted. **Fixed**: the sentence now
   names **every** grouping dimension ("…by borrower type and region").

## 4. The fail-closed guard

`run_mi_agent_query` now calls `check_dimension_invariant` immediately after
execution. If any parsed dimension was silently dropped, the query is **refused**
(`ok=False`, a validation error, and the reason surfaced in `warnings`/`error`)
rather than answered. The result also carries:

- `dimension_invariant` — `{ok, applied, rejected, dropped}`
- `query_trace` — the end-to-end trace (below), surfaced by the API as `queryTrace`.

## 5. Query trace diagnostics

`build_query_trace` (contract) assembles, and the adapter surfaces as
`response.queryTrace`:

`rawQuery`, `normalisedQuery`, `intent`, `parserMode`/`parserConfidence`,
`metric`/`aggregation`/`weightField`, `dimensionsParsed` (key→canonical→business
name), `filtersParsed`, `rejectedDimensions`/`rejectedFilters` (with reasons),
`executedGroupFieldKeys`/`executedGroupCols`, `resultType`/`resultColumns`,
`chartAxes` (back-filled from the emitted artifact), `topN`/`sortBy`,
`portfolioLens`, `reconciliation`, and the `invariant` block. This makes it
immediately obvious whether a fault is parser-, executor- or renderer-side.

## 6. Generative golden harness

`mi_agent/mi_query_harness.py` generates the suite **from the semantic registry**
(dimensions, their business names/synonyms, and measures) — not a hardcoded list:

- `build_fixture()` — a deterministic funded tape with canonical columns.
- `probe_usable_dimensions()` — discovers which registry dimensions parse **and**
  execute against the fixture (schema-driven; nothing hardcoded).
- `generate_cases()` — single/two/three-dimension groupings, filter+group,
  top-N, ranking (largest/smallest), weighted-average, count, and unsupported-
  concept rejection cases, each over business names **and** synonyms.
- `evaluate_case()` — runs the REAL pipeline and applies the parser/executor/
  renderer invariant checks (dims applied-or-rejected, metric in payload, chart
  axes/table cover applied dims, two categorical dims never collapse to a bar).

Current suite: **229 cases, 100% pass, 5/5 unsupported concepts refused**
(`docs/mi_query_calibration_report.md`).

## 7. What was found & fixed beyond the dimension bug

The harness also caught a **metric-level silent substitution**: "valuation by
broker" and "original balance by broker" resolved to the default
`current_outstanding_balance`. The deterministic parser's metric grammar was a
curated token list that omitted these governed measures. **Fixed** with a
registry-driven metric-synonym pass (`_registry_metric_terms` in
`llm_query_parser.py`): governed measures resolve to their own field
(multi-word phrases beat curated single tokens; over-generic single tokens are
blocklisted so a synonym can never hijack an unrelated question). The requested
metric now appears in the payload instead of being silently swapped.

## 8. Files changed

| File | Change |
|---|---|
| `mi_agent/mi_query_executor.py` | `_all_group_dims`; all group paths honour `dimensions[]`; record `group_field_keys` + `rejected_dimensions` |
| `mi_agent/mi_query_contract.py` | **new** — parsed-query contract, `check_dimension_invariant`, `build_query_trace` (object- and dict-spec aware) |
| `mi_agent/mi_agent_workflow.py` | fail-closed invariant guard; attach `dimension_invariant` + `query_trace` |
| `mi_agent/llm_query_parser.py` | registry-driven metric-synonym resolution (no silent metric substitution) |
| `mi_agent_api/adapters.py` | promote 2-dimension bar/line to heatmap; surface `queryTrace` + `dimensionInvariant` |
| `frontend/.../responsePresenter.ts` | grounded sentence names every grouping dimension |
| `mi_agent/mi_query_harness.py` | **new** — generative golden harness |
| `mi_agent/tests/test_mi_query_invariants.py` | **new** — invariant regression (CI) |
| `scripts/mi_query_calibration.py` | **new** — calibration report generator |
| `docs/mi_query_calibration_report.md` | **new** — generated report |

## 9. Test commands

```bash
# The fail-closed invariant regression (deterministic, offline — CI-safe):
python -m pytest mi_agent/tests/test_mi_query_invariants.py -q

# Regenerate the calibration report:
python scripts/mi_query_calibration.py            # → docs/mi_query_calibration_report.md

# Broader parser/executor/golden regression:
python -m pytest mi_agent -q

# Adapter/chart/workflow API tests:
python -m pytest mi_agent_api -q -k "adapter or chart or artifact or workflow or query"

# Frontend:
cd frontend/mi-agent-ui && npx vitest run
```

No external LLM is used in the deterministic tests (the parser falls back to the
deterministic grammar). An optional live-LLM mode can be layered by the caller;
it is not part of the CI path.

## 10. Previously-failing → now-passing

- `Show balance by borrower type by region` → groups by **both** dimensions;
  chart is a heatmap; both canonical columns present; invariant `ok`.
- `balance by broker by region` (and 59 other two-dimension pairs) → both dims
  survive to result + chart.
- `valuation by broker` / `original balance by broker` → resolve to the
  **requested** measure, not the default balance.
- Unsupported concepts (arrears, defaults, NNEG, credit score, recoveries) →
  **refused**, never answered with a stand-in.

## 11. Remaining unsupported shapes & hardening recommendations

- **3+ dimensions:** two survive on the grid; extras are **rejected with a
  reason** (surfaced in the trace), not silently dropped. A pivot/nested-table
  renderer would let all three display — future work.
- **LLM path:** the invariant guards ANY parser (deterministic or LLM), so an
  LLM that emits a 2-dimension spec is covered end to end. Recommend running the
  harness in a gated live-LLM mode periodically to calibrate the LLM parser.
- **Metric vocabulary:** now registry-driven; adding a measure synonym to the
  registry is picked up with no code change. Keep single-word synonyms out of
  the generic blocklist collision set (`_GENERIC_METRIC_TOKENS`).
- **Parser hardening:** consider promoting `rejected_dimensions` /
  `rejectedFilters` into a visible UI chip so an analyst always sees when a
  requested slice could not be honoured.
