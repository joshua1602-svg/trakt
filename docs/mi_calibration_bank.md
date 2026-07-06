# MI Agent Curated Calibration Bank (250+)

A curated bank of realistic business-user MI questions with declared **expected
semantic behaviour** — additional to the generated registry harness
(`mi_agent/mi_query_harness.py`), not a replacement.

## Files

| File | Role |
|---|---|
| `config/mi/golden_questions/ere_mi_calibration_250.yaml` | The bank — 252 curated questions + per-case expectations |
| `scripts/build_mi_calibration_bank.py` | Regenerates the YAML (curated questions + calibrated expectations) |
| `mi_agent/mi_calibration.py` | Evaluation engine — runs each question through the real deterministic MI path and checks its expectations |
| `mi_agent/tests/test_mi_calibration_bank.py` | Pytest suite enforcing the bank (known-gaps xfailed) |
| `scripts/mi_query_calibration.py` | Calibration report now includes the curated-bank section |
| `docs/mi_query_calibration_report.md` | Generated report (generated harness + curated bank) |

## Count & category breakdown (252 questions)

| Category | Count |
|---|---:|
| basic_kpi | 32 |
| single_dim | 64 |
| two_dim | 26 |
| filtered | 37 |
| ranking | 23 |
| pipeline | 12 |
| forecast | 9 |
| risk | 20 |
| data_quality | 9 |
| unsupported | 10 |
| ambiguous | 10 |

## Per-case expectation schema

`id, category, question, expected_status (answer|refuse|clarify), execution
(full|parse_only), expected_scope, expected_metric(s), expected_dimensions,
expected_filters, expected_artifact_type (kpi|table|bar|heatmap|line|treemap|none),
expected_reconciliation, expected_dimension_invariant_ok,
expected_filter_invariant_ok, expected_warnings, expected_columns_include,
expected_min_columns, known_gap, notes`.

Each **answer** case asserts: metric resolved, dimensions resolved AND applied,
filters resolved AND applied, `dimensionInvariant.ok`, `filterInvariant.ok`,
artifact type compatible, reconciliation present when expected, and required
columns present. Each **refuse/clarify** case asserts it is not answered with
data and a reason is surfaced (no KPI/chart/table artifact).

`pipeline`/`forecast` run in `parse_only` mode: their execution needs the runtime
chat-routing harness (pipeline/forecast data), so the deterministic funded suite
validates them at parse level (no hallucinated fields; metric where determinable).

## How to run

```bash
python -m pytest mi_agent/tests/test_mi_calibration_bank.py -q   # curated bank
python -m pytest mi_agent/tests/test_mi_query_invariants.py -q   # generated harness
python scripts/mi_query_calibration.py                          # regenerate the report
python scripts/build_mi_calibration_bank.py                     # regenerate the YAML
```

## Result

**0 hard failures** — every non-known-gap case holds its declared expectation
(234/252 pass; 18 xfailed known gaps). Unsupported concepts (NNEG, credit score,
defaulted balance, arrears, recoveries, indexed value) are all correctly refused
with a reason, never fabricated.

## Discovered limitations (marked `known_gap`, not loosened)

Each states the IDEAL behaviour and is xfailed with a follow-up reason:

- **Ambiguous answered instead of clarified** — "show best brokers", "show bad
  regions", "show profitability by region", "show me interesting regions" are
  silently answered as *balance by dimension*. Follow-up: detect subjective /
  unavailable concepts and clarify/refuse.
- **3rd dimension silently dropped at parse** — "balance by region by borrower
  type by LTV bucket" keeps only two dims; the third is dropped at parse time
  (`dim_keys[:2]`) with no warning, and the dimension invariant cannot catch it
  because it never reaches the spec. Follow-up: warn at the parser, or route 3+
  dims to a pivot table.
- **Place-name exposure not filtered** — "exposure to London / the South East"
  returns whole-book balance (no region filter). Follow-up: parse place names as
  region-value filters.
- **`concentration by <bucket>`** pulls the bucket's measure (WA LTV / mean age)
  instead of balance. Follow-up: prefer balance for "concentration by <bucket>".
- **`largest regional concentration`** resolves to a loan-level table, not a
  region concentration bar.
- **`property value band`** mis-maps to the age bucket (no valuation-band dim).
- **Bare `balance where <filter>`** (no total/how-many cue) resolves the metric to
  LTV and drops the filter — use "total balance where …".
- **`average loan size`** parses "size" as the ticket-size dimension.
- **Data-quality counts** ("missing region count", "loans excluded from LTV
  analysis") are not a first-class intent — coverage is delivered via the
  reconciliation block on a normal result. Follow-up: a dedicated data-quality
  intent.
- **Filtered time-series** ("balance trend where LTV above 50%") answers an
  unfiltered trend (the line path does not attach filters) — a fail-closed gap.
