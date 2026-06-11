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

`build_mi_semantics_registry.py` reads the canonical registry and selects a
field if **any** of these hold (OR, not AND):

- `core_canonical: true`, **or**
- `layer` in `{performance, product}`, **or**
- `category: analytics`.

Each selected field records *why* it was chosen in `source_criteria`
(e.g. `["core_canonical"]`, `["layer:performance"]`, `["category:analytics"]`).

For every field the build heuristically infers:

- **role**: `metric | dimension | date | identifier | flag | unknown`
- **format**: `currency | percent | integer | decimal | date | string | boolean`
- **chartable**, **allowed_aggregations** / **default_aggregation**
- **allowed_chart_roles** / **default_chart_role**
- **weight_field** (balance field used for weighted averages of rates/LTVs)
- **bucket_field** (suggested bucketing dimension, e.g. `age_bucket`, `ltv_bucket`,
  `ticket_bucket`, `vintage_year`)

Fields that cannot be classified safely are marked `role: unknown` with a note
that they *require manual analytics classification*.

The output also carries top-level `metadata` (generation timestamp, source
registry path, selection rules, field count, version, default weight field).

> The generated metadata is heuristic and **requires human review before
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

- **No chart rendering yet** — a spec describes a chart; it does not draw one.
- **No Streamlit chat UI yet.**
- **No MI pipeline integration** — not wired into `trakt_run.py` or any gate.
- **LLM parser is a skeleton** — the live Claude path is optional and mockable;
  the deterministic parser handles only a handful of example phrasings.
- **No scenario support yet.**
- **Semantics are heuristic** and must be reviewed by a human before production
  use (especially `role: unknown` fields).
- The LLM is only ever shown the data-free semantic catalogue; it never sees raw
  dataset values, and generated content is parsed as data only — never executed.
