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
  the deterministic parser handles only a handful of example phrasings and does
  not yet consult `synonyms` for field resolution (a v2 item).
- **No scenario support yet.**
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
