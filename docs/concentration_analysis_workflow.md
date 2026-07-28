# Concentration Analysis — governed workflow

**Workflow id:** `concentration_analysis` · **Calculation version:** 1.1.0

**Artefacts:**

- Workflow package: `mi_workflows/` (`semantics.py`, `engine.py`,
  `concentration_analysis.py`)
- Recogniser + adapter: `mi_agent_api/chat_routing.py`
  (`_route_concentration`, registered as `concentration_analysis`)
- Tests: `tests/test_concentration_analysis_workflow.py`,
  `mi_agent_api/tests/test_concentration_analysis_route.py`

Deterministic measurement of **how exposure is distributed across governed
dimensions** for one portfolio scope at one reporting date. The workflow
measures concentration; it never determines whether concentration is
acceptable, excessive or within appetite — those judgements belong to future
covenant, eligibility and warehouse-monitoring workflows, which are expected
to consume this workflow's measurements. Not portfolio risk comparison,
period change analysis, covenant monitoring, eligibility testing, risk
appetite assessment or forecasting.

## Execution path

```
User question
    → ParsedQuestion (single parse; spec.dimension resolves user vocabulary
      to canonical fields, parse meta says whether the dimension was
      explicitly requested or a parser default)
    → Recogniser Registry (concentration_analysis, priority 66,
      confidence 0.7; Recogniser.metadata declares the BSR axes consumed)
    → chat_routing adapter (resolves frame + portfolio registry + BSR +
      workspace scope; NO calculations)
    → mi_workflows.concentration_analysis (pure)
         scope resolution   portfolio_lens (question text wins over the
                            workspace selection) → trakt_core.portfolio
                            .resolve_scope → mi_agent.portfolio_scope
                            .apply_scope
         semantics          mi_workflows.semantics (BSR v0.2.0, schema 2)
         calculations       mi_workflows.engine (shared deterministic engine)
    → chat envelope → mi_service GovernedResult (+ audit event) → presenters
```

## Recognition — and what it deliberately does not own

Matches: `concentration` / `concentrated`, `diversification`, `<family> mix`
(product / geographic / broker / originator / currency / …), `exposure by X`,
`distribution by X`, `largest / top [N] exposures | loans | borrowers |
obligors`, `single name`.

Rejects (owned elsewhere, checked first):

- **period change** language and any parse with `temporal_mode`/`forecast_mode`;
- **limit / covenant / headroom / appetite** framings and any parse with
  `risk_limit_query` — the risk-limit monitor owns concentration *limits*;
- **location + superlative/where-is** questions ("largest geographic area
  concentration", "where is the book concentrated", "show geographic
  concentration") — the ITL3 geographic exposure capability owns those and
  resolves postcodes to governed ITL3 areas; this workflow still serves
  geography through "geographic mix" / "distribution by region" phrasings;
- **explicitly grouped rankings** ("top 5 brokers *by balance*") and
  numeric-band stratifications ("exposure by LTV band") — the point-in-time
  executor honours the metric, top-N and lens for those;
- **cross-portfolio comparisons** — portfolio risk comparison;
- eligibility, raw exports, reconciliation;
- **HHI / Herfindahl / concentration indices** — measurement only, no index.

## Concentration bases

Two governed bases only, no arbitrary bases:

| Basis | Meaning | When |
|---|---|---|
| `exposure` | current outstanding balance | default wherever the scope is internally currency-consistent and the column carries positive exposure |
| `count` | loans | always computed; the sole basis when the mixed-currency guard suppresses exposure |

Count is returned alongside exposure on every category. The shared
mixed-currency guard decides: an internally mixed scope suppresses exposure
concentration with an explanation, count concentration continues, and **no FX
conversion is ever performed**.

## Dimension governance (BSR)

A candidate dimension qualifies only when the registry declares:

- `analytical_role: dimension`, **and**
- the governed `concentration` category in `categories`;
- asset-specific dimensions additionally need the scope to declare the single
  matching `asset_class`;
- on a scope spanning several portfolios, `requires_scale_alignment` /
  `not_comparable` vocabularies are excluded with the reason recorded (their
  categories would mix originator vocabularies; no heuristic mapping is
  created) — within a single portfolio they are measured.

Nothing is inferred from a field's name. Modes:

1. **Requested dimension** — the parse's canonical dimension, when the user
   explicitly named it (`parse_meta.explicit_dimension_requested`; a parser
   default never masquerades as a request). If governed-but-unusable here,
   the other governed dimensions of the same concept answer, with the
   exclusion recorded.
2. **Requested concept** — question language mapped to one analytical concept
   (geography, product_mix, origination, collateral, …).
3. **Overview** ("how diversified", "largest concentrations") — every
   governed concentration dimension, deterministically bounded to one per
   concept (confidence desc, field name asc); considered-but-not-selected is
   recorded in the audit.
4. **Single name** ("top exposures", "largest borrowers") — governed
   identifiers only: `loan_identifier`; `borrower_identifier` →
   `borrower_1_id`; `original_obligor_identifier` → `new_obligor_identifier`
   (declared precedence, first present wins). No heuristic grouping, no
   connected-counterparty logic.

## Calculations (shared engine only)

`engine.ranked_distribution` — a post-ordering of the existing governed
`distribution` primitive, so shares can never disagree across workflows — per
category: exposure, exposure share, count, count share, rank, cumulative
share; deterministic ordering (basis value desc, category asc); top-5 /
top-10 / user-supplied-N cumulative shares. Unknown values are always an
explicit block (count, count share, exposure, exposure share) with the full
in-scope denominator, never silently excluded; the unknown bucket is
disclosed but not ranked, so cumulative shares can only reach
1 − unknown share.

## Result contract

`workflow_id`, `available`, `reason`, `portfolio_scope` (context, label,
portfolio ids, row count, currencies, mode, dataset), `reporting_date`,
`concentration_basis`, `dimension_results` (canonical field, analytical
concept, confidence, basis, population, total exposure, ranked categories,
top-N shares, unknown block), `single_name_results` (kind, governed
identifier, distinct names, bounded listing with explicit remainder share,
top-N shares, unknown block), `warnings`, `limitations`, `summary`
(observational sentences only — "Largest X is Y", "Top 5 account for Z%",
"Unknown values represent W%"), `evidence` (populations, denominators,
source), `audit`.

## Audit

BSR version + schema, reporting date, portfolio scope, dataset, question,
mode, concentration basis, top-N, currencies + whether exposure concentration
was allowed, dimensions considered / selected / excluded with reasons,
single-name identifiers considered, per-result denominators (count and
exposure), calculation version, outcome. Controlled failures carry the same
audit identity with `outcome: controlled_failure`.

## Explicitly not implemented

HHI or any concentration index, concentration limits, covenant logic,
warehouse tests, eligibility rules, materiality thresholds, risk appetite,
forecasting, FX conversion, connected-counterparty grouping, LLM-derived
calculations.
