# Portfolio Risk Comparison — governed workflow

**Workflow id:** `portfolio_risk_comparison` · **Calculation version:** 1.0.0

**Artefacts:**

- Workflow package: `mi_workflows/` (`semantics.py`, `engine.py`,
  `portfolio_risk_comparison.py`)
- Recogniser + adapter: `mi_agent_api/chat_routing.py`
  (`_route_portfolio_comparison`, registered as `portfolio_risk_comparison`)
- BSR resolver wiring: `mi_agent_api/dependencies.py`
- Tests: `tests/test_portfolio_risk_comparison_workflow.py`,
  `mi_agent_api/tests/test_portfolio_risk_comparison_route.py`

Deterministic comparison of **two governed portfolio scopes at one reporting
date**. Reports observed characteristics side by side; never determines
whether a portfolio is acceptable, compliant, within appetite, safer or
preferable. Not period change analysis, concentration analysis, covenant
monitoring, eligibility testing, forecasting or risk scoring.

## Execution path

```
User question
    → ParsedQuestion (single parse; semantics_context now populated by the
      Business Semantics Registry resolver wired in build_dependencies)
    → Recogniser Registry (portfolio_risk_comparison, priority 65,
      confidence 0.7 — a workflow question outranks single-capability routes;
      Recogniser.metadata declares the BSR terms consumed)
    → chat_routing adapter (resolves frame + portfolio registry + BSR;
      NO calculations)
    → mi_workflows.portfolio_risk_comparison (pure)
         scope resolution   portfolio_lens.resolve_comparison_lenses
                            → trakt_core.portfolio.resolve_scope
                            → mi_agent.portfolio_scope.apply_scope
         semantics          mi_workflows.semantics (BSR v0.2.0, schema 2)
         calculations       mi_workflows.engine (shared deterministic engine)
    → chat envelope → mi_service GovernedResult (+ audit event) → presenters
```

## The shared analytical engine

`mi_workflows.engine` is the single deterministic calculation engine for
governed workflows. This workflow is its first full consumer; the engine is
deliberately workflow-neutral so period-change analysis and later workflows
compute through the same primitives:

| Primitive | Contract |
|---|---|
| `aggregate` | sum / average / weighted average / share, driven entirely by the BSR's `default_aggregation` / `weight_field` / `share_basis`; discloses population, valid population, exclusions and the denominator used |
| `distribution` | categorical mix with count share, exposure share and an explicit `(unknown)` bucket; shares are over all rows in scope so they always sum to 1 |
| `compare_values` | absolute = A − B; relative = (A − B)/\|B\|; the only comparison formula |
| `directionality_verdict` | observed relation interpreted through governed directionality; never aggregated into a score |
| `currency_profile` / `mixed_currency_guard` | monetary comparison only when currency profiles match; no FX conversion, suppression is disclosed |
| `unit_for_field` | unit from the runtime MI semantics layer's per-field `format` |

## Recognition

Matches: `compare`, `versus`, `vs`, `differences between`, comparative
which-questions (`which portfolio has higher …`), over portfolio nouns
(portfolios, books, SPVs, warehouse pools, originators, cohorts), and any
question naming two governed scopes (`direct_001 vs acquired_001`,
originated vs acquired).

Rejects (owned elsewhere, checked first): period/temporal language and any
parse with `temporal_mode ∈ {compare, trend}` or a forecast mode;
concentration; covenant/appetite/limit; eligibility; raw loan exports;
reconciliation.

## Scope, reporting date, currency, asset class

- Sides resolve to explicit portfolio-id lists through the governed registry;
  an unknown scope, a single-portfolio deployment, a no-provenance tape, more
  than two named scopes, or an empty side is a **controlled failure** naming
  what is available.
- Comparison requires one common reporting date; differing or ambiguous dates
  are refused. A tape without the column is one governed snapshot and that is
  disclosed.
- The mixed-currency guard suppresses monetary metrics (unit `currency`) on
  any intra- or inter-scope mismatch; non-monetary comparison continues;
  within-portfolio exposure *shares* survive a cross-portfolio currency
  difference but not an internally mixed scope.
- Fields are compared per the BSR: `comparable` directly;
  `requires_scale_alignment` excluded with an explanation (no heuristic
  mappings); `within_asset_class_only` and asset-specific fields only when
  both scopes declare the same single governed asset class (`asset_class`
  column); `not_comparable` excluded.

## Modes

1. **Requested metric** — the parse's canonical metric, if BSR-governed.
2. **Requested concept** — question language mapped to one analytical concept
   (payment performance, credit quality, geography, …).
3. **Overview** — measures and dimensions tagged `portfolio_comparison` under
   exposure, leverage, payment performance, credit quality, collateral,
   valuation, pricing, maturity, product mix, geography; deterministically
   bounded per concept (by confidence, then field name) with everything
   considered-but-not-selected recorded in the audit.

## Result contract

`workflow_id`, `available`, `reason`, `comparison_scope`, `reporting_date`,
`portfolio_results`, `metric_comparisons` (field, concept, aggregation,
weight basis, unit, per-side value + populations + denominator, absolute and
relative difference, declared directionality + interpretation, comparability,
confidence), `distribution_comparisons` (count/exposure/unknown shares per
category with `present_in`), `warnings`, `limitations`, `summary`
(observational sentences only), `evidence` (populations, denominators,
source), `audit` (BSR version + schema, reporting date, selected portfolios,
dataset, currencies, asset classes, fields considered/selected/excluded with
reasons, comparability decisions, aggregations, weights, denominators,
calculation version).

## Explicitly not implemented

Concentration indices or limits, covenant testing, warehouse eligibility,
materiality thresholds, composite risk scores, forecasting, FX conversion,
LLM-derived calculations.
