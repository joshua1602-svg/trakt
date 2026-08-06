# mi_agent_pptx — MI Agent-native investor/funder PPTX pack

Generates a standardised **12–15 slide institutional investor/funder PowerPoint
deck** as a by-product of a completed MI Agent pipeline run. The deck updates
automatically whenever a new file is loaded and the MI Agent pipeline has
re-run: point the generator at the run directory and it regenerates the pack
from the latest canonical/analytics artifacts.

This stack is **MI Agent-native**. It does **not** depend on the legacy
Streamlit app (`streamlit_app_erm.py`), legacy Streamlit state/filters/chart
wrappers, or the legacy `analytics/generate_pptx_client.py`. The legacy code was
used only for visual inspiration.

## Source-of-truth principles

1. Consumes the MI Agent **canonical registries** only:
   - `mi_agent/mi_semantics_field_registry.yaml` — field labels, formats,
     aggregations, weighting fields.
   - `config/mi/buckets.yaml` — bucket edges (via `analytics_lib.buckets`).
   - `config/mi/stratification_catalogue.yaml` — dimension ↔ field ↔ state
     eligibility.
   - `config/mi/state_library.yaml`, `config/routes/mi_route.yaml`.
   - `config/mi/mi_equity_release_uk_applicability.yaml` — field applicability
     (broker-channel suppressibility).
2. Consumes **post-pipeline artifacts** from a run directory (`out/runs/<run_id>`):
   canonical typed tape, pipeline tape, and any analytics / metric / chart /
   validation / risk-monitor / forecast / strapline JSON artifacts already
   produced by MI Agent.
3. **No economic derivations** in the PPTX layer beyond the aggregation methods
   the semantic registry already declares (sum / avg / weighted_avg / count).
   All bucketing is delegated to the registry-authorised `analytics_lib`.
4. Missing fields/artifacts produce **branded placeholders + appendix coverage
   notes** — never a crash.
5. Fully **config-driven**: slides, metrics, chart specs, field bindings, lens
   eligibility and broker suppression live in `configs/pptx/investor_pack.yaml`.

## CLI

```bash
python -m mi_agent_pptx.cli \
    --run-dir out/runs/<run_id> \
    --deck-config configs/pptx/investor_pack.yaml \
    --client-name "Client Name" \
    --as-of-date "YYYY-MM-DD" \
    --output reports/client_investor_pack_YYYYMMDD.pptx
```

Optional flags:

- `--portfolio-context total|direct|acquired|<source_portfolio_id>` — the
  **governed analytical scope** the report covers. Resolved through
  `mi_agent_api.portfolio_context.resolve_context()`, the same service React and
  Copilot use. Separate from `--client-id`, which is tenant/client identity.
- `--tenant-id` — owning tenant (defaults to the client id).
- `--lens` — deprecated alias for `--portfolio-context`.
- `--consolidated` — accepted for compatibility (no effect).
- `--work-dir` — where intermediate chart PNGs are written.

Exit codes: `0` publishable · `3` generated but **blocked from publication** by a
mandatory preflight gate (see below) · non-zero otherwise.

## Module layout

| Module | Responsibility |
|---|---|
| `registry_loader.py`  | Read-only access to the MI Agent canonical registries. |
| `artifact_loader.py`  | Discover & load run-directory artifacts (CSV + JSON). |
| `deck_config.py`      | Parse the YAML deck config. |
| `data_resolver.py`    | Normalise the typed tape; materialise registry buckets. |
| `metric_resolver.py`  | Resolve KPI metrics (analytics artifact → registry aggregation → placeholder). |
| `chart_resolver.py`   | Render static charts (matplotlib) onto the theme panel, or a placeholder. |
| `insight_resolver.py` | Straplines: LLM artifact → deterministic template (≤24 words, no fabrication). |
| `pptx_theme.py`       | Brand theme mirroring the MI Agent **React** dashboard. |
| `placeholders.py`     | Branded placeholder charts + appendix coverage notes. |
| `pptx_builder.py`     | Assemble the 16:9 deck (title + strapline + footer per slide). |
| `validation.py`       | Enforce 12–15 slides, straplines populated, mandatory-content checks. |
| `cli.py`              | Command-line entry point. |

## Portfolio context, composition and governed commentary

The deck is **portfolio-aware**. Three modules carry that:

| Module | Responsibility |
|---|---|
| `deck_context.py` | `DeckPortfolioContext` — scope, constituent books, per-book reporting dates, and a governed funded snapshot per portfolio type. Composes existing contracts; computes no economic value. |
| `composition.py` | Decides which slides the portfolio justifies (`when:` conditions + per-type data guards) and records every omission with a reason. |
| `insights.py` | The deterministic executive summary. Extends the governed `Insight` contract from `mi_agent_api`. **No LLM.** |
| `preflight.py` | Publication gates. A deck that fails one is generated but withheld. |

Consequences:

- every deck states its **reporting scope, constituent books and reporting
  dates** on the cover, and carries a scope stamp in every slide footer — a
  single-book pack can no longer be read as a total-portfolio pack;
- there are **no placeholder slides**: a section with no data is omitted and
  explained in the appendix's omission ledger;
- when both a direct and an acquired book are in scope, the pack adds a
  **Portfolio Composition** slide and a **Direct vs Acquired** movement
  attribution, so a blended total cannot hide a growing book offset by a
  redeeming one.

## Deck structure (`configs/pptx/investor_pack.yaml`)

The deck has **no fixed length** — composition decides it. The configured
sequence is: Cover · Executive Summary (governed observations) · Portfolio
Composition · Direct vs Acquired _(mixed books only)_ · Funded Key Measures ·
Stratifications I–III · Multi-Dimensional Risk Analytics · Geographic Exposure ·
Funded Balance Evolution _(≥2 periods)_ · Origination Vintages · Pipeline
Overview / Evolution / Funnel / Flow _(pipeline only)_ · Forecast Bridge /
Projection _(forecast only)_ · Risk Limits _(limits only)_ · Methodology ·
Appendix.

Cover, Methodology and Appendix are mandatory — they are the pack's disclosure
spine, and a deck missing one fails preflight.

## Publication gates (`preflight.py`)

Checked against the **rendered file**, not the build record: the deck opens; the
mandatory slides are present; the reporting scope, every reporting date and
every constituent book are rendered; direct + acquired reconcile to the total in
both balance and loan count; an executive summary was generated where
observation was possible; no placeholder slide reached the deck.

On failure the deck is still written (an operator needs it to diagnose the run),
a `<deck>.preflight.json` sidecar records the verdict, the CLI returns `3`, and
`pptx_stage` withholds durable publication and marks the artefact
`generated_not_published`.

## Charting

Static matplotlib PNGs that reproduce the React dashboard's visual language:

- the signature breakdown visual is the dashboard **BarList** — horizontal
  periwinkle bars with the category label left and a right-aligned mono value
  (`£X.XMM`), ordered by the registry bucket order (LTV/age ascending);
- time series use monotone lines with a top→bottom gradient area fill;
- the heatmap uses the dashboard navy→periwinkle→mint ramp with contrast-
  flipping cell values;
- each figure is rendered at the **exact width×height of its slide panel**, so
  python-pptx never stretches it (no distortion), onto the panel background
  (`#12152b`) so there are no white boxes.

Colours/typography mirror the dashboard (navy `#232D55`, periwinkle `#919DD1`,
`£X.XMM`/`£XK` value format). No `plotly`/`kaleido`/Chrome dependency — it runs
headless in Azure Functions.

## Pipeline canonicalisation

A pipeline run may land as the raw central tape (`18a_central_pipeline_tape.csv`)
whose columns carry source-alias headers ("Loan Amount", "Status", "Broker",
"Property Region"…). `pipeline_prep.canonicalise_pipeline` maps these onto the
canonical field names using the shippable `config/mi/pipeline_field_contract.yaml`
aliases, normalises the stage vocabulary, and derives the registry forecast
inputs (completion probability, weighted expected amount, expected completion
date) from `config/client/pipeline_expected_funding.yaml`. So pipeline &
forecast charts render from real data — only the Risk Monitor is a genuine v1
placeholder.

## Lenses & deltas

Every metric and slide declares a **lens** (`funded` / `pipeline` / `forecast`)
and resolves against that lens's frame only — the pipeline total is never the
funded total. A slide whose lens has no data renders a branded placeholder
instead of borrowing another lens's numbers. Pass `--prior-run-dir <dir>` to
render **prior-period deltas** on the KPI tiles ("▲ +£0.7MM vs prior"); with no
prior run, tiles show the value without a fabricated delta.

## Tests

```bash
python -m pytest tests/mi_agent_pptx/ -q
```

Covers artifact loading, data resolution, registry-authorised metric
resolution, missing-field fallbacks, chart + placeholder creation, straplines,
broker suppression, and end-to-end PPTX creation/validation.
