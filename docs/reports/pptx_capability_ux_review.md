# PPTX Capability & UX Review

**Status:** audit / design review only. No production code, template, component,
MI calculation or configuration was changed.
**Base:** `main` @ `e7678c8`.
**Method:** source tracing plus three executed probes (see
[Appendix A — evidence probes](#appendix-a--evidence-probes)). Every claim below
cites a file and, where it matters, a line. Where a module exists but is not on
the production path, it is called out as such rather than counted as capability.

---

## 1. Executive answer

**How much of the desired PPTX can be produced today?**
More than the brief assumes. The deck that ships today is **23 configured slides**
(`configs/pptx/investor_pack.yaml`), composition-gated down to what the portfolio
justifies, and it already calls the *same in-process compute functions* as the
React dashboard's `/mi/*` endpoints (`mi_agent_pptx/mi_api.py`). Of the ~18
distinct visuals in the proposed scope, **11 already exist in the deck**, **5 need
renderer/layout work only over payloads that are already resolved or already
governed**, **2 need a small composition of existing primitives**, and **0 need a
genuinely new MI calculation**.

**How much is renderer/UX work?**
The overwhelming majority. The single largest gap in the product is not analysis —
it is that the deck and the dashboard **re-implement the same presentation
decisions in three separate stacks** (React/Recharts, matplotlib, server-side
Plotly), and those three have already drifted on bucket ordering, currency symbol,
number formatting and colour scales. That is the work worth doing.

**How much genuinely requires new MI?**
Essentially none for the proposed deck. Two proposed items (£200m scale target;
pipeline stratification by LTV/age/ticket/rate) are *configuration* and *a
breakdown call over an already-bucketed frame* respectively — not new economics.

**Can PPTX genuinely match the React Dashboard UX?**
Yes on data and semantics — it substantially already does, and there is a
committed cross-channel parity test that proves it
(`tests/mi_agent_pptx/test_channel_parity.py`, 15 passed / 1 skipped when run for
this review). No on pixel identity, and that is fine. What it *cannot* do today
without work is guarantee **the same bucket order, the same currency symbol and
the same label hygiene**, because those live only in TypeScript.

**Recommended architecture.**
Option **C+ — one governed chart-data payload, two renderers, plus a shared
presentation contract**. Concretely: move ordering, label hygiene, currency
symbol, series colour role and number format *out of the React components and out
of the matplotlib renderer* and into the payload the API already emits. This is
the lowest-drift option that is actually feasible here (screenshotting React from
an Azure Function is not — see §9). It is also small: the payloads already exist
and already travel to both surfaces.

**The one-line summary:** the analysis is already shared; the *presentation
grammar* is not, and that is where every observed divergence lives.

---

## 2. Existing PPTX

### 2.1 Production entry point and data flow

```
Azure blob trigger
  → apps/blob_trigger_app/orchestrator_invoke.py:308
  → apps/blob_trigger_app/pptx_stage.generate_investor_pptx()      (pptx_stage.py:410-472)
        argv: --run-dir --deck-config --client-name --output
              [--as-of-date --portfolio-context --client-id --tenant-id --output-root]
  → mi_agent_pptx.cli.run(argv)                                     (pptx_stage.py:102-109)
        ├─ _load_slides(configs/pptx/investor_pack.yaml)
        ├─ mi_api.build_dashboard_data(...)      ← calls the SAME compute fns as /mi/*
        ├─ deck.DeckBuilder(data, ctx).build(slides, output)
        │     ├─ composition.build_facts / select_slides   (gate + omission ledger)
        │     └─ per-slide handler → mi_agent_pptx/render.py (matplotlib PNG) → python-pptx
        └─ preflight.run_preflight(report, data) → <deck>.preflight.json
  → validate_deck_file / deck_checksum / persist_investor_deck (durable blob)
  → run_state.json artifacts["investor_pack_pptx"]
```

There is a **second, on-demand** entry point for the same generator:
`POST /mi/decks/generate` (`mi_agent_api/app.py:1543`) →
`mi_agent_api/deck_generation.py:282-284` → the *same*
`pptx_stage.generate_investor_pptx`. The React control is
`frontend/mi-agent-ui/src/components/DeckDownloadMenu.tsx`, which polls the job and
distinguishes `generating / completed / blocked / failed / unavailable`. So the
deck is genuinely wired into the dashboard, not a batch side-product only.

| Concern | Where |
|---|---|
| Entry point | `apps/blob_trigger_app/pptx_stage.py`; `mi_agent_api/deck_generation.py` |
| Template | **None.** No `.potx`. The deck is assembled shape-by-shape in `mi_agent_pptx/deck.py` on a 13.33×7.5in blank presentation. |
| Slide builders | `mi_agent_pptx/deck.py` (2,151 lines, 24 handlers, `_DISPATCH` at :2090) |
| Chart renderers | `mi_agent_pptx/render.py` (matplotlib) + `chart_resolver.render_bridge_waterfall` |
| Screenshot pipeline | **None exists.** No Playwright/Chrome/kaleido anywhere in the deck path. |
| Data input | `mi_agent_pptx/mi_api.build_dashboard_data` — in-process calls to `mi_agent_api.*` |
| MI services consumed | `snapshots.compute_funded_snapshot`, `evolution.{funded_evolution, funded_bridge, funded_cohort_progression, pipeline_evolution, pipeline_funnel_evolution, forecast_evolution}`, `cohorts.cohort_analysis`, `geo.exposure_by_itl3`, `pipeline_contract.{load_prepared_pipeline, compute_pipeline_snapshot, compute_prior_week_aggregates, build_pipeline_history}`, `forecast_bridge.compute_forecast_bridge`, `workspace.forecast_breakdowns`, `forecast_extrapolation.build_extrapolation`, `concentration_tests_api.compute_concentration_tests`, `risk_limits.compute_risk_limits`, `portfolio_scope.resolve/apply_scope`, `funded_prep.prepare_funded_mi_dataset`, `datasets._resolve_pipeline_source` |
| Config consumed | `configs/pptx/investor_pack.yaml`; transitively `config/mi/buckets.yaml`, `config/mi/stratification_catalogue.yaml`, `config/risk/concentration_test_library.yaml`, `config/client/pipeline_expected_funding.yaml`, `config/mi/pipeline_field_contract.yaml` |
| Client branding | `--client-name` string + `deck.logo_path` (**currently `null`**, `investor_pack.yaml:34`). There is no per-client brand registry, and React uses Trakt's own `trakt-mark.svg`. Effectively: **no client branding on either surface.** |
| Reporting period / version | `--as-of-date`, `run_state.json.reporting_date`, `snapshots.infer_reporting_date`; the preflight sidecar records `template_version`, `insight_version`, `source_runs`, `input_snapshots` |
| Output | `<run_dir>/reports/investor_pack.pptx` (+ `.preflight.json`), then durable blob, served via `/mi/decks/download` |
| Tests | `tests/mi_agent_pptx/` — 15 files, incl. `test_channel_parity.py` |

### 2.2 Current slide list

Composition (`mi_agent_pptx/composition.py`) evaluates a restricted-AST `when:`
expression over 24 governed facts, then a per-type data guard. Anything dropped
is recorded with an investor-facing reason and rendered in the methodology page's
omission ledger. **There are no "no data" placeholder pages.** Max rendered = 22
(`risk` and `concentration` are mutually exclusive).

| # | Slide (`type`) | Purpose / metrics | Chart | Calculation source | Rendered-from-dashboard? | Styling shared? | Data/config shared? | Independent PPTX logic? |
|---|---|---|---|---|---|---|---|---|
| 1 | Cover (`cover`) | Entity, governed scope, constituent books, every reporting date | — | `deck_context.build_context` | n/a | duplicated tokens | yes | no |
| 2 | Executive Summary (`exec_insights`) | ≤N deterministic observations, no LLM | text | `mi_agent_pptx/insights.py` (12 generators) | **no** — separate generator set from React's Weekly Brief | duplicated | figures yes, prose no | **prose selection only** |
| 3 | Portfolio Composition (`portfolio_composition`) | Total then per-type on identical measures | stacked bar + cards | `compute_funded_snapshot` per type | B | duplicated | yes | no |
| 4 | Movement & Drivers (`movement_drivers`) | Why funded AuM moved, by dimension | waterfall | `evolution.funded_bridge` via `mi_agent_pptx/movement.py` | **not in React panels** (React exposes movement via `/mi/insight/movement-detail` drawer) | duplicated | yes | no |
| 5 | Funded Key Measures (`kpi_summary`) | First 10 KPI tiles, 5-col grid | tiles | `compute_funded_snapshot.kpis` **verbatim** | B — same tile objects React's `FundedSnapshotPanel` renders | duplicated | yes | no |
| 6–8 | Stratifications I–III (`strat_barlists` ×3) | ltv+ticket, age+borrower_type, broker+region | BarList ×2 | `snapshots._funded_stratifications` **+ PPTX-only `_extra_stratifications`** | B, partially | duplicated | **partly** | **yes — see §4** |
| 9 | Multi-Dimensional Risk Analytics (`multidim`) | LTV×Age, LTV×BorrowerType, LTV×Region | heatmap ×2–3 | **`mi_api._matrix` / `_multidim` — PPTX-only cross-tab** | **no React equivalent** | duplicated | buckets yes, cross-tab no | **yes** |
| 10 | Geographic Exposure (`geo`) | 4 tiles + top-12 areas | tiles + BarList | `geo.exposure_by_itl3` | B — but React draws a **choropleth**, deck draws a bar list | different visual | yes | `top5/total` ratio computed in deck |
| 11 | Funded Balance Evolution (`funded_evolution`) | balance, WA LTV | 2 line charts | `evolution.funded_evolution` | B — React shows **4** series (adds loan_count, WA rate) | duplicated | yes | no |
| 12 | Vintage Formation (`cohorts`) | Vintage composition table | table | `cohorts.cohort_analysis` | B | duplicated | yes | no |
| 13 | Cohort Progression (`cohort_progression`) | Static-pool seasoning per vintage | lines + change table | `evolution.funded_cohort_progression`, one call per vintage | B — same call the React Cohorts tab issues | duplicated | yes | cohort *selection* (`cohorts.select_cohorts`, ≥2% share, max 4) |
| 14 | Pipeline Overview (`pipeline_summary`) | 4 tiles + stage + broker/region | tiles + BarList ×2 | `pipeline_contract.compute_pipeline_snapshot` | B — React shows 8 tiles + 5 panels | duplicated | yes | `avg_case = amount/cases` |
| 15 | Pipeline Evolution (`pipeline_evolution`) | Pipeline stock over time | lines | `evolution.pipeline_evolution` | B | duplicated | yes | no |
| 16 | Origination Funnel (`funnel`) | Latest weekly flow by stage (falls back to current cases by stage) | BarList | `evolution.pipeline_funnel_evolution` | B — but **drops the conversion payload entirely** | duplicated | yes | no |
| 17 | Origination Flow (`origination_flow`) | Weekly KFI/completion run-rate | lines ×2 | `evolution.pipeline_funnel_evolution` | B | duplicated | yes | no |
| 18 | Forecast Bridge (`forecast_bridge`) | Funded → +expected completions by month → Forecast | waterfall | `forecast_bridge.compute_forecast_bridge` + `workspace.forecast_breakdowns` | B | duplicated | yes | month head/tail split (presentational) |
| 19 | Forecast Projection (`forecast_projection`) | Run-rate scale-up, milestone dates | lines + table | `forecast_extrapolation.build_extrapolation` | B | duplicated | yes | no |
| 20 | Portfolio Health & Watch Items (`watchlist`) | ≤5 watch items, ≤3 positives | text | `mi_agent_pptx/watchlist.py` over governed payloads | **no React equivalent** | duplicated | figures yes | selection/ranking only |
| 21 | Concentration Tests & Headroom (`concentration`) | Current / Expected / Full-pipeline stress, headroom, RAG, breach horizon | utilisation chart + table | `concentration_tests_api.compute_concentration_tests` | B — React `RiskLimitsWorkspace` | duplicated | yes | status vocabulary mapping only |
| 22 | Risk Limits (`risk`) | Legacy extracted monitor, only when no approved config | table | `risk_limits.compute_risk_limits` | B | duplicated | yes | no |
| 23 | Data and Methodology (`methodology` → `slide_appendix`) | Scope, dates, coverage, basis, omission ledger | text | `mi_api.diagnostics` + composition omissions | n/a | duplicated | yes | no |

**Two handlers exist but are never configured, so they do not run in production:**
`portfolio_comparison` (deliberately retired, per the comment at
`investor_pack.yaml:60-64`) and **`forecast_evolution`** (`deck.py:1526`, registered
at `deck.py:2104`, but no slide of that type exists in the YAML). The latter matters
— see §6.

### 2.3 "The PPTX currently renders dashboard charts into the deck" — what that actually means

**Answer: B, with one qualification.**

- **Not A** (rendering the React component). There is no browser, no Playwright, no
  Chrome, no kaleido in the deck path. `mi_agent_pptx/render.py:15-17` is
  `matplotlib.use("Agg")`; the README explicitly states the deck "runs headless in
  Azure Functions" with "no plotly/kaleido/Chrome dependency".
- **Not C** (screenshot). Nothing in `pptx_stage.py`, `deck.py` or `render.py`
  captures an image from the dashboard.
- **It is B**: the deck resolves the *identical payload objects* the dashboard's
  `/mi/*` handlers return — by calling the same functions in-process
  (`mi_agent_pptx/mi_api.py`, entire module) — and then draws them with a second,
  hand-written matplotlib renderer whose visual vocabulary was written to imitate
  the React one (`render.py:1-8`: "a faithful export of the dashboard's
  Recharts/BarList/stat-tile components"; `render.py:36` copies React's
  `EvolutionPanel` palette by value).
- **The qualification (this is D, and it is where the risk lives):** for
  *presentation* decisions the payload does not carry, the two surfaces decide
  independently. Bucket order, label hygiene, currency symbol and colour scale are
  all decided *after* the shared payload — in TypeScript on one side and in Python
  on the other. So it is **B for numbers and D for presentation grammar.**

There is also a **third** chart stack in the product: `mi_agent/mi_chart_factory.py`
(server-side Plotly, live via `mi_agent_workflow.py:1222`), which produces the chat
artifact figures. It carries its own light theme (`mi_chart_factory.py:65-90`), its
own categorical palette, its own `compact_currency` (`:162`) and its own bucket
sorter (`:476`). React re-skins it at render time
(`frontend/.../lib/plotlyTheme.ts`). That is three formatting implementations and
four bucket-ordering implementations for one product.

### 2.4 Dead code inside `mi_agent_pptx` (do not count it as capability)

The README's module table describes the **v1 architecture, which is no longer on
the production path**. Verified by grep across the repo excluding tests:

| Module / class | Lines | Referenced by |
|---|---|---|
| `pptx_builder.py` (whole module, incl. `PptxDeckBuilder`) | 497 | **nothing** |
| `chart_resolver.ChartResolver` | ~440 | only `pptx_builder.py` |
| `insight_resolver.StraplineResolver` | 148 | only `pptx_builder.py` |
| `validation.py` | 91 | **nothing** |
| `data_resolver.py` / `metric_resolver.MetricResolver` | 260 + class | only via the `__init__` re-export and `chart_resolver` |

Live from those files: `metric_resolver.compact_currency/compact_number/format_percent`
and `chart_resolver.render_bridge_waterfall` only. **≈1,400–1,700 of 10,247 lines in
`mi_agent_pptx` are v1 residue.** This is the main reason a reader of the README
comes away with the wrong mental model of what the deck does.

---

## 3. Existing React Dashboard

### 3.1 Structure

`AppShell.tsx` is a chat-plus-workspace application, not a report:

- **Header** (`HeaderBar.tsx`): portfolio, run, reporting period, identity,
  portfolio-scope selector, `DeckDownloadMenu`, `ExportMenu`.
- **Weekly Portfolio Brief** (`WeeklyBriefPanel.tsx`) — **flag-gated off by
  default**: `/mi/insights/weekly-brief` 404s unless `TRAKT_MI_WEEKLY_BRIEF` is set
  (`mi_agent_api/app.py:1146-1149`).
- **Core Dashboard** — four lens tabs (`ViewToggle`): **Funded**, **Pipeline**,
  **Forecast**, **Risk Limits**, each capability-gated with a
  `PortfolioScopeBanner` disclosure.
  - Funded → Stratifications (`FundedSnapshotPanel`) · Geography
    (`GeographyPanel` + `UkChoropleth`) · Evolution (`EvolutionPanel`) · Cohorts
    (`CohortsPanel`).
  - Pipeline → Stratifications (`PipelineSnapshotPanel`, `PipelineWatchlist`) ·
    Evolution (`EvolutionPanel` tabs `pipeline` + `origination`).
  - Forecast → Projection (`ForecastView` + `ForecastExtrapolationPanel`) ·
    Forecast Evolution (`EvolutionPanel` tab `forecast`).
  - Risk Limits → `RiskLimitsWorkspace` + `ConcentrationDetailPanel`
    (drill-through, history, pipeline drivers).
- **Artifact workspace** (`ArtifactCanvas` / `ArtifactRenderer`) — chat answers as
  KPI / chart / table / validation / risk / scenario cards.

**Critical for §11: there is no executive landing page and no combined
pipeline+funded+forecast+risk tile row anywhere in React.** The nearest thing is
`FundedSnapshotPanel`'s funded-only KPI grid, which the deck already reproduces
verbatim on slide 5.

### 3.2 Measures, dimensions and views actually surfaced

| Surface | Content | Source |
|---|---|---|
| Funded KPIs | balance, loans, WA current LTV, WA original LTV, avg balance, WA rate, WA months on book, WA youngest age, % single borrowers, WA property value, MoM balance, MoM loans, **exited/redeemed loans**, NNEG *or* arrears risk tile | `snapshots.py:613-741` |
| Funded stratifications | **8**: ltv, age, region, rate, product, vintage, status, equity — each with `availability` + `reason` | `snapshots.py:333-342` |
| Geography | ITL3 choropleth + ranked list, coverage % | `geo.exposure_by_itl3` |
| Funded evolution | funded_balance, loan_count, wa_ltv, wa_interest_rate | `EvolutionPanel.tsx:970-981` |
| Cohorts | vintage composition; progression metrics: funded_balance, wa_ltv, wa_interest_rate, **nneg_headroom_pct**, loan_count | `EvolutionPanel.tsx:603-608` |
| Pipeline | 8 tiles; stage amount, stage count, weighted-expected by completion month, broker, region | `PipelineSnapshotPanel.tsx:159-205` |
| Origination / funnel | weekly flow by stage, 5-week average, **cohort conversion + weekly velocity + lag disclosure + sufficiency gating** | `EvolutionPanel.tsx:280-330`, `evolution.py:934-1120` |
| Forecast | forecast balance **by region**, **by LTV bucket**, by completion month; run-rate & KFI-conversion projection with milestone ladder | `ForecastView.tsx:57-67`, `forecast_extrapolation.py` |
| Forecast evolution | funded actual / weighted pipeline / forecast; **actual vs prior-run forecast** | `EvolutionPanel.tsx:1137-1149` |
| Risk | approved concentration tests, current/expected/stress, headroom, RAG, drill-through, **history**, **pipeline drivers** | `RiskLimitsWorkspace.tsx`, `ConcentrationDetailPanel.tsx` |

### 3.3 The visual system, and how centralised it is

| Element | Centralised? | Where |
|---|---|---|
| Brand + RAG + categorical palette | **Yes, per surface — but duplicated across three surfaces** | `src/lib/theme.ts` (React) · `mi_agent_pptx/pptx_theme.py` (deck) · `mi_agent/mi_chart_factory.DEFAULT_THEME` (chat). React/deck values match today; the chart factory's do **not** (light paper, Calibri, different categorical hues, inverted sequential ramp). |
| Design tokens (surfaces, lines, ink) | Yes in CSS (`src/index.css` `--surface-dashboard`, `--color-line`, …); **copied by value** into `pptx_theme.py:42-53` | duplicated |
| Typography | Inter + fallbacks, restated in both | `theme.ts` / `pptx_theme.py:87-91` / `render.py:27-32` |
| Sequential (heatmap) scale | **Three different ones**: `plotlyTheme.TRAKT_SEQUENTIAL` (`#11162e→#3d4a82→peri`), `pptx_theme.sequential` (same three), and `UkChoropleth`'s "cool→warm" ramp (`#1a2340,#2f4a94,#4f74d6,#37b9a6,#e8b13c`) | divergent |
| Number formatting | **Three implementations**: `lib/utils.formatGBP` · `mi_agent_pptx/metric_resolver.compact_currency` · `mi_chart_factory.compact_currency`. React and the deck agree on thresholds (K/MM/BN); the chart factory does not (`k` only from £100k, `m` lower-case, no `BN`). | divergent |
| Currency | **Governed service exists** (`mi_agent_api/currency.py`, request-scoped ContextVar, `resolve_currency_code`), applied only by the API request path (`datasets.py:873-887`, called from `app.py:96` and `mi_service.py:1227`). React honours it via `setDisplayCurrency`. **The deck never calls it and hard-codes `£`.** | **divergent — see §4 RED-1** |
| Bucket ordering | **Four implementations**: `analytics_lib.stratify` (payload: balance desc) · `lib/stratOrder.sortStratBars` (React: ordinal ascending, Unknown last) · `mi_chart_factory._bucket_sort_key` · the deck (none — payload order as-is). | **divergent — see §4 RED-2** |
| Label hygiene | `lib/stratOrder.cleanBucketLabel` (`"2008.0" → "2008"`) — React only. | divergent |
| Chart components | React: `pipeline/bits.BarList` + `StatTile`, `EvolutionPanel.EvoLineChart`, `UkChoropleth`, `artifacts/{Chart,Heatmap,Treemap,Risk,Plotly}ArtifactView`. Deck: `render.draw_{barlist,lines,bars_with_line,bubble,heatmap,diverging,utilisation_tests,table}` + `render_bridge_waterfall`. | parallel implementations |
| Cards / tiles / spacing | Tailwind classes per component (React); an explicit content band `CONTENT_L=0.55 / CONTENT_R=12.78` (deck, `deck.py:244-246`). | not shared, but internally disciplined |

**Verdict on centralisation:** the *palette* is centralised within each surface and
duplicated across them. The *presentation grammar* (order, format, label hygiene,
currency) is not centralised at all — it is per-surface, and in React's case it
lives in components downstream of the API payload, which is precisely what makes it
un-shareable today.

---

## 4. Current Dashboard/PPTX parity

### A. Semantic parity — **strong**

Metric names, definitions and stratification bands come from the same governed
services. The deck imports the governed `Insight` contract rather than inventing a
vocabulary (`insights.py:28-32`). Concentration statuses are normalised onto the
approved vocabulary and the legacy monitor is explicitly disclosed as *not*
operator-approved (`concentration.py:44-47`). Movement language is deliberately
conservative — "reduction", not "redemption" (`movement.py:19-23`).

One asymmetry: three deck stratification dimensions (**broker**, **borrower type**,
**ticket size**) do not exist in the dashboard's funded snapshot at all, and five
dashboard dimensions (**rate, product, vintage, status, protected equity**) never
appear in the deck. Same words, different sets.

### B. Data parity — **strong, and tested**

`tests/mi_agent_pptx/test_channel_parity.py` drives the real React HTTP routes and
the real deck build over one fixture book and compares underlying values, not
formatted strings. Run for this review: **15 passed, 1 skipped.** It asserts
headline funded figures, portfolio-context totals, stratifications bar-for-bar,
funded evolution series, cohorts, geo exposure, concentration evaluations,
reporting date, scope narrowing, vintage table, cohort basis, static-pool
progression, and WA-LTV unit.

Two gaps in the guard itself:
- the stratification test compares only **`set(theirs) & set(mine)`**
  (`test_channel_parity.py:180-182`), so the three PPTX-only dimensions are outside
  the guard by construction;
- `test_the_deck_layer_owns_no_economic_calculation` (`:289`) is a **grep over
  module text** for governed function names. It cannot detect
  `mi_api._ticket_series`'s hard-coded bins or `mi_api._matrix`, and does not.

### C. Visual parity — **close in intent, divergent in three provable places**

Same navy/periwinkle, same tile grammar, same BarList idiom, same 16:9 dark
surface. Divergences below.

### D. Implementation parity — **weak. This is the finding.**

There is one shared definition underneath the *numbers* and **no shared definition
underneath the presentation**. Colour tokens are copied by value; ordering,
formatting and currency are re-decided per surface.

### Classification of every existing PPTX visual

| Slide / visual | Class | Why |
|---|---|---|
| Funded KPI tiles | 🟠 AMBER | Same tile objects, independently styled |
| Portfolio composition | 🟠 AMBER | Same per-type snapshots, deck-specific layout |
| Movement & drivers waterfall | 🟠 AMBER | Same `funded_bridge`; no React panel to match |
| Stratifications — **ltv, age, region** | 🔴 **RED-2** | Same payload, **different display order** (see below) |
| Stratifications — **broker, borrower_type, ticket** | 🔴 **RED-3** | Dimension computed only in the deck |
| Multi-dimensional heatmaps | 🔴 **RED-4** | Cross-tab computed only in the deck |
| Geography | 🟠 AMBER | Same data; choropleth vs bar list; `top5/total` computed in deck |
| Funded evolution | 🟠 AMBER | Same series; deck shows 2 of React's 4 |
| Vintage formation / cohort progression | 🟢 GREEN | Same governed calls, same basis, guarded by tests |
| Pipeline overview | 🟠 AMBER | Same snapshot; `avg_case` divided in deck; fewer panels |
| Pipeline evolution / origination flow | 🟠 AMBER | Same series, different renderer |
| Origination funnel | 🟠 AMBER | Same payload; conversion block dropped |
| Forecast bridge | 🟠 AMBER | Same bridge; monthly split is presentational |
| Forecast projection | 🟢 GREEN | Straight render of `build_extrapolation` |
| Concentration & headroom | 🟢 GREEN | Governed evaluator end-to-end |
| Risk limits (legacy) | 🟢 GREEN | Governed, and disclosed as legacy |
| Executive summary / watchlist | 🟠 AMBER | Governed figures, deck-only prose generators |
| **All money on every slide** | 🔴 **RED-1** | Currency symbol hard-coded |

### The RED items, proved

**RED-1 — the deck hard-codes `£` and ignores the governed reporting currency.**
`mi_agent_api/currency.py` is the governed authority ("every surface — dashboard,
MI Query Agent, Copilot/Teams, **decks** — reads the one request-scoped code
resolved here"). It is applied only by `datasets._apply_request_currency`, called
from `app.py:96` and `mi_service.py:1227`. `mi_agent_pptx` never calls it, and
`metric_resolver.compact_currency(value, symbol="£")` (`:46`) defaults the symbol at
every one of its ~20 call sites. Executed probe:

```
governed current_symbol():                €
deck compact_currency(124_600_000):       £124.6MM
```

A EUR book renders `€124.6MM` in the dashboard and `£124.6MM` in the investor pack.
Severity: high — a wrong currency symbol on an investor deck is a distribution
incident, not a cosmetic one. Fix: LOW effort.

**RED-2 — stratification bars are ordered differently in the two channels.**
`analytics_lib/stratify.py:152-156` sorts every stratification by `balance_sum`
**descending**. React then re-sorts ordinal dimensions into natural bucket order
(`lib/stratOrder.sortStratBars`, `FundedSnapshotPanel.tsx:195`). The deck renders
`st["bars"]` untouched (`deck.py:875-877` → `render.draw_barlist`, which neither
sorts nor cleans labels). Executed probe on the committed parity fixture:

```
ticket: 200-300k | 300-500k | 100-150k     ← deck order (payload / balance desc)
        100-150k | 200-300k | 300-500k     ← React order (sortStratBars)
```

This affects every ordinal dimension (LTV, age, rate, vintage, ticket, time-on-book)
whenever balance rank ≠ bucket rank — the normal case. React also applies
`cleanBucketLabel`, so a vintage bar reads `2008` in the dashboard and `2008.0` in
the deck. Fix: LOW effort, but it must be fixed **in the payload**, not by copying
`sortStratBars` into Python (that would be a fifth implementation).

**RED-3 — three deck stratifications are computed in the PPTX layer.**
`mi_api._extra_stratifications` (`:508-520`) adds broker / borrower-type / ticket by
picking a column from a deck-local candidate list (`_broker_series` `:459`,
`_borrower_type_series` `:435`, `_ticket_series` `:446`) and stratifying itself
(`_stratify_dim` `:485`, which re-sorts by balance descending and truncates to 12).
`_ticket_series` carries **its own bucket edges**, contradicting the governed
registry. Executed probe:

```
governed balance_band labels (config/mi/buckets.yaml:124-137):
  ['<50k','50-100k','100-150k','150-200k','200-300k','300-500k','500k-1m','>=1m']
deck _ticket_series fallback:
  bins   = [0, 100_000, 150_000, 200_000, 250_000, 300_000, 400_000, 1e12]
  labels = ["<£100K","£100–150K","£150–200K","£200–250K","£250–300K","£300–400K","£400K+"]
```

**Mitigating fact, stated precisely:** `_ticket_series` prefers an existing
`ticket_bucket` column, and `funded_prep` materialises exactly that column from the
governed `balance_band` (`funded_prep.py:60`, `analytics_lib/buckets.materialise_buckets`).
The probe above confirms the deck's live labels are the governed ones
(`200-300k`, not `£200–250K`). So the ungoverned bins are **latent, not active** on
a prepared tape — they would fire silently on any frame that skipped or failed the
prep. They are still a second economic definition sitting in the renderer, and they
are outside the parity test. All three dimensions **are** in the governed
stratification catalogue (`config/mi/stratification_catalogue.yaml`:
`broker_channel`, `borrower_structure`, `balance_band`) — they are simply not in
`snapshots._STRAT_DIMS`. The correct fix is to add them there, not to keep them in
the deck.

**RED-4 — the multi-dimensional cross-tab exists only in the deck.**
`mi_api._matrix` / `_multidim` (`:523-559`) build LTV×Age, LTV×BorrowerType and
LTV×Region balance matrices in the PPTX layer. The bands come from governed
services (`cohorts._dimension_series`) and only balances are summed, so it is close
to presentation arithmetic — but the *analysis definition* (which pairs, which
ordering, what "Unknown" does) lives nowhere else, and the dashboard cannot show
the same chart because there is no endpoint for it. Fix: promote to
`mi_agent_api` as a `/mi/multidim` payload; LOW–MEDIUM.

**Target state: zero RED. All four are achievable without new analysis.**

---

## 5. Proposed-scope capability matrix

Classes: 1 already in PPTX · 2 dashboard has it, PPTX doesn't · 3 MI has it,
neither surface shows it · 4 exists but needs a better visual · 5 small composition
of existing primitives · 6 genuinely new MI · 7 not meaningful until seasoned ·
8 not worth adding.

### 1. Executive dashboard slide

| Item | Class | Underlying MI | Source | React | PPTX today | Effort | Recommendation |
|---|---|---|---|---|---|---|---|
| Funded headline tiles | **1** | `compute_funded_snapshot.kpis` | `snapshots.py:613` | `FundedSnapshotPanel` | `slide_kpi_summary` (`deck.py:397`) | — | Keep |
| Pipeline headline tiles | **2** | `compute_pipeline_snapshot` | `pipeline_contract.py:876` | `PipelineSnapshotPanel` | on slide 14 only | LOW | Promote 2 tiles to slide 1 |
| Weighted-average / expected forecast | **2** | `forecast_bridge.compute_forecast_bridge` | `forecast_bridge.py:425-437` | `ForecastView` | on slide 18 only | LOW | Promote 1 tile |
| Headline risk / concentration | **2** | `compute_concentration_tests` (worst utilisation, headroom, breach horizon) | `concentration_tests_api.py:340`, `forward.py:448` | `RiskLimitsWorkspace` | on slide 21 only | LOW | Promote 1 tile + RAG chip |
| **A single mixed pipeline+funded+forecast+risk tile row** | **4** | all four above, already resolved on one `DashboardData` | `mi_api.py:589` | **does not exist in React either** | does not exist | LOW–MED | Build it; see §11 |
| Sparklines on tiles | **5** | `funded_evolution.periods[].metrics` already resolved | `evolution.py:251` | — | `render.draw_lines` exists | MED | Only where ≥3 periods |

### 2. Pipeline slides

| Item | Class | Underlying MI | Source | React | PPTX today | Effort | Recommendation |
|---|---|---|---|---|---|---|---|
| 2A Pipeline stratification 2×2 — **region, broker, stage, expected-completion month** | **1** | `_dimension_breakdown` | `pipeline_contract.py:787,812,715,872-874` | `PipelineSnapshotPanel` | slide 14 (2 of 4) | LOW | Make it a true 2×2 |
| 2A Pipeline stratification — **LTV, borrower age, ticket, rate** | **3** | `analytics_lib.buckets` already materialises `ltv_bucket / age_bucket / ticket_bucket / interest_rate_bucket` **on the prepared pipeline frame** | `pipeline_prep.py:602-608`; contract `pipeline_field_contract.yaml:280` `reused: [ltv_bucket, age_bucket, ticket_bucket, interest_rate_bucket]`; reported in `dimensions_available` (`pipeline_prep.py:776`) | **no** | **no** | LOW | **Highest-value cheap win.** One `_dimension_breakdown` call per dimension, added to `compute_pipeline_snapshot`; both surfaces gain it at once. Not new MI. |
| 2B Pipeline evolution matrix | **1** | `evolution.pipeline_evolution` | `evolution.py:701` | `EvolutionPanel` | slide 15 | — | Keep; add the count series React shows |
| 2C Pipeline conversion over time | **2** | `pipeline_funnel_evolution` — cohort conversion, weekly velocity, lag, sufficiency | `evolution.py:934-1120` | `EvolutionPanel` conversion disclosure | **dropped**; slide 16 renders only latest weekly flow | LOW | Add a conversion line chart. The payload is already on `DashboardData.funnel`. |
| 2D Balance by LTV × borrower age | **1** | `mi_api._multidim` | `mi_api.py:546` | no | slide 9 | — | Keep, but promote the cross-tab to the API (RED-4) |
| 2D Balance by region × LTV | **1** | `mi_api._multidim` | `mi_api.py:557` | no | slide 9 | — | Keep |
| Bubble variant | **8** | — | — | — | `render.draw_bubble` exists but is no longer used | — | **Do not reinstate.** `deck.py:1356-1364` already records the reasoning: bubble area encodes value while position encodes nothing. Heatmap is correct. |

### 3. Funded slides

| Item | Class | Underlying MI | Source | React | PPTX today | Effort | Recommendation |
|---|---|---|---|---|---|---|---|
| 3A Funded stratification matrix | **1 / 4** | `_funded_stratifications` (8 dims) | `snapshots.py:504` | `FundedSnapshotPanel` | 3 slides × 2 | LOW | Compress to **one 2×2** on the governed dims; fix RED-2 ordering |
| — rate / product / vintage / status / equity dims | **2** | same service | `snapshots.py:333-342` | yes | **no** | LOW | Make the 2×2 configurable; offer as deep dive |
| 3B Funded evolution matrix — balance, count, WA LTV, WA rate | **2** | `evolution.funded_evolution` | `evolution.py:251` | 4 charts | **2 charts** | LOW | Add the missing 2 → a true 2×2 |
| 3B Funded balance by region, stacked over time | **6 (small)** | `funded_bridge` gives two-point-by-dimension; `funded_frames` gives the frames | `evolution.py:196,324` | no | no | MED | **Defer.** A per-period × per-dimension series is a genuinely new query shape (`analytics_lib/history.py:10-16` says exactly this). Low executive value versus the bridge you already have. |
| 3C Funded deep dives — heatmaps | **1** | `mi_api._multidim` | — | no | slide 9 | — | Keep |
| 3C Treemap | **8** | `mi_chart_factory._build_treemap` (chat only) | `mi_chart_factory.py:696` | `TreemapArtifactView` | **no renderer** | MED | **Do not add.** See §8. |
| 3D Cohort / vintage composition | **1** | `cohorts.cohort_analysis` | `cohorts.py` | `CohortsPanel` | slide 12 | — | Keep |
| 3D Cohort progression / seasoning | **1 / 7** | `evolution.funded_cohort_progression` | `evolution.py:526` | `EvolutionPanel` cohort mode | slide 13, gated on `has_cohort_progression` | — | Keep; already correctly gated |
| 3D Quarterly originations split by year | **5** | `cohort_analysis(grain="Q")` — grain is already a parameter | `mi_api.py:917-921` (deck currently pins `grain="Y"`) | grain selector exists in React | Y only | LOW | Make grain configurable in the deck YAML |

### 4. Forecast slides

| Item | Class | Underlying MI | Source | React | PPTX today | Effort | Recommendation |
|---|---|---|---|---|---|---|---|
| 4A Expected funded evolution from conversion | **1** | `compute_forecast_bridge` + `forecast_breakdowns` | `forecast_bridge.py`, `workspace.py:312` | `ForecastView` | slide 18 | — | Keep |
| 4A Forecast by region / by LTV bucket | **2** | `workspace.forecast_breakdowns` — **already resolved onto `DashboardData.forecast["forecastBreakdowns"]` and then discarded** | `mi_api.py:877-880`; deck uses only `byCompletionMonth` (`deck.py:1454-1457`) | `ForecastView.tsx:57-67` | resolved, not drawn | **LOW** | **Free win.** Two BarLists, zero MI cost. |
| 4B Time to £100m | **1** | `_THRESHOLDS = [25m, 50m, 75m, 100m, 150m]` + milestone dates per scenario | `forecast_extrapolation.py:31,162-190` | `ForecastExtrapolationPanel` | slide 19 | — | Keep |
| 4B Time to **£200m** / client-configured targets | **5** | `build_extrapolation(extra_thresholds=…)` already accepts extra targets (`:362`); the ladder itself is a module constant, not client config | `forecast_extrapolation.py:31,415` | — | not passed | LOW | Move the ladder to client config and thread it from the deck YAML. Not new MI. |
| 4C Actual vs prior-run forecast (calibration) | **2** | `evolution.forecast_evolution` periods | `evolution.py:1129` | `EvolutionPanel.tsx:909-918` — **derived in the browser** (lag-1 shift) | handler exists (`deck.py:1526`) but **no slide is configured** | LOW | Add the slide **and** move the lag-1 shift server-side so both surfaces read one series. |

### 5. Concentration / risk slide

| Item | Class | Underlying MI | Source | React | PPTX today | Effort | Recommendation |
|---|---|---|---|---|---|---|---|
| Current / Expected / Stress, limit, headroom, RAG | **1** | `compute_concentration_tests` + `forward.evaluate_forward_states` | `concentration_tests_api.py:340`, `forward.py:250` | `RiskLimitsWorkspace` | slide 21 | — | Keep. This is the deck's strongest slide. |
| Forecast breach horizon | **1** | `forward.expected_breach_horizon` | `forward.py:448` | yes | `concentration.py:97`, `deck.py:1830` | — | Keep |
| Emerging risks | **2** | `forward.identify_emerging_risks` | `forward.py:517` | yes | **no** | LOW | Add to the watchlist slide |
| Concentration movement over time | **2** | `concentration_tests_api.compute_history` | `app.py:1774` | `ConcentrationDetailPanel` | **no** | LOW–MED | Add a sparkline per test |
| Pipeline drivers of the expected move | **2** | `forward.compute_drivers_for_test` | `forward.py:689` | yes | **no** | MED | Optional deep dive only |

**Nothing in the proposed scope is class 6 (genuinely new MI).** The two closest —
stacked funded balance by region over time, and a client-configurable target ladder
— are respectively a deferrable new query shape and a config move.

---

## 6. Existing MI capability omitted from the proposed scope

| Capability | Status | Implementation | Surfaced today? |
|---|---|---|---|
| Balance movement / change attribution by dimension | **SUPPORTED NOW** | `evolution.funded_bridge` (`evolution.py:324`) — per-category deltas sum exactly to close − open, with a reconciling "Other" | Deck slide 4; React via `/mi/insight/movement-detail` |
| **Opening → new → exited → continuing → closing balance bridge** | **SUPPORTED NOW** | `mi_agent/period_change/bridge.py` — loan-identity based, reconciliation *checked* not asserted, composite key with `source_portfolio_id`, explicit refusal on missing/duplicate ids | **Neither.** Chat only, via `period_change_route` (`chat_routing.py:3648`). No React panel references `periodChange`. |
| **Distribution movement between periods** | **SUPPORTED NOW** | `mi_agent/period_change/distribution.py` — per-category count share + balance share movement, ranked top ±5 (`ranking.py`) | **Neither.** Chat only. |
| Actual vs prior forecast | **SUPPORTED NOW** (as a lag-1 read of `forecast_evolution`) | `evolution.forecast_evolution:1129`; shift done in React at `EvolutionPanel.tsx:909-918` | React only; **deck handler exists but no slide is configured** |
| Forecast calibration / accuracy statistics | **PARTIALLY SUPPORTED** | The two series exist; no error metric (MAE/bias/hit-rate) is computed anywhere. `forecast_extrapolation.py:346` marks one model `withdrawn_pending_calibration` — a status, not a calibration. | no |
| Conversion by vintage / period / dimension | **SUPPORTED NOW** for period; **PARTIAL** for dimension | `evolution.pipeline_funnel_evolution:934-1120` — cohort conversion, weekly velocity, lag-adjusted, sufficiency-gated. No by-dimension cut. | React funnel; **deck drops it** |
| Concentration movement | **SUPPORTED NOW** | `/mi/concentration-tests/history` → `compute_history` (`app.py:1774`) | React only |
| Concentration headroom | **SUPPORTED NOW** | `concentration_tests_api.py:294` | Both |
| Forecast concentration (Expected + all-pipeline stress) | **SUPPORTED NOW** | `mi_agent/concentration_tests/forward.py:250,619` | Both |
| Emerging-risk / exception analysis | **SUPPORTED NOW** | `forward.identify_emerging_risks:517`, `expected_breach_horizon:448` | React; deck carries horizon only |
| Largest deteriorating concentration | **SUPPORTED NOW** | `forward.identify_emerging_risks` + `compute_history` | React |
| Bucket / LTV migration (transition matrix) | **PARTIALLY SUPPORTED** | `mi_agent/risk_monitor/migration.migration_matrix:78` + `per_loan_movement:156` are real. `analytics_lib/migration.py` is an explicit **stub that raises `NotImplementedError`** (`:26-33`). The working implementation is wired to the **agent tool surface** (`trakt_tools/handlers/history.py:667`), not to `/mi/*` or React. | Agent tools only |
| Cohort / vintage performance, seasoning | **SUPPORTED NOW** | `evolution.funded_cohort_progression:526`; governed seasoning axis in `mi_agent/seasoning.py` + `config/mi/buckets.yaml` `seasoning:` block (front/back book, lending windows, month bands) | Deck slides 12–13; React Cohorts |
| Redemption / run-off | **PARTIALLY SUPPORTED** | Three distinct things exist: (a) the **"Exited / redeemed loans"** KPI, loan-identity based (`snapshots.py:725-731`); (b) `forecast_bridge` **`balanceRetentionFactor`** (`:324`); (c) a full observed **`prepayment_rate` / SMM-CPR** in `analytics_lib/history.py:353`, plus `default_rate:871`, `cure_rate:1017`, `loss_and_recovery:562`, `classify_exits:191`, `portfolio_series:111` — all wired to `trakt_tools`, **not** to `/mi/*` or React. | KPI in both; the rate library in agent tools only |
| Borrower-age migration | **NOT SUPPORTED** as a migration; the age *band* is governed (`age_bucket`) and `risk_monitor.migration` could take it, but nothing calls it that way | — | no |
| Valuation / HPI-driven movement | **PARTIALLY SUPPORTED** | `analytics_lib/valuation_age.valuation_age_profile:92` measures valuation **staleness and method** — explicitly a data-quality finding, not an HPI revaluation. There is no HPI index anywhere. | Agent tools only |
| Top movers | **SUPPORTED NOW** | `period_change/ranking.rank_movement`; `funded_bridge` top-N; `movement_detail` components (new / removed / progressed_out / increased / decreased / unchanged) | Chat + React drawer |
| Funding / securitisation target trajectory | **SUPPORTED NOW** | `forecast_extrapolation` milestone ladder | Both |
| Contractual WAL / amortisation cashflows | **SUPPORTED NOW** | `analytics_lib/contractual.py` (refuses where the contract stops determining the answer) | Agent tools only |
| Weekly pipeline movement attribution | **SUPPORTED NOW** | `movement_detail.py` — mutually exclusive, exhaustive components summing to the headline | React drawer |

### Which of these actually belong in the deck

Only three earn a place, and the bar is "an investor asks this on every call":

1. **Actual vs prior-run forecast** (forecast credibility). Class 2, LOW.
2. **Conversion rate over time** (the single best leading indicator for a growing
   book). Class 2, LOW.
3. **Concentration movement** — a sparkline of each approved test's utilisation
   across snapshots, beside the current/expected/stress table. Class 2, LOW–MED.

**Deliberately excluded, with reasons:**
- *Period-change balance bridge* — genuinely excellent and genuinely governed, but
  it answers the same executive question as the movement-drivers waterfall the deck
  already carries, and adding both invites a reader to reconcile two bridges. Revisit
  only if the loan-identity bridge is chosen to *replace* the dimensional one.
- *Distribution shift / bucket migration* — a strong seasoned-book slide, but the
  ranked-shift output is dense and there is no React panel to keep it honest.
  Post-go-live, and put it in React first.
- *CPR / default / cure / loss rates* — these are **arrears- and cashflow-shaped**
  and live on the agent tool surface. Surfacing them in an investor deck for an
  equity-release book without first proving field coverage would be a
  credibility risk, not a win. Post-go-live, gated on data coverage.
- *Valuation age profile* — a diligence exhibit, not an investor-deck exhibit.

---

## 7. New vs seasoned portfolio

### What the system already knows about history

**A conditional composition engine already exists.** `mi_agent_pptx/composition.py`:
`build_facts` (`:126-183`) derives 24 governed facts from the resolved payloads;
`evaluate_condition` (`:68`) evaluates the `when:` expression as a restricted AST
walk (names, and/or/not, comparisons, `in` — never `eval` of arbitrary code); a
per-type `will_render` guard then checks the actual payload; every drop is recorded
as a `SlideOmission` with an investor-facing reason. **There is no
`seasoned=true/false` flag anywhere, and none is needed.**

Facts available today:

`scope · is_total · portfolio_count · type_count · has_direct · has_acquired ·
is_mixed · mixed_reporting_dates · has_funded · has_stratifications · has_movement ·
has_attribution · has_funded_history · has_geo · has_cohorts ·
has_cohort_progression · has_multidim · has_pipeline · has_pipeline_history ·
has_funnel · has_forecast · has_forecast_projection · has_forecast_history ·
has_risk · has_concentration · has_concentration_forward`

The sufficiency rules that matter are already enforced:

- **`has_funded_history`** = `≥2` funded periods and not `singlePeriod`
  (`composition.py:99-104`).
- **`has_cohort_progression`** = a cohort holds loans in **≥2 reporting periods**,
  asked of the cohort adapter itself so the guard and the slide can't disagree
  (`composition.py:107-121`, `cohorts.progression_is_meaningful:369`). The YAML
  comment states the reason: "joining one point into a line would be the misleading
  trend the sufficiency rules exist to prevent."
- **Cohort selection**: ≥2% of book share, max 4 series
  (`cohorts.py:53-56`, `select_cohorts:324`).
- **Conversion sufficiency** is enforced upstream in the MI engine — velocity is
  marked provisional below a minimum week count (`evolution.py:859`,
  `EvolutionPanel.tsx:323-327`).

### The minimum sensible additions

Only three facts are missing, and all three are read-only reads of already-resolved
payloads:

| New fact | Derivation | Enables |
|---|---|---|
| `funded_periods: int` | `len(funded_evolution["periods"])` | `when: funded_periods >= 4` for cohort/vintage depth; `>= 6` for a distribution-shift module |
| `funded_balance: float` | `funded["kpis"]` → `balance.raw` | `when: funded_balance >= 50_000_000` to expand the risk section |
| `pipeline_share: float` | `pipeline.pipelineAmount / (funded_balance + pipelineAmount)` | `when: pipeline_share >= 0.10` to keep the pipeline/forecast block prominent on a young book |

Plus one optional: `forecast_history_periods: int`, to gate a
forecast-vs-actual slide at `>= 3` rather than `>= 2` (two points is a line, not a
track record).

**Recommendation: do not build a "seasoned deck" and a "new-book deck". Build one
deck config with `when:` expressions over these facts.** The engine already
guarantees the omission is explained rather than silent, which is what makes a
conditional pack safe to hand to an investor.

**Explicitly do not add** a `seasoned` boolean. It would be a manually maintained
fact that can disagree with the data, which is exactly the failure mode the
fact-derived design avoids.

---

## 8. Multidimensional visual capability

| Visual | MI can produce the dataset? | Rendered anywhere? | Chart lib support? | PPTX renders faithfully? | Same spec both sides? | Simpler alternative better? |
|---|---|---|---|---|---|---|
| **LTV × borrower age (heatmap)** | Yes — `mi_api._matrix` over `cohorts._dimension_series` bands | Deck slide 9 only | React `HeatmapArtifactView` (native grid) + `mi_chart_factory._build_heatmap`; deck `render.draw_heatmap` | **Yes.** Contrast-flipping cell values, exact panel dimensions, theme sequential ramp | **Not today** — no `/mi/multidim` endpoint (RED-4) | No — a cross-tab is the right shape |
| **LTV × region (heatmap)** | Yes | Deck slide 9 | as above | Yes, and it takes full width because region labels are long (`deck.py:1384-1392`) | Not today | No |
| **LTV × borrower type (heatmap)** | Yes | Deck slide 9 | as above | Yes | Not today | **Yes** — 2 columns × N LTV bands is a grouped bar, not a heatmap. Consider demoting. |
| **Bubble** | Yes (`points` projection already produced) | **Nowhere** — deliberately removed | `render.draw_bubble` still exists; React `ChartArtifactView` handles `bubble` | Yes | n/a | **Yes.** `deck.py:1356-1364`: bubble area encodes value, position encodes nothing; the reader compares circles by eye where a heatmap lets them read the number. **Do not reinstate.** |
| **Treemap** | Yes for a single hierarchy (`mi_chart_factory._build_treemap:696`, requires hierarchy cols + value) | React `TreemapArtifactView` (chat artifacts only) | Recharts Treemap in React; **no PPTX renderer** | **No** — would need a new matplotlib renderer, and treemaps degrade badly at slide scale with UK region names | No | **Yes.** A BarList already ranks the same magnitudes and is readable at 6in. **Do not add.** |
| **Choropleth** | Yes — `geo.exposure_by_itl3` + `uk_itl3_paths.json` atlas | React `UkChoropleth` | React only | Would need SVG-path rendering in matplotlib | Data yes, visual no | **Judgement call.** A static map is a strong investor visual and the atlas is already in the repo. MEDIUM effort. Optional, not core. |

**Recommendation:** keep exactly two multidimensional visuals — **LTV × borrower
age** and **LTV × region** — both as heatmaps on one colour-scale methodology, and
promote the cross-tab to a governed `/mi/multidim` payload so the dashboard can show
the identical chart. Drop LTV × borrower type to a grouped bar or cut it. Add no
bubble, no treemap.

---

## 9. Recommended shared UX architecture

### Ranking the four options against what actually exists

| Option | Verdict | Why |
|---|---|---|
| **A. PPTX captures/renders the actual React visual** | **AVOID** | Requires a browser in the deck path. The deck runs as an Azure Functions stage (`pptx_stage.py`) and the whole renderer was deliberately built to avoid Chrome/kaleido (`README.md:144-146`). It would also make deck generation depend on a running frontend and an authenticated session, converting a deterministic batch artefact into a distributed-system failure mode. The `blocked / failed` states in `DeckDownloadMenu` exist because determinism matters here. |
| **B. Shared chart *specification*** | **ACCEPTABLE, but not first** | Attractive in principle. In practice the three renderers have genuinely different capability envelopes (Recharts vs matplotlib vs Plotly), so a spec rich enough for all three becomes a fourth abstraction to maintain. Worth adopting **only for the narrow presentation contract in C+**, not for full chart specs. |
| **C. Same governed chart-data payload, separate renderers** | **BEST — and it is already 80% built** | This is exactly what `mi_agent_pptx/mi_api.py` does, and `test_channel_parity.py` proves it holds for numbers. The gap is that the payload stops at *values* and leaves *presentation grammar* to each renderer. |
| **D. PPTX independently rebuilds charts** | **AVOID** | This is what produced RED-1 through RED-4. |

### The recommendation: **C+**

> **Define the analysis once. Define its presentation grammar once, in the same
> payload. Render it in React. Render the same payload in PowerPoint.**

Concretely, extend the governed payloads so that each chart-bearing block carries
its own presentation contract, and make both renderers consume it rather than
decide for themselves:

| Field | Today | Under C+ |
|---|---|---|
| `bars[].label` | raw (`"2008.0"`) | cleaned server-side once (`cleanBucketLabel` logic moves into `analytics_lib.stratify`) |
| `displayOrder` | absent — React re-sorts, deck doesn't | emitted with the payload; **both renderers iterate in payload order** |
| `ordinal: bool` | absent | declared from the bucket registry, not sniffed from labels |
| `valueFormat` | implicit | `"gbp" \| "pct" \| "count" \| "decimal"` — already a React concept, promote it |
| `currency` | React reads it from the envelope; deck ignores it | on the envelope, and **both** money formatters read it |
| `seriesColorRole` | hard-coded per component/renderer | `"primary" \| "secondary" \| "positive" \| "negative" \| "neutral"` — the palette stays per-surface, the **role** is shared |
| `ragStatus` | already governed | keep |
| `title` / `subtitle` / `footnote` | per surface | governed for the metric, decorated per surface |
| `reportingDate` / `scope` / `lineage` | already on the payloads | keep |

Then **one shared token file per surface, generated, not hand-copied.** Today
`pptx_theme.py:42-96` is a by-value copy of `index.css` + `theme.ts`. Emit the token
set from one source (a small JSON) and generate both `theme.ts` and `pptx_theme.py`
from it in CI, with a test that fails on divergence. That removes the whole class of
"the values match today" risk.

### What must be shared, and what may differ

**Shared (one definition, consumed by both):**
chart data · display order · bucket definitions and labels · metric definition ·
series labels and colour *roles* · number format · currency · RAG semantics ·
footnotes and lineage · reporting period · scope and constituent books · client
branding inputs · data version / run id.

**Appropriately different (PowerPoint is static):**
aspect ratio and panel geometry (the deck's fixed 13.33×7.5in content band) ·
annotation density (the deck must label values on the mark because there is no
hover) · axis-label thinning and label truncation · legend placement · slide
composition and how many panels share a page · explanatory prose and the
methodology page · interaction affordances (drill-through, measure toggles,
grain selectors) — these have **no** static equivalent and should be replaced by an
editorial choice recorded in the deck config, not silently dropped.

### One structural consequence worth stating

The three `compact_currency` implementations and four bucket sorters are not a
tidiness problem; they are the *mechanism* by which the surfaces drift. C+ removes
the mechanism. Everything else in this review is downstream of that.

---

## 10. Proposed final deck

Principles: executive length, one question per slide, nothing that cannot be
sourced from a governed payload, and every omission explained by the existing
composition ledger. **CORE** = always. **CONDITIONAL** = `when:` rule. **DEEP DIVE**
= available, not standard.

### A. New / non-seasoned portfolio — 11 slides

| # | Title | Business question | Visual(s) | Existing capability | Dashboard equivalent | Inclusion rule |
|---|---|---|---|---|---|---|
| 1 | Executive Position | Where is the book today, and what is coming? | Mixed tile row + 2 sparklines | `compute_funded_snapshot`, `compute_pipeline_snapshot`, `forecast_bridge`, `compute_concentration_tests` | none (new composition, §11) | CORE |
| 2 | Executive Summary | What changed this period? | Governed observations | `mi_agent_pptx/insights.py` | Weekly Brief (flag-gated) | CORE |
| 3 | Portfolio Composition | What do I own today? | Stacked bar + per-type cards | per-type `compute_funded_snapshot` | `PortfolioScopeBanner` + snapshot | CORE |
| 4 | Funded Key Measures | How large is the book and what are its characteristics? | 10 KPI tiles | `compute_funded_snapshot.kpis` | `FundedSnapshotPanel` | CORE |
| 5 | Funded Stratifications | How is exposure distributed? | **2×2 BarList** (LTV, ticket, age, region) | `_funded_stratifications` | `FundedSnapshotPanel` | CORE |
| 6 | Geographic Exposure | Where is the collateral? | 4 tiles + top-12 BarList | `geo.exposure_by_itl3` | `GeographyPanel` | `when: has_geo` |
| 7 | Pipeline Overview | What is likely to fund next? | 4 tiles + **2×2** (stage, broker, region, expected-completion month) | `compute_pipeline_snapshot` | `PipelineSnapshotPanel` | `when: has_pipeline` |
| 8 | Origination Funnel & Conversion | How does the pipeline convert, and is it improving? | Stage BarList + **conversion line** | `pipeline_funnel_evolution` | `EvolutionPanel` origination | `when: has_pipeline` |
| 9 | Forecast Bridge | Where is the book heading on the current pipeline? | Waterfall + **forecast by region / by LTV** | `compute_forecast_bridge` + `forecast_breakdowns` | `ForecastView` | `when: has_forecast` |
| 10 | Time to Scale | When do we reach £100m / £200m? | Projection lines + milestone table | `build_extrapolation` | `ForecastExtrapolationPanel` | `when: has_forecast_projection` |
| 11 | Concentration & Headroom | Am I within approved limits, now and on forecast? | Utilisation chart + table (Current / Expected / Stress) | `compute_concentration_tests` | `RiskLimitsWorkspace` | `when: has_concentration` |
| — | Data and Methodology | What does this cover, as at when, how produced? | Text + omission ledger | `mi_api.diagnostics` + composition | — | CORE (mandatory gate) |

Deliberately **absent** on a young book: cohort progression, funded evolution,
vintage depth, movement attribution — the composition engine already suppresses
each, and the methodology page says why.

### B. Seasoned portfolio — 13 slides

| # | Title | Business question | Visual(s) | Existing capability | Dashboard equivalent | Inclusion rule |
|---|---|---|---|---|---|---|
| 1 | Executive Position | — | Mixed tiles + sparklines | as A1 | none | CORE |
| 2 | Executive Summary | What changed? | Observations | `insights.py` | Weekly Brief | CORE |
| 3 | Portfolio Composition | What do I own? | Stacked bar + cards | per-type snapshot | snapshot | CORE |
| 4 | Portfolio Movement & Drivers | Why did funded AuM change? | Waterfall + takeaways | `evolution.funded_bridge` | movement-detail drawer | `when: has_attribution or has_movement` |
| 5 | Funded Key Measures | Characteristics? | KPI tiles | `compute_funded_snapshot` | `FundedSnapshotPanel` | CORE |
| 6 | Funded Stratifications | Distribution? | 2×2 BarList | `_funded_stratifications` | `FundedSnapshotPanel` | CORE |
| 7 | Funded Balance Evolution | How is the book changing? | **2×2**: balance, loan count, WA LTV, WA rate | `evolution.funded_evolution` | `EvolutionPanel` funded | `when: has_funded_history` |
| 8 | Vintage Formation | How much sits in each vintage? | Composition table | `cohorts.cohort_analysis` | `CohortsPanel` | `when: has_cohorts` |
| 9 | Cohort Progression | How have vintages seasoned? | Static-pool lines + change table | `funded_cohort_progression` | `EvolutionPanel` cohort mode | `when: has_cohort_progression` |
| 10 | Multi-Dimensional Risk | Where do risk dimensions concentrate together? | LTV×Age, LTV×Region heatmaps | `_multidim` (→ `/mi/multidim`) | none today | `when: has_multidim` |
| 11 | Geographic Exposure | Where is the collateral? | Tiles + BarList | `exposure_by_itl3` | `GeographyPanel` | `when: has_geo` |
| 12 | Concentration & Headroom **+ movement** | Within limits, and moving which way? | Utilisation + table + **per-test sparkline** | `compute_concentration_tests` + `compute_history` | `RiskLimitsWorkspace` + `ConcentrationDetailPanel` | `when: has_concentration` |
| 13 | Portfolio Health & Watch Items | What needs attention? | ≤5 items, ≤3 positives | `watchlist.py` (+ `identify_emerging_risks`) | none | CORE |
| — | Data and Methodology | — | Text + ledger | — | — | CORE |

### C. Full capability / mixed portfolio — 16 slides

A + B merged, deduplicated, in this order:

1 Executive Position · 2 Executive Summary · 3 Portfolio Composition ·
4 Portfolio Movement & Drivers · 5 Funded Key Measures · 6 Funded Stratifications ·
7 Funded Balance Evolution · 8 Vintage Formation · 9 Cohort Progression ·
10 Multi-Dimensional Risk · 11 Geographic Exposure · 12 Pipeline Overview ·
13 Origination Funnel & Conversion · 14 Forecast Bridge · 15 Time to Scale ·
16 Concentration & Headroom + movement · 17 Portfolio Health & Watch Items ·
Data and Methodology.

(That is 17 + methodology. If 18 pages is too long for the audience, the first cut
is slide 10 → deep dive and slide 8 folded into slide 9.)

### Deep dives — configured, not standard

Pipeline Evolution · Origination Flow (weekly run-rate) · Forecast Evolution &
Actual-vs-Forecast · Concentration Drill-Through and Pipeline Drivers · Funded
Stratifications II (rate, product, status, protected equity) · Legacy Risk Limits ·
Direct vs Acquired comparison (handler already exists, currently unconfigured).

### What to remove from the deck as it ships today

- **Three stratification slides → one 2×2.** Six BarLists over three pages is a data
  dump; the executive question is answered by four.
- **`slide_funnel`'s current-week fallback to case counts** should be labelled as a
  different measure, not silently substituted (`deck.py:1425-1429`) — a reader
  cannot tell weekly *flow* from current *stock* from the chart alone.

---

## 11. Slide 1 recommendation

### What exists to build on

- **Which dashboard headline tiles already exist:** funded only —
  `FundedSnapshotPanel`'s KPI grid (`snapshots.py:613-741`), which the deck already
  reproduces verbatim on `slide_kpi_summary`.
- **There is no React landing page to mirror.** `AppShell` opens on the Funded lens
  of a tabbed Core Dashboard. So slide 1 is not "export the React landing page" — it
  is a genuinely new composition, and it should be **designed for the deck and then
  offered back to React**, not the other way round.
- **Sparklines:** `render.draw_lines` already renders a line series at arbitrary
  panel size and is used on five slides; `funded_evolution.periods[]` is already
  resolved on `DashboardData`. A sparkline is a size and axis-suppression choice, not
  a new renderer.
- **Everything needed is already on one object.** `build_dashboard_data` resolves
  funded, pipeline, forecast, concentration, evolution and insights in a single pass
  (`mi_api.py:589-760`). Slide 1 costs **zero extra MI calls**.

### Can pipeline + funded + forecast + risk fit without overload?

Yes, at **7 tiles + 2 sparklines + 1 RAG strip**, using the deck's existing
`_tile_grid` (5 columns) and `_strip` primitives. Ten numbers is the practical
ceiling for an executive page; the proposal below is nine plus a status strip.

### Proposed composition

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ <Client>  ·  Investor & Funder MI Pack        Reporting date · 30 June 2026   │
│ Scope: Total portfolio (Direct + Acquired) · 2 books                          │
├──────────────────────────────────────────────────────────────────────────────┤
│  FUNDED BALANCE   LOANS FUNDED   WA CURRENT LTV   PIPELINE (WTD EXPECTED)  …  │
│  £124.6MM         1,284          46.3%            £18.2MM                     │
│  ▲ +£3.1MM        ▲ +34          ▼ −0.4pp         ▲ +£1.4MM vs prior wk       │
├──────────────────────────────────────────────────────────────────────────────┤
│  FORECAST FUNDED  TIME TO £100M  TOTAL PIPELINE   CASES        AVG CASE       │
│  £142.8MM         reached        £31.4MM          212          £148K          │
│  funded + wtd     — / Q1 2027                                                 │
├───────────────────────────────┬──────────────────────────────────────────────┤
│  Funded balance — 12 periods  │  Weighted expected pipeline — 12 weeks        │
│  ▁▂▃▃▄▅▅▆▇▇██  (sparkline)    │  ▂▃▂▄▅▄▆▅▇▆██  (sparkline)                    │
├───────────────────────────────┴──────────────────────────────────────────────┤
│  CONCENTRATION  ● 6 within limit  ● 1 approaching  ● 0 breach                 │
│  Tightest headroom: Region — South East, £2.1MM (8.4% of limit)               │
└──────────────────────────────────────────────────────────────────────────────┘
```

Sources, tile by tile — **all class 1 or 2, none new**:

| Tile | Source |
|---|---|
| Funded balance, Loans funded, WA current LTV (+ MoM deltas) | `compute_funded_snapshot.kpis` — reuse the tile objects verbatim, exactly as `slide_kpi_summary` does today |
| Pipeline weighted expected, Total pipeline, Cases, Avg case (+ prior-week deltas) | `compute_pipeline_snapshot` incl. `priorWeek` (`pipeline_contract.py:876-905`) |
| Forecast funded balance | `forecast_bridge.forecastFundedBalance` (`forecast_bridge.py:436`) |
| Time to £100m | `build_extrapolation` milestone ladder (`forecast_extrapolation.py:162-190`) |
| Funded sparkline | `funded_evolution.periods[].metrics.funded_balance` |
| Pipeline sparkline | `pipeline_evolution.periods[].metrics.weighted_expected_funded_amount` |
| Concentration strip | `compute_concentration_tests` summary + `headroom` (`concentration_tests_api.py:294`) |

### Behaviour under thin data

Every tile must degrade the way the deck already degrades: `_tile` renders `—` when
`available` is false or the value is empty (`deck.py:195-199,214`). Sparklines follow
the existing ≥2-period rule (`_evolution_lines`, `deck.py:936`) and are simply
**omitted** — not replaced by a placeholder — when history is short, since slide 1
must never look broken. The concentration strip is dropped when
`has_concentration` is false, and the tile row reflows from 5 columns to 4.

### How closely it can mirror React

It cannot mirror what does not exist. The honest sequencing is: **build this
composition once, in the payload, and render it on both surfaces** — a
`/mi/executive-summary` block returning exactly these tiles plus the two series,
consumed by a new React landing card and by slide 1. That makes slide 1 the first
thing built under the C+ architecture rather than the last thing bolted on.

---

## 12. Implementation gap

### A. PPTX UX / layout only

| Work | Effort |
|---|---|
| Slide 1 executive composition (tiles + sparklines + RAG strip) | MED |
| Collapse 3 stratification slides → one 2×2 | LOW |
| Funded evolution 2 charts → 2×2 (add loan count, WA rate) | LOW |
| Draw the already-resolved `forecastBreakdowns.byRegion` / `byLtvBucket` | **LOW — free** |
| Add the conversion line to the funnel slide (payload already on `DashboardData`) | LOW |
| Configure the existing `forecast_evolution` handler into the YAML | **LOW — free** |
| Label the funnel's current-week fallback as a different measure | LOW |
| Pipeline slide → true 2×2 | LOW |
| Make cohort grain (`Y`/`Q`) configurable in the deck YAML | LOW |
| Retire the v1 dead path (`pptx_builder`, `ChartResolver`, `StraplineResolver`, `validation.py`, `data_resolver`) and correct the README | LOW |

### B. Shared visual-system work

| Work | Effort |
|---|---|
| **Deck reads the governed currency** (`currency.current_symbol()`), removing the `£` default — **RED-1** | LOW |
| Generate `theme.ts` and `pptx_theme.py` from one token source; CI test on divergence | LOW–MED |
| Move `sortStratBars` + `cleanBucketLabel` semantics into the payload as `displayOrder` / cleaned labels; both renderers iterate payload order — **RED-2** | LOW–MED |
| Promote `valueFormat`, `currency`, `seriesColorRole` onto the governed payloads | MED |
| Reconcile `mi_chart_factory`'s formatter and theme with the other two (or make it consume the shared tokens) | MED |
| One sequential scale across choropleth / heatmap / Plotly | LOW |

### C. Dashboard/PPTX chart-reuse work

| Work | Effort |
|---|---|
| Promote `_multidim` to a governed `/mi/multidim` payload; React renders the same heatmaps — **RED-4** | MED |
| Move broker / borrower_type / ticket into `snapshots._STRAT_DIMS` and delete `_extra_stratifications` + `_ticket_series` — **RED-3** | LOW–MED |
| Move the forecast-variance lag-1 shift from `EvolutionPanel.tsx:909-918` into `evolution.forecast_evolution` | LOW |
| Extend `test_channel_parity` to compare **all** stratification keys and **display order**, not just shared keys and values | LOW |
| Replace the grep-based `test_the_deck_layer_owns_no_economic_calculation` with an import-boundary check (deck may import `mi_agent_api` / `analytics_lib`, may not define bin edges) | LOW |

### D. Small MI composition (existing primitives)

| Work | Value | Effort | Before go-live? |
|---|---|---|---|
| Pipeline stratification by LTV / age / ticket / rate — one `_dimension_breakdown` per already-materialised bucket column | **High** — closes the biggest asymmetry between the two lenses, and both surfaces gain it | LOW | **Yes** |
| Add facts `funded_periods`, `funded_balance`, `pipeline_share` to `composition.build_facts` | **High** — unlocks the conditional deck without a `seasoned` flag | LOW | **Yes** |
| Move `forecast_extrapolation._THRESHOLDS` to client config and thread `extra_thresholds` from the deck YAML (adds £200m) | Medium | LOW | **Yes** — trivial, and the brief asks for £200m |
| `/mi/executive-summary` block backing slide 1 and a React landing card | Medium-high | MED | Nice to have |
| Concentration-test sparkline from `compute_history` | Medium | LOW–MED | Nice to have |

### E. Genuinely new MI capability

| Work | Value | Effort | Before go-live? |
|---|---|---|---|
| Forecast calibration statistics (MAE / bias / hit-rate over `forecast_evolution`) | Medium — the two series and the chart are enough for v1 | MED | **No** |
| Per-period × per-dimension funded series (stacked balance by region over time) | Medium — the bridge already answers the executive question | MED–HIGH | **No** |
| Wire `analytics_lib.history` (CPR/SMM, default, cure, loss) into `/mi/*` and React | Medium, and coverage-dependent | HIGH | **No** |
| Wire `risk_monitor.migration` (LTV / bucket migration) into `/mi/*` and React | Medium, seasoned books only | HIGH | **No** |
| Surface `period_change` (loan-identity bridge + distribution shift) as a dashboard panel | Medium-high long-term | HIGH | **No** |
| HPI / valuation-driven movement | Low — no index exists in the repo | HIGH | **No** |

**Default answer for D and E is NO unless it materially improves the client-facing
product.** Four D items pass that bar; **no E item does.**

---

## 13. Recommendation

### MUST DO BEFORE GO-LIVE — 6 items, all LOW effort

Every one is a correctness or credibility defect, not a feature.

1. **Deck honours the governed reporting currency.** Remove the `£` default in
   `metric_resolver.compact_currency` and resolve through
   `mi_agent_api.currency`. *(RED-1 — a non-GBP book currently ships an investor deck
   with the wrong currency symbol.)*
2. **One display order across both channels.** Emit `displayOrder` and cleaned
   labels from the stratification payload; both renderers iterate payload order.
   *(RED-2 — LTV/age/ticket/vintage/rate bands currently read in different orders in
   the dashboard and the deck.)*
3. **Move broker / borrower_type / ticket into `snapshots._STRAT_DIMS`; delete
   `_extra_stratifications` and `_ticket_series`.** *(RED-3 — removes the only
   second economic definition in the renderer, and brings all six deck dimensions
   inside the parity test.)*
4. **Extend the parity test** to all stratification keys and to display order, and
   replace the grep-based no-calculation test with an import-boundary check.
5. **Draw the two forecast breakdowns that are already resolved and discarded**
   (`byRegion`, `byLtvBucket`) and **configure the existing `forecast_evolution`
   handler into the YAML.** Two slides' worth of content for a config change and one
   handler call.
6. **Add the conversion line to the funnel slide**, and label the current-week
   fallback as the different measure it is.

*Not on this list, deliberately:* the multidim promotion (RED-4). It is a real
architectural gap, but the deck's cross-tabs are correct today and no reader is
misled. It is the first thing after go-live.

### HIGH-VALUE NICE TO HAVE BEFORE GO-LIVE — 5 items

1. **Slide 1 executive composition.** The deck's weakest point relative to the brief
   is that it opens on a cover and a text summary. Zero extra MI calls. (MED)
2. **Pipeline stratification by LTV / age / ticket / rate.** The buckets are already
   materialised on the prepared pipeline frame; this is one breakdown call per
   dimension and both surfaces gain it. (LOW)
3. **Collapse three stratification slides into one 2×2, and complete the funded
   evolution 2×2.** Turns a data dump into an executive page. (LOW)
4. **Add `funded_periods` / `funded_balance` / `pipeline_share` facts** so the
   new-book vs seasoned-book behaviour is driven by data, not by editing the YAML per
   client. (LOW)
5. **Generate the two theme files from one token source, with a CI divergence
   test.** Cheap insurance against the palettes silently parting. (LOW–MED)

### POST-GO-LIVE

- Promote `_multidim` to `/mi/multidim`; React renders the same heatmaps (RED-4).
- `/mi/executive-summary` payload; React landing card mirroring slide 1.
- Concentration movement sparklines from `compute_history`; emerging risks onto the
  watchlist slide.
- Forecast-variance shift moved server-side; then forecast calibration statistics.
- Full `valueFormat` / `seriesColorRole` presentation contract on all payloads;
  reconcile `mi_chart_factory`.
- Surface `period_change` (loan-identity bridge, distribution shift) in React first,
  then consider a seasoned-book deck module.
- Consider a static choropleth for the deck's geography slide.
- Retire the v1 dead path and rewrite `mi_agent_pptx/README.md` to describe the
  architecture that actually runs.

---

## Final questions, answered

**1. Can the existing MI engine already support most of the proposed deck?**
Yes — comfortably. Of the ~18 distinct visuals proposed, 11 already render in the
deck, 5 need renderer work over payloads that are already governed (two of them
already *resolved and discarded*), and 2 need a small composition of existing
primitives. The engine is ahead of the deck, not behind it.

**2. Which proposed slides require genuinely new calculations?**
**None.** The two closest are: "funded balance by region as stacked bars over time",
which needs a per-period × per-dimension series that does not exist (and which the
existing movement bridge already answers well enough for an executive audience); and
the **£200m** scale target, which is a *configuration* change — the milestone
machinery exists and `build_extrapolation` already accepts `extra_thresholds`; the
ladder is simply a module constant today.

**3. What high-value MI capability exists today that you failed to include?**
Three worth adding: **actual funded vs prior-run forecast** (in React, a deck handler
exists but is unconfigured); **conversion rate over time** (rich in the engine —
cohort conversion, weekly velocity, lag adjustment, sufficiency gating — and the deck
drops the whole block); and **concentration movement** (`/mi/concentration-tests/history`,
in React, absent from the deck). Two more are real and deliberately excluded for now:
the governed **loan-identity balance bridge** and **distribution shift** in
`mi_agent/period_change/` — genuinely good, fully governed, surfaced *only* through
chat, and overlapping the movement waterfall the deck already carries. Also present
but agent-tool-only: observed **prepayment/CPR, default, cure, loss/recovery**
(`analytics_lib/history.py`), **migration matrices**
(`mi_agent/risk_monitor/migration.py`) and **valuation-age profiling** — all real
implementations, none wired to `/mi/*` or React, and none appropriate for an
investor deck until their field coverage is proven.

**4. Is the current PPTX genuinely rendering the Dashboard, or merely recreating its
charts?**
**Both, and the split is the point.** It renders the *same payloads* — the deck calls
the dashboard's compute functions in-process, and a committed parity test proves the
numbers agree (15 passed for this review). It *recreates* the visuals with an
independent matplotlib renderer, because there is no browser in the Azure Functions
deck path by design. So: **option B for the analysis, option D for the presentation
grammar.** The D half is where all four RED findings live — currency, ordering, the
deck-only ticket bins, and the deck-only cross-tab.

**5. What is the best technically feasible way to make PPTX and React look and behave
like two surfaces of the same product?**
Option **C+**: keep the shared governed payload (already built), and extend it to
carry the *presentation contract* — display order, cleaned labels, value format,
currency, series colour role, RAG semantics, footnotes, reporting period. Generate
the theme tokens for both surfaces from one source with a CI divergence test.
Screenshotting React (option A) is not feasible here and would trade determinism for
fidelity — the wrong trade for an artefact that carries publication gates.

**6. Can new/non-seasoned and seasoned books use one conditional deck architecture
without client-specific PPTX code?**
**Yes — and it already does.** `mi_agent_pptx/composition.py` derives 24 governed
facts from the resolved payloads, evaluates a restricted-AST `when:` expression per
slide, applies a data guard, and records every omission with an investor-facing
reason. There is no `seasoned` flag and none should be added. Three additional facts
(`funded_periods`, `funded_balance`, `pipeline_share`) are all that stand between
today and full new/seasoned/mixed behaviour from a single deck config.

**7. If we freeze MI today and only improve the PPTX renderer, how good can the
resulting product be?**
**Very good — a genuinely credible institutional pack.** With MI frozen you can
deliver all three deck variants in §10, including slide 1, the funded and pipeline
2×2s, forecast-by-region and by-LTV, the conversion line, forecast-vs-actual,
multidimensional heatmaps, cohort seasoning, time-to-scale, and the
concentration/headroom page with forecast and stress states. The only things you
cannot have are stacked funded balance by region over time, forecast accuracy
statistics, and the run-off/migration family. None of those is a reason an investor
would decline the pack.

**8. What is the smallest implementation sprint that gets to a polished,
commercially credible final automated PPTX?**
**One sprint, roughly 8–11 working days:**

- **Days 1–3 — correctness (the MUST DO list).** Governed currency; one display
  order in the payload; fold broker/borrower_type/ticket into the governed
  stratifications and delete the deck's own; extend the parity test to all keys and
  to order.
- **Days 4–6 — content already paid for.** Draw the two discarded forecast
  breakdowns; configure the existing forecast-evolution slide; add the conversion
  line; add pipeline stratification over the already-materialised buckets.
- **Days 7–9 — composition.** Slide 1; three stratification slides → one 2×2;
  funded evolution → 2×2; add the three composition facts and rewrite the deck YAML
  as one conditional config.
- **Days 10–11 — hygiene.** Generate both theme files from one token source with a
  CI test; retire the v1 dead path; correct the README.

Everything in that sprint is renderer, payload-shape or configuration work. **No new
MI calculation is required to reach a polished, commercially credible deck.**

---

## Appendix A — evidence probes

Three read-only probes were executed against the repository at `e7678c8`. No source
file was modified; the probe scripts were written to the session scratchpad, and the
one temporary test file was deleted immediately after the run.

**Probe 1 — committed cross-channel parity suite.**
```
$ python -m pytest tests/mi_agent_pptx/test_channel_parity.py -q
.....s..........
15 passed, 1 skipped in 12.23s
```

**Probe 2 — governed currency vs the deck formatter, and governed ticket bands vs
the deck's fallback.**
```
governed current_symbol():                  €
deck compact_currency(124_600_000):         £124.6MM
governed balance_band labels:               ['<50k','50-100k','100-150k','150-200k',
                                             '200-300k','300-500k','500k-1m','>=1m']
deck _ticket_series fallback bins:          [0, 100_000, 150_000, 200_000, 250_000,
                                             300_000, 400_000, 1e12]
deck _ticket_series fallback labels:        ["<£100K","£100–150K","£150–200K",
                                             "£200–250K","£250–300K", …]
```

**Probe 3 — stratification bar order in the deck payload** (a temporary test reusing
the committed `test_channel_parity` fixtures; removed after the run):
```
ltv:    40-50% | 50-60% | 60-70%
age:    65-70 | 70-75 | 75-80
ticket: 200-300k | 300-500k | 100-150k      ← balance-descending, not bucket order
```
The `ticket` labels confirm the deck is consuming the **governed** `ticket_bucket`
column on a prepared tape (so the ungoverned fallback bins are latent, not active),
and simultaneously that the deck renders the payload's balance-descending order where
React's `sortStratBars` would render `100-150k | 200-300k | 300-500k`.

---

## Appendix B — key source references

| Concern | Reference |
|---|---|
| Deck production entry | `apps/blob_trigger_app/pptx_stage.py:102-109,410-472` |
| On-demand generation | `mi_agent_api/app.py:1543-1566`; `mi_agent_api/deck_generation.py:282-284` |
| Shared payload bridge | `mi_agent_pptx/mi_api.py` (whole module); entry `:589` |
| Slide handlers / dispatch | `mi_agent_pptx/deck.py:2090-2107` |
| Deck configuration | `configs/pptx/investor_pack.yaml` |
| Conditional composition | `mi_agent_pptx/composition.py:68-183` |
| Publication gates | `mi_agent_pptx/preflight.py:498-516` |
| Matplotlib renderer | `mi_agent_pptx/render.py:55-462` |
| Deck theme (duplicated tokens) | `mi_agent_pptx/pptx_theme.py:37-96` |
| React theme | `frontend/mi-agent-ui/src/lib/theme.ts`; `src/index.css` |
| React Plotly re-skin | `frontend/mi-agent-ui/src/lib/plotlyTheme.ts` |
| Third chart stack | `mi_agent/mi_chart_factory.py:65-90,162,476,737-740`; live via `mi_agent/mi_agent_workflow.py:1222` |
| React bucket ordering | `frontend/mi-agent-ui/src/lib/stratOrder.ts:17,34,51` |
| Payload bucket ordering | `analytics_lib/stratify.py:152-156` |
| Governed bucket edges | `config/mi/buckets.yaml` (`balance_band` `:124-137`; `seasoning` `:147-200`) |
| Governed currency | `mi_agent_api/currency.py`; applied at `mi_agent_api/datasets.py:873-887` |
| Funded snapshot / stratifications | `mi_agent_api/snapshots.py:333-342,504-575,613-741` |
| Pipeline snapshot | `mi_agent_api/pipeline_contract.py:787-935` |
| Pipeline bucket materialisation | `mi_agent_api/pipeline_prep.py:602-608`; `config/mi/pipeline_field_contract.yaml:280` |
| Evolution / bridge / cohorts / funnel | `mi_agent_api/evolution.py:251,324,526,701,934,1129` |
| Forecast | `mi_agent_api/forecast_bridge.py:425-445`; `mi_agent_api/workspace.py:312-335`; `mi_agent_api/forecast_extrapolation.py:31,162-190,358-415` |
| Concentration | `mi_agent_api/concentration_tests_api.py:340`; `mi_agent/concentration_tests/forward.py:250,448,517,619,689`; `mi_agent_api/app.py:1774` |
| Period change (unsurfaced) | `mi_agent/period_change/{bridge,distribution,ranking,workflow}.py`; `mi_agent_api/period_change_route.py`; routed at `mi_agent_api/chat_routing.py:3648` |
| Observed-behaviour library (agent tools only) | `analytics_lib/history.py:111,191,353,562,871,1017`; `trakt_tools/handlers/history.py` |
| Migration (stub vs real) | `analytics_lib/migration.py:26-33` (raises) vs `mi_agent/risk_monitor/migration.py:78,156` |
| Parity tests | `tests/mi_agent_pptx/test_channel_parity.py:175-189,289-320` |
| React shell / tabs | `frontend/mi-agent-ui/src/components/AppShell.tsx:403-514` |
| React panels | `FundedSnapshotPanel.tsx`, `PipelineSnapshotPanel.tsx`, `EvolutionPanel.tsx:603-608,904-918,970-1149`, `ForecastView.tsx:57-67`, `GeographyPanel.tsx`, `UkChoropleth.tsx`, `risk/RiskLimitsWorkspace.tsx`, `risk/ConcentrationDetailPanel.tsx` |
| Weekly brief (flag-gated) | `mi_agent_api/app.py:1137-1149`; `mi_agent_api/insight_generators.py` |
| Deck insight generators | `mi_agent_pptx/insights.py:28-32,137-530`; `mi_agent_pptx/watchlist.py` |
| Dead v1 path | `mi_agent_pptx/{pptx_builder,insight_resolver,validation,data_resolver}.py`; `chart_resolver.ChartResolver` |
