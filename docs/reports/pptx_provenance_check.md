# PPTX Provenance Check

**Purpose:** prove which PPTX implementation is connected to the current React
application, before acting on `docs/reports/pptx_capability_ux_review.md`.
**Method:** traced from the React UI downward. No code changed.
**Base:** `main` @ `e7678c8`.

**Headline:** the two estates are **completely disjoint** — zero imports in either
direction, and separated at the deployment boundary by explicit manifests. The
previous audit was about the React-connected estate throughout. **One claim in it
was wrong and is corrected below (finding 2), and two are refined (11, 12).**

---

## 1. Current React PPTX entry point

**The current React application DOES expose PPTX generation.** Full chain, every
hop proven:

| # | Hop | Exact location |
|---|---|---|
| 1 | React component | `components/DeckDownloadMenu.tsx` — mounted at `components/HeaderBar.tsx:140`, itself mounted at `components/AppShell.tsx:284`. Not orphaned. |
| 2 | Frontend functions | `client.generateDeck` (`DeckDownloadMenu.tsx:181`), `client.getDeckGeneration` (`:105`), `client.deckDownloadUrl` (`:126,133`), `client.downloadDeck` (`:158`) — declared on `api/AgentClient.ts:165-196` |
| 3 | HTTP | `api/HttpAgentClient.ts:366` `POST /mi/decks/generate` · `:387` `GET /mi/decks/generate/{jobId}` · `:319` `GET /mi/decks` · `:326` `GET /mi/decks/download` |
| 4 | Backend route | `mi_agent_api/app.py:1543` `generate_deck` · `:1559` `generate_deck_status` · `:1458` `list_decks` · `:1482` `download_deck` |
| 5 | Service | `mi_agent_api/deck_generation.py` → `request_generation` → `_run_generation` (`:269`) |
| 6 | Orchestration stage | `apps/blob_trigger_app/pptx_stage.generate_investor_pptx` (`deck_generation.py:282-284` → `pptx_stage.py:414`) |
| 7 | PPTX generator | `pptx_stage._invoke_generator(argv)` (`:102-109`) → **`mi_agent_pptx.cli.run`** |
| 8 | Builder | `mi_agent_pptx/cli.py:317` `DeckBuilder` + `DeckContext` (`mi_agent_pptx/deck.py`); slides from `configs/pptx/investor_pack.yaml` |
| 9 | Chart rendering | `mi_agent_pptx/render.py` (matplotlib `Agg`) + `chart_resolver.render_bridge_waterfall` |
| 10 | MI/data functions | `mi_agent_pptx/mi_api.build_dashboard_data` (`cli.py:318`) → `mi_agent_api.{snapshots, evolution, cohorts, geo, pipeline_contract, forecast_bridge, forecast_extrapolation, concentration_tests_api, risk_limits, workspace, funded_prep, datasets}` |
| 11 | File returned/stored | `<run_dir>/reports/investor_pack.pptx` + `.preflight.json` → `pptx_stage.persist_investor_deck` → deck store → `decks_mod.resolve_deck_local` (`artefacts.py:142`) → `FileResponse` (`app.py:1508`) |

**Empirical proof, executed:**

```
$ python -m pytest tests/test_deck_generation_route.py::test_react_button_generates_a_real_downloadable_deck -q
1 passed in 4.04s
```

That test drives `POST /mi/decks/generate` → poll → `GET /mi/decks` → `GET
/mi/decks/download` through `TestClient(mi_agent_api.app)` and asserts the response
is real PowerPoint bytes (`PK\x03\x04`), opens it with `python-pptx`, and finds ≥10
slides containing "Executive Summary".

**`mi_agent_pptx` is definitively the React-connected PPTX implementation.**

Also checked: `components/ExportMenu.tsx` (the artifact-card export) offers
PNG/SVG/CSV/XLSX/JSON only — **no PPTX**. `DeckDownloadMenu` is the sole PPTX action
in the React app.

---

## 2. Legacy Streamlit PPTX path

| # | Hop | Exact location |
|---|---|---|
| 1 | Entry point | `analytics/streamlit_app_erm.py` (116,965 bytes), run by `Dockerfile.streamlit` `ENTRYPOINT … streamlit run streamlit_app_erm.py` |
| 2 | PPTX action | `streamlit_app_erm.py:1002` `st.button("Generate PowerPoint")` → writes `temp_pptx_data/temp_data_*.csv` → `subprocess.run([sys.executable, "generate_pptx_client.py", "--input", temp_csv, "--output", output_pptx])` (`:1013-1021`) |
| 3 | Builder | `analytics/generate_pptx_client.py` (114,513 bytes) — its own `Presentation()`, `add_cover_slide`, `add_kpi_slide`, `add_chart_slide`, `add_static_pools_slides`, `add_risk_monitoring_slide`, `add_scenario_analysis_slide` |
| 4 | Chart renderer | Its own matplotlib (53 `plt.` call sites): `save_ltv_dual_chart`, `save_geographic_treemap`, `save_bubble_ltv_vs_age`, `save_bubble_balance_vs_value`, `save_nneg_distribution`, `save_vintage_distribution`, `render_risk_limits_table_png`, … **zero Plotly** |
| 5 | Data source | **A CSV file passed on the command line.** No MI API, no governed compute function, no registry resolution. |
| 6 | Return | `st.download_button` with the file bytes (`:1025-1032`) |

**Modules exclusive to this estate** (proven by import graph, not by name):
`analytics/streamlit_app_erm.py`, `analytics/generate_pptx_client.py`,
`analytics/charts_plotly.py`, `analytics/tab_pipeline.py`,
`analytics/mi_prep.py`, `analytics/risk_monitor.py`,
`analytics/scenario_engine.py`, `analytics/static_pools_core.py`,
`analytics/pipeline_tab_helpers.py`, `analytics/pipeline_forward_risk.py`,
`analytics/pipeline_snapshot_selector.py`, `analytics/portfolio_semantics.py`,
`analytics/blob_storage.py`, plus `Dockerfile.streamlit`, `deploy-streamlit.sh`,
`docker/start-streamlit.sh`, `.github/workflows/deploy_streamlit_dashboard.yml`.

**`generate_pptx_client.py` has exactly one caller in the entire repository:**
```
$ grep -rn "generate_pptx_client" --include=*.py --include=*.yml --include=*.sh .
./analytics/streamlit_app_erm.py:1013
```

### Cross-estate import check (both directions)

```
React estate → legacy analytics:   NONE
legacy analytics → mi_agent_pptx / mi_agent_api:   NONE
legacy analytics → mi_agent:   NONE
```

`analytics_lib/` is a **different package** from `analytics/` (no `__init__.py` in
the latter) and belongs to the current estate. `mi_agent/mi_chart_factory.py:66`
states its theme is "Aligned with analytics/charts_plotly.py (**recreated here, not
imported**)" — a copy, not a dependency.

### One genuine, narrow exception — and it is not PPTX

`engine/orchestrator/trakt_run.py:871-878` lazily imports four modules from
`analytics/`, behind a config flag (`maybe_persist_forward_exposure`, disabled
unless `pipeline_persistence.enabled`):
`analytics.pipeline_expected_funding`, `analytics.pipeline_persistence`,
`analytics.pipeline_prep`, `analytics.pipeline_reconciliation`.

These are **pipeline data-preparation** helpers. Neither `streamlit_app_erm.py` nor
`generate_pptx_client.py` is among them. So `analytics/` is *not wholly legacy* —
a pipeline slice is shared — but the **PPTX slice of `analytics/` is reachable only
from Streamlit**.

**Incidental pre-existing defect found while proving this** (working tree clean, so
it is on `main`, not caused by this session):
```
$ python -m pytest tests/test_mi_api_appservice_packaging.py -q
FAILED TestRequirementsSufficiency::test_every_reachable_distribution_is_declared
  ['analytics (provides: analytics)', 'lxml (provides: lxml)']
```
`analytics/` is excluded from **both** deployment artefacts (absent from
`deploy/trakt-mi-api/package_contents.txt`; `.funcignore:33`), so if
`maybe_persist_forward_exposure` is ever enabled in a deployed host it raises
`ImportError` at that line. Unrelated to the PPTX audit; flagged for the record.

---

## 3. Side-by-side path table

| Component | Current React PPTX | Legacy Streamlit PPTX | Shared | Dead |
|---|:---:|:---:|:---:|:---:|
| **Entry points** |
| `components/DeckDownloadMenu.tsx` → `HeaderBar.tsx:140` | ✅ | | | |
| `mi_agent_api/app.py` `/mi/decks{,/download,/generate,/generate/{id}}` | ✅ | | | |
| `mi_agent_api/deck_generation.py` | ✅ | | | |
| `apps/blob_trigger_app/pptx_stage.py` | ✅ | | | |
| `mi_agent_pptx/cli.py` (`run`) | ✅ | | | |
| `analytics/streamlit_app_erm.py:1002-1032` | | ✅ | | |
| **Builders** |
| `mi_agent_pptx/deck.py` (`DeckBuilder`, 24 handlers) | ✅ | | | |
| `analytics/generate_pptx_client.py` | | ✅ | | |
| `mi_agent_pptx/pptx_builder.py` | | | | ✅ **fully orphaned** |
| **Templates** |
| `configs/pptx/investor_pack.yaml` | ✅ | | | |
| No `.potx` anywhere (both estates assemble shape-by-shape) | — | — | — | — |
| Legacy `config/client/*.yaml` logo/branding (`_load_client_config`) | | ✅ | | |
| **Chart renderers** |
| `mi_agent_pptx/render.py` (matplotlib) | ✅ | | | |
| `mi_agent_pptx/chart_resolver.render_bridge_waterfall` | ✅ | | | |
| `mi_agent_pptx/chart_resolver.ChartResolver` (class) | | | | ✅ test-only |
| `analytics/generate_pptx_client.py` `save_*` (53 `plt.`) | | ✅ | | |
| `analytics/charts_plotly.py` | | ✅ | | |
| React Recharts / `artifacts/*ArtifactView.tsx` | ✅ (dashboard) | | | |
| `mi_agent/mi_chart_factory.py` (Plotly, chat artifacts) | ✅ | | | |
| **MI API adapters** |
| `mi_agent_pptx/mi_api.py` | ✅ | | | |
| `mi_agent_api/{snapshots,evolution,cohorts,geo,pipeline_contract,forecast_bridge,forecast_extrapolation,concentration_tests_api,risk_limits,workspace}` | ✅ | | ✅ (React dashboard + deck) | |
| `analytics/mi_prep.py`, `analytics/risk_monitor.py`, `analytics/static_pools_core.py` | | ✅ | | |
| `analytics_lib/` (buckets, stratify, cohort, numeric) | ✅ | | ✅ (API + deck + tools) | |
| `analytics/pipeline_{prep,persistence,reconciliation,expected_funding}.py` | | ✅ | ⚠️ also `engine/orchestrator/trakt_run.py:871-878` | |
| **matplotlib code** |
| `mi_agent_pptx/render.py`, `chart_resolver.py` | ✅ | | | |
| `analytics/generate_pptx_client.py` | | ✅ | | |
| **Plotly code** |
| `mi_agent/mi_chart_factory.py` | ✅ | | | |
| `frontend/.../lib/plotlyTheme.ts`, `PlotlyArtifactView.tsx` | ✅ | | | |
| `analytics/charts_plotly.py` | | ✅ | | |
| **Screenshot / image rendering** |
| **None in either estate.** No playwright/selenium/puppeteer/html2image. `kaleido` is a guarded optional import (`mi_chart_factory.py:139`) and deliberately absent from `deploy/trakt-mi-api/requirements.txt:57`. | — | — | — | — |
| **Styling / config** |
| `mi_agent_pptx/pptx_theme.py` | ✅ | | | |
| `frontend/.../lib/theme.ts`, `src/index.css` | ✅ | | | |
| `config/mi/buckets.yaml`, `stratification_catalogue.yaml`, `config/risk/*` | ✅ | ✅ | ✅ (config data, both read it) | |
| `mi_agent_pptx/{data_resolver,metric_resolver.MetricResolver,insight_resolver,validation}.py` | | | | ✅ (`validation.py` orphaned; rest test-only) |
| **Tests** |
| `tests/test_deck_generation_route.py`, `test_pptx_orchestration_stage.py`, `test_deck_publication.py`, `mi_agent_api/tests/test_decks.py` | ✅ | | | |
| `tests/mi_agent_pptx/*` (16 files) | ✅ | | | |
| `tests/mi_agent_pptx/test_charts_and_placeholders.py` (covers v1 classes) | | | | ✅ tests dead code |
| No test anywhere covers `analytics/generate_pptx_client.py` | | ✅ (untested) | | |
| **README / docs** |
| `mi_agent_pptx/README.md` — **mixed**: its "Module layout" table lists the v1 (dead) modules and omits `deck.py`, `mi_api.py`, `render.py`, `composition.py`, `preflight.py`, `insights.py`, `watchlist.py`, `movement.py`, `cohorts.py`, `concentration.py`, `deck_context.py` entirely; its later sections correctly describe the v2 path | ✅ (subject) | | | partly describes dead code |
| `analytics/generate_pptx_client.py` docstring ("Mirrors … streamlit_app_erm.py") | | ✅ | | |

---

## 4. Recheck of every previous finding

| # | Previous finding | Estate | Verdict |
|---|---|---|---|
| 1 | `mi_api.py` calls the same compute functions as `/mi/*` | **A** | **STANDS.** `mi_api.py` is reached from `cli.run` (`cli.py:318`), which is reached from the React route via `pptx_stage._invoke_generator`. |
| 2 | `test_channel_parity.py` compares React HTTP routes with the real current deck generation path | **A, but the claim is WRONG** | **WITHDRAWN — see §5.** It compares React HTTP routes with `build_dashboard_data`, not with a built deck. |
| 3 | PPTX hard-codes GBP | **A** | **STANDS.** `metric_resolver.compact_currency(value, symbol="£")` is imported by `deck.py` (`:515,666,686,749,902,…`) on the live path. Legacy has its own separate `£` handling; irrelevant. |
| 4 | PPTX bucket ordering differs from React | **A** | **STANDS, and is strengthened.** The divergence sits strictly downstream of where the parity test stops (§5), which is exactly why it was never caught. |
| 5 | Three PPTX stratifications are calculated inside the renderer | **A** | **STANDS.** `mi_api._extra_stratifications` (`:508`), `_ticket_series` (`:446`), `_stratify_dim` (`:485`) — all in the React-connected module. |
| 6 | Multidimensional cross-tab exists only in PPTX | **A** | **STANDS.** `mi_api._matrix`/`_multidim` (`:523-559`), rendered by `deck.slide_multidim` (`:1353`). (The legacy estate has its *own* bubble/treemap charts — unrelated code, not the same thing.) |
| 7 | Actual-vs-prior-forecast handler exists | **A** | **STANDS.** `deck.slide_forecast_evolution:1526`, registered `deck.py:2104`, **absent from `investor_pack.yaml`** → unreachable by configuration, not by estate. |
| 8 | Conversion-over-time capability exists | **A** | **STANDS.** `mi_agent_api/evolution.pipeline_funnel_evolution:934-1120`, resolved onto `DashboardData.funnel` (`mi_api.py:720`), dropped by `deck.slide_funnel`. |
| 9 | `forecastBreakdowns.byRegion` / `byLtvBucket` available to the current deck | **A** | **STANDS.** Set at `mi_api.py:877-880`; `deck.py:1454-1457` consumes only `byCompletionMonth`. |
| 10 | Conditional-deck system / facts / `when:` evaluator belongs to the current PPTX | **A** | **STANDS.** `mi_agent_pptx/composition.py`, called from `deck.DeckBuilder.build:2118-2121` on the live path. (Correction: I said "24 facts"; the dict at `composition.py:142-182` contains **26** keys.) |
| 11 | 1,400–1,700 lines of dead PPTX code | **D** | **STANDS, REFINED.** Of the v1 residue: `pptx_builder.py` (497 lines) and `validation.py` (91) have **zero references anywhere, tests included** — fully orphaned. `ChartResolver`, `StraplineResolver`, `MetricResolver`, `data_resolver` are **test-only** (`tests/mi_agent_pptx/test_charts_and_placeholders.py:7,9`, `test_data_and_metrics.py:8`) — unreachable from production but not un-referenced. "≈1,400 lines unreachable from production" is right; "unreferenced" applies to ~590 of them. |
| 12 | README describes legacy rather than current behaviour | **A + D** | **STANDS, REFINED.** It does *not* describe the **Streamlit** estate — it explicitly disclaims it (`README.md:9-12`). It describes `mi_agent_pptx`'s own **v1 (dead) architecture**: the "Module layout" table lists `pptx_builder.py` and `validation.py` and omits `deck.py`, `mi_api.py`, `render.py`, `composition.py`, `preflight.py` and six more live modules. Later sections correctly describe the v2 path. So: **stale about its own estate, not about the wrong estate.** |

**No previous finding belonged to the legacy Streamlit estate.** Categories B and C
are empty.

---

## 5. Test provenance — the correction

### What `test_channel_parity.py` actually invokes

```python
@pytest.fixture()
def deck(book):
    from mi_agent_pptx.mi_api import build_dashboard_data      # ← line 116-120
    return build_dashboard_data(book, client_id=CLIENT, as_of=AS_OF, ...)

@pytest.fixture()
def react(book):
    from fastapi.testclient import TestClient                   # ← line 123-137
    from mi_agent_api.app import app
    client = TestClient(app)                                    # real HTTP routes
```

Proven by grep:

```
$ grep -n "DeckBuilder\|cli.run\|generate_investor_pptx\|\.pptx" tests/mi_agent_pptx/test_channel_parity.py
(no match — no deck is ever rendered in this file)

$ grep -ln "DeckBuilder\|cli.run\|generate_investor_pptx" tests/mi_agent_pptx/*.py
test_build_end_to_end.py  test_concentration_flexible.py  test_concentration_states.py
test_investor_v2.py  test_investor_v21.py  test_investor_v22.py

$ grep -ln "TestClient\|mi_agent_api.app" tests/mi_agent_pptx/*.py
test_channel_parity.py
```

**The two sets are disjoint.** The only test that touches React never builds a deck;
the six that build a deck never touch React.

### Classification

**ENGINE / PPTX PARITY ONLY** — more precisely, **payload parity**.

It proves `mi_agent_pptx.mi_api.build_dashboard_data` returns the same *values* as
the `/mi/*` HTTP routes. `build_dashboard_data` **is** on the production React path
(`cli.py:318`), so this is not testing an unused function — but it stops one hop
short of the deck. It cannot observe `DeckBuilder`, `render.draw_barlist`,
`compact_currency`, slide composition, or anything a reader sees.

### Why this matters for the earlier findings

The visual divergences sit on **both sides of where the test stops**: React's
`sortStratBars` runs inside `FundedSnapshotPanel.tsx:195` (downstream of the HTTP
payload the test reads), and the deck's rendering runs inside `render.draw_barlist`
(downstream of the `DashboardData` the test reads). The test is structurally
incapable of catching RED-2, and the deck-building tests have no React side to
compare against. **The gap explains the defect.**

**Correction to the previous report:** it said the test "drives the real React HTTP
routes and the real deck build". The React half is accurate. The deck half is not —
I repeated the test's own docstring (`test_channel_parity.py:3-4`), which overstates
what its fixtures do. That docstring should be corrected too.

**What this does NOT change:** every finding was derived from reading the
React-connected source and from two independent probes (currency, bucket edges) plus
a fixture-driven order probe — not from trusting the test. Findings 1 and 3–10 stand
on their own evidence.

---

## 6. Deployment proof

| Question | Answer | Proof |
|---|---|---|
| Which PPTX module is packaged in the React production deployment? | **`mi_agent_pptx`** | `deploy/trakt-mi-api/package_contents.txt` lists `mi_agent_pptx  # investor deck rendering (deck-generation route)`, `apps  # apps.blob_trigger_app: storage, pptx_stage, layout`, `configs  # configs/pptx/investor_pack.yaml — the deck definition`. The workflow stages exactly these paths (`deploy-mi-api.yml:76-84`). |
| Which PPTX route is exposed? | `POST /mi/decks/generate`, `GET /mi/decks/generate/{job_id}`, `GET /mi/decks`, `GET /mi/decks/download` | `mi_agent_api/app.py:1458,1482,1543,1559` |
| Is `analytics/` (legacy) in the React deployment? | **No** | `grep -c "^analytics$" deploy/trakt-mi-api/package_contents.txt` → `0` |
| Is `analytics/` in the Functions deployment? | **No** | `.funcignore:33` `analytics/`, under the heading `# --- legacy Streamlit MI app (replaced; not used by the Functions host) ---` (`:18`) |
| Is Streamlit packaged/deployed at all? | **Yes — still live** | `.github/workflows/deploy_streamlit_dashboard.yml` deploys image `trakt-streamlit` to Azure Web App `trakt-dashboard` (RG `trakt`) on every push to `main` touching `analytics/**`, `config/**`, `Dockerfile.streamlit`, `requirements.txt` |
| What does the Streamlit image contain? | **Only `analytics/` + `config/`** | `Dockerfile.streamlit`: `COPY analytics/ ./analytics/`, `COPY config/ ./config/`, then `ENTRYPOINT … streamlit run streamlit_app_erm.py`. **`mi_agent_pptx`, `mi_agent_api` and `frontend/` are absent from the image.** |
| Does any production manifest reference the legacy PPTX path? | **No** | `generate_pptx_client.py` appears in no workflow, Dockerfile, manifest or requirements file — only in `analytics/streamlit_app_erm.py:1013` |

**Conclusion:** three deployment artefacts, cleanly separated.
`trakt-mi-api` (App Service, serves React) and the Functions host both carry
`mi_agent_pptx` and exclude `analytics/`. The `trakt-streamlit` container carries
`analytics/` and nothing else. **No artefact contains both PPTX implementations.**

---

## 7. Final answer

**1. What exact code generates PPTX for the current React application?**
`DeckDownloadMenu.tsx` → `HttpAgentClient.generateDeck` → `POST /mi/decks/generate`
→ `mi_agent_api/deck_generation._run_generation` →
`apps/blob_trigger_app/pptx_stage.generate_investor_pptx` → `mi_agent_pptx.cli.run`
→ `mi_agent_pptx/deck.DeckBuilder` + `mi_agent_pptx/mi_api.build_dashboard_data` +
`mi_agent_pptx/render.py`, driven by `configs/pptx/investor_pack.yaml`.

**2. What exact code generated PPTX for the old Streamlit application?**
`analytics/streamlit_app_erm.py:1002` "Generate PowerPoint" → CSV to a temp file →
`subprocess` → `analytics/generate_pptx_client.py` (its own matplotlib renderers,
its own `Presentation`, its own client config) → `st.download_button`. **Still
deployed** as `trakt-dashboard`, but from a container that contains no part of the
React estate.

**3. Which modules are genuinely shared?**
Between the two **PPTX** estates: **none.** Zero imports in either direction.
Shared only as *config data* read independently by both: `config/mi/buckets.yaml`,
`config/mi/stratification_catalogue.yaml`, `config/risk/*`, `requirements.txt`.
One code overlap exists outside PPTX: `analytics/pipeline_{prep,persistence,
reconciliation,expected_funding}.py`, lazily imported by
`engine/orchestrator/trakt_run.py:871-878` behind a config flag.

**4. Which modules are legacy/dead?**
*Legacy (live but separate product):* the `analytics/` Streamlit estate listed in
§2 — deployed, not dead.
*Dead (unreachable from any production path):* `mi_agent_pptx/pptx_builder.py` and
`mi_agent_pptx/validation.py` (fully orphaned, ~590 lines); `ChartResolver`,
`StraplineResolver`, `MetricResolver`, `data_resolver.py` (test-only, ~800 lines).
Plus two configured-out handlers: `deck.slide_portfolio_comparison` and
`deck.slide_forecast_evolution`.

**5. Did the previous audit accidentally include any legacy findings?**
**No.** Every module cited in it — `mi_api.py`, `metric_resolver.py`, `deck.py`,
`composition.py`, `render.py`, `pptx_theme.py`, `investor_pack.yaml` — is on the
React-connected path proven in §1. Nothing from `analytics/` was cited. Category B
(legacy only) and category C (both) are empty in §4.

**6. Does `test_channel_parity.py` genuinely prove React ↔ CURRENT PPTX parity?**
**No.** It proves React ↔ **`build_dashboard_data`** parity — the data layer of the
current PPTX, which is genuinely on the production path, but one hop short of the
deck. **ENGINE/PPTX PARITY ONLY.** No test in the repository compares a rendered
deck against the React dashboard. Your instinct was right, and this is the one
material correction to the previous report.

**7. Which previous findings remain valid for the production React-connected PPTX?**
Findings **1, 3, 4, 5, 6, 7, 8, 9, 10** — all valid, all category A. Finding **11**
valid with the orphaned/test-only split. Finding **12** valid, reclassified as
"stale about its own v1 architecture" rather than "describing the Streamlit estate".

**8. Which previous findings must be withdrawn or reclassified?**
**Withdrawn:** the characterisation of `test_channel_parity.py` as covering "the
real deck build". The parity evidence it provides is *payload* parity only, and the
report must say so.
**Reclassified:** finding 11 (≈590 orphaned + ~800 test-only, not 1,400–1,700
un-referenced); finding 12 (stale-about-itself, not wrong-estate); the fact count in
finding 10 (26, not 24).

**9. Does the 8–11 day sprint still stand after removing legacy estate from scope?**
**Yes — no legacy estate was ever in scope, so nothing is removed.** One item gains
weight and one is added:

- **Item 4 ("extend the parity test") is promoted from hygiene to a MUST DO in its
  own right.** There is currently *no* test anywhere that compares a rendered deck
  to the dashboard, which is precisely why the ordering defect survived. The fix is
  a new assertion class — build the deck via `cli.run` (as the six existing deck
  tests do) *and* drive `TestClient(app)` in the same fixture — not an extension of
  the existing value comparison.
- **Add: correct the `test_channel_parity.py` module docstring** (`:3-4`) and the
  `mi_agent_pptx/README.md` module table. Both currently misdescribe the system to
  the next reader, which is how this confusion started.

Everything else — currency, ordering, the three renderer-side stratifications, the
discarded forecast breakdowns, the unconfigured forecast-evolution slide, the
conversion line, slide 1 — is unchanged and unaffected by the estate question.

**Estimate: unchanged at 8–11 days**, with the deck-vs-React rendering test absorbed
into the Days 1–3 correctness block.
