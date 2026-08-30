# PPTX UX Parity & Final Deck — Implementation Report

## 1. Executive result

**PASS.**

**Is the final deck production-ready?** Yes. Five representative packs — new,
seasoned and growing books, in sterling and euro — were generated through the
production React route, passed all 21 publication gates, and were read slide by
slide. The pack is coherent, correctly scoped, correctly ordered and correctly
denominated.

**Is React/PPTX presentation parity now structurally enforced?** Yes, and by the
product rather than only by tests. Category order, label hygiene, currency symbol
and the multidimensional cross-tab are each decided once, above both surfaces.
Two publication gates check the **rendered** deck against those decisions, and a
deck that fails either is generated but withheld from publication — the operator
gets a diagnosable artefact and the client gets nothing wrong. Both defects were
deliberately reintroduced to confirm this: the pound sign failed three tests and
blocked publication; the balance ordering failed five tests and blocked
publication.

**Did any slide require new MI?** No. Every chart in the final deck is a governed
compute function's output, a composition of two of them, or a re-indexing of one.
Nothing was stopped for want of a primitive, so the audit's central finding held
under implementation.

---

## 2. Starting point

| | |
|---|---|
| Starting `main` SHA | `e7678c81100f562c25cb39cf1cbf69798e13a5ed` |
| Branch | `claude/pptx-ux-parity-implementation` |
| Audit reports used | `docs/reports/pptx_capability_ux_review.md`, `docs/reports/pptx_provenance_check.md` (brought across from `claude/pptx-capability-ux-audit-e0dhph`; nothing else was taken from that branch) |
| Commits | `b8af72e` docs · `3cceeca` parity foundation · `970b4d0` parity tests · `030910a` deck composition · `419d46a` visual QA and fixes |
| Diff | 27 files, +4,033 / −262 |

---

## 3. Architecture, before and after

**Before — drift below the payload.**

```
governed compute  ──►  payload  ──┬──►  React component decides order, labels
(shared, correct)                 │     (lib/stratOrder.ts) ──► Recharts
                                  │
                                  └──►  deck renderer decides currency, order
                                        (metric_resolver, mi_api) ──► matplotlib
```

The analysis was shared; every presentation decision after it was made twice. The
audit's four RED findings were all instances of that one shape.

**After — one decision, two renderers.**

```
governed compute  ──►  mi_agent_api.presentation  ──►  payload carries
(unchanged)            (order · labels · ordinality)   bars in display order
                       mi_agent_api.currency          + displayOrder: governed
                       (symbol, already governed)     + the resolved currency
                                       │
                       ┌───────────────┴───────────────┐
                       ▼                               ▼
                React renders                    deck renders
                what it is given                 what it is given
                (orderBarsForDisplay             (draw_barlist records
                 returns governed                 what it drew)
                 payloads unchanged)                    │
                                                        ▼
                                              publication gates check the
                                              RENDERED result against the
                                              governed answer
```

`presentation.py` is 269 lines and answers three questions: what order, what
label, is this ordinal. It computes no economic value, selects no population and
owns no threshold. The order is not a heuristic — `config/mi/buckets.yaml`
already declared the ladder for every banded dimension, and that declaration is
the authority.

---

## 4. The four RED findings

### RED-1 — the deck hard-coded a pound sign

**Root cause.** `metric_resolver.compact_currency(value, symbol="£")` defaulted
the symbol at ~20 call sites, and three further private money formatters in
`movement.py`, `watchlist.py` and `concentration.py` each carried their own
literal. The governed service (`mi_agent_api.currency`) resolved the client's
actual reporting currency and the deck never asked it. **A second instance was
found during implementation:** `insight_generators.money` — the governed prose
formatter behind React's weekly brief — carried the same literal, so that surface
was wrong too while its KPI tiles were right.

**Fix.** One symbol source. `compact_currency` defaults to
`currency.current_symbol()`; the three private formatters are deleted and call
the governed one; `insight_generators.money` routes through
`currency.format_money`. `mi_api` resolves the code exactly as the API resolves it
per request, and `use_currency` — a new scoping helper over the **existing**
ContextVar, not a second resolution path — holds it across the data build, the
render and preflight, then restores what it found. No PPTX-specific currency
resolution was added.

**Evidence.** `test_a_euro_book_renders_euro_on_both_surfaces` asserts the
dashboard returns `currencyCode: EUR`, renders euro tiles, and that the
downloaded `.pptx` contains `€` and **no** `£`. Reintroducing the literal:

```
FAILED test_a_euro_book_renders_euro_on_both_surfaces
FAILED test_the_deck_records_the_currency_it_reported_in
FAILED test_both_channels_agree_on_the_headline_balance_and_its_currency
Preflight: BLOCKED — [FAIL] governed_currency: the deck reports in EUR (€) but also renders £
```

### RED-2 — the two surfaces ordered ordinal bands differently

**Root cause.** `analytics_lib.stratify` ranks by balance descending. React
re-sorted ordinal dimensions into their natural ladder in the browser; the deck
drew the payload order. Neither was wrong alone; having both is the defect.

**Fix.** `snapshots._funded_stratifications` selects by materiality (top twelve by
balance — unchanged) then orders the survivors through `presentation.order_bars`,
tidies their labels, and tags the payload `displayOrder: "governed"`. React's
`orderBarsForDisplay` returns a governed payload unchanged and keeps the old
heuristic only for mock and legacy shapes. `stratOrder.ts` was not ported to
Python: the order comes from the bucket registry, which is where it was declared.

**Evidence.** The parity fixture's LTV bands are chosen so balance order and band
order genuinely disagree, and one test asserts they still do, so the suite cannot
quietly stop being able to catch this. Reintroducing the balance ordering:

```
FAILED test_the_dashboard_serves_bands_in_the_governed_ladder
FAILED test_the_deck_draws_bands_in_the_same_order_the_dashboard_serves
FAILED test_the_drawn_ltv_order_is_the_ladder_and_not_the_balance_ranking
FAILED test_every_banded_bar_list_passed_the_publication_gate
FAILED test_the_deck_and_react_cross_tab_share_axes_and_totals
Preflight: BLOCKED — [FAIL] governed_bucket_order: ltv, ticket, age
```

### RED-3 — three stratifications computed inside the renderer

**Root cause.** `mi_api._extra_stratifications` added broker, borrower type and
ticket size by picking its own source columns and stratifying itself.
`_ticket_series` carried bin edges (250k, 400k boundaries) that contradict
`config/mi/buckets.yaml` `balance_band`. All three are governed dimensions in
`config/mi/stratification_catalogue.yaml`; they had simply never been declared in
`snapshots._STRAT_DIMS`.

**Fix.** Declared there, alongside the other eight, computed by the same engine.
The renderer's copies — `_broker_series`, `_borrower_type_series`,
`_ticket_series`, `_stratify_dim`, `_extra_stratifications` — are **deleted**, not
relocated: no parallel PPTX helper was created. Ticket size reads the canonical
`ticket_bucket` column with **no fallback banding**, so a frame that never went
through the governed prep reports the dimension unavailable rather than inventing
a second ladder.

**Evidence.** `grep 'pd.cut|groupby|bins ='` across the live deck estate returns
nothing. The dashboard and the deck now serve eleven stratifications from one
engine, and the parity test compares them.

### RED-4 — the cross-tab existed only inside the deck

**Root cause.** `mi_api._matrix` / `_multidim` built the LTV × age, LTV × borrower
type and LTV × region matrices in the PPTX layer. The React product had no way to
reach that grouping.

**Fix.** `snapshots.cross_tab` / `snapshots.multidimensional` own it, with the
pair list as data rather than renderer code, axis labels ordered by the same
governed ladder the one-dimensional stratifications use. Served at
**`GET /mi/multidim`**. The deck calls the same function in-process. It is a
composition — governed band series, one balance sum per cell — not a new
calculation.

**Evidence.** `test_react_can_reach_the_multidimensional_analysis`,
`test_the_deck_and_react_cross_tab_share_axes_and_totals`,
`test_the_cross_tab_axis_order_is_the_governed_ladder`.

---

## 5. Final deck structure

25 slides configured; composition decides how many render. `risk` and
`concentration` are mutually exclusive, so the ceiling is 24.

**CORE — always**

| Slide | Business question |
|---|---|
| Cover | What is this report about? |
| **Executive Position** | Where is the portfolio today, what is coming, is anything near a limit? |
| Executive Summary | What changed this period? |
| Portfolio Composition | What do I own today? |
| Funded — Key Measures | How large is the book and what are its characteristics? |
| Funded Stratifications (2 × 2) | How is exposure distributed by risk, size, borrower, geography? |
| Multi-Dimensional Risk Analytics | Where do risk dimensions concentrate together? |
| Geographic Exposure | Where is the collateral? |
| Portfolio Health and Watch Items | What needs attention before the next period? |
| Data and Methodology | What does this cover, as at when, produced how? |

**CONDITIONAL — governed `when:` over the composition facts**

| Slide | Rule |
|---|---|
| Portfolio Movement and Drivers | `has_attribution or has_movement` |
| Funded Balance Evolution (2 × 2) | `has_funded_history` |
| Vintage Formation | `has_cohorts and funded_periods >= 2 and cohort_count >= 2` |
| Cohort Progression | `has_cohort_progression` |
| Pipeline Stratifications (2 × 2) | `has_pipeline` |
| Pipeline Overview | `has_pipeline` |
| Pipeline Evolution | `has_pipeline_history` |
| Origination Funnel + conversion | `has_pipeline` |
| Origination Flow | `has_pipeline_history` |
| Forecast Bridge + region/LTV cuts | `has_forecast` |
| Forecast Projection — time to scale | `has_forecast_projection` |
| **Forecast Evolution — Actual vs Prior Forecast** | `has_forecast_history and forecast_periods >= 3` |
| Concentration Tests and Headroom | `has_concentration` |
| Risk Limits (legacy monitor) | `has_risk and not has_concentration` |

**OPTIONAL DEEP DIVE**

| Slide | Rule |
|---|---|
| Funded Stratifications — Secondary Dimensions (rate, vintage, broker, borrower type) | `has_stratifications and funded_balance >= 25000000` |

### Resulting composition, from the generated packs

| | New book | Seasoned book | Growing book |
|---|---:|---:|---:|
| Slides rendered | **17** | **23** | **23** |
| Movement and Drivers | omitted (no prior period) | ✓ | ✓ |
| Funded Evolution | omitted (one period) | ✓ | ✓ |
| Vintage Formation / Cohort Progression | omitted (one vintage year) | ✓ | ✓ |
| Forecast Projection / Actual vs Forecast | omitted (no history) | ✓ | ✓ |
| Pipeline block (5 slides) | ✓ | ✓ | ✓ |

No `seasoned` flag exists, no client-specific Python was written, and every
omission is recorded with a business reason in the methodology page's omission
ledger.

---

## 6. What React and PPTX now share

| Shared, one definition | Owner |
|---|---|
| Metric definition, measure, values | the governed compute functions (unchanged) |
| Dimension and bucket definitions | `config/mi/buckets.yaml` via `analytics_lib.buckets` |
| **Category display order** | `mi_agent_api.presentation.order_bars` |
| **Category label hygiene** | `mi_agent_api.presentation.clean_label` |
| **Ordinality** (ladder vs names) | `mi_agent_api.presentation.is_ordinal` |
| **Currency code and symbol** | `mi_agent_api.currency` |
| Number format | `currency.format_money` (tiles `BN/MM/K`, prose `bn/m/k`) |
| RAG vocabulary and status | the concentration evaluator |
| Multidimensional axes and cells | `snapshots.cross_tab` |
| Prior-forecast series | `evolution.forecast_evolution` |
| Reporting period, scope, constituent books | the portfolio-context resolver |
| Scale targets | `configs/pptx/investor_pack.yaml` `deck.scale_targets` |

**Deliberately not shared** — these differ because PowerPoint is static, and none
of them is in the shared payload: aspect ratio and panel geometry, annotation
density (the deck labels values on the mark because there is no hover), axis-tick
thinning, legend placement, slide composition, explanatory prose, and interaction
affordances. No pixel dimensions, PowerPoint coordinates, React component details
or matplotlib properties were put into shared configuration.

---

## 7. The parity test

`tests/test_presentation_parity.py` — 17 tests. It drives **both production
paths**: React through `TestClient(mi_agent_api.app)`, and the pack through
`POST /mi/decks/generate` → job poll → `GET /mi/decks/download`. What it inspects
is the PowerPoint a user would receive.

**Why it is split in two.** Text that reaches a slide as text — titles, KPI tile
values, tables, straplines — is read out of the `.pptx` with python-pptx. Bar-list
category order cannot be: the deck draws a bar list as a matplotlib PNG, so the
labels are pixels by the time the file exists. For those the renderers record what
they drew **at the moment they drew it** (`render.record_renders`), and the record
travels into the deck's own preflight sidecar — a production artefact the
publishing stage already writes and gates on, not a test fixture and not the
pre-render payload.

**What it catches:** currency on both surfaces (EUR and GBP), headline metric
values, governed bucket order in the payload and in the drawn chart, cross-tab
axis order, series names, slide titles and uniqueness, stratification labels,
reporting period, client identity, RAG vocabulary, and that the publication gates
passed.

**One test guards the suite itself:**
`test_the_drawn_ltv_order_is_the_ladder_and_not_the_balance_ranking` asserts that
the fixture's band order and balance order still differ — so the suite cannot
quietly lose the ability to catch the defect it exists for.

Two publication gates were added alongside, because a test that fails is a report
and a gate that fails is a deck that does not ship: `governed_bucket_order` checks
every banded bar list against the ladder, and `governed_currency` checks that no
foreign symbol reached the page.

---

## 8. Visual QA

`scripts/pptx_visual_qa.py` generates each pack through the React route, then
inspects the downloaded file for off-canvas shapes, overlapping chart images,
type below 7.5pt, orphaned slides, duplicated titles, foreign currency symbols,
bar lists out of governed order or too dense to read, and failed gates.

| Deck | Book | Currency | Pipeline | Limits | Slides | Result |
|---|---|---|---|---|---:|---|
| `artifacts/pptx_qa/new_book_gbp.pptx` | new | GBP | ✓ | ✓ | 17 | clean |
| `artifacts/pptx_qa/new_book_eur.pptx` | new | EUR | ✓ | ✓ | 17 | clean |
| `artifacts/pptx_qa/seasoned_book_gbp.pptx` | seasoned | GBP | ✓ | ✓ | 23 | clean |
| `artifacts/pptx_qa/seasoned_book_eur.pptx` | seasoned | EUR | — | ✓ | 17 | clean |
| `artifacts/pptx_qa/mixed_book_gbp.pptx` | growing | GBP | ✓ | ✓ | 23 | clean |

All five pass all 21 gates with zero findings.

**Automated checks are not the QA.** Every slide of three decks was converted and
read. Five defects were found that way, four of which no assertion would have
caught:

1. *(caught by the new gate, before I read anything)* the forecast bridge's region
   and LTV cuts were drawn in amount order — fixed upstream in
   `workspace.forecast_breakdowns`, so React's Forecast view gets it too;
2. seven forecast bands in a one-inch panel drew their labels on top of one
   another — `draw_barlist` now scales its type to the row band it has, and the
   caller shows only the rows a panel can carry;
3. the funnel rendered weekly flow **amounts** as bare numbers, indistinguishable
   from the **case counts** its single-extract fallback shows — each now carries
   its own units;
4. the Forecast Evolution slide promised "actual vs prior forecast" and drew
   neither;
5. slide one first crowded the risk strip under a chart border, and then, when
   given clearance, dropped the chart entirely — it now budgets its page.

The QA harness was extended to see the bar-list density class it had missed.

*Environment note:* LibreOffice Impress was absent from the container and had to
be installed to convert the decks; it also failed on a trivial python-pptx file
beforehand, so that was the tool, not the artefacts.

---

## 9. Regression

Every count below is measured against the starting SHA in a worktree, on the same
machine, with the same command.

| Suite | Starting SHA `e7678c8` | This branch | New failures |
|---|---|---|---|
| `tests/mi_agent_pptx/` + deck route + orchestration + publication + `mi_agent_api/tests/` | 3 failed, 1570 passed, 3 skipped | **3 failed, 1587 passed, 3 skipped** | **0** |
| `mi_agent/tests/` (MI Query Agent) + stratify | 12 failed, 982 passed, 264 skipped, 7 xfailed | **11 failed, 983 passed, 264 skipped, 7 xfailed** | **0** |
| concentration · analytics_lib · cohort identity · risk monitor · packaging | — | 1 failed, 266 passed, 17 skipped | 1 pre-existing (§10) |
| React (`vitest`) | — | **66 files, 509 tests, all passed** | 0 |
| React typecheck (`tsc --noEmit`) | — | clean | 0 |

The +17 on the first row is the new parity suite. The one fewer failure on the
second row is `test_registry_governance::test_checked_in_registry_matches_generator`,
which fails at the starting SHA and does not reproduce here; it was not touched by
this work.

**The three pre-existing failures**, each confirmed identical at the starting SHA:
`test_currency_authority::test_client_1_gbp_comes_from_the_governed_client_configuration`
(looks for a client config file the repo does not ship),
`test_chat_routing_e2e::test_cumulative_cohort_conversion_routes`, and
`test_single_parse_and_substitution::test_an_unavailable_dimension_is_refused_not_substituted`.
None was fixed here — they are outside this sprint.

---

## 10. Deferred

**In scope but not done, deliberately:**

- **A React landing page mirroring slide 1.** The executive composition is built
  from `DashboardData` with no extra compute call, specifically so it can become a
  `/mi/executive-summary` payload and a React card later. Building the React page
  was explicitly out of scope for this sprint.
- **Concentration movement on the slide.** `concentration_tests_api.compute_history`
  is now resolved onto `DashboardData.concentration_history` through the governed
  service, but no slide renders it yet. The concentration page is already dense and
  earning it a panel is a design decision worth taking deliberately.
- **Retiring the v1 dead estate** (`pptx_builder.py`, `validation.py`,
  `ChartResolver`, `StraplineResolver`, ~1,400 unreachable lines) and correcting
  `mi_agent_pptx/README.md`, whose module table still describes that dead path.
  Neither obstructed the work, and the sprint said to leave cleanup.
- **A Trakt-styled treemap or choropleth for the deck.** No renderer exists and
  neither earned a page over the visuals that do.

**Kept separate — the `analytics/` packaging finding.** `engine/orchestrator/trakt_run.py:871-878`
lazily imports four pipeline modules from `analytics/` behind a disabled config
flag; `analytics/` is excluded from both deployment artefacts, so
`tests/test_mi_api_appservice_packaging.py::test_every_reachable_distribution_is_declared`
fails on `main`. It still fails here, unchanged. It is **not** on the React PPTX
production route — that route was traced hop by hop in the provenance check and
touches none of it — so it was not fixed in this sprint. Pre-existing debt,
recorded and left alone.

---

## 11. Go-live recommendation

**Is the automated PPTX now suitable to be generated from the production React
application and supplied to a client as a polished Trakt reporting output?**

**YES.**

The React button generates it, the gates pass, the numbers reconcile to the
dashboard by a test that drives both real paths, the currency is the client's own,
the bands read in the order the business declared them, and the pack composes
itself to the book rather than to a template. Five representative packs were
generated and read end to end.

Two things a reader should know rather than discover:

- **Slide 1 has no dashboard twin yet.** It is built from governed payloads and
  cannot disagree with the pack behind it, but a client comparing screen to deck
  will find this page only in the deck until the React card is built.
- **Client branding is still only a name.** `deck.logo_path` is `null` and there is
  no per-client brand registry on either surface, so the pack is Trakt-branded with
  the client named. That was true before this sprint and is unchanged by it; if
  client branding is expected at go-live, it is a real gap — but it is a gap on
  both surfaces, not a PPTX defect.

Neither is a blocker to generating and supplying the pack.
