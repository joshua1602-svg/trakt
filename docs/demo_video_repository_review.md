# Repository review and implementation note — synthetic product demonstration

**SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER.**

This note records the assessment that preceded the demonstration build: what the
repository already implements, which execution paths were traced, what was reused
unchanged, what needed a demo-specific wrapper, and the four production seams that
had to be touched.

It is written to be useful after the fact — if the demonstration stops
reconciling, the reasoning behind each decision is here.

---

## 1. Architecture, as traced

Filenames were not trusted; each capability below was traced to the code that
performs it.

| Capability | Implementation path | Verdict |
|---|---|---|
| Gate pipeline | `engine/orchestrator/trakt_run.py` — `--mode mi` runs Gate 1 (`gate_1_alignment/semantic_alignment.py`), the canonical transform (`gate_2_transform/canonical_transform.py`), Gate 2 (`gate_3_validation/validate_canonical.py`), Gate 2.5 (`gate_2_transform/lineage_tracker.py`), Gate 3 (`gate_3_validation/validate_business_rules.py`), Gate 3b aggregation | **Reused unchanged** |
| Source provenance | `engine/provenance.py` — six stamped fields, `direct`/`acquired`, fails closed rather than assigning "unknown". `_ID_RE` requires a lowercase slug; `derive_portfolio_type` only understands the `direct_`/`acquired_` prefixes | **Reused unchanged** |
| Onboarding Agent | Two stacks exist. `agents/onboarding_agent.py` is the earlier one; `engine/onboarding_agent/` is the current operator workflow (`workflow.run_operator_workflow`, `file_profiler`, `mapping_proposer`, `gap_analyzer`, `review_pack_builder`, `central_tape_builder`, and a worked example in `demo_onboarding_v1.py`) | **Engine stack reused** for profiling and mapping |
| Assembler | `engine/platform_assembler.py` — latest accepted canonical per `source_portfolio_id`, composite-key uniqueness, a manifest carrying per-portfolio balances and content hashes | **Reused unchanged** |
| MI data source | `mi_agent_api/data_source.py` — environment-resolved. `KIND_PLATFORM_CANONICAL` applies `augment_platform_canonical_dimensions` plus bucket materialisation at read time | **Reused via environment** |
| Deterministic MI | `POST /mi/query` → `chat_routing.try_route` (compare / evolution / bridge / geo / risk / forecast / scenario) → else `mi_agent/mi_agent_workflow.run_mi_agent_query` → `mi_query_executor` → `adapters.adapt_workflow_result` | **Reused; one route added** |
| Multi-period metrics | `mi_agent_api/evolution.py` — `funded_frames`, `assemble_funded_evolution` (funded_balance, loan_count, wa_ltv, wa_interest_rate, avg_borrower_age), `funded_bridge` whose per-category deltas sum exactly to the net change | **Reused unchanged** |
| Copilot layer | `mi_agent_api/copilot_actions.py` — `askTraktMi` delegates to `/mi/query`; `getLatestInvestorDeck`; `getLatestCanonicalTape`; HMAC-signed short-lived downloads. Package in `deploy/copilot-agent/` | **Reused unchanged** |
| React MI Agent | `frontend/mi-agent-ui` — React 18, Vite, Tailwind v4, recharts/plotly. Tokens in `src/index.css`, palette in `src/lib/theme.ts` (navy `#232D55`, periwinkle `#919DD1`), mark at `public/trakt-mark.svg` | **Tokens + layout mirrored** |
| Investor deck | `mi_agent_pptx/cli.py` + `configs/pptx/investor_pack.yaml`; reads the same `/mi/*` computations the dashboard uses | **Reused unchanged** |
| Risk monitoring | `mi_agent_api/risk_limits.py` over `mi_agent/risk_monitor/` (Schedule 8 extractor, concentration, migration) | **Reused; two fixes** |
| Regulatory | `engine/gate_4_projection/regime_projector.py` (ESMA Annex 2–9), `engine/gate_5_delivery/xml_builder*.py` | **Attempted, excluded — see §5** |
| Synthetic fixtures | `synthetic_demo/` (36 loans, one period), `synthetic_onboarding_pack_domain_based/generate_pack.py`, `frontend/mi-agent-ui/scripts/generate_funded_fixtures.py` (33/73 loans) | **Reference only** — three orders of magnitude too small |
| Remotion / video | **None existed.** The only JS package was `frontend/mi-agent-ui` | **New project** |

---

## 2. The two findings that shaped the design

### 2.1 The production platform layout runs locally

`apps/blob_trigger_app/storage.py` maps `blob://{container}/{key}` onto
`{TRAKT_LOCAL_BLOB_ROOT}/{container}/{key}` when the backend is `file`. Combined
with `mi_agent_api/platform_snapshots_blob.py`, which enumerates dated platform
canonicals under a `blob://` root, this means the **production-shaped** layout

```
processed/platform/{client}/{YYYY-MM-DD}/platform_canonical_typed.csv   (dated cuts)
processed/platform/{client}/latest/platform_canonical_typed.csv         (current)
```

resolves entirely on the filesystem. The demonstration therefore gets the real
portfolio selector, the real multi-period evolution series and the real
attribution bridge with **no production change and no Azure dependency**. This is
why `demo_platform/config.mi_env()` is a handful of environment variables rather
than a shim layer.

### 2.2 The readable region label is already first in the region family

`chat_routing._REGION_FAMILY` puts `collateral_geography` ahead of the regulatory
NUTS fields, and `canonical_transform.normalize_geography` relocates readable
region labels there while deriving ITL3 codes from postcodes. So a source `Region`
column becomes a readable label ("South East") in `collateral_geography` **and** a
consistent ITL3 code in `geographic_region_*_itl3`.

The generator exploits this properly rather than working around it: each loan's
postcode district is drawn from inside the ITL1 region it claims, using the
repository's own `uk_itl_master_lookup_v2.csv`. The readable label and the derived
ITL3 code agree by construction, and a test asserts it over a sample of every
period.

---

## 3. The capability gap, and how it was closed

No single governed route answered *"what has changed versus the prior month?"* as a
composite. `_route_compare` answers one metric at a time; `_route_bridge` answers
attribution only. Worse, for that exact phrasing the deterministic parser produces
`compare_periods = ['latest', 'prior month']`, so `_route_compare` renders "moved
from £1.96bn in 2026-06 to £1.95bn in 2026-05 — a change of £18.1m (down)": the
right magnitude, the wrong direction, and a chronologically reversed sentence.

Two options were rejected:

- **fake the answer in Remotion** — the film would state a figure the product does
  not produce, which is exactly what the accuracy rules forbid;
- **rewrite `_route_compare`** — it serves explicit two-period comparisons
  correctly and has existing tests; changing its wording to suit a narrower
  phrasing would be a regression risk with no upside.

What was built instead: `mi_agent_api/movement_summary.py`, which **composes** the
existing services (`funded_evolution` for the metrics, `funded_bridge` for the
attribution, `portfolio_lens` for the per-book scoping) into two governed
answers — a current-period portfolio summary and a month-on-month movement. It
introduces no new metric definition. `chat_routing` routes only the narrow
"summarise the portfolio" and "what has changed versus the prior *period*"
intents to it, and both return `None` (or a controlled insufficient-data envelope)
when they cannot answer, so every existing route is untouched.

One discipline worth recording: the movement answer only says *"driven by
completions in the South East"* when that is measurable. `period_movement`
computes the balance of loans originated **inside** the reporting month within the
primary region, and the completion wording is used only when those loans account
for the majority of that region's increase. On the current data that is 64%. If it
were not, the answer would fall back to a neutral "the largest contribution came
from…", and the assertion gate checks the evidence rather than the wording.

---

## 4. Production changes

Four files, all additive, all off by default. Each was needed to make the
demonstration honest rather than to make it easier.

| File | Change | Why |
|---|---|---|
| `engine/gate_1_alignment/semantic_alignment.py` | `--extra-aliases-dir` (repeatable) layers a client's approved onboarding-contract aliases on top of the global files; an overlay entry wins, and every override is recorded in the Gate 1 report. Fails loudly on a missing directory rather than running with a silently empty contract | Two genuinely different source schemas need ~24 client-specific header decisions. Adding them to `config/system/aliases_*.yaml` would widen every other client's matching surface — precisely what the brief says not to do |
| `engine/orchestrator/trakt_run.py` | `--extra-aliases-dir` passthrough, resolved to an absolute path before handing it to the Gate 1 subprocess | Gate 1 resolves relative paths against its own module directory, so a relative path would have silently produced an empty overlay (it did, on the first attempt) |
| `mi_agent_api/movement_summary.py` (new) + `chat_routing.py` | the composed summary and movement routes described in §3 | Closes the gap without duplicating calculation logic or diverging the two surfaces |
| `mi_agent/portfolio_lens.py` | an **explicit** lens selection may name any provenance-valid `source_portfolio_id`, not only the `direct_NNN`/`acquired_NNN` convention. Natural-language detection is unchanged | `/mi/source-portfolios` already exposes any provenance slug as a selectable cohort, but `lens_from_selection` silently fell back to *total* for anything outside that convention — a portfolio you could see in the dropdown but not scope the chat to |

Plus two fixes found while wiring the risk-monitoring output, both in
`mi_agent_api/risk_limits.py`:

- **blob-root awareness.** `compute_risk_limits` resolved its funded frame through
  `snapshots.discover_snapshots`, which is filesystem-only and keyed on the
  onboarding `18_` tape layout. Under a `blob://` platform root it enumerated
  nothing, so every limit was reported "unavailable" against a book that was
  plainly there. It now falls back to `evolution.funded_frames` — the same
  blob-aware resolution the evolution, bridge and compare services already use, so
  the observed concentrations reconcile to the dashboard.
- **region column.** A concentration limit is written against a *readable* region
  ("the South East"), but the monitor tested it against `geographic_region_obligor`,
  which after the transform holds ITL3 codes. A "South East ≤ 30%" limit was
  therefore reporting 0.0% against a book that is 26.3% South East. It now uses the
  same region-column preference order as the bridge.

Both are genuine defects that would affect any client on the blob platform path,
not demo scaffolding.

---

## 5. What was excluded, and why

- **ESMA Annex 2 delivery.** `demo_platform/artefacts.regulatory_output()` runs the
  projector so the outcome is a measured fact rather than an assumption. It stops
  at the enum-review gate: `Field 'purpose' has unmapped enum values in strict
  Annex2 mode: ['NULL']`. That is correct production behaviour — a regime
  projection must not proceed on unreviewed enums — so the artefact catalogue
  records it as unavailable with that exact reason and Scene 7 omits the card. The
  demonstration does not claim a regulatory delivery it did not perform.
- **`/mi/snapshot` month-on-month deltas.** That endpoint resolves its prior run
  through the same filesystem-only walk described above. Rather than change a
  second production endpoint to suit the film, Scene 5 takes its deltas from the
  governed movement service. Recorded as a limitation.
- **The Tier 7 LLM field mapper.** Requires human confirmation before any mapping
  is applied (`engine/gate_1_alignment/agent_orchestrator.py`), and the build runs
  offline and deterministically. Headers the deterministic tiers cannot resolve are
  shown as referred for review and closed by the client's alias contract — the same
  review-first path, without the LLM.
- **The legacy Streamlit UI.** The React MI Agent is the intended client interface,
  so `analytics/streamlit_app_erm.py` and `mi_agent/streamlit_mi_agent.py` do not
  appear in the film.

---

## 6. Scale and the engineered narrative

The existing synthetic fixtures carry 36 and 73 loans over a single period, which
cannot support a consolidated month-on-month narrative. The generator models
**11,035 current loans across two books and three month-ends** (7,126 origination,
3,909 acquired), which is fast enough for a full pipeline run in ~120 seconds while
being commercially credible.

The narrative is produced by **engineering the source data**, never by patching an
output. `demo_platform/generator.py` models each loan's own economics — advance,
fixed roll-up rate, regional valuation, completion date — and derives every
reported figure from it, so LTV always reconciles to balance and valuation and
month-on-month balances always reconcile to the engineered movement. A
deterministic calibration then solves four scalars (an initial-LTV shift, an
origination-age shift, Portfolio A's new-business volume, Portfolio B's
reserve-drawdown volume) against the same formulas the pipeline applies, and the
solved values are recorded in the demo manifest.

One economic point is worth recording because it constrains the whole design: a
lifetime-mortgage book's balance grows by interest roll-up and shrinks by
redemption, and on a *closed* book those two roughly cancel — net growth is
~0.04% a month. Portfolio B therefore cannot contribute £6.6m of a £18.2m monthly
increase from roll-up alone. Its growth comes from roll-up **plus
reserve-facility drawdowns** (further advances on existing loans), which is a real
and ubiquitous feature of the product, at ~2.7% monthly utilisation. Portfolio A
grows through completions plus roll-up, less redemptions.

`demo_platform/assertions.py` then re-verifies the produced pipeline output
independently — 33 checks — and the run fails closed on any drift.

---

## 7. Reuse ledger

**Reused directly, no wrapper:** `trakt_run.py`, `canonical_transform.py`,
`validate_canonical.py`, `validate_business_rules.py`, `lineage_tracker.py`,
`platform_assembler.py`, `provenance.py`, `file_profiler.py`, `HeaderMapper`,
`evolution.py`, `movement_summary.py`, `chat_routing.py`, `adapters.py`,
`mi_query_executor.py`, `funded_prep.py`, `data_source.py`, `snapshots.py`,
`platform_snapshots_blob.py`, `geo.py`, `risk_limits.py`, `decks.py`,
`copilot_actions.py`, `mi_agent_pptx/cli.py`, `storage.py`,
`uk_itl_master_lookup_v2.csv`, `config/system/fields_registry.yaml`,
`config/system/aliases_*.yaml`, `config/system/enum_synonyms.yaml`,
`config/mi/buckets.yaml`.

**Needed a demo-specific wrapper or fixture:** the source generator (no generator
at this scale existed); the two source schemas; the per-portfolio alias contracts;
the demo client master config; a synthetic Schedule 8 so the risk extractor has a
governing document to parse; the metric/surface export; the reconciliation and
safety gates; and the Remotion project.

**Mirrored, not imported:** the visual identity. `demo-video/src/design/tokens.ts`
carries the palette and type tokens verbatim from
`frontend/mi-agent-ui/src/lib/theme.ts` and `src/index.css`, because the Remotion
bundle does not run the product's Tailwind build. If the product's palette changes,
change it there and mirror it here — the file says so at the top.
