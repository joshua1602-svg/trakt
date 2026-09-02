# Pipeline stage transitions — React and PPTX exposure

Sprint: exposing the governed stage-transition capability through the **existing**
architecture. Presentation only — no analytical code.
Branch: `claude/pipeline-stage-transition-engine-2ewble`
Starting `main`: `d2ca730` (*Merge pull request #383 — pipeline stage transition engine*).
Engine report: `docs/reports/pipeline_stage_transition_engine.md`.

---

## 1. Executive verdict

**YES.** The capability is exposed through the architecture that already existed.

**NEW HTTP ROUTES ADDED: 0** — the app's route set is byte-identical to
`d2ca730` (39 routes before, 39 after, `diff` clean). Both consumers read the
same governed object, and a parity test asserts the deck's payload and the
dashboard's HTTP response are *the same object*, not two agreeing computations.

One correction to the brief's premise, stated up front because it changed what
"upgrade" could mean: **there was no existing React stage-movement panel and no
existing "Pipeline Stage Movement" slide in this repository.** The brief asks to
upgrade both. Neither exists — `grep` for `stage.movement|StageMovement|
stage_movement` across `.tsx`/`.ts`/`.py` returns only `insight_generators.py`
(a completions-movement insight) and the engine sprint's own test. The deck's
slide inventory has `pipeline_summary`, `pipeline_evolution`, `funnel`,
`origination_flow` and `movement_drivers` (funded, not pipeline); React has
`PipelineSnapshotPanel`, `EvolutionPanel` and the `insight/` hover components.

So the end state the brief describes — a Pipeline Stage Movement surface driven
by the governed transition payload — was **created** rather than upgraded, in
both channels, inside the existing composition architecture. Nothing was
displaced, and the net-movement surfaces that do exist are untouched.

---

## 2. Existing architecture traced

Four comparable capabilities traced end to end before any code was written.

### A. Movement detail — the closest comparable (same engine module)

```
movement_detail.resolve_movement_detail(detailType)
  → GET /mi/insight/movement-detail?detailType=…            [app.py:1071]
  → HttpAgentClient.getMovementDetail()                     [one route, N types]
  → useMovementDetail()  (debounce, dedupe, cache)
  → MovementDetail  [domain/insightDetail.ts]
  → EnhancedMetricTooltip / InsightDetailDrawer
```

**The decisive finding.** This route was already parameterised by detail type
and already validated `detailType` against a tuple of two. It is the existing
semantic owner of "what changed between two governed snapshots", and its
extension point is a third member of that tuple. Not consumed by PPTX today.

### B. Pipeline evolution

```
evolution.pipeline_evolution
  → GET /mi/evolution/pipeline → getPipelineEvolution → EvolutionPanel   [React]
  → mi_api._pipeline_evo → DashboardData.pipeline_evolution
      → composition guard "pipeline_evolution" → deck.slide_pipeline_evolution
      → configs/pptx/investor_pack.yaml                                   [PPTX]
```

### C. Origination funnel / flow

```
evolution.pipeline_funnel_evolution
  → GET /mi/evolution/funnel → getFunnelEvolution → EvolutionPanel funnel cards
  → mi_api._funnel → DashboardData.funnel
      → guards "funnel" / "origination_flow" → slide_funnel / slide_origination_flow
```

### D. Concentration tests

```
concentration_tests_api
  → GET /mi/concentration-tests → getConcentrationTests → ConcentrationDetailPanel
  → mi_api._concentration → DashboardData.concentration
      → guard "concentration" → deck.slide_concentration
```

### The pattern both channels share

* **React** reaches a governed engine result through **one HTTP route per
  capability family**, via `AgentClient` → `CachingAgentClient` → a hook → a
  domain type → a component.
* **PPTX** never speaks HTTP. It calls the **same Python engine functions**
  directly, stores each result on the shared `DashboardData` aggregate, gates
  the slide in `composition.py`, renders in `deck.py`, and declares the slide in
  `configs/pptx/investor_pack.yaml`.

Stage transitions were fitted to exactly this, with no new mechanism in either
channel.

---

## 3. Exposure owner

| Question | Answer |
|---|---|
| 1. Which response already represents pipeline movement detail? | `GET /mi/insight/movement-detail`, built by `movement_detail.py` — the same module the engine lives in |
| 2. Who already fetches it? | React, via `HttpAgentClient.getMovementDetail` → `useMovementDetail` → the hover/drawer components |
| 3. Can the transition detail be added additively? | **Yes** — as a third `detailType`. The route already branches on it; the two existing types are byte-unchanged |
| 4. Can React consume it without new HTTP surface? | **Yes** — the route it already calls, the client it already holds, the caching layer it already uses. One request when the panel is shown, exactly as the movement hover works |
| 5. Can PPTX consume it through its existing `mi_api`? | **Yes** — `_stage_transitions()` beside `_pipeline_evo()` / `_funnel()`, onto the shared `DashboardData` |
| 6. Does this preserve one computation / one contract? | **Yes** — one engine function, one payload; both channels read it verbatim |

**Not chosen, and why.** Adding the block to `/mi/pipeline/snapshot` would have
let React consume it with no request at all, but that route is a *single-snapshot*
contract; transitions are a two-snapshot capability, and computing them on every
snapshot request would put the cost on a hot route and change a shape every
existing consumer reads. The movement route already owns two-snapshot semantics,
so it is the smaller and more honest seam.

---

## 4. Route impact

**NEW HTTP ROUTES ADDED: 0**

Proved, not asserted: the full route set was dumped at `d2ca730` and at HEAD and
diffed — identical, 39 routes both sides. A permanent test
(`TestNoNewRoute`) fails if any path containing `transition` or `stage-movement`
ever appears, and pins that exactly one endpoint contains `movement`.

The capability also rides the **existing** feature flag
(`TRAKT_MI_ENHANCED_HOVERS`) rather than introducing a second switch, and an
unknown `detailType` is still a 400.

---

## 5. Shared payload

The route returns `resolve_stage_transition_detail(...)` **verbatim** —
`test_the_payload_equals_resolve_stage_transition_detail` compares the whole
object. Nothing is flattened, renamed or dropped:

`detail_type`, `available`, `reason`, `reason_code`, `identifier`, `measure`,
`stage_field`, `as_of_date`, `comparison_date`, `counts`, `transitions[]`,
`new_arrivals[]`, `stayers[]`, `departures[]`, `event_totals`,
`reconciliation{by_stage[], count_reconciliation_residual,
amount_reconciliation_residual, global, amount_tolerance, count_identity,
amount_identity}`, `methodology`, `source_dates`, `sources`.

The React domain type (`StageTransitionDetail` in `domain/insightDetail.ts`)
mirrors it field for field. No concept was renamed in either consumer.

---

## 6. React

**Created** (no existing panel to upgrade — see §1), inside the existing view:

| File | Role |
|---|---|
| `domain/insightDetail.ts` | `+160` lines, additive types beside the existing `MovementDetail` |
| `api/AgentClient.ts` | one optional method on the existing interface |
| `api/HttpAgentClient.ts` | calls the **same** `/mi/insight/movement-detail` route |
| `api/CachingAgentClient.ts` | same dedupe/cache discipline as movement detail |
| `api/MockAgentClient.ts` | demo dataset refuses with the engine's own reason code |
| `hooks/useStageTransitionDetail.ts` | the movement hook's discipline, without the hover debounce |
| `components/pipeline/StageTransitionPanel.tsx` | the panel |
| `components/EvolutionPanel.tsx` | `+8` lines — mounted under the existing "Pipeline by stage over time" chart |

**Why a new component was justified.** The brief says not to add one unless the
current component cannot reasonably consume the payload. `EnhancedMetricTooltip`
and `InsightDetailDrawer` are typed to `MovementDetail` — a headline metric plus
ranked contributor lists. A transition matrix is a different shape, not a
different dataset, so they cannot render it without becoming a union of two
unrelated contracts. Everything *around* the new component is reused: route,
client, caching, feature flag, view, and the `pipeline/bits.tsx` visual idiom.

**Design.** Four side-by-side blocks — **Moved stage**, **New arrivals**,
**Stayed in place**, **Left the pipeline** — over a per-stage
opening→closing reconciliation table, with a Cases/Value toggle and a footnote
carrying the identifier, the population and both residuals. Deliberately not a
Sankey: the repo has no Sankey grammar, and four labelled blocks make the
distinction the brief actually requires (arrival vs stayer vs transition vs
departure) more legible than a flow diagram would.

The Cases/Value toggle serves §6 from **one payload** — the test asserts the
client is called exactly once across both measures.

---

## 7. PPTX

**Created** as slide 14b, "Pipeline Stage Movement", immediately after Pipeline
Evolution — the gross companion to that slide's stock view.

| File | Change |
|---|---|
| `mi_api.py` | `DashboardData.stage_transitions` field + `_stage_transitions()` resolver, beside `_pipeline_evo` / `_funnel` |
| `composition.py` | `has_stage_transitions` fact + a guard that surfaces the **engine's own reason** on omission |
| `deck.py` | `slide_stage_transitions` + `_DISPATCH` entry |
| `configs/pptx/investor_pack.yaml` | the slide spec |

**Layout** (verified by rendering, §11):

* top-left — **Cases that moved stage**: Movement / Cases / Prior / Latest / Change
* top-right — **Arrivals, stayers and departures**: every non-transition event
* bottom-left — **Case reconciliation by stage**: Open / New / In / Out / Left / Close
* bottom-right — **Stage value — opening vs closing**
* footnote — identifier, population, both residuals, and the identity note

**A defect found and fixed during visual QA.** The value card first showed
`Opening | New | Amended | Closing`. That is a *partial* identity — the
transferred-in and transferred-out terms are missing — so the columns did not
add up on the page and invited the reader to conclude the numbers were wrong.
The full seven-term identity cannot be rendered legibly in half a slide, so the
card now shows opening and closing stock only, titled as a comparison rather
than a reconciliation. The value identity is still proved: by the residual in
the footnote, and in full in the payload and its tests.

---

## 8. Parity

`tests/mi_agent_pptx/test_stage_transition_parity.py` — the discipline
`test_channel_parity.py` already applies to the funded surfaces, pointed at this
capability. It drives the **real React HTTP route** and the **real deck data
path** over one fixture, both pointed at the same governed pipeline pack.

The primary assertion is the strongest available form:

```python
def test_the_whole_governed_object_is_identical(deck, react):
    assert deck.stage_transitions == react()      # modulo resolution provenance
```

Everything below it is a readable restatement, so a failure names the block:

| Requirement | Test | Fixture truth |
|---|---|---|
| identifier | `test_the_identifier_matches` | `pipeline_case_identifier` |
| source→destination | `test_the_source_destination_transitions_match` | 3 flows |
| counts | `test_the_transition_counts_and_amounts_match` | 2 / 2 / 1 |
| KFI→Application | `test_the_kfi_to_application_flow_matches` | 2 cases, £920,000 latest |
| Application→Offer | `test_the_application_to_offer_flow_matches` | 2 cases, −£10,000 |
| completion | `test_the_completion_flow_matches` | Offer→Completed, 1 case |
| arrivals | `test_the_arrivals_match` | KFI 1, Application 1; **no `source_stage` on either side** |
| stayers + amendments | `test_the_stayers_and_their_amendments_match` | KFI +£20,000, Application −£20,000 |
| departures + outcomes | `test_the_departures_and_their_outcomes_match` | 2 governed, 2 `unclassified_departure` |
| residuals | `test_the_reconciliation_and_residuals_match` | 0 cases / 0.00 value |

No consumer value is ever compared against a separately recalculated one.

**The React fixture is the engine's real output**, generated by
`frontend/mi-agent-ui/scripts/generate_stage_transition_fixture.py`, and
`test_the_committed_react_fixture_still_equals_the_engine` re-runs the engine and
compares — so the React suite cannot go on passing against a contract the
backend no longer produces.

---

## 9. Unavailability

Neither consumer decides availability. Both read the engine's typed result.

| State | React | PPTX |
|---|---|---|
| `no_prior_snapshot` | `stage-transitions-unavailable`, engine's `reason` shown | slide omitted, engine's reason in the omission ledger |
| `missing_case_identifier` | same | same |
| `duplicate_case_identifiers` | same | same |
| `no_governed_cases` | same | same |
| request rejected (layer off, 404, abort) | "no detail", never a visible error | n/a |

A refusal is **never** rendered as an empty matrix, which a reader would take as
"nothing moved" — asserted in both channels.

**A design flaw found and fixed by the tests.** The slide spec first carried
`when: has_stage_transitions`. That coarse condition short-circuited composition
and produced *"the reporting condition 'has_stage_transitions' was not met"* —
burying the engine's actual reason. The `when` is now the coarse
`has_pipeline` gate and the guard carries the engine's own wording, so an
omission reads *"2 duplicate pipeline_case_identifier value(s)…"*. A test pins
it.

---

## 10. Existing-output non-regression

Every governed output both channels produce was dumped at `d2ca730` and at HEAD
over the same fixture book and compared byte-for-byte:

funded snapshot · **pipeline overview** · **pipeline stratifications** ·
forecast (incl. **weighted expected funded**) · funded evolution ·
**pipeline evolution** · **funnel / conversion** · forecast evolution ·
**cohorts** · cohort progression · cohort series · geo · risk ·
**concentration** · extrapolation · multidim · insights · watchlist ·
movement · plus the React channel's `pipeline_evolution` and
`pipeline_funnel_evolution`.

**Result: identical — 134,213 bytes, `sha256 c43a556a…`.**

The two existing detail types are additionally pinned by test: same
`detail_type`, still carrying `contributors` and `components`, and still
carrying **no** `transitions` key.

---

## 11. Visual QA

**React** — the real component, rendered in a real browser (Chromium via
Playwright) against the engine fixture, at 1280px, in both measures.

* Four populations visually distinct; `unclassified_departure` rows dimmed so an
  unevidenced exit cannot be mistaken for a governed one.
* No clipping — the longest label wraps rather than truncating.
* Reconciliation reads correctly on the page: KFI `4 + 1 + 0 − 2 − 0 = 3`.
* No synthetic stage: arrivals render as "New into KFI", never `KFI → KFI`.
* Value view shows `+£20K` green / `−£20K` red on stayers; one payload, one
  request.

**PPTX** — built through the **real production CLI**
(`mi_agent_pptx.cli.run` with `configs/pptx/investor_pack.yaml`): 21 slides,
0 placeholders, **Preflight PASS — 19 gates, 0 failures, 0 warnings**. The four
rendered panels were extracted from the deck and inspected as images.

Two defects were found this way and fixed:

1. **Label collision.** "Left from Application — outcome not evidenced" overran
   the Cases column. Shortened to "Left from Application — unclassified" —
   the engine's own word, so no meaning was lost — and re-rendered to confirm
   clearance. `Application` is the longest canonical stage, so this is the worst
   case. React was aligned to the same wording so the channels read alike.
2. **Invisible amendment.** Application→Offer rendered `£1.3MM → £1.3MM`,
   hiding the −£10,000 change at compact scale. A **Change** column was added;
   it now reads `-£10K`.

Checks passed: no clipping, readable stage labels, readable counts and amounts,
no impossible flows, source/destination unambiguous, totals reconcile visually,
no synthetic stage presented as a real one.

*(LibreOffice could not load the generated `.pptx` for PDF conversion in this
container. The slide's content is rendered PNG panels, so those were extracted
from the deck itself and inspected directly — the same pixels a reader sees.)*

---

## 12. MI Query

**MODIFIED: NO.**

No parser, recogniser registry, query spec, executor, chat routing, vocabulary
or question bank was touched — confirmed by the changed-file list, and enforced
permanently by `TestNoQueryAgentChange`, which fails if any Query file ever
mentions `PIPELINE_STAGE_TRANSITION`, `stage_transition` or
`resolve_stage_transition_detail`.

---

## 13. Regression — baseline `d2ca730` vs HEAD

| Scope | Baseline | HEAD | New failures |
|---|---|---|---|
| Targeted (movement detail + api, pipeline prep/stock/evolution/source/runtime, forecast, weekly brief, serving cache, stage contract, **all PPTX**) | 500 passed, 2 failed, 2 skipped | 518 passed, 2 failed, 2 skipped | **0** |
| Broad `mi_agent_api/tests` (76 files) | 1376 passed, 3 failed, 1 skipped | 1393 passed, 3 failed, 1 skipped | **0** |
| Repo-level pipeline **+ MI Query replay** surface (27 files) | 352 passed, 26 failed, 125 skipped | 352 passed, 26 failed, 125 skipped | **0** |
| React (`vitest`) | 509 passed | 526 passed | **0** |
| React typecheck (`tsc --noEmit`) | clean | clean | — |

Failure sets were **diffed, not merely counted**, and are identical on both
sides at every scope. The pre-existing failures (2 PPTX omission-ledger, 3
`mi_agent_api`, 26 repo-level) are unrelated to this sprint and were not
repaired.

Pass counts rise by exactly the new tests: +18 parity, +17 exposure, +17 React.

*Environment note.* Test dependencies are absent from a fresh container and were
installed to run anything at all (Python: pandas pinned to the `<3.0.0` range
`requirements.txt` declares, plus the analytics and FastAPI stack; Node: `npm
install`). A `npm i -D playwright` for the screenshot step briefly modified
`package.json`/`package-lock.json`; that was reverted and Playwright installed
in scratch space instead — the manifests are unchanged in the diff. No
repository file was changed for any of this, and both baseline and HEAD were
measured in the identical environment.

---

## 14. Sprint 3 readiness

**YES.** The shared governed result is stable enough for the existing
stage-movement question bank to route against.

* One capability, one payload, one `detail_type`, reachable by one route that
  Query can call with no new plumbing.
* Both presentation consumers now depend on that exact shape and would fail
  loudly if it changed — the contract is pinned from three directions (engine
  tests, exposure tests, parity tests) rather than by convention.
* Availability is typed and enumerable (`no_prior_snapshot`,
  `missing_case_identifier`, `duplicate_case_identifiers`,
  `no_governed_cases`), so a router can refuse in the existing pattern.
* Stage tokens remain the canonical vocabulary the question layer already reads.
* The payload answers gross source→destination, arrivals, stayers, departures,
  amendments and per-stage opening/closing directly — no derivation required of
  the Query layer.

---

## 15. Merge recommendation

**YES.** Additive; 0 new routes; the route set, and every existing governed
output in both channels, byte-identical to `d2ca730`; zero new test failures at
every scope; both surfaces visually QA'd through their real production paths;
MI Query untouched.

---

## Return schedule

| # | Item | Answer |
|---|---|---|
| 1 | Starting main SHA | `d2ca730` |
| 2 | Branch | `claude/pipeline-stage-transition-engine-2ewble` (restarted from merged main) |
| 3 | Commits | one |
| 4 | Comparable patterns inspected | movement detail; pipeline evolution; origination funnel/flow; concentration tests — all traced engine → owner → payload → React component, and → `mi_api` → `DashboardData` → composition → slide |
| 5 | Existing service/payload extended | `GET /mi/insight/movement-detail` (React) and `DashboardData` via `mi_api._stage_transitions` (PPTX) |
| 6 | New routes added | **0** — route set diffed identical, 39 both sides |
| 7 | Engine capability consumed | `resolve_stage_transition_detail`, `detail_type = PIPELINE_STAGE_TRANSITION` |
| 8 | React component | `pipeline/StageTransitionPanel.tsx`, mounted in the existing `EvolutionPanel` pipeline view (no prior panel existed — see §1) |
| 9 | PPTX slide | `stage_transitions` → "Pipeline Stage Movement", after Pipeline Evolution (no prior slide existed — see §1) |
| 10 | Transition data shown | source→destination matrix with cases, prior, latest, change; arrivals; stayers; departures; per-stage opening→closing; residuals |
| 11 | New-arrival handling | `source_stage` stays null; rendered "New into &lt;stage&gt;"; never `KFI → KFI` — asserted both channels |
| 12 | Departure handling | engine's `governed_outcome`; "Left after Completed/Withdrawn" where evidenced, "Left from &lt;stage&gt; — unclassified" where not; never inferred |
| 13 | Stayer handling | one row, same stage, amendment shown as a signed change — never an exit plus an arrival |
| 14 | Amendment handling | surfaced as net amount change on persisting cases; no repricing/scope-change attribution invented |
| 15 | Unavailable behaviour | engine's typed `reason_code` + `reason`; React shows the controlled unavailable state, PPTX omits the slide with the engine's own reason in the ledger |
| 16 | KFI→Application parity | deck == dashboard: 2 cases, £900,000 → £920,000, +£20,000 |
| 17 | Application→Offer parity | deck == dashboard: 2 cases, £1,300,000 → £1,290,000, −£10,000 |
| 18 | Completion parity | deck == dashboard: Offer→Completed, 1 case, £800,000 |
| 19 | Global React/PPTX parity | whole governed object identical (`assert deck.stage_transitions == react()`), 18 parity tests green |
| 20 | Pipeline stock unchanged | **confirmed** — byte-identical |
| 21 | Pipeline evolution unchanged | **confirmed** — byte-identical |
| 22 | Forecast unchanged | **confirmed** — byte-identical (incl. weighted expected funded) |
| 23 | MI Query modified | **NO** — enforced by test |
| 24 | Visual QA | React screenshotted in Chromium (both measures); PPTX built via the real CLI (21 slides, preflight PASS) and its panels inspected as images; two defects found and fixed |
| 25 | Targeted tests | 35 new Python (17 exposure + 18 parity) + 17 React, all green; targeted suite 500 → 518 |
| 26 | Broad regression | `mi_agent_api` 1376 → 1393 (same 3 pre-existing); repo-level pipeline + Query replay identical (352 passed / 26 pre-existing); React 509 → 526 |
| 27 | Report path | `docs/reports/pipeline_stage_transition_surfaces.md` |
| 28 | Merge recommendation | **MERGE** |

### Also not modified

Query parser, recogniser registry, query spec, executor, chat routing, question
banks; the engine (`movement_detail.py` analytical code, `pipeline_prep.py`);
pipeline-stock semantics; forecast methodology; concentration; cohort; funded
analytics; Annex 2; canonical schema; field registry. No new analytical owner
was created in either consumer: React and PPTX order, label and format, and
compute no transition, arrival, departure, stayer, amendment or reconciliation
of their own.
