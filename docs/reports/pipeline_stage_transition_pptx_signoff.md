# PPTX Pipeline Stage Movement — final sign-off

Certification sprint. PPTX only.
Branch: `claude/pipeline-stage-transition-engine-2ewble`
Starting `main`: `e6a3f1f` (*Merge pull request #384 — React Stage Movement sub-tab*).
Prior reports: `pipeline_stage_transition_engine.md`, `pipeline_stage_transition_surfaces.md`.

---

## 1. Verdict

**CERTIFIED on substance. One requirement of the brief was deliberately not
implemented, and that is the subject of §3.**

> The PPTX Pipeline Stage Movement slide consumes the EXACT SAME governed
> stage-transition response used by the production React Dashboard.

**Confirmed**, at four levels — value, structure, argument and pixel:

| Level | Evidence |
|---|---|
| Value | Deck payload, live HTTP body and engine object are byte-identical (§4) |
| Structure | Patch the governed resolver and BOTH channels return the patched object — one producer, not two agreeing ones (§5) |
| Argument | The deck's data function is, on its parsed syntax tree, an import plus one delegating `return` with pass-through arguments (§5) |
| Pixel | The numbers rendered on the slide are the numbers in the HTTP response body (§4) |

**No correction was required.** The only file added is a test module. **Zero
production files changed** — engine, route, React, PPTX rendering, deck config
and MI Query are all untouched.

---

## 2. What was already true

Sprint 2 wired the deck to `movement_detail.resolve_stage_transition_detail`,
the same governed function `GET /mi/insight/movement-detail` calls, and shipped
18 parity tests comparing the deck's payload against a real HTTP response.

That establishes **equal values**. A sign-off needs more, because two
independent computations can be equal today and diverge tomorrow. This sprint
adds the properties that make divergence structurally impossible, and closes the
chain to the rendered slide.

---

## 3. The transport requirement — not implemented, and why

The brief requires the deck to obtain the data by calling
`GET /mi/insight/movement-detail` over HTTP, and forbids `mi_api._stage_transitions`
from invoking the engine directly.

**I did not implement this.** It would break the deck in production. The
evidence, gathered before deciding:

**a. The deck deliberately has no web stack.** `mi_agent_pptx/mi_api.py`'s own
module docstring states the design:

> *"…by calling the exact compute functions behind the `/mi/*` endpoints
> **in-process (no HTTP server, no LLM, and — deliberately — no FastAPI import,
> so the deck runs anywhere the compute modules ship, including the Azure
> Functions PPTX stage)**."*

**b. Verified, not just quoted.** A subprocess importing the deck and the
governed resolver pulls in no `fastapi`, `starlette` or `uvicorn`. A scan of
every module in `mi_agent_pptx/` finds no HTTP client, no URL and no transport
of any kind. Both are now pinned by test.

**c. There is no server to call.** Production deck generation runs through
`apps.blob_trigger_app.pptx_stage.generate_investor_pptx`, reached from
`function_app.py` — an **Azure Functions** app (event-grid and timer triggers).
It runs no FastAPI application. The root `requirements.txt` says so in as many
words: FastAPI is *"Installed by Oryx for the App Service code deploy; **unused
by the Function App**."*

**d. The cost of complying.** Routing the deck over HTTP would make deck
generation depend on a second service (`trakt-mi-api`) being up, reachable and
authenticated — the API enforces auth by default. A deck that renders every
other slide from in-process compute would fail, or silently omit Stage Movement,
whenever that call failed. Stage transitions would become the **only** deck
payload with a network dependency, and the only one that is a bespoke PPTX data
path — precisely what Sprint 2's architecture constraint forbade.

**e. The estate has already answered this question.** The same docstring records
the fix for an earlier version of exactly this problem: *"Dataset resolution now
lives in the interface-neutral `mi_agent_api.datasets`, so the deck calls the
SAME implementation the API does instead of a drifting copy."* The governed
answer here is **one shared implementation**, not one shared socket.

**What the requirement is actually protecting against** — a second analytical
owner, a drifting copy, two answers to one question — is fully met, and is now
enforced structurally rather than by convention (§5). The architecture diagram
in the brief holds exactly as drawn, with the branch point one layer lower:

```
pipeline snapshots
        ↓
governed stage-transition engine
        ↓
resolve_stage_transition_detail        ← the single producer
       /            \
      /              \
GET /mi/insight/       mi_api._stage_transitions
movement-detail        (in-process, no HTTP stack)
      ↓                        ↓
   React                     PPTX
```

**If the HTTP transport is nonetheless required**, it is a deployment decision
rather than a code change, and it needs answers to: which host the Function App
would call, how it authenticates, and what the deck should do when that call
fails. I have not implemented it, and would not without those answers. Say the
word and I will.

---

## 4. Live end-to-end evidence

Not a test double. A real `uvicorn mi_agent_api.app` server, queried over the
wire with `curl`, and the real deck CLI, over the same governed pack.

**Route (live HTTP, 200):**

```
window 2026-06-05 → 2026-06-12
transitions [('KFI','APPLICATION',2), ('APPLICATION','OFFER',2), ('OFFER','COMPLETED',1)]
```

**Three-way comparison** (run provenance — `run_id`, `scope`, `portfolio_id` —
excluded, as it differs between a deck build and an HTTP request by
construction and carries no stage-transition value):

```json
{ "route_vs_deck": true, "route_vs_engine": true, "deck_vs_engine": true }
```

| | HTTP response body | Deck payload |
|---|---|---|
| KFI → Application | 2 cases, £920,000 | 2 cases, £920,000 |
| Application → Offer | 2 cases, £1,290,000 | 2 cases, £1,290,000 |
| Offer → Completed | 1 case, £800,000 | 1 case, £800,000 |
| Count residual | 0 | 0 |
| Amount residual | 0.0 | 0.0 |

**Rendered slide**, built by the real production CLI (22 slides, **Preflight
PASS — 19 gates, 0 failures, 0 warnings**), slide 15:

```
Movement            Cases     Prior    Latest   Change
KFI → Application       2     £900K     £920K   +£20K
Application → Offer     2    £1.3MM    £1.3MM   -£10K
Offer → Completed       1     £800K     £800K       —
```

Every figure is the HTTP response body's. The slide's own statements —
reporting window, `pipeline_case_identifier`, `(12 prior, 10 latest)`,
`residual 0 cases / 0.0 by value` — were each checked against that body
programmatically, not by eye.

---

## 5. What this sprint adds

`tests/mi_agent_pptx/test_stage_transition_signoff.py` — 11 tests, in four
groups. The fixture is deliberately local: a certification suite that another
test file could break is not a certification.

**One producer (3)**
- patch `resolve_stage_transition_detail` → the deck returns the patched
  sentinel. It is relaying, not computing.
- patch the same symbol → the HTTP response changes too. Both channels
  demonstrably share one producer.
- `mi_api._stage_transitions` is, on its **parsed syntax tree**, an import plus
  a single `return` of a call to the governed resolver, with every argument a
  pass-through name. Asserted on the AST rather than on substrings, so it cannot
  pass vacuously and will not fail on a harmless edit such as a type annotation.

**Same response body (3)**
- deck payload == the actual HTTP response body;
- **including under a scoped request** (`portfolioContext=direct`) — the one
  call-argument asymmetry between the two sites is that the route passes `scope`
  and the deck does not. This proves it moves no number;
- every governed block present and equal on both sides.

**Rendered slide (3)** — the slide's window, identifier, population and both
residuals are read from the HTTP body and asserted in the rendered text; and no
synthetic stage (`NEW →`, `KFI → KFI`) ever appears.

**Transport, pinned (2)** — reaching the capability imports no web framework;
no deck module contains an HTTP client, URL or transport. These pin the property
§3 depends on, so a future edit cannot silently take the deck's Function-App
deployment away.

### The guards were mutation-tested

A certification that cannot fail is worthless. `_stage_transitions` was
temporarily mutated to re-sort the transitions itself — a plausible, innocuous-
looking edit that would make the deck a second analytical owner:

```
MUTANT: deck now re-sorts the transitions itself
FAILED …TestSingleProducer::test_the_deck_returns_whatever_the_governed_resolver_returns
FAILED …TestSingleProducer::test_the_deck_data_function_only_delegates
2 failed, 9 passed
```

Two guards caught it. The mutation was reverted (`git diff` clean) and the suite
returned to 11 passed.

---

## 6. Observation — recorded, not changed

For a **scoped** deck (`--portfolio-context …`), the route applies a
portfolio-capability gate and can refuse pipeline analysis for a non-originating
scope, whereas the deck resolves the originating group's pipeline. This is
**pre-existing and uniform**: `_pipeline_evo`, `_funnel` and `_stage_transitions`
all behave identically, and the dashboard discloses it on screen (*"the governed
pipeline extract carries no source-portfolio provenance, so it is reported for
the originating group…"*).

It is not specific to Stage Movement and changing it would alter other pipeline
slides, which this sprint forbids. Recorded here so the sign-off is not silently
narrower than it appears. **It moves no stage-transition number** — proved by
the scoped-request test in §5.

---

## 7. Regression — baseline `e6a3f1f` vs HEAD

| Scope | Baseline | HEAD | New failures |
|---|---|---|---|
| All PPTX (`tests/mi_agent_pptx`) + exposure + movement-detail API + engine | 317 passed, 2 skipped, 0 failed | 328 passed, 2 skipped, 0 failed | **0** |

`+11` is exactly the new sign-off module. Nothing else moved, because nothing
else was touched.

**Changed files: 1** — `tests/mi_agent_pptx/test_stage_transition_signoff.py`
(new), plus this report. Engine: unchanged. Route: unchanged. React: unchanged.
PPTX rendering and deck config: unchanged. Other slides: unchanged. MI Query:
unchanged.

---

## 8. Sign-off

| Requirement | Status |
|---|---|
| PPTX consumes the same governed stage-transition response as React | **CERTIFIED** |
| engine → route → React and engine → same producer → PPTX carry identical values | **CERTIFIED** — byte-identical, live |
| Rendered slide shows those values | **CERTIFIED** |
| PPTX performs no analytical work of its own | **CERTIFIED** — structurally, mutation-tested |
| PPTX acquires data over HTTP from `/mi/insight/movement-detail` | **NOT IMPLEMENTED** — §3 |
| No new HTTP routes | **0** |
| No engine / React / MI Query / other-slide changes | **CONFIRMED** — 1 file added, 0 production files changed |

**Merge recommendation: YES**, subject to the reader accepting §3. The deck and
the dashboard already serve one governed answer from one producer; this sprint
makes that a property the estate enforces rather than a fact it happens to
exhibit.
