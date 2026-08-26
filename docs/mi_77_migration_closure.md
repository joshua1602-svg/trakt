# MI Agent 7/7 migration closure — MI-only assurance, C7 post-claim cleanup, target state

Base `5f8a9b6`. Pre-registration `cb8ead9` (before the edit). Closure `2a3012f`.

---

## 1. The MI-only test boundary, from the import graph

A plain forward closure from the MI packages said **377 of 477** test modules
"reach MI" — every OCC, onboarding, Annex 2 and mail suite among them. Traced,
the coupling was **one edge**:

```
tests.test_onboarding_annex2_workflow
  → engine.onboarding_agent.workflow
  → engine.onboarding_agent.onboarding_handoff
  → mi_agent.risk_monitor.risk_limits_contract
```

Reaching MI *through* another application is not being MI. Paths now stop at a
non-MI application package, with the narrow leaves MI genuinely imports carved
back in — **measured, not assumed**: `engine.provenance`,
`operations_control.stores`, `apps.blob_trigger_app.source_registry` are the
only barrier-package modules the MI packages import directly. The task's
shared-module rule is honoured separately: a test that *lives with* a shared
library MI depends on (`trakt_core`, `analytics_lib`, `snapshot`,
`trakt_notifications`, `trakt_tools`) is in; a test that merely touches one is
not.

```
MI production surface (transitive)   302 modules
test modules INCLUDED                278
test modules EXCLUDED                199
```

Probed: `test_onboarding_annex2_workflow` **OUT**; OCC, mail, demo-platform
**OUT**; `test_compound_canary_bank`, `test_mi_query_invariants`,
`test_failclosed_route_execution` **IN**. The four remaining paths matching an
"unrelated" word (`test_gate4_mi_readiness_flow`, `test_mi_regulatory_separation`,
`test_readiness_agent_surface`, `test_readiness_framework`) are in because they
import MI directly — 30 to 42 MI modules each.

Manifest: `migration_phase0/MI_REGRESSION_MANIFEST.txt` (278 paths).
Instrument: `migration_phase0/mi_test_boundary.py`.

## 2. MI-only baseline and regression

| | baseline | after |
|---|---|---|
| modules | 278 | 278 |
| passed | 5957 | 5957 |
| failed | 81 | 81 |
| skipped | 711 | 711 |
| xfailed / xpassed | 15 / 0 | 15 / 0 |
| errors | 4 | 4 |
| timeouts | 1 | 1 |
| tests executed | 6768 | 6768 |
| failing/erroring names | 85 | 85 |

```
INTRODUCED:    0
FIXED/REMOVED: 0
modules with a different summary: none
```

The one timeout (`tests/test_assurance_measurement_failure.py`) is identical in
both runs and reported as a timeout, not a pass. The manifest was frozen before
the guard module existed, so `tests/test_post_claim_semantic_guard.py`
(14 passed) is measured separately and is not in the 278.

## 3. The three C7 post-claim sites, recomputed and closed

Equivalence measured over **882 corpus questions before the edit**:

| site | replaced by | agreement |
|---|---|---|
| `_period_request.requested_span(question)` | `span_from_claim(contract.time)` | 882 / 882 |
| `chat_routing._resolve_lens(question, source_lens)` | lens mapped from `contract.source_scope` | 882 / 882, and 400 / 400 with a workspace selection present |
| `recognise(question, …)` inside the handler | the registry's own pre-claim reading, carried | deterministic 882 / 882 |

`TimeClaim.window_periods` exists for the first of these — its docstring names
`requested_span(question)` as *"a second read of the sentence for a fact the
contract had already claimed"*. Conversion 2 closed against it; C7 had not.

For population, `mi_agent.portfolio_lens` remains the only thing that decides
what a scope MEANS; the route maps the contract's claim onto that owner's own
constructors. No new scope capability. Source lens and row predicates remain
different axes. A contract with no scope claim yields a **deferral**, not a
second population owner reachable exactly when the first one failed.

For recognition, `RouteRequest` gained a **generic** pre-claim memo — a place to
put a value, not a slot named after any route — so the reading the registry
already produced is carried instead of rebuilt. No new flag, no route-local
concept.

**A correction recorded rather than hidden.** The first caller-precedence check
passed `{"scope": "acquired"}`, a shape `lens_from_selection` does not accept;
neither side saw a selection and its "agree 400" meant nothing. Re-run with a
selection the resolver accepts, provenance became `caller_context` 399 /
`explicit_user` 1 and the agreement became real.

## 4. Zero blast

882 questions through the live `/mi/query` path in a worktree at `cb8ead9` and
in the working tree, one shared fixture, comparing route, `ok`, answer, warnings
and every artifact field by field:

```
QUESTIONS CHANGED 0     ROUTES CHANGED 0
ok=True   before/after  381 / 381
artifacts before/after  932 / 932
exceptions   0 / 0      keys removed/added 0 / 0
```

Canary **0 invariant breaches**, nothing DROPPED, 21 UNEVIDENCED and 5
unexercised families unchanged. Independent audit **10 / 10**. Route-substitution
detector **0 of 2**.

## 5. The structural guard

`tests/test_post_claim_semantic_guard.py` — AST, 14 tests, covering all seven
handlers. It detects four shapes of the defect: the question handed to a
resolver, matched against a regex, tested for a keyword, or compared directly.

Five mutation controls each reintroduce a real defect into a **parsed copy** and
require the guard to fail: `_resolve_lens` reinstated, `requested_span`
reinstated, post-claim `recognise` reinstated, a measure resolver added to a
C1–C6 handler, and a new three-word movement vocabulary declared and used. A
sixth test proves none of them is left in the source.

**Discrimination, not agreement.** Run against the pre-fix worktree the guard
**fails on C7 with exactly the three sites** and passes C1–C6; run here it
passes all seven.

## 6. The seven-route census, recomputed

| | route | pre-claim recognition reads | post-claim raw-question reads | route-local vocabulary | semantic decision sites |
|---|---|---|---|---|---|
| C1 | portfolio_summary | wording + spec | 0 | 0 | 0 |
| C2 | period_movement | wording | 0 | 0 | 0 |
| C3 | geo_exposure | wording + spec + view | 0 | 0 | 0 |
| C4 | funded_bridge | wording + spec | 0 | 0 | 0 |
| C5 | temporal_compare | **contract only** | 0 | 0 | 0 |
| C6 | evolution (+funnel, +stage) | wording + spec | 0 | 0 | 0 |
| C7 | period_change_analysis | wording + spec + view | 0 | 0 | 0 |
| | **total** | **6 of 7** | **0** | **0** | **0** |

By concept — dataset, measure, population/scope, time/comparison,
LEVEL/MOVEMENT, grouping dimension, alternate dimension, filters/predicates,
ranking requested/direction/basis/limit: **no second owner**. LEVEL/MOVEMENT has
one owner (`lexical.temporal_aspect`) with five delegating consumers; ordering
has one owner (`lexical.ordering_request`) with two.

Cross-checked by `semantic_owner_inventory`, an independently written
instrument, which now reports `period_change_analysis` **CLEAN** and all seven
migrated routes clean.

## 7. Cost

| | added | deleted |
|---|---|---|
| C7 post-claim cleanup (`period_change_route.py`) | 86 (52 code) | 9 |
| router plumbing (`recogniser_registry.py`, `chat_routing.py`) | 22 (9 code) | 0 |
| tests (guard) | 229 | 0 |
| test harness adapted (`test_period_change_route.py`) | 25 | 0 |
| assurance (`mi_test_boundary.py` + manifest) | 538 | 0 |

**This is not deletion-heavy, and the number is not shaped to look like it.**
The population mapping has to live somewhere and it landed as one function. It
is not new semantic infrastructure — no new concept, no new primitive, no new
route shape — but it is a net addition of ~61 production code lines.

## 8. Target-state audit

| property | verdict | evidence |
|---|---|---|
| **Interpretation singularity** | **PROVEN** for the seven routes | census: 0 post-claim reads, 0 route-local vocabularies, no second owner for any of the fourteen concepts. Cross-checked by a second instrument. |
| **No post-claim reinterpretation** | **PROVEN** for handlers; **PARTIALLY PROVEN** for the estate | the AST guard covers the seven handlers and discriminates against the pre-fix tree. `analytical_plan` never receives the question (`assert_no_question_read`). The period-change engine carries `question` only in `to_dict()`, for audit. Two GENERIC-path routes outside the seven — `concentration_analysis` and `portfolio_risk_comparison` — still hand the whole question to a workflow that runs its own recognition. |
| **Generic composition** | **PROVEN** | *"Which three regions added the most balance since last month?"* — a limit never in the corpus — is answered, ranked, receipted, with **zero code change**. Filtered ranked movement composed from existing primitives (22 lines in one shared parser function, no new primitive). |
| **Deterministic truth** | **PROVEN after interpretation** | the contract → plan → execution → receipt path is deterministic; a model may participate in producing the *spec* (`llm_enabled`, deterministic-first) but owns nothing downstream of it. |
| **Evidence singularity** | **PROVEN for ranked movement**; **PARTIALLY PROVEN estate-wide** | prose, table and `metadata.rankedMovement` are projections of one `MovementReceipt`; mutating it moves the live answer, and substituting the grouping dimension is caught and refused. Only `period_change` publishes that receipt; the other routes publish the older `execution_receipt` channel, which this programme has not re-measured. |
| **Fail closed** | **PROVEN** | detector 2 of 2 substituted pre-fix / 0 of 2 after; F1 and F2 fail against the pre-fix tree; pre-claim decline preserved (F3, F3b). |
| **Honest insufficiency** | **PROVEN** | no implicit measure, no implicit comparison period; the no-period refusal is unchanged in the sweep; a dimension the governed registry does not admit is refused by name with the reason. |
| **No capability-specific route branching** | **PARTIALLY PROVEN** | a new *combination* needs no new shape (above). A new *analysis* still needs a recogniser: recognition remains wording-based on 6 of 7 routes. |

### Residuals, stated

1. **Recognition reads wording on 6 of 7 routes.** Lives in the registered
   `recognise` predicates. It remains because this task explicitly permitted it
   and forbade a recogniser redesign. It decides **which analysis runs**, so it
   can change analytical meaning. It does not block composition of already-
   governed concepts; it blocks contract-driven *route selection*. C5 shows the
   alternative — recognised from `spec.temporal_mode` alone.
2. **`concentration_analysis` and `portfolio_risk_comparison`** hand the whole
   question to a workflow with its own recognition. They are on the generic
   funded-book path but are **not among the seven**. They can change analytical
   meaning. They block the claim that the generic estate has one interpreter.
3. **Evidence singularity is proven only for ranked movement.** The other routes
   were not re-measured against the receipt-derivation standard.
4. **A governed dimension can still be refused by configuration.**
   `broker_channel` is tagged `period_change` and its data is present, but its
   `asset_applicability` is `[equity_release]` against an unidentified portfolio
   asset class, so `_base_filter` excludes it. **This is config, not route
   shape** — `_select_requested` reads only the registry and the policy — and
   the refusal is honest and names the field. It does not block composition; it
   is a registry-tagging decision.
