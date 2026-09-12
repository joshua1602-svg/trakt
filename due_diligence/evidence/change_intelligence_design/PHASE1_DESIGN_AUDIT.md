# Change intelligence — Phase 1 design audit

```
PRODUCT_BASELINE = 00bb3e9d8175a395b6643772379866d6bc6169eb
PRODUCT_HEAD     = 00bb3e9d8175a395b6643772379866d6bc6169eb
PRODUCT_FILES_CHANGED = 0     LIVE_MODEL_CALLS = 0     DEPLOYMENTS = 0
```

The brief gates implementation on this audit. The audit's finding is that **most
of the requested composition architecture already exists**, and that the total
ask is a programme rather than one sprint. Both facts change what should be
built next.

## The headline: the composition layer is already here

`mi_agent_api/insight_engine.py` + `insight_contract.py` + `insight_config.py`
is the "Weekly Portfolio Brief", and it already implements — deterministically,
without an LLM — almost exactly the mechanism the brief specifies:

| Brief requirement | Existing implementation |
|---|---|
| typed finding contract | `insight_contract.Insight` — type, severity, headline, metrics, contributors, components, methodology, source_dates, data_quality, priority, stable `insight_id` |
| deterministic ranking | `insight_engine.rank_key` — severity, then type priority, then type name, then discriminator (total order, stable every run) |
| selection / suppression | `insight_engine.select` — orders, caps per type, cuts to a limit |
| evidence for suppressed findings | `Omission(type, reason, "capped")` — "a brief that silently truncated would read as 'this is everything', which it would not be" |
| risk above ordinary KPIs | `TYPE_PRIORITY`: CONCENTRATION_PROXIMITY 100, DATA_QUALITY 90, then movement 70/65/60, then ticket/LTV 40/38, then mix 30/28 |
| severity classes | CONCERN 3 > ATTENTION 2 > INFO 1 |
| one governed config source | `insight_config` over `config/mi/insights.yaml` — documented defaults, `TRAKT_MI_INSIGHTS_CONFIG` override, and the stated rule: *"The engine reads thresholds ONLY through this module — no threshold is written in the calculation layer."* |
| quiet-period result | `build_brief(..., status=..., reason=...)` |

**So `MISSING_COMPOSITION_LAYER` is not "missing". It is missing three specific
things:** a funded basis, status-transition findings, and reachability from a
plan.

## What it does NOT do

1. **It is bound to the weekly PIPELINE extract.** `build()` returns
   `status="unavailable"`, *"No governed weekly pipeline extract is available"*,
   when `current_extract` is absent. The nine generators are
   PIPELINE_MOVEMENT, COMPLETIONS_MOVEMENT, TICKET_SIZE, WEIGHTED_LTV,
   TICKET_MIX_SHIFT, LTV_MIX_SHIFT, CONVERSION_CONTEXT,
   CONCENTRATION_PROXIMITY, DATA_QUALITY. **There is no funded-balance,
   loan-count or funded-KPI generator, and no funded monthly snapshot pair.**

2. **It has no status-transition finding — the brief's PRIORITY 1.**
   `CONCENTRATION_PROXIMITY` is proximity within one state, not
   breached→cured or pass→fail between two. The limit and eligibility owners
   (`concentration_tests.evaluation.evaluate_active_tests`,
   `forward.evaluate_forward_states`, `mi_agent_api.risk_limits`) each take
   **one** frame. A transition therefore requires calling an existing
   single-period owner **twice, once per snapshot, and diffing two governed
   outputs** — legitimate composition with no new arithmetic, but not something
   the current engine does.

3. **It is unreachable from a GovernedQueryPlan.** `build(root, client_id, *,
   tenant_id, as_of, …)` takes a client and a root, not a plan. It is also
   feature-flagged (`weekly_brief_enabled()`).

## Owners

```
SEMANTIC_SCHEMA_OWNER      mi_agent/interpretation_v2/intent.py (CandidateIntent)
                           + plan.py (GovernedQueryPlan)
NORMALISATION_OWNER        mi_agent/interpretation_v2/normalise.py  (exists, wired
                           at compiler.py:298, records rewrites at :1010)
COMPILER_OWNER             mi_agent/interpretation_v2/compiler.py
CAPABILITY_DISPATCH_OWNER  mi_agent/plan_serving_canary.py (+ plan_runtime_adapter,
                           plan_temporal_runtime, plan_pipeline_runtime)
MATERIALITY_POLICY_OWNER   mi_agent_api/insight_engine.py (rank_key / select)
MATERIALITY_CONFIG_OWNER   mi_agent_api/insight_config.py over config/mi/insights.yaml
PERIOD_CHANGE_OWNER        mi_agent/period_change/workflow.py
                           ::run_period_change_analysis
                           (modes: requested_metric, concept, portfolio_overview)
BRIDGE_OWNER               mi_agent/period_change/bridge.py::balance_bridge
                           + mi_agent_api/movement_detail.py
CONCENTRATION_OWNER        mi_agent/concentration_tests/{evaluation,metrics,forward}.py
LIMIT_OWNER                mi_agent_api/risk_limits.py + concentration_tests.evaluation
ELIGIBILITY_OWNER          mi_agent/concentration_tests/evaluation.py (active tests)
```

## Phase 1 answers

```
EXISTING_REUSABLE_COMPONENTS = Insight/Omission contract; rank_key; select;
    insight_config + config/mi/insights.yaml; build_brief status/reason;
    concentration + data-quality generators; period_change workflow and its
    three modes; balance_bridge; movement_detail; concentration/limit/
    eligibility evaluators; normalise.py
MISSING_SEMANTIC_PRIMITIVE  = form-of-change, independent of measure and
    capability (confirmed from the previous sprint's evidence)
MISSING_COMPOSITION_LAYER   = NOT the mechanism. Missing: (a) a funded monthly
    snapshot-pair basis, (b) status-transition findings, (c) plan reachability
CONFIG_REUSE                = config/mi/insights.yaml via insight_config
NEW_CONFIG_REQUIRED         = a funded section (balance, loan count, WA LTV,
    WA rate, composition shares) and status-transition rules, added UNDER the
    existing loader — no second threshold file
```

## One design conflict to resolve, not override

The brief asks for **absolute** movement thresholds. The existing materiality
owner deliberately forbids them:

> *"Every threshold is relative — never an absolute currency amount, which
> would be the whole story on a small book and noise on a large one."*

That is a considered, lender-agnostic decision and it is the reason the config
is portable across book sizes. Adding absolute thresholds would weaken exactly
the property that makes it work for a second lender. **Recommendation: keep
relative and percentage-point thresholds, and express "economic significance"
as a relative floor plus a minimum-exposure gate (which the config already does
via `min_case_count`) rather than a currency amount.** Flagged for your decision
rather than silently resolved either way.

## Scope finding: this is a programme, not a sprint

The brief asks, in one pass, for: a new semantic primitive across intent + plan
+ provenance + receipt; an interpreter change to populate it; a four-way
compiler mapping; a general `generic_analysis` ownership invariant across every
emitted plan; a funded material-change composition layer with five priority
classes including transitions that have no generator today; a materiality config
extension; a twelve-category three-lender offline semantic bank; twelve
deterministic materiality fixtures; a receipt gate; a full regression gate;
eighteen live Opus calls; and twelve live canary calls.

The brief's own code-quality guard asks to stop if the design requires "another
general query framework", and prefers "one new semantic primitive; one existing
compiler normalisation path; one thin composition owner". Those are two
different sizes of work, and the second half is the larger.

**Recommended split:**

```
SPRINT A — the semantic contract (small, self-contained, unblocks everything)
  * form-of-change as a first-class slot in CandidateIntent and the plan
  * ONE principled interpreter change to populate it (no benchmark phrases)
  * compiler maps form -> owner deterministically; unsupported -> CLARIFY/REFUSE
  * the generic_analysis invariant: no plan may name an owner that cannot
    execute its shape
  * offline semantic bank + replay; zero new analytics; no execution change
  Success: measure no longer determines analytical form; Q18/Q19/Q20 stable
  for semantic reasons; coverage deliberately unchanged.

SPRINT B — funded material-change composition (larger, depends on A)
  * extend insight_engine to a funded snapshot-pair basis
  * add status-transition findings by calling existing single-period owners
    twice and diffing governed outputs
  * funded KPI + composition generators over period_change
  * extend config/mi/insights.yaml under the existing loader
  * the twelve materiality fixtures and the receipt gate
```

Sprint A is the one that matches the change budget and the architecture guard.
Sprint B is where the brief's PRIORITY 1–5 hierarchy actually gets built, and it
needs A's semantic primitive to be reachable at all.

**Nothing implemented. Awaiting instruction on the split and on the
absolute-threshold question.**
