# Autonomous Portfolio Review Agent — Minimum Production Slice

Implementation report for the work in commits `9aac7d1` … `a761c49`, following
`docs/autonomous_portfolio_review_agent_readiness_review.md`.

**Production recipients remain OFF.** `config/mi/teams_notifications.yaml` is
unchanged: `enabled: false`, `recipients: []`.

---

## 0. Re-verification of the six identified changes

Checked against `origin/main` (`7fe8ccd`) before any code was written.

| # | Change | Status | Evidence at HEAD-of-main |
|---|---|---|---|
| 1 | False-clear concentration defect | **CONFIRMED** | `sources.py:132-136` skipped `_resolve_pipeline_side` on a funded-only approval; that was the only caller of `_concentration`. `risk_review.py:252` computed `partial = bool(unavailable)`, which was empty because nothing was *attempted*. |
| 2 | Pipeline product attribution | **CONFIRMED** | `movement_detail.py:78` `DIMENSIONS = (brokers, regions)`. `product_type` is a prepared pipeline column (`pipeline_prep.py:653`) and a governed dimension (`pipeline_field_contract.yaml:151`), unused for attribution. |
| 3 | Funded generator set + thresholds | **CONFIRMED** | `insight_engine.build` steps were pipeline-only; `insights.yaml` had no funded section; the funded card printed its figures unconditionally. |
| 4 | "Five pipeline tools" | **REQUIRES DIFFERENT MINIMAL CHANGE** | The count holds; the composition does not. `period_change` already covers funded headline movement, distribution shift and the balance bridge, so a fifth *pipeline* tool was not needed. What was missing was **four pipeline primitives plus one funded one** — `funded_composition`, because no tool could say *why* the book moved. Also found: `evaluate_covenants` projects a fixed field list (`covenants.py:58`) that excludes `emergingRisks` and the forward `states`, so forward concentration needed its own tool rather than an argument to that one. |
| 5 | Portfolio Review controller on the readiness loop | **CONFIRMED** | `readiness_agent/agent.py:218` is reusable, but `SYSTEM_PROMPT` and `SUBMIT_TOOL` were module constants and `import anthropic` was unconditional. Both addressed additively. |
| 6 | Enable by configuration only | **CONFIRMED** | Unchanged and still off. |

**Additional finding, ahead of implementation:** acquisition attribution needed
no schema change. `engine/provenance.py:47` already defines
`source_portfolio_id` (mandatory on every row), `source_portfolio_type`,
`source_portfolio_label` and `acquisition_date`, and `derive_portfolio_type`
(`provenance.py:115`) resolves direct/acquired from the id. The smallest
scalable rule was therefore **portfolio presence across two governed frames**,
which needs no new metadata at all — see §A.3.

---

## A. What changed

### A.1 The false-clear defect — `9aac7d1`

| Module | Change |
|---|---|
| `trakt_notifications/sources.py` | `_resolve_concentration` hoisted out of the pipeline side and run once for **every** update type, before either side. Both shapes of absence recorded: an exception (already handled by `_safe`) and a governed `{"available": false}`, which was not. |
| `trakt_notifications/sources.py` | `REQUIRED_RISK_CAPABILITIES` + `unevaluated_risk_domains()` — a required domain with no *positive evidence* it ran is added to the unavailable list whatever the resolver did. `unavailable_summary` now unions the two. |

The guard asks for evidence rather than trusting the absence of an error, so a
future resolver change that skips a domain degrades the message instead of
silently widening what it claims.

**Regression evidence:** 9 of the 10 new tests in
`tests/notifications/test_risk_evidence.py` fail against the unfixed module.
They drive `sources.resolve` and stub only at the governed *service* boundary —
the original defect survived because the fixture supplied `concentration`
directly.

### A.2 Pipeline product attribution — `063ffca`

| Module | Change |
|---|---|
| `mi_agent_api/movement_detail.py:78` | `("products", "product_type")` added to `DIMENSIONS`. Everything downstream iterates the tuple, so the projection, ranking, reassignment counts and empty-payload shape follow. |
| `mi_agent_api/insight_generators.py` | Products carried on `PIPELINE_MOVEMENT` and `COMPLETIONS_MOVEMENT`; the summary names the product lead separately from the broker lead. |
| `mi_agent_api/insight_generators.py:71` | `_names` skips the `Unknown` bucket. |
| `trakt_notifications/portfolio_update.py` | The card names the product lead first, one lead per dimension. |

**One deliberate change to an existing answer:** a summary that previously read
"led by Unknown" no longer names it. `Unknown` is a real analytical fact and is
still never dropped from the attribution — it is simply not the name of a
driver, and adding a dimension many books will not carry made that sentence
common rather than rare. No other summary changes.

### A.3 Acquisition-aware movement — `cc7637e`

New module `mi_agent_api/funded_composition.py`. One partition of both frames:

```
opening balance
  + portfolio additions      source portfolios present now, absent prior
  - portfolio disposals      present prior, absent now
  + organic new lending      new loans inside CONTINUING portfolios
  - exits                    departed loans inside continuing portfolios
  +/- existing-book movement loans present in both
= closing balance
```

Components sum to the movement **by construction**, not by tolerance; the
residual is reported regardless.

**Where "acquisition" comes from.** A source portfolio appearing for the first
time is an observed *portfolio addition* — a fact about identity. Whether it is
an *acquisition* is answered only by an explicit `source_portfolio_type` on the
rows, or by `engine.provenance.derive_portfolio_type` over the governed id. When
neither answers, it is `unclassified` and is described as a new source
portfolio. **A balance jump inside one portfolio produces no addition at all**
(`test_a_balance_jump_alone_is_never_called_an_acquisition`).

**Underlying-book lens** reuses `evolution._scope_frame_lens` over the
continuing ids. No second population definition.

### A.4 Funded materiality — `f5598d8`

| Module | Change |
|---|---|
| `mi_agent_api/insight_contract.py` | Six funded types + their `TYPE_PRIORITY`. `RISK_LIMIT_TRANSITION` ranks above `CONCENTRATION_PROXIMITY`; `FUNDED_COMPOSITION` above the headline it decomposes. |
| `mi_agent_api/insight_generators_funded.py` | New. Six generators: headline movement, composition, underlying book, mix over seven governed dimensions, weighted LTV, risk-limit transitions. |
| `mi_agent_api/insight_engine.py` | `resolve_funded_inputs` + `build_funded` + `select_funded`. `select` refactored onto a shared `_select_with` so the *ordering rule* cannot diverge between weekly and monthly. |
| `config/mi/insights.yaml`, `insight_config.py` | Five funded threshold sections + `funded_brief` limits. |
| `trakt_notifications/sources.py` | Resolves the monthly brief; new `funded_brief` slot kept separate from `brief`. |
| `trakt_notifications/portfolio_update.py` | Funded observations taken from the gated set, in ranked order, using the generators' own summaries. |
| `trakt_notifications/risk_review.py` | `RISK_LIMIT_TRANSITION` merged into the risk findings at rank 1. |

`risk_limit_transition` reads `statusTransition` / `priorStatus` /
`deteriorated` off the approved concentration evaluation. **It defines no limit,
status or threshold** — its only config key decides whether an *improvement*
earns a card.

**No funded data-quality generator** was built. The governed funded
data-quality signals exist as agent tools over the lineage index, not as
anything the notification resolver holds; reaching them would mean a second
validation path inside notifications. The agent reads those tools directly.

**One deliberate change to an existing answer:** the funded card's cohort, LTV
and region bullets are now emitted only when the gated set produced no
observation (`_ungated_attribution`). Where the generators ran, `FUNDED_COMPOSITION`
and the region mix shift are the gated versions of exactly those facts, so
keeping both would duplicate. Where they did not, the ungated lines still run —
nothing is lost.

### A.5 Governed tools — `d90ba9c`

`trakt_tools/handlers/portfolio_review.py` — five reusable primitives:

| Tool | Wraps |
|---|---|
| `pipeline_position` | `pipeline_contract.compute_pipeline_snapshot` |
| `pipeline_movement` | `movement_detail.resolve_movement_detail` |
| `pipeline_conversion` | `evolution.pipeline_funnel_evolution` (lagged as the dashboard lags it) |
| `funded_composition` | `funded_composition.composition_movement` |
| `forward_concentration` | `concentration_tests_api.compute_concentration_tests` |

`ToolDependencies.pipeline_root` added with the same test seam `output_root`
has. Pipeline tools accept a pipeline-pinned resource and refuse a funded one —
the funded tools' rule in the other direction. SPV refusal is checked in each,
not inherited. `deploy/agent-api/trakt-agent-openapi.yaml` regenerated by
`scripts/build_agent_openapi.py`.

### A.6 The controller — `a344bb8`

`portfolio_review/` — `objective.py` (two objectives, one system prompt,
`SUBMIT_REVIEW`) and `controller.py` (`resolve_period`, `run_review`,
`ReviewOutcome`).

`readiness_agent.run_assessment` gains `system_prompt` and `submit_tool`
parameters, defaulted to the readiness values, and its `anthropic` import
becomes lazy so the injection seam works without the SDK. No other change to
that agent.

The objectives name **no metric, no ordering and no first call**
(`test_no_objective_names_a_metric_or_an_ordering`). The opening message carries
dates and no figures — the rule that every number came from a tool call must
hold of the model's *input* too.

---

## B. What did NOT change

| Area | Status |
|---|---|
| Canonical architecture / schema | **Unchanged.** No field added, removed or redefined. `engine/provenance.py` untouched. |
| Ingestion architecture | **Unchanged.** `function_app.py`, `occ_intake`, the orchestrator and the gates are untouched. |
| MI Query Agent semantics | **Unchanged.** No edit to `mi_query_spec`, `mi_query_executor`, `mi_query_validator`, `llm_query_parser`, `chat_routing` or the recogniser registry. |
| Risk-limit ownership | **Unchanged.** Limits, statuses and transitions remain the approved concentration configuration's. Nothing added defines one. |
| Teams delivery architecture | **Unchanged.** No edit to `teams_bot.py`, `teams_client.py`, `delivery.py`, `outbox.py`, `recipients.py`, `contract.py` or `cards.py`. |
| A2A | **Unchanged and not a dependency.** `trakt_a2a/` untouched. |
| Client isolation | **Unchanged.** Tenancy, entitlement and the five recipient gates untouched. |
| Storage model | **Unchanged.** No new store, container, layout or prefix. |
| OCC approval | **Unchanged.** `operations_control/engine.py` untouched. |

### Change budget (actual)

| Module | Purpose | Lines | Seam reused | Risk |
|---|---|---|---|---|
| `trakt_notifications/sources.py` | resolve concentration always; evidence guard; monthly brief | +153/−13 | the existing resolver | **Low** — additive; the one behaviour change is the defect fix |
| `mi_agent_api/funded_composition.py` | movement decomposition | +330 (new) | `evolution.funded_frames`, `_scope_frame_lens`, `_cohorts` | **Low** — new module, no existing caller |
| `mi_agent_api/insight_generators_funded.py` | funded materiality | +493 (new) | the weekly generator contract | **Low** — new module |
| `mi_agent_api/insight_engine.py` | funded resolve/build/select | +279/−4 | `build_brief`, `rank_key`, `_safe` | **Low** — `select` refactored onto a shared helper, same behaviour |
| `trakt_tools/handlers/portfolio_review.py` | five tools | +639 (new) | `execute_governed_tool`, `_scope_block` | **Low** — new module |
| `trakt_tools/handlers/__init__.py` | registration | +112 | the registry | **Low** |
| `portfolio_review/` | controller + objectives | +463 (new) | `run_assessment`, `GovernedSession` | **Medium** — the only module that runs a model |
| `trakt_notifications/portfolio_update.py` | gated funded observations | +90/−34 | the message contract | **Medium** — changes card content |
| `mi_agent_api/insight_contract.py` | six types + priorities | +30 | the contract | **Low** |
| `mi_agent_api/insight_config.py`, `config/mi/insights.yaml` | thresholds | +39, +73 | the config loader | **Low** |
| `mi_agent_api/movement_detail.py` | product dimension | +13 | `DIMENSIONS` | **Low** |
| `mi_agent_api/insight_generators.py` | product in summaries | +33/−5 | the generators | **Low** |
| `trakt_notifications/risk_review.py` | transitions as findings | +15/−3 | `_INSIGHT_RISKS` | **Low** |
| `readiness_agent/agent.py` | parameterise prompt/tool; lazy import | +33/−7 | itself | **Low** — defaults preserve behaviour |
| `trakt_tools/execution.py` | `pipeline_root` dependency | +16 | `ToolDependencies` | **Low** |

**Totals: 2,854 production lines added, 34 deleted; 2,481 test lines added.**
No stop condition in §23 was reached.

---

## C. Weekly pipeline capability

From `tests/notifications/test_pipeline_product_attribution.py` and the existing
weekly path. Figures are the fixture's, stated in the test.

```
Weekly Pipeline Update — 7 August

Pipeline contains 812 cases totalling £503.4m.

• Case count is 9.0% above its five-week average.
• Completions totalled 41 cases and £18.7m, 14.0% below average.
• Weighted expected funding is £236.2m.
• Growth of £2.1m was led by Lump Sum, Broker A and London.
• Weekly conversion to completion is 4.2% by value over the governed
  5-week window.

Pipeline five-week comparisons are based on 5 observed weeks.
```

The fourth bullet is what this work added: **product first**, then one lead per
dimension. Each dimension's contributions reconcile to the £2.1m independently
(`test_each_dimension_reconciles_independently`) — which is why they are named
rather than added.

---

## D. Monthly funded capability — a normal month

```
Monthly Funded Update — 31 July

Funded balance is £1.42bn, +£24.1m on the month.

• Loan count is 4,820 (+59 on the month).
• Largest book contribution: Direct at +£18.4m.
• Weighted-average LTV is 31.4%, +0.3pp on the month.
• Largest regional contribution: South East at +£9.2m.
```

Movement is +1.73% — above the 1.0% gate, so the headline insight is produced.
LTV moved 0.3pp — below the 0.5pp gate, so it is **suppressed with a stated
omission** and the ungated line carries the level instead. That distinction did
not exist before: every figure printed unconditionally.

---

## E. Acquisition month capability

From `tests/notifications/test_acquisition_acceptance.py`, which runs the whole
pathway. Every figure is the fixture's, computed by the production code.

```
Monthly Funded Update — 31 July

Funded balance is £184.0m, +£72.0m on the month.

• Loan count is 5 (+3 on the month).
• £68.0m of the £72.0m movement is £68.0m portfolio additions, £3.0m new
  lending. £68.0m of the £72.0m movement reflects the acquisition of
  Portfolio B.
• Excluding the 1 portfolio(s) added this period, the existing book
  increased by £4.0m (+3.6%) to £116.0m.

Risk Review — london exposure: pass → warning
London exposure deteriorated from pass to warning (pass → warning).
Utilisation is 94.7% of the limit. Headroom +1.6pp.
```

The governed decomposition behind it:

| Component | Amount |
|---|---|
| opening balance | £112.0m |
| portfolio additions (`portfolio_beta`, type `acquired`) | +£68.0m |
| organic new lending | +£3.0m |
| existing-book movement | +£1.0m |
| **closing balance** | **£184.0m** |
| reconciles | `True`, residual `0.0` |

The month is +64% at the headline and **+3.6% underlying**. Both are stated, so
the acquisition cannot hide what the incumbent book did.

---

## F. Scale

`test_a_further_acquired_portfolio_needs_no_production_change` adds a third and
fourth acquired book (`portfolio_gamma`, `portfolio_delta`) by changing the
frame alone. No module edited, no id registered, no branch added; additions
total £104.0m and still reconcile.

`test_no_production_module_names_a_portfolio_id` scans the five new production
modules for `acquired_00N` / `direct_00N` / `portfolio_alpha` / `SPV2` and finds
none. Every test in `test_funded_composition.py` uses generic names.

`test_an_explicit_type_column_beats_the_id_prefix` covers a client whose ids
carry no Trakt prefix: the `source_portfolio_type` column is the primary
authority, the prefix only a fallback.

---

## G. Remaining gaps

| Gap | Class | Note |
|---|---|---|
| The controller has never run against a real model | **PRE-GO-LIVE** | Tested against a scripted model. Needs a shadow run against a real month, compared with the deterministic brief, before it drives anything a user sees. |
| The controller is not wired to the trigger | **PRE-GO-LIVE** | Deliberate. Phase 1 (deterministic brief) should ship and be watched before an agent is put in front of it. |
| Full-suite regression not completed | **PRE-GO-LIVE** | See §H. A targeted regression across every affected area passed; the whole suite exceeds the time available here. |
| No funded data-quality generator | **POST-GO-LIVE** | The signals exist as agent tools; the notification resolver has no path to the lineage index and should not grow one. |
| Forward-concentration signals not consumed by the monthly brief | **POST-GO-LIVE** | `forward_concentration` exists as a tool; the monthly brief reads the funded evaluation only. |
| Per-recipient content | **PERSONALISATION** | Out of scope, as required. The seam is preserved: see §I. |
| User preference profile | **PERSONALISATION** | Telemetry began 2026-09-01; no corpus yet. |
| Annex 2 / readiness deltas in the review | **FUTURE SECURITISATION** | `regulatory_readiness`, `evaluate_rule_packs`, `data_completeness` and `list_validation_exceptions` are already callable by the controller; nothing consumes them as period-over-period signals. |
| Departed portfolios in `period_movement` | **POST-GO-LIVE** | `movement_summary.period_movement` iterates the current frame's cohorts (`movement_summary.py:365`), so a disposed portfolio lands in its residual. `funded_composition` reports disposals correctly; the older service still has the gap. Not touched — out of scope and it changes a governed answer. |

---

## H. Regression

**Targeted, before and after.** The same suites were run on `origin/main` (in a
worktree) and on HEAD: `tests/notifications/`, `tests/operations_control/`,
`tests/concentration_tests/`, `mi_agent_api/tests/`, `mi_agent/tests/`, the
agent-tool and A2A suites, the readiness suites, and the movement/receipt
suites.

| | Baseline (`origin/main`) | HEAD |
|---|---|---|
| passed | 4,214 | **4,285** (+71) |
| failed | 22 | **21** |
| skipped / xfailed | 316 / 7 | 316 / 7 |
| duration | 11:00 | 10:59 |

**New failures introduced: none.** The failure lists were diffed test by test;
`comm -13` over the two sorted lists is empty. All 21 HEAD failures are present
on clean `main` — they live in `test_mi_predicate_extraction`,
`test_mi_trust_hardening`, `test_p0_execution_receipt`,
`test_p1c_ranked_movement`, `test_parser_cost_hardening`,
`test_chat_routing_e2e`, `test_currency_authority`,
`test_pipeline_stage_transition`, `test_single_parse_and_substitution` and
`test_conversion2_period_movement`, none of which this work touches.

The 22nd baseline failure —
`test_registry_governance::test_checked_in_registry_matches_generator` — is **not
a fix**. It asserts an absolute path against the checked-in registry, so it fails
from any worktree and passes only when run from `/home/user/trakt`. Verified by
running it standalone on both: it fails on HEAD too.

**Synthetic portfolio artefacts.** The first pass of this work tested the new
funded code only against frames stated inline in the tests. That was the weaker
evidence, and it has since been closed:
`tests/notifications/test_funded_composition_on_real_canonical.py` runs the
decomposition over `synthetic_demo/output/multibook/` — two consecutive platform
canonicals the pipeline actually produced, 116 and 118 loans across
`alp_acquired` / `alp_origination` / `spv1_sponsored`. None of those ids carries
the `direct_` / `acquired_` prefix, so classification runs through the
`source_portfolio_type` column on data never shaped to exercise that path. The
partition reconciles at full precision (residual `0.0`).

The `tests/test_simulation_*.py` suites — the asset-class hardening framework,
which drives generated portfolios through the real gates — were **not** in the
targeted set above and were run separately: **19 failed / 136 passed / 3 errors
on both `origin/main` and HEAD**, failure lists diffed and identical. They are
pre-existing and environment-related in this container.

**Not completed:** the entire repository suite. It is simulation-heavy and did
not finish in the time available; two full runs reached ~10% after fifteen
minutes each. This is stated rather than glossed: the targeted set covers every
module touched and every area §19 names, but it is not the whole suite.

**Environment note.** This container ships a broken `cryptography` /
`_cffi_backend`, which fails ten `test_teams_package.py` tests identically on
clean `main`. Reinstalling `cffi` fixes it. Any run before that repair shows
those ten as failures on both sides.

---

## I. The personalisation seam

Not implemented, and preserved:

* **Mandatory findings are already separable.** `Insight.notification_eligible`
  (`insight_contract.py:129`) is `severity in (attention, concern)` — every
  breach, near-breach and material deterioration the funded generators produce
  is already flagged, including every `RISK_LIMIT_TRANSITION` into `warning` or
  `breach`.
* **Ranking is already separate from content.** `rank_key` orders on severity
  then type priority; `priority` is stamped by the selector, not the generator.
  A preference weighting slots in as a third term without touching what an
  insight claims.
* **Routing already fans out per recipient.** `outbox.enqueue` writes one item
  per (message, recipient). Only the *content* is shared —
  `delivery.py:168` renders from the batch — so per-recipient content is a
  change to one render call, not to the identity, dedupe or correction logic.

A future profile can therefore weight the personalised tier without being able
to suppress the mandatory one.

---

## J. Readiness

| Capability | Before | After |
|---|---|---|
| Snapshot comparison | EXISTS | EXISTS |
| Pipeline growth | EXISTS | EXISTS |
| Funded growth | EXISTS | EXISTS |
| Pipeline→funded movement | EXISTS | EXISTS |
| Product drivers | PARTIAL | **EXISTS** |
| Regional drivers | EXISTS | EXISTS |
| LTV movement | EXISTS | EXISTS |
| Borrower-age analysis | PARTIAL | **EXISTS** (mix dimension, gated) |
| Joint-borrower analysis | PARTIAL | **EXISTS** (mix dimension, gated) |
| Vintage analysis | EXISTS | EXISTS |
| Concentration analysis | EXISTS | EXISTS |
| Risk-limit status | EXISTS | EXISTS |
| Forecast | EXISTS | EXISTS |
| Securitisation signals | PARTIAL | PARTIAL (callable, not consumed) |
| MI user telemetry | EXISTS | EXISTS |
| User preference profile | MISSING | MISSING (out of scope) |
| Proactive Teams messaging | EXISTS | EXISTS |
| Per-user message routing | PARTIAL | PARTIAL (routing yes, content no) |
| Semantic MI execution API | PARTIAL | PARTIAL |
| Evidence / provenance | EXISTS | EXISTS |
| **Acquisition attribution** | *(not scored)* | **EXISTS** |
| **Underlying-book analysis** | *(not scored)* | **EXISTS** |
| **Funded materiality** | *(not scored)* | **EXISTS** |
| **Autonomous review controller** | MISSING | **EXISTS** (untested against a real model) |

**16 EXISTS, 3 PARTIAL, 1 MISSING** over the original twenty, plus four
capabilities that did not exist to be scored. On the same weighting as the
review: **17.5 / 20 ≈ 88%**, from 75%.

### Is it safe to enable for Client 1?

**The deterministic brief: yes, after a shadow month.** The false-clear defect
is fixed and regression-tested; the monthly review now has the same materiality
discipline as the weekly one; an acquisition month is described as an
acquisition month. Recommended sequence:

1. Enable `TRAKT_MI_WEEKLY_BRIEF` and read the briefs without delivering them.
2. Run one full reporting cycle in shadow — the batch is stored even with no
   recipient (`trigger.py:228`), so the audit record of what *would* have been
   said exists without anything being sent.
3. Authorise one operational recipient via the existing CLI.
4. Board users only after the weekly and monthly cards have been read by
   someone operational.

**The autonomous controller: not yet.** It has never run against a real model.
It should run in shadow against the deterministic brief for at least one weekly
and one monthly cycle, and be read for the two failure modes its prompt guards —
adding contributor dimensions together, and calling a movement an acquisition —
before it drives anything a user sees.

**Board vs operational permissions** remain unmodelled: `portfolio_contexts` is
a portfolio scope, not a role. Both user types authorised for `total` receive an
identical card. That is the current production model and was explicitly out of
scope, but it should be decided before board users are authorised.
