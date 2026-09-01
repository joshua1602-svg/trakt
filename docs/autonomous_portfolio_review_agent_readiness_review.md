# Autonomous Portfolio Review Agent — Repository & Architecture Review

**Status:** review only. Nothing was implemented. No code was changed.
**Scope:** how close the current Trakt production architecture is to autonomously
determining material portfolio developments and delivering a curated, per-user
Microsoft Teams briefing after each reporting cycle.

---

## A. Executive conclusion

### Readiness: **MOSTLY READY**

The prompt assumes a system that must be taught to send proactive Teams
briefings. That assumption is wrong, and the correction is the single most
important finding in this review:

> **Trakt already generates and proactively delivers a governed, two-message
> portfolio briefing to authorised Microsoft Teams users after an approved
> reporting run.** The pathway is built end to end, tested end to end, wired
> into the production approval hook and the production Azure Function timer,
> and switched **off** by configuration.

The evidence chain is unbroken and production-resident:

```
blob arrival        function_app.py:113            on_raw_blob_event
  → OCC intake      apps/blob_trigger_app/occ_intake.handle_arrival
  → approval        operations_control/engine.py:2466  approve_publication
  → notify hook     operations_control/engine.py:2516  _notify_publication
  → trigger         trakt_notifications/trigger.py:94  on_publication_approved
  → governed MI     trakt_notifications/sources.py:104 resolve
  → two messages    trakt_notifications/generate.py:59 build
  → outbox          trakt_notifications/outbox.py:185  enqueue
  → timer worker    function_app.py:83                 deliver_teams_notifications
  → Teams card      trakt_notifications/teams_client.py
```

The bot endpoint that captures the conversation reference a proactive send needs
is `mi_agent_api/teams_bot.py:256` (`POST /v1/teams/bot/messages`), mounted at
`mi_agent_api/app.py:1963`. Recipient authorisation is a separate, operator-only
step (`trakt_notifications/recipients.py:228`), so installing the app authorises
nothing.

So the question is not "can Trakt send a briefing". It is **"can Trakt decide
what to say, and say it differently to different people"**. Against that
question:

| Dimension | Assessment |
|---|---|
| Governed analytics to reason over | **Present and large.** ~30 governed service functions across pipeline, funded, movement, concentration, forecast, cohort and readiness. |
| Deterministic materiality | **Present but pipeline-only.** Nine generators, config-driven thresholds, explicit omissions (`mi_agent_api/insight_engine.py:230`). Nothing equivalent exists for the funded book. |
| Autonomous investigation loop | **Present, for a different objective.** `readiness_agent/` is already a governed agentic loop with structural no-arithmetic enforcement. |
| Proactive Teams delivery | **Present, complete, tested, off.** |
| Per-user personalisation | **Absent, and architecturally excluded by design today.** |
| MI telemetry to learn preferences from | **Present, resolved-semantics-based, but three weeks old with no accumulated history.** |

### Is this integration work, analytics work, or AI work?

**Predominantly integration work**, in that order of weight:

- **~65% integration** — connecting components that already exist but do not
  currently call each other (funded-side insight generation; the readiness agent
  loop pointed at a portfolio-review objective; the telemetry store read as a
  preference source; per-recipient rendering in the outbox).
- **~25% new deterministic analytics** — a funded/monthly materiality generator
  set mirroring the weekly one, and a small number of missing attribution
  dimensions (product on pipeline movement, borrower structure on funded).
- **~10% new AI work** — an investigation controller and narrative synthesiser.
  Both are variations of `readiness_agent/agent.py`, not new infrastructure.

### The largest genuine blockers

1. **A false clear-state defect on the monthly funded path (correctness, not
   capability).** See §C.20 and the box in §B. On a funded-only approval the
   resolver never runs the concentration tests, and the Risk Review then emits
   the *unqualified* "No material portfolio risks were identified from this
   update." This is precisely the failure mode the package was written to
   prevent, and the test fixture masks it. **This must be fixed before the
   feature is switched on at all**, autonomous agent or not.
2. **No funded-side materiality engine.** The monthly card is assembled directly
   from `movement_summary.period_movement` with no threshold gating, no
   omissions and no severity. The richer monthly review the brief asks for has
   no materiality layer beneath it.
3. **One batch, many recipients.** `trakt_notifications/sources.py:17-19` states
   the design rule explicitly: *"One resolution per update, not per recipient."*
   The outbox already fans out per recipient (`outbox.py:185`), but the *content*
   is shared — `delivery.py:168` renders from `batch.message(item.message_type)`.
   Personalisation requires a per-recipient message layer that does not exist.
4. **Telemetry has no history.** `mi_agent_api/query_telemetry.py` landed on
   2026-09-01 (commit `31eb910`). The schema is excellent; the corpus is empty.
   A preference profile cannot be derived from data that has not accumulated.
5. **The agent tool surface is funded-only.** `trakt_tools/handlers/` registers
   27 tools, none of which reach the weekly pipeline; `covenants.py:27` refuses a
   pipeline-pinned resource outright. The readiness agent cannot see the pipeline.

### Does the architecture already contain the necessary primitives?

**Yes, with two exceptions.** Every analytical primitive the worked example in
§5 of the brief requires — headline movement, dimensional attribution,
concentration utilisation, pipeline drivers, forward concentration under an
expected-completion state — exists and is governed. The two genuinely absent
primitives are a **user preference profile** and a **per-recipient message
rendering path**. Everything else is wiring.

**Percentage of required capability already present: ~75%** (derivation in §C).

---

## B. Current architecture (evidence-based)

### B.1 Ingestion → canonical → snapshot

```
Blob upload (container TRAKT_BLOB_CONTAINER, e.g. raw-v2)
   │  function_app.py:113  on_raw_blob_event  → _dispatch (function_app.py:125)
   ▼
apps/blob_trigger_app/occ_intake.handle_arrival
   │  registers file, recognises roles, evaluates pack + config readiness,
   │  mints the immutable run manifest, starts the governed run
   ▼
engine/orchestrator_agent  (per portfolio)
   Onboarding → Transformation → Validation → provenance stamp (engine/provenance.py)
   ▼
engine/platform_assembler.py / engine.assembler_agent
   latest accepted snapshot per source_portfolio_id → platform_canonical_typed.csv
   ▼
operations_control/engine.py:2466  approve_publication
   promotes artefacts to processed latest + period, then calls _notify_publication
```

The `_READY.json` sentinel path is explicitly retired (`function_app.py:130-133`);
OCC owns readiness. Legacy Streamlit (`analytics/streamlit_app_erm.py`) is not on
this path and was excluded from all conclusions below.

### B.2 Canonical → MI

Two governed dataset families, deliberately kept apart:

| | Funded | Pipeline |
|---|---|---|
| Tape | `18_central_lender_tape.csv` | `18a_central_pipeline_tape.csv` / governed KFI extracts |
| Discovery | `mi_agent_api/snapshots.py:159` `discover_snapshots` | `mi_agent_api/pipeline_contract.py:390` `weekly_extract_inventory` |
| Prep | `mi_agent_api/funded_prep.prepare_funded_mi_dataset` | `mi_agent_api/pipeline_prep.prepare_pipeline_mi_dataset` |
| Contract | `mi_agent_api/mi_dataset_contract.build_dataset_contract` | `pipeline_contract.py:695` `build_pipeline_dataset_contract` |
| Reporting date | `snapshots.py:91` `infer_reporting_date` (column first, then `mi_YYYY_MM` → month-end) | folder/filename date, `pipeline_contract.py:97` `_extract_date` |
| Prior period | `snapshots.py:202` `find_prior_run` | `movement_detail.py:458` `select_pair` |

### B.3 MI → MI Query Agent

```
question ──► ParsedQuestion.parse            mi_agent/parsed_question.py
             (llm_query_parser.parse_with_repair; deterministic by default)
        └──► MIQuerySpec                     mi_agent/mi_query_spec.py:279
                │
                ├─► try_route                mi_agent_api/chat_routing.py:3754
                │     recogniser registry → 15 governed route handlers
                │     (compare, evolution, forecast, scenario, concentration
                │      tests, risk, bridge, cohort progression, geo, conversion,
                │      portfolio summary, period movement, stage movement, …)
                │
                └─► run_mi_agent_query       mi_agent/mi_agent_workflow.py:479
                      └─► execute_mi_query   mi_agent/mi_query_executor.py:1511
                          (flat point-in-time: measures / filters / group-bys /
                           ranking / share / contribution / distribution)
```

Production entry point: `POST /mi/query` (`mi_agent_api/app.py:1914`) → `mi_service`.

### B.4 MI → Teams

```
trakt_notifications/sources.py:104  resolve(...)
   pipeline side (sources.py:141):
     datasets._pipeline_history          → historical completion model
     concentration_tests_api.compute_concentration_tests
     evolution.pipeline_funnel_evolution (lag_weeks passed as the dashboard does)
     evolution.pipeline_evolution        → weekly series + fiveWeekAverage
     insight_engine.build                → THE governed insight set
     concentration_tests_api.compute_pipeline_drivers (warning/breach tests only)
   funded side (sources.py:261):
     movement_summary.period_movement
   ▼
generate.build → NotificationBatch (2 messages, deterministic ids)
   portfolio_update.build   trakt_notifications/portfolio_update.py:51
   risk_review.build        trakt_notifications/risk_review.py:91
   ▼
Outbox (per message × per recipient)  → DeliveryWorker → Adaptive Card
```

`trakt_notifications` contains **no LLM call of any kind** — verified by grep.
`recommendation.py:26` states it: *"No language model participates."*

> ### ⚠ Defect found during this review — monthly funded false clear state
>
> `operations_control/engine.py:2541` passes `datasets=[run.delivery.get("dataset", "funded")]`
> — a single dataset, defaulting to `funded`. For a funded-only approval,
> `generate.update_type_for` returns `UPDATE_FUNDED`, so `trigger.py:167-169`
> calls `sources.resolve(want_pipeline=False, want_funded=True)`.
>
> `sources.resolve` (sources.py:132-136) then **never calls `_resolve_pipeline_side`**,
> which is the only place `compute_concentration_tests` is invoked. Therefore:
>
> - `inputs.concentration is None`
> - `inputs.brief is None`
> - `inputs.unavailable == {}` — nothing was *recorded* as unavailable, because
>   nothing was attempted
>
> `risk_review._collect` (risk_review.py:149) reads `emergingRisks` off
> `inputs.concentration` and returns `[]`. `_clear_message` (risk_review.py:244)
> computes `partial = bool(unavailable)` → `False` → emits `CLEAR_STATEMENT`:
> the **unqualified** claim that no material portfolio risks were identified.
>
> No concentration test ran. No risk limit was evaluated. The monthly funded
> card asserts a clear state on the strength of checks that did not execute —
> the exact failure the module docstring (risk_review.py:17-27) exists to prevent.
>
> The test suite does not catch it: `tests/notifications/conftest.py:143` sets
> `inputs.concentration` by hand in the `funded_inputs()` fixture, so
> `test_messages.py` exercises the builder with concentration present while
> production never supplies it on that path.
>
> **Smallest fix:** resolve concentration on the funded side too (it is a funded
> measure — `_concentration` at sources.py:214 already takes `output_root`, not
> `pipeline_root`), or, at minimum, record `CAP_CONCENTRATION` as unavailable
> when `want_pipeline` is `False` so the message degrades to `PARTIAL_STATEMENT`.
> Either is a handful of lines. It should ship before the flag is turned on.

### B.5 Telemetry

```
mi_service._finish              mi_agent_api/mi_service.py:343
  ├─ emit_audit_event(result)   → App Insights "trakt.audit" (identifiers only)
  └─ query_telemetry.record     mi_agent_api/query_telemetry.py:231
       → OpsStore.save_mi_query  operations_control/stores.py:517
         (client-scoped, day-partitioned: mi_query_uri(client, day, query_id))
Operator surface: operations_control/api/mi_query_routes.py:159
```

---

## C. Capability matrix

| # | Capability | Status | Evidence |
|---|---|---|---|
| 1 | Snapshot comparison | **EXISTS** | funded: `snapshots.py:202` `find_prior_run`, `:579` `compute_funded_snapshot` (loan-id-level new/exited at `:692-706`); pipeline: `movement_detail.py:458` `select_pair` over `pipeline_contract.py:390` |
| 2 | Pipeline growth | **EXISTS** | `evolution.py:701` `pipeline_evolution`, `:808` `five_week_average`, `movement_detail.py:341` `build_movement_detail` (`DETAIL_PIPELINE`) |
| 3 | Funded growth | **EXISTS** | `movement_summary.py:255` `period_movement`; `evolution.py:251` `funded_evolution` |
| 4 | Pipeline→funded movement | **EXISTS** | `pipeline_history.py:47` `build_historical_completion_model` (tracks the same case across weekly snapshots by KFI/account/application id); `evolution.py:934` `pipeline_funnel_evolution` (lagged conversion); `movement_detail.py:657` `stage_transition_events`, `:749` `transition_matrix`, `:847` `stage_reconciliation` |
| 5 | Product drivers | **PARTIAL** | funded: `evolution.py:324` `funded_bridge` accepts any dimension column. pipeline: `movement_detail.py:78` `DIMENSIONS` is `(brokers, regions)` only — **product_type is a governed pipeline dimension** (`config/mi/pipeline_field_contract.yaml:151`) but is not an attribution dimension |
| 6 | Regional drivers | **EXISTS** | `movement_detail.py:78`; `movement_summary.py:134` `_regional_exposure`, `primaryRegion`; `evolution.py:324` |
| 7 | LTV movement | **EXISTS** | `insight_metrics.py:124` `weighted_ltv` (coverage-gated), `:182` `band_mix` over `LTV_BAND`; funded delta `wa_ltv_points` in `movement_summary.py:255` |
| 8 | Borrower-age analysis | **PARTIAL** | levels: `snapshots.py:656` `wa_age` KPI, `_STRAT_DIMS` "By borrower age"; series: `evolution.py` `avg_borrower_age`; pipeline: `pipeline_prep.py:362` `_derive_youngest_age`. **No materiality generator, no period-over-period gate.** |
| 9 | Joint-borrower analysis | **PARTIAL** | derived on both sides (`pipeline_prep.py:197` `_derive_borrower_type`; funded `borrower_type`); `snapshots.py:664` `pct_single_borrowers` KPI. **Not in `_STRAT_DIMS`, not an attribution dimension, no movement measure.** |
| 10 | Vintage analysis | **EXISTS** | `cohorts.py:200` `cohort_analysis`, `:406` `cohort_formation`, `:497` `cohort_static_pool`; `evolution.py:526` `funded_cohort_progression`; routes `/mi/cohorts`, `/mi/cohorts/vintages`, `/mi/cohorts/progression` |
| 11 | Concentration analysis | **EXISTS** | `mi_agent/concentration_tests/` (approved config → `evaluation`), `concentration_tests_api.py:340` `compute_concentration_tests`, `:493` `compute_pipeline_drivers`, `:541` `compute_history` |
| 12 | Risk-limit status | **EXISTS** | `risk_limits.py:585` `compute_risk_limits` (Schedule 8 extracted + actuals, green/amber/red/needs_review/unavailable, movement vs prior run); approved-config precedence disclosed at `concentration_tests_api.py:5-18` |
| 13 | Forecast | **EXISTS** | `forecast_extrapolation.py:358` `build_extrapolation` (3 models); `evolution.py:1129` `forecast_evolution`; **forward concentration**: `mi_agent/concentration_tests/forward.py:250` `evaluate_forward_states` (funded / expected_forecast / full_pipeline), `:448` `expected_breach_horizon`, `:336` `pipeline_drivers` |
| 14 | Securitisation signals | **PARTIAL** | tools exist: `regulatory_readiness` (Annex 2 blocking vs ND-permitting gaps, `trakt_tools/handlers/readiness.py:383`), `evaluate_rule_packs` (`trakt_core/readiness.py:468`, one fact / many rulebooks / authority-tagged), `data_completeness`, `list_validation_exceptions`. **None is consumed by the notification pathway.** |
| 15 | MI user telemetry | **EXISTS** | `query_telemetry.py:147` `build_record` — 40+ fields including resolved interpretation. **Caveat: landed 2026-09-01; no accumulated corpus.** |
| 16 | User preference profile | **MISSING** | grep for `preference_profile` / `user_preference` / `affinity` returns nothing |
| 17 | Proactive Teams messaging | **EXISTS** | `teams_client.py` (Bot Framework client-credentials + `POST {serviceUrl}/v3/conversations/{id}/activities`, service-URL host allowlist), `delivery.py` worker, `function_app.py:83` 5-minute timer |
| 18 | Per-user message routing | **PARTIAL** | routing exists (`outbox.py:185` one item per message × recipient; `recipients.py:243` `select` with five explicit gates); **content does not** — `delivery.py:168` renders the shared batch message |
| 19 | Semantic MI execution API | **PARTIAL** | see §C-note below |
| 20 | Evidence / provenance | **EXISTS** | `insight_contract.py:115` deterministic `insight_id`; `contract.py` `notification_batch_id` + `reporting_key`; `methodology_versions`; `mi_agent/execution_receipt.py`; `mi_agent_api/movement_receipt.py` ("sufficient to re-derive the answer"); `trakt_core/envelope.GovernedResult` |

**Score: 12 EXISTS + 6 PARTIAL + 2 MISSING → (12 + 3) / 20 = 75%.**

### C-note — the semantic execution seam (§4 of the brief)

The brief asks whether the MI execution layer can be invoked programmatically
without pretending to be a human. **Yes — through two seams, and one is already
used in production for exactly this purpose.**

**Seam 1 — the spec executor (flat shapes).**
`execute_mi_query(spec: MIQuerySpec, data, semantics)` at
`mi_agent/mi_query_executor.py:1511` is a pure spec-in / result-out function,
exported from `mi_agent/__init__.py`. `MIQuerySpec` (`mi_query_spec.py:279`) is a
plain serialisable dataclass with `from_dict`/`to_dict`. It covers measures,
multi-measure composition, filters (with a rich predicate normaliser), grouping,
hierarchy, ranking/top-N/sort, share, contribution, distribution, buckets and
the portfolio lens. Fail-closed dimension invariant at `mi_query_contract.py:29-38`.

`ParsedQuestion` (`mi_agent/parsed_question.py`) is likewise a plain dataclass, so
`run_mi_agent_query(question=..., parsed=ParsedQuestion(question, spec, meta))`
executes a machine-built spec through the full workflow (validation, receipt,
lens, reconciliation) without invoking the parser.

**Seam 2 — the governed service functions (composite shapes).**
The compositional, temporal, risk and forecast capabilities are **not** reachable
from a `MIQuerySpec`; they live behind `try_route` (`chat_routing.py:3754`),
which takes a `question: str` and dispatches on recognisers. But the services
those routes call are ordinary typed Python functions. **`trakt_notifications/sources.py`
already proves the pattern**: it calls `evolution.pipeline_evolution`,
`evolution.pipeline_funnel_evolution`, `concentration_tests_api.compute_concentration_tests`,
`movement_summary.period_movement` and `insight_engine.build` directly, with
arguments, and its docstring names the containment rule (sources.py:2-8).

**Seam 3 — the tool registry (agent-facing).**
`trakt_tools/registry.py` + `trakt_tools/handlers/` publish 27 governed tools with
JSON Schemas, reached through `execute_governed_tool` with capability and
entitlement checks. `readiness_agent/session.py:127` is the agent door.

**Recommendation: do not build an English-question round trip, and do not build a
third execution engine.** Use Seam 3 for the investigation loop (it already
carries authorisation, refusal semantics and an audit transcript) and extend it
with the pipeline tools it lacks. Use Seam 2 directly where a tool would be
gratuitous. Seam 1 is available for ad-hoc stratifications the tool set does not
name.

There is one further object worth knowing about: `mi_agent_api/analytical_plan.py`
is a **compositional plan layer** that takes a `QuestionInterpretation` (not a
question string — `assert_no_question_read` enforces it structurally, docstring
lines 27-32) and builds a `Plan` over six named primitives (`stack_periods`,
`select_population`, `resolve_measure`, `group`, `rank`, `compare`). Five routes
are converted onto it (`portfolio_summary`, `period_movement`, `geo_exposure`,
`temporal_compare`, `funded_bridge`). It is the right long-term home for a
programmatic plan API, but `QuestionInterpretation` is a *linguistic* contract
carrying raw-text spans (`question_interpretation/schema.py:11-23`), so
synthesising one from an agent is awkward today. **Not on the critical path.**

---

## D. Existing MI analytical universe

### D.1 Measures

| Family | Measures | Source |
|---|---|---|
| Funded levels | funded balance, loan count, average loan balance, WA current LTV, WA original LTV, WA interest rate, WA months on book, WA youngest borrower age, WA property value, single-borrower share, NNEG exposure (ERM) / arrears balance | `snapshots.py:579` |
| Funded series | `funded_balance`, `loan_count`, `avg_balance`, `wa_ltv`, `wa_interest_rate`, `avg_borrower_age` | `evolution.py:116` `assemble_funded_evolution` |
| Pipeline levels | `pipeline_amount`, `pipeline_case_count`, `weighted_expected_funded_amount`, stage breakdown, expected-completion breakdown | `pipeline_contract.py:845` `compute_pipeline_snapshot` |
| Pipeline movement | new / removed / progressed_out / increased / decreased / unchanged components, per dimension, reconciling to the net | `movement_detail.py:98` `COMPONENTS`, `:168` `movement_components` |
| Weekly metrics | ticket size (mean + median, separately gated), weighted LTV with coverage, band mix, data quality | `insight_metrics.py:88,124,182,233` |
| Concentration | share, top-N share, utilisation, headroom, breach amount, status; 39 registered metric evaluators | `mi_agent/concentration_tests/metrics.py`, `trakt_core/readiness.py:41` |
| Conversion | weekly conversion by value/count, lagged, cohort conversion, sufficiency flag | `evolution.py:934` |
| Forecast | run-rate model, KFI×conversion model, weighted pipeline model, scenario bands, milestone dates | `forecast_extrapolation.py:83,245,358` |
| Cohort | per-vintage balance, count, book share, WA LTV / rate / months-on-book; static pool; formation | `cohorts.py:200,406,497` |
| Contractual | WAL, YTM, CPR, SMM, CDR, cure rate, default stock, stale balance | `trakt_tools/handlers/contractual.py`, digest keys at `readiness_agent/session.py:221` |

### D.2 Dimensions

- **Funded stratifications** (`snapshots.py:333` `_STRAT_DIMS`): LTV, borrower age,
  region, rate band, product, origination vintage, account status, protected equity.
- **Funded attribution** (`evolution.py:324` `funded_bridge`): any present column,
  candidate lists supported, **fails closed** on a requested-but-absent dimension.
- **Pipeline dimensions** (`config/mi/pipeline_field_contract.yaml`):
  `borrower_type`, `geographic_region_obligor`, `broker_channel`, `product_type`,
  plus derived LTV / youngest age / stage / expected completion.
- **Pipeline movement attribution** (`movement_detail.py:78`): **brokers, regions only.**
- **Portfolio lens** (`mi_agent/portfolio_lens.py`): total / direct / acquired / cohort.
- **Bucket families** (`config/mi/buckets.yaml`): ticket, LTV, rate, age, equity.

### D.3 Filters

`mi_query_executor.py:497` `governed_predicate_mask` + `:668` `_apply_filters`:
equality, membership, numeric ops (`gt`/`ge`/`lt`/`le`/`between`), domain-value
resolution, case-insensitive categorical matching. Unfoldable predicates are
**never dropped** — they land in `spec.unavailable_filters` and surface as
warnings (`mi_query_spec.py:82-90`).

### D.4 Periods and comparisons

`latest` | `as_of` | `compare` | `trend` (`TEMPORAL_MODES`, `mi_query_spec.py:56`);
trend grains daily/weekly/monthly/quarterly; `compare_periods`; funded bridge with
`start_period` or `window_periods`; five-week trailing average and trailing flow
(`evolution.py:808,889`); pipeline lag handling for conversion (`evolution.py:916`).

### D.5 Analytical shapes NOT supported

- Redemption / prepayment / performance **curves** in the MI path —
  `cohorts.py:8-11` says so explicitly.
- Loan-level pipeline→funded lineage as a *join*. The relationship is
  established by **observing the same case identifier across consecutive weekly
  snapshots** (`pipeline_history.py:1-16`), not by a key into the funded tape.
  Conversion is empirical and stage-based, gated at `MIN_OBSERVATIONS = 12`.
- Cross-portfolio (multi-tenant) analysis. Tenancy is a path segment
  (`recipients.py:166-171`); there is no cross-client analytic and there should not be.
- Product attribution on pipeline movement (§C.5).
- Any funded-side materiality gating (§H.2).

---

## E. Telemetry assessment

### E.1 Actual schema — `mi_agent_api/query_telemetry.py:147`

Against the fields the brief speculates about:

| Asked for | Present | Field |
|---|---|---|
| user_id | ✅ | `user_id` (= `audit.actor_id`; for Entra paths this is the **`oid`**, `react_auth.py:337`) |
| client_id | ✅ | `client_id` (= `result.tenant_id`) |
| timestamp | ✅ | `asked_at`, `day` |
| raw_question | ✅ | `question` |
| resolved intent | ✅ | `interpretation.intent`, `.output_type`, `.execution_mode` |
| measure | ✅ | `interpretation.metric`, `.measures`, `.aggregation`, `.weight_field` |
| filters | ✅ | `interpretation.filters`, `.state_filters`, `.portfolio_lens`, `.segment` |
| dimensions | ✅ | `interpretation.dimension`, `.dimensions`, `.hierarchy`, `.bucket_field`, `.concentration_dimension` |
| time period | ✅ | `.as_of_date`, `.reporting_date`, `.start_date`, `.end_date`, `.temporal_mode`, `.trend_grain` |
| comparison period | ✅ | `.baseline_date`, `.current_date`, `.compare_periods` |
| semantic plan | ✅ (projection) | `_spec_interpretation` — 30 whitelisted spec keys, `query_telemetry.py:119-135` |
| execution route | ✅ | `route`, `capability`, `engine`, `execution_mode` |
| answer status | ✅ | `outcome` (ANSWERED/REFUSED/ERROR), `governed_status` |
| refusal reason | ✅ | `refusal_reason`, `error_code`, `error_category`, `warnings` |
| response | ✅ | `answer` (verbatim, as the user saw it) |
| latency | ✅ | `duration_ms` |

Plus, beyond the brief: `snapshot_id`, `content_hash`, `source_kind`,
`reporting_period`, `dataset_view`, `data_source_kind`, `parser` provenance,
`row_count`, `lens_applied`, `artifact_kinds`, `metric_defaulted`,
`unavailable_filters`, `organisation_id`, `correlation_id`, and an operator
review block (`review.classification`, 8-value vocabulary).

### E.2 Can preferences be derived from resolved semantics rather than re-reading English?

**Yes — this is the schema's strongest property for this purpose.** The brief's
worked example maps directly onto stored fields with **no LLM re-interpretation**:

```
high_affinity  ← count of ANSWERED records per (user_id, interpretation.dimension)
                 and per (user_id, interpretation.metric) over a trailing window
medium_affinity← the same counts at a lower rank
low_evidence   ← governed dimensions/measures with zero records for this user
```

"Joint borrowers / youngest borrower age / borrower age distribution" resolve to
`borrower_type` and `youngest_borrower_age` in `interpretation.dimension` /
`.metric`. "Pipeline / conversion / London / funding requirement" resolve to
`dataset_view`, `route`, `interpretation.filters` and
`weighted_expected_funded_amount`. These are governed field keys, already
disambiguated by the parser and already validated by the executor's fail-closed
dimension invariant. Re-asking an LLM what the user meant would discard that work
and reintroduce ambiguity the system has already resolved.

Two important qualifications:

- **`metric_defaulted`** must be honoured. A measure the parser *substituted*
  is not a measure the user *chose* (`mi_query_spec.py:287-295`). Counting it as
  affinity would manufacture a preference from a default.
- **Refusals are signal, not noise.** A user repeatedly asking something Trakt
  refuses is expressing an interest Trakt cannot yet serve. That belongs in the
  profile as an explicit "wanted, unavailable" category, not as absence.

### E.3 Additional telemetry to retain from now on

Small, and only three items:

| Addition | Why | Size |
|---|---|---|
| `microsoft_tenant_id` (from `audit.microsoft_tenant_id`, already on `AuditMetadata` per `mi_service.py:371`) | Teams `recipient_id` is `sha256(microsoft_tenant_id \| entra_object_id)` (`recipients.py:57`). Without the directory id, joining a telemetry `user_id` to a Teams recipient requires an indirection through `config/organisations.yaml`. One line in `build_record`. | 1 line |
| `interpretation.route_family` (the settled analytical family from `mi_workflows/analytical/intent.py`) | The family is already computed at `chat_routing.py:3854` and then discarded. It is a much better affinity key than the route id, which is an implementation detail. | ~3 lines |
| A retention/decay note in the record or the profile builder | A preference from March should not outweigh one from last week. Recording `day` already permits windowing; the decision belongs with the profile, not the record. | 0 (design) |

**Everything else needed is already recorded.** No schema redesign. Note that the
store is day-partitioned per client (`stores.py:162`), so a trailing-window scan
per user is cheap.

**The binding constraint is history, not schema.** Telemetry started on
2026-09-01. With 7 users, a meaningful affinity profile needs on the order of
weeks-to-months of usage. **Personalisation must therefore be Phase 3, not
Phase 1**, whatever the architecture allows.

---

## F. Autonomous investigation feasibility

The brief's worked example, step by step, against demonstrated capability:

| # | Step | Today | Evidence |
|---|---|---|---|
| 1 | "Funded balance increased £8.4m" | ✅ | `movement_summary.py:255` `period_movement` → `delta.funded_balance` |
| 2 | "Analyse source of growth" | ✅ | `evolution.py:324` `funded_bridge(dimension_col=...)` — per-category deltas sum exactly to the net |
| 3 | "65% originated from Lump Sum" | ✅ | `funded_bridge` on `erm_product_type` / `product`; `_STRAT_DIMS` carries "By product" |
| 4 | "Inspect geography" | ✅ | `funded_bridge` on the region family; `movement_summary.py:134` `_regional_exposure`; `primaryRegion` with a stated causality discipline (`movement_summary.py:36-41`) |
| 5 | "London = 28.4% vs 30% limit" | ✅ | `concentration_tests_api.py:340` → utilisation, headroom, status against the **operator-approved** configuration |
| 6 | "Inspect pipeline: London pipeline £5.1m" | ✅ | `concentration_tests_api.py:493` `compute_pipeline_drivers` → `forward.py:336` `pipeline_drivers` (governed contributor aggregates, never per-case) |
| 7 | "Estimate governed forward concentration" | ✅ | `forward.py:250` `evaluate_forward_states` — three labelled states (`funded`, `expected_forecast`, `full_pipeline`) with a governed treatment table per metric family; `:448` `expected_breach_horizon` |
| 8 | "Material finding" | ✅ | `forward.py:517` `identify_emerging_risks` — fixed rank order: current breach → expected breach → low expected headroom → deterioration → stress-only → limitation |
| 9 | **Decide which of 1–8 to run, and when to stop** | ❌ | **The only missing step.** |
| 10 | Rank findings across families | ⚠️ | `insight_engine.py:52` `rank_key` ranks *weekly pipeline* insights; `identify_emerging_risks` ranks *concentration* findings; `risk_review.py:149` merges the two. No cross-family ranker spanning funded movement, cohort, forecast and readiness. |
| 11 | Weight to this user | ❌ | No preference profile (§C.16). |

**Eight of eleven steps are shipped capability. The gap is the controller, not the analytics.**

And even the controller has a working precedent. `readiness_agent/agent.py:218`
`run_assessment` is exactly this loop:

- objective in, no metric list, no ordering (`agent.py:44-50`);
- the model calls `portfolio_capabilities` first to learn what is knowable
  *before computing anything* (`agent.py:61-63`);
- six distinct unavailability states the model must not conflate
  (`agent.py:65-70`) — `UNAVAILABLE` ≠ `NOT_APPLICABLE` ≠ `METHODOLOGY_NOT_APPROVED`;
- **"You perform NO arithmetic"** enforced *structurally*, not by prompt:
  `GovernedSession` hands out three verbs and no DataFrame
  (`session.py:10-25`) — *"Denying the agent raw data is not a security measure
  here; it is a correctness measure."*;
- "Drill into what looks interesting, and stop when further calls would not
  change your conclusion" (`agent.py:97-101`);
- structured submission via one tool (`agent.py:112`), scored not parsed;
- step ceiling as the only control (`agent.py:41`, `DEFAULT_MAX_STEPS = 40`);
- efficiency telemetry including `repeated_calls` as a looping signal
  (`session.py:176-197`).

**The two changes needed to point it at portfolio review:**

1. **Pipeline tools.** The registry is funded-only. `covenants.py:27` refuses a
   pipeline- or forecast-pinned resource by design. A portfolio review agent
   needs `pipeline_snapshot`, `pipeline_movement`, `pipeline_funnel`,
   `forward_concentration` and `period_movement` as registered tools — each a
   thin wrapper over a function that already exists (`pipeline_contract.py:845`,
   `movement_detail.py:341`, `evolution.py:934`, `forward.py:250`,
   `movement_summary.py:255`), following the existing handler pattern.
2. **A period-scoped objective**, replacing `OBJECTIVE` at `agent.py:46`, that
   names the two snapshots rather than the portfolio.

---

## G. Teams delivery assessment

### G.1 Current capability — more than the brief assumes

| Question | Answer | Evidence |
|---|---|---|
| Azure/Teams architecture | Bot Framework bot + declarative Copilot agent in one Teams app package | `tests/notifications/test_teams_package.py:56,62` |
| Bot entry point | `POST /v1/teams/bot/messages` | `teams_bot.py:256`, mounted `app.py:1963` |
| Inbound authentication | Bot Framework JWT: signature via JWKS from `login.botframework.com`, issuer `api.botframework.com`, audience = bot app id, expiry. **Fails closed when unconfigured** | `teams_bot.py:102-136` |
| Tenant isolation | Trakt tenant from deployment config only (`TRAKT_TEAMS_TRAKT_TENANT`); Microsoft tenant from validated `channelData`; recipient store keyed by tenant path segment | `teams_bot.py:189-200`, `recipients.py:166-171` |
| User identity | `aadObjectId` (directory identity), never display name — *"it is user-controlled in most directories"* | `teams_bot.py:166-175` |
| Teams user → Trakt user | `recipient_id = sha256(microsoft_tenant_id \| entra_object_id)[:24]` | `recipients.py:57` |
| User → client | **Operator mapping only.** `store.authorise(tenant, rid, portfolio_contexts, actor)` | `recipients.py:228` |
| Proactive messages | ✅ Implemented | `teams_client.py`, `delivery.py` |
| Conversation references retained | ✅ Verbatim, in the SDK's own shape | `teams_bot.py:142-156`, `recipients.py:77-79` |
| Can initiate without inbound message | ✅ — that is the whole design; capture happens at install | `teams_bot.py:234` `capture_conversation` |
| Adaptive Cards | ✅ | `trakt_notifications/cards.py`, severity→container style at `contract.py:79` |
| Deep links to React | ✅ | `trakt_notifications/deep_links.py` — per-tab, per-context, per-as-of-date |
| A2A integration | Exists (`trakt_a2a/server.py`) but is **not** on the production API path — its only caller is `scripts/run_a2a_eval.py` | grep |

### G.2 Can a user receive another client's information?

Five gates, each explicitly reported on failure (`recipients.py:243`
`select` → `Refusal`): notifications enabled → app installed → conversation
captured → **portfolio context authorised** → Microsoft tenant matches. The
tenant on the batch comes from the governed run, not a caller — `trigger.py:106-108`
states *"There is no parameter by which a caller can name a different tenant,
which is what makes cross-tenant delivery unreachable rather than merely guarded."*

Additionally: the service URL is host-allowlisted before a send
(`teams_client.py:40-48`) so a tampered conversation reference cannot redirect a
card *and the bot's credentials* to an arbitrary endpoint.

### G.3 Board vs operational users

**Not modelled.** `portfolio_contexts` is a list of portfolio scopes
(`total`/`direct`/`acquired`/cohort), not a role. Today a board user and an
operational user authorised for `total` receive an identical card. If board
users are to see a different message, the distinction has to be represented —
and the cheapest honest representation is a **recipient role tag** consumed by
the personalisation layer, *not* a second entitlement model.

### G.4 Is A2A required?

**No.** Recommended invocation path:

```
approve_publication (existing)
  → trakt_notifications.trigger.on_publication_approved (existing hook)
      → PortfolioReviewController          ← NEW, in-process
          → GovernedSession                  (readiness_agent/session.py, reused)
              → execute_governed_tool         (trakt_tools, reused)
          → findings + evidence
      → generate.build (existing, extended for per-recipient messages)
      → outbox (existing) → DeliveryWorker (existing) → Teams
```

A2A is a **delegation boundary for external agents** (`trakt_a2a/server.py:1-32`) —
it exists so a client's own agent can ask Trakt for an assessment. Routing an
internal Trakt agent through a JSON-RPC task boundary to reach a function in the
same process would add polling, task state and serialisation for no isolation
benefit, and it would put a network hop inside an approval-triggered flow the
whole design keeps synchronous-and-cheap.

**Keep the Teams bot as a pure delivery/interaction channel.** It already is one
— `teams_bot.py:6-10`: *"It answers no questions and returns no portfolio data."*
That separation is worth preserving.

### G.5 Likely changes to the Teams layer

| Change | Size |
|---|---|
| Per-recipient message on the outbox item (or a per-recipient batch), so `delivery.py:168` renders personalised content | Moderate — touches `contract.py`, `outbox.py`, `delivery.py`, `store.py` |
| Recipient role tag (`board` / `operational`) on `Recipient` + a `--role` flag on `cli.py authorise` | Small |
| Nothing else | — |

---

## H. Missing architecture

Only genuinely missing components. No duplicate engines are proposed.

### H.1 Funded/monthly insight generators

| | |
|---|---|
| **Why required** | The monthly funded card is assembled with **no materiality gate**: `portfolio_update._funded_items` (portfolio_update.py:365) emits loan count, cohort contribution, LTV delta and primary region unconditionally. A £2k move and a £24m move produce the same card. The weekly pipeline side has nine gated generators; the funded side has none. |
| **Extends** | `mi_agent_api/insight_generators.py` + `insight_config.py` + `insight_engine.py:230` (add a funded step list); new types in `insight_contract.py:54` |
| **Size** | Moderate. ~6–8 generators (balance movement, product mix shift, regional mix shift, LTV movement, borrower-structure movement, vintage development, rate profile, exits/runoff), each ~40 lines following the existing `Result = (List[Insight], List[Omission])` shape. Thresholds go in `config/mi/insights.yaml` under a new `funded:` section — **relative only**, per the file's own rule (`insights.yaml:11-16`). |
| **AI vs deterministic** | **Deterministic.** This is the layer that must not become an LLM judgement. |
| **Production risk** | Low. Additive; the engine already isolates a failing generator into an `Omission` with category `error` (`insight_engine.py:250-255`). |

### H.2 Portfolio Review investigation controller

| | |
|---|---|
| **Why required** | Step 9 of §F. Nothing decides *which* governed analyses to run for a given period, when a result warrants a drill-down, or when further analysis is immaterial. |
| **Extends** | `readiness_agent/agent.py` (the loop) and `readiness_agent/session.py` (the door) — **reused, not reimplemented**. New: a period-scoped objective, a `submit_review` tool schema replacing `SUBMIT_TOOL`, and a snapshot-pair resolver that pins the controller to (current, prior). |
| **Size** | Moderate. ~250–350 lines, most of it the submission schema and the objective. The loop itself is ~80 lines and already exists. |
| **AI vs deterministic** | **AI**, bounded exactly as the readiness agent is: no arithmetic, no DataFrame, refusals are findings. |
| **Production risk** | Medium — bounded by the step ceiling and by the fact that it cannot compute. Mitigate by shipping the deterministic Phase 1 brief first and running the controller in shadow against it. |

### H.3 Pipeline tools in the governed tool registry

| | |
|---|---|
| **Why required** | The tool surface is funded-only; a weekly review needs the pipeline. |
| **Extends** | `trakt_tools/registry.py` + a new `trakt_tools/handlers/pipeline.py`, each handler wrapping an existing function (see §F). Also requires extending `trakt_core/resource.py` or the covenant guard so a pipeline-scoped call is not blanket-refused (`covenants.py:27`). |
| **Size** | Small–moderate. 5 tools × ~50 lines of handler + schema. **No new analytics.** |
| **AI vs deterministic** | Deterministic. |
| **Production risk** | Low. The registry refuses duplicate names at startup (`registry.py:41-45`), and every tool is entitlement-checked. |

### H.4 User preference profile

| | |
|---|---|
| **Why required** | §C.16. |
| **Extends** | `operations_control/stores.py` `list_mi_queries` as the source; a new `mi_agent_api/preference_profile.py` producing `{user_id, high_affinity[], medium_affinity[], low_evidence[], wanted_unavailable[], window, computed_at, evidence_counts}`. |
| **Size** | Small. ~150 lines, **all deterministic counting over resolved semantics** — no LLM. |
| **AI vs deterministic** | **Deterministic.** An LLM re-reading historic questions would discard the parser's already-resolved field keys (§E.2). |
| **Production risk** | Low, but it is **useless until telemetry accumulates**. |

### H.5 Two-tier finding selection (mandatory vs personalised)

| | |
|---|---|
| **Why required** | §9 of the brief. This is the safety property that makes personalisation acceptable. |
| **Extends** | `insight_contract.py:129` `notification_eligible` — which already exists for exactly this and is documented as *"set but not acted on"* — plus the selector at `insight_engine.py:68`. |
| **Size** | Small. |
| **AI vs deterministic** | **Deterministic gate; AI ordering within the personalised tier only.** |
| **Production risk** | Low if built as described in §I. High if built as "let the model decide what the user sees". |

### H.6 Per-recipient message rendering

| | |
|---|---|
| **Why required** | §C.18 / §G.5. The current design resolves once and renders once. |
| **Extends** | `contract.py` (`NotificationBatch` gains per-recipient message variants, or `OutboxItem` gains a rendered message), `outbox.py:185`, `delivery.py:168`, `store.py`. |
| **Size** | Moderate — it changes a load-bearing contract, so the deterministic ids and the dedupe/correction semantics (`contract.py:23-40`) must be preserved. `message_id` and `item_id` already separate message identity from recipient identity, so the change is tractable. |
| **AI vs deterministic** | Deterministic plumbing. |
| **Production risk** | Medium — the idempotency and correction/supersession logic is the most safety-critical code in the package. |

### H.7 Fix: monthly funded false clear state

| | |
|---|---|
| **Why required** | See the box in §B.4. This is a correctness defect in shipped code, not a missing feature. |
| **Extends** | `trakt_notifications/sources.py:104` `resolve` — resolve concentration on the funded side (it is a funded measure), or record `CAP_CONCENTRATION` as unavailable when `want_pipeline` is `False`. Add a test that drives `sources.resolve` rather than the hand-built fixture. |
| **Size** | **Very small** — a handful of lines plus a test. |
| **Production risk** | Low to fix; **high to leave**. |

---

## I. Proposed minimum architecture

```
                    approve_publication                      EXISTING
                 operations_control/engine.py:2466
                              │
                    _notify_publication                      EXISTING
                 operations_control/engine.py:2516
                              │
                on_publication_approved                      EXISTING
                trakt_notifications/trigger.py:94
                              │
              ┌───────────────┴───────────────┐
              │                               │
      Snapshot pair resolver          Recipient selection      EXISTING
      snapshots.find_prior_run        recipients.select
      movement_detail.select_pair     (5 gates, refusals reported)
              │                               │
              ▼                               │
   ┌──────────────────────────┐               │
   │  Portfolio Review        │  NEW (H.2)    │
   │  investigation controller│               │
   │  = readiness_agent loop  │               │
   │    + period objective    │               │
   └──────────┬───────────────┘               │
              │ three verbs only               │
              ▼                                │
   GovernedSession.call        REUSED (readiness_agent/session.py:127)
              │                                │
              ▼                                │
   execute_governed_tool       EXISTING (trakt_tools/execution.py)
     ├─ funded tools (27)                      │
     └─ pipeline tools         NEW (H.3)       │
              │                                │
              ▼                                │
   ┌──────────────────────────────────────┐    │
   │  GOVERNED DETERMINISTIC LAYER        │    │   THE BOUNDARY
   │  evolution · movement_summary ·      │    │   No LLM below this line.
   │  movement_detail · concentration_    │    │   No LLM arithmetic above it.
   │  tests · forward · risk_limits ·     │    │
   │  forecast_extrapolation · cohorts ·  │    │
   │  snapshots · pipeline_contract ·     │    │
   │  readiness · execute_mi_query        │    │
   └──────────────┬───────────────────────┘    │
                  │ governed values only        │
                  ▼                             │
   Candidate findings                           │
                  │                             │
                  ▼                             │
   ┌──────────────────────────────────────┐     │
   │  Materiality / risk evaluation       │     │
   │  insight_engine.select   EXISTING    │     │
   │  + funded generators     NEW (H.1)   │     │
   │  identify_emerging_risks EXISTING    │     │
   └──────────────┬───────────────────────┘     │
                  │                             │
      ┌───────────┴────────────┐                │
      ▼                        ▼                │
  MANDATORY tier          PERSONALISED tier     │   NEW (H.5)
  notification_eligible   ranked by profile     │
  (severity ≥ attention)  within the remaining  │
  — every authorised      card budget           │
    recipient, always                           │
      │                        ▲                │
      │                        │                │
      │              User preference profile    │   NEW (H.4)
      │              ← OpsStore.list_mi_queries │   EXISTING SOURCE
      │                (resolved semantics)     │
      └───────────┬────────────┘                │
                  ▼                             │
   Narrative synthesis  (templates now,         │
   LLM prose in Phase 2)                        │
                  │                             │
                  ▼                             │
   Evidence object / receipt   EXISTING SHAPE   │
   insight_id · batch_id · reporting_key ·      │
   methodology_versions · movement_receipt ·    │
   tool-call transcript (session.transcript)    │
                  │                             │
                  ▼                             ▼
   ┌──────────────────────────────────────────────┐
   │ generate.build → per-recipient messages      │  H.6
   │ outbox.enqueue (message × recipient)         │  EXISTING
   │ DeliveryWorker → Adaptive Card → Teams       │  EXISTING
   └──────────────────────────────────────────────┘
```

**Two properties this diagram is designed to hold:**

1. **The AI never crosses the boundary line.** It selects *which* governed call
   to make and *how* to say the result. It never receives a frame. This is not a
   convention — `GovernedSession` has no method that returns one.
2. **The mandatory tier is computed before the profile is consulted, and is not
   subject to it.** A user's demonstrated interest in borrower age cannot cause a
   London concentration warning to be dropped, because the mandatory tier is
   filled from `notification_eligible` (severity ≥ `attention`) *first*, and the
   profile only ranks what remains within the card budget
   (`maximum_update_items: 5`, `maximum_risk_items: 3`). Personalisation is
   **additive weighting inside the residual budget, never a filter over the whole
   set** — and the Risk Review's clear-state discipline (§B.4) means the absence
   of a mandatory finding is itself an explicit statement.

### On the securitisation lens (§6 of the brief)

The brief's own instinct is right and the repository supports it. The Portfolio
Review Agent should **consume high-level readiness signals, not perform an
Annex 2 audit weekly**. Every signal it names already exists as a governed
output:

| Signal | Source |
|---|---|
| "Annex 2 coverage deteriorated" | `regulatory_readiness` → `blocking_gaps` vs ND-permitting gaps (`trakt_tools/handlers/readiness.py:383`) |
| "Blocking validation exceptions increased" | `list_validation_exceptions` (+ `lineage_available` guard, which distinguishes "clean" from "unknown") |
| "Specific vintage has recurring data gaps" | `data_completeness` × `cohorts.cohort_analysis` |
| "Concentration moved closer to a transaction limit" | `evaluate_rule_packs` — **authority-tagged**, so a Trakt screening FLAG can never be rendered as a warehouse BREACH (`trakt_core/readiness.py:10-22`) |
| "Out-of-perimeter exposure increased" | `evaluate_rule_packs` against a supplied criteria pack |

A full readiness assessment stays where it is: `readiness_agent`, invoked
deliberately, and reachable by an external party over A2A. The economics support
this — the weekly brief's cold cost is documented at 5.6s with the funnel alone
accounting for 3.5s of it (`insight_config.py:52-56`); a weekly Annex 2 sweep
would be a different order of cost for information that changes monthly at most.

---

## J. Delivery phases

### Phase 0 — Correctness and switch-on (prerequisite)

| | |
|---|---|
| **Modules touched** | `trakt_notifications/sources.py`, `tests/notifications/conftest.py`, `tests/notifications/test_messages.py` |
| **New modules** | None |
| **Tests** | A test that drives `sources.resolve(want_pipeline=False, want_funded=True)` end to end and asserts the Risk Review does **not** emit `CLEAR_STATEMENT`; the existing 80+ notification tests must stay green |
| **Acceptance** | A funded-only approval either evaluates concentration or names it as an unavailable check. No unqualified clear state is reachable without every check having run. |
| **Dependencies** | None |
| **Complexity** | **Very small.** Hours, not days. |

This is not optional and does not belong inside a later phase. The feature is
currently disabled (`config/mi/teams_notifications.yaml` `enabled: false`,
`recipients: []`; the brief itself is behind `TRAKT_MI_WEEKLY_BRIEF`), so the
defect is not live — but it is the first thing that becomes live on switch-on.

### Phase 1 — Deterministic weekly + monthly briefing

| | |
|---|---|
| **Modules touched** | `insight_engine.py` (funded step list), `insight_contract.py` (new types), `insight_config.py` + `config/mi/insights.yaml` (funded thresholds), `sources.py` (resolve funded generators' inputs), `portfolio_update.py` (consume gated funded insights instead of raw movement), `movement_detail.py:78` (add `("products", "product_type")`) |
| **New modules** | `mi_agent_api/insight_generators_funded.py` |
| **Tests** | Per-generator threshold tests mirroring `mi_agent_api/tests/test_insight_engine.py`; a monthly end-to-end mirroring `tests/notifications/test_end_to_end.py`; an assertion that an immaterial month produces an `Omission` with category `immaterial`, not silence |
| **Acceptance** | For a real month, every statement in §11/§12 of the brief that §C marks EXISTS is produced with a threshold behind it, an `insight_id` on it and an explicit omission where it was suppressed. Statements marked PARTIAL/MISSING are *absent*, not approximated. |
| **Dependencies** | Phase 0 |
| **Complexity** | **Moderate.** The single largest deterministic piece of work in the programme. |

**Supportability of the brief's §11 target output, today:**

| Statement | Status |
|---|---|
| "Pipeline increased by £Xm and X cases to £Ym" | ✅ `pipeline_evolution` + `fiveWeekAverage` |
| "Growth driven by: 1. Product A — £Xm, 2. Product B — £Xm" | ⚠️ **product is not a pipeline attribution dimension** (`movement_detail.py:78`) — one-line fix, then ✅ |
| "Region A and Region B represented X% of new pipeline" | ✅ `movement_components` contributors |
| "Weighted average LTV moved from X% to Y%" | ✅ `insight_metrics.weighted_ltv` (coverage-gated at 90%) |
| "X cases / £Xm moved to funded" | ✅ `DETAIL_COMPLETIONS` + funnel `latestFlowValue`/`latestFlowCount` |
| "Current funded balance is £Xm" | ✅ `period_movement` |
| "All portfolio limits remain compliant, London is Xpp from its limit" | ✅ `compute_concentration_tests` → utilisation/headroom |
| "Joint-borrower cases increased by X, youngest-borrower age moved X→Y" | ⚠️ levels available both sides; **no movement measure or materiality gate** — Phase 1 work |
| "Based on this user's prior MI activity…" | ❌ **Phase 3** |
| "No other material developments identified" | ✅ — and correctly qualified when a check did not run (once Phase 0 lands) |

### Phase 2 — Autonomous investigation

| | |
|---|---|
| **Modules touched** | `trakt_tools/registry.py`, `trakt_tools/handlers/__init__.py`, `readiness_agent/session.py` (unchanged if the resource model extends cleanly), `trakt_notifications/trigger.py` (invoke the controller) |
| **New modules** | `trakt_tools/handlers/pipeline.py` (H.3); `portfolio_review/controller.py` + `portfolio_review/objective.py` (H.2) |
| **Tests** | A scripted-model harness following `tests/test_agent_governed_execution.py`; an assertion that no number in a submitted review is absent from the tool transcript; `repeated_calls` bounded; a shadow-mode comparison against the Phase 1 deterministic brief |
| **Acceptance** | On a real month the controller reaches the Phase 1 findings **plus** at least one drill-down Phase 1 does not produce, with a complete evidence transcript, inside the step ceiling and the cost budget. Every number traces to a tool call. |
| **Dependencies** | Phase 1 (it is the shadow baseline) |
| **Complexity** | **Moderate.** Reuses the loop; the work is tools and the objective. |

### Phase 3 — User preference weighting

| | |
|---|---|
| **Modules touched** | `query_telemetry.py` (add `microsoft_tenant_id`, `route_family` — §E.3); `contract.py`/`outbox.py`/`delivery.py`/`store.py` (H.6); `recipients.py` (role tag) |
| **New modules** | `mi_agent_api/preference_profile.py` (H.4); the two-tier selector (H.5) |
| **Tests** | The filter-bubble test is the acceptance test: **a user whose entire history is borrower-age questions still receives a London concentration breach as the lead item.** Plus: a user with no history receives the mandatory tier and a governed default ordering; a profile built entirely from `metric_defaulted` measures is empty. |
| **Acceptance** | Mandatory findings are provably independent of the profile. Two users with different histories and the same portfolio receive the same mandatory items in the same order, differing only in the residual budget. |
| **Dependencies** | Phase 2, **and** enough accumulated telemetry. Do not start before there is a corpus. |
| **Complexity** | **Moderate**, dominated by H.6 rather than by the profile itself. |

### Phase 4 — Securitisation-readiness signals

| | |
|---|---|
| **Modules touched** | `sources.py` (resolve readiness signals), `risk_review.py` (a readiness finding category) |
| **New modules** | A thin signal extractor over `regulatory_readiness` / `evaluate_rule_packs` / `data_completeness` / `list_validation_exceptions` — **deltas only**, monthly cadence |
| **Tests** | Authority-labelling: a Trakt screening FLAG must never render with breach language (`recommendation.py:43` `FORBIDDEN_PHRASES` already maintains a forbidden-wording list) |
| **Acceptance** | The monthly review carries readiness *signals* with their authority intact. It never carries a full Annex 2 audit. |
| **Dependencies** | Phase 1 |
| **Complexity** | **Small.** All four sources exist as governed tools. |

**On ordering:** the repository evidence supports the brief's proposed sequence
with one change — **Phase 0 is inserted ahead of everything**, and **Phase 4 is
independent of Phases 2–3** and can be pulled forward if warehouse-funder or
rating-agency scrutiny arrives before the telemetry corpus does.

---

## Final question

> *If we wanted to ship the first production version of the Autonomous Portfolio
> Review Agent without compromising the governed MI architecture, what is the
> smallest set of changes required, and what percentage of the required
> capability already exists today?*

### Smallest set of changes

**Six changes. Four are small. None requires a new analytics engine, a new metric
ontology, a new LLM stack, A2A, or any change to the pipeline.**

| # | Change | Where | Size |
|---|---|---|---|
| 1 | Resolve concentration on the funded path (or record it unavailable), so a monthly card can never assert an unearned clear state | `trakt_notifications/sources.py:104` | **Very small** |
| 2 | Add `("products", "product_type")` to the pipeline movement attribution dimensions | `mi_agent_api/movement_detail.py:78` | **One line** |
| 3 | Add a funded generator set + a `funded:` threshold section, so the monthly review has the materiality layer the weekly one has | new `insight_generators_funded.py`; `insight_engine.py:230`; `config/mi/insights.yaml` | **Moderate** |
| 4 | Register 5 pipeline tools wrapping existing functions, so the agent surface can see the weekly book | new `trakt_tools/handlers/pipeline.py` | **Small–moderate** |
| 5 | A Portfolio Review controller = the `readiness_agent` loop with a period objective and a `submit_review` schema | new `portfolio_review/` | **Moderate** |
| 6 | Turn the flags on: `TRAKT_MI_WEEKLY_BRIEF`, `teams_notifications.enabled`, `TRAKT_TEAMS_BOT_ENABLED`, and authorise recipients via the existing operator CLI | config + `python -m trakt_notifications.cli authorise` | **Configuration** |

That is a **generic, governed, autonomously-investigated weekly and monthly
briefing, delivered proactively to authorised Teams users, with a full evidence
trail** — the first production version. Personalisation (H.4, H.5, H.6) is
deliberately excluded from it, because the telemetry corpus does not yet exist
and because per-recipient rendering is the riskiest change in the programme.
Ship the agent first; personalise it once there is something to personalise from.

### Percentage of required capability that already exists

**≈ 75%.**

Derived in §C: 12 of 20 required capabilities are fully present, 6 are partial,
2 are absent. Weighted at half-credit for partial, that is 15/20.

Decomposed by kind of work:

| | Existing | Gap |
|---|---|---|
| Data pipeline, snapshots, comparability | ~95% | Nothing material |
| Governed analytics surface | ~90% | Product-on-pipeline attribution; borrower-structure movement |
| Materiality engine | ~50% | Weekly complete; funded absent |
| Autonomous investigation | ~70% | Loop, session, tools and refusal semantics exist; pipeline tools and the objective do not |
| Teams proactive delivery | ~95% | Switched off; one correctness defect |
| Telemetry schema | ~95% | Two fields |
| Telemetry **history** | ~0% | Time, not code |
| Personalisation | ~10% | Routing exists; content and profile do not |
| Evidence / provenance | ~90% | Needs one review-level object composing what already exists |

**The honest summary:** Trakt is not being asked to build an autonomous portfolio
review agent from nothing. It is being asked to point an agent loop it already
has at a governed analytical surface it already has, gate the funded side with
the materiality discipline it already applies to the pipeline side, and switch on
a Teams delivery pathway it already built and tested. The genuinely new work is
one controller, one generator set, five tool wrappers — and, later and separately,
a preference layer that cannot be built until people have used the system.

---

*Prepared as an evidence-first architecture review. Every material conclusion
above cites `path:line`. Where the repository contradicted the architecture
assumed in the brief — proactive Teams delivery already existing, the autonomous
agent loop already existing, A2A not being on the production path, and the
monthly funded clear-state defect — the evidence is reported rather than the
assumption preserved.*
