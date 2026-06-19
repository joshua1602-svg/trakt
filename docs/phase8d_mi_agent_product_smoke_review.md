# Phase 8D — MI Agent product smoke review and demo readiness

*Status: review / documentation / demo-readiness only. No runtime features, no
refactors, no new tests. This document describes the system as it exists after
Phase 8C.*

---

## 1. Executive summary

After Phase 8C, the MI Agent has a **proven end-to-end natural-language-to-result
path on synthetic/local data**:

```
natural-language question
  → interpreter (deterministic baseline OR Anthropic-style client)
  → MIQuerySpec v2
  → MIQuerySpec.normalized()
  → validate_query_spec()
  → run_mi_query()
  → deterministic MI result (rows / metrics / chart instruction)
```

The boundary of what is true today is deliberately narrow and must be stated
honestly:

* **The end-to-end NLQ-to-runtime path is proven on synthetic/local data only.**
  Snapshots are small, in-memory, deterministic frames in a local filesystem
  store — not real client data.
* **The LLM / interpreter proposes a spec only.** It emits MIQuerySpec-v2 JSON
  (or a clarification). It never computes numbers, aggregates, trends, or
  migrations.
* **Validation gates execution.** A spec is normalised and validated before it
  runs; ambiguous, malformed, or invalid interpretations are blocked and never
  reach the engine.
* **`run_mi_query` performs all deterministic analytics.** It is the single
  execution engine for state, temporal, and risk queries over the snapshot layer.
* **Production onboarding, Azure/cloud storage, UI/export, real-client data
  mapping, and the M&A route runtime remain deferred.**

In one line: *we can demonstrate that a plain-English MI question is safely
interpreted into a governed query and answered deterministically — on synthetic
data, locally, with the model strictly confined to interpretation.*

---

## 2. Current product capability map

Supported question families, the spec they map to, how they execute, and the
evidence. "Spec fields" lists the discriminating fields (route/client and
governance defaults omitted). Evidence references the test suites that exercise
each path.

| # | Example question | Generated MIQuerySpec fields | Runtime mode | Output type | Evidence | Caveat |
|---|---|---|---|---|---|---|
| 1 | "show total funded" | `state=total_funded, temporal_mode=latest` | state | table (rows) + total balance | `test_phase8c_nlq_to_runtime_smoke.py::test_show_total_funded`; `test_phase6b…::test_total_funded_latest` | Latest snapshot only |
| 2 | "show total pipeline" | `state=total_pipeline, temporal_mode=latest` | state | table + total balance | `…8c::test_show_total_pipeline`; `…6b::test_total_pipeline_latest` | Latest snapshot only |
| 3 | "show forecast funded" | `state=total_forecast_funded, temporal_mode=latest` | state | metric (`forecast_funded_total`) | `…8c::test_show_forecast_funded`; `…6b::test_total_forecast_funded_latest` | Probability source defaults apply |
| 4 | "trend funded balance over the last three months" | `state=total_funded, temporal_mode=trend, start_date, end_date, trend_grain=monthly` | temporal | series + `chart_instruction={"chart_type":"line"}` | `…8c::test_trend_funded`; `…6b::test_funded_trend` | Dates resolved from context anchors only |
| 5 | "trend pipeline over the last three months" | `state=total_pipeline, temporal_mode=trend, start_date, end_date` | temporal | series + line chart instruction | `…6b::test_pipeline_trend`; golden `…8a` | Dates from context anchors only |
| 6 | "compare funded balance to last month" / "what changed since last month" | `state=total_funded, temporal_mode=compare, baseline_date, current_date` | temporal | comparison row (baseline/current/change, new/retained counts) | `…8c::test_compare_funded`; `…6b::test_funded_compare_baseline_current` | Two anchored snapshots required |
| 7 | "show funded balance by portfolio" | `risk_monitor_mode=concentration, state=total_funded, dimension=portfolio_id` | risk (concentration) | grouped table (`balance_sum`, status) | `…8c::test_funded_by_portfolio`; `…6b::test_funded_by_portfolio` | "portfolio" needs a portfolio reference config/context, else clarify |
| 8 | "show funded balance by region" | `risk_monitor_mode=concentration, state=total_funded, dimension=geographic_region_obligor` | risk (concentration) | grouped table + status | `…8c::test_funded_by_region`; `…6b::test_funded_by_region` | Region = obligor region |
| 9 | "show pipeline by stage" | `risk_monitor_mode=concentration, state=total_pipeline, dimension=pipeline_stage` | risk (concentration) | grouped table | `…8c::test_pipeline_by_stage`; `…6b::test_pipeline_by_stage` | "stage" only valid in a pipeline context, else clarify |
| 10 | "show concentration by region" | `risk_monitor_mode=concentration, state=total_funded, dimension=geographic_region_obligor` | risk (concentration) | grouped table + RAG status | `…8c::test_concentration_by_region`; `…6b::test_concentration_warning` | Thresholds from risk-monitor config |
| 11 | "are we too concentrated by broker" | `risk_monitor_mode=concentration, state=total_funded, dimension=broker_channel` | risk (concentration) | grouped table + status | golden `…8a` ("…by broker"); `…6b::test_forecast_funded_by_broker` (broker grouping) | Broker = `broker_channel` |
| 12 | "show risk grade migration" | `risk_monitor_mode=migration, dimension=internal_risk_grade, baseline_date, current_date` | risk (migration) | migration matrix (`from_value`/`to_value`/`movement_type`) | `…8c::test_risk_grade_migration`; `…6b::test_risk_grade_migration` | Needs two anchored snapshots |
| 13 | "show IFRS stage migration" | `risk_monitor_mode=migration, dimension=ifrs9_stage, baseline_date, current_date` | risk (migration) | migration matrix | `…8c::test_ifrs_migration`; `…6b::test_ifrs9_migration` | Needs two anchored snapshots |
| 14 | "show PD bucket migration" | `risk_monitor_mode=migration, dimension=pd_bucket, baseline_date, current_date` | risk (migration) | migration matrix | `…8c::test_pd_bucket_migration`; `…6b::test_pd_bucket_migration` | PD already bucketed in source data |
| 15 | "show balance buckets" / "show interest rate buckets" / "show time on book buckets" | `risk_monitor_mode=concentration, dimension=balance_band\|interest_rate_bucket\|time_on_book_bucket, bucket_strategy=quantile` | risk (concentration) | grouped table (intended) | Interpreted + validated: golden `…8a`; `parse`/validate via `…8b`. Quantile engine: `quantile_buckets.py` + `…6::quantile` tests | **Expressible but not fully runtime-materialised** — see note below |

### Note on bucket dimensions (item 15)

These questions are **interpreted and validated correctly** into a governed
spec, and a standalone quantile-bucketing engine exists
(`mi_agent/quantile_buckets.py`, asset-agnostic quartiles). However,
`run_mi_query` does **not** currently call the bucket materialiser as part of its
risk/concentration dispatch — the bucket column (`balance_band`, etc.) is not
synthesised from its source field (`current_outstanding_balance`,
`current_interest_rate`, `months_on_book`) inside the runtime. So a "by balance
bucket" query will validate and route, but the grouped result is only meaningful
if the bucket column is already present in the snapshot frame. LTV / age buckets
rely on the existing configured-band engine and are likewise not wired through
the NLQ runtime path. Treat bucketed concentration as **demonstrable at the spec
level, not yet a runtime-materialised analytic.** (This matches the Phase 6E
audit finding that Step-0 resolvers/bucketers are not invoked by `run_mi_query`.)

---

## 3. Blocked / clarification cases

Questions that must **not** execute automatically. Each yields either a
clarification (interpreter asks back) or a structured validation block; in the
bridge these surface as `executed=False`, `runtime_result is None`.

| Question / input | Why unsafe or ambiguous | Expected issue / code | Expected behaviour | Evidence |
|---|---|---|---|---|
| "show risk" | Too vague — migration vs flags vs concentration | clarification (no code; `not_executed_clarification_required` at bridge) | Ask which risk view | golden `…8a` ("show risk"); `…8a::test_ambiguous_questions_clarify`; `…8c::test_ambiguous_question_via_deterministic_not_executed` |
| "show changes" | No period/date context | clarification | Ask over what period | golden `…8a` ("show changes") |
| "show stage" | Bare "stage" outside a pipeline context (pipeline vs IFRS vs risk stage) | clarification (`ambiguous_dimension` if forced into a spec) | Ask which stage | golden `…8a` ("show stage"); spec-graded `ambiguous_dimension` |
| "show portfolio" (no config/context) | "portfolio" needs a Trakt portfolio reference config | clarification | Ask for portfolio reference | golden `…8a` (`portfolio_config_available: false`) |
| "show rate" | Unclear: average rate vs rate buckets, and population | clarification | Ask metric vs buckets | golden `…8a` ("show rate") |
| Malformed LLM output | Not parseable as a spec | `llm_malformed_json` (or `llm_output_contains_code` / `llm_output_not_object`) → clarification | Block, ask to rephrase | `…8b::test_parse_*`; `…8c::test_malformed_output_not_executed` |
| Hallucinated field (semantics supplied) | Dimension names no known semantic field | `llm_hallucinated_field` → clarification | Block | `…8b::test_hallucinated_field_*`; `…8c::test_hallucinated_field_not_executed` |
| Invalid enum (`state=not_a_state`) | Value outside controlled vocabulary | `invalid_enum_value` → clarification | Block | `…8b` validation; `…8c::test_invalid_enum_not_executed` |
| Missing temporal dates (compare/trend) | Selector incomplete | `temporal_selector_incomplete` | Block | `…8c::test_missing_temporal_dates_not_executed`; `…8b::test_invalid_spec_compare_without_dates_clarifies` |
| Regulatory route + MI state | Regulatory routes may not run MI state/temporal/risk | `invalid_route_for_state` | Block | `…8c::test_regulatory_route_with_mi_state_not_executed`; golden `…8a` |
| M&A route + pipeline/forecast state | M&A route cannot run pipeline states | `invalid_route_for_state` | Block | `…8c::test_mna_route_with_pipeline_state_not_executed`; golden `…8a` |
| Unsupported chart type | Chart type outside the governed library | `llm_unsupported_chart_type` → clarification | Block | `…8b::test_unsupported_chart_type_forces_clarification` |

The governing principle: **`result.ok` is true only when the spec validated and
no adapter-level error occurred; otherwise the bridge clarifies and never
executes.**

---

## 4. Demo narrative (6–8 steps)

A short, honest walkthrough. Use the deterministic interpreter or fake-client
specs — no live LLM required.

1. **Set the frame (caveat first).** "This is a *synthetic, local* proof. The
   data is a small set of fabricated snapshots on this machine. There is no real
   client data, no cloud, and no UI yet. What we're proving is the *governed
   path*, not production scale."
2. **Explain the split.** "The language model only *interprets* the question into
   a structured query. It never calculates. All numbers come from our
   deterministic engine, `run_mi_query`."
3. **Simple state question.** Ask *"show total funded."* Show the generated spec
   (`state=total_funded, temporal_mode=latest`) and the result (3 funded loans,
   balance 620).
4. **Trend question.** Ask *"trend funded balance over the last three months."*
   Show the temporal spec, the series `[300, 620, 620]`, and that the engine
   returns a line-chart instruction.
5. **Concentration question.** Ask *"show funded balance by region."* Show the
   concentration spec and the grouped result (North 400, South 220) with a RAG
   status column from the risk-monitor config.
6. **Risk migration question.** Ask *"show risk grade migration."* Show the
   migration spec and the matrix highlighting a B→C *deteriorated* movement
   between two snapshots.
7. **Blocked / clarification example.** Ask *"show risk."* Show that the system
   *refuses to guess* and asks a clarifying question instead of executing —
   emphasise this is the safety gate, not a failure.
8. **Close on the boundary.** "Every answer went through normalise → validate →
   execute. Invalid or ambiguous questions stop before the engine. Production
   onboarding, cloud storage, a UI, and live model integration are the next
   decisions — not built yet."

---

## 5. Technical flow diagram

```
            ┌──────────────────────────────────────────────────────────────┐
            │                        User question                          │
            │                      (natural language)                       │
            └───────────────────────────────┬──────────────────────────────┘
                                             │
                                             ▼
                       ┌───────────────────────────────────────┐
                       │   Interpreter                          │
                       │   • deterministic baseline, OR         │
                       │   • Anthropic-style client (8B adapter)│
                       │   PROPOSES A SPEC ONLY — no analytics  │
                       └──────────────────┬────────────────────┘
                                          │
                  clarification / ────────┤  (ambiguous → STOP, ask user)
                  malformed / code        │
                                          ▼
                            ┌───────────────────────────┐
                            │       MIQuerySpec v2       │
                            └─────────────┬─────────────┘
                                          ▼
                            ┌───────────────────────────┐
                            │   MIQuerySpec.normalized() │  (convenience → canonical)
                            └─────────────┬─────────────┘
                                          ▼
                            ┌───────────────────────────┐
                            │    validate_query_spec()   │
                            └─────────────┬─────────────┘
                         INVALID ─────────┤  (bad enum / route / dates /
                                          │   hallucinated field → STOP)
                              VALID       │
                                          ▼
                            ┌───────────────────────────┐
                            │        run_mi_query()      │  ← the ONLY engine
                            │   flat | state | temporal  │
                            │            | risk          │
                            └─────────────┬─────────────┘
                                          ▼
                            ┌───────────────────────────┐
                            │        RuntimeResult       │
                            │  rows / metrics / issues / │
                            │      chart_instruction     │
                            └─────────────┬─────────────┘
                                          ▼
                            ┌───────────────────────────┐
                            │     chart / table output   │
                            └───────────────────────────┘
```

**Where invalid specs stop:** at the **interpreter gate** (a clarification or
unparseable/code output is never turned into a runnable spec) and at the
**validation gate** (`validate_query_spec` failures — bad enums, route/state
conflicts, incomplete temporal selectors, hallucinated fields — are blocked).
The bridge (`interpret_and_run_mi_query`) enforces both before calling the
engine, so `executed=False` whenever either gate trips.

---

## 6. What is real vs synthetic

| Category | Components |
|---|---|
| **Real / generic platform components** | Canonical field registry & aliases; MI semantic registry; route contracts; analytics library (states, temporal compare/trend, risk migration/concentration); MIQuerySpec v2 + `normalized()`; `validate_query_spec()`; `run_mi_query` dispatch; snapshot model & `SnapshotStore` interface; deterministic interpreter; Anthropic-first interpreter adapter (mockable boundary); NLQ→runtime bridge. These are asset-agnostic and not tied to any one client. |
| **Local-only components** | `LocalFsSnapshotStore` (local filesystem persistence); snapshots registered on a temp/local path; risk-monitor & portfolio-reference example configs. |
| **Synthetic-only proofs** | All Phase 6B/6C/8C snapshot frames (fabricated loans across three reporting dates); the golden question dataset; fake Anthropic client responses; expected numeric results in tests. |
| **Docs / demo-only artefacts** | This document and the prior phase docs; the dev-only smoke script `scripts/mi_nlq_dev_smoke.py` (manual, not in CI). |
| **Not yet built** | Production onboarding/consolidation mapping; client portfolio reference configs at scale; lineage/data-quality workflow; Azure/cloud storage & ingestion; UI/export layer; auth/permissions; production Anthropic integration controls; monitoring/audit logs; M&A route runtime; runtime materialisation of quantile/configured buckets through the NLQ path. |

---

## 7. What must be true before a client demo (internal/synthetic)

Minimum readiness for an honest, synthetic demo:

- [ ] **PRs merged** — Phases 0–8C merged to `main` (8D docs optional for the demo itself).
- [ ] **Tests green** — run the MI suites and confirm pass:
  ```bash
  python -m pytest tests/test_phase6_mi_runtime.py \
    tests/test_phase6b_mi_runtime_smoke_pack.py \
    tests/test_phase6c_multi_artifact_consolidation.py \
    tests/test_phase7_mi_query_spec_v2.py \
    tests/test_phase8a_mi_interpreter_harness.py \
    tests/test_phase8b_anthropic_interpreter_adapter.py \
    tests/test_phase8c_nlq_to_runtime_smoke.py -q
  ```
- [ ] **Dev smoke command rehearsed** (optional, only if showing a real model):
  ```bash
  ANTHROPIC_API_KEY=sk-... python scripts/mi_nlq_dev_smoke.py "show total funded"
  ```
  Otherwise demo with the deterministic interpreter / fake-client specs.
- [ ] **Sample synthetic dataset loaded** — the Phase 6B three-snapshot set (or the dev-smoke builder) registered in a local store.
- [ ] **Known caveats prepared** — synthetic/local only; buckets are spec-level only; latest-snapshot semantics; dates come from context anchors.
- [ ] **No live-client-data claim** — explicitly state the data is fabricated.
- [ ] **No production-mapper claim** — no onboarding/consolidation pipeline is being shown end-to-end from raw client files.

---

## 8. What must be true before real client use (production gaps)

- **Config-driven onboarding / consolidation** — map real, messy client files to
  canonical fields and consolidate multi-artefact deliveries reliably.
- **Client portfolio reference config** — real per-client portfolio/SPV/acquired
  -portfolio reference data so "by portfolio" resolves safely.
- **Production lineage / data-quality workflow** — provenance, validation, and
  exception handling on ingested data before it reaches the snapshot store.
- **Azure / cloud storage and ingestion** — a production `SnapshotStore`
  implementation (blob-backed) and ingestion pipeline replacing the local FS.
- **UI / export layer** — a user surface to ask questions and render/export
  tables and charts.
- **Authentication / permissions** — per-client isolation, access control, and
  row/route-level authorisation.
- **Real Anthropic integration controls** — key management, rate limits,
  timeouts/retries, cost controls, prompt/version pinning, and logging of model
  inputs/outputs.
- **Monitoring / audit logs** — execution audit trail, issue/clarification
  tracking, and operational monitoring.
- **Runtime bucket materialisation** — wire the quantile/configured bucket
  engines into the NLQ runtime path so bucketed concentration is a real analytic.

---

## 9. Recommended next decisions

*No further build phase is started automatically.* These are options; pick based
on the immediate goal.

| Option | Choose this when… |
|---|---|
| **Pause and demo internally** | The goal is to validate direction with stakeholders before more investment. The synthetic path is demo-ready today; this is the lowest-risk next step. |
| **Harden interpreter coverage** | Demos surface many phrasings the deterministic baseline doesn't map yet, or you want higher confidence before any live model. Expand golden questions + adapter parsing without touching the runtime. |
| **Connect a real Anthropic dev smoke** | You want to see genuine Claude interpretation (not fake clients) on synthetic data, in a controlled dev setting. Use the existing `scripts/mi_nlq_dev_smoke.py` with a key; cheap and reversible. |
| **Build production onboarding / consolidation** | A real client dataset is available and the priority is getting *real data* into the canonical/snapshot layer. This is the biggest unlock toward real use. |
| **Build Azure snapshot adapter** | Onboarding is in hand and the blocker is durable, cloud-hosted, multi-tenant storage. Do this when local FS is the limiting factor. |
| **Build UI / export layer** | Stakeholders need a self-serve surface and the analytics/interpretation are trusted enough to expose. Do this once question coverage and data are credible. |
| **Build M&A route runtime** | M&A analytics are an explicit near-term commercial need. This is a distinct runtime workstream; sequence it only when MI is stable and demanded. |

**Recommended default:** *pause and demo internally* first — it converts the work
to date into feedback at minimal cost and informs which of the build options is
actually the right next investment.

---

## Appendix — exclusions honoured by this phase

Phase 8D is documentation only. No runtime features, refactors, Azure adapter,
onboarding orchestration, M&A runtime, Annex 2/XML/regulatory changes, new chart
types, production UI, or live LLM calls were introduced. No test changes were
required for documentation accuracy.
