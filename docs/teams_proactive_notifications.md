# Teams proactive portfolio notifications — v1

Delivery-only capability. Two messages per approved data update, pushed to a
personal Teams chat from the **existing** Trakt Teams application package.

---

## 1. Audit — what already exists

### 1.1 The Teams application

| Artefact | What it is |
| --- | --- |
| `deploy/copilot-agent/manifest.json` | Teams app manifest, schema **v1.19**, app id `89c9db43-…`, `copilotAgents.declarativeAgents` only |
| `deploy/copilot-agent/declarativeAgent.json` | Declarative agent v1.2 — instructions, conversation starters, one action reference |
| `deploy/copilot-agent/ai-plugin.json` | API plugin v2.2 — `askTraktMi`, `getArtifact`, `OAuthPluginVault` runtime |
| `deploy/copilot-agent/trakt-copilot-openapi.yaml` | OpenAPI for the two Copilot actions |
| `deploy/copilot-agent/package_agent.py` | Builds the sideloadable zip and generates placeholder icons |
| `mi_agent_api/copilot_auth.py` | Entra bearer validation for `/v1/copilot/*` (JWKS, issuer, audience, fail closed) |
| `mi_agent_api/auth.py` | Easy Auth / SWA principal guard for `/mi/*` |

**Platform conclusion.** Teams manifest v1.19 permits `bots` and `copilotAgents`
in the same manifest. A declarative agent and a bot are separate *capabilities*
of one app, not competing app types. No separate application is required, so no
stop condition is triggered. The bot is added to the same package, the same app
id, the same branding, and the existing declarative agent and its two Copilot
actions are untouched.

### 1.2 Insight Engine and shared MI services

| Capability | Module | Reused for |
| --- | --- | --- |
| Governed insight object | `mi_agent_api/insight_contract.py` | Deterministic ids, severity vocabulary, `notification_eligible`, no-loan-level rule |
| Weekly Portfolio Brief | `mi_agent_api/insight_engine.py` | The whole insight set for one week, with omissions |
| Insight generators | `mi_agent_api/insight_generators.py` | Pipeline movement, completions, ticket, LTV, mix, conversion, concentration, data quality |
| Materiality config | `mi_agent_api/insight_config.py` → `config/mi/insights.yaml` | Every threshold |
| Movement attribution | `mi_agent_api/movement_detail.py` | Broker / region contributors on pipeline movement |
| Funded movement | `mi_agent_api/movement_summary.py::period_movement` | Funded balance, loan count, WA LTV, cohort (direct/acquired), region contributions |
| Pipeline snapshot | `mi_agent_api/pipeline_contract.py::compute_pipeline_snapshot` | Case count, pipeline amount, weighted expected funded |
| Pipeline evolution | `mi_agent_api/evolution.py::pipeline_evolution` | Governed weekly series |
| Funnel evolution | `mi_agent_api/evolution.py::pipeline_funnel_evolution` | Completions flow + governed five-week averages + conversion |
| Concentration tests | `mi_agent_api/concentration_tests_api.py`, `mi_agent/concentration_tests/*` | Utilisation, expected forecast, all-pipeline stress, breach horizon |
| Emerging risks | `mi_agent/concentration_tests/forward.py::identify_emerging_risks` | The ranked, deterministic risk vocabulary |
| Pipeline drivers | `mi_agent/concentration_tests/forward.py::pipeline_drivers` | Reconciling per-case expected contributions |
| Portfolio context | `mi_agent_api/portfolio_context.py` | Scope resolution shared by React / Copilot / PPTX |
| Execution context | `trakt_core/context.py`, `trakt_core/tenancy.py` | Trusted tenant, never request-supplied |

`insight_contract.Insight.notification_eligible` was written for exactly this
phase and is finally consumed here.

### 1.3 Ingestion lifecycle

```
blob arrival (Event Grid)
  → apps/blob_trigger_app/occ_intake.handle_arrival
  → operations_control.engine.OpsEngine.create_batch / register_batch_file
  → assess_batch → start_batch (auto when ready)
  → _execute → run_orchestration (onboard → transform → validate → assemble)
  → _apply_run_state → RUN_AWAITING_PUBLICATION + _prepare_publication
  → OPERATOR APPROVAL
  → OpsEngine.approve_publication  ◄── the only path to processed-v2/.../latest
  → RUN_PUBLISHED
```

`approve_publication` is the single governed point at which an update is
*approved*, artefacts are promoted to `latest`, and governed MI outputs are safe
to consume. It carries `run.client_id`, `run.portfolio_id`,
`run.reporting_period`, `run.orchestrator_run_id`, `run.delivery["dataset"]`
(`pipeline` | `funded`) and the publication document (version, previous
publication id, source artefacts). Every field the notification contract needs
is already there. No new lifecycle event is invented.

---

## 2. Shared-service extensions — exactly two, both additive

The Teams layer performs **no** calculation. Two gaps were found in the governed
layer and are closed once, in the shared services, where React, PPTX and Copilot
can also use them.

1. **Total-pipeline five-week average.** `pipeline_funnel_evolution` publishes
   `fiveWeekAvgFlow*` / `fiveWeekAvgStock*` per funnel *stage*; nothing publishes
   a five-week average for the *total* pipeline. `pipeline_evolution` gains an
   additive `fiveWeekAverage` block computed by the same `_trailing_avg` helper,
   over the same governed weekly series, with the same 5-week window. No
   existing field changes.

2. **Dimension contributors for concentration drivers.** `pipeline_drivers`
   returns per-case rows carrying `caseId` — loan-level, and therefore
   unusable in a notification. A new function `driver_contributors` aggregates
   the *already computed* governed expected contributions by broker, by the
   test dimension, and by their intersection. It adds no economics: it groups
   numbers the evaluator already produced and reconciles to
   `expectedNumeratorMovement`. `pipeline_drivers` itself is not modified.

Both are exposed through `mi_agent_api/concentration_tests_api.py` /
`mi_agent_api/evolution.py` so every channel shares them.

---

## 3. Target architecture

```
approve_publication (operations_control/engine.py)
      │  best-effort, after publish, never blocks the operator
      ▼
trakt_notifications.trigger.on_publication_approved
      ├─ eligibility  — enabled? dataset known? run published? not superseded?
      ├─ sources      — governed MI resolution (no recalculation)
      ├─ generate     — deterministic two-message batch (immutable)
      ├─ recipients   — authorised, tenant-scoped destinations
      └─ outbox       — one durable record per (batch, message, recipient)
                              │
                              ▼
              trakt_notifications.delivery worker
                     (timer / CLI, independent of MI)
                              │
                              ▼
              Bot Framework REST → Teams personal chat
                              │
                              ▼
                    deep link → React MI view
```

---

## 4. File impact

### New — `trakt_notifications/` (the whole delivery capability)

| File | Responsibility |
| --- | --- |
| `contract.py` | Versioned envelope, two message types, deterministic ids, severity vocabulary |
| `config.py` | `teams_notifications` settings (delivery only, never thresholds) |
| `layout.py` | Blob URIs under `trakt-state/notifications/` |
| `sources.py` | Governed MI resolution — adapter, no calculation |
| `portfolio_update.py` | Message 1 |
| `risk_review.py` | Message 2 |
| `recommendation.py` | Controlled action-language levels 0/1/2 |
| `deep_links.py` | React deep-link construction |
| `recipients.py` | Durable recipient registry + authorisation |
| `outbox.py` | Durable outbox, states, claim, idempotency, supersede |
| `cards.py` | Two Adaptive Card templates |
| `teams_client.py` | Bot Framework REST proactive send |
| `delivery.py` | Delivery worker |
| `telemetry.py` | Structured, redacted operational events |
| `trigger.py` | Approval-hook entry point |
| `cli.py` | Operator diagnostics + worker entry point |

### Changed — narrow, additive

| File | Change |
| --- | --- |
| `mi_agent_api/evolution.py` | `pipeline_evolution` gains `fiveWeekAverage` (additive key) |
| `mi_agent/concentration_tests/forward.py` | New `driver_contributors` function appended |
| `mi_agent_api/concentration_tests_api.py` | `compute_pipeline_drivers` gains additive `contributors` |
| `mi_agent_api/app.py` | Mounts the Teams bot router (feature-flagged) |
| `operations_control/engine.py` | One guarded call at the end of `approve_publication` |
| `deploy/copilot-agent/manifest.json` | Adds `bots` + `webApplicationInfo`; declarative agent unchanged |
| `deploy/copilot-agent/package_agent.py` | Validates the bot block when packaging |
| `config/mi/teams_notifications.yaml` | New delivery configuration |

### Not changed

Canonical pipeline, MI methodology, forecasting, concentration methodology,
React analytics, PPTX analytics, Copilot reasoning.

---

## 5. Azure resources required

| Resource | Purpose | Status |
| --- | --- | --- |
| Azure Bot (Bot Service) registration | Bot app id + messaging endpoint for Teams | **New** |
| Entra app registration (multi-tenant) for the bot | Bot credentials, external-tenant consent | **New** |
| Key Vault / App Service settings entry | `TRAKT_TEAMS_BOT_APP_ID` / `…_APP_PASSWORD` | Existing mechanism |
| Blob container `trakt-state` | Recipients, batches, outbox | **Existing** |
| App Service `trakt-mi-api` | Hosts `/v1/teams/bot/messages` | **Existing** |
| Function App timer trigger | Delivery worker | **Existing app**, new function |

No Redis. No new datastore technology. No event bus.

---

## 6. Security posture

* Tenant comes from the trusted `ExecutionContext` / the governed run, never a caller.
* Recipient must be authorised for the portfolio context before an outbox row is written.
* Microsoft tenant id on the inbound bot activity must match the recipient record; a mismatch is a permanent failure, never a re-route.
* No loan-level data or PII in any card (asserted by test).
* Content is immutable after generation; a correction creates a new, labelled batch.
* Missing bot credentials or recipient mapping fails closed.
* Run ids and blob URIs are never rendered to a user.

---

## 7. Deliberate v1 limitations

Personal scope only; named pilot recipients only; no channels, group chats,
email, self-service subscriptions, interactive actions, or Graph-based mass
installation. Card editing on correction is not implemented — a clearly labelled
correction message is sent instead.
