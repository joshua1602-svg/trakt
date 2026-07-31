# OCC Agent — owning the onboarding operating process

The OCC Agent coordinates the complete onboarding operating process, from a
human's first instruction through to starting the existing Onboarding Agent on
the client's first real delivery.

It is not a second onboarding capability and not a second pipeline. Every part
of the work already exists somewhere in Trakt:

* **Client Onboarding** (`operations_control/onboarding/`) owns the case, the
  field catalogue, validation, inference, information requests, approval and —
  in `activate()` — the only place active configuration is created;
* **the delivery pipeline** — the Onboarding Agent, the Orchestration Agent, the
  Assembler Agent and the governed gates — owns ingestion;
* **the OCC's engine** owns the governed input pack (`create_batch`,
  `upload_batch_files`, `start_batch`).

What did not exist is the *process* joining them: someone has to decide what to
ask a client, ask it, take the answers in whatever form they arrive, record them
with their provenance, resolve what is missing, put a complete package in front
of a human, and — only after that human says so — activate and start ingestion.
That is what this package is.

## The nine stages

| # | Stage | Where it happens |
|---|---|---|
| 1 | A human initiates the onboarding, in their own words | `service.create_case` → `interpretation` → `extraction` → `planning` |
| 2 | Generate the complete structured onboarding pack | `pack.build` — a projection of the governed catalogue |
| 3 | Draft the outbound client communication | `communication` — `DRAFTED → HUMAN_REVIEW_REQUIRED → APPROVED_TO_SEND → SENT` |
| 4 | Consume client responses in any form | conversation (`extraction`), structured answers (`save_step`), uploaded files (`artefacts`), operator input |
| 5 | Persist responses into the governed configuration, with provenance | `OnboardingService.save_step` + `OnboardingCase.provenance_class` |
| 6 | Resolve outstanding questions, with full disclosure | `planning.ApplicationPlan.disclosure()` |
| 7 | Submit for human review with a complete package | `review.build` |
| 8 | Trigger the existing Onboarding Agent after approval | `adapters.LiveExecutionAdapter` |
| 9 | — | — |

Stages 1–7 are the same whether the case is being rehearsed or performed. Only
stage 8 differs, and that difference is a single injected adapter.

## Two lifecycles, and one waypoint

A case is two records, never merged: an `OnboardingCase` (Client Onboarding's
own, with its own reference, status and event history) and a `SyntheticRun`
beside it holding what Client Onboarding has no concept of.

`states.py` covers the **execution** lifecycle only:

```
AWAITING_ONBOARDING
  → PACK_DRAFTED → PACK_REVIEW_REQUIRED → PACK_APPROVED_TO_SEND → PACK_SENT
  → READY_TO_RUN → SYNTHETIC_ONBOARDING_RUNNING
      ↘ EXCEPTIONS_REQUIRE_INPUT ↗
  → SYNTHETIC_ONBOARDING_PASSED → ORCHESTRATION_PLAN_GENERATED
  → EXECUTION_APPROVAL_REQUIRED → READY_FOR_EXECUTION        ← a WAYPOINT
  → READY_FOR_REVIEW → APPROVED_FOR_ACTIVATION
  → ACTIVATION_CONFIRMATION_REQUIRED → ACTIVATING → INGESTION_STARTED
```

plus `ACTIVATION_FAILED`, `BLOCKED` and `CANCELLED`. The terminal states are
`INGESTION_STARTED` and `CANCELLED`.

`READY_FOR_EXECUTION` used to be the end. It is not the end of the operating
process — it is the point at which a rehearsal has passed and a human can be
asked to approve the real thing.

Onboarding actions are not gated by this table at all: Client Onboarding's own
transition table decides those, and a test asserts no onboarding status appears
here.

## Rehearsal and performance

Two adapters, one workflow:

* `SyntheticExecutionAdapter` — the default. Runs the real Onboarding,
  Orchestration and Assembler agents over files in an isolated sandbox, derives
  the live locations without writing to them, and **refuses activation**.
* `LiveExecutionAdapter` — sequences the platform's own governed path:

  ```
  OnboardingService.activate()   writes the configuration, registers the sources
  engine.create_batch()          opens a governed input pack
  engine.upload_batch_files()    places files where the intake expects them
  engine.start_batch()           starts the existing Onboarding Agent
  ```

  Every one of those already exists and is already governed. This adapter
  sequences them; it reimplements none of them, and a test asserts it makes no
  storage, HTTP or subprocess call of its own.

Live execution is **off by default** (`OCC_AGENT_LIVE_ENABLED`), and the flag
fails closed on anything it does not recognise.

## The one activation gate

Every path to production goes through `adapters.assert_may_activate`, which
checks all of these together and reports **every** unmet reason, not the first:

1. the live feature flag is enabled;
2. the run's mode is explicitly `live`;
3. the onboarding case is approved;
4. a human has explicitly approved the configuration;
5. every deterministic readiness criterion passed;
6. the tenant, client and portfolio are all identified;
7. the configuration validates;
8. the required artefacts are all present;
9. the approval is in the audit trail;
10. activation has not already happened;
11. a separate, explicit confirmation has been given.

Approval of the configuration is **not** the trigger — points 4 and 11 are
distinct states (`APPROVED_FOR_ACTIVATION` and
`ACTIVATION_CONFIRMATION_REQUIRED`), and a bare "yes" can never satisfy the
second. Immediately before production the operator is shown a confirmation
naming the client, the portfolio, the files, the target live locations and every
action that will occur.

A test asserts `assert_may_activate` is called from exactly one place in the
package.

## Only what the client can answer

Catalogue coverage is not the same as a good client experience. Projecting every
collected field produced 58 questions for a straightforward equity-release
onboarding, 20 of which the client could not answer at all.

`classification.py` puts every field in exactly one of five categories, derived
from the catalogue's own `source` axis:

| # | Category | What happens |
|---|---|---|
| 1 | Already known | pre-populated; offered for confirmation when material |
| 2 | **Only the client can answer** | **the pack** |
| 3 | Trakt works it out | automatic, with provenance |
| 4 | Learned from the first delivery | never in the initial pack |
| 5 | Internal operator decision | OCC workflow only |

For scenario A that is **22 client questions instead of 58**, of which 10 are
required. Everything not asked is *reported* with a reason, so "why is that not
being asked?" always has an answer.

`client_form.py` is how those questions are put and how the answers come back:
grouped into client-facing steps, conditional on product, asset class, portfolio
structure and delivery method, progressive (a follow-up appears when its trigger
is answered), and multi-portfolio without repeating client-level questions.

Every answer is keyed by an authoritative catalogue key — `contacts.email` or
`portfolios[0].portfolio_type` — checked against the form the client was served,
and written through `OnboardingService.save_step` **exactly as submitted**. A
test asserts `client_form.py` cannot even import an interpreter.

**There is no secure external client portal today.** Nothing serves a page to a
client or accepts a submission from outside the operator's network. What exists
is the domain and API contract one needs; the OCC's own surface renders it in
the meantime. Building the portal is named under production enablement below.

## Field mappings are not collected — deliberately

The platform learns field mappings from the first representative delivery: the
existing capability proposes them, an operator reviews and approves them during
that first ingestion, and they are then promoted and fingerprinted through the
governed path. The catalogue records that decision in its own `not_collected`
list (`file_role_schemas`, `expected_schema_fingerprint`).

Pushing mappings into onboarding would contradict a governed decision, so the
pack does not ask for them and says so (`pack.MAPPING_STATEMENT`), and the
review package tells the approver the same thing in as many words
(`review.MAPPING_NOTE`).

What the pack *does* collect is what Trakt cannot work out on its own — what the
numbers mean. The catalogue's `data_definitions` section asks for the source
file, its description, proprietary field definitions, units and currency, the
balance definition, date conventions, whether a measure is point-in-time or
cumulative, and known data-quality limitations.

## User access is a requirement, not a grant

`config/onboarding/field_catalogue.yaml` has an `access` section — user name,
email, role, scope, OCC access, dashboard access, report recipient — and every
field writes to the existing governed `onboarding_record` artefact. No new
user-management or identity framework was built.

Trakt reads its operators from environment configuration, so the review package
emits **structured operator actions**, each marked `not_provisioned`, rather
than claiming anybody now has access. See `review.access_actions`.

## Nothing is applied that was not understood

`planning.py` computes an instruction's effect **without writing anything**. A
plan carries four populations and reports all four: what was understood, what is
proposed, what the agent needs to ask, and what it could not read at all. A plan
with anything in the last two is refused — `PartiallyUnderstood` — unless the
human explicitly confirms the disclosed remainder.

A repeatable item is never silently replaced. The rules are exhaustive and there
is no fallback:

| Incoming item | Outcome |
|---|---|
| identifier matches an existing item | update that item |
| identifier differs | **add a new item** |
| no identifier, nothing yet | add |
| no identifier, one existing | *propose* updating it — never assumed |
| no identifier, several | a question; nothing is applied |

## The conversation reads the catalogue

`extraction.py` derives its patterns from
`config/onboarding/field_catalogue.yaml` rather than from a hand-written list:
cues from each field's own label, key and implied acronym; value patterns from
its declared `validation` or `type`; options from its own option list; asset
vocabulary from the product profiles' own signal tokens. A field added to the
catalogue is asked for by the pack *and* understood in conversation, with no
change here — and a parametrised test asserts exactly that, field by field.

## Switching it on

Two flags for visibility, and a third for live execution. All fail closed.

```bash
# The tab and the routes
export OCC_AGENT_SYNTHETIC_ENABLED=true
VITE_OCC_AGENT_SYNTHETIC_ENABLED=true npm run build

# Live execution. NOT enabled anywhere in this repository.
export OCC_AGENT_LIVE_ENABLED=false
```

With `OCC_AGENT_SYNTHETIC_ENABLED` unset the OCC is byte-for-byte what it was:
no tab, no routes, nothing in this package imported.

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `OCC_AGENT_SYNTHETIC_ENABLED` | *(unset — off)* | Mounts the routes and shows the tab. |
| `OCC_AGENT_LIVE_ENABLED` | *(unset — off)* | Permits live activation. One of eleven preconditions; on its own it grants nothing. |
| `TRAKT_OCC_AGENT_SYNTHETIC_CONTAINER` | `operations-control-synthetic` | Where practice cases are stored. Refused if it names the live operations container. |
| `TRAKT_OCC_AGENT_SANDBOX_ROOT` | `.occ-agent-synthetic` | Filesystem sandbox for artefacts and working files. |
| `TRAKT_OCC_AGENT_RETENTION_DAYS` | `30` | How long a practice case is kept before `purge_expired` will remove it. |

No Azure credentials are needed for a rehearsal; a test asserts the whole
process runs with every credential variable absent.

## What is reused, and what is simulated

Reused for real, on the live code path:

* **Client Onboarding** — `start_new_client`, `save_step` (with its inference,
  derivation and validation), `client_checklist`, `create_request` /
  `record_response` / `review_response`, `readiness`, `preview`,
  `submit_for_approval`, `approve`, `withdraw` and — through the live adapter
  only — `activate`;
* **The field catalogue** — what the interpreter validates against, what the
  pack projects, and what the conversation derives its patterns from;
* **Onboarding Agent components** — `file_profiler.profile_file` and
  `gate_1_alignment.semantic_alignment.HeaderMapper`;
* **Canonical transformation** — `gate_2_transform.canonical_transform.apply_types`;
* **Validation and materiality** — `gate_3_validation.validate_business_rules.run_rules`
  and `aggregate_validation_results`, driven by `config/asset/issue_policy.yaml`;
* **Orchestration Agent** — `engine.orchestrator_agent.orchestrator.run_orchestration`;
* **Assembler Agent** — the inherited, unmodified `stamp_provenance` / `assemble` /
  `route_mi`;
* **Input requirements** — `config/system/workflow_input_requirements.yaml`;
* **Product profiles** — `config/asset/product_profiles.yaml` signal tokens;
* **Decision contract** — the existing `34_target_first_decisions.yaml` shape,
  read by the existing `operations_control.adapters.extract_mapping_decisions`;
* **The governed intake** — `engine.create_batch` / `upload_batch_files` /
  `start_batch`, through the live adapter.

Simulated in a rehearsal, and labelled as such on the run and in the package:

* **Regime projection** — the command is built with the real
  `engine.assembler_agent.build_regime_command` and recorded, never run;
* **Live handoff** — refused by the synthetic policy, and the refusal audited.

Every stage records which of the five outcomes it reached, so a simulated stage
can never read as a completed one.

## Sending

Delivery is behind `communication.CommunicationAdapter`. The only implementation
is `RecordOnlyAdapter`, which records exactly what was approved and to whom, and
says plainly that **nothing was sent** — its receipt carries `sent: False`, and
every surface reads that rather than assuming. An operator who approved a pack
and sees "recorded, not sent" knows to send it themselves.

## Operational behaviour

* **Durability** — artefacts and packages are written to the container and
  cached in the sandbox. `store.materialise()` rebuilds a case's working files,
  so the instance that runs a case need not be the one that received the upload.
* **Background execution** — `start_synthetic_onboarding` runs the pipeline pass
  on its own thread, the same pattern `Engine.start` already uses, so an API
  request never holds one open.
* **Retention** — `store.expired()` / `purge_expired()` / `purge()`. A purge
  reports what it could **not** remove rather than pretending it is gone; on a
  blob backend the container's own lifecycle rule is the mechanism.
* **Identifier reservations** — nothing is activated in a rehearsal, so Client
  Onboarding's own collision check cannot see one practice case from another.
  `store.reserve_identifier` closes that gap inside the practice container.
* **The production source registry is read, never written.** Writing it is
  `activate()`'s job. `store.assert_no_live_registry_write` names the guarantee
  so a test asserts it rather than a comment claiming it.

## The scenarios

`fixtures.py` holds five deterministic scenarios, each blocked (or not) by a
different control, and `scenarios.run_scenario` drives one through the SAME
service calls the UI makes.

| Scenario | Onboarding | Run ends at | What it demonstrates |
|---|---|---|---|
| A — Clean onboarding | approved | `ACTIVATION_CONFIRMATION_REQUIRED` | The whole process with nothing in the way |
| B — Ambiguous mapping | approved | `ACTIVATION_CONFIRMATION_REQUIRED` | A halt the engine cannot settle, and the rerun after a human decision |
| C — Missing mandatory artefact | approved | `BLOCKED` | The configured input requirements refusing an incomplete pack |
| D — Product information gap | in review | `PACK_SENT` | A product question the client left unanswered, refused by Client Onboarding's own validator |
| E — Material business-rule failure | approved | `BLOCKED` | A deterministic blocker natural language cannot bypass |

A rehearsal stopping at `ACTIVATION_CONFIRMATION_REQUIRED` is the correct end:
everything is approved, and the one act that would reach production has
deliberately not been performed.

## Tests

`tests/operations_control/occ_agent/`:

| File | Covers |
|---|---|
| `test_planning.py` | merge rules, disclosure, provenance |
| `test_client_experience.py` | the five categories; only category 2 reaches a client; structured persistence |
| `test_boundary.py` | the OCC Agent / Onboarding Agent line, enforced by AST |
| `test_conversation_coverage.py` | every collected catalogue field, answerable in words |
| `test_pack_and_review.py` | the pack is the catalogue; the workflow; the review package; access |
| `test_activation.py` | the flag, the gate, approval ≠ activation, the live contract (fakes only) |
| `test_operational.py` | durability, retention, reservations, isolation, background runs |
| `test_workflow.py` | the five scenarios end to end |
| `test_state_model.py` | the two lifecycles, and that there are exactly two |
| `test_synthetic_safety.py` | the boundary |
| `test_component_reuse.py` | the real components are the ones that run |
| `test_interpretation.py` | schema validation of anything an interpreter produces |
| `test_tenancy_and_api.py` | isolation and the routes |

Frontend: `frontend/operations-control-ui/src/screens/agent/AgentTab.test.tsx`.

## Remaining production-enablement steps

Nothing in this repository enables live execution. To do so deliberately:

1. set `OCC_AGENT_LIVE_ENABLED=true` in the target environment;
2. construct `OccAgentService` with the OCC engine, so `_default_adapter`
   returns a `LiveExecutionAdapter` (a live case must not use the practice
   container — point it at the real `OpsStore`);
3. set the run's `mode` to `live` for cases intended for production;
4. register a real `CommunicationAdapter` in `communication.default_adapter` if
   the pack is to be emailed rather than issued by hand;
5. review the tenancy model: a live case is a production record, and the
   practice container's retention and reservation behaviour does not apply;
6. run the live adapter's contract tests against a staging Azure environment
   before any production client.
