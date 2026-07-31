# OCC Agent — practising an onboarding, end to end

A natural-language operating layer that joins two things the Operations Control
Centre already has, and which have never met:

* **Client Onboarding** (`operations_control/onboarding/`) — the governed
  standing-configuration capability, with its own case model, field catalogue,
  validation, information requests, approval and activation;
* **the delivery pipeline** — the Onboarding Agent, the Orchestration Agent, the
  Assembler Agent and the governed gates.

Client Onboarding stops at activation and never runs a pipeline. The pipeline
starts from a client that already exists. Between them sits the question an
operator actually has — *if we onboard this client, will their data go through?*

The OCC Agent answers it **without creating anything**. An operator describes a
new client in their own words; a real onboarding case is opened, answered,
asked-about, approved — and then a practice execution runs the real agents and
the real controls over practice files to `READY_FOR_EXECUTION`.

It is a pre-scale and exit-readiness capability. It is **not** required for
Client 1 delivery, it is off unless explicitly enabled, and it cannot activate a
configuration, touch live storage, the live pipeline, production configuration,
email or publication.

## The one line it never crosses

`OnboardingService.activate()` is, by its own docstring, *"the only place active
configuration is created, and only from an approved case"*. The OCC Agent names
it as a capability (`activate_configuration`), refuses it, and asserts it: a
readiness criterion — `no_configuration_written` — fails if a case were ever
activated, and a test drives all five scenarios with `activate` monkeypatched to
raise.

So a practice case ends **approved but never activated**. `preview()` shows
exactly what activation *would* create; nothing writes it.

## Where it lives

It is part of the existing OCC, not a second application:

| Layer | Location |
|---|---|
| Tab | `frontend/operations-control-ui/` — one entry in `components/Shell.tsx`, two routes in `App.tsx`, screens under `src/screens/agent/` |
| API | `operations_control/occ_agent/api.py`, mounted into the existing FastAPI app by `operations_control/api/app.py::mount_occ_agent` |
| Service | `operations_control/occ_agent/service.py` — the bounded tool surface |
| Onboarding | `operations_control.onboarding.OnboardingService`, constructed with an `OpsStore` pinned to the synthetic container |
| Storage | container `operations-control-synthetic` + a filesystem sandbox; never the live `operations-control` container |

## Two records, two lifecycles

A practice case is **not** a third case model. It is:

* an `OnboardingCase` — Client Onboarding's own, with its own reference
  (`ONB-2026-0001`), its own status (`draft → … → approved`), its own transition
  table and its own event history;
* a `SyntheticRun` beside it, keyed by that reference, holding only what Client
  Onboarding has no concept of: which pipeline stages ran and how honestly, the
  mapping report, the open decisions, the artefacts and their intended live
  locations, the orchestration and assembler plans, and the readiness verdict.

`states.py` covers the **execution** lifecycle only —
`AWAITING_ONBOARDING → READY_TO_RUN → SYNTHETIC_ONBOARDING_RUNNING →
(EXCEPTIONS_REQUIRE_INPUT) → SYNTHETIC_ONBOARDING_PASSED →
ORCHESTRATION_PLAN_GENERATED → EXECUTION_APPROVAL_REQUIRED →
READY_FOR_EXECUTION`, plus `BLOCKED` and `CANCELLED`. Onboarding actions are not
gated by it at all: Client Onboarding's own table decides those, and a test
asserts no onboarding status appears as an execution state.

## Switching it on

Two flags, same name, read independently — the UI hides the tab, the API does
not mount the routes. Both fail closed.

```bash
# Backend (the API process)
export OCC_AGENT_SYNTHETIC_ENABLED=true

# Frontend (Vite build time)
VITE_OCC_AGENT_SYNTHETIC_ENABLED=true npm run build
```

With the flag unset the OCC is byte-for-byte what it was: no tab, no routes, and
nothing in this package is imported.

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `OCC_AGENT_SYNTHETIC_ENABLED` | *(unset — off)* | Mounts the routes and shows the tab. Anything other than `1/true/yes/on/enabled` is off. |
| `TRAKT_OCC_AGENT_SYNTHETIC_CONTAINER` | `operations-control-synthetic` | Where practice cases — onboarding cases included — are stored. Refused if it names the live operations container. |
| `TRAKT_OCC_AGENT_SANDBOX_ROOT` | `.occ-agent-synthetic` | Filesystem sandbox for artefacts, run working files and readiness packages. |

No Azure credentials are needed; a test asserts the whole process runs with every
credential variable absent.

## What is reused, and what is simulated

Reused for real, on the live code path:

* **Client Onboarding** — `start_new_client`, `save_step` (with its inference,
  derivation and validation), `client_checklist`, `create_request` /
  `record_response` / `review_response`, `readiness`, `preview`,
  `submit_for_approval`, `approve` and `withdraw`. Every answer reaches the case
  through *its* writer, so its event history is complete;
* **The field catalogue** — `config/onboarding/field_catalogue.yaml` is what the
  interpreter validates against, so a model behind the `Interpreter` seam cannot
  invent a field or reach a section the wizard does not have;
* **Onboarding Agent components** — `file_profiler.profile_file` (source
  inspection) and `gate_1_alignment.semantic_alignment.HeaderMapper` (the real
  tiered alias/fuzzy mapper, against the real field registry and alias
  directory);
* **Canonical transformation** — `gate_2_transform.canonical_transform.apply_types`;
* **Validation and materiality** — `gate_3_validation.validate_business_rules.run_rules`
  and `aggregate_validation_results` driven by `config/asset/issue_policy.yaml`;
* **Orchestration Agent** — `engine.orchestrator_agent.orchestrator.run_orchestration`,
  the real conductor, over an adapter that subclasses the real `AgentAdapters` seam;
* **Assembler Agent** — the inherited, unmodified `stamp_provenance` / `assemble` /
  `route_mi`, i.e. real provenance stamping and `engine.assembler_agent`;
* **Input requirements** — `config/system/workflow_input_requirements.yaml`;
* **Decision contract** — halts are written in the existing
  `34_target_first_decisions.yaml` shape and read by the existing
  `operations_control.adapters.extract_mapping_decisions`.

Simulated, and labelled as such on the run and in the readiness package:

* **Regime projection** — the projector command is built with the real
  `engine.assembler_agent.build_regime_command` (so an invalid call still fails)
  and recorded, never run. Stage outcome: `execution_simulated`.
* **Live handoff** — activating the configuration, writing to Blob, triggering
  the pipeline, sending an information request by email and promoting production
  configuration all exist as named seams that the synthetic policy refuses.

Every stage records which of the five outcomes it reached — deterministic
execution completed, contract validation completed, execution simulated, human
input required, hard blocked — so a simulated stage can never read as a
completed one.

## The synthetic boundary

`policy.py` is the floor. Its capability set is immutable and every permission is
denied by construction:

```
runtime_mode: synthetic
allow_external_email: false
allow_live_blob_write: false
allow_live_pipeline_trigger: false
allow_production_config_write: false
allow_publish: false
allow_live_case_access: false
allow_activate_configuration: false
```

A prohibited call raises `SyntheticBoundaryError` (an `OpsError`, so the existing
API envelope carries it), fails closed, and writes a `blocked` audit event
against the case. Path safety lives here too: case references and file names are
validated, and `sandbox_path` refuses traversal, absolute components and symlink
escapes.

## The scenarios

`fixtures.py` holds five deterministic scenarios, each blocked (or not) by a
different control. Each carries what the operator says at the start *and* what
the client eventually sends back, keyed by the catalogue's own `section.field`.

| Scenario | Onboarding | Run | What it demonstrates |
|---|---|---|---|
| A — Clean onboarding | approved | `READY_FOR_EXECUTION` | The whole process with nothing in the way |
| B — Ambiguous mapping | approved | `READY_FOR_EXECUTION` | A halt the engine cannot settle, and the rerun after a human decision |
| C — Missing mandatory artefact | approved | `BLOCKED` | The configured input requirements refusing an incomplete pack |
| D — Product information gap | in review | `AWAITING_ONBOARDING` | A product-specific question the client left unanswered, refused by Client Onboarding's own validator |
| E — Material business-rule failure | approved | `BLOCKED` | A deterministic blocker natural language cannot bypass |

Run one end to end through the same service calls the UI makes:

```python
from operations_control.occ_agent.scenarios import run_scenario
run = run_scenario(service, "scenario_a_clean", tenant="client_a", actor="Alice")
print(run.onboarding_progression, run.progression, run.stopped_because)
```

## Tests

`tests/operations_control/occ_agent/` — the safety boundary, component reuse,
the workflow and scenarios, the two state models, interpretation, tenancy and the
API. Frontend:
`frontend/operations-control-ui/src/screens/agent/AgentTab.test.tsx`, which
drives the mock through the real application root; the mock composes
`MockOnboarding` for the same reason the service composes `OnboardingService`.

## Deferred

Genuinely live functionality, deliberately not built: activating a practice
case's configuration, external email sending, live Blob writes, live pipeline
handoff, live publishing, and converting a practice case into a production one.
Each has a named seam that the policy refuses today, so connecting one later is a
policy and adapter change rather than a redesign.
