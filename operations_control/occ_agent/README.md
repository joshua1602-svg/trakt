# OCC Agent — the synthetic onboarding tab

A natural-language operating layer over the existing Operations Control Centre.
An operator describes a new client in their own words; the OCC Agent creates a
**synthetic** onboarding case and works it through the whole operating process —
onboarding pack, client response, configuration, mapping, validation,
orchestration plan — to `READY_FOR_EXECUTION`.

It is a pre-scale and exit-readiness capability. It is **not** required for
Client 1 delivery, it is off unless explicitly enabled, and it cannot touch live
storage, the live pipeline, production configuration, email or publication.

## Where it lives

It is part of the existing OCC, not a second application:

| Layer | Location |
|---|---|
| Tab | `frontend/operations-control-ui/` — one entry in `components/Shell.tsx`, two routes in `App.tsx`, screens under `src/screens/agent/` |
| API | `operations_control/occ_agent/api.py`, mounted into the existing FastAPI app by `operations_control/api/app.py::mount_occ_agent` |
| Service | `operations_control/occ_agent/service.py` — the bounded tool surface |
| Storage | container `operations-control-synthetic` + a filesystem sandbox; never the live `operations-control` container |

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
| `TRAKT_OCC_AGENT_SYNTHETIC_CONTAINER` | `operations-control-synthetic` | Where synthetic cases are stored. Refused if it names the live operations container. |
| `TRAKT_OCC_AGENT_SANDBOX_ROOT` | `.occ-agent-synthetic` | Filesystem sandbox for artefacts, run working files and readiness packages. |

No Azure credentials are needed; a test asserts the whole process runs with every
credential variable absent.

## What is reused, and what is simulated

Reused for real, on the live code path:

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
* **Configuration hierarchy** — `operations_control.configuration.resolver.EffectiveConfigResolver`
  and the Annex 2 preflight, against a synthetic store;
* **Input requirements** — `config/system/workflow_input_requirements.yaml`;
* **Decision contract** — halts are written in the existing
  `34_target_first_decisions.yaml` shape and read by the existing
  `operations_control.adapters.extract_mapping_decisions`.

Simulated, and labelled as such on the case and in the readiness package:

* **Regime projection** — the projector command is built with the real
  `engine.assembler_agent.build_regime_command` (so an invalid call still fails)
  and recorded, never run. Stage outcome: `execution_simulated`.
* **Live handoff** — writing to Blob, triggering the pipeline, sending the pack
  by email and promoting the configuration all exist as named seams that the
  synthetic policy refuses.

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
```

A prohibited call raises `SyntheticBoundaryError` (an `OpsError`, so the existing
API envelope carries it), fails closed, and writes a `blocked` audit event
against the case. Path safety lives here too: case identifiers and file names are
validated, and `sandbox_path` refuses traversal, absolute components and symlink
escapes.

## The scenarios

`fixtures.py` holds five deterministic scenarios, each blocked (or not) by a
different control:

| Scenario | Outcome | What it demonstrates |
|---|---|---|
| A — Clean onboarding | `READY_FOR_EXECUTION` | The whole process with nothing in the way |
| B — Ambiguous mapping | `READY_FOR_EXECUTION` | A halt the engine cannot settle, and the rerun after a human decision |
| C — Missing mandatory artefact | `BLOCKED` | The configured input requirements refusing an incomplete pack |
| D — Product configuration gap | `BLOCKED` | A product-specific requirement, loaded through the product framework |
| E — Material business-rule failure | `BLOCKED` | A deterministic blocker natural language cannot bypass |

Run one end to end through the same service calls the UI makes:

```python
from operations_control.occ_agent.scenarios import run_scenario
run = run_scenario(service, "scenario_a_clean", tenant="client_a", actor="Alice")
print(run.progression, run.case.state)
```

## Tests

`tests/operations_control/occ_agent/` — safety boundary, component reuse,
workflow and scenarios, the state model, interpretation and the pack, tenancy and
the API. Frontend: `frontend/operations-control-ui/src/screens/agent/AgentTab.test.tsx`.

## Deferred

Genuinely live functionality, deliberately not built: external email sending,
live Blob writes, live pipeline handoff, live publishing, and converting a
synthetic case into a production case. Each has a named seam that the policy
refuses today, so connecting one later is a policy and adapter change rather than
a redesign.
