# 10 — Effective configuration resolution and the narrowed Onboarding Agent

Implementation plan + current-state findings (investigated before coding).

## 1. Current-state findings

| Concern | Current state (evidence) |
|---|---|
| Merged-config utility | `config/system/config_resolver.py` — deterministic layer merge with per-key provenance, self-described "Pass 1 scaffold… not wired into runtime entrypoints yet". **Reused as the resolver's merge engine.** |
| Configuration layers | Directories are the taxonomy: `config/system` (registry, aliases, enums, modes), `config/regime` (Annex 2 universe/rules), `config/asset` (product profiles/defaults), `config/client` (client YAML), plus OCC rule store (client/portfolio/file scopes) and run decisions. No versioning, no activation flow, live files read directly at run time. |
| Onboarding agent config intake | `run_operator_workflow(..., registry=, aliases_dir=, regime_config=, asset_config=, target_first_decisions=)` — paths to LIVE repo YAML, chosen by the caller (`RealAgentAdapters`). Some internal reads (onboarding modes, run-context policy, product profiles) use fixed repo paths. |
| Agent decision persistence | The managed run path **does not write** shared config or client memory (no `save_entry` in `onboarding_orchestrator`). Writers are CLI verbs (`cli.py:563`), the legacy Streamlit workbench, and `mapping_persistence` sinks A/B/C (CLI approve flows). |
| OCC configuration handling | Preflight (client details), `client_rule` store + overlay materialisation, rule projection to client memory — already OCC-owned. No system/regime/asset governance, no immutable per-run configuration record. |
| Legacy onboarding callers | `orchestrator_agent/adapters.py` (production), `engine/onboarding_agent/cli.py`, `agents/` v1 package, `ui/onboarding_review.py` + `cli/onboarding_review_cli.py` (legacy review UIs), `apps/blob_trigger_app/decisions_bridge.py`, `frontend/mi-agent-ui/scripts/generate_funded_fixtures.py`, `trakt_run.py` (TODO comment only). |

## 2. Responsibility map

| Responsibility | Current owner | Target owner | Change |
|---|---|---|---|
| Configuration selection/precedence | implicit in `RealAgentAdapters` arg defaults | OCC resolver | new `operations_control/configuration/` |
| System/regime/asset versioning + activation | none (live files, git only) | OCC config packages (admin) | new package store, draft→validate→activate→rollback |
| Client configuration | repo YAML + OCC `client_rule` overlay | OCC (unchanged, formalised as a layer) | resolver integration |
| Portfolio overrides | OCC portfolio-scoped rules | OCC (unchanged) | resolver integration |
| Run decisions | OCC decision/rule store | OCC (unchanged) | `decision_set_version` pinning |
| Effective configuration | none (agent reads live files) | OCC — immutable per-run `EffectiveConfiguration` | new contract + persistence + snapshots |
| Onboarding execution | `run_operator_workflow` (paths + flags) | narrowed `run_onboarding(files, effective_config, context)` | new facade `engine/onboarding_agent/narrowed.py` (additive file; no existing agent file modified) |
| Decision requests | scattered artefacts (33/34/36) | structured `DecisionRequired` (category/severity/observed values/affected counts) | additive contract fields |
| Approvals/rule scoping/persistence | OCC | OCC (scopes extended: file>run>portfolio>client>asset>global; asset/global admin-only) | additive |
| Config mutation by agent | absent on managed path | forbidden by contract + test | formalised |

## 3. Precedence model (formal contract)

```
system defaults < regime < asset < client < portfolio < approved run decisions
```

Most-specific wins; every effective value carries `provenance[key] = layer`
(via the existing `config_resolver` merge) and overrides record original value,
reason, approver, timestamp and scope (from the rule store records). Rule-set
precedence (mappings/aliases/enums) keeps the established OCC order with the
new scopes inserted: `file > run > portfolio > client > asset > global`.
Repository-observed exception preserved and documented: run-approved onboarding
decisions > client memory > registry aliases (the agent's own application
order) — the resolver feeds those artefacts, it does not reorder them.

## 4. EffectiveConfiguration contract

Immutable, hashed, persisted to the operations-control container BEFORE the
agent runs; every layer pinned by content hash; materialised as a read-only
snapshot directory the agent receives instead of live YAML paths. Fields per
the specification (`operations_control/configuration/contract.py`). Reruns
after decision approval produce a NEW version (bumped `decision_set_version`);
historic runs keep their persisted version.

## 5. Existing-file changes that may affect runtime behaviour (flagged)

1. **OCC files** (`engine.py`, `contracts.py`, `rules.py`, `auth.py`,
   `api/app.py`, presenters, conftest) — OCC-owned; the behavioural change is
   that OCC-driven onboarding receives **hash-verified snapshot copies** of
   the same configuration files. Canonical output is unchanged by
   construction (byte-identical inputs) and verified by regression
   (central-tape hash equality, legacy vs narrowed path).
2. **`engine/onboarding_agent/narrowed.py`** — new file only; no existing
   agent module is edited. Production `RealAgentAdapters` untouched; the OCC
   routes through a subclass.
3. **No pipeline, calculation, regime, projector, XML or repo-config file is
   modified.** Config packages snapshot file *contents* into the governed
   container; the repo files remain the authoritative storage format and the
   seed of version 1.

## 6. Compatibility and legacy callers

| Caller | Disposition |
|---|---|
| OCC engine (`GovernedAdapters`) | **migrated** to the authoritative path (resolver → EffectiveConfiguration → narrowed agent) |
| `apps/blob_trigger_app` production trigger | **wrapped-compatible, deferred** — untouched this session (out of OCC scope; same underlying workflow function). Migration = pointing its invoker at the narrowed facade; requires separate approval. |
| `engine/onboarding_agent/cli.py`, `decisions_bridge` | **deferred** — legacy operational tools; still call the workflow directly |
| `agents/` v1, `ui/onboarding_review.py`, Streamlit workbench, mi-agent-ui fixture script | **obsolete tier** — unchanged, listed for later removal |
| `run_onboarding_legacy(...)` compatibility wrapper | provided in the narrowed module; resolves raw paths into an ad-hoc effective configuration and emits a deprecation warning |
