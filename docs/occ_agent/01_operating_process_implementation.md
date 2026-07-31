# OCC Agent — the onboarding operating process

What was built, why it is shaped that way, and what remains before it can be
switched on in production.

The previous implementation answered *"if we onboard this client, will their
data go through?"* — a useful pre-flight capability, but not the objective. The
objective is that **the OCC Agent owns and coordinates the complete onboarding
operating process**, from the human's initial instruction through client
information collection, governed configuration, human approval, and initiation
of the existing live Onboarding Agent.

---

## 1. Revised architecture

### The shape

```
      human instruction
             │
             ▼
 ┌───────────────────────┐   reads     ┌────────────────────────────┐
 │  extraction.py        │◀────────────│ config/onboarding/         │
 │  cues + value shapes  │             │   field_catalogue.yaml     │
 │  derived per FIELD    │             │ config/asset/              │
 └──────────┬────────────┘             │   product_profiles.yaml    │
            │                          └────────────────────────────┘
            ▼                                        ▲
 ┌───────────────────────┐                           │ projects
 │ interpretation.py     │  Interpretation           │
 │ instruction / action  │  (steps, delivery,        │
 └──────────┬────────────┘   provenance, misses)     │
            │                                        │
            ▼                                 ┌──────┴──────┐
 ┌───────────────────────┐  ApplicationPlan   │  pack.py    │
 │ planning.py           │  understood /      │  the client │
 │ WHAT IT WOULD DO      │  proposed /        │  pack       │
 │ (writes nothing)      │  questions /       └──────┬──────┘
 └──────────┬────────────┘  unrecognised             │
            │                                        ▼
            │                              ┌────────────────────┐
            │                              │ communication.py   │
            │                              │ DRAFTED → REVIEW → │
            │                              │ APPROVED → SENT    │
            │                              └────────────────────┘
            ▼
 ┌────────────────────────────────────────────────────────────────┐
 │ service.py — the bounded tool surface                          │
 │   every write goes through OnboardingService.save_step         │
 └───────┬──────────────────────────────────────┬─────────────────┘
         │                                      │
         ▼                                      ▼
 ┌────────────────┐                    ┌──────────────────────┐
 │ onboarding/    │  the governed      │ review.py            │
 │ OnboardingCase │  record: answers,  │ the package a human  │
 │ + provenance   │  provenance,       │ approves             │
 │ + events       │  events            └──────────┬───────────┘
 └────────────────┘                               │
         ▲                                        ▼
         │                            ┌───────────────────────────┐
         │                            │ adapters.py               │
         │                            │  assert_may_activate      │
         │                            │  ── THE ONE GATE ──       │
         │                            └────┬─────────────────┬────┘
         │                                 │                 │
         │                    SyntheticExecutionAdapter   LiveExecutionAdapter
         │                    (refuses, audited)          (flag-gated)
         │                                                      │
         └──────────────────────────────────────────────────────┘
                          activate() → create_batch()
                          → upload_batch_files() → start_batch()
```

### What changed, and why

**`READY_FOR_EXECUTION` became a waypoint.** It used to be terminal, which
encoded the old objective. The lifecycle now runs on through review, approval,
a distinct confirmation, activation and `INGESTION_STARTED`. The terminal states
are `INGESTION_STARTED` and `CANCELLED`.

**One workflow, two effects.** Stages 1–7 are identical whether the case is
being rehearsed or performed. Only what happens at the end differs, and that is
a single injected `ExecutionAdapter`. There is no "synthetic mode" branch
running through the process — a rehearsal and a real onboarding are the same
code, the same case, the same catalogue, the same controls, the same approvals.

**Planning was separated from applying.** `planning.py` computes an
instruction's whole effect without writing anything, so the operator sees a
proposal *before* it is applied rather than discovering afterwards what a
sentence did.

**Recognition is derived from configuration.** `extraction.py` builds its cues
and value patterns from each catalogue field's own declaration. The previous
hand-written vocabulary guaranteed the agent would eventually ask for something
it could not itself understand.

**`activate()` is no longer only a boundary.** It is simultaneously the line a
rehearsal must never cross and the mechanism stage 8 requires. Both are true:
the synthetic adapter names it as a capability and takes the audited refusal;
the live adapter calls it. Neither reimplements it.

### What was deliberately NOT built

* No second onboarding model, catalogue, validator or activation path.
* No second pipeline. The live adapter sequences four existing calls.
* No user-management, identity or entitlement framework (see §2 below).
* No parallel field-mapping process (see §2 below).
* No business logic in React. Every panel renders a decision the server made.
* No mail integration. `RecordOnlyAdapter` records and says it sent nothing.

---

## 2. The three decisions, as implemented

### Decision 1 — field mappings stay where they are

Mappings continue to be learned from the first delivery, proposed by the
existing capability, reviewed through the existing approval process, promoted
and fingerprinted through the existing governed path.

* `file_role_schemas` and `expected_schema_fingerprint` remain in the
  catalogue's `not_collected` list — asserted by
  `test_the_catalogue_still_records_mappings_as_not_collected`.
* The pack states the decision to the client (`pack.MAPPING_STATEMENT`).
* The review package states it to the approver (`review.MAPPING_NOTE`):
  *"Field mappings are NOT part of this configuration and were not collected …
  Approving this activation does not approve any mapping."*
* A new governed `data_definitions` catalogue section collects exactly the
  permitted scope — source file, description, proprietary fields, units and
  currency, balance definition, date conventions, point-in-time vs cumulative,
  known limitations — all writing to `onboarding_record`.

### Decision 2 — user access is a requirement, not a grant

A new governed `access` catalogue section collects user name, email, role,
scope, OCC access, dashboard access and report-recipient status. Every field
writes to the existing `onboarding_record` artefact — asserted by
`test_access_is_collected_into_the_governed_onboarding_record`.

Trakt reads its operators from environment configuration, so
`review.access_actions` turns each collected row into structured operator
actions marked `not_provisioned`, and the package's `ACCESS_NOTE` says so
outright. `test_nothing_claims_access_was_provisioned` asserts the wording.

### Decision 3 — the live adapter exists and is disabled

`OCC_AGENT_LIVE_ENABLED` is off by default and fails closed on anything it does
not recognise. It is not set anywhere in this repository, and
`test_the_repository_never_ships_the_flag_enabled` asserts that.

All eleven preconditions are checked in one function,
`adapters.assert_may_activate`, which reports every unmet reason rather than the
first. Approving the configuration and confirming activation are distinct states
and distinct calls; a bare "yes" can never be the confirmation
(`pending_confirmation_action` deliberately omits
`ACTIVATION_CONFIRMATION_REQUIRED`).

The live adapter was tested through fakes only. No Azure resource was contacted
at any point.

---

## 3. Full file ledger

### New backend modules

| File | Lines | Purpose |
|---|---:|---|
| `operations_control/occ_agent/extraction.py` | 849 | Catalogue-derived recognition: cues, value shapes, options, asset signal tokens |
| `operations_control/occ_agent/planning.py` | 379 | What an instruction would do. Merge rules and disclosure. Writes nothing |
| `operations_control/occ_agent/pack.py` | 383 | The client pack, projected from the governed catalogue |
| `operations_control/occ_agent/communication.py` | 142 | The four-state issue workflow and the delivery seam |
| `operations_control/occ_agent/review.py` | 358 | The package a human approves; provenance; operator actions |
| `operations_control/occ_agent/adapters.py` | 332 | The one activation gate; synthetic and live execution adapters |

### Modified backend

| File | Δ | What changed |
|---|---:|---|
| `operations_control/occ_agent/service.py` | +848 | Planning, pack workflow, review, activation, background jobs, provenance |
| `operations_control/occ_agent/interpretation.py` | −734/+ | Rewritten onto `extraction`; six-value provenance; new actions |
| `operations_control/occ_agent/states.py` | +274 | Ten new states; `READY_FOR_EXECUTION` demoted to a waypoint |
| `operations_control/occ_agent/store.py` | +253 | Durable artefacts, packages, retention, purge, identifier reservations |
| `operations_control/occ_agent/api.py` | +143 | Pack, review and activation routes |
| `operations_control/occ_agent/run.py` | +36 | `mode`, pack fields, review ref, activation intent and result |
| `operations_control/occ_agent/scenarios.py` | +44 | Drives the pack and the review/approval half |
| `operations_control/occ_agent/fixtures.py` | +27 | Expectations follow the extended lifecycle |
| `operations_control/onboarding/case.py` | +10 | `provenance_class` beside the existing free-text `provenance` |
| `operations_control/onboarding/service.py` | +5 | `data_definitions` and `access` steps |
| `config/onboarding/field_catalogue.yaml` | +192 | The two new governed sections |

### Frontend

| File | Δ | What changed |
|---|---:|---|
| `src/api/agentTypes.ts` | +177 | Pack, review, activation and disclosure types |
| `src/api/HttpOpsClient.ts` | +78 | Nine new calls |
| `src/api/OpsClient.ts` | +24 | The interface those satisfy |
| `src/api/MockAgent.ts` | +505 | The extended lifecycle, pack, review and refused activation |
| `src/api/MockOpsClient.ts` | +53 | Wiring |
| `src/lib/copy.ts` | +55 | Operator-facing wording |
| `src/screens/agent/AgentCase.tsx` | +458 | `Disclosure`, `PackPanel`, `ReviewPanel`, `ActivationPanel` |
| `src/screens/agent/AgentCases.tsx` | +5 | "Ready" filters on the readiness status, not the position |

### Tests

| File | Lines | Covers |
|---|---:|---|
| `test_planning.py` | 219 | Merge rules, disclosure, provenance |
| `test_conversation_coverage.py` | 186 | Every collected field, answerable in words (150 cases) |
| `test_pack_and_review.py` | 327 | Pack ↔ catalogue, the workflow, the review package, access |
| `test_activation.py` | 343 | Flag, gate, approval ≠ activation, live contract via fakes |
| `test_operational.py` | 216 | Durability, retention, reservations, isolation, background runs |

Plus updates to `test_workflow.py`, `test_state_model.py`,
`test_interpretation.py`, `test_synthetic_safety.py`,
`test_component_reuse.py`, `test_tenancy_and_api.py` and `AgentTab.test.tsx`.

**Total: 38 files, +7,521 / −783.**

---

## 4. Evidence: the pack is derived from the governed catalogue

Two tests, in both directions:

* `test_every_question_traces_to_a_catalogue_field` — for every question the
  pack produces, `cat.field(section, field)` exists, is `collected`, and the
  question's label and `writes_to` are the field's own.
* `test_every_client_facing_catalogue_field_appears_in_the_pack` — for every
  catalogue field that is collected and not answered by Trakt, a question
  exists. Nothing the catalogue asks is silently dropped.

The frontend asserts the same property against the mock
(`every pack question is a field the governed catalogue declares`).

Observed on scenario A: **58 questions across 9 sections** — `client`,
`entities`, `contacts`, `portfolios`, `sources`, `reporting`, `presentation`,
`data_definitions`, `access` — with 29 outstanding.

---

## 5. Evidence: conversation coverage for all collected fields

`test_conversation_coverage.py` is parametrised over
`extraction.collected_fields(catalogue())` — every field the catalogue says is
collected — and asserts twice per field:

1. a sentence built from that field's **own label** is read back to that field;
2. the value survives the round trip, in the shape the field's declaration
   coerces it to.

**150 test cases, all passing.** Add a field to the catalogue and the suite
starts asking about it with no change to the test.

`test_every_checklist_item_can_be_answered` closes the specific loop: it takes
the checklist the agent itself generates and answers every item back to it.

Four real gaps were found and fixed by writing this test:
`portfolios.owning_entity` (an `entity_reference` took no free-text value),
`data_definitions.units_and_currency` (the conjunction splitter cut a field
label containing "and"), `presentation.brand_colour` (`#112233` was read as
`#112`), and `portfolios.portfolio_type` (an option token appearing in its own
field's label was treated as ambiguous).

---

## 6. Evidence: partial instructions cannot apply silently

```
--- 6. disclosure: a partly-read instruction ---
instruction: "The LEI is 894500SYNTHETIC00042 and sort out the other thing."
{
  "understood":   ["Legal Entity Identifier (Kestrel Mutual): 894500SYNTHETIC00042"],
  "proposed":     "Legal Entity Identifier (Kestrel Mutual): 894500SYNTHETIC00042.",
  "questions":    [],
  "unrecognised": ["sort out the other thing"]
}
  complete: False  -> apply_plan raises PartiallyUnderstood
  -> OCC_AGENT_PARTIALLY_UNDERSTOOD
  LEI on the case afterwards: (nothing)
```

The mechanism:

* `ApplicationPlan.complete` is false if anything is unrecognised or unresolved;
* `service.apply_plan` raises `PartiallyUnderstood` unless `confirm=True`;
* `instruct()` forces a proposal for any material plan and attaches the
  disclosure, so the UI shows all four populations and the sentence *"Nothing
  has been applied."*

Tests: `test_a_partly_read_instruction_applies_nothing`,
`test_the_disclosed_remainder_can_be_confirmed`,
`test_a_turn_reports_all_four_populations`,
`test_a_half_understood_sentence_reports_the_half_it_missed`.

---

## 7. Evidence: two portfolios cannot be destructively merged

The defect: "They also have portfolio id direct_102, equity release." renamed
`direct_101` to `direct_102` and moved its delivery registration with it, while
the proposal summary read as though something had been added.

The fix is an exhaustive rule table in `planning._plan_repeatable` with **no
fallback**. The old `len(incoming) == 1 and len(merged) == 1` shortcut is gone.

```
--- P1: two portfolios are two portfolios ---
  before: ['direct_501'] -> after: ['direct_501', 'direct_502']
  deliveries still registered for: ['direct_501', 'direct_502']
```

Regression tests: `test_a_second_portfolio_is_added_not_merged_over_the_first`,
`test_adding_two_portfolios_in_sequence_keeps_all_three`,
`test_the_first_portfolios_delivery_survives_a_second_being_added`,
`test_naming_the_same_identifier_updates_that_book`,
`test_a_change_with_no_identifier_and_several_books_asks_which`,
`test_a_change_with_no_identifier_and_one_book_is_only_proposed`,
`test_a_rename_is_never_produced_by_a_new_identifier`.

---

## 8. Synthetic workflow demonstration

Driven through `scenarios.run_scenario`, i.e. the same service calls the UI
makes:

```
execution states : AWAITING_ONBOARDING → PACK_REVIEW_REQUIRED
                 → PACK_APPROVED_TO_SEND → PACK_SENT → READY_TO_RUN
                 → SYNTHETIC_ONBOARDING_PASSED → EXECUTION_APPROVAL_REQUIRED
                 → READY_FOR_EXECUTION → READY_FOR_REVIEW
                 → ACTIVATION_CONFIRMATION_REQUIRED
onboarding states: draft → information_requested → in_review
                 → ready_for_approval → approved
stopped because  : the case is waiting for an explicit confirmation to
                   activate, which a rehearsal never gives

--- the pack workflow ---
  DRAFTED                by Alice  the agent drafted the pack from the catalogue
  HUMAN_REVIEW_REQUIRED  by Alice  a human must read it before it goes out
  APPROVED_TO_SEND       by Alice  Reviewed in a practice case.
  SENT                   by Alice  Recorded as issued. Trakt did not send it…
  receipt.sent = False

--- provenance on the governed record (20 classified fields) ---
  client.client_name                     human_supplied
  client.jurisdiction                    human_approved
  contacts.reporting_contact_email       client_supplied
  entities[0].country_of_establishment   client_supplied
  …

--- access: a requirement, not a grant ---
  [not_provisioned] occ_access           Dana Fox <dana@kestrel.example>
  [not_provisioned] report_distribution  Dana Fox <dana@kestrel.example>
  [not_provisioned] approval_role        Dana Fox <dana@kestrel.example>

--- the confirmation gate ---
  adapter: synthetic   live_enabled: False
  would do:
    - Write 4 configuration artefact(s) for NORTHSTAR, as a new governed version.
    - Register the expected source deliveries in the production source registry.
    - Place 2 file(s) in the production raw location.
    - Start the existing Onboarding Agent, which will profile, map, transform,
      validate and assemble the delivery.
  target:
    blob://raw-v2/NORTHSTAR/direct/funded/monthly/direct_101/2026-06-30/…
  refused for:
    - This case is in rehearsal mode.
    - Live execution is not switched on in this environment.
  confirm_activation -> OCC_AGENT_ACTIVATION_REFUSED

--- the boundary ---
  files in the LIVE container: 0
  onboarding status          : approved
  activated_version          : None
  audit chain intact         : True
  audit events               : 20
  includes activation_refused: True
```

---

## 9. Live-adapter contract demonstration (fakes only)

`test_activation.py` drives `LiveExecutionAdapter` against `FakeOnboarding` and
`FakeEngine`. No Azure client is constructed and the flag is never set on the
real environment.

**The governed path, in order.**
`test_the_live_adapter_calls_the_governed_path_in_order` asserts:

```
onboarding.calls == [("activate", "ONB-2026-0001", "Alice")]
engine.calls     == ["create_batch", "upload_batch_files", "start_batch"]
result.version == 1, batch_id == "bat_0001", workflow_id == "wf_0001"
engine.uploads == [("loans.csv", b"a,b\n1,2\n")]
```

**It refuses before it touches anything.** With `confirmed=False`, both fakes
record zero calls.

**A partial activation reports what had already happened.** Parametrised over a
failure at each of the three engine calls: the result carries the version that
was written, the calls that succeeded, and *"Activation stopped part-way."*

**It does not reimplement the pipeline.**
`test_the_live_adapter_does_not_reimplement_the_pipeline` inspects the source
for the four governed calls and asserts the absence of `write_bytes`,
`BlobServiceClient`, `run_orchestration`, `requests.` and `subprocess`.

**The gate is singular.** `test_there_is_exactly_one_gate_in_the_codebase`
asserts `_adapters.assert_may_activate(` appears exactly once in `service.py`
and that no other module in the package calls it.

**Each precondition refuses on its own.** Fourteen parametrised cases, one per
precondition, each breaking exactly one and asserting the specific reason.

---

## 10. Test results

| Suite | Result |
|---|---|
| `tests/operations_control/occ_agent` | **209 passed** |
| `tests/operations_control` (incl. Client Onboarding's own 327) | **792 passed** |
| Frontend `vitest run` | **143 passed** (18 files) |
| Frontend `npm run lint` (`tsc --noEmit`) | clean |

New tests contributed: 150 conversation-coverage cases, 49 activation cases,
27 pack-and-review cases, 15 operational cases, 15 planning cases.

---

## 11. Remaining production-enablement steps

Nothing here enables live execution. To do so deliberately:

1. **Set the flag** — `OCC_AGENT_LIVE_ENABLED=true` in the target environment.
   On its own this grants nothing: it is one of eleven preconditions.
2. **Construct the service with the engine** — `OccAgentService(..., engine=…)`
   so `_default_adapter` returns a `LiveExecutionAdapter`. A live case must not
   use the practice container: point the `OnboardingService` at the real
   `OpsStore` rather than `synthetic_ops_store`.
3. **Set the run's mode** — `SyntheticRun.mode = "live"` for cases intended for
   production. There is no code path that flips this implicitly.
4. **Decide the tenancy model for live cases.** The practice container's
   retention (30 days) and identifier reservations are rehearsal mechanisms; a
   production case is a permanent record and neither should apply to it.
5. **Register a communication adapter**, if the pack is to be emailed rather
   than issued by hand. `communication.default_adapter` is a lookup, so this is
   a registration and not a rewrite. Until then the receipt honestly says
   nothing was sent.
6. **Exercise the live adapter against staging.** Its contract is tested with
   fakes; the real `create_batch` / `upload_batch_files` / `start_batch` have
   never been called from this code path.
7. **Provision access separately.** The review package's operator actions are
   the input to that; Trakt's operator list still comes from
   `TRAKT_OPS_OPERATORS`.

### Known limitations

* **Retention on a blob backend.** `store.purge` removes what the storage
  abstraction lets it and *reports* what it could not. A container lifecycle
  rule is the real mechanism, and it is not configured by this code.
* **The production source registry is read.** `EffectiveConfigResolver` and the
  onboarding store share a `Storage` client, so a practice case reads live
  registrations. It never writes them —
  `store.assert_no_live_registry_write` names the guarantee and a test asserts
  the refusal — but the read is real and is documented in `store.py`.
* **Recognition is deterministic, not exhaustive.** `extraction.py` reads what
  the catalogue lets it read. Anything else is reported as unrecognised rather
  than guessed, which is the intended failure mode, but it does mean an operator
  writing very freely will be asked to rephrase.
