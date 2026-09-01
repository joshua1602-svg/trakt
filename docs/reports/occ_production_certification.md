# OCC Production Certification

Final pre-go-live control-plane review of the Operations Control Centre.

---

## 1. Executive verdict

**PASS WITH OPERATING CONDITIONS.**

A trained Trakt operator can take a lender from onboarding through activation,
a complete reporting cycle, exception resolution, publication and the next
reporting period, without a developer editing code, YAML, CSVs, state files,
databases or pipeline artefacts. This was executed, not asserted: two unrelated
synthetic lenders were driven through the real production entrypoint — one
MI-only, one regime-required — for two reporting periods each. Every
intervention was a business or control action taken through the OCC decision
API. **Zero developer actions were required in either rehearsal.**

That is the proven claim, and it is a **first-pass-correct** claim. The
conditions below are all about what happens when something goes wrong, and the
first is the one that matters most.

1. **An approved mapping decision cannot be changed by an operator.** The Rules
   Library is read-only: `GET /ops/rules` and `GET /ops/rules/{id}/history`
   exist, and there is no route — and no engine method — to retire, amend or
   supersede a standing rule. `RuleStore.retire()` exists in the code and
   nothing calls it. Demonstrated at runtime: an operator answered the loan-key
   question with `CUST_ID` (the customer) instead of `ACCT_REF` (the account);
   the period published keyed on the customer; the next period, with different
   content, asked **0 questions**, silently reused the wrong key, and published
   again. Correcting it means editing the rule store — a developer intervention.
   The brief's own list of allowed business actions includes "approve/**change**
   mappings"; changing is not currently possible. **P1 with a developer-only
   consequence on the error path.** Not fixed here: Part 11 directs that a
   potential P0 requiring architectural expansion is reported, not implemented,
   and a rule-supersession surface is exactly that.
2. **The Gate 3 validation exception is a blind override.** When Gate 3 halts,
   OCC asks one question — "Some checks did not pass. Do you want Trakt to
   continue and prepare the report anyway?" — with *no evidence attached at
   all*: no failing field, no rule, no row count, no severity. Approving it
   force-publishes past a **core canonical contract violation**, and the stage
   record is then rewritten to `status: completed`, `summary: "All checks
   passed."`, `blockers: []`. Demonstrated at runtime: a book with
   `account_status` blank on all 30 rows was published to MI with all 30 blanks
   intact. Acceptance criterion 7 therefore does not hold in the presence of an
   operator override. **P1.**
3. **The file-role answer is never learned.** A lender whose file name is
   outside the built-in vocabulary is asked "what is this file?" on *every*
   reporting period, forever. The answer is recorded against the batch and
   never promoted to the source registry's `file_role_schemas` /
   `file_role_aliases`, although the machinery to do so exists and is used by
   the legacy route. One extra operator click per period. **P1.**
4. **Expected external infrastructure and human controls** — Azure Event Grid,
   blob storage, App Service, authentication, and publication approval itself —
   remain outside OCC by design. These are not defects.

There are **no P0 blockers**: nothing prevents a commercial reporting cycle in
the intended current scope, and two were completed end to end. Nothing in OCC
orchestration should prevent onboarding Client 1 and commencing the
managed-service production acceptance checks, provided conditions 1 and 2 are
operated around or closed first.

---

## 2. Starting point

| | |
|---|---|
| Starting `main` SHA | `e7678c81100f562c25cb39cf1cbf69798e13a5ed` |
| Branch | `claude/occ-production-certification-axdj1z` |
| Date | 2026-08-30 |
| Production code changed | **No.** Audit and runtime certification only. |
| Working tree at start | Clean, at the `main` SHA above |

The rehearsal harnesses written for this certification live outside the
repository (session scratchpad). The only committed artefact is this report.

---

## 3. Actual production lifecycle

### 3.1 The route, as it actually executes

```
Lender writes to  raw-v2/{client}/{book}/{dataset}/{frequency}/{portfolio}/{period}/{file}
   │
   ▼  Event Grid
function_app.py :: on_raw_blob_event → _dispatch
   │  classify_blob_event (container filter) → occ_intake.handle_arrival
   ▼
apps/blob_trigger_app/occ_intake.py
   │  parse_blob_path            — an unparseable path is refused, not guessed
   │  outcome_for_source         — mi | mi_annex2, from the source registry's
   │                               regime_required; non-funded datasets
   │                               short-circuit to MI whatever the registry says
   │  engine.create_batch  →  engine.register_batch_file
   ▼
operations_control/engine.py :: assess_batch
   │  intake.classify            — file roles, header-first then filename
   │  _client_is_activated       — FAIL CLOSED for an ungoverned client
   │  _config_resolver().resolve — effective configuration readiness
   │  _sync_file_role_decisions  — "what is this file?" where unrecognised
   │  intake.assess              — batch state
   │  start_batch (auto)         — manifest → delivery → workflow → start()
   ▼
OpsEngine.start → daemon thread → OpsEngine._execute
   │  rules.applicable → project_rules_to_client_memory
   │  _approved_decisions_digest — Gate 1 output is stale if answers changed
   │  _resolve_effective_config  — PINNED for the life of the run
   │  _write_approved_overrides_file
   ▼
engine/orchestrator_agent/orchestrator.py :: run_orchestration (full_pipeline=True)
   │  onboard    Gate 1  — mapping, target coverage, handoff readiness
   │  transform  Gate 2  — refuses a package Gate 1 has not declared ready
   │  validate   Gate 3  — canonical core-field contract
   │  stamp/assemble     — governed canonical (platform_canonical_typed.csv)
   ▼
_apply_run_state → translate_run_state → GovernedAgentResult per stage
   │
   ├── outcome = mi         → _prepare_publication → awaiting_publication
   │
   └── outcome = mi_annex2  → _run_annex2_chain
         │  regulatory_config  annex2/preflight.run_preflight
         │  projection         Gate 4  engine/gate_4_projection/regime_projector.py
         │  delivery_prep      Gate 4b engine/gate_4b_delivery/annex2_delivery_normalizer.py
         │  xml_delivery       Gate 5  engine/gate_5_delivery/xml_builder_annex2.py
         │                             + lxml XSD validation against
         │                             DRAFT1auth.099.001.04_1.3.0.xsd
         ▼
       _prepare_publication → awaiting_publication
   ▼
Operator: POST /ops/workflows/{id}/publish
   ▼
approve_publication → ProductionPersistence
   │  persist_platform     blob://processed-v2/platform/{client}/latest|{period}/
   │  persist_regime_dir   regime prefix (XML + interventions + effective config)
   │  _promote_source      (new_client / new_portfolio only)
   ▼
   published  →  MI Query / dashboard read the published governed canonical
```

### 3.2 Stage map (A–U)

| # | Stage | Entry point | Production module / function | Persisted state | Operator surface | Next state | Hold / error state | Resumable |
|---|---|---|---|---|---|---|---|---|
| A | Client onboarding | `POST /ops/onboarding/cases` | `onboarding/service.py::start_new_client`, `save_step` | onboarding case doc (`_onboarding/cases`) | Onboarding case wizard, 7 steps | `ready_for_approval` | `changes_required`, `information_requested` | Yes — case is durable |
| B | Portfolio / source onboarding | `PUT /ops/onboarding/cases/{id}` steps `portfolios`, `sources` | `onboarding/service.py::save_step`, `_sync_derived` | case `answers.portfolios/sources` | Same wizard | `ready_for_approval` | readiness `blocking` problems | Yes |
| C | Activation | `POST /ops/onboarding/cases/{id}/approve` then `/activate` | `service.py::approve` → `activate` → `onboarding/artefacts.py::plan`/`apply` | generated client config, portfolio registry, client index, **source registry record** (`regime_required`), version + content hash | Approve / Activate buttons | `activated` | `OPS_ONBOARDING_INCOMPLETE`, `OPS_ONBOARDING_NOT_APPROVED` | Yes — terminal once activated |
| D | Raw delivery intake | Event Grid → `function_app.py::_dispatch` | `occ_intake.handle_arrival` → `engine.create_batch` / `register_batch_file` | input batch doc, `source_prefix`, file SHA-256 | Input packs list | `classifying` | `unparseable_path` (no batch), `legacy_sentinel_ignored` | Yes — idempotent on batch id |
| E | Workflow selection | inside `handle_arrival` | `occ_intake.outcome_for_source` | `batch.workflow_type` | none (deterministic) | batch assess | registry read failure → logs and defaults to `mi` | Yes |
| F | Gate 1 | `OpsEngine._execute` | `orchestrator.run_orchestration` step `onboard` (Onboarding Agent) | `run_state.json`, `18_central_lender_tape.csv`, `24_onboarding_handoff_manifest.json` | Review Centre queue | Gate 2 | `halted` → `needs_review` | Yes — orchestrator resume state |
| G | Review Centre | `GET /ops/reviews`, `POST /ops/reviews/{id}/decision` | `engine.resolve_decision` → `_persist_rule` → `_write_approved_decisions_file` | decision docs, `RuleRecord` versions, `34_target_first_decisions_approved.yaml` | approve / amend / defer / reject, with scope | Gate 1 rerun | `OPS_*` refusals | Yes |
| H | Gate 1 deterministic rerun | `POST /ops/workflows/{id}/rerun`, or automatic when the queue empties | `engine.rerun` → `start` → `_execute` with `redo_onboarding` on digest change | `onboarding_decisions_digest`, `rerun_count` | "Run again" | Gate 2 | `OPS_ALREADY_RUNNING` | Yes |
| I | Gate 2 | orchestrator step `transform` | `engine/transformation_agent` | `31_transformed_canonical_tape.csv` | none (refuses, does not ask) | Gate 3 | `halted` when Gate 1 not ready | Yes |
| J | Gate 3 | orchestrator step `validate` | `engine/gate_3_validation/validate_canonical.py` | `43_validation_issues.csv`, validation manifest | validation-exception decision (see section 8C) | assembly | `halted` → `needs_review` | Yes |
| K | Governed canonical / assembly | orchestrator `stamp` / `assemble` | `engine/assembler_agent.py` | `out_platform/platform_canonical_typed.csv`, `state.central_canonical_path` | none | publication prepared | `halted`/`failed` → `blocked` | Yes |
| L | MI publication | `POST /ops/workflows/{id}/publish` | `engine.approve_publication` → `persistence.persist_platform` | publication doc (id, version, rule_versions, source_artefacts), `blob://processed-v2/platform/{client}/latest` and `/{period}/` | Publish / Hold | `published` | `OPS_PUBLICATION_NOT_PREPARED`, `OPS_ALREADY_PUBLISHED` | Terminal |
| M | MI Query availability | — | reads `platform_latest_uri` / `platform_period_uri` | same blobs | MI Query Agent | — | — | n/a |
| N | Regulatory branch selection | `_apply_run_state` | `run.outcome == OUTCOME_MI_ANNEX2` → `_run_annex2_chain` | `run.annex2` | none (from approved configuration) | Gate 4 | — | Yes |
| O | Gate 4 | `_run_annex2_chain` | `annex2/stages.py::run_projection` → `engine/gate_4_projection/regime_projector.py` | `annex2_ESMA_Annex2_projected.csv`, `population_reconciliation.json` | ND / enum / source / constant decisions | Gate 4b | `_park_regulatory` → `needs_review`, MI unaffected | Yes — `done()` skips completed stages |
| P | Gate 4b | `_run_annex2_chain` | `stages.py::run_normalisation` → `engine/gate_4b_delivery/annex2_delivery_normalizer.py` | `annex2_ESMA_Annex2_delivery_ready.csv`, `effective_delivery_rules.yaml` | same decision surface | Gate 5 | `_park_regulatory` | Yes |
| Q | Gate 5 XML | `_run_annex2_chain` | `stages.py::run_xml` → subprocess `engine/gate_5_delivery/xml_builder_annex2.py` | `annex2_submission.xml`, `builder_interventions.json`, `xml_sha256` | none | publication prepared | `_park_regulatory`, temp file deleted | Yes |
| R | XSD validation | inside `run_xml` | `lxml` inside the Gate 5 builder, `--xsd DRAFT1auth.099.001.04_1.3.0.xsd` | `xsd`, `xsd_result` on the run and publication | none | — | XML written to `*.building` and **deleted unless XSD passes** | Yes |
| S | Regulatory artefact publication | `POST /ops/workflows/{id}/publish` | `approve_publication` → `persistence.persist_regime_dir` | publication `annex2` block: xml, xsd, xsd_result, hashes, effective config | Publish / Hold | `published` | as L | Terminal |
| T | Next-period processing | new blob arrival | same route; standing rules applied by `rules.applicable` | new workflow, new publication version | only the questions not already answered | as above | as above | Yes |
| U | Failure / retry / recovery | `POST /ops/workflows/{id}/rerun`; API startup | `engine.rerun`; `engine.recover_on_startup` (`api/app.py` lifespan) | `interrupted` flag, lease (`owner_pid`, `started_at`) | "Run again" | `running` | `failed` → rerunnable | Yes |

The derived state machines behind this route — onboarding case, input batch
and workflow run, with entry, exit, actor, persistence and restart behaviour for
every state — are in section 9.1, where they feed the dashboard inventory in
section 10.

### 3.3 Steps executed outside OCC

| Step | Status |
|---|---|
| Event Grid subscription, blob storage, App Service, Azure auth | **External infrastructure.** Not OCC's to control. |
| Teams notification delivery | OCC writes durable intent; a separate timer trigger (`function_app.py::deliver_teams_notifications`) drains the outbox. Deliberately decoupled so delivery cannot fail a publication. |
| Investor PPTX generation | **Not on the OCC production route.** `pptx_stage` is invoked only by `apps/blob_trigger_app/orchestrator_invoke.default_orchestrator_invoker`, which is reached only from the legacy `router.handle_blob_event`. `operations_control/engine.py` never imports `orchestrator_invoke` except for `resolve_llm_policy`, and calls `run_orchestration` directly. Verified by import trace. |
| LLM mapping resolver | Wired and policy-gated (`TRAKT_LLM_ENABLED` / `TRAKT_LLM_MODE`), off in both rehearsals by recommendation. Recommender only. |
| Runtime logs / metrics | Azure-side. See sections 9 and 10. |

**Nothing was labelled automated on the strength of code existing.** Every stage
in the table above executed during the rehearsals in sections 5 and 6, and each is
evidenced by a persisted artefact named in the "persisted state" column.

---

## 4. Onboarding proof

Both lenders were created through the OCC onboarding case only. Every value is
an answer an operator types; no bespoke Python, no repository file edited.

**Path executed:** `start_new_client` → steps `client`, `entities`, `contacts`,
`portfolios`, `reporting`, `sources`, (`regime` where applicable) → `approve`
→ `activate` → source eligible for production intake.

### What OCC generates and owns

| Item | Where it lands | Verified |
|---|---|---|
| Client identity (`client_id`, name, jurisdiction, currency, time zone) | generated client config + client index | Yes — Client A and Client B configs differ |
| Portfolio identity, asset class, structure, period convention | portfolio registry | Yes |
| Country / base currency | generated client config | Yes |
| Source / dataset role, cadence, channel, format | **source registry record** | Yes — `records_for_dataset` returns the record after activation |
| Reporting product selection | `pipeline.mi_enabled` / `pipeline.esma_enabled` | Yes |
| `regime_required` | source registry record | Yes — `True` for Client B, `False` for Client A |
| Originator name / LEI / establishment country | `defaults.originator_*` in the generated client config, derived from the entity carrying the `originator` role | Yes — Client B's own LEI appears 31 times in its return, the incumbent's zero times |
| Asset/product configuration selection | asset pack resolved from portfolio type (`ASSET_PACKS`) | Yes — Gate 4 completed on the ERM product layer |
| Required client/deal regulatory facts | onboarding readiness, **enforced before approval** | Yes — see below |

### Required regulatory facts are enforced at onboarding, not discovered at Gate 4

Two probes attempted to activate a regime-required client with a fact missing.
Both were refused **before any configuration was written**:

```
no entity holds the originator role
  → OPS_ONBOARDING_INCOMPLETE
    "Regulatory reporting names an originator. Give one entity the originator role."
    blocking, owner: operator

originator present, LEI absent
  → OPS_ONBOARDING_INCOMPLETE
    "Legal Entity Identifier for No LEI Lending Limited is needed."
    blocking, owner: client
```

The `owner` field is the useful part: OCC distinguishes a fact the operator can
supply from one that must be requested from the lender. A Gate 4
`regulatory_config` decision surface exists as a backstop for clients activated
before these checks (migration cases); it was never reached in either rehearsal
because onboarding had already captured the facts.

### Settings still required in a repository configuration file

None for an OCC-activated client on the certified route. Two residual items
carried over from the previous sprint and unchanged here:

* `enrichment.uk_nuts3` (postcode → ITL3) is **not** generated by OCC, so a
  second UK lender receives no geography enrichment. It did not block either
  rehearsal (Annex 2 geography is satisfied by the `uk_geography` GBZZZ
  override, which OCC does generate). **P2, pre-existing, reported not fixed.**
* `analytics/streamlit_app_erm.py` still reads the incumbent client file's
  branding unconditionally. Dashboard presentation only. **P2, pre-existing.**

---

## 5. MI-only reporting proof — Client A

A brand-new equity release lender whose core system writes terse codes
(`ACCT_REF`, `BAL_OS`, `STATUS_CD`), present in no alias library.

| | Period 1 | Period 2 |
|---|---|---|
| Reporting period | 2025-11-30 | 2025-12-31 |
| Run ID | `wf_fe5f18136a36` | `wf_0d561e52acb0` |
| Client / portfolio | ALPHA / direct_001 | ALPHA / direct_001 |
| Source file | `LoanExtract.csv` at the governed blob prefix | same |
| Workflow selection | `mi` (no ESMA product) | `mi` |
| Workflow type | `new_portfolio` | `recurring` |
| File-role questions | 0 (filename recognised) | 0 |
| Review questions | **41** (40 target-first decisions + 1 loan-key) | **0** |
| Gate 1 | done | done |
| Gate 2 | done | done |
| Gate 3 | done | done |
| Assembly | completed | completed |
| Canonical | 30 rows, 30 distinct loan ids | 30 rows, 30 distinct |
| Publication | v1, `published` | v1, `published` |
| Rerun count | 1 | 0 |
| Elapsed | 15.4 s | 5.4 s |
| Shell / file manipulation | **none** | **none** |

**Manual actions, period 1:** 6 onboarding step answers, approve case, activate
client, choose the loan identifier (`ACCT_REF`), answer the mapping queue
(12 columns named, 3 static values, 25 marked not applicable), approve
publication. **Period 2: approve publication only.**

The loan-key question is worth naming: three columns in Client A's tape are
unique per row — `ACCT_REF`, `ROLL_NO`, `CUST_ID` — and only one keys the loan.
OCC raises a non-blocking question listing all three, states what it used, and
the operator's answer governs the canonical identity thereafter. Before the
answer the tape was keyed on the customer; after it, on the account. Period 2
used the same key without being asked.

### MI publication and MI Query availability

Both periods reached the published governed data:

```
blob://processed-v2/platform/ALPHA/latest/platform_canonical_typed.csv        exists
blob://processed-v2/platform/ALPHA/2025-11-30/platform_canonical_typed.csv    exists
blob://processed-v2/platform/ALPHA/2025-12-31/platform_canonical_typed.csv    exists
```

Read back from the published `latest` blob: 30 rows, 37 columns, carrying
`loan_identifier`, `current_outstanding_balance`, `account_status`,
`current_interest_rate`. An MI-style aggregate over it returns a total balance
of £6,849,569.00 and a status breakdown of `{LIVE: 28, REDEEMED: 2}`. The
governed data MI Query reads is present, complete and queryable.

**Acceptance:** standing mappings reused automatically ✓; standing client facts
reused ✓; no developer intervention ✓; no repeat questions ✓; correct new
reporting period and publication ✓; publication reached the same final state ✓.

---

## 6. Regime-required reporting proof — Client B

A second, unrelated lender: prose headers, a *policy* book rather than a loan
book, `regime_required` true, and management statuses ESMA has no code for.

| | Period 1 | Period 2 |
|---|---|---|
| Reporting period | 2026-01-31 | 2026-02-28 |
| Run ID | `wf_45eae6f11bda` | `wf_3388882d2489` |
| Client / portfolio | BETA / direct_002 | BETA / direct_002 |
| Source file | `PolicyExtract.csv` | `PolicyExtract.csv` |
| Workflow selection | **`mi_annex2`** from `regime_required` | `mi_annex2` |
| File-role questions | 1 | **1 (re-asked — see section 8B)** |
| Review questions | **48** over 3 rounds | **0** |
| Gate 1 / 2 / 3 | done / done / done | done / done / done |
| Assembly | completed | completed |
| Gate 4 projection | completed | completed |
| Gate 4b delivery prep | completed | completed |
| Gate 5 XML | completed | completed |
| XML | `annex2_submission.xml`, 588,379 bytes, **30 records, 107 fields** | same shape |
| `xml_sha256` | `51e893dc…ced78c` | `b65445c6…783ba1` (different content) |
| XSD | `DRAFT1auth.099.001.04_1.3.0.xsd` | same |
| `xsd_result` | **PASSED** | **PASSED** |
| Independent re-validation (this certification, `lxml.etree.XMLSchema`) | **valid: True**, 0 errors | **valid: True**, 0 errors |
| `nd_injected` / `review_required_instances` | 0 / 0 | 0 / 0 |
| Publication | v1, `published` | v1, `published` |
| Elapsed | 89.0 s | 32.8 s |
| Shell / file manipulation | **none** | **none** |

### The questions, in the order they were asked

| Round | Asked | Category |
|---|---|---|
| 1 | 1 | `regulatory_source` — "Which column holds the new obligor identifier?" (RREL5), offering the tape's own column headers |
| 2 | 43 | 42 × `regulatory_no_data` — "How should the regulator be told about *X*?", each offering only that field's permitted ND codes; + 1 `regulatory_source` (pool addition date, answered as a value) |
| 3 | 4 | `regulatory_enum_translation` — "How should 'In possession' be reported to the regulator?", offering that field's 15 permitted ESMA codes |
| 4 | 0 | complete → `awaiting_publication` |

Not one question names a lender, a file path or an internal identifier. Every
option list is the regulator's own vocabulary.

### The twelve required proofs

| # | Requirement | Result | Evidence |
|---|---|---|---|
| 1 | MI derives from the same governed canonical as the regime branch | **Yes** | `_run_annex2_chain` runs after `_apply_run_state` on `state.central_canonical_path`; both publication `source_artefacts.central_canonical` and the projection input are that path |
| 2 | Management values remain lender-native | **Yes** | The canonical carries all five lender statuses — `In possession`, `Live`, `Moved to LTC`, `Probate - awaiting sale`, `Redeemed` — unchanged, both periods |
| 3 | Regulatory translations do not rewrite the management canonical | **Yes** | `Live→PERF`, `Probate - awaiting sale→OTHR` were approved and applied; the canonical still reads the lender's words. Translations are merged into `effective_delivery_rules.yaml` at Gate 4b, never into client mapping memory |
| 4 | Real data wins over defaults / no-data treatments | **Yes** | `nd_injected: 0` — the builder inserted no no-data value of its own. Every ND code in the return is an operator's approved answer for a field the lender does not carry |
| 5 | Deterministic derivations used where inputs exist | **Yes** | `review_required_instances: 0`; RREL5 was asked rather than derived precisely because the lender's tape carries a candidate column |
| 6 | Client/deal facts come from the client's own effective configuration | **Yes** | `_run_annex2_chain` resolves `client_config` via `EffectiveConfigResolver.client_config_for(run.client_id)` and materialises `effective_client_config.yaml` per run |
| 7 | Asset defaults come from the correct asset pack | **Yes** | `product_defaults_ERM.yaml` resolved from the portfolio type; Gate 4 completed with the ERM product layer |
| 8 | Portfolio regulatory treatments come only from approved operator decisions | **Yes** | 42 ND treatments + 4 enum translations + 1 constant + 1 source column, each a recorded `decision_approved` audit entry with a `RuleRecord` |
| 9 | XML written by the production Gate 5 path | **Yes** | `stages.py::run_xml` runs `engine/gate_5_delivery/xml_builder_annex2.py` as a subprocess with the workbook, code order and XSD |
| 10 | Real auth.099 XSD validation passes | **Yes** | Builder reports `XSD Validation: PASSED`; **independently re-validated in this certification** with `lxml.etree.XMLSchema` against the repository XSD — valid, zero errors, both periods |
| 11 | The return carries the correct client's identity / LEI | **Yes** | Client B's LEI appears **31 times** (1 header + 30 records) |
| 12 | No other client's configuration or decisions observable in the output | **Yes** | The incumbent ERM UK LEI appears **0 times**; rule store returns nothing across the client boundary |

**Period 2 acceptance:** zero repeat questions where standing decisions existed
✓; valid MI ✓; valid XML ✓; XSD PASS ✓; no developer intervention ✓. The single
file-role question is a repeat and is reported as a P1 in section 8B.

---

## 7. Operator decision coverage

Every hold category OCC can surface in a normal reporting cycle, and whether an
operator can resolve it. "Refuses invalid" means the engine rejects a bad answer
*before* it is persisted, naming the permitted alternatives.

| Hold | Raised as | Exposed | Operator action | Refuses invalid | Persists at | Consumed on rerun | Reused next period | Changeable later | File edit needed |
|---|---|---|---|---|---|---|---|---|---|
| Unknown file role | `file_role` / `input_batch` | ✓ blocking, 6 role options | choose the role | ✓ `OPS_VALUE_REQUIRED` on empty | batch (`override_classification`) | ✓ batch becomes ready | **✗ re-asked (P1)** | n/a — asked again | No |
| Unknown source column mapping | `field_mapping` / `target_first_decisions` | ✓ | approve · amend · **point at a column** · not applicable · defer | ✓ `OPS_SOURCE_COLUMN_REQUIRED`, `OPS_SOURCE_FILE_REQUIRED` | portfolio (`RuleRecord`) | ✓ digest change forces Gate 1 redo | ✓ 0 questions in period 2 | **✗ no route (P1)** | No |
| Ambiguous loan identifier | `field_mapping` / `central_tape_gaps` | ✓ non-blocking, every candidate listed, recommendation shown | choose the column | ✓ option list constrained | portfolio | ✓ canonical re-keyed | ✓ | **✗ no route (P1)** | No |
| Missing client/deal fact | onboarding readiness `blocking`; Gate 4 `regulatory_config` backstop | ✓ | supply the value; readiness names `owner: operator` vs `owner: client` | ✓ `OPS_ONBOARDING_INCOMPLETE` before any config is written | client | ✓ | ✓ | ✓ amendment case | No |
| Missing mandatory regulatory value | `regulatory_source` | ✓ | point at a column, **or amend with a value** where no row carries it | ✓ | portfolio | ✓ | ✓ | **✗ no route (P1)** | No |
| Permitted ND treatment | `regulatory_no_data` | ✓, offering only that field's permitted codes | approve a code | ✓ **`OPS_TREATMENT_NOT_PERMITTED`** — "Permitted: ND5." | portfolio (`nd_defaults`) | ✓ | ✓ | **✗ no route (P1)** | No |
| Regulatory enum translation | `regulatory_enum_translation` | ✓, offering that field's 15 ESMA codes | approve a code | ✓ **`OPS_CODE_NOT_PERMITTED`** — lists all permitted | client | ✓ | ✓ | **✗ no route (P1)** | No |
| Pool/report-level regulatory constant | `regulatory_constant` | ✓ | record the value | ✓ | portfolio (`regulatory_constants`) | ✓ | ✓ | **✗ no route (P1)** | No |
| Deferred lender response | `defer` action on a target-first decision | ✓ available on every treatment question | defer with a reason | ✓ `OPS_REASON_REQUIRED`; item stays **open**, run stays held | decision `deferrals` list + audit | run stays held | n/a | ✓ answer it later | No |
| Publication approval | `publication` | ✓ | Publish / Hold | ✓ `OPS_PUBLICATION_NOT_PREPARED`, `OPS_ALREADY_PUBLISHED`, `OPS_REASON_REQUIRED` on hold | publication doc | n/a | n/a | n/a | No |
| Gate 3 validation exception | `validation_exception` / `validation_halt` | ✓ **but with no evidence at all** | proceed / stop | — no validity concept | file scope (**not admin-only**) | ✓ force-publishes | n/a | **✗ no route (P1)** | No |

### Dead-end questions

**None at first ask.** Every question OCC raises has an action that resolves it,
and each resolution is consumed on the next run. The previous sprint's dead end
— a regulatory field the lender does not carry, with no way to record a
permitted no-data code — is closed: 42 such questions were raised and answered
in Client B's first period, and reused silently in the second.

### Two structural weaknesses in the decision surface

**A dead end on the *second* ask.** Once a decision is approved it becomes a
standing rule and the question is never raised again — which is the point, and
what makes month two silent. But there is no governed way to reopen it. An
operator who answers wrongly has no route back, and the answer governs every
subsequent period. Runtime proof in section 8F. This is the certification's most
consequential finding, because its remedy today is a developer editing the rule
store.

**One over-permissive question.** The Gate 3 validation exception is the inverse
of a dead end: an action too powerful for the information it is presented with —
no evidence at all, available at a non-admin scope, and it carries a
core-contract violation into a published MI version. Detail in section 8C'.

---

## 8. Failure-mode results

All cases were executed on the production route (blob arrival → OCC intake →
gates), not by invoking a gate directly.

### A. Unknown client — **PASS, fails closed**

```
delivery for STRANGER, never onboarded
  workflow created           : False
  batch status               : blocked
  status_reason              : "Complete client onboarding and activate the client
                                configuration before processing this delivery."
  publication record         : None
  known clients              : []
then activate STRANGER through onboarding and re-deliver
  workflow created           : True
```

No fallback configuration, no other client's settings, no publication. The block
is about governance, not about the files — the same delivery runs once the
client is activated.

*Observation (P1, latent):* `EffectiveConfigResolver.client_config_for()`
returns the repository's incumbent client file for a client with no generated
configuration. On the certified route this is unreachable — `_client_is_activated`
refuses before the resolver is consulted, and an activated client always has its
own generated config. It remains a documented legacy-adoption path and the only
thing standing between it and a wrong LEI is that guard.

### B. Unknown file role — **PASS on first contact, P1 on recurrence**

```
Whatever_Lender_Calls_It.csv
  batch status               : review_required
  missing_input_roles        : ['loan_extract']
  workflow before answer     : ""            (nothing proceeded)
  questions raised           : 1, blocking
  title                      : "Identify the file 'Whatever_Lender_Calls_It.csv'"
  options                    : loan_extract, collateral_extract, cashflow_extract,
                               funder_pi_extract, pipeline_report, property_extract
  empty answer               : REFUSED — OPS_VALUE_REQUIRED "Choose what this file is."
after the operator answers
  batch status               : running
  missing_input_roles        : []
  run                        : awaiting_publication, all four gates done
NEXT PERIOD, same lender, same filename, same headers
  batch status               : review_required
  missing_input_roles        : ['loan_extract']
  workflow created without asking : False
  file-role questions re-asked    : 1
  registry file_role_schemas      : {}
  registry file_role_aliases      : {}
```

OCC asks, refuses an empty answer, persists the decision and proceeds — all
correct. It does **not** learn. `_promote_source` (engine.py:2441) calls
`approvals.write_pending` without `role_schemas` or `role_aliases`, so the
approval artefact carries `{}` and `approvals.promote` — which *does* learn them
when present — has nothing to record. Confirmed independently in the Client B
rehearsal (`file_role_schemas: {}` after a fully published period).

**Impact:** one extra operator click per reporting period for any lender whose
file name is outside the built-in vocabulary — which is every genuinely new
lender. Not a blocker; it is a business action an operator can always take.
**P1.**

### C. Required core data missing — **PASS**

| Case | Run status | Gates | Blocking validation field | Assembly | Publication | Platform blob |
|---|---|---|---|---|---|---|
| `BAL_OS` column removed | `needs_review` | onboard done, transform done, **validate halted** | `current_principal_balance` | waiting | `OPS_PUBLICATION_NOT_PREPARED` | not written |
| `STATUS_CD` present but blank | `needs_review` | onboard done, transform done, **validate halted** | `account_status` | waiting | `OPS_PUBLICATION_NOT_PREPARED` | not written |

Publication could not be forced through `approve_publication` in either case; no
partial publication occurred; the platform canonical was never written.

**Reason visibility is where this falls short.** The blocker shown to the
operator is the generic sentence *"Some figures did not pass Trakt's checks and
need your review."* The actual issue — `account_status`, CORE002, 30 rows — sits
in `43_validation_issues.csv` in staging and is **not** surfaced in the stage
result or the decision. Part 6C's "reason visible to OCC/operator" requirement
is therefore **not met**.

### C′. The validation exception can bypass the core canonical contract — **FINDING**

A targeted probe asked whether the operator override defeats criterion 7. It does.

```
account_status blank on all 30 rows  (CORE002)
  Gate 3                       : halted, blocking field account_status     ✓ correct
  question offered             : "Some checks did not pass. Do you want Trakt to
                                  continue and prepare the report anyway?"
  options                      : ['proceed', 'stop']
  decision evidence            : []          ← empty
  decision observed_values     : []          ← empty
  decision affected_record_count: 0          ← zero
  stage evidence               : []          ← empty
  scope                        : "file"  — NOT in ADMIN_ONLY_SCOPES, so any operator

operator approves "proceed"
  validate step                : force-marked done
  run status                   : awaiting_publication
  stage validation             : completed
  stage summary                : "All checks passed."      ← false
  blockers                     : []                        ← cleared
  canonical account_status     : 30 of 30 null
  PUBLISHED                    : v1, blob://processed-v2/platform/FORCE/latest/…
  published rows               : 30, account_status blank on all 30
  publication record marker    : none
```

Three separate weaknesses compound:

1. **The operator cannot see what they are accepting.** Evidence, observed
   values and affected record count are all empty. The one sentence offered does
   not distinguish a core-contract violation from a soft format warning.
2. **The stage record is rewritten to a false statement.** After the override the
   validation stage reads `"All checks passed."` with no blockers and no
   warnings. Anyone reviewing the run later — or the future dashboard — is told
   the opposite of the truth.
3. **The publication record carries no marker.** `configuration_versions`,
   `source_artefacts` and `agent_versions` are silent on the override.

The fact is not lost entirely: the hash-chained audit log records
`rule_persisted {kind: validation_exception, scope: file}` and
`decision_approved {value: proceed}` at sequences 138–139. Reconstruction is
possible; it is just not visible anywhere an operator would look.

**Classification: P1** by the fix policy's own definition — the process
completes, so it is not a P0; operational control and auditability are
materially weak, which is P1 exactly. Per Part 11 this session does not fix P1.
The correction would be narrow and is specified in section 14.

### D. Regulatory-only failure — **PASS**

Proven directly from Client B's first period rather than by construction. At
rounds 1–3 the regulatory branch was blocked while the management report stood
ready:

| Round | Run status | validation | assembly | projection | delivery_prep | xml_delivery | **publication** |
|---|---|---|---|---|---|---|---|
| 1 | needs_review | completed | completed | **blocked** | waiting | waiting | **ready** |
| 2 | needs_review | completed | completed | **blocked** | waiting | waiting | **ready** |
| 3 | needs_review | completed | completed | completed | **blocked** | waiting | **ready** |
| 4 | awaiting_publication | completed | completed | completed | completed | completed | ready |

MI remained valid and publishable throughout; the regulatory artefact was
withheld; the failure was explicit (`_park_regulatory` writes a blocked stage
result with named blockers and raises the resolving decisions); and once the
operator answered, the regulatory branch reran and completed. The reverse cannot
happen — the chain only runs after Gate 3 passed and the canonical was assembled.

### E. Invalid regulatory enum / treatment — **PASS**

```
no-data treatment  allocated_losses (RREL73)  ← "ND9"
  REFUSED  OPS_TREATMENT_NOT_PERMITTED
  "The regulator does not allow that treatment for this field. Permitted: ND5."
  decision status after refusal : open        (not silently consumed)

enum translation  account_status "In possession"  ← "NOT_A_CODE"
  REFUSED  OPS_CODE_NOT_PERMITTED
  "The regulator does not accept that code for this field. Permitted: ARRE, DADB,
   DFLT, DTCR, NDFT, OTHR, PERF, RARR, RDMD, REBR, REDF, REOT, RERE, RESS, RNAR."
  decision status after refusal : open

management canonical after both refusals
  ['In possession', 'Live', 'Moved to LTC', 'Probate - awaiting sale', 'Redeemed']
```

Refused before approval, not discovered at build time. Permitted alternatives
named in the error. Management canonical untouched.

### F. Rerun / idempotency — **PASS within a run; a gap across runs**

**Within a run — the decision digest.** This is the mechanism, and it works.
`OpsEngine._execute` compares `_approved_decisions_digest(run)` against
`run.onboarding_decisions_digest`; when they differ it sets `redo_onboarding`,
which resets **every** step (not only `onboard`) and `assemble` / `route` /
`project` to `pending` and clears `central_canonical_path`. Approved decisions
are applied *by* Gate 1, so its previous output is stale the moment they change,
and nothing downstream of its tape is reused.

Proven by the identity change in Client A's first period: Gate 1 initially keyed
the tape on the customer; after the operator chose `ACCT_REF` the digest changed,
Gate 1 was reproduced, and the canonical came back keyed on the account
(`100000`, 30 distinct ids). Stale output was not reused; `rerun_count` advanced
to 1; all four gates ran again; publication was offered once.

**Duplicate delivery — correctly a no-op.** Re-landing a byte-identical file for
a published period opens a successor pack (`..._v2`) and then declines to
register the file: `register_file` hashes the content, finds it already
registered, and records `duplicate_status: duplicate_ignored` in the audit
without mutating the completed run. No duplicate workflow, no second
publication, no version corruption.

**Across runs — the gap.** A decision that is already standing is never
re-raised, and there is no route to change it.

```
period 1  operator answers the loan-key question: CUST_ID   (the customer — wrong)
          canonical keyed on 9000, 9001
          published

period 2  same lender, different content (40 rows, new period)
          questions asked          : 0
          loan-key question re-asked: 0
          canonical keyed on       : 9000, 9001     ← the wrong key, silently reused
          published

standing rule  rule_e251e549a45d  v1  active
               payload {source_column: CUST_ID, canonical_field: loan_identifier}

routes to change it
  GET  /ops/rules                 read-only listing
  GET  /ops/rules/{id}/history    read-only history
  engine method to retire a rule  none      (hasattr → False)
  RuleStore.retire()              exists, called by nothing
  onboarding amendment case       amends case answers (client, entities,
                                  portfolios, sources, regime) — not rule records
  admin config layers             system | regime | asset — repository packages,
                                  not per-client mapping rules
```

All four candidate surfaces were checked. None can change an approved mapping,
enum translation, no-data treatment or loan-key decision. `rules.approve()`
supersedes by subject key and creates version n+1 — but only when the same
question is raised again, and it is not raised again precisely because the rule
answers it.

**Consequence.** An ordinary human error on a question OCC itself flags as
ambiguous — three plausible loan-identifier candidates, one correct — becomes
permanent, and every subsequent period inherits it. Undoing it requires editing
the persisted rule store, which is a developer intervention and the one class of
action this certification exists to exclude.

Part 6F's four requirements: the necessary upstream stage reruns ✓; stale output
is not silently reused ✓; duplicate publication / version corruption does not
occur ✓; output can be traced to the final approved decision set ✓ (publication
`rule_versions` records the exact rule id → version map used). The gap is not in
rerun mechanics but in the absence of a way to *change* what is rerun.

### G. Deferred lender response — **PASS**

```
target-first decision options : provide_source_mapping, configure_static_value,
                                confirm_default_or_nd, mark_not_applicable, defer
defer without a reason        : REFUSED — OPS_REASON_REQUIRED
                                "Please say what you are waiting for."
after deferring
  decision status             : open          (not resolved, not hidden)
  deferrals                   : [{by: Operator, at: 2026-08-30T15:26:50Z,
                                  reason: "waiting on the lender's data dictionary"}]
  audit                       : decision_deferred
  run status                  : needs_review
  still-open questions        : 41  (the deferred item among them)
  publication stage           : waiting
  forced publication          : REFUSED — OPS_PUBLICATION_NOT_PREPARED
```

Deferring records who is waiting for what, and cannot be used to make a blocking
question disappear: the item stays open, the run stays held, no rule is written,
and publication is refused.

---

## 9. What remains outside OCC

| Item | Class | Note |
|---|---|---|
| Azure Event Grid subscription, blob storage, App Service, Managed Identity, Entra auth | **1 — Expected external infrastructure** | Not a defect |
| Storage backend selection and connection strings | **1** | `decide_backend()` logs the decision at startup |
| Teams / email notification transport | **1** | OCC writes durable intent; delivery is a separate timer trigger by design |
| Publication approval (MI and regulatory) | **2 — Expected human business control** | Deliberate. Not a defect |
| Onboarding approval and activation | **2** | Two-person control: `approve` records the decision, `activate` performs it |
| Deferring a genuinely unresolved lender fact | **2** | The item stays open and the run stays held — deferring cannot quietly close a blocking question |
| Runtime logs, latency, error rates | **3 — Operating tooling gap** | Visible only in Azure. The OCC system dashboard is the next sprint; not built here |
| Health of the MI API and the React app | **3** | OCC exposes its own `/health` only |
| A run interrupted by an Azure worker recycle | **3 / 4** | See below |
| Investor PPTX | **4 — Product/orchestration gap, out of scope** | Not on the OCC route at all (section 3.3). Being finalised separately; explicitly not an acceptance blocker |
| Changing an approved decision | **4 — Product/orchestration gap** | Section 8F. The most consequential gap: its workaround today is a developer edit. P1 |
| Learning a file role into the source registry | **4 — Product/orchestration gap** | Section 8B. P1 |
| Surfacing Gate 3 validation issues on the exception decision | **4** | Section 8C'. P1 |

### The interrupted-run gap, stated precisely

`OpsEngine.start` executes a run on an **in-process daemon thread**, and
`occ_intake.handle_arrival` returns to the Azure Function as soon as the thread
is launched. If the Functions host recycles or scales in the worker mid-run, the
thread dies and the workflow is left `running` in the store with a live lease.

Recovery exists — `recover_on_startup` marks such runs `interrupted` and
`blocked` with the message *"This run was interrupted. Choose 'Run again' to
continue where it left off"* — but it runs **only in the OCC API's FastAPI
lifespan hook**, in a different deployed service. Until that App Service
restarts, the run shows as `running` indefinitely and nothing sweeps it.

The signal to detect it is already persisted: the lease document carries
`owner_pid` and `started_at` and is cleared on completion, so a lease older than
any plausible run is an unambiguous stuck-run marker. Recovery itself needs no
developer — the operator presses "Run again". **P1, and the most valuable single
input to the dashboard sprint.**

---

## 9.1 Reporting-cycle state model

Derived from the code as it behaves, not imposed on it. Three state machines run
in sequence and are persisted separately; the dashboard sprint will need all
three.

### Onboarding case — `operations_control/onboarding/case.py::TRANSITIONS`

```
                    ┌──────────────── withdrawn ◀── (from any non-terminal state)
                    │
draft ──▶ information_requested ──▶ awaiting_client ──▶ in_review
  │              │                        │                │
  │              └────────────────────────┴──────┬─────────┘
  │                                              ▼
  └──────────────────────────────────▶ ready_for_approval ──▶ approved ──▶ activated
                                              ▲                    │            (terminal)
                                              └── changes_required ◀┘
```

| State | Entry | Exit | Who acts | Persisted | After restart |
|---|---|---|---|---|---|
| `draft` | `start_new_client` / `start_migration` / `start_amendment` | any step saved, or submit | Operator | case doc + answers | Resumes; nothing written outside the case |
| `information_requested` → `awaiting_client` | a request is created and marked sent | a response is recorded and reviewed | Operator / client | request + response records | Resumes |
| `in_review` | submit for approval | readiness passes or changes needed | Operator | readiness snapshot | Resumes |
| `ready_for_approval` | readiness has no blocking problems | `approve` | Administrator | — | Resumes |
| `approved` | `approve` with a mandatory reason | `activate` | Administrator | `approved_by/at`, reason | Resumes. **Approval writes no configuration** |
| `activated` | `activate` | terminal | Administrator | generated client config, portfolio registry, client index, **source registry record**, version + content hash | Terminal; a change needs a new amendment case |

The two-step approve/activate split is the control: approval records the
decision, activation performs it, and activation is the only place active
configuration is created.

### Input batch — `operations_control/intake.py::assess`

```
receiving ──▶ classifying ──▶ ┌── review_required ──┐
                              ├── incomplete ───────┤──▶ ready ──▶ running ──▶ completed
                              ├── configuration_required ─┘                └─▶ failed
                              └── blocked  (client not activated — terminal for this pack)
```

| State | Entry | Exit | Who acts | Persisted | After restart |
|---|---|---|---|---|---|
| `receiving` | pack created, no files yet | first file registered | blob trigger / operator upload | batch doc | Resumes; idempotent on batch id |
| `blocked` | `_client_is_activated` is false | complete onboarding, redeliver | Administrator | `status_reason` + `delivery_blocked_client_not_activated` audit | Stays blocked — fail closed |
| `review_required` | a file's role is ambiguous, **or** an open blocking decision exists | operator identifies the file | Operator | file-role decision + `override_classification` | Resumes |
| `incomplete` | a required input role has not arrived | the missing file arrives | lender | `missing_input_roles` | Resumes |
| `configuration_required` | effective configuration resolves `BLOCKED` | configuration supplied | Administrator | `effective_config_status` | Resumes |
| `ready` | roles satisfied, config resolvable, no blocking decisions | `start_batch` (auto when `auto_start_when_ready`) | system | manifest, `idempotency_key` | Resumes; duplicate start suppressed by `workflow_id` |
| `running` / `completed` / `failed` | mirrors the workflow run | — | — | `workflow_id` | A late file opens a successor pack `_v2`; identical content is ignored as a duplicate |

### Workflow run — `operations_control/contracts.py::RUN_TRANSITIONS`

```
received ──▶ running ──┬──▶ needs_review ──┬──▶ running   (rerun)
                       │        │          ├──▶ published (MI, while the
                       │        │          │              regulatory branch is held)
                       │        │          └──▶ held
                       ├──▶ blocked ───────────▶ running   (rerun)
                       ├──▶ failed ────────────▶ running   (rerun)
                       └──▶ awaiting_publication ─┬──▶ published  (terminal)
                                                  ├──▶ held ──▶ awaiting_publication
                                                  └──▶ running
   any non-terminal ──────────────────────────────────▶ cancelled  (terminal)
```

| State | Entry | Exit | Who acts | Persisted | After restart / rerun |
|---|---|---|---|---|---|
| `received` | `create_workflow` | `start` | system | run doc, pinned `effective_config` | Rerunnable |
| `running` | `start` — an in-process **daemon thread** | orchestrator finishes | system | lease (`owner_pid`, `started_at`), `started` event | **The gap:** a worker recycle leaves it `running`; only the OCC API's startup hook marks it `interrupted` + `blocked` |
| `needs_review` | a stage returns `needs_review`, or `_park_regulatory` holds the regime branch with MI ready | every blocking decision answered → automatic rerun | Operator | decision docs, stage GARs | Resumes from orchestrator state; a changed decision digest reproduces Gate 1 and everything downstream |
| `blocked` | a stage returns `blocked`, or configuration is missing, or a run was interrupted | `rerun` | Operator | `blockers`, `interrupted` | Rerunnable |
| `awaiting_publication` | orchestrator `done` (MI) or the Annex 2 chain completed | `approve_publication` / `reject_publication` | Operator | publication doc `prepared` | Publication is idempotent — a second approval is refused |
| `held` | `reject_publication` with a mandatory reason | `rerun`, or approve later | Operator | publication `held` + reason | Returns to `awaiting_publication` |
| `published` | `approve_publication` | terminal | Operator | publication `published`, `published_artefacts`, promoted source record | Terminal. A corrected delivery opens a new pack version and a new publication version |
| `failed` | uncaught execution error | `rerun` | Operator | `execution_error` event | Rerunnable |
| `cancelled` | `cancel` with a reason | terminal | Operator | open decisions marked `superseded`, never approved or rejected | Terminal |

### Stage lattice inside a run

`received → understanding → mapping → validation → assembly →`
*(regime only)* `regulatory_config → projection → delivery_prep → xml_delivery →`
`publication`

Each carries its own status from `waiting · running · needs_review · blocked ·
ready · approved · rejected · completed`, and each writes a
`GovernedAgentResult` with a history copy, so a stage's earlier verdicts survive
a rerun. The regime stages are skipped entirely for an `mi` outcome —
`WorkflowRun.applicable_stages` filters them — which is why an MI run has six
stages and a regime run ten.

The one asymmetry worth carrying into the dashboard: a run held on the
**regulatory** branch sits at `needs_review` while its publication stage is
`ready`. Run status alone therefore does not tell an operator whether MI can be
published; the publication stage does. `approve_publication` checks the stage,
not the status, for exactly this reason.

---

## 10. OCC system-dashboard observability inventory

Classification: **A** already persisted and directly available · **B** in logs
but not structured · **C** not currently captured · **D** requires independent
Azure monitoring.

### System health

| Item | Class | Where |
|---|---|---|
| OCC API health | **A** | `GET /health` → `{ok, storage_ok, auth_configured}` |
| MI API health | **C** | OCC has no probe for it |
| Dashboard / React app health | **D** | Static Web App; Azure-side |
| Last successful health check | **C** | `/health` is a live probe; no result is stored |
| Latency | **D** | Azure Application Insights |
| Availability / error condition | **D** | Azure |
| Storage backend actually selected | **B** | `decide_backend()` logged once at startup |

### Reporting runs

| Item | Class | Where |
|---|---|---|
| Client, portfolio, period | **A** | `WorkflowRun.client_id / portfolio_id / reporting_period` |
| Run ID | **A** | `workflow_id`, plus `orchestrator_run_id` |
| Current stage | **A** | `WorkflowRun.stages` — status + `result_id` per stage |
| Current status | **A** | `WorkflowRun.status`, and the batch's own status |
| Start / end / duration | **A** | Per-workflow event stream: `started` / `execution_finished` pairs with ISO timestamps (10 events across Client B's first period) |
| Previous successful period | **A (stale)** | `SourceRecord.last_successful_reporting_period` — **but written only by `_promote_source`, which runs for `new_client`/`new_portfolio` only.** It froze at 2025-11-30 for Client A after period 2 published. **P1 for the dashboard.** |
| Retry / rerun state | **A** | `rerun_count`, `interrupted`, `workflow_rerun` audit events |
| Stuck-run detection | **A** | Lease `owner_pid` + `started_at`, cleared on completion — no sweeper reads it yet |
| Stage-level duration | **B** | `GovernedAgentResult.started_at` / `completed_at` exist but are stamped at the same instant for OCC-owned stages |

### Exceptions

| Item | Class | Where |
|---|---|---|
| Blocking issue | **A** | `WorkflowRun.blockers`, `GovernedAgentResult.blockers` |
| Gate / source | **A** | `GovernedAgentResult.stage` |
| Timestamp | **A** | GAR `completed_at`, audit `at` |
| Operator action required | **A** | `decisions_required` on the GAR; `open_decisions` |
| Last failure | **A** | `execution_error` event, `RUN_FAILED` |
| Retry outcome | **A** | `rerun_count` + subsequent `execution_finished` |
| **Which validation rule actually failed** | **C** | `43_validation_issues.csv` is written to staging but never lifted into a GAR or decision (section 8C') |

### OCC questions

| Item | Class | Where |
|---|---|---|
| Open / answered / deferred | **A** | Decision docs: `open`, `approved`, `rejected`, `superseded` |
| Category | **A** | `kind` + `subject.artefact` — 11 distinct artefact types |
| Scope | **A** | `resolution_scope`, `allowed_scopes` |
| Age | **A** | decision `created_at` / `resolved_at` |
| Operator | **A** | `resolved_by`, and the audit `actor` |
| Deferral reason | **A** | `resolution_reason` |

### MI Query usage

| Item | Class |
|---|---|
| Client, user, timestamp | **C** — OCC does not observe MI Query at all |
| Natural-language question | **C** |
| Interpretation / route | **C** |
| Answer / refusal / error classification | **C** |
| Latency | **C / D** |
| Queried MI / data version | **A (derivable)** — the publication record names the exact `platform_canonical_typed.csv` version MI Query reads; joining a query to it needs instrumentation on the MI side |

MI Query usage telemetry is the single largest **C** block. It is not an OCC
regression — OCC was never asked to observe it — but the dashboard cannot show
it without new instrumentation in the MI API.

### Publication

| Item | Class | Where |
|---|---|---|
| Canonical version | **A** | `publication.source_artefacts.central_canonical` + `orchestrator_run_id` |
| MI version | **A** | `publication.version`, `previous_publication_id` |
| Regulatory projection | **A** | `publication.annex2.projected_csv` / `delivery_ready_csv` |
| XML | **A** | `publication.annex2.xml` + `xml_sha256` |
| XSD status | **A** | `publication.annex2.xsd` (name + size + mtime fingerprint) and `xsd_result` |
| Approval state | **A** | `prepared` → `approved` → `published`, with `approved_by` / `approved_at` / `published_at` |
| Published artefact locations | **A** | `published_artefacts.latest` / `.period` / `.regime` |
| Future PPTX | **C** | Not on the OCC route |
| **Effective configuration used, on the publication record** | **B** | `WorkflowRun.effective_config` carries id, version, content hash and package versions; the publication's own `configuration_versions.client_config` reads the literal string **`"repository default"`** for every MI run — misleading. **P2.** |
| **Force-publish marker** | **C** | section 8C' |

### Audit

| Item | Class | Where |
|---|---|---|
| Who approved what, when | **A** | Hash-chained audit log — 23 distinct event types observed across one regime cycle, each with `actor`, `at`, `prev_hash`, `record_hash` |
| Which decision set / config version was used | **A** | `effective_config_resolved` audit event with `effective_config_id`, `version`, `content_hash`, `decision_set_version`; also pinned on the run |
| Tamper evidence | **A** | `prev_hash` / `record_hash` chain |

**Summary for the dashboard sprint:** run state, stage state, exceptions,
questions, publication and audit are all class **A** — the dashboard is largely
a reader over existing persisted state. Four things need work first: MI Query
usage instrumentation (C), an MI API health probe (C), lifting the validation
issue detail into the exception record (C), and fixing
`last_successful_reporting_period` so it advances on recurring runs (A-stale).

---

## 11. Developer-intervention test

Every manual action taken across both rehearsals, in order.

### Client A (MI only)

| # | Action | Class | Surface |
|---|---|---|---|
| 1 | Answer onboarding steps: client, entities, contacts, portfolios, reporting, sources | Business | `PUT /ops/onboarding/cases/{id}` |
| 2 | Approve the onboarding case | Control | `POST .../approve` |
| 3 | Activate the client | Control | `POST .../activate` |
| 4 | Choose the loan identifier (`ACCT_REF` of three candidates) | Business | `POST /ops/reviews/{id}/decision` |
| 5 | Answer the mapping queue — 12 columns named, 3 static values, 25 not applicable | Business | `POST /ops/reviews/{id}/decision` |
| 6 | Approve publication, period 1 | Control | `POST /ops/workflows/{id}/publish` |
| 7 | Approve publication, period 2 | Control | `POST /ops/workflows/{id}/publish` |

### Client B (MI + Annex 2)

| # | Action | Class | Surface |
|---|---|---|---|
| 1 | Answer onboarding steps, including the regime step | Business | `PUT /ops/onboarding/cases/{id}` |
| 2 | Approve the onboarding case | Control | `POST .../approve` |
| 3 | Activate the client | Control | `POST .../activate` |
| 4 | Identify `PolicyExtract.csv` as the loan extract, period 1 | Business | `POST /ops/reviews/{id}/decision` |
| 5 | Point RREL5 at the `Borrower Ref` column | Business | `POST /ops/reviews/{id}/decision` |
| 6 | Record the pool addition date | Business | `POST /ops/reviews/{id}/decision` |
| 7 | Approve 42 permitted no-data treatments | Business | `POST /ops/reviews/{id}/decision` |
| 8 | Approve 4 enum translations from ESMA's own code list | Business | `POST /ops/reviews/{id}/decision` |
| 9 | Approve publication, period 1 | Control | `POST /ops/workflows/{id}/publish` |
| 10 | Identify `PolicyExtract.csv` again, period 2 (**repeat — P1, section 8B**) | Business | `POST /ops/reviews/{id}/decision` |
| 11 | Approve publication, period 2 | Control | `POST /ops/workflows/{id}/publish` |

**Developer-only actions: 0.**

No source code was edited. No generated configuration was edited. No canonical
tape was edited. No XML was edited. No repair script was run. No persisted
decision was modified behind OCC's back. No lender-specific alias or config was
hard-coded. No XSD or validation rule was changed to obtain a pass. Every one of
the 18 actions above went through the OCC decision, onboarding or publication
API, and each maps to a route a React operator screen already calls.

---

## 12. Acceptance table

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | OCC can activate a governed new client/portfolio without bespoke Python | **YES** | Both lenders onboarded through the case flow only; every value an operator answer |
| 2 | OCC can receive/classify a production-style raw delivery | **YES** | Event Grid path → `occ_intake.handle_arrival`; both lenders' tapes registered and classified |
| 3 | OCC selects MI vs MI+regime from approved configuration | **YES** | Client A → `mi`, Client B → `mi_annex2`, from `regime_required` on the source record set at activation |
| 4 | Every reporting run passes through Gates 1–3 | **YES** | `onboard/transform/validate/stamp` all `done` on all four published runs |
| 5 | Every legitimate Gate 1 blocking issue is actionable by the operator | **YES** | 41 questions on Client A's first contact, each with executable options; no dead end at first ask. Reopening an answered one is a separate gap — criterion 19 |
| 6 | Approved mapping decisions persist and are consumed on rerun | **YES** | `onboarding_decisions_digest` forces a Gate 1 redo; canonical re-keyed after the loan-key answer. They persist permanently — see criterion 19 and section 8F |
| 7 | An economically incomplete canonical cannot reach MI | **NO — qualified** | Absolute on the default path (both C cases blocked, publication refused, no blob written). Defeated by the operator's blind validation exception: 30 blank `account_status` rows published. section 8C'. **P1** |
| 8 | MI publication can be completed without developer intervention | **YES** | Both clients, all four periods, published through `POST /ops/workflows/{id}/publish` |
| 9 | MI Query can query the published governed data | **YES** | `platform/ALPHA/latest` + both period blobs exist; 30 rows, 37 columns read back and aggregated |
| 10 | A regime-required run uses the same governed canonical | **YES** | `_run_annex2_chain` projects from `state.central_canonical_path`, the same path the publication record names |
| 11 | Every legitimate regulatory blocker is actionable, or explicitly held for lender information | **YES** | 48 questions, all answerable; deferral keeps the item open and the run held |
| 12 | Gate 4/4b/5 complete without developer intervention after legitimate decisions | **YES** | All three stages `completed`, both periods, after operator answers only |
| 13 | Real XML is produced and real XSD validation passes | **YES** | 30 records × 107 fields, `xsd_result: PASSED`, **independently re-validated with `lxml` here** — valid, 0 errors, both periods |
| 14 | A regulatory failure does not corrupt/withdraw otherwise valid MI | **YES** | Rounds 1–3: projection/delivery_prep blocked while `publication: ready`; MI publishable throughout |
| 15 | Client configuration/rules/identity are isolated | **YES** | Client B's LEI 31×, incumbent LEI 0×; rule store returns nothing across the boundary; per-client generated configs differ |
| 16 | Unknown clients fail closed | **YES** | Batch `blocked`, no workflow, no publication, no fallback config |
| 17 | The next reporting period reuses standing decisions automatically | **YES — one exception** | 0 mapping/regulatory questions in period 2 for both clients; the file-role question is re-asked (section 8B, P1) |
| 18 | No developer edits generated files/state to make the second period work | **YES** | Period 2 required publication approval only (Client A) or file identification + approval (Client B) |
| 19 | Reruns after decision changes consume the new decision set | **YES — within a run** | The decision digest resets every step and clears the canonical path; Client A's tape was re-keyed from the customer to the account. Across runs the criterion cannot be fully exercised: a standing decision cannot be changed at all (section 8F) |
| 20 | OCC has a traceable state/output record sufficient to reconstruct the cycle | **YES — qualified** | Hash-chained audit (23 event types), per-run event stream, per-stage GARs with history, pinned effective config, versioned publication records. Two blemishes: the validation stage reads "All checks passed." after an override (section 8C') and `configuration_versions.client_config` reads `"repository default"` on MI publications (section 10) |

**16 YES · 3 YES with a stated qualification · 1 NO.**

**Criteria that could not be demonstrated because the capability is
intentionally external/manual:** none. Publication approval is deliberately
manual and was exercised as an operator control, which is the intended
managed-service operating model, not a gap.

**The one NO.** Criterion 7 holds absolutely on the default path — a book missing
its balance column and a book with a blank status column both stopped at Gate 3,
publication was refused with `OPS_PUBLICATION_NOT_PREPARED`, and no platform
blob was written. It is defeated only by an explicit operator action. That action
is a governed business control rather than a developer intervention, so the
session's central proposition is unaffected; but the criterion as written is a
hard guarantee, and the guarantee does not hold.

---

## 13. Regression

No production code was changed, so per Part 13 the sweep is scoped to the OCC
and go-live acceptance suites and the production-route tests, not the whole
repository. Every run is at the starting `main` SHA on a clean tree.

| Suite group | Modules | Passed | Failed | Skipped |
|---|---|---|---|---|
| `tests/operations_control/` — OCC engine, intake, onboarding, publication, rules, tenancy, recovery, workflow, OCC Agent | 39 | **1101** | 0 | 0 |
| `tests/test_occ_go_live_e2e.py` — the go-live acceptance suite | 1 | **27** | 0 | 0 |
| Annex 2 production route | 13 | **331** | 1 | 1 |
| Gates, orchestrator, assembler, client isolation, tenancy, approval policy | 14 | **310** | 2 | 1 |
| MI Query Agent / analytics acceptance | 15 | **451** | 2 | 2 |
| **Total** | **82** | **2220** | **5** | **4** |

### Every failure compared against a pre-change baseline

Part 13 forbids dismissing a failure as unrelated without comparing it to the
starting `main` SHA. This certification changed nothing, so every failure is
pre-existing on `main` by construction — which is a weak statement. The stronger
comparison was made instead: each was re-run in a clean worktree at
**`fce4bd7`**, the commit immediately *before* the previous Annex 2 sprint.

| Failing test | At `main` `e7678c8` | At `fce4bd7` | Verdict |
|---|---|---|---|
| `test_transformation_agent_workflow.py::TestTransformationRun::test_numeric_normalised` | FAIL | **FAIL** | Long-standing; predates the previous sprint |
| `test_transformation_agent_workflow.py::TestTransformationRun::test_readiness_flags_distinct` | FAIL | **FAIL** | Long-standing; predates the previous sprint |
| `test_phase8b_anthropic_interpreter_adapter.py::test_golden_valid_via_fake_anthropic[compare funded balance to last month]` | FAIL | **FAIL** | Long-standing; MI interpreter surface |
| `test_analytical_capability_layer.py::TestSecondBookAcceptance::test_q7_compares_the_two_governed_sides_and_reconciles` | FAIL | **FAIL** | Long-standing; MI analytics surface |
| `test_annex2_field_xsd_path_map.py::TestGeneratorReproducible::test_regeneration_matches_committed` | FAIL | **PASS (15/15)** | **Regression introduced by the previous sprint** |

**The one real regression.** `config/delivery/annex2_field_xsd_path_map.yaml` no
longer matches what `scripts/build_annex2_field_xsd_path_map.py` produces — the
`mapping_status` values drifted when commit `1f0be01` re-pointed the analysis
scripts at the derived contract, and the committed artefact was not regenerated.
The previous sprint's 31-suite Annex 2 sweep did not catch it.

It is **not on the OCC production route**. The map is read only by
`engine/delivery_xml_agent` (the XSD *preview* surface,
`production_xsd_mapping_configured: false`, all production gates asserted false)
and by four analysis scripts. OCC's Gate 5 runs `xml_builder_annex2.py` directly
with the mapping workbook, the code-order file and the XSD, and produced a
schema-valid return twice in this certification. **P2** — regenerate the
committed artefact; not fixed here under the P0-only fix policy.

**What the two Gate 2 failures actually pin.** The fixture declares two contract
rows targeting `current_principal_balance`: one source-mapped from the lender's
principal column (`"177,334.06"`), one `semantic_derivation_required` from
`current_outstanding_balance` (`"180000.00"`), commented *"must not alias"*.
Gate 2 lets the second overwrite the first, so the principal balance becomes the
outstanding balance. Harmless in UK equity release, where the two are equivalent
— which is exactly the R-2 balance-vocabulary risk the previous sprint recorded
as post-go-live hardening, and it would not stay harmless for a book that
services interest. Out of scope here, still current, and unchanged by this
certification. **P2 for the current mono-product scope; revisit before any
interest-servicing book.**

### A note on environment, recorded rather than buried

An earlier run of the MI group reported 5 failures and 23 errors. Twenty-three
errors and three of the failures were an artefact of this container
(`ModuleNotFoundError: _cffi_backend`); after installing `cffi` the group settles
at 451 / 2 / 2. Recorded because dismissing them undiagnosed would have been
precisely the mistake Part 13 warns against.

**MI Query Agent behaviour did not change as a side effect** — no MI Query Agent
code, dashboard calculation or capability was touched, because no production
code was touched at all.

---

## 14. Pre-go-live recommendation

> **Is there anything in OCC orchestration that should prevent us from onboarding
> Client 1 and commencing the managed-service production acceptance checks?**

# NO.

Nothing in OCC orchestration is a blocker. If Client 1 were activated today and
its reporting file arrived tomorrow, the entire reporting process could be
operated through OCC as a managed service, with human involvement only where a
business or control decision is genuinely required, and completed without a
developer. That was executed twice here, for two reporting periods each, once
with a real Annex 2 return that passes the real auth.099 schema.

The honest qualifier: that is a first-pass-correct claim. If an operator answers
a mapping question wrongly, there is today no way back without a developer. That
is a real risk to run during the acceptance checks, and it is the first thing to
close — but it is not a reason to delay starting them.

### Conditions to carry into the acceptance checks

None of these is a blocker. They are the things to fix, or to operate around,
in order.

**1. No route to change an approved decision (P1, close first).**
This is the one that turns an ordinary operator mistake into a developer job.
Until it is closed, operate Client 1 under a four-eyes rule on the first
period's review queue — in particular the loan-identifier question, where OCC
itself says the answer is ambiguous — because a wrong answer there silently
re-keys every subsequent period and nothing will ask again.

The correction is architectural rather than narrow, which is why it is reported
here rather than implemented: it needs an engine method over the existing
`RuleStore.retire()` / `approve()` supersession, a governed route to reach it, a
rule for what happens to runs already published under the superseded version,
and a way to force the affected question back into the review queue. Part 11
directs that a potential P0 requiring architectural expansion is reported before
implementation. **Reporting it, not implementing it, is the deliberate choice
here.**

**2. The blind Gate 3 validation exception (P1).**
Until fixed, operate under a standing instruction that "Continue — I accept the
flagged items" is not used, and that a Gate 3 halt is escalated rather than
overridden. This correction *is* narrow:

* attach the existing `43_validation_issues.csv` content to the decision as
  evidence — failing field, rule id, severity, affected row count — so the
  operator can see what they are accepting;
* separate a core-canonical violation (CORE001/CORE002) from a soft check, and
  either withhold the override for the former or require an admin scope;
* stop rewriting the validation stage to `"All checks passed."` after an
  override — carry the force-publish forward as a warning on the stage and a
  marker on the publication record.

Not fixed here because the fix policy scopes automatic correction to P0 defects,
and this is P1 by that policy's own definition.

**3. The file-role answer is not learned (P1).**
Expect one extra operator click per reporting period for Client 1 if its file
name is outside the built-in vocabulary. The fix is a single call site:
`_promote_source` should pass the pack's `role_schemas` and `role_aliases` to
`approvals.write_pending`, which `approvals.promote` already knows how to record.

**4. `last_successful_reporting_period` freezes after the first period (P1).**
Only `_promote_source` writes it, and that runs only for `new_client` /
`new_portfolio` workflows. The dashboard sprint should not build "previous
successful period" on this field until it advances on recurring runs.

**5. No sweeper for an interrupted run (P1).**
A run whose Azure worker recycles stays `running` until the OCC API restarts.
Recovery needs no developer once noticed — the operator presses "Run again" —
but nothing surfaces it. The lease document (`owner_pid`, `started_at`) is
already the signal; a stale-lease sweep belongs in the dashboard sprint.

**6. Three misleading or stale records (P2).**
`configuration_versions.client_config` reads `"repository default"` on every MI
publication although the run used the client's own generated configuration;
`config/delivery/annex2_field_xsd_path_map.yaml` is stale against its generator
(section 13); and `.funcignore` still describes the root `function_app.py` as a
shim re-exporting the legacy router, which it is not.

### Findings by priority

| Priority | Finding | Fixed here |
|---|---|---|
| **P0** | *(none — no finding prevents a commercial reporting cycle in the intended current scope)* | — |
| **P1** | No operator route to change an approved decision; a wrong answer is permanent and correcting it requires editing the rule store | No — architectural; Part 11 directs reporting, not implementing |
| **P1** | Gate 3 validation exception: no evidence shown, non-admin scope, bypasses the core canonical contract, and rewrites the stage to "All checks passed." | No — P1, per the fix policy |
| **P1** | File-role decision never promoted to the source registry; re-asked every period | No — P1 |
| **P1** | `last_successful_reporting_period` written only on onboarding runs, so it freezes after period 1 | No — P1 |
| **P1** | No sweeper for a run interrupted by a worker recycle; it shows as `running` until the OCC API restarts | No — P1 |
| **P1** | `EffectiveConfigResolver.client_config_for()` falls back to the incumbent client file (unreachable on the certified route; the activation guard is the only barrier) | No — P1, latent |
| **P2** | `annex2_field_xsd_path_map.yaml` stale against its generator — a regression from the previous sprint, on the preview surface only | No — pre-existing |
| **P2** | Gate 2 lets a `semantic_derivation_required` mapping overwrite an already source-mapped canonical field (the R-2 balance vocabulary) | No — pre-existing, long-standing |
| **P2** | `configuration_versions.client_config: "repository default"` on MI publications | No |
| **P2** | Two long-standing MI interpreter / analytics failures | No — pre-existing |
| **P2** | `enrichment.uk_nuts3` not generated by OCC; incumbent branding read unconditionally by the ERM Streamlit app | No — pre-existing |
| **P2** | Stale `.funcignore` comment describing the deployed entrypoint | No |

---

*Certification performed against `main` at `e7678c81100f562c25cb39cf1cbf69798e13a5ed`
on branch `claude/occ-production-certification-axdj1z`. No production code was
changed.*
