# OCC Governance Follow-Up — Intent vs Implementation

Follow-up to `occ_production_certification.md` (commit `a88b53f`, branch
`claude/occ-production-certification-axdj1z`, `main` at `e7678c8`).

Purpose: distinguish **intentional governed operating controls** from
**accidental widening**, **incomplete implementation**, **latent defects** and
**genuine go-live blockers**. No production code changed.

---

## 0. Headline

Two of the six P1 findings turn out to be *less* serious than certified, one
turns out to be *more*, and none is a P0.

| | |
|---|---|
| **The Gate 3 exception is a designed control.** | Its file/reporting-period containment **works** — proven at runtime, the exception does not carry to the next period. What is defective is that a 2026-08-29 structural rule inherited a 2026-08-04 generic route, and that the control's own designed evidence requirements were never implemented. |
| **Correcting a wrong decision is not new architecture.** | `retire()` + `approve()` — both already implemented **and tested** — fully correct a wrong mapping. Proven end to end: period 2 asked 0 questions and produced the corrected canonical. What is missing is a service method and a route. |
| **The interrupted-run finding is worse than certified.** | Recovery marks the run resumable and tells the operator "Nothing has been lost. You can try running this step again." Proven: after the recycle that triggers recovery, "Run again" **fails**, because the delivery's input files lived in the same ephemeral `/tmp` staging. The message is false and the offered affordance does not work. |

**Recommended verdict change: PASS WITH OPERATING CONDITIONS → PASS WITH
PRE-GO-LIVE HARDENING.** The conditions are no longer open-ended things to
operate around; they are a small, bounded change set that is almost entirely
the exposure and enforcement of mechanisms that already exist.

---

## 1. Gate 3 validation exception — original intent

### 1.1 When it was introduced

The repository's history is a series of bulk snapshots until late August 2026,
then granular commits. `KIND_VALIDATION_EXCEPTION` is present in the **first
snapshot that contains OCC at all** — `3f8628a` (2026-08-04),
`operations_control/contracts.py:138`. No incremental commit introduced it: it
arrived as part of the original OCC design, three weeks before the core-canonical
contract.

### 1.2 What it was intended to solve, and what an exception was meant to be

Three independent sources agree, and none of them describes "ignore any Gate 3
failure".

**`docs/operations_control_centre/08_operational_rule_model.md` §3** — the rule
model's own payload specification:

| Kind | Payload (essence) | Projection target |
|---|---|---|
| `exception` | `{rule_id_ref, condition, disposition, justification}` | Validation exception artefacts / remediation ledger, re-read on rerun |

An exception was designed to name **which rule** it excepts (`rule_id_ref`),
**under what condition**, and **why** (`justification`). It is a per-rule
instrument, not a per-run switch.

**§2, scope semantics** — the canonical example of a file-scoped rule:

> **Current file** — one-off; applies to this delivery only (e.g. *a known bad
> value in one month's tape*). Never auto-applied to future files.

**The legacy exception ledger** (`exception_db.py`, `exception_queue.py`), which
doc 06 names as the Phase-2 read source for validation exceptions, is the most
explicit evidence of intent. Its schema:

```sql
CREATE TABLE findings (
    id, snapshot_id, rule_id, severity, field_name, row_index,
    message, classification, materiality, status DEFAULT 'open', created_at);

CREATE TABLE remediations (
    id, finding_id  REFERENCES findings(id),   -- PER FINDING, not per run
    action           TEXT NOT NULL,            -- accept | override | escalate
    field_name, row_index,
    original_value, override_value,            -- the corrected value
    rule_id          TEXT NOT NULL,
    justification    TEXT NOT NULL,            -- MANDATORY
    user_id          TEXT NOT NULL,
    user_name        TEXT NOT NULL,
    created_at, record_hash, prev_hash);       -- hash-chained
```

with `MATERIALITY_COLOURS = {"BLOCKING", "REVIEW", "INFO"}` and finding statuses
`open | accepted | overridden | escalated`. The UI refuses an override with no
value: *"Override value is required when action is 'override'."*

### 1.3 The eight questions, answered

| # | Question | Answer |
|---|---|---|
| 1 | When was `validation_exception` introduced? | With OCC itself, snapshot `3f8628a`, 2026-08-04. Never an incremental addition. |
| 2 | What problem was it intended to solve? | A **known, identified data-quality finding in one delivery** that the business accepts with a stated reason, so one month's tape is not held hostage to a value the lender cannot restate. |
| 3 | What was intended to be overrideable? | Findings in the legacy ledger's sense: a named `rule_id` against a named `field_name`, with a `materiality`. `accept` / `override` (with a corrected value) / `escalate`. Nothing in any source contemplates a field being structurally absent or wholly empty. |
| 4 | Intentionally file / reporting-period scoped? | **Yes**, explicitly — doc 08 §2, and `_persist_rule` forces `scope = "file"` regardless of what the caller passes, with the comment *"validation exceptions never generalise silently"*. |
| 5 | Intentionally non-persistent as a standing rule? | **Yes in effect.** The rule record persists for audit, but the force-publish flag is read by `_validation_exception_approved`, which scans decisions **for this `workflow_id` only** — so it cannot reach another period. Proven in §1.5. |
| 6 | What evidence / rationale / authority was originally required? | `justification NOT NULL`, `rule_id`, `field_name`, `row_index`, `original_value` / `override_value`, `user_id` + `user_name`, hash-chained. |
| 7 | "Accept a known business-rule exception" or "ignore any Gate 3 failure"? | Unambiguously the former. |
| 8 | Which tests historically defined the behaviour? | Exactly one: `tests/operations_control/test_workflow_engine.py::TestValidationException::test_validation_halt_needs_approval_then_force_publishes`. Its own reason string is **`"known data issue this month"`** and it asserts *"The exception rule is file-scoped — it never generalises silently."* |

### 1.4 What the implementation actually does

```python
elif kind == KIND_VALIDATION_EXCEPTION:
    payload = {"check": subject.get("artefact", "validation"),   # "validation_halt"
               "disposition": value,                             # "proceed"
               "justification": reason}                          # not required
    desc  = "Accepted flagged checks for this delivery."
    scope = "file"
```

The rule persisted by the certification's runtime probe:

```json
{ "kind": "validation_exception", "scope": "file",
  "file_ref": "sha256:feef829b…",
  "payload": { "check": "validation_halt", "disposition": "proceed",
               "justification": "" },
  "approved_by": "Operator", "reason": "" }
```

| Designed | Implemented |
|---|---|
| `rule_id_ref` — which rule is excepted | `check: "validation_halt"` — a constant |
| `condition` — under what condition | absent |
| `justification` **NOT NULL** | `""` — accepted, never required |
| per **finding** (`finding_id` FK) | per **run** |
| `original_value` / `override_value` | absent |
| `accept` / `override` / `escalate` | one action: `proceed` |
| `user_id` + `user_name`, `role` | `approved_by` only; `file` scope is **not** in `ADMIN_ONLY_SCOPES = (asset, global)`, so any operator with client access qualifies |

### 1.5 What is genuinely correct — proven, and it corrects the certification

The certification did not test whether the exception leaks into the next period.
It does not.

```
period 1   account_status blank on all 30 rows, operator approves "proceed"
           → awaiting_publication, validation "completed", published

period 2   SAME schema fingerprint (sha256:feef829b… — identical), SAME defect,
           no operator action at all
           → run status            needs_review
           → validation            needs_review
           → publication           waiting
           → exception RE-ASKED    1
           → approve_publication   REFUSED  OPS_PUBLICATION_NOT_PREPARED
```

The design's promise — *"one-off; applies to this delivery only … never
auto-applied to future files"* — **holds**. It holds because
`_validation_exception_approved` keys on `workflow_id`, not because of the rule
scope: the file-scoped rule record *is* still returned by `applicable()` for
period 2 (file scope keys on the schema fingerprint, which is stable across
periods), it simply has no consumer for this purpose. The containment is real;
the rule record is decorative for it.

---

## 2. CORE001 / CORE002 chronology

| # | Question | Answer |
|---|---|---|
| 1 | Which commit introduced them? | The **checks** predate OCC — `validate_canonical.py` has carried CORE001/CORE002 since the bulk snapshots. What is new is their **enforcement in the agentic chain**: `a43f0ca`, *"Enforce the canonical core-field contract at Gate 3"*, 2026-08-29, the first commit of the go-live sprint. |
| 2 | Why? | Its own message: *"the Validation Agent never called that enforcement … so the agentic chain could report `ready_for_validation_complete` on a canonical with no balance at all. The definition was authoritative; nothing consulted it."* |
| 3 | What invariant? | That a canonical cannot reach `ready_for_validation_complete` without the economically essential fields. `rules_adapter.validate_core_canonical_presence` docstring: *"An `error` violation is blocking for validation, which is what makes readiness fail."* |
| 4 | Did it consider the existing exception route? | **No.** The commit message is long and careful — it discusses readiness flags, applicability relaxations, two defects in `is_blank` and `validate_core_presence`, the balance equivalence, and `originator_*` applicability. It does not mention `validation_exception`, `force_publish`, or the Review Centre once. The reasoning stops at *"readiness fails"*. |
| 5 | Tests added? | For core-field failure: yes (fixtures moved to carry the full core set). For publication blocking: yes, later, in `test_occ_go_live_e2e.py`. **For exception / override interaction: none.** `a43f0ca` touched two test files and added no exception test; no test anywhere asserts what an override does to a CORE failure. |
| 6 | Did CORE inherit the generic machinery? | **Yes, by construction.** `validate_core_canonical_presence` emits `_result(..., b_val=True)` → `blocking_for_validation` → `ready_for_validation_complete` False → the orchestrator adapter returns `blocking=True` → `STEP_HALTED` → OCC raises the single generic `validation_halt` decision. Every Gate 3 blocking finding, structural or not, arrives at the operator through one undifferentiated question. |

> **Were structural failures intentionally made overrideable?**
> **No.** The exception route is three weeks older than the core-canonical
> enforcement. The commit that created the enforcement reasoned only to readiness
> and never mentioned the override. No test, design document or policy statement
> anywhere contemplates accepting a structurally absent or wholly empty core
> field. CORE001/CORE002 fell inside an older generic mechanism because both are
> expressed as `blocking_for_validation`.

---

## 3. The actual Gate 3 failure taxonomy

Derived from the code, using the repository's own terminology
(`engine/validation_agent/validation_agent.py`). There is no BLOCKING/REVIEW/INFO
taxonomy in the live path — that vocabulary is the legacy ledger's. The live one
is `validation_classification` × `downstream_owner` × three independent blocking
flags.

**`validation_classification`** — 8 values:
`validation_pass` · `validation_warning` · `validation_failure` ·
`operator_required` · `config_required` · `projection_required` ·
`acceptable_downstream_gap` · `semantic_derivation_required`

**`downstream_owner`** — 5: `validation` · `transformation_validation` ·
`projection` · `operator` · `config_policy`

**Blocking flags** — three, deliberately separate:
`blocking_for_validation` · `blocking_for_projection` · `blocking_for_xml_delivery`

| Category | Example rule / issue type | Classification | Severity | Meaning | Structurally invalidates the canonical? | Currently overrideable | Historically intended overrideable |
|---|---|---|---|---|---|---|---|
| **Core-canonical presence** | `CORE001-{field}` (column absent) | `validation_failure` | error | An economically essential field is not there at all | **Yes** | **Yes** | **No** — no source contemplates it |
| **Core-canonical population** | `CORE002-{field}` (column present, wholly blank) | `validation_failure` | error | An economically essential field is empty on every row | **Yes** | **Yes** | **No** |
| **Mandatory-field presence** | `VR-{field}-presence`, `core=True` | `validation_failure` | error | A registry-mandatory management field is absent | **Yes** | **Yes** | **No** |
| **Regulatory-only presence** | `VR-{field}-presence`, `core=False` | `validation_warning` | warn | Required by the regulator, not by the figures | No | n/a — blocks projection, not validation | n/a (correct as is) |
| **Parse failure** | `date_parse_failed`, `numeric_parse_failed`, `boolean_parse_failed` | `validation_failure` | error | A value exists but cannot be typed | Borderline — the value is present but unusable | **Yes** | Plausibly yes, with a corrected value (`override`) |
| **Invalid configured default** | `invalid_default`, `invalid_nd_default` | `validation_failure` | error | A default in the asset/regime config is itself invalid | No — a configuration error | **Yes** | No — owner is `config_policy`, the fix is the config |
| **Unmapped enum, mandatory** | `enum_unmapped` + `mandatory` + `enforce_presence` | `validation_failure` | error | A lender value has no accepted code | No | **Yes** | Plausibly yes (`accept` with rationale) |
| **Unmapped enum, optional** | `enum_unmapped` otherwise | `config_required` | warn | Extend the mapping | No | not blocking | n/a |
| **Business-rule violation** | `validate_business_rules`, `validate_uniqueness` | `validation_failure` / `validation_warning` | error / warn | A deterministic relationship fails | No | **Yes** | **Yes — this is the designed case** |
| **Source absent** | `source_absent` | `validation_warning` / `config_required` | warn | The lender does not supply it | No | not blocking | n/a |
| **Operator decision pending** | `operator_decision_pending` | `operator_required` | warn | Awaiting an answer | No | not blocking | n/a |
| **Unknown management enum** | `ENUM_INVALID` | `validation_warning` | warn | Lender's own word, no ESMA code | No | not blocking | n/a — correct |

The estate already distinguishes structure from business rule. The
classification is computed, recorded per issue in `43_validation_issues.csv`,
and then **collapsed to a single boolean** (`ready_for_validation_complete`)
before the operator ever sees it.

---

## 4. What an approved exception is supposed to mean

| # | Question | Answer from the estate |
|---|---|---|
| 1 | Is evidence / comment mandatory? | **Yes by design, no in implementation.** `justification TEXT NOT NULL` in the ledger; the ledger UI validates it; doc 08 lists it in the payload. `resolve_decision` requires a reason for `defer` and `reject` and **not** for `approve`. Runtime proof: `"justification": ""`. |
| 2 | Is privileged approval required? | The ledger carries `users.role`. OCC has `ADMIN_ONLY_SCOPES = (asset, global)`; `file` is not among them, so any operator with client access can approve. No evidence the exception was *intended* to be admin-only — but no evidence it was intended to be open to everyone for a structural failure either, because that case was never contemplated. |
| 3 | Scope | **This file / this delivery**, explicitly, and it behaves that way — §1.5. |
| 4 | Is the original finding retained in the audit record? | **No.** The audit chain records `rule_persisted {kind: validation_exception}` and `decision_approved {value: proceed}`, but *what* failed is only in `43_validation_issues.csv`, which lives under `TRAKT_OPS_STAGING_ROOT` — `/tmp/trakt/ops_staging` on the deployed App Service. Ephemeral. The durable record never carries the finding. |
| 5 | Should the stage ever read `All checks passed`? | **No.** After an override the validation GAR is rewritten to `status: completed`, `summary: "All checks passed."`, `blockers: []`, `warnings: []`. That is a false statement in the system of record. |
| 6 | Does an explicit exception state already exist? | **Not for stages.** `STAGE_STATUSES` has no `passed_with_exception`. But two existing vocabularies serve: `ST_APPROVED = "approved"` is already a stage status distinct from `completed`, already used when an operator resolves a review batch; and the ledger's `accepted / overridden / escalated`. **No new status needs inventing.** |

---

## 5. Reassessment of the Gate 3 finding

**B and C. Not A, and not D.**

| Part | Verdict |
|---|---|
| The exception capability exists deliberately | **A — intentional and correct.** In OCC from inception, named in the rule model, covered by a test. |
| File / reporting-period containment | **A — intentional and correct, and it works.** Proven: period 2 re-asks and refuses. |
| Never generalises to portfolio or client scope | **A — correct.** `_persist_rule` forces `scope = "file"`. |
| **Structural / core-canonical failures are eligible** | **B — valid capability, implementation too broad.** A 2026-08-29 integrity rule inherited a 2026-08-04 generic route because both are `blocking_for_validation`. Never intended, never tested, never discussed. |
| **Justification, evidence, finding reference, authority, honest status** | **C — valid capability, control implementation incomplete.** Every one of the designed requirements is absent: no mandatory justification, no `rule_id_ref`, no `condition`, no evidence on the decision (`evidence: []`, `observed_values: []`, `affected_record_count: 0`), no privileged approval, no durable retention of the finding, and a stage record that says the opposite of the truth. |

It is **not D**. A true bypass would let the exception escape its period, generalise
across deliveries, or apply without any operator act. None of those is the case.

---

## 6. Wrong approved decision — is the correction mechanism really absent?

### 6.1 The lifecycle already exists, and is already tested

| # | Question | Answer |
|---|---|---|
| 1 | Why does `RuleStore.retire()` exist? | Because `retired` is one of the three designed rule states — doc 08 §1: `"status": "active \| superseded \| retired"`. It is fully implemented (sets status, records `reason` and `approved_by`, writes both the version and current) and **covered by `tests/operations_control/test_rules.py::test_retire`**. |
| 2 | Was operator correction originally intended? | **Explicitly.** Doc 08 §5: an item conflicting with an active rule offers *"keep the rule (treat file as exception) / **update the rule (new version)** / file-scope override."* §2: *"approving a new value for an existing (kind, scope, subject) creates **version n+1** of the same rule, superseding n."* Covered by `test_reapproval_supersedes_same_subject_same_scope`. |
| 3 | Is there already a method that could expose it? | `RuleStore.retire()` and `RuleStore.approve()` are both public and complete. There is no `OpsEngine` wrapper and no route — `GET /ops/rules` and `GET /ops/rules/{id}/history` are the only rule endpoints, both read-only. |
| 4 | Can a new record supersede an old one? | Yes — `approve()` → `_find_same_subject()` → version n+1, old marked `superseded`. **With one important caveat in §6.3.** |
| 5 | Does persistence preserve history? | Yes. Every version is written to its own URI (`rule_version_uri`) plus a `current.json`; `history()` reads them all. Proven: after retirement the v1 record survives with its original payload and the retirement reason. |
| 6 | What downstream state must be invalidated? | Nothing that is not already regenerated. `_write_approved_overrides_file` says so itself — *"regenerated from the governed rule store on every approval"* — and `_execute` calls it on **every** run. Proven: the corrected column took effect on the next period with no rerun of the corrected period. |
| 7 | New architecture, or unexposed lifecycle? | **Unexposed lifecycle.** The correction is two calls to existing, tested methods. |

### 6.2 Proven end to end, using only existing public methods

```
period 1   operator answers the loan-key question CUST_ID  (the customer — wrong)
           canonical keyed 9000, 9001 → published

correction rules.retire("CMP", old.rule_id, by="Administrator",
                        reason="CUST_ID is the customer, not the account")
           rules.approve(RuleRecord(... source_column="ACCT_REF" ...))

           old rule    status  retired,  payload CUST_ID   (history preserved)
           new rule    status  active,   payload ACCT_REF
           applicable  ["ACCT_REF"]                        (exactly one)

period 2   questions asked   0
           canonical keyed   100000, 100001            ← CORRECTED
           status            awaiting_publication
```

No code was changed and no state was hand-edited. The correction propagates
because Gate 1's overrides artefact is rebuilt from the rule store on every run.

### 6.3 Two things a naive exposure would get wrong

**`subject_key` for a field mapping is the *source column*, not the canonical
field:**

```python
if self.kind in ("field_mapping", "alias"):
    return f"{self.kind}:{_norm(p.get('source_column') or p.get('alias', ''))}"
```

So "the same canonical field, a different column" is a **different subject** and
`approve()` alone will **not** supersede — it leaves two active rules both
claiming `loan_identifier`, and `applicable()` returns both. The working
correction is **retire the old, then approve the new**, which is what was proven
above.

**Retirement alone is not a correction.** Proven separately: retiring the wrong
loan-key rule removed it from `applicable()` correctly, but the next period asked
**0 questions** and produced the **same wrong identifiers** — because the
deterministic default reverts to the same guess the operator was originally asked
to settle. Removing an answer does not re-raise the question.

---

## 7. File-role learning

| # | Question | Answer |
|---|---|---|
| 1 | Designed to be learned? | **Yes, explicitly.** `SourceRecord.file_role_schemas`: *"Approved per-role header/column signatures … **captured at promotion**. The PRODUCTION role-detection rule is header-first: an incoming file whose normalised headers match one of these signatures is assigned that logical role **regardless of filename**."* |
| 2 | What structure already exists? | `SourceRecord.file_role_schemas` (`role → [columns]`, primary) and `file_role_aliases` (`role → [name patterns]`, documented as a *"FALLBACK hint only"*). Both are read on every intake by `assess_batch`. |
| 3 | Repeated questioning intentional for safety? | **No.** `operations_control/occ_agent/pack.py` states the governed decision: field mappings *"are learned from the first representative delivery and approved through the existing mapping path"*, and the catalogue records `file_role_schemas` as *"Learned at mapping approval from a representative pack"*. The legacy router implements it (`router.py:607`). Only the OCC promotion path omits it. |
| 4 | Does the approved answer carry enough provenance? | Yes. The answer is an `override_classification` against a named `source_file_id` in a batch whose headers, SHA-256 and `source_uri` are all recorded, and it is promoted under an approval artefact already keyed to client / portfolio / dataset / frequency / fingerprint. |
| 5 | What should period 2 do? | Recognise the file **header-first, regardless of filename**, and ask nothing. |

**Cause:** `_promote_source` (engine.py:2441) calls `approvals.write_pending(...)`
without `role_schemas=` or `role_aliases=`, so the artefact carries `{}` and
`approvals.promote` — which *does* record them when present — has nothing to
record. Incomplete promotion path, not a safety choice.

---

## 8. `last_successful_reporting_period`

| # | Question | Answer |
|---|---|---|
| 1 | Wrong, or stale UI metadata? | **The field is wrong**, and it is not UI-only — it feeds a classification branch. |
| 2 | Could stale state change workflow/routing? | It changes the **classification label**, not execution. `classification.py:92` — a delivery whose period precedes `last_successful_reporting_period` is classified `WF_BACKFILL` instead of `WF_RECURRING`. Because only `_promote_source` writes it, and that runs only for `new_client` / `new_portfolio`, the anchor is frozen at the **first** published period — so a genuine backfill of a mid-history period is mislabelled `recurring`. `WF_BACKFILL`'s only consumers are `language.py`'s label (*"Historical backfill"*) and its explanatory sentence. No gate, publication or regime behaviour changes. |
| 3 | Will the dashboard rely on it? | It is the only registry field for "previous successful period", which the certification's observability inventory names as a dashboard item. **It should not be built on.** |
| 4 | Authoritative alternative today | `store.list_publications(client_id)` — one record per period carrying `reporting_period`, `version`, `status: published` and `published_at` — and the per-client workflow index. Both are accurate and per-period. |

The legacy router updates this field on **every** successful run
(`router.py:538-539`). The OCC route updates it only at onboarding. The same
pattern as file-role learning: the older route implements it, the OCC route
does not.

---

## 9. Interrupted run — and the finding the certification understated

### 9.1 The recovery mechanism is right; its trigger is not

Tested with a second `OpsEngine` over the same store — which is exactly what a
different process is:

```
run status                              running
lease                                   {owner_pid: 2078, started_at: 2026-08-30T16:41:32Z}
other process sees _is_executing        False
operator presses "Run again"            REFUSED  OPS_ALREADY_RUNNING
recover_on_startup()                    ["wf_251f8601b5c9"]
  → status blocked, interrupted True
  → "This run was interrupted. Choose 'Run again' to continue where it left off."
operator presses "Run again"            ACCEPTED → running
```

So: while a run is stranded at `running`, the operator has **no route at all** —
`start()` refuses any status outside `received | needs_review | blocked | failed
| held`. `recover_on_startup()` produces exactly the right state and the right
affordance, and it fires only in the OCC API's FastAPI lifespan hook. Nothing
fires it while the API is up.

### 9.2 The part the certification missed: "Run again" does not work

`TRAKT_OPS_STAGING_ROOT` is `/tmp/trakt/ops_staging` on the deployed OCC API
(`deploy/trakt-ops-api/provision.sh:91`), and the delivery's **input files live
there**: `intake.batch_dir()` = `staging_root/{client}/batches/{batch_id}/files`.
So does the orchestrator's `run_state.json` and `43_validation_issues.csv`.

Simulating the recycle by clearing the staging root:

```
raw file still in durable blob           True
staging root after recycle               gone
recover_on_startup()                     ["wf_5d8324679180"]
  status blocked, interrupted True
  "This run was interrupted. Choose 'Run again' to continue where it left off."
run.delivery.input_path exists now       False
operator presses "Run again"             → status  blocked
                                           understanding  blocked
                                           mapping        blocked
  blockers: "Something did not go as expected on our side.
             Nothing has been lost. You can try running this step again."
canonical rebuilt                        False
```

The operator is told *"Nothing has been lost. You can try running this step
again"*, tries again, and gets the same result. Both sentences are false: the
staged inputs **were** lost, and running it again cannot succeed.

### 9.3 Answers

| # | Question | Answer |
|---|---|---|
| 1 | Process dies mid Gate 2/3/4/5? | The run stays `running` in durable store with a live lease. All governed state (run doc, decisions, rules, audit, publications) survives — it is in blob. All **working** state (input files, orchestrator state, validation issues) is in ephemeral `/tmp`. |
| 2 | Does Event Grid redeliver? | **No.** The Function returns successfully once `handle_arrival` has registered the file; the pipeline runs on a detached daemon thread afterwards. Event Grid only retries on handler failure, and the handler did not fail. |
| 3 | Does OCC recognise the existing run? | Partly. A redelivery of the same blob finds the batch in `running`, which is excluded from the reuse branch, so it opens a successor pack `…_v2`; `register_file` then dedupes the identical content and returns without registering, leaving the successor at `receiving` forever. |
| 4 | Is a stale lease releasable? | Only by `recover_on_startup()`, which clears it. The lease carries `owner_pid` and `started_at`, so it is a perfectly good staleness signal — nothing reads it as one. |
| 5 | Can the operator restart without a developer? | While `running`: **no**. After recovery: yes, but the resume fails if staging was lost. There is a real route out — `POST /ops/batches/{batch_id}/upload`, the manual delivery upload — but nothing tells the operator that is what is needed. |
| 6 | Can a run be stranded indefinitely? | Until the OCC API restarts, yes, and it is indistinguishable from a healthy `running` run. |
| 7 | Is a sweeper the right fix? | **No.** Two smaller things, both reusing what exists: (a) on rerun, re-stage any registered file whose `storage_reference` is missing from its durable `source_uri` (recorded on every file record, with a `sha256` to verify); (b) let a run whose lease is older than a threshold be recovered without an API restart — `recover_on_startup()` already does the work, it just needs a second trigger that uses lease age rather than the process-local `_is_executing`. |

**Risk classification:** an operator-visible stranding, with a misleading
message, recoverable today only by an undocumented manual re-upload.

---

## 10. Client-config fallback

| # | Question | Answer |
|---|---|---|
| 1 | Which one? | `operations_control/configuration/resolver.py::EffectiveConfigResolver.client_config_for` (resolver.py:65), defaulting to `config/client/config_client_ERM_UK.yaml`. |
| 2 | Historical reason? | Stated in its own docstring: *"A client that has not [been onboarded] is governed by the repository file exactly as before, so adopting a client is a decision, never a side effect of this code shipping."* A deliberate legacy-adoption path for the incumbent, which predates OCC generating per-client configuration. |
| 3 | Which production paths reach it? | Three, all in `engine.py`: `assess_batch` (651) and `start_batch` (778) via `resolve()`, and `_run_annex2_chain` (1491) directly. |
| 4 | Does every caller pass the activation guard? | Yes. `assess_batch` calls `_client_is_activated` at line 647, **before** line 651. `start_batch` requires `status == "ready"`, only reachable through `assess_batch`. `_run_annex2_chain` is only reached after a successful orchestration started by `start_batch`. |
| 5 | Background / admin routes that bypass it? | None found. The three callers are the whole surface. |
| 6 | Can a malformed / missing activated config still cause fallback? | **Yes — this is the live risk.** The guard and the resolver consult **different blobs**. `_client_is_activated` passes if the client appears in the onboarding **index** (`onboarded_clients()` → `_index_uri()`). `client_config_for` needs the client's **artefact** (`read_artefact(client_id, client_config_rel(client_id))`) and silently falls back on any exception or empty result. Index present + artefact absent, empty or transiently unreadable → the regulatory return is built against the incumbent's LEI, originator name and establishment country. |
| 7 | Any legitimate reason for it now? | For the incumbent, whose `client_id` matches the repository file, yes — that is the third arm of `_client_is_activated`. For anyone else, no. |

**Classification: LATENT ISOLATION RISK.** Not unreachable legacy debt — it is
live code on the production path, and the only thing between it and a
cross-client identity leak is a guard that reads a different document. Not a live
defect — no production path was demonstrated that reaches it with the wrong
client, and both rehearsals produced correctly isolated configurations.

There is an established precedent for the fix in this estate: the previous sprint
changed `mi_agent_api.currency.client_config_path()` to return `None` rather than
the incumbent's file, for exactly this reason.

---

## 11. XSD preview regression

| # | Question | Answer |
|---|---|---|
| 1 | What changed? | Commit `1f0be01` re-pointed `scripts/build_annex2_field_xsd_path_map.py` from the retired `annex2_delivery_rules.yaml` to `as_delivery_rules(build_contract())`. The generator grades a mapping `confirmed` only when the contract's `workbook_semantic` token matches an XSD leaf **by exact name**. The retired file carried a bare leaf token; the derived contract carries a **path-qualified** one — `Dtls/PrprtyTp`, `ScrtstnRpt/CutOffDt`, `BalDtls/PrrPrncplBal`. Path-qualified tokens miss the leaf-name index and fall to the fuzzy branch. |
| 2 | Obsolete assertion? | **No.** The test asserts *"statuses are stable across regeneration"*, and they are not: 14 of 107 codes differ. The committed artefact and its own generator genuinely disagree. |
| 3 | Real regression? | **Yes, but a confidence-grading regression, not a path regression.** 10 codes dropped `confirmed → inferred_high_confidence` (RREC9, RREL2, 6, 22, 25, 31, 32, 40, 41, 69); 4 improved `unresolved → inferred_low_confidence` (RREC1, RREC2, RREL67, RREL83) because the derived contract supplies tokens the retired file lacked. No path became wrong. `workbook_semantic` is present for 104 of 107 codes — the evidence did not disappear, the generator stopped understanding its shape. |
| 4 | Does any operator surface consume it? | **No.** `config/delivery/annex2_field_xsd_path_map.yaml` is read only by `engine/delivery_xml_agent/{preview_readiness,xsd_structured_preview_builder}.py` and four analysis scripts. Nothing in `operations_control/`, `apps/`, `mi_agent_api/` or `function_app.py` imports them; the only non-test importer is `scripts/inspect_delivery_xml_readiness.py`. OCC's Gate 5 runs `xml_builder_annex2.py` directly against the workbook and the XSD, and produced a schema-valid return twice in the certification. |
| 5 | Update the test, or restore behaviour? | **Neither as posed — fix the generator, then regenerate.** Teach it to accept a path-qualified `workbook_semantic` (match the full path, or split on `/` and take the leaf), then rebuild the committed artefact. That restores the intended grading and the test passes on its own terms. Weakening the assertion would discard the only thing keeping the artefact honest. |

Kept separate from the OCC governance findings: this is developer introspection,
not a control.

---

## 12. Reclassification of the six findings

| # | Finding | Original | Intended design | Actual defect? | Production consequence | Pre-Client-1 priority |
|---|---|---|---|---|---|---|
| 1 | Approved decision cannot be changed | P1 "architectural" | **Correction was designed**: doc 08 §5 offers *"update the rule (new version)"*; `status: active\|superseded\|retired`; `retire()` and supersession both implemented **and tested** | **Yes — exposure only.** Two existing method calls fully correct a wrong mapping, proven end to end. Not architecture | A wrong answer to a question OCC itself flags as ambiguous is permanent; the only route today is editing the rule store | **P1A** |
| 2 | Gate 3 validation exception | P1 | A per-finding, justified, file-scoped acceptance of a **known data issue in one delivery** | **Partly.** Capability, scoping and period-containment are correct and proven. Two defects: structural/core failures are eligible when nothing ever intended them to be (**B**), and every designed control — justification, finding reference, evidence, authority, honest status, durable retention — is missing (**C**) | A single undifferentiated click can publish an economically incomplete canonical, and the record then reads "All checks passed." Contained to one period | **P1A** |
| 3 | File role not learned | P1 | **Designed and documented**: signatures *"captured at promotion"*, recognition *"header-first … regardless of filename"*; the legacy router implements it | **Yes — two omitted kwargs** in `_promote_source` | One extra operator click per period for any lender whose filename is outside the built-in vocabulary | **P1A** |
| 4 | `last_successful_reporting_period` stale | P1 | Updated on every successful run (as the legacy router does) | **Yes**, but consequence is narrow: it feeds `WF_BACKFILL` vs `WF_RECURRING`, whose only effect is the operator-facing label | A mid-history backfill is mislabelled "recurring". No gate, publication or regime behaviour changes | **P2** — but the dashboard sprint must not build on it; use `list_publications` |
| 5 | Interrupted-run recovery | P1 "no sweeper" | `recover_on_startup()` marks the run resumable and tells the operator to press "Run again" | **Yes, and worse than certified.** The state and affordance are right; the resume **cannot succeed** after the recycle that triggers it, because the run's inputs share the ephemeral `/tmp` staging. The message *"Nothing has been lost"* is false | Operator-visible stranding with a misleading message and no working offered route; recoverable only by an undocumented manual re-upload | **P1A** |
| 6 | Client-config fallback | P1 latent | The fallback is a deliberate incumbent-adoption path predating per-client generation | **Latent.** All three callers pass the activation guard, but the guard reads the onboarding **index** while the resolver needs the **artefact** — two independent blobs that can diverge | If they diverge, a regulatory return carries another lender's LEI and originator identity. Not demonstrated in production | **P1A** |

**P0: none.** No finding prevents a commercial reporting cycle in the intended
current scope, and two full cycles were completed in the certification.

---

## 13. Minimum pre-go-live hardening sprint

Six changes. Five are exposure or completion of mechanisms that already exist;
none introduces an abstraction.

### H1 — Exclude structural failures from the exception route *(the one that matters)*

| | |
|---|---|
| **Defect** | CORE001 / CORE002 / mandatory-presence failures inherit the generic `validation_halt` route (§2) |
| **Reuse** | `validation_classification` and the three blocking flags already computed per issue in `43_validation_issues.csv`; `_result(...)` already distinguishes `core_canonical_presence` as a `check_type` |
| **Files** | `operations_control/adapters.py` (the `va.status == STEP_HALTED` branch that builds the decision) |
| **Behaviour change** | Yes, deliberately: a run halted **only** on structural findings offers no "proceed" — it states which fields and how many rows, and the route forward is to correct the source, the mapping or the value |
| **Acceptance test** | Blank `account_status` on all rows → no `validation_exception` decision is offered; publication stays refused. Business-rule-only halt → the exception is still offered and still works |
| **Regression** | `tests/operations_control/` (esp. `test_workflow_engine.py::TestValidationException`), `test_occ_go_live_e2e.py`, `test_validation_agent_workflow.py` |
| **Effort** | **MEDIUM** |

### H2 — Complete the exception's designed controls

| | |
|---|---|
| **Defect** | No mandatory justification, no finding reference, no evidence on the decision, and the stage reports "All checks passed." after an override (§1.4, §4) |
| **Reuse** | `justification` is already in the payload shape; `43_validation_issues.csv` already holds field / rule / severity / counts; `DecisionRequired` already has `evidence`, `observed_values`, `affected_record_count`; `ST_APPROVED` already exists as a stage status distinct from `completed`; `OPS_REASON_REQUIRED` already exists |
| **Files** | `operations_control/adapters.py` (attach evidence), `operations_control/engine.py` (require a reason for this kind; stop rewriting the stage summary) |
| **Behaviour change** | Yes: approving without a reason is refused; the stage reads `approved` with the original blockers retained as warnings rather than `completed` / "All checks passed." |
| **Acceptance test** | Approve with no reason → `OPS_REASON_REQUIRED`. After a valid override, the stage is not `completed` and the original blockers survive on the record |
| **Regression** | `tests/operations_control/`, `test_occ_go_live_e2e.py` |
| **Effort** | **LOW–MEDIUM** |

### H3 — Expose the rule correction that already exists

| | |
|---|---|
| **Defect** | `RuleStore.retire()` / `approve()` are complete and tested but unreachable from OCC (§6) |
| **Reuse** | Both methods, unchanged; `_write_approved_overrides_file` already regenerates from the rule store on every run, so no invalidation machinery is needed |
| **Files** | `operations_control/engine.py` (one `correct_rule(client_id, rule_id, …)` method doing retire-then-approve), `operations_control/api/app.py` (one route), Rules Library UI action |
| **Behaviour change** | Additive |
| **Acceptance test** | The proven scenario: answer the loan key wrongly, publish, correct through the new method, next period asks 0 questions and produces the corrected canonical; the retired version and its reason survive in `history()` |
| **Regression** | `tests/operations_control/test_rules.py`, `test_workflow_engine.py`, `test_occ_go_live_e2e.py` |
| **Effort** | **LOW** — but mind `subject_key` (§6.3): retire-then-approve, not approve alone |
| **Open question for the owner** | Whether correcting a rule should flag periods already published under the superseded version. Recommend recording it and surfacing it, not auto-republishing |

### H4 — Finish the file-role promotion path

| | |
|---|---|
| **Defect** | `_promote_source` omits `role_schemas=` / `role_aliases=` (§7) |
| **Reuse** | `approvals.write_pending` already accepts both; `approvals.promote` already records them; `router.role_schemas_for_pack` / `aliases_for_pack` already compute them |
| **Files** | `operations_control/engine.py::_promote_source` |
| **Behaviour change** | Yes, and it is the documented design: period 2 recognises the file header-first and asks nothing |
| **Acceptance test** | Deliver a file whose name is outside the vocabulary, answer once, publish; next period creates a workflow with no file-role question and `file_role_schemas` non-empty in the registry |
| **Regression** | `tests/operations_control/test_intake.py`, `test_classification.py`, `test_occ_go_live_e2e.py`, `test_blob_trigger_app.py` |
| **Effort** | **LOW** |

### H5 — Make "Run again" actually work after a recycle

| | |
|---|---|
| **Defect** | Recovery promises a resume that cannot succeed once ephemeral staging is gone, and says "Nothing has been lost" (§9.2) |
| **Reuse** | Every registered file already carries a durable `source_uri` (`blob://raw-v2/…`) and a `sha256`; `Storage.download_file` is the same call intake already uses; `recover_on_startup()` already does the recovery work; the lease already carries `owner_pid` and `started_at` |
| **Files** | `operations_control/intake.py` or `engine.py` (re-stage a file whose `storage_reference` is missing, verifying the sha256); a lease-age trigger for the existing `recover_on_startup()`; correct the blocker sentence |
| **Behaviour change** | Yes: a rerun after staging loss succeeds instead of failing, and a stranded run becomes recoverable without an API restart |
| **Acceptance test** | Run to a resting state, delete the staging root, recover, press "Run again" → the canonical is rebuilt and the run reaches its previous state |
| **Regression** | `tests/operations_control/test_recovery.py`, `test_intake.py`, `test_workflow_engine.py` |
| **Effort** | **MEDIUM** — a lease-age trigger must not mark a run healthy in another process as interrupted; `_is_executing` is process-local, which is why the current trigger is startup-only |

### H6 — Close the client-config fallback

| | |
|---|---|
| **Defect** | `client_config_for` silently returns the incumbent's file when a client's own artefact is missing, empty or unreadable (§10) |
| **Reuse** | The precedent already applied in this estate to `mi_agent_api.currency.client_config_path()` — return `None` rather than the incumbent's file |
| **Files** | `operations_control/configuration/resolver.py` |
| **Behaviour change** | Yes: a client whose configuration cannot be read is **blocked** rather than processed under someone else's identity. Keep the incumbent arm only where `client_id` equals the configured client id — the same test `_client_is_activated` already uses |
| **Acceptance test** | A client in the onboarding index whose config artefact is absent → the delivery is blocked with a configuration message, and no other client's LEI appears anywhere |
| **Regression** | `tests/operations_control/test_effective_configuration.py`, `test_tenancy.py`, `test_occ_go_live_e2e.py` (client isolation) |
| **Effort** | **LOW** |

**Deliberately excluded:** `last_successful_reporting_period` (P2 — one line, but
no production consequence; do it with the dashboard sprint) and the XSD path-map
generator (P2 — developer introspection, no operator surface).

---

## 14. The Gate 3 model the evidence supports

This is **already the intended architecture**. Every distinction below is
computed today and then discarded at the operator boundary. Only the last column
is a new policy decision.

| Gate 3 outcome | Repository terminology | What the operator is offered | Status recorded |
|---|---|---|---|
| **Structural / core-canonical failure** | `validation_classification: validation_failure`, `check_type: core_canonical_presence`, rules `CORE001` / `CORE002`, and `VR-{field}-presence` where the registry says `core_canonical: true` | **No acceptance.** The finding is stated — field, rule, row count — and the routes forward are the ones that already exist: point the field at a column, supply a value, or correct the source. `blocking_for_validation` stands | stage `blocked` / `needs_review`, blockers retained |
| **Business-rule failure** | `validation_failure` from `validate_business_rules`, `validate_uniqueness`, `date/numeric/boolean_parse_failed`, mandatory `enum_unmapped` | **Governed validation exception**, as designed: the specific finding named, a **mandatory** justification, scoped to this file / delivery, the original finding retained | stage `approved` (existing `ST_APPROVED`), original blockers carried as warnings |
| **Warning** | `validation_warning`, `config_required`, `operator_required`, `projection_required`, `acceptable_downstream_gap`, `semantic_derivation_required`, `ENUM_INVALID` | Recorded, non-blocking for validation; `blocking_for_projection` continues to hold the regulatory branch only | stage `completed` with warnings |

**Already intended, needs enforcement:**
- the three-way distinction (it is `validation_classification` + the three
  blocking flags, computed per issue today);
- mandatory justification (`justification NOT NULL` in the ledger; in doc 08's
  payload);
- naming the specific finding (`rule_id_ref` in doc 08's payload;
  `finding_id` FK in the ledger);
- file / delivery scoping (already enforced, already works);
- not reporting "All checks passed" after an exception (no source ever asked for
  this; `ST_APPROVED` already exists to say what actually happened).

**A new policy decision, requiring owner approval — two questions:**
1. **Should a structural core-canonical failure be absolutely non-overrideable,
   or overrideable by an administrator only?** The evidence says it was never
   *intended* to be overrideable, but "never intended" is not the same as "must
   be forbidden". Recommendation: absolutely non-overrideable. A core field that
   is absent or wholly empty is not an exception with a rationale; it is a report
   that cannot be produced, and the operator already has governed routes to
   supply the missing mapping or value.
2. **Should a business-rule exception require an admin, or any operator with a
   justification?** The legacy ledger carries `users.role` but never gated on it.
   Recommendation: any operator, with a mandatory justification and the finding
   named — the audit chain and the file scoping are the control, and requiring an
   admin for a routine monthly data issue would push operators toward the wrong
   answer.

---

## 15. Final answers

**1. Why was Gate 3 made overrideable?**
To let an operator accept a **known, identified data-quality finding in one
month's tape** with a stated reason, rather than hold a whole delivery for a
value the lender cannot restate. It arrived with OCC itself (`3f8628a`,
2026-08-04), three weeks before the core-canonical contract existed.

**2. Was CORE001/CORE002 intentionally made overrideable?**
**No.** The enforcement commit (`a43f0ca`) reasons only as far as "readiness
fails", never mentions the exception route, and adds no interaction test. No
design document, policy or test anywhere contemplates accepting a structurally
absent or wholly empty core field. It inherited an older generic route because
every Gate 3 blocking finding is expressed as `blocking_for_validation`.

**3. Valid capability, accidental widening, incomplete control, or true bypass?**
**B and C — accidental widening *and* incomplete control implementation.** Not a
true bypass: the capability is designed, the scoping is correct, and the
containment to one reporting period is real and proven.

**4. Which Gate 3 failures should be overrideable?**
Business-rule failures — deterministic relationship checks, uniqueness, parse
failures, mandatory unmapped enums. Not structural core-canonical failures. Not
invalid configured defaults (the owner is `config_policy`; the fix is the
configuration).

**5. What should an exception require?**
A mandatory justification; the specific finding named (rule, field, affected row
count) and shown to the operator before approval; file/delivery scope (already
enforced); the original finding retained durably; and a stage status that says
`approved`, not `completed` / "All checks passed."

**6. Was the inability to amend an approved mapping intentional immutability?**
**No.** Doc 08 §5 explicitly offers *"update the rule (new version)"*; `status`
includes `retired`; both `retire()` and same-subject supersession are implemented
**and covered by tests**. It is an unexposed lifecycle capability. Proven: two
existing method calls correct the mistake and the next period comes out right
with zero questions.

**7. Should file-role decisions be learned?**
**Yes** — the `SourceRecord` docstring states the production rule is header-first
recognition from signatures *"captured at promotion"*, and the legacy router
implements it. The OCC promotion path simply omits two arguments.

**8. Is stale `last_successful_reporting_period` operationally consequential?**
Barely. It feeds `WF_BACKFILL` vs `WF_RECURRING`, whose only effect is the
operator-facing label; no gate, publication or regime behaviour depends on it. It
**is** wrong, and the dashboard must use `list_publications` instead.

**9. What happens after a worker recycle, and what is the minimum adequate
recovery?**
The run stays `running` with a live lease; Event Grid does not redeliver; the
operator cannot rerun (`OPS_ALREADY_RUNNING`); `recover_on_startup()` fixes it
but fires only at API startup. Worse: once ephemeral `/tmp` staging is gone, the
"Run again" the recovery message promises **fails**, and the message claims
"Nothing has been lost". Minimum adequate fix is **not a sweeper** — re-stage
inputs from the durable `source_uri` each file already carries, and give the
existing `recover_on_startup()` a lease-age trigger.

**10. Is the client-config fallback genuinely reachable in production?**
Not through any demonstrated path — all three callers sit behind the activation
guard. But the guard reads the onboarding **index** and the resolver reads the
per-client **artefact**, and those are independent documents. If they diverge, a
regulatory return is built with the incumbent's LEI and originator identity.
**LATENT ISOLATION RISK.**

**11. What is the XSD preview regression?**
The derived Annex 2 contract supplies `workbook_semantic` as a path-qualified
token (`BalDtls/PrrPrncplBal`) where the retired rules file supplied a bare leaf.
The generator matches leaves by exact name, so 10 codes dropped `confirmed →
inferred_high_confidence` and 4 improved `unresolved → inferred_low_confidence`.
A confidence-grading regression, not a path regression; no operator surface
consumes it; fix the generator and regenerate rather than weaken the test.

**12. Which findings genuinely need fixing before Client 1?**
Five, all P1A: **H1** exclude structural failures from the exception route,
**H2** complete the exception's controls, **H3** expose the rule correction,
**H4** finish the file-role promotion, **H5** make "Run again" work after a
recycle, **H6** close the client-config fallback. (Six changes; H1 and H2 are
one finding.)

**13. What is the minimum change set?**
Section 13. Five of the six are the exposure or completion of mechanisms that
already exist and are already tested; no new abstraction, no new status, no
sweeper, no rule-versioning architecture. Effort: three LOW, two MEDIUM, one
LOW–MEDIUM.

**14. Does the verdict stand?**
**It should become PASS WITH PRE-GO-LIVE HARDENING.**

Not `PASS`: the certification's central claim — that a trained operator can run
the whole cycle without a developer — remains proven, but it is a
first-pass-correct claim, and three of the findings (H3, H5, H1/H2) are exactly
the paths a first period will actually exercise when something goes wrong.

Not `FAIL`: there is no P0. Two complete reporting cycles ran end to end, one
producing a schema-valid Annex 2 return, with zero developer actions.

`PASS WITH OPERATING CONDITIONS` understated what is now known in one direction
and overstated it in another. The Gate 3 exception is a real designed control
whose containment works, and correcting a decision is two existing method calls
away — both better than certified. The interrupted-run path is worse: it strands
a run and tells the operator nothing has been lost. What separates the estate
from `PASS` is no longer a set of conditions to operate around; it is a bounded
hardening sprint, most of which is turning on machinery that is already written
and already tested.

---

*Follow-up performed against `main` at `e7678c81100f562c25cb39cf1cbf69798e13a5ed`,
branch `claude/occ-production-certification-axdj1z`, on top of certification
commit `a88b53f`. No production code was changed.*
