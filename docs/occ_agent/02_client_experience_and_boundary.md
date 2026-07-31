# The client experience, and the agent boundary

Two changes. The first makes the onboarding pack something you could actually
send a client. The second makes the line between the OCC Agent and the
Onboarding Agent checkable rather than merely stated.

---

## 1. Before and after: question classification

### Why scenario A produced 58 questions

The pack projected **every collected catalogue field**. `_client_visible` kept a
section if any of its fields was `collected and not answered_by_trakt`, and then
included every `collected` field in it. `collected` deliberately spans four
sources — client, operator, default and inferred — because inference *discharges*
a requirement rather than removing one, which is right for validation and wrong
for a questionnaire.

So the 58 broke down as:

| Source | Count | Could the client answer it? |
|---|---:|---|
| `client_supplied` | 38 | yes |
| `inferred` | 9 | no — the Onboarding Agent reads these from the first file |
| `operator_supplied` | 7 | no — Trakt's operator decides them |
| `trakt_default` | 4 | no — a governed default already applies |

Twenty of the 58 were not answerable by a client at all. Twenty more were
already answered by the operator's opening instruction and asked again anyway.
And the `data_definitions` section asked a client to describe files they had not
yet told Trakt about.

### The five categories

`operations_control/occ_agent/classification.py` assigns every field exactly one
category, **derived** from the catalogue's own `source` axis plus two
adjustments it cannot express: an already-answered field is category 1 whatever
its source, and `DEFERRED_SECTIONS` names sections whose questions need
something else to exist first. A five-entry `OVERRIDES` map covers fields whose
`source` is right for generation but wrong for who to ask.

### Scenario A, after

| # | Category | Count | What happens |
|---|---|---:|---|
| 2 | **Only the client can answer** | **22** | the pack |
| 1 | Already known | 22 | pre-populated; 11 offered for confirmation |
| 3 | Trakt works it out | 0 | see note below |
| 4 | Learned from the first delivery | 10 | never in the initial pack |
| 5 | Internal operator decision | 7 | OCC workflow only |
| — | Does not apply to this client | 2 | excluded by a catalogue condition |
| | **Total classified** | **63** | |

**58 client-facing questions → 22.** Ten are required.

**On category 3 being zero.** It is not empty because nothing is derived — it is
empty because by the time the pack is built, everything derivable has already
been derived. Those values appear in category 1 with their true origin in the
reason: *"already answered (a Trakt default)"*, *"already answered (derived by
Trakt)"*, *"already answered (read from a delivery)"*. Category 3 is reserved
for a value Trakt will derive but has not yet; on a case opened with a thinner
instruction it is populated. Reporting it as zero rather than folding it into
category 1 keeps the count honest.

**The 63 vs the catalogue's 84.** Regime fields for products this client did not
select are excluded entirely — they are not questions, not deferred and not
internal; they simply do not exist for this client.

### The full listing for scenario A

**Category 2 — asked (22)**

`entities[0]`: `lei`*, `country_of_establishment`*, `registration_number`,
`registered_address` · `contacts`: `reporting_contact_name`*,
`reporting_contact_email`*, `operational_contact_name`*,
`operational_contact_email`*, `reporting_contact_phone`, `authorised_approver` ·
`portfolios[0]`: `portfolio_type`* · `presentation`: `brand_colour`,
`logo_uri`, `disclaimer`, `reporting_calendar_note` · `access[0]`: `user_name`*,
`user_email`*, `user_role`*, `scope_note`, `occ_access_required`,
`dashboard_access_required`, `report_recipient`  *(\* = required)*

**Category 1 — pre-populated (22)** — `client.client_name`, `jurisdiction`,
`reporting_currency`, `time_zone`, `environment`; `entities[0].entity_id`,
`legal_name`, `roles`; `portfolios[0].asset_class`, `display_name`,
`originates`, `owning_entity`, `period_convention`, `reporting_currency`;
`presentation.day_count_convention`, `payment_frequency`, `report_title`;
`reporting.products`; `sources[0].cadence`, `cut_off_convention`,
`portfolio_id`, `regime_required`

**Category 4 — deferred (10)** — all eight `data_definitions` fields, plus
`sources[0].file_format` and `sources[0].expected_files`

**Category 5 — internal (7)** — `client.client_id`,
`portfolios[0].portfolio_id`, `structure`, `sources[0].dataset`,
`delivery_channel`, `mapping_complete`, `sample_provided`

**Not applicable (2)** — `contacts.investor_report_recipients` (no investor
reporting selected), `sources[0].source_party` (the book is not acquired)

### Conditional and progressive

`asked_when` is a new axis on the catalogue, using the same expression language
as `required_when` and evaluated by the same function. It decides whether a
question is worth putting to anyone at all, as distinct from whether an asked
question must be answered.

| Conditional on | Declared where | Effect |
|---|---|---|
| **Selected product** | `contacts.investor_report_recipients`, four `presentation` fields | branding is only asked of a client who receives an MI surface; an investor-report list only of one who selected investor reporting |
| **Portfolio structure** | `sources.source_party` (`portfolio_type == acquired`) | who sends the file is only asked when it is not the client |
| **Asset class** | regime sections, via `from_regime` + `f.product` | a product's standing regulatory fields exist only for a client who selected it |
| **Delivery method** | `sources.delivery_channel` is category 5 | the operator sets the channel; the client is not asked to choose Trakt's plumbing |

Progressive behaviour is demonstrable: answering
`portfolios[0].portfolio_type = acquired` unlocks `sources[0].source_party`,
which was previously reported *"does not apply here"*. A deferred section is
reported as **locked** with its trigger, not hidden:

```json
{"step": "data", "label": "How to read the data",
 "unlocked_by": "Asked once the client has listed the files they will send."}
```

### Multi-portfolio

Client-level questions are asked once; each book gets its own group.

| | one portfolio | two portfolios |
|---|---:|---:|
| Total questions | 22 | 24 |
| `contacts.*` | 6 | **6** |
| `portfolios[*]` groups | 1 | 2 |

The two extra questions are the second book's, and only the second book's.

---

## 2. Sample client-facing pack — scenario A

```markdown
# Onboarding — Northstar Lending

Reference: ONB-2026-0001

There are 22 questions for you, 10 of them required. Everything else Trakt
either already holds or works out itself.

## Please check these are right

Trakt already holds these. Tell us if any are wrong.

- **Client name**: Northstar Lending
- **Jurisdiction**: GB
- **Reporting currency**: GBP
- **Reporting time zone**: Europe/London
- **Legal name** — Northstar Lending: Northstar Lending
- **Roles** — Northstar Lending: originator
- **Portfolio name** — Northstar Lending portfolio: Northstar Lending portfolio
- **Asset class** — Northstar Lending portfolio: equity_release
- **Reporting period convention** — Northstar Lending portfolio: calendar_month_end
- **Expected cadence** — direct_101/funded: monthly
- **Reporting products**: mi

## About your business

The legal entity behind the portfolio, and who we should talk to.

### Northstar Lending

- [required] **Legal Entity Identifier**
      Twenty characters. Required for any entity named in regulatory reporting.
- [optional] **Company registration number**
- [required] **Country of establishment**
- [optional] **Registered address**

### Contacts and distribution

- [required] **Primary reporting contact**
- [required] **Reporting email**
- [optional] **Reporting contact telephone**
- [required] **Operational contact**
- [required] **Operational email**
- [optional] **Authorised approver**

## Your portfolios

One set of answers per book. Everything above is asked once.

### Northstar Lending portfolio

- [required] **How the book was acquired**

## Your reports

What you receive, and how it should look.

- [optional] **Brand colour** — A hex colour, for example #1F3B5C.
- [optional] **Logo**
- [optional] **Disclaimer**
- [optional] **Reporting calendar**

## Who needs access

People at your end who need Trakt, or who receive reports.

- [required] **Name**  [required] **Email address**  [required] **Role**
- [optional] **Which clients or portfolios**
- [optional] **Needs Operations Control Centre access**
- [optional] **Needs dashboard access**
- [optional] **Receives reports by email**

## Files to send

- **Primary loan tape** — required
- Cash-flow tape / Collateral tape / Funder P&I tape / Property tape — optional

## How to send them

- How often: monthly
- direct_101 funded: `raw-v2/NORTHSTAR/direct/funded/monthly/direct_101/2026-06-30/`

## What we are NOT asking you for

Trakt does not ask you to map your fields to ours. Send a representative file
and Trakt will propose the mapping itself; an operator reviews and approves it
during the first ingestion, and it is then fixed for every later delivery. What
Trakt cannot work out on its own is what your numbers MEAN — Trakt asks about
that once it has seen your files, not before.

Trakt reads these from the first file you send, so there is nothing for you to
fill in:

- Date conventions · Expected files · Fields that need explaining · File format
- Known data-quality limitations · Point in time or cumulative · Source file
- Units and currency · What a balance means · What the file contains
```

---

## 3. Sample internal operator review view

The operator sees three things a client never does: the full classification,
the internal decisions, and the provenance of every value.

```markdown
# Review — Northstar Lending

Reference: ONB-2026-0001

## What Trakt holds

### About the client
- **Client name**: Northstar Lending — an operator told Trakt
- **Client identifier**: NORTHSTAR — the client's name, checked against
                                     identifiers already in use
- **Jurisdiction**: GB — an operator approved it
- **Reporting currency**: GBP — the currency used in GB
- **Reporting time zone**: Europe/London — the time zone for GB

### Legal and reporting entities
- **Legal Entity Identifier** (Northstar Lending): 894500SYNTHETIC00042
                                                   — the client told Trakt
- **Country of establishment** (Northstar Lending): GB — the client told Trakt

### Expected deliveries
- **File format** (direct_101/funded): csv — the sample the client supplied
- **Expected files** (direct_101/funded): northstar_loan_extract_202606.csv,
                                          northstar_cashflow_202606.csv
                                          — the files in the sample

## Field mappings

Field mappings are NOT part of this configuration and were not collected. They
are proposed by Trakt from the first representative delivery, reviewed and
approved by an operator during that first ingestion, and then fingerprinted and
fixed. Approving this activation does not approve any mapping.

## Actions for an administrator
- [not_provisioned] Dana Fox <dana@…> — Add to the OCC operator list …
- [not_provisioned] Dana Fox <dana@…> — Add to the report distribution list …

## What activation would do
- Write 4 configuration artefact(s) for NORTHSTAR, as a new governed version.
- Register the expected source deliveries in the production source registry.
- Place 2 file(s) in the production raw location, through the platform's own
  governed intake.
- Start the existing Onboarding Agent, which will profile, map, transform,
  validate and assemble the delivery.
```

`GET /ops/agent/cases/{ref}/classification` returns the operator's category
view: every field, its category, its reason and its provenance.

---

## 4. How structured client answers are persisted

```
  ┌──────────────────────┐
  │ OCC Agent            │  the operator's instruction, in their own words
  │ conversation         │
  └──────────┬───────────┘
             │ extraction + planning (LLM-free, catalogue-derived)
             ▼
  ┌──────────────────────┐
  │ onboarding case      │  pre-populated answers + provenance_class
  └──────────┬───────────┘
             │ classification: 5 categories
             ▼
  ┌──────────────────────┐
  │ client_form.build()  │  ONLY category 2, conditional + progressive
  └──────────┬───────────┘
             │  GET /ops/agent/cases/{ref}/form
             ▼
  ┌──────────────────────────────────────────────────────┐
  │ structured client form                               │
  │  key = "contacts.reporting_contact_email"            │  ← authoritative
  │  key = "portfolios[0].portfolio_type"                │     catalogue keys
  └──────────┬───────────────────────────────────────────┘
             │  POST /ops/agent/cases/{ref}/form
             ▼
  ┌──────────────────────┐
  │ plan_response()      │  PURE. No interpreter import — a test asserts it.
  │  · key not in the catalogue      → UnknownAnswerKey     (nothing saved)
  │  · key not on the served form    → NotAClientQuestion   (nothing saved)
  │  · value outside declared options→ RESPONSE_INVALID     (nothing saved)
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ OnboardingService    │  save_step(step, payload) — the platform's own
  │ .save_step()         │  writer, with its validation, inference and events
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ governed case        │  answers + provenance_class = "client_supplied"
  │                      │  + audit: client_response_submitted
  └──────────┬───────────┘
             │
             ▼   the agent resumes: follow-ups, contradictions, exceptions
  ┌──────────────────────┐
  │ human review         │  review package → approval → confirmation
  └──────────────────────┘
```

**The agent is used for** generating and pre-populating the form, interpreting
optional free-text explanations, explaining questions, spotting contradictions
and drafting targeted follow-ups. **It is not used** to decide what a structured
answer meant.

Two structural guarantees, both tested:

* `test_the_form_module_cannot_reach_an_interpreter` parses `client_form.py`'s
  imports and asserts it cannot reach `interpretation`, `extraction` or
  `planning`;
* `test_submitting_a_response_never_calls_the_interpreter` replaces the
  service's interpreter with one that raises, then submits the answer *"map
  balance to nonsense"* — it is stored as that string, verbatim.

### There is no secure external client portal

Nothing in Trakt today serves a page to a client, authenticates one, or accepts
a submission from outside the operator's network. **This is stated plainly
rather than implied.** What exists is the domain and the API contract such a
portal needs: the form definition, the authoritative keys, the validation and
the deterministic persistence. The OCC's own operator-facing surface renders it
in the meantime, and an operator types what a client sends back.

Building the portal is an infrastructure and identity task — transport,
authentication, tenant isolation, rate limiting, file upload — and no part of it
was attempted here. Critically, **no separate questionnaire system was built**:
the portal, when it arrives, serves `GET /form` and posts to `POST /form`.

---

## 5. Responsibility matrix

| Capability | OCC Agent | Client form | File transfer (`upload_batch_files`) | Onboarding Agent |
|---|:---:|:---:|:---:|:---:|
| Interpret the operator's instruction | ● | | | |
| Decide what to ask the client | ● | | | |
| Present questions / collect answers | | ● | | |
| Validate an answer against the catalogue | ● | ● | | |
| Persist answers to the governed case | ● | | | |
| Tell the client which files are required | ● | | | |
| Generate delivery instructions | ● | | | |
| Record expected files | ● | | | |
| Associate received files with the case | ● | | | |
| Verify required artefacts are **present** | ● | | | |
| Obtain human approval | ● | | | |
| Activate the approved configuration | ● | | | |
| Place files in the governed location | | | ● | |
| Register a file in the intake | | | ● | |
| Start ingestion | ● *(instructs)* | | | ● *(performs)* |
| **Inspect a source schema** | ○ | | | ● |
| **Create canonical mappings** | ○ | | | ● |
| **Transform loan data** | ○ | | | ● |
| **Validate a loan tape** | ○ | | | ● |
| Assess materiality | ○ | | | ● |
| Assemble the canonical | ○ | | | ● |

● owns it · ○ **must not, and is tested for**

### The rehearsal, precisely

`execution.py` *does* profile, map, transform and validate — by calling
`file_profiler.profile_file`, `semantic_alignment.HeaderMapper`,
`canonical_transform.apply_types` and `validate_business_rules.run_rules` over
an isolated sandbox. That is **reuse**, not duplication, and the tests pin the
distinction down:

* `test_no_coordination_module_processes_a_file` — parametrised over all 15
  coordination modules, asserting none mentions `read_csv`, `DataFrame`,
  `profile_file`, `HeaderMapper`, `apply_types`, `run_rules`,
  `semantic_alignment`, `canonical_transform` or `validate_business_rules`;
* `test_the_rehearsal_imports_the_real_components` — the inverse, on
  `execution.py`: if it stopped importing them, that would mean it had grown its
  own;
* `test_the_rehearsal_defines_no_mapping_or_validation_of_its_own` — no
  `_map_headers`, `_infer_mapping`, `_apply_types`, `_fuzzy_match`, `_score_alias`;
* `test_the_live_adapter_makes_no_other_outbound_call` — an AST walk enumerating
  every call the live adapter makes on an injected collaborator, asserted equal
  to exactly the four governed calls.

---

## 6. `create_batch` / `upload_batch_files` / `start_batch`

### Which model is intended

**Model B: the OCC provides a thin governed transfer mechanism that places
submitted files into Blob before the Onboarding Agent starts.**

This is not a choice made here — it is what the platform already implements, and
what the operator's existing manual-delivery screen already uses.
`OpsEngine.upload_batch_files` says so in its own docstring:

> Take files uploaded by an operator into the **SAME governed intake the blob
> trigger uses**. The destination is derived here from the batch's own
> controlled fields — the browser sends file content and a name, never a
> location.

That is precisely "a thin governed transfer mechanism", and it is infrastructure:
it derives the destination, sanitises the filename, writes the object and
registers it. It does not read the file.

Model A is also supported by the platform — files landing in the governed
location directly are picked up by the blob trigger — and the OCC Agent does not
need to implement it, because a case that has files in hand should not have to
put them somewhere and wait to be noticed.

### The sequence, and what each call is for

| Call | Owner | What it does | Reads file contents? |
|---|---|---|---|
| `OnboardingService.activate()` | Client Onboarding | writes the versioned configuration, registers sources | no |
| `engine.create_batch()` | OCC intake | opens a governed input pack; refuses an outcome/dataset combination that could route a pipeline book into regulatory reporting | no |
| `engine.upload_batch_files()` | OCC intake | derives the destination, sanitises, writes, registers; readiness assessed after the whole pack is present | no |
| `engine.start_batch()` | OCC intake | resolves the effective configuration, writes the immutable run manifest, starts governed execution | no |
| — | **Onboarding Agent** | profile, map, transform, validate, stamp, assemble | **yes** |

### A defect this review found and fixed

The live adapter was calling `create_batch` **without `workflow_type`**, which
the engine requires and which decides whether a delivery may reach regulatory
reporting at all. It would have raised `TypeError` on the first live activation.

It survived because the test fake accepted `**kwargs`. Both are fixed:
`ActivationIntent` now carries `outcome` and `cadence` from the confirmed
intent, and `FakeEngine` declares the engine's real signatures with
`test_the_adapter_calls_the_engine_with_the_engines_own_signature` asserting
they still match. A fake that accepts anything cannot catch a caller that has
drifted.

---

## 7. Confirmation: no logic duplicated

| Concern | Where it lives | The OCC Agent |
|---|---|---|
| Field catalogue | `config/onboarding/field_catalogue.yaml` | reads; added two sections and one axis |
| Validation | `onboarding/validator.py` | delegates |
| Inference / derivation | `onboarding/inference.py`, `derivation.py` | delegates |
| Case lifecycle | `onboarding/case.py` | never restates — asserted |
| Configuration generation | `onboarding/generation.py` | delegates via `preview` / `activate` |
| Activation | `OnboardingService.activate()` | calls; never reimplements |
| Source profiling | `file_profiler` | rehearsal calls it |
| Header mapping | `gate_1_alignment.semantic_alignment` | rehearsal calls it |
| Canonical transform | `gate_2_transform.canonical_transform` | rehearsal calls it |
| Business rules / materiality | `gate_3_validation`, `issue_policy.yaml` | rehearsal calls it |
| Orchestration | `orchestrator_agent.run_orchestration` | rehearsal calls it |
| Assembly | `engine.assembler_agent` | rehearsal calls it |
| Governed intake | `OpsEngine.create_batch/upload/start` | live adapter calls it |
| Blob path rules | `manual_intake.derive_raw_prefix`, `path_parser` | reads |

New code added by this change is coordination and presentation only:
classification (which category a field is in), the client form (which questions
to show and where the answers go), the pack, the review package and the
activation gate. None of it reads a data file.

---

## 8. Test results

| Suite | Result |
|---|---|
| `tests/operations_control/occ_agent` | **489 passed** |
| — of which new here | `test_client_experience.py` 34, `test_boundary.py` 23 |
| Frontend `vitest run` | 143 passed |
| `npm run lint` (`tsc --noEmit`) | clean |

### Whole-repo comparison

| | failed | passed | skipped | errors |
|---|---:|---:|---:|---:|
| Base `3d507dc` (before any of this work) | 45 | 3451 | 32 | 10 |
| HEAD | 45 | 3711 | 32 | 10 |

Identical failure and error counts. The 45 are pre-existing branch-vs-`main`
drift — the largest group is `test_no_regulatory_or_annex2_files_modified`,
which diffs the branch against `main` and forbids changes under `config/regime/`,
`engine/gate_*` and similar; this change touches none of those.
