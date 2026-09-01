# MI Query Live Telemetry — Day-1 Calibration Feedback Loop

**Branch:** `claude/mi-query-live-telemetry`
**Starting SHA:** `d355065b9c502c8eee8575a68cda159735d496fc`
(the Day-1 hardening tip `e78e648` merged with `origin/main` `95dbbda`)

**Objective.** Capture how real users interrogate their portfolio through the MI
Query Agent from their first question onwards, so that live questions become
governed evidence for improving recognition, interpretation and capability
coverage — without changing a single answer the agent gives.

**Scope discipline.** This sprint records; it does not calculate, judge or
intervene. It is not the OCC system-monitoring dashboard, and no uptime, run,
gate, health or publication monitoring was built.

---

## 1. The actual production MI Query path

Before deciding where telemetry belongs, the real path was traced rather than
assumed. Every channel converges on one governed function:

```
React MI Agent  ──┐
M365 Copilot    ──┤
Python / job    ──┼──►  mi_agent_api.mi_service.execute_governed_mi_query(request, context)
future agent    ──┘
```

`mi_agent_api/mi_service.py` is explicitly documented as **the** governed MI
capability (`capability id: mi.question.answer`), with every interface an adapter
over it. Inside one execution the stages run in this order:

| Stage | Where | What it produces |
|---|---|---|
| Scope check | `trakt_core.context.ExecutionContext` | authorised caller, or `SCOPE_MISSING` |
| Portfolio authorisation | `trakt_core.tenancy` | `AuthorisedPortfolio`, or `PORTFOLIO_NOT_AUTHORISED` |
| Source-approval policy | `trakt_core.policy` | approved / `DATA_SOURCE_NOT_APPROVED` |
| Dataset resolution | `mi_agent_api.datasets` | the snapshot actually read |
| **Interpretation** | parser → `spec` | the structured question |
| **Execution** | routing → deterministic capability | the calculated payload |
| **Answer** | narration / artifacts | the words and charts the user saw |
| Governed envelope | `trakt_core.envelope.GovernedResult` | status, error code, audit metadata |
| Audit | `trakt_core.audit.emit_audit_event` | one `trakt.audit` log line |

Two things follow, and they determined the whole design:

1. **The architectural separation the brief asks to preserve already exists in
   the data.** By the time execution reaches the end of `execute_governed_mi_query`,
   the question, the structured interpretation (`result["spec"]`), the executed
   route, the answer text and the governed outcome are all separately addressable
   fields of one object. Telemetry does not have to reconstruct or infer any of
   them — it projects what the pipeline already decided.
2. **There is exactly one seam.** Instrumenting anywhere else would either miss a
   channel or duplicate itself across three.

## 2. Where telemetry is captured

At the single point where a governed MI execution finishes and audits itself.

`mi_service.py` previously ended each of its four exit paths with
`emit_audit_event(result); return result`. All four now return through one
helper:

```python
def _finish(result, request):
    emit_audit_event(result)                 # unchanged — Application Insights
    try:
        from . import query_telemetry
        from operations_control.stores import OpsStore
        query_telemetry.record(OpsStore.from_env(), result,
                               question=request.question,
                               requested_portfolio=request.effective_portfolio_id())
    except Exception:
        logger.warning("mi query telemetry unavailable for request_id=%s", ...)
    return result
```

Three properties are deliberate:

* **It is downstream of everything.** The answer has already been computed,
  guarded and returned-shaped. Telemetry observes a finished result; it cannot
  reach back into parsing, routing or calculation.
* **It cannot fail a query.** `record()` is itself wrapped in a bare `except`,
  and the call site is wrapped again. A storage outage costs a telemetry record,
  never an answer.
* **It records nothing the model "thought".** Only `GovernedResult` fields and
  the explicit structured `spec` the parser emitted. No prompts, no
  chain-of-thought, no hidden reasoning, no tokens — none of which are available
  at this seam in any case.

## 3. Storage split: OCC vs Azure

No new storage subsystem was introduced. The audit of existing storage found a
governed, client-scoped, day-partitionable document store already serving the
Operations Control Centre (`OpsStore` / `OpsLayout`, `operations_control/stores.py`),
sitting in the `operations-control` blob container behind the same tenancy rules
as every other OCC document. It supports hundreds to thousands of records,
user/time filtering, a 72-hour view, a long-term corpus and client isolation —
so it is what telemetry writes to.

The split, in evidence. A single live question,
*"What is the weighted average LTV?"*, produced both of the following:

**Application Insights — the `trakt.audit` line (unchanged by this sprint):**

```json
{"actor_id": "alice@lender.example", "actor_type": "user",
 "capability": "mi.question.answer", "channel": "react", "duration_ms": 890,
 "error_code": null, "outcome": "success", "portfolio_id": "client_001",
 "request_id": "req_cff87df2dd604479", "schema_version": "1.1.0",
 "snapshot_id": "SYNTHETIC_ERE_Portfolio_012026_canonical_typed@949d6a19d19e",
 "source_kind": "synthetic_demo", "started_at": "...", "tenant_id": "client_001"}
```

Checked directly: the question text is **not** in that line, and the answer is
**not** in that line. `trakt_core/audit.py` enforces this with a
`_FORBIDDEN_KEYS` set covering `answer`, `result`, `rows`, `artifacts`, tokens,
URLs and paths.

**The governed OCC store — the telemetry record** (same execution) carries the
exact question, the exact answer, the interpretation, the capability, the data
version and the outcome, client-scoped at
`{client_id}/mi-queries/{day}/{query_id}.json`.

So: **operational metadata to Application Insights; user-visible question and
answer content to the governed store, and only there.** The full question and
answer are not sprayed into ordinary application logs merely because Application
Insights exists.

## 4. Exact telemetry schema

`mi_agent_api/query_telemetry.py`, `SCHEMA_VERSION = "1.0.0"`. One JSON document
per question. Fields, grouped by the stage they come from — the architectural
separation is visible in the record itself:

**Identity — who asked, and when**
`query_id`, `asked_at`, `day`, `client_id`, `portfolio_id`, `user_id`,
`user_type`, `channel`, `organisation_id`, `request_id`, `correlation_id`

`user_id` is `AuditMetadata.actor_id`, which comes from the authenticated
`ExecutionContext` — never from a client-supplied string in the request body.

**The question — what they asked**
`question` (verbatim, as typed)

**Data context — which version was queried**
`snapshot_id`, `content_hash`, `source_kind`, `reporting_period`,
`dataset_view`, `data_source_kind`, `data_source_label`

**Interpretation — what Trakt thought the question meant**
`interpretation` (the structured `spec`), `parser`

`interpretation` copies only keys the spec actually carries, from a fixed list
covering population (`portfolio_lens`, `segment`, `state_filters`, `filters`),
measurement (`metric`, `measures`, `aggregation`, `weight_field`), cut
(`dimension`, `hierarchy`, `bucket_field`, `concentration_dimension`,
`ranking_mode`, `top_n`, `sort_by`), period (`as_of_date`, `start_date`,
`end_date`, `compare_periods`, `temporal_mode`, `trend_grain`, `cohort_grain`),
requested shape (`intent`, `output_type`, `chart_type`, `execution_mode`) and
what the parser could not honour (`unavailable_filters`, `metric_defaulted`).

A route that exposes no structured spec records an **empty** interpretation
rather than a fabricated one. The absence is itself a finding an operator should
see; inventing an interpretation would be the one way this telemetry could lie.

**Execution — which deterministic capability answered it**
`route`, `capability`, `engine`, `execution_mode`, `result_type`, `row_count`,
`lens_applied`, `artifact_kinds`

`route` is `spec.route_id` where a routed capability ran, falling back to the
metadata `route`. It is recorded as it is rather than back-filled, so "no named
route" stays visible.

**Answer — what Trakt actually told the user**
`answer` (the exact narration text the user saw)

**Outcome — answered, refused or errored**
`outcome`, `governed_status`, `refusal_reason`, `error_code`, `error_category`,
`message`, `warnings`

**Performance**
`duration_ms` (from `AuditMetadata`, the timing already measured)

**Quality review — set only by a human, later**
`review: {classification, reviewer, reviewed_at, note}`, initialised to
`classification: "UNREVIEWED"`.

### How ANSWERED / REFUSED / ERROR is derived

From the existing governed vocabulary only — nothing new was invented:

* `status == "success"` → **ANSWERED**
* `CALCULATION_FAILED`, or any error whose `ErrorCategory` is `INFRASTRUCTURE`
  → **ERROR**
* everything else with an error code → **REFUSED**

The reasoning: a capability-level non-delivery (`UNSUPPORTED_QUESTION`,
`AMBIGUOUS_QUESTION`, `NO_MATCHING_RECORDS`, `DATA_SOURCE_NOT_APPROVED`,
`PORTFOLIO_NOT_AUTHORISED`) is a *governed refusal* — the request was well formed
and correctly authorised, and the governed answer is "we did not compute one". A
broken calculation or failed infrastructure is a *fault*. Conflating them would
make the Day-1 refusal analysis useless, because genuine capability gaps would
hide inside an error rate.

### What is never recorded

Model reasoning or chain-of-thought · prompts · tokens or credentials · stack
traces · connection strings, signed URLs or paths · anything the audit event's
forbidden-key list already excludes.

## 5. OCC query-review UI

A new screen, `MI Query usage` (`/mi-queries`), in the existing OCC console —
added to both the manual-first and agent-first navigation hierarchies.

* **Window tabs**: 24h · 72h · 7d · all.
* **Counters**: total questions, unique users, answered / refused / errors with
  percentages, median and p95 latency, and the review counters described in §6.
* **Filters**: client, outcome, user, portfolio, review state (including a
  `PROBLEMATIC` shorthand), and free-text over the question.
* **Query log**: one row per question — time, user, channel, the question,
  outcome, route, refusal reason, latency, review state.
* **Detail panel**, laid out along the same separation as the pipeline:
  *Question* → *What Trakt understood* → *Which capability ran* → *Which data
  version* → *Answer* → *Outcome* → *Quality review*.
* **Review control**: the eight classifications and a free-text note.

Backend routes, all on the OCC API, all behind the same `require_client` guard as
the rest of it:

| Route | Purpose |
|---|---|
| `GET /ops/mi-queries/summary` | window counters |
| `GET /ops/mi-queries` | filterable query log |
| `GET /ops/mi-queries/{query_id}` | full record |
| `POST /ops/mi-queries/{query_id}/review` | record an operator judgement |
| `GET /ops/mi-queries/export/calibration` | the safe calibration export |

**Relation to the future OCC monitoring dashboard.** `mi_query_routes.py` reads
one governed record type and knows nothing about runs, gates, publications or
service health. It can therefore later become one module of the wider operations
console — the dashboard can link to `/mi-queries` — without the dashboard having
to absorb it or this having to reach into the dashboard's concerns.

## 6. Quality-review model

**The MI Query Agent never marks its own answer correct.** Nothing in the write
path can set `review.classification`; the field is initialised `UNREVIEWED` and
changed only by `POST /ops/mi-queries/{id}/review`, which stamps the
authenticated operator's name and the time.

The classification list is closed. An unrecognised value is rejected with
`OPS_BAD_CLASSIFICATION`:

`CORRECT` · `WRONG_INTERPRETATION` · `WRONG_CALCULATION` · `RENDERING_ERROR` ·
`PARTIALLY_CORRECT` · `APPROPRIATE_REFUSAL` · `SHOULD_HAVE_ANSWERED` ·
`NEEDS_INVESTIGATION`

**A review never changes the answer already given to the client.** The record's
`answer` field is written once at execution time and is not touched by the review
route; the review writes only into the `review` block. The answer was served
before any review existed, and a later classification is calibration evidence
about it, not a correction of it. This is covered by a dedicated test.

Every review is written to the client's hash-chained OCC audit trail as
`mi_query_reviewed`, with the reviewer, the query id and the classification.

### How correctness is reported

Only over what was actually reviewed, with the denominator travelling beside it.
The summary returns:

```
total_questions, answered, refused, errors,
reviewed, unreviewed, reviewed_correct, reviewed_problematic,
reviewed_correctness_pct, review_breakdown{...}
```

so the UI states, for example:

> Reviewed responses: 74 · Correct: 72 · Problematic: 2 · **Reviewed correctness: 97.3%**

and never "AI accuracy = 98%". `reviewed_correctness_pct` is computed with
`len(reviewed)` as its denominator and is `null` when nothing has been reviewed —
there is no arithmetic path by which an unreviewed corpus can produce a
confidence figure. (`CORRECT` and `APPROPRIATE_REFUSAL` both count as correct: a
refusal Trakt was right to make is a right response.)

## 7. Client isolation

* Records are stored under a client-scoped key, `{client_id}/mi-queries/...`, in
  the same container and under the same layout discipline as every other OCC
  document.
* Every route — summary, log, detail, review and export — resolves the client set
  through `principal.visible_clients(...)` and then calls `require_client()` on
  each. A detail read for a query id belonging to another client returns
  `OPS_NOT_FOUND`, not the record.
* A query with no tenant is **not** recorded at all: the store is client-scoped,
  and a record with no client could not be isolated.
* `record()` calls the existing idempotent `store.register_client()` so a client
  that asks MI questions before it has an OCC workflow still appears in the
  operator's client index — otherwise its questions would be recorded and then
  invisible.
* OCC/admin access is the OCC API's existing authentication; no new, weaker path
  to this data was created.

## 8. Privacy boundary

The position being protected: **portfolio rows, portfolio values, aggregates and
calculated answers are not sent to Anthropic / Claude for MI Query
interpretation.** This sprint does not change that, and adds nothing that could.

* **Inside the governed environment**, telemetry stores the exact answer. That is
  the point — an operator cannot judge whether a response was right without
  seeing what the user was told.
* **No workflow in this sprint sends anything to an external model.** There is no
  automatic export, no scheduled job, no outbound call. The calibration export is
  a read-only endpoint an operator invokes deliberately.
* **The export defaults to the safe metadata form** and is labelled as such:
  `export_kind: "external_model_safe"`.

The two forms are not silently mixed: the governed record and the export are
different shapes produced by different code paths, and the export is built by
naming the fields it includes rather than by removing fields from the record — so
a new field added to the record in future is excluded from the export by default,
not included by accident.

## 9. Calibration export

`GET /ops/mi-queries/export/calibration?window=72h&reviewed_only=true`

**Included** (all of it recognition and routing evidence, none of it portfolio
content): `query_id`, `asked_at`, `question`, `interpretation`, `parser`,
`route`, `capability`, `outcome`, `refusal_reason`, `error_code`,
`dataset_view`, `reporting_period`, `snapshot_id`, `quality_classification`,
`reviewer_note`.

**Excluded by construction**, and declared in the response body under
`excludes`: the answer text · artifacts · aggregate values · deterministic
payload values · loan rows · portfolio values · the snapshot `content_hash`.

`reviewed_only` defaults to `true`: an unreviewed question carries no verdict, so
it is not calibration evidence yet.

The governed record still holds the answer for operator review inside OCC. It is
this export, and only this export, that is shaped to be safe to hand to an
external model — and even then, handing it over remains a deliberate human act,
not something the system does.

## 10. First-72-hour workflow

The intended Day-1 loop, using only what now exists:

1. Open **MI Query usage**, window **72h**.
2. Read the counters: how many questions, how many users, and the
   answered / refused / error split.
3. Filter to **REFUSED** and group by refusal reason. `AMBIGUOUS_QUESTION` and
   `UNSUPPORTED_QUESTION` concentrations are the capability-coverage signal —
   repeated unsupported question patterns are visible by filtering to refusals
   and reading the questions, which are stored verbatim.
4. Filter to **ERROR**. These are faults, and are a different queue from
   refusals: they are defects, not gaps.
5. Sample the **ANSWERED** ones. For each, compare *what Trakt understood*
   against *what the user asked*, then the capability and the answer. Classify.
6. Where a refusal should have been an answer, classify `SHOULD_HAVE_ANSWERED` —
   that is the capability backlog, generated from real usage rather than
   speculation.
7. At the end of the window, export the reviewed set for calibration.

The counters answer the owner's Day-1 questions directly: who asked, what they
asked, what Trakt gave them, what Trakt thought they meant, which capability
answered, whether it answered / refused / errored, how long it took, which data
version, and — after review — whether it was any good.

## 11. Tests and regression

### New tests

`tests/operations_control/test_mi_query_telemetry.py` — **32 tests**, mapped to
the brief's acceptance list:

| | Requirement | Evidence |
|---|---|---|
| A | Answered query | user, exact question, exact answer, interpretation, capability, data version, `ANSWERED`, latency, `UNREVIEWED` initially — all asserted; plus "no model reasoning or secrets" and "an interpretation is never invented" |
| B | Refused query | structured refusal reason recorded, **no fake answer** |
| C | Error query | `ERROR` not `REFUSED`, query/correlation id present, no stack trace, outcome split follows the existing error vocabulary |
| D | Human review | classification + note recorded; **the user's answer is unchanged**; invented classifications refused; the review is audited |
| E | Client isolation | both directions, cross-client detail read refused, export scoped |
| F | Data version | a query names the exact snapshot it was answered from |
| G | Period filter | the window excludes older questions; summary counts what the window holds; correctness only over reviewed |
| H | Calibration export | carries no portfolio content; carries what calibration needs; reviewed-only by default; the governed record still holds the answer |
| I | Parity | see below, plus: telemetry never fails a query, nothing recorded without a configured store, no tenant → no record |

### I — Parity, proven directly

A fixed set of 12 real questions (8 answerable, 1 capability error, 2 refusals, 1
cross-cut) was run through `execute_governed_mi_query` — the same entry point
every channel uses — three times: twice with telemetry **off**, once with it
**on**. The full analytical payload was captured each time and compared.

The only run-to-run differences were per-run artefact identifiers (`art_…`,
`kpi_…`) and artefact `createdAt` timestamps. These were shown to be ordinary
noise, not telemetry: **two consecutive telemetry-off runs differ in exactly the
same 41 places, of exactly the same kind.** With those normalised, all three runs
hash identically:

```
telemetry OFF  run 1 : dae9347a064070b63aa80a18f9e992308b17fa8ea86c1701c702247b870dd9e4
telemetry OFF  run 2 : dae9347a064070b63aa80a18f9e992308b17fa8ea86c1701c702247b870dd9e4
telemetry ON         : dae9347a064070b63aa80a18f9e992308b17fa8ea86c1701c702247b870dd9e4
```

Status, error code, capability, warnings, interpretation, route, row counts and
the exact answer text are identical in every case, refusals and the error
included. During the ON run, 12 telemetry records were written — so the
instrumentation was demonstrably active while making no difference.

### Regression counts

| Suite | Result |
|---|---|
| `tests/operations_control/` (OCC backend, incl. the 32 new tests) | **1133 passed, 0 failed** |
| `mi_agent_api/tests/` | 1417 passed, 1 skipped, **5 failed — all pre-existing** |
| MI Query acceptance suites in `tests/` (phase7 spec v2, phase8a interpreter harness, capability registry, recognition diagnosis, portfolio lens wiring, governed MI integration defects, MI render) | 217 passed, **5 failed — all pre-existing** |
| `frontend/operations-control-ui` (Vitest) | 23 files, **208 passed** |
| `tsc --noEmit` | clean (apart from a pre-existing `tsconfig` `baseUrl` deprecation warning) |

**Pre-existing failure baseline, verified not assumed.** The 10 failures were
reproduced at the clean merge base `d355065` in a separate `git worktree`, with
the same test ids and the same assertions, before any file in this sprint
existed:

*`mi_agent_api/tests/`* — `test_chat_routing_e2e.py::test_cumulative_cohort_conversion_routes`,
`test_currency_authority.py::test_client_1_gbp_comes_from_the_governed_client_configuration`,
`test_pipeline_stage_transition.py::TestClassification::test_a_new_arrival_is_not_given_a_prior_stage_it_never_had`,
`test_pipeline_stage_transition.py::TestClassification::test_a_prior_only_identifier_is_a_departure`,
`test_single_parse_and_substitution.py::test_an_unavailable_dimension_is_refused_not_substituted`

*`tests/test_phase8a_mi_interpreter_harness.py`* —
`test_golden_example[compare funded balance to last month]`,
`test_golden_example[show changes]`,
`test_supported_valid_questions_validate[compare funded balance to last month]`,
`test_ambiguous_questions_clarify[show changes]`,
`test_interpreter_always_validates_supported_specs`

The baseline run was `5 failed, 217 passed` and this branch's run is
`5 failed, 217 passed` — the same five. **This sprint introduced no new test
failures.**

## 12. Intentionally deferred

Not built, deliberately:

* **The OCC system-monitoring dashboard** — uptime, run, gate, health and
  publication monitoring. Explicitly out of scope; §5 explains how this screen
  stays linkable from it later.
* **Any change to MI Query behaviour** — parser, interpretation, routing,
  capability selection, deterministic calculation, refusal policy, narration,
  supported question surface or answer wording. Proven unchanged in §11.
* **Automatic calibration** — no mechanism feeds reviewed questions back into the
  parser, and no export leaves the environment on its own. The loop is closed by
  a human deciding to close it.
* **Retention and archival policy** — records accumulate with no expiry. Flagged
  rather than guessed: choosing how long a client's questions and answers are
  kept is an owner policy decision, not a developer default, and the wrong
  default here is a privacy decision made by accident. Recommend setting this
  before the corpus grows.
* **Cross-client aggregate analytics** — deliberately excluded; every route is
  client-scoped, and a cross-tenant view would need its own authorisation model.
* **Free-text search at scale** — the current filter scans the window's records,
  which is right for hundreds to thousands and would not be for millions. No new
  database was introduced, per the brief; revisit only with evidence of volume
  the existing store cannot serve.
* **Alerting on refusal spikes** — monitoring, not telemetry. Belongs with the
  future dashboard.

---

## 13. Final answers

**1. Where is detailed MI Query telemetry stored?**
In the existing governed OCC document store (`OpsStore`, the `operations-control`
blob container), client-scoped and day-partitioned at
`{client_id}/mi-queries/{day}/{query_id}.json`. No new storage subsystem was
introduced.

**2. What is stored in Azure / Application Insights?**
Exactly what was already stored: the one `trakt.audit` line per execution —
actor, actor type, channel, capability, request id, correlation id, tenant,
portfolio, outcome, error code, snapshot id, source kind, start time, duration.
No question text, no answer text, no figures. Unchanged by this sprint.

**3. Can I see user + exact question + exact answer?**
Yes, all three, in the OCC record and on the query detail panel. The user comes
from the authenticated identity, never from a request field.

**4. Can I see how Trakt interpreted the question?**
Yes — the structured `interpretation` block: metric, aggregation, weighting,
filters, dimension, period, requested output shape and anything the parser could
not honour. Where a route publishes no structured spec, the interpretation is
recorded empty rather than invented.

**5. Can I see which deterministic capability answered it?**
Yes — `route` (the routed capability id), `capability`, `engine`,
`execution_mode`, `result_type`, `row_count`.

**6. Can I distinguish answered / refused / error automatically?**
Yes, and it is derived from the existing governed status and error vocabulary
rather than guessed: success → ANSWERED; `CALCULATION_FAILED` or any
infrastructure-category error → ERROR; every other governed error code →
REFUSED, with the reason recorded.

**7. Can I manually classify correct / wrong / etc.?**
Yes — the eight fixed classifications plus a note, via
`POST /ops/mi-queries/{id}/review` and the UI. The agent never classifies its
own answer; records start `UNREVIEWED`; and the review never alters the answer
the client was given.

**8. Can I reproduce which data version was queried?**
Yes — `snapshot_id`, `content_hash`, `source_kind`, `reporting_period`,
`dataset_view`, `data_source_kind`, `data_source_label`.

**9. Can I filter to the first 72 hours?**
Yes — a 72h window on both the summary and the log (also 24h, 7d and all),
applied over day partitions and then per-record timestamps.

**10. Can I identify repeated unsupported question patterns?**
Yes — filter to `outcome=REFUSED` and read the reasons; `UNSUPPORTED_QUESTION`
and `AMBIGUOUS_QUESTION` concentrations are the capability-coverage signal, and
the question text is stored verbatim so the pattern is readable. Free-text search
over questions is also available.

**11. Can I export reviewed questions for future calibration?**
Yes — `GET /ops/mi-queries/export/calibration`, reviewed-only by default,
client-scoped, and labelled `export_kind: "external_model_safe"`.

**12. What information is excluded from any external-model-safe export?**
The answer text, artifacts, aggregate values, deterministic payload values, loan
rows, portfolio values and the snapshot content hash. The export is built by
naming what it includes, so future record fields are excluded by default rather
than leaked by accident. Nothing is sent to any external model automatically; no
such workflow was created.

**13. Did any MI Query behaviour change?**
No. Parser, interpretation, routing, capability selection, deterministic
calculation, refusal policy, narration, supported question surface and answer
wording are untouched. Proven by direct A/B: 12 questions run with telemetry off
(twice) and on, all three producing the identical payload digest
`dae9347a064070b63aa80a18f9e992308b17fa8ea86c1701c702247b870dd9e4` once
per-run artefact ids and timestamps — shown to differ identically between two
telemetry-off runs — are normalised. 12 telemetry records were written during
the ON run, so instrumentation was demonstrably active while changing nothing.

**14. Regression counts.**
OCC backend `tests/operations_control/`: **1133 passed, 0 failed**.
`mi_agent_api/tests/`: 1417 passed, 1 skipped, 5 failed — all five reproduced at
the clean merge base `d355065`.
MI Query acceptance suites in `tests/`: 217 passed, 5 failed — the baseline run
at `d355065` was the same `5 failed, 217 passed`, same test ids.
Frontend Vitest: 23 files, 208 passed. **No new failures introduced.**

**15. Commit SHA and pushed branch.**
Pushed to branch `claude/mi-query-live-telemetry`, as the single commit on top
of the starting SHA `d355065b9c502c8eee8575a68cda159735d496fc` recorded at the
head of this report. Not merged to main; no pull request opened. (The commit SHA
itself is quoted in the sprint response rather than inside the file it commits,
which cannot contain its own hash.)

---

# Appendix — Operator explanation

## When a user asks an MI question, what can I now see?

When somebody at a client asks Trakt a question about their portfolio, Trakt now
keeps a record of it. You can open that record in the Operations Control Centre,
under **MI Query usage**.

For every question asked, you can see:

* **Who asked it** — the signed-in person, and which client they belong to.
* **When**, and how long Trakt took to reply.
* **What they actually typed**, word for word.
* **What Trakt thought they meant** — the measure it settled on, how it added the
  numbers up, how it split them, and the date it used. This is the most useful
  part when something has gone wrong, because it usually shows you *where* it
  went wrong.
* **Which part of Trakt answered it** — the specific calculation that ran.
* **Which version of the client's data** the answer came from, so you can go back
  to the same figures later.
* **What Trakt told them**, word for word.
* **Whether Trakt answered, declined, or hit a problem.** Declining is not a
  fault: it means Trakt was asked something it does not cover, or something too
  vague to be sure about, and said so rather than guessing. A problem is
  different — that is something to fix.

### Judging whether the answer was any good

Trakt does not mark its own homework. Every record starts as **not yet
reviewed**, and stays that way until a person reads it and says what they think.
You choose from a short fixed list — right, wrong interpretation, wrong
calculation, display problem, partly right, right to decline, should have
answered, needs investigating — and you can add a note.

Two things to know about this:

* **Your review never changes what the client was told.** That answer was given
  at the time and stands. Your review is a note about it, for improving Trakt.
* **The figures never overstate what has been checked.** The screen says
  something like *"Reviewed: 74 · Correct: 72 · Reviewed correctness: 97.3%"* —
  always with the number reviewed shown next to it. It will never tell you Trakt
  is "98% accurate" on the strength of a handful of checked answers.

### What to do in the first few days

Set the window to the last 72 hours and work through three groups:

1. **The ones Trakt declined.** If the same kind of question keeps being declined,
   that is your list of what to teach Trakt next.
2. **The ones that hit a problem.** These are faults, and they need fixing.
3. **A sample of the ones it answered.** Read what the person asked, then what
   Trakt understood, then what it said. Mark each one.

### On privacy

The full question and the full answer are kept inside Trakt's own governed
storage, kept separate per client — one client's operator cannot see another
client's questions. The ordinary system logs, which go to Microsoft's monitoring,
carry only timings and identifiers: no questions, no answers, no figures.

If you ever want to use these questions to help improve Trakt's understanding
using an outside AI service, there is a separate export that deliberately leaves
out the client's numbers — no answers, no balances, no totals, no loan data. It
carries only the question, what Trakt understood, which calculation ran, what
happened, and your verdict. Nothing is sent anywhere automatically; that would
always be somebody's deliberate decision.
