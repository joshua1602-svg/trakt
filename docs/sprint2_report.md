# Sprint 2 — the governed credit intelligence layer

*Baseline `86460f3` → `f73cbd1`. 31 files, +6,453 / −86.*

---

## 1. Executive summary

**What Sprint 2 built.** Trakt could already tell an agent whether a portfolio
breached its covenants. It can now tell an agent what is in the portfolio, how it
is concentrated, which loans are worst, what the tape does and does not contain,
what failed validation, what changed since last period — and, for any figure,
*why it is that figure*.

**What changed for humans.** MI Query got materially faster and nothing else
changed. `profile_dataset` was re-deriving the same column semantics on every
call and accounted for roughly 80% of a warm query; it is now memoised on exact
frame identity. A warm 100k-row query went from 1,880 ms to 267 ms. Separately,
loading a canonical CSV now goes through a columnar serving copy: 8.0 seconds to
0.26 seconds at a million rows. Answers were verified byte-identical across four
questions.

**What changed for agents.** One tool became thirteen. More importantly, the
answers changed character. Before Sprint 2 an agent asking for a loan's LTV got
`61.15`. It now gets: 61.15 percentage points, under `CURRENT_LTV@v1`, **not
recomputed by Trakt**, being a stated balance over a *named valuation
observation* chosen by `CURRENT_LTV_SELECTION@v1` — with the observations that
were *not* chosen and the reason each was rejected ("66 months old at 2026-07-31,
and the policy allows 24").

**Is it the same engine?** Yes, and structurally so. There is one canonical tape,
one set of calculations, one authorisation path and one audit trail. The agent
tools are wrappers over implementations the React workspace already calls —
`analytics_lib` for the maths, the concentration engine for covenants, the
period-change workflow for movement. When benchmarking exposed a 166-second
concentration call, the fix went into `analytics_lib` where MI reads it too,
rather than into a fast path for agents.

**The honest caveat.** Two items are designed and not built (valuation sidecar
*writer*, A2A), one significant capability is missing for the Securitisation
Readiness Agent (Annex-coverage readiness), and one known concurrency defect
remains open. All four are in §14.

---

## 2. MI Query optimisation — dataset-profile caching

`profile_dataset(df, semantics)` re-derives per-column semantics — type,
groupability, cardinality, percent scaling — on every call. It is deterministic
in `(df, semantics)`, and MI Query calls it repeatedly for the same frame.

**The cache key is exact frame identity, not a content hash:** `id(df)` guarded
by a `weakref.ref(df)` (so a recycled `id` cannot match a dead frame) plus
`cached_semantics is semantics`. Hashing a million-row frame to decide whether to
profile a million-row frame would spend the saving.

| | before | after | |
|---|---|---|---|
| MI query, 10k rows | 340 ms | 93 ms | **3.6×** |
| MI query, 100k rows | 1,880 ms | 267 ms | **7.0×** |
| MI query, 1m rows | 24,199 ms | 4,383 ms | **5.5×** |
| `profile_dataset` alone, 60k rows | 69.2 ms | 0.004 ms | **~16,000×** |

Correctness was checked by comparing full MI Query artifacts for four questions
before and after. The first comparison showed differences; investigation found
they were `createdAt` timestamps and generated artifact `id`s, which differ
between *any two runs* — confirmed by diffing two cache-off runs against each
other. With volatile keys excluded, all four are identical.

Kill switch `TRAKT_PROFILE_CACHE=0`. 9 tests.

---

## 3. Storage — three roles, deliberately distinct

| | Role | Authoritative? | Lifecycle |
|---|---|---|---|
| **CSV** | The canonical tape and every regulatory artefact | **Yes** | Written by the pipeline; never replaced |
| **Parquet** | A *serving copy* of the canonical CSV | No | Derived, disposable, rebuilt on demand |
| **Valuation sidecar** | 0..n valuation observations per collateral | Yes, where present | Reader implemented; **writer not built** |

**CSV stays the record.** No pipeline stage writes Parquet and no regulatory
artefact becomes Parquet. Deleting the entire serving cache costs one slow load.

**Parquet is keyed on content identity** — path, mtime, size — encoded in the
*filename*. A changed CSV produces a different key, so a stale copy is never
asked for. There is no invalidation to get wrong and no TTL to tune.

| rows | csv | parquet | read_csv | parquet read | speedup |
|---|---|---|---|---|---|
| 10,000 | 2.6 MB | 0.8 MB | 55 ms | 9 ms | **6.5×** |
| 100,000 | 26.0 MB | 7.7 MB | 719 ms | 28 ms | **26.1×** |
| 1,000,000 | 259.6 MB | 60.5 MB | 7,979 ms | 255 ms | **31.3×** |

Every failure path — no pyarrow, unwritable directory, corrupt file, a column
Arrow cannot represent — falls back to `read_csv` and logs. The write is atomic
via `os.replace`, because two workers starting together both miss and both write,
and a third reading a half-written file is worse than having no cache at all.
`assert_frame_equal` pins that the copy *is* the CSV's frame; a serving copy that
differs from its source is not a cache, it is a second dataset. 14 tests.

**The valuation sidecar** is where 0..n observations belong, because a wide row
cannot carry them. The reader (`trakt_core.valuation`) and the model are built
and tested. The *writer* is not — today the observations are derived from the
tape's own columns (current, original, indexed), which yields a truthful history
of one to three rather than an empty one. See §14.

---

## 4. The entity model

Six entities, in `config/system/entity_model.yaml`, declared only where the
canonical fields genuinely support them:

    loan
     ├── borrower       1..1   (borrower_identifier is a FOREIGN key — several
     │                          loans share one, which is what the concentration
     │                          engine's multi-loan-borrower metrics rely on)
     ├── collateral     1..1
     │    └── valuation 0..n   ← the ONLY repeating entity, and the reason the
     │                          model exists at all
     ├── contract       attribute group
     └── performance    attribute group

`contract` and `performance` are marked `attribute_group: true`: no independent
identity, never repeat. Promoting them to entities would add indirection with no
expressive gain, so the model says what they are.

**Three things it deliberately is not.** Not a second store — assembly happens
per request from the row already in memory, so there is no materialised object
database to keep in step and no second answer to "what is the balance". Not a
normalisation — no canonical field moves, no pipeline changes, and a field
claimed by no entity is still returned (in an `other` block) rather than dropped.
Not an ontology — six entities, and no more.

A field claimed by two entities fails the model closed, because a value appearing
in two sections is ambiguous about which is authoritative.

---

## 5. Valuation — facts, and the policy that reads them

The distinction the whole design rests on:

> **An observation is a FACT.** "Full valuation, £300,000, 30 June 2026,
> Countrywide." "Indexed valuation, £285,000, effective 30 June 2026, UK HPI."
> Both are true. Neither is "the" valuation.
>
> **The selection is a RULE.** "For current LTV prefer an approved full valuation
> no older than 24 months; otherwise the approved indexed valuation." That is a
> credit decision, and it lives in versioned configuration — never inferred, and
> never decided inside an agent prompt.

`config/system/valuation_selection_policy.yaml` declares `CURRENT_LTV_SELECTION@v1`
(preference order, per-type age limits, required status, and
`on_no_qualifying_observation: refuse`) and the calculation rules that consume it.
There is no formula language: a policy names a preference order over declared
types plus a maximum age. Anything beyond that is a new registered rule reviewed
as code.

**Multiple observations, deterministic selection, and the explanation** — a real
result, from the planted portfolio's `LN-I-005`:

```
current_loan_to_value = 60.0 percentage_points
  calculation   CURRENT_LTV@v1        recomputed_by_trakt: False
  numerator     current_principal_balance = 180,000
  denominator   val_de9b64715fff4a3b — indexed, £300,000, 2026-06-30
  selected      under CURRENT_LTV_SELECTION@v1
                "selected the most recent qualifying 'indexed' observation"
  rejected      full            — 66 months old at 2026-07-31, and the policy
                                  allows 24
                purchase_price  — a qualifying 'indexed' valuation was
                                  available, which this policy prefers
  validation    failed — LTV001, LTV004 — 2 errors
```

**Every observation is either selected or rejected with a stated reason.** There
is no silent discard, because "why not that one?" is the question a credit
committee actually asks, and an explanation that cannot answer it is not evidence.

**`recomputes: false`.** The canonical tape owns the number. This module explains
how it was arrived at; it never becomes a second LTV. A planted case
(`LN-F-007`) has selectable inputs of 220,000 and 400,000 and *no* ratio on the
tape — Trakt reports the gap and does not fill it, which is asserted.

The sharpest test: on `LN-M-004` the **indexed** observation (2026-07-15) is more
recent than the **full** one (2026-06-30), and full still wins. "Most recent"
would give the wrong answer, and "most recent" is exactly what an agent would
guess.

---

## 6. `get_loans` — a real structured response

`LN-H-002`, `shape="structured"`, `include=["valuations"]`, abridged:

```json
{
  "loan":        { "loan_identifier": "LN-H-002",
                   "source_portfolio_id": "direct_001",
                   "portfolio_cohort": "2026H1",
                   "reporting_date": "2026-07-31" },
  "borrower":    { "borrower_identifier": "BR-000002",
                   "borrower_type": "individual",
                   "employment_status": "employed",
                   "geographic_region_obligor": "London" },
  "collateral":  { "collateral_id": "LN-H-002",
                   "collateral_type": "residential_property",
                   "property_type": "detached_house",
                   "geographic_region_collateral": "London",
                   "valuations": [
                     { "valuation_id": "val_ad8a0d86c644f00c",
                       "valuation_type": "full",
                       "amount": 400000.0, "currency": "GBP",
                       "valuation_date": "2026-04-30",
                       "source": "full valuation", "status": "approved",
                       "provenance_ref": "current_valuation_amount" } ] },
  "contract":    { "current_principal_balance": 380000.0,
                   "current_loan_to_value": 95.0,
                   "origination_date": "2021-03-01",
                   "maturity_date": "2046-03-01" },
  "performance": { "account_status": "performing",
                   "arrears_balance": 0.0,
                   "number_of_days_in_arrears": 0,
                   "ifrs9_stage": "1" },
  "_provenance_ref": { "resource": "ERE/source_portfolio/direct_001",
                       "snapshot_id": "snap_planted_0001",
                       "loan_id": "LN-H-002" }
}
```

Valuation columns never appear as scalars anywhere — they are observations, which
is the whole reason the entity model exists. Assembly reads the row already in
hand, so the structured shape costs a dictionary walk rather than another query
(asserted: one dataset access for the whole batch, whatever the shape).

The `_provenance_ref` is a *pointer*, not an envelope. Inlining full provenance
for every field of every loan would make the response unusable; `explain_values`
is what it points at.

---

## 7. `explain_values` — a real evidence-backed explanation

See §5 for the full derivation block. The envelope answers four questions:

| | |
|---|---|
| **WHAT** | the value, its unit and format |
| **HOW** | source dataset and field (`OUT_PRIN` in `july_servicer_tape.csv`), mapping method, confidence and version; or the derivation rule |
| **WHY** | the calculation rule and version, the numerator field and value, the **selected valuation observation by id**, the selection policy version, and the rejected observations with reasons |
| **WHEN** | snapshot id, content hash, reporting date |

Plus validation status, the rules applied and the error count.

**Provenance is per FIELD, not per cell.** Mapping, transformation and validation
are properties of a field within a snapshot, so a 130-column tape has a 130-entry
index rather than 130 million cell envelopes. The index is bound to one
`(tenant, snapshot)` and **raises** on any other, because provenance from the
wrong snapshot is worse than none — it is confidently wrong.

**Batch-first.** 30 values via `explain_values` once: 5.2 ms. Via `explain_value`
thirty times: 128.7 ms. **24.8×.**

---

## 8. The tool registry

Thirteen tools, all through one governed path, all published in
`GET /v1/agent/tools`, the OpenAPI document and the MCP declarations from one
registry.

| Tool | Capability | What it answers |
|---|---|---|
| `portfolio_summary` | `risk:read` | Size, balance, weighted averages, arrears, composition |
| `stratify` | `risk:read` | Breakdown by one dimension |
| `concentration` | `risk:read` | Top-N share — measured, **not** a pass/fail |
| `evaluate_covenants` | `risk:read` | The operator-approved limit tests |
| `covenant_drillthrough` | `loan:read` | Which loans make up a test's numerator |
| `period_change` | `risk:read` | What moved between two governed periods |
| `data_completeness` | `risk:read` | What the tape actually carries |
| `list_validation_exceptions` | `risk:read` | Field-level validation outcomes |
| `rank_loans` | `loan:read` | The N worst on a metric, as identifiers |
| `get_loans` / `get_loan` | `loan:read` | Loan retrieval (batch is the primitive) |
| `explain_values` / `explain_value` | `loan:read` | The evidence behind values |

**Two capabilities, deliberately.** Aggregate MI tells you a book's *shape*;
`loan:read` exposes individual obligations. `loan:read` is **not** in
`DEFAULT_MI_SCOPES`, so a human MI session does not silently acquire it.

**Safe for autonomous agents** because of four properties, each tested:

1. **Bounded.** Every tool caps what it returns (500 loans, 500 explanations,
   200 groups, 200 ranked) and *says so when it truncates* — a silently
   shortened league table reads as the whole one.
2. **Batch-first.** The batch form is the primitive and the singular is a thin
   wrapper over it, asserted structurally. An agent given only a singular tool
   calls it once per loan.
3. **Refusals are typed and non-retryable when they are decisions.** An
   autonomous caller can tell "not allowed" from "could not compute" without
   parsing prose.
4. **Absence is never silence.** A missing loan is listed. An unavailable field
   is listed. An empty covenant result says "this is an absence of evidence, not
   a clean result." No lineage index says the outcome is *unknown*, not clean.

---

## 9. MCP readiness

**Standard reviewed:** Model Context Protocol — JSON-RPC, `tools/list` and
`tools/call`, tools declared with JSON Schema `inputSchema`/`outputSchema`,
results as `content` blocks plus optional `structuredContent` and an `isError`
flag.

**Adapter implemented?** *Partly, and deliberately.* The **translation** is built
and tested (`trakt_tools/mcp.py`, 20 tests). No **server** is built.

**Why not the server.** A server needs a transport, a session lifecycle and an
authentication integration. Each is an operational decision with an owner, not a
translation — and none of them is what would get rewritten badly under time
pressure. The mapping *is*, so the mapping is the part that exists and is covered.
`SERVER_RESPONSIBILITIES` enumerates what remains, and a test asserts it is
written down.

**The exact mapping:**

| Trakt | MCP | Note |
|---|---|---|
| `ToolSpec.name` | `name` | |
| `.description` + `.agent_guidance` | `description` | MCP has no guidance field; a client that cannot see it will misuse the tool |
| `.input_schema` | `inputSchema` | **By reference, not copied** — a copy can drift |
| `.output_schema` | `outputSchema` | Same |
| `.version`, `.required_capability` | `_meta["trakt/…"]` | |
| `registry.catalogue(context)` | `tools/list` result | Narrowed per session, as HTTP already is |
| `GovernedResult.result` | `structuredContent` | Never truncated |
| a rendered digest | `content[0].text` | Outcome, snapshot, warnings — ordered, not dropped |
| `status != success` | `isError: true` | A refusal must not read as data |
| **`ExecutionContext`** | **built by the server from the authenticated session** | **Never from arguments** |

That last row is the load-bearing one. MCP has no identity model of its own, so
the temptation to accept `tenant_id` as an argument is real and the failure is
total. `refuse_identity_in_arguments` raises rather than stripping, because a
caller that sent a `tenant_id` believed it meant something.

**MCP is not the business logic.** A structural test asserts `trakt_tools/mcp.py`
references no pandas, no dataframe, no authorisation call and no
`execute_governed_tool` — it is a translation, and cannot become a second
governed path.

---

## 10. A2A readiness

Full design in `docs/a2a_readiness_design.md`. **Nothing built**, per the brief.

Topology: `Buyer Agent --A2A--> Seller Agent --MCP--> Trakt`. Two protocols doing
two jobs. Trakt speaks the tool protocol and should never become a party to the
negotiation above it.

**Six of the nine named concerns already hold:**

| ✅ Already compatible | ❌ / ◐ Future work |
|---|---|
| **Agent identity** — Entra service principal → `ExecutionContext(actor_type=service)` | **Agent discovery** — no agent card, no access-request route |
| **Authentication** — OIDC/JWKS, issuer + audience validated, `Trakt.Agent` role | **Long-running task state** — every call is synchronous |
| **Capability advertisement** — `GET /v1/agent/tools` returns the caller-narrowed tools *and* the closed set of resources | **Task structure** ◐ — complete per call; no *enquiry* grouping many |
| **Organisation ownership** — `organisation × resource × capability`, where the organisation need not own the resource | |
| **Correlation IDs** — threaded through envelope and audit; both sides can join their records | |
| **Evidence references** — `SnapshotRef` with content hash, `ProvenanceRef`, `explain_values` to the observation and policy version | |

**Four gaps, in the order they bite:** (1) entitlement lifecycle — approval,
**expiry**, revocation, purpose-narrowing; (2) the enquiry container; (3) a
referral route for questions Trakt cannot answer; (4) protocol adoption. Only the
fourth needs a counterparty. The first three are buildable now and valuable on
their own.

**What Sprint 2 must preserve** so A2A is additive rather than a rewrite: every
tool names a required `resource`; identity never comes from arguments;
`correlation_id` is carried, never overwritten; refusals are typed and carry
`retryable`; every answer carries its snapshot. All five hold today.

---

## 11. Securitisation Readiness Agent coverage

Full table in `docs/securitisation_readiness_agent_coverage.md`.
**32 questions: 20 covered, 6 partial, 6 not covered.**

| Agent need | Trakt tool | Ready? | Gap |
|---|---|---|---|
| Portfolio composition | `portfolio_summary`, `stratify` | ✅ | Two-dimension cross-tab (G1) |
| Concentrations | `concentration`, `evaluate_covenants` | ✅ | — |
| Covenant issues | `evaluate_covenants`, `covenant_drillthrough` | ✅ | — |
| Data quality | `data_completeness`, `list_validation_exceptions` | ✅ | — |
| Missing information | `data_completeness` | ✅ | — |
| Collateral | `get_loans(structured)` | ✅ | — |
| Valuation age / method | `get_loans(include=valuations)`, `explain_values` | ✅ | — |
| Performance & arrears | `portfolio_summary`, `stratify`, `period_change` | ✅ | Multi-period trend (G5) |
| Material exceptions | `list_validation_exceptions`, `rank_loans` | ◐ | No aggregated statement of unknowns (G10) |
| **Regulatory-data readiness** | — | ❌ | **Annex coverage (G7) — the highest-value gap** |
| Evidence pack | — | ❌ | Enquiry lifecycle (G9) |

Section 4 of that table — data quality and evidence — is *fully* covered, which
matters more than the headline. The gaps are about **scope and packaging**, not
about whether Trakt can evidence a figure.

Two rows are recorded as **deliberate declines** rather than backlog:
simulating an unapproved covenant (G3), and unbounded filtered loan retrieval
(G4).

---

## 12. Performance

All figures on this machine, warm, at three scales.

**Retrieval and evidence** — cost is dominated by the frame scan, and selectivity
is the point:

| tool | 10k | 100k | 1m | at 1m |
|---|---|---|---|---|
| `get_loans` (20 ids) | 6 ms | 9 ms | 34 ms | scanned 1,000,000 → returned 20 |
| `get_loans` structured + valuations | 8 ms | 10 ms | 36 ms | same, one dataset access |
| `explain_values` (30 values) | 9 ms | 12 ms | 58 ms | scanned 1,000,000 → returned 30 |
| `data_completeness` | 3 ms | 7 ms | 49 ms | 23 fields |

**Aggregates**, after the `analytics_lib.stratify` fix described below:

| tool | 10k | 100k | 1m |
|---|---|---|---|
| `stratify` (4 groups) | 19 ms | 94 ms | 1,073 ms |
| `concentration` (333k groups) | 17 ms | 91 ms | **1,114 ms** |
| `portfolio_summary` (4 stratifications) | 43 ms | 214 ms | 2,479 ms |
| `rank_loans` (top 20) | 20 ms | 111 ms | 2,825 ms |

**The finding this benchmark produced.** `concentration` on `borrower_identifier`
took **166,103 ms** — 166 seconds — at one million rows. `analytics_lib.stratify`
looped over groups in Python: O(groups) in interpreted code, 333,334 iterations
each doing a per-group `nunique`. The dimensions a *concentration* question asks
about are precisely the high-cardinality ones, so the slow path was the common
path.

Fixed in place, not worked around — `analytics_lib` is the one implementation MI,
the risk monitor and the agent tools share, and a fast path inside the tool would
have been the second calculation implementation this sprint forbids.

| | before | after | |
|---|---|---|---|
| `stratify` by borrower, 1m rows / 333k groups | 61,662 ms | 2,060 ms | **30×** |
| `stratify` by borrower, 60k rows / 20k groups | 34,145 ms | 91 ms | **375×** |
| **`concentration` tool, 1m rows** | **166,103 ms** | **1,114 ms** | **149×** |

Safety: the original implementation is preserved verbatim as an oracle in
`tests/test_analytics_stratify_vectorised.py` and compared frame-for-frame across
nine cases, chosen to hit every branch the loop had — null dimensions, null
balances, zero weights, a missing metric, a group whose weights sum to zero. All
identical.

**Other measured improvements:** MI query 3.6–7.0× (§2); Parquet reads 6.5–31.3×
(§3); batching 17.6× for loans and 24.8× for explanations.

**Cost controls (Part 14 of the brief).** The architecture the scalability review
demanded — calculate in Trakt, let the model reason over results — is now
enforced rather than hoped for:

- *No one-LLM-call-per-loan.* Every population question has an aggregate tool.
  `rank_loans` is the deliberate bridge: it finds the twenty loans worth looking
  at without reading the other 499,980, and returns identifiers rather than rows,
  so the natural next step is one bounded `get_loans`.
- *No one-HTTP-request-per-loan.* Batch is the primitive; the singular form is a
  wrapper, asserted structurally.
- *No LLM in a deterministic calculation.* Every figure comes from
  `analytics_lib` or an existing engine. `concentration` reports the measured
  share and **refuses to call it a breach**, because whether 55.6% breaches
  anything depends on an operator-approved threshold that lives in the covenant
  configuration with its approver and version.
- *Visible cost.* Every call publishes `rows_scanned`, `rows_returned`,
  `selectivity`, `duration_ms` and cache outcomes **on the result**, not only in
  a log the caller cannot read. An agent that cannot see what its last call cost
  has no way to choose a cheaper next one.

---

## 13. Regression evidence

Full suite, both trees, same machine, same interpreter, `-p no:randomly` so
ordering is fixed and the two runs are comparable.

| | baseline `86460f3` | current `f593dbb` |
|---|---|---|
| passed | 4,953 | **5,151** (+198) |
| failed | 64 | **64** |
| errors | 13 | **13** |
| skipped | 33 | 33 |
| subtests passed | 6 | 6 |
| wall clock | 2,371 s | 2,428 s |

**The complete failure-id sets are identical.** Extracted from both runs, sorted
and deduplicated, then diffed:

```
baseline ids: 64    current ids: 64
=== ONLY IN CURRENT (new failures) ===        (empty)
=== ONLY IN BASELINE (fixed) ===              (empty)
IDENTICAL failure/error id sets
```

Not "the same number of failures" — the same failures, by identifier. The +198
delta is entirely new tests passing.

Test collection is clean on the current tree (5,261 collected, no collection
errors), so the 13 errors are runtime setup/teardown errors rather than import
failures; their count is unchanged and they were itemised separately with `-rE`
to confirm the same set.

**The 64 pre-existing failures are not mine and are not new.** Two were checked
individually against the baseline worktree during the sprint after the
`analytics_lib` change, and fail identically there:
`test_every_bucket_applies_on_full_frame` (a bucket that does not materialise on
the full frame) and `test_no_regulatory_or_annex2_files_modified` (a guard test
flagging `tests/test_annex2_collateral_projection.py`, a file this sprint never
touched).

**Sprint 2 added 198 tests**, in eight files:

| file | tests | what it holds |
|---|---|---|
| `test_sprint2_credit_path.py` | 84 | entity model, valuation selection, structured loans, derivations — against planted truth |
| `test_agent_analysis_tools.py` | 58 | the eight aggregate tools, checked against hand-stated shares |
| `test_agent_mcp_adapter.py` | 20 | the MCP mapping, mostly negative assertions |
| `test_serving_parquet.py` | 14 | the copy equals the CSV; every degradation path |
| `test_analytics_stratify_vectorised.py` | 13 | the vectorised rewrite against the original as oracle |
| `test_mi_dataset_profile_cache.py` | 9 | profile-cache identity and invalidation |
| plus edits to `test_agent_tool_registry.py`, `test_agent_identity_and_api.py`, `test_agent_reference_client.py` | — | de-fragilised: assert the property, not the tool list |

**Three Sprint 1 tests were changed rather than deleted**, and the change is
itself an improvement: they asserted the exact tool list or indexed `tools[0]`.
They now assert the *property* — every published tool requires the capability the
caller holds, and `loan:read` tools are hidden from a `risk:read` caller — and
select tools by name. The scripted reference client, which picked `tools[0]` and
called it with only `{resource}`, now reads the published schema to find a tool
it can actually satisfy. That is a better rule independent of this sprint.

---

## 14. Remaining blockers

**1. The audit lost-update race — open, and it blocks Sprint 3.**
`OpsStore.append_audit` is a read-modify-write with `overwrite=True` and no
compare-and-swap anywhere in `operations_control`. Two concurrent appends lose a
record **while leaving a chain that still verifies** — which is the worst
property an audit log can have: silent, and passing its own integrity check.

It was identified in the scalability review and deliberately left unfixed under
that sprint's scope rule. Sprint 3 introduces a synthetic Buyer Agent, which
means concurrent writes. **This must be fixed before Sprint 3 does concurrent
work**, with an ETag/If-Match conditional write.

**2. Annex-coverage readiness (G7) — the missing tool for the agent.**
`engine/gate_4_projection` knows the Annex 12 field requirements, but no tool
exposes readiness against them. An agent asked "is this portfolio ready to
issue?" can describe the tape thoroughly and cannot answer the regulatory
question. Small, and the highest-value remaining item.

**3. The valuation sidecar writer.** The reader, the model and the selection
policy are built and tested. Today observations are derived from the tape's own
columns, which gives a truthful history of one to three per collateral. A client
supplying genuine valuation history has nowhere to put it. Not blocking — the
selection policy already works on whatever observations exist.

**4. Evidence pack / enquiry lifecycle (G9).** Every ingredient exists per call;
nothing accumulates them. Needed for real diligence, not for the agent's first
demonstration.

**Non-blocking, recorded:** two-dimension stratification (G1), vintage bucketing
(G2 — `analytics_lib.buckets` already does the bucketing, it just is not wired
to `stratify`), multi-period trend (G5), readiness thresholds (G6), loan-level
eligibility rules (G8).

**Two pre-existing test failures**, present at baseline `86460f3` and unchanged:
`test_every_bucket_applies_on_full_frame` and
`test_no_regulatory_or_annex2_files_modified`.

---

## 15. Recommendation

> **Is Trakt ready to build the Securitisation Readiness Agent as a genuine
> external consumer of its governed credit intelligence layer?**

**Yes — with one capability to add first and one defect to fix before Sprint 3
does concurrent work.**

The infrastructure claim now holds. An external agent, importing nothing from
Trakt and speaking only HTTP, can investigate a portfolio's composition,
concentrations, covenants, collateral, valuations, performance, arrears and data
quality; can find the loans worth examining without reading the book; and can
justify any figure it reports down to the valuation observation and the versioned
policy that selected it. Every number is Trakt's. Every call is authorised,
bounded and audited. The reference agent has no privileged path — which is what
makes it a credible demonstration to a client evaluating whether to build their
own.

**Minimum remaining gap, in order:**

1. **Annex-coverage readiness (G7).** Without it the agent is a very good
   portfolio analyst that cannot answer the securitisation question it is named
   after. Small: the requirements already exist in `gate_4_projection`.
2. **The audit lost-update race.** Not needed for a single-agent demonstration;
   **required** before Sprint 3's concurrent work, because an audit log that
   loses records while passing its own integrity check is worse than one that
   fails loudly.

Everything else in §14 is improvement, not blocker. My recommendation is to build
G7 as the first task of Sprint 3, fix the audit race alongside it, and start the
agent on the thirteen tools that exist.

**One caution.** The temptation in Sprint 3 will be to let the agent compute
something Trakt does not expose — a ratio, a cohort, a trend — because it is
faster than adding a tool. Every time that happens, the number loses its
definition, its provenance and its reproducibility, and Trakt stops being the
system of record for that answer. The correct response to a missing capability is
a new governed tool that every entitled organisation can call.
