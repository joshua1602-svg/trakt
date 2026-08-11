# Trakt A2A Scalability and Efficiency Review

**Status:** Read-only review. No production code was modified.
**Date:** 2026-08-11
**Branch:** `claude/trakt-a2a-architecture-review-ogm81v`
**Predecessor:** [`a2a_architecture_readiness_review.md`](a2a_architecture_readiness_review.md)
**Subject:** the Sprint 1 agent tool layer (commit `64e5d51`) and the canonical
read/load paths beneath it.

**Question:** will this architecture scale from one client / one portfolio / one
agent to hundreds of organisations, thousands of portfolios and millions of
loans *without a rewrite*?

**Method:** measurement, not inference. Every number in this document was
produced by running the actual code in this repository. Synthetic configuration
and canonical-shaped frames (100 columns, matching the 66–130 columns observed in
real `*_canonical_typed.csv` outputs) were generated to the four scale envelopes.
Measurements were taken on this development container; treat them as *relative*
evidence — the ratios are the finding, not the absolute milliseconds.

---

## 1. Executive conclusion

**Is the current A2A-native architecture scalable?** The *contracts* are. The
*implementation* has one defect that makes it unusable beyond a pilot, and it is
about thirty lines to fix.

The tool layer itself is close to free — schema validation 6 µs, catalogue
assembly 24 µs, audit event 10 µs, and the schemas are import-time literals, not
rebuilt per request. Handlers are pure functions with no per-request module
state. The governed envelope, the resource-reference indirection and the
`ToolSpec` registry are all format-agnostic, so storage can change underneath
them without touching a single agent-facing contract. That is the property this
review was meant to test, and it holds.

**To approximately what scale without major change?** Roughly **Scale A
(pilot)** as it stands: 1–5 clients, <20 portfolios, <250k loans. The binding
limit is not data volume — it is per-request configuration reloading.

**What is the first likely bottleneck?** Not storage. **Governance configuration
is re-parsed and re-validated from YAML on every single governed call**, and the
cost is linear in the number of organisations, resources and grants:

| Envelope | orgs | resources | grants | Config cost **per governed tool call** |
|---|---:|---:|---:|---:|
| A — pilot | 2 | 4 | 4 | **7 ms** |
| B — early platform | 30 | 250 | 150 | **239 ms** |
| C — established | 150 | 2,100 | 750 | **1,744 ms** |
| D — A2A infrastructure | 400 | 6,000 | 2,000 | **5,098 ms** |

That is *before any data is touched*, on every call, per worker. At Scale C a
single tool call spends 1.7 seconds re-reading YAML it read on the previous call.
This is the top finding of the review and the one thing that must change before
Sprint 2 — because Sprint 2 adds `get_loan` and `explain_value`, which are
high-frequency tools, and the overhead compounds per call.

**Is the canonical file-based model currently a problem?** Not yet, and not for
the reason usually assumed. Once a frame is in memory, everything is fast: at 1M
rows × 100 columns a groupby is 52 ms, a filter 659 ms, a loan lookup 12 ms. The
problem is purely **load cost**, and the existing signature-keyed dataset cache
(`data_source._ACTIVE_CACHE`) already absorbs most of it. What does not scale is
the *cold* read:

| rows | CSV size | CSV read | Parquet read | Parquet, 5 columns | in-memory |
|---:|---:|---:|---:|---:|---:|
| 100,000 | 56 MB | **2.3 s** | 0.32 s | **0.045 s** | 87 MB |
| 1,000,000 | 562 MB | **30.8 s** | 7.2 s | **0.16 s** | 869 MB |

**Should Parquet be introduced?** Yes — as a **serving artefact alongside CSV,
not as a replacement and not as the authoritative format.** Column projection is
the decisive number: 157 ms versus 30.8 s at 1M rows, a **190× improvement**, and
agent tools read 5–15 columns, not 100. CSV stays authoritative because the
regulatory pipeline, the XSD path and human inspectability all depend on it.

**Is a database required now?** No. Nothing measured here justifies one. Triggers
are defined in §4; the honest first trigger is *concurrent writes to workflow
state*, not read volume — and that arrives with Sprint 3's DD objects, not
Sprint 2.

**Is the repeatable Valuation entity design scalable?** Yes, and it is the right
call — with one condition: valuation observations must live in a **separate
narrow dataset keyed by collateral, not as additional columns on the wide tape**.
A wide tape cannot represent 0..n observations without either exploding column
count or losing history. §5.

**Is `get_loan` assembling objects on demand the right approach?** Yes — but
`get_loan` must not be the primary contract. Measured: 20 individual loan lookups
at 1M rows cost **1,617 ms**; the same 20 loans in **one batched call cost 76 ms**
— a **21× penalty** for the N+1 shape (37× at 100k rows). `get_loans(loan_ids[])`
must be the primitive and `get_loan` sugar over it, decided *before* the contract
ships. §6.

**Is on-demand `explain_value` sustainable?** Yes for single values, no for DD
sweeps, unless two things are built with it: an **ingestion-time lineage index**
(the current `field_lineage.json` is a whole-file JSON parse per lookup) and a
**bulk `explain_values([...])`** from day one. §7.

**Does anything need changing before Sprint 2?** Three things, all small:

1. cache the five governance registries on file identity (~30 lines);
2. make `get_loans` the primitive, `get_loan` the convenience wrapper;
3. add `explain_values` alongside `explain_value` and build the lineage index at
   ingestion.

Everything else is either already right, or a defined migration trigger.

---

## 2. Scalability scorecard

Legend: ✅ comfortable · ⚠️ works with the stated change · ❌ blocking.

| Area | Pilot (A) | Early platform (B) | Established (C) | A2A (D) | Main constraint |
|---|:--:|:--:|:--:|:--:|---|
| **Configuration / entitlements** | ⚠️ 7 ms | ❌ 239 ms | ❌ 1.7 s | ❌ 5.1 s | Re-parsed and re-validated per call; O(orgs+resources+grants). **Fix before Sprint 2.** |
| **Storage (canonical)** | ✅ | ⚠️ | ❌ | ❌ | CSV cold read 30.8 s/1M rows. Parquet serving copy at Scale B. |
| **Analytics (in-memory)** | ✅ | ✅ | ⚠️ | ⚠️ | 869 MB/1M rows bounds portfolios per worker; full-frame `.copy()` 115 ms/query. |
| **Agent API / tool execution** | ✅ | ✅ | ✅ | ✅ | 6–24 µs overhead. Stateless. Genuinely not a constraint. |
| **Loan-level access** | ✅ | ⚠️ | ❌ | ❌ | N+1 shape: 21–37× penalty. **Contract shape must change before Sprint 2.** |
| **Provenance (`explain_value`)** | ✅ | ⚠️ | ❌ | ❌ | Whole-file JSON parse per lookup; no bulk form. |
| **Audit (log line)** | ✅ | ✅ | ✅ | ✅ | 10 µs/event. Not a bottleneck. |
| **Audit (hash-chained)** | ⚠️ | ❌ | ❌ | ❌ | Read-modify-write with no CAS — **lost updates under concurrency**, not just slow. |
| **Workflow state** | ✅ | ⚠️ | ❌ | ❌ | `OpsStore` list-and-read-all query pattern; no optimistic locking. Sprint 3 concern. |
| **Identity / machine auth** | ✅ | ✅ | ✅ | ⚠️ | JWKS cached; org resolution is the config problem above, not an auth problem. |
| **Multi-tenancy** | ✅ | ⚠️ | ❌ | ❌ | `get_dataframe()` takes no tenant: the canonical read path is **deployment-scoped**. |

---

## 3. Bottleneck map — top 10, ranked

Ranked by *expected cost of leaving it*: likelihood × impact × how much harder it
gets later.

| # | Bottleneck | Where | Appears at | Impact | Cost to fix later | Rank basis |
|---|---|---|---|---|---|---|
| 1 | **Governance config re-parsed per call** | `trakt_core/{organisation,resource,entitlement,principal,tenancy}.py` — no caching in any of the five loaders | Scale B | 239 ms → 5.1 s per call | Low (30 lines) but the *symptom* is misread as "the platform is slow" and invites wrong fixes | Certain; measured |
| 2 | **N+1 loan access** | Sprint 2 contract, not yet written | Scale A with an agent | 21–37× | **High** — an agent-facing contract, once published, is the hardest thing to change | Certain if `get_loan` ships alone |
| 3 | **CSV cold read** | `data_source._load_active` → `pd.read_csv` | Scale B | 30.8 s/1M rows per cold worker | Medium — Parquet is additive behind an unchanged contract | Certain; measured |
| 4 | **Hash-chained audit lost updates** | `OpsStore.append_audit` (`operations_control/stores.py:371`) | Any concurrency | **Correctness**, not perf | Medium — but a corrupted evidence chain is not retro-fixable | High likelihood at Sprint 3 |
| 5 | **`explain_value` whole-file JSON parse** | `field_lineage.json` / `value_lineage.json` artefacts | Scale B, or any DD sweep | Linear in file size per lookup | Medium — index is additive | Certain during DD |
| 6 | **Single-slot, deployment-scoped dataset cache** | `data_source._ACTIVE_CACHE`; `get_dataframe()` has no tenant argument | Scale C (multi-tenant process) | Cache thrash; forces deployment-per-tenant | Medium | Only if multi-tenant processes are wanted |
| 7 | **Full-frame `.copy()` per query** | `mi_query_executor` (`work = work.copy()`, line 366; `_apply_filters` returns `work.copy()`, 478) | Scale C | 115 ms + 659 ms per call at 1M rows | Low | Certain but modest |
| 8 | **Memory ceiling per worker** | 869 MB per 1M-row × 100-col frame | Scale C | ~4–8 cached portfolios per 8 GB worker | Low (projection fixes it) | Certain at Scale C |
| 9 | **`OpsStore` list-then-read-all queries** | `list_audit`, `list_decisions` (`stores.py:395`) | Scale C / Sprint 3 | O(n) blob reads per query | Medium | Sprint 3 |
| 10 | **Threadpool saturation** | Sync `def` routes → anyio threadpool (default 40) with GIL-bound pandas | Scale C | Concurrency ceiling ≈ 40 blocking calls | Low (config) | Moderate |

**Deliberately *not* in this list**, having been checked and found sound: the tool
registry, schema validation, JSON schema generation, the audit *log* path, the
Copilot/agent JWKS cache, `request_scope` and `currency` (both `contextvars` —
correctly request-isolated), and `serving_cache.BoundedCache` (identity-keyed,
tenant-scoped, bounded, no TTL — the right design already present in the repo).

---

## 4. Storage recommendation

### Where the cost actually is

The instinct "CSV does not scale" is half right and misleadingly stated. Measured
on canonical-shaped data (100 columns):

```
                 CSV      Parquet   Parquet     Parquet
                 read     read      5 columns   +predicate      in-memory ops
  100k rows      2.3 s    0.32 s    0.045 s     0.18 s          groupby   8 ms
                                                                filter   59 ms
                                                                lookup    4 ms
  1M rows       30.8 s    7.2 s     0.157 s     0.75 s          groupby  52 ms
                                                                filter  659 ms
                                                                lookup   12 ms

  size         562 MB    141 MB                                 869 MB RAM
```

Three facts follow, and they drive the whole recommendation:

1. **Analytics are not the problem.** A groupby over a million loans is 52 ms.
   The deterministic engines are fine.
2. **Load is the problem, and only when cold.** `data_source._ACTIVE_CACHE` is
   signature-keyed (blob ETag, or `path:mtime:size`) with a 30 s re-check, so a
   warm worker pays nothing. A cold worker at 1M rows pays 31 seconds — and every
   worker pays it independently.
3. **Column projection is the single highest-leverage change available.**
   157 ms versus 30.8 s. Agent tools want 5–15 columns; the tape has 66–130.

### Recommendation

| Question | Answer |
|---|---|
| **Authoritative format now** | **CSV, unchanged.** `*_canonical_typed.csv` remains the system of record. The regulatory chain (`regime_projector` → `annex2_delivery_normalizer` → `xml_builder_annex2` → XSD) reads it, humans inspect it, and `content_sha256` in `platform_canonical_manifest.json` is computed over it. Do not disturb any of that. |
| **Serving format now** | **Add Parquet as a derived serving copy**, written next to the CSV by the existing assembler, with the CSV's `content_sha256` recorded inside the Parquet metadata so the two can never silently diverge. |
| **Should Parquet replace CSV?** | **No.** Not now, probably not ever for the regulatory path. |
| **Should Parquet become authoritative?** | **Only** if and when the CSV write itself becomes a measured ingestion bottleneck. That is a different trigger, and it is not close. |
| **What stays CSV** | Everything a human or a regulator reads: canonical typed output, regime projections, mapping/validation reports, the provenance companion. |

**Migration path that breaks nothing:** the serving copy is generated *after* the
CSV, from the CSV, by `engine/platform_assembler.py`. A missing or stale Parquet
file falls back to reading the CSV. So the change is additive, revertible by
deleting a file, and invisible to every existing pipeline — precisely the shape
`serving_cache` already uses for its kill switches.

### When a relational database or warehouse becomes necessary

Not on read volume. The specific triggers, in the order they will actually arrive:

| Trigger | Observable condition | Store implied |
|---|---|---|
| **T1 — concurrent workflow writes** | Two agents mutating DD/workflow state for the same engagement, or `append_audit` contention (see §11) | **Transactional store for workflow + audit only** (SQLite → Postgres). *This is the first real trigger and it arrives with Sprint 3, not Sprint 2.* |
| **T2 — administrative scale** | >~50 organisations, or access changes more than weekly (see §13) | Governed config store (the OCC `ConfigPackageStore` already models draft→activate→rollback) |
| **T3 — loan-level random access dominates** | Median tool call touches <100 loans but pays a full-portfolio load; or valuation history exceeds ~10 observations/collateral | Indexed serving layer — **Parquet with row-group statistics first**, DuckDB over Parquet second. Postgres only if writes are needed. |
| **T4 — cross-portfolio / cross-client analytics** | A single question spans >50 portfolios, or >10M loans in one query | Warehouse/lakehouse. Not before. |

**Separate the two concerns explicitly**, because they have opposite
characteristics and conflating them is what forces premature migration:

* **Analytical storage** — canonical tapes, snapshots, valuations. Append-only,
  immutable, read-heavy, wide, well served by files + Parquet, and cacheable on
  content identity.
* **Transactional / workflow storage** — DD requests, decisions, escalations,
  audit chain, entitlements. Small, mutable, concurrent, needs atomicity and
  indexes, and is badly served by JSON blobs.

`OpsStore` is currently doing the second job with the first job's technology.
That is fine at pilot scale and is the first thing that will genuinely need a
database — for **write correctness**, not for size.

---

## 5. Entity model and the Valuation decision

### The pattern is right

```
   canonical analytical record   (wide, flat, immutable, one row per loan)
              ↓  entity assembly (config-driven, no second store)
   agent-facing object           (nested, typed, provenanced)
```

Assembling on demand from the canonical row is correct and should be preserved.
The rule from the previous review holds: **do not create a second canonical truth
store merely to make agent objects convenient.** Nothing measured here changes
that — assembly from an in-memory row is microseconds; the cost is the *load*,
which is shared with every other tool.

### Confirmed: Valuation is a repeatable dated entity

This was the open question from Sprint 1, and the answer is **yes, a repeatable
dated observation — not a flat attribute group**. Three reasons, in order of
weight:

1. **A wide tape cannot express 0..n.** The tape carries
   `current_valuation_amount` / `_date` / `_method` and `indexed_value` /
   `indexed_loan_to_value`. Those are *the currently selected* valuation, not the
   valuation history. Adding history to the tape means either `valuation_1_*`,
   `valuation_2_*` … (which breaks the field registry's one-concept-one-field
   discipline and is unbounded) or losing history entirely.
2. **`explain_value(current_loan_to_value)` is unanswerable without it.** The
   honest answer to "why is LTV 61%?" is "because we selected the full valuation
   of £300k dated 30 June 2026 over the indexed £285k, under selection policy
   v3". If only the selected value exists, Trakt can state the input but not the
   *choice* — and the choice is the part a credit committee argues about.
3. **The concentration library already anticipates it.** `valuation_indexed` is a
   declared field role with `implementation_status: interface_only`, refusing
   with `external_reference_unconfigured` rather than simulating. That is a
   modelled gap waiting for exactly this entity.

### Minimum Valuation identity

Nine fields. Everything else proposed in the brief is deferred until something
needs it.

```yaml
valuation:
  valuation_id:       # stable, deterministic: hash(collateral_id, type, valuation_date, source)
  collateral_id:      # FK — the parent
  valuation_type:     # full | drive_by | desktop | avm | indexed | purchase_price
  amount:             # numeric
  currency:           # ISO code
  valuation_date:     # when the valuation was performed  }  kept separate: an
  effective_date:     # the date it speaks to             }  indexed valuation
                      #                                      is performed later
                      #                                      than it speaks to
  source:             # provider / index name — who says so
  provenance_ref:     # → the snapshot + source field this observation came from
```

**Deliberately excluded for now**, with the trigger that would add each:

| Field | Excluded because | Add when |
|---|---|---|
| `methodology` | `valuation_type` carries the distinction that matters today | A client supplies two AVMs with different methodologies |
| `supersedes` / `superseded_by` | Derivable from `(collateral_id, valuation_type, valuation_date)` ordering | An observation is genuinely retracted rather than superseded |
| `status` | Selection policy (below) decides usability; a status field would be a second, competing answer | An operator must be able to reject one observation without deleting it |

### Storage for valuation observations

**A separate narrow dataset, Parquet, keyed by `collateral_id` — not the wide
tape, not a relational child table (yet).**

```
platform/{tenant}/{snapshot}/canonical_typed.csv        (authoritative, unchanged)
platform/{tenant}/{snapshot}/canonical_typed.parquet    (serving copy, §4)
platform/{tenant}/{snapshot}/valuations.parquet         (NEW — narrow, 9 columns)
```

Why this and not the alternatives:

* **not the wide tape** — cannot express 0..n, and widening a 130-column tape
  toward 499 makes every load worse for every caller;
* **not a relational child table yet** — valuations are append-only observations
  with no concurrent writers. A database buys atomicity nobody needs, at the cost
  of a migration nobody has justified. Trigger T3 above is when this changes;
* **Parquet, not CSV** — a valuations dataset is tall and narrow (n_collateral ×
  n_observations), exactly the shape where columnar projection and predicate
  pushdown pay, and it has no regulatory consumer requiring CSV.

Sizing: 1M loans × 3 observations × 9 narrow columns ≈ 3M rows ≈ well under
200 MB in Parquet, and a `collateral_id` predicate pushdown reads a fraction of
that.

### Observations vs selection policy — confirmed, and this is the important part

The separation proposed in the brief is correct and should be built as stated:

```
  VALUATION OBSERVATIONS  =  facts        (immutable, dated, sourced)
      "full £300,000, valued 30 Jun 2026, source: Countrywide"
      "indexed £285,000, effective 30 Jun 2026, source: UK HPI"
              ↓
  SELECTION POLICY        =  governed rule (versioned, operator-approved)
      "for current LTV prefer an approved full valuation ≤12 months old;
       otherwise the approved indexed valuation"
              ↓
  DERIVED VALUE           =  current_loan_to_value
```

This is the same shape the codebase already uses successfully for concentration
tests — `config/risk/concentration_test_library.yaml` declares *what a test is*
and the operator-approved `ActiveConfiguration` declares *the thresholds*, with
"no formula language and no path from client text to executable code". A
valuation selection policy should be a registered, versioned rule in exactly that
mould, not an expression language.

It makes `explain_value` genuinely answerable:

```jsonc
{ "value": 61.0, "canonical_field": "current_loan_to_value",
  "calculation": { "method": "balance_over_selected_valuation",
                   "method_version": "1.2.0" },
  "inputs": [
    { "canonical_field": "current_outstanding_balance", "value": 183450.00 },
    { "entity": "valuation", "valuation_id": "val_…", "amount": 300000.00,
      "valuation_type": "full", "valuation_date": "2026-06-30" } ],
  "selection": { "policy": "ltv_valuation_selection", "policy_version": "3",
                 "selected": "val_…", "rejected": [
                   { "valuation_id": "val_…", "reason": "indexed; a full "
                     "valuation ≤12 months old was available" } ] } }
```

**Scalability of the pattern:** excellent, because the policy is evaluated once
per loan during canonical transformation (vectorised over the frame), not per
request. `explain_value` then *reports* the recorded decision rather than
re-deriving it. Record the selected `valuation_id` and `policy_version` as
columns on the tape at transform time — two narrow columns — and the explanation
becomes a lookup rather than a re-computation.

### Minimum entity model before implementing `get_loan`

Six entities. Not eight, not the credit universe.

| Entity | Key | Parent | Cardinality | Source |
|---|---|---|---|---|
| `loan` | `loan_identifier` | — | 1 row on the tape | tape |
| `borrower` | `borrower_identifier` | loan (FK) | 1..n loans per borrower | tape columns |
| `collateral` | `collateral_id` *(derive as `loan_identifier` where absent)* | loan | 1:1 today | tape columns |
| `valuation` | `valuation_id` | collateral | **0..n** | `valuations.parquet` |
| `contract` | — (attribute group) | loan | 1:1 | tape columns |
| `performance` | — (attribute group) | loan | 1:1 point-in-time | tape columns |

Note `contract` and `performance` are deliberately **attribute groups, not
entities**: they have no independent identity and no repetition. Promoting them
to entities would add indirection with no expressive gain. `valuation` is the
only one that earns entity status, because it is the only one that repeats.

Declared additively in `config/system/fields_registry.yaml` (`entity`,
`entity_role`, `references` per field) plus one small
`config/system/entity_model.yaml`. No new store, no graph, no ORM.

---

## 6. `get_loan` design

### The measurement that decides the contract

```
                        20 loans, one at a time    20 loans, one batched call
  100k-row portfolio            117 ms                      3.2 ms      (37×)
    1M-row portfolio          1,617 ms                     76.5 ms      (21×)
```

And that is *in-process*, excluding HTTP, identity resolution and the governance
chain. Over the wire each of those 20 calls also pays the per-call config cost
from §1 — 7 ms at pilot, 239 ms at Scale B. At Scale B, 20 individual `get_loan`
calls cost ~6 seconds of pure governance overhead alone.

An agent asked to review 200 exceptional loans will call `get_loan` 200 times
unless the contract makes the batch form the obvious one.

> A curiosity worth recording: `df.set_index(...).loc[[key]]` **one at a time**
> measured *slower* than a full scan (408 ms vs 117 ms for 20 lookups at 1M
> rows) — per-call overhead dominates. The win is **batching, not indexing.**
> Indexing only pays once lookups are already batched.

### Contract

**`get_loans` is the primitive. `get_loan` is sugar.** Ship both, implement one.

```
get_loans(resource, loan_ids[], fields?, include?) -> { loans: [...], not_found: [...] }
get_loan (resource, loan_id,   fields?, include?) -> { loan: {...} }        # == get_loans([id])
```

| Aspect | Decision |
|---|---|
| `loan_ids` bound | **500 per call.** Above that the agent is doing analysis, not investigation, and should use `query_loans` / `stratify`. |
| `fields` | Optional canonical field list. Absent ⇒ a **curated default projection** (~25 fields), never all 130. This is what makes the Parquet column projection pay. |
| `include` | `["valuations"]` etc. — opt-in child entities. Absent ⇒ scalar attributes only. |
| Provenance | **Compact stub per field** (`source_field`, `effective_date`, `validation_status`), plus one `provenance_ref` the agent can expand via `explain_value`. Full envelopes for 25 fields × 500 loans is an unusable response. |
| `not_found` | Explicit array. Never silently omit — an agent that gets 498 loans back from 500 ids must be told which two are missing, or it will treat absence as a finding. |
| Response cap | Hard byte ceiling (~5 MB) with `truncated: true` and a `next_cursor`. |

### What must NOT be returned

* every canonical column by default (130 columns × 500 loans);
* full provenance envelopes inline;
* borrower free-text / PII beyond what the tool's capability grants;
* nested valuations unless `include` asked for them;
* an unbounded list.

### Preventing N+1

Four mechanisms, in order of effectiveness:

1. **Contract shape** — `get_loans` documented as the normal form, `get_loan`
   described in its `agent_guidance` as "for a single loan you are already
   investigating; to look at several, call `get_loans` once".
2. **Tool descriptions carry the guidance the model reads.** The `agent_guidance`
   field already exists on `ToolSpec` and is published in the catalogue; this is
   what it is for.
3. **Richer upstream tools that answer without loan fetches.** `covenant_drillthrough`
   should return the top contributing loans *plus compact evidence* — the numbers
   an agent would otherwise make 50 calls to assemble. This is the highest-value
   anti-N+1 measure and it costs nothing extra at query time.
4. **Telemetry, then budgets** (§8). Measure the pattern before policing it.

### Request-pattern boundaries

| Pattern | Right tool | Shape |
|---|---|---|
| single-loan investigation | `get_loan` | synchronous, <100 ms warm |
| 20 exceptional loans | `get_loans` | one call, synchronous |
| a cohort (100–500) | `get_loans` + `fields` projection | one call, synchronous, capped |
| "the worst 50 by LTV" | `rank_loans` | never `get_loans` over the whole book |
| portfolio-wide analysis | `stratify` / `concentration` / `portfolio_summary` | aggregated — the agent must never receive 100k rows |
| >500 loans of detail | `query_loans` with cursor | paginated |
| whole-portfolio evidence pack | asynchronous job (§10) | job id + poll |

---

## 7. `explain_value` design

### Execution flow, and where it hurts

```
explain_value(resource, loan_id, canonical_field)
  1  authorise resource + loan:read                        ~0 ms (in-memory)
  2  resolve snapshot for the resource                     cached
  3  read the canonical row                                12 ms cold scan / <1 ms batched
  4  read field mapping   ← field_lineage.json             WHOLE-FILE JSON PARSE
  5  read value lineage   ← value_lineage.json (optional)  WHOLE-FILE JSON PARSE
  6  read validation outcomes for (loan, field)            whole-file read
  7  read snapshot manifest (hash, dates, source portfolio) cached
  8  if calculated: selection + policy version              tape columns (§5)
  9  assemble the envelope                                  microseconds
```

Steps 4–6 are the problem. They are **whole-file JSON parses per lookup**, and
they are not snapshot-indexed. A DD sweep calling `explain_value` 500 times
re-parses the same lineage files 500 times.

### Recommendation

1. **Build a lineage index at ingestion, not at request time.** The pipeline
   already produces `field_lineage.json`; have `lineage_tracker` additionally
   emit a narrow `lineage_index.parquet` keyed `(canonical_field)` → mapping tier,
   source field, alias file, mapping version, derivation rule. Field-level
   lineage is **per field, not per cell** — for a 130-column tape that is 130
   rows, not 130 million. This is the single change that makes provenance cheap,
   and it materialises nothing per cell.
2. **Keep value-level lineage as an optional artefact**, read only when a caller
   asks about a field the index marks as value-traced. Most fields are not.
3. **Add `explain_values([...])` from day one.** Same 21–37× argument as
   `get_loans`. Bound at 500 `(loan_id, field)` pairs.
4. **Add an evidence pack as an asynchronous job** for the "explain everything
   material about this portfolio" case (§10), reusing the existing job pattern.

### Caching

Safe, because provenance is a pure function of immutable inputs:

| | |
|---|---|
| **Key** | `(tenant_id, snapshot_id, content_hash, canonical_field)` — field-level, not cell-level |
| **Scope** | process-local `serving_cache.BoundedCache` (the existing class already refuses to build a key without tenant and scope) |
| **Snapshot binding** | `snapshot_id` **and** `content_hash` both in the key. A republished snapshot changes the hash and therefore misses. |
| **Invalidation** | None needed — immutable identity, no TTL. This is exactly `serving_cache`'s stated design. |
| **What is never cached** | the authorisation decision, and the loan row itself |

**The rule that must not bend:** the cache key is derived from *content identity*,
never from time. A cache that could return provenance from the wrong snapshot or
the wrong tenant would be worse than no provenance at all, because it would be
confidently wrong. `serving_cache.key_for` already enforces tenant-and-scope in
the key by refusing to build one without them — reuse it rather than writing a
new cache.

---

## 8. Agent efficiency controls

Autonomous agents are, as the brief says, potentially inefficient and adversarial
consumers of compute. Four layers, cheapest first:

**1 — Tool design (free, most effective).** Every tool that answers a question an
agent would otherwise assemble from many calls removes those calls permanently.
Concretely: `covenant_drillthrough` returning top contributors *with* compact
evidence; `stratify` returning shares and counts so no follow-up is needed;
`rank_loans` so the agent never scans; `data_completeness` returning the whole
RAG profile in one call.

**2 — Batching.** `get_loans`, `explain_values`, and array arguments wherever a
tool is naturally per-entity. Bound every array at 500.

**3 — Limits and pagination.** Every list-returning tool: default page 100, max
500, hard response ceiling ~5 MB, explicit `truncated` + `next_cursor`. Never
silently truncate — an agent that receives 100 of 4,000 rows without being told
will report the 100 as the portfolio.

**4 — Budgets, per agent session (keyed on `correlation_id`).** Introduce
**after** telemetry shows real distributions, not before:

| Budget | Suggested opening value | Rationale |
|---|---|---|
| tool calls | 200 | a thorough DD review measured in the tens, not hundreds |
| rows returned | 50,000 | above this the agent should be aggregating |
| bytes returned | 50 MB | protects the model's context as much as the server |
| expensive calls (`period_change`, full stratification) | 20 | these are the ones that load frames |
| wall-clock | 10 min | a runaway loop should end |

Exceeding a budget must return a **typed, non-retryable error** — a new
`BUDGET_EXHAUSTED` code in the existing taxonomy — so an agent stops rather than
retries. Retryability is already a first-class property of `TraktError`.

**Telemetry first, and it is cheap.** `AuditMetadata` already carries
`duration_ms`. Add four fields — `rows_scanned`, `rows_returned`, `bytes_returned`,
`cache_hit` — to the tool result path, and one `tool_calls_in_session` counter
keyed by correlation id. That is enough to answer the question that actually
matters: *what does one autonomous portfolio review cost?* Model token usage
belongs at the harness, not in Trakt — the server makes no LLM call on the agent
path, and should not start.

---

## 9. Caching design

The repository already contains the right primitive: `mi_agent_api/serving_cache.py`
— identity-keyed (URI+ETag / `mtime_ns:size`), tenant and scope mandatory in the
key, bounded LRU, **no TTL**, per-process, with per-cache kill switches. Every
recommendation below reuses it rather than inventing a second mechanism.

| Item | Cache? | Key | Scope | Invalidation | Shared / local |
|---|:--:|---|---|---|---|
| **Governance registries** (orgs, resources, entitlements, principals, tenancy) | **YES — the fix** | file path + `mtime_ns:size` | global (config is server-side, not tenant data) | file change | process-local |
| Resolved entitlements per organisation | YES | `(organisation_id, entitlement_store.version)` | organisation | store fingerprint changes | process-local |
| Tool catalogue (per capability set) | YES | `frozenset(scopes)` | caller scopes | registry is import-time immutable | process-local |
| Field catalogue / entity model | YES | registry file identity | global | file change | process-local |
| Canonical dataset handle | **already cached** | source signature (ETag / mtime) | **deployment** ⚠️ see §10/§14 | signature change | process-local |
| Prepared MI dataset | already cached | `serving_cache` key | tenant + scope | source identity | process-local |
| `portfolio_summary` | YES | `(tenant, snapshot_id, content_hash, scope, filters_hash)` | tenant+scope | content hash | process-local |
| Covenant results | YES | `(tenant, snapshot_id, content_hash, config_version, scope)` | tenant+scope | either hash changes | process-local |
| Period-change results | YES | `(tenant, from_snapshot, to_snapshot, calculation_version)` | tenant | snapshot ids | process-local |
| **Provenance / lineage index** | YES | `(tenant, snapshot_id, content_hash, canonical_field)` | tenant+snapshot | content hash | process-local |
| Snapshot manifests | YES | `(tenant, snapshot_id)` | tenant | immutable | process-local |
| **Authorisation decisions** | **NEVER** | — | — | — | — |
| **Loan rows / any borrower data** | **NEVER separately** | — | — | — | (they live in the cached frame) |

Three rules, all already stated in `serving_cache`'s own docstring and worth
restating because A2A raises the stakes:

* **Immutable source identity, never time.** No TTLs anywhere in this table. A
  TTL is simultaneously unsafe (stale inside the window) and pointless (changed
  content already misses).
* **Tenant and scope always in the key.** Not a discipline — enforced by the key
  builder refusing to construct a key without them.
* **Never cache an authorisation decision.** Caching the *config* an
  authorisation reads is safe and is the fix in §1. Caching the *decision* means
  a revoked grant keeps working, which is the one failure mode this architecture
  must not have. The distinction is exactly why `ExecutionContext.entitlement_version`
  exists — it is the fingerprint that makes a revocation invalidate a cache.

---

## 10. Concurrency, async work, and multi-tenancy

### Concurrency model as it stands

Agent routes are sync `def`, so FastAPI runs them in an anyio threadpool (default
40 threads). Blocking pandas work is therefore already off the event loop —
correct. Two consequences worth knowing: the concurrency ceiling is the threadpool
size, and pandas holds the GIL for parts of its work, so 40 concurrent
million-row filters will not deliver 40× throughput. At Scale C this argues for
more workers rather than more threads, which in turn argues for the Parquet
projection (each worker's memory footprint is what limits worker count).

**Handlers are stateless.** Verified: `trakt_tools` module state is `_REGISTRY`
and `_LOADED` (both write-once at import) plus frozen constants. Nothing
per-request, nothing per-tenant. **Multiple instances can be added freely** — the
tool layer has no horizontal-scaling obstacle at all.

**A single agent session can safely issue concurrent read-only calls.**
`ExecutionContext` is a frozen dataclass; `currency` and `request_scope` are
`contextvars`-based and therefore per-task; the cached frame is not mutated in
place (the executor copies before filtering).

### Sync vs async

| Stays synchronous | Becomes asynchronous |
|---|---|
| `get_loan` / `get_loans` (≤500) | whole-portfolio DD sweep |
| `explain_value` / `explain_values` (≤500) | securitisation-readiness review |
| `portfolio_summary`, `stratify`, `concentration` | full period comparison over large history |
| `evaluate_covenants`, `covenant_drillthrough` | regulatory generation (already job-shaped) |
| tool catalogue | bulk evidence pack |
| `rank_loans` | cross-portfolio analytics |

**Trakt already has the job infrastructure.** `POST /mi/decks/generate` +
`GET /mi/decks/generate/{job_id}` is exactly the pattern; `operations_control`
carries `WorkflowRun` with an enforced transition table, `idempotency_key` and
restart safety; `trakt_notifications/outbox.py` is a durable at-least-once outbox.
**Do not add Celery, Temporal or a broker.** The gap, when it arrives, is one
generic `agent_job` resource reusing the deck-generation shape — and it is a
Sprint 3+ concern, not Sprint 2.

### Multi-tenancy — the honest position

**The canonical read path is deployment-scoped, not tenant-scoped.**
`data_source.get_dataframe()` takes no tenant argument; `resolve_data_source()`
reads deployment environment variables (`MI_AGENT_PLATFORM_URI`,
`MI_AGENT_CLIENT_ID`); `_ACTIVE_CACHE` is a single slot. One process serves one
tenant's data. Today that is correct and safe — deployment-per-tenant is the
stated production model and it gives physical isolation for free.

It is also the constraint that decides Scale C. Serving 150 clients means either
150 deployments (operationally heavy but genuinely isolated) or a tenant-keyed
dataset cache (one code change, `_ACTIVE_CACHE` → `BoundedCache` keyed on tenant
+ signature, plus threading the tenant through `get_dataframe`). The second is
not hard, but it must be a deliberate decision, because it trades physical
isolation for logical isolation.

Checked and sound today: agent identities are organisation-bound and resolved
only from a signature-verified directory; `ResourceRef` is `{tenant}/{kind}/{id}`
so portfolio ids **cannot collide across tenants**; `authorise_resource_access`
re-checks `ref.tenant_id != context.tenant_id` as defence in depth even though
config validated it; audit records carry both `tenant_id` and `organisation_id`.

### Cross-organisation A2A — architectural prerequisites only

The Sprint 3+ situation: Buyer Agent (org A) interacts through Trakt with Seller
Agent (org B) about a portfolio owned by org B.

| Concern | Prerequisite |
|---|---|
| **Permissions** | A holds `dd:request` on B's resource; B holds `dd:respond`. Asymmetric grants in the existing model — **no new mechanism needed**. |
| **Workflow state ownership** | The DD engagement is a **third-party object**, owned by neither. Store it under the *data-owning* tenant (B) with A recorded as counterparty, so evidence never leaves B's partition. |
| **Evidence ownership** | Values in a `DDResponse` must be re-resolved by Trakt from B's governed data and stamped with B's provenance — never authored by A's or B's agent. This is a control, not an optimisation. |
| **Audit** | Two views of one chain: B's tenant partition holds the authoritative record; A receives its own calls. Correlation id joins them. **Do not write a cross-tenant chain** — it would put two organisations' records in one integrity boundary. |
| **Provenance leakage** | A `provenance_ref` must never be dereferenceable by an organisation without a grant on the referenced resource. Today refs are opaque strings; before A2A they need an authorisation check on dereference. |

---

## 11. Audit scalability

**Two audit paths exist and they must stay separate. That separation is already
right and should be preserved deliberately.**

| | `trakt_core.audit` (log line) | `OpsStore.append_audit` (hash chain) |
|---|---|---|
| Cost | **10 µs** measured | read + 2 writes per record |
| Purpose | operational observability | tamper-evident evidence |
| Volume | every tool call | governed decisions |
| Scaling | trivially — it is a log line | **does not scale as written** |

### The log path is not a bottleneck

10 µs per event, no lock beyond `logging`'s. At 10,000 tool calls/hour that is
0.1 seconds of CPU per hour. Nothing to do.

### The hash chain has a correctness defect, not merely a performance one

`OpsStore.append_audit` (`operations_control/stores.py:371`):

```python
head = _read_json(...)              # read seq = N
seq  = int(head.get("seq") or 0) + 1
...
_write_json(self.storage, self.layout.audit_uri(client_id, seq), record)   # write N+1
_write_json(self.storage, self.layout.audit_head_uri(client_id), {...})    # write head
```

`_write_json` is atomic *per document* (temp+rename locally, atomic blob upload)
but there is **no compare-and-swap**: `upload_blob(overwrite=True)`, and no etag
precondition is used anywhere in `operations_control`. Two concurrent appends for
the same client both read `seq=N`, both write to the *same* URI for `N+1` — one
silently overwrites the other — and both write the head.

The result is a **lost audit record in a chain that still verifies**, because the
surviving record's `prev_hash` matches. `verify_audit_chain` cannot detect it.
For an evidence trail in institutional credit, that is the worst failure shape:
silent, and invisible to the integrity check built to catch it.

At pilot scale, with a human clicking through the OCC, this effectively never
fires. The moment agents write concurrently — which is exactly Sprint 3 — it
becomes likely.

### Recommendations

1. **Do not hash-chain every tool call.** Chain *governed decisions and workflow
   transitions* — DD requests, responses, decisions, escalations, approvals. Tool
   calls go to the log path and to a partitioned append-only store. A chain over
   high-frequency reads buys nothing: nobody disputes a read, and every entry is
   a serialization point.
2. **Chain per engagement, not per client.** Each DD engagement gets its own
   chain. Independent engagements then have no contention with each other, and
   the sequence that matters (what happened in *this* negotiation) is exactly
   what is chained.
3. **Add compare-and-swap before any concurrent writer exists.** The `Storage`
   abstraction already exposes `etag()`; the head write needs an
   if-match precondition and a bounded retry. This is the smallest change that
   turns a silent lost update into a visible retry. **Do it before Sprint 3, not
   after.**
4. **Partition and archive.** `list_audit` currently lists a prefix and reads
   every blob — O(n) per query. Partition by `{tenant}/{engagement}/{yyyy-mm}/`,
   keep a small index document per partition, and archive closed partitions to
   cool storage. Retention is cheap; queryability is what costs.

| Layer | Immutable? | Fast query? | Retention |
|---|---|---|---|
| Operational log (`trakt.audit`) | no (rotates) | via log store | 90 days |
| Tool-call record (partitioned, append-only) | yes | by correlation id | 1–3 years |
| Decision chain (hash-chained per engagement) | **yes, verifiable** | by engagement | 7+ years, archived |

Preserved throughout: actor, organisation, capability, resource, outcome,
correlation, decision basis. Never stored: model reasoning, prompts, answer
bodies, loan rows — `trakt_core.audit._FORBIDDEN_KEYS` already enforces this on
the log path and the same denylist should apply to the evidentiary path.

---

## 12. Workflow state and configuration at scale

### Workflow state (`OpsStore`)

| Scale | Verdict |
|---|---|
| **Pilot** | ✅ Appropriate. JSON documents in blob/file storage, human-paced writes, no contention. |
| **Early platform** | ⚠️ Works with CAS on state transitions (§11) and a per-engagement index document to avoid list-and-read-all. |
| **High-volume A2A** | ❌ Migrate workflow + audit to a transactional store. |

Specific gaps, all Sprint 3 concerns rather than Sprint 2:

* **atomic transitions** — the transition *table* is enforced
  (`contracts.transition`, `IllegalTransition`) but the *write* is not
  conditional. Two agents can drive one object into an inconsistent state;
* **optimistic locking** — no `version` field on `WorkflowRun` or `OnboardingCase`.
  Adding one now is nearly free and makes CAS trivial later;
* **idempotency** — present where it matters most already (`WorkflowRun.idempotency_key`,
  `snapshot/store.py` content-hash registration, `annex_delivery_agent` run
  identity). DD request/response writes need the same, keyed on
  `(engagement, request_id, actor)`;
* **query patterns** — `list_decisions` / `list_audit` read every document.
  Acceptable at hundreds, not at tens of thousands.

**Migration trigger (T1):** more than one concurrent writer per engagement, or
`>1,000` workflow objects per tenant, or a p95 workflow query above ~500 ms.
Postgres (or SQLite for a single-writer deployment) for workflow + audit
**only** — canonical analytical data stays in files.

### Configuration

YAML is genuinely the right choice today and should be defended, not apologised
for: it is diffable, reviewable, versionable in git, and it makes an access
change a pull request. Keep it.

It stops being right at a measurable point, and the point differs per file:

| Config | Keep as YAML until | Then |
|---|---|---|
| `fields_registry.yaml`, `business_semantics_registry.yaml`, `entity_model.yaml`, `field_roles.yaml`, `concentration_test_library.yaml` | **indefinitely** — these are platform definitions that change with releases, not with customers | — |
| `organisations.yaml`, `principals.yaml` | ~50 organisations, or changes more than weekly | governed store via `access_admin` |
| `resources.yaml` | ~500 resources, or when resources are derived from the portfolio registry rather than hand-written | generate from the registry (`resources_from_portfolio_records` already exists) |
| `entitlements.yaml` | ~500 grants, or when a single review cannot hold the whole file in mind | governed store |
| per-client covenant/portfolio config | already in the OCC store | — |

**The migration path already exists and does not need designing.**
`operations_control/access_admin/` implements
`AccessChangeSet → apply → validate → DRAFT → named-human confirm → ACTIVE →
materialise` over `ConfigPackageStore` (content-hashed, immutable, versioned,
rollback-capable) with hash-chained audit. Crucially, it **materialises published
YAML** that `load_organisation_registry()` and friends read. So the store can
become the administration surface with the runtime unchanged — the loaders keep
reading files. Versioning and auditability are preserved by construction.

**Do not migrate now.** Two organisations and four grants belong in a file.

---

## 13. Cost model

Assumptions, clearly labelled and adjustable:

* container ≈ £0.15/hour compute; blob storage ≈ £0.02/GB/month;
* canonical tape 100 columns; agent projection ~15 columns;
* warm dataset cache (a cold worker adds the CSV/Parquet load once, not per call);
* LLM: input ≈ £2.50/M tokens, output ≈ £10/M tokens;
* a competent review issues ~40 tool calls, ~8k input / 1.5k output tokens each
  once results and history are in context.

### One autonomous securitisation-readiness review

| | 1,000 loans | 10,000 loans | 100,000 loans |
|---|---|---|---|
| **Trakt — data load** (cold, CSV) | 0.02 s | 0.23 s | 2.3 s |
| **Trakt — data load** (cold, Parquet projected) | <0.01 s | 0.02 s | 0.05 s |
| **Trakt — analytics** (~40 calls: stratify, concentration, covenants, period change) | ~1 s | ~3 s | ~15 s |
| **Trakt — `get_loans`** (≤500 exceptions, batched) | 0.01 s | 0.02 s | 0.08 s |
| **Trakt — `explain_values`** (≤500, indexed) | 0.05 s | 0.05 s | 0.05 s |
| **Trakt compute total** | **~1 s** | **~3 s** | **~18 s** |
| **Trakt compute cost** | **<£0.01** | **<£0.01** | **~£0.01** |
| **LLM cost** (~40 calls) | **~£1.40** | **~£1.40** | **~£1.40** |
| **Storage** (tape + valuations + lineage, Parquet) | negligible | ~£0.01/mo | ~£0.10/mo |
| **Total per review** | **~£1.40** | **~£1.40** | **~£1.45** |

**The finding is the shape, not the number: cost is dominated by the LLM and is
almost flat in portfolio size.** That is exactly what a well-designed agentic
architecture should look like — the agent reasons over *aggregates and
exceptions*, and Trakt does the per-loan work vectorised.

Now the anti-patterns, priced. Same 100,000-loan review, done badly:

| Anti-pattern | Consequence | Cost |
|---|---|---|
| **one LLM call per loan** | 100k calls | **~£3,500** — 2,400× worse |
| **one `get_loan` per loan** | 100k calls × (config 7 ms + lookup 12 ms) | ~32 min of server time, plus context overflow |
| **the same at Scale B config cost** | 100k × 239 ms | **~6.6 hours** of pure config parsing |
| **repeated full-dataset scan** (no cache) | 40 calls × 30.8 s CSV | **~20 min** vs ~18 s |
| **provenance reconstructed per cell** | 100k × 130 fields | unbounded |

Every one of these is avoided by architecture rather than by discipline: batching
in the contract, caching on content identity, aggregate-first tool design, and a
lineage index that is per-field rather than per-cell.

---

## 14. Target performance characteristics

Ranges, not point targets, and split by whether the operation should *feel*
interactive. Warm cache assumed; the cold-start path is a separate concern
addressed by the Parquet serving copy.

| Operation | Target p95 | Feel |
|---|---|---|
| `GET /v1/agent/tools` | < 50 ms | instant |
| `get_loan` | < 150 ms | interactive |
| `get_loans` (≤500, projected) | < 500 ms | interactive |
| `explain_value` | < 200 ms | interactive |
| `explain_values` (≤500) | < 1 s | interactive |
| `portfolio_summary` | < 500 ms | interactive |
| `evaluate_covenants` | < 1.5 s | interactive |
| `stratify` / `concentration` | < 1 s | interactive |
| `covenant_drillthrough` | < 1.5 s | interactive |
| `period_change` (two snapshots) | 2–10 s | slow but synchronous |
| whole-portfolio evidence pack | minutes | **job** |
| securitisation-readiness review | minutes | **job** |
| regulatory generation | minutes | **job** (already job-shaped) |

Two service-level statements matter more than any of these numbers:

* **cold start is a separate budget.** A worker that has just started pays the
  dataset load once. Target < 5 s with the Parquet serving copy; it is 31 s
  today at 1M rows with CSV.
* **governance overhead must be < 10 ms of any call, at any scale.** It is
  currently 7 ms at pilot and 5,098 ms at Scale D. That single line is the
  clearest statement of the top finding.

---

## 15. Findings classified

### MUST CHANGE BEFORE SPRINT 2

Three items. All small; all become materially harder once `get_loan` /
`explain_value` contracts exist.

| # | Change | Why before | Effort |
|---|---|---|---|
| **M1** | **Cache the five governance registries on file identity** (`mtime_ns:size`), and reuse a resolved `EntitlementStore` across requests keyed on its fingerprint | Sprint 2 adds high-frequency tools; 7 ms → 239 ms → 1.7 s per call is paid on every one of them. Fixing it after means every Sprint 2 performance number is measured against noise. | ~30 lines + tests |
| **M2** | **Make `get_loans(loan_ids[])` the primitive; `get_loan` sugar over it** | 21–37× measured penalty, and a published agent contract is the hardest thing in this system to change. This is a *contract* decision, not an optimisation. | Design only |
| **M3** | **Add `explain_values([...])` alongside `explain_value`, and emit a per-field `lineage_index` at ingestion** | Same contract argument as M2; and the index is per-field (130 rows), not per-cell, so it is cheap and must not be confused with materialising provenance | Design + small pipeline addition |

### SHOULD CHANGE DURING SPRINT 2

| # | Change | Note |
|---|---|---|
| S1 | Parquet serving copy written alongside the canonical CSV, with the CSV's `content_sha256` embedded; fall back to CSV when absent | Unlocks the 190× column-projection win |
| S2 | Default field projection (~25 fields) for `get_loans`, `fields` parameter to widen | What makes S1 pay |
| S3 | `valuations.parquet` sidecar + the six-entity model in `fields_registry.yaml` / `entity_model.yaml` | §5 |
| S4 | Record `selected_valuation_id` + `valuation_policy_version` as tape columns at transform time | Makes `explain_value(current_loan_to_value)` a lookup, not a re-derivation |
| S5 | Result caps, pagination, `truncated` + `next_cursor` on every list-returning tool | Cheap now, awkward to retrofit into a published contract |
| S6 | Telemetry fields (`rows_scanned`, `rows_returned`, `bytes_returned`, `cache_hit`) on the tool result path | Measure before policing |
| S7 | `serving_cache` entries for `portfolio_summary`, covenant results, provenance | Reuse the existing class; do not write a new cache |

### DEFER — no need at current scale

Graph database. Warehouse/lakehouse. Microservices. Kubernetes. Celery/Temporal.
A separate agent-facing data store. Materialised per-cell provenance. Field-level
ABAC. Cross-tenant shared cache. Read replicas. A second calculation
implementation of anything.

### MIGRATION TRIGGERS — define now, build when observed

| Trigger | Observable condition | Change then |
|---|---|---|
| **T1 — transactional workflow store** | >1 concurrent writer per engagement, or >1,000 workflow objects/tenant, or p95 workflow query >500 ms | Postgres/SQLite for workflow + audit only |
| **T2 — governed config store** | >50 organisations, or >500 grants, or access changes more than weekly | `access_admin` becomes the admin surface; loaders unchanged |
| **T3 — indexed serving layer** | Median tool call touches <100 loans but pays a full load; or >10 valuation observations per collateral | Parquet row-group stats → DuckDB over Parquet |
| **T4 — warehouse** | A question spans >50 portfolios, or >10M loans in one query | Lakehouse; canonical files remain the source |
| **T5 — tenant-keyed dataset cache** | One process must serve >1 tenant | `_ACTIVE_CACHE` → `BoundedCache` keyed on tenant + signature |
| **T6 — audit CAS** | **Before Sprint 3** — any concurrent audit writer | etag if-match + bounded retry on the head write |
| **T7 — job infrastructure** | An agent operation exceeds ~30 s | Generic `agent_job` reusing the deck-generation pattern. Not a broker. |

---

## 16. Evolution roadmap

```
NOW  ── Sprint 2 ─────────────────────────────────────────────────────────
  authoritative : CSV                    unchanged
  serving       : + Parquet copy         additive, falls back to CSV
  valuations    : valuations.parquet     new, narrow, sidecar
  config        : YAML, CACHED           M1 — the one blocking fix
  contracts     : get_loans / explain_values are the primitives
  caching       : serving_cache, content-identity keys
  audit         : log line per call; chain for decisions only
  workflow      : OpsStore (+ version field, added cheaply now)
        │
        │  trigger T1: concurrent workflow writers  (arrives with Sprint 3)
        ▼
NEXT ── Sprint 3 A2A ─────────────────────────────────────────────────────
  + audit compare-and-swap (T6) — before any concurrent writer
  + per-engagement audit chains, partitioned
  + DD objects with idempotency keys and optimistic locking
  + generic agent_job for long operations (T7)
        │
        │  triggers T2 (>50 orgs) · T3 (loan-level access dominates)
        ▼
SCALE ── established platform ────────────────────────────────────────────
  + workflow + audit in a transactional store
  + access administration through the governed config store
  + DuckDB over Parquet for loan-level serving
  + tenant-keyed dataset cache (T5) OR deployment-per-tenant
        │
        │  trigger T4: cross-client analytics · >10M loans
        ▼
FUTURE ── A2A infrastructure ─────────────────────────────────────────────
  + lakehouse for cross-client analytics
  + canonical files STILL the authoritative source and the regulatory path
```

**The property that makes this roadmap safe:** every stage happens *behind an
unchanged tool contract*. `get_loans(resource, loan_ids)` returns the same
conceptual object whether it is served from CSV, Parquet, DuckDB or a warehouse,
because the agent names a **resource**, never a dataset, a path or a format — and
`ResolvedResource.predicate()` already produces the filter server-side. That
indirection, built in Sprint 1 for security reasons, turns out to be the thing
that makes storage replaceable.

---

## 17. Final recommendation

**If this were my company, I would spend the next sprint on three things and
deliberately not on infrastructure.**

**1. Fix the configuration reload before writing a line of Sprint 2.** It is
thirty lines. It is worth 7 ms today and 1.7 seconds at the scale the business
plan implies. More importantly, it is the difference between measuring Sprint 2's
real performance and measuring YAML parsing. `serving_cache` already contains the
right pattern — identity-keyed, bounded, no TTL — so this is reuse, not design.
The one discipline to hold: cache the *configuration*, never the *decision*, and
keep `entitlement_version` in any downstream key so a revoked grant invalidates
everything derived from it.

**2. Get the batch contracts right the first time.** `get_loans` and
`explain_values` as the primitives, with `get_loan` and `explain_value` as
documented conveniences. This costs nothing now and is close to unfixable later —
an agent contract published to a client's own agent is the one interface in this
system you cannot quietly change. The measured 21–37× penalty is the argument,
and the £3,500-versus-£1.40 line in §13 is what it looks like in money.

**3. Add Parquet as a serving copy and stop there.** Not as a replacement, not as
authoritative, not as a migration. One derived file next to the CSV, with the
CSV's hash inside it, falling back to CSV when absent. It buys a 190× improvement
on the access pattern agents actually have — narrow projections over wide tapes —
and it is deletable if it disappoints. Everything else in the storage debate
(DuckDB, Postgres, lakehouse) has a defined trigger and none of those triggers is
lit.

**What I would explicitly not do:** migrate to a database because agents are
coming; adopt a graph because entities now exist; introduce a job broker because
some operations are slow; split the API into services because it will grow. Each
of those would be justified by a hypothetical and paid for immediately. The
architecture reviewed here is unusually well positioned precisely because its
predecessors resisted that instinct — `trakt_core` has no framework dependency,
`serving_cache` has no TTL, the tool layer has no state, and the canonical
pipeline has no second implementation of anything.

**The biggest technical risk** is not scale. It is the audit chain's lost-update
race (§11). It is silent, it is invisible to the integrity check built to catch
it, and it becomes likely at exactly the moment the system starts making claims
worth disputing. Fix it before Sprint 3, with a compare-and-swap on the head
write — not because of volume, but because an evidence trail that can lose a
record without saying so is worse than one that does not exist.

**The biggest efficiency risk** is treating the agent as a well-behaved caller.
It is not one. It will call `get_loan` a hundred thousand times if the contract
lets it, and it will do so at £0.035 a call. The defence is not rate limiting —
it is designing tools that answer the question in one call, and publishing
`agent_guidance` that says so. `ToolSpec.agent_guidance` already exists and is
already in the catalogue; Sprint 2 should treat it as load-bearing.

---

## Appendix — measurement provenance

Every figure was produced by running this repository's code in this container.
Ratios are the evidence; absolute milliseconds are container-specific.

| Measurement | Method |
|---|---|
| Config load by scale | Synthetic `organisations/resources/entitlements` YAML at four envelopes; timed `load_organisation_registry` ×2 + `load_entitlement_store` + `load_resource_catalogue` (the calls one governed request makes) |
| YAML parses per request | Instrumented `pathlib.Path.read_text` and `yaml.safe_load` around `identity.context_from_agent_principal` + `authorise_resource_access` |
| CSV/Parquet | 100-column canonical-shaped frame (matching the 66–130 columns in `*_canonical_typed.csv`), 100k and 1M rows; `pd.read_csv`, `pd.read_parquet` full / 5-column / predicate-filtered |
| In-memory ops | `df.copy()`, boolean filter, `groupby().sum()`, single-loan scan on the same frames |
| N+1 | 20 loan lookups individually (scan and `.loc`) versus one `.loc[[...]]` batch |
| Memory | `df.memory_usage(deep=True).sum()` |
| Audit / registry / schema | `emit_audit_event`, `trakt_tools.catalogue`, `schema.validate` over 2k–20k iterations |
| Concurrency, statelessness, mutation | Source inspection of `agent_api`, `trakt_tools`, `data_source._ACTIVE_CACHE`, `mi_query_executor` copy sites, `currency`/`request_scope` contextvars, `OpsStore.append_audit`, `Storage.write_bytes` |
