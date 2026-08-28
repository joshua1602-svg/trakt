# Semantic coverage — architecture diagnosis and recommendation

Baseline `137dbb3`. **No production code was changed.** Diagnosis, measurement
and architecture selection only.

---

## 1. Q16B: the exact first divergence

Traced over 10 live runs, arm active.

| | proposals | outcome |
|---|---|---|
| 9 runs | `["drawdown","balance","geography","ltv band"]` | CORRECT, 39 groups |
| **1 run** | **`[]` — the model returned no concepts at all** | WRONG, 42 groups, whole book |

The omission run is not a mis-binding or a wrong role. It is an **empty
proposal**: the model produced nothing, the arm degraded exactly as designed
(`applied: []`, deterministic contract untouched), and the deterministic
contract for this question never had `drawdown` either — it is the same
contract that grades WRONG with the arm off.

**The first point the architecture could have known** is the moment the facet
ledger is built, before reconciliation. And it is precisely there that it
cannot know, for a structural reason:

```
_facets = detect_requested_facets(question, semantics, frame=df,
                                  requested_dimensions=_requested_dims,
                                  resolved_filters=set(spec.filters))
```

Measured directly on Q16B with no resolved filters — the omission situation:

```
requested_dimension_terms  -> [collateral_geography, ltv_bucket]
detect_requested_facets    -> grouping_dimension 'geography'
                              grouping_dimension 'ltv band'
                              (NOTHING for drawdown)
```

**The execution-gating ledger is derived partly from the contract it exists to
check.** A concept that never reached `spec.filters` raises no facet, so there
is nothing for reconciliation to mark lost and nothing for the guard to refuse.
Absence is unobservable to the object whose job is to observe it.

### The evidence was already in the building

A different owner sees it. On the same question:

```
completeness.stated_concepts(...) ->
   ('dimension', 'collateral_geography', ...)
   ('dimension', 'ltv_bucket', ...)
   ('value',     'erm_product_type', 'drawdown')     <-- here
   ('measure',   'current_outstanding_balance', ...)
```

`stated_concepts` reads the question against the **book's own value catalogue
and the registry's dimension terms**, through the existing owners
(`categorical_spans.value_field`, `requested_dimension_terms`,
`portfolio_lens`, with owner precedence and span masking). It is not a phrase
list and it has no lexicon of its own.

## 2. Why the existing controls cannot see it

Every current control is a **relation between two records that both exist**:

| control | compares | blind to |
|---|---|---|
| monotonic merge | proposal vs deterministic slot | a proposal never made |
| `OperationProfile` role rules | proposed role vs operation | a role never proposed |
| binding (ambiguous/unregistered) | proposal vs registry | a proposal never made |
| receipt facet reconciliation | facet ledger vs execution | a facet never raised |
| requirement coverage | plan vs intent requirements | concepts, which it does not model |

None of them is wrong. They are all **conditional on a claim existing**. The
missing invariant is the only unconditional one: *did every governed concept
the question states reach the contract at all?*

## 3. Which measured failures share the class

Simulated `stated_concepts` + `unresolved_concepts` against the **live
envelope** for 1,612 questions (166 graded bank + 1,446 surface), deterministic
arm; and 166 on the Opus arm. Nothing was changed — this measures what a gate
*would* do.

| failure | shares the class? | caught? |
|---|---|---|
| **Q16B** drawdown omitted | yes — value concept lost | **yes** |
| **Q17C** borrower-age omitted | yes — measure concept lost | **yes** |
| Q03A, Q05C, Q07B (deterministic) | yes | **yes** |
| **Q19A** | yes — `scope portfolio_lens=direct` never carried | **yes** |
| Unknown/Missing age | partly — 3/6 whole-book substitution | see §6 |
| **Q04C** | **no** — right population, wrong output shape | no |
| **Q10B** | **no** — "size" is not in any registered vocabulary, so no owner states it | no |

Q04C and Q10B are **not omission failures** and coverage does not claim them.
Q04C loses nothing; it aggregates at the wrong grain. Q10B's concept is
unstateable — no owner recognises the bare word "size", so nothing can report
it missing. Stating that plainly matters more than the headline count.

## 4. Semantic-evidence inventory

Everything needed already exists:

| object | what it already provides |
|---|---|
| `question_interpretation/schema.py` | `Span` (char offsets), `Slot.state` ∈ {filled, empty, unresolvable}, provenance ∈ {explicit_user, caller_context, model_inferred, default, unresolved}, and **`residue: "Wording no interpreter claimed"`** |
| `completeness.stated_concepts` | governed concepts read from the question by the **existing owners**, with precedence and masking |
| `completeness.ExecutedContract.from_envelope` | "THE ONE ADAPTER from an answer envelope" |
| `completeness._carried` | mature reconciliation: role disagreement is not loss; bands resolve via registry `derived_from`; scope must be *applied*, not merely resolved; dataset must be *reconciled*, not merely decided |
| `concept_proposal` / `claim_merge` | proposals, spans (`covers`), governed binding, monotonic merge, `OperationProfile` |
| receipt / reconciliation | per-facet applied/lost/unsupported records |

**A documented limit, and it is the reason a span ledger cannot be the
foundation:** `Slot` states that *"170 of the dimension and filter claims
raised across 690 real-surface questions have no recoverable span"*. Spans are
optional in this estate and a consumer *"must never require it"*. A design that
gates on span coverage would be gating on evidence that is absent a quarter of
the time.

**The authoritative owner should be `question_interpretation.completeness`,
promoted from a library nobody calls to the pre-answer coverage gate.** It is
already the only object that (a) enumerates stated concepts from governed
owners and (b) owns the single envelope adapter. Nothing else is a candidate,
and adding a second would recreate the two-owners defect this programme has
already hit three times.

## 5–6. Candidate comparison, against the measured corpus

| candidate | verdict | evidence |
|---|---|---|
| **A** registry-backed concept coverage | **core of the answer** | this *is* `stated_concepts`. Catches 6/8 deterministic WRONG including Q16B. Adds no lexicon: it asks the value catalogue and registry terms, so a product type added tomorrow is covered with no code |
| **B** independent LLM critic | **reject as the mechanism** | the omission run returned an *empty* proposal — a failure mode a second constrained pass can share. Unbounded latency/cost for a control that A already provides deterministically. Its only unique reach is unstateable concepts (Q10B), which it cannot bind anyway |
| **C** repeated extraction + union | **reject** | measured omission ≈1/10; two passes → ~1%, not an invariant. This is the brief's own "run Opus twice and hope". Doubles cost, closes nothing structurally |
| **D** span/disposition ledger | **shape of the answer, not its evidence** | correct as the *contract*; cannot be founded on spans, per the 170/690 measurement above. Found on **concepts**, which every owner does emit |
| **E** hybrid with a critic | **reject the critic half** | measured false-refusal on correct answers is already **0**; there is no residual for a critic to earn its cost against |

### The measurement that decides it

Deterministic arm, 1,612 questions:

| | flagged | clean |
|---|---|---|
| CORRECT | **0** | 117 |
| WRONG | **6** | 2 (Q04C, Q10B) |

Opus arm, 166 bank:

| | flagged | clean |
|---|---|---|
| CORRECT | **0** | 122 |
| WRONG | 3 | 2 |

Five former regressions: **all clean**. Seven recoveries: five clean; **Q16B
and Q17C flagged precisely on the runs where the concept was lost** — the same
questions are clean and CORRECT on runs where Opus proposes the concept. The
control fires on the omission, not on the question. That selectivity is the
whole requirement.

### The honest qualification

Across all 980 questions that answer, **43 (4.4%) would be flagged**, and the
graded slice understates this because the 1,446 surface is ungraded. Split:

| cause | n | is it a wrong answer? |
|---|---|---|
| routes that narrow correctly and **publish no record** (`portfolio_summary`, `funded_bridge`, `evolution`, `cohort_progression`) | 12 | **no** — recording gap |
| **dataset not reconciled** on forecast/analytical routes | 13 | **no** — recording gap |
| candidate genuine losses (Q03A/Q05C/Q07B/Q16B/Q17C class) | 18 | **yes** |

`Summarise the acquired book` answers correctly — 199 loans, £54.7m — and is
flagged only because `portfolio_summary` never publishes `scopeApplied`. That
is the standing finding *"OPEN · Two routes narrow correctly and record
nothing"*, still open, and **it is the gating blocker: 25 of the 43 are routes
failing to disclose, not answers failing to narrow.**

This is why the recommendation ships in two stages rather than one.

---

## 7–10. Recommended architecture

**Promote `question_interpretation.completeness` to the authoritative
pre-answer semantic-coverage gate, at the single envelope seam, in two stages.
No second model call.**

### A. Authoritative objects — one each, no competition

| concern | owner |
|---|---|
| semantic claims stated by the question | `completeness.stated_concepts` (delegating to existing owners) |
| what execution recorded | `completeness.ExecutedContract.from_envelope` |
| **coverage / disposition** | `completeness.unresolved_concepts` — **the new authority** |
| canonical binding | unchanged: registry / `concept_proposal.bind` |
| execution eligibility | the gate, at the envelope seam in `mi_service` |

### B. Runtime flow

```
question
  ├─ deterministic parse ──────────────► spec (claims + provenance)
  ├─ [MODEL] concept proposal (Opus) ──► registered vocabulary only
  ├─ deterministic binding ────────────► governed fields (registry)
  ├─ monotonic merge + OperationProfile► contract          [unchanged]
  ├─ route / planner / executor ───────► answer envelope   [unchanged]
  │
  └─ COVERAGE GATE  (deterministic, no model call)
        stated_concepts(question)   ⨯   ExecutedContract.from_envelope(envelope)
                     │
             unresolved_concepts
                     │
        ┌────────────┴────────────┐
     none missing              ≥1 missing
        │                          │
     ANSWER                 STAGE 1: disclose + instrument
                            STAGE 2: REFUSE, naming the concept
```

Every model boundary is upstream of the gate; the gate itself never calls a
model and never edits the contract.

### C. Fail-closed rule — exact

> Execution may be **answered** only if every concept in
> `stated_concepts(question)` is `_carried` by `ExecutedContract.from_envelope(envelope)`.
> Any concept not carried is `UNRESOLVED`, and an answer carrying an
> `UNRESOLVED` concept is refused, naming the concept and the words that
> produced it.

No confidence scores, no thresholds, no model judgement. Ordinary language is
not in the ledger at all: a word no governed owner claims never becomes a
concept, which is why the measured false-refusal rate on correct answers is 0
rather than "tuned low".

### D. Q16B, both runs

**Opus proposes `drawdown`** — bound to `erm_product_type`, merged, executed;
`filters` contains `erm_product_type`; `_carried` → true; **answer stands**
(39 groups). Measured: 9/10 runs, CORRECT, unflagged.

**Opus omits it** — nothing merged; the contract has no `erm_product_type` in
`filters`, `dimensions` or `applied_fields`; `stated_concepts` still reports
`('value','erm_product_type','drawdown')` from the book's value catalogue;
`_carried` → false → **UNRESOLVED → refuse**. Measured on the live envelope for
exactly this contract: flagged. **Whole-book execution becomes a named
refusal.** Never silent.

### E. Model reach preserved

The gate reads only the question and the envelope. It never blocks a proposal,
so every recovery the model earns still lands: Q01C, Q02B, Q03C, Q05C, Q03A
measured CORRECT and unflagged on the Opus arm, and all five former regressions
clean. It is not a phrase parser because it owns no phrases — it asks the same
owners the parser asks, and its reach grows when theirs does.

### F. Extension model

A new product type, portfolio, field, dimension or capability enters the value
catalogue / registry, and `stated_concepts` covers it **with no change here**,
because it enumerates from those sources rather than from a list. The one
obligation a new *route* carries is the same one this exercise exposes: publish
what it narrowed. That is a route contract, not coverage code.

---

## 11–15. Impact

**Measured**

* Q16B omission ≈ 1/10 runs (10 traced) and 1/6 (6 traced, separate session).
* 5 of 1,446 model replies unparseable in the last full sweep — total omissions.
* Coverage over 1,612 deterministic: 0/117 correct flagged, 6/8 wrong caught.
* Coverage over 166 Opus: 0/122 correct flagged.
* 43/980 answering questions flagged; 25 attributable to route non-disclosure.
* Latency/cost: **zero additional model calls**; `stated_concepts` runs in the
  same pass that already builds facets. Measured indirectly: the 1,612-question
  simulation added no perceptible time to a run that is dominated by I/O.

**Inferred**

* Neutralises Q16B, Q17C and Q19A as *wrong answers*; Q19A becomes a refusal
  rather than the correct answer the reverted route gate produced.
* Unknown/Missing age: the 3/6 whole-book runs state
  `('measure','youngest_borrower_age')` and answer over 640 loans — expected to
  flag, not separately confirmed run-by-run.

**Unknown until implementation**

* The true correct/incorrect split of the 18 candidate genuine losses on the
  ungraded surface.
* Whether closing route disclosure introduces its own movement.

---

## 16–17. Implementation plan — not implemented

### Minimum production-safe version

**Stage 1 — ledger + disclosure (safe to ship immediately)**

* `mi_agent_api/mi_service.py`: call the gate at the single envelope seam, both
  routed and point-in-time. **~40 LOC.**
* `question_interpretation/completeness.py`: add `coverage_report(question,
  envelope, semantics, …) -> {stated, carried, unresolved}` wrapping the two
  existing functions. **~40 LOC.**
* publish `metadata.semanticCoverage` with each concept and its disposition
  (`RESOLVED` / `UNRESOLVED`); no behaviour change.
* instrumentation: log unresolved concepts with question id, route, kind, field.

**Stage 2 — route disclosure (the actual blocker)**

Close the 25 recording gaps so the gate can fail closed without refusing
correct answers. Each route publishes what it already did:

* `portfolio_summary`, `funded_bridge`, `evolution`, `cohort_progression` →
  `metadata.scopeApplied` (~10–20 LOC each);
* forecast/analytical routes → derive `reconciliation.dataset` via the existing
  `workspace.reconciliation_for` primitive already adopted at five sites.

**Stage 3 — flip to fail-closed.** One flag. Acceptance below.

**Production footprint: ~80 LOC new + ~100 LOC route disclosure.** No new
module, no new ontology, no model call.

### Optional later enhancement

Extend `Slot.span` capture so the refusal can quote the exact words. Valuable
for the user-facing message; **the safety property does not depend on it**, and
the 170/690 span-absence measurement says it must not.

---

## 18–19. Acceptance protocol

Stochastic, so every criterion is over **≥10 independent Opus invocations**.

**Safety** — Q16B wrong frequency **0/10** (refusal is an acceptable outcome);
Unknown/Missing age wrong frequency 0/10; no must-refuse question answered;
no deterministic claim disappears; no model-selected canonical field; no
invented period/metric; no dataset substitution. Q04C/Q19A: coverage claims
**Q19A only** — Q04C must be reported as still wrong, not silently counted.

**Utility** — the seven recoveries available ≥ their current rate; five former
regressions 10/10 correct; **flagged-correct count must be 0** on the graded
banks; deterministic arm byte-identical.

**Blast** — full 1,446 both arms, 75, CFO 91, six pipeline answers, frozen
278-module manifest exactly 85. Every movement enumerated and categorised.

**Gate-specific** — Stage 1 must ship with `semanticCoverage` published and
**zero** answer movement. Stage 3 may not be flipped until the Stage 1
telemetry shows the flagged-answering-question count is composed only of
genuine losses.

## 20. Residual risks

1. **Unstateable concepts remain invisible.** Q10B proves the limit: no owner
   recognises "size", so nothing reports it missing. Coverage raises the floor
   to *what the estate can name*, not to what a person can say.
2. **Wrong-grain answers are not omissions.** Q04C keeps its right population
   and wrong shape; this control cannot see it.
3. **Stage 3 depends on Stage 2.** Flipping to fail-closed before route
   disclosure closes would refuse ~25 correct answers. That ordering is the
   single most important implementation constraint here.
4. **`_carried` is now load-bearing.** It is currently a library with tests; as
   a gate its rules (role disagreement, `derived_from` bands, scope-applied)
   become production semantics and need adversarial review.
5. Concept-level coverage does not prove *value* correctness — that a filter
   was applied, not that it was applied to the right rows.

---

# IMPLEMENT THIS ARCHITECTURE

It closes the measured hole: Q16B's omission run becomes a named refusal
instead of a whole-book answer, verified against the live envelope for exactly
that contract. It preserves the model's reach — 0 of 122 correct Opus answers
flagged, all five corrections and the proposing runs of every recovery clean.
It scales through the registry and value catalogue rather than through phrases,
so a product type added tomorrow is covered without an edit. And it is small:
the enumerator, the envelope adapter and the reconciliation rule all exist and
are tested; what is missing is a call site and a route-disclosure backlog.

The gating condition is honest and specific: **do not flip it to fail-closed
until the 25 route non-disclosures are closed.** Stage 1 is safe to ship as it
stands and will measure the rest.
