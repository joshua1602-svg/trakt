# C7 post-claim closure — pre-registration

**Written and committed BEFORE the production edit.** Base `5f8a9b6`. Nothing
here is adjusted afterwards; a breach is reported as a breach.

---

## 1. The three sites, recomputed from the current tree

`migration_phase0.analytical_meaning_census` re-run on this tree still reports
exactly three post-claim raw-question semantic reads, all in
`route_period_change`, and none anywhere else in C1–C6.

| # | site | line | decides | contract field that already carries it |
|---|---|---|---|---|
| 1 | `recognise(question, spec, view, semantics_context)` | 302 | `matched` (the decline), `mode`, `requested_fields`, `period_request`, `include_bridge`, `composition_focus` | — it is the SAME call the registry already made pre-claim |
| 2 | `chat_routing._resolve_lens(question, source_lens)` | 307 | the source-portfolio lens | `interpretation.source_scope` (`state`, `scope`, `portfolio_ids`, `provenance`) |
| 3 | `_period_request.requested_span(question)` | 372 | the requested time span | `interpretation.time.window_periods` via the existing `span_from_claim` |

Site 3's contract field was added FOR THIS: `TimeClaim.window_periods`'s own
docstring names `requested_span(question)` as *"a second read of the sentence
for a fact the contract had already claimed"*. C2 was closed against it; C7 was
not.

## 2. Equivalence, measured over 882 corpus questions before any edit

```
SITE 3  requested_span(question)  vs  span_from_claim(contract.time)
        agree 882   disagree 0

SITE 2  _resolve_lens(question, None)  vs  lens built from contract.source_scope
        agree 882   disagree 0     contract source_scope state: filled 882

SITE 1  recognise() on identical inputs
        identical 882   differing 0
```

Caller precedence measured separately, because a workspace selection is the case
where the two could legitimately differ. With `source_lens="acquired"` over the
first 400 questions: **agree 400, disagree 0**, contract provenance
`caller_context` 399 / `explicit_user` 1 — the one question that names its own
scope correctly overrides the selection.

**A correction recorded rather than hidden.** The first run of that precedence
check passed `{"scope": "acquired"}`, which `lens_from_selection` does not
recognise; both sides therefore saw no selection and "agree 400" meant nothing.
It is re-run above with a selection the resolver actually accepts, which is what
made the `caller_context` provenance appear.

## 3. The intended change

* **Site 1** — the registry already runs this recogniser pre-claim and discards
  the result. Memoise it on the `RouteRequest`, exactly as `interpretation` is
  memoised, and have the handler consume it. The wording read moves to
  **pre-claim**, which this task permits. **No new flag and no new concept.**
* **Site 2** — build the lens from `interpretation.source_scope` by MAPPING the
  contract's scope onto `mi_agent.portfolio_lens`'s own constructors
  (`total_lens`, `lens_from_term`, `_selection_lens`). `portfolio_lens` remains
  the only thing that decides what a scope MEANS; this transports its answer,
  the way Conversion 1 does. Source lens and row predicates stay separate axes.
* **Site 3** — replace with the existing governed `span_from_claim(qi.time)`.

## 4. Pre-registered blast

**Expected: 0.** All three are equivalence-preserving substitutions measured at
882/882 before the edit. Any movement in a normal-path answer, route, economics
or receipt is a breach, not a finding.

Preserved: C1–C6; C7 ranked and filtered ranked movement; the live movement
receipt; LEVEL/MOVEMENT; ranking direction/basis/limit; D1, D2, D4; fail-closed
routing; pre-claim fallback; predicate execution parity; stage/funnel;
source-portfolio scope behaviour; honest missing-measure and missing-period
refusals; frozen canary history.

## 5. Stop conditions

* **STOP — C7 POPULATION CONTRACT INCOMPLETE** if any supported population case
  cannot be represented. *Not triggered:* 882/882 filled and agreeing.
* **STOP — C7 TIME CONTRACT INCOMPLETE**. *Not triggered:* 882/882 agreeing.
* **STOP — C7 STILL DEPENDS ON ROUTE-LOCAL MEANING** if removing `recognise`
  needs a new route-specific concept. *Not triggered:* the pre-claim result is
  carried, not replaced.
* **STOP — POST-CLAIM CLEANUP CAUSED BLAST** on any unexplained movement.
* **STOP — TARGET STATE NOT YET REACHED** if closure needs broad new semantic
  infrastructure.
