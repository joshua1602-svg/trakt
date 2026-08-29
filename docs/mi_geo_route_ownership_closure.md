# MI Agent — Geography Ranking vs Specialist Exposure: route-ownership closure

**STOP — CONTRACT STILL CANNOT SEPARATE ROUTE OWNERSHIP**

**Verdict: NOT READY FOR REAL-CLIENT-DATA ACCEPTANCE.** The blocker is unchanged
and is now diagnosed to the field level. Section 2 below is the report §5 asks
for: exactly what semantic information is missing.

Branch `claude/mi-query-agent-c7-2tlhr6`; baseline for every before/after number
is commit `d2543df`.

---

## A. ROOT CAUSE

Why generic regional ranking is claimed by specialist geographic exposure:

`chat_routing._is_geo_exposure` claims a question when a GEO TERM and a GEO
MARKER both appear:

```
_GEO_TERMS   = ("geograph", "region", "area", "postcode", …)
_GEO_MARKERS = ("concentrat", "largest", "biggest", "most exposed", "top ",
                "hotspot", "where is", "where are", "where's",
                "which region", "which area", "exposure")
```

Four of those markers — `largest`, `biggest`, `top `, `which region` /
`which area` — are RANKING words, not concentration words. So a plain ranked
stratification is captured whenever its axis happens to be geographic.

There is already an exclusion for exactly this, and it misses:

```
_GROUPED_RANKING_RE = r"\b(?:top|bottom|largest|biggest|smallest|lowest|highest)\b[^?]*?\bby\b"
```

It requires the literal word **`by`**. So:

| question | `_GROUPED_RANKING_RE` | owner |
|---|---|---|
| "What are the top three regions **by** balance?" | matches | generic ✓ |
| "Which region has the largest balance?" | **does not match** | geo_exposure ✗ |
| "Which region has the smallest balance?" | **does not match** | geo_exposure ✗ |

One analytical shape, two owners, decided by whether the sentence contains the
word "by". Traced on the shipped `/mi/query` path:

```
'Which region has the largest balance?'
  claim order   : [('geo_exposure', 0.5, 60)]
  _is_geo_exposure: True      _GROUPED_RANKING_RE: False
  answer        : "I can't build a geographic exposure view for this book:
                   no ITL3 field and no property postcode on the tape."
```

while the same book answers `"Show balance by region."` over seven governed
values of `geographic_region_obligor`.

---

## B. CONTRACT DISTINCTION — **there is none**

The prior revert's stated reason was *"no contract field separates 'where is the
book concentrated' from 'which region has the largest balance'"*. That was
re-measured against the current tree rather than assumed, because the contract
has gained `OperationClaim.type` and the four `ordering_*` fields since.

**The contract now separates SOME of the pairs, and not the one that matters.**

Measured on the shipped path (`scratchpad/geo_disc.py`, contract resolved
through `RouteRequest.resolve_interpretation()`):

| question | intended owner | `operation.type` | `ordering_direction` | `ordering_basis` | `ordering_limit` | grouping dim |
|---|---|---|---|---|---|---|
| Which region has the largest balance? | generic | **ranking** | increase | absolute | None | filled |
| Which region has the smallest balance? | generic | **ranking** | decrease | absolute | None | filled |
| What are the top three regions by balance? | generic | **ranking** | increase | absolute | 3 | filled |
| Which broker channel has the largest balance? | generic | **ranking** | increase | absolute | None | filled |
| Where is the book concentrated geographically? | specialist | amount | None | None | None | filled |
| Show geographic exposure. | specialist | amount | None | None | None | filled |
| Analyse geographic concentration. | specialist | count | None | None | None | — |
| **What is the largest geographic area concentration?** | **specialist** | **ranking** | **increase** | **absolute** | **None** | **filled** |
| **Which area has the largest concentration?** | **specialist** | **ranking** | **increase** | **absolute** | **None** | **filled** |
| Where are we most exposed geographically? | specialist | amount | None | None | None | filled |

The last two rows are the finding. **`OperationClaim.type = RANKING` plus a
governed grouping dimension is NOT sufficient**, because a specialist
concentration question phrased with a superlative claims exactly the same
shape as a generic ranking. Every other governed field is identical too:

```
                                    generic                      specialist
                        "…region has the largest      "…largest geographic area
                                    balance?"                 concentration?"
operation.type          ranking                      ranking
operation.state         filled                       filled
ordering_direction      increase                     increase
ordering_basis          absolute                     absolute
ordering_limit          None                         None
ordering_of             None                         None
modifiers               ()                           ()
subject.state           filled                       filled
subject.candidate       current_outstanding_balance  current_outstanding_balance
subject.span            Span(0, 37)  (whole question) Span(0, 50)  (whole question)
dimensions              [(collateral_geography,      [(collateral_geography,
                          'grouping'), …]              'grouping'), …]
residue                 []                           []
```

`ordering_requested` does not exist on `OperationClaim`; the ordering values are
the four typed fields above.

### What semantic information is missing

Either of these would separate the shapes, and the contract carries neither:

1. **A governed ANALYTIC / CAPABILITY claim** — a slot recording that the
   question asked for *geographic concentration* as an analysis, rather than
   for a measure ranked over a geographic axis. `QuestionInterpretation` has
   `operation`, `subject`, `dimensions`, `filters`, `time`, `target`,
   `population`, `row_predicates`, `source_scope`, `dataset`, `residue`,
   `notes`. None of them names an analytic. The word "concentration" leaves no
   governed trace at all: `residue` — the slot for wording no interpreter
   claimed — is EMPTY for both questions.

2. **Provenance on the ranked SUBJECT** — whether the reader NAMED the measure
   or the parser DEFAULTED it. In "which region has the largest **balance**?"
   the measure is in the sentence; in "what is the largest geographic area
   **concentration**?" it is not, and `current_outstanding_balance` is a parser
   default. `SubjectClaim` records neither a provenance flag nor a usable span:
   `subject.span` is the WHOLE QUESTION in every case, so it cannot say which
   words named the measure.

Adding either is a contract change — new capability — which this brief
prohibits. Hence the STOP.

**A wording-only fix exists and was deliberately not taken.** Narrowing
`_GEO_MARKERS` to genuinely specialist words (dropping `largest`, `biggest`,
`top `, `which region`, `which area`, keeping `concentrat`, `most exposed`,
`hotspot`, `where is/are`, `exposure`) separates all ten questions correctly.
It is a wording list, not the contract, and §4/§5 direct the contract to
determine entitlement. It is offered as the fallback if the contract is not to
be extended.

---

## C. ROUTE OWNERSHIP BEFORE / AFTER

| question | before | after |
|---|---|---|
| Which region has the largest balance? | geo_exposure | **geo_exposure (unchanged — blocked)** |
| Which region has the smallest balance? | geo_exposure | **geo_exposure (unchanged — blocked)** |
| Show balance by region. | generic | generic |
| Which broker channel has the largest balance? | generic | generic |
| What are the top three regions by balance? | generic | generic |
| Where is the book concentrated geographically? | geo_exposure | geo_exposure |
| Show geographic exposure. | geo_exposure | geo_exposure |
| Which region added the most balance since last month? | period_change_analysis | period_change_analysis |
| Which region lost the most balance since last month? | period_change_analysis | period_change_analysis |

**No route ownership changed.** An entitlement rule keyed on
`OperationClaim.type == RANKING` + a governed grouping dimension was built and
run end to end; it released A and B correctly, and it ALSO released
`"What is the largest geographic area concentration?"`, breaking
`test_geographic_exposure_routes_to_itl3_engine` — a guard protecting a
genuinely specialist request (a postcode tape, an ITL3 answer naming Bristol).
Per §6 that guard is valid and was preserved: the rule was reverted rather than
the test rewritten.

`chat_routing.py` is byte-for-byte unchanged by this task.

---

## D. FAIL-CLOSED PRESERVATION

Nothing in this task touches dispatch, and the previously removed post-claim
fallthrough was not reinstated. Proven, not asserted:

* **Mutation 3 — fault the specialist handler AFTER claim.** `_route_geo` was
  made to raise at its execution point. Both geography questions returned:

      ok: false
      "I could not complete this analysis: it failed while running.
       I have not answered your question with a different analysis instead."

  No generic route answered either one. Mutation restored.
* **Route substitution detector: `SUBSTITUTIONS: 0 of 2`**, with the boundary
  derived from the detector's own run (`fault executed 1 · handlers entered
  ['period_movement'] · alternate executed 0 · execution failure True`).
* `tests/test_failclosed_route_execution.py` — 9 passed.

Mutations 1 and 2 (remove the ranking exclusion; make a specialist request look
generic) target the entitlement rule that is not shipped, so they have no
subject. The evidence they were designed to produce is instead the measurement
in section B, which is why the rule is not shipped.

---

## E. QUALITY MOVEMENT

Two defects were found while measuring the targeted cases and are fixed. Neither
is a route change; both are attributed below.

**1. A grouped ranking ignored the direction the question asked for.**
`mi_query_executor` sorted grouped output `ascending=False` unconditionally,
although the spec had already resolved `sort_direction`. Measured at `d2543df`
on a NON-geographic dimension, which the geographic route never touched:

    "Which broker channel has the smallest balance?"
        Delta Advisers   £49,050,182     <- the LARGEST, returned first
        Alpha Network    £41,654,473
        Gamma Direct     £40,884,938
        Beta Partners    £40,465,954     <- the answer, returned last

"largest" and "smallest" returned byte-identical results. The loan-level ranking
path and `_apply_top_n` both honoured `sort_direction`; only the grouped path did
not. `sort_direction` defaults to `"desc"`, so the fix is a no-op for any spec
that did not ask to ascend, and `concentration_pct` is a per-row share that does
not depend on row order.

**2. A ranking question was answered with a group count.** The lead sentence was
*"Here is the bar for your query, covering 4 groups"* — the same sentence
whichever direction was asked for. It now names the group, from the first row of
the already-ordered result and through the same formatter the KPI and table
artifacts use:

    Delta Advisers has the highest Current Outstanding Balance: £49.1MM (4 groups).
    Beta Partners has the lowest Current Outstanding Balance: £40.5MM (4 groups).

A plain breakdown ("Show balance by region.") keeps the neutral lead.

`tests/test_grouped_ranking_direction.py` (9 tests) pins both. Against the
pre-fix tree, 4 of the 9 fail. Under Mutation 4 (direction reversed), 5 fail.

### Frozen surfaces — before (`d2543df`) and after

| surface | before | after | gate |
|---|---|---|---|
| CFO bank (91) | EXACT 66 · FALSE_REFUSAL 10 · TRUE_REFUSAL 13 · DISCLOSED 2 · **WRONG/SILENT 0** | **identical** | WRONG/SILENT = 0 ✓ |
| supplement (24) | CORRECT 20 · UNCLEAR 1 · SAFE REFUSAL 3 · **WRONG/SILENT 0** | **identical** | WRONG/SILENT = 0 ✓ |
| categorical sweep (69) | CORRECT NARROWING 56 · HONEST REFUSAL 10 · UNCLEAR 3 · **SILENT DROP 0** | **identical** | SILENT DROP = 0 ✓ |
| collision sweep, real book (46) | CORRECT 42 · HONEST REFUSAL 2 · SILENT DROP 0 · WRONG ADDITIONAL CLAIM 2 | **identical** | SILENT DROP = 0 ✓ · WRONG ADDITIONAL CLAIM = 2 ✗ |
| collision sweep, synthetic (58) | CORRECT 54 · HONEST REFUSAL 2 · SILENT DROP 0 · WRONG ADDITIONAL CLAIM 2 | **identical** | as above |

Zero movement on every frozen bank; the two fixes above are the only behavioural
change and neither is exercised by a bank question. The two WRONG ADDITIONAL
CLAIMs are the documented single-token `origination_channel = direct` residual,
preserved untouched as instructed.

---

## F. REGRESSION

Authoritative 278-module MI regression, run alone:

| | baseline | after |
|---|---|---|
| passed | 5957 | **5957** |
| failed | 81 | **81** |
| skipped | 711 | **711** |
| xfailed | 15 | **15** |
| errors | 4 | **4** |
| hung | 0 | **0** |
| failing/erroring names | 85 | **85** |

**introduced = 0 · fixed/removed = 0.** Exact-name diff empty in both
directions.

*(A first run of this regression was launched concurrently with five bank
processes and reported 5936 passed with a per-test timeout in
`test_mi_query_invariants`. That is contention, not a measurement, and it is
excluded. The clean re-run above is the reported one.)*

### Architecture controls

| control | result |
|---|---|
| post-claim raw-question semantic reads | **0** in all eight categories |
| route-local semantic vocabularies | **0** |
| substitution detector | **0 of 2** |
| compound canary | **11 passed, 0 breaches** |
| semantic / migration guard suite | **239 passed**, 45 skipped |

No frozen migration guard was weakened, rewritten or skipped.

---

## G. MANUAL CLIENT-FACING REVIEW

| question | answer | reading |
|---|---|---|
| Which region has the largest balance? | "I can't build a geographic exposure view for this book: no ITL3 field and no property postcode on the tape." | **the blocker, unchanged** |
| Which region has the smallest balance? | as above | **the blocker, unchanged** |
| Show balance by region. | "Here is the bar for your query, covering 7 groups." | correct |
| Where is the book concentrated geographically? | specialist degradation, honest | correct for its data requirement |
| Show geographic exposure. | as above | correct |
| Which region added the most balance since last month? | "…ranked by absolute balance movement across Geographic Region Obligor, Scotland increased the most: Scotland £16.5m → £28.9m (+£12.4m, +75.0%)…" | unchanged, correct |
| Which region lost the most balance since last month? | "…London decreased the most: London £24.8m → £22.4m (−£2.4m, −9.6%)… 5 further categories did not decrease." | unchanged, correct |

No answer exposes a route name or implementation rationale.

---

## H. REMAINING BLOCKERS

1. **Generic point-in-time regional ranking is claimed by specialist geographic
   exposure**, and the governed contract cannot be used to stop it: a
   specialist concentration question phrased with a superlative carries an
   identical `OperationClaim`. Resolving it requires one of:
   * extending the contract with a governed ANALYTIC claim, or with subject
     provenance / a narrow subject span (section B) — a contract change; or
   * narrowing `_GEO_MARKERS` to genuinely specialist wording — a wording
     change, which separates all ten measured questions correctly but is not
     contract-driven.

   Both are decisions for the guard's author. Neither was taken here.

2. **A route that cannot build its view still returns `ok: true`** with a
   non-answer. Independent of (1), and pinned by
   `test_geographic_exposure_degrades_honestly_without_itl3_or_postcode`, which
   asserts `ok is True` for that envelope.

Also observed, out of scope, unchanged, and not fixed:
`"What are the top three regions by balance?"` returns all seven groups. The
contract carries `ordering_limit = 3`; `spec.top_n` is `None`, because the
parser's top-N detector does not read the word "three". The limit is dropped
without disclosure. Pre-existing at `d2543df` and not caused by this task.
