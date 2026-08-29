# Standing findings — MI Query Agent interpretation programme

Findings that outlived the stage that produced them. Each is here because it
changed how the next piece of work was scoped, and each is stated so that it can
be checked rather than believed.

---

## F1 · A vocabulary gap does not produce silence. It produces the nearest expressible thing.

**Measured, Stage 3.** The concept vocabulary offered to the model was
constrained hard, and both constraints held exactly as designed:

- **zero** raw governed field keys proposable;
- **zero** fields this book does not carry proposable.

And the vocabulary still produced a three-way narrowing error.

There is no concept kind for a numeric threshold, so *"borrowers over 75"*
cannot be said in registered concepts at all. The model did not fall silent. It
reached for the nearest expressible things: the FIELD as a `measure` — declined
32 times — and the BUCKET VALUES as category values, `75-80`, `80-85`, `85+`,
the first of which filled an empty slot. *"Balance for borrowers over 75"*
became *"balance for `age_bucket` == 75-80"*, on a question that is EXACT today.
8 of 157 correct-today questions would have taken that fill.

**The rule.** A constraint that cannot express a concept does not stop the model
reaching for it. When designing a constrained vocabulary, the question is not
"can it say anything wrong?" — it is **"can it say everything the sentence
says?"** Where it cannot, enumerate what the nearest expressible substitutes are
and measure whether they are reached, because the substitution is silent by
construction: every proposal is in-vocabulary, every binding is registry-owned,
and every guard reports green.

This generalises past this programme. It is the same shape as a validator that
accepts only known enum members and a producer that has no member for the state
it is in.

---

## F2 · An API key in the environment switches on the shipped free-form parser arm, so any "deterministic before" captured that way is void.

**Measured, Stage 3.** `mi_agent_api.datasets._mi_llm_config` runs `auto` by
default and sets `enabled = has_key`. With `ANTHROPIC_API_KEY` present, every
`/mi/query` call is parsed by the free-form LLM arm — the one that emits a whole
`MIQuerySpec` — which is the arrangement the concept-proposal split exists to
replace.

The first full Stage 3 run was captured that way and had to be discarded. It was
caught only because two must-refuse questions answered while the merge had
filled nothing on either, and it took a stashed working tree to establish that
the merge was not the cause.

**The guard, and it stays.** Any harness measuring a deterministic baseline
while holding a key for its own model calls must refuse to run unless
`_mi_llm_config()` reports `enabled=False available=False`. `MI_AGENT_LLM_PARSER=off`
forces that while leaving the key available for direct calls. The check belongs
at the top of the harness, as a hard exit, not in its documentation:
`migration_phase0/must_refuse_both_arms.py` and the Stage 3 harness both carry
it.

---

## F3 · An instrument that cannot be measured must report NOT MEASURED, never clean.

**Stage 3.** `question_interpretation.mi_recognition_diagnosis` refused to run —
`TRAKT_RUNTIME_MODE` resolved to `production`, so `trakt_core.policy` would
refuse both books as synthetic fixtures and every shape would rate ABSENT. A
clean-looking zero would have been indistinguishable from a passing surface.

Reported as **not measured**. That stays the convention: the absence of a
finding is evidence about the instrument before it is evidence about the
product.

---

# TOP OF THE OPEN LIST

The two items below lead deliberately. The first is a deployment-shaped defect that
degrades governed answers with nothing in any bank to catch it. The second is a guard
evasion held open on purpose rather than closed as a side effect.

---

## OPEN · Five readers decide axis-vs-measure and give three answers

**A consolidation, and it needs its own scope. It is not a precedence rule and
must not be attempted as one.**

On the single word `ltv`, in one module:

| reader | says |
|---|---|
| `_explicit_dimensions` (`terms_map`: 251 registry + 56 curated) | not an axis |
| `_explicit_dimensions(grouping=True)` extra map | not an axis |
| `_classify_segment` | numeric axis → `ltv_bucket` |
| `_NUMERIC_AXIS_BUCKET` | `ltv_bucket` |
| `_detect_metric` | measure `current_loan_to_value` |

Three answers, five readers. Two of the disagreements are not about the sentence
at all — they are the same knowledge held twice, at different sizes:

* **Two bare-to-banded maps, 1,100 lines apart.** `_NUMERIC_AXIS_BUCKET`
  (llm_query_parser:1962) holds nine terms over four buckets. The
  `grouping=True` extra map (llm_query_parser:521) holds **one**,
  `{"age": "age_bucket"}`. So `grouping=True` resolves `borrower age` and not
  `ltv`, while the nine-entry map resolves both.
* **The registry declares the relation completely and no reader reads it.**
  `derived_from` carries 20 relations. Nine hand-written terms and one
  hand-written term stand in for it.
* **One list is simultaneously the axis map and part of the measure
  vocabulary.** `_metric_side_residue` (llm_query_parser:836) folds
  `_NUMERIC_AXIS_BUCKET`'s nine strings into the terms that make a word a
  measure. The same nine strings decide both roles, 1,100 lines apart.

Downstream, `execution_receipt.requested_dimension_terms` and
`concept_proposal.vocabulary` both delegate to `_explicit_dimensions` — the
reader that agrees with the branch that gets Q17C wrong. That is why
`notApplied` is empty on a question that lost two of three axes: the facet layer
is consulting the reader that already agreed with the mistake.

**Why this is a consolidation and not a rule.** Every one of the five readers is
internally correct for its own view of the sentence, and a genuine multi-measure
request — *"For the London book, give me balance, number of loans,
weighted-average LTV and average borrower age"* — needs its bare terms to STAY
measures. Any single precedence rule has to separate that sentence from Q17C,
and both put bare measure words after a preposition. Choosing between the five
without first deciding what each is FOR would pick a winner by branch order
again, which is the defect.

Measured in `migration_phase0/MI_COORDINATED_AXIS_SCOPE.md` §5.

---

## F6 · A configuration path resolved against the process working directory

`trakt_core.capability.load_registry` resolves
`config/system/mi_capability_registry.yaml` **relative to the process cwd**, and
`mi_agent_workflow._capability_explanation` wraps the load in a bare
`except Exception: return None`.

Run from anywhere but the repository root, the `FileNotFoundError` is swallowed
and two capability-aware refusals silently degrade:

```
from the repo root   "Cure Rate is measured ACROSS governed snapshots and MI Query
                      answers from a single dataset… request it through the governed
                      history tools, where the snapshot window is resolved."
from anywhere else   "I couldn't map this question to a governed analytic."
```

No wrong number either way — both refuse. What is lost is the governed
methodology and the route to the number, replaced by an admission of ignorance
about a question the estate can name precisely. **A deployment's answer quality
should not depend on which directory it was started from**, and an absorbed
`FileNotFoundError` is how it came to.

Found while checking review-pack fidelity: two CFO questions differed between
two runs of the same code on the same tape, and the only difference was cwd.

**Owner: separate work.** Every measurement in this programme is now run from
the repository root.

---

## OPEN · A documented precondition nothing enforces — fail-open guard, THIRD instance

Worth naming as a pattern, because three separate defects in this programme have
had the same cause: the estate knew a rule, wrote it down, and had no mechanism
that could tell whether it held.

| | the precondition | what enforced it |
|---|---|---|
| Q19C | routes must publish the narrowing they performed | nothing, until `metadata.scopeApplied` |
| `_unknown_named_book` | a capitalised run before a book noun is a name UNLESS generic | a hand-maintained word list, short by four |
| **the dataset class** | **routes ask `workspace.resolve_dataset` for the dataset** | **nothing — the sentence exists only in comments** |

The third is the cleanest specimen. `chat_routing.py` states it twice —

```
:262   "`workspace.resolve_dataset`, which is the single owner. Routes ask that owner"
:3495  "the route asks `workspace.resolve_dataset` for it"
```

— and **all four occurrences of `resolve_dataset` in the file are comments.**
`try_route` receives `view` and spends it on the value catalogue, the
interpretation projection and the ownership re-read; never on the answering
frame. `_route_portfolio_summary` takes no dataset parameter at all, so it could
not honour the request even if it asked.

**A precondition written in a comment is a wish.**

### THE TEST, and it needs no knowledge of this codebase

> **When a comment says another component "asks", "must", "already handles" or
> "is responsible for" — check that a call exists.**

Two lines of `grep` settle it. Run against `chat_routing.py` it returns four
mentions of `resolve_dataset` and zero calls; run against `_unknown_named_book`
it returns a guard list and no test of the property the comment describes. Both
times the estate had written the rule down and had nothing that could tell
whether it held.

The test generalises because the failure does: a comment is the cheapest place
to record an obligation and the only place that cannot enforce one. Anywhere a
component's correctness depends on a peer behaving a certain way, either the
call is there or the obligation is fiction — and prose about it reads exactly the
same in both cases.

Scoped in `migration_phase0/MI_COMPOSITE_BOUNDARY_SCOPE.md`.

---

## CLOSED · The parser emitted `intent='chart'` with `chart_type='none'`

One builder, one spec kind: `_bridge_recognizer` declared an output shape it did
not name, because the waterfall is built by the `funded_bridge` route rather than
the chart factory. 23 of 1,446 corpus specs carried the pair, all from that
builder. Invisible while the route claimed every such question; the moment one was
declined the reader saw *"chart_type 'none' is not valid for intent 'chart'"* — an
internal message, not a governed refusal.

Fixed at the owner (`intent="table"`, the pair the validator permits) and the
invariant enforced: `_deterministic_parse` is now a wrapper that **fails closed**
rather than emitting a self-contradictory spec.

**Only the self-contradiction, and measurement drew the line.** The validator's
other chart rules ask whether a chart can be RENDERED — 22 corpus questions carry
such a spec and are answered correctly by a route today. Failing those closed
would convert working answers into refusals. Zero answer movement.

---

## OPEN · CR4 — twenty-four questions with a proven capability and no recognition

The largest recoverable population in the estate, catalogued in
`migration_phase0/MI_CAPABILITY_RECOVERY.md`. Every one has either a CORRECT
sibling formulation of the same case or is answered by the merge arm, so the
analysis demonstrably works and the wording fails **before capability selection**.

Includes the funded/pipeline contribution family: *"What is the weighted expected
pipeline contribution?"* classifies into the same families as three sibling
wordings that build `['funded_balance_forecast', 'pipeline_completion_forecast']`,
and `plan_for` returns `None` for it.

**Not repairable with more deterministic phrase rules** — that is the F1 failure
mode at scale. Recorded for the semantic layer.

---

## OPEN · `funded_bridge` shadows a family whose replacement is unreachable

`funded_bridge` claims *"Show funded vs pipeline contribution."* and *"What is the
weighted expected pipeline contribution?"* and answers both with a funded-balance
bridge. The capability that computes them exists —
`executors.funded_balance_forecast` produces current funded, expected completions
from the open pipeline, the completed/withdrawn exclusion, and the forecast total.

**It cannot be narrowed yet.** Both wordings are CR4 and never reach that
capability, so narrowing would leave one answering a pipeline total that is not a
funded-vs-pipeline contribution and the other refusing. The replacement path must
be proven first.

Established while tracing: `weighted_expected_funded_amount` is **NaN for
COMPLETED and WITHDRAWN**, so the open-pipeline sum and the all-rows sum are the
same £1,320,000 and no double-count is possible through that column.

---

## OPEN · A public forecast capability with no MI caller

`mi_agent_api/forecast_bridge.py::compute_forecast_bridge` is public and
deterministic. Its only live callers are in `mi_agent_pptx` — the deck builder.
No MI route reaches it.

CR2-shaped, and **not actionable today**: the questions it might serve are CR4 and
never reach a planner, so wiring it would recover nothing. Recorded so the find is
not lost.

---

## CLOSED · `FORECAST_BREACH` now gates the current-state limits route

Two bounded rulings, implemented together.

**`risk_limits` is the governed owner of the approved client concentration tests
on today's funded book, and now declines a forward question.** Eligibility asks
`mi_workflows.analytical.intent.classify` whether the sentence carries
`FORECAST_PROJECTION` or `FORECAST_BREACH` — the governed intent owner, not a
second reading of the sentence and not a word test. `run_funded_vs_forecast` was
deliberately NOT wired in: it forecasts group shares against placeholder RAG
thresholds, names no approved test, needs a caller-supplied dimension and a
SnapshotStore the MI path does not construct.

**The intent owner was extended by one construction.** `" at risk of breach "` /
`" at risk of breaching "`, gated on `limiting` so it cannot widen any other
family, and deliberately NOT added to `_FORECAST_TERMS`, which every family
reads. `_LIMIT_HEADROOM_TERMS` already contains `" at risk "` — correct for
*"which limits are most at risk?"*, a ranking of today's headroom — but *"at risk
of BREACHING"* names a breach that has not happened, which is what *"do we expect
to breach"* already carries.

Verified apart: *"which limits are most at risk?"* still classifies HEADROOM,
current-only; *"which region is most at risk?"* is untouched.

**Measured. Exactly three answers moved, on each arm, all of them Q25:**

| id | before | after |
|---|---|---|
| Q25A | WRONG — today's status | **controlled refusal** |
| Q25B | WRONG — today's status | **controlled refusal** |
| Q25C | WRONG — today's status | `analytical_composition` — see below |

WRONG falls 11→8 (deterministic arm) and 6→3 (merge). CFO 91 byte-identical.
The six registered pipeline answers byte-identical. 278 modules: 85 failures
before and after. No new WRONG, no previously CORRECT answer changed.

The refusal is written by the existing facet layer, not at the route:

> *"…this asks about concentration limits, which are governed by the portfolio's
> limit schedule rather than by the loan tape; and this asks for a forward-looking
> figure, which needs a governed forecast rather than a current position. I have
> NOT substituted a current-position figure, because that would answer a
> different question from the one you asked."*

---

## CLOSED · Partial requirement coverage must decline, not substitute

A plan may claim a question only if it satisfies **every** governed requirement
the intent owner emitted. Partial coverage declines; it does not substitute the
part it can do.

**The defect, measured.** *"Based on the current book and forward pipeline, which
concentration tests are we at risk of breaching?"* emits
`requirements=('limit_evidence', 'forecast')`. The planner could supply
`forecast` — `funded_balance_forecast` + `pipeline_completion_forecast` — so it
claimed the question and answered *"Current funded balance is £172.1m… Forecast
funded balance: £173.4m."* Truthful, honestly reconciled against
`funded+pipeline`, and about **balance**, to a reader who asked which
concentration **tests** are at risk. Half the requirement met, the other half
quietly dropped.

**Enforced at the narrowest existing seam** — `planner.plan_for`'s return, via
`_unprovided_requirements`. The requirement comes from the governed intent owner;
the answer from each capability's own `produces` declaration
(`concentration_limits` produces `limit`, `funded_balance_forecast` produces
`forecast`). No word test, no question exception, no new intent family, no new
capability, no forecast methodology. A capability that later produces limit
evidence satisfies the requirement with no edit.

**Only the requirements a PLAN can honestly be judged on, and measurement decided
which.** `intent.unmet_requirements` judges what was ACTUALLY EXECUTED — its
`dataset`, `periods`, `grouping` and `populations` keys describe a finished
answer. The general form applied at the plan seam **over-declines**: six
legitimate funded-balance-forecast plans read `funded+pipeline` and failed a
`dataset != "pipeline"` string equality, and two pipeline plans failed a period
count no plan carries. Those stay where they already work — on the executed
answer in `mi_service._fail_closed_analytical`.

**Blast radius: 1 of 1,446.** 55 questions build a plan; exactly one declines.
The other 54 are unchanged, including every forecast question:

```
What funded balance should we expect once the current pipeline flows through?     unchanged
Show forecast balance by expected completion month.                               unchanged
If the current pipeline converts as expected, what will our funded balance be?    unchanged
How much do we currently have at offer and how much of it is likely to complete?  unchanged
```

**One answer moved per arm** — Q25C, from the balance composition to the same
controlled refusal Q25A and Q25B give, citing the same governed boundary. CFO 91
byte-identical. Six registered pipeline answers byte-identical. 278 modules: 85
failures before and after. No new WRONG, no previously CORRECT answer changed, no
movement outside Q25C.

Q25A, Q25B and Q25C now all controlled-refuse for one boundary, and forward
approved-limit testing remains the separate future capability it was.



---

## OPEN · The funded-vs-forecast concentration primitive cannot answer the limits question

`mi_agent.risk_monitor.run_funded_vs_forecast` is real, deterministic, and runs
clean on the live MI frames — funded 640, pipeline 8, forecast 645 — returning
per-group `baseline_share`, `current_share`, `share_change`, `increasing`,
`status_current`.

**It has no concentration TESTS in it.** `status_current` is RAG against
`get_concentration_thresholds` — `{"amber": 0.20, "red": 0.30}` — a generic share
band. The approved tests live in `config/client/risk_limits_config.py::ALL_LIMITS`,
which is what `risk_limits` reads, and `config/mi/risk_monitor.yaml` says so
outright: *"Thresholds below are PLACEHOLDER defaults… client concentration limits
will live in a separate config/client/ file."*

```
risk_limits             "6 breach(es)… Nearest to limit: Top 3 brokers
                         (-31.5 pp headroom)"        <- named tests + headroom
run_funded_vs_forecast   eight regional shares, all green, no test named
```

Three further blockers, each sufficient on its own: **no `SnapshotStore` exists
on the MI path** (abstract, one adapter, zero references under `mi_agent_api/`,
none in the fixture); it requires a **caller-chosen `dimension`** the question
does not supply; and its thresholds are not the approved limits.

And the route that DOES own the approved limits never gets the question:
`mi_workflows/analytical/planner.py::plan_for` returns **None for all six**
concentration questions, current and forward, so `analytical_composition` never
claims one — even though `concentration_limits` (`funded`, `limits`) and
`funded_balance_forecast` (`funded`, `pipeline`) both exist as capabilities.

**Owner: separate work, and it is a methodology question, not a routing one.**

---

## OPEN · The separator evasion in `_unknown_named_book`

Deliberately NOT closed by the sentence-position property that shipped in
`8a4f2ab`, and deliberately not closed as a side effect of anything since.

"Direct-book" yields the token `Direct-`. `_PROPER_TOKEN_RE` admits an internal
hyphen, so the trailing separator comes with it, and `direct-` matches no entry in
`_GENERIC_BOOK_WORDS` or any other word list — the run is therefore judged a proper
name **whatever the list holds**. Same shape as the verb defect the property fixed
(a token the guard cannot match, rather than a word nobody listed); different
mechanism.

Four questions still refuse with `'Direct- book'`. One survives in the data-claim
audit's `QUOTES_A_MANGLED_PHRASE` class, which fell 2 → 1 rather than to 0 for
exactly this reason.

**It is coupled to the unshipped hyphen fix** in `portfolio_lens._qualified_span_re`,
and that coupling is why it waits. A simulated edge-punctuation normalisation removes
all four (7 fragments removed, 0 raised) — but it changes behaviour on the same
questions the separator fix touches, so shipping it alone would decide the hyphen
question by side effect. Measured in `migration_phase0/MI_DIRECT_COLLISION_SCOPE.md`.

---

## OPEN · A live wrong answer on the shipped path, independent of this split

**Not caused by this programme, and not fixed by it.** Logged here so it is not
absorbed into the interpretation work.

With an `ANTHROPIC_API_KEY` present and `MI_AGENT_LLM_PARSER` unset:

| question | outcome | repeats |
|---|---|---|
| `What changed?` | refused | 5 / 5 |
| `Show me the trend.` | **ANSWERED** | 5 / 5 |
| `Compare us with the market.` | **ANSWERED** | 5 / 5 |

*"Compare us with the market."* is answered with `640 loans · Current
Outstanding Balance: £172.1MM`. **There is no market data in this platform.** A
whole-book figure is returned to a question asking for a comparison against
something the estate has never held, and nothing in the answer says so.

All three refuse on the deterministic arm and the frozen CFO bank has expected
`REFUSE` for all three since it was written. The mechanism is that the free-form
arm supplies the missing element itself, so no governed default is ever
recorded and the guards that fire on a recorded default never see one — the same
mechanism the Opus acceptance run walked through.

**Owner: separate work.** Recorded every run by
`migration_phase0/must_refuse_both_arms.py`, which exits non-zero only if the
DETERMINISTIC arm stops refusing and prints the LLM arm in full regardless.
Failing the instrument on a known-open finding would only teach the estate to
stop running it.

---

## OPEN · The receipt stamps a threshold applied/lost the wrong way round

**Found while measuring the threshold kind. Not caused by it, not fixed by it.**

*"How much outstanding balance do we have where borrower age exceeds 75 and LTV
is over 40%?"* (Q02B) publishes:

```
served facets : ('threshold', 'LTV over 75', 'applied')
                ('threshold', 'LTV over 40', 'lost')
spec filters  : ('current_loan_to_value',)
```

Both labels are wrong — `execution_receipt._detect_thresholds` does not resolve
the field, so the borrower-age threshold is labelled `LTV` — and the two
statuses are **inverted against the contract**: the facet stamped *applied*
names a predicate the spec does not carry, and the facet stamped *lost* names
one it does.

The consequence for anything reading the receipt is that a threshold the
contract holds reads as lost and one it does not hold reads as satisfied. It is
why Q02B was classified as a threshold loss and is not one.

**Owner: separate work.** Recorded in
`migration_phase0/MI_THRESHOLD_KIND_RESULT.json` under
`prediction_B.why_the_third_did_not_land`.

---

## OPEN · A registry gap the estate reports to the reader as a data gap

**Found in Stage 4. Three questions, one registry entry.**

*"Show a table of balance by LTV bucket and interest-rate bucket."* (Q13A, and
its two siblings Q13B and Q13C) is answered with:

> 'interest rate bucket' is not available in this dataset. This book does not
> report it, so the question cannot be answered from the current data (no value
> was fabricated).

**The book reports it.** `interest_rate_bucket` is a fully populated column on
the acceptance tape — 640 of 640 rows, five bands: `4-5%`, `5-6%`, `6-7%`,
`7-8%`, `>=8%`. What is missing is the **registry declaration**: the field is
not in `semantics["fields"]`, so `requested_dimension_terms` resolves nothing
for any spelling of the term, no owner claims it, and the concept vocabulary
cannot offer it either.

The refusal is honest about having fabricated nothing and **false about the
data**. A reader is told the book does not hold something it holds.

This was classified as a capability gap in the frozen 75-bank grades. It is
not. Every other bucket axis on this tape — `ltv_bucket`, `age_bucket`,
`ticket_bucket` — is declared and works.

**Owner: separate work, and the cheapest remedy measured in Stage 4** — three
questions for one registry entry and no code, against the threshold kind's
three questions for a new concept kind.

---

## CLOSED · The registry gap the estate reported as a data gap

Fixed in `9968025`. `interest_rate_bucket` declared from the same template as
the other derived buckets; Q13A/B/C `FALSE_REFUSAL -> CORRECT` on both arms at
30 cells against the frozen truth; `data_claim_audit`'s
`FALSE_about_the_book` class fell 3 -> 0 and its pre-registered set is now
**empty**, so any refusal that starts lying about a client's book fails the
audit on its first run.

---

## F1, SECOND INSTANCE · A guard list is not a guard

The standing finding says *a vocabulary gap does not produce silence, it
produces the nearest expressible thing*. `portfolio_lens._unknown_named_book`
was the second instance, and it is worth recording because the failure was
louder than the first.

The function reads a run of capitalised tokens before "Book"/"Portfolio" as a
proper name. Its only guard was `_GENERIC_BOOK_WORDS`, whose last block is
commented *"question scaffolding that can be sentence-initial or capitalised"*
and holds `show`, `give`, `summarise`, `provide`, `list`, `report`. It did not
hold `break`, `split`, `plot` or `which`. So

    "Break Direct portfolio balance down across LTV, ticket size and borrower age."

refused with *"'Break Direct portfolio' is not a governed portfolio for this
book"* — the reader's own verb quoted back as the name of a book they never
named. Not silence: the nearest expressible thing.

**The fix was a property, not a longer list.** A sentence-initial token is
capitalised by orthographic convention whatever it is, so its capital is no
evidence of a name; discount it. Measured over 1,446 corpus questions: three
fragments removed, none raised, every genuine unknown book name still caught.
A list is a closed set; sentence position is a property of every sentence.

**The corollary, recorded because it is the more useful half.** The list was
not two words short. It was short by however many words nobody had thought of,
and there was no way to know which — `which` is not even a verb. Whenever a
guard is a hand-maintained list of the members of an open class, the question
to ask is what property the members share, not which ones are missing.

**Still open in the same function:** a token carrying a trailing separator.
"Direct-book" yields `Direct-`, and `direct-` matches no entry in any word
list, so the run is judged a name whatever the list holds. Same shape — a token
the guard cannot match rather than a word nobody listed. Coupled to the
unshipped qualifier/noun separator fix; four questions still refuse with
`'Direct- book'`.

---

## OPEN · Coordinated axis lists are read as measures — DIAGNOSED, NOT FIXED

Scoped in `1aaf52f`. **Not fixed**, and the diagnosis changed what it is.

**It is a bounded rule, not coordination parsing.** The coordination is already
parsed: `_grouping_segments` splits Q17C into three segments and
`_classify_segments` resolves all three axes with their bands. The correct
builder exists and already handles three or more axes
(`_build_multi_dim_table_spec`). The failure is **branch order**: the
multi-measure branch returns first because `detect_measure_set` finds balance +
LTV + borrower age, masks their spans, and keeps one dimension.

**Bare vs banded is the variable, not coordination.** A bare axis fails ALONE —
"Show balance by LTV." parses zero dimensions. Coordination enters only as an
arity effect, and backwards: the TWO-item bare coordination works, the
THREE-item one fails, because the third bare term is the second measure.

**It is NOT Q22C.** Q17C's segmenter returns three segments; Q22C's returns
zero. Q22C needs an elided head noun distributed over both conjuncts and nothing
in the estate does that. Q17C does not join it.

**Worth one question.** Across 1,446 corpus questions, 152 name two or more
resolvable axes and **three** lose one. **No currently CORRECT answer is in the
affected set, on either arm.** Recovery is Q17C (WRONG on the deterministic arm,
**already CORRECT on the merge arm**) and Q12C (FALSE_REFUSAL, and an adjacent
trigger — the word "plot" sets `explicit_plot` and skips the grouping branch).
The third refuses for an unrelated reason.

**The risk is not in the affected set.** Those three bound the recovery. A
precedence change is judged against the **149** questions that name two or more
axes and get them all today, and against genuine multi-measure requests whose
bare terms must stay measures.

Measured in `migration_phase0/MI_COORDINATED_AXIS_SCOPE.md`.

---

## OPEN · The comparison recogniser does not accept "How do X and Y differ?"

    "Compare the Direct and Acquired books."             -> portfolio_risk_comparison
    "How do the Direct and Acquired portfolios differ?"  -> no route

The second falls to the generic path. On the deterministic arm it answers with a
plain 640-loan count — an answer to a different question, escaping a WRONG grade
only because there is no computable truth for it. On the merge arm the merge
fills `source_portfolio_type`, the generic path cannot honour it, and the
silently-dropped-dimension guard refuses. Every component behaved correctly; the
gap is the recogniser's phrasing coverage.

Recorded here because it was previously mislabelled — by me — as a merge-arm
duplicate dimension. It is not: `spec.dimensions=['source_portfolio_type']` with
`spec.dimension='source_portfolio_type'` is the documented single-axis
convention, and the "duplicate" was an artefact of the review pack's own capture
concatenating the two spec fields.

---

## OPEN · Three routes answer from the funded book when the question named the pipeline

Diagnosed in `963a60d`, and **half addressed**: all five hard-coded sites now
DERIVE what they read. The read itself is unchanged and the narrowing decision is
still open.

**Four questions, not fifteen.** Eleven of the fifteen were a NAMING mismatch:
`analytical_composition` and `forecast_extrapolation` publish
`datasetContext: forecast` beside `reconciliation.dataset: funded+pipeline`, and
the forecast frame IS funded plus pipeline — measured at 645 rows = 640 + 5.

**The four, and their routes:**

```
portfolio_summary  Summarise the current pipeline.
risk_limits        Based on the current book and forward pipeline, which
                   concentration tests are we at risk of breaching?
funded_bridge      Show funded vs pipeline contribution.
funded_bridge      What is the weighted expected pipeline contribution?
```

**The data is present and the capability works.** `_resolve_frame('pipeline')`
returns 8 rows, £3.6m, 5 stages, and a sibling formulation already answers from
it — *"Give me an overview of the pipeline by size and stage"* falls to the
point-in-time path and reconciles `pipeline`. The point-in-time path honours the
dataset; the routed path does not.

### What deriving established, and it is the input to the narrowing decision

All five sites now call `workspace.datasets_read(output_root=...)` and report what
that returns. **The derived value equals the value they used to assert, at every
site: `funded`.** Nothing moved — no answer, no grade, no `reconciliation.dataset`
string, on either arm of the 166 and across the 1,446.

That null result is the finding. **The substitution is a READ failure, not a
LABELLING failure.** The three routes were not mislabelling a pipeline read as
funded; they genuinely read only the funded book, and none of the five even takes
a `pipeline_root` parameter. A labelling change therefore cannot fix these four
questions, and no amount of better disclosure will: only narrowing, refusing, or
routing elsewhere can.

What deriving DID buy is that the claim is now produced by the same code path that
did the reading, so a route that later consumes `pipeline_root` says so without
anyone remembering to edit a string — and the class cannot silently regrow, which
is why all five were converted rather than the three that had been reached.

### The narrowing decision, resolved on evidence

Diagnosed in `MI_NARROWING_DECISION.md` from live call paths — reads intercepted
inside `try_route`, one fresh process per question, caches off.

**Nine of nineteen live routes genuinely support the pipeline. None of the three
substituting routes is among them,** and none takes a `pipeline_root` parameter.
`risk_limits`' apparent pipeline read is `portfolio_context.
discovered_pipeline_portfolios` — **portfolio discovery, computing nothing.**
Counting an enumeration read as a capability would have credited a route with
semantics it does not have; reads are classified by call stack into *answer* and
*discovery*, and only *answer* reads count.

Simulating the three declining a non-funded question decides each case:

| case | outcome if the route declines | decision |
|---|---|---|
| Summarise the current pipeline. | `(point-in-time)` · `pipeline` · **8 loans · £3.6MM** — the constructible truth | **(a) route elsewhere** |
| …concentration tests at risk? | `(point-in-time)` · `pipeline` · a **governed refusal** naming the capability boundary | **(b) refuse** |
| Show funded vs pipeline contribution. | ✗ *"chart_type 'none' is not valid for intent 'chart'"* | **(b), BLOCKED** |
| Weighted expected pipeline contribution? | ✗ same | **(b), BLOCKED** |

**Cases 3 and 4 are blocked by a defect that has nothing to do with datasets.**
`_deterministic_parse` emits `intent='chart'` with `chart_type='none'` for both,
with no routing involved — an internally inconsistent spec, pre-existing and
currently MASKED because `funded_bridge` claims the questions before the spec is
validated. Narrowing first would trade a wrong answer for an internal error
message, which is a different failure rather than a better one.

**No capability is broadened.** `portfolio_summary` computes funded balance and
WA LTV, which the pipeline frame does not carry (its measure is
`weighted_expected_funded_amount`); `funded_bridge` attributes movement in funded
balance, which a pipeline case has none of. Adding `pipeline_root` would invent a
capability, not honour a request.

**Order: narrow `portfolio_summary`, then `risk_limits`, then fix the spec
inconsistency, then narrow `funded_bridge`.** Nothing built yet.

**Owner: separate work.** Measured in
`migration_phase0/MI_DATASET_CLASS_SCOPE.md` and
`migration_phase0/MI_COMPOSITE_BOUNDARY_SCOPE.md`.

---

## F3, SECOND INSTANCE — INSIDE THE INSTRUMENT THAT ENFORCES F3 · CLOSED

F3 says *an instrument that cannot be measured must report NOT MEASURED, never
clean.* It does **not** say that not-measured may overrule a measurement somebody
else made. The grader that enforces F3 across this programme was doing exactly
that.

`grade_75` returned `NO_COMPUTABLE_TRUTH` the moment `independent_truth` was
null — **over the top of a `grade` field sitting in the same row of the same
file**, recorded by a human who had read the answer. Twenty-two pack rows carried
that label. Five of them were recorded wrong verdicts:

| id | buried verdict | frozen rationale |
|---|---|---|
| Q10A | WRONG / SILENT | answered from the FUNDED book; the question named the pipeline dataset |
| **Q07B** | WRONG / SILENT | both scopes dropped; a whole-book figure answered a comparison question |
| Q25A/B/C | CURRENT-STATE SUBSTITUTION | a FORWARD question answered with today's risk-limit status |

**Q07B has nothing to do with the dataset class it was found beside.** It was
buried by a label that reads as innocuous, in every pack this programme has
published.

**Fixed, and the fix is deference gated on fidelity.** Precedence is now: a
refusal is a refusal; then the COMPUTED truth where one exists, because it is the
stronger oracle and it is what caught Q19C; then the frozen human grade, **only
where today's answer is byte-identical to the one that grade was recorded
against**; then `NO_COMPUTABLE_TRUTH`, which now means nobody has measured this
rather than "I could not, so nothing else counts".

The fidelity gate is load-bearing. A frozen grade judges a PARTICULAR ANSWER, not
a question, and code has shipped since that run. Deferring to a grade recorded
against a different answer would assert a stale verdict with a human's authority
behind it — a worse failure than the one being fixed. Where the answer has moved,
the grader says so in its reason rather than silently deferring.

**Measured, both arms, zero answers moved** (a grader is not the product):

| | CORRECT | FALSE_REFUSAL | NO_COMPUTABLE_TRUTH | TRUE_REFUSAL | WRONG |
|---|---:|---:|---:|---:|---:|
| off before | 102 | 21 | 22 | 15 | 6 |
| **off after** | **118** | 21 | **1** | 15 | **11** |
| merge before | 109 | 20 | 20 | 15 | 2 |
| **merge after** | **123** | 20 | **2** | 15 | **6** |

Five recorded wrong verdicts surfaced on the deterministic arm and four on the
merge arm; sixteen answers a human had graded correct stopped being reported as
unmeasured. `NO_COMPUTABLE_TRUTH` now falls to 1 and 2 — which is what an honest
"nobody has measured this" should look like on a bank this well covered.

**The grader is now in the repository**, at `migration_phase0/pack_grader.py`. It
graded a committed, hand-reviewed artefact from an ephemeral scratch directory
where no one could read it. A review artefact whose oracle is not reviewable is
an assertion, not evidence.

---

## F6 CONFIRMED IN PRACTICE · a `cd` silently changed the pack

While re-running the pack for the grader fix, two CFO refusals changed wording
between two runs of identical code. The cause was F6 exactly: one shell
invocation began with `cd` into the scratch directory, so
`trakt_core.capability.load_registry` — which resolves
`config/system/mi_capability_registry.yaml` **relative to the process cwd** —
raised `FileNotFoundError`, `_capability_explanation` swallowed it, and two
governed refusals degraded from naming their owned metric and its methodology to
*"I couldn't map this question to a governed analytic."*

Instrumented, the failure fires on the **first** call in such a process and every
call after it. Re-run from the repository root, the rich wording returns and the
pack is byte-reproducible.

No wrong number either way — but a measurement harness that changes directory
produced a different pack, silently, and nothing said so. **F6 stays at the top of
the open list; this is the second time it has cost a measurement in this
programme.**

---

## OPEN · Two routes narrow correctly and record nothing

The precondition for gating on the completeness check, and the reason the wiring
stopped.

```
Summarise the acquired book              -> 199 loans, £54.7m   correct, narrowed
Summarise the direct book                -> 441 loans, £117.4m  correct, narrowed
portfolio summary for the acquired book  -> 199 loans           correct, narrowed
                                            …all three: scopeApplied = None
```

`portfolio_summary` and `funded_bridge` apply the lens and publish no record of
having done so. The completeness check is right that nothing positively records
the narrowing — Q19C proved an envelope indistinguishable from these carried a
figure wrong by £10.2m — but as a GATE the check would refuse three correct
answers, and eight questions are in the class.

Stage 1's own record reads *"the two scope routes now publish
`metadata.scopeApplied`"*. These are further routes with the same silence,
invisible on the 157-question calibration surface and visible on 854.

**Owner: separate work, and it is step 1 of the wiring order.** Make them publish
and the class empties.

---

## CLOSED · A narrowing that ran, in a form nothing could read

The section above ("Two routes narrow correctly and record nothing") has a third
instance, and it is the one that actually refused an answer.

An analytical plan narrowed twice through the portfolio lens, measured both
books to the penny, and refused both questions on the ground that neither
narrowing had been applied:

```
Q22B  "Did Direct or Acquired add more balance during the last month?"
Q22C  "Which of the Direct and Acquired books drove more of the
       month-on-month balance increase?"

Direct    £105.0m -> £117.4m  (+£12.4m)   441 loans
Acquired   £44.5m ->  £54.7m  (+£10.2m)   199 loans
                                          narrowedTo: []
```

The difference from the two open cases is worth stating precisely, because it
changes what the fix has to be. Those routes record **nothing**. This plan
recorded the narrowing **in a form no consumer could parse**:

```
predicate = "portfolio lens = Direct (direct_001)"
```

`narrowedTo` — the one channel the receipt reads for *"was this answer scoped to
X?"* — is built by splitting field-named predicates apart. Everything
`mi_agent.population` produces is field-named. The lens clause names a lens, so
splitting it yields the field `"portfolio lens"`, which is not a field, and the
builder skipped it with a `continue`. The narrowing was performed, described,
and invisible.

**Carried means a consumer can read it.** A record in a form only a human can
parse is a record the contract does not carry, and it fails the standing test
the same way silence does.

Fixed by giving the population contract a machine-readable twin
(`PopulationRef.narrowed_on`) for exactly the case where the predicate text
names no field. No guard changed; both refusals lifted through readers that were
already there and whose strictness is unchanged.

### The pattern this adds, and it is the fail-open guard's sibling

There were **two** producers of the lens predicate text. The second composed it
itself, under a comment asserting it matched the first:

> *"the same text ``populations.apply`` records, so a population reads
> identically whichever capability produced it."*

True when written. The fix went into `apply`; the capability that runs these two
questions used the other; both answers stayed refused **with the fix in the
tree**. The comment was the only thing holding the two in agreement.

**Portable test, alongside the fail-open one:** *when a comment says its output
matches another component's — check that it calls it.* An assertion of agreement
between two implementations is a statement about the past.
