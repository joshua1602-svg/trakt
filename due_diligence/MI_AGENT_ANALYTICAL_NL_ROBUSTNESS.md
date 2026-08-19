# MI Agent — Analytical Capability Layer: NL Robustness & Generalisation Gate

**Mode:** test / measure / report only. No production code, parser rule,
recogniser, semantic registry, capability registry, planner or test bank was
modified. Working tree clean throughout.
**Baseline:** branch `claude/mi-analytical-capability-layer-vlkjfw`, SHA
`f6360f6`, tree clean, remote in sync.
**Question:** can a CFO express the nine analytical needs in ordinary ways and
get the same governed answer?

---

## 1. Executive verdict

**No. The architecture is sound; its recognition surface is not.**

| | |
|---|---|
| Runs | **752** (376 production, 376 forced-LLM) |
| Genuine LLM parses | **648 (86%)** — `parser_used == "llm"` |
| CORRECT + CORRECT_WITH_DISCLOSED_LIMITATION | **56.4%** *(gate requires ≥90%)* |
| SAFE_REFUSAL | 19% |
| **SILENT_SEMANTIC_ERROR** | **147 (20%)** |
| **INCORRECT_SUCCESSFUL** | **40 (5%)** |
| HARD_FAILURE | **0** |
| Unsafe total | **187 / 752 (25%)** |

The gate fails, and it fails on the criterion that matters most: **a quarter of
ordinary phrasings return a confident answer to a different question.**

Three findings decide how to read that number.

**Every figure the analytical layer produced was right.** 2,686 numeric findings
reconciled independently against the fixture CSVs with pandas — **zero
mismatches**. When the layer runs, it is correct. The deterministic engines are
not implicated in a single failure.

**Every unsafe outcome originates UPSTREAM of the analytical planner — 187 of
187, 100%.** Zero failures were caused by capability planning, capability
contracts, deterministic execution or narrative composition. The layer is not
the defect; it is bypassed before it can act.

**The failures are deterministic, not flaky.** 172 of 176 variation×book×arm
groups produced an *identical* grade across every repeat (98%), and all 44
variations produced the *same verdict on both books* (100%). A phrasing either
always works or always fails. This is vocabulary coverage, not model variance —
which is good news, because deterministic faults are fixable and measurable.

## 2. Test method

44 variations × 2 books × 3–5 repeats × 2 arms = **752 runs**, all through the
production `POST /mi/query` entrypoint.

| Arm | Setting | Why |
|---|---|---|
| **production** | exactly as deployed | the commercial gate |
| **forced_llm** | zero-cost-first disabled *in this process only* | production skips the LLM when the deterministic parser is confident; this arm forces a model parse so repeat-variance has something to vary. No file modified. |

Only `parser_used == "llm"` counts as a genuine LLM parse;
`deterministic_fallback_after_llm_failure` is recorded separately, per §4.

**Grading is contract-based.** The expected route owner, analytical intent and
required capabilities for all nine intents were declared *before* any result was
read (`nl_score.py::EXPECTED`). Q2/Q4/Q5/Q6 are owned by existing routes under
the documented deference rule, so for those the **owning route is the correct
outcome** and the analytical layer claiming one would itself be the defect.

**Two grading rules were added after inspecting cases. Both are declared here
because they change the numbers.**

1. *Consistency.* A route-owned intent diverted elsewhere with `guard=ok` is
   graded SILENT_SEMANTIC_ERROR, the same as an analytical intent. This moved
   items into the **more** severe class, not the more forgiving one.
2. *"Materially answers".* Condemning every route divergence was wrong. Q7.3 was
   answered by the generic executor grouping by `seasoning_segment` with six
   governed measures across both sides — that **is** the comparison asked for,
   reached by another mechanism. A structural test now reads the executed spec
   (population, measures, period structure) and grades such a run
   CORRECT_WITH_DISCLOSED_LIMITATION. This is the only rule that moved anything
   in the product's favour, and it moved 20 runs.

## 3. The 44-question bank

The brief's 36, with one operator-directed change: Q8 names single vs joint
borrowers, and **neither book carries a borrower-structure field** (the only
borrower columns are identifiers, age and an impairment flag). Inventing one was
forbidden. Per the operator's ruling Q8 ran against **all three governed X/Y
pairs**, using the **same four sentence shapes** for each, so the paraphrase axis
is held constant and only the population resolver varies:

| Pair | Resolver under test | A vs B |
|---|---|---|
| provenance | portfolio lens → governed registry | direct vs acquired |
| seasoning | governed binary partition | front book vs back book |
| dimension_value | literal category match | Alderbridge South East/London; Kestrelmoor North West/Scotland |

44 variations per book; 88 book/question pairs.

## 4. Results by analytical intent

| Intent | Runs | OK | Safe refusal | **Unsafe** | Verdict |
|---|---|---|---|---|---|
| **Q1** new-origination profile | 80 | **0 (0%)** | 0 | **80 (100%)** | **systemic failure — all four paraphrases** |
| Q2 £100m milestone | 48 | 36 (75%) | 12 | 0 | robust |
| Q3 offers + completion | 80 | 37 (46%) | 43 | 0 | safe; two of four refuse |
| **Q4** completion run rate | 48 | 12 (25%) | 12 | **24 (50%)** | **systemic — two of four** |
| Q5 forecast breach | 80 | 60 (75%) | 20 | 0 | robust |
| **Q6** limits closest to breach | 48 | 24 (50%) | 1 | **23 (48%)** | **systemic — two of four** |
| **Q7** vintages vs front book | 80 | 40 (50%) | 20 | **20 (25%)** | one paraphrase unsafe |
| **Q8** relative movement | 240 | 180 (75%) | 20 | **40 (17%)** | one shape unsafe, on two of three pairs |
| Q9 forecast funded balance | 48 | 36 (75%) | 12 | 0 | robust |

**Four intents are completely safe** (Q2, Q3, Q5, Q9 — zero unsafe in 256 runs).
**Q1 is a total failure**: not one of its 80 runs produced an acceptable outcome.

## 5. Results by paraphrase

Majority verdict per variation, production arm, identical on both books:

| Variation | Verdict | What happened |
|---|---|---|
| Q1.1 *"profile of our new lending…"* | SILENT | → `period_change_analysis`, whole book |
| Q1.2 *"originating different types… vs a few months ago"* | SILENT | → generic executor: **count by origination_date, 3,722 groups, one date** |
| Q1.3 *"recent lending vs earlier in the year"* | SILENT | → same, total balance |
| Q1.4 *"risk and borrower profile of new business"* | INCORRECT | → `period_change_analysis` |
| Q4.1 *"what completion rate are we running at?"* | SILENT | → **total balance £1.96bn** |
| Q4.2 *"how many loans are we completing?"* | SILENT | → **count of all 11,035 loans** |
| Q6.2 *"where are we closest to our limits?"* | SILENT | → WA LTV / arrears **by region** |
| Q6.3 *"which of our limits are most at risk?"* | SILENT | → total balance **by account status** |
| Q7.2 *"are older loans riskier…?"* | SILENT | → generic executor |
| Q8.3/provenance *"developing differently over time?"* | INCORRECT | → `cohort_progression` |
| Q8.3/seasoning *"developing differently over time?"* | SILENT | → `evolution` |
| the other 33 | CORRECT / DISCLOSED / SAFE_REFUSAL | — |

**Q4.2 is the one to show a board.** *"How many loans are we completing at the
moment?"* returns the total number of loans on the book — a plausible figure a
CFO could act on, with `ok=True` and a green guard.

**Q8.3 is the sharpest diagnostic in the bank.** The identical sentence shape —
*"Are X and Y balances developing differently over time?"* — is CORRECT for two
regions, INCORRECT for direct/acquired, and SILENT for front/back. Same words,
three population resolvers, three different outcomes.

## 6. Repeated-LLM consistency

| | |
|---|---|
| Variation×book×arm groups with an identical grade across every repeat | **172 / 176 (98%)** |
| Groups that varied | 4 |

The four: `Q3.3` (CORRECT ×4, SAFE_REFUSAL ×1, on three arms) and `Q6.3`
(SILENT ×2, SAFE_REFUSAL ×1). Both vary between a *safe* and an *unsafe-or-ok*
outcome, never between two different wrong answers.

**This is the most important structural result in the report.** The brief's
premise was that repeated LLM runs would expose instability. They did not.
Outcomes are governed by which phrases the deterministic vocabularies contain,
so a failing phrasing fails every time. Robustness here is a **coverage**
problem, not a **variance** problem — and coverage is testable.

## 7. Two-book consistency

**44 of 44 variations produced the same verdict on both books (100%).** No
variation works on one book and fails on the other; no failure differs in kind
between them. Every difference observed was a DATA difference (Kestrelmoor has
no Schedule 8 document, so Q5/Q6 report limits unavailable) and none was a
planner or semantic difference.

## 8. Intent-selection accuracy

| | |
|---|---|
| Runs reaching the intended owner | **420 / 752 (56%)** |
| Runs reaching the analytical layer when it should own the question | see below |

Per §8 of the brief:

* **A. Did all four variants select the same intent?** No — for Q1 (0/4 reach the
  layer), Q4 (2/4), Q6 (2/4), Q7 (2/4) and Q8 (10/12).
* **B. Materially equivalent plans?** Yes, wherever the layer was reached: every
  analytical run planned the declared intent with the declared capabilities.
* **C. Did the planner ever…** select a simple MI route instead — **yes, 107
  runs fell to the generic executor**; omit a required capability — **no**; add
  an unrequested capability — **no**; fabricate a population — **no**; fabricate
  a period — **no**; answer only part of a multi-part question — **yes**, Q3's
  two refusing paraphrases.
* **D. Harmless or material?** Material. 187 unsafe runs.

## 9. Analytical-plan consistency

Where the analytical layer was reached, it behaved perfectly: **every run planned
the expected intent, called the expected capabilities, resolved the expected
populations, and produced findings that reconciled.** There is not one instance
in 752 runs of a wrong plan, a missing capability, an invented population or an
invented period.

## 10. Numerical reconciliation

| Arm | Findings reconciled | Mismatches |
|---|---|---|
| Alderbridge production | 669 / 669 | 0 |
| Kestrelmoor production | 674 / 674 | 0 |
| Alderbridge forced-LLM | 669 / 669 | 0 |
| Kestrelmoor forced-LLM | 674 / 674 | 0 |
| **Total** | **2,686 / 2,686** | **0** |

Truth computed independently with pandas from the fixture CSVs. Populations
verified by row count as well as by value. **Zero unexplained variance.**

## 11. Safe refusals and partials

140 runs (19%) refused safely, every one naming a reason. Notable:

* **Q3.1 / Q3.4** refuse the offer-pipeline question on both books — an honest
  refusal of a question the product *can* answer under other phrasings (Q3.2,
  Q3.3 succeed). Lost capability, not a wrong answer.
* **Q4.4** refuses with *"'Reporting Date' is not available in this dataset"* —
  correct and specific.
* **Q5.4, Q7.1, Q8.3/dimension_value, Q9.1** refuse cleanly.

No refusal was silent; no refusal lacked a reason.

## 12. Failure classification and clustering

**By cause, across 187 unsafe runs** (a run may carry several):

| Cause | Runs | Share | Intents affected |
|---|---|---|---|
| **route contention** | 187 | **100%** | Q1, Q4, Q6, Q7, Q8 |
| **guard coverage** | 147 | **79%** | Q1, Q4, Q6, Q7, Q8 |
| comparison-period recognition | 100 | 53% | Q1, Q8 |
| parser / business semantics | 87 | 47% | Q1, Q4, Q6 |
| population recognition | 80 | 43% | Q1 |
| capability planning | **0** | 0% | — |
| deterministic execution | **0** | 0% | — |
| narrative / presentation | **0** | 0% | — |

**Clustered by shared root cause, there are only two defects, not eleven.**

**CLUSTER 1 — "the generic executor answers anything" (107 runs, 57%).**
No route claims the question, so it falls through to the point-in-time executor,
which has no concept of a run rate, a limit or a forecast and answers with
whatever measure and dimension the parse produced. The guard passes because it
raised **zero facets** — outside its phrase vocabulary nothing can be "lost", so
nothing is refused. Affects Q1, Q4, Q6, Q7.

**CLUSTER 2 — "a neighbouring route claims it and answers narrowly" (80 runs,
43%).** `period_change_analysis` takes Q1 (40 runs); `cohort_progression` and
`evolution` take Q8.3 (40 runs). Each answers its own narrower question
correctly. Affects Q1, Q8.

Where the guard stood: `ok` on **147 of 187** unsafe runs (79%); no verdict on
the remaining 40.

## 13. Commercial-readiness implication

**The nine-intent analytical architecture is sound. Its front door is not.**

Stated precisely, because the distinction decides what to fix:

* the **capability contract, planner, orchestrator, findings and narrative** did
  not cause a single failure in 752 runs;
* every figure they produced was right, 2,686 times over;
* they generalise across two materially different books with 100% verdict
  agreement;
* but they are reached for only **56%** of ordinary phrasings, and when they are
  not, something else answers **confidently and wrongly** in a quarter of cases.

A CFO using this today would get an excellent answer to five of nine questions
however they phrased them, and a plausible wrong answer to Q1 every single time.

**This is not a "make the LLM better" problem.** 86% of runs were genuine LLM
parses and outcomes were 98% repeatable. The model is not the variable; the
governed vocabularies and the fall-through behaviour are.

## 14. Recommended targeted fixes

Not implemented — measurement only, per §12 of the brief.

**Two generic fixes remove the large majority of unsafe outcomes.**

**FIX A — never let the generic executor answer an analytically-marked
question.** When a question carries analytical intent markers (comparison,
forecast, run-rate, limit, profile language) and *no* specialist route claimed
it, refuse instead of falling through. **Converts 107 of 187 unsafe runs (57%)
into safe refusals**, and is purely additive: it cannot change any answer a route
already produces. This is the single highest-value change available.

**FIX B — widen facet detection to match the analytical vocabularies.**
`_detect_comparison_period` misses *"compared with a few months ago"* and
*"earlier in the year"* while catching *"changed in the last few months"*. Raising
the facet makes the existing P1L/P0 machinery refuse on its own. Addresses the
**guard coverage** cause present in 147 of 187 unsafe runs (79%), and overlaps
heavily with Fix A — together they cover essentially all of Cluster 1.

**FIX C — decide what "new lending" means, once.** `mi_agent/seasoning.py`
deliberately excludes *"new lending"*, *"originating"* and *"recent lending"* from
the governed segment vocabulary on the grounds that they read as flow measures.
That decision is defensible, and it is why **Q1 fails 100% of the time**. This is
a **business-methodology decision, not a code fix**, and it needs an owner's
ruling before anyone touches the vocabulary. Q1 alone is 80 of 187 unsafe runs
(43%).

**FIX D — route precedence for Q8.3.** *"Are X and Y developing differently over
time?"* is claimed by `cohort_progression` / `evolution` before the analytical
layer. 40 runs. Lowest value of the four and the most likely to disturb working
behaviour; it should wait until A–C are measured.

Sequencing recommendation: **A, then B, then re-measure.** Together they should
convert ~57–79% of unsafe outcomes to safe refusals without changing a single
correct answer. C requires a decision, not an implementation. D last.

---

**Stop conditions.** No deterministic-calculation corruption was found (2,686/2,686
reconciled). No regression was introduced by the analytical capability layer —
zero failures originate inside it. Measurement therefore ran to completion, as
directed.

ANALYTICAL NL ROBUSTNESS: FAIL
