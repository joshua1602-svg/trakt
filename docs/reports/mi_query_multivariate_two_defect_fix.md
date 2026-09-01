# MI Query Agent — multivariate pipeline, two targeted defect fixes

| | |
|---|---|
| starting main SHA | `95dbbda` — merged `main` (Stage Movement merged in #387) |
| branch | `claude/mi-query-multivariate-two-defect-fix` |
| production files changed | **4** · **+84 / −7 executable lines** |
| new routes / recognisers / engines / MI capabilities | **0** |
| arms measured | governed engine, and the concept-merge language layer (`claude-opus-5`) |

---

## 0. One thing to check before reading further

The audit that defined these two defects **was not merged**. Its artefacts — the
`pipeline_multivariate` fixture, the 43-question bank and the two analysis
scripts — are the instrument that measures both defects, so this branch starts
at merged `main` and carries that audit commit forward as its **first commit**
(`16cf56a`), unchanged.

That commit contains **zero production changes**; `git diff 95dbbda 16cf56a` over
`mi_agent`, `mi_agent_api`, `mi_workflows`, `question_interpretation`,
`frontend` and `engine` is empty. Every production change in this report is in
the second commit. If the audit is merged separately, this branch rebases onto
it with no conflict; if it is not, merging this branch brings it along.

---

## 1. Executive verdict

**Both defects are repaired. Nothing else moved.**

| gate | required | measured |
|---|---|---|
| authoritative 166 bank | zero previously correct lost | **0 questions moved** |
| Stage Movement (36) | remain 36/36 | **36/36 · 0 moved** |
| Stage Movement near neighbours (13) | remain 13/13 | **13/13 · 0 moved** |
| multivariate (43) | every change attributable to A or B | **5 changes, 5 attributable, 0 unattributed** |
| silent wrong answers | none new | **3 → 2** |
| stage + temporal | unchanged | **byte-identical, all four** |
| language layer agrees | same delta | **24 → 29, same five questions** |

Multivariate went 24 → 29 correct (55.8% → 67.4%), and the entire delta is the
five questions the two defects owned. No previously correct answer was lost.

Measured on **both** arms independently, and they agree: the governed engine
(language layer off) and the concept-merge language layer (`claude-opus-5`) each
move the same five questions and no others. Silent wrong answers fall 3 → 2 on
the engine and 2 → 1 with the language layer. Full comparison in §6.

---

## 2. Defect A — the share denominator

### Root cause

`share` divides one population by another and had only ever been given one. The
whole-book denominator was **deliberate and documented**, in two places:

* `mi_query_spec.py`: *"P1A — a filtered population expressed as a share of the
  whole book. Distinct from the aggregations above because it needs TWO
  populations."*
* `mi_query_executor._execute_share`: *"the filtered population over the
  WHOLE-BOOK population"*, taking the unfiltered frame `df` as the denominator
  and the filtered `work` as the numerator.

So the numerator was never wrong. `"What share of Offer pipeline is joint
borrowers?"` bound **both** filters and counted 6 joint Offer cases correctly —
then divided by all 40 pipeline cases instead of the 10 at Offer.

The estate had **no representation anywhere** of which population a share is
*of*: one filter channel (`spec.filters`), and an execution receipt that labels
both the stage and the borrower type identically as `row_population`.

### Existing owner modified

`mi_agent/mi_query_executor._execute_share` — the single governed share
implementation. No second share implementation was created.

### The fix, and why it reads the selection side

The reader claims the **numerator** side of "what share of X is Y" and treats
everything else in `spec.filters` as the population the share is *of*.

That direction is the safe one, and the reason is concrete: the stage filter is
**not** contributed by this parser module at all — the governed stage reader
injects it later. A reader that tried to resolve filters *inside* the "of X"
phrase found nothing (measured: `_parse_filters("Offer pipeline")` → `{}`) and
the fix did not work. Reading the selection side instead means any narrowing
contributed by any other owner stays with the denominator by default, and a
selection this reader cannot resolve leaves the denominator exactly where it has
always been.

| file | change |
|---|---|
| `mi_agent/llm_query_parser.py` | `_SHARE_SELECTION_RE` + `_share_selection_fields`, wired at the three existing share construction sites |
| `mi_agent/mi_query_spec.py` | one field, `share_selection_fields: List[str]` |
| `mi_agent/mi_query_executor.py` | `_execute_share` narrows the denominator by `filters − selection`, using `_apply_filters` — the same mask builder the numerator uses, so the two populations cannot drift |

### Before → after

```
"What share of Offer pipeline is joint borrowers?"

before   Current Outstanding Balance Share Pct: 19.2% · 6 loans · Population Total: 40
after    Current Outstanding Balance Share Pct: 59.7% · 6 loans · Population Total: 10

governed 2,960,000 / 4,960,000 = 59.68%
```

### Proof the denominator is the Offer population

`Population Total: 10`. The fixture holds exactly ten Offer cases; the whole
pipeline holds forty. The receipt states the population it divided by, so the
proof is on the answer itself rather than in this document. Pinned by
`test_the_denominator_population_is_stated_and_is_the_stage`.

### Proof whole-book share semantics did not change

Measured on the funded book, all still dividing by 640:

| question | after |
|---|---|
| What share of the book is drawdown? | 49.3% · **Population Total: 640** |
| What proportion of the book is below 75% LTV? | 99.9% · **Population Total: 640** |
| What proportion of the book is in Scotland? | 21.5% · **Population Total: 640** |

And structurally: on every one of those, `_share_selection_fields` returns `[]`,
so no contextual narrowing exists and the denominator is the whole frame — the
pre-existing code path, unchanged. Whole-book shares are preserved **by
construction, not by exception**.

`test_the_executor_divides_by_the_whole_frame_when_nothing_is_claimed` exercises
both settings on one frame: nothing claimed → denominator 6 rows; the stage
claimed as context → denominator 2 rows.

---

## 3. Defect B — region filter binding on pipeline

### Root cause

`pipeline_prep._apply_group_aliases` **copies** `collateral_geography` into
`geographic_region_obligor` and records the alias
(`"geographic_region_obligor<-collateral_geography"`). Both names are registered
dimensions in the semantics registry, so `execution_receipt.book_values`
catalogued **the same values twice**:

```
pipeline catalogue (before)
  collateral_geography        london, midlands, north west, scotland, south east, wales
  geographic_region_obligor   london, midlands, north west, scotland, south east, wales
```

`categorical_spans.value_field` then did exactly what it is written to do —
*"A value two governed fields both claim returns None: an ambiguous narrowing
must be disclosed, never resolved by preference"* — and returned `None`. No
filter bound, and the downstream refusal told the reader their book held no
London loans.

The rule was right; the premise was false. The two names select **identical
rows**, so there was no ambiguity to disclose.

Two facts rule out simpler explanations, both measured: the catalogue **did**
contain London, and the failure reproduced for **Scotland and Wales**, which the
funded book also carries.

### Existing owner modified

`mi_agent/execution_receipt.book_values` — the catalogue owner, and the only
place that holds both the registry and the frame.

### The fix

A dimension whose column is **element-wise identical** to one already catalogued
is published once. The test is the data, not a list of names:

* two names for one column → one catalogue entry, the value binds;
* two fields sharing a vocabulary but differing on any row → both entries stay,
  and the ambiguity rule still fires.

### Before → after

```
"How much Application-stage pipeline is in London?"

before   ok=false  "No loans in this book match that filter ('london'), so there
                    is nothing to calculate."
         filters = {"pipeline_stage": "APPLICATION"}

after    Balance: £1.9MM · 4 loans.
         filters = {"collateral_geography": "London",
                    "pipeline_stage": "APPLICATION"}

governed 1,910,000 across 4 cases
```

All four formulations of MV03 now answer correctly.

### Proof the mechanism is governed, not hard-coded

No region name appears anywhere in the change. The rule is a column comparison.
Evidence:

| probe | result |
|---|---|
| another governed region — Offer / Scotland | **£450K · 1 loan** (governed: 450,000 · 1) |
| funded region filter | unchanged, still binds `geographic_region_obligor`, £37.6MM · 141 loans |
| funded catalogue | unchanged — funded carries only one region column, so nothing collapses |

`test_the_catalogue_collapses_a_duplicated_dimension_not_a_named_one` builds a
three-column frame where two columns are identical and a third shares the same
words while differing per row: the identical pair collapses, the third survives,
and `value_field` still returns `None` for the genuinely ambiguous case.

### Proof unknown and zero-match regions stay safe

| question | after |
|---|---|
| "…pipeline is in Atlantis?" | **refuses**, names `'atlantis'`, no whole-book figure substituted |
| "How much Completed-stage pipeline is in Wales?" | no borrowed figure — Wales holds pipeline cases but none at COMPLETED |

Both pinned by tests.

---

## 4. Regression gates

### Existing 166 bank

| | before | after |
|---|---|---|
| CFO91 CORRECT | 63 | 63 |
| CFO91 TRUE_REFUSAL | 16 | 16 |
| CFO91 FALSE_REFUSAL | 11 | 11 |
| CFO91 NO_COMPUTABLE_TRUTH | 1 | 1 |
| CFO91 **WRONG** | **0** | **0** |
| BANK75 DELIVERED | 44 | 44 |
| BANK75 DECLINED | 31 | 31 |

**Questions moved: 0.** Not merely the same totals — every answer, route and
verdict byte-identical.

### Stage Movement and near neighbours

```
bank_stage        36/36 correct   0 question(s) moved
bank_neighbours   13/13 own route 0 question(s) moved
```

### Multivariate 43

| | before | after |
|---|---:|---:|
| CORRECT | 24 | **29** |
| WRONG (silent) | 3 | **2** |
| HONEST DECLINE | 12 | 8 |
| SAFE REFUSAL | 4 | 4 |
| correct rate | 55.8% | **67.4%** |
| safe rate | 93.0% | **95.3%** |

**Every change attributed, none unattributed:**

| question | before → after | defect |
|---|---|---|
| MV02B What share of Offer pipeline is joint borrowers? | WRONG → CORRECT | **A** |
| MV03A How much Application-stage pipeline is in London? | DECLINE → CORRECT | **B** |
| MV03B What is the Application pipeline balance for London? | DECLINE → CORRECT | **B** |
| MV03C How much pipeline at Application is in London? | DECLINE → CORRECT | **B** |
| MV03D Show Application-stage exposure in London. | DECLINE → CORRECT | **B** |

Previously correct multivariate questions lost: **none**.

The two remaining silent wrongs are the deferred gaps, untouched: MV02C
(`"…how much is joint borrower exposure as a percentage?"` — a measure-recognition
gap, answers an absolute) and MV06C (`"LTV distribution"` read as a scalar).

### Stage + temporal — deliberately untouched

All four MV09 formulations are **byte-identical** before and after, and still
refuse without substituting a prior-week figure for a prior month. Pinned by
`test_stage_plus_previous_month_still_refuses_without_substituting`.

**Temporal pipeline semantics modified: NO.**

### Targeted tests

`mi_agent_api/tests/test_multivariate_two_defect_fix.py` — **22 passed**, 12
subtests. Roughly half assert what must *not* change: whole-book share
denominators, the funded region filter, unknown and zero-match regions, the
deferred temporal and recognition gaps, and the seven working constructions the
audit measured — including the hardest, `WA LTV for joint borrowers in
Application` at 53.7% with both narrowings still bound and the weighting basis
intact.

### Broad serial regression

Run serially at both SHAs — not under `xdist`, whose module-level state
instability was demonstrated in the previous sprint.

```
HEAD      83 failed · 5994 passed · 709 skipped · 15 xfailed   (9m20s)
95dbbda   83 failed · 5994 passed · 709 skipped · 15 xfailed   (9m15s)

failures identical in both        82
failures ONLY on HEAD              1   environmental — see below
failures ONLY on the baseline      1   worktree artefact — see below
```

**Zero new failures attributable to this change.** Both single-sided failures
were verified rather than assumed:

* `tests/test_serving_parquet.py::test_the_serving_copy_is_materially_faster_and_smaller`
  — a performance assertion with a 2x floor. Re-run three times on HEAD: **2.4x
  pass, 3.0x pass, 1.6x fail**, the failure landing while the language-layer arm
  was loading the box. Its own docstring says it measures the ORDER of the
  difference on shared hardware. Nothing in this change touches serving or
  parquet.
* `mi_agent/tests/test_registry_governance.py::test_checked_in_registry_matches_generator`
  — **passes at HEAD** (5 passed). The baseline ran in a git worktree and the
  checked-in registry records an absolute path, so the comparison was
  `/home/user/trakt/config/...` against `/tmp/wt_base2/config/...`. A worktree
  artefact, and explicitly **not** claimed as a fix.

No pre-existing failure was fixed or touched.

---

## 5. Scope

| | |
|---|---|
| production files changed | 4 |
| executable lines | **+84 / −7** |
| new Query routes | **0** |
| new recognisers | **0** |
| new interpreters / parser frameworks | **0** |
| new analytical implementations | **0** |
| new MI capabilities | **0** |
| new datasets / API routes | **0** |

| file | executable | what |
|---|---:|---|
| `mi_agent/execution_receipt.py` | +27 / −2 | catalogue publishes a duplicated dimension once |
| `mi_agent/llm_query_parser.py` | +35 / −4 | the share's selection-side reader, wired at three existing sites |
| `mi_agent/mi_query_executor.py` | +21 / −1 | `_execute_share` divides by the named population |
| `mi_agent/mi_query_spec.py` | +1 / −0 | one field |

Two repairs to existing composition seams. The LLM's responsibility is
unchanged: language interpretation only, with deterministic governed code owning
every calculation.

---

## 6. Recommendation

**MERGE.**

Both defects are repaired, on both arms, and nothing else moved. The two arms
were measured separately because they fail differently: the deterministic
engine arm is the reproducible instrument, and the concept-merge language layer
is what actually serves.

### Both arms, side by side

| multivariate 43 | engine before | engine after | language before | language after |
|---|---:|---:|---:|---:|
| CORRECT | 24 | **29** | 24 | **29** |
| WRONG (silent) | 3 | **2** | 2 | **1** |
| HONEST DECLINE | 12 | 8 | 13 | 9 |
| SAFE REFUSAL | 4 | 4 | 4 | 4 |
| correct rate | 55.8% | **67.4%** | 55.8% | **67.4%** |
| safe rate | 93.0% | **95.3%** | 95.3% | **97.7%** |

The two arms agree on the delta, checked **per question** rather than inferred
from totals — the language-arm before/after verdicts differ on exactly five
questions, and they are the same five:

```
MV02B  WRONG          -> CORRECT      defect A
MV03A  HONEST_DECLINE -> CORRECT      defect B
MV03B  HONEST_DECLINE -> CORRECT      defect B
MV03C  HONEST_DECLINE -> CORRECT      defect B
MV03D  HONEST_DECLINE -> CORRECT      defect B
```

Same five questions, same +11.6 points of correctness. The arms differ only in
MV06C (`"LTV distribution"`), which the language layer already answered before
this change and still answers after — exactly as the audit recorded. **This change repairs what the language
layer could not.**

On the regression banks the language arm at HEAD reads:

```
CFO91   CORRECT 63 · TRUE_REFUSAL 16 · FALSE_REFUSAL 11 · NO_COMPUTABLE_TRUTH 1 · WRONG 0
BANK75  DELIVERED 49 · DECLINED 26
stage   36/36        near neighbours  13/13 own route
```

Stated precisely, because it matters: the **matched** before/after comparison on
the 166 bank is the engine arm, where every answer, route and verdict is
byte-identical and **0 questions moved**. The language arm was run at HEAD only
in this sprint. Its figures sit inside the variance already observed between two
runs of *identical* code in the previous sprint — CFO91 63 and 62, BANK75 49
both times — so 63/49 is the shipping record reproduced, not evidence of a
change. It is a corroboration, not a second controlled experiment, and should
not be read as one.

### The decision rule, condition by condition

| # | condition | measured | |
|---:|---|---|:--:|
| 1 | Defect A repaired | share of Offer that is joint = **59.7%**, denominator 10 Offer cases | ✅ |
| 2 | Defect B repaired | Application + London = **£1.9MM · 4 loans**; all four MV03 formulations correct | ✅ |
| 3 | No previously correct answer lost, any bank | 166: 0 moved · stage: 0 moved · neighbours: 0 moved · multivariate: 0 lost | ✅ |
| 4 | Stage Movement stays 36/36 | **36/36**, both arms | ✅ |
| 5 | Near neighbours stay 13/13 | **13/13** kept their own owner, both arms | ✅ |
| 6 | Every multivariate change attributable to A or B | **5 changes, 5 attributed** (MV02B→A; MV03A–D→B), 0 unattributed | ✅ |
| 7 | No new silent wrong answers | engine **3 → 2**, language **2 → 1**; no question became wrong | ✅ |
| 8 | Temporal pipeline behaviour unchanged | all four MV09 formulations **byte-identical**, still a safe refusal | ✅ |
| 9 | No new route, recogniser, interpreter, engine, capability, dataset or API route | **0 of each**; 4 existing files, +84/−7 executable lines | ✅ |
| 10 | Broad regression shows no new failures | serial run at both SHAs: **82 identical failures**, 1 each side, both verified as artefacts | ✅ |

All ten hold.

### What a reviewer must decide, and it is not the code

One thing in this branch is not a code question. Its first commit (`16cf56a`) is
the **unmerged audit** — the fixture, the 43-question bank and the two analysis
scripts. It carries zero production change, and it is also the only instrument
that can measure either defect. A reviewer should decide deliberately whether to
take it here or merge it separately; what it must not be is merged by accident.
Section 0 has the diff proof.

### What this change does not claim

* MV02C (`"…how much is joint borrower exposure as a percentage?"`) is still a
  silent wrong on the engine arm — a measure-recognition gap that answers an
  absolute where a percentage was asked. Deferred deliberately: fixing it means
  touching measure recognition, which is outside these two defects.
* MV06C (`"LTV distribution"` read as a scalar) is unchanged on both arms.
* The other audit failures — `"currently"`, `"over 500k"`, bare `"amount"`,
  `"status"`, and the WA-LTV executor issue — were not touched, by instruction.
* The two single-sided regression failures are argued as artefacts with evidence
  (a re-run distribution and a worktree path), not asserted as passes.

