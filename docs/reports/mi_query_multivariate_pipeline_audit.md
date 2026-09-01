# MI Query Agent — multivariate pipeline audit

| | |
|---|---|
| starting SHA | `95dbbda` — merged `main`, with the Pipeline Stage Movement work merged in #387 |
| branch | `claude/mi-query-stage-movement-cqko43` (restarted from merged main) |
| kind | **AUDIT ONLY** — production Query files changed: **0** |
| fixture | `tests/fixtures/pipeline_multivariate` (new, analysis-only) |
| arms | governed engine alone, and the concept-merge language layer (`claude-opus-5`) |

---

## 1. Executive verdict

> **How well does the current MI Query Agent compose multiple governed pipeline
> concepts?**

**Better than expected on filters, and it does not fabricate. It fails on two
specific compositions, and one of those fails silently.** Measured on both arms:
the language layer changes one verdict out of forty-three and fixes neither
headline gap.

Of eleven core constructions, **seven work**:

| construction | works | evidence |
|---|---|---|
| stage + borrower type + balance | ✅ | 4/4 |
| stage + borrower type + **weighted** LTV | ✅ | 4/4 |
| stage + LTV threshold + balance | ✅ | 3/4 |
| stage + LTV band grouping | ✅ | 3/4 |
| stage + weighted LTV | ✅ | 3/4 |
| stage + region **grouping** | ✅ | 3/4 |
| stage × borrower type (two-dimensional) | ✅ | 3/4 |
| stage + **region filter** | ❌ | 0/4 |
| stage + **share of stage total** | ❌ | 0/4 |
| stage + **historical comparison** | ❌ | 0/4 |
| stage + ticket threshold | partial | 1/3 |

The single most important finding is that **three of the four failures do not
produce a wrong number**. The estate's fail-closed machinery catches them and
refuses. The exception is the share question, which is the one silent error in
the bank and the highest-priority finding.

**No new MI capability is required.** Every figure the failing questions ask for
is computable from the governed prepared frame — proven independently in §4.
Every failure is a Query-side **binding** or **composition** gap.

---

## 2. Baseline continuity

The audit must not itself change anything. All three existing banks were run at
`95dbbda` before the audit and again after, with the same runner:

| bank | before | after | moved |
|---|---|---|---:|
| authoritative 166 · CFO91 | 63 correct · 16 correctly declined · 11 false refusal · 1 no-truth · **0 wrong** | identical | **0** |
| authoritative 166 · BANK75 | 44 delivered · 31 declined | identical | **0** |
| Stage Movement (36) | 36 correct · 0 wrong | identical | **0** |
| Stage Movement near neighbours (13) | 13/13 own route | identical | **0** |

```
   bank_166         0 question(s) moved
   bank_stage       0 question(s) moved
   bank_neighbours  0 question(s) moved
```

These are the same figures recorded before #387 merged, so the merge itself
moved nothing either.

---

## 3. Question bank

`tests/fixtures/mi_query_stage_movement/MULTIVARIATE_PIPELINE_BANK.yaml` — the
**same case/formulation schema and the same `must` / `must_not` grading shape**
the Stage Movement bank already uses. No new oracle framework, question schema
or scoring methodology was introduced.

**11 core business questions (10 required + 1 optional) × 3–4 formulations = 43.**

| id | core business question | construction |
|---|---|---|
| MV01 | How much of the pipeline in Offer stage is joint borrowers? | stage + categorical filter + balance |
| MV02 | What percentage of Offer-stage pipeline is joint borrowers? | stage + categorical filter + share |
| MV03 | How much Application-stage pipeline is in London? | stage + region filter + balance |
| MV04 | Show Offer-stage pipeline by region. | stage + 1-D grouping |
| MV05 | How much Offer-stage pipeline has LTV above 60%? | stage + numeric threshold |
| MV06 | Show Application-stage pipeline by LTV band. | stage + governed bucket grouping |
| MV07 | What is WA LTV for Offer-stage pipeline? | stage + weighted measure |
| MV08 | What is WA LTV for joint borrowers in Application? | stage + filter + weighted measure |
| MV09 | Application-stage pipeline vs the previous month | stage stock + historical comparison |
| MV10 | Show pipeline by stage and borrower type. | two-dimensional grouping |
| MV11 | *(optional)* Offer-stage pipeline on cases over 500k | stage + measure threshold |

None asks about stage-to-stage movement. The bank keeps **current position**,
**historical comparison** and **stage-to-stage movement** distinct; only the
first two appear.

---

## 4. Fixture proof

### Why a new fixture was necessary

Neither committed pipeline pack can discriminate these questions:

| pack | borrower type | LTV |
|---|---|---|
| `pipeline_transition_2w` | **joint 10 of 10** | **constant 0.5** |
| `pipeline_history_5w` | **joint 8 of 8** | **constant 0.5** |

On either pack, "what share of Offer pipeline is joint borrowers?" is 100% and
"LTV above 60%" is empty. A bank measured there would score an agent that
**silently dropped the borrower-type filter as correct** — the exact failure the
audit exists to detect. Neither pack was edited: both are pinned inputs for
governed outputs already asserted elsewhere.

`tests/fixtures/pipeline_multivariate` is written in the canonical M2L weekly
extract schema, discovered by `pipeline_contract._PIPELINE_SOURCE_GLOBS` and
prepared by the ordinary `prepare_pipeline_mi_dataset`. **No production branch,
alias or validation change exists for it.** Forty cases at 2026-06-26; four
extracts across a month boundary (29 May, 5 / 12 / 26 June).

### Every expected answer, computed from the governed prepared frame

`scripts/prove_multivariate_pipeline_fixture.py` reads the output of production
`load_prepared_pipeline` and computes each expectation directly. It also fails
if any filter would be indistinguishable from its absence.

| question | required governed fields | expected | the figure a dropped concept gives |
|---|---|---|---|
| MV01 | `pipeline_stage`, `borrower_type`, balance | **2,960,000 · 6 cases** | 4,960,000 · 10 cases |
| MV02 | + share of stage subtotal | **59.68%** | 19.2% (share of all 40) |
| MV03 | `geographic_region_obligor` | **1,910,000 · 4 cases** | 4,345,000 · 12 cases |
| MV04 | region grouping | 6 regions, London 1,930,000 | — |
| MV05 | `current_loan_to_value` > 0.60 | **2,790,000 · 4 cases** | 4,960,000 · 10 cases |
| MV06 | `ltv_bucket` grouping | ≤50% 1,590,000 · 50-60% 620,000 · 60-70% 2,135,000 | — |
| MV07 | balance-weighted LTV | **58.383%** | unweighted mean 53.1% |
| MV08 | + borrower filter | **53.671%** | 55.915% (filter dropped) / 49.714% (unweighted) |
| MV09 | prior-month extract | 4,345,000 vs 3,075,000 (+1,270,000) | prior *week* 2026-06-12 |
| MV10 | stage × borrower cross-tab | 10 cells | — |
| MV11 | balance > 500,000 | **2,790,000 · 4 cases** | 4,960,000 |

```
DISCRIMINATING: every filter changes the answer it would be confused with.
```

**`erm_product_type` is absent from the prepared pipeline frame**, so the
optional product question was dropped rather than scored as a Query failure.
This is a fixture/preparation fact, not an agent fact.

**One governed detail that matters:** pipeline LTV is a **ratio** (0.31–0.78),
normalised by `_to_ratio`. A correct answer to "LTV above 60%" has to compare
against 0.60, not 60. The agent does — see §5.

---

## 5. Current results

Governed engine arm, 43 questions:

| verdict | count |
|---|---:|
| CORRECT | **24** |
| WRONG (silent) | **3** |
| HONEST DECLINE | 12 |
| SAFE REFUSAL on an answerable question | 4 |

| by case | correct | outcome |
|---|---|---|
| MV01 stage + borrower type | **4/4** | both filters bound and applied |
| MV02 share | 0/4 | 2 wrong, 2 declined — **denominator** |
| MV03 region filter | 0/4 | 4 refusals, each a **false claim about the book** |
| MV04 region grouping | 3/4 | 1 recognition miss on "currently" |
| MV05 LTV threshold | 3/4 | 1 measure-binding slip |
| MV06 LTV grouping | 3/4 | 1 wrong — "distribution" read as a scalar |
| MV07 WA LTV | 3/4 | 1 executor message leak |
| MV08 stage + filter + WA LTV | **4/4** | the hardest composition, clean |
| MV09 stage + prior month | 0/4 | 4 safe refusals — **composition** |
| MV10 stage × borrower type | 3/4 | 1 recognition miss on "status" |
| MV11 ticket threshold | 1/3 | 2 recognition misses |

---

## 6. Success rate

```
CORRECT RATE   24 / 43 = 55.8%
SAFE RATE      40 / 43 = 93.0%      correct + declines + safe refusals
SILENT WRONG    3 / 43 =  7.0%
```

### Both arms

| | governed engine | language layer (`claude-opus-5`) |
|---|---:|---:|
| CORRECT | 24 | 24 |
| WRONG (silent) | **3** | **2** |
| HONEST DECLINE | 12 | 13 |
| SAFE REFUSAL | 4 | 4 |
| correct rate | 55.8% | 55.8% |
| safe rate | 93.0% | **95.3%** |

Exactly **one** verdict moves between the arms: MV06C
(*"What is the LTV distribution…"*) goes from a silent wrong answer to an honest
decline. The language layer removes one silent error and adds none.

**It fixes neither headline gap.** The region filter (MV03) and the share
denominator (MV02) fail identically on both arms — which is the evidence that
both are architectural rather than language-understanding problems, and the
reason §15 recommends code work rather than vocabulary work for them.

For comparison, the wider MI Query Agent's historical shipping figure is ≈82%
correct on the 166 bank. **The multivariate bank is materially harder and scores
materially lower**, which is the expected shape: every question here demands at
least two governed concepts to bind and compose.

The safe rate is the more important number. The agent's fail-closed machinery is
doing its job on 16 of the 19 non-correct questions: it refuses rather than
publishing a plausible figure.

**One qualification on "safe".** Four of the twelve declines (all of MV03) are
*safe* but not *honest*: see §7.

---

## 7. Silent-error review

### The one silent wrong number — MV02, share denominator

```
Q   "What share of Offer pipeline is joint borrowers?"
A   Current Outstanding Balance Share Pct: 19.2% · 6 loans · Population Total: 40
    Calculated: Share of Balance · Borrower Type = Joint · Pipeline Stage = OFFER
                · 6 qualifying loans of 40
Correct: 59.68%  (2,960,000 of the 4,960,000 at Offer)
```

Both filters bound. The **numerator is right** (6 joint Offer cases). The
denominator is **all 40 pipeline cases**, not the 10 at Offer. The receipt's own
facet label states the governed concept plainly: *"a proportion of the book"*.
So the estate has exactly one share denominator — the dataset population — and
no governed notion of a share of a narrowed subtotal.

This is a **silent error relative to the question asked**: 19.2% is a true
statement about a different denominator, presented as the answer to a question
about Offer. It is partially disclosed ("of 40"), which a careful reader could
catch, but the headline percentage is wrong for the question.

`MV02C` is the second: asked "as a percentage", answered with an absolute
balance (£3.0MM). Measure binding, disclosed in the receipt but not in the
figure.

### The second silent wrong — MV06C, a distribution answered as a scalar

```
Q   "What is the LTV distribution of pipeline currently at Application?"
A   Weighted-average Current LTV: 55.9% · 12 loans
```

The stage bound correctly. "Distribution" did not bind to a grouping, so a
single weighted average stands in for a distribution across bands. The receipt
discloses what was computed; the reader asked for something else.

### Four false refusals that make a false claim about the client's data — MV03

```
Q   "How much Application-stage pipeline is in London?"
A   "No loans in this book match that filter ('london'), so there is nothing to
     calculate. I have not returned a whole-book figure in its place."
Truth: London holds 4 Application cases worth 1,910,000 in this very fixture.
```

No wrong figure is published, so this is not a silent numeric error — but the
sentence is a **false statement about the book**, which
`migration_phase0/data_claim_audit.py` already classifies as the most damaging
refusal shape. It reproduces for **every** region tested, including regions
present in both the funded tape and the pipeline:

| region | in funded book | in pipeline fixture | filter binds |
|---|---|---|---|
| Scotland | ✅ 141 loans | ✅ | ❌ |
| Wales | ✅ | ✅ | ❌ |
| London | ❌ | ✅ 4 Application cases | ❌ |
| North West | ❌ | ✅ | ❌ |

**Fully honest rate**, excluding these four: **36 / 43 = 83.7%**.

---

## 8. First-failure decomposition

Attributed from the captured envelope evidence — the parsed spec, the receipt
and the guard facets — not from the verdict label.

| first failure layer | count | questions |
|---|---:|---|
| **Filter binding** — region value never binds on pipeline | 4 | MV03A–D |
| **Composition** — stage filter lost on the temporal route | 4 | MV09A–D |
| **Recognition / vocabulary** — "currently", "over 500k", bare "amount" | 3 | MV04C, MV11A, MV11C |
| **Share denominator** — dataset population, not the narrowed one | 3 | MV02B, MV02C, MV02D |
| **Measure binding** — measure substituted, caught by the guard | 1 | MV05B |
| **Grouping binding** — "distribution" not read as a grouping | 1 | MV06C |
| **Executor** — "bar chart requires a dimension" leaked to the reader | 1 | MV07D |
| **Filter binding** — "status" bound to `account_status` | 1 | MV10D |
| **Dataset selection** | 0 | — |
| **Stage recognition** | 0 | — |
| **Underlying analytics missing** | **0** | — |

Two layers are conspicuously clean: **dataset selection never failed** (every
question resolved to `pipeline`), and **stage recognition never failed** — the
stage bound in every question where anything bound at all.

---

## 9. Stage + dimension composition

**Composition works.** This is the audit's main positive finding, and it is the
question §21 asks directly: can Query combine a stage with another filter
without one displacing the other?

```
"How much Offer pipeline is joint borrowers?"
  spec.filters = {"borrower_type": "Joint", "pipeline_stage": "OFFER"}
  Balance: £3.0MM · 6 loans
  Calculated: Total Balance · Borrower Type = Joint · Pipeline Stage = OFFER · 6 loans
```

Both filters appear in the spec, both in the receipt, and the figure is the
governed 2,960,000 across 6 cases. **No silent filter loss was observed on any
categorical or numeric filter that bound at all.** Where a filter could not
bind, the guard refused rather than answering over the wider population.

| second concept | binds with stage | evidence |
|---|---|---|
| borrower type | ✅ | MV01 4/4, MV08 4/4 |
| LTV threshold | ✅ | MV05 — `{"current_loan_to_value": {"op":"gt","value":60.0}}` + stage, 4 cases |
| balance threshold | ✅ | MV11B — `Balance > 500000 · Pipeline Stage = OFFER`, 4 cases |
| region (as **grouping**) | ✅ | MV04 — `grouped by Region · 6 groups` within `Pipeline Stage = OFFER` |
| region (as **filter**) | ❌ | MV03 — `filters = {"pipeline_stage": "APPLICATION"}` only |

**The LTV ratio question is answered correctly.** §8 of the brief warned against
inferring pipeline support from funded support; measured, the threshold is
applied correctly against the governed ratio and returns the right 4 cases.

**Region is the one asymmetry**, and it is narrow: the same dimension binds as a
grouping on pipeline and as a filter on funded, but not as a filter on pipeline.

---

## 10. Weighted measures

**Weighting integrity holds.** The fixture was built so an unweighted mean is
detectable, and the agent never returned one.

| question | governed WA | unweighted (wrong) | answered |
|---|---|---|---|
| MV07 Offer WA LTV | 58.383% | 53.1% | **58.4%** ✅ |
| MV08 Application joint WA LTV | 53.671% | 49.714% | **53.7%** ✅ |

MV08 is the strongest single result in the audit: stage + categorical filter +
balance-weighted measure, correct on all four formulations, with the receipt
naming both narrowings and the weighting basis. Dropping the borrower filter
would have given 55.9%; it did not appear.

The one MV07 miss (`"average LTV, weighted by balance"`) is not a weighting
failure — it is an executor message reaching the reader: *"bar chart requires a
dimension (or x)"*. No figure was substituted.

---

## 11. Historical comparison

**"Previous month" IS a governed construction. Stage + time is not.**

This corrects the hypothesis in §12 of the brief. The captured evidence:

```
MV09A  route = temporal_compare   spec.temporal_mode = compare
  facets: comparison_period "comparison period (last month)"  APPLIED
          granularity "month"                                 APPLIED
          grouping_dimension "application stage"              LOST
  → refused: "…application stage … could not be applied to the calculation…
              I have not substituted a broader figure."
```

The temporal route resolved the period and the month granularity. What it could
not carry is the **stage narrowing**, and the guard refused rather than
publishing a whole-pipeline month-on-month comparison.

* **No period substitution occurred.** The prior *week* (2026-06-12) was never
  silently used in place of the prior month.
* The outcome is **safe**, but it is a false refusal of an answerable question:
  the governed May extract exists (3,075,000 at Application against 4,345,000).
* The refusal **misdescribes the obstacle** — it says the answer "covers the
  whole population", which is about the stage, and says nothing about the
  period the reader asked for.

Classification: **composition gap, not a temporal semantics gap.**

---

## 12. Share-of-total

Covered in §7. The finding restated for §23's terms:

* the **numerator** correctly honours both the stage and the borrower filter;
* the **denominator** is the whole dataset population (40 cases), never the
  narrowed one (10 at Offer);
* the governed concept is named in the estate's own facet label as *"a
  proportion of the book"* — so this is a **definition** boundary, not a bug in
  filter binding;
* consequently there is currently **no way to ask for a share of a stage
  subtotal**, and asking produces a true statement about a different denominator.

---

## 13. Multidimensional grouping

**Two-dimensional grouping on pipeline works.**

```
"Break down pipeline balance by stage and single versus joint borrower."
  spec.dimensions = ["pipeline_stage", "borrower_type"]
  Here is the heatmap for your query, covering 10 groups.
  Calculated: Total Balance · grouped by Pipeline Stage and Borrower Type
              · 10 groups · 40 loans
```

Ten cells — exactly the governed cross-tab from §4. Three of four formulations
succeed. The fourth (`"by stage and joint/single status"`) binds *status* to
`account_status`, which the pipeline genuinely lacks, and refuses honestly.

The existing Query plan representation, executor and renderer therefore already
support two groupings on the pipeline dataset. No new interpreter, and no new
seam, is needed for §24.

---

## 14. Existing capability vs Query gap

| failed family | gap class | why |
|---|---|---|
| MV03 region filter | **C · recognition / binding** | the value exists in the prepared frame, binds as a grouping on pipeline and as a filter on funded |
| MV02 share denominator | **B · composition** | both primitives exist — the filtered subtotal and the stage subtotal are each computable today |
| MV09 stage + prior month | **B · composition** | `temporal_compare` resolves the period; the stage filter works elsewhere; they do not compose |
| MV06C "distribution" | **C · recognition** | the LTV bucket grouping works when named "band" or "bucket" |
| MV04C / MV11A / MV11C | **C · vocabulary** | "currently", "over 500k", bare "amount" |
| MV10D "status" | **C · recognition** | binds to the wrong governed field |
| MV05B / MV07D | **C · measure binding / executor message** | both caught by the guard |

**Class E — genuinely missing MI: none.** Every required figure was computed
from the governed prepared frame by the fixture-proof script, using no capability
the estate does not already have.

---

## 15. Remediation recommendations

Not implemented. Smallest safe fix per family.

| # | family | remediation | likely owner | size | regression sensitivity | before go-live? |
|---|---|---|---|---|---|---|
| R1 | MV03 region filter on pipeline | **FILTER-BINDING FIX** — let a categorical value bind against the dataset the answer will be computed on | the value catalogue handed to the parser (`mi_service._book_values` / `categorical_spans`) | small | **HIGH** — the catalogue feeds span ownership for every route | **Yes** — it publishes a false claim about client data |
| R2 | MV02 share denominator | **EXISTING PLAN COMPOSITION** — a governed share-of-narrowed-population, or refuse when the question names a subtotal denominator | share aggregation owner + `execution_receipt` share facet | small–medium | medium | **Yes** — it is the only silent wrong number |
| R3 | MV09 stage + prior month | **EXISTING EXECUTOR EXTENSION** — carry the stage row-predicate into `temporal_compare` | `chat_routing._route_compare` / `temporal_compare` | medium | medium | No — currently refuses safely |
| R4 | MV06C, MV04C, MV11A/C, MV10D | **VOCABULARY / RECOGNITION ONLY** — "distribution" as a grouping request; "currently"; "over Nk"; "status" | existing parser vocabulary | small | low–medium | No |
| R5 | MV07D | **EXISTING EXECUTOR** — an internal chart-construction message reaches the reader | executor / renderer | very small | low | No — cosmetic, but it is user-visible implementation language |
| R6 | product-type questions | **NO CHANGE** — `erm_product_type` is absent from the prepared pipeline frame; a preparation question, not a Query one | `pipeline_prep` | n/a | n/a | No |

R1's regression sensitivity is the reason this audit does not implement it: the
value catalogue is the same mechanism that gives recognisers their span
ownership, and the Stage Movement near-neighbour bank depends on it.

---

## 16. Go-live impact

| finding | blocker? | reasoning |
|---|---|---|
| MV03 false claim about the book | **Yes** | the agent tells a client their book lacks a region it holds. No figure is wrong, but the sentence is, and it is repeatable for every region |
| MV02 share denominator | **Yes** | the only silent wrong number in the bank; a reader asking for a share of a stage gets a share of the dataset |
| MV09 stage + prior month | No | refuses safely, publishes nothing wrong |
| Recognition/vocabulary misses | No | all refuse or disclose |
| MV06C distribution-as-scalar | Borderline | discloses what it computed in the receipt, but the headline answers a different question |

---

## 17. Recommendation

**TARGETED QUERY COMPOSITION WORK WARRANTED.**

Not *"material analytical gaps exist"* — there are none: every failing question
is computable from capabilities the estate already owns, and the fixture proof
demonstrates it. Not *"current model sufficient"* either: two findings publish
something untrue relative to the question asked, and both are reachable from
ordinary business language.

The work is two narrow fixes (R1, R2), each in an existing seam, each with a
bank that can now measure it. The remaining gaps refuse safely and can wait.

**What the audit also establishes, and should not be lost:** stage + filter +
weighted measure composes cleanly on the pipeline dataset, two-dimensional
grouping works, thresholds apply correctly against the governed LTV ratio, and
no filter that bound was ever silently dropped.

---

## Appendix · production files changed

```
QUERY PRODUCTION FILES CHANGED: 0
ENGINE FILES CHANGED:           0
REACT FILES CHANGED:            0
PPTX FILES CHANGED:             0
```

`git diff --stat` against `95dbbda` is empty — no tracked file was modified.
Everything added is new and analysis-only:

| path | kind |
|---|---|
| `tests/fixtures/pipeline_multivariate/` | analysis fixture (+ its builder) |
| `tests/fixtures/mi_query_stage_movement/MULTIVARIATE_PIPELINE_BANK.yaml` | question bank, existing schema |
| `scripts/prove_multivariate_pipeline_fixture.py` | fixture oracle |
| `scripts/run_mi_query_multivariate_audit.py` | audit harness |

An `__init__.py` was briefly created under `scripts/` to import the existing
bank runner and was **removed**: making `scripts/` a package changes import
semantics for every other script in the tree, which an audit may not do. The
runner is loaded by path instead.
