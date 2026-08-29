# The dataset class — diagnosis

Code base identical to `31c3257` (HEAD `1aaf52f` adds one markdown file and no
code). **Nothing built, no design proposed.** The gate, `scopeApplied`, the
coordinated-axis read and the five-reader consolidation are untouched.

**Two corrections to my own premise come first, because both change the answer.**

> **It is four questions, not fifteen.** Eleven of the fifteen are a *naming*
> mismatch between two correct descriptions of the same composed frame, not a
> substitution. I asserted "all genuine losses" in the wiring scope on a test that
> cannot tell a composite from a substitution. That was wrong.
>
> **They did not survive every bank — the bank caught them and my grader
> discarded the verdict.** The frozen readiness bank grades "Summarise the current
> pipeline" **`WRONG / SILENT`**, rationale *"answered from the FUNDED book; the
> question named the pipeline dataset"*. My `grade_75` returns
> `NO_COMPUTABLE_TRUTH` whenever `independent_truth` is null and never reads the
> frozen human `grade` sitting in the same row. Five pack rows are affected.

---

## 1. Where the substitution happens

**Three routes, and the dataset stops being honoured at the route boundary.**

The decision itself is correct throughout. Traced live on "Summarise the current
pipeline.":

```
resolve_dataset(question)  -> 'pipeline'          the owner is right
_resolve_frame('pipeline') -> 8 rows              the frame IS loaded
try_route(view='pipeline')                        the route IS told
   -> route portfolio_summary                     …and answers 640 funded loans
```

`try_route` receives `view` and uses it in exactly three places — the book-value
catalogue, the interpretation projection, and the ownership re-read. **It is never
used to select the answering frame.** The handlers load from `output_root`, which
is the funded onboarding output. `_route_portfolio_summary` does not take a
dataset parameter at all:

```python
def _route_portfolio_summary(question, spec, spec_dict, *, client_id, run_id,
                             output_root, portfolio_id, as_of, source_lens=None,
                             interpretation=None)
```

The three implicated routes and their questions, complete:

| route | question | named | reconciled |
|---|---|---|---|
| `portfolio_summary` | Summarise the current pipeline. | pipeline | **funded** |
| `risk_limits` | Based on the current book and forward pipeline, which concentration tests are we at risk of breaching? | pipeline | **funded** |
| `funded_bridge` | Show funded vs pipeline contribution. | pipeline | **funded** |
| `funded_bridge` | What is the weighted expected pipeline contribution? | pipeline | **funded** |

**The routed path is not broken as a class.** On the same corpus, `evolution`
(6), `evolution_pipeline_stage` (4) and 14 further route-claimed questions
reconcile against `pipeline` correctly. Three routes, not the mechanism.

### The eleven that are NOT substitutions

`analytical_composition` (8) and `forecast_extrapolation` (3) publish
`datasetContext: forecast` beside `reconciliation.dataset: funded+pipeline`.
`workspace.build_forecast_view_frame` composes the funded book with the pipeline
— measured, the forecast frame is **645 rows** = 640 funded + 5 pipeline. So the
route read exactly what a forecast is, and reconciled against it under its
compositional name. **Two names for one frame, not a wrong frame.** The
completeness check flags them because it compares strings; that is a limit of the
check, and it is why "datasetContext != reconciliation.dataset" is not a
substitution test.

---

## 2. Is it Q19C's shape? Two of three causes, and the third is inverted

| Q19C's cause | here |
|---|---|
| **1. A lens naming a book while carrying nothing to narrow by** | **Present.** `datasetContext: pipeline` is published on an answer carrying only funded rows. Same shape: the envelope names the thing and the figure is not of it. |
| **2. A documented precondition nothing enforced** | **Present, and stated twice.** `chat_routing.py:262` — *"`workspace.resolve_dataset`, which is the single owner. Routes ask that owner"*. `chat_routing.py:3495` — *"NO `view=`. The dataset is the question's, and the route asks `workspace.resolve_dataset` for it."* **`chat_routing.py` never calls `resolve_dataset`.** All four mentions in the file are comments. The three implicated handlers do not take a parameter that could carry the answer. |
| **3. Routes narrowing silently** | **INVERTED, and favourably.** These routes DO publish `reconciliation.dataset`, and it disagrees with `datasetContext`. The record exists and contradicts itself in one envelope — which is why a deterministic check found all of them with no new reader. Q19C's routes published nothing at all. |

So: the same two structural causes, and the disclosure Q19C lacked is already
present. That is the material difference and it is what makes this cheaper to
detect than Q19C was.

---

## 3. Is the pipeline data present? Yes — this is a routing failure

```
_resolve_frame('pipeline', 'client_001/mi_2026_06') -> 8 rows, err=None
columns: record_type, current_outstanding_balance, pipeline_stage,
         kfi_date, application_date, offer_date, expected_completion_date, …
current_outstanding_balance: £3,600,000     pipeline_stage: 5 distinct
```

The tape carries it, the estate loads it, and **another formulation of the same
question already answers from it**:

```
"Give me an overview of the pipeline by size and stage."
   route=None (point-in-time) · reconciled=pipeline
   -> "Total Balance · grouped by Pipeline Stage · 5 groups · 8 loans."
```

**The point-in-time path honours the dataset. The routed path does not.** The
only difference between Q10A and Q10B is which route claims the sentence.

So the refusal-that-lies class does **not** apply: these are answers, not
refusals, and no claim about missing data is made. The correct behaviour is not a
refusal naming what is missing — nothing is missing.

---

## 4. Why there is no computable truth, and whether one can be built

**A truth is fully constructible, by the same method as every other truth in the
bank.** Computed from the pipeline frame:

```json
{"rows": 8, "balance": 3600000.0, "stages": 5}
```

The answer given is 640 loans and £172.1m. Any of the three fields separates them.

**So the gap is in the bank — and worse, in my instrument.** `independent_truth`
is null for these cases, and `grader.grade_75` returns `NO_COMPUTABLE_TRUTH` the
moment it is null, **without ever reading the frozen human `grade` in the same
row**. The bank had already judged them:

| id | my grade | frozen human grade | frozen rationale |
|---|---|---|---|
| **Q10A** | NO_COMPUTABLE_TRUTH | **WRONG / SILENT** | answered from the FUNDED book; the question named the pipeline dataset |
| **Q07B** | NO_COMPUTABLE_TRUTH | **WRONG / SILENT** | both scopes dropped; a whole-book figure answered a comparison question |
| **Q25A** | NO_COMPUTABLE_TRUTH | **CURRENT-STATE SUBSTITUTION** | a FORWARD question answered with today's risk-limit status |
| **Q25B** | NO_COMPUTABLE_TRUTH | **CURRENT-STATE SUBSTITUTION** | as above |
| **Q25C** | NO_COMPUTABLE_TRUTH | **CURRENT-STATE SUBSTITUTION** | as above |

Of the 22 pack rows my grader called `NO_COMPUTABLE_TRUTH`, 17 carry a frozen
grade of EXACT or SUBSTANTIVELY CORRECT — where the label is harmless — and **five
carry a recorded wrong verdict that the label buried.**

This is F3's family, and on my own instrument: *an instrument that cannot measure
must report NOT MEASURED, never clean.* `NO_COMPUTABLE_TRUTH` reported not-measured
honestly and then **out-ranked a verdict that had been measured by someone else.**
It should defer to a recorded human grade rather than replace it.

---

## 5. Blast radius

| | pack (166) | corpus (1,446) |
|---|---:|---:|
| name a non-default dataset | 10 | 115 |
| claimed by a route | 4 | 56 |
| fell to point-in-time | 6 | 59 |
| **deliver reconciled against `funded`** | **2** | **4** |

Every affected question is listed in §1. The pack's two are Q10A and Q25C.

**Currently CORRECT answers on the non-default set: 5 in the pack**, all
reconciled `pipeline`, none in the substitution set:

```
What is the pipeline balance?          CORRECT   route=None        pipeline
How many cases are in the pipeline?    CORRECT   route=None        pipeline
Show the pipeline by stage.            CORRECT   route=None        pipeline
How has the pipeline evolved?          CORRECT   evolution         pipeline
Show pipeline evolution by stage.      CORRECT   evolution_…stage  pipeline
```

### Recovery

**Four answers, three routes.** Two of them (Q10A, Q25C) already carry a frozen
wrong verdict, so recovering them converts two recorded wrong answers. The other
two are corpus-only.

A second, larger recovery is available **without touching any route**: making
`grade_75` defer to the frozen human grade when no truth is computable would
surface five recorded wrong verdicts the pack currently hides — including Q07B,
which has nothing to do with datasets. That is an instrument fix, not a product
fix, and it is the cheaper half.

### Risk — stated separately

The four bound the recovery. They do **not** bound the risk.

**The risk zone is 32 questions**: delivering, route-claimed, naming a non-default
dataset, and currently reconciled against something other than `funded`:

```
pipeline -> pipeline        14      already correct
forecast -> forecast         7      already correct
forecast -> funded+pipeline  6      the composite; correct under another name
pipeline -> forecast         3      the composite; correct under another name
pipeline -> funded+pipeline  2      the composite
```

Any change that makes a route honour the named dataset is judged against these 32,
and the nine `funded+pipeline` cases are the hazard: a rule that forces a route to
read *only* the named dataset would break the composition that makes a forecast a
forecast. **The composite is not a bug and must not be "fixed".** Separating "read
the pipeline" from "compose funded and pipeline" is the actual difficulty, and it
is not measured here because measuring it means building.

The five currently-correct pack answers sit outside the substitution set but
inside the same routing surface, so they are the regression check any change owes.

---

## 6. The answer to the question you asked

**Bounded — more bounded than I reported.** Three routes, four questions, one
missing enforcement of a precondition the code states twice in comments. The
dataset owner is already correct, the frame is already loaded, the route is
already told, and the point-in-time path already honours it.

**Recovery: four answers, two of them already graded wrong by the frozen bank.
Plus five recorded verdicts recoverable by fixing my grader instead of the
product.**

**Risk: 32 questions, nine of which are correct compositions that a naive
"honour the named dataset" rule would break.**

Not fixed. Nothing built.

### Environment
`MI_AGENT_LLM_PARSER=off` throughout (F2), run from the repository root (F6).
**Successful model responses: 0.**
