# Fabricated Population — Targeted Safety Fix

**Scope:** prevent a governed population concept the user never requested from executing.
Nothing else.
**Baseline:** `8a1a766` (second-book review escalation).

---

## 1. Root cause

`P1L` protects one direction: a population the question requested must reach execution or the
answer refuses. Nothing protected the mirror.

> **"What is the balance of the sponsored book?"** — a governed `ENTIRE_AUM` phrase (P1I-A).

The LLM emitted `seasoning_segment = back book`. That predicate was then **correctly** applied:
the frame narrowed, rows-before/rows-after were recorded, and the population facet was
legitimately `APPLIED`. Every guard reported success on an answer covering the wrong book.

Traced at the parse seam, the fault is unmistakable:

```
seasoning.segments_named("...sponsored book?")   -> []      # no segment requested
portfolio_lens.names_total_scope(...)            -> True    # ENTIRE_AUM requested
LLM spec filters                                 -> {"seasoning_segment": "back book"}
scope_role_rejected                              -> []      # governs absent fields only
seasoning_population                             -> None    # nothing to resolve, nothing removed
DETERMINISTIC filters                            -> {}      # correct
```

`reject_scope_role_filters` (P1I-A) only rejects filters on fields **absent** from the dataset;
`seasoning_segment` is a real column with a real value, so the spec validated cleanly. Nothing
downstream could tell that the user had asked about the whole book.

## 2. Smallest correction

One function, in the module that already models populations — no second population model.

```python
mi_agent/population.py :: fabricated_concepts(filters, question) -> List[str]
```

It compares **governed concepts**, never literal words:

| Concept | Executed when the spec filters on | Requested when the question |
|---|---|---|
| seasoning | `seasoning_segment`, `seasoning_bucket`, `months_on_book` | names a segment (`seasoning.segments_named`) or the seasoning field vocabulary |
| provenance | `source_portfolio_type` | resolves to a direct/acquired lens (`portfolio_lens.resolve_lens`) |

The vocabularies are read from the modules that already own each concept, so the guard cannot
drift from the code that decides what those phrases mean.

Wired at the same normalisation seam that owns the other population roles, and classified
**non-repairable**: a model that invented one population may invent a different one on a retry —
Kestrelmoor returned the front book twice and the back book once for the same question.
Recovery is the **existing sanctioned deterministic interpretation** (already used when an LLM
spec fails validation), which resolves the governed scope phrase correctly, or a refusal. No new
recovery behaviour was introduced.

## 3. Before / after — the sponsored book

| Book | Before | After |
|---|---|---|
| **Alderbridge** | 3/3 → `Seasoning Segment = back book`, **£1,793,150,141.49** | 3/3 → **£1,964,886,258.21**, 11,035 loans, *entire funded portfolio* |
| **Kestrelmoor** | 2/3 → front book **£238,685,188.37**; 1/3 → back book **£1,092,962,806.49** | 3/3 → **£1,331,647,994.86**, 12,255 loans, *entire funded portfolio* |

Both recover via `parser=deterministic_fallback` — the sanctioned path — and reconcile exactly to
independently computed AuM.

## 4. Legitimate derivation vs fabrication

The distinction the guard has to get right, and does:

| Spec | Question | Verdict |
|---|---|---|
| `months_on_book > 12` | "average LTV of the **back book**" | **legitimate** — the segment authorises its own derivation |
| `seasoning_bucket = 25-60m` | "the back book by seasoning band" | **legitimate** |
| `months_on_book > 60` | "loans with more than **60 months on book**" | **legitimate** — concept named directly |
| `direct + Back Book` | "the **direct back book**" | **legitimate** — both concepts named |
| `acquired + Front Book` | "the **acquired front book**" | **legitimate** |
| `seasoning_segment = Back Book` | "the **sponsored** book" | **fabricated** |
| `direct + Back Book` | "the **direct** book" | **fabricated** (seasoning invented) |
| `acquired + Back Book` | "the **back book**" | **fabricated** (provenance invented) |
| `months_on_book > 12` | "the **sponsored** book" | **fabricated** — derivation without the concept |

A user never has to say "months_on_book"; they have to have asked for the concept.

## 5. Blast radius

The success rule was: known fabricated-population cases change, everything else stays
semantically unchanged. That held.

| Asset | Before | After | Changed |
|---|---|---|---|
| Alderbridge Commercial Beta bank — deterministic | 29 / 5 | **29 / 5** | **0** |
| Alderbridge Commercial Beta bank — genuine LLM | 30 / 4 | **30 / 4** | 1 refusal *message* (see below) |
| Immutable 40-question bank | 14/40 | **14/40** | **0** |
| Kestrelmoor bank — deterministic | 48 / 2 | **48 / 2** | **0** |
| Kestrelmoor bank — genuine LLM | 47 correct, 1 incorrect | **49 correct, 0 incorrect** | K25 (the fix), K46 (see below) |
| P-gates + `mi_agent` + `mi_agent_api` suites | — | **2,922 passed** | 0 failed |

**Two apparent changes were investigated and are not caused by the fix:**

- **C21** ("how has the portfolio balance changed since last month?") — the refusal *wording*
  varies run to run. Measured directly: **3/5 vs 2/5** across five runs, the parse is stable at
  `validation_failed / missing_dataset_columns`, and `fabricated_concepts` returns `[]`. Both
  outcomes are safe refusals of the same question; the classification is unchanged.
- **K46** ("show balance by LTV by age") — 5 parse runs show `filters={}` and no fabricated
  concept in every one; the guard never fires. 4/5 produce a dimension spec that refuses,
  1/5 falls back to deterministic and renders the bubble. Pre-existing nondeterminism, already
  recorded in the multi-dimension review.

No new refusals, no changed populations, no altered correct answers.

## 6. Genuine-LLM repeated gate

Live API, **5 runs per case, both books**, provenance captured at the parse seam.

| Population | Alderbridge | Kestrelmoor |
|---|---|---|
| sponsored book | PASS — £1,964,886,258.21 / 11,035 (`deterministic_fallback` ×5) | PASS — £1,331,647,994.86 / 12,255 (`deterministic_fallback` ×5) |
| whole book | PASS — 11,035 (`llm` ×5) | PASS — 12,255 (`llm` ×5) |
| direct book | PASS — £1,385,508,582.98 / 7,126 | PASS — £436,631,033.20 / 5,612 |
| acquired book | PASS — £579,377,675.23 / 3,909 | PASS — £895,016,961.66 / 6,643 |
| front book | PASS — £171,736,116.72 / 1,177 | PASS — £238,685,188.37 / 3,074 |
| back book | PASS — £1,793,150,141.49 / 9,858 | PASS — £1,092,962,806.49 / 9,181 |
| direct back book | PASS — £1,237,547,386.29 / 6,199 | PASS — £265,071,428.39 / 3,274 |
| acquired front book | PASS — £23,774,920.03 / 250 | PASS — £67,125,583.56 / 736 |

**Both gates GREEN.** Every case **1 distinct outcome across 5 runs**.

```
FABRICATED_POPULATION  = 0
INCORRECT_SUCCESSFUL   = 0
SILENT_SEMANTIC_ERROR  = 0
HARD_FAILURE           = 0
```

## 7. Independent truth

Every figure above recomputed with pandas from each fixture; the MI executor was not used as its
own oracle. Populations verified by row count as well as balance, so a right number over the
wrong rows could not pass. Zero variance throughout.

Sponsored book reconciles to `ENTIRE_AUM` on both books, which was the specific requirement.

## 8. Regression results

`tests/test_fabricated_population.py` adds **21** tests: every fabrication case the brief names,
every legitimate governed derivation, the derivation-without-the-concept case, ordinary row
filters left untouched, and the parser-seam classification.

Accumulated suites: **2,922 passed, 1 skipped, 21 xfailed, 0 failed.** No existing test was
deleted, weakened or modified — the fix required no change to any prior expectation, which is
the strongest evidence that it did not disturb governed behaviour.

**Full repository suite:**

```
8,854 passed, 30 skipped, 21 xfailed, 48 warnings, 6 subtests passed
0 failed                                        in 2622.40s (0:43:42)
```

Baseline at `aa5436c` (P1N) was 8,833 passed; the increase is the 21 tests this fix adds. Green
on the first run — no test required correction, which for a change that touches the parse seam
is the outcome worth noting.

## 9. Beta-blocker verdict

**The fabricated-population Beta blocker is CLOSED.**

- A governed population concept the question did not request can no longer execute silently.
- The check is generic across the existing governed population concepts, not a sponsored-book
  keyword patch — it catches invented seasoning *and* invented provenance, in either direction.
- Legitimate governed derivations are untouched, including compound populations and bare
  `months_on_book` predicates.
- P1L is unchanged and still authoritative for the losing direction.
- No trade-off was accepted: nothing previously correct changed.

**Still open, separately:** the bubble-chart sampling disclosure, which the product owner has
ruled leaves bubble charts not commercially ready until fixed. That is untouched here.

---

FABRICATED POPULATION SAFETY FIX: PASS
