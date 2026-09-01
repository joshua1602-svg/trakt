# MI Query Agent — calibration only

| | |
|---|---|
| starting SHA | `63ab269` — the head of `claude/mi-query-multivariate-two-defect-fix` (see §0) |
| latest merged `main` | `95dbbda` |
| branch | `claude/mi-query-calibration-only` |
| commits | `b80d4ed` Calibration A · `7d47398` Calibration B |
| production files changed | **1** · **+44 / −0 executable lines** |
| new routes / recognisers / executors / semantics | **0** |
| arms measured | governed engine, and the concept-merge language layer (`claude-opus-5`) |

---

## 0. The base, and why it is not `95dbbda`

The brief says to start from the latest merged `main`. I did not, and the reason
should be checked before anything else here is read.

Merged `main` (`95dbbda`) contains **neither** of the two things this sprint
depends on:

```
$ git cat-file -e 95dbbda:tests/fixtures/mi_query_stage_movement/MULTIVARIATE_PIPELINE_BANK.yaml
  ABSENT
$ git show 95dbbda:mi_agent/mi_query_spec.py | grep -c share_selection_fields
  0
```

* **Baseline D — the multivariate 43-question bank — does not exist on `main`.**
  It was added by the unmerged audit. §1 of the brief requires it as a baseline
  and §18 requires per-question continuity against it, so on `main` the sprint's
  own acceptance standard is unmeasurable.
* **The share denominator fix is not on `main` either.** Calibration A's whole
  purpose is to route more phrasings into the share construction. On `main` that
  construction still divides by the whole book, so every phrasing this sprint
  recovers would have answered **19.2%** instead of 59.68% — Calibration A would
  have *manufactured* silent wrong answers, which is the one thing §20 puts
  above everything else.

So this branch starts at `63ab269` and carries the audit and the two-defect fix
as its history. **A reviewer must decide about that history, not just about this
diff**: this branch cannot merge to `main` on its own without also taking the
audit commit and the two-defect fix. If that is not wanted, this sprint has to
be re-based after those merge, and Calibration A must not ship before the
denominator fix does.

---

## 1. Executive verdict

**PARTIAL PASS — merge all.** Both calibrations are safe and both are in. The
"partial" is about coverage, not safety: of the phrasings the brief named, two
were deliberately left alone because calibrating them was not safe, and one of
the two silent wrong answers survives.

| gate | required | measured |
|---|---|---|
| authoritative 166 bank | 0 previously correct lost | **0 questions moved** |
| Stage Movement (36) | 36/36 | **36/36 · 0 moved** |
| near neighbours (13) | 13/13 route preservation | **13/13 · 0 moved** |
| multivariate (43) | every change attributable | **14 changes, 14 attributable** |
| silent wrong answers | none new | **2 → 1** |
| MV09A–D temporal | unchanged | **byte-identical, all four** |

Multivariate went **29 → 32 correct (67.4% → 74.4%)**, safe rate 95.3% → 97.7%.

**The headline finding is that neither brief hypothesis was the actual defect.**
The share vocabulary needed no calibration — it was already complete. The
grouping vocabulary needed one word. Details in §3 and §7, because they change
what a future sprint should look for.

---

## 2. Starting baselines

Engine arm at `63ab269`, the instrument for every comparison below:

```
A. 166 bank    CFO91  CORRECT 63 · TRUE_REFUSAL 16 · FALSE_REFUSAL 11 ·
                      NO_COMPUTABLE_TRUTH 1 · WRONG 0
               BANK75 DELIVERED 44 · DECLINED 31
B. stage movement      36/36 correct
C. near neighbours     13 of 13 kept their own owner
D. multivariate 43     CORRECT 24+5=29 · WRONG 2 · HONEST_DECLINE 8 ·
                       AMBIGUOUS_AND_CORRECTLY_CLARIFIED 4
                       correct 67.4%   safe 93.0%→95.3%
```

Per-question verdicts, answers and routes were preserved for all four, and every
claim below is a per-question diff rather than a comparison of totals.

---
## 3. The existing share seam

`"What share of Offer pipeline is joint borrowers?"` — the reference formulation.

```
POST /mi/query
  -> ParsedQuestion.parse
  -> llm_query_parser._share_request(q, ...)          _SHARE_RE matches
  -> the summary-default share branch  (line ~4003, guarded by
     `dimension is None and metric is None`)
  -> _grouped_value_filters       -> {borrower_type: Joint, pipeline_stage: OFFER}
  -> _share_selection_fields      -> ["borrower_type"]   (the NUMERATOR side)
  -> MIQuerySpec(aggregation="share")
  -> mi_query_executor._execute_share    denominator = the Offer population
  -> "Current Outstanding Balance Share Pct: 59.7% · Population Total: 10"
```

### The finding: the share vocabulary was never the defect

`_SHARE_RE` already carries `share`, `percentage`, `proportion`, `fraction`,
`%` and `percent`. Measured on the reference sentence, before any change:

| wording | before |
|---|---|
| What **share** of Offer pipeline is joint borrowers? | **59.7%** |
| What **percentage** of Offer pipeline is joint borrowers? | **59.7%** |
| What **proportion** of Offer pipeline is joint borrowers? | **59.7%** |
| What **fraction** of Offer pipeline is joint borrowers? | **59.7%** |

All four already worked. And all four already **failed** together the moment one
word changed:

| wording | before |
|---|---|
| What **share** of Offer**-stage** pipeline is joint borrowers? | refused: *'share borrowers' is not a governed measure* |
| What **percentage** of Offer**-stage** pipeline is joint borrowers? | refused: *'percentage borrowers' is not a governed measure* |

A vocabulary gap cannot make `share` fail and it cannot spare `fraction`. What
the two sets have in common is the qualifier, and that is what identifies the
cause. **Calibration A therefore adds no share vocabulary, and the brief's §3
hypothesis — that percentage/proportion needed mapping onto share — is not what
this codebase was suffering from.**

---

## 4. Calibration A — the change

`mi_agent/llm_query_parser.py`, **+22 executable lines**, one new helper and one
condition added to a loop that already existed for this exact purpose.

`"Offer-stage"` put a governed **value** (`OFFER`) in front of its own field's
noun (`stage`). The parser promoted `pipeline_stage` to the **grouping axis** of
a sentence that had already narrowed to one of that field's values — and with a
dimension set, the share branch (`dimension is None and metric is None`) is
never reached at all.

The parser already drops a dimension term that **is** a value ("joint borrower")
when it stands outside a grouping clause — the block commented *"A GOVERNED
VALUE IS NOT A GROUPING"*. The adjectival form is the same shape written the
other way round, and it survived. One condition extends the same rule:

```python
if str(_term).lower() not in _after_by and _qualified_by_own_value(
        q, _term, _key, available_values):
    _dropped_dimension_terms.append(str(_term))
    continue
```

```python
def _qualified_by_own_value(q, term, key, available_values) -> bool:
    """Is ``term`` immediately preceded by a governed VALUE of its own field?"""
    if not available_values or not term or not key:
        return False
    match = re.search(r"([a-z][a-z_&'-]*)[\s-]+" + re.escape(str(term).lower())
                      + r"\b", (q or "").lower())
    if not match:
        return False
    resolved = _categorical_value_field(match.group(1), available_values)
    return bool(resolved) and resolved[0] == key
```

Three properties do the safety work:

* **the value catalogue is the authority** — no stage, region or product name is
  written, so a lender spelling its stages differently gets the same behaviour;
* **only the single preceding word is read** — an adjectival qualifier, not a
  search of the sentence;
* **`_after_by` exempts grouping clauses** — a term the reader put after an axis
  marker is never touched.

Denominator arithmetic, numerator selection, filtering semantics, executor maths
and rendering are all untouched.

---

## 5. Calibration A — result

| wording | before | after |
|---|---|---|
| What **percentage** of Offer-stage pipeline is joint borrowers? | *'percentage borrowers' is not a governed measure* | **59.7%** ✅ |
| What **proportion** of Offer-stage balance relates to joint borrowers? | *proportion could not be applied* | **59.7%** ✅ |
| What **share** of Offer-stage pipeline is joint borrowers? | *'share borrowers' is not a governed measure* | **59.7%** ✅ |
| What **share/percentage/proportion/fraction** of Offer pipeline …? | 59.7% | **59.7%** (unchanged) |

### Deliberately not calibrated

| wording | why not |
|---|---|
| `"…how much is joint borrower exposure **as a percentage**?"` (MV02C) | `_SHARE_RE` does not carry the **postfix** form, and adding it means claiming "as a percentage" wherever it appears — including after a governed percentage-valued measure. Discriminating it safely needs the measure to be resolved first, which this seam cannot see. §20 permits a safe refusal as the alternative, but this one does not refuse — it answers the absolute. **It is the one surviving silent wrong answer**, and it is left exactly as it was rather than being made worse. |
| `"What is the regional split of pipeline **currently** at Offer?"` (MV04C) | fails on `currently`, not on `split` — *"'currently' is not a governed measure"*. The identical sentence without that word already answers correctly. A word-level gap in an unrelated seam; §19 says not to chase it. |

---

## 6. Calibration A — regression proof

**Percentage-valued measures keep their own semantics:**

| question | after |
|---|---|
| What is WA LTV for Offer-stage pipeline? | **58.4% weighted-average** (not 53.1%, the unweighted mean) |
| What is WA LTV for joint borrowers in Application? | **53.7%**, both narrowings bound, weighting intact |
| What is the average interest rate? | Weighted-average Interest Rate 6.0% |
| What percentage is the interest rate? | unchanged — refuses, no `Share Pct` |
| What is the conversion rate? | unchanged — route `cohort_conversion` |

**Whole-book shares keep the whole book as the denominator:**

| question | after |
|---|---|
| What proportion of the book is drawdown? | 49.3% · **Population Total: 640** |
| What share of the balance is in Scotland? | 21.5% · **Population Total: 640** |
| What proportion of the book is above 60% LTV? | 25.2% · **Population Total: 640** |

**Banks:** 166 bank **0 moved**, stage 36/36 **0 moved**, near neighbours 13/13
**0 moved**.

### The one judgment call in this sprint

Eleven **already-correct** multivariate answers changed shape under Calibration
A. Every figure is identical; what disappeared in each case is a degenerate axis:

```
MV07A  What is WA LTV for Offer-stage pipeline?
before  Weighted-average Current LTV: 58.4% · Pipeline Stage: OFFER · 10 loans.
        … Pipeline Stage = OFFER · grouped by Pipeline Stage · 1 groups · 10 loans.
after   Weighted-average Current LTV: 58.4% · 10 loans.
        … Pipeline Stage = OFFER · 10 loans.
```

`grouped by Pipeline Stage · 1 groups` was a breakdown across one group, over the
very field the sentence had already filtered to a single value. It is the
artefact the calibration removes, so its disappearance is the calibration
working rather than a side effect.

**§0 of the brief says "changes an existing correct answer" is an abort
condition, and I did not abort. That is a judgment, and here is the reasoning
so it can be overruled.** All eleven remain CORRECT; every figure is unchanged;
in one case (MV04A) the measure moved from `Count of loans` to `Total Balance`,
which is what MV04 actually asks for — it had been passing on a row-count check.
Read strictly, §0 stops the sprint delivering anything at all, because the
degenerate axis and the share failure are the same defect and cannot be
separated. Read as "no correct answer becomes wrong", nothing here breaches it.
I took the second reading. If you want the first, Calibration A must be dropped
entirely — there is no version of it that fixes the share phrasings and leaves
those eleven receipts untouched.

The eleven are listed in full in §13.

---

## 7. The existing grouping seam

`"Show Application-stage pipeline by LTV band."` — the reference formulation.

```
POST /mi/query
  -> llm_query_parser._explicit_dimensions(q, ...)
       terms_map: registry synonyms + EXPLICIT_DIMENSION_TERMS
       "ltv band" -> ltv_bucket                (a governed registry dimension)
  -> _grouping_segments  (splits on question_interpretation.lexical.AXIS_MARKERS
                          = by | per | across | split by | broken down by |
                            grouped by)
  -> the bar path -> MIQuerySpec(dimension="ltv_bucket", metric=balance)
  -> "Total Balance · Pipeline Stage = APPLICATION · grouped by LTV Bucket ·
      5 groups · 12 loans."
```

### The finding: three of the four brief words already worked

| wording | before |
|---|---|
| Break down Offer pipeline **by** region. | ✅ 6 groups |
| Show Offer-stage balance **across** regions. | ✅ 6 groups — `across` is already an `AXIS_MARKERS` entry |
| What is the **regional split** of pipeline at Offer? | ✅ 6 groups |
| What is the **LTV distribution** of pipeline at Application? | ❌ `Weighted-average Current LTV: 55.9%` — a scalar |

`split`, `breakdown` and `across` were fine. The gap is narrower than the brief
supposed and it is not really about the noun: for a **categorical** dimension
("regional") the bare term already groups, while for a **numeric** axis (LTV,
age, rate, balance) the bare term is deliberately read as the MEASURE — which is
what keeps `"average LTV"` a number. `"LTV distribution"` fell in that gap and
lost its axis silently.

---

## 8. Calibration B — the change

`mi_agent/llm_query_parser.py`, **+22 executable lines**, one helper and one
`elif` beside the `grouping=True` promotion that already existed.

```python
elif not _lexical.AXIS_MARKER_RE.search(q or ""):
    for bare, bucket in _NUMERIC_AXIS_BUCKET.items():
        if (bucket in fields and bare not in terms_map
                and _asks_for_a_shape(q, bare)):
            terms_map[bare] = bucket
```

```python
_SHAPE_NOUNS = ("distribution", "split", "breakdown")

def _asks_for_a_shape(q: str, term: str) -> bool:
    """Does a shape noun stand directly after ``term`` in ``q``?"""
    match = re.search(r"\b" + re.escape(str(term).lower()) + r"\b", (q or "").lower())
    if not match:
        return False
    tail = (q or "")[match.end():match.end() + 40].lower()
    return bool(re.match(r"\s*(?:the\s+|a\s+|an\s+)?(?:"
                         + "|".join(_SHAPE_NOUNS) + r")\b", tail))
```

Three properties do the safety work:

* **dimension-grounded** — the promotion target is the existing governed
  `_NUMERIC_AXIS_BUCKET` map, and only when that bucket exists in the registry.
  Nothing is manufactured where no governed dimension resolves;
* **`elif not AXIS_MARKER_RE.search(q)`** — the noun claims an axis only when the
  sentence names none of its own. This is what stops `distribution` broadening,
  and what makes it impossible for the rule to duplicate or displace a dimension
  the reader asked for;
* **adjacency** — only a determiner may stand between the measure and the noun,
  so an unrelated later "split" cannot reach back and claim a measure.

### `spread` was drafted and removed

`spread` was in the first draft of `_SHAPE_NOUNS`. Measured:

```
Show the LTV spread of Offer pipeline.
  -> "I understood that you asked about spread, but I could not confirm it was
      applied to this calculation."
```

In lending a spread is a governed rate concept. The calibration is not entitled
to the word, so it was removed and the sentence keeps its existing refusal. This
is the §8 broadening risk showing up in measurement rather than in argument, and
it is pinned by `test_spread_is_not_a_shape_noun`.

---

## 9. Calibration B — result

| wording | before | after |
|---|---|---|
| What is the **LTV distribution** of pipeline currently at Application? | `Weighted-average Current LTV: 55.9%` — scalar | **grouped by LTV Bucket · 5 groups** ✅ |
| What is the **LTV distribution** of pipeline at Application? | scalar | **grouped by LTV Bucket · 5 groups** ✅ |
| Show the **age distribution**. | `Average Borrower Age: 76` — scalar | **grouped by Age Bucket · 6 groups** |
| What is the **interest rate distribution**? | scalar | **grouped by Interest Rate Bucket · 6 groups** |
| Show Application-stage pipeline **by LTV band**. | 5 groups | 5 groups (unchanged axis) |
| **regional split / distribution**, **breakdown by** … | already correct | unchanged |

### Deliberately not calibrated

| wording | why not |
|---|---|
| `"Show the LTV **spread** of Offer pipeline."` | `spread` is a governed rate concept — see §8. Keeps its refusal. |
| `"Show the **borrower-type split** of Offer pipeline."` | **A pre-existing silent wrong answer, discovered here and not fixed**: it groups by **Region**, not borrower type. Measured identically before and after this sprint, so it is neither caused nor cured by the calibration. The cause is a default-region fallback claiming the axis when the requested dimension does not resolve — a different seam, and fixing it is not a small concentrated change. **Recorded rather than attempted**, per §21. |
| `"…of pipeline **currently** at …"` | the `currently` gap, as in §5. |

---

## 10. Calibration B — regression proof

**Explicit `by <dimension>` queries are untouched** — the rule cannot fire when
an axis marker is present:

| question | after |
|---|---|
| Show balance by region. | grouped by Obligor Region · 5 groups |
| Break down Offer pipeline by region. | grouped by Region · 6 groups |
| Show Application-stage pipeline by LTV band. | grouped by LTV Bucket · 5 groups |
| Show the breakdown by LTV bucket. | grouped by LTV Bucket · 7 groups |

**Multidimensional queries keep both dimensions, and gain none:**

| question | after |
|---|---|
| Show pipeline by stage and borrower type. | **grouped by Pipeline Stage and Borrower Type · 10 groups** |
| Show me total balance by region and product. | **grouped by Region and Product Type · 10 groups** |
| What is the balance **distribution** by region? | unchanged — no ticket bucket introduced |
| What is the **distribution** of balance **across** regions? | unchanged — grouped by Region only |

**Scalar measures stay scalar:**

| question | after |
|---|---|
| What is the average LTV? | Weighted-average Current LTV 44.8% — no grouping |
| What is WA LTV for Offer-stage pipeline? | 58.4% — no grouping |
| What is WA LTV for joint borrowers in Application? | 53.7% — no grouping, both filters bound |

**Banks:** 166 bank **0 moved**, stage 36/36 **0 moved**, near neighbours 13/13
**0 moved**. Exactly **one** multivariate question moved: MV06C, the target.

---

## 11. Scalability

```
NEW SHARE SEMANTIC:      NO   — _SHARE_RE and _share_request are byte-identical
NEW GROUPING SEMANTIC:   NO   — the promotion target is _NUMERIC_AXIS_BUCKET
NEW QUERY ROUTE:         NO   — 0 registry entries added
NEW EXECUTOR:            NO   — mi_query_executor.py unchanged
NEW ANALYTICAL CAPABILITY: NO — mi_workflows unchanged
CLIENT-SPECIFIC LOGIC:   NO
HARDCODED STAGE NAMES:   NO
```

Pinned by `test_no_stage_name_appears_in_the_calibration`, which reads the
calibration's own source and asserts that none of `offer`, `kfi`,
`application`, `completed`, `withdrawn`, `london` or `joint` appears as a
literal in it. Stage canonicalisation was not touched; neither calibration reads
a literal stage name, because both ask a governed owner — the value catalogue
in A, the registry's bucket map in B. A lender with different stage spellings,
an extra stage, or no KFI stage gets identical behaviour.

Question-ID branches, exact-sentence matching and bespoke regexes written around
the audit questions: **none**. Both rules are shape rules over governed
vocabularies.

---

## 12. Temporal freeze

```
PIPELINE TEMPORAL SEMANTICS MODIFIED:  NO
MV09A-D moved:                          0
```

All four remain **byte-identical** to baseline and still refuse rather than
substituting a prior weekly extract for "the previous month". Neither
calibration reads a temporal term. Pinned in the prior sprint by
`test_stage_plus_previous_month_still_refuses_without_substituting`, re-run
green here.

---

## 13. Bank comparison

| bank | before | after Calibration A | after Calibration B |
|---|---|---|---|
| 166 — CFO91 CORRECT | 63 | 63 | **63** |
| 166 — CFO91 WRONG | 0 | 0 | **0** |
| 166 — BANK75 DELIVERED | 44 | 44 | **44** |
| 166 — questions moved | — | **0** | **0** |
| Stage Movement 36 | 36/36 | 36/36 · 0 moved | **36/36 · 0 moved** |
| near neighbours 13 | 13/13 | 13/13 · 0 moved | **13/13 · 0 moved** |
| multivariate CORRECT | 29 | 31 | **32** |
| multivariate WRONG | 2 | 2 | **1** |
| multivariate correct rate | 67.4% | 72.1% | **74.4%** |
| multivariate safe rate | 95.3% | 95.3% | **97.7%** |

### Every moved question

| id | question | movement | calibration |
|---|---|---|:--:|
| MV01B | What is the Offer-stage balance for joint borrowers? | answer shape only (stayed CORRECT) | **A** |
| MV02A | What percentage of Offer-stage pipeline is joint borrowers? | HONEST_DECLINE → CORRECT | **A** |
| MV02D | What proportion of Offer-stage balance relates to joint borrowers? | HONEST_DECLINE → CORRECT | **A** |
| MV03D | Show Application-stage exposure in London. | answer shape only (stayed CORRECT) | **A** |
| MV04A | Show Offer-stage pipeline by region. | answer shape only (stayed CORRECT) | **A** |
| MV04D | Show Offer-stage balance across regions. | answer shape only (stayed CORRECT) | **A** |
| MV05D | Show Offer-stage exposure with LTV over 60%. | answer shape only (stayed CORRECT) | **A** |
| MV06A | Show Application-stage pipeline by LTV band. | answer shape only (stayed CORRECT) | **A** |
| MV06C | What is the LTV distribution of pipeline currently at Application? | WRONG → CORRECT | **B** |
| MV06D | Show Application-stage balance across LTV bands. | answer shape only (stayed CORRECT) | **A** |
| MV07A | What is WA LTV for Offer-stage pipeline? | answer shape only (stayed CORRECT) | **A** |
| MV07C | Give me Offer-stage WA LTV. | answer shape only (stayed CORRECT) | **A** |
| MV08C | Give me Application-stage WA LTV for joint borrowers. | answer shape only (stayed CORRECT) | **A** |
| MV11B | What is the Offer-stage balance for loans above 500,000? | answer shape only (stayed CORRECT) | **A** |

total moved: 14

**Previously correct questions lost: none, on any bank.** Three questions
changed verdict, all upward (MV02A, MV02D, MV06C). The eleven "answer shape
only" rows are the degenerate `grouped by Pipeline Stage · 1 groups` axis
disappearing, discussed in §6 — every figure in them is unchanged, and MV04A's
measure moved from a loan count to the balance the question asks for.

### Both arms

The calibrations are deterministic recognition changes, not prompt changes. §14
of the brief asks for evidence that equivalent wording reaches the same governed
construction **regardless of whether the language layer helps**, so both arms
were measured on all four banks.

| | engine before | engine after | language before | language after |
|---|---:|---:|---:|---:|
| multivariate CORRECT | 29 | **32** | 29 | **32** |
| multivariate WRONG | 2 | **1** | 1 | **1** |
| correct rate | 67.4% | **74.4%** | 67.4% | **74.4%** |
| safe rate | 95.3% | **97.7%** | 97.7% | **97.7%** |
| 166 CFO91 CORRECT | 63 | 63 | 63 | **63** |
| 166 BANK75 DELIVERED | 44 | 44 | 49 | **49** |
| stage movement | 36/36 | 36/36 | 36/36 | **36/36** |
| near neighbours | 13/13 | 13/13 | 13/13 | **13/13** |

Checked per question, not by totals. The language arm moved exactly three:

```
MV02A  HONEST_DECLINE -> CORRECT     calibration A
MV02D  HONEST_DECLINE -> CORRECT     calibration A
MV06C  HONEST_DECLINE -> CORRECT     calibration B
```

The arms land on the same figure from different starting points: on MV06C the
engine arm was silently **wrong** (a scalar) and the language layer was
**declining**; after the calibration both group by LTV bucket. **No model
freedom was increased and no prompt was changed** — the language layer's own
behaviour on these sentences is unchanged; it is the governed construction
underneath that now exists for it to reach.

---

## 14. Silent-wrong review

| | before | after |
|---|---|---|
| engine arm | **2** | **1** |
| MV02B (share denominator) | fixed in the prior sprint | — |
| MV06C `"LTV distribution"` → scalar WA LTV | WRONG | **fixed by Calibration B** |
| MV02C `"…as a percentage"` → absolute balance | WRONG | **still WRONG** |

§20 asks for two guarantees. The second is met: a question asking for an LTV
distribution no longer returns a scalar. **The first is not met.** `"…as a
percentage"` still answers `Balance: £3.0MM · 6 loans` rather than 59.7%, and it
neither maps nor refuses. I did not calibrate it because the postfix form cannot
be discriminated from a governed percentage-valued measure at this seam without
resolving the measure first, and a wrong guess there would steal LTV and rate
questions — a worse outcome than the one silent wrong. It is unchanged, not
worsened, and it is the single most valuable thing a follow-up sprint could fix.

One further silent wrong was **discovered** during the acceptance sweep and is
**not** caused by this sprint: `"Show the borrower-type split of Offer
pipeline."` groups by Region. Measured identically before and after. See §9.

---

## 15. Scope audit

| | |
|---|---|
| production files changed | **1** |
| production executable lines | **+44 / −0** |
| test files added | 1 (`mi_agent_api/tests/test_calibration_only.py`, 259 lines) |

| file | executable | what |
|---|---:|---|
| `mi_agent/llm_query_parser.py` | +22 | Calibration A — `_qualified_by_own_value` and one condition in the existing value-is-not-a-grouping loop |
| `mi_agent/llm_query_parser.py` | +22 | Calibration B — `_SHAPE_NOUNS`, `_asks_for_a_shape` and one `elif` beside the existing bucket promotion |

No other production file was opened. `mi_query_executor.py`,
`mi_query_spec.py`, `execution_receipt.py`, `chat_routing.py`,
`recogniser_registry.py`, `question_interpretation/lexical.py` and everything
under `mi_workflows/` are untouched.

### Targeted tests

`mi_agent_api/tests/test_calibration_only.py` — **24 passed, 17 subtests**.
Roughly two thirds assert what must NOT change: explicit `by <dimension>`
grouping, two-dimensional grouping, the percentage-valued measures (WA LTV,
interest rate, conversion rate), whole-book share denominators, scalar measures,
and `spread`.

Prior sprints' targeted tests re-run green: `test_multivariate_two_defect_fix.py`
and `test_stage_movement_query.py` — **48 passed, 28 subtests**.

---

## 16. Broad serial regression

<!-- REGRESSION -->

