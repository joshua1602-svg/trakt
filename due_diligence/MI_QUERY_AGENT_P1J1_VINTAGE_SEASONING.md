# MI Query Agent — P1J-1: Governed Vintage & Seasoning

**Objective:** implement vintage/seasoning as a governed MI semantic axis,
completely independent of provenance; correct the existing conflation; expose the
existing vintage analytics; unlock B04/B09/B28; preserve the zero-silent-error
boundary.

**Result:** immutable 40-bank **11/40 → 14/40**, exactly B04, B09 and B28 changed,
**zero regressions**, all truth reconciled to 1e-9.

---

## 1. Architecture before

Almost all of the machinery already existed, in four places:

| Component | State before P1J-1 |
|---|---|
| `origination_date` | **present** on the tape, 11,035 / 11,035 non-null, 2014-01-01 → 2026-06-30 |
| `vintage_year`, `months_on_book` | governed in `mi_agent/mi_semantics_field_registry.yaml`; **derived** in `funded_prep._derive_source_fields` against the reporting date |
| `config/mi/buckets.yaml` | bucket-edge config + `analytics_lib.buckets` engine, applied by `data_source._materialise_mi_buckets` |
| `execution_receipt._COHORT_CONCEPTS` | already carried the vintage CONCEPT vocabulary ("back book", "seasoned", "front book", "new origination") and the groupable vintage dimensions |
| `portfolio_lens` | owned the PROVENANCE vocabulary — and carried five seasoning phrases inside it |

So the vintage axis was governed, derivable, and already had a safety guard
watching for it. What it did not have was a materialised dimension on the read
path the MI Agent actually uses, or a vocabulary that kept it separate from
provenance.

---

## 2. Root cause of the vintage/provenance conflation

Three independent conflations, at three different layers.

### 2.1 Vocabulary (`mi_agent/portfolio_lens.py`)

```
_ACQUIRED_TERMS ⊃ "back book", "backbook", "legacy book"
_DIRECT_TERMS   ⊃ "new origination", "newly originated"
```

Verified live before the fix:

```
"the back book"     -> lens=acquired  (source_portfolio_type = acquired)
"the legacy book"   -> lens=acquired
"newly originated"  -> lens=direct    (source_portfolio_type = direct)
```

Asking for "the back book" silently returned the **acquired** book: every
seasoned *direct* loan excluded, every recent *acquired* loan included. On this
book that is not hypothetical — **250 loans are Acquired + Front Book** and
**6,199 are Direct + Back Book** (§9), so the two axes disagree about 6,449 loans.

### 2.2 Routing (`mi_workflows/portfolio_risk_comparison.py`)

"Compare the front book with the back book" matched `has_marker and has_noun`
("compare" + "book") and was recognised as a **source-portfolio** comparison. It
was answered by sourcing — the receipt said *"the books were compared by
sourcing, which is not how long the loans have been on the book"* — and the P1G
cohort-identity guard refused it. **Never wrong, but a refusal is not the right
answer to an answerable question.**

### 2.3 Data (`mi_agent_api/funded_prep.py`)

`_derive_source_fields` already derived `vintage_year` and `months_on_book`
correctly, against the reporting date — but only on the **funded-tape prep**. The
MI Agent reads the **platform canonical**, whose read-time derivation
(`augment_platform_canonical_dimensions`) derived borrower type, youngest age and
region, and **skipped vintage entirely**. So a book carrying an origination date
for all 11,035 loans reported *"'Vintage' is not available in this dataset"*.

That is why P1J classified this as a **false missing-data** case, and it is the
single largest cause of the refusals.

---

## 3. The governed semantic model

One model, one config block, two outputs — so the bands and the binary split can
never disagree:

```
origination_date
  └─ months_on_book            (vs the governed reporting date)
       ├─ vintage_year         (origination-year cohort)
       ├─ seasoning_bucket     (analytical bands)
       └─ seasoning_segment    (binary front/back — layered on the same model)
```

* **Nothing is stored.** `seasoning_segment` and `seasoning_bucket` are derived at
  read time, exactly as `vintage_year`, `age_bucket` and `borrower_type` already
  were. Front/back is never the fundamental data model.
* **Front/back is not a second mechanism.** It is a boundary on the same model.
  Proven by test: the Front Book population and the `0-12m` band are the *same
  1,177 rows*, by index, not merely the same count.
* `front_book_max_months` is a deliberate separate scalar rather than "the first
  bucket edge", so a client can move the front/back boundary without re-cutting
  the analytical bands, and vice versa.

New module: `mi_agent/seasoning.py` (config loader, `SeasoningConfig`,
`derive_seasoning`, `resolve_segment_population`, `describe_segment`).

---

## 4. Configuration

No new configuration framework: the governed `seasoning:` block was added to the
existing `config/mi/buckets.yaml`, read through the existing loader.
`load_bucket_config()` returns only the `buckets` mapping, so the added top-level
key is inert for every existing consumer (verified).

```yaml
seasoning:
  front_book_max_months: 12          # client-configurable; never hard-coded
  buckets:
    - {name: "0-12m",  min_months: 0,  max_months: 12}
    - {name: "13-24m", min_months: 13, max_months: 24}
    - {name: "25-60m", min_months: 25, max_months: 60}
    - {name: "60m+",   min_months: 61}
```

Proven config-driven, not code-driven: a loan 18 months on book is **Back Book**
under the governed 12-month default and **Front Book** under a client file
setting `front_book_max_months: 24` — same code, same loan.

`config/mi/buckets.yaml::time_on_book_bucket` is a **separate, pre-existing
QUANTILE dimension** (`mi_agent/quantile_buckets.py`) and is deliberately
untouched.

---

## 5. Phrase resolution

| Phrase | Resolves to | Role |
|---|---|---|
| back book, backbook, legacy book, seasoned book, seasoned loans | `seasoning_segment` = **Back Book** | population |
| front book, new origination(s), recent origination(s), newly originated, recently originated | `seasoning_segment` = **Front Book** | population |
| both sides named ("front book **vs** back book", B04) | `seasoning_segment` | **grouping** (comparison) |
| vintage, vintages, vintage year, origination year(s), origination vintage, cohort year | `vintage_year` | grouping |
| seasoning bucket / band, months-on-book band | `seasoning_bucket` | grouping |
| **new lending** | dimension synonym only — **never** selects a population | fails safe |
| **older vintages** | **unresolved** | fails safe |
| direct / directly originated / organic / own book / in-house | provenance lens **direct** | scope |
| acquired / purchased / bought book / acquisition / inorganic / m&a | provenance lens **acquired** | scope |

Three deliberate refusals to over-map, per the brief's "do not blindly make every
phrase an exact synonym":

* **"new lending"** reads as a FLOW measure at least as naturally as a population
  ("the run rate of new lending" — B08). It stays a dimension synonym but never
  narrows the book. B08 is asserted unchanged.
* **"older vintages"** more naturally asks for analysis *across* vintage cohorts
  than for `back_book = true`. It resolves to nothing and fails safe.
* **bare "quality"** is not a measure synonym; only "credit quality" / "quality of
  the book" / "book quality" are, per ruling R1 (§7).

**The two vocabularies may not overlap**, in either direction — asserted
structurally by `test_the_two_vocabularies_do_not_overlap`, so the conflation
cannot creep back.

### Role resolution: population vs axis

`resolve_seasoning_role` decides the ROLE once, at normalisation, before the spec
is validated — the same discipline as P1I-A's scope resolution:

* names **one** segment → **population filter** ("the average LTV of the back
  book" narrows to 9,858 loans);
* names **both** → **grouping** (the comparison IS the question);
* names none → untouched.

This closed a live silent semantic error: "the average LTV of the back book" was
resolved as a *grouping*, so it answered for front **and** back — all 11,035
loans — while presenting itself as an answer about the back book.

---

## 6. Analytics reused

Per the brief, exposure and composition, not a second vintage engine:

| Reused | Where |
|---|---|
| vintage / months-on-book derivation | `funded_prep._derive_source_fields` — extracted to `_derive_vintage_and_seasoning` and now **shared** with the platform-canonical path |
| bucket materialisation | `analytics_lib.buckets` via `data_source._materialise_mi_buckets` (unchanged) |
| grouped multi-measure execution | the existing P1E machinery (B04 returns balance + count + WA LTV by segment) |
| ranking over a dimension | the existing ranking path (B09) |
| cohort-identity guard | `execution_receipt._COHORT_CONCEPTS` — extended with `seasoning_segment`, not replaced |
| semantic resolution | `semantic_resolver.resolve_dimension` over registry synonyms (registry regenerated from curation, per its contract) |

No duplicate maths was written. The only new calculation is the band/segment
assignment itself, which is a lookup over `months_on_book`.

---

## 7. Rulings applied

| # | Ruling |
|---|---|
| **R2a** (given) | Front/back = vintage; direct/acquired = provenance; independent axes; 12-month default; configurable. |
| **R1** (P1J §0) | On a fully-performing book (arrears, defaults, impairment all zero) **credit quality = weighted-average current LTV**, lower is better. Curated as synonyms on `current_loan_to_value` with the reasoning recorded in the build script; the receipt always names the executed measure, so the interpretation is visible. |

---

## 8. B04 / B09 / B28 — before → after

| ID | Exact bank wording | Before | After | Why |
|---|---|---|---|---|
| **B04** | "Is the credit quality of new origination better or worse than the back book?" | REFUSAL — *"a comparison by how long the loans have been on the book … could not be applied"* | **CORRECT** | vintage materialised; compares **Front vs Back Book** on WA current LTV (R1); never direct/acquired |
| **B09** | "Which vintages have the highest LTV?" | REFUSAL — *"ranking by vintage … the requested breakdown was not applied"* | **CORRECT** | `vintage_year` materialised + plural synonym; 13 vintages ranked by WA current LTV |
| **B28** | "Show me the quality of the book by origination vintage." | REFUSAL — *"'Vintage' is not available in this dataset"* | **CORRECT** | as B09, plus R1 quality→LTV and the line-coercion fix below |

All three are **consequences of the reusable capability**, not string-matched:
none of the three bank strings appears anywhere in the implementation.

**A silent semantic error found and fixed en route.** A vintage request forced a
LINE chart, whose x-axis is date-coerced. Every integer year became epoch month
`1970-01`, collapsing all thirteen vintages into **one row** carrying the
whole-book LTV — and still reporting itself as *"by origination vintage"*. A
vintage is a cohort label, not a point on a time axis; it is now a grouped bar,
which is what the ranking path already produced correctly.

---

## 9. Independent truth reconciliation

Recomputed with pandas from `origination_date` vs `data_cut_off_date`. The
seasoning model is never its own oracle.

**B04 — Front vs Back at the governed 12-month cutoff**

| Segment | Loans | Balance | WA current LTV |
|---|---|---|---|
| Front Book | 1,177 | £171,736,116.72 | 34.706481 |
| Back Book | 9,858 | £1,793,150,141.49 | 43.965661 |
| **Total** | **11,035** | **£1,964,886,258.21** | 43.156246 |

Verdict: newer origination is **better** (lower LTV). Agent output matches to
1e-9. *(The P1J report's illustrative 24-month figures — 35.5 / 44.8 — are
superseded by these governed 12-month figures.)*

**B09 / B28 — WA current LTV by vintage** (agent vs truth, all MATCH):
2014 54.4736 · 2015 51.3951 · 2016 48.3335 · 2017 46.7085 · 2018 44.3469 ·
2019 49.0084 · 2020 45.9713 · 2021 44.0098 · 2022 41.6309 · 2023 39.3765 ·
2024 37.4288 · 2025 35.0876 · 2026 34.5862.

A monotone decline with one 2019 step — older equity-release roll-up loans
accrete interest, so LTV rises with seasoning. That is the business story the
capability now tells.

**Seasoning bands** (reconcile exactly to the book):

| Band | Loans | Balance | WA LTV |
|---|---|---|---|
| 0-12m | 1,177 | £171,736,116.72 | 34.7065 |
| 13-24m | 1,270 | £186,979,928.03 | 36.4202 |
| 25-60m | 3,943 | £687,429,173.02 | 40.8811 |
| 60m+ | 4,645 | £918,741,040.44 | 47.8092 |
| **Total** | **11,035** | **£1,964,886,258.21** | |

**Derived columns** are byte-identical to the independent recompute
(`months_on_book` and `seasoning_segment` both `.all()` equal).

**Unexplained variance: 0.**

---

## 10. Provenance × seasoning cross-product

All four cells populated — the axes are genuinely orthogonal on this book, so
neither can be predicted from the other:

| | Front Book | Back Book |
|---|---|---|
| **Direct** | 927 loans · £147,961,196.69 · 35.6379 | 6,199 loans · £1,237,547,386.29 · 44.2761 |
| **Acquired** | 250 loans · £23,774,920.03 · 28.9101 | 3,659 loans · £555,602,755.20 · 43.2742 |

**250 Acquired + Front Book loans** and **6,199 Direct + Back Book loans** are the
direct empirical refutation of "back book = acquired".

Combined query truth — "the average LTV of the back book in the direct book" =
6,199 loans, £1,237,547,386.29, WA LTV 44.276076 — and the agent returns exactly
that, with **both** axes in the receipt.

---

## 11. Deterministic acceptance

`tests/test_p1j1_vintage_seasoning.py` — **47 tests, 47 passing**, covering all 15
required adversarial proofs:

| # | Proof | Test |
|---|---|---|
| 1-5 | back book / seasoned book / legacy book / newly originated / new origination are **never provenance** | `test_a_seasoning_phrase_is_never_provenance` (10 phrases) |
| 6-7 | acquired book / direct book (and every other provenance synonym) **still resolve** | `test_provenance_vocabulary_still_resolves` |
| — | the two vocabularies structurally cannot overlap | `test_the_two_vocabularies_do_not_overlap` |
| 8 | a five-year-old direct loan is Direct + Back Book | `test_a_seasoned_direct_loan_is_direct_and_back_book` |
| 9 | a recent acquired loan is Acquired + Front Book | `test_a_recent_acquired_loan_is_acquired_and_front_book` |
| 10 | historical snapshot uses seasoning **as at that date** | `test_seasoning_is_measured_against_the_reporting_date`, `test_a_historical_snapshot_carries_its_own_seasoning` |
| 11 | changing `front_book_max_months` changes the population, no code change | `test_changing_the_cutoff_changes_the_population_without_code_changes` |
| 12 | seasoning buckets reconcile exactly to the total | `test_seasoning_buckets_reconcile_to_the_total_population` |
| 13 | Front + Back reconcile exactly | `test_front_and_back_reconcile_to_the_governed_population` |
| 14 | no loan is both Front and Back | `test_no_loan_is_both_front_and_back` |
| 15 | provenance and seasoning combine, neither replacing the other | `test_provenance_and_seasoning_combine_in_one_query` |

Plus: role resolution (population vs grouping), the ambiguity refusals, the
run-rate non-regression, the three bank questions with truth reconciliation, and
receipt-distinguishability.

**Four pre-existing tests were updated, not weakened.** They asserted "this book
has no vintage, therefore refuse" — a premise P1J-1 deliberately changes. They now
assert the *enduring* invariant: a vintage question is **never answered by
sourcing**, and when it answers it must have answered on the vintage axis.

Targeted suites: `test_p1j1_vintage_seasoning` + `test_p0_cohort_identity` +
`test_p1g_measure_identity` + `test_p1i_scope_resolution` = **156 passed**.

### Semantic identity — the receipt

A reader can tell the four executions apart from the receipt alone:

```
provenance       Calculated: Weighted-average Current Loan To Value · Direct vs Acquired.
binary seasoning Calculated: Weighted-average Current LTV · grouped by Seasoning Segment · 2 groups
population       Calculated: Weighted-average Current LTV · Seasoning Segment = Back Book · 9,858 loans
vintage year     Calculated: Weighted-average Current LTV · grouped by Vintage · 13 groups
combined         Calculated: Weighted-average Current LTV · Seasoning Segment = Back Book · Source Portfolio in alp_origination
```

`SeasoningConfig.describe_segment` renders the governed boundary
(`Front Book (0-12 months)` / `Back Book (13+ months)`) so the configurable cutoff
is stateable rather than assumed.

---

## 12. Immutable 40-bank — before → after

Run unchanged, deterministic path, exact bank wording.

```
BEFORE 11/40   ->   AFTER 14/40
changed: B04 (False->True), B09 (False->True), B28 (False->True)
REGRESSED (was correct, now not): NONE
```

| Classification | Count | Detail |
|---|---|---|
| Correct | **14** | +B04, +B09, +B28 |
| Explicit partial | unchanged | A4, B22 as before |
| Safe refusal | 26 | all previously-refusing questions except the three |
| **Incorrect successful** | **0** | |
| **Silent semantic error** | **0** | two were *removed* (§8 line-coercion, §5 grouping-vs-population) |
| **Hard failure** | **0** | |

No unrelated question changed **at all** — not its `ok` flag and not its answer
text — so the +3 is genuine added answerability, not churn.

---

## 13. Full suite

`mi_agent/tests`, `mi_workflows`, `mi_agent_api/tests`, `trakt_core`, `tests` —
run against the final code with nothing else competing for the machine:

```
8645 passed, 30 skipped, 21 xfailed, 48 warnings, 6 subtests passed in 1715.85s (0:28:35)
```

**0 failed.** (P1I-A baseline was 8,587; the +58 are this phase's new tests.)

Six pre-existing tests had to be corrected on the way, across four files. Five
encoded the conflation or its arithmetic and are recorded in §11 and in the
commit messages; the sixth was a **governance violation this phase introduced** —
"seasoning" briefly became a synonym of both `months_on_book` and
`seasoning_bucket`, making resolution order-dependent.
`test_no_synonym_maps_to_two_fields` caught it, and `months_on_book` kept the
bare word.

That six pre-existing tests asserted the vintage/provenance conflation as
*expected behaviour* is itself a finding: the two axes were entangled in the
test suite as well as in the code.

One further failure was investigated and deliberately **not** changed:
`test_serving_parquet::test_the_serving_copy_is_materially_faster_and_smaller`
failed in a concurrent run. It is a timing test on shared hardware whose own
docstring calls the 2× threshold deliberately loose, it touches only parquet/CSV
reading, and in isolation it passes at 3.3×. It passed in another full run of the
same code. It was contention, not a regression — so the fix was to stop running
three heavy jobs at once, not to touch the test.

---

## 14. Genuine-LLM acceptance

Five runs per case through the real model, `zero_cost_first` forced off in the
harness so every question reaches the LLM. **Parser provenance is recorded per
run**, and a deterministic fallback is never labelled an LLM result.

| case | axis | correct | safe refusal | substitution | hard failure |
|---|---|---|---|---|---|
| back book | seasoning | 5 | 0 | 0 | 0 |
| seasoned book | seasoning | 5 | 0 | 0 | 0 |
| legacy book | seasoning | 5 | 0 | 0 | 0 |
| front book | seasoning | 5 | 0 | 0 | 0 |
| newly originated | seasoning | 5 | 0 | 0 | 0 |
| B04 (vintage comparison) | seasoning | 5 | 0 | 0 | 0 |
| B09 (vintages) | seasoning | 5 | 0 | 0 | 0 |
| B28 (by vintage) | seasoning | 5 | 0 | 0 | 0 |
| acquired book | provenance | 5 | 0 | 0 | 0 |
| direct book | provenance | 5 | 0 | 0 | 0 |
| A8 (provenance comparison) | provenance | 5 | 0 | 0 | 0 |
| combined (seasoning + provenance) | both | 5 | 0 | 0 | 0 |

```
parser provenance: {'llm': 55, None: 5}      llm calls: 55
SEMANTIC GATE: GREEN
```

**PROVENANCE_SUBSTITUTION = 0 · SEASONING_SUBSTITUTION = 0 · AXIS_DROPPED = 0 ·
HARD_FAILURE = 0.** The five `None`-provenance runs are the routed provenance
comparison (A8), which answers through `portfolio_risk_comparison` and bypasses
the parser by design — not a fallback.

### What the LLM run found that the deterministic bank did not

Three real defects, all fixed, all now covered by regressions:

1. **A dropped seasoning population degraded to a "partial".** "The average LTV
   of the back book in the direct book" returned a **count of the whole direct
   book** — 7,126 loans — marked `ok`, with the lost population mentioned only as
   a not-applied breakdown. Seasoning is now a **subject** facet: it refuses.
2. **A population honoured as a filter read as LOST**, because the reconciler only
   looked at group keys. The correct answer was being disclosed as a partial.
3. **An unhandled crash.** The model expressed "newly originated" as
   `origination_date >= "2024-01-01"`; the executor forced it through `float()`
   and raised `ValueError`, surfacing as *"The MI Agent could not complete this
   query."* Dates are now compared as dates, and a value that is neither numeric
   nor a date fails in a controlled way.

Two honest notes on the measurement itself:

* An intermediate re-run reported **GREEN with `llm calls: 0`** — the key had run
  out of credit and every question silently fell back to the deterministic
  parser. That green was meaningless and was **discarded, not reported**. It is
  why parser provenance is recorded per run rather than assumed.
* My first scorer counted the crash above as a *safe refusal* because `ok` was
  `False`. A crash and a refusal are not the same outcome; the scorer now
  separates them, which is what turned that case red and got it fixed.
* One scorer correction was made after seeing results: a **routed** provenance
  comparison carries no spec filter or dimension, so the original check could not
  see the axis and marked A8 `NO_AXIS`. That was a measurement bug — A8's answer
  ("Direct has higher observed Current Loan To Value than Acquired") is
  independently correct — and fixing the scorer is not the same as relaxing the
  standard. It is recorded here so the change is visible rather than silent.

---

## 15. Residual limitations

* **A combined two-axis COMPARISON refuses.** "How does the seasoned direct book
  compare with recent acquired loans?" names two provenance scopes *and* two
  seasoning segments; it reaches the provenance workflow and the cohort guard
  refuses it. That is the deliberate safe outcome — better a refusal than a
  silently dropped axis — but it is a real breadth limit. A governed two-axis
  comparison would be its own phase.
* **A combined FILTER + scope works** ("the back book in the direct book"), so the
  limitation is specific to comparing on both axes at once.
* **"Older vintages" and "new lending" deliberately do not select a population.**
  Both are genuinely ambiguous; they fail safe rather than guess.
* **The 2019 step** in the vintage series is a property of the synthetic fixture
  (origination volumes jump from ~330/yr to ~1,280/yr in 2019), not a defect.
* **`time_on_book_bucket` remains a quantile dimension** with different edges from
  the governed seasoning bands. Two similar-sounding dimensions now coexist;
  consolidating them was out of scope and would change existing pptx/stratification
  output.
* **Loans with no origination date** would derive a null segment and fall out of
  both populations. Every loan on this fixture has one, so the reconciliation is
  exact here; a book with gaps would need an explicit unknown-bucket policy.

---

## 16. Status and next breadth increment

**Recommended next increment (from P1J's ranking): P1J-2 — PROJECTION** (A7, B19).
It is the next-highest breadth-per-risk: `mi_agent_api/forecast_extrapolation.py`
already computes run-rate milestones including a £100m threshold, so it is pure
exposure with no new maths, and it serves a Treasury question asked constantly.

---

## 17. Gate

| criterion | required | measured |
|---|---|---|
| INCORRECT_SUCCESSFUL | 0 | **0** |
| SILENT_SEMANTIC_ERROR | 0 | **0** (two were removed) |
| HARD_FAILURE | 0 | **0** (one was found and fixed) |
| PROVENANCE_SUBSTITUTION | 0 | **0** (55 genuine LLM calls) |
| SEASONING_SUBSTITUTION | 0 | **0** |
| targeted bank | green | **53 / 53** |
| immutable 40-bank | no regression | **11/40 → 14/40**, nothing regressed |
| full suite | green | **8,645 passed, 0 failed** |
| truth reconciliation | exact | **0 unexplained variance** |

P1J-1 GOVERNED VINTAGE & SEASONING: PASS
