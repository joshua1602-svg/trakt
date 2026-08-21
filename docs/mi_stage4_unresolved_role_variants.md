# The unresolved-role default — three variants, measured side by side

Base: merge-base `4e051f3` on `claude/mi-analytical-capability-layer-vlkjfw`;
release candidate `28ece25`; both ancestors of the branch head. Deterministic arm
of both surfaces throughout, per standing condition 1 as amended.

**No variant is chosen here.** This records what each does.

---

## 0. Two corrections to the premise, before any numbers

### 0.1 Governed populations ARE in the unresolved set

The request asked me to confirm, in all three variants, that "governed
populations are no longer misread as unresolved, and the seasoning families are
not in the unresolved set". Measured, the second half is false, and my own
earlier commit wording said otherwise and was wrong.

Of the 64 unresolved facets the detector produces across the corpus, **14 are
governed lending windows**, and every one of them is a seasoning-family
robustness question — 7 questions across 2 books:

| id | question | field | governed as |
|---|---|---|---|
| Q1.1 | How has the profile of our new lending changed over the last few months? | `seasoning_segment` | `months_on_book le 1` |
| Q7.1 | How does the front book compare with our older lending from a risk perspective? | `seasoning_segment` | `seasoning_segment = Front Book` |
| Q7.3 | How different is the risk profile of recent originations versus the back book? | `seasoning_segment` | `seasoning_segment = Front Book` |
| Q7.4 | Compare the credit profile of the front book with our seasoned loans. | `seasoning_segment` | `seasoning_segment = Front Book` |
| Q8.1 | How has lending to the front book changed compared with the back book? | `seasoning_segment` | `seasoning_segment = Front Book` |
| Q8.3 | Are the front book and the back book balances developing differently over time? | `seasoning_segment` | `seasoning_segment = Front Book` |
| Q8.4 | Compare how the the front book and the back book books have changed over the last few months. | `seasoning_segment` | `seasoning_segment = Front Book` |

Resolved through `_governed_population_predicates` — the same route the
population check uses since the governed comparison went in, keyed on the
facet's wording rather than its field key, so this instrument cannot disagree
with the code it measures.

### 0.2 Variant 1 does not return a whole-book number with a note

The request frames variant 1 as "whole-book number with a disclosure attached",
and frames the choice as "one risks a wrong number in front of a CFO; the other
costs a round trip". That asymmetry does not obtain in this tree.

`SHAPE_FACETS` still names `KIND_GROUPING` and `assess` still has a
`VERDICT_PARTIAL` branch for it, which reads like a disclosure path. It is
unreachable. `_blocks` returns `True` for **any** non-applied grouping, and the
blocking test runs before the partial test — the honour-or-clarify rule was
extended to groupings and populations in Tranche D. Measured across 252
calibration cases and 88 robustness runs: **zero `partial` verdicts, in every
variant.**

So today an unresolved dimension already refuses. The measured choice is
between a **refusal** and a **question**, not between a wrong number and a
round trip. `question_interpretation/tests/test_unresolved_role_variants.py`
pins this, with the can-fail half asserting the partial branch still exists so
the first test cannot pass vacuously.

---

## 1. What was measured

`mi_agent.execution_receipt.UNRESOLVED_ROLE_DEFAULT` — a measurement switch,
default `"grouping"`, i.e. current behaviour. Where no source supplies a role —
the field is in neither `spec.filters` nor on an axis — it decides the facet's
fate:

| variant | what `_split_named_dimension_roles` does |
|---|---|
| `grouping` | leaves the facet alone (current) |
| `population` | `KIND_POPULATION`, relabelled `the population {field}: {wording}` |
| `clarify` | `KIND_UNRESOLVED_ROLE`; `assess` asks before it refuses |
| `population_bare` | **not a candidate** — see §4 |

A fourth arm exists only to make the recurrence claim falsifiable.

---

## 2. Side by side

### 2.1 Robustness bank — deterministic arm, both books, 44 × 2

| variant | alderbridge | kestrelmoor | seasoning family | both books agree |
|---|---|---|---|---|
| grouping | 32 C / 10 SR / 2 CWDL | 32 / 10 / 2 | Q1 4, Q7 4, Q8 12 — all CORRECT | 44/44 |
| population | 32 C / 10 SR / 2 CWDL | 32 / 10 / 2 | Q1 4, Q7 4, Q8 12 — all CORRECT | 44/44 |
| clarify | 32 C / 10 SR / 2 CWDL | 32 / 10 / 2 | Q1 4, Q7 4, Q8 12 — all CORRECT | 44/44 |

Identical. **The seasoning families hold, 20 of 20 by name, in all three.**

### 2.2 Answer text — 340 answers, byte-identical diffing

| variant | identical | moved |
|---|---|---|
| grouping | 340 | 0 |
| population | 337 | `filt_129`, `filt_135`, `filt_151` |
| clarify | 337 | `filt_129`, `filt_135`, `filt_151` |

No robustness answer moves in any variant. All movement is three calibration
cases, all `borrower_type`.

> An earlier pass of this table reported 3 moved for the calibration half and 0
> for the robustness half because `answer_diff` and
> `run_robustness_deterministic` both re-invoke themselves per book as a
> subprocess and forwarded only `--only-book` / `--book`, not the variant. Those
> runs measured the default and labelled it the variant. Both now forward every
> argument that changes what is measured. See §5.

### 2.3 Verdicts — 252 calibration cases

| variant | ok | no guard | refuse | clarify | partial |
|---|---|---|---|---|---|
| grouping | 136 | 112 | 4 | – | 0 |
| population | 136 | 112 | 4 | – | 0 |
| clarify | 136 | 112 | 1 | 3 | 0 |

`population` moves **no verdict at all**. It changes three facets' kind and
therefore the wording of three refusals; the refusals stay refusals.

### 2.4 The three cases, in full

| case | grouping | population | clarify |
|---|---|---|---|
| `filt_129` | refuse — `grouping_dimension/unavailable` | refuse — `row_population/lost` | **clarify** — `unresolved_role/lost` |
| `filt_135` | refuse | refuse | **clarify** |
| `filt_151` | refuse | refuse | **clarify** |

`filt_135`, "WA LTV by region for joint borrowers":

* **grouping** — *I understood that you asked for joint borrower, but that could
  not be applied to the calculation (joint borrower (Borrower Type) — field is
  unavailable in this dataset). I have not substituted a broader figure.*
* **population** — *I understood that you asked for the population
  borrower_type: joint borrower, but that could not be applied to the
  calculation (the population borrower_type: joint borrower (Borrower Type)). I
  have not substituted a broader figure.*
* **clarify** — *I could not tell how you meant joint borrower. Did you want the
  book split by it, or narrowed to one value of it? I have not answered over the
  whole book in the meantime.*

Two things to note about `population` that are not visible in the counts. It
leaks the field key into user-facing prose, and the status moves from
`unavailable` to `lost`, which drops "field is unavailable in this dataset" —
the only part of the sentence that told the reader *why*. Both are properties of
this implementation rather than of the direction, but neither is free to fix:
the label must carry the field name or B5 becomes reachable (§3).

### 2.5 B5, under every variant

| variant | population facets built | carrying a field key | label omits its field |
|---|---|---|---|
| grouping | 83 | 83 | **0** |
| population | 147 | 147 | **0** |
| clarify | 83 | 83 | **0** |
| population_bare | 147 | 147 | **0** |

B5 unreachable in all four.

---

## 3. What moves, what blocks, what clarifies — per variant

**`grouping` (current).** Moves nothing. Blocks 4 calibration cases, 3 of them
on an unresolved role. Clarifies nothing. Seasoning families hold. The reader of
`filt_135` is told the question cannot be answered as asked, and told which term
was the problem. They are *not* told that the system could not tell whether they
meant a breakdown or a filter — the refusal presents a single confident reading
of their sentence and rejects it.

**`population`.** Moves 3 facets from `grouping_dimension` to `row_population`
and 3 answers' wording. Blocks the same 4 cases. Clarifies nothing. Seasoning
families hold. Against the current default it buys a more accurate facet kind in
the receipt and costs the "field is unavailable" reason and a field key in the
prose. On this corpus it does not change what any reader can or cannot get.

**`clarify`.** Moves 3 facets to `unresolved_role` and converts 3 refusals into
3 questions. Blocks 1 case (unrelated to roles). Clarifies 3. Seasoning families
hold. It is the only variant that changes what the reader can do next: a refusal
ends the exchange, a question continues it.

---

## 4. Can 32c263a recur? Measured, not assumed — and the honest answer

32c263a assigned `KIND_GROUPING` only where the words justified it and let
everything else fall to `POPULATION`, the blocking side. The over-assignment cost
160 runs, concentrated in Q1.1 and the Q7/Q8 seasoning family.

**The premise that rules it out is false (§0.1): 14 governed lending windows sit
in the unresolved set, and they are exactly the questions 32c263a broke.** So
the reason it cannot recur has to be found somewhere other than where the
request expected it.

Measured, it is this: **of the 64 unresolved facets the detector produces, 3
reach `_split_named_dimension_roles` on the serving path.** All three are
`borrower_type`, on `filt_129` / `filt_135` / `filt_151`. The other 61 —
including all 14 governed windows — never arrive. All 14 route through
`analytical_composition`, which never calls the point-in-time executor and
therefore never calls `reconcile_facets`, where the split lives.

The recurrence is out of reach, not out of the set. That is a weaker guarantee
than "governed populations are no longer misread", and it is contingent on
routing rather than on the governed comparison: a future change that sent one of
those questions down the point-in-time path would put a governed window in front
of the variant with nothing measured about it.

### 4.1 The falsifying arm, and what it failed to show

`population_bare` is `population` with the facet relabelled to the bare field
name, destroying the wording `_governed_population_predicates` reads. It was
built to *produce* the 32c263a failure, so that ruling the failure out would mean
something.

**It did not produce it.** Both books, all 14 governed windows, still CORRECT;
seasoning family still 20 of 20; the same 3 answers move and no others. That is
not evidence that the wording does not matter — it is a second, independent
measurement of the reach finding above. The facets never get there, so nothing
done to their labels can break them.

The wording mechanism itself is real and is proven at unit level, where the
corpus cannot exercise it:
`test_population_keeps_the_wording_the_governed_check_reads` shows
`_governed_population_predicates` resolving `front book` under `population` and
returning nothing under `population_bare`. Generated coverage, because the
corpus has no construct that reaches it.

---

## 5. Two measurement limitations, to travel with any number quoted from here

**L1 — the split is inert on this tape.** The alderbridge book carries no
`borrower_type` column, so the parser correctly drops that filter and the
role split proper reclassifies nothing on the production parse. Parsed without
column filtering, the same corpus moves 9 facets, all `borrower_type`, on the
questions the inventory named. Every end-to-end figure above therefore rests on
the unfiltered parse for the split's own behaviour and on 3 cases for the
variant's. Do not quote the unfiltered parse-level figures (64 / 55 unresolved
facets, 9 moved) as production behaviour.

**L2 — the B5 scanner watched the wrong facets.** It was built to guard the
label construction that the role split changes, and it originally scanned only
detection-time facets while the split happens at reconcile — it would have
missed precisely the facets it exists to guard. This is recorded in the standing
rules as a **pattern**, not an incident: it is the seventh instance in this
programme of an instrument carrying the defect it was built to find. Two more
instances surfaced during this measurement (§2.2), both of the same shape.

---

## 6. Reproducing

```
python -m question_interpretation.unresolved_role_variants --json out.json
python -m question_interpretation.run_robustness_deterministic --all-books --unresolved-role VARIANT
python -m question_interpretation.answer_diff --unresolved-role VARIANT --against question_interpretation/answer_baseline_stage1.json
python -m question_interpretation.b5_reachability --unresolved-role VARIANT
python -m pytest question_interpretation/tests/test_unresolved_role_variants.py
```

The switch defaults to `"grouping"`. With it untouched every surface is
byte-identical to before it existed — 340 of 340 answers, both books 44/44,
verdicts unchanged. Two of the four arms are to be deleted once one is chosen;
this must not become a deployment knob.

---

## 7. Stopping condition hit — a regression on HEAD, found while measuring

Not part of the variant question, and reported rather than absorbed, per the
standing instruction.

**`e35a01b` (Stage 4, 2/n — the role split) breaks 5 tests in
`tests/test_p1j1_vintage_seasoning.py`.** They pass at `43f264a`, the commit
before it. My report on that commit said nothing moved; the two surfaces and the
calibration bank all agreed, and all three were blind to this.

### The mechanism

    "What is the balance of newly originated loans?"

    split in :  ('grouping_dimension', 'newly originated',  'seasoning_segment')
    split out:  ('row_population',     'the population seasoning_segment = Front Book')
    receipt  :  row_population / LOST      →      REFUSE

`reconcile_facets`'s `KIND_POPULATION` branch upgrades a facet to `APPLIED` only
when the route is analytical; otherwise it "keeps the status the population
ledger already stamped". On the point-in-time path the only thing that stamps a
population is `reconcile_population`, and that runs over the facets
`population_facets(spec)` raises — a list the split-created facet was never in.

So the split takes a facet `reconcile_facets` *can* stamp (a grouping, resolved
against `group_field_keys`) and turns it into one that nothing on that path can
stamp. It is therefore LOST, and a lost population blocks.

### Why every surface missed it

The condition needs a field that is both in `spec.filters` and raised as a
dimension facet. On the calibration and robustness corpora that is
`borrower_type` alone, and the alderbridge tape does not carry it — the same
inertness recorded as L1. The `p1j1` fixture book carries `seasoning_segment`,
so it fires there. L1 was reported as a limitation on the *strength* of the
Stage 4 evidence; it turns out to have been hiding a defect, not merely
weakening a proof.

### What this does to the numbers above

The variant measurement was taken on this tree. It is unaffected in substance —
the three variants differ only on unresolved facets, and this defect is in the
*resolved* filter branch, which all four arms share — but every figure in §2 was
measured on a tree with a live refusal regression in it, and should be re-taken
after the fix.

### Fix shape, not applied

The split must not create a population that no ledger owns. Either the
point-in-time path gains the same `population_facets` / `reconcile_population`
wiring the routed path has, or the split defers to the ledger where one already
covers the field. The second is smaller; the first is where the duplication
actually is. Both change a grading path, which this programme does not do
mid-measurement without a decision.

Separately, `tests/test_p1l_population_propagation.py` has 2 failures that
predate all of this — they are live at the release candidate `28ece25` and are
stale assertions from before Tranche D moved populations to the blocking side.
Not caused here, and not fixed here.
