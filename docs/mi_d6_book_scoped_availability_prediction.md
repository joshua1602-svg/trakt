# D6 (B14) — "is this field available in the book being asked about"

Written before implementing. §2 is BASELINE measurement of `9bf7b9c`, taken
before the fix was designed.

Base: HEAD `9bf7b9c`; merge-base `4e051f3`; `4e051f3` and `28ece25` both
ancestors; clean tree.

---

## 1. THE SURFACE, AND THE BASELINE, AS SEQUENCED STEPS

B16a named its surface in advance and still predicted the wrong differ result,
because the cases postdated the baseline being compared against. That is a step,
not a correction, and it is executed here in order:

| # | step | result |
|---|---|---|
| 1 | **Name the surface.** The defect changes which route answers, which facet status is stamped, and the SENTENCE the reader sees. The routed surface asserts the first two; nothing asserted the third on this path. | routed surface + `answer_diff` |
| 2 | **Put the cases in, declared failing.** `rt_014` (from D7), `rt_017`, `rt_018` (control), `rt_019` (B19). | 21 of 21, 3 declared expected-to-fail |
| 3 | **Extend the differ so the sentence is measurable.** `answer_diff` gains a fifth surface — the routed bank's own questions, as text. These are the ONLY questions in any surface that reach a non-funded view. | 5 surfaces |
| 4 | **RE-RECORD the baseline, with the cases in and before the fix.** | **718 answers across 5 surfaces** → `answer_baseline_d6.json` |
| 5 | pre-register (this document) | — |
| 6 | implement, and verify the owner list against each site | — |
| 7 | measure against the baseline from step 4 | — |

## 2. Baseline, measured

### 2.1 The frame is chosen by a keyword scan, before the question is parsed

```python
def resolve_active_view(question, dataset_context):
    q = (question or "").lower()
    if "forecast" in q:  return "forecast"
    if "pipeline" in q:  return "pipeline"
    if "funded" in q:    return "funded"
```

The word *forecast* **anywhere** in the question selects the forecast view. This
runs before parsing, before routing, before anything knows what the question is
about. The census called it "a routing guess made before the question is
answered"; it is narrower and blunter than that — it is a substring test.

### 2.2 The forecast view is a 12-column projection of a 76-column book

```
funded    rows=11,035   cols=76     seasoning_segment ✓  vintage_year ✓  account_status ✓  amortisation_type ✓
forecast  rows=11,035   cols=12     seasoning_segment ✗  vintage_year ✗  account_status ✗  amortisation_type ✗
                                    collateral_geography ✓
```

`build_forecast_view_frame` copies twelve `_SHARED_DIMS` and drops the other
sixty-four. Every availability check downstream reads those twelve.

### 2.3 What a reader is told, and the control that proves it wrong

| question | field | in the frame | in the book | the receipt says |
|---|---|---|---|---|
| What is the forecast run rate **for the front book**? | `seasoning_segment` | ✗ | ✓ | *"field is unavailable in this dataset"* — **false** |
| What is the forecast run rate **by vintage**? | `vintage_year` | ✗ | ✓ | *"field is unavailable in this dataset"* — **false** |
| What is the forecast run rate **by region**? | `collateral_geography` | ✓ | ✓ | *"this answer covers the whole population"* — **true** |

**The third is the control and it settles what the correct outcome is.** The same
question shape, on a field the projection happens to carry, already produces the
honest sentence. Both refuse; only one of them is honest about why. So the fix is
not "stop refusing" — it is **read the book, and let the field fall through to
LOST**, which is what a field present in the book but absent from the frame
actually is.

### 2.4 The corpora cannot reach it, and the reason is instructive

```
questions by view: funded 610 · pipeline 60 · forecast 27
non-funded-view questions claiming a field is unavailable: 0
```

Eighty-seven questions take a projected view and **not one** names a field the
projection drops. Reading the 27 forecast questions shows why: they are
*forecast balance by region · by broker · by LTV bucket · by expected completion
month · by stage* — built from the fields the forecast frame carries.

**The family was enumerated from the projection, so it cannot exercise the
projection's gap.** That is the corpus limitation this work order asks to be
recorded, arriving again and unprompted.

## 3. The class, and the illustration

**The class:** *every reader that asks whether a field is available asks it of
the BOOK being reported on, not of whichever projection a keyword scan caused to
be loaded. A field the book carries and the loaded frame does not is LOST — the
request did not reach execution — never UNAVAILABLE, which says the book does not
report it.*

**The illustration:** `rt_014`, `rt_017` and the `rt_018` control. All
constructed; §2.4 measured that no corpus question reaches the class.

**Explicitly NOT in the class, and pinned separately as B19:** *"What is the
forecast run rate for active loans?"* answers `ok` with the **whole-book**
run rate — £16.3m/month, byte-identical to the unqualified question — and **no
facet at all**. B16a's value allowlist is built from the loaded frame, so a
projection with no `account_status` column recognises no value of it and raises
no narrowing. Same root, worse symptom, and a different change: D6 scopes the
check to the book's **schema**; recognising the value needs the book's **rows**.
`rt_019` pins it. **Not fixed here.**

## 4. The owners — provisional, to be VERIFIED during implementation

The work order is explicit that this list is worth something made against each
piece of the design and has now missed twice when made against its headline.
So this is a starting list, not a finding, and the implementation report will
state what changed:

| # | site | what it reads today |
|---|---|---|
| 1 | `reconcile_facets` `KIND_GROUPING` | `canonical not in columns` → UNAVAILABLE |
| 2 | `reconcile_routed_facets` `KIND_GROUPING` | `not any(c in columns)` → UNAVAILABLE |
| 3 | `reconcile_routed_facets` `KIND_GEOGRAPHIC_SCOPE` | "no geographic field in this dataset carries that value" |
| 4 | `reconcile_population(dataset_columns=…)` | → UNSUPPORTED |
| 5 | `dimension_role` source 4 | "the book cannot express it" → AXIS |
| 6 | `geographic_values(frame, …)` | `column not in frame.columns` |
| 7 | `dimension_values(frame, …)` | `column not in frame.columns` |
| 8 | `seasoning.resolve_population_predicate(text, columns)` | returns None when a predicate field is absent |
| 9 | `requested_dimension_terms(available_columns=…)` | the availability-filtered second resolution |
| 10 | `_deterministic_parse(available_columns=…)` | slot assignment |
| 11 | `mi_calibration._absent_required_fields` | the bank's own check |

Sites **6 and 7 read VALUES, not schema**, and must keep reading the loaded
frame: a value can only be recognised in rows that exist. That is the B19
boundary, and mistaking it for a schema check is how D6 would silently become
B19 and take the wrong-number class with it.

## 5. The rule

**The book's columns travel with the frame.**

`_resolve_query_frame` already HAS the funded book in hand when it builds a
derived view — `build_forecast_view_frame(funded_df, pipeline_df)` takes it as an
argument and drops sixty-four of its columns. So the schema is loaded and
discarded, which is the carriage pattern for the sixth time. The frame is stamped
with `attrs["book_columns"]` instead, and one accessor — `book_columns(frame)` —
is what every schema check consults. On the funded view the two are the same
list, so nothing changes there.

Then, at each schema site: a field in `book_columns` is **not** UNAVAILABLE. It
falls through to whatever the branch says next, which for a grouping is LOST and
for a population is LOST — both true, both blocking, neither claiming the book
does not report it.

## 6. Pre-registered prediction

### 6.1 What moves

**Two answers, and both only in their REASON.**

| id | today | predicted |
|---|---|---|
| `rt_014` | `grouping_dimension seasoning_segment UNAVAILABLE`, *"field is unavailable in this dataset"* | `... LOST`, *"this answer covers the whole population…"* |
| `rt_017` | `grouping_dimension vintage_year UNAVAILABLE` | `... LOST` |

Both already refuse and both continue to refuse. **No verdict changes, no number
changes, no route changes.** `answer_diff`: **716 identical, 2 moved**, both on
`routed_surface`.

`rt_014` and `rt_017` flip to passing; `rt_019` stays declared-failing.

### 6.2 What must not move

1. **`rt_018` keeps its `lost` status.** It is the control; if it moves the change
   is not reading the book, it is blanketing.
2. **`rt_019` keeps failing.** A fix that closed B19 as a side effect would mean
   the schema check had reached into values.
3. **No funded-view answer moves at all** — `book_columns` equals `frame.columns`
   there. 610 of 697 corpus questions and both banks.
4. **The seasoning families stay at their by-name counts**, both books.
5. **Robustness stays `32/10/2`** on both books; calibration stays `259/259`,
   0 hard fails, 0 known gaps.
6. **No lexical decision moves.** 693 of 693.
7. **A field genuinely absent from the BOOK is still UNAVAILABLE** — the
   `broker_channel` family, which this tape does not carry.

### 6.3 Stop conditions

* `rt_018` moving;
* `rt_019` passing;
* any funded-view answer moving;
* any verdict, number or route changing anywhere;
* a field absent from the book being reported as anything but unavailable;
* any seasoning family count moving.

### 6.4 Acceptance

* one accessor, consulted by every SCHEMA site; §4 verified site by site during
  implementation and the differences reported;
* the value sites (6, 7) demonstrably still read the loaded frame;
* the book's columns are carried, not re-loaded;
* all five surfaces, deterministic arm, both books; seasoning by name;
* `rt_014` and `rt_017` flip; `rt_018` and `rt_019` hold.
