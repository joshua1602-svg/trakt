# B16a — a narrowing the sentence marked and execution did not apply

Written before implementing. §1 and §2 are BASELINE measurement of `f6d10b7`,
taken before the fix was designed.

Base: HEAD `f6d10b7`; merge-base `4e051f3`; `4e051f3` and `28ece25` both
ancestors; clean tree.

---

## 1. THE SURFACE, NAMED FIRST — and it fails today

Carried into this work order because D7 moved five answers while all three
standing surfaces reported nothing, the fourth occurrence of a change measured
by instruments that cannot see it.

### 1.1 Where the defect lives

**The point-in-time path.** Every live instance returns `route=None`: no route
claims *"what is the balance where account status is active"*. So the pattern
the work order names — *any change to the routed path is invisible to surfaces
built around the point-in-time path* — **does not apply here, and its converse
does**: this is a point-in-time defect, and the routed surface added in D7 is the
wrong instrument for it.

### 1.2 No standing corpus contains the construct

Measured across all 693 questions: **15** named dimensions preceded by a
selector preposition and not by `by`. Every one is already accounted for:

| group | n | what happens today |
|---|---|---|
| *"for joint borrowers"* / *"for single borrowers"* / *"for broker Alpha"* | 9 | already refuses — `borrower_type` and `broker_channel` are not columns of this book, so the facet is UNAVAILABLE and honour-or-clarify blocks. **No wrong number.** |
| *"of the front book"* / *"of recent originations"* | 6 | already correct — the seasoning owner (`7c46f81`) governs, and `analytical_composition` answers |

**Zero live wrong numbers on the corpora.** And the narrower construction — a
condition opener, the dimension, then a copula and a non-numeric operand
(*"where account status is active"*) — occurs **0 times in 693**.

### 1.3 The surface that will show it: the CALIBRATION BANK

It is the only surface that grades **numbers**, and it is point-in-time, which is
where the defect is. Four cases are added **in this commit, before the fix**, and
the bank already carries the `rt_013` pattern under the name `known_gap` — it
reports `known_gaps_xfailed` separately from hard failures, so a declared defect
is loud rather than quiet, and flips to `known_gaps_passing` when fixed.

```
before:  generated 249/249; curated 255/255 held, 0 hard fails, 0 known gaps
after adding the cases:
         generated 249/249; curated 257/259 held, 0 hard fails, 2 KNOWN GAPS
```

| case | states | today |
|---|---|---|
| `b16_001` What is the balance where account status is active? | `refuse` | **fails** — answers, whole book grouped by status, breakdown certified |
| `b16_002` What is the balance for interest roll-up loans? | `refuse` | **fails** — answers, whole book, **no facet at all** |
| `b16_003` What is the balance by account status? | `answer`, dimension applied | passes, and must keep passing |
| `b16_004` What is the balance of the front book? | `answer`, `seasoning_segment` filter | passes, and must keep passing |

`b16_003` and `b16_004` are the can-fails, and they are not decoration.
`b16_003` breaks if the fix refuses every question naming a dimension;
`b16_004` breaks if it reaches past the narrowings somebody already resolved —
which would undo `7c46f81`.

**Secondary instruments**, both of which will also move once the bank cases
exist: `answer_diff`'s calibration surface feeds from this bank, so the two
refusals become byte-diff movements; and `rt_015` on the routed surface already
fails today for the same reason.

---

## 2. A correction to my own framing of B16, before anything is built on it

The D2 and D7 documents both say:

> `lexical.is_filter_subject` is the declared lexical owner of "this mention is a
> selector", and nothing reads it.

**The first half is wrong in a way that matters.** `is_filter_subject` is an
owner of "this mention is a selector" **for numeric predicates only** — both its
patterns require a comparator or a `<>=` symbol or a leading digit. So does
`condition_cut`, which needs a digit after the opener before it will cut.
Measured:

```
"how many loans have LTV above 50%"                is_filter_subject(ltv)            = True
"balance where LTV above 50%"                      is_filter_subject(ltv)            = True
"What is the balance where account status is active?"
                                                   is_filter_subject(account status) = False
                                                   condition_cut                     = None
"total balance where repayment type is interest roll-up"
                                                   is_filter_subject(repayment type) = False
"balance for joint borrowers"                      is_filter_subject(joint)          = False
"show loans in the South East"                     is_filter_subject(south east)     = False
```

**The lexical layer's predicate machinery is entirely numeric, and the numeric
cases are exactly the ones the parser already resolves** — four of the five
fields that ever reach `spec.filters` are numeric bounds on measures.

So B16a as scoped — *"read the existing mark"* — **is not implementable as
stated. The mark does not exist.** The categorical selector has **zero** owners,
not one unread owner.

That changes the shape and is recorded here rather than discovered mid-fix.

---

## 3. The class, and the illustration

**The class:** *a dimension the sentence used to SELECT rows, which no source
resolved into a predicate, must be recorded as a narrowing that was requested and
lost — never answered over the whole book, and never asserted as a breakdown.*

**The illustration:** the four bank cases. All four are **constructed**, and that
is itself the finding: 693 corpus questions contain no live instance. Per the
standing rule, movement outside the four is expected and is not a stop condition;
movement outside the class is.

**Two shapes inside the class, and the second is the harder one:**

| shape | example | today |
|---|---|---|
| the dimension term IS raised | *balance where **account status** is active* | answered as a breakdown, `grouping_dimension applied` — a wrong claim to catch |
| the dimension term is NOT raised | *balance for **interest roll-up** loans* | answered over the whole book with **no facet at all** — nothing to catch |

A fix that only repairs the first satisfies `b16_001` and leaves `b16_002`.

---

## 4. The rule

Three parts, and the first is the one that needs care.

### 4.1 One lexical owner for a decision that has none

`lexical.selector_mark(text, start, end)` — *does the sentence use this mention
to select rows?* A **new decision with exactly one owner**, not a second reader
of an existing one. `is_filter_subject` keeps the numeric predicate and is not
touched; `condition_cut` and therefore `subject_side` are not touched, because
`subject_side` is the byte-identical Stage 3 conversion and widening it would
move every `where`-question's measure slot.

The construction, deliberately narrow:

* a condition opener (`where`, `with`, `for`, `whose`, `having`, `of`)
  immediately before the mention, **or** the mention immediately followed by a
  copula (`is`, `are`, `=`, `equals`) and a non-numeric operand;
* and **not** preceded by a grouping marker (`by`, `per`, `across`, `split by`).

Narrow because the cost of a false positive is a refused answer to a question
that was fine, which is `32c263a`'s cost.

### 4.2 The role owner consumes it — source 3's other half

`dimension_role` already reserves this. Its source 3 answers AXIS via
`grouping_cut`, and its docstring records the FILTER half as deliberately
unimplemented because *"a population facet must carry a resolved predicate"*.
That reasoning stands: the owner returns a **third answer**, not FILTER —
`ROLE_LOST_NARROWING`, meaning *the sentence selected, and no source resolved a
value*.

### 4.3 A facet kind that records the loss

`KIND_LOST_NARROWING` — LOST at construction, like `KIND_UNRESOLVED_MEASURE`,
carrying the field name and no predicate. It blocks, so honour-or-clarify
refuses and no whole-book figure stands in for the narrowed one. It carries **no
value**, so `_analytical_population_satisfies` never sees it and B5 is not
reopened.

Registered in `RECLASSIFICATION_TARGETS`, so `test_reclassification_targets.py`
demands a receiver — the closed class doing its job on the third new kind since
it was built.

**No parser change. B16b stays behind D6 and B1**, as scoped: resolving the value
needs the allowlist (B1) and the book-scoped availability check (D6), and folding
it in here would multiply the fabricated-binding class.

---

## 5. Every place the owner's answer arrives — and what was already deriving it

| # | site | derives it today? | after |
|---|---|---|---|
| 1 | `lexical.is_filter_subject` | owns the **numeric** selector mark | untouched. The new owner is a different decision, not a second reader of this one |
| 2 | `lexical.condition_cut` → `subject_side` → `answer_type.subject_side` | owns where a condition clause opens, numerically | **untouched**, deliberately. Widening it moves the Stage 3 byte-identical conversion |
| 3 | `dimension_role` | owns axis-or-filter; its source 3 has a documented hole exactly here | consumes the new owner; gains a third answer |
| 4 | `_split_named_dimension_roles` | consumes `dimension_role` | consumes one more answer from it; no new decision |
| 5 | `requested_dimension_terms` | raises every named dimension as a grouping | unchanged — the role owner reclassifies downstream, as it already does for populations |
| 6 | `seasoning.resolve_population_predicate` | **owns** the seasoning vocabulary and RESOLVES it | the new owner must never reach a phrase it took. `b16_004` is the test |
| 7 | `_deterministic_parse` filter slots | resolves numeric bounds and one categorical geography | unchanged. Where it resolves, `dimension_role` source 1 wins before source 3 is consulted |
| 8 | `assess` / `_blocks` | consume statuses | gains one blocking kind; `RECLASSIFICATION_TARGETS` is what requires the receiver |
| 9 | `reconcile_facets` / `reconcile_routed_facets` | stamp statuses | the kind is LOST at construction with no execution evidence to weigh — the `KIND_UNRESOLVED_MEASURE` precedent, and the registry records the exemption |

**Nothing else decides "did the sentence select on this mention".** Nothing can:
§2 measured that the decision has no owner at all today.

---

## 6. Pre-registered prediction

### 6.1 What moves

* **`b16_001` and `b16_002` flip** from `known_gaps_xfailed` to
  `known_gaps_passing`. Bank: `259/259 held, 0 hard fails, 2 known gaps passing`.
* **`rt_015` flips** to passing on the routed surface.
* **`answer_diff` moves exactly 2**, both on the `calibration_bank` surface,
  both `ok: True -> False`.
* **Nothing else.**

### 6.2 What must not move

1. **`b16_003` and `b16_004` keep passing.** Either failing stops the work.
2. **The seasoning families stay at their by-name counts**, both books —
   `Q1 4, Q7 4, Q8 12`, all CORRECT. This is the vocabulary `32c263a` broke.
3. **Robustness stays `32/10/2`, both books.**
4. **The other 255 bank cases stay held**, 0 hard fails.
5. **No lexical decision moves.** 693 of 693. The new owner is additive and
   `lexical_decisions` records the existing consumers; if `subject_side` or
   `requested_unit` moves, the change reached into an owner it must not touch.
6. **`service_path` and both robustness surfaces stay byte-identical** — 691 of
   693 in the differ, with only the two calibration cases moving.
7. **The stamping matrix stays at 0 live holes** and
   `RECLASSIFICATION_TARGETS` stays closed.

### 6.3 Stop conditions

* `b16_003` or `b16_004` failing;
* any seasoning family count moving;
* any answer moving outside the two predicted;
* any lexical decision moving;
* a live hole in the stamping matrix, or a facet reaching a reconciler with no
  branch to receive it;
* the new owner firing on a phrase `resolve_population_predicate` already took.

### 6.4 Acceptance

* one owner for a decision that had none, and §5 confirmed line by line;
* `is_filter_subject`, `condition_cut` and `subject_side` **unchanged** —
  asserted by the lexical diff, not by inspection;
* both shapes closed, the raised and the unraised;
* all four surfaces, deterministic arm, both books; seasoning by name;
* the new kind registered and its receiver proven by the closed-class test.

---

## 7. The carriage pattern, recorded as instructed

Third instance of **correct information produced and then discarded**:

1. **Stage 5** — the time grain was read from the question and dropped before
   anything could act on it;
2. **Stage 2** — the parser's filter offsets were computed and discarded, never
   reaching the spec (still open as B0);
3. **D7** — `concentration_analysis` published `workflow.dimension_results`,
   the exact canonical field keys it grouped by, on every envelope, and the
   receipt reader ignored it and guessed from a display column called
   `category`.

In all three the fix was **a reader, not a producer**, and in all three the
instinct was to add a producer. Going into the standing rules as:

> **Before adding a producer, check whether the answer is already being published
> and merely unread.**

**Applied to B16a, and it changes nothing here** — §2 measured that the
categorical selector mark is produced by nothing, which is why this commit adds
an owner rather than a reader. The check is recorded as performed, with its
result, rather than assumed.

---

# Amendment, written before implementing and after measuring

§4's design does not survive contact with the measurement. Three things were
found in order, each one changing the design, and they are recorded here rather
than discovered inside the fix. **§6's prediction is unchanged** — the same two
answers move — so this amends the mechanism, not the claim.

## A1. The mechanism already exists, for exactly one dimension

`funded_filtered_qa_018` — *"What is the balance of South East loans with LTV
above 50%?"* — already refuses, with `geographic_scope collateral_geography
LOST`. Reading that path:

* `geographic_values(frame, semantics)` builds `{value → field key}` **from the
  loaded book**, low-cardinality dimension columns only — the profiled allowlist
  B1 wants, already built;
* `_detect_geographic_scope` raises a facet for a named value;
* `KIND_GEOGRAPHIC_SCOPE` is in `NUMBER_OR_SUBJECT_FACETS`, so it **blocks**;
* both reconcilers have branches for it.

**Every part of B16a exists and is restricted to geography by one `if` in
`geographic_values`.** That is a fourth instance of the carriage pattern in a
variant the standing rule did not name: not *produced and unread*, but **built
for one field when it generalises to all**. The rule is amended to say so.

**And it settles comparisons without any lexical rule.** *"How has lending to
South East changed compared with London"* raises **two** scope facets and both
are **APPLIED**, because `analytical_composition` declares `narrowedTo` for both
sides. Execution proves the comparison; nothing has to recognise it in the
sentence. That is the contract working, and it is a better answer than the
comparison guard I was about to write.

## A2. But the geography restriction is load-bearing, and removing it is a disaster

Measured: generalising `geographic_values` to every low-cardinality dimension
adds 40 value tokens and would newly raise a scope facet on **101 of 697
questions**.

```
by field: origination_channel 78 · seasoning_segment 25 · source_portfolio_type 8
          vintage_year 5 · amortisation_type 2
```

The cause is one collision: **`Broker` is a VALUE of `origination_channel` and
also the ordinary word for the dimension.** So *"balance by broker"* — a plain
breakdown, 78 times over — would be read as a narrowing to
`origination_channel = Broker` and refuse. `front book` / `back book` add 25 more
in the seasoning owner's territory.

Geography values (*London*, *South East*, *Wales*) do not collide with dimension
words. General dimension values do, badly. **The restriction is not an arbitrary
limitation; it is what keeps the allowlist safe without a mark.**

## A3. So the mark IS needed — and the safe rule is the conjunction

Neither half works alone. The allowlist without a mark hits 101; a mark without
the allowlist cannot name the field. Measured across all 697 questions, the
conjunction fires **5 times**:

```
value allowlist  AND  selector mark  AND  not already resolved
                 AND  no comparison marker  AND  exactly one value of that field
```

| id | question | today |
|---|---|---|
| `b16_001` | balance where account status is active | **answers** — the defect |
| `b16_002` | balance for interest roll-up loans | **answers** — the defect |
| `pipe_191` | pipeline by stage for broker Alpha | already refuses: no governed pipeline data |
| `funded_filtered_qa_018` | balance of South East loans with LTV above 50% | already refuses: `geographic_scope` LOST |
| `data_quality_006` | How complete is broker? | already refuses at measure resolution |

**Three of the five already refuse for their own reasons, so their outcomes
cannot move. The other two are the defect.** That is why §6.1's prediction stands
unchanged.

The intermediate rules are recorded because their failures are the evidence for
the final one: allowlist + mark alone → **28** hits, reaching Q7.3/Q7.4/Q8.x on
both books — the seasoning family, `32c263a`'s family. Adding *exactly one value
named* → **13**, still reaching them, because *"the front book"* versus *"our
seasoned lending"* names its second side with a synonym the frame does not
contain, and *"direct"* versus *"acquired"* splits across two fields so each sees
one value. Adding the comparison marker → **5**. **Each guard was earned by a
family it would otherwise have broken.**

## A4. The amended design

1. **`lexical.selector_mark(text, start, end)`** — the one owner of *"the sentence
   uses this mention to select rows"*. New decision, zero owners today (§2).
   `is_filter_subject`, `condition_cut` and `subject_side` are untouched.
2. **`execution_receipt.dimension_values(frame, semantics)`** — `geographic_values`'
   scan without the geography `if`. The existing function keeps its restriction,
   because its consumers are geography-specific and its narrowness is what makes
   it safe unaccompanied.
3. **`KIND_LOST_NARROWING`** — modelled on `KIND_GEOGRAPHIC_SCOPE`, not on
   `KIND_UNRESOLVED_MEASURE` as §4.3 said. It is **provable**: the point-in-time
   reconciler stamps it APPLIED from `applied_filter_fields` and the routed one
   from `_analytical_narrowed_to`, exactly as the geographic facet is. That is
   strictly better than §4.3's LOST-at-construction, because a narrowing that
   execution DID apply must be recorded as honoured, and because it is what makes
   the comparison families safe by evidence rather than by the lexical guard
   alone.
4. **The guards from A3**, each with the family that earned it named in the code.

§5's arrival table gains one row and loses none: `_detect_geographic_scope` and
`geographic_values` are **not** modified, so nothing that reads them changes.
