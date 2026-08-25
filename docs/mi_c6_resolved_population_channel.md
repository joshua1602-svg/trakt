# C6 filter binding, steps 1–2 — the resolved population channel

Base `97bbdc1` (the scoping report). **Steps 1 and 2 only.** No plan primitive
added, no route switched, no duplicate owner removed, C6 not executed.

---

## 1. What was asked, and what each half returned

| | asked | result |
|---|---|---|
| Step 1 | add the resolved population channel | **DONE** |
| Step 2 | prove 121/121 agreement | **PROVEN — 121/121 on field, operator and value** |

Step 2 also carried a stronger check that was **not** asked for and that
**fails**: whether the claims, applied through `apply_population`, select the
same rows `_apply_filters` selects. They do not, on 71 of 119 questions. That
check is the stated precondition for **step 3**, so its failure is reported
here rather than discovered inside the conversion. Section 5.

---

## 2. Step 1 — the channel

`RowPredicateClaim` on `question_interpretation/schema.py`, populated by
`projection._row_predicates`, carried on `QuestionInterpretation.row_predicates`
and in `as_dict()`.

```
Show funded balance evolution by month for loans above 50% LTV.
   current_loan_to_value  gt  50.0
Show funded balance for borrowers over 75 with LTV above 50.
   youngest_borrower_age  gt  75.0
   current_loan_to_value  gt  50.0
Show funded balance evolution by month.
   (none)
```

It is a **third** channel, and deliberately so. The scoping report's ruling
stands: `FilterClaim` says what the question SAID and carries no field on
purpose; `PopulationClaim` names a population by INTENT and is kept apart from
its resolution on purpose. Adding a field to either would collapse a distinction
each exists to hold. So the brief's instruction — *"do not decide in advance
that `field` must simply be added to `FilterClaim`"* — resolves as: it must not
be, and it is not.

The channel **resolves nothing**. `llm_query_parser._filter_field_of` already
bound the clause, once, upstream of every route, and
`population.material_predicates` already normalises the result into
`Predicate(field, op, value)`. `_row_predicates` calls that same normaliser. No
second binder exists, so the "no silent guessing" rule is satisfied
structurally: there is no new place that could guess.

`source_portfolio_id` cannot appear on this channel. `mi_agent.population`
excludes it by name under the P1I-A ruling, and a test asserts the exclusion.

Production cost, in the canonical unit (raw added + raw deleted production
diff lines, never net-executable):

```
question_interpretation/schema.py       47 added,  0 deleted
question_interpretation/projection.py   36 added,  1 deleted
                                        --------------------
                                        84 raw production diff lines
```

Most of it is docstring: the two rulings this channel sits between
(`FilterClaim` says what was SAID, `PopulationClaim` names an INTENT) have to be
recorded on the new claim, or the next reader collapses them.

---

## 3. Step 2, first result — the description agrees, 121/121

`migration_phase0/row_predicate_agreement.py`, over the 119 corpus questions
that carry filters:

```
predicates on the contract : 121
FIELD  agreement : 121/121
OP     agreement : 121/121
VALUE  agreement : 121/121
```

Families covered — the brief required the binding be generic across them:

```
LTV 56 · borrower age 20 · balance 15 · borrower type 15 · geography 9 ·
months on book 4 · interest rate 2
```

### 3.1 The count reached 121 only after the harness was corrected

The first run reported **117**, and the four missing predicates were the whole
finding of the sub-investigation:

```
How has the profile of our new lending changed over the last few months?
    spec.filters = {'months_on_book': {'op': 'le', 'value': 1}}
    claims       = []
```

`projection.project` is the read-only Stage 1 harness: it calls
`_deterministic_parse` **directly**. `resolve_seasoning_role` — which is where
"new lending" becomes `months_on_book <= 1` — runs inside `parse_with_repair`
and **mutates `spec.filters`** afterwards. Production assembles through
`from_parts` on the spec `ParsedQuestion.parse` returns, so production sees the
seasoning predicate and the harness did not.

Measuring `project()` would have scored the contract against filters the
executor never received, and would have silently exempted the entire seasoning
family. The instrument and the tests both now go through the production shape.

---

## 4. Step 2, second result — the EXECUTION check fails, 48/119

Row-set identity between `_apply_filters(frame, spec)` and
`apply_population(frame, claims)` on the real governed frame (11,035 rows):

```
identical : 48/119
DIFFERENT : 56          executor raised (controlled validation failure): 15
```

Every one of the 71 is classified; none is unexplained:

```
  56  percent_scale   (executor rescales points -> fraction)
  15  absent_column   (executor raises; apply_population widens)
```

**Percent scale.** `current_loan_to_value` is stored as a fraction
(`0.0000 .. 1.0456`). `_apply_filters` divides a percent-format threshold by 100
before comparing — `mi_query_executor.py:555-565`, "the single percent-scale
source of truth, never re-guessed downstream". `apply_population._mask`
delegates only the **comparator** to `_apply_numeric_op` and never the **value
normalisation**, so `gt 50.0` is compared against fractions:

```
_apply_filters   : 1,889 rows
apply_population :     0 rows
```

**Absent column.** For `borrower_type`, `_require_column` **raises** — a
controlled validation failure, fail-closed. `apply_population` records the
predicate as `unavailable`, leaves the frame alone and returns **all 11,035
rows**. That is the silent-widening shape this programme has closed twice
already; here the caller is expected to refuse on the evidence, which makes the
widening safe today and unsafe the moment a caller forgets.

This contradicts `apply_population`'s own docstring:

> Reuses the executor's own comparison semantics rather than reimplementing
> them, so a route and the point-in-time path cannot disagree about what
> "age > 85" means.

It reuses the comparator. It does not reuse the normalisation, and the two
paths do disagree.

---

## 5. Blast radius — the divergence is LATENT, not live

Measured by spying on `mi_agent.population.apply_population` while running all
119 filtered questions through `execute_governed_mi_query`:

```
questions reaching apply_population today : 4/119
   [('months_on_book','le',1)]   How has the profile of our new lending changed…
   [('months_on_book','le',3)]   Are we originating different types of loans now…
   [('months_on_book','le',3)]   How does recent lending compare with what we…
   [('months_on_book','le',1)]   Has the risk and borrower profile of new business…

of those, DISAGREEING (a live wrong answer) : 0
```

All four are `months_on_book` — an integer field, no percent format, column
present — and all four are in the **agree** set. Confirmed end to end: the LTV
questions answer correctly today (1,889 loans, £472.5m on the one that routes),
because they never reach `apply_population`.

**So nothing is broken in production and nothing was changed to make it so.**
The divergence becomes live the moment `apply_population` becomes the plan-level
`SELECT_POPULATION` primitive, which is exactly step 3.

---

## 6. What this does to step 3

The scoping report recorded step 3 as:

> Add the second `SELECT_POPULATION` kind and its reader; prove
> `apply_population` ≡ `_apply_filters` per period on the four delivered cases.

The equivalence has now been measured **early**, on the whole corpus rather than
four cases, and it does not hold. Step 3 as written would route the LTV family
(56 questions) into a primitive that selects zero rows, and the borrower-type
family (15) into one that silently returns the whole book.

Step 3 therefore acquires a prerequisite, and it is small and generic:

- **Move the value normalisation to where the binding is.** The rescale is not a
  comparison concern; it is part of resolving a percent threshold against a
  governed field. Doing it once, where the predicate is built, keeps
  `_apply_filters` and `apply_population` agreeing by construction rather than
  by maintenance — the same failure mode as the grain claim and the stage
  claim.
- **Make an unappliable predicate fail closed inside the population helper**, so
  the helper cannot return a wider frame than asked for regardless of whether a
  caller reads the evidence.

Both are changes to a **shared** helper already consumed by `population_facets`,
`reconcile_population`, `mi_service._population_frame`,
`mi_workflows.analytical.populations.apply` and the threshold receipt. **Neither
is made in this task** — steps 1–2 were the authorised scope, and a shared-owner
change wants its own pre-registration and blast-radius proof.

---

## 7. Tests

`question_interpretation/tests/test_row_predicate_claim.py`, 9 tests, through
the production assembly path.

Positive: numeric threshold bound to its governed field; two clauses bound
independently; a categorical clause carried as a value; a derived population
("new business") carried as the predicate it executes.
Negative: an unfiltered question carries none; a source-portfolio scope never
becomes a row predicate; `FilterClaim` still carries no field.

Non-vacuity, both mutations applied to `projection.py`:

```
MUTATION A  _row_predicates call removed   -> 6 failed, 3 passed
MUTATION B  field_key=None                 -> 6 failed, 3 passed
unmutated                                  -> 9 passed
```

The three survivors are the negative controls, which must pass when the channel
is absent — that is what they assert.

---

## 8. Files

```
question_interpretation/schema.py                        production  +47 -0
question_interpretation/projection.py                    production  +36 -1
question_interpretation/tests/test_row_predicate_claim.py test        new
migration_phase0/row_predicate_agreement.py               assurance   new
docs/mi_c6_resolved_population_channel.md                 doc         new
```

## 9. Status

# STEPS 1–2 CLOSED. STEP 3 BLOCKED ON A NAMED, MEASURED PREREQUISITE.
