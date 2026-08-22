# The P0 refusal that was true of the route and false of the product

**Fixed before P1.** Three defects were stacked behind one refusal. All three are
small, none needed the build, and the P0 guard is not weakened by any of them.

The refusal:

> `How have direct and acquired balances moved over the periods?`
> → *"I understood that you asked for Direct and Acquired tracked separately,
> but that could not be applied to the calculation."*

The same request with a comparison verb — `Compare balance over time for direct
and acquired` — composed and answered completely. **The statement was true of the
route the question took and false of the product.**

---

## The diagnosis: three gates, all closed for the same reason

The question was **not** blocked by a missing capability. It was blocked three
times over by the sentence not containing a comparison verb, and once by a
plural.

### Gate 1 — the metric did not resolve

```
"How have direct and acquired balances moved over the periods?"  -> metric = None
"How have direct and acquired balance  moved over the periods?"  -> current_outstanding_balance
```

The registry carries `balance`, `outstanding balance`, `exposure` — **singular
only**. A lender types *balances*. With no metric, no analytical plan can name a
measure, so no plan is built.

This was never specific to the P0 question: `Compare balances over time for
direct and acquired` — the *working* phrasing, pluralised — failed identically.

`_METRIC_TERMS` already carried the plural convention for other measures
(`redemptions`/`redemption`, `recoveries`/`recovery`). Balance was simply missed.

### Gate 2 — the planner required a comparison VERB to accept a pair it could see

`planner._population_pair` detects `direct` **and** `acquired` through the
governed portfolio lens — but only inside `if _is_comparative(text)`. The
sentence carries no term from `_COMPARISON_TERMS`, so the branch was never
entered.

### Gate 3 — the same vocabulary, a second time

`_plan_population_movement_comparison` returns `None` unless
`SIGNAL_COMPARISON` is in the intent reading. `classify` raised that signal from
its **own copy** of the comparison vocabulary — the exact drift
`is_comparative`'s docstring warns about:

> *"Public because the planner asks it rather than keeping a second comparison
> vocabulary — which is how the same question came to resolve two different ways
> depending on which of the two lists happened to contain its wording."*

The warning was accurate and the drift had already happened.

### And meanwhile, the guard read the same sentence correctly

`execution_receipt.segments_named_in(question, values)` matched the sentence
against **governed dimension values** and returned `["direct", "acquired"]` — which
is how the refusal knew to name them. **Two mechanisms were deciding "did the
user ask for these split?" by different logic, and the weaker one (a verb
vocabulary) gated the route while the stronger one (governed value matching)
drove the refusal.**

---

## The fix

**Naming both values of one governed binary dimension IS the comparison.** A
comparison verb adds nothing: there is nothing else *"direct and acquired"* with
a measure attached could be asking for.

1. `mi_agent/llm_query_parser.py` — plurals for the curated balance terms, in the
   convention already used for redemptions and recoveries; plus plural
   registration for registry-derived measure phrases, through the same ambiguity
   filter, so a colliding plural is dropped rather than guessed.
2. `mi_workflows/analytical/intent.py` — `names_both_sides_of_a_pair()`, and
   both `is_comparative` and `classify` now read it, so the two cannot drift
   again.
3. `mi_workflows/analytical/planner.py` — the pair strategy no longer requires a
   verb.

### Result

```
How have direct and acquired balances moved over the periods?
  ok=True  route=analytical_composition
  "Across 2026-04-30 → 2026-06-30, Direct, 7,126 loans: Current Outstanding
   Balance £1.36bn → £1.39bn (+£21.5m). Across 2026-04-30 → 2026-06-30,
   Acquired, 3,909 loans: £568.3m → £579.4m (+£11.1m)."
```

**Shape 8 is now PROVEN — 3 of 3 declared phrasings**, on both books.

---

## Why the P0 guard is not weakened

The guard is untouched. It still refuses a whole-book series returned for a
segmented request. **It stops firing on this question because the route now
produces the segmented series it was protecting.**

The other two named P0 refusals are unaffected, and structurally so:

| P0 refusal | names a value of a governed pair? | still refuses |
|---|---|---|
| `Show me balance by month by region and LTV band` | **no** — two DIMENSIONS | ✅ |
| `balance by month broken down by LTV band and region` | **no** — two DIMENSIONS | ✅ |

`segments_named_in` returns `[]` for both. They coordinate two **dimensions**,
not two **values of one dimension**, so they cannot reach the changed branch at
all. `tests/test_p0_segment_pair_routing.py` pins this as a structural property,
not an incidental one.

---

## Verification

* `tests/test_p0_segment_pair_routing.py` — 18 tests: the comparison definition,
  the non-comparative cases including both surviving P0 refusals, a drift test
  asserting `classify` and `is_comparative` cannot disagree, and plural/singular
  metric parity.
* **212 passed** across the P0 and analytical-boundary suites
  (`test_analytical_intent_boundary`, `test_p0_temporal_honouring`,
  `test_p0_cohort_identity`, `test_p0_time_axis_request`).
* Failure sets diffed **before and after** across the analytical, parser, intent,
  planner, measure, lens and population suites: **4 failures before, 4 after,
  identical.** No regression introduced, none fixed. (The 4 are pre-existing;
  `test_q7_compares_the_two_governed_sides_and_reconciles` fails identically on
  unmodified code.)

---

## Recorded as a standing pattern

This is the **fourth instance** of two mechanisms answering one question with
different logic, where the weaker gated the route and the stronger drove the
refusal. See `docs/mi_dual_mechanism_pattern.md`, which carries the standing
constraint: **an instance of this pattern is not closed by adding a phrase to the
weak list.**

## A limit of the fix, recorded

`names_both_sides_of_a_pair` recognises the governed binary pairs by their own
vocabularies. It does **not** recognise elided coordination:

```
"front book and back book movement"                    -> DELIVERS
"how do the front and back books compare over the periods" -> still refused
```

The second shares one noun between two modifiers (*"the front and back books"*).
That is a real lender phrasing and it is not covered. It is recorded rather than
patched, because covering it well means parsing coordination properly rather than
adding another vocabulary entry — which is how this defect arose in the first
place.

**This limit must not be closed with a vocabulary entry.** Adding
`"front and back books"` to a list closes the example and leaves the mechanism
intact; the next elision fails identically and the pattern gains a fifth
instance. The constraint is recorded in `docs/mi_dual_mechanism_pattern.md` and
any change that closes it by adding a phrase should be rejected in review on the
strength of it.
