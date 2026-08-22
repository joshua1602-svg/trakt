# B22 — a bare substring must not silently narrow the population

Written before implementing. §3 is BASELINE measurement of `cd4a005`.

Base: HEAD `d815883`; merge-base `4e051f3`; clean tree.

---

## 1. CONSTRUCTED COVERAGE — stated first, because it qualifies everything below

**The portfolio lens narrows ZERO of the 697 corpus questions.** Every question
that names both provenance families resolves to Total through the comparison
guard, and none names one alone. Its only coverage anywhere in the estate is
`tests/test_p1i_scope_resolution.py`.

**So every case in this commit is constructed from the first line, and no
standing corpus result — before or after — is evidence about this fix.**

What that means for the confidence anyone should place in the result, stated
plainly:

* **A green corpus proves the fix did not reach the corpus.** It does not prove
  the fix is right. The 697 questions are the wrong instrument here and will read
  identically whether the change is correct, inert, or wrong in a direction the
  constructed cases do not probe.
* **The constructed cases carry the entire claim.** Their value is bounded by
  whether I imagined the right sentences. I have written five; there is no
  reason to believe five exhausts ordinary English about buying property.
* **The confidence to place in "B22 is closed" is therefore: the six sentences
  measured here behave correctly, the module's own qualification test now gates
  the decision, and the class beyond those six is argued rather than measured.**

This is the corpus limitation already in the due diligence pack —
*a family that never asks about provenance cannot exercise the provenance lens* —
in its most complete form: not a gap in coverage, but its absence.

## 2. The sequence, steps 1–4 complete before this document

| # | step | result |
|---|---|---|
| 1 | name the surface | routed surface + `answer_diff` surface 5 |
| 2 | cases in, declared failing, **each stating its evidence** | `rt_023`–`rt_027` |
| 3 | **the named surface could not see the defect** — extended | see §2.1 |
| 4 | **re-record the baseline, before the fix** | `answer_baseline_b22.json` |

### 2.1 The surface reported a live defect as FIXED

Step 2 put three cases in expecting the correct outcome. The surface reported
all three **FIXED** — passing — against a live defect.

It asserts route, `ok`, verdict, facet kinds and statuses. **None of those moves
when a population is silently narrowed from 11,035 to 3,909.** The narrowing
lives in `executionSummary.population` and `filtersApplied`, which the surface
did not read.

`observe` now carries both, `check` asserts them through `expect_population` and
`expect_filters`, and the self-test's can-fail probes cover the two new
assertions. **A surface reporting a live defect as closed is worse than one that
is silent about it**, and this is the fourth instrument in this programme found
inadequate by the change it was supposed to measure.

## 3. Baseline, measured

| question | `scope_phrase_spans` | lens | population |
|---|---|---|---|
| balance **of the acquired book** | `"of the acquired book"` | Acquired | 3,909 ✓ |
| balance **of the direct book** | `"of the direct book"` | Direct | 7,126 ✓ |
| loans **purchased** at auction | *none* | **Acquired** | **3,909 ✗** |
| the borrower **acquired** the property | *none* | **Acquired** | **3,909 ✗** |
| **directly** held collateral | *none* | **Direct** | **7,126 ✗** |
| balance **excluding the acquired book** | `"the acquired book"` | **Acquired** | **3,909 ✗** |

The last row is the one that shapes the fix. **It has a qualified mention**, so
the qualification test that settles the three above leaves it untouched — and it
is the worst of the six: the reader excluded a cohort and received only that
cohort.

## 4. The rule

### 4.1 One helper, two callers

`portfolio_lens.scope_phrase_spans` is the helper. It already exists, already
covers `direct`, `acquired`, `purchased` and `funded` in `_SCOPE_QUALIFIERS`, and
already requires them to qualify a book noun. `resolve_lens` becomes a caller of
it. Duplicating the test would create a second owner of the decision this fix
exists to consolidate.

### 4.2 A disclaiming construction DECLINES; it does not select the opposite

*"Excluding the acquired book"* states what is **not** wanted. Inferring
"therefore the direct book" is a guess about scope, and a guess about scope has
been treated as a substitution throughout this programme.

So the lens returns Total — **and that is not the end of it.** Returning Total
silently widens the answer to include the very cohort the reader excluded, which
is the fail-open direction honour-or-clarify forbids. The exclusion is a
narrowing that was requested and not applied, so it is recorded as
`KIND_LOST_NARROWING` — B16a's kind, unchanged, for exactly the case it was built
for — and the answer refuses rather than quietly covering everything.

**The lens owner reports that it declined; it does not raise the facet itself.**
A second raiser would be the defect this programme removes.

### 4.3 A bare TERM is not a question

The critical hazard, found while enumerating callers and to be verified site by
site during implementation: several callers pass a **term**, not a question.

```
mi_workflows/analytical/planner.py:641      resolve_lens(key)
mi_workflows/analytical/populations.py:112  resolve_lens(lens_term)
mi_workflows/analytical/populations.py:174  resolve_lens(f"the {spec.lens_term} book")
```

The third is the tell: that call site **already knows** a bare term must be
qualified before `resolve_lens` will read it correctly, and wraps it by hand.
Requiring qualification without separating the two entry points would break the
first two — a caller that has already established it holds a book name would be
told it has not.

So: `resolve_lens(text)` resolves from a QUESTION and requires qualification;
term callers get an explicit entry point that says so in its name. **One
decision, two entry points, and the distinction stated rather than implied by a
hand-built string.**

## 5. Every place the owner's answer arrives — provisional, VERIFIED in implementation

Fifteen call sites across six modules, listed in the pre-registration and
confirmed one by one during implementation. The classification that matters is
question-caller versus term-caller, and it is not inferable from the call: the
argument names (`question`, `key`, `population`, `lens_term`, `spec.lens_term`)
are the only signal, and one of them lies — `populations.py:112` names its
argument `lens_term` and receives it from a spec field.

## 6. Pre-registered prediction

### 6.1 What moves

| id | today | predicted |
|---|---|---|
| `rt_023` purchased at auction | Acquired, 3,909 | **Total, 11,035, no filter** |
| `rt_025` borrower acquired the property | Acquired, 3,909 | **Total, 11,035, no filter** |
| `rt_027` directly held collateral | Direct, 7,126 | **Total, 11,035, no filter** |
| `rt_026` excluding the acquired book | Acquired, 3,909, `ok` | **refuses**, the exclusion recorded as lost |
| `rt_024` of the acquired book | Acquired, 3,909 | **unchanged** |

`answer_diff`: **4 moved**, all `routed_surface`.

### 6.2 What must not move

1. **`rt_024` unchanged.** The can-fail: a fix that stopped the lens narrowing
   would satisfy four cases and destroy the decision.
2. **No corpus answer moves.** The lens narrows none of the 697 — and per §1
   this proves the fix did not reach the corpus, not that it is right.
3. **`tests/test_p1i_scope_resolution.py` passes unchanged.** It is the only
   pre-existing coverage of this decision, and it asks about "the direct book" —
   a qualified phrase.
4. **The seasoning families stay at their by-name counts**, both books.
5. **Robustness `32/10/2`; calibration `259/259`, 0 hard fails, 0 known gaps.**
6. **No lexical decision moves.** 693 of 693.
7. **The analytical comparison paths are unchanged** — `resolve_comparison_lenses`
   is a different function and this fix does not touch it.

### 6.3 Stop conditions

* `rt_024` moving;
* any corpus answer moving;
* `test_p1i_scope_resolution` failing;
* any term-caller resolving to Total where it resolved to a cohort before;
* a second raiser of the lost-narrowing facet appearing.

### 6.4 Acceptance

* one helper, two callers; `resolve_lens` consults `scope_phrase_spans` and does
  not reimplement it;
* question-callers and term-callers separated explicitly, §5 verified site by
  site and the differences reported;
* a disclaiming mention declines AND is recorded, so the answer neither inverts
  nor silently widens;
* all five surfaces, deterministic arm, both books; seasoning by name;
* the constructed-coverage position in §1 restated in the commit and the pack.
