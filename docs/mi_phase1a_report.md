# Phase 1A — report

**Status: GO-READY.** The Phase 0 blocker is cleared. `portfolio_summary` is
ready for a **separately authorised** conversion task and was **not switched
here**.

Nothing client-visible shipped. T3–T7 remain closed. One production change
outside the migration (the packaging fix), and one inside it (the contract
extension).

Commits: `6bc308f`, `e1941e9`, `3631cf8`, plus the estate follow-up.

---

## 1. Deployment packaging — CONFIRMED and closed

**Confirmed from the deployment path, not from the test's report.** Staging
exactly what `.github/workflows/deploy-mi-api.yml` stages and importing the app
from that directory alone:

```
ModuleNotFoundError: No module named 'question_interpretation'
```

**Cause.** `mi_agent/llm_query_parser.py:29` and
`mi_agent/execution_receipt.py:45` carry **module-level**
`from question_interpretation import lexical as _lexical`. `mi_agent_api.app`
imports `mi_agent`, so the App Service fails at **startup** — not on the first
request that reaches it, which is what the existing test's message predicted.

That package is the single lexical owner the question-interpretation programme
created. Consolidating those readings into one module put it on the API's import
path; the deploy manifest was never updated with it.

**Change.** One entry in `deploy/trakt-mi-api/package_contents.txt`. No
packaging restructuring, no workflow change, no requirements change — it is
pure-Python repo code with no new third-party dependency.

**Test.** The completeness test already caught this and was failing; it is not
what was missing. What was missing is the App Service equivalent of the
guarantee the container image gets from its build-time smoke import — *a
manifest can list every package the closure names and still produce an artefact
that does not import*. `TestStagedArtefactImports` stages by the workflow's own
rules and imports `mi_agent_api.app` in a subprocess whose `sys.path` holds the
staging directory and **not** the repo (otherwise it would pass on a manifest
that stages nothing).

**Evidence it cannot recur through the same mechanism** — manifest reverted to
its defective state:

```
FAILED TestArtefactCompleteness::test_every_reachable_repo_package_is_staged
FAILED TestStagedArtefactImports::test_the_staged_artefact_imports_the_asgi_app
```

With the fix: **14 passed**, and the staged artefact imports the ASGI app.

Pre-existing and unrelated: reproduces at `42cef00` **and** `8066b2f`.

---

## 2. Estate baseline — introduced-name set

Captured by exact test name at `091d2ea`:

```
107 failed, 9375 passed, 36 skipped, 13 xfailed, 28 errors
```

The `+2 passed / +5 xfailed` against Phase 0's first run is exactly
`tests/test_migration_preregistered.py`, committed between the two — fully
accounted for.

**Introduced-name set: EMPTY**, established three ways rather than asserted:

* product code at `091d2ea` byte-identical to `42cef00`;
* a clean `git worktree` at `42cef00` reproducing the failures;
* a name-by-name diff on the MI path — **HEAD 18, base 19, in-HEAD-not-in-base: none.** The one base-only name (`test_registry_governance`) is a worktree artefact and passes at HEAD.

**A Phase 0 baseline correction.** `question_interpretation/tests/` was **never
run** at Phase 0 — a whole directory outside A5's registered surfaces. It holds
**566 passing tests and one pre-existing failure**
(`test_p0_time_axis_request` · *"balance by each month"*), which reproduces
identically at `42cef00`. Pre-existing, but it should have been recorded and was
not.

---

## 3. Interpretation-contract design

**Existing owner:** `mi_agent.portfolio_lens`. Unchanged, and still the only
thing that decides what *"the acquired book"* means.

**Representation chosen:** a new single-valued `SourceScopeClaim` on
`QuestionInterpretation.source_scope` — **not** an entry in `population`.

Two reasons, both load-bearing:

1. **A source lens and a seasoning segment are different axes.** *"the front
   book"* is a seasoning population; *"the acquired book"* is a source lens; a
   question can name both and neither implies the other.
2. **`population` is a list of narrowings, and `total` is not a narrowing.**
   Putting *"no source narrowing"* into a list of narrowings is exactly how
   absence and Total became indistinguishable.

**How the five cases are distinguished** — by `state`, never by absence:

| state | scope | meaning |
|---|---|---|
| `FILLED` | `total` | the owner **looked** and found no source narrowing |
| `FILLED` | `direct` / `acquired` | the type lens |
| `FILLED` | `cohort` | named book(s); `portfolio_ids` carries them |
| `EMPTY` | — | the owner was **not consulted**. **Not Total.** |
| `UNRESOLVABLE` | — | consulted, could not resolve; `reason` says why. **Not Total.** |

`__post_init__` refuses a `FILLED` claim with no scope, and refuses a scope
outside the vocabulary — an unknown lens kind is recorded `UNRESOLVABLE` rather
than mapped onto the nearest member, because that substitution would be
invisible downstream.

**Why this is not a second semantic owner.** The projection makes **one call**
to `resolve_lens` and records the answer. There is **no vocabulary** in
`question_interpretation` for this — no phrase list, nothing to match against —
so widening the owner widens this automatically. If the owner is unavailable or
raises, the claim is `UNRESOLVABLE`, never silently Total. A test asserts the
contract **agrees with its owner on every case** rather than trusting the
wiring.

---

## 4. Implementation

**Production modules changed — two, 138 lines added, 4 removed:**

| file | + | − |
|---|---:|---:|
| `question_interpretation/schema.py` | 81 | 1 |
| `question_interpretation/projection.py` | 61 | 3 |

Plus `deploy/trakt-mi-api/package_contents.txt` (5 lines, objective 1).

**Tests added/changed:**

* `question_interpretation/tests/test_source_scope_claim.py` — **19 tests**, new
* `tests/test_mi_api_appservice_packaging.py` — 1 test added (14 total)
* `question_interpretation/tests/test_schema.py` — 1 assertion, the pre-registered key-set movement

**No phrase lists added.** Confirmed structurally: `build_plan` calls none of
`resolve_lens`, `resolve_lens_with_default`, `lens_from_term`,
`resolve_comparison_lenses`, `segments_named`, `resolve_population_predicate`;
`lens_for` calls `lens_from_selection` and not `resolve_lens`, and mentions no
`question` name or attribute — asserted over the **parsed AST with docstrings
stripped**, after a first version of that test matched the word "question" in
its own prose and proved nothing.

---

## 5. Before/after contract evidence

`python -m migration_phase0.probe_source_scope`

**Before:** `source_scope` did not exist; `population` was `[]` for every case
below, so Total and Acquired were indistinguishable.

| question | `source_scope` | `population` |
|---|---|---|
| *Please provide a portfolio summary* | `filled` / `total` / narrows=**False** | `[]` |
| *Summarise the direct book* | `filled` / `direct` / narrows=True | `[]` |
| *Summarise the acquired book* | `filled` / `acquired` / narrows=True | `[]` |
| *Summarise the acquired_001 book* | `filled` / `cohort`, ids `('acquired_001',)` | `[]` |
| *Summarise the front book* | `filled` / `total` / narrows=**False** | `['seasoning_segment']` |
| *Summarise the front book in the acquired portfolio* | `filled` / `acquired` | `['seasoning_segment']` |

**Not conflated, in both directions.** A seasoning question carries a seasoning
population and `scope=total`; a lens question carries a lens and no seasoning
claim; and the last row carries **both**, separately. The probe checks every
case against what the owner independently says and reports
`AGREES WITH ITS OWNER ON EVERY CASE`.

---

## 6. `portfolio_summary` shadow result

| | Phase 0 | Phase 1A |
|---|---:|---:|
| cases on the surface | 9 | **9** |
| plans constructible | 0/9 | **9/9** |
| externally supplied lenses | 9/9 | **0/9** |
| economic differences | 0 | **0** |

Economics unchanged, field for field — availability, period, reporting date,
period count, region column, all five metrics, every `topRegions` entry
(`region`, `balance`, `share`), the cohort set:

* Total £1,964,886,258.21 / 11,035 loans
* Acquired £579,377,675.23 / 3,909
* Direct £1,385,508,582.98 / 7,126

**Structural differences observed: none beyond the intended one.** The executor
now takes nothing from the harness but the plan, and `lensFromPlan` records the
scope the plan selected — checked against what the shipped route resolved
independently. That check is why "identical economics" now means something it
could not in Phase 0, when the harness supplied the lens itself.

**Still not compared, and still stated rather than glossed:** answer text (the
baseline does not record `portfolio_summary` prose as stable) and receipt facets
(the shadow path produces no envelope). Both are conversion-commit work.

---

## 7. Regression result — every registered A5 surface

| surface | baseline | after Phase 1A |
|---|---|---|
| calibration bank | 267 passed | **267 passed** |
| robustness 44 | 32 / 6 / 4 / 2 | **32 / 6 / 4 / 2** |
| — seasoning families **by name** | Q1 4 · Q7 4 · Q8 12 | **Q1 4 · Q7 4 · Q8 12** |
| shipped shapes | 15 correct, 0 wrong | **0 wrong; C1–C5 reconcile** |
| routed surface | 31 passed, `rt_004` | **31 passed, `rt_004`** |
| recognition (61) | 15 / 7 / 10 / 29, 13 no-route | **identical, by shape row for row** |
| time-series surface | T1 PROVEN … T8 ABSENT | **identical** |
| **silent drops** | **0** | **0** |
| `question_interpretation/tests` | *(not in baseline)* | 566 passed, 1 pre-existing |
| shadow equivalence | 0 diffs, 9 blocked | **0 diffs, 0 blocked** |

**Case-name movements: none.** No delivered answer changed economically, no
governed refusal became an answer, no recognition movement, no new
interpretation owner, no raw-question read in the plan builder.

---

## 8. Final status

# GO-READY

Every condition met:

* the source lens is fully represented by the interpretation contract, including the distinction between *explicitly unrestricted* and *unresolved*;
* one semantic owner — `mi_agent.portfolio_lens` — with no vocabulary added downstream;
* 9/9 shadow plans construct without external lens injection;
* economic equivalence exact;
* all regression gates clean **by name**;
* no abort condition fires.

`portfolio_summary` is ready for a separately authorised conversion. **It was
not switched here.**

---

## 9. Measured effort

| | |
|---|---|
| production lines changed | **143** (138 contract + 5 manifest) |
| production modules changed | **3** |
| tests added | 20 (19 contract boundary + 1 packaging smoke) |
| tests changed | 1 (pre-registered key-set movement) |
| commits | 4, separable |
| baselines updated to make something green | **0** |

### Dependencies discovered

* **`lens_from_selection` falls back to Total for anything it does not recognise.** Convenient for a UI dropdown, dangerous for a plan: an unchecked call would silently widen the population. `lens_for` checks the rebuilt lens against the claimed scope and refuses on mismatch. **Any later route conversion consuming a governed scope needs the same check.**
* **`question_interpretation/tests/` was outside the Phase 0 baseline.** Now measured; it belongs in A5's registered surfaces permanently.
* The contract's `as_dict()` is published on every MI answer's payload (`mi_agent_workflow.py:1003`), so **any contract extension is a payload change** — additive here, but the next one may not be.

### Does this change the expected cost of later route migrations?

**Yes, downwards for the contract work and upwards for the checking.**

* **Downwards:** the extension was **138 production lines** and needed no new owner, no phrase list and no change to the lens resolver. Phase 0 called the contract gap the blocker and it cost less than the shadow plan that found it. If the remaining gaps (filter-clause join, rank parameters, comparison pairing) are of this shape, the contract work is bounded.
* **Upwards:** the `lens_from_selection` fallback shows that **carrying a governed decision into the contract is not sufficient — rebuilding it downstream needs a mismatch check**, because the owner's convenience paths default rather than fail. That is a per-primitive cost the scoping study did not anticipate, and it applies to every governed concept a plan reconstructs.
* **Unchanged:** the study's warning that blast radius does not predict conversion cost still stands. `portfolio_summary` needed a contract change no blast-radius metric predicted.
