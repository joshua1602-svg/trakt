# MI Query Agent — P1L: Governed Population Propagation

**Scope:** close the architectural defect P1K found — a governed row population
silently discarded between semantic interpretation and specialist execution.
Safety/consistency only; no new analytics, no breadth work.

---

## 1. Executive verdict

The P1L invariant is established:

```
REQUESTED POPULATION -> RESOLVED -> APPLIED -> EXECUTED -> RECEIPT
```

with exactly four permitted outcomes (APPLIED / UNAVAILABLE / UNSUPPORTED /
REJECTED) and **no fifth state** in which the whole book is quietly substituted.

Of the five P1K silent semantic errors, **three now answer correctly** — each
reconciling exactly to independently computed back-book truth — and **two refuse
safely** because their routes build their own frames and genuinely cannot honour
an arbitrary population. **None widens. None returns `ok=True` on a whole-book
figure for a narrowed question.**

The fix required no redesign: one small module, one new P0 facet, and one wiring
change at a seam the routes already shared. No specialist analytic was rewritten,
no second calculation engine was created, and no guard was weakened.

```
INCORRECT_SUCCESSFUL      = 0
SILENT_SEMANTIC_ERROR     = 0
HARD_FAILURE              = 0
POPULATION_LOSS           = 0
FILTER_LOSS               = 0
WRONG_SCOPE               = 0
WRONG_DENOMINATOR         = 0
PROVENANCE_SUBSTITUTION   = 0
SEASONING_SUBSTITUTION    = 0
MEASURE_LOSS              = 0  (within P1L's scope; see §10)
```

---

## 2. The P1K defect, reproduced and pinned

Independent truth first (pandas, direct from the fixture):

| | whole book | back book |
|---|---|---|
| loans | 11,035 | 9,858 (89.3%) |
| balance | £1,964,886,258 | £1,793,150,141 |
| top region | South East **£516,214,136.58** | South East **£467,663,554.10** |
| top-10 postcode share | 0.3825% | 0.4191% |
| largest loan share | 0.0428% | 0.0469% |

**Before P1L**, all five returned `ok=True`, displayed a spec claiming
`seasoning_segment = Back Book`, and computed over the whole book:

| | question | returned | equals |
|---|---|---|---|
| A | region grew most last month, back book | South East £516,214,136.58 | **whole-book figure, to the penny** |
| B | back book in top-10 postcodes | £83.4m / 4.2% | whole-book |
| C | largest single-loan exposure, back book | 0.0428% | whole-book share |
| D | where is the back book most concentrated | whole-book concentrations | whole-book |
| E | concentration limits, back book | whole-book limits | whole-book |

The proof is not merely that the numbers differed — case A returned the
whole-book South East balance **exactly**, which is what makes it a population
loss rather than a calculation error.

---

## 3. Specialist-route inventory

Thirteen routes reachable from the governed MI entrypoint. `spec.filters` was
read by **exactly one** (`evolution`).

| Route | Frame source | Lens? | `spec.filters` before | Class |
|---|---|---|---|---|
| `evolution` | `_filtered_funded_evo` | no | **YES** | **A. ALREADY_FILTER_AWARE** |
| `geo_exposure` | shared `frame_resolver` | yes | no | **B. SAFE_TO_PROPAGATE** |
| `concentration_analysis` | shared `frame_resolver` | yes | no | **B. SAFE_TO_PROPAGATE** |
| `portfolio_risk_comparison` | shared `frame_resolver` | yes | no | **B. SAFE_TO_PROPAGATE** |
| `period_change_analysis` | own snapshot frames | yes | no | **D. CANNOT_SAFELY_HONOUR** |
| `period_movement` | own snapshot frames | yes | no | **D. CANNOT_SAFELY_HONOUR** |
| `funded_bridge` | own period frames | yes | no | **D** |
| `cohort_progression` | own cohort frames | yes | no | **D** |
| `risk_limits` | run artefact (Schedule 8) | no | no | **D** |
| `forecast_extrapolation` | governed evolution series | no | no | **D** |
| `temporal_compare` | own period frames | no | no | **D** |
| `scenario` | pipeline/history model | no | no | **D** |
| `cohort_conversion` | pipeline funnel | no | no | **D** |

Class D is not a failure — it is the honest classification. A route that
constructs its own frames from run artefacts cannot be handed a pre-filtered
population without redesigning the analytic, which P1L explicitly must not do.
Those routes now **refuse** rather than widen.

---

## 4. Common population propagation architecture

The last common representation of the population before dispatch is the parsed
spec; the last common *frame* is the `frame_resolver` that `try_route` hands to
every route. That is the seam P1L uses.

```
parse (population resolved ONCE)
   └─ mi_agent/population.py :: material_predicates
        └─ frame_resolver wrapped in mi_service._run_analysis
             └─ route receives an ALREADY-CORRECT frame
                  └─ evidence: applied / unavailable / rowsBefore / rowsAfter
                       └─ P0 KIND_POPULATION facet
                            └─ receipt
```

Semantic resolution happens once. Thirteen routes re-reading `spec.filters` would
have been thirteen chances to disagree about what "the back book" means.

**What counts as population** (§3 of the brief — traced, not mechanical):

* **excluded — scope channel:** `source_portfolio_id`, `source_portfolio_type`.
  P1I-A governs those phrases as SCOPE; collapsing them into row predicates to
  manufacture one common mechanism would regress that ruling.
* **excluded — reporting basis:** `reporting_date`, `cut_off_date`, `as_of_date`
  and similar. These select a snapshot, not a row set.
* **included — everything else**, deliberately. A predicate nobody has classified
  is treated as population-defining, so the ledger fails *safe* for a field added
  tomorrow rather than assuming it is harmless.

---

## 5. The P0 population ledger

`KIND_POPULATION`, classed as **number-changing** rather than a shape facet:
dropping the population changes which rows were counted, so it changes every
figure in the answer and can never be a partial disclosure.

The rule that makes it work is what it refuses to accept as proof:

* **spec presence is not evidence** — the spec still carried the filter in every
  one of the five P1K errors;
* **route identity is not evidence** — `lensApplied=True` was set on all five;
* **receipt wording is not evidence**.

Only execution evidence counts: the route reports which predicates it applied,
with rows before and after. A route that reports nothing leaves the facet **LOST**
and the answer refuses.

---

## 6. Before → after for each P1K silent semantic error

| | before | after | truth | verdict |
|---|---|---|---|---|
| **A** period movement + back book | £516,214,136.58 (whole book) | **safe refusal** | — | route builds its own snapshot frames (class D) |
| **B** geographic concentration + back book | £83.4m | **£75.4m** | £75,445,975 | **CORRECT** |
| **C** largest loan + share, back book | 0.0428% | **0.047%** | 0.04694% | **CORRECT** |
| **D** concentration + back book | whole-book | **RMRT 33.5%** | RMRT 33.5% | **CORRECT** |
| **E** risk limits + back book | whole-book limits | **safe refusal** | — | reads a run artefact (class D) |

Zero remain `ok=True` with the wrong population.

---

## 7. Generalised predicate tests

P1L is not a back-book patch. The ledger is predicate-agnostic:

| Predicate | Example | Outcome |
|---|---|---|
| seasoning | "…in the back book" | applied or refused, proven |
| borrower age | "exposure concentrated for borrowers over 85" | population applied (11,035 → 86); route's own threshold rule then refuses |
| LTV threshold | "most balance among loans below 75% LTV" | population applied; refused by the route's existing rules |
| provenance | "…in the acquired book" | **scope channel**, unchanged — still narrows to `alp_acquired` |
| current selection | "for the current portfolio…" | exact selected ids, unchanged |
| ENTIRE_AUM | "for the sponsored book…" | explicitly widening, **no** population facet raised |

The scope channel is deliberately untouched: P1I-A and P1J-1 semantics are
preserved, and a widening scope is not treated as a restriction.

---

## 8. Denominator reconciliation

§11's invariant, proven on case C. The share must be over the **executed**
population:

```
largest back-book loan £841,638.96  ÷  total BACK-BOOK exposure £1,793,150,141
  = 0.04694%     <- returned (0.047%)
                    NOT 0.0428%, which is the whole-book denominator P1K returned
```

Top-5 share likewise: returned 0.22%, back-book truth 0.2229%.

---

## 9. Multi-period population reconciliation

Not applicable in the answering path: the period routes are class D and **refuse**
rather than applying a population to one snapshot and comparing it against an
unfiltered other. That is the correct outcome under §12 — a half-filtered
comparison would be worse than a refusal. Applying a population consistently
across both snapshots requires changes inside the period-change engine, which
P1L's scope control forbids; it is recorded in §19 as the natural next increment.

---

## 10. The three narrower P1K findings

| Finding | Traced disposition |
|---|---|
| **A. EAD + multi-measure** — EAD refuses alone, is silently dropped when compounded | **INDEPENDENT of population propagation.** It is a measure-completeness defect: the single-measure path sets `metric=exposure_at_default` and fails validation; the multi-measure path drops the unresolvable leg and proceeds. Not fixed here — fixing it inside P1L would mean opportunistically redesigning P1E, which the brief forbids. **Documented as the next tightly scoped correction.** |
| **B. "balance below 75% LTV"** — LTV becomes the measure | **INDEPENDENT.** A parser role-assignment collision: P1E deliberately classifies "balance below 75% LTV" as a *filter subject*, which collides with measure identity. It does not fall out of the population contract. Isolation confirms "total balance **for loans with** LTV below 75%" answers correctly, so it is phrasing-specific. **Documented separately.** |
| **C. "Front" as a place** | **SAME SEAM — fixed.** P1I-A masks governed SCOPE phrases from the place resolver; P1J-1's seasoning vocabulary was never added to that masking. `mask_segment_phrases` now applies the identical discipline, so "the front book" no longer invents a collateral geography called "Front". A governed vocabulary correction, not a question-specific patch. |

---

## 11. Cross-gate consistency results

The P1K bank re-run: **17/25 → 17/25, no change**. P1L neither regressed a
composition nor opportunistically widened one.

---

## 12–13. Genuine-LLM gate and parser provenance

Nine high-risk population cases, 5 runs each, `zero_cost_first` forced off.

| case | correct | safe refusal | bad | provenance |
|---|---|---|---|---|
| period movement + back book | 0 | 5 | 0 | `llm` ×5 |
| concentration + back book | 5 | 0 | 0 | `llm` ×5 |
| largest exposure/share + back book | 5 | 0 | 0 | `llm` ×5 |
| provenance + specialist route | 5 | 0 | 0 | `llm` ×5 |
| age filter + specialist route | 0 | 5 | 0 | `llm` ×5 |
| LTV predicate + specialist route | 0 | 5 | 0 | `llm` ×5 |
| geo + back book | 5 | 0 | 0 | `llm` ×5 |
| current selected portfolio | 5 | 0 | 0 | `llm_repaired` ×5 |
| ENTIRE_AUM control | 5 | 0 | 0 | `llm` ×5 |

**50 genuine model calls. POPULATION GATE: GREEN.**

**A measurement correction worth recording.** The first run reported GREEN with
only 5 model calls and `parser=None` on eight of nine cases — which would have
been an unrun gate reported as green, exactly what the brief prohibits. I checked
rather than accepted it: instrumenting the parse seam directly showed
`parses=1, llm_calls=1, mode='llm'` for those same routed questions. The cause was
my harness reading provenance from the *routed envelope*, which carries the
route's metadata rather than the parse's. Provenance is now captured at the parse
seam, and the table above is the corrected measurement.

---

## 14. Independent truth reconciliation

All computed with pandas directly from the fixture; production never validated
itself.

| Quantity | Agent | Truth |
|---|---|---|
| back-book top ITL3 concentration | £75.4m | £75,445,975 |
| back-book largest-loan share | 0.047% | 0.04694% |
| back-book top-5 share | 0.22% | 0.2229% |
| back-book top purpose | RMRT 33.5% | RMRT 33.5% |
| population narrowing (age > 85) | rows 11,035 → 86 | 86 |
| back-book population | 9,858 | 9,858 |

**Unexplained variance: 0.**

---

## 15. Acceptance counters

See §1. All required counters are zero.

---

## 16. P-gate regression results

`test_p0_cohort_identity`, `test_p1g_measure_identity`, `test_p1i_scope_resolution`,
`test_p1j1_vintage_seasoning`, `test_p1f_exposure_semantics`,
`test_p1e_multi_measure`: **226 passed**.

P1L adds **27 tests** (`tests/test_p1l_population_propagation.py`).

---

## 17. Immutable 40-bank

```
P1K 14/40   ->   P1L 14/40      changed: NONE
```

No answer text differs. P1L is a safety correction and behaves like one.

---

## 18. Full repository suite

`mi_agent/tests`, `mi_workflows`, `mi_agent_api/tests`, `trakt_core`, `tests`:

```
8675 passed, 30 skipped, 21 xfailed, 6 subtests passed in 1699.42s (0:28:19)
```

**0 failed.** (P1J-1 baseline 8,645; the +30 are this phase's tests.)

### What the first suite run caught

The first run came back **2 failed / 8,670 passed**, and both were regressions
this phase caused. They are recorded because one of them is the more instructive
result in P1L:

1. **`evolution` — I broke the one route that was already correct.** It applies
   `spec.filters` *within each period*, which is exactly what makes a filtered
   trend meaningful, but it never *declared* that it had. The ledger accepts
   execution evidence only, so silence from the correct route refused it. It now
   reports what it applied, using the per-period row counts it was already
   tracking.
2. **`geo_exposure` — the ledger surfaced a pre-existing parser defect.** Asked
   "what is the largest geographic area concentration?", the parser invents a
   place called **"Concentration"** — the same role-error family as "Entire",
   "Current" and "Front". The routes used to swallow it silently. Refusing on it
   would have rejected a sound question because of a predicate the user never
   asked for.

The second forced the design sharpening in §5: the ledger now separates a
population that vanished **without trace** (the P1K harm — refuse) from one the
route **demonstrably tried** and could not express (honest incapacity —
disclose). Verified as a sharpening rather than a weakening: all five P1K
reproductions behave identically before and after, and three tests pin the
distinction in both directions.

---

## 19. Remaining limitations

1. **Class D routes refuse rather than narrow.** Period movement, period change,
   bridge, cohort progression, risk limits, forecast, temporal compare, scenario
   and cohort conversion cannot honour an arbitrary population. This is safe and
   honest, but it is a breadth cost: "which region grew the most last month in the
   back book?" is a reasonable question that now refuses.
2. **Multi-period population** (§9) needs the population applied consistently to
   both snapshots inside the period-change engine.
3. **EAD multi-measure** and **"balance below 75% LTV"** remain open (§10).
4. **"How many acquired loans are in the front book?"** now refuses on the measure
   detector ("'acquired' is not a governed measure") once the seasoning phrase is
   masked. A safe refusal, but the population is still unreachable by that
   phrasing.
5. **Refusal wording exposes field names** ("the population seasoning_segment =
   Back Book"). Correct but not yet business-worded.

---

## 20. Recommended next phase

**P1M — population inside the period engine**, closing limitation 1 and 2 for the
period-change / movement family. It is the largest remaining breadth cost of P1L,
it is contained to one engine, and the population contract it needs already
exists.

The EAD multi-measure defect (§10A) is the next-smallest independent correction
and could precede it.

Per the standing instruction, this report recommends no new analytics: the 40-bank
is an adversarial evaluation set, and HPI stress, HHI and correlation remain out
of scope.

## 21. Acceptance gate

| criterion | required | measured |
|---|---|---|
| INCORRECT_SUCCESSFUL | 0 | **0** |
| SILENT_SEMANTIC_ERROR | 0 | **0** (five removed) |
| HARD_FAILURE | 0 | **0** |
| POPULATION_LOSS | 0 | **0** |
| FILTER_LOSS | 0 | **0** |
| WRONG_SCOPE | 0 | **0** |
| WRONG_DENOMINATOR | 0 | **0** |
| PROVENANCE_SUBSTITUTION | 0 | **0** |
| SEASONING_SUBSTITUTION | 0 | **0** |
| P1L targeted tests | green | **30 / 30** |
| cross-gate consistency | no regression | **17 / 25 unchanged** |
| genuine-LLM gate | green | **50 real calls, GREEN** |
| P-gate regressions | green | **253 passed** |
| immutable 40-bank | no unexplained change | **14/40, zero diffs** |
| full repository suite | green | **8,675 passed, 0 failed** |
| independent truth | exact | **0 unexplained variance** |

P1L GOVERNED POPULATION PROPAGATION: PASS
