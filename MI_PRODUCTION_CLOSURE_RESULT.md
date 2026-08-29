# Final production-closure sprint — disclosures, coverage enforcement, availability safety

Start `6a2f224` (clean) → end `7a2fdfe` + this report.

The sprint closed the five disclosure gaps that forced the coverage gate to be
reverted, found and closed a sixth of the same class, restored the gate, and
added the rule that an unavailable augmentation call refuses rather than
quietly executing a narrower reading. Every gate in the brief was run from
scratch. Nothing is spliced from an earlier run.

---

## 1–3 · Commits, files, lines

| commit | phase |
|---|---|
| `cdfe83c` | 1 — the five manifest disclosure gaps |
| `02edd77` | 2 — the sixth gap the census found, and the U1/U2/U3/U4 classification |
| `9017828` | 3 — the coverage gate restored |
| `7a2fdfe` | 4 — model-availability safety |

Production files, `6a2f224..7a2fdfe`:

| file | + / − |
|---|---|
| `question_interpretation/completeness.py` | +45 −4 |
| `mi_agent_api/mi_service.py` | +108 −1 |
| `mi_agent_api/chat_routing.py` | +56 −5 |
| `mi_agent/execution_receipt.py` | +16 −1 |
| `mi_agent_api/concept_merge_arm.py` | +9 −1 |
| `tests/test_p4_model_availability.py` (new) | +288 |

Four production files carry real logic; the fifth is one renamed constant. No
prompt change, no grammar, no ontology, no new capability, no new model call.

---

## 4 · The five manifest disclosure defects, and the exact fixes

Each was proved to be an execution-disclosure defect before anything was
touched, by reading the envelope and finding the narrowing present in the
answer and absent from the receipt.

| test | question | route | stated concept | what execution actually did | what was missing | producer used |
|---|---|---|---|---|---|---|
| `test_query_applies_drill_through_filters` | Show balance by region | point-in-time | facet `collateral_geography` "region" | grouped `geographic_region_obligor` | the facet concept lost the alternates its dimension twin keeps | `requested_dimension_terms`' own alternates |
| `test_cohort_progression_route_returns_metric_line` | how has funded balance evolved for the direct book | `cohort_progression` | scope `portfolio_lens` | narrowed the cohort by `lens.filters` | no `scopeApplied` | `_declare_lens_scope` → `_declare_scope` |
| `test_kfi_trend_by_week_e2e` | Show KFI trend by week. | `evolution_funnel` | value `pipeline_stage` "KFI" | built the series for one governed stage | nothing named the stage | `metadata.populationApplied` |
| `test_a_weekly_funnel_question_is_no_longer_told_it_is_monthly` | Show KFI trend by week. | `evolution_funnel` | same | same | same | same |
| `test_q8_two_populations_move_independently_and_reconcile` | How has the balance from direct lending changed relative to acquired | `analytical_composition` | value `source_portfolio_type` "acquired" | split 441 Direct / 199 Acquired and published it in `narrowedTo` | the coverage adapter never read `narrowedTo` | `route.narrowed_entries`' existing record |

Two of the five were **reader** gaps, not route gaps: the evidence was already
published and the coverage adapter did not read it. Two were routes that
disclosed nothing. One was an owner disagreement inside the ledger itself.

Phase 2's census found a sixth of the same class, on the surface the census
reaches most: a question naming several governed measures executes all of them,
records the set in `spec.measures`, folds only the first into `spec.metric`, and
the adapter read the singular alone. Four flagged instances resolved.

Where a format rule now has two readers it is written once. `populationApplied`'s
field-name parsing moved to `execution_receipt.declared_population_fields`, and
both the receipt and the coverage adapter call it. `_declare_scope` remains the
only writer of `scopeApplied`.

---

## 5 · Proof no calculation changed

* **The 166-question acceptance bank, deterministic arm, run at `6a2f224` and
  again at final HEAD: 162 of 166 envelopes carry identical answer text.** The
  four that differ are the four coverage refusals. CORRECT is unchanged at 117.
* **Deterministic 166 sweep corpus, real book — byte-identical** between `6a2f224` and the
  end of Phase 2. The comparison strips only artifact uuids and `createdAt`, and
  was falsified before use: it still catches a one-penny `rawValue` change and a
  one-character prose edit. Across all 166 the only raw-differing key was
  `artifacts`.
* **The five manifest questions — 13 envelopes, 0 differ** on `ok`, `error`,
  verdict, spec, dataset, artifact row counts and key sets, and prose with
  numerals masked. Exact byte identity is not measurable there and the report
  says so: those fixtures synthesise balances randomly, and one commit run twice
  gives 11,864,085.41 and 12,583,856.48 for the same question.
* **Frozen 278-module manifest — 85 names, exactly, at every phase boundary.**
* The deterministic 1,446 surface answers 851 questions before enforcement —
  the same 851 as the previous sprint's baseline.

---

## 6 · Coverage census before enforcement

1,612 questions, ledger present on 1,612.

| | answering questions carrying UNACCOUNTED |
|---|---|
| start of sprint | 19 |
| after the `spec.measures` reader fix | 15 |

The 15 remaining, over 11 distinct questions, classified:

**U1 — genuine semantic omission (10 questions, 14 instances).** Verified against
the book, not the prose:

* "how many drawdown loans have LTV above 50%" answers 144 — LTV>50 across the
  whole book. The drawdown intersection is 45.
* "WA LTV for lump sum lending in the Direct portfolio" answers 441, the whole
  Direct book. Direct ∩ lump sum is 278.
* "how do the Direct and Acquired portfolios differ" and "Summarise the
  Acquired" both answer 640, the whole book. Acquired is 199.
* "Break drawdown balance down by geography and LTV band" groups the whole book.
* Two questions name `current_valuation_amount`, a governed registry field this
  book does not carry, and silently substitute: one filters balance instead of
  valuation, the other groups by age bucket instead of property value band.
* Two answer pipeline questions from the funded bridge.
* One substitutes a completion run-rate for the KFI→completion conversion rate
  it was asked for, and discloses that only in prose.

**U3 — semantic-estate limitation (1 question).** "the balance where the borrower
acquired the property recently": the estate cannot name recent property
acquisition, and the value owner reads the verb as the Acquired book. Recorded,
not fixed — no vocabulary was added in this sprint.

**U2 — remaining disclosure defect: none.** That was the entry gate for Stage 3.

**U4 — known non-coverage defect:** Q04C, right population and wrong output
grain, which coverage is not built to see.

---

## 7 · Enforcement result

Frozen manifest run first, as required: **85 → 85, name for name.** The five
false refusals that forced the revert are gone.

Final census under enforcement: **1,612 questions, 0 answering questions carry an
UNACCOUNTED concept.** Exactly 15 envelopes moved, all `ok: True → False`, all 15
the cases classified above. Zero correct answers lost, zero prose changed
elsewhere.

Deterministic 1,446 surface: 851 → 840 answers, the 11 SURFACE-bank instances.

---

## 8 · The six WRONG → refusal cases

On the deterministic 166 the gate moves **four**, every one from a wrong answer
to a controlled refusal, each verified against the book: Q03A, Q05C, Q07B, Q16B.
Zero correct answers move; zero refusals become answers.

The prototype moved six. **Two of those six were false refusals**, and closing
the disclosures proved it:

* **Q19A** now answers correctly — Direct 441, Acquired 199, both narrowed, both
  published in `narrowedTo` — and was refused only because the adapter could not
  read that record.
* **Q17C** was refused on "borrower age", which the answer computes and
  publishes as a governed measure.

Q17C carries a residual and this report does not hide it: deterministically the
answer breaks down by ticket size alone, and bare "LTV" in that phrasing is
named by no axis owner ("LTV band" is). The question is still incompletely
answered deterministically, coverage cannot see the axis that makes it so, and
the gate no longer catches it — it was catching it by accident, on a different
concept. Under the healthy Opus arm the axis is recovered and Q17C is CORRECT
6/6.

---

## 9 · Frozen manifest

**85, name for name**, at Phase 1, Phase 2, Phase 3, Phase 4 and at final HEAD.
278 modules, 0 hung, 0 new names, 0 names gone.

The new `tests/test_p4_model_availability.py` is deliberately **not** added to
the frozen module list — that list is the baseline and is not absorbed into. It
runs separately: 15 passed.

---

## 10 · The model-availability state machine

```
                      apply()
  ┌──────────────────────────────────────────────────────────────┐
  │ call raised — provider error, usage limit, timeout, unreachable │──┐
  │ reply unreadable — not JSON, no concepts key, wrong shape       │──┤
  │ the arm was enabled and could not be built or run at all        │──┤
  └──────────────────────────────────────────────────────────────┘  │
                                                                    ▼
                                                        status = proposal_unavailable
                                                        (no `proposed` key at all)
                                                                    │
  ┌──────────────────────────────────────────────────────────────┐  │
  │ call succeeded, model proposed nothing  → status = no_change  │  │
  │ call succeeded, proposals bound + merged → status = applied   │  │
  └──────────────────────────────────────────────────────────────┘  │
                                                                    ▼
                                  _governed_context: coverage gate, then availability gate
                                                                    │
                              ok and status == proposal_unavailable ─┴─→ controlled refusal
```

`proposal_unavailable` is never inferred from `[]`. A successful call that
proposes nothing returns `no_change` **with** `proposed: []`; an unavailable call
publishes no proposals at all. The two are distinguishable at the arm's own
boundary and that is asserted directly.

One hole was closed: `apply` reports its own failures, but building the
interpretation or the value catalogue could fail around it, and the evidence was
then left at `None` — indistinguishable from the arm being switched off. An
enabled arm that could not run now records that it could not run.

**No exception is admitted, and the sprint asked me to decide that rather than
assume it.** The estate has no completeness proof independent of the
deterministic parse: the coverage ledger and the execution receipt are both
built from the same owners that produced the reading, so neither can certify a
reading whose gap is a term no owner names. "Size" is exactly such a term. The
one proof that would be independent — that the merge can only fill slots the
deterministic parse left empty, so a contract with no empty slot cannot be
changed by any proposal — is one I would have had to construct, and constructing
it was out of scope. So unavailability refuses, including on questions the
deterministic estate answers perfectly.

---

## 11 · Forced-failure results

15 injection tests, none consuming credit — every one injects at the model call.

| injected | arm status | outcome |
|---|---|---|
| usage limit (`rate_limit_error`) | `proposal_unavailable` | controlled refusal |
| provider error (529 overloaded) | `proposal_unavailable` | controlled refusal |
| timeout | `proposal_unavailable` | controlled refusal |
| unreachable (connection) | `proposal_unavailable` | controlled refusal |
| prose instead of JSON | `proposal_unavailable` | controlled refusal |
| single-quoted pseudo-JSON | `proposal_unavailable` | controlled refusal |
| JSON with no `concepts` key | `proposal_unavailable` | controlled refusal |
| `concepts` not a list | `proposal_unavailable` | controlled refusal |
| a concept with no term | `proposal_unavailable` | controlled refusal |
| **valid successful empty proposal** | `no_change`, `proposed: []` | **answer stands** |
| **valid successful non-empty proposal** | `applied` | **answer stands, and narrows** |
| arm switched off | no evidence published | answer stands |

The failure cases are probed with a question whose coverage ledger is clean, so
the only thing that can be refusing them is availability — not coverage under
another name.

Forced model-unavailable acceptance suite, 31 questions × 5 injected modes
(Q16B, Q10B, the seven recoveries, five former regressions, the protected set,
known residuals, six deterministic-easy questions, three must-refuse controls):

| | every mode |
|---|---|
| answered (`ok: True`) | 0 |
| controlled refusals | 31 |
| **WRONG** | **0** |
| whole-book answers | 0 |
| Q16B whole-book | 0 |
| Q10B refused | yes |
| must-refuse answered | 0 |

---

## 12 · Q16B

**Healthy, 20 independent invocations:** 20/20 reached the model, 20/20 model id
`claude-opus-5`, 20/20 valid proposals, 20/20 `applied`, 20/20 CORRECT,
population 244 on every run — which is the drawdown count in this book. **0
whole-book answers, 0 refusals, 0 wrong.**

In the six-repeat control matrix it is 6/6 CORRECT, improved from the frozen
baseline's 5 CORRECT / 1 WRONG.

**Unavailable:** controlled refusal in all five injected modes, whole-book count
0. This is the case the build was commissioned on — during the credit outage,
20 of 20 runs returned a whole-book answer — and it is closed.

## 13 · Q10B

**Healthy:** CORRECT 6/6 in the control matrix and CORRECT in the full Opus bank
run — 8 groups, stage × size, which is the independently established truth.

**Unavailable:** controlled refusal in all five injected modes. No stage-only
answer is produced, and no claim is made that a clean ledger makes the
stage-only answer safe: bare "size" has no governed owner, so the ledger cannot
see its loss, and the refusal comes from the availability rule instead.

**Boundary, stated plainly:** that guarantee holds *when the arm is enabled*.
With the arm switched off entirely there is no augmentation to be unavailable,
and deterministic Q10B is a known WRONG — 5 stage-only groups — which coverage
cannot detect. See residual risks.

---

## 14 · Seven CR4 recoveries, repeated runs

Healthy model, 6 independent invocations each, 42 calls:

| Q01C | Q02B | Q03A | Q03C | Q05C | Q16B | Q17C |
|---|---|---|---|---|---|---|
| 6/6 CORRECT | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |

## 15 · Five former Opus regressions, repeated runs

| Q23A | Q23C | CFO74 | CFO63 | CFO65 |
|---|---|---|---|---|
| 6/6 CORRECT | 6/6 | 6/6 | 6/6 | 6/6 |

Every one of the 31 control questions matches the frozen control baseline
exactly, with one improvement (Q16B 5/6 → 6/6). 180 of 180 calls reached the
model; 180 of 180 recorded `claude-opus-5`.

## 16 · 24 CR4 final disposition

| | CORRECT | FALSE_REFUSAL | WRONG |
|---|---:|---:|---:|
| deterministic | 1 | 20 | 3 (Q04C, Q17C, Q19A) |
| **Opus** | **8** | **14** | **2 (Q04C, Q19A)** |

Of the 20 deterministic refusals, 16 are the pre-existing safe refusals and 4
are new coverage refusals of answers the book proves were wrong. Zero recoveries
regressed.

## 17 · 75 bank and CFO 91

Both arms were run at `6a2f224` and again at final HEAD, so the movement below
is this sprint's and nothing else's.

| deterministic | CORRECT | FALSE_REF | NO_TRUTH | TRUE_REF | WRONG |
|---|---:|---:|---:|---:|---:|
| at `6a2f224` | 117 | 22 | 4 | 15 | **8** |
| **at final HEAD** | **117** | 26 | 4 | 15 | **4** |

Four movements, all WRONG → controlled refusal: Q03A, Q05C, Q07B, Q16B. Nothing
else moved.

| Opus arm | CORRECT | FALSE_REF | NO_TRUTH | TRUE_REF | WRONG |
|---|---:|---:|---:|---:|---:|
| at `6a2f224` | 125 | 20 | 4 | 15 | **2** |
| **at final HEAD** | **124** | 21 | 4 | 15 | **2** |

**One movement, and it is not the coverage gate.** CFO40, "For loans with
borrower age above 75, balance by region", hit a transient call failure in both
runs; at `6a2f224` the deterministic reading was executed anyway and happened to
be right, and now it refuses. Re-run 8 times immediately afterwards: 8/8
answered correctly.

This is worth stating plainly rather than dressing up: **on the healthy Opus
path the coverage gate fires zero times**, because the model recovers every
concept it would otherwise catch. Opus WRONG was already 2 before this sprint
and is 2 after. The gate's measured value is on the deterministic path and under
model unavailability — which is exactly where the risk was.

Remaining WRONG under Opus: Q04C and Q19A, both pre-existing and both out of
scope by the brief.

## 18 · Six pipeline questions

| question | deterministic | Opus |
|---|---|---|
| Summarise the current pipeline. (Q10A) | reconciles `pipeline`, 8 loans | same |
| Give me an overview of the pipeline by size and stage. (Q10B) | WRONG (stage only) | **CORRECT** (8 groups) |
| What does the current pipeline look like? (Q10C) | safe refusal | safe refusal |
| What funded balance should we expect once the pipeline flows through? | answered, `funded+pipeline` | answered |
| Show forecast balance by expected completion month. | answered, `funded+pipeline` | one transient availability refusal; 6/6 answered on re-run |
| How much do we have at offer and how much is likely to complete? | answered, `pipeline` | answered |

## 19 · Full 1,446 surface

| | answers | refusals | arm reached | model |
|---|---:|---:|---:|---|
| deterministic | 840 | 606 | — | — |
| **Opus** | **848** | 598 | 1436 / 1446 | `claude-opus-5` on all 1,446 |

Deterministic movement is exactly the 11 coverage refusals. Opus movement against
the previous sprint's baseline is 17, every one attributed:

* **7 availability refusals** — transient call failures;
* **7 coverage refusals** — the U1/U3 cases the model does not recover;
* **1 answered → refused with `applied`** — "pipeline by stage for broker Alpha":
  the arm applied the Broker filter, the executor found no matching rows and
  declined. The pipeline holds exactly one broker and the deterministic answer
  silently dropped "Alpha", so this refusal is correct;
* **2 refused → answered** — both stochastic arm behaviour, neither wrong.

**0 new WRONG on the Opus surface.**

Merge audit over the 166-question Opus bank: 206 findings agreed with the
deterministic claim, 36 filled an empty slot, and **33 declined** — 30 because a
person had already chosen the slot, 1 because the field was already placed in
another role, 1 because the slot carried a governed default, 1 because the slot's
provenance was never recorded. Every fill landed on a canonical registry field
chosen by the registry from a term the model proposed. **0 deterministic claims
overwritten, 0 dataset slots filled, 0 model-selected canonical fields.**

## 20 · Remaining known wrong answers

| | arm | why | in scope? |
|---|---|---|---|
| Q04C | both | right population, wrong output grain — coverage is not built to see shape | no, U4 by the brief |
| Q19A | both | the intent owner reads a two-period delta and a window progression identically | no, reverted in a previous sprint |
| Q10B | deterministic only | bare "size" has no governed owner | no, out of scope by the brief |
| Q17C | deterministic only | bare "LTV" in that phrasing is named by no axis owner | newly visible; no vocabulary added |

---

## 21 · Capability progress

**New analytical capability: 0.** No calculation, methodology, route or metric
was added in this sprint, and none in the programme this report closes.

**Existing capability newly reachable through natural language: 8.** The seven
CR4 recoveries (Q01C, Q02B, Q03A, Q03C, Q05C, Q16B, Q17C) plus Q10B. Each is a
deterministic calculation the estate already performed and could not be asked
for in that wording; the augmentation arm proposes the term, the registry binds
the field, and the merge fills a slot the deterministic parse left empty. On the
full surface this shows as 127 questions where a proposal bound and applied.
None of it is new analysis.

**Existing capability unblocked — falsely refused or shadowed, now executing: 4.**
Q10A (the pipeline summary was answering from the funded tape and now reconciles
to `pipeline`); Q22B and Q22C (the lens narrowing ran and was undisclosed, so the
receipt refused a correct answer); Q19A's composition question, which this sprint
proved was being refused for a record the adapter could not read. Q23A also moved
from false refusal to CORRECT under the arm.

**Safety-only — no additional reach, silent wrong behaviour converted to
controlled refusal: 15 + all-of-availability.**

* 15 answering questions across 1,612 lost a governed concept the estate names
  and now refuse naming it. 14 were verified wrong against the book; 1 is an
  estate limitation the refusal handles conservatively. On the 166-question
  acceptance bank this is deterministic WRONG 8 → 4. On the healthy Opus path it
  is worth nothing — the gate fires zero times there — and that is not a defect:
  the model already recovers those concepts, and the gate exists for the paths
  where it does not.
* Every unavailable augmentation call now refuses instead of executing a reading
  that may be narrower than the sentence. Measured cost: 12 transient failures in
  1,842 live calls, **0.65%**.
* Five disclosure gaps and one reader gap closed, which changed no answer at all
  but is what made enforcement possible without refusing correct work.

The reachability figures are not new capability and are not described as such.

---

## 22 · Residual risks

1. **With the arm enabled, ~0.65% of questions will be refused for transient
   provider reasons.** Measured over 1,842 live calls. Each is recoverable by
   asking again — every transient refusal investigated answered correctly on
   re-run (CFO40 8/8, the forecast-by-month question 6/6). This is the price of
   correct-or-refuse and it is the largest single behavioural change here.
2. **The availability rule refuses everything, including trivially deterministic
   questions.** That is deliberate and it is the brief's instruction, because no
   independent completeness proof exists. If one is ever wanted, the candidate is
   named in §10: a contract with no empty slot cannot be changed by any proposal.
   It was not built.
3. **The arm is off by default.** `MI_AGENT_CONCEPT_MERGE` defaults to `off` and
   a key alone does not enable it, so a default deployment sees neither the new
   reach nor the new refusals. The Q10B and Q17C deterministic residuals are
   live in that configuration and coverage cannot see either.
4. **Coverage is concept-founded and cannot see shape.** Q04C has the right 24
   loans summing to the right figure in the wrong artifact shape, and no
   disposition will ever fire on it.
5. **`_governed_context` returns early if `metadata` is not a dict**, bypassing
   both gates. Not reachable in practice — the ledger was present on 1,612 of
   1,612 census envelopes — but it is a fail-open path and is recorded rather
   than changed, since the brief restored the gate rather than redesigning it.
6. **The coverage refusal does not set `controlledRefusal`.** The restored gate
   is byte-identical to the prototype and this was not changed. A consumer that
   distinguishes controlled refusals from errors by that flag will read these
   four as errors. The availability refusal does set it.

---

## 23 · Recommendation

Every hard gate in the brief is met:

| gate | result |
|---|---|
| frozen manifest exactly 85 | ✅ at every phase boundary and at final HEAD |
| no correct answer refused for missing disclosure | ✅ — and two such refusals from the prototype were found and removed |
| Q16B whole-book widening = 0 | ✅ 0 of 20 healthy, 0 of 5 unavailable modes |
| forced model-unavailable suite = 0 WRONG | ✅ 0 of 155 (31 × 5 modes) |
| Q10B cannot silently degrade to stage-only when augmentation is unavailable | ✅ controlled refusal in all five modes |
| 0 new WRONG / SILENT, 0 new WRONG / DISCLOSED | ✅ both arms measured at `6a2f224` and at final HEAD: deterministic WRONG 8→4, Opus WRONG 2→2 |
| 0 must-refuse → answer | ✅ 18 of 18 healthy, 15 of 15 unavailable |
| 0 model-selected canonical fields | ✅ 36 fills, every field chosen by the registry |
| 0 deterministic claims overwritten | ✅ 33 declines, 0 overwrites |
| 0 invented required metric/period | ✅ governed defaults declined 1/1 |
| 0 dataset substitution | ✅ 0 dataset slots filled; 0 pipeline questions reconciled to funded alone |
| seven recoveries available with healthy model | ✅ 6/6 each, 42/42 |
| five former regressions still fixed | ✅ 6/6 each, 30/30 |
| deterministic arm unchanged but for authorised safety refusals | ✅ 162 of 166 bank envelopes identical; the sweep corpus byte-identical; 11 coverage refusals on the 1,446 surface and nothing else |
| every remaining known WRONG identified | ✅ §20 |

The two guarantees the programme set out to establish now hold and are measured:
when the model is healthy Trakt gets the reach already demonstrated, and when it
is unavailable Trakt answers less but never answers a different question; and any
governed concept the reader states and the estate can name either survives into
execution or causes a controlled refusal.

FREEZE FOR PRODUCTION
