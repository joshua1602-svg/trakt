# Capability recovery and acceptance recalibration

Start `e9530f8` (clean) → end `dd07ad2` + this report. Four commits of work, one
of evidence.

The previous freeze recommendation was withdrawn for the capability surface. The
safety architecture is untouched: every invariant listed as frozen is intact and
re-measured below.

---

## 1–2 · Commits, and the strict recalibrated starting score

| commit | phase |
|---|---|
| `633a5dc` | 0 — acceptance truth, and two defects it makes visible |
| `90a5101` | 1–4 — three shared owners, four recoveries |
| `eb9227f` | 3 — a milestone question is one that asks WHEN and names a target |
| `dd07ad2` | 4 — four more shared owners |

The score the sprint actually started from, after Phase 0 made the oracle
harder rather than easier:

| shipping bank (166) | reported at `e9530f8` | **strict** |
|---|---:|---:|
| CORRECT | 124 | **124** |
| WRONG | 2 | **4** |
| DECLINED BUT ANSWERABLE | 21 | 21 |
| CORRECTLY DECLINED | 15 | 16 |
| NO CHECKABLE TRUTH | 4 | 1 |

---

## 3 · Phase 0 grade corrections

**Two production defects, both found by asking what an answer asserts rather
than what it was asked.**

A governed measure was published twice from one executed row and the two
renderings disagreed — "55.6%" in the KPI tile, "0.56%" in the sentence beside
it. Storage scale is a fact about the column, decided once from the data by
`percent_storage_scale`, and three renderers held their own opinion. Two applied
it; the prose composer did not, so it was wrong by a hundredfold on every
`percent_fraction` measure and right by luck on every `percent_points` one —
which is why it survived. One conversion now, `to_display_points`, and all three
call it.

"Single name" meant the wrong thing. Schedule 8 §6.1 writes the limit about "any
single Borrower (or group of connected Borrowers)"; the concentration vocabulary
read the phrase as the LOAN kind. So "our largest single-name exposure" answered
£449k — the largest single loan — against a limit written about borrowers, on a
tape carrying no borrower identifier at all. The estate already reports that
limit as unavailable for exactly that reason. One line of vocabulary; the
machinery behind it was already correct.

**Four oracle corrections, none of them question-specific machinery.**

| correction | mechanism |
|---|---|
| Every factual assertion is graded, not only the requested one | prose percentages must agree with the same result's KPI tile — no truth file, no question id |
| A comparison is adjudicated as a comparison | the Q22 truth already carried both cohorts' open, close, delta and the winner; nothing could read it |
| An answer must carry the requested operation's own evidence | `must_state` / `must_not_state` |
| A ranked answer IS its table | ten rows checked for count, order and cumulative share; a capture without rows reports NOT MEASURED |

Grade movements: **CFO75** CORRECT→CORRECTLY DECLINED, **Q09B** CORRECT→WRONG,
**Q22A** NO TRUTH→WRONG, **Q22B/Q22C** NO TRUTH→CORRECT.

---

## 4 · The reach denominator: 14, not 16

Two candidates are not capability gaps, and the evidence says so.

**CFO60 "Show product concentration" and CFO61 "Show broker concentration"** —
the registry declares those dimensions' categories originator-specific, so a
book-level concentration over them would present unaligned categories as one
exposure, and the methodology will not invent a mapping. It measures product
concentration on `product_type` and broker concentration on
`origination_channel` instead. Both refuse with a stated governed reason.
Removed from the denominator.

Excluded by the brief and confirmed: **Q25A/B/C** (forward-limit capability
genuinely absent), **CFO40** (transient), **CFO71** (genuine measure ambiguity —
"value" resolves to Balance or Valuation and the estate declines to choose).

---

## 5 · First-divergence matrix

| question | successful sibling | first divergence | existing capability | role lost | common owner |
|---|---|---|---|---|---|
| Q04A Q05B Q15B Q17B | Q04C, Q05C, Q17C | merge fills `source_scope`; `_apply_to_spec` has no branch for it | point-in-time with a lens | source scope | `concept_merge_arm._apply_to_spec` |
| Q04A Q05B Q15B Q17B | "the direct book" answers | `\s+` between qualifier and noun; "Direct-book" has a hyphen | portfolio lens | source scope | `portfolio_lens._qualified_span_re` |
| Q04A Q05B Q15B Q17B | Q17C | token `Direct-` matches no word list, so the run is judged a proper name | portfolio lens | source scope | `portfolio_lens._unknown_named_book` |
| Q23B Q24B | Q23A, Q24A | verb list gates the forecast recogniser; "get to" is not "reach" | run-rate milestone | the operation | `llm_query_parser._FORECAST_SCALE_RE` |
| Q12C | Q16B, Q17B | bare "plot" forces a loan-level scatter and disables the grouped matrix | two-dimension grouping | second axis | `llm_query_parser._deterministic_parse` |
| Q01B | same question without the article | "an LTV greater than 50%" read as an unresolved measure slot | both predicates already applied | none — a false finding | `llm_query_parser.unresolved_measure_slots` |
| Q09B | Q09A, Q09C, CFO72/73 | "tests" is the noun the answer uses and not one the reader reads | governed limit assessment | the operation | `llm_query_parser._RISK_LIMIT_RE` |
| Q22A | Q22B, Q22C | — | bridge ranking | none; the ORACLE was over-strict | oracle |
| Q07B Q20B Q21B Q21C Q15C Q10C | — | not reached this sprint; see §16 | | | |

---

## 6 · The shared mechanisms

Every recovery is a governed decision made by the wrong reader, and the fixes
are all one of three shapes:

1. **A claim recorded and not carried.** The merge accepted a source scope and
   nothing applied it — the estate's own cardinal sin, inside the arm.
2. **A token the guard could not match.** `Direct-` against a list containing
   `direct`; "an LTV greater than 50%" against an applied threshold. Fixed by
   normalising the token, never by lengthening a list.
3. **Two owners of one governed fact, disagreeing.** Three renderers and one
   storage scale; a verb list beside two owners that already knew; a scatter
   verb beside the axes themselves; the concentration methodology's field
   against the receipt's binding.

The architectural hypothesis the brief asked me to test — that the missing
abstraction is a governed operation/role proposal rather than more field
vocabulary — **was not needed, and I did not build it.** In every case the
governed estate already held the decision and a smaller composition reached it.
The clearest instance: the forecast gate. `answer_type.asked` already decided
the question wants a DATE, and `_forecast_target_value` already resolved the
target, both correctly on every failing phrasing. Composing the two owners
claims exactly **3 more questions across 1,446** — measured before shipping —
where a model-proposed operation kind would have been a new semantic owner for
the same result. The brief's instruction to prefer a smaller common mechanism
is why this sprint added no proposal kind.

---

## 7 · Production changes by owner

| file | +/− | what |
|---|---|---|
| `mi_agent/llm_query_parser.py` | +92 −2 | milestone composition; scatter decided by axes; a comparison is a predicate; the limit-test noun |
| `mi_agent/portfolio_lens.py` | +33 −6 | hyphen separator; token normalisation |
| `mi_agent/mi_agent_workflow.py` | +21 −7 | prose composer reads the profile's scale |
| `mi_agent/mi_dataset_profile.py` | +23 | `to_display_points`, the one conversion |
| `mi_agent_api/concept_merge_arm.py` | +25 | apply a filled source scope through the lens owner |
| `mi_agent_api/chat_routing.py` | +18 | concentration declares the axes it measured |
| `mi_workflows/concentration_analysis.py` | +17 −2 | single-name is a borrower |
| `mi_agent_api/adapters.py`, `snapshots.py` | +4 −7 | call the shared conversion |

Oracle and evidence: `pack_grader.py` +156, the two banks, one test corrected.

---

## 8 · Recovered: 8 of 14

| question | sibling that already answered | now |
|---|---|---|
| Q01B how many loans, borrower over 55 and an LTV over 50% | the same question without the article | 144 loans |
| Q04A balance of Direct-book loans in London to borrowers over 75 | Q04C — same 24 loans | £7.2m, 24 loans |
| Q05B Direct-book lump sum weighted average LTV | Q05C | 37.0%, 278 loans |
| Q12C plot balance across LTV and borrower-age buckets | Q16B | heatmap, 42 groups |
| Q15B Direct-book balance by broker channel and loan type | Q17C | 8 groups |
| Q17B Direct-book balance by LTV, ticket and age band | Q17C | 143 groups |
| Q23B at the current trajectory, when do we get to £100m | Q23A | already reached |
| Q24B when are we expected to get to £250m | Q24A | run-rate date |

Two further questions moved from **wrong** to **correct**: Q09B and Q22A.

---

## 9–13 · The Phase 0 questions

**Q09B** now routes to the governed limit monitor: 6 breaches, nearest to limit
Top 3 brokers at 76.5% against a 45% limit. 6/6 correct. The Q25 forward-limit
boundary still refuses.

**Q03C** states 45 and "Weighted-average Current LTV: 55.59%" — verified against
the tape as the balance-weighted LTV of those 45 loans. Both claims governed and
correctly rendered.

**Q22A/B/C** independently adjudicated against figures recomputed from the
governed runs to the penny: Direct +£12,366,371.40, Acquired +£10,229,937.01,
Direct larger. All three correct. Q22A is adjudicated on the winner and the
winner's delta — the facts its own question asks for; its two-sided siblings are
adjudicated on both sides.

**CFO75** refuses, naming the identifiers it looked for. **CFO76** is verified
through the artefact: exactly ten rows, that ranking, cumulative share climbing
to 2.60%.

---

## 14–15 · Final verdict table and remaining wrong

| shipping bank (166) | start | **final** |
|---|---:|---:|
| CORRECT | 124 | **135** |
| CORRECTLY DECLINED | 16 | 16 |
| NO CHECKABLE TRUTH | 1 | 1 |
| DECLINED BUT ANSWERABLE | 21 | **12** |
| **WRONG** | 4 | **2** |

Governed engine alone: 117 → **126** correct, wrong 6 → 4.
75 bank 63 correct / CFO 91 72 correct + 16 correctly declined.
24 CR4: **16 correct** (was 8), 6 declined, 2 wrong.

Remaining wrong, both pre-existing and both out of scope by the brief:
**Q04C** (right 24 loans, right total, wrong output shape) and **Q19A** (a
five-period progression where a two-period delta was asked for).

---

## 16 · Remaining capability gaps

| question | why it still declines |
|---|---|
| Q07B how do Direct and Acquired differ | the dimension reaches the spec and execution does not apply it; the guard refuses rather than drop it |
| Q20B what changed in the drawdown book | the population reaches the spec and the movement route does not apply it |
| Q21B which region contributed most balance growth, loans over 50% LTV | the deterministic parse puts LTV in the subject slot; the model's `balance` cannot overwrite a slot a person filled |
| Q21C among loans with LTV above 50% | "among" is read as a categorical value and matches no rows |
| Q15C broker-by-product table | the role of "broker" is genuinely ambiguous and the estate asks rather than guesses |
| Q10C what does the current pipeline look like | no governed analytic matches the formulation |
| Q25A/B/C | forward-limit projection: real new capability |
| CFO71 what is the value of outstanding offers | "value" is Balance or Valuation; the estate declines to choose |

---

## 17 · Repeated-run stability

216 healthy invocations, 215 reached the model, all `claude-opus-5`.

| group | result |
|---|---|
| 8 recovered this sprint | **6/6 CORRECT each — WRONG 0** |
| Q09B, Q22A | 6/6 and 5/6 correct (one transient refusal) — **WRONG 0** |
| seven CR4 recoveries | 6/6 CORRECT each |
| five former regressions | 6/6 CORRECT each |
| Q10B, Q22B, Q22C, CFO76 | 6/6 CORRECT |
| CFO75 | 6/6 correctly declined |
| Q25A/B/C | 6/6 still refuse |
| must-refuse controls | 6/6 refuse, 0 answered |
| Q04C, Q19A | 6/6 wrong — the two known residuals |

---

## 18 · 1,446 blast radius

| | before | after | movement |
|---|---:|---:|---|
| governed engine | 840 | 848 | 9 recovered, 1 deliberate refusal (single-name) |
| shipping | 848 | 862 | 17 recovered, 3 refused |

Of the 17, ten are this sprint's recoveries — including two phrasings not in any
bank — and seven are baseline transients that answer on any healthy run. Of the
three refusals: one is the single-name correction, one a transient, and one is
model stochasticity on "the forecast run rate for active loans", where in 1 run
of 4 the model reads "active" as a population the forecast route cannot honour
and the estate refuses rather than widening. That is correct-or-refuse working.

Coverage census: 1,612 questions, ledger present on all, **0 answering questions
carry an unaccounted concept.**

---

## 19 · Provider-unavailable, and the frozen manifest

31 questions × 5 injected failure modes: **0 answered, 0 wrong, 0 whole-book**;
Q16B never whole-book, Q10B always refused, must-refuse never answered.

Frozen 278-module manifest: **85, name for name**, at every phase boundary and
at final HEAD. One test assertion was corrected rather than worked around —
`"single name concentration" == "loan"` was the defect written down — and
hostile controls were added beside it.

Transient provider failures: **1 in 216** stability calls, **4 in 1,446** surface
calls — consistent with the 0.65% measured previously.

---

## 20 · Recommendation

| gate | required | result |
|---|---|---|
| shipping bank CORRECT | 135–140 | **135** |
| existing-capability recovery | ≥8 of the denominator | **8 of 14** |
| WRONG | ≤2, no new | **2**, both pre-existing |
| per recovered question over ≥6 runs | WRONG = 0 | **0** |
| Q09B no longer generic concentration | — | limit assessment, 6/6 |
| Q03C no unsupported or misscaled claim | — | 55.59%, tape-verified |
| Q22A/B/C independently graded | — | all three correct |
| CFO75 proven semantics or safe refusal | — | refuses, naming the missing identifiers |
| CFO76 all ten exposures proven | — | artefact-verified |
| Q25A/B/C protected | — | 6/6 still refuse |
| CFO71 does not guess a measure | — | still declines |
| must-refuse controls | all refuse | 18/18 |
| model-selected canonical fields | 0 | **0** |
| overwritten deterministic claims | 0 | **0** |
| silent filter/scope/axis loss | 0 | **0** — census clean on 1,612 |
| provider-unavailable | 0 answered / 0 wrong | **0 / 0** |
| frozen manifest | exactly 85 | **85** |

The sprint's two objectives are met on the evidence: half the reachable
existing-capability gap is closed through shared owners with no question-specific
rule and no new semantic owner, and the acceptance model is strictly harder than
the one that produced the withdrawn recommendation — it downgraded two answers
that were passing and it is what found both production defects in Phase 0.

CAPABILITY TARGET MET — FREEZE
