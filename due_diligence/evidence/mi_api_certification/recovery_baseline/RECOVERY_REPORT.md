# MI Query Agent V1 — Semantic Recovery Sprint

_Phases 0–9 complete. Phase 10 (deploy one SHA, rotate the token, run the frozen
135 live) is NOT started and is blocked on the user — see section L._

Every number here was measured. Where something is unproven, it says so.

---

## A. Recovery baseline (Phase 0)

- Frozen bank sha256 `3c20668904c8855605f721f3a7feec35b26b228955e5f6e85e683318ac535e94`
  — unchanged; no question, expectation or threshold was edited at any point.
- Harness defects E1–E8 closed (harness only), then the corrected live baseline
  at deployed SHA `ea8c65b…` (harness 191fa31, run 34323550096):

| | |
|---|---|
| canonical cases | 45 |
| EXACT | 16 |
| WRONG | 18 |
| UNDELIVERED | 53 |
| CRASH | 0 |
| NOT_EXECUTABLE | 3 |
| paraphrase consistency | 26 / 45 |
| independent truth agreement | 34 / 91 |

## B. Architectural invariants (Phase 1)

Seven invariants, written BEFORE any consolidation (commit `12ccff6`):
**31 red, 6 already holding.** Final state: **66 green** across
`test_invariants.py`, `test_model_layer.py` and `test_phase3_truth.py`.

## C. Consolidations (Phase 2–3), one commit each

| Commit | Consolidation | Old owners → new owner |
|---|---|---|
| `8e386ce` | **G2** one normaliser | 9 private punctuation rules → `question_interpretation.normalise` |
| `05dbe95` | **G1** one period-pair owner | bridge index arithmetic (falling to the EARLIEST), movement `periods[-1-span]`, compare token table, 4 period readers → `analytical_plan.resolve_period_pair` over the governed `period_change.periods.resolve_periods` |
| `f6fa1f8` + `8a1b419` | **G3** owner-aware claiming + the claims ledger | a closed exemption list, and a guard that re-read the raw sentence → each owner's `claims_word`/`temporal_word`/`movement_word`/`status_word`/`window_word`, recorded once in `mi_agent.semantic_claims` |
| `ae916af` | **G7-lite** the final interpretation includes the merge | a fallback COUNT recorded as a filled subject → subject EMPTY unless asked; a model fill carries its own aggregation and `model_inferred` provenance |
| `10ccb25` | **CAP** capability ownership | none (a generic measure collision) → `mi_agent.capability_ownership` + `borrowing_base.capacity` |
| `a4712b7` + `9abbc28` | **G6** one geography field owner | **five** private column orders → `mi_geography.region_field` / `code_field` / `region_candidates`, harmonised tier first |
| `2ef7fb3` | **P0-E / P1-B** | the measure vocabulary read the noun in "balance-weighted"; a parallel question lost its second measure → `statistic` owns the weighting grammar and asks the measure owner; the slot owner reads parallel interrogatives |
| `2f25afd` | Phase 7 vertical suites | — |

## D. Semantic census (Phase 6) — machine-checked

**What "the corpus" is here.** The sprint names 882 questions. The estate's
own corpus (`question_interpretation/stage2_corpus.json`) carries 939 rows,
which contain **843 distinct questions** — the census runs the distinct set, so
a question asked twice cannot count as two movements. The frozen bank's 135 are
censused separately and never merged with it.

`recovery_baseline/census/classify_movements.py` asserts that every question
that moved between the baseline and HEAD moved in a NAMED step for a NAMED
reason. It exits non-zero on any unexplained movement.

```
corpus: 843 questions, 210 moved     bank: 135 questions, 72 moved
  G2         2                         G2         5
  G1         1                         G3         8
  G3         2                         G7-lite   62
  G7-lite  207                         CAP        2
                                       P3         1
  every movement is attributed to a named step. None unexplained.
```

The 207/62 G7-lite movements are **contract fields only** — 26 asked counts
gaining `explicit_user` provenance and 36 unasked counts emptying a subject
they never should have filled. Zero metric, aggregation, filter, dimension or
route movement among them (checked field by field, 0 unclassified).

### Defects the census caught — mine, fixed before landing
1. **The weighting pattern was one word too general.** `[a-z]+ weighted` matched
   *the* weighted and *plus* weighted and *show* weighted, turning a SUM into a
   weighted average. The first fix grew a list of words that cannot be weights;
   the census taught it three of them, one wrong answer at a time. The rule is
   now that a weighting names a MEASURE, and the measure owner is asked.
2. **The borrowing-base claim over-reached.** With `lend` in its verb list,
   "the front BOOK compare with our older LENDING FROM a risk perspective" was
   claimed as a borrowing-base question.
3. **G6 dropped the harmonised region column** (found by
   `test_three_spellings_of_one_region_are_one_bar`, not by the census — the
   demo book carries no harmonised columns). This would have re-opened the
   defect where one region spelled three ways is tested as three 25% bars and a
   40% limit reported COMPLIANT against a 75% concentration. Fixed in `9abbc28`.

### One error in the record, corrected here
The `ae916af` commit message states 39 frozen-bank movements for G7-lite. The
true count is **62** (26 + 36 above). The classification in that message is
right; the count was estimated from a truncated listing instead of counted.

### The classified service defects, and where each was fixed

The 19 SERVICE_DEFECT rows from the earlier classification fall into five
groups. Each was fixed by removing an OWNER, never by special-casing a
question:

| Group | Fixed by |
|---|---|
| **P0-A** a period word read as a categorical filter value ("prior", "previous", "live", "new", "exited") | G3 — the claimant asks the period, status, movement and seasoning owners |
| **P0-B** a point-in-time question about a PAST period answered as at the current one, or read as a trend | the claims ledger — `period_as_at` refuses, and a LEVEL is no longer re-read as MOVEMENT by a second raw reader |
| **P0-C** explicit period language not honoured | G1 — one period-pair owner; a named period the frames lack refuses instead of falling to the earliest |
| **P0-D** a multi-measure request narrowed or refused | G2 (the hyphenated measure now binds, so C11 reads both) and P1-B (a coordinated slot is named, not dropped) |
| **P0-E** a weighting qualifier read as a measure | `statistic` owns the weighting grammar and asks the measure owner |

**P1-A paraphrase reachability** is addressed structurally — one owner per
concept is what makes paraphrases converge, and the model may only bind
REGISTERED concepts (proved by P3 in section E). Its measurement is the live
bank's paraphrase-consistency count, which needs a live run.

## E. Model-layer proof (Phase 4)

`tests/semantic_recovery/test_model_layer.py` — six properties, on REPLAYED
proposals through the real arm and the real contract, no network:

| | |
|---|---|
| P1 fill-only | an empty subject is filled, disclosed `model_inferred`, and takes the measure's own aggregation |
| P2 never overwrite | a reader-filled subject stands; an asked count stands; the disagreement is reported |
| P3 registered concepts only | an unregistered term is rejected — never a nearest match |
| P4 no ambiguity by preference | a value two governed fields claim is rejected as ambiguous |
| P5 unavailable is a refusal | the arm reports `PROPOSAL_UNAVAILABLE` and the service refuses on it — never a silent downgrade |
| P6 determinism | the same replay yields the same contract twice |

## F. Focused truth tests (Phase 5)

`tests/semantic_recovery/test_phase3_truth.py` — the weighting qualifier, the
coordinated measure list, and the ledger's refusals, all on wording authored
here rather than taken from the bank.

## G. Broad non-regression (Phase 7)

**Vertical suites** (`vertical_suites.py`, 48 authored questions, five
verticals, through the real app): baseline vs HEAD — **48 compared, 5 moved,
every one an intended remediation, 0 regressions.**

| | before → after |
|---|---|
| "How much collateral value are we able to borrow against?" | £12.9m VALUATION TOTAL, ok=true → the governed no-facility refusal |
| "How much can we draw against the facility?" | "36 loans · £5.4MM", ok=true → the same governed refusal |
| "…the funded balance at the previous reporting date?" | refused for the WRONG reason ("no loans match filter 'previous'") → "it asks about an earlier reporting period" |
| "…against the large-loan limit" | all 12 limit tests → the 2 large-loan tests |
| the bridge's insufficient-period refusal | evolution's sentence → the period owner's |

**Targeted regression per step** — every test file touching each changed owner,
run at HEAD and at the untouched baseline, compared by test id:

| Step | files | new failures |
|---|---|---|
| G2 | 391 tests | 0 |
| G1 | 40 files | 0 |
| G3 | 41 files | 0 |
| G7-lite | 154 files | 0 |
| CAP / G6 / G6-completion | 135 + 20 files | 0 |

**Full suite** (`tests`, `mi_agent_api/tests`, `mi_agent/tests`,
`question_interpretation/tests`): see section I — it is the last gate and its
result is reported there, not assumed here.

## H. Ownership review (Phase 8)

Every duplicate owner named in the architecture review is gone from live code:

| Deleted | live references |
|---|---|
| `_ITL3_FIELDS`, `_REGION_FAMILY`, `REGION_COLUMNS`, `risk_limits._REGION_COLUMNS` (as a chooser) | 0 |
| `DEFAULT_SPAN_PERIODS`, `span_periods`, `bridge_window_periods`, `funded_bridge(window_periods=)` | 0 |
| `temporal_compare._RELATIVE_PRIOR` | 0 |
| the claimant's exemption list | 0 |

Net: **+1,868 / −253** lines of Python across 9 commits, of which ~570 are the
three new owner modules (`normalise`, `semantic_claims`, `capability_ownership`
+ `borrowing_base.capacity`) and ~640 are tests.

## I. Go / no-go (Phase 9)

The sprint's targets, each answered by a measurement rather than a score:

| Target | Verdict | Evidence |
|---|---|---|
| zero silent semantic substitutions | **MET for every mechanism found** | the two wrong answers in the vertical suites are gone; the substituting owners are deleted, not patched |
| zero silently lost explicit filters / dimensions / periods / scopes | **MET** | a named period the frames lack now REFUSES (G1); a claimed lending window, pipeline status or earlier period refuses on the point-in-time path (G3 ledger); a dropped measure slot is named (P1-B) |
| zero wrong dataset / measure | **MET for the mechanisms found** | capability ownership (CAP), the weighting qualifier (P0-E), the one normaliser (G2) |
| equivalent phrasing → equivalent semantics | **MET where a normaliser decides it** | hyphenated and spaced spellings converge, proved end to end on the routed path |
| unsupported long tail refuses safely | **MET** | "churn" is still unclaimed and still refuses — no bank-copied alias was added to make a question pass |
| certified calculations unchanged | **MET** | 0/843 and 0/135 census movements for G1, G6 and G6-completion; the G7-lite movements are contract fields only, with zero metric/aggregation/filter/route movement |
| P0 counts zero | **NOT ESTABLISHED** | the P0 counts are a property of the LIVE run against a deployed SHA. This sprint has not run one — see section L |

**DEPLOY VERDICT: NO — not yet, and not because the work failed.**
The sprint's own rule is to propose a deploy only when all P0 counts are zero,
and P0 counts come from the live acceptance. The live acceptance cannot run:
MI_BEARER is expired and no candidate SHA is deployed. Deploying now would
change production with no acceptance run able to validate it, which is the one
outcome worse than not deploying.

### The six verdict lines, as far as this sprint can honestly answer them

| | |
|---|---|
| `SEMANTIC_OWNERSHIP_CONSOLIDATED` | **YES** — 8 consolidations, every duplicate owner deleted, 0 live references remaining |
| `INVARIANTS_HOLD` | **YES** — 66 tests, written before the code |
| `NO_UNEXPLAINED_SEMANTIC_MOVEMENT` | **YES** — machine-checked over 843 + 135 questions |
| `BROAD_NON_REGRESSION` | **YES for every changed surface** — 0 new failures against the untouched baseline in every targeted file set; the whole-estate suite is the one gate still running at the time of writing |
| `MI_QUERY_AGENT_V1_LIVE_READY` | **NOT ESTABLISHED** — requires a live run |
| `DEPLOY_RECOMMENDED` | **NO** — blocked on a token and a deploy, both the user's to give |

## J–M. Deployment and the live run (Phase 10)

**NOT STARTED, and blocked on the user.** The sprint requires the token to be
rotated IMMEDIATELY BEFORE the run, a single candidate SHA deployed and proved,
and the frozen 135 executed exactly once against it. Three of those steps are
not mine to take:

1. **MI_BEARER is expired.** The last measured state was
   `EXPIRED at 2026-09-08T22:20:16Z`. The harness refuses to start without a
   live credential and aborts mid-run on a 401 — by design, so a partial run is
   never scored as a result.
2. **Nothing is deployed.** These nine commits are on
   `claude/borrowing-base-mi-query-cvplz0` and have not been pushed. No deploy
   has been triggered and no SHA has been promoted.
3. The live acceptance runs in GitHub Actions (egress is blocked here), so the
   run must be dispatched against a deployed SHA.

**What is needed to finish:** push the branch, deploy ONE candidate SHA, rotate
MI_BEARER, then dispatch the frozen bank once. The harness already asserts the
bank hash, the deployed SHA, and the credential's remaining lifetime before it
starts, and aborts rather than half-answering.
