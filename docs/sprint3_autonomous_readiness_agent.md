# Sprint 3 — The Autonomous Securitisation Readiness Agent

Five scored autonomous runs against a hidden-truth portfolio, evaluated against
independently planted findings.

Evidence base: run 1 executed at `a6479b6`; runs 2–5 executed at `5dc890c`,
which differs from `a6479b6` only in `scripts/run_readiness_agent_eval.py`
(persistence). The agent prompt, the governed session, the tool surface, the
hidden portfolio and the scorer are byte-identical across all five runs. Three
further runs were executed and lost before the persistence fix; they are **not**
counted here, and their absence is a finding in its own right (§18).

---

## 1. Executive conclusion

**Can the Securitisation Readiness Agent autonomously investigate an unfamiliar
portfolio through Trakt without being told which metrics to inspect?**

## YES.

The agent receives one sentence — *"Assess this portfolio for securitisation
readiness. Identify material strengths, weaknesses, risks and areas requiring
further diligence. Support every conclusion with governed Trakt evidence."* — and
no metric list, no ordering, and no hint that anything is wrong. Across five
independent runs it discovered a mean **91.5%** of the planted findings
(88.5–92.3%), made **zero** methodology errors, committed **zero** epistemic
failures, and tripped **neither** planted false-positive trap.

The qualification is narrow and specific rather than structural: the agent
reliably finds what is *there* and is markedly weaker at noticing what it did
**not** ask for. Its single systematic miss (§16) is a question it never posed,
not a question it answered wrongly.

One claim in this report is deliberately softer than the raw score suggests. The
scorer flagged two false positives; reading the agent's actual words shows both
are scorer artifacts, not agent errors (§7). Conversely, reading found one
incorrect claim the scorer does not check (§17). The score is a floor and a
ceiling on different axes, and neither is the finding — the reading is.

---

## 2. What the Agent does

**Trakt** owns the data and every calculation. It holds the loan tape, the
canonical field model, the governed metric registry, the methodology
identifiers, and the rule packs. It decides what can be calculated, refuses what
cannot, and returns a reason with every refusal.

**The Agent** owns the investigation. It decides what to look at, what a result
means, whether a finding deserves a follow-up, when it has enough evidence, and
what to conclude. It weighs materiality and writes the narrative.

**The LLM does not** perform arithmetic. Not one number in any of the five
assessments was computed by the model. This is not enforced by instruction — it
is enforced structurally: `GovernedSession` hands out three verbs
(`capabilities()`, `call()`, `transcript()`) and no DataFrame, no file path, no
analytics function and no storage handle. The agent sees governed tool results,
never tapes. If the only way to obtain a number is to ask Trakt for it, then
every number in the conclusion carries Trakt's methodology and provenance with
it.

That is the whole design. An agent handed a DataFrame will compute — fluently
and wrongly. Denying it raw data is a correctness measure, not a security one.

---

## 3. Synthetic portfolio

| Property | Value |
|---|---|
| Loans | 400 |
| Periods | 6 monthly snapshots, 2025-08-31 → 2026-01-31 |
| Balance | £250,000 per loan, uniform |
| Asset class | UK residential mortgages, secured |
| Regions | London, South East, North West, Midlands, Scotland |
| Amortisation | 300 loans FIXE, 100 loans FRXX |
| Rate type | 300 loans FXRL (fixed for life), 100 loans FLIF (floating) |
| Vintages | 2021–2024 |
| Rule packs | A warehouse agreement, a proposed-securitisation criteria set, and Trakt's own internal screening thresholds — three distinct authorities over the same facts |

The design principle is that **no single metric reveals the portfolio**. The
headline LTV is comfortable; the tail beneath it is not. The 30+ arrears series
contains a spike that resolves and an underlying trend that does not. The
prepayment trend is declining, which is notable and not automatically adverse.
One concentration fact passes one rulebook, breaches another, and flags a third.

Answers were defined independently of the code that computes them, and
`ANSWER_KEY` is never importable from the agent's runtime — the evaluation
runner imports the portfolio *builder* only, so the agent cannot reach the thing
it is being scored against.

---

## 4. Hidden answer key

Sixteen planted cases, thirteen of which the agent is expected to discover.

| ID | Category | Planted fact | Value | Discover? |
|---|---|---|---|---|
| CONC-01 | concentration | London share of balance | 31.0% | ✅ |
| CONC-02 | **false-positive trap** | South East share of balance | 22.0% — passes every supplied limit | ❌ |
| LTV-01 | collateral | Weighted-average current LTV | 62.08% | ✅ |
| LTV-02 | collateral | Share of balance above 80% LTV | 12.0% | ✅ |
| VAL-01 | valuation | Share on stale valuations (>24m) | 12.0% — the *same* loans as LTV-02 | ✅ |
| VINT-01 | cohort | 2024 vintage carries all high-LTV and all 90+ DPD exposure | 120 loans, 30% of balance | ✅ |
| ARR-01 | arrears | 90+ DPD trajectory | 1.25 → 10.0% over six periods | ✅ |
| ARR-02 | **false-positive trap** | 30+ DPD spike at 2025-10-31 that resolves | the *spike* is not material; the underlying rise is | ❌ |
| DEF-01 | default | Periodic default rate, rising every period | `OBSERVED_DEFAULT_RATE_CDR@v1` | ✅ |
| CURE-01 | cure | Cure confined to one period, zero in five | `[0, 0, 40.9, 0, 0]` | ✅ |
| CPR-01 | prepayment | Prepayment declining — notable, **not** automatically adverse | 900k → 300k unscheduled | ✅ |
| LOSS-01 | loss | Observed loss severity | 40.0% (vs a **supplied LGD of 25.0**) | ✅ |
| WAL-01 | contractual | Contractual WAL on the fixed sub-book | AVAILABLE — 300 loans FIXE+FXRL | ✅ |
| YTM-01 | epistemic | Contractual YTM portfolio-wide | ASSUMPTION_REQUIRED — 100 loans FRXX+FLIF | ✅ |
| EPI-01 | **epistemic trap** | Expected WAL | MODEL_REQUIRED — must not be quoted | ❌ |
| DATA-01 | data readiness | `employment_status` missing | 25% of loans | ✅ |

Three of these are traps. **LOSS-01 is the sharpest**: the portfolio carries a
supplied loss-given-default estimate of 25.0 alongside a realised severity of
40.0. Quoting 25.0 *as* severity is the exact defect Sprint 2.5E found in MI
Query, planted deliberately to see whether the agent repeats it.

---

## 5. Representative autonomous run

Run 1, observable actions only. No model reasoning is stored or shown — what the
agent *did* is auditable; what it *thought* is not evidence.

**Objective given:** the single sentence in §1, plus the portfolio resource
identifier and a note that the portfolio is being considered for a term
securitisation.

**Discovery (call 1).** `portfolio_capabilities` → 28 capabilities: 27
AVAILABLE, 1 MODEL_REQUIRED. The agent learns what Trakt can and cannot produce
before computing anything.

**Orientation (calls 2–9).** `portfolio_summary` → `readiness_framework` →
`data_completeness` → `evaluate_rule_packs` → `regulatory_readiness` →
`evaluate_covenants` → `valuation_age_profile` → `portfolio_history`. It
establishes the shape of the book and the rulebooks that apply *before* looking
for problems.

**The pivot (calls 10–11).** `stratify` on `current_loan_to_value` returns the
distribution — and the agent immediately follows with `cohort_comparison` on the
92.0% LTV band, measuring arrears within it. This is the run's most important
moment: nothing told it that a comfortable 62.08% headline might conceal a tail,
and nothing told it to test that tail for performance. It inferred both.

**Trajectory (calls 12–15).** `period_change` → `transition_analysis` →
`default_analysis` → `prepayment_analysis`. It moves from levels to movement
unprompted.

**Drill-down (calls 16–30).** Concentration by region; a governed metric batch;
a failed `stratify` on `product_type` (the field does not exist — the agent
records the gap rather than substituting); interest-rate-type stratification;
contractual WAL; origination vintage; loss analysis; the ten highest-LTV loans;
`get_loans` on five of them by identifier; region stratification with weighted
metrics; arrears shares; a Scotland cohort comparison; validation exceptions;
and finally defaulted loans by region.

**Conclusion.** `MATERIAL_REMEDIATION_REQUIRED`, with 7 material findings, 5
strengths, 4 could-not-assess entries and 7 further-diligence items. 30 tool
calls, **0 repeated**, 161.9 seconds.

The shape is investigative, not procedural: orient → notice → drill → confirm →
widen. No two runs produced the same sequence.

---

## 6. Discovery results

| | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|---|---|---|---|---|---|
| **Discovery** | 92.3% | 92.3% | 92.3% | 88.5% | 92.3% |
| Found | 12 | 12 | 12 | 11 | 12 |
| Partial | 0 | 0 | 0 | 1 | 0 |
| Missed | 1 | 1 | 1 | 1 | 1 |

**Mean 91.5%, range 88.5–92.3%.**

Per-case, across all five runs (F = found, p = partial, – = missed):

| Case | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| CONC-01 | F | F | F | **p** | F |
| LTV-01 | F | F | F | F | F |
| LTV-02 | F | F | F | F | F |
| VAL-01 | F | F | F | F | F |
| VINT-01 | F | F | F | F | F |
| ARR-01 | F | F | F | F | F |
| DEF-01 | F | F | F | F | F |
| CURE-01 | F | F | F | F | F |
| CPR-01 | F | F | F | F | F |
| LOSS-01 | F | F | F | F | F |
| WAL-01 | F | F | F | F | **–** |
| YTM-01 | **–** | **–** | **–** | **–** | **F** |
| DATA-01 | F | F | F | F | F |

**Ten of thirteen cases were found in every run. Nothing was never found.**

The three inconsistent cases are worth separating, because they are not the same
kind of instability:

- **WAL-01 / YTM-01 are a trade, not noise.** Every run called
  `contractual_analytics` exactly once. Four runs asked only for
  `contractual_wal` and reported it. Run 5 asked for *both* — and then reported
  the yield with its limitation intact but omitted the life. The agent's failure
  is not computation; it is deciding what is worth saying.
- **CONC-01 in run 4** was quoted correctly (31%) but compared against only two
  of the three rulebooks, so it scored partial rather than found.

---

## 7. False positives

**Confirmed material false positives: zero.**

Neither planted trap was tripped in any run:

- **CONC-02** (South East at 22%, inside every supplied limit) was never raised
  as a concern.
- **ARR-02** (the 30+ DPD spike that resolves) was never mistaken for the
  underlying trend.
- **EPI-01** (expected WAL) was never quoted. Every run that touched contractual
  life distinguished it from expected life.

**The scorer reported two false positives. Both are scorer defects, and I am
recording them as such rather than against the agent.** The CONC-02 check tests
whether `"south east"` appears anywhere in the assessment **and** whether
`"breach"`, `"exceed"`, `"concern"` or `"issue"` appears anywhere in the
assessment — a document-wide conjunction with no proximity requirement. Reading
what the two flagged runs actually wrote:

> **Run 3, in full:** *"London 31%, South East 22%, North West 19%, Midlands
> 16%, Scotland 12%."* — a factual distribution list. No judgement attached.

> **Run 4:** *"Five UK regions represented (London 31%, South East 22%, …),
> providing geographic spread despite the London breach."* — South East is cited
> as **diversification**, and the word "breach" refers to London.

Run 4 cites South East as a *strength*. The scorer counted it as a concern
because the word "breach" appears elsewhere in the same paragraph, attached to a
different region.

I have not corrected the scorer, because the freeze on it is what keeps the five
runs comparable. Corrected by reading, the false-positive count is **0 across
5 runs**, not 2.

This is the third time in this programme that a defect was found by reading a
definition next to the thing claiming to implement it. The scorer passed its own
tests.

---

## 8. Numerical correctness

**Correct numerical claims: every value the scorer pins, in all five runs.**

The scorer verifies planted values at stated tolerances and checks two specific
traps in the arrears series — quoting the 30+ share as 8.5% (the delinquency
band alone, omitting defaulted loans, which are also more than 30 days down) or
the 90+ share as 4.0% (the same omission). **No run made either error.** These
are exactly the mistakes a careless aggregation produces, and the agent avoided
them because it asked Trakt for the metric rather than assembling one.

**Wrong methodology: zero.** No run quoted the supplied LGD of 25.0 as realised
severity. Every run that discussed severity reported the observed 40.0% under
`OBSERVED_LOSS_SEVERITY@v1` and, where it mentioned the supplied estimate,
labelled it as someone's model. **The Sprint 2.5E defect did not reproduce.**

**Wrong aggregation: zero. Wrong period: zero.**

**One incorrect claim, found by reading rather than by the scorer.** Run 4 wrote,
inside the `rule` field of a finding:

> *"BREACH: Scotland at 31% (actually London at 31%; Scotland is 12% but is the
> weakest cohort)."*

The agent caught its own error mid-sentence and corrected it in place — but left
the incorrect assertion standing in delivered output. Both facts are individually
right (London is 31%, Scotland is 12%) and the same run files a separate,
entirely correct London concentration finding. A reader skimming the rule field
would nonetheless take "BREACH: Scotland at 31%" at face value. See §17.

**What I did not check.** The scorer verifies the values the answer key pins
exactly; I additionally read every South East mention, every WAL/YTM mention,
and run 4's concentration findings in full. Numbers outside that set — the
weighted rates, the sub-cohort percentages, the £ figures — are unchecked.
Silence on them means *unverified*, not *correct*.

---

## 9. Investigation quality

The behaviour the experiment was built to test is whether a finding triggers a
*further* question that nobody asked for. It does, consistently:

| Trigger | Autonomous follow-up | Runs |
|---|---|---|
| Comfortable 62.08% headline LTV | `cohort_comparison` on the 92% LTV band, measuring arrears **within** it | all 5 |
| 12% high-LTV tail | `valuation_age_profile` → recognition that the stale-valuation pocket is the **same 12%** | all 5 |
| Stale + high-LTV overlap | `cohort_comparison` on Scotland, isolating the cohort | 4 of 5 |
| Rising arrears | `transition_analysis` across periods → `default_analysis` on the owned CDR metric | all 5 |
| `stratify` on `product_type` fails | Gap recorded in `could_not_assess`; **no substitute metric offered** | 3 of 5 |
| Ten highest-LTV loans ranked | `get_loans` on named identifiers to inspect them individually | 3 of 5 |

The second row is the strongest single result. Two independently planted
findings — a high-LTV tail and a stale-valuation pocket, each 12% of balance —
are the *same loans*. Nothing in the objective, the prompt, or any tool
description says so. Every run discovered both; four of five explicitly stated
the overlap and reasoned about what it means (the reported 92% LTV rests on
valuations Trakt flags as stale, so the true credit position is unknowable
without fresh appraisals).

Tool-call efficiency: 29–35 calls per run, **0 repeated identical calls in every
run**. The agent is not looping; it is investigating.

---

## 10. Fact / rule / judgement

The prompt requires every material conclusion to separate what Trakt measured,
what threshold applies and whose authority it carries, and what the agent itself
concludes. Three examples from the runs:

**Example 1 — one fact, three authorities (run 5).**
- **FACT:** London accounts for 31% of balance, measured by `concentration` on
  `geographic_region_collateral`.
- **RULE:** Warehouse agreement permits ≤35% → **PASS**. Proposed
  securitisation criteria permit ≤27% → **BREACH**. Trakt internal screening
  flags >25% → **FLAG**.
- **JUDGEMENT:** *"This is a RULE difference, not a portfolio deterioration: the
  same 31% fact yields a pass under warehouse terms and a breach under
  transaction terms."*

That sentence is the single best output of the sprint. The agent understood that
a PASS under one authority is not clearance under another, and said so in those
terms — the distinction the prompt asks for and the one a careless report
collapses.

**Example 2 — observed is not modelled (run 1).**
- **FACT:** Realised loss severity 40.0%, under `OBSERVED_LOSS_SEVERITY@v1`.
- **RULE:** No severity threshold in the supplied criteria.
- **JUDGEMENT:** The supplied LGD estimate of 25.0% is someone's model and is
  60% below what this book actually realised; the divergence is itself the
  finding.

**Example 3 — a limit with no rule (run 4).**
- **FACT:** 34 loans in Arrears carry 92% weighted-average LTV and a 5.75%
  weighted rate, against 59.4% and 4.54% for Performing.
- **RULE:** *"No status-mix threshold exists in the supplied criteria, but the
  LTV and rate differential is a measurable fact."*
- **JUDGEMENT:** The higher rate on distressed borrowers may reflect
  affordability stress or penalty rates.

The agent declines to invent a threshold where none was supplied, and says so
explicitly rather than reaching for an industry convention.

---

## 11. Capability-state handling

| State | Exercised? | Evidence |
|---|---|---|
| **AVAILABLE** | ✅ all runs | 27 of 28 capabilities at discovery; every reported number came from one |
| **UNAVAILABLE** | ✅ all runs | Borrower concentration (no borrower identifier on the tape), product mix (`product_type` absent), period-change attribution (*"fewer than two governed portfolio snapshots available"*) |
| **MODEL_REQUIRED** | ✅ all runs | Expected WAL — 1 of 28 at discovery; never quoted by any run |
| **ASSUMPTION_REQUIRED** | ✅ run 5 only | Contractual YTM: *"100 floating-rate loans (25% of balance) excluded from yield calculation because Trakt does not assume future rate paths. Reported 4.45% yield covers fixed-rate portion (75% of balance) only."* |
| **METHODOLOGY_NOT_APPROVED** | ⚠️ run 1, **wrongly** | See below |
| **NOT_APPLICABLE** | ❌ **not exercised** | The fixture is secured residential with full geography; no condition in it produces this state |

Run 5's ASSUMPTION_REQUIRED handling is exactly right: it reports the number it
*can* have, states the population it covers, names the population it excludes,
and gives the reason. It does not extrapolate the fixed-rate yield across the
floating book.

**Two defects in state handling:**

**Run 1 conflated "I asked wrongly" with "Trakt hasn't settled a definition."**
It called `readiness_metrics` with `vintage_share` (the correct identifier is
`composition_vintage_share`), received *"vintage_share is not in the governed
metric library"*, and filed the result under **METHODOLOGY_NOT_APPROVED**. Trakt
has an approved vintage methodology; the agent used the wrong id. The label is
materially misleading in a governance report — it tells a reader Trakt has an
unsettled definition when Trakt has a settled one the agent failed to name.

**Runs 4 and 5 invented states outside the vocabulary.** The `could_not_assess`
schema types `status` as a free string, not an enum, and the agent used
`FIELD_GAP`, `PARTIAL`, `Parameter issue` and `ASSUMPTION_REQUIRED (partial)`.
None is one of the six governed states. `FIELD_GAP` is borrowed from the Sprint
2.5C WAL/YTM classification taxonomy — a different vocabulary for a different
purpose — which suggests the model is pattern-matching across concepts the
system deliberately keeps apart. The six states carry different meanings and
different remediation paths; a free-text status field invites exactly this.

**The NOT_APPLICABLE gap is a fixture limitation, not an agent result.** The
sprint asked for examples of five states and this report can honestly show four.

---

## 12. Multi-run consistency

| | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean |
|---|---|---|---|---|---|---|
| Verdict | MATERIAL_REMEDIATION_REQUIRED | ← same | ← same | ← same | ← same | **5/5 identical** |
| Discovery | 92.3% | 92.3% | 92.3% | 88.5% | 92.3% | 91.5% |
| False positives (scored) | 0 | 0 | 1 | 1 | 0 | — |
| False positives (**verified by reading**) | 0 | 0 | **0** | **0** | 0 | **0** |
| Methodology errors | 0 | 0 | 0 | 0 | 0 | 0 |
| Epistemic failures | 0 | 0 | 0 | 0 | 0 | 0 |
| Numerical errors (scored) | 0 | 0 | 0 | 0 | 0 | 0 |
| Unsupported claims (**by reading**) | 0 | 0 | 0 | **1** | 0 | — |
| Tool calls | 30 | 35 | 31 | 29 | 31 | 31.2 |
| Repeated calls | 0 | 0 | 0 | 0 | 0 | **0** |
| Material findings | 7 | 7 | 8 | 8 | 8 | 7.6 |
| Model steps | 11 | 11 | 12 | 9 | 10 | 10.6 |

**The verdict is unanimous.** Five independent runs, no shared state, reached
`MATERIAL_REMEDIATION_REQUIRED` — the same category, on the same portfolio, by
five different investigative routes.

Tool selection is stable where it matters and varies where it should. Fifteen
tools were called in all five runs (`portfolio_capabilities`, `portfolio_summary`,
`readiness_framework`, `data_completeness`, `evaluate_rule_packs`,
`regulatory_readiness`, `valuation_age_profile`, `portfolio_history`,
`period_change`, `transition_analysis`, `default_analysis`,
`prepayment_analysis`, `loss_analysis`, `list_validation_exceptions`,
`contractual_analytics`). Exploration depth varies — `stratify` ranged 3–9 calls,
`rank_loans` 1–3 — which is investigative freedom, not instability.

**Consistency is high but not total, and the variance is concentrated in what the
agent chooses to report rather than in what it can compute.** Every run had WAL
available; four reported it. One run had the yield and reported it; four never
asked.

---

## 13. Audit evidence

Every governed call is recorded with sequence, tool, full arguments, status,
elapsed milliseconds, error code, and a bounded digest of the result:

```json
{
  "sequence": 1,
  "tool": "portfolio_capabilities",
  "arguments": { "resource": "ERE/source_portfolio/direct_001",
                 "include_definition": true },
  "status": "success",
  "elapsed_ms": 234.18,
  "error_code": null,
  "result_digest": { "summary": { "AVAILABLE": 27, "MODEL_REQUIRED": 1 },
                     "metrics_count": 28 }
}
```

From a completed run you can reconstruct: the objective, the actor, the
organisation, the request id, the portfolio, the capabilities discovered, every
call in order with its inputs and a quotable summary of its result, the timing
of each, and the efficiency profile. `test_the_run_is_reconstructible_from_the_audit_record`
asserts this.

**What is deliberately absent: the model's reasoning.**
`test_the_audit_record_stores_actions_not_reasoning` enforces it. A stored chain
of thought reads as though it were evidence and is not — only tool inputs and
Trakt's outputs are reconstructible facts about what happened. Narration the
model emitted alongside its tool calls is kept separately as observable output,
not folded into the audit trail.

**One gap.** `_digest` extracts a fixed key list from the top level of the
payload. `contractual_analytics` nests its results, so run 5's WAL/YTM call
digests to `{}` — the audit record shows that the call was made and succeeded
but carries nothing quotable from it. Not fixed here, because the session is
frozen for comparability.

---

## 14. Security

Isolation is tested at the session boundary, where an external client-owned
agent would arrive. All 30 surface tests pass at HEAD.

| Test | Asserts |
|---|---|
| `test_an_agent_cannot_reach_a_portfolio_it_was_not_granted` | Resource-level authorisation |
| `test_an_agent_cannot_reach_another_tenants_portfolio` | Tenant isolation |
| `test_an_agent_without_the_capability_is_refused_the_tool` | Capability-scoped tool access |
| `test_a_refusal_is_recorded_in_the_transcript_like_any_other_call` | Refusals are auditable, not silent |
| `test_the_agent_package_cannot_reach_trakts_calculations` | AST-level: the agent package imports no analytics function |
| `test_the_session_exposes_three_verbs_and_no_data` | No DataFrame, no path, no storage handle |
| `test_the_answer_key_is_not_importable_from_the_agent_package` | The agent cannot reach its own scoring truth |

Every call goes through `execute_governed_tool` — the same entry point an
external HTTP client reaches. **The reference agent has no privilege a
third-party agent would lack.** That is the property that makes the A2A claim in
§19 meaningful rather than aspirational.

Credential handling for this evaluation: the temporary API key was held outside
the repository at mode 600, injected only into the evaluation process
environment, and shredded on completion. It was never written to source, tests,
fixtures, logs, commits or documentation, and never appears in any artefact.

---

## 15. Performance and cost

| Run | Wall | Trakt analytics | LLM | Trakt share | Calls | Steps |
|---|---|---|---|---|---|---|
| 1 | 161.9 s | 4.67 s | 157.2 s | 2.9% | 30 | 11 |
| 2 | 202.9 s | 9.63 s | 193.2 s | 4.7% | 35 | 11 |
| 3 | 193.5 s | 3.93 s | 189.6 s | 2.0% | 31 | 12 |
| 4 | 216.6 s | 3.54 s | 213.0 s | 1.6% | 29 | 9 |
| 5 | 180.9 s | 1.32 s | 179.6 s | 0.7% | 31 | 10 |
| **Mean** | **191.2 s** | **4.6 s** | **186.5 s** | **2.4%** | **31.2** | **10.6** |

**Trakt's deterministic analytics account for 2.4% of wall-clock time.** Thirty-one
governed calls over a 400-loan, six-period book complete in under five seconds.
The remaining 97.6% is model inference. Any latency work belongs at the agent
loop, not the analytics layer — and correctness was not traded for speed
anywhere.

| Run | Input tokens | Output tokens | Cost |
|---|---|---|---|
| 1 | 373,742 | 6,487 | $1.22 |
| 2 | 380,729 | 7,810 | $1.26 |
| 3 | 419,181 | 8,083 | $1.38 |
| 4 | 292,054 | 8,456 | $1.00 |
| 5 | 334,473 | 7,311 | $1.11 |
| **Total** | **1,800,179** | **38,147** | **$5.97** |

At Sonnet-tier list price ($3.00/$15.00 per MTok). **A complete governed
securitisation readiness assessment costs about $1.19.**

### Input-token investigation

Conducted after the experiment, as instructed. The observed band across five
runs is 292k–419k input tokens — wider than the 335k–384k seen in the lost runs.
Two structural causes, in order of size:

**1. No prompt caching is configured anywhere in the agent loop.** There is no
`cache_control` in `readiness_agent/`, and the usage accounting reads only
`input_tokens` and `output_tokens` — `cache_read_input_tokens` and
`cache_creation_input_tokens` are never recorded. Every step re-sends the entire
conversation at full input price.

**2. The fixed prefix is large and is re-sent on every step.** The tool surface
is generated from Trakt's whole registry — 28 tools, 39,773 characters of JSON
schema — plus a 2,960-character system prompt. That is 42,733 characters
(≈10,700 tokens) of *byte-identical* content re-sent on each of the 9–12 model
calls per run:

| Run | Steps | Input tokens | Fixed prefix × steps | Share |
|---|---|---|---|---|
| 1 | 11 | 373,742 | 117,513 | 31% |
| 2 | 11 | 380,729 | 117,513 | 31% |
| 3 | 12 | 419,181 | 128,196 | 31% |
| 4 | 9 | 292,054 | 96,147 | 33% |
| 5 | 10 | 334,473 | 106,830 | 32% |

**Across five runs, ~566,000 of 1,800,000 input tokens (31%) are the same
unchanging prefix, paid for 52 times.** The other ~69% is the growing message
history — also re-sent uncached, and growing quadratically in total cost as the
investigation lengthens.

Token count per run correlates with step count, not tool-call count: run 2 made
the most calls (35) but run 3 consumed the most tokens (12 steps). The unit of
cost is the model round-trip, not the Trakt query.

Both causes are fixable without touching the agent's reasoning. The tool schemas
and system prompt render before messages and never change within a run, so a
single cache breakpoint on the last tool or system block would cache them; a
breakpoint on the last block of each turn would cache the conversation prefix
incrementally. The certain saving is the 31% fixed prefix; the conversational
prefix is a larger but harder-to-bound additional saving.

**Two caveats on this analysis.** Token counts are estimated at 4 characters per
token — I did not call `count_tokens`, since the temporary key is deleted and
re-provisioning one for an accounting refinement is not warranted. And
`run_assessment` accumulates usage across steps without recording per-step
figures, so this is a structural analysis of *why* the totals are what they are,
not a measurement of each request. Both would be resolved by recording per-step
usage including the cache fields — a change to the runner, not the agent.

---

## 16. Most important Agent miss

**YTM-01: the agent never asked what the portfolio yields, in four of five runs.**

This is the sprint's only systematic miss and it is not a computation failure.
Every run called `contractual_analytics` exactly once. Four asked only for
`contractual_wal`. Trakt would have returned the yield — correctly, with its
limitation attached — had they asked. Run 5 asked, and got precisely the honest
partial answer the case was planted to elicit: 4.45% covering the fixed-rate 75%
of balance, with the 100 floating-rate loans excluded because Trakt does not
assume future rate paths.

**Why it matters more than a missing metric.** YTM-01 was planted as an
*epistemic* case, not an analytical one. Its purpose is to test whether the agent
can hold a partial answer honestly — reporting what it can have, naming what it
cannot, and refusing to extrapolate across the gap. Run 5 proves the agent
*can*. The other four never reached the test.

**The likely cause, stated as a hypothesis.** `contractual_analytics` takes a
`metrics` list. An agent that has decided it wants "the contractual analytics"
must still choose which members of that list to name, and the WAL is the more
familiar securitisation figure. The agent is not weighing WAL against YTM and
choosing WAL; it appears not to be considering YTM at all. Run 5's behaviour —
asking for both, then reporting only one — points the same way: the omission is
in what the agent decides is worth *saying*, one layer above what it decides to
*ask*.

**What this does not mean.** It is not evidence of an epistemic weakness. Zero
epistemic failures were recorded across five runs, and no run quoted an expected
WAL. The agent respects boundaries it encounters. It just did not walk up to this
one.

---

## 17. Most important false positive

**No planted trap was tripped, so the most important false positive is one
nobody planted: run 4 asserted a breach against the wrong region.**

> *"BREACH: Scotland at 31% (actually London at 31%; Scotland is 12% but is the
> weakest cohort)."*

Scotland is 12% of balance and breaches nothing. London is 31% and breaches the
securitisation criterion. The agent produced the wrong claim, recognised the
error inside the same sentence, corrected it parenthetically — and shipped both
the error and the correction in the delivered `rule` field of a material finding.

**Why this is the most important one.** The scorer does not catch it, and would
not catch its more dangerous variant. It sits in the `rule` field — the field
whose entire purpose is to state a threshold and its authority — which is
precisely where a reader looks for the compliance answer. A reader skimming
findings sees "BREACH: Scotland at 31%" as a heading-level claim; the correction
is inside a parenthesis. And the same run files a separate, entirely correct
London concentration finding, so the report contains both a right answer and a
wrong one about the same fact.

**What it is not.** It is not a hallucinated number: both 31% and 12% are real,
correctly measured, correctly attributed elsewhere in the same assessment. It is
a mis-association of a correct value with the wrong entity, surviving into output
because the agent's self-correction was textual rather than structural — it
rewrote the sentence in place instead of rewriting the finding.

**Frequency: 1 occurrence in 5 runs**, in the run that also scored lowest on
discovery (88.5%) — the only run where the two signals agree.

---

## 18. Regression

**Sprint 3 substrate regression, `3ca848b` → `a6479b6`:**

| | Baseline | Candidate | Δ |
|---|---|---|---|
| Passed | 5,299 | 5,330 | **+31** |
| Failed | 64 | 64 | 0 |
| Errors | 13 | 13 | 0 |

All four directional difference sets were empty — no test that passed at
baseline fails at candidate, no failure or error id appears or disappears in
either direction. Run under pinned immutable git worktrees, created with an
explicit `git checkout --detach` and verified by `rev-parse` before and after
execution, and not edited while tests ran. (That verification exists because an
earlier sprint's `git worktree add -f` silently no-op'd on an existing directory
and tested the wrong commit — a defect found only by checking the SHA.)

**Post-substrate commits.** Two further commits land after `a6479b6`:
`5dc890c` (runner persistence) and `4663e77` (the scoring runner). Both are
scripts under `scripts/`, neither is collected by pytest, and `grep` confirms no
test imports either. 7,702 tests collect at HEAD. The 30-test agent surface suite
passes at HEAD. A full-suite run at HEAD was launched to confirm the totals are
unchanged; **its result is not yet in hand at the time of writing, and this
section should be treated as complete only for `a6479b6`.**

**The lost runs.** Runs 2–4 of the first execution completed and were destroyed:
the runner accumulated results in memory and wrote once after the loop, and run 5
died on an exhausted credit balance, taking three paid-for assessments with it.
Their observable metadata survived only in console output (34/29/30 tool calls;
384,112/335,361/344,520 input tokens; 190.3/169.3/169.1 s). **They are excluded
from every figure in this report**, as instructed. The fix — persist after every
run, atomically via write-and-rename, and resume onto an existing file rather
than overwriting — is `5dc890c`, and its behaviour was verified end-to-end
against a temporary file rather than only checked for syntax. The lost runs are
themselves the strongest available evidence for the discipline: the destroyed
code imported cleanly and passed everything it was asked to pass.

---

## 19. A2A readiness

**Could the Securitisation Readiness Agent be placed behind a real A2A
interaction with a client enterprise or Copilot agent without redesigning its
core analytical behaviour?**

## Yes — the analytical core needs no redesign.

The reason is architectural rather than fortunate. The agent already speaks to
Trakt through `execute_governed_tool`, the same entry point an external
client-owned agent reaches over HTTP. It holds no DataFrame, no file path and no
private analytics function; it cannot widen its own authority; and tenant,
resource and capability isolation are enforced and tested at that boundary. A
client's agent placed on the other side of an A2A handshake would get exactly the
surface this agent gets — no more, and no less.

Remaining integration work, none of which touches analytical behaviour:

1. **The handshake and transport.** Agent card / capability advertisement, task
   lifecycle, and streaming progress. Not built, not faked.
2. **Identity propagation.** The client agent's principal must map to a Trakt
   `ExecutionContext` with the right tenant and capabilities. The context object
   exists and is honoured; the mapping from an external identity does not.
3. **Per-step usage and cache accounting** (§15), so a client-facing service can
   attribute and cap cost per task.
4. **The `could_not_assess` status vocabulary should be an enum, not free text**
   (§11). A consuming agent needs to branch on the six governed states; today it
   would receive `FIELD_GAP` and `Parameter issue` alongside them.
5. **A structured-output contract for findings.** `submit_assessment` is already
   structured; what a peer agent consumes should be a versioned schema rather
   than the evaluation harness's shape.

Items 4 and 5 are small and would materially improve machine consumption. Item 1
is the real work.

---

## 20. Landing-page demo candidate

Drawn only from behaviour that actually occurred, in every run. Roughly 25
seconds:

| Beat | ~Time | On screen |
|---|---|---|
| **The objective** | 0–3 s | One sentence, typed: *"Assess this portfolio for securitisation readiness."* Nothing else. |
| **Governed discovery** | 3–7 s | `portfolio_capabilities` → *27 AVAILABLE · 1 MODEL_REQUIRED*. Trakt states what it can and cannot produce before anything is computed. |
| **The comfortable headline** | 7–11 s | `stratify` → **weighted-average LTV 62.08%** — visibly reassuring. |
| **The agent doesn't stop** | 11–16 s | Unprompted: `cohort_comparison` on the 92% LTV band → **12% of balance above 80% LTV**, and its arrears are far worse than the book's. |
| **The compounding finding** | 16–20 s | `valuation_age_profile` → **12% on valuations older than 24 months** — *the same loans*. The reassuring 92% LTV rests on stale evidence. |
| **One fact, three rulebooks** | 20–24 s | London 31%: **PASS** (warehouse ≤35%) · **BREACH** (securitisation ≤27%) · **FLAG** (Trakt screening >25%). |
| **The conclusion** | 24–26 s | `MATERIAL_REMEDIATION_REQUIRED`, with the audit trail behind it. |

Why this sequence: it opens on an objective rather than a dashboard, shows
governance as an *enabling* step rather than a gate, contains a genuine reversal
(the comfortable number is the wrong number), demonstrates autonomous
investigation nobody scripted, and lands on the rulebook distinction — the thing
Trakt does that a spreadsheet cannot. Every beat is real and reproduced in all
five runs.

**Not built. Not animated.** This is a proposal.

---

## 21. Next recommendation

**Harden the Agent. Specifically, close the reporting gap between what it
computes and what it says.**

The evidence supports this and not the alternatives. The analytical layer is not
the constraint: 2.4% of runtime, zero methodology errors, zero epistemic
failures, zero numerical errors on every pinned value, and no planted trap
tripped in 155 governed calls. The governance layer is not the constraint
either: refusals are respected, states are honoured, isolation holds. **Every
weakness this sprint found is in the last mile — deciding what to ask for and
what to report.**

Four concrete items, in order:

1. **The `contractual_analytics` metric selection (§16).** The agent computes
   what it asks for and asks for less than it should. This is the single miss
   across five runs.
2. **The status vocabulary (§11).** Make `could_not_assess.status` an enum of the
   six governed states. This alone would have prevented `FIELD_GAP`,
   `Parameter issue` and `ASSUMPTION_REQUIRED (partial)`, and would have forced
   run 1 to reconsider `METHODOLOGY_NOT_APPROVED` for what was an id error.
3. **Metric-id resolution (§11).** `vintage_share` → `composition_vintage_share`
   should resolve or suggest, not refuse flatly. The registry has synonyms; this
   path does not consult them.
4. **Self-correction discipline (§17).** A corrected claim should replace the
   original, not sit beside it in delivered output.

**Do not build the A2A handshake yet** — not because it is wrong, but because
items 2 and 4 change what a peer agent consumes, and building the transport
first means building it against a contract that is about to change.

**Do not build the landing-page demo yet.** The sequence in §20 is real and will
still be real after hardening.

**Do not run another infrastructure sprint.** The infrastructure did its job. The
five runs cost $5.97 and produced a defensible institutional credit assessment
from a one-sentence objective; what is left to improve is judgement, not
plumbing.

---

## Appendix — reproducing this evaluation

```bash
# 1. Run (requires ANTHROPIC_API_KEY; resumes onto an existing file)
python scripts/run_readiness_agent_eval.py --runs 5 --out <path>

# 2. Score (free, repeatable, never contacts a model)
python scripts/score_readiness_agent_eval.py --runs-file <path>
```

Keep the run file outside the repository: a run record contains a full
assessment and is evaluation evidence, not source.
