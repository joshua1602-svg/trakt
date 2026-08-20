# MI Query Agent — Client Readiness

Making the agent production-ready for Client 1 onboarding across three shipped
capabilities: historical MI answerability; slicing, dicing and charting governed
fields into bespoke views; and reasonable answering of the nine multi-layered
intents. Forecasting is not in the shipped scope — it is measured so the
decision to enable it later is evidenced, and it ships gated.

**Status: Tranche D complete. Tranches E, F and G not started.**

---

## 1. Readiness verdicts

Stated separately, because they are different questions, and both are interim
until the client-shaped fixtures of Tranche E exist.

### Onboarding-ready — what tomorrow requires

**On track.** What onboarding needs is ingest, field mapping, governed config,
and a safe refusal on everything not yet configured. The measurable part of that
is *no unsafe answers and no silent substitution*, and Tranche D moved it:

| | before D | after D |
|---|---|---|
| unsafe outcomes over 752 runs | 0 | **0** |
| calibration bank | 245 passed / 13 xfailed | **259 passed / 1 xfailed** |
| silent substitutions found by an independent type sweep | 5 | **0** |
| parser self-disagreement (176 cells) | 10 (5.7%) | **2 (1.1%)** |

The single remaining `xfail` is a fixture gap, not a product gap, and says so in
its own reason field: `exposure to London` resolves the governed region filter
and refuses safely when nothing matches, but the synthetic 400-row tape carries
no London. Tranche E re-bases the bank onto a real book, where the value exists.

### CFO-ready — the question surface

**Not yet demonstrable.** The three shipped capabilities are evidenced only on
three funded snapshots and eleven weekly extracts. Client 1 has twelve months
and fifty-two weeks. Period handling across thirteen snapshots is not a safe
extrapolation from three, and Tranche D found two questions that were being
answered over a period the reader did not ask for — which is precisely the class
of defect a longer series changes. This verdict is deferred to Tranche E, by
design rather than by omission.

## 2. Slot assignment: the root cause, and what changed

### 2.1 The brief's premise was wrong in three ways, and it is corrected here

Reported rather than quietly worked around, because each correction changes what
the tranche had to do.

1. **There are seventeen `known_gap` entries, not thirteen.** Thirteen were
   xfailing; four more carried the flag while passing.
2. **Four of the seventeen passed *because their own expectation declared the
   defect*.** `risk_222` — "balance by property value band" — declares
   `expected_dimensions: [age_bucket]`, the mis-mapping its own gap note
   describes. A case that documents a defect in its expectation will never fail
   for it.
3. **Two of the four cases the brief lists as returning confident wrong numbers
   were already refusing safely.** `balance where LTV above 50%` answered *"the
   answer reports ltv, but the question asked about balance. I have not returned
   the substituted breakdown"*; `exposure to the South East` likewise. Those were
   **false refusals**, not silent substitutions — a different defect needing the
   opposite fix.

### 2.2 The mechanism, traced before anything changed

`_detect_metric` walks a **fixed-priority vocabulary tuple** and returns the
highest-priority entry appearing **anywhere** in the text it is handed. It has
no notion of sentence position or grammatical role. `"ltv"` precedes `"balance"`
in `_METRIC_TERMS`, so in "balance where LTV above 50%" the head noun loses to a
word inside the condition.

That single mechanism explains three of the seventeen. It is **not** one root
cause: four further mechanisms are distinct.

| # | mechanism | cases |
|---|---|---|
| A | metric slot captured by a clause that names a field | `filt_159`, `risk_217`, `risk_218` |
| B | an explicit COUNT dropped, defaulting to balance | `filt_126`, `filt_149`, `filt_150`, `risk_208`, and two shipped bank questions |
| C | a recognised scope or bound that produces no filter, answered unfiltered | `risk_211`, `risk_212`, `risk_223` |
| D | a subjective or absent concept substituted with a default | `ambig_247`–`ambig_250` |
| E | a missing intent answered with the adjacent question | `dq_228`–`dq_231` |
| F | a dimension attached that the question did not request | `kpi_032`, `risk_207`, `risk_222` |

### 2.3 What was changed — precedence first, vocabulary only where justified

**Precedence.** `_metric_slot` hands the detector only the subject side,
truncating at a filter clause. The clause counts only when a numeric bound
follows its opener, so "loans with LTV above 50%" is cut and "regions with the
highest LTV" is not. The grouping fallback that read the raw question now reads
the dimension-blanked text.

**Four further fixes, each its own mechanism.** An explicit COUNT is a named
measure, not the absence of one. `exposure to <place>` names a geography filter —
admitted only in that idiom, because admitting `to` outright bound a geography
called "Complete" out of "expected to complete". A qualitative bound with no
number ("high age", "large loans") clarifies, because there is no defensible
default for "high". "How much data is missing" is not a governed analytic here
and says so.

**Vocabulary, stated as vocabulary.** Three additions, each justified rather
than pattern-matched: `regional` as the adjectival form of a governed dimension;
`loan/ticket/deal size` as the measure reading of a word that is also a bucket,
disambiguated by precedence (an aggregator in front of a bucket synonym names a
measure — you cannot average a band); and `best`/`worst` **removed** from ranking
framing, since they name a judgement whose basis the question never gives.

Every added or removed term was then run across all five banks plus seventeen
deliberately non-idiomatic probes, controlled against the pre-D tree:
**no regressions**. `tranche_d/vocab_blast_radius.txt`.

## 3. The seventeen known gaps, case by case

`xfail` is no longer an acceptable end state for any of them. Sixteen are now
hard-asserted.

| case | question | disposition |
|---|---|---|
| `kpi_032` | average loan size | **answer** — was a breakdown by ticket bucket; the aggregator now names the measure |
| `filt_159` | balance where LTV above 50% | **answer** — was a false refusal; the metric slot is no longer captured by the condition |
| `risk_207` | largest regional concentration | **answer** — `regional` resolves the governed region dimension |
| `risk_211` | exposure to London | **fixture gap** — resolves the filter and refuses safely; the tape has no London |
| `risk_212` | exposure to the South East | **answer** — was a false refusal |
| `risk_217` | concentration by LTV bucket | **answer** — the grouping term no longer supplies the measure |
| `risk_218` | concentration by age bucket | **answer** — same |
| `risk_222` | balance by property value band | **answer** — expectation retained; §2.1 records that it declares the mis-mapping |
| `risk_223` | high age borrower exposure | **clarify** — status changed; no defensible default for "high" |
| `dq_228`–`dq_231` | missing X count | **clarify** — not a governed analytic here |
| `ambig_247` | show best brokers | **clarify** — "best" names a basis the question does not give |
| `ambig_248`–`ambig_250` | bad / profitability / interesting | **clarify** — already correct; notes were stale |

## 4. Parser stability, before and after

### 4.1 The measurement, on the same 176-cell basis

A cell is one variation, on one book, in one arm. It disagrees when repeats of
the same question, on the same commit, do not agree.

| run set | cells disagreeing |
|---|---|
| V1 baseline (pinned) | 11 / 176 (6.2%) |
| before D4 | 10 / 176 (5.7%) |
| after the constrain/extend split | 3 / 176 (1.7%) |
| **after the granularity rule** | **2 / 176 (1.1%)** |
| deterministic parse, either tree | 0 / 176 |

### 4.2 Constrain where the parser invents; extend where it is right

Enumerating every filter shape across the 752 runs showed these are not one
problem, so they did not get one fix.

**Constrained — five shapes the question does not support.**
`origination_date ge 2024-01-01` on a question naming no date;
`reporting_date ge 2024-11-15` likewise; `arrears_balance gt 0` where nothing
says zero. Applying an invented bound silently narrows the book to a population
the reader never described, which is worse than the refusal it replaces.

The guard checks **both halves**: the field must be one the governed registry
carries, and the value must be one the question states. Checking the value alone
lets a fabricated field through whenever its invented number happens to appear —
`unicorn_score gt 50` survives "balance where LTV above 50%".

**`fabricated_concepts` does not cover this.** It reads the same filter slots,
but only for the two population *concepts* it knows, seasoning and provenance,
so any other field is invisible to it. It is a concept guard, not a field guard.

**Normalisation must not cause a false rejection**, and is tested for it: `50%`
held as 50 or as 0.5, `£1.5m` as 1500000, `£250,000` and `£1,500,000` with their
separators. Rejecting a real filter is worse than the defect the guard catches.

**Extended — one shape the route already resolves.** `pipeline_stage = Offer` is
legitimate and derived from the question; the route resolves OFFER from the
intent either way and declares it. A filter naming the population the plan
already resolved is a **no-op**, accepted on the plan's own declaration and only
when it names both the same field and the value asked for. `pipeline_stage = KFI`
still refuses; `account_status = offer` still refuses.

**`origination_date ge LAST_4_WEEKS` is reported separately and is not a
fabricated bound.** It is a template token that reached the output unfilled — the
*name* of a bound rather than one. It does not come from our prompt: the built
prompt contains no `SCREAMING_SNAKE` token at all, so the symbol is the model's
own. Fixable at source only by instructing the prompt that date bounds must be
literal, which has blast radius on every LLM parse; proposed, not done.

### 4.3 The residual, since it is not zero

Both remaining cells are **Q3.4**, one shape: `account_status = offer`. The model binds a pipeline stage to a funded-book field. Both
halves of the bounds guard pass — `account_status` **is** governed and "offer"
**is** in the question — so what is wrong is neither the field nor the bound but
the **value's membership of the field's domain**. A funded loan's
`account_status` is Active or Redeemed, never "offer".

Not built. It is a third guard shape with its own blast radius across every
categorical filter, the current outcome is a safe refusal on 0.7% of runs rather
than a wrong answer, and it belongs in field resolution rather than in a fourth
envelope guard. §12 carries it.

### 4.4 The first sprint effect that clears the noise floor

This matters more than the headline, and the direction is what proves it.

The earlier hardening sprint produced **31 grade differences on the LLM path,
17 one way and 14 the other** — bidirectional, inside a self-disagreement floor
of 6–10% of cells, and therefore noise. It is why that sprint's attribution
claims were made on the deterministic comparator instead.

Tranche D produced **17 grade differences, and every one is attributable by
name**:

| variation | direction | count | why |
|---|---|---|---|
| Q1.3 | refusal → correct | 3 | the invented date bound is rejected; the governed L3M window answers |
| Q3.3 | refusal → correct | 2 | the redundant `pipeline_stage` filter is a no-op |
| Q3.4 | refusal → correct | 5 | same |
| Q3.4 | correct → refusal | 1 | `account_status = offer` — §4.3, the residual |
| Q4.4 | correct → refusal | 4 | the granularity rule: weeks asked, months available |
| Q6.3 | refusal → correct | 2 | the invented `arrears_balance` bound is rejected |

**Twelve one way, five the other, and the five are named.** A 17-and-14 split
with no account of either direction is what a coin does. This is the first change
in the programme whose effect on the LLM path is larger than that path's own
instability — and it is legible *because* the floor was measured first.

### 4.5 The 90.7% understates, and by a knowable amount

**Four of the five reverse-direction changes are Q4.4 clarifying** — the four
runs where "based on the last few weeks" now says the run-rate cannot express
weeks, which §10.1 establishes is the correct behaviour. The frozen scorer grades
them `SAFE_REFUSAL`, because it cannot distinguish *a correct clarification* from
*a failure to answer*. Both are the absence of a figure; only one is a defect.

The scorer is **not** changed. It is the control, it graded both sides of every
comparison in this programme, and a control edited to flatter a result is not a
control. So the headline stands as measured — but it is stated here that it
understates, and by how much: **four runs of 752**, named, on one variation, for
one reason.

This is the mirror image of the byte-equality finding. There, a frozen baseline
was **protecting wrong answers** — "How many loans are in the back book?" had
been answering with a balance and passing, because the only test was that it
matched last time. Here, a frozen scorer is **penalising a right change**. Both
follow from the same property: a control that pins behaviour cannot also judge
whether the behaviour is correct. Both need an independent declared expectation
to sit beside them, which is what `expected_answer_type` and
`answer_types_44.yaml` now provide for type, and what nothing yet provides for
the answer/clarify boundary.

## 5. Tranche E — client-shaped fixtures

### 5.1 A defect in the EXISTING generator, found before building the new one

The completion distribution in the current twelve weekly extracts is not a book
converting. It is a generator that could not have produced completions any
earlier.

| extract | 04-13 … 06-15 (ten) | 06-22 | 06-29 |
|---|---|---|---|
| COMPLETED cases visible, Alderbridge | **0** | 10 | 32 |
| COMPLETED cases visible, Kestrelmoor | **0** | 15 | 46 |

**The mechanism, confirmed arithmetically rather than inferred.**
`tests/analytical/pipeline_fixture.py` creates every case *inside* the published
window: cohort 0 enters at the first extract. A case reaches COMPLETED after
`10 + pace` weeks, `pace ∈ {0,1,2,3}`. With twelve extracts (indices 0–11), only
cohort 0 at paces 0 and 1, and cohort 1 at pace 0, can ever reach ten weeks.
Every other cohort is arithmetically incapable of completing. The pack has **no
pre-history** — no population already in flight when the window opens.

Two further consequences of the same cause: **no case withdraws at any point in
the pack**, and the dwell times are compressed against the governed lags —
KFI→COMPLETED is 70 days where `stage_days_to_fund` says 90, APPLICATION→
COMPLETED 49 against 60, OFFER→COMPLETED 28 against 30.

### 5.2 What this does and does not invalidate

Stated precisely, because "the fixture was wrong" is not the same claim as "the
figures were wrong".

**Not invalidated — Tranche B.** Excluding a settled case from a *forward*
forecast is correct whatever the data looks like; the population decision does
not depend on the conversion pattern. The independent recomputation that agreed
to the penny read the same extracts and checked arithmetic, and that check
stands. What is fixture-specific is the **magnitudes**: £9,625,160.91 is an open
book weighted by rates of 0.0635 / 0.1016 / 0.1693, and those rates will move
materially on a fixture where cases convert throughout. They were always going
to move on real data; what is new is knowing they came from a near-degenerate
sample rather than a thin one.

**Not invalidated — Tranche C. Strengthened.** C found this and said so: *"every
observed completion happened at exactly the same elapsed time"*, *"every
completion falls in the last two weekly extracts"*, *"there is not one withdrawal
in the entire twelve-extract history"*, and *"the magnitudes above are a property
of the generator, and no claim about a real client's conversion rate should be
drawn from them."* C's conclusion — that the data cannot identify a completion
rate at the horizons the forecast assumes — is exactly what this mechanism
predicts. C described the symptom; this is the cause.

**What does change: the standing of the Tranche F question.** F is built to test
whether an empirical rate can displace a configured one where a maturity test
passes. On the current fixture that test cannot pass at any stage, so F would be
testing its own fallback. The client's extracts carry completions in material
volume throughout, so the fixture is less demanding than reality **in exactly the
dimension F exists to measure**. That is why the generation spec below changes.

### 5.3 Generation specification

Added to E1/E3, and binding on the new pack:

* **Completions distributed across the whole fifty-two-week window**, at a pace
  consistent with each stage's `stage_days_to_fund` — KFI 90d, APPLICATION 60d,
  OFFER 30d — not bunched at the end.
* **Cases enter at KFI and progress with realistic dwell times**, completing or
  withdrawing continuously from early in the window. This requires a **warm-up
  cohort**: cases entering before the first *published* extract, so a population
  is already in flight when the window opens.
* **Withdrawals occur throughout**, at a rate that makes the governed
  `exclude_stages` contract meaningful rather than vacuous.
* **The stationary pattern is built first.** The declining-KFI stress case only
  means something if there is a normal conversion pattern underneath it to
  decline from; the decline is then applied to new KFI intake in the final three
  months on the nominated book, with the other book held stationary as a control.
* **The completions-per-week series and the matured observation count per stage
  are reported for both books BEFORE anything else runs on the new fixtures.**
  If completions are still bunched, work stops: the fixture is wrong and every
  downstream measurement inherits it.

### 5.4 The V1 set is untouched

The existing three funded snapshots and eleven weekly extracts, and every
artefact hashed against them, stay byte-identical. The new pack is an additional
set with its own manifest section and its own baseline. If a tool would rewrite
an existing fixture, work stops.

## 6–9. Tranches F and G

*(Conversion methodology and concentration limits. Not started.)*

## 10. Bank results after Tranche D

Deterministic parse unless stated.

| bank | result |
|---|---|
| 252-case calibration | **259 passed, 1 xfailed** (from 245 / 13) |
| 30-question simple-MI | 28 ok / 30; 1 answer changed, a count question that had been answering with a balance |
| 80-question wide | 66 ok / 80; 3 false refusals became answers, 2 answers became clarifications |
| 44-variation NL, LLM path | unsafe **0**; CORRECT 675 → **682** (90.7%); substantive 622 → **628** (83.5%) against the 82.7% floor; refusals 77 → **70** |
| type conformance, four banks | **0** findings, from 5 |

**Two answers became clarifications, both individually justified**, as the
anti-gaming rule requires. *"How much has the book grown this year?"* asked for
12 reporting periods where the book carries 3. *"How has LTV moved over the last
three months?"* asked for a 3-period span where the furthest reach is 2. Neither
is a capability loss: both were previously answering over a period the reader
did not ask for.

### 10.1 Honour the stated period, or clarify

A declared period that cannot be honoured is a clarification, not a narrower
answer with a note attached. Disclosure is not honouring.

| question | before | after |
|---|---|---|
| "over the last 2 months" | 31 May → 30 Jun | **30 Apr → 30 Jun** (honoured) |
| "this year" | 31 May → 30 Jun | **clarifies** — 12 asked, 3 available |
| "over the last three months" | answered | **clarifies** — 3 asked, 2 reachable |
| "what has changed?" | latest pair | unchanged |

Written that way it needs no revision once Tranche E supplies twelve months: the
same code answers instead of clarifying.

**The same rule applied to granularity.** *"Based on the last few weeks, what
level of completions are we achieving?"* pins no count, so there is no span to
fail — but it pins a **unit**, and the run-rate is measured from month-end
funded snapshots. It first gained a disclosure ("based on 2 month(s) of funded
growth"), which named the window without answering the question asked. The
weekly extracts cannot stand in: **ten of the twelve carry no completion at
all**, so a weekly completion rate would rest on two observations in the final
fortnight — the censoring artefact Tranche C documents, not a rate. So it
clarifies, on four runs of the bank. Questions that name no unit ("what
completion rate are we running at?", "what's our recent run rate?") answer, with
the window disclosed.

## 11. What ships gated and what ships open

*(Settled after Tranche F. Forecasting ships gated by construction; Tranche D
changed nothing about that.)*

## 12. Post-onboarding backlog

**Deferred by the brief, and not started:**

* **A time-series intent family with a line artifact.** The largest genuine
  capability gap against historical answerability, and the first thing after
  onboarding. **The type-conformance sweep's specialist-route blind spot belongs
  with this work**: the sweep types an answer from the spec's metric slot, so a
  route carrying its measure in its own identity is invisible to it — the
  balance bridge, the period-movement digest, the conversion-rate answer. Those
  are the routes that will carry most of the load once twelve months of history
  arrives, so the harness extension (type from the structured findings, as the
  44-bank sweep already does) should land with that capability rather than float
  free. The claim today covers **313 of 366** executed cases, listed by name in
  `tranche_d/sweep_caveat.md`.
* **NNEG exposure.** The field does not exist in the tape; it stays a refusal and
  is disclosed to the client up front.
* **Any new analytical family, vocabulary expansion, or planner redesign.**

**Added by Tranche D, against the clause-splitter work:**

* **Categorical domain validation**, sited in **field resolution** rather than as
  an envelope guard. Whether "offer" is a valid `account_status` is a question
  about what the field means — business semantics resolving a value inside a
  filter span. §4.3 is the case that raised it.
* **Guard convergence.** There are now three guards doing partial versions of one
  job: the concept guard (two population concepts only), the bounds guard
  (values), and the proposed domain guard (categories). Each was added when a
  defect exposed a gap; none covers the whole. The convergent form is a **single
  validation pass over the resolved filter — field exists, value in domain, bound
  present in the question**. One check, three conditions, one place it can be
  wrong. This is the same fragmentation pattern as the whole-string scans in
  §13, and should be fixed the same way.

## 13. A standing rule the tranche produced

The same defect appeared **four times, in four independent places**:

| where | what it did |
|---|---|
| `llm_query_parser._detect_metric` | "balance where LTV above 50%" resolved to weighted-average LTV |
| `llm_query_parser.wants_balance_too` | "how many loans have a balance above £250k" answered with a balance |
| `answer_type.asked`, first version | typed "balance by region where borrower age is over 70" as an AGE question |
| `answer_type.asked`, second version | typed "balance by LTV bucket" as a RATE question |

Two of those are in the instrument built to find the other two. Four
occurrences is a design rule, not four bugs:

> **Anywhere wording is scanned to fill a slot, it reads the SUBJECT SIDE, never
> the whole string.** A measure named inside a condition is the field being
> filtered on. A measure named inside a grouping clause is the axis, not the
> answer. Both belong to clauses already consumed by another slot, and a scanner
> that sees them is competing for a word that is spoken for.

Two helpers implement it — `llm_query_parser._metric_slot` and
`answer_type.subject_side` — and they agree on the conservative rule for
conditions.

**Where the pattern may still exist, unaudited.** Every occurrence so far was
found by a defect rather than by a search. The shape to look for is a regex or
vocabulary scan over the whole question that decides one slot:
`_explicit_dimensions`, `_parse_filters` and the categorical-value scan,
`_detect_periods`, `_relative_mode`, `_forecast_question_kind`,
`_risk_limit_category`, `detect_measure_set`. None is asserted defective; they
are the population a systematic audit should cover, and that audit is in the
backlog rather than done here.

## 14. Evidence

Under `due_diligence/evidence/forecast_composition_hardening/tranche_d/`:
the type sweep across four banks, its **pre-D control run** proving it finds the
three defects D fixed, the vocabulary blast-radius sweep, the self-disagreement
measurement before and after, the deterministic-substitution check against the
frozen expectations, the LLM run files after D4, and the written caveat on the
sweep's own reach.

Two new expectation controls ship with the tranche:
`expected_answer_type` on all 252 calibration cases, enforced by the evaluator
with a test that it can fail; and `answer_types_44.yaml`, a companion to the
frozen expectation file which remains **byte-identical** and still verifies
against the manifest.
