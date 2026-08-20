# MI Query Agent — Client Readiness

Making the agent production-ready for Client 1 onboarding across three shipped
capabilities: historical MI answerability; slicing, dicing and charting governed
fields into bespoke views; and reasonable answering of the nine multi-layered
intents. Forecasting is not in the shipped scope — it is measured so the
decision to enable it later is evidenced, and it ships gated.

**Status: Tranches D and E complete. Tranches F and G not started.**

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
| calibration bank (against `build_fixture`) | 245 passed / 13 xfailed | 259 passed / 1 xfailed — **both figures withdrawn, §5.11** |
| calibration bank (against a real funded tape) | — | **231 passed / 21 xfailed** |
| silent substitutions found by an independent type sweep | 5 | **0** |
| parser self-disagreement (176 cells) | 10 (5.7%) | **2 (1.1%)** |

The single remaining `xfail` refuses safely and fabricates nothing. **The
reason recorded against it was wrong, and Tranche E proved it wrong** — see
§5.11a. It said `exposure to London` would answer once the bank ran on a real
book, because the synthetic tape carries no London. A real book carries 1,380
London loans and the case still fails: the governed region filter targets
`geographic_region_obligor`, which holds ITL3 codes on every real book here.
The reason field is corrected in place; the `xfail` stands.

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

### 5.5 The stop-gate: the new pack, before anything else ran on it

Both books, stationary, `decline=None`. The gate was mandated before any
downstream measurement, and the condition for stopping was "completions still
bunched".

| | week-1 stock inherited from warm-up | flow, weeks 2–52 | zero-weeks | expected steady state |
|---|---|---|---|---|
| alderbridge | 211 completed, 902 withdrawn | mean **8.12**/wk, sd 2.68, range 4–13 | **0** | 42 × 0.20 = 8.4 |
| kestrelmoor | 341 completed, 1,264 withdrawn | mean **12.41**/wk, sd 3.49, range 3–20 | **0** | 61 × 0.20 = 12.2 |

Drift across the window is −0.08/wk and −0.76/wk: noise, not trend. Withdrawals
occur in **every** week (34/wk and 49/wk) where the V1 pack had none anywhere,
and intake balances — 42 = 34 withdrawn + 8 completed.

One point of interpretation, because it would otherwise be misread as the old
defect: **week 1 is stock, not flow.** At the first published extract there is
no prior snapshot, so every case the warm-up already carried to a terminal state
is indistinguishable from one that completed that week. That is what a real first
extract looks like. The bunching defect was its mirror image — an *empty* start
and a spike at the *end*.

**Matured observations per stage.** A case observed at stage S in week *w* is
matured only if *w* + `stage_days_to_fund[S]` lands on or before the last
extract.

| book | stage | matured | immature | converted | observed rate | target | delta |
|---|---|---|---|---|---|---|---|
| alderbridge | KFI | 1,734 | 546 | 345 | 0.199 | 0.20 | −0.001 |
| | APPLICATION | 851 | 167 | 373 | 0.438 | 0.45 | −0.012 |
| | OFFER | 524 | 52 | 407 | 0.777 | 0.75 | +0.027 |
| kestrelmoor | KFI | 2,527 | 793 | 521 | 0.206 | 0.20 | +0.006 |
| | APPLICATION | 1,260 | 228 | 568 | 0.451 | 0.45 | +0.001 |
| | OFFER | 810 | 79 | 624 | 0.770 | 0.75 | +0.020 |

This is the property the V1 pack could not supply and Tranche F requires: the
generating rates are **recoverable from the data**, so an empirical estimator can
be judged against ground truth rather than against itself. The immature column is
the maturity trap made concrete — 546 KFI observations on alderbridge whose
outcome the window cannot yet know.

**Gate passed.** Nothing was stopped.

### 5.6 Two fixture properties that needed deciding, and were not in the spec

**A retention window on terminal cases.** The first build carried every
terminated case forever. The live pipeline was stationary — 250 cases in week 1,
249 in week 52 — but the extract grew 882 → 3,024 rows and the live share fell
28.3% → 8.2%, purely from stock accumulation. A time-series question over the
pipeline dataset would read that monotone drift as a real trend. Terminal cases
are now carried 26 weeks and then dropped, which is what an operational feed
does. This is safe for the historical model, which unions case timelines across
all snapshots rather than reading the last one.

**A warm-up long enough to cover the retention window.** 20 weeks was enough to
put a funnel in flight but not enough to open with a fully accumulated terminal
stock, so composition still drifted to a plateau *inside* the observation window
— a smaller version of the defect being corrected. `WARMUP_WEEKS` is 40: the
longest KFI-to-completion dwell (~13 weeks) plus the 26-week retention.

After both, the extract is stationary in **size and composition**, not just in
flow:

| week | rows | live | completed | withdrawn | live % |
|---|---|---|---|---|---|
| 2025-07-04 | 1,347 | 234 | 211 | 902 | 17.4% |
| 2025-11-21 | 1,341 | 253 | 205 | 883 | 18.9% |
| 2026-04-10 | 1,361 | 260 | 212 | 889 | 19.1% |
| 2026-06-26 | 1,342 | 250 | 210 | 882 | 18.6% |

### 5.7 The retention window immediately caught a defect — in the measurement, not the fixture

The first matured-observation run after retention showed every rate roughly
halved: OFFER 0.387 against a target of 0.75. The fixture was not wrong. **The
measurement script was**: it read each case's outcome off the *final* extract, so
any case the feed had already dropped counted as a non-conversion.

That is precisely the trap Tranche F exists to avoid, reproduced inside the
instrument built to measure it — the same shape as the answer-type classifier
that carried the subject-side defect it was written to find. Correcting it to
union outcomes across snapshots, as `pipeline_history` does, recovers the rates
in the table above. The naive figure is kept in the evidence because the gap
between the two columns is the finding:

| book | stage | correct rate (union across snapshots) | naive rate (final extract only) |
|---|---|---|---|
| alderbridge | OFFER | 0.777 | 0.387 |
| kestrelmoor | OFFER | 0.770 | 0.375 |

### 5.8 What was built

Three additive modules, none of which touches a V1 generator:

| module | what it produces |
|---|---|
| `tests/analytical/client_fixture.py` | 52 weekly pipeline extracts per book, warm-up, governed dwell times, ground-truth stage rates, withdrawals, retention |
| `tests/analytical/client_funded.py` | 13 monthly funded snapshots per book, 2025-06-30 → 2026-06-30 |
| `tests/analytical/client_pack.py` | the single entry point every E/F/G measurement runs against |

`second_book.build()` gained keyword parameters for the client, portfolio set,
region weights, seed and reporting dates. Every one defaults to the module
constant it replaced, and that was **verified rather than assumed**: building
with the pre-change module and the current one produced ten artefacts identical
by SHA-256.

Two properties worth stating because they bound what a later measurement can be
blamed on:

* **Deterministic.** Two independent builds produce the same pack digest,
  `4ee0282d…`. 163 artefacts, 219,353,539 bytes.
* **Strict superset.** The thirteen-month kestrelmoor history reproduces the V1
  three cuts **byte-for-byte**. Any measurement that moves between the two
  fixtures is therefore attributable to the added history, not to a regenerated
  book.

The pack is **not committed**. That follows the repository's own convention
rather than departing from it — the V1 funded tapes under
`demo_platform/workspace/` are gitignored and reproduced from a committed
generator, with only their hashes in the manifest. At 211 MB raw (~30 MB
gzipped) a committed copy would prove strictly less than the digest does: a
digest fails when the generator drifts, a copy silently diverges from it.
`due_diligence/evidence/client_readiness/hash_fixture_pack.py --check` rebuilds
and verifies; `tests/test_evidence_manifest.py` runs it in the suite.

### 5.9 The nominated stress book

**kestrelmoor** carries the declining-KFI profile; **alderbridge** is the
stationary control. Two reasons, both about measurability: kestrelmoor has the
larger pipeline (61 new cases/week against 42), so a 65% fall in intake still
leaves a usable sample in the final weeks; and alderbridge is the book the V1
evidence and the demonstration platform are anchored to, so leaving it stationary
keeps a control comparable with every existing measurement.

The decline runs over the final thirteen weeks and applies to **new KFI intake
only** — a lender's existing pipeline does not evaporate when origination slows.
The resulting shape is the point of the stress case:

| | week 39 | week 52 |
|---|---|---|
| new KFI intake | 61/wk | **21/wk** |
| live pipeline | 368 | 203 |
| KFI stock | 218 | **87** |
| OFFER stock | 59 | 50 |
| completions | 11/wk | 13/wk |

The fall is **top-weighted** and completions **lag rather than collapse**. A
forecast extrapolating an older run-rate over-states; a maturity treatment that
reads the thin recent KFI cohort as a wave of non-conversions under-states.

### 5.10 E3 baseline: the nine acceptance questions on the new fixtures

Deterministic parse, both books, recorded in
`evidence/client_readiness/baseline_e3.json`. **5 of 9 carry structured findings
on each book.** Q2 and Q4 answer through the run-rate route, which emits no typed
finding — the specialist-route gap already logged in Tranche D. Q5 and Q6 return
the controlled "no Schedule 8 limits available" refusal, which is Tranche G's
subject.

The longer history did what it was built to do. **Q8** now answers across
2025-06-30 → 2026-06-30 where it previously had two months to work with, and
**Q4** reports a run-rate "based on 12 month(s) of funded growth". The Tranche D
prediction — *"written that way it needs no revision when twelve months of
history arrive: the same code answers instead of clarifying"* — holds.

**One new defect, and the longer fixture is the only reason it is visible.**

Q1 asks *"How has the profile of new originations changed in the last few
months?"* On the new fixture it is answered over **2025-06-30 → 2026-06-30**:
twelve months, `method: explicit_dates`. The cause is direct — `requested_span`
returns `None` for a vague span:

```
'in the last few months'   -> None
'a few months ago'         -> None
'this year'                -> SpanRequest(label='this year', periods=12)
'over the last three months'-> SpanRequest(label='the quarter', periods=3)
```

so the route falls through to the widest available pair. **On the V1
three-snapshot fixture that fallback gave two months, which reads as "a few".**
The defect was always there; a fixture whose history happened to match the
stated span was concealing it.

It is the exact mirror of the "this year" defect Tranche D fixed. There, a
*shorter* window was substituted for the one asked. Here, a *longer* one is —
and it is the more dangerous direction, because a twelve-month comparison of a
rolling front-book cohort is a materially different question from a three-month
one, and nothing in the answer says the span was chosen rather than asked for.

**Not fixed here, and deliberately so.** The repair needs a semantic ruling this
report should not make unilaterally: what "a few months" resolves to. The three
options are to pin it to a number (3 is the natural reading), to clarify rather
than answer, or to honour the widest span while disclosing that the stated period
was imprecise. The first two are consistent with the honour-or-clarify rule
already in force; the third is the current behaviour minus the silence.
**Recommendation: clarify.** "A few" is genuinely imprecise, the codebase already
has the clarification path from D, and pinning a number invents a precision the
question does not carry. This is flagged for approval rather than implemented
because it changes answers across the 44-variation bank.

### 5.11 E4: the calibration bank on a real book

The 252-case bank has been graded, for its whole life, against
`mi_agent.mi_query_harness.build_fixture` — 400 rows with every column drawn
independently from a uniform distribution over its range. No nulls, no
correlation between any two fields, no skew, no relationship between LTV and
valuation. It is a *shape*, not a book: the right instrument for "does this query
parse and execute", the wrong one for "is the answer right on client data".

Re-pointed onto a real funded tape, unchanged:

| fixture | cases passed |
|---|---|
| `build_fixture` (400 synthetic rows) | **251 / 252** |
| a real book (11,035 loans) | **125 / 252** |

**126 regressions.** The identical 126 cases regress on the committed V1 demo
book *and* on the newly generated one, so none of this is attributable to the new
fixture. Decomposed, because "126 defects" would be a false headline:

| n | class | is it a product defect? |
|---|---|---|
| 60 | Field absent from any real book — `erm_product_type` (32), `broker_channel` (25), `borrower_type` (22), `term_bucket` (3). Refused cleanly, nothing fabricated. | **No.** The system is correct; the bank's expectation is unachievable on a real book. |
| 50 | `geographic_region_obligor` not applied. The field exists but holds **ITL3 codes** (`TLH12`); region *names* live in `collateral_geography`. | **Partly.** The bank pins the wrong canonical field. Which field "region" should mean is a real, open question. |
| 13 | Absent field, and a **different dimension substituted** — Amortisation Type ×12, Age Bucket ×1. | **Yes.** Same class as the D2 no-silent-substitution work. Fail-closed caught every one, so no wrong number shipped. |
| 3 | Answered where the bank expects a refusal — `default_amount` / `arrears_balance` exist on a real book (all zero) but not in the fixture. | **No.** The expectation was fitted to the fixture. |

Confirmed against the committed demo book: `build_fixture` carries five columns
— `borrower_type`, `borrower_structure`, `broker_channel`, `erm_product_type`,
`term_bucket` — that **no real book in this repository has**, and puts English
region names in a field that on every real book holds NUTS3 codes.

**What this does not mean.** It is not a claim that 126 answers are wrong. In
123 of the 126 the system either refused correctly or was caught by the
fail-closed guard; nothing fabricated a figure. What it means is narrower and
worse: **the bank's 251/252 was never evidence about client data**, and the four
classes above were invisible for as long as the only book it ran against was one
that carried every column it asked for.

**Not repaired here.** Re-pointing the bank permanently, deciding what "region"
resolves to, and re-declaring the three refusal expectations are all changes to
a control file with blast radius across every prior measurement. They are put to
approval, not taken.

### 5.11a A prediction this report made, and Tranche E falsified

§1 said of the single remaining `xfail`: *"Tranche E re-bases the bank onto a
real book, where the value exists."* It was tested and it is false.

`risk_211` — *exposure to London* — still fails on a real book. Not because
London is absent: the demonstration book carries **1,380 London loans** in
`collateral_geography`. It fails because the governed region filter is applied
to `geographic_region_obligor`, which holds `TLI43`, not `London`.
`build_fixture` put region *names* in that field, which is precisely why the
case read as a data gap rather than a field-resolution question.

The `known_gap` reason in the bank is corrected in place, and it now records
that the earlier diagnosis was wrong rather than quietly replacing it. The
`xfail` itself is unchanged — the diagnosis moved, the result did not. That
distinction is the point: this is a documentation defect repaired as one, not a
number reworded to fit.

### 5.12 A manifest failure of mine, found by this tranche

Running the new pack's manifest surfaced a failure in the V1 one:
`forecast_composition_hardening/three_axis.py` no longer hashed to its recorded
value. **Tranche D extended that file in place** to add the answer-type axis and
did not update the manifest, so V1 verification failed from commit `c9ac20b`
onward — through the rest of the tranche — and nothing caught it, because no test
ran the verifier. The manifest did its job; the process around it did not.

Repaired additively, the same way as everything else in this sprint:

* `three_axis.py` restored to its hashed content. The V1 manifest is untouched
  and verifies: **124 of 124** artefacts.
* The extended instrument moved to `tranche_d/three_axis_typed.py`. It
  reproduces the recorded `det_post` figures exactly — 176 answer-type matches,
  0 diverge, 740/752 semantic, diverging variation Q2.3 — so the Tranche D typed
  numbers stand on an instrument that is now itself hashed.
* `tests/test_evidence_manifest.py` runs both verifications in the suite.

The general point is the one the byte-equality finding already made from the
other side: **a control that nothing runs is not a control.**

### 5.13 The four decisions, as ruled and implemented

All four were put up, ruled on, and implemented. The rulings, and what each
turned out to mean once built:

**1. Vague recency resolves to the governed window — not a clarification.**
The silent widening goes regardless of what replaces it. What replaces it is the
seasoning configuration's own `lending_windows.recent_max_months`, because that
convention exists precisely to settle this and declining to apply our own
governed configuration is hard to defend on a phrase this common. Every vague
phrasing resolves identically, since handling near-identical wordings
differently is worse than either choice. Clarification stays reserved for spans
with no governed convention — "the last few weeks" keeps the granularity path.
The window is read from config, not pinned in code, and disclosed in the answer.
Q1 now answers 2026-03-31 → 2026-06-30; Q8, which names no period, correctly
keeps the widest window.

**2. The bank is permanently on a real book; the old figure is withdrawn.**
`run_bank` defaults to a real funded tape and `default_bank_frame` **raises**
rather than falling back — a silent fallback is how the bank came to measure
something other than what it claimed. 251/252 and 245/13 are withdrawn from the
V1 report and this one rather than caveated. The number is **231/252 passed, 21
xfailed**.

**3. Region resolves through the mapping the transformation already builds.**
The correction to the framing was the substance: raw tapes do not generally
carry ITL codes at all, so this was never a choice between two representations.
A field declares `value_domain: uk_region` in business semantics and the
executor asks the semantics what a value means — postcode district, postcode
area, ITL3 code, ITL3/ITL2/ITL1 name and common aliases all reach the same rows.
The check that matters: on a book stripped of its readable region column,
"London" resolves to 1,380 rows, exactly the count the name column gives.

But the measurement that followed changed what this decision was *for*. On a
real book the product **already** resolved "region" correctly — "balance by
region" groups by `collateral_geography`, "exposure to London" filters it to
1,380 rows. All 50 region regressions were the bank's expectation pinning the
NUTS3 field because `build_fixture` had no readable one. So the resolver is not
what fixed those 50 cases; re-pointing the expectation was. The resolver earns
its place for the case a real client will actually present — a book whose tape
carries a postcode and nothing else — and for the terms no client would ever
type a code for.

**4. The three refusal expectations are re-declared, with the reason per case.**
Against the test asked for — answerable on *any* real book, or only on this one?
`default_amount` and `arrears_balance` are canonical Annex-2 fields on every book
onboarded through the transformation; `build_fixture` simply omits them. On the
demonstration book they correctly return 0.00 across 11,035 loans. Refusing to
report a field the book does report is the wrong behaviour, and zero is an
answer. They use the same `requires_fields` mechanism as the other 76, so a book
that genuinely lacks the field still refuses — no special case.

**The mechanism the first three decisions produced.** 76 cases now declare
`requires_fields`. That is a **prerequisite, not a re-declared expectation** —
the distinction the fourth decision insisted on, generalised. "Balance by broker"
is answerable on any book reporting a broker; no real book here does. Where the
field is absent the evaluator demands a controlled refusal that **names** it,
which is stricter than the original expectation rather than weaker.

**What re-pointing then exposed.** The 21 remaining failures are one defect
class at two severities, and the more serious one was invisible before:

* **13 cases** — the resolver reaches for a *different* dimension (Amortisation
  Type, Age Bucket) when `borrower_type` is absent. Fail-closed catches the
  substitution and refuses, so no wrong number ships.
* **8 cases** — the answer **discloses** that the field is unavailable and still
  emits a data artifact computed over the broader population. *"How many joint
  borrowers are there"* returns a KPI of 11,035 loans and £1.96bn: the whole
  book. This is **disclosure without honouring** — the exact pattern Tranche D
  ruled against for periods, never applied to populations. A declared element
  that cannot be honoured is a clarification, not a broader answer with a note
  attached.

Both were invisible for as long as the only book the bank saw fabricated the
column. The second is the most consequential finding of the tranche and is
recorded in the backlog as the population half of the honour-or-clarify rule.

### 5.14 The retention window, verified rather than assumed

The 26-week retention window was a modelling assumption about a feed nobody had
looked at. Checking it produced a finding worth more than the assumption.

**There is no real M2L extract anywhere in this repository.** Both multi-week
packs in the tree are synthetic — `tests/fixtures/client_001_mi_pack` is written
by "Synthetic Lender Ltd" — and they contradict each other:

| pack | what happens to a terminated case |
|---|---|
| `client_001_mi_pack` (3 monthly extracts) | dropped at the **very next** extract — withdrawn and completed cases present in October are gone in November |
| `hist_api_qvyl2kad` (3 monthly extracts) | **never** dropped; all 16 cases persist across all three |

Neither is 26 weeks, neither is evidence about the client's feed, and both are
too small (8–16 cases) to be authoritative about anything. So the assumption
cannot be verified from this tree, and saying so is the honest position.

**What was done instead: measure whether it matters.** The pack was built at
three settings spanning the whole plausible range and every property Tranche E
rests on was recomputed at each.

| retention | rows wk1 → wk52 | live wk1 → wk52 | completions/wk | KFI | APP | OFFER |
|---|---|---|---|---|---|---|
| 1 week | 286 → 295 | 234 → 250 | 8.12 | 0.199 | 0.438 | 0.777 |
| 26 weeks | 1,347 → 1,342 | 234 → 250 | 8.12 | 0.199 | 0.438 | 0.777 |
| never drop | 1,722 → **3,864** | 234 → 250 | 8.12 | 0.199 | 0.438 | 0.777 |

Every recovered rate, the live pipeline and the completion flow are **identical
across the entire range**. The only thing retention changes is the extract's
size, and only the unbounded extreme misbehaves — the monotone drift this
window was introduced to remove.

So the fixture-realism gap was closed without opening another: no Tranche E
figure depends on the assumption, and the only property that does — "not
unbounded" — holds at every setting a real feed could plausibly have. The window
remains a parameter to be **measured against the client's actual feed at
onboarding**, and it is recorded in the backlog as such rather than defended
here.

That the recovered rates are unchanged even at one-week retention is also an
independent check on the union-across-snapshots correction of §5.7: a terminal
state observed in a single snapshot is enough, exactly as the method claims.

## 6–9. Tranches F and G

*(Conversion methodology and concentration limits. Not started.)*

## 10. Bank results after Tranche D

Deterministic parse unless stated.

| bank | result |
|---|---|
| 252-case calibration, real book | **231 passed, 21 xfailed** (the `build_fixture` figures are withdrawn, §5.11) |
| 30-question simple-MI | 28 ok / 30; 1 answer changed, a count question that had been answering with a balance |
| 80-question wide | 66 ok / 80; 3 false refusals became answers, 2 answers became clarifications |
| 44-variation NL, LLM path | unsafe **0**; CORRECT 675 → **682** (90.7%); substantive 622 → **628** (83.5%) against the 82.7% floor; refusals 77 → **70** |
| type conformance, four banks | **0** findings, from 5 |
| `tests/` + `mi_agent/tests/` + `mi_agent_api/tests/` | **9,036 passed, 0 failed**, 26 skipped, 9 xfailed |

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

### 12a. Added by Tranche E

**Honour-or-clarify for POPULATIONS (highest priority).** The rule is
implemented for periods and not for populations. Eight calibration cases pin it:
a question naming a field the book does not report is answered over the broader
population with a disclosure attached, rather than clarified. `borrower_type` is
the field that exposes it; the defect is general. Fix beside the field resolver.

**Field resolution must not substitute a dimension the question did not name.**
Thirteen cases, caught fail-closed today. Same class as the D2 no-silent-
substitution work, and the same site as the categorical domain validation
already queued.

**Measure the client's actual extract retention at onboarding.** §5.14 shows no
Tranche E figure depends on the 26-week assumption, and that the only property
that matters — bounded rather than unbounded — holds across the whole plausible
range. It is still a parameter to be measured against the real feed rather than
assumed, and there is no real M2L extract in this repository to measure against.

**A second `value_domain` will test whether the seam is real.** Region is the
first. The claim that adding a domain is "a registry entry and a resolver, not a
change to the query path" is unproven until a second one exists.

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

## 13a. Two patterns this programme keeps producing, and the rules that follow

Both are recorded as patterns rather than incidents because each has now
happened enough times to be predictable, and a predictable failure that is not
guarded is a choice.

### 13a.1 An instrument reproduces the exact defect it was built to find

**Five occurrences.** Not five variations on a theme — five instances of the
same mechanism, each found only because something independent disagreed with the
instrument.

| # | instrument | the defect it was built to find | the defect it contained |
|---|---|---|---|
| 1 | `answer_type.asked` | a metric slot filled from the wrong part of the question | scanned the WHOLE question, typing "balance by region where borrower age is over 70" as an AGE question |
| 2 | `answer_type.asked`, after the first fix | the same | read PAST `by`, typing "balance by LTV bucket" as a RATE question |
| 3 | `three_axis.py` figure check | figures printed that no finding holds | flagged bucket labels ("30-40%") and rounded renderings ("£2.3m") — it measured the extractor, not the product |
| 4 | the P1A test | a population computed over the wrong rows | asserted the defect in its own comment: "deliberately not attempted inside P1A" |
| 5 | `matured.py` (Tranche E) | conversion rates read off censored data | read outcomes off the FINAL extract, scoring every dropped case as a non-conversion and halving every rate |

The mechanism is the same each time: **the instrument and its subject share an
assumption, so the instrument cannot see the defect it is looking for.** Number 5
is the clearest — a script written specifically to avoid the maturity trap fell
into a neighbouring one, and only the retention window introduced in the same
commit exposed it.

> **STANDING RULE — every measurement instrument ships with a test proving it can
> fail.** Not a test that it runs, and not a test that it passes on good input: a
> test that feeds it the defect it exists to detect and requires it to report
> that defect. An instrument that has never been observed failing is not evidence
> that its subject is clean; it is an untested claim about the instrument.

Applied in this tranche: the answer-type check ships with a case run against a
deliberately wrong declared type; the pre-D2 control run proves the type sweep
finds the three defects D2 fixed; the manifest verifier now has a test that
corrupts an evidence hash and requires a non-zero exit; the region resolver has a
test that an unmatched term yields no rows rather than broadening.

### 13a.2 Controls need controls

**Two occurrences.** Both silent, both found by accident, both a control that
had stopped controlling while continuing to look like one.

**The stale manifest.** Tranche D edited a hashed instrument in place. Manifest
verification failed from `c9ac20b` onward and nobody noticed for the rest of the
tranche, because nothing ran the verifier. Repaired in §5.12, and
`tests/test_evidence_manifest.py` now runs it in the suite.

**The verifier that cried wolf.** Adding it to the suite immediately failed on
two production files this sprint legitimately changed — and investigating rather
than regenerating found the deeper defect. The manifest's `code+config` group
records which production code the manifest was generated against; all fifteen
entries hash to tree `34611f0`. Production code is precisely what this programme
exists to change, so **the verifier failed on every legitimate code change**.

That is not a separate problem from the first one. It is its cause. A control
that fails on ordinary work is a control that gets ignored, and an ignored
control is where a real failure hides — which is exactly what happened: the one
genuine alarm was indistinguishable from the noise it had been emitting all
along.

Repaired in the verifier rather than the manifest, which stays immutable:
`code+config` entries are verified against the tree they were recorded at, and
working-tree divergence is reported as classified INFO naming each file;
evidence drift stays fatal.

> **STANDING RULE — a control is not finished until something runs it on every
> change, and until it distinguishes the failures that matter from the movement
> it should expect.** A control nobody runs is documentation. A control that
> fires on routine work trains its readers to ignore it, which is worse than
> having none, because it also supplies false assurance.

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
