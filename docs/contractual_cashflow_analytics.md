# Contractual cash-flow analytics

*Sprint 2.5D. Baseline `3ddf7af` → `7026281`.*

*Prior analysis: `docs/mi_metric_methodology_review.md` §11–§12.*

## 1. Executive answer

> **Were contractual WAL and YTM genuinely missing data capabilities, or were
> they mostly latent deterministic analytics?**

**Latent, and by a wide margin.** Building them added **no canonical field, no
regime mapping and no alias**. Every input came off the existing governed model:
RREL35 amortisation type, RREL42 rate type, RREL30 balance, RREL24 maturity,
RREL37/38 frequency, RREL39 payment due, RREL41 balloon, RREL43 rate, RREL34
price, RREL6 cut-off. What was missing was arithmetic, not information.

Two genuine gaps survive, and both are narrow: `day_count_convention` exists
only as CREL122 on Annex 3, which bounds the yield to a **periodic** basis on
residential tapes; and floating-rate exposures need a future index level, which
is an assumption rather than a field. Neither was fixed here, and neither was
faked.

The one case that is *not* a gap at all is Trakt's own book. Equity release
reports `OTHR`, repays on death, sale or long-term care, and therefore has **no
contractual WAL to compute**. That is encoded as `NOT_APPLICABLE` with a reason,
and excluded from the readiness framework by asset class, so it cannot be
mistaken for something to go and build later.

---

## 2. `outstanding_balance_period_*` — investigated, not assumed

Eight fields, investigated before anything was built, because if they *were* a
received schedule the whole enumerator would have been unnecessary.

**What the repository actually contains:**

| Evidence sought | Found |
|---|---|
| Canonical registry entries | Yes — eight, all `category: analytics`, `layer: core` |
| Regime code on any annex | **None** |
| `allowed_values`, aliases | **None** |
| Declared format | `_1` and `_600` are `decimal`; the four `_date` fields are `date`; **`_2_120` and `_121_599` have `format: null`** — the registry does not know their type |
| Sample data, fixture, workbook, generator | **None anywhere in the repository** |
| Consumer — code, config, test, route | **None** |
| Introducing commit | Traces to a merge of an unrelated OCC config-admin PR; no specification accompanies them |
| Prior documentation | Two reviews, and they **disagree** |

The disagreement is the decisive evidence. `mi_agent/reports/mi_semantics_review.md`
calls them *"arrears-ageing buckets, not a portfolio balance concept"*.
`docs/business_semantics_registry_review.md` calls them *"Balance-period buckets
(technical)"*. Both **excluded** them; neither cites a specification.

**Classification: UNKNOWN.**

Not CONFIRMED SCHEDULE DATA — nothing documents them and nothing populates them.
Not confidently LEGACY either — there is no evidence they were ever used. If the
first review is right and these are arrears ageing, consuming them as a balance
path would be catastrophically wrong in a way that would look plausible in every
output.

**They are not consumed by any production analytic in this sprint**, and the
uncertainty is recorded here rather than resolved by inference. Resolving it is
a question for whoever specified them.

---

## 3. Payment-date anchoring

Annex 2 carries no payment-date field. `next_payment_date` (CREL104),
`payment_date` (CREL102) and `start_date_of_amortisation` (CREL16) are all
Annex 3. So dates must be anchored, and the choice is made from the regime's own
wording rather than convenience.

**Methodology `CONTRACTUAL_SCHEDULE@v1`, anchor preference:**

1. **`next_payment_date` (CREL104) where present.** Where the tape states the
   date, deriving one instead would substitute arithmetic for a fact. Published
   as `anchor_basis: explicit_next_payment_date`.
2. **Otherwise count back from `maturity_date` (RREL24)** at the contractual
   frequency to the first date after the as-of date. Published as
   `anchor_basis: counted_back_from_maturity`.

**Why backwards, not forwards from origination.** Two field definitions settle
it:

- **RREL41** defines the balloon as "principal repayment to be paid **at the
  maturity date**". The terminal flow must land exactly on maturity. Counting
  back guarantees that; counting forward leaves a short final period and puts
  the balloon in the wrong place — and the balloon is the largest single flow in
  any schedule that has one.
- **RREL58** counts "payments made prior to the exposure being transferred to
  **the securitisation**". It anchors to the transfer, not to the reporting
  date, so it cannot locate the next payment relative to `data_cut_off_date` at
  all. It is not a usable forward anchor, whatever its arithmetic appeal.

**Limitation, disclosed rather than hidden:** any stub period therefore falls at
the **front** of the schedule, and the schedule says so in its notes. Where the
frequency is `OTHR` or absent, no dates are constructed and the result is
`FIELD_GAP` — monthly is **not** assumed.

---

## 4. Decision matrix

Rate types per RREL42: **FXRL** fixed for life; **FXPR/FLCF** fixed until a
contractual revision; **FLIF/FINX/FLFL/CAPP/FLCA** floating.

| Amortisation | Rate type | Schedule | WAL | YTM | Classification |
|---|---|---|---|---|---|
| **BLLT** | FXRL | principal: 1 flow at maturity; interest: known | ✅ | ✅ | **LATENT DETERMINISTIC MI → now READY** |
| **BLLT** | floating / resetting | principal known, interest not | ✅ | ❌ | WAL **READY**; YTM **ASSUMPTION REQUIRED** |
| **FIXE** | FXRL | constant principal; interest on declining balance | ✅ | ✅ | **LATENT DETERMINISTIC MI → now READY** |
| **FIXE** | floating / resetting | principal known, interest not | ✅ | ❌ | WAL **READY**; YTM **ASSUMPTION REQUIRED** |
| **FRXX** | FXRL | constant total instalment split period by period | ✅ | ✅ | **LATENT DETERMINISTIC MI → now READY** |
| **FRXX** | FXPR / FLCF, revision inside the schedule | deterministic only to RREL51 | ❌ | ❌ | **ASSUMPTION REQUIRED** |
| **FRXX** | FXPR / FLCF, revision after the last payment | fully determined | ✅ | ✅ | **READY** |
| **FRXX** | floating | instalment resets; principal split moves with it | ❌ | ❌ | **ASSUMPTION REQUIRED** |
| **DEXX** | FXRL | interest-only first instalment, then as FRXX | ✅ | ✅ | **LATENT DETERMINISTIC MI → now READY** |
| **DEXX** | floating / resetting | as FRXX | ❌ | ❌ | **ASSUMPTION REQUIRED** |
| **OTHR** (ERM) | any | no contractual principal profile | — | — | **NOT APPLICABLE — model-dependent** |
| any | frequency `OTHR` / absent | no payment interval | — | — | **FIELD GAP** |
| any | Annex 2, day-count-exact yield | — | — | ❌ | **FIELD GAP** — CREL122 is Annex 3 only |

**The asymmetry that drives the whole design.** WAL is principal-weighted, so it
depends on the rate *only where the amortisation type makes principal depend on
it*. Under BLLT and FIXE the contract fixes principal directly, so a floating
loan keeps its WAL and loses only its yield. Under FRXX and DEXX the contract
fixes the *total*, so principal is the residual after interest and the two fall
together. The schedule therefore carries `principal_deterministic` and
`interest_deterministic` as **separate flags** — collapsing them into one would
have thrown away the floating-BLLT WAL, which is a real capability.

---

## 5. How each schedule is constructed

**BLLT.** One principal flow, equal to the current balance, on the maturity
date. Interest accrues each period on the full balance. WAL reduces to the time
to maturity, which is what "full principal repaid in the last instalment" means.

**FIXE.** Constant principal per instalment.
`regular_principal_instalment` is used where the tape carries it; it has **no
regime code on any annex**, so where it is absent the constant is recovered as
`(balance − balloon) ÷ amortising periods`. That inference is made **only**
because RREL35 has already asserted the per-instalment principal is constant —
outside FIXE the identical arithmetic would be a fabrication, and the schedule
note says so. The balloon lands on maturity per RREL41, and the final instalment
trues up so the schedule closes exactly on the balance.

**FRXX.** The contract fixes the total. `payment_due` (RREL39) — "the next
contractual payment due by the obligor according to the payment frequency" — *is*
that constant for a constant-total structure, and it is carried on all five
annexes. Where absent, the annuity factor supplies it from balance, rate and
term, which is still contractual arithmetic with no behavioural input; the note
records which was used. Then, period by period:

```
interest_t  = rate × opening balance_(t−1)
principal_t = instalment − interest_t
```

with the final instalment closing the loan.

**DEXX.** As FRXX, except that the first instalment is interest-only per RREL35
— no principal is allocated, and the balance is unchanged going into period two,
so period two accrues on the full balance. This lengthens the WAL against the
same loan amortising from period one, which is asserted.

**OTHR.** No schedule. `NOT_APPLICABLE`, with the reason attached.

---

## 6. Contractual WAL

**Methodology `CONTRACTUAL_WAL@v1`:**

```
WAL = Σ(tᵢ × Pᵢ) ÷ ΣPᵢ
```

| Definition | Choice |
|---|---|
| Time unit | years |
| Date basis | **ACT/365F**, named explicitly — the WAL time axis is a measure, not an accrual basis, but "do not invent a convention" applies to measures too |
| As-of | `data_cut_off_date` (RREL6), or an explicit override |
| Stubs | at the front, per §3, and disclosed |
| Balloons | on the maturity date per RREL41, and in the numerator like any other principal flow |
| Interest | **excluded** — WAL is principal-weighted, which is exactly why a floating bullet has one |
| Missing schedule | the schedule's status and reason are returned; **never a number** |

**Portfolio WAL aggregates the cash flows** rather than averaging loan WALs. The
two are mathematically equivalent when the weights are total principal, and
aggregation is the one that cannot be got wrong by weighting on *balance*
instead — which is precisely where a book with balloons would silently drift.
The number of loans excluded, and by which status, is part of the answer: a
portfolio WAL that quietly dropped the ERM loans would be a different
portfolio's WAL.

**Test evidence** (all hand-computed, independent of the code):

| Case | Expected | Basis |
|---|---|---|
| BLLT 5-year | 5.002740 yrs | 1,826 days ÷ 365 (2028 is a leap year) |
| FIXE 4 × annual, level | 2.501370 yrs | (365+730+1096+1461)/4/365 |
| FIXE + £400k balloon | 2.801644 yrs | (200k×1 + 200k×2 + 200k×3 + 400k×4)/1m, ACT/365F |
| FRXX £100k @ 10% | 2.718 yrs | from the hand-worked split in §7 |
| DEXX vs FRXX | longer | deferring principal one period must lengthen the life |
| ERM | `NOT_APPLICABLE` | no number at all |

---

## 7. Contractual YTM

**Methodology `CONTRACTUAL_YTM_PERIODIC@v1` — periodic, and named so.**

| Component | Basis |
|---|---|
| **Price** | `purchase_price` (RREL34): "the price, relative to par, at which the underlying exposure was purchased by the SSPE. Enter 100 if no discounting was applied." Already a percentage — no cash consideration needed. **A missing price is not treated as par**, because that would turn every yield into the coupon |
| **Cash flows** | principal + interest from the schedule, unchanged |
| **Date/frequency** | discounted at the contractual payment frequency, annualised `(1+y)^(periods per year) − 1` |
| **Day count** | **NOT APPLIED.** `day_count_convention` is CREL122, Annex 3 only. A day-count-exact yield is not claimed and a convention is not invented |
| **Solver** | bisection — chosen over Newton because it cannot diverge on an awkward first period, and a failure to converge is visible rather than silent |

**Supported:** any amortisation type in §4 where the rate is contractually fixed
for the life of the schedule — including a revision date that falls *after* the
final payment, where refusing would be conservative to the point of wrong.

**Limitations, stated rather than buried:**

- Periodic, not day-count exact, on any tape without CREL122.
- Floating and resetting exposures return `ASSUMPTION_REQUIRED`, not a yield
  computed from today's rate as though it were fixed.
- **On a book bought at par, RREL34 is 100 and the yield collapses to the
  coupon.** That is a correct answer that reads like a broken calculation, and
  anyone quoting it should expect the question.

**Worked and asserted:** a 5% bullet at par gives periodic 0.4166667% monthly
and 5.116190% annualised — `(1 + 0.05/12)^12 − 1`, by hand. Bought at 98 the
yield rises above the coupon and at 102 falls below it; the direction is
asserted because a sign error would be invisible in a single number.

---

## 8. Equity release

Explicitly confirmed, and encoded so it cannot be undone by accident:

| | ERM |
|---|---|
| **Legal maturity** | **Present.** `maturity_date` (RREL24) is populated — typically a long-stop age |
| **Contractual WAL** | **NOT APPLICABLE.** Repayment is contingent on death, sale or long-term care. No contractual repayment date exists to weight |
| **Expected WAL** | **MODEL REQUIRED** — mortality, morbidity, voluntary redemption. **Not built** |
| **Contractual YTM** | **NOT APPLICABLE.** Price and legal maturity both exist, which is the trap: the *inputs* are there and the *meaning* is not, because the payment timing itself is contingent |
| **Mortality model** | **None added.** No table, no assumption, no placeholder |

`config/asset/product_defaults_ERM.yaml` already reasons this out — a lifetime
mortgage "rolls up interest and repays at death/sale ... it is **NOT** a
scheduled bullet amortisation under the Annex 2 definition" — and reports `OTHR`.

Three guards make the misleading number hard to produce: the schedule returns
`NOT_APPLICABLE` with the reason in it; the readiness framework excludes both
metrics from `equity_release` by `asset_class_applicability`, so their absence
is not a readiness failure; and a test asserts that the refusal carries a reason
mentioning mortality rather than a bare null, because an agent given `null`
cannot tell "not applicable" from "failed".

---

## 9. Yield to worst

**Not implemented, and not close.** YTW is the minimum yield across meaningful
alternative contractual redemption outcomes, and Trakt does not represent the
option set:

- No canonical field carries a tranche or class balance, coupon, attachment
  point or paydown — the liability side is not modelled at all.
- On the asset side, `prepayment_lock_out_end_date` (RREL60),
  `percentage_of_prepayments_allowed_per_year` (RREL59) and `prepayment_fee`
  (RREL61) describe the *cost and permission* to prepay, not a contractual call
  schedule with dates and prices.
- The Annex 12 deal template's `cashflow_items` and `triggers_tests_events`
  lists are both empty.

**Classification: EXPOSURE GAP.** What would be required is a represented option
set — call dates and call prices, or a note-level structure. Simulating
prepayment scenarios and labelling the minimum "YTW" would be a forecast wearing
a contractual name, and is exactly what the brief forbids.

---

## 10. Shared MI integration

One calculation path, and it is asserted rather than asserted-to.

- The calculation lives in `analytics_lib/contractual.py`, beside
  `analytics_lib/history.py`. There is no securitisation-specific cash-flow
  module and no agent-specific fast path.
- **One tool, not one per KPI.** `contractual_analytics` serves both metrics
  from a single schedule pass, so the two cannot disagree on the same loan and
  the book is not enumerated twice.
- The handler contains no arithmetic. It resolves the governed frame, applies
  filters, and translates.
- Enum normalisation **reuses the repository's own maps** from
  `annex2_delivery_rules.yaml` — the delivery path already maps "French" to
  FRXX. A second mapping in Python would have drifted the first time a synonym
  was added to only one of them.

**Methodology catalogue, extended:**

| Identifier | Metric |
|---|---|
| `CONTRACTUAL_SCHEDULE@v1` | contractual cash-flow enumeration |
| `CONTRACTUAL_WAL@v1` | contractual weighted average life |
| `CONTRACTUAL_YTM_PERIODIC@v1` | contractual periodic yield |

alongside the observed family (`OBSERVED_SMM@v2`, `OBSERVED_CPR@v2`,
`OBSERVED_LOSS@v2`, `OBSERVED_RECOVERY@v2`, `OBSERVED_LOSS_SEVERITY@v1`,
`OBSERVED_SERIES@v1`) and `CURRENT_LTV@v1`. The `CONTRACTUAL_` / `OBSERVED_`
prefix is the vocabulary boundary, visible in every result.

---

## 11. Readiness integration

Two metrics added: `COMP_CONTRACTUAL_WAL` and `COMP_CONTRACTUAL_YTM`, both
`fact_tool: contractual_analytics`, both `status: READY`. Framework coverage
moves from 46/48 to **48/50 (96.0%)**.

**Neither is cross-asset.** Both declare
`asset_class_applicability: [residential_mortgage, commercial_mortgage]`, so
`applies_to("equity_release")` returns `False` and the metrics are excluded from
an ERM assessment entirely — not reported as gaps on a book where they are not
economically defined. A test asserts this, including that `cross_asset` is not
present, since that alone would sweep equity release back in.

The agent guidance carries the three things a reader can get wrong: that this is
contractual rather than expected and should be read beside the observed CPR;
that the legal long-stop must never be substituted for a WAL; and that the yield
is periodic because CREL122 is Annex 3 only.

---

## 12. Explainability

From synthetic governed data — £1,000,000 FIXE, 5% fixed for life, four annual
instalments, £400,000 balloon, bought at 98. No LLM involved: every line is in
the payload.

**"Why is contractual WAL 2.801644 years?"**

```
amortisation_type          FIXE          rate_type              FXRL
schedule_method            CONTRACTUAL_SCHEDULE@v1
anchor_basis               counted_back_from_maturity
payment_frequency_months   12            as_of                  2026-01-31
final_payment_date         2030-01-31    time_basis             ACT/365F
principal_payments         4             total_principal        1,000,000.00
balloon                    400,000.00    balloon_treatment      paid at maturity per RREL41

cash flows   2027-01-31  principal 200,000  interest 50,000
             2028-01-31  principal 200,000  interest 40,000
             2029-01-31  principal 200,000  interest 30,000
             2030-01-31  principal 400,000  interest 20,000
```

plus the two notes: that dates were counted back from maturity because RREL41
requires the balloon to fall there, and that the constant principal was derived
as `(balance − balloon) ÷ periods` **because RREL35 = FIXE says it is constant**.

**"Why is contractual YTM 5.804661%?"**

```
price_basis           purchase_price (RREL34) = 98.0% of par; consideration
                      980,000.00 on an opening balance of 1,000,000.00
interest_rate_basis   current_interest_rate (RREL43), contractually fixed for
                      the life of the schedule
date_frequency_basis  periodic: 12 month(s) per period, 1 period per year,
                      annualised as (1+y)^periods_per_year − 1
day_count             NOT APPLIED — this is a periodic yield.
                      day_count_convention exists only as CREL122 (Annex 3)
                      and is not invented for tapes without it
irr_method            bisection on the discounted cash flows
periods               4        methodology_version  CONTRACTUAL_YTM_PERIODIC@v1
```

A 5% coupon bought at a discount yields 5.80%, which is the right direction and
the right order of magnitude.

---

## 13. Performance

Measured, and improved only after profiling.

| Workload | Before | After |
|---|---|---|
| Schedule enumeration, 10k loans (763k cash flows) | 5,081 ms | **1,396 ms** |
| Schedule enumeration, 100k loans (7.6m cash flows) | 49,600 ms | **11,551 ms (4.3×)** |
| Portfolio WAL, 100k loans | — | 1,010 ms |
| Periodic YTM, 5,333 eligible loans | — | 1,698 ms (0.32 ms/loan) |

**Two guesses avoided by profiling first.** The profile put **49%** of the run
in `_as_date` → `pd.to_datetime`, because calling it on a *scalar* re-runs the
format guesser every time — 3.0 of 10.7 seconds in
`_guess_datetime_format_for_array` alone. ISO strings now parse directly with
pandas retained as the fallback, so nothing stops being accepted. A further
**33%** was month arithmetic constructing two `date` objects per call to
subtract ordinals; that is now a table lookup.

**And the benchmark found a defect the tests did not.** The IRR solver divided
by zero at the bottom of its bracket: `(1 + −0.9999)^600` is `1e-4^600`, which
underflows to exactly `0.0`, and the power form then divides by it. The solver
now accumulates the discount factor forwards, where an underflowed factor
correctly zeroes the remaining terms. Thirty-eight passing methodology tests had
not reached it, because none of them was long enough.

Correctness took precedence over further optimisation: 100k loans producing 7.6m
contractual cash flows in 11.6 seconds is adequate for portfolio analysis, and
the remaining cost is genuinely per-cash-flow.

---

## 14. Regression

Baseline `3ddf7af`, candidate `7026281`, each run from a pinned, clean
worktree, verified by `rev-parse` before and after.

| | Baseline `3ddf7af` | Candidate `7026281` | Δ |
|---|---|---|---|
| passed | 5,207 | 5,245 | +38 |
| failed | 64 | **67** | **+3** |
| errors | 13 | 13 | 0 |
| skipped | 33 | 33 | 0 |

**The result was NOT neutral, and the three new failures were mine.**

```
FAILED only at candidate:
  tests/test_agent_openapi_document.py::test_the_document_is_not_stale
  tests/test_agent_openapi_document.py::test_every_registered_tool_has_a_route
  tests/test_agent_openapi_document.py::test_each_route_publishes_the_registrys_own_schema

FAILED only at baseline : none
ERROR  only at candidate: none
ERROR  only at baseline : none
```

**Cause.** `deploy/agent-api/trakt-agent-openapi.yaml` is *generated* from the
tool registry and checked in. Registering `contractual_analytics` without
running `scripts/build_agent_openapi.py` left it stale, so the tool existed on
the Python surface and not on the HTTP one — an agent reading the published
contract would not have found it.

**Why it was missed.** The sprint ran targeted test subsets before committing
and the full suite only afterwards, in the regression. The guard existed and
worked; it simply was not run at the point it would have helped. This is the
second process failure of this kind in the programme, after the stale-worktree
one in 2.5C, and both share a shape: a check that exists, is correct, and is
skipped because a faster subset felt sufficient.

**Fixed** in `0149e20` by regenerating the document (25 tools) and recording
the required step at the registration site in `trakt_tools/handlers/__init__.py`
— the file someone adding a tool is already editing. The three tests pass.

**Regression neutrality is therefore NOT claimed for `7026281`.** It is claimed
for the corrected tree, and re-verified in the Sprint 2.5E regression, whose
baseline includes this fix.

---

## 15. Remaining gaps

| Gap | Type | Detail |
|---|---|---|
| Day-count-exact YTM on Annex 2 | **FIELD GAP** | `day_count_convention` is CREL122, Annex 3 only. Periodic YTM is exposed instead, named as such |
| Floating and resetting cash flows | **ASSUMPTION REQUIRED** | the future index level. RREL48/49 bound it, RREL47/51 date it, nothing supplies it |
| ERM expected WAL | **MODEL REQUIRED** | mortality, morbidity and redemption behaviour. Not built |
| YTW | **EXPOSURE GAP** | no represented option set; no liability-side entity |
| `outstanding_balance_period_*` | **UNKNOWN** | undocumented; two prior reviews disagree on what they are. Not consumed |
| Loans with frequency `OTHR` | **FIELD GAP** | the payment interval itself is unstated |
| Stub-period placement | **IMPLEMENTATION** | stubs fall at the front by construction; a tape with an explicit first-payment date on Annex 2 would do better, and no such field exists |
| Scheduled-principal denominator for SMM | **IMPLEMENTATION, deliberately deferred** | see below |

**One deliberate non-change.** The deterministic schedule *would* now support a
more precise scheduled-principal denominator for SMM on supported amortisation
types. That is a methodology upgrade to a published observed metric and is
**not** being made silently in a contractual sprint. It is recorded here as a
candidate for separate review. Meanwhile a test asserts `history.py` never
references `payment_due`: RREL39 is principal **plus** interest, and using it as
scheduled principal would inflate the deduction and overstate every prepayment
rate Trakt publishes.

---

## 16. Sprint 3 readiness

> **Is the deterministic portfolio-intelligence foundation now sufficient to
> proceed to the Securitisation Readiness Agent?**

**Yes, with two named caveats rather than blockers.**

The agent's deterministic surface is now genuinely broad: observed prepayment,
arrears, defaults, losses, recoveries and severity; contractual life and yield;
LTV and valuation evidence; concentrations; regulatory field readiness. Every
one carries a versioned methodology identifier and a status, and — the part that
matters most for an autonomous consumer — **the refusals explain themselves**.
An agent asking for a WAL on an equity-release book gets `NOT_APPLICABLE` and
the reason, not a null it has to guess about and not a plausible number derived
from a legal long-stop.

**The two genuine blockers from 2.5C remain, and neither is a cash-flow
question:** default rate and cure rate are still methodology gaps where the data
is settled and the definitions are unowned. An agent quoting either today would
be relying on a convention nobody has chosen. They should be finished first, and
they are small.

**The caveat worth carrying into Sprint 3** is scope of applicability, not
correctness. Contractual WAL and YTM are real capabilities on amortising
residential and commercial books, and they compute nothing at all on Trakt's own
equity-release portfolio — correctly, but it means this sprint's headline
metrics will be absent from the very book most likely to be demonstrated. That
is a presentation problem to plan for, not a defect to fix.
