# MI metric methodology review

*Sprint 2.5C. Baseline `cdefc25` → `7d3c05f`.*

> **Scope warning.** This sprint specifies 25 work parts and a 20-section
> report. **It is partially delivered.** The audit found and fixed two
> label/calculation defects and one performance defect, and built the test
> apparatus for methodology checking. Parts 3 and 7–16 — external market-source
> verification and the systematic audit of default, cure, loss, recovery, LTV,
> concentration, vintage, completeness and period-change methodologies — are
> **not done**. §9 lists exactly what remains. Nothing below claims coverage it
> does not have.

---

## 1. Executive summary

> **Can we now trust the economically meaningful metrics in the MI registry to
> mean what a market professional would reasonably expect?**

**NO — not yet, across the board.** Two metrics are now demonstrably correct
where they previously were not, and the audit apparatus exists. But the majority
of the metric universe has **not** been audited against external sources, so a
blanket assurance would be exactly the kind of unearned confidence this sprint
exists to remove.

What changed:

| | Before | After |
|---|---|---|
| "Prepayment Speed (CPR)" | a **sum of £ amounts** labelled as an annualised rate | renamed to what it shows; genuine SMM/CPR exists separately |
| "30+ DPD" | **two definitions** — 25% and 50% on the same book | one definition, inclusive boundary, structurally shared |
| Redemption | any disappearance counted as prepayment | classified; unexplained exits excluded |
| `prepayment_rate` at 100k × 12 | 9,501 ms | **681 ms** |

The finding that most deserves attention is the second, and not because of its
size: **I introduced half of it in Sprint 2.5B.** A metric can be correct in
each of two places and still be wrong as a system, and no test caught it because
each side passed its own.

---

## 2. Metric universe reviewed

Repository-derived. **Audited in this sprint:**

| Metric | Location | Verdict |
|---|---|---|
| SMM / CPR | `analytics_lib.history.prepayment_rate` | **Corrected** (§3) |
| "Prepayment Speed (CPR)" chart | `config/asset/static_pools_config_erm.yaml` | **WRONG → renamed** |
| Arrears share / 30-60-90 DPD | `concentration_tests.metrics._eval_arrears_share` | **Corrected** (§4) |
| Arrears measures | `trakt_tools.handlers.history._measure_functions` | **Corrected** — now delegates |
| Exit classification | `analytics_lib.history.classify_exits` | **New** (§5) |
| WA LTV weighting | `_eval_weighted_average`, history `wa_ltv` | Verified balance-weighted |
| Loss denominators | `analytics_lib.history.loss_and_recovery` | Verified — all four reported |

**Identified but NOT audited:** default rate, cure rate, roll/transition rates,
loss severity, recovery-rate denominators, original-vs-current LTV, high-LTV
exposure basis, concentration denominators, WA coupon/rate/age/seasoning/term,
vintage comparability, completeness definitions, period-change stock-vs-flow.

---

## 3. CPR / SMM

**Previous state.** No SMM or CPR calculation existed anywhere in the
repository. A chart titled *"Prepayment Speed (CPR)"* was specified as
`metric: prepayment_amount, agg: sum` — cumulative redemptions in pounds,
plotted, with an annualised-rate label.

**Corrected methodology** (`OBSERVED_SMM@v1` / `OBSERVED_CPR@v1`):

```
SMM  = (qualifying unscheduled principal + evidenced redemptions)
       / opening principal balance of the period

CPR  = 1 − (1 − mean monthly SMM)^12
```

**Numerator** — `unscheduled_principal_collections` plus full redemptions with
qualifying evidence. **Denominator** — the opening balance of the period.
**Excluded by name** — scheduled amortisation, contractual maturity,
default-related reduction, unexplained exits.

**Synthetic proof**, hand-worked and asserted:

```
opening exposed balance   £100,000,000
qualifying prepayment          £500,000
SMM                               0.5%
CPR = 1 − (1 − 0.005)^12        5.8377%
```

And the negative case: a book falling £1m purely through **scheduled** principal
reports **SMM 0.0000%**.

**The React chart has been renamed, not converted.** It is now *"Cumulative
Redemptions by Cohort"*. Genuine CPR is `prepayment_analysis`. A test fails if
any chart summing an amount carries a rate-implying label.

**Known limitation:** the denominator is the full opening balance, not opening
balance net of scheduled principal. Both conventions exist; Trakt's choice is
stated but **not yet verified against an external source** — see §9.

---

## 4. Arrears

**The defect.** `_eval_arrears_share` masked on `> min_days`; the Sprint 2.5B
history measure masked on `>= minimum`. Same nominal metric, same book:

```
loans at 0, 29, 30, 31 days — equal balances

concentration library  "30+ DPD"  →  25.0%   (only the loan at 31)
history measure        "30+ DPD"  →  50.0%   (the loans at 30 and 31)
```

**Fixed structurally.** The history measure now *calls* the library metric.
Agreement by hand is what drifts; delegation cannot.

**The boundary is now explicit and inclusive by default** — `min_days: 30` means
30 or more, matching the "N+ DPD" label every consumer uses. `dpd_boundary:
exclusive` remains available for a contract that genuinely says "more than N
days", but must be requested.

**Direction of change is conservative:** inclusive can only report *more*
arrears, so it cannot hide a breach. No operator-approved test config pins the
parameter — only Trakt's own rule packs do.

**Verified properties:** balance-weighted (a £900k delinquent loan gives 90%,
not the 50% a count would give); bands are **cumulative, not mutually
exclusive** (30+ ⊃ 60+ ⊃ 90+), asserted so a reader cannot double-count.

---

## 5. Redemption identification

**Previous state (Sprint 2.5B).** Any loan present at the open and absent at the
close was counted as a voluntary redemption.

**Why that is wrong.** A loan leaves a tape because it redeemed, defaulted and
was written off, matured, was sold or transferred — or because the extract
broke. Counting all of them as prepayment inflates the rate *most* in the
situation where it matters: a book shedding defaulted loans.

**Corrected.** `classify_exits` requires qualifying evidence, all from fields
that already exist in the canonical model:

| Evidence | Classification | In the prepayment numerator? |
|---|---|---|
| `loan_redemption_flag` set | `redemption` | **Yes** |
| `default_date`, or a defaulted `account_status` | `default_exit` | No |
| `maturity_date` on or before the close | `maturity` | No |
| none of the above | **`UNKNOWN_EXIT`** | **No** |

A defaulted exit is never reclassified as a redemption however the flag was
left. `UNKNOWN_EXIT` balance is reported and excluded — an unexplained
disappearance is a data-quality finding, not a prepayment.

**Demonstrated:** five loans exit with £1.5m opening balance; only the £100k
carrying redemption evidence enters the numerator, and the £400k unexplained is
disclosed in the notes.

---

## 6. Weighted averages (partial)

**Verified:** WA LTV is balance-weighted in both the library evaluator
(`weighting` defaults to `current_balance`) and the history measure. Asserted on
a two-loan book where the arithmetic mean is 50.0% and the balance-weighted
answer is 82.0%.

**Not verified:** WA coupon, WA interest rate, WA borrower age, WA seasoning, WA
remaining term, WA property value. Each may legitimately want a different
weight, and none was checked.

---

## 7. Loss denominators (verified, not extended)

`loss_and_recovery` reports four denominators and chooses none:

| Rate | Synthetic result |
|---|---|
| cumulative loss / original balance | 0.428% |
| cumulative loss / opening balance | 0.380% |
| cumulative loss / current balance | 0.453% |
| recoveries / losses | 40.0% |

Three answers, one portfolio, 19% apart. **Loss severity** (realised loss over
defaulted exposure) is **not implemented** — §9.

---

## 8. Performance

The 100k × 12-period benchmark omitted in Sprint 2.5B was run, and it found a
real bottleneck.

| Workload | Before | After |
|---|---|---|
| `prepayment_rate` 100k × 12 | 9,501 ms | **681 ms (14×)** |
| `portfolio_series` 12p × 7 measures, 1.2m rows | — | 860 ms |
| `loss_and_recovery` 12 periods | — | 4 ms |

**Root cause, found by profiling after two wrong guesses** (which produced 1.1×
and 1.2×): pandas' Arrow-backed string `isin` falls back to a **Python list
comprehension** over 1.1m elements — 9.0 of 9.05 seconds. Converting to object
dtype routes to the numpy hashtable.

Optimised in the shared implementation. No agent-specific fast path.

---

## 9. Remaining exceptions

**Not audited at all** — the systematic methodology review of:

- **default rate** — event definition, population, absorbing vs reversible,
  period vs cumulative, balance vs count;
- **cure rate** — eligible states, what constitutes a cure, observation window;
- **roll / transition rates** — count vs balance weighting, treatment of exits;
- **loss severity** — not implemented;
- **recovery-rate denominators** — defaulted balance vs charged-off vs gross
  loss vs balance at default; currently only recoveries/losses exists;
- **LTV** — original vs current distinction, missing/zero valuation handling,
  whether "% > 80% LTV" is of loans or of balance in every consumer;
- **concentration** — balance vs count denominators, missing-category treatment;
- **vintage comparability** — calendar-time vs months-on-book alignment, which
  materially affects any "the 2025 vintage is underperforming" claim;
- **completeness** — canonical completeness vs ESMA mandatory-field readiness
  are currently distinct tools but their percentages are not formally defined;
- **period change** — percentage points vs percent change is asserted in one
  test but not audited across consumers.

**Not done — external verification (Part 3).** No authoritative external source
was consulted. Every methodology decision here rests on repository evidence,
the canonical field semantics, and stated reasoning. The market-source register
the brief requires is therefore **empty, and deliberately so** — citing sources
I did not review would be worse than reporting none.

**Not done — the formal catalogue (Part 2).** Registry metadata for
`numerator` / `denominator` / `weighting` / `time_basis` / `annualisation` /
`exclusions` / `methodology_version` per metric was not added. Methodology
identifiers exist (`OBSERVED_SMM@v1`, `OBSERVED_CPR@v1`, `OBSERVED_LOSS@v1`,
`CURRENT_LTV@v1`) but are not yet uniform across the universe.

**Ambiguous, needing a deliberate decision:**

- SMM denominator — full opening balance vs opening balance net of scheduled
  principal. Both conventions exist in the market.
- DPD bucket convention — 1-29/30-59/60-89/90+ vs 1-30/31-60/61-90/91+. Trakt
  now uses inclusive (30 means 30+), which is stated but not externally
  validated.

---

## 10. Sprint 3 recommendation

> **Are the shared MI metrics now sufficiently methodologically trustworthy for
> an autonomous agent to use in a production-style securitisation review?**

**Not yet — but the blocker is narrow and specific.**

The metrics an agent would lean on hardest are in materially better shape:
prepayment is real rather than a mislabelled sum, arrears has one definition
instead of two, redemption requires evidence, and LTV weighting is verified. The
audit apparatus exists and catches this defect class.

**The genuine blocker is that most of the universe has not been audited.** An
agent presenting a default rate or a recovery rate to a counterparty today would
be relying on a definition nobody has checked — and this sprint found that two
of the first three metrics examined were wrong. That base rate is the argument
against assuming the rest are fine.

**Recommended before the agent build:** complete Parts 7–16 for default, cure,
recovery denominators, LTV distribution basis and vintage comparability — the
five that a securitisation review quotes most. Parts 2, 3 and the remaining
weighted averages can follow, because they improve documentation and confidence
rather than correctness.

**One observation worth carrying forward.** Both defects were found by reading
the calculation next to its label, not by running anything. Every test passed
throughout. A metric audit is not a testing activity, and scheduling it as one
is how these survive.
