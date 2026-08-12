# Field availability → market-standard KPI methodology

*Sprint 2.5C, completed. Baseline `cdefc25` → this commit.*

The question this sprint was reopened to answer:

> **Given the core, canonical and mapped regulatory fields Trakt already holds,
> which market-standard credit and securitisation KPIs can Trakt calculate
> correctly and deterministically?**

The short answer is that Trakt's field universe is much stronger than its
metric universe, and the earlier review mistook the second for the first. Of
the thirty economic concepts a securitisation review quotes, **all thirty are
present as governed canonical fields**; twenty-seven carry ESMA codes, and
twenty-four of those on three or more annexes. Almost everything the previous
pass listed as a data gap was a calculation nobody had written.

Three things were wrong in the repository and are now fixed. Each was found by
reading a definition next to the code that claimed to implement it, and none
was caught by a test:

| | Was | Is |
|---|---|---|
| Loss severity | not implemented; recorded as needing collateral-proceeds data Trakt lacks | `allocated_losses` ÷ `default_amount`, both Mandatory on every annex |
| `net_loss` (mine, 2.5B) | allocated losses **less** cumulative recoveries | withdrawn — RREL73 is already stated after recoveries, so it deducted them twice |
| Annex 12 arrears buckets IVSS38–44 | a **balance** summed into a field ESMA types `PercentageRate`, with `min: 0` on the "1–29 days" band and `min: 50` on the "60–89" band | share of pool balance, bands per the schema, seven codes, no test previously |

---

## 1. What was searched, and in what order

The brief's first non-negotiable is *do not label something a field gap until
the complete canonical + regime field universe has been searched*. The search
order was:

1. `config/system/fields_registry.yaml` — **499 canonical fields**
   (295 core, 93 performance, 57 product, 54 collateral; 316 regulatory,
   183 analytics).
2. Their `regime_mapping` blocks — ESMA Annex 2 (107 fields), Annex 3 (238),
   Annex 4 (120), Annex 8 (84), Annex 9 (85).
3. `config/regime/annex2_field_universe.yaml` — the regulator's own field
   *definitions* and ND-fallback rules, which turned out to settle three
   methodology questions on their own.
4. `config/business_semantics_registry.yaml` — 242 of the 499 fields carry
   analytical metadata (concept, role, temporality, aggregation, weight).
5. `config/risk/concentration_test_library.yaml` — 42 MI metrics and the 50
   field *roles* they resolve through.
6. `config/regime/annex12_template.yaml` and the Annex 12 XSD.

Only after all six did anything get called a field gap.

**Two things worth stating about the regime coverage.** `BOE_Cashflow` is
declared as a priority regime for the `BOE_Securitisation` consumer and **no
field maps to it** — the consumer resolves to nothing. And there is **no FCA
field mapping anywhere in the repository**: "FCA" appears only in prose
describing UK ITL3 geography fields for MI drill-down. The regulatory-leverage
table in §3 therefore has no FCA column, because populating one would mean
inventing it.

---

## 2. Field → KPI capability map

`CORE` = canonical core field. `REG` = carries an ESMA code (annexes listed).
`DERIVED` = computed from canonical fields. `HIST` = needs two or more governed
snapshots.

| KPI | Required source fields | Available? | Where | Historical? | Existing calculation? | Can calculate now? |
|---|---|---|---|---|---|---|
| **Arrears share, N+ DPD** | `number_of_days_in_arrears`, `current_principal_balance` | Yes | REG A2/3/4/8/9 (RREL68, RREL30) | No | `perf_arrears_share` | **Yes — READY** |
| **Arrears stock (£)** | `arrears_balance` | Yes | REG (RREL67) | No | none | **READY, not exposed** |
| **Principal vs interest arrears** | `principal_arrears_amount`, `interest_arrears_amount` | Yes | Analytics (no code) | No | none | **READY, not exposed** |
| **Arrears roll / migration** | `days_in_arrears_prior` + `number_of_days_in_arrears` | Yes | Analytics + REG | **No — one snapshot** | `transition_analysis` (two snapshots) | **READY by a second route** |
| **Delinquency bucket distribution** | `number_of_days_in_arrears`, balance | Yes | REG | No | Annex 12 IVSS38–44 | **Yes — fixed this sprint** |
| **SMM / CPR** | unscheduled principal, scheduled principal, opening balance, exit evidence | **Partly** | `unscheduled_principal_collections` is **Annex 3 only** | Yes | `prepayment_rate` @v2 | **Yes for CRE; derived for RMBS** |
| **Default rate** | `default_date`, `default_amount`, `account_status` | Yes | REG (RREL71/72/69) | Either | none as a *rate* | **METHODOLOGY GAP** |
| **Cure rate** | `account_status`, `days_in_arrears_prior`, `date_last_in_arrears` | Yes | REG + analytics | Either | none | **METHODOLOGY GAP** |
| **Cumulative loss rate** | `allocated_losses`, balances | Yes | REG (RREL73) | Yes | `loss_and_recovery` | **Yes — READY** |
| **Loss severity** | `allocated_losses`, `default_amount` | Yes | REG (RREL73, RREL71) | No | **built this sprint** | **Yes — READY** |
| **Recovery rate** | `cumulative_recoveries`, `default_amount` | Yes | REG (RREL74, RREL71) | Yes | **corrected this sprint** | **Yes — READY** |
| **Liquidation decomposition** | `liquidation_expense`, `net_proceeds_received_on_liquidation` | CRE only | REG A3 (CREL138/139) | No | **built this sprint** | **Yes for CRE** |
| **Current LTV** | `current_loan_to_value` or balance ÷ valuation | Yes | REG (RREC12) + valuation observations | No | `ltv_weighted_average`, `ltv_above_share` | **Yes — READY** |
| **Original / indexed LTV** | `original_loan_to_value`, `indexed_loan_to_value` | Yes | REG (RREC16) / analytics | No | roles exist; no distinct metric | **READY, not exposed** |
| **Valuation staleness** | valuation date + selection policy | Yes | collateral layer | No | `valuation_age_profile` | **Yes — READY** |
| **WAC / WA margin** | `current_interest_rate`, `current_interest_rate_margin`, balance | Yes | REG (RREL43, RREL46) | No | `rate_gross_wac`, `rate_net_wac` | **Yes — READY** |
| **WA seasoning** | `origination_date` | Yes | REG (RREL23) | No | `composition_vintage_share` only | **READY, not exposed** |
| **WA remaining term** | `maturity_date` | Yes | REG (RREL24) | No | `maturity_within_horizon_share` only | **READY, not exposed** |
| **Concentrations (geo, borrower, top-N)** | geography, `borrower_identifier`, balance | Yes | analytics + REG | No | 17 library metrics (geography 6, borrower 6, balance 5) | **Yes — READY** |
| **Vintage / cohort performance** | `origination_date` + any measure | Yes | REG | Yes | `cohort_comparison` | **Yes, with a caveat (§7)** |
| **Contractual WAL — bullet / fixed amortisation** | `maturity_date`, RREL37 frequency, `regular_principal_instalment`, `balloon_amount`, RREL35 = BLLT or FIXE | Yes | REG ×5 + one analytics field | No | none | **READY, not exposed (§11)** |
| **Contractual WAL — French / German** | the above + per-period P&I split | **Partly** | RREL35 = FRXX/DEXX | No | none | **METHODOLOGY GAP (§11)** |
| **Contractual WAL — ERM (RREL35 = OTHR)** | — | — | — | — | — | **UNDEFINED — no contractual repayment date (§11)** |
| **Observed portfolio life / runoff** | governed snapshots | Yes | HIST | Yes | `portfolio_series` (partly) | **READY, not exposed** |
| **Expected WAL** | the above **+ a prepayment assumption** | No | — | — | none | **ASSUMPTION-MODEL REQUIRED** |
| **YTM — bullet / fixed amortisation** | `purchase_price` RREL34, `current_interest_rate`, balance, maturity, frequency | Yes | REG ×5 | No | none | **READY, not exposed (§12)** |
| **YTM — French / German** | the above + per-period P&I split | **Partly** | REG ×5 | No | none | **METHODOLOGY GAP (§12)** |
| **YTW** | the above **+ call/prepayment scenarios**, at note level | **No** | no tranche/note data model exists | — | none | **EXPOSURE GAP + ASSUMPTION-MODEL** |
| **PD / LGD** | `bank_internal_loss_given_default_lgd_estimate`, `loss_given_default` | Yes, **as supplied values** | analytics layer | No | none | **READY to report, never to model** |
| **Regulatory field readiness** | ND rules + `regime_mapping` | Yes | `annex2_field_universe.yaml` | No | `regulatory_readiness` | **Yes — READY** |

---

## 3. Regulatory-field leverage

The concepts a credit KPI needs, and where each already lives. "Used in MI"
means a shared calculation reads it today.

| Economic concept | Core field | ESMA field(s) | FCA | Canonical representation | Used in MI? |
|---|---|---|---|---|---|
| Current exposure | `current_principal_balance` | RREL30 / CREL23 / CRPL39 / LESL28 / ESTL28 | — | monetary, point_in_time | **Yes** |
| Original exposure | `original_principal_balance` | RREL29 / CREL24 / CRPL38 / LESL27 / ESTL27 | — | monetary, static_baseline | **Yes** |
| Days past due | `number_of_days_in_arrears` | RREL68 / CREL130 / CRPL78 / LESL56 / ESTL54 | — | integer, point_in_time | **Yes** |
| Arrears stock | `arrears_balance` | RREL67 / CREL129 / CRPL77 / LESL55 / ESTL53 | — | monetary, point_in_time | No |
| Prior-period DPD | `days_in_arrears_prior` | — (analytics) | — | integer, point_in_time | No |
| Account state | `account_status` | RREL69 / CREL136 / CRPL79 / LESL57 / ESTL55 | — | enum | **Yes** |
| Default event | `default_date` | RREL72 / CREL133 / CRPL82 / LESL60 / ESTL58 | — | date | **Yes** (exit classification) |
| Defaulted exposure | `default_amount` | RREL71 / CREL132 / CRPL81 / LESL59 / ESTL57 | — | monetary — *gross, before proceeds* | **Yes — new this sprint** |
| Realised loss | `allocated_losses` | RREL73 / CREL137 / CRPL83 / LESL61 / ESTL59 | — | monetary — *after sale proceeds* | **Yes** |
| Recoveries | `cumulative_recoveries` | RREL74 / CREL141 / CRPL84 / LESL62 / ESTL60 | — | monetary, cumulative | **Yes** |
| Liquidation proceeds | `net_proceeds_received_on_liquidation` | CREL138 | — | monetary | **Yes — new this sprint** |
| Liquidation costs | `liquidation_expense` | CREL139 | — | monetary | **Yes — new this sprint** |
| Collateral sale | `sale_price` | RREC21 / CRPC17 / ESTC19 | — | monetary | No |
| Unscheduled principal | `unscheduled_principal_collections` | **CREL98 only** | — | monetary, period_flow | **Yes** |
| Scheduled principal | `regular_principal_instalment` | — (analytics) | — | monetary, period_flow | **Yes (SMM denominator)** |
| Principal frequency | `scheduled_principal_payment_frequency` | RREL37 / CREL90 / CRPL48 / LESL33 / ESTL33 | — | enum (MNTH/QUTR/SEMI/YEAR) | No |
| Amortisation shape | `amortisation_type` | RREL35 / CREL87 / CRPL46 / LESL31 / ESTL31 | — | enum (French/German/…) | via `payment_option` |
| Balloon | `balloon_amount` | RREL41 / CRPL51 / ESTL37 | — | monetary | as a role only |
| Maturity | `maturity_date` | RREL24 / CREL18 / CRPL34 / LESL23 / ESTL25 | — | date | **Yes** |
| Origination | `origination_date` | RREL23 / CREL15 / CRPL33 / LESL22 / ESTL24 | — | date | **Yes** |
| Original term | `original_term` | RREL25 / CREL19 / LESL24 | — | integer months | via `extension_share` |
| Coupon | `current_interest_rate` | RREL43 / CREL110 / CRPL53 / LESL36 / ESTL39 | — | percent | **Yes** |
| Margin | `current_interest_rate_margin` | RREL46 / CREL113 / CRPL56 / LESL39 / ESTL42 | — | percent | **Yes** |
| Rate type | `interest_rate_type` | RREL42 / CREL109 / CRPL52 | — | enum | **Yes** |
| Purchase price | `purchase_price` | RREL34 / CREL28 / CRPL43 / LESL29 / ESTL30 | — | **percent of par** | No |
| Current valuation | `current_valuation_amount` | RREC13 / CREC15 / CRPC10 / LESL75 / ESTC10 | — | monetary + observation | **Yes** |
| Original valuation | `original_valuation_amount` | RREC17 / CRPC13 / LESL72 / ESTC14 | — | monetary | **Yes** |
| Current LTV | `current_loan_to_value` | RREC12 / CREL76 | — | percent | **Yes** |
| Supplied LGD | `bank_internal_loss_given_default_lgd_estimate` | — (analytics) | — | percent | No |
| Grace period | `principal_grace_period_end_date` | RREL36 / CREL88 / CRPL47 / LESL32 / ESTL32 | — | date | No |

**Twenty-four of the thirty carry ESMA codes on at least three annexes.** The
concepts MI does not read are not missing; they are unread.

---

## 4. Field gap vs KPI gap

| Metric | Classification | Why |
|---|---|---|
| Arrears share, DPD bands | **READY** | fields and calculation both present |
| Cumulative loss rate | **READY** | four denominators published, none chosen |
| **Loss severity** | **was METHODOLOGY GAP → now READY** | RREL71 and RREL73 were always there |
| **Recovery rate on defaulted exposure** | **was METHODOLOGY GAP → now READY** | previously divided by the wrong denominator |
| Arrears stock, principal-vs-interest split | **METHODOLOGY GAP** | fields present, no metric exposes them |
| Default rate | **METHODOLOGY GAP** | see §6 — the event definition is the work, not the data |
| Cure rate | **METHODOLOGY GAP** | needs an observation window, which is a choice |
| WA seasoning / remaining term / borrower age | **METHODOLOGY GAP** | the weighted-average evaluator already exists |
| SMM/CPR numerator on **RMBS** | **FIELD GAP, asset-class-scoped** | `unscheduled_principal_collections` is Annex 3 only |
| Contractual WAL, bullet / fixed | **READY, not exposed** | the cash flows are an enumeration over fields already held |
| Contractual WAL, French / German | **METHODOLOGY GAP** | Trakt receives an instalment amount, not a per-period principal series |
| Contractual WAL, ERM (`OTHR`) | **not applicable** | no contractual repayment date exists to weight |
| Expected WAL | **ASSUMPTION-MODEL REQUIRED** | out of scope by the brief |
| YTM, bullet / fixed | **READY, not exposed** | RREL34 supplies price relative to par; cash flows as above |
| YTM, French / German | **METHODOLOGY GAP** | same per-period decomposition as WAL |
| YTW | **EXPOSURE GAP** | Trakt holds no tranche or note-level data at all |
| PD / LGD modelling | **out of scope** | supplied estimates may be reported; none may be built |
| Submission acceptance state | **DATA GAP** | genuinely external evidence Trakt does not hold |

Exactly one true field gap was found, and it is scoped to an asset class.

---

## 5. Prepayment: SMM and CPR

**Denominator, corrected earlier in this sprint and verified externally:**

```
SMM = unscheduled principal ÷ (beginning balance − scheduled principal)
CPR = 1 − (1 − SMM)^12
```

Scheduled principal was never available to prepay, so it does not belong in the
denominator. Worked and asserted: £100m opening, £1m scheduled, £500k
unscheduled → 0.505051% SMM → 5.895058% CPR. Every period publishes
`denominator_basis`, and a tape with no scheduled-principal field says so
rather than quietly using the opening balance.

**The numerator is where the asset-class gap sits.**
`unscheduled_principal_collections` carries **CREL98 and nothing else** — it is
a commercial-real-estate field. Annex 2 has no equivalent. For a residential
book the numerator is assembled from redemption evidence
(`loan_redemption_flag`, `redemptions_received_in_period`) and balance
movement, which is weaker. That is a real field gap, and it is the only one in
this report.

**Exits still require evidence.** `loan_redemption_flag` → redemption;
`default_date` or a defaulted `account_status` → `default_exit`;
`maturity_date` on or before the close → `maturity`; anything else →
`UNKNOWN_EXIT`, excluded from the numerator and disclosed. Disappearance is
never read as redemption.

---

## 6. Arrears: stock, flow, rate, movement, migration

Trakt holds all five, and the earlier review only used two.

| Kind | Fields | Exposed today |
|---|---|---|
| **Stock (£)** | `arrears_balance` (RREL67), `principal_arrears_amount`, `interest_arrears_amount` | No |
| **Stock (days)** | `number_of_days_in_arrears` (RREL68) | Yes |
| **Rate (%)** | balance in band ÷ pool balance | Yes — `perf_arrears_share` |
| **Movement** | `days_in_arrears_prior`, `loan_entered_arrears`, `date_last_in_arrears` (RREL66) | No |
| **Migration** | prior vs current DPD | Only via two snapshots |

`days_in_arrears_prior` is the interesting one: it is *"prior-period days past
due; comparison basis for arrears deterioration"*. A roll rate normally needs
two periods. Trakt carries the prior state **on the row**, so a single snapshot
can produce one. It is deliberately **not** built here — `transition_analysis`
already computes roll rates from two snapshots, and a second implementation of
one metric is the exact defect this sprint spent its first half removing. It is
registered as latent MI in §10 with the condition attached: build it only as an
alternative *input* to the existing calculation, never as a second calculation.

**The boundary question is now settled by an authority, not by argument.** The
ESMA Annex 12 schema defines its bands as *"between 30 and 59 days
(inclusive)"*. Trakt's inclusive default — `min_days: 30` means 30 or more — is
the regulator's convention. The `dpd_boundary: exclusive` option remains, and
it now has a named justification too: CRR Article 178 defines default as *past
due **more than** 90 days*, so a rule pack expressing the regulatory default
definition should ask for it.

**Which also means arrears is not default.** "90+ DPD" as Trakt computes it has
no materiality threshold (CRR sets €100 retail / €500 other, and 1% of the
obligor's aggregate exposure) and no consecutive-day requirement. It is a
delinquency measure. Default is read from `default_date` and `account_status`,
and the two must not be presented as the same fact.

---

## 7. Loss, recovery and severity

**Severity is now implemented, and the regime wrote the formula.** Reading the
two field definitions side by side:

> **RREL71 Default Amount** — "Total **gross** default amount **before** the
> application of sale proceeds and recoveries."
>
> **RREL73 Allocated Losses** — "The allocated losses to date, net of fees,
> accrued interest etc. **after application of sale proceeds** ... as
> recoveries are collected and the work out process progresses."

The market definition of severity — net loss over unpaid balance at
liquidation — is exactly RREL73 over RREL71. Both are Mandatory on all five
annexes. Nothing needed to be added to the field model.

**And the same two sentences found a defect in code I wrote in Sprint 2.5B.**
`net_loss` was `allocated_losses − cumulative_recoveries`. RREL73 is *already*
after recoveries, so that subtraction removes the same money twice. On the test
book it reports 30% severity where the answer is 40%. `net_loss` and
`net_loss_rate_on_original_pct` are **withdrawn**; `recovery_rate_on_losses_pct`
is renamed `recoveries_against_residual_loss_pct` because that is what it
measures; and a genuine `recovery_rate_on_defaulted_pct` — recoveries over
RREL71 — has been added. `OBSERVED_LOSS@v1` → `@v2`, because output fields
changed meaning and a finding that cited v1 must keep meaning what it meant.

Where a CRE tape carries `liquidation_expense` and
`net_proceeds_received_on_liquidation`, severity is also built the long way —
balance at default + expenses − proceeds — and published as a **cross-check**
with an explicit `agrees_with_allocated_losses` flag. A disagreement is a data
finding about the tape, not a second severity.

**A pool with no defaults reports severity as unavailable, not 0%.** Zero
asserts that defaults happened and cost nothing.

**Still not audited:** default rate and cure rate. Both are methodology gaps
where the *data* question is settled and the *definition* question is not:
absorbing versus reversible default, count versus balance, period versus
cumulative, and what observation window makes a cure a cure. Neither should be
guessed.

---

## 8. Weighted metrics

Every weighted average in the library runs through one evaluator, and it does
declare all four things the brief asks for:

| | |
|---|---|
| **Value field** | resolved through a named role (`interest_rate`, `ltv_current`, …) |
| **Weight field** | the `weighting` parameter; default `current_balance` |
| **Eligible population** | rows where value **and** weight are both non-null |
| **Missing-data policy** | pairwise deletion, **and disclosed** — `denominator_value`, `denominator_basis`, `loans_in_numerator` and `total_loans` all published |

Verified balance-weighted: WA LTV, WA gross coupon, WA net coupon, WA property
valuation, WA balance. On a two-loan book the arithmetic mean is 50.0% and the
balance-weighted answer 82.0%, asserted.

WA seasoning, WA remaining term and WA borrower age are **not defective —
they do not exist**. The evaluator would serve them unchanged; only library
entries are missing. §10.

---

## 9. Concentration and portfolio characteristics

The library's 42 metrics break down as geography 6, borrower 6, property
value 5, loan balance 5, rate and product 5, performance 4, LTV 3, composition
3, maturity 2, and one each for residual value, external index and the generic
filtered-share primitive. Each declares `denominator_options` of
`current_balance` / `original_balance` / `loan_count` and publishes which was
used. **`denominator_floor` is contractual and is never inferred from data** —
a covenant with a floored denominator has to say so.

**Not audited, and stated as such:** missing-category treatment across the
`share_of_balance` family — whether a null region lands in the denominator only
or in neither. It is a real question and it was not answered here.

---

## 10. Latent MI register

Metrics Trakt can calculate today from governed fields and does not expose.

| Rank | Metric | Fields | Why it ranks there |
|---|---|---|---|
| **HIGH** | **Loss severity** | RREL73 ÷ RREL71 | first question asked of a defaulted book — **built this sprint** |
| **HIGH** | **Recovery rate on defaulted exposure** | RREL74 ÷ RREL71 | the previous ratio used a denominator already net of the numerator — **corrected this sprint** |
| **HIGH** | **Annex 12 delinquency bands** | RREL68 + balance | a required regulatory disclosure that was emitting a balance into a percentage field — **fixed this sprint** |
| **HIGH** | Arrears stock in £, principal vs interest | RREL67, `principal_arrears_amount`, `interest_arrears_amount` | a £ arrears figure is what a servicer reconciles against; only a % exists |
| **HIGH** | Single-snapshot arrears roll rate | `days_in_arrears_prior` | roll rates normally need two periods; the prior state is on the row. **Build as an input to `transition_analysis`, never as a second calculation** |
| **HIGH** | Annex 12 IVSS22 annualised CPR | the existing `prepayment_rate` | the regulator asks for CPR, the template's `method` is **empty**, and Trakt now has one. Needs two periods, which the single-period projector cannot supply — an architecture change, not a formula |
| **MEDIUM** | WA seasoning, WA remaining term, WA borrower age | RREL23, RREL24, borrower age roles | quoted in every pool summary; the evaluator already supports them |
| **MEDIUM** | Original / indexed LTV as distinct metrics | RREC16, `indexed_loan_to_value` | roles resolve; no library metric distinguishes them from current LTV |
| **MEDIUM** | Observed portfolio runoff / life to date | governed snapshots | the honest, assumption-free half of WAL (§11) |
| **MEDIUM** | Contractual WAL, bullet and fixed-amortisation | `maturity_date`, RREL37, `regular_principal_instalment`, RREL41 | an enumeration, not an engine — **but rates LOW on Trakt's own ERM book, where RREL35 is `OTHR` and contractual WAL is undefined** (§11) |
| **MEDIUM** | Loan-level YTM, same exposures | the above + `purchase_price` RREL34 | price relative to par is already Mandatory on all five annexes (§12) |
| **LOW** | `outstanding_balance_period_*` as a received schedule | eight analytics fields | shaped like a dated balance path; **no code, no semantics, no consumer — needs a specification decision before any use** (§11) |
| **MEDIUM** | Supplied LGD reporting | `bank_internal_loss_given_default_lgd_estimate` | reporting the originator's own estimate is not modelling; **Trakt still builds no PD/LGD** |
| **MEDIUM** | Payment-frequency and amortisation-shape mix | RREL37, RREL35 | eligibility screens ask for these directly |
| **LOW** | Grace-period exposure | RREL36 | narrow, but free |
| **LOW** | Collateral sale-price outcomes | RREC21 | overlaps the liquidation decomposition on CRE |
| **LOW** | Annex 12 IVSS24 gross charge-offs | — | `method` empty and `source_field` null; needs a definition decision first |

---

## 11. Weighted average life

An earlier draft of this section said WAL was blocked behind a "contractual
schedule builder". That was inferred from the absence of a WAL function rather
than established from the field universe, and re-checking the universe shows it
was wrong in **both** directions. Trakt can compute contractual WAL directly
today for part of the book, and no builder would ever produce one for the part
that matters most to Trakt.

### What Trakt actually receives

| Ingredient | Field | Coverage |
|---|---|---|
| Periodic principal amount | `regular_principal_instalment` | **analytics only — no regime code on any annex** |
| Periodic interest amount | `regular_interest_instalment` | analytics only |
| Next payment amount (P&I combined) | `payment_due` RREL39 | all five annexes — but a **scalar**, not a vector |
| Principal payment frequency | `scheduled_principal_payment_frequency` RREL37 | all five annexes |
| Interest payment frequency | `scheduled_interest_payment_frequency` RREL38 | all five annexes |
| Next payment **date** | `next_payment_date` CREL104 | **Annex 3 only** |
| Amortisation start date | `start_date_of_amortisation` CREL16 | **Annex 3 only** |
| Payments already made | `number_of_payments_before_securitisation` RREL58 | all five annexes |
| Maturity | `maturity_date` RREL24 | all five annexes |
| Terminal principal | `balloon_amount` RREL41 | A2/A4/A9 — **absent from Annex 3** |
| Amortisation shape | `amortisation_type` RREL35 | all five annexes |
| Current period scheduled P&I | `total_scheduled_principal_interest_due` CREL123 | **Annex 3 only** |

So the honest characterisation is: **Trakt receives a periodic instalment
amount and a frequency, not a dated schedule of principal.** One amount per
loan, not a vector by date. That distinction is what decides the question, and
it decides it differently for each amortisation type.

### Whether that is sufficient depends entirely on RREL35

The regime defines five values, and they do not behave alike:

| RREL35 | Definition | Principal per period | Contractual WAL from held fields? |
|---|---|---|---|
| **BLLT** | "full principal amount is repaid in the last instalment" | one cash flow at maturity | **Yes — directly, today.** WAL *is* the term to maturity |
| **FIXE** | "the principal amount repaid in each instalment is the same" | constant | **Yes — directly**, where `regular_principal_instalment` is populated |
| **FRXX** | "the total amount — principal plus interest — repaid in each instalment is the same" | **varies every period** | **No** — see below |
| **DEXX** | "the first instalment is interest-only and the remaining instalments are constant" | varies, plus a phase change | **No** |
| **OTHR** | — | undefined | **No — and no builder would help** |

**For BLLT and FIXE, no builder is required and none should be written.** The
cash flows are an enumeration: frequency from RREL37, count from the balance
divided by the instalment (or the term), terminal amount from RREL41. That is
arithmetic over fields already held, with no assumption anywhere in it.
`Σ(Pᵢ·tᵢ)/ΣPᵢ` follows immediately. **This is a real capability Trakt has and
does not expose**, and the previous draft wrongly listed it as blocked.

**For FRXX and DEXX it genuinely is insufficient, and the reason is specific.**
Under French amortisation the constant quantity is the *total* instalment; the
principal portion rises every period as the interest portion falls. A single
`regular_principal_instalment` scalar cannot represent a series that changes
each period. Recovering it means iterating

```
principal_t = instalment − rate_t × balance_(t−1)
balance_t   = balance_(t−1) − principal_t
```

which is an amortisation computation, not a lookup — and it needs the rate to
be projected, which for a variable-rate loan (`interest_rate_type` RREL42) is
an assumption rather than a fact. **That** is the blocker, and it is narrower
than "a schedule builder": it is the principal/interest decomposition for two
of five amortisation types.

**Two timing caveats, both real and neither fatal.** Annex 2 carries **no
payment-date field** — `next_payment_date` and `start_date_of_amortisation` are
Annex 3 codes. For a residential loan the payment dates must be anchored from
`maturity_date` counted back by RREL37, or from `origination_date` plus RREL58.
Both are deterministic where the frequency is regular, and RREL37 admits
`OTHR`, where neither is. And `regular_principal_instalment` carries no regime
code at all, so on a purely regulatory tape it may simply be absent — in which
case even FIXE needs the instalment derived from balance and term.

### The part that no builder fixes

Trakt's primary book is equity release, and the repository already reasons this
out in `config/asset/product_defaults_ERM.yaml`: a lifetime mortgage "rolls up
interest and repays at death/sale ... it is **NOT** a scheduled bullet
amortisation under the Annex 2 definition", and Trakt therefore reports it as
`OTHR`.

That is the substantive answer for the book Trakt actually holds. There are no
contractual principal payments before an event — death, sale, entry to care —
which is not contractually dated. **A contractual WAL for an ERM loan is not
blocked by missing data or missing code; it is undefined.** Any ERM WAL is an
*expected* WAL and needs mortality and voluntary-redemption assumptions, which
is forecasting and is out of scope by the brief.

### One undocumented family worth a decision

Eight canonical fields — `outstanding_balance_period_1`,
`outstanding_balance_period_2_120`, `outstanding_balance_period_121_599`,
`outstanding_balance_period_600` and a `_date` companion for each — have the
shape of a **received, dated projected balance path**. If a tape populates
them, they are the closest thing to an amortisation schedule Trakt is given,
and a WAL could be read off them directly.

They also have **no regime code, no business-semantics entry, no alias, and no
consumer anywhere in the repository**. They are declared and inert. Nothing
should be built on them until someone states what they mean; that is a question
for whoever specified them, not something to infer from four field names.

### Summary

| Question | Answer |
|---|---|
| Contractual WAL, bullet and fixed-amortisation exposures | **Computable now. Latent MI, not a gap** |
| Contractual WAL, French / German amortisation | Needs a principal-interest decomposition per period, and a rate projection if variable-rate |
| Contractual WAL, ERM (`OTHR`) | **Undefined** — no contractual repayment date exists |
| Observed portfolio life / runoff | Computable now from governed snapshots, no assumptions |
| Expected WAL | Assumption model. Out of scope |

---

## 12. Yield to maturity and yield to worst

**YTM inherits §11 exactly, and the price term is already there.**
`purchase_price` (RREL34) is Mandatory on all five annexes and is defined as
*"the price, relative to par, at which the underlying exposure was purchased by
the SSPE. Enter 100 if no discounting was applied"* — precisely the price basis
an IRR needs, and on a percentage-of-par scale that needs no consideration
figure.

So the same split applies, for the same reason:

- **Bullet and fixed-amortisation exposures: YTM is computable today.** Price
  from RREL34, cash flows by enumeration, coupon from `current_interest_rate`
  (RREL43). For a bullet loan it is a single-cash-flow discount.
- **French and German: blocked on the same per-period decomposition**, not on
  price and not on data.
- **ERM (`OTHR`): the cash-flow *dates* are behavioural**, so a contractual YTM
  is undefined for the same reason its WAL is.

One caveat worth stating before anyone quotes it: on a warehouse book bought at
par, RREL34 is 100 for every loan and YTM collapses to the coupon. That is a
correct answer that will look like a broken calculation.

**YTW remains an exposure gap, and for a different reason entirely.** Yield to
worst is the minimum yield across call and prepayment scenarios, and where it
is normally quoted it is a **note-level** measure. Trakt does not model the
liability side: no canonical field carries a tranche or class balance, coupon,
attachment point or paydown. The nine fields whose names mention notes,
seniority or subordination — `noteholder_consent`, `seniority`,
`principal_payment_allocation_to_senior_loan`,
`restrictions_on_sale_of_subordinated_loan` and the rest — are all attributes
of the **underlying exposure**, not of a tranche. The Annex 12 deal template's
`cashflow_items` and `triggers_tests_events` lists are both empty. YTW is
blocked by an entity Trakt does not model, and no field or formula fixes that.

**None of these should be built on synthetic assumptions.** A YTM from an
invented price, or a YTW from invented call dates, is worse than not having one.

---

## 13. Not duplicating regulatory calculations

Two places already hold regulator-defined calculations, and the audit found
Trakt drifting toward re-implementing them.

**Annex 12 arrears bands.** The template computes IVSS38–IVSS44 itself. Three
defects, all fixed, all verified against the schema's own wording:

| Code | Was | Consequence | Now |
|---|---|---|---|
| IVSS38–44 | `BUCKET_SUM` of `current_outstanding_balance` | an amount into a `PercentageRate` element, in fields the constraint file marks as admitting **no** ND fallback | `BUCKET_SHARE` — band balance ÷ pool balance × 100 |
| IVSS38 | `min: 0` on a band named "1–29 days" | the entire performing book reported as in arrears — on the test pool, 70.0% instead of 10.0% | `min: 1` |
| IVSS40 | `min: 50` on a band named "60–89 days" | 50–59 days counted in IVSS39 **and** IVSS40; the bands stopped partitioning the book | `min: 60` |

The denominator is the regulator's, quoted rather than chosen: *"relative to the
total outstanding principal amount of **all** exposures as at the data cut-off
date"* — the whole pool, not the arrears book. Seven codes had **no test of any
kind**; there are now eight.

**Annex 12 CPR.** `IVSS22_annualised_constant_prepayment_rate` has an empty
`method`, so the projector skips it and the field falls to a No-Data value. The
correct response is *not* a second CPR in the projector — it is to feed the
existing `prepayment_rate` into it. That needs the projector to see two
periods, which today it cannot. Recorded in §10 as an architecture item.

---

## 14. Market-methodology verification

The previous pass consulted no external sources and said so. This one did. Each
source below was read, not cited from memory.

| Source | Type | What it settled |
|---|---|---|
| **ESMA Annex 12 XSD** (`DRAFT1auth.098.001.04_1.3.0.xsd`, `ArrearsData2`) — in repository | Regulator | Delinquency bands are **percentages**, the denominator is **all exposures**, and the bands are **inclusive at both ends**. Settled §6 and §13 outright |
| **ESMA Annex 2 field universe** (`annex2_field_universe.yaml`, RREL71/73/74/34) — in repository | Regulator | Default amount is gross **before** proceeds; allocated losses are **after** them; purchase price is **relative to par**. Settled §7 and §12 |
| **EBA / CRR Article 178** and the EBA definition-of-default guidelines | Regulator | Default is *more than* 90 days past due, on a **material** obligation (€100 retail / €500 other, and 1% of aggregate exposure), counted over **consecutive** days. Settled that arrears ≠ default (§6) |
| Structured-finance WAL convention (SIFMA-style `Σ(Pᵢ·tᵢ)/ΣPᵢ`; industry treatments) | Convention | Confirmed WAL needs a principal schedule, and that with prepayment permitted it is necessarily an **estimate**. Settled §11 |
| RMBS loss-severity literature (Urban Institute; Philadelphia Fed working paper; Moody's *Measuring Loss Severity Rates of Defaulted RMBS*) | Industry / rating agency | Severity = net loss ÷ **UPB at liquidation**, net loss = balance + expenses − proceeds − recoveries. Settled §7 |
| SMM/CPR convention (prior pass, `CPR = 1 − (1 − SMM)¹²`, independently worked example 0.5% → 5.84%) | Convention | Settled §5 |

**Where sources disagree, both are kept and named.** The DPD boundary is the
clearest case: investor reporting says inclusive, CRR Article 178 says
exclusive. Trakt defaults to inclusive and offers the other on request, and
neither is presented as the only convention.

---

## 15. Metric methodology catalogue

The catalogue exists in configuration rather than prose, which is the right
place for it: `config/risk/concentration_test_library.yaml` already carries
`numerator`, `denominator_options`, `required_roles`, `optional_roles`, `unit`,
`aggregation`, `output_precision`, `implementation_status` and `version` for
each of its **42 metrics** — 39 implemented, 2 declared-not-implemented,
1 interface-only. The history metrics carry versioned methodology identifiers:

| Identifier | Metric | Changed this sprint |
|---|---|---|
| `OBSERVED_SMM@v2` | single monthly mortality | v1 → v2: denominator now nets scheduled principal |
| `OBSERVED_CPR@v2` | annualised prepayment | follows SMM |
| `OBSERVED_LOSS@v2` | loss and recovery | v1 → v2: double-counted netting withdrawn |
| `OBSERVED_RECOVERY@v2` | recovery rates | v1 → v2: denominator corrected to defaulted exposure |
| `OBSERVED_LOSS_SEVERITY@v1` | loss severity | new |
| `OBSERVED_SERIES@v1` | period series | unchanged |
| `CURRENT_LTV@v1` | LTV from valuation observations | unchanged |

**Still uneven, and stated plainly:** the two families express methodology
differently — the library through YAML metadata, history through identifiers in
code — and `time_basis` / `annualisation` / `exclusions` are not yet uniform
across either. Unifying them is worth doing and was not done here.

---

## 16. Readiness framework integration

The framework does consume shared MI calculations — every `fact_tool` names a
registered tool and every `calculation_source` names a real module — but it had
**drifted again, in the direction the existing guard could not see**.

`PERF_PREPAYMENT` and `PERF_LOSSES` were still published as `SMALL_GAP` with
`fact_tool: null` and guidance reading *"Report as unavailable"*, a full sprint
after `prepayment_analysis` and `loss_analysis` were built, registered and
tested. The Sprint 2.5 guard only fires when a metric **names** a tool, so a
null `fact_tool` slipped past it. An agent reading the framework would have
declined to measure prepayment on a book where Trakt could measure it.

Both now name their tool, their calculation source and `prior_snapshot` as
required evidence. Published coverage moves from 44/48 (91.7%) to **46/48
(95.8%)**. The two remaining non-READY metrics are honest:
`REG_SUBMISSION_STATE` is external evidence Trakt does not hold, and
`TREND_DATA_QUALITY_DRIFT` is explicitly a judgement.

The structural fix closes the direction that leaked. A framework metric linked
by `metric_id` to an unimplemented library metric must now either **be** a gap
or declare `supersedes_library_metric: true` and name a registered tool — both
assertions tested. That is the exact shape of this defect: prepayment can never
have a single-snapshot library evaluator, because a rate needs two periods, so
the framework and the library will always disagree here and now have to say why.

---

## 17. What was implemented, and what was deliberately not

**Implemented** — all five conditions met (fields exist, methodology clear,
metric high-value, change belongs in shared analytics, implementation
contained):

- loss severity, in `analytics_lib.history` — one function, published through
  the existing `loss_analysis` tool;
- the recovery-rate correction and the withdrawal of the double-counting net
  loss;
- the Annex 12 arrears bucket fixes — bounds and share basis;
- the readiness framework status correction and its structural guard.

**Deliberately not implemented, with the reason:**

- **WAL and YTM for bullet and fixed-amortisation exposures** — computable
  today from fields already held (§11), and left unbuilt only because Trakt's
  primary book is equity release, where `amortisation_type` is `OTHR` and a
  contractual WAL does not exist to be computed. High value on a CRE or
  amortising residential book; near-zero on this one. Registered as latent MI
  rather than built blind.
- **WAL and YTM for French and German amortisation** — need a per-period
  principal/interest decomposition, because Trakt receives an instalment
  amount rather than a principal series. A calculation, not a field gap (§11).
- **YTW** — needs a note-level entity Trakt does not model.
- **Expected WAL, any PD/LGD** — assumption models. Out of scope by the brief,
  and supplied LGD values may be reported but never authored.
- **Single-snapshot roll rate** — buildable today, and *not* built, because
  `transition_analysis` already computes roll rates. A second implementation of
  one metric is the defect this sprint exists to remove.
- **Annex 12 CPR wiring** — needs the projector to see two periods.
- **Default rate, cure rate** — the data is settled, the definitions are not,
  and guessing them would put an unowned convention into production.

---

## 18. Regression

Run under the discipline the brief requires: both trees committed and
immutable, neither edited while its run executed, baseline from a worktree
pinned at `cdefc25` and candidate from a worktree pinned at the delivered
commit.

*The candidate run against the final tree is in flight; the comparison — totals,
complete failure IDs, complete error IDs, and differences in both directions —
is recorded below when it completes. **No claim of regression neutrality is made
until then.***

One earlier attempt is worth recording as a process finding: a `git worktree
add -f` against an existing directory **fails silently**, and a run launched
that way tested a stale revision while appearing to test HEAD. It was caught by
checking `rev-parse` on the worktree rather than trusting the command that
created it. Verifying the tree is part of the regression, not preparation for it.

---

## Sprint 3 readiness

> **Are the shared MI metrics now methodologically trustworthy enough for an
> autonomous agent to use in a production-style securitisation review?**

**For prepayment, arrears, loss, recovery, severity, LTV and concentration —
yes.** Those are the metrics such a review leans on hardest, each now has one
definition, and the definitions have been checked against the regulator's own
words rather than against each other.

**For default rate and cure rate — no, and they should be finished first.** Not
because the data is missing, but because the definitions are unowned, and this
sprint's base rate is unkind: of the metrics examined closely, prepayment,
arrears, severity, net loss and three Annex 12 bucket definitions were all
wrong. Assuming the unexamined ones are fine is not supported by the evidence.

**A 2.5D cash-flow sprint is justified, but it is smaller than the last draft
of this document claimed.** Re-checking the field universe rather than
inferring from the absence of a function changed the shape of it: WAL and YTM
for bullet and fixed-amortisation exposures need no new engine at all — the
cash flows are an enumeration over `maturity_date`, RREL37 frequency,
`regular_principal_instalment` and `balloon_amount`. What genuinely needs
building is narrower: the per-period principal/interest decomposition for the
French and German amortisation types, where Trakt receives an instalment
amount rather than a principal series.

Two things should be settled before that sprint rather than during it. The
`outstanding_balance_period_*` family — eight fields with the shape of a
received, dated balance path and no documentation, no regime code and no
consumer — may already supply the schedule, and nobody should build one until
that is answered. And on Trakt's own book the whole question changes: equity
release reports `OTHR`, repays on death or sale, and has no contractual
repayment date, so contractual WAL is undefined rather than unbuilt. The
cash-flow sprint is worth doing for CRE and amortising residential; it is not
what unlocks ERM.

The Annex 12 CPR wiring is a separate, smaller item — the calculation exists,
and only the projector's single-period input is in the way. YTW is separate
again, and is a note-level data question rather than a methodology one.

**The observation from the last review still holds, and got stronger.** Every
defect in this sprint was found by reading a definition next to the code that
claimed to implement it. Not one was found by running anything — every test
passed throughout, including the tests I had just written. The most productive
hour of this sprint was spent reading `annex2_field_universe.yaml`, a file the
repository already had.
