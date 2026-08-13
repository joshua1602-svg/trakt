# Field availability → market-standard KPI methodology

*Sprint 2.5C, completed. Baseline `cdefc25` → this commit.*

> **Superseded in part.** Sprint 2.5D built the contractual analytics this
> report classified as latent, and the §11–§12 matrix was refined again in
> doing so — `payment_due` (RREL39) makes fixed-rate FRXX and DEXX
> deterministic, which this report had reached but not implemented. See
> `docs/contractual_cashflow_analytics.md`.

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
| **Contractual WAL — BLLT / FIXE, any rate type** | RREL30, RREL24, RREL37, RREL41 | Yes | REG ×5 (RREL41 ×3) | No | none | **LATENT DETERMINISTIC MI (§11)** |
| **Contractual WAL — FRXX / DEXX, fixed rate** | + RREL39 instalment, RREL43 rate, RREL42 = FXRL | Yes | REG ×5 | No | none | **LATENT DETERMINISTIC MI (§11)** |
| **Contractual WAL — FRXX / DEXX, floating or resetting** | + the future index level | **No** | RREL44–49 describe the structure, not the level | No | none | **ASSUMPTION REQUIRED (§11)** |
| **Contractual WAL — ERM (RREL35 = OTHR)** | a repayment date that does not contractually exist | **No** | — | — | — | **MODEL REQUIRED (§11)** |
| **Observed portfolio life / runoff** | governed snapshots | Yes | HIST | Yes | `portfolio_series` (partly) | **READY, not exposed** |
| **Expected WAL** | the above **+ a prepayment assumption** | No | — | — | none | **ASSUMPTION-MODEL REQUIRED** |
| **Contractual YTM — fixed rate (RREL42 = FXRL), any amortisation type** | RREL34 price, RREL43 rate, RREL38 frequency, RREL6 as-of, + the §11 principal series | Yes | REG ×5 | No | none | **LATENT DETERMINISTIC MI (§12)** |
| **Contractual YTM — floating or resetting** | + the future index level | **No** | — | No | none | **ASSUMPTION REQUIRED (§12)** |
| **Day-count-exact YTM on Annex 2** | `day_count_convention` | **No** | CREL122 — **Annex 3 only** | No | none | **FIELD GAP (§12)** |
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

Five classifications, used consistently from here on. The distinction that
matters most is between the second and the last two: *"Trakt doesn't have WAL"*
and *"Trakt holds everything contractual WAL needs for this amortisation type
and has not exposed the arithmetic"* are materially different statements, and
the earlier drafts of this report ran them together.

| Classification | Meaning |
|---|---|
| **READY** | the calculation exists and its inputs are sufficient |
| **LATENT DETERMINISTIC MI** | every economically necessary input exists, the result is deterministic, and only the enumeration is unwritten |
| **FIELD GAP** | a genuinely necessary contractual input is unavailable |
| **ASSUMPTION REQUIRED** | the result depends on an unknown future variable — a floating reference rate |
| **MODEL REQUIRED** | the meaningful result depends on behavioural assumptions — ERM mortality and redemption |


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
| Contractual WAL, BLLT / FIXE, any rate type | **LATENT DETERMINISTIC MI** | the principal series is rate-independent and contractually fixed |
| Contractual WAL, FRXX / DEXX, fixed rate | **LATENT DETERMINISTIC MI** | RREL39 supplies the constant instalment; the split is arithmetic |
| Contractual WAL, FRXX / DEXX, floating or resetting | **ASSUMPTION REQUIRED** | the principal split moves with an unknown future index |
| Contractual WAL, ERM (`OTHR`) | **MODEL REQUIRED** | repayment is contingent and undated; a legal maturity is not a WAL |
| Expected WAL | **ASSUMPTION-MODEL REQUIRED** | out of scope by the brief |
| Contractual YTM, any amortisation type, fixed rate (FXRL) | **LATENT DETERMINISTIC MI** | price RREL34, rate RREL43, as-of RREL6 all present |
| Contractual YTM, floating or resetting | **ASSUMPTION REQUIRED** | future index level, not a field |
| Day-count-exact YTM on Annex 2 | **FIELD GAP** | `day_count_convention` is CREL122 — Annex 3 only |
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
| **MEDIUM** | Contractual WAL — BLLT / FIXE, any rate type | RREL30, RREL24, RREL37, RREL41 | the principal series is rate-independent; pure enumeration. **Rates LOW on Trakt's own ERM book, where RREL35 is `OTHR`** (§11) |
| **MEDIUM** | Contractual WAL — FRXX / DEXX at a **fixed** rate | + RREL39, RREL43 | `payment_due` supplies the constant instalment on all five annexes; the principal split is arithmetic, not a model (§11) |
| **MEDIUM** | Contractual YTM — any amortisation type at a fixed rate | + RREL34, RREL6 | price relative to par and the as-of date are both Mandatory-scale (§12) |
| **LOW** | `outstanding_balance_period_*` as a received schedule | eight analytics fields | shaped like a dated balance path; **no code, no semantics, no consumer — needs a specification decision before any use** (§11) |
| **MEDIUM** | Supplied LGD reporting | `bank_internal_loss_given_default_lgd_estimate` | reporting the originator's own estimate is not modelling; **Trakt still builds no PD/LGD** |
| **MEDIUM** | Payment-frequency and amortisation-shape mix | RREL37, RREL35 | eligibility screens ask for these directly |
| **LOW** | Grace-period exposure | RREL36 | narrow, but free |
| **LOW** | Collateral sale-price outcomes | RREC21 | overlaps the liquidation decomposition on CRE |
| **LOW** | Annex 12 IVSS24 gross charge-offs | — | `method` empty and `source_field` null; needs a definition decision first |

---

## 11. Weighted average life, by amortisation type

Two earlier drafts of this section got this wrong in opposite directions. The
first said WAL was blocked behind a "contractual schedule builder", inferred
from the absence of a WAL function. The second split it BLLT/FIXE versus
FRXX/DEXX and treated the latter as blocked outright. Both were too coarse,
because **RREL35 is itself a governed field**: the contract tells Trakt what
happens to principal, so the question is not "does a schedule exist?" but
"given what the contract says, are the governed fields sufficient to enumerate
the cash flows without assuming anything?"

Asked that way, the answer separates cleanly — and the separator is not the
amortisation type alone. It is the amortisation type crossed with RREL42.

### The distinction that does the work

**WAL is a principal-weighted measure. It does not depend on the interest rate
at all — unless the amortisation type makes the principal series depend on it.**

- Under **BLLT** and **FIXE** the contract fixes the *principal* series
  directly ("full principal repaid in the last instalment"; "the principal
  amount repaid in each instalment is the same"). The rate is irrelevant to
  WAL. A floating-rate FIXE loan has a fully determined principal schedule.
- Under **FRXX** and **DEXX** the contract fixes the *total* instalment, so
  `principal_t = instalment − rate_t × balance_(t−1)`. The principal series is
  a function of the rate, and WAL inherits whatever uncertainty the rate has.

YTM, by contrast, needs the interest series in every case, so it always
inherits the rate's uncertainty.

That single observation dissolves most of the supposed blocker.

### What the governed fields supply

Every input below is canonical, and the annex coverage is stated rather than
assumed:

| Input | Field | Annexes |
|---|---|---|
| Amortisation shape | `amortisation_type` RREL35 | **5** |
| Rate type | `interest_rate_type` RREL42 | 3 (A2/A3/A4) |
| Outstanding balance | `current_principal_balance` RREL30 | **5** |
| Maturity | `maturity_date` RREL24 | **5** |
| Principal frequency | `scheduled_principal_payment_frequency` RREL37 | **5** |
| Interest frequency | `scheduled_interest_payment_frequency` RREL38 | **5** |
| **Constant total instalment** | `payment_due` RREL39 — *"the next contractual payment due by the obligor according to the payment frequency"* | **5** |
| Current rate | `current_interest_rate` RREL43 | **5** |
| Terminal principal | `balloon_amount` RREL41 | 3 (**not Annex 3**) |
| Principal grace | `principal_grace_period_end_date` RREL36 | **5** |
| Rate reset interval | `interest_rate_reset_interval` RREL47 | **5** |
| Rate cap / floor | `interest_rate_cap` RREL48 / `interest_rate_floor` RREL49 | **5** |
| Next contractual rate change | `interest_revision_date_1..3` RREL51/53/55 | 2 |
| As-of date | `data_cut_off_date` RREL6 | **5** |
| Price relative to par | `purchase_price` RREL34 | **5** |
| Payments already made | `number_of_payments_before_securitisation` RREL58 | **5** |
| Per-period principal | `regular_principal_instalment` | **0 — analytics only** |
| Day-count convention | `day_count_convention` CREL122 | **1 — Annex 3 only** |

`payment_due` (RREL39) is the field the previous draft missed, and it is the
one that matters most: for a French loan the total instalment *is* constant, so
RREL39 **is** the constant the decomposition needs — Mandatory-scale coverage on
all five annexes. And `regular_principal_instalment` having no regime code
turns out not to block FIXE either, because constant principal is recoverable
as `(balance − balloon) ÷ remaining periods`.

> **One trap this section creates, closed here before someone falls into it.**
> RREL39 is *principal **plus** interest*. It must never be substituted for
> `regular_principal_instalment` in the SMM denominator (§5), which needs
> scheduled **principal** alone. Doing so would inflate the deduction by the
> whole interest instalment, shrink the exposed balance and overstate every
> prepayment rate Trakt publishes. The current fallback — when
> `regular_principal_instalment` is absent, use the full opening balance and
> label the result an approximation — is the conservative direction and is the
> right one to keep.

### The matrix

Rate types per RREL42: **FXRL** fixed for life; **FXPR** fixed with future
periodic resets; **FLCF** fixed with compulsory future switch to floating;
**FLIF / FINX / FLFL / CAPP / FLCA** floating variants; **DISC / SWIC / OBLS /
MODE / OTHR** special forms.

| RREL35 | Rate type | Required inputs available? | Cash flows deterministic? | Contractual WAL | Contractual YTM | Exact blocker |
|---|---|---|---|---|---|---|
| **BLLT** | any | Yes — RREL30, RREL24 for WAL; + RREL43, RREL38, RREL34, RREL6 for YTM | **Principal: yes, unconditionally.** Interest: only if fixed | **LATENT DETERMINISTIC MI** — WAL *is* the term to maturity | **LATENT DETERMINISTIC MI** (FXRL) / **ASSUMPTION REQUIRED** (floating) | none for WAL; future index for floating YTM |
| **FIXE** | any | Yes — RREL30, RREL37, RREL24, RREL41; instalment derivable if `regular_principal_instalment` absent | **Principal: yes, unconditionally** — constant principal is rate-independent | **LATENT DETERMINISTIC MI** | **LATENT DETERMINISTIC MI** (FXRL) / **ASSUMPTION REQUIRED** (floating) | none for WAL; future index for floating YTM |
| **FRXX** | **FXRL** | Yes — RREL39 instalment, RREL43 rate, RREL30, RREL37, RREL24, RREL36 | **Yes** — iterate `principal_t = RREL39 − r × balance_(t−1)` | **LATENT DETERMINISTIC MI** | **LATENT DETERMINISTIC MI** | none — arithmetic only |
| **FRXX** | FXPR / FLCF | Partly — RREL51 gives the *date* of the change, not the rate | Deterministic **only to the first revision date** | **ASSUMPTION REQUIRED** beyond RREL51 | **ASSUMPTION REQUIRED** beyond RREL51 | the reset rate at RREL51; RREL50 gives the margin, not the index |
| **FRXX** | FLIF / FINX / FLFL / CAPP / FLCA | Rate structure known (RREL44–49), future index level not | **No** — instalment resets, and the principal split moves with it | **ASSUMPTION REQUIRED** | **ASSUMPTION REQUIRED** | the future reference index from the next reset (RREL47) onward. RREL48/49 bound it; they do not determine it |
| **DEXX** | **FXRL** | Yes — as FRXX, plus the interest-only first period, datable from RREL23/RREL58/RREL36 | **Yes** | **LATENT DETERMINISTIC MI** | **LATENT DETERMINISTIC MI** | none — arithmetic only |
| **DEXX** | resetting / floating | as FRXX above | No | **ASSUMPTION REQUIRED** | **ASSUMPTION REQUIRED** | as FRXX above |
| **OTHR** — Trakt's ERM book | any | Balance, legal maturity and rate all present | **No** — repayment is contingent on death, sale or long-term care, none of which is contractually dated | **not applicable** — see below | **not applicable** | no contractual repayment date exists |
| **OTHR** — anything else | any | unknown by construction | Unknown | **FIELD GAP** — the contract's principal behaviour is not stated | **FIELD GAP** | RREL35 = OTHR conveys no principal profile |
| any | frequency RREL37 = `OTHR` | Frequency unknown | **No** | **FIELD GAP** | **FIELD GAP** | the payment interval itself |

### Two conventions to state, neither of which is an assumption about the future

**Payment-date anchoring on Annex 2.** Annex 2 carries no payment-date field —
`next_payment_date` and `start_date_of_amortisation` are CREL104 and CREL16,
Annex 3 only. Residential payment dates must be anchored either by counting
back from `maturity_date` at RREL37 frequency, or forward from
`origination_date` plus RREL58 payments made. Both are deterministic
arithmetic, they can disagree by a stub period, and choosing between them is a
**stated convention**, not a forecast. It belongs in a methodology identifier,
not in a gap register.

**Day count.** `day_count_convention` exists only as CREL122, Annex 3. A
**period-basis** YTM — discount at the payment frequency, annualise by it —
needs no day-count convention and is exact on regular schedules. A
**day-count-exact** YTM on residential does need it, and that is the one
genuine, precisely nameable field gap in this whole section: *Annex 2 has no
day-count convention field.*

### ERM, kept separate

Three things must not be run together, and the previous draft nearly did:

| | ERM |
|---|---|
| **Contractual / legal maturity** | Exists. `maturity_date` RREL24 is populated — often a long-stop age |
| **Contractual WAL** | **Does not exist.** There are no contractual principal payments before an undated contingent event |
| **Expected WAL** | **MODEL REQUIRED** — mortality, morbidity, voluntary redemption |

`config/asset/product_defaults_ERM.yaml` already reasons this out: a lifetime
mortgage "rolls up interest and repays at death/sale ... it is **NOT** a
scheduled bullet amortisation under the Annex 2 definition", and Trakt
therefore reports `OTHR`. A legal long-stop maturity must not be pressed into
service as a contractual WAL — it would produce a confident number that is
wrong by decades. **No mortality modelling is proposed, and none should be
built into Trakt MI.**

### The undocumented family, unchanged

`outstanding_balance_period_1 / _2_120 / _121_599 / _600`, each with a `_date`
companion, have the shape of a received, dated balance path — which for `OTHR`
loans would be the only route to any WAL at all. They carry no regime code, no
semantics entry, no alias and **no consumer anywhere in the repository**.
Whether they are a projection, a contractual schedule, or something else is a
specification question for whoever defined them. Nothing should be built on
them until it is answered.

---

## 12. Yield to maturity and yield to worst

**YTM needs three things: a price, dated contractual principal, and dated
contractual interest.** Taking them in turn against the governed fields:

| Component | Status |
|---|---|
| **Price** | **Present.** `purchase_price` RREL34, all five annexes, defined as *"the price, relative to par, at which the underlying exposure was purchased by the SSPE. Enter 100 if no discounting was applied"* — already a percent-of-par scale, so no cash consideration is needed |
| **Dated contractual principal** | Exactly as §11 — unconditional for BLLT/FIXE, deterministic for fixed-rate FRXX/DEXX, assumption-dependent for resetting and floating |
| **Dated contractual interest** | `current_interest_rate` RREL43 (all five) × balance × `scheduled_interest_payment_frequency` RREL38 (all five). Deterministic **only where the rate is contractually fixed** (RREL42 = FXRL) |
| **Discount anchor** | **Present.** `data_cut_off_date` RREL6, all five annexes |
| **Day-count basis** | **Present on Annex 3 only** (CREL122). Not required for a period-basis YTM; required for a day-count-exact one |

So the answer to "is something else missing?" is specific rather than general:
**nothing is missing for a period-basis contractual YTM on a fixed-rate loan of
any amortisation type.** The two named gaps are the Annex 2 day-count
convention (for day-count exactness only) and the future index level (for
floating and resetting loans, which is an assumption, not a field).

One caveat before anyone quotes the result: on a warehouse book bought at par,
RREL34 is 100 for every loan and YTM collapses to the coupon. That is a correct
answer that will be read as a broken calculation.

**YTW remains an exposure gap, and for an unrelated reason.** Yield to worst is
the minimum yield across call and prepayment scenarios and, where it is
normally quoted, it is a **note-level** measure. Trakt does not model the
liability side: no canonical field carries a tranche or class balance, coupon,
attachment point or paydown, and the nine fields whose names mention notes,
seniority or subordination are all attributes of the **underlying exposure**.
The Annex 12 deal template's `cashflow_items` and `triggers_tests_events` lists
are both empty. No field or formula fixes that.

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
pinned at `cdefc25` and candidate from a worktree pinned at `dada1b9`. Both
worktrees verified clean (`git status --porcelain` empty) and verified by
`rev-parse` *before* and *after* their runs, not merely by the command that
created them.

| | Baseline `cdefc25` | Candidate `dada1b9` | Δ |
|---|---|---|---|
| passed | 5,176 | 5,207 | **+31** |
| failed | 64 | 64 | **0** |
| errors | 13 | 13 | **0** |
| skipped | 33 | 33 | 0 |
| subtests passed | 6 | 6 | 0 |
| collected | 5,273 | 5,304 | +31 |
| elapsed | 2,475.35s | 2,414.76s | −60.6s |

**Full ID comparison, both directions:**

```
FAILED ids present at baseline and absent at candidate  :  none
FAILED ids present at candidate and absent at baseline  :  none
ERROR  ids present at baseline and absent at candidate  :  none
ERROR  ids present at candidate and absent at baseline  :  none
```

The 64 failure IDs and 13 error IDs are **identical sets**, compared by sorted
full node ID rather than by count — a count match would survive one failure
being traded for another, and that is the case this comparison exists to catch.

**The +31 is fully accounted for, with nothing unexplained:**

| Source | Tests |
|---|---|
| `tests/test_metric_methodology.py`, created earlier in this sprint at `7d3c05f` | 15 |
| loss severity and the recoveries-double-count guard (this commit) | 6 |
| `tests/test_annex12_arrears_buckets.py` (this commit) | 8 |
| readiness framework / library agreement guards (this commit) | 2 |
| **Total** | **31** |

Confirmed by direct collection in the candidate worktree: the two changed test
files collect 29 (21 + 8), and the readiness file's two additions make 31.

**Regression neutrality is claimed, and this is the basis for it:** no test
passing at `cdefc25` fails at `dada1b9`, no new error appears, and every
additional passing test is a test this sprint wrote.

**A process finding worth recording.** An earlier attempt tested the wrong
revision: `git worktree add -f` against an existing directory **fails
silently**, so a run launched that way reported on a stale commit while
appearing to test HEAD. It was caught by checking `rev-parse` on the worktree
rather than trusting the command that created it. Verifying which tree ran is
part of the regression, not preparation for it — and this is the second time
this sprint that regression discipline failed in a way the output did not show.

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

**Where WAL and YTM actually stand.** Re-checking against RREL35 and RREL42
rather than inferring from the absence of a function moved most of this out of
the gap column:

| Asset / amortisation case | Contractual WAL | Contractual YTM | Expected WAL / yield | Classification |
|---|---|---|---|---|
| **BLLT**, any rate type | **Yes** — the term to maturity | Yes if RREL42 = FXRL | n/a | **LATENT DETERMINISTIC MI** (YTM: ASSUMPTION REQUIRED if floating) |
| **FIXE**, any rate type | **Yes** — constant principal is rate-independent | Yes if FXRL | n/a | **LATENT DETERMINISTIC MI** (YTM: ASSUMPTION REQUIRED if floating) |
| **FRXX / DEXX**, fixed rate (FXRL) | **Yes** — iterate RREL39 against RREL43 | **Yes** | n/a | **LATENT DETERMINISTIC MI** |
| **FRXX / DEXX**, fixed with resets (FXPR / FLCF) | To RREL51 only | To RREL51 only | needs a rate path | **ASSUMPTION REQUIRED** beyond the revision date |
| **FRXX / DEXX**, floating (FLIF / FINX / FLFL / CAPP / FLCA) | No | No | needs a rate path | **ASSUMPTION REQUIRED** |
| **ERM** (RREL35 = `OTHR`) | **Does not exist** | Does not exist | needs mortality / redemption | **MODEL REQUIRED** — and not to be built |
| Any type where RREL37 = `OTHR` | No | No | — | **FIELD GAP** — the payment interval itself |
| Day-count-exact YTM, Annex 2 | — | No | — | **FIELD GAP** — `day_count_convention` is CREL122, Annex 3 only |

**1. Contractual WAL today, from existing governed fields:** BLLT and FIXE for
*any* rate type, because their principal series is contractually fixed and
rate-independent; and FRXX and DEXX where RREL42 = FXRL.

**2. Contractual YTM today:** every amortisation type in answer 1, restricted
to fixed-rate loans, on a period basis. Price (RREL34), rate (RREL43),
frequency (RREL38) and as-of date (RREL6) are all Mandatory-scale on five
annexes.

**3. Only a small deterministic implementation:** all of the above. One
schedule-enumeration function plus an IRR solver — no engine, no model, no new
field. The heaviest case, FRXX at a fixed rate, is a single recurrence:
`principal_t = RREL39 − r × balance_(t−1)`.

**4. Genuinely missing fields:** two, and both are narrow. `day_count_convention`
does not exist on Annex 2 (CREL122 is Annex 3), which blocks a day-count-exact
YTM but not a period-basis one. And where RREL37 or RREL35 is `OTHR`, the
contract's own terms are not conveyed — for those loans the gap is real.

**5. Assumptions or models:** floating and resetting loans need a future index
path (RREL48/49 bound it, RREL47/51 date it, nothing supplies it). ERM needs
mortality and redemption behaviour and is out of scope — a legal long-stop
maturity must never be dressed up as a contractual WAL.

**6. Is "a contractual schedule builder is the blocker" still supportable? No —
that description is now too broad and should be retired.** It conflates a
missing calculation with missing information. The evidence says the opposite:
for the majority of amortisation types the contractual cash flows are
*deterministically enumerable from fields Trakt already holds*, and what remains
unwritten is arithmetic. The genuine blockers are three specific things — the
future index level on floating loans, the Annex 2 day-count convention, and the
absence of any contractual repayment date on ERM — and only the last is
structural.

**What that means for a 2.5D sprint.** It is smaller than previously scoped and
should be sequenced by asset class, not by metric. On CRE and amortising
residential the work is one enumeration function and an IRR solver over fields
already present. On Trakt's own ERM book it delivers nothing, because the
question is undefined there. Two things should be settled first: what the
undocumented `outstanding_balance_period_*` family actually contains, since it
may already *be* the schedule; and which payment-date anchoring convention
Annex 2 uses, since counting back from maturity and forward from origination can
disagree by a stub period and the choice belongs in a methodology identifier.

The Annex 12 CPR wiring is a separate, smaller item — the calculation exists and
only the projector's single-period input is in the way. YTW is separate again,
and is a note-level data question rather than a methodology one.

**The observation from the last review still holds, and got stronger.** Every
defect in this sprint was found by reading a definition next to the code that
claimed to implement it. Not one was found by running anything — every test
passed throughout, including the tests I had just written. The most productive
hour of this sprint was spent reading `annex2_field_universe.yaml`, a file the
repository already had.
