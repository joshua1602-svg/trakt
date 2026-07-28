# Business Semantics Registry — Review and Validation

**Artefacts:**

- Registry: `config/business_semantics_registry.yaml`
- Generator: `scripts/build_business_semantics_registry.py`
- Tests: `tests/test_business_semantics_registry.py`
- Source: `config/system/fields_registry.yaml` (canonical field registry)

**Scope.** This review covers the creation of the governed Business Semantics
Registry: the subset of canonical registry entries that represent meaningful
analytical concepts or metrics, enriched with controlled metadata for future
reasoning workflows (period-change analysis and portfolio risk comparison in
this first version, plus ranking and monitoring tags). Every classification is
grounded in the canonical registry metadata (category / format / layer /
portfolio_type / regime mapping), the curated MI semantics layer
(`mi_agent/build_mi_semantics_registry.py`, `mi_agent/mi_semantics_field_registry.yaml`)
or the risk-monitor configuration (`config/mi/risk_monitor.yaml`). No
calculation logic, materiality thresholds, covenant limits, risk weights or
business definitions were invented.

The source field registry was **not modified**.

---

## 1. Coverage summary

| Measure | Count | % of source |
|---|---:|---:|
| Source registry entries reviewed | **499** | 100% |
| Included in the Business Semantics Registry | **242** | 48.5% |
| Excluded | **257** | 51.5% |
| — of which clear exclusions | 232 | 46.5% |
| — of which uncertain, excluded pending human review | 25 | 5.0% |
| Included entries flagged ambiguous (in registry, needs review) | 1 (`equity`) | — |

Note: the task brief referenced "1,000+ entries"; the canonical field registry
(`config/system/fields_registry.yaml`) contains **499** field entries (the
file's ~5,800 lines include per-regime mapping blocks under each field, which
is likely the source of the higher estimate). All 499 were reviewed
individually.

Every field was reviewed against the inclusion criteria (measurable exposure,
balance/value, performance, payment performance, risk, leverage,
concentration, liquidity, valuation, cashflow, return/pricing,
maturity/duration, collateral, credit quality, delinquency, loss, eligibility,
forecast, operational performance, data quality). Uncertain fields were **not**
forced into the registry — they are tracked in section 5 with a proposed
classification and a recommended human decision.

---

## 2. Included-entry summary

All values below are drawn from the generated registry (they are re-derivable
by loading the YAML; the validation tests assert taxonomy conformance).

### By analytical concept (primary, one per entry)

Counts reflect the **v2 schema** (see section 8): `concentration` is no
longer a primary concept — its 22 former entries were re-homed to their
substantive concepts (geography, product_mix, origination, credit_quality,
collateral, exposure, operational_performance).

| Concept | Count | | Concept | Count |
|---|---:|---|---|---:|
| credit_quality | 49 | | geography | 5 |
| collateral | 29 | | operational_performance | 5 |
| cashflow | 22 | | coverage | 4 |
| loss | 21 | | eligibility | 4 |
| exposure | 18 | | liquidity | 4 |
| valuation | 17 | | origination | 4 |
| payment_performance | 14 | | product_mix | 4 |
| leverage | 13 | | data_quality | 2 |
| pricing | 13 | | forecast | 1 |
| maturity | 12 | | tail_risk | 1 |

### By analytical role (v2)

| Role | Count | Meaning |
|---|---:|---|
| measure | 159 | Quantitative metrics (sum / average / weighted_average / share aggregation) |
| dimension | 69 | Grouping/mix dimensions (distribution aggregation); concentration analysis follows from this role plus workflow tags |
| derived_input | 10 | Snapshot-pair and prior-period fields that exist only to feed migration/movement computation — never reported as standalone metrics |
| supporting_attribute | 4 | Event/recency attributes (`current_valuation_date`, `date_last_in_arrears`, `default_date`, `date_of_restructuring`) |

### By temporality (v2)

| Temporality | Count | Period-change handling |
|---|---:|---|
| point_in_time | 198 | Compare snapshot stocks directly |
| static_baseline | 26 | Never expect loan-level movement; use as comparison baseline only |
| period_flow | 13 | Compare period totals directly |
| cumulative | 5 | **Must be differenced before period comparison** (`cumulative_*` fields and `allocated_losses`, the ESMA losses-to-date cumulative) |

### By portfolio comparability (v2)

| Comparability | Count |
|---|---:|
| comparable | 210 |
| requires_scale_alignment (lender/servicer scales & vocabularies) | 13 |
| within_asset_class_only (lease/occupancy income-property concepts) | 9 |
| not_comparable (the 10 derived_input fields) | 10 |

### By category (secondary, one or more per entry)

credit_quality 53 · collateral 52 · cashflow 42 · concentration 33 · loss 29 ·
risk 26 · exposure 23 · payment_performance 22 · valuation 21 · income 19 ·
leverage 18 · maturity 18 · obligor_financials 17 · pricing 15 ·
delinquency 10 · recovery 10 · operational_performance 8 · eligibility 7 ·
liquidity 7 · tail_risk 7 · data_quality 6 · product_mix 6 · affordability 5 ·
geography 5 · coverage 4 · return 2 · seasoning 2 · forecast 1

### By workflow tag

| Workflow tag | Count | Notes |
|---|---:|---|
| portfolio_comparison | 204 | Metrics/dimensions meaningfully comparable across portfolios |
| period_change | 106 | Metrics reasonably assessable across previous run / MoM / QoQ / YTD / YoY / user-selected periods. Static baselines (`original_*`, `*_at_securitisation*`) deliberately do **not** carry this tag |
| monitoring | 100 | Risk-relevant metrics and the concentration dimensions monitored by `config/mi/risk_monitor.yaml` |
| ranking | 28 | Loan-level magnitude metrics suitable for top-N / ranking views |

### By confidence

| Confidence | Count |
|---|---:|
| high | 71 |
| medium | 171 |
| low | 0 — every candidate that would have carried `low` confidence was excluded pending human review instead (section 5), per "do not force uncertain fields into the registry" |

### By asset applicability

| Asset applicability | Count |
|---|---:|
| cross_asset (canonical `portfolio_type: common`) | 189 |
| sme | 18 |
| equity_release | 14 |
| commercial_real_estate | 10 |
| equipment_leasing | 7 |
| residential_mortgage | 4 |

Asset applicability is derived mechanically from the canonical registry's
`portfolio_type` (with a small number of curated overrides where the MI layer
scopes a common field to equity release, e.g. `negative_equity_guarantee`).
No applicability was invented beyond what the source registry declares.

### By default aggregation (renamed from `aggregation_type` in v2)

| Default aggregation | Count | Typical use |
|---|---:|---|
| sum | 89 | Balances, amounts, cashflows |
| distribution | 77 | Status, grade, mix and cohort dimensions |
| weighted_average | 32 | Rates, LTV, PD/LGD, ratios — every one carries `weight_field: current_outstanding_balance` (the governed MI default weight) |
| share | 24 | Y/N flags — every one carries `share_basis: count` (the governed MI flag default); balance-weighted variants are a future per-workflow decision |
| average | 20 | Days past due, ages, terms, incomes |

### By directionality

| Directionality | Count |
|---|---:|
| neutral | 80 |
| context_dependent | 59 |
| higher_is_worse | 56 |
| higher_is_better | 45 |
| lower_is_better | 1 (`expected_timing_of_recoveries`) |
| lower_is_worse | 1 (`property_leasehold_expiry`) |

Directionality was only asserted where the direction is unambiguous from the
metric's meaning (e.g. arrears, losses, PD, LGD, LTV higher-is-worse;
coverage, recoveries, valuations higher-is-better). Rates, prepayments and
ordinal scales without a registry-defined ordering are `context_dependent`;
sizes and mixes are `neutral`.

---

## 3. Exclusion summary

257 fields were excluded (232 clear + 25 uncertain-pending-review). Main
exclusion reasons for the 232 clear exclusions (counts are from a keyword
classification of the excluded set and are indicative):

| Exclusion reason | ~Count | Examples |
|---|---:|---|
| Non-analytical date / event timestamp | 63 | `payment_date`, `pool_addition_date`, `interest_revision_date_1..3`, `servicer_watchlist_date`, `borrower_1_DOB`, at-securitisation date stamps |
| Technical / structural / descriptive enums & terms | 39 | `day_count_convention`, `waterfall_type` companions, `covenant_breach_trigger`, `scheduled_*_payment_frequency`, `risk_model_version` (provenance) |
| Identifier / reference | 29 | `loan_identifier`, `unique_identifier`, `*_legal_entity_identifier`, `international_securities_identification_number`, `pool_identifier`, `source_portfolio_id` (segmentation key, not a metric) |
| Technical / structural flags | 20 | `noteholder_consent`, `managed_by_clo`, `collection_of_escrows`, `servicing_standard`, `main_residence` (covered by `occupancy_type`) |
| Swap / waterfall structural fields | 18 | `currency_swap_notional`, `breakage_costs_*`, `waterfall_a_b_*` (consistent with the MI layer's explicit exclusion of swap/waterfall fields) |
| Static descriptive facts | 15 | `commercial_area`, `net_square_metres`, `floor_of_property`, `year_last_renovated`, `rounding_increment`, `revision_margin_1..3` (contractual schedule detail) |
| Currency codes | 13 | `*_currency` fields (the analytical currency-mix dimension is carried by `exposure_currency_denomination`, which **is** included) |
| Name / free text | 12 | `borrower_legal_name`, `property_name`, `description`, `sponsor`, `tenant_name`, `name_of_valuer_at_securitisation` |
| Insufficient definition (no format, duplicate concept) | 11 | `payment_type`, `ranking` (vs `lien`), `prepayment_penalty` (vs `early_repayment_charge`), `interest_cap_rate` (vs `interest_rate_cap`) |
| Duplicative external codes | 5 | rating-agency industry codes (`fitch/moody_s/s_p/other_industry_code`), `obligor_tax_code` — industry concentration is carried by `nace_industry_code` |
| Balance-period buckets (technical) | 4 | `outstanding_balance_period_*` (consistent with the MI layer's exclusion) |
| Address | 3 | `property_address`, `property_post_code`, `property_postcode` (geography is carried by the NUTS/region fields and MI-curated `postcode`) |

Per the brief, excluded fields are not individually listed beyond the
examples above; the authoritative excluded set is *(source fields) −
(registry fields)* and is enforced by the validation tests for representative
cases.

---

## 4. Full included-entry inventory (242 fields)

<!-- BEGIN GENERATED INVENTORY (from config/business_semantics_registry.yaml) -->

| Field | Display name | Concept | Role | Temporality | Categories | Workflow tags | Directionality | Confidence | Rationale |
|---|---|---|---|---|---|---|---|---|---|
| `account_status` | Account Status | credit_quality | dimension | point_in_time | credit_quality, payment_performance | period_change, portfolio_comparison, monitoring | context_dependent | high | Current account/loan status (performing, arrears, default, redeemed); primary book-composition status dimension (MI core). |
| `accrued_interest_in_period` | Accrued Interest In Period | cashflow | measure | period_flow | cashflow, exposure | period_change | neutral | high | Interest accrued in the reporting period; period income accrual and balance roll-up component. |
| `actual_default_interest` | Actual Default Interest | cashflow | measure | period_flow | cashflow, payment_performance | period_change | neutral | medium | Default interest actually charged/collected in the period (ESMA CREL135). |
| `allocated_losses` | Allocated Losses | loss | measure | cumulative | loss, risk | period_change, portfolio_comparison, ranking, monitoring | higher_is_worse | high | Losses allocated to the loan (MI core); realised credit loss measure. |
| `amortisation_type` | Amortisation Type | cashflow | dimension | point_in_time | cashflow, concentration, product_mix | period_change, portfolio_comparison, monitoring | context_dependent | high | Amortisation/repayment profile (MI core enum); interest-only vs repayment mix monitored by the risk monitor. |
| `amounts_added_to_escrows_in_current_period` | Amounts Added To Escrows In Current Period | liquidity | measure | period_flow | liquidity, cashflow | period_change | neutral | medium | Escrow additions in the period (CRE); reserve funding flow. |
| `amounts_held_in_escrow` | Amounts Held In Escrow | liquidity | measure | point_in_time | liquidity, collateral | period_change, monitoring | higher_is_better | medium | Amounts held in escrow (CRE); reserve liquidity supporting the exposure. |
| `arrears_balance` | Arrears Balance | payment_performance | measure | point_in_time | payment_performance, delinquency, risk | period_change, portfolio_comparison, ranking, monitoring | higher_is_worse | high | Total amount in arrears; core delinquency metric (MI core). |
| `asset_insurance` | Asset Insurance | collateral | measure | point_in_time | collateral, risk | portfolio_comparison | higher_is_better | medium | Asset insurance flag (SME); insured share of collateral. |
| `asset_type` | Asset Type | collateral | dimension | point_in_time | concentration, product_mix | portfolio_comparison | neutral | medium | Asset type (SME, governed enum); financed-asset mix dimension. |
| `balloon_amount` | Balloon Amount | maturity | measure | point_in_time | maturity, cashflow, exposure | portfolio_comparison, monitoring | context_dependent | medium | Balloon repayment amount; repayment concentration at maturity (refinancing risk). |
| `bank_internal_loss_given_default_lgd_estimate` | Bank Internal Loss Given Default LGD Estimate | loss | measure | point_in_time | loss, credit_quality | portfolio_comparison, monitoring | higher_is_worse | medium | Bank internal LGD estimate (SME); loss severity assumption. |
| `bank_internal_loss_given_default_lgd_estimate_down_turn` | Bank Internal Loss Given Default LGD Estimate Down Turn | loss | measure | point_in_time | loss, credit_quality, tail_risk | portfolio_comparison | higher_is_worse | medium | Bank internal downturn LGD estimate (SME); stressed loss severity. |
| `bank_internal_rating` | Bank Internal Rating | credit_quality | dimension | point_in_time | credit_quality, risk | portfolio_comparison, monitoring | context_dependent | medium | Bank internal obligor rating (SME); internal credit-quality scale without a registry-defined ordering. |
| `borrower_1_age` | Borrower 1 Age | maturity | measure | point_in_time | maturity, tail_risk | portfolio_comparison | context_dependent | medium | Primary borrower age (equity release); duration/longevity input. |
| `borrower_2_age` | Borrower 2 Age | maturity | measure | point_in_time | maturity, tail_risk | portfolio_comparison | context_dependent | medium | Second borrower age (equity release); joint-life duration input. |
| `borrower_basel_iii_segment` | Borrower Basel Iii Segment | credit_quality | dimension | point_in_time | concentration, credit_quality | portfolio_comparison | neutral | medium | Borrower Basel III segment (SME); regulatory risk-segmentation mix. |
| `borrower_jurisdiction` | Borrower Jurisdiction | geography | dimension | point_in_time | concentration, geography | portfolio_comparison | neutral | medium | Borrower legal jurisdiction (MI core); country-mix dimension. |
| `broker_channel` | Broker Channel | origination | dimension | point_in_time | concentration | period_change, portfolio_comparison, monitoring | neutral | high | Broker/origination channel (MI core); channel concentration monitored by the risk monitor. |
| `charge_type` | Charge Type | collateral | dimension | point_in_time | collateral | portfolio_comparison | neutral | medium | Charge type (governed enum); nature of the charge over collateral. |
| `collateral_geography` | Collateral Geography | geography | dimension | point_in_time | concentration, geography, collateral | period_change, portfolio_comparison, monitoring | neutral | high | Readable collateral region label (MI core Region); geographic concentration dimension. |
| `collateral_type` | Collateral Type | collateral | dimension | point_in_time | collateral, concentration | portfolio_comparison, monitoring | neutral | high | Collateral type (governed enum); collateral mix dimension. |
| `collateral_value` | Collateral Value | valuation | measure | point_in_time | valuation, collateral | period_change, portfolio_comparison, ranking, monitoring | higher_is_better | high | Collateral value; basis of collateral cover and LTV. |
| `collateralisation_ratio` | Collateralisation Ratio | collateral | measure | point_in_time | collateral, leverage | period_change, portfolio_comparison, monitoring | higher_is_better | medium | Collateralisation ratio (SME); collateral cover of the exposure. |
| `commercial_liabilities` | Commercial Liabilities | credit_quality | measure | point_in_time | obligor_financials, leverage | portfolio_comparison | context_dependent | medium | Obligor commercial liabilities (SME financial statements). |
| `committed_undrawn_facility_underlying_exposure_balance` | Committed Undrawn Facility Underlying Exposure Balance | exposure | measure | point_in_time | exposure, liquidity | period_change, portfolio_comparison, monitoring | context_dependent | medium | Committed undrawn facility balance; contingent exposure and funding commitment measure. |
| `contractual_annual_rental_income` | Contractual Annual Rental Income | cashflow | measure | point_in_time | cashflow, income | period_change, portfolio_comparison, ranking, monitoring | higher_is_better | high | Contractual annual rental income (MI extended); income-producing collateral cashflow. |
| `corporate_guarantor_bank_internal_1_year_probability_default` | Corporate Guarantor Bank Internal 1 Year Probability Default | credit_quality | measure | point_in_time | credit_quality, collateral | portfolio_comparison, monitoring | higher_is_worse | medium | Guarantor one-year PD; credit quality of guarantee support. |
| `credit_impaired_obligor` | Credit Impaired Obligor | credit_quality | measure | point_in_time | credit_quality, risk | period_change, portfolio_comparison, monitoring | higher_is_worse | high | Credit-impaired obligor flag (ESMA); share of book with impaired obligors. |
| `cumulative_accrued_interest` | Cumulative Accrued Interest | exposure | measure | cumulative | exposure, cashflow | period_change, portfolio_comparison, monitoring | context_dependent | high | Cumulative accrued (rolled-up) interest; drives balance compounding, central to equity-release roll-up exposure. |
| `cumulative_drawn_amount` | Cumulative Drawn Amount | exposure | measure | cumulative | exposure, cashflow | period_change, portfolio_comparison | neutral | medium | Cumulative amount drawn on the facility; exposure build-up over time. |
| `cumulative_prepayments` | Cumulative Prepayments | cashflow | measure | cumulative | cashflow | period_change, portfolio_comparison, monitoring | context_dependent | high | Cumulative unscheduled prepayments (MI extended); prepayment behaviour measure. |
| `cumulative_recoveries` | Cumulative Recoveries | loss | measure | cumulative | recovery, loss | period_change, portfolio_comparison | higher_is_better | high | Cumulative recoveries to date (ESMA); realised recovery performance. |
| `current_debt_service_coverage_ratio` | Current Debt Service Coverage Ratio | coverage | measure | point_in_time | coverage, cashflow, credit_quality | period_change, portfolio_comparison, ranking, monitoring | higher_is_better | high | Current DSCR (MI extended); income cover of debt service, core CRE/income-property risk metric. |
| `current_default_interest_rate` | Current Default Interest Rate | pricing | measure | point_in_time | pricing, payment_performance | portfolio_comparison | neutral | medium | Current default interest rate charged on defaulted amounts. |
| `current_index_rate` | Current Index Rate | pricing | measure | point_in_time | pricing | period_change, monitoring | neutral | medium | Current index/reference rate underlying floating pricing. |
| `current_interest_coverage_ratio` | Current Interest Coverage Ratio | coverage | measure | point_in_time | coverage, cashflow | period_change, portfolio_comparison, ranking, monitoring | higher_is_better | high | Current interest coverage ratio; income cover of interest cost. |
| `current_interest_rate` | Current Interest Rate | pricing | measure | point_in_time | pricing, return | period_change, portfolio_comparison, ranking, monitoring | context_dependent | high | Current interest rate (MI core); weighted-average portfolio coupon/yield measure. |
| `current_interest_rate_index` | Current Interest Rate Index | pricing | dimension | point_in_time | pricing, concentration | portfolio_comparison | neutral | medium | Reference index (governed enum); basis-risk mix of floating-rate exposures. |
| `current_interest_rate_margin` | Current Interest Rate Margin | pricing | measure | point_in_time | pricing, return | period_change, portfolio_comparison, ranking, monitoring | context_dependent | high | Current interest margin/spread (MI extended). |
| `current_loan_to_value` | Current Loan To Value | leverage | measure | point_in_time | leverage, collateral, risk | period_change, portfolio_comparison, ranking, monitoring | higher_is_worse | high | Current LTV (MI core); weighted-average leverage metric for collateral and portfolio risk, with governed LTV buckets. |
| `current_outstanding_balance` | Current Outstanding Balance | exposure | measure | point_in_time | exposure, risk | period_change, portfolio_comparison, ranking, monitoring | neutral | high | Primary current-exposure metric; the MI layer's default balance/weighting field for portfolio aggregation. |
| `current_principal_balance` | Current Principal Balance | exposure | measure | point_in_time | exposure | period_change, portfolio_comparison, ranking, monitoring | neutral | high | Current outstanding principal; core exposure measure used across regulatory annexes and MI. |
| `current_residual_value_of_asset` | Current Residual Value Of Asset | valuation | measure | point_in_time | valuation, collateral | period_change, portfolio_comparison, monitoring | higher_is_better | medium | Current residual value of the leased asset (equipment); residual value risk measure. |
| `current_valuation_amount` | Current Valuation Amount | valuation | measure | point_in_time | valuation, collateral | period_change, portfolio_comparison, ranking, monitoring | higher_is_better | high | Most recent collateral valuation (MI core). |
| `current_valuation_date` | Current Valuation Date | valuation | supporting_attribute | point_in_time | valuation, data_quality | monitoring | neutral | medium | Date of the most recent valuation (MI extended); valuation staleness profile. |
| `current_valuation_method` | Current Valuation Method | valuation | dimension | point_in_time | valuation, data_quality | portfolio_comparison | context_dependent | medium | Valuation method (governed enum); valuation reliability mix (e.g. full survey vs indexed/AVM). |
| `customer_type` | Customer Type | credit_quality | dimension | point_in_time | concentration | portfolio_comparison | neutral | medium | Customer type (ESMA enum); borrower-type mix dimension. |
| `date_last_in_arrears` | Date Last In Arrears | payment_performance | supporting_attribute | point_in_time | payment_performance, delinquency | monitoring | neutral | medium | Date last in arrears (MI extended); recency of arrears experience. |
| `date_of_lease_expiration` | Date Of Lease Expiration | maturity | dimension | point_in_time | maturity, cashflow | monitoring | neutral | medium | Lease expiration date (equipment); income and asset return horizon. |
| `date_of_restructuring` | Date Of Restructuring | credit_quality | supporting_attribute | point_in_time | credit_quality, payment_performance | monitoring | neutral | medium | Date the loan was restructured (MI extended); forbearance recency profile. |
| `days_in_arrears_prior` | Days In Arrears Prior | payment_performance | derived_input | point_in_time | payment_performance, delinquency | period_change | higher_is_worse | medium | Prior-period days past due; comparison basis for arrears deterioration flags (risk_monitor arrears_days_increase). |
| `debt_service_coverage_ratio_at_the_securitisation_date` | Debt Service Coverage Ratio At The Securitisation Date | coverage | measure | static_baseline | coverage, cashflow | portfolio_comparison | higher_is_better | medium | DSCR at securitisation; static baseline for coverage deterioration comparison. |
| `debt_to_income_ratio` | Debt To Income Ratio | leverage | measure | point_in_time | leverage, affordability, risk | period_change, portfolio_comparison, ranking, monitoring | higher_is_worse | high | Debt-to-income ratio (MI extended); borrower leverage and affordability metric. |
| `debt_type` | Debt Type | product_mix | dimension | point_in_time | concentration, product_mix | portfolio_comparison | neutral | medium | Debt type (governed enum); instrument-mix dimension. |
| `default_amount` | Default Amount | loss | measure | point_in_time | loss, credit_quality, exposure | period_change, portfolio_comparison, ranking, monitoring | higher_is_worse | high | Balance at the point of default (MI core); defaulted exposure measure. |
| `default_date` | Default Date | credit_quality | supporting_attribute | point_in_time | credit_quality, loss | monitoring | neutral | medium | Date of default (MI extended); default timing/vintage profile. |
| `default_or_foreclosure_on_the_loan_per_basel_iii_definition` | Default Or Foreclosure On The Loan Per Basel Iii Definition | credit_quality | measure | point_in_time | credit_quality, loss | period_change, portfolio_comparison, monitoring | higher_is_worse | medium | Default/foreclosure flag under the Basel III definition; default incidence measure. |
| `default_or_foreclosure_on_the_loan_per_the_transaction_definition` | Default Or Foreclosure On The Loan Per The Transaction Definition | credit_quality | measure | point_in_time | credit_quality, loss | period_change, portfolio_comparison, monitoring | higher_is_worse | medium | Default/foreclosure flag under the transaction definition; default incidence measure. |
| `defaulted_underlying_exposure_purchase_price` | Defaulted Underlying Exposure Purchase Price | valuation | measure | static_baseline | valuation, loss | portfolio_comparison | neutral | medium | Purchase price of defaulted exposures; NPL acquisition pricing measure. |
| `deferred_interest` | Deferred Interest | payment_performance | measure | point_in_time | payment_performance, cashflow | period_change, portfolio_comparison, monitoring | higher_is_worse | medium | Deferred interest amount; unpaid interest carried forward. |
| `deposit_amount` | Deposit Amount | leverage | measure | static_baseline | leverage, collateral | portfolio_comparison | higher_is_better | medium | Deposit amount; borrower stake reducing effective leverage. |
| `dominion_bond_rating_service_dbrs_public_rating_equivalent` | DBRS Public Rating Equivalent | credit_quality | dimension | point_in_time | credit_quality | portfolio_comparison, monitoring | context_dependent | medium | DBRS public-rating equivalent; external credit-quality scale. |
| `down_payment_amount` | Down Payment Amount | leverage | measure | static_baseline | leverage, collateral | portfolio_comparison | higher_is_better | medium | Down payment amount; borrower stake reducing effective leverage. |
| `drawdown_facility` | Drawdown Facility | exposure | measure | point_in_time | exposure | period_change, portfolio_comparison | neutral | medium | Facility drawdown amount; exposure origination measure. |
| `duration_of_extension_option` | Duration Of Extension Option | maturity | measure | point_in_time | maturity | portfolio_comparison | context_dependent | medium | Duration of the extension option; potential maturity extension. |
| `ead_bucket` | EAD Bucket | exposure | dimension | point_in_time | exposure, concentration | period_change, portfolio_comparison, monitoring | neutral | high | Banded EAD stratification dimension defined at the analytics layer (config/mi/buckets.yaml). |
| `early_repayment_charge` | Early Repayment Charge | cashflow | measure | point_in_time | cashflow, pricing | portfolio_comparison | context_dependent | medium | Early repayment charge applies (MI extended); prepayment protection share. |
| `earnings_before_interest_taxes_depreciation_and_amortisation_ebitda` | Earnings Before Interest Taxes Depreciation And Amortisation EBITDA | credit_quality | measure | point_in_time | obligor_financials, income | portfolio_comparison | higher_is_better | medium | Obligor EBITDA (SME financial statements). |
| `earnings_before_interest_taxes_ebit` | Earnings Before Interest Taxes EBIT | credit_quality | measure | point_in_time | obligor_financials, income | portfolio_comparison | higher_is_better | medium | Obligor EBIT (SME financial statements). |
| `ebitda` | EBITDA | credit_quality | measure | point_in_time | obligor_financials, income | period_change, portfolio_comparison | higher_is_better | medium | Obligor EBITDA; core debt-service capacity input. |
| `economic_occupancy_at_securitisation` | Economic Occupancy At Securitisation | operational_performance | measure | static_baseline | operational_performance, income | portfolio_comparison | higher_is_better | medium | Economic occupancy at securitisation (CRE); income-generation baseline. |
| `employment_status` | Employment Status | credit_quality | dimension | point_in_time | credit_quality, affordability | portfolio_comparison | neutral | medium | Borrower employment status (governed enum, MI extended); income stability mix. |
| `energy_performance_certificate_value` | Energy Performance Certificate Value | collateral | dimension | point_in_time | collateral, eligibility | portfolio_comparison | context_dependent | medium | EPC rating (governed enum); collateral energy-efficiency mix with eligibility relevance. |
| `enterprise_size` | Enterprise Size | credit_quality | dimension | point_in_time | concentration, obligor_financials | portfolio_comparison | neutral | medium | Obligor enterprise-size classification (ESMA enum); size-mix dimension for SME books. |
| `enterprise_value` | Enterprise Value | valuation | measure | point_in_time | valuation, obligor_financials | portfolio_comparison | higher_is_better | medium | Obligor enterprise value (SME); valuation of the obligor business. |
| `equity` | Equity | collateral | measure | point_in_time | collateral, obligor_financials, leverage | period_change, portfolio_comparison | higher_is_better | medium | Equity cushion; the MI layer defines this as borrower equity in the property while the registry types it as an SME financials field — see the uncertain-entries review. |
| `erm_product_type` | ERM Product Type | product_mix | dimension | point_in_time | concentration, product_mix | period_change, portfolio_comparison, monitoring | neutral | high | Equity-release product type (MI core); product-mix dimension monitored by the risk monitor. |
| `erm_sub_product_type` | ERM Sub Product Type | product_mix | dimension | point_in_time | concentration, product_mix | portfolio_comparison | neutral | medium | Equity-release sub-product type (MI core); product-variant mix. |
| `expected_timing_of_recoveries` | Expected Timing Of Recoveries | forecast | measure | point_in_time | forecast, recovery, loss | monitoring | lower_is_better | medium | Expected timing of recoveries (ESMA); forward-looking recovery horizon. |
| `exposure_at_default` | Exposure At Default | exposure | measure | point_in_time | exposure, risk, credit_quality | period_change, portfolio_comparison, ranking, monitoring | higher_is_worse | high | EAD from the risk model layer (Phase 0B); expected exposure amount at default, curated in the MI risk registry. |
| `exposure_currency_denomination` | Exposure Currency Denomination | exposure | dimension | point_in_time | concentration, risk | portfolio_comparison | neutral | medium | Exposure currency (core canonical); currency-mix and FX-risk dimension. |
| `financial_expenses` | Financial Expenses | credit_quality | measure | point_in_time | obligor_financials | portfolio_comparison | context_dependent | medium | Obligor financial expenses (SME financial statements); debt-cost burden. |
| `finished_property` | Finished Property | collateral | measure | point_in_time | collateral, risk | portfolio_comparison | higher_is_better | medium | Whether the property is finished; development/completion risk share. |
| `fitch_public_rating_equivalent` | Fitch Public Rating Equivalent | credit_quality | dimension | point_in_time | credit_quality | portfolio_comparison, monitoring | context_dependent | medium | Fitch public-rating equivalent; external credit-quality scale. |
| `foreclosure_cost` | Foreclosure Cost | loss | measure | point_in_time | loss, recovery | portfolio_comparison | higher_is_worse | medium | Foreclosure costs; workout cost reducing net recovery. |
| `free_cashflow` | Free Cashflow | cashflow | measure | point_in_time | cashflow, obligor_financials | portfolio_comparison | higher_is_better | medium | Obligor free cashflow; debt-service capacity measure. |
| `further_advance_amount` | Further Advance Amount | exposure | measure | point_in_time | exposure, cashflow | period_change, portfolio_comparison | neutral | medium | Amount of further advances taken; incremental exposure on existing loans. |
| `further_advance_flag` | Further Advance Flag | exposure | measure | point_in_time | exposure | period_change, portfolio_comparison | neutral | medium | Whether a further advance has been taken (MI-curated flag); share of book with incremental borrowing. |
| `geographic_region_collateral` | Geographic Region Collateral | geography | dimension | point_in_time | concentration, geography, collateral | period_change, portfolio_comparison, monitoring | neutral | high | Collateral NUTS3 region (MI extended); geographic concentration dimension. |
| `geographic_region_obligor` | Geographic Region Obligor | geography | dimension | point_in_time | concentration, geography | period_change, portfolio_comparison, monitoring | neutral | high | Obligor NUTS3 region (MI core); geographic concentration dimension monitored by the risk monitor. |
| `guarantee_type` | Guarantee Type | collateral | dimension | point_in_time | collateral, credit_quality | portfolio_comparison | neutral | medium | Guarantee type; form of credit support mix. |
| `guarantor_type` | Guarantor Type | collateral | dimension | point_in_time | collateral, credit_quality | portfolio_comparison | neutral | medium | Guarantor type (governed enum); guarantee-support mix. |
| `health_impairment_flag` | Health Impairment Flag | eligibility | measure | point_in_time | eligibility, tail_risk | portfolio_comparison | context_dependent | medium | Health impairment flag; enhanced-terms eligibility and longevity assumption input. |
| `ifrs9_stage` | IFRS 9 Stage | credit_quality | dimension | point_in_time | credit_quality, loss | period_change, portfolio_comparison, monitoring | higher_is_worse | high | IFRS 9 impairment stage; ordered Stage 1 to Stage 3 per risk_monitor deterioration orderings. |
| `ifrs9_stage_current` | IFRS 9 Stage Current | credit_quality | derived_input | point_in_time | credit_quality, loss | period_change | higher_is_worse | medium | Current-snapshot IFRS 9 stage; input to stage migration matrix (risk_monitor migration pairs). |
| `ifrs9_stage_previous` | IFRS 9 Stage Previous | credit_quality | derived_input | point_in_time | credit_quality, loss | period_change | higher_is_worse | medium | Previous-snapshot IFRS 9 stage; input to stage migration matrix (risk_monitor migration pairs). |
| `income_expiring_13_24_months` | Income Expiring 13 24 Months | cashflow | measure | point_in_time | cashflow, income, maturity | portfolio_comparison | context_dependent | medium | Lease income expiring in 13-24 months; income rollover profile. |
| `income_expiring_1_12_months` | Income Expiring 1 12 Months | cashflow | measure | point_in_time | cashflow, income, maturity | portfolio_comparison, monitoring | context_dependent | medium | Lease income expiring within 12 months; near-term income rollover risk. |
| `income_expiring_25_36_months` | Income Expiring 25 36 Months | cashflow | measure | point_in_time | cashflow, income, maturity | portfolio_comparison | context_dependent | medium | Lease income expiring in 25-36 months; income rollover profile. |
| `income_expiring_37_48_months` | Income Expiring 37 48 Months | cashflow | measure | point_in_time | cashflow, income, maturity | portfolio_comparison | context_dependent | medium | Lease income expiring in 37-48 months; income rollover profile. |
| `income_expiring_49_months` | Income Expiring 49 Months | cashflow | measure | point_in_time | cashflow, income, maturity | portfolio_comparison | context_dependent | medium | Lease income expiring beyond 49 months; income rollover profile. |
| `indexed_loan_to_value` | Indexed Loan To Value | leverage | measure | point_in_time | leverage, collateral, risk | period_change, portfolio_comparison, ranking, monitoring | higher_is_worse | high | Indexed LTV (MI core); leverage against index-updated valuation. |
| `indexed_value` | Indexed Value | valuation | measure | point_in_time | valuation, collateral | period_change, portfolio_comparison, monitoring | higher_is_better | high | Valuation indexed to the current period (MI core). |
| `interest_arrears_amount` | Interest Arrears Amount | payment_performance | measure | point_in_time | payment_performance, delinquency | period_change, portfolio_comparison, ranking, monitoring | higher_is_worse | high | Interest amount in arrears (MI core). |
| `interest_collections_in_period` | Interest Collections In Period | cashflow | measure | period_flow | cashflow, income | period_change, portfolio_comparison | higher_is_better | high | Interest collected in the period; core cash income measure. |
| `interest_coverage_ratio_at_the_securitisation_date` | Interest Coverage Ratio At The Securitisation Date | coverage | measure | static_baseline | coverage, cashflow | portfolio_comparison | higher_is_better | medium | ICR at securitisation; static baseline for coverage comparison. |
| `interest_in_arrears` | Interest In Arrears | payment_performance | measure | point_in_time | payment_performance, delinquency | period_change, portfolio_comparison, monitoring | higher_is_worse | high | Whether the loan is in arrears (MI core flag); arrears incidence share of the book. |
| `interest_rate_at_the_securitisation_date` | Interest Rate At The Securitisation Date | pricing | measure | static_baseline | pricing | portfolio_comparison | neutral | medium | Interest rate at securitisation; static pricing baseline. |
| `interest_rate_cap` | Interest Rate Cap | pricing | measure | point_in_time | pricing, risk | portfolio_comparison | context_dependent | medium | Contractual interest rate cap; bounds borrower payment shock and lender yield. |
| `interest_rate_floor` | Interest Rate Floor | pricing | measure | point_in_time | pricing, risk | portfolio_comparison | context_dependent | medium | Contractual interest rate floor; bounds yield downside. |
| `interest_rate_reset_interval` | Interest Rate Reset Interval | pricing | measure | point_in_time | pricing, risk | portfolio_comparison | context_dependent | medium | Interest-rate reset interval; repricing frequency and rate-risk measure. |
| `interest_rate_type` | Interest Rate Type | pricing | dimension | point_in_time | pricing, concentration | period_change, portfolio_comparison, monitoring | context_dependent | high | Interest rate type (MI core enum); fixed/floating mix and rate risk profile. |
| `internal_risk_grade` | Internal Risk Grade | credit_quality | dimension | point_in_time | credit_quality, risk | period_change, portfolio_comparison, ranking, monitoring | higher_is_worse | high | Internal obligor risk grade (Phase 0B risk model); ordered A (best) to G (worst) per risk_monitor deterioration orderings. |
| `internal_risk_score` | Internal Risk Score | credit_quality | measure | point_in_time | credit_quality, risk | period_change, portfolio_comparison, monitoring | context_dependent | medium | Numeric internal/behavioural risk score; score direction is not defined in the registry so directionality is context-dependent. |
| `internal_risk_stage` | Internal Risk Stage | credit_quality | dimension | point_in_time | credit_quality, risk | period_change, portfolio_comparison, monitoring | context_dependent | high | Internal monitoring/watchlist stage (lender taxonomy, unordered per risk_monitor; distinct from IFRS 9). |
| `leveraged_transaction` | Leveraged Transaction | leverage | measure | point_in_time | leverage, credit_quality | portfolio_comparison | higher_is_worse | medium | Leveraged-transaction flag; share of highly leveraged exposures. |
| `lgd_bucket` | LGD Bucket | loss | dimension | point_in_time | loss, concentration | period_change, portfolio_comparison, monitoring | higher_is_worse | high | Banded LGD stratification dimension; ordered banding defined in risk_monitor deterioration orderings. |
| `lgd_current` | LGD Current | loss | derived_input | point_in_time | loss, credit_quality | period_change | higher_is_worse | medium | Current-snapshot LGD; input to LGD migration analysis. |
| `lgd_previous` | LGD Previous | loss | derived_input | point_in_time | loss, credit_quality | period_change | higher_is_worse | medium | Previous-snapshot LGD; input to LGD migration analysis. |
| `lien` | Lien | collateral | dimension | point_in_time | collateral, credit_quality | portfolio_comparison, monitoring | higher_is_worse | high | Lien/charge ranking (MI extended); first-charge positions carry stronger recovery claims than junior positions. |
| `liquidation_expense` | Liquidation Expense | loss | measure | point_in_time | loss, recovery | period_change, portfolio_comparison | higher_is_worse | medium | Liquidation expenses; costs reducing net recoveries. |
| `litigation` | Litigation | credit_quality | measure | point_in_time | credit_quality, risk | portfolio_comparison, monitoring | higher_is_worse | medium | Litigation flag; share of book subject to legal action. |
| `loan_hedged` | Loan Hedged | pricing | measure | point_in_time | pricing, risk | portfolio_comparison, monitoring | higher_is_better | medium | Whether the loan is hedged; hedged share of rate risk. |
| `loan_redemption_flag` | Loan Redemption Flag | cashflow | measure | point_in_time | cashflow | period_change, portfolio_comparison, monitoring | context_dependent | high | Whether the loan has redeemed (MI core); redemption/attrition incidence. |
| `long_term_debt` | Long Term Debt | leverage | measure | point_in_time | leverage, obligor_financials | portfolio_comparison | higher_is_worse | medium | Obligor long-term debt (SME financial statements). |
| `loss_given_default` | Loss Given Default | loss | measure | point_in_time | loss, credit_quality, collateral | period_change, portfolio_comparison, ranking, monitoring | higher_is_worse | high | Loss given default (0-1) from the Phase 0B risk model; loss severity assumption. |
| `ltv_cap` | LTV Cap | eligibility | measure | point_in_time | eligibility, leverage | portfolio_comparison | neutral | medium | Product LTV cap (equity release); maximum permitted leverage constraint. |
| `market_value` | Market Value | valuation | measure | point_in_time | valuation | period_change, portfolio_comparison | higher_is_better | high | Market value of the asset/collateral. |
| `maturity_date` | Maturity Date | maturity | dimension | point_in_time | maturity | portfolio_comparison, monitoring | neutral | high | Scheduled maturity date (MI core); maturity/refinancing profile with governed maturity-year cohorts. |
| `modification` | Modification | credit_quality | dimension | point_in_time | credit_quality, payment_performance | period_change, monitoring | context_dependent | medium | Loan modification type; forbearance and restructuring incidence. |
| `moody_s_public_rating_equivalent` | Moody's Public Rating Equivalent | credit_quality | dimension | point_in_time | credit_quality | portfolio_comparison, monitoring | context_dependent | medium | Moody's public-rating equivalent; external credit-quality scale. |
| `most_recent_capital_expenditure` | Most Recent Capital Expenditure | operational_performance | measure | point_in_time | operational_performance, cashflow | period_change | context_dependent | medium | Most recent capital expenditure (CRE); property investment level. |
| `most_recent_operating_expenses` | Most Recent Operating Expenses | operational_performance | measure | point_in_time | operational_performance, cashflow | period_change, portfolio_comparison | higher_is_worse | medium | Most recent operating expenses (CRE); property cost performance. |
| `most_recent_revenue` | Most Recent Revenue | credit_quality | measure | point_in_time | obligor_financials, income | period_change | higher_is_better | medium | Most recent reported obligor revenue; updated financial-strength measure. |
| `nace_industry_code` | NACE Industry Code | credit_quality | dimension | point_in_time | concentration | portfolio_comparison, monitoring | neutral | medium | NACE industry code (governed enum); industry concentration dimension. |
| `negative_amortisation` | Negative Amortisation | exposure | measure | point_in_time | exposure, payment_performance | period_change, portfolio_comparison, monitoring | higher_is_worse | medium | Negative amortisation amount; balance growth from unpaid interest capitalisation. |
| `negative_equity_guarantee` | Negative Equity Guarantee | tail_risk | measure | point_in_time | tail_risk, collateral, eligibility | portfolio_comparison, monitoring | higher_is_worse | high | No-negative-equity guarantee flag (MI core); NNEG share drives lender tail risk from collateral shortfall at redemption. |
| `net_internal_floor_area_validated` | Net Internal Floor Area Validated | data_quality | measure | point_in_time | data_quality, collateral | portfolio_comparison, monitoring | higher_is_better | medium | Whether the floor area was validated; collateral data-quality share. |
| `net_operating_income_at_securitisation` | Net Operating Income At Securitisation | cashflow | measure | static_baseline | cashflow, income | portfolio_comparison | higher_is_better | high | NOI at securitisation (MI extended); static income baseline for coverage comparison. |
| `net_proceeds_received_on_liquidation` | Net Proceeds Received On Liquidation | loss | measure | point_in_time | recovery, loss | period_change, portfolio_comparison | higher_is_better | medium | Net liquidation proceeds; realised recovery outcome. |
| `net_profit` | Net Profit | credit_quality | measure | point_in_time | obligor_financials, income | portfolio_comparison | higher_is_better | medium | Obligor net profit (SME financial statements). |
| `new_build` | New Build | collateral | measure | static_baseline | collateral | portfolio_comparison | context_dependent | medium | New-build flag (RRE); new-build share of the book. |
| `new_or_used` | New Or Used | collateral | dimension | static_baseline | collateral | portfolio_comparison | context_dependent | medium | New or used asset (governed enum); asset condition mix. |
| `non_recoverability_determined` | Non Recoverability Determined | loss | measure | point_in_time | loss | period_change, monitoring | higher_is_worse | medium | Determination that amounts are non-recoverable; write-off pipeline indicator. |
| `number_of_bedrooms` | Number Of Bedrooms | collateral | dimension | point_in_time | collateral | portfolio_comparison | neutral | medium | Number of bedrooms (MI extended); property-size mix dimension. |
| `number_of_days_in_arrears` | Number Of Days In Arrears | payment_performance | measure | point_in_time | payment_performance, delinquency | period_change, portfolio_comparison, ranking, monitoring | higher_is_worse | high | Days past due; standard delinquency severity measure with governed arrears buckets at the analytics layer. |
| `number_of_days_in_interest_arrears` | Number Of Days In Interest Arrears | payment_performance | measure | point_in_time | payment_performance, delinquency | period_change, portfolio_comparison, ranking, monitoring | higher_is_worse | high | Days in interest arrears (MI extended). |
| `number_of_days_in_principal_arrears` | Number Of Days In Principal Arrears | payment_performance | measure | point_in_time | payment_performance, delinquency | period_change, portfolio_comparison, ranking, monitoring | higher_is_worse | high | Days in principal arrears (MI extended). |
| `number_of_leased_objects` | Number Of Leased Objects | collateral | measure | point_in_time | collateral | portfolio_comparison | neutral | medium | Number of leased objects (equipment); collateral pool breadth per lease. |
| `number_of_payments_before_securitisation` | Number Of Payments Before Securitisation | maturity | measure | static_baseline | seasoning, maturity | portfolio_comparison | context_dependent | medium | Payments made before securitisation; seasoning measure. |
| `number_of_properties_at_data_cut_off_date` | Number Of Properties At Data Cut Off Date | collateral | measure | point_in_time | collateral, exposure | portfolio_comparison | neutral | medium | Number of properties securing the loan (MI extended); collateral pool breadth. |
| `number_of_units` | Number Of Units | collateral | measure | point_in_time | collateral | portfolio_comparison | neutral | medium | Number of units in the property; income-property scale measure. |
| `obligor_basel_iii_segment` | Obligor Basel Iii Segment | credit_quality | dimension | point_in_time | concentration, credit_quality | portfolio_comparison | neutral | medium | Obligor Basel III segment; regulatory risk-segmentation mix. |
| `obligor_reporting_breach` | Obligor Reporting Breach | credit_quality | measure | point_in_time | credit_quality, data_quality | monitoring | higher_is_worse | medium | Obligor breach of financial-reporting obligations; covenant-compliance signal (SME). |
| `occupancy_type` | Occupancy Type | collateral | dimension | point_in_time | collateral, concentration | portfolio_comparison | context_dependent | high | Occupancy type (MI core); owner-occupied vs rental mix. |
| `option_to_buy_price` | Option To Buy Price | valuation | measure | point_in_time | valuation, cashflow | portfolio_comparison | neutral | medium | Lessee option-to-buy price (equipment); residual-value realisation reference. |
| `original_interest_rate` | Original Interest Rate | pricing | measure | static_baseline | pricing | portfolio_comparison | neutral | medium | Interest rate at origination; static pricing baseline. |
| `original_loan_to_value` | Original Loan To Value | leverage | measure | static_baseline | leverage, collateral | portfolio_comparison, ranking | higher_is_worse | high | LTV at origination (MI core); static underwriting-leverage baseline. |
| `original_principal_balance` | Original Principal Balance | exposure | measure | static_baseline | exposure | portfolio_comparison, ranking | neutral | high | Principal at origination; static exposure baseline for comparison and vintage analysis (does not move period to period). |
| `original_residual_value_of_asset` | Original Residual Value Of Asset | valuation | measure | static_baseline | valuation, collateral | portfolio_comparison | neutral | medium | Original residual value of the leased asset (equipment); static baseline. |
| `original_term` | Original Term | maturity | measure | static_baseline | maturity | portfolio_comparison | neutral | high | Original loan term (MI extended) with governed term buckets. |
| `original_valuation_amount` | Original Valuation Amount | valuation | measure | static_baseline | valuation, collateral | portfolio_comparison | neutral | high | Valuation at origination (MI extended); static valuation baseline. |
| `origination_channel` | Origination Channel | origination | dimension | point_in_time | concentration | period_change, portfolio_comparison, monitoring | neutral | high | Origination channel (governed enum); channel concentration monitored by the risk monitor. |
| `origination_date` | Origination Date | maturity | dimension | static_baseline | maturity, seasoning | portfolio_comparison | neutral | high | Origination date (MI core); vintage cohort basis and seasoning input. |
| `originator_affiliate` | Originator Affiliate | eligibility | measure | point_in_time | eligibility | portfolio_comparison | context_dependent | medium | Whether the originator is an affiliate (MI extended); risk-retention/eligibility relevant share. |
| `originator_name` | Originator Name | origination | dimension | point_in_time | concentration | portfolio_comparison, monitoring | neutral | high | Originator (MI core); originator concentration dimension. |
| `other_public_rating` | Other Public Rating | credit_quality | dimension | point_in_time | credit_quality | portfolio_comparison | context_dependent | medium | Other public rating; external credit-quality scale. |
| `pari_passu_underlying_exposures` | Pari Passu Underlying Exposures | leverage | measure | point_in_time | leverage, exposure | portfolio_comparison | higher_is_worse | medium | Equal-ranking debt amount alongside the exposure; dilutes effective collateral coverage. |
| `payment_due` | Payment Due | payment_performance | measure | period_flow | payment_performance, cashflow | period_change | neutral | medium | Scheduled payment due in the period; denominator for collection performance. |
| `payment_in_kind` | Payment In Kind | payment_performance | measure | point_in_time | payment_performance, credit_quality | portfolio_comparison, monitoring | higher_is_worse | medium | Payment-in-kind flag; interest satisfied other than in cash, a credit-stress indicator. |
| `pd_bucket` | PD Bucket | credit_quality | dimension | point_in_time | credit_quality, concentration | period_change, portfolio_comparison, monitoring | higher_is_worse | high | Banded PD stratification dimension; ordered banding defined in risk_monitor deterioration orderings. |
| `pd_current` | PD Current | credit_quality | derived_input | point_in_time | credit_quality, risk | period_change | higher_is_worse | medium | Current-snapshot PD; input to PD migration and deterioration flags (risk_monitor). |
| `pd_previous` | PD Previous | credit_quality | derived_input | point_in_time | credit_quality, risk | period_change | higher_is_worse | medium | Previous-snapshot PD; input to PD migration and deterioration flags (risk_monitor). |
| `penalty_interest_balance` | Penalty Interest Balance | payment_performance | measure | point_in_time | payment_performance, cashflow | period_change, monitoring | higher_is_worse | medium | Accrued penalty interest balance; consequence of payment underperformance. |
| `penalty_interest_rate` | Penalty Interest Rate | pricing | measure | point_in_time | pricing | portfolio_comparison | neutral | medium | Penalty interest rate applied to amounts in default. |
| `physical_occupancy_at_securitisation` | Physical Occupancy At Securitisation | operational_performance | measure | static_baseline | operational_performance, income | portfolio_comparison | higher_is_better | medium | Physical occupancy at securitisation (CRE); utilisation baseline. |
| `postcode` | Postcode | geography | dimension | point_in_time | concentration, geography | portfolio_comparison | neutral | medium | Property postcode (MI extended); granular geographic drilldown dimension. |
| `prepayment_fee` | Prepayment Fee | cashflow | measure | period_flow | cashflow, pricing | period_change | neutral | medium | Prepayment fee amount; compensation income on early repayment. |
| `prepayment_interest_excess_shortfall` | Prepayment Interest Excess Shortfall | cashflow | measure | period_flow | cashflow | period_change | context_dependent | medium | Prepayment interest excess/shortfall; signed cashflow variance on prepayment. |
| `primary_income` | Primary Income | credit_quality | measure | point_in_time | affordability, credit_quality, income | portfolio_comparison | higher_is_better | medium | Primary borrower income (ESMA); affordability input. |
| `primary_income_verification` | Primary Income Verification | data_quality | dimension | point_in_time | data_quality, credit_quality, affordability | portfolio_comparison, monitoring | context_dependent | medium | Income verification basis (governed enum); underwriting evidence quality mix (e.g. verified vs self-certified). |
| `principal_advances_in_period` | Principal Advances In Period | cashflow | measure | period_flow | cashflow, exposure | period_change | neutral | high | Principal advanced in the period; new-lending cash outflow. |
| `principal_arrears_amount` | Principal Arrears Amount | payment_performance | measure | point_in_time | payment_performance, delinquency | period_change, portfolio_comparison, ranking, monitoring | higher_is_worse | high | Principal amount in arrears (MI core). |
| `prior_principal_balances` | Prior Principal Balances | exposure | derived_input | point_in_time | exposure | period_change | neutral | medium | Prior-period principal balance; the comparison basis for balance movement analysis. |
| `probability_of_default` | Probability Of Default | credit_quality | measure | point_in_time | credit_quality, risk | period_change, portfolio_comparison, ranking, monitoring | higher_is_worse | high | Probability of default (0-1) from the Phase 0B risk model; core credit-quality metric. |
| `product_type` | Product Type | product_mix | dimension | point_in_time | concentration, product_mix | portfolio_comparison, monitoring | neutral | high | Product type (governed enum); product-mix concentration dimension. |
| `property_leasehold_expiry` | Property Leasehold Expiry | collateral | dimension | point_in_time | collateral, maturity | portfolio_comparison, monitoring | lower_is_worse | medium | Leasehold expiry date (CRE); short unexpired terms erode collateral value. |
| `property_portfolio_value_at_securitisation_date` | Property Portfolio Value At Securitisation Date | valuation | measure | static_baseline | valuation, collateral | portfolio_comparison | neutral | medium | Property portfolio value at securitisation; static baseline. |
| `property_status` | Property Status | collateral | dimension | point_in_time | collateral | portfolio_comparison | context_dependent | medium | Property status (governed enum); collateral condition/status mix. |
| `property_type` | Property Type | collateral | dimension | point_in_time | collateral, concentration | portfolio_comparison, monitoring | neutral | high | Property type (governed enum); collateral mix dimension. |
| `protected_equity_flag` | Protected Equity Flag | collateral | measure | point_in_time | collateral, eligibility | portfolio_comparison | context_dependent | medium | Protected-equity flag (MI core, derived from the percentage); share of loans with ring-fenced borrower equity. |
| `protected_equity_percentage` | Protected Equity Percentage | collateral | measure | point_in_time | collateral, leverage | portfolio_comparison | context_dependent | medium | Protected-equity share of the property (MI core); reduces the collateral available to the lender. |
| `purchase_price` | Purchase Price | valuation | measure | static_baseline | valuation | portfolio_comparison | neutral | medium | Purchase price of the asset; acquisition valuation basis. |
| `purpose` | Purpose | credit_quality | dimension | point_in_time | concentration, credit_quality | portfolio_comparison | neutral | medium | Loan purpose (governed enum); purpose-mix dimension. |
| `real_estate_sale_price` | Real Estate Sale Price | loss | measure | point_in_time | loss, recovery, valuation | period_change, portfolio_comparison | higher_is_better | medium | Realised real-estate sale price; recovery outcome measure. |
| `reason_for_default_or_foreclosure` | Reason For Default Or Foreclosure | loss | dimension | point_in_time | loss, credit_quality | portfolio_comparison | neutral | medium | Reason for default or foreclosure (governed enum); default-driver mix. |
| `recourse` | Recourse | collateral | measure | point_in_time | collateral, credit_quality | portfolio_comparison | higher_is_better | medium | Recourse flag; share of exposures with recourse to the obligor. |
| `recoveries_in_period` | Recoveries In Period | loss | measure | period_flow | recovery, cashflow, loss | period_change, portfolio_comparison, monitoring | higher_is_better | high | Recoveries received in the period (MI core). |
| `recovery_source` | Recovery Source | loss | dimension | point_in_time | recovery, loss | portfolio_comparison | neutral | medium | Source of recoveries (governed enum); recovery-channel mix. |
| `redemptions_received_in_period` | Redemptions Received In Period | cashflow | measure | period_flow | cashflow | period_change, portfolio_comparison, monitoring | context_dependent | high | Redemption proceeds received in the period (MI core); attrition cashflow. |
| `revenue` | Revenue | credit_quality | measure | point_in_time | obligor_financials, credit_quality, income | period_change, portfolio_comparison | higher_is_better | medium | Obligor revenue; financial-strength input for corporate/SME/CRE credit assessment. |
| `risk_grade_current` | Risk Grade Current | credit_quality | derived_input | point_in_time | credit_quality, risk | period_change | higher_is_worse | medium | Current-snapshot risk grade; input to the risk-grade migration matrix (risk_monitor migration pairs). |
| `risk_grade_previous` | Risk Grade Previous | credit_quality | derived_input | point_in_time | credit_quality, risk | period_change | higher_is_worse | medium | Previous-snapshot risk grade; input to the risk-grade migration matrix (risk_monitor migration pairs). |
| `s_p_public_rating_equivalent` | S&P Public Rating Equivalent | credit_quality | dimension | point_in_time | credit_quality | portfolio_comparison, monitoring | context_dependent | medium | S&P public-rating equivalent; external credit-quality scale. |
| `sale_price` | Sale Price | loss | measure | point_in_time | loss, recovery, valuation | period_change, portfolio_comparison | higher_is_better | medium | Realised sale price of the asset; recovery outcome measure. |
| `secondary_income` | Secondary Income | credit_quality | measure | point_in_time | affordability, income | portfolio_comparison | higher_is_better | medium | Secondary borrower income (RRE); affordability input. |
| `securitised_residual_value` | Securitised Residual Value | exposure | measure | static_baseline | exposure, valuation | portfolio_comparison | context_dependent | medium | Residual value securitised (equipment); residual-value exposure. |
| `security_type` | Security Type | collateral | dimension | point_in_time | collateral | portfolio_comparison | neutral | medium | Security type (governed enum); form of security mix. |
| `seniority` | Seniority | credit_quality | dimension | point_in_time | credit_quality, collateral | portfolio_comparison | context_dependent | medium | Debt seniority ranking; claim priority mix of the book. |
| `servicer_name` | Servicer Name | operational_performance | dimension | point_in_time | concentration, operational_performance | portfolio_comparison | neutral | medium | Servicer; servicer concentration and operational-dependency dimension. |
| `servicer_watchlist_code` | Servicer Watchlist Code | credit_quality | dimension | point_in_time | credit_quality, operational_performance | period_change, monitoring | context_dependent | medium | Servicer watchlist code; servicer-flagged credit concern status. |
| `short_term_financial_debt` | Short Term Financial Debt | leverage | measure | point_in_time | leverage, liquidity, obligor_financials | portfolio_comparison | higher_is_worse | medium | Obligor short-term financial debt; near-term refinancing burden. |
| `source_portfolio_type` | Source Portfolio Type | origination | dimension | point_in_time | concentration | portfolio_comparison | neutral | medium | Direct vs acquired book (MI segmentation dimension); origination-source mix. |
| `special_scheme` | Special Scheme | eligibility | dimension | point_in_time | eligibility, concentration | portfolio_comparison | neutral | medium | Special scheme participation; government/support-scheme mix. |
| `special_servicing_status` | Special Servicing Status | credit_quality | measure | point_in_time | credit_quality, loss | period_change, portfolio_comparison, monitoring | higher_is_worse | medium | Whether the loan is in special servicing; distressed-management incidence. |
| `status_of_properties` | Status Of Properties | collateral | dimension | point_in_time | collateral | portfolio_comparison | context_dependent | medium | Status of properties (CRE enum); collateral status mix. |
| `stressed_LTV` | Stressed LTV | leverage | measure | point_in_time | leverage, tail_risk, collateral | portfolio_comparison, ranking, monitoring | higher_is_worse | high | Stressed LTV; leverage under a stressed valuation, a tail-risk measure. |
| `target_escrow_amounts_reserves` | Target Escrow Amounts Reserves | liquidity | measure | point_in_time | liquidity | monitoring | neutral | medium | Target escrow/reserve amounts (CRE); required reserve level for adequacy monitoring. |
| `tenure` | Tenure | collateral | dimension | point_in_time | collateral, concentration | portfolio_comparison | context_dependent | high | Property tenure (MI core, RRE); freehold/leasehold mix affecting collateral quality. |
| `total_credit_limit` | Total Credit Limit | exposure | measure | point_in_time | exposure | portfolio_comparison, monitoring | neutral | medium | Total committed credit limit; upper bound of potential exposure. |
| `total_debt` | Total Debt | leverage | measure | point_in_time | leverage, obligor_financials | portfolio_comparison | higher_is_worse | medium | Obligor total debt; leverage input. |
| `total_liabilities_excluding_equity` | Total Liabilities Excluding Equity | leverage | measure | point_in_time | leverage, obligor_financials | portfolio_comparison | higher_is_worse | medium | Obligor total liabilities excluding equity (SME financial statements). |
| `total_other_amounts_outstanding` | Total Other Amounts Outstanding | exposure | measure | point_in_time | exposure | period_change, portfolio_comparison | context_dependent | medium | Other amounts outstanding (fees, charges) beyond principal and interest; part of total exposure. |
| `total_proceeds_from_other_collateral_or_guarantees` | Total Proceeds From Other Collateral Or Guarantees | loss | measure | point_in_time | recovery, collateral | period_change, portfolio_comparison | higher_is_better | medium | Proceeds from other collateral or guarantees; recovery source measure. |
| `total_reserve_balance` | Total Reserve Balance | liquidity | measure | point_in_time | liquidity | period_change, monitoring | higher_is_better | medium | Total reserve balance; liquidity support available to the structure. |
| `total_scheduled_principal_interest_due` | Total Scheduled Principal Interest Due | cashflow | measure | period_flow | cashflow, payment_performance | period_change | neutral | medium | Total scheduled principal and interest due; scheduled cashflow expectation for the period. |
| `total_scheduled_principal_interest_paid` | Total Scheduled Principal Interest Paid | cashflow | measure | period_flow | cashflow, payment_performance | period_change, portfolio_comparison | higher_is_better | medium | Total scheduled principal and interest actually paid; collection performance versus schedule. |
| `total_shortfalls_in_principal_interest_outstanding` | Total Shortfalls In Principal Interest Outstanding | payment_performance | measure | point_in_time | payment_performance, delinquency, cashflow | period_change, portfolio_comparison, ranking, monitoring | higher_is_worse | high | Outstanding principal and interest shortfalls; unresolved payment underperformance. |
| `turnover_of_obligor` | Turnover Of Obligor | credit_quality | measure | point_in_time | obligor_financials, income | portfolio_comparison | higher_is_better | medium | Obligor turnover; business-scale and financial-strength measure. |
| `undrawn_facility` | Undrawn Facility | exposure | measure | point_in_time | exposure, liquidity | period_change, portfolio_comparison, monitoring | context_dependent | high | Undrawn facility amount; contingent exposure that can convert to drawn balance. |
| `unscheduled_principal_collections` | Unscheduled Principal Collections | cashflow | measure | period_flow | cashflow | period_change | context_dependent | medium | Unscheduled principal collected in the period (CRE); prepayment cashflow. |
| `vacant_possession_value_at_securitisation_date` | Vacant Possession Value At Securitisation Date | valuation | measure | static_baseline | valuation, collateral | portfolio_comparison | neutral | medium | Vacant-possession value at securitisation (CRE); conservative valuation basis. |
| `valuation_at_securitisation` | Valuation At Securitisation | valuation | measure | static_baseline | valuation, collateral | portfolio_comparison | neutral | medium | Collateral valuation at securitisation (CRE); static baseline. |
| `valuation_type` | Valuation Type | valuation | dimension | point_in_time | valuation, data_quality | portfolio_comparison | context_dependent | medium | Valuation type/basis (MI core, RRE); valuation reliability mix. |
| `weighted_average_lease_terms` | Weighted Average Lease Terms | maturity | measure | point_in_time | maturity, cashflow | portfolio_comparison | higher_is_better | medium | Weighted average lease term (equipment); contracted income duration. |
| `weighted_average_life` | Weighted Average Life | maturity | measure | point_in_time | maturity, cashflow | period_change, portfolio_comparison | context_dependent | high | Weighted average life; expected duration of the exposure. |
| `work_out_process_complete` | Work Out Process Complete | loss | measure | point_in_time | loss, operational_performance | period_change, monitoring | higher_is_better | medium | Whether the workout process is complete (SME); resolution progress share. |
| `workout_strategy_code` | Workout Strategy Code | loss | dimension | point_in_time | loss, operational_performance | monitoring | neutral | medium | Workout strategy code (CRE); resolution-approach mix. |
| `year_built` | Year Built | collateral | dimension | static_baseline | collateral | portfolio_comparison | neutral | medium | Year the property was built; property age profile. |
| `year_of_manufacture_construction` | Year Of Manufacture Construction | collateral | dimension | static_baseline | collateral, valuation | portfolio_comparison | neutral | medium | Year of manufacture/construction (equipment); asset age and depreciation profile. |
| `youngest_borrower_age` | Youngest Borrower Age | maturity | measure | point_in_time | maturity, tail_risk, risk | period_change, portfolio_comparison, monitoring | context_dependent | high | Age of the youngest borrower (MI core); drives expected loan duration and NNEG exposure on joint-life equity-release loans. |

<!-- END GENERATED INVENTORY -->

---

## 5. Uncertain entries (require human decision)

25 fields were **excluded pending review** (not written to the registry) and
1 included entry is **flagged ambiguous**. Per the brief, none of these were
forced into the registry.

<!-- BEGIN GENERATED UNCERTAIN -->

### `bank_internal_rating_prior_to_default` — Excluded pending review

- **Current definition:** analytics decimal, performance layer (SME)
- **Proposed classification:** credit_quality (pre-default rating, distribution)
- **Reason for uncertainty:** Niche loss-analysis field; rating scale and ordering are not defined in the registry.
- **Recommended human decision:** Confirm scale/ordering before enabling comparisons.

### `borrower_1_date_of_death` — Excluded pending review

- **Current definition:** analytics date (equity release)
- **Proposed classification:** cashflow (mortality/redemption event input)
- **Reason for uncertainty:** Event date, not a metric; mortality-experience analysis would consume a derived rate, which does not yet exist.
- **Recommended human decision:** Keep excluded until mortality-experience analytics are defined.

### `borrower_1_gender` — Excluded pending review

- **Current definition:** analytics string (equity release); MI core dimension
- **Proposed classification:** concentration (demographic mix for joint-life mortality analysis)
- **Reason for uncertainty:** Demographic attribute, not a metric; relevant to equity-release mortality/NNEG modelling but no governed mortality analytics exist yet.
- **Recommended human decision:** Include when mortality/longevity analytics are built.

### `borrower_2_date_of_death` — Excluded pending review

- **Current definition:** analytics date (equity release)
- **Proposed classification:** cashflow (mortality/redemption event input)
- **Reason for uncertainty:** Same as borrower_1_date_of_death.
- **Recommended human decision:** Keep excluded until mortality-experience analytics are defined.

### `borrower_2_gender` — Excluded pending review

- **Current definition:** analytics string (equity release); MI core dimension
- **Proposed classification:** concentration (demographic mix for joint-life mortality analysis)
- **Reason for uncertainty:** Same as borrower_1_gender.
- **Recommended human decision:** Include when mortality/longevity analytics are built.

### `borrower_2_income` — Excluded pending review

- **Current definition:** analytics decimal (equity release)
- **Proposed classification:** credit_quality (affordability, average)
- **Reason for uncertainty:** Income is not an underwriting driver for equity-release lifetime mortgages; analytical relevance unclear.
- **Recommended human decision:** Include only if affordability analysis is extended to ERM books.

### `borrower_deposit_amount` — Excluded pending review

- **Current definition:** analytics field, no format defined (SME)
- **Proposed classification:** leverage (borrower stake, sum)
- **Reason for uncertainty:** No format defined; overlaps deposit_amount / down_payment_amount.
- **Recommended human decision:** Deduplicate against deposit_amount, then include one.

### `customer_segment` — Excluded pending review

- **Current definition:** analytics field, no format defined (SME)
- **Proposed classification:** concentration (segment mix, distribution)
- **Reason for uncertainty:** No format or value set defined in the registry.
- **Recommended human decision:** Define the segment taxonomy, then include as a mix dimension.

### `date_of_financials` — Excluded pending review

- **Current definition:** regulatory date (ESMA)
- **Proposed classification:** data_quality (financials staleness input)
- **Reason for uncertainty:** Raw date; the analytical concept is a derived staleness measure that does not yet exist.
- **Recommended human decision:** Define a governed staleness metric if financials recency monitoring is required.

### `final_margin` — Excluded pending review

- **Current definition:** analytics field, no format defined
- **Proposed classification:** pricing (post-revision margin, weighted_average)
- **Reason for uncertainty:** No format or definition distinguishing it from current_interest_rate_margin.
- **Recommended human decision:** Confirm definition vs current margin, then include or retire.

### `ground_rent_payable` — Excluded pending review

- **Current definition:** regulatory decimal (ESMA)
- **Proposed classification:** cashflow (leasehold cost burden, sum)
- **Reason for uncertainty:** Analytically relevant for UK leasehold risk but no existing MI usage or derived metric to anchor it.
- **Recommended human decision:** Include if leasehold-cost analysis is required.

### `interest_reset_period` — Excluded pending review

- **Current definition:** analytics field, no format defined
- **Proposed classification:** pricing (repricing frequency, average)
- **Reason for uncertainty:** Duplicates interest_rate_reset_interval without a defined format.
- **Recommended human decision:** Retire in favour of interest_rate_reset_interval or define separately.

### `loan_entered_arrears` — Excluded pending review

- **Current definition:** analytics decimal, performance layer
- **Proposed classification:** payment_performance (arrears-entry incidence, share)
- **Reason for uncertainty:** Decimal type for what reads as an event/flag; semantics (count? flag? amount?) are not defined.
- **Recommended human decision:** Clarify whether this is a flag, count or amount, then classify.

### `maximum_balance` — Excluded pending review

- **Current definition:** analytics field, no format defined in the registry
- **Proposed classification:** exposure (facility maximum balance, sum)
- **Reason for uncertainty:** No format or business definition in the canonical registry; overlaps total_credit_limit.
- **Recommended human decision:** Confirm definition vs total_credit_limit, then include or retire.

### `most_recent_financials_as_of_end_date` — Excluded pending review

- **Current definition:** regulatory date (ESMA)
- **Proposed classification:** data_quality (financials staleness input)
- **Reason for uncertainty:** Same as date_of_financials.
- **Recommended human decision:** Define a governed staleness metric if required.

### `number_of_collateral_items_securing_the_loan` — Excluded pending review

- **Current definition:** analytics field, no format defined
- **Proposed classification:** collateral (pool breadth, average)
- **Reason for uncertainty:** No format defined; overlaps number_of_properties_at_data_cut_off_date.
- **Recommended human decision:** Deduplicate against the properties count, then include one.

### `number_of_employees` — Excluded pending review

- **Current definition:** analytics decimal, SME obligor size
- **Proposed classification:** credit_quality (obligor size, average)
- **Reason for uncertainty:** Firm-size fact with weak direct analytical use; enterprise_size already provides the governed size mix.
- **Recommended human decision:** Include only if headcount-based analysis is required.

### `prior_balances` — Excluded pending review

- **Current definition:** analytics field, no format defined
- **Proposed classification:** exposure (prior-period balance, sum)
- **Reason for uncertainty:** No format defined; prior_principal_balances (typed) covers the same concept.
- **Recommended human decision:** Retire in favour of prior_principal_balances.

### `regular_interest_instalment` — Excluded pending review

- **Current definition:** analytics field, no format defined
- **Proposed classification:** cashflow (scheduled instalment, sum)
- **Reason for uncertainty:** No format defined; overlaps total_scheduled_principal_interest_due.
- **Recommended human decision:** Confirm definition, then include or retire.

### `regular_principal_instalment` — Excluded pending review

- **Current definition:** analytics field, no format defined
- **Proposed classification:** cashflow (scheduled instalment, sum)
- **Reason for uncertainty:** No format defined; overlaps total_scheduled_principal_interest_due.
- **Recommended human decision:** Confirm definition, then include or retire.

### `rent_payable` — Excluded pending review

- **Current definition:** regulatory decimal (ESMA)
- **Proposed classification:** cashflow (rental obligation or income; unclear)
- **Reason for uncertainty:** Direction of the rent (payable by borrower vs received) is ambiguous across annex contexts.
- **Recommended human decision:** Confirm whether this is an income or a cost field per annex, then classify.

### `securitised_loan_amount` — Excluded pending review

- **Current definition:** analytics field, no format defined
- **Proposed classification:** exposure (securitised amount, sum)
- **Reason for uncertainty:** No format defined; overlaps current/original principal balances at securitisation.
- **Recommended human decision:** Confirm definition, then include or retire.

### `total_credit_limit_used` — Excluded pending review

- **Current definition:** analytics field, no format defined
- **Proposed classification:** exposure (limit utilisation input, sum)
- **Reason for uncertainty:** No format defined; utilisation is better expressed as a derived ratio of drawn/limit which does not yet exist.
- **Recommended human decision:** Define a governed utilisation metric instead of the raw field.

### `unconditional_corporate_third_party_guarantee_amount` — Excluded pending review

- **Current definition:** analytics field, no format defined (SME)
- **Proposed classification:** collateral (guarantee support amount, sum)
- **Reason for uncertainty:** No format defined in the registry.
- **Recommended human decision:** Confirm the field is populated/typed, then include.

### `unconditional_personal_guarantee_amount` — Excluded pending review

- **Current definition:** analytics field, no format defined (SME)
- **Proposed classification:** collateral (guarantee support amount, sum)
- **Reason for uncertainty:** No format defined in the registry.
- **Recommended human decision:** Confirm the field is populated/typed, then include.

### `equity` — INCLUDED (flagged ambiguous)

- **Current definition:** analytics decimal, portfolio_type sme; MI layer describes it as borrower equity in the property
- **Proposed classification:** collateral (equity cushion, sum) — included with medium confidence
- **Reason for uncertainty:** The canonical registry types equity as an SME balance-sheet field while the MI semantics layer uses it as property equity for equity release.
- **Recommended human decision:** Confirm a single definition (or split into two fields).

<!-- END GENERATED UNCERTAIN -->

---

## 6. Coverage gaps

Analytical concepts that are underrepresented or absent in the canonical
registry (no fields were invented to fill them):

- **tail_risk** — only `negative_equity_guarantee` carries it as primary
  concept (with `stressed_LTV` and downturn LGD in categories). There is no
  canonical *NNEG exposure amount* field; NNEG exposure appears only as a
  computed metric in the analytics layer, not as a registry field.
- **forecast** — only `expected_timing_of_recoveries` exists at loan level.
  The forecast fields used by MI (`forecast_funded_balance`,
  `forecast_funding_probability`, `forecast_funding_date`) are *virtual*
  pipeline/state-layer dimensions in the MI semantics registry, not canonical
  registry fields, so they are out of scope here.
- **data_quality** — only two per-loan quality flags exist
  (`primary_income_verification`, `net_internal_floor_area_validated`).
  Portfolio-level data-quality metrics (completeness, validation pass rates)
  live in the validation/exception layer, not the field registry.
- **liquidity** — limited to escrow/reserve fields (CRE-centric). There is no
  borrower- or portfolio-level liquidity metric in the registry.
- **return** — the registry has pricing inputs (rates, margins) but no
  realised-return or yield field; portfolio yield is a derived analytic.
- **concentration** — concentration is representable only via mix dimensions
  (region, channel, product, industry). Share-of-balance concentration
  metrics (e.g. `regional_balance_share`) are computed by the risk monitor,
  not stored as fields.
- **eligibility** — thin (LTV cap, special scheme, health impairment,
  originator affiliate); there is no general criteria/eligibility flag set.
- **tenant concentration (CRE)** — only free-text `tenant_name` exists; no
  coded tenant dimension, so CRE tenant concentration cannot be classified.
- **payment performance ratios** — collection rate (paid vs due) and limit
  utilisation are natural derived metrics; the inputs exist
  (`total_scheduled_principal_interest_paid`/`_due`,
  `cumulative_drawn_amount`/`total_credit_limit`) but no governed derived
  metric is defined. (Defining them is out of scope for this task.)

---

## 7. Method, quality controls and boundaries

**Method.** Selection is an explicit curated allowlist in the generator
(`CURATION`), mirroring the repository's established pattern
(`mi_agent/build_mi_semantics_registry.py` v0.2 moved from broad rules to a
curated allowlist for exactly this reason). Each entry carries a one-line
rationale referencing what grounds it (MI curation tier, risk-monitor
monitored/migration dimensions and orderings, ESMA field meaning, registry
format/enum).

**Determinism & governance.** The generator is deterministic (no timestamps,
sorted keys/taxonomy) and **refuses to overwrite** the existing reviewed
output unless `--force` is passed; `--check` verifies the committed file
matches regeneration. Validation tests (33) cover: every registry field
exists in the source registry; no duplicate field keys; controlled-taxonomy
values only (concepts, roles, temporality, categories, workflow tags,
default aggregation, directionality, confidence, comparability, asset
applicability); every entry has a rationale and at least one
concept/category; representative excluded fields are absent; YAML loads;
generation is deterministic and byte-identical with the committed file;
overwrite protection works; plus the v2 assertions listed in section 8.

**Boundaries respected.** No reasoning workflows were built; the MI query
engine, calculation definitions, canonical transformations and onboarding
logic are untouched; no materiality thresholds, covenant limits or risk
weights were added; the source registry was not modified.

---

## 8. Version 2 schema amendment

A tightly scoped schema amendment applied before the registry freeze. The
included field set (242) and the v1 curation are unchanged except for the
concept re-homing below. Content version `0.2.0`, `schema_version: 2`.

### Migration summary

| Change | v1 | v2 |
|---|---|---|
| Versioning | `version` only | `schema_version: 2` (shape) + `version: 0.2.0` (content) |
| `analytical_role` | — (implicit in aggregation) | `measure` \| `dimension` \| `derived_input` \| `supporting_attribute` on every entry |
| `temporality` | — (implicit in names/tags) | `point_in_time` \| `period_flow` \| `cumulative` \| `static_baseline` on every entry |
| `aggregation_type` | single value | renamed **`default_aggregation`** (mechanical; same values; taxonomy key `default_aggregations`) |
| `weight_field` | — (MI convention implicit) | nullable; `current_outstanding_balance` on all 32 weighted-average entries (governed MI default weight) |
| `share_basis` | — | nullable; `count` on all 24 share entries (governed MI flag default); null otherwise |
| `comparable_across_portfolios` (bool, uniformly true) | replaced | **`portfolio_comparability`**: `comparable` \| `requires_scale_alignment` \| `within_asset_class_only` \| `not_comparable` |
| `concentration` as primary concept | 22 entries | removed from the concepts taxonomy; entries re-homed (below). Concentration suitability now follows from `analytical_role: dimension` + workflow tags; `concentration` remains a category |

New concepts added to the controlled list to receive the re-homed
dimensions: `geography`, `product_mix`, `origination`. The broader taxonomy
redesign (splitting `credit_quality`; promoting `obligor_financials`,
`recovery`, `prepayment`; removing the `risk`/`delinquency` categories;
unifying concepts/categories into one faceted vocabulary) is **deliberately
deferred**.

### Concentration re-homing (all 22 changed concepts)

| Fields | New concept |
|---|---|
| `geographic_region_obligor`, `geographic_region_collateral`, `collateral_geography`, `postcode`, `borrower_jurisdiction` | geography |
| `broker_channel`, `origination_channel`, `originator_name`, `source_portfolio_type` | origination |
| `product_type`, `erm_product_type`, `erm_sub_product_type`, `debt_type` | product_mix |
| `purpose`, `customer_type`, `enterprise_size`, `obligor_basel_iii_segment`, `borrower_basel_iii_segment`, `nace_industry_code` | credit_quality |
| `asset_type` | collateral |
| `exposure_currency_denomination` | exposure |
| `servicer_name` | operational_performance |

### v2 classification rules (mechanical, with reviewed override sets)

- **analytical_role** — `distribution` → `dimension`, everything else →
  `measure`; then overrides: the 10 snapshot-pair/prior-period fields
  (`pd/lgd/risk_grade/ifrs9_stage` previous+current pairs,
  `days_in_arrears_prior`, `prior_principal_balances`) → `derived_input`
  (they exist only to feed migration/movement computation per
  `config/mi/risk_monitor.yaml` and must not be reported as standalone
  metrics); four event/recency dates (`current_valuation_date`,
  `date_last_in_arrears`, `default_date`, `date_of_restructuring`) →
  `supporting_attribute`.
- **temporality** — names containing `_in_period` / `_in_current_period`
  plus an explicit flow list → `period_flow`; `cumulative_*` plus
  `allocated_losses` (ESMA losses-to-date) → `cumulative` (**must be
  differenced before period comparison**); origination/securitisation-
  anchored values and immutable physical facts (`original_*`,
  `*_securitisation*`, `origination_date`, purchase/deposit amounts,
  build/manufacture year, `new_build`, `new_or_used`) → `static_baseline`;
  otherwise `point_in_time`. Contractual product terms without an explicit
  origination anchor (`ltv_cap`, `negative_equity_guarantee`, rate
  caps/floors) deliberately stay `point_in_time` (current recorded terms)
  rather than over-claiming immutability.
- **portfolio_comparability** — `derived_input` fields → `not_comparable`;
  lender/servicer scales and originator vocabularies (internal grades,
  scores, stages, bank-internal ratings and LGD estimates, guarantor
  bank-internal PD, `servicer_watchlist_code`, plus label vocabularies
  `erm_product_type`, `erm_sub_product_type`, `broker_channel`,
  `guarantee_type`, `special_scheme`) → `requires_scale_alignment`;
  income-property lease/occupancy concepts typed `common` in the canonical
  registry (`contractual_annual_rental_income`, the five
  `income_expiring_*` buckets, both at-securitisation occupancy measures,
  `number_of_units`) → `within_asset_class_only`; everything else →
  `comparable`.

### Classifications made with less than full confidence

Flagged for review; none block workflow use:

- `allocated_losses` → `cumulative` on the ESMA "allocated losses to date"
  definition; if a client tape supplies period losses, this needs a
  per-source override.
- `further_advance_amount` and `negative_amortisation` are kept
  `point_in_time`; both could plausibly be cumulative-to-date depending on
  the source tape convention.
- `pd_previous` / `lgd_previous` carry `weight_field:
  current_outstanding_balance` per the single governed MI weighting
  convention; a true prior-period weighted average would weight by
  `prior_principal_balances`, but no such convention exists yet and none
  was invented (they are `derived_input`, so aggregate reporting on them
  is not expected).
- `share_basis: count` uniformly follows the MI flag default; risk
  workflows may later prefer balance-weighted shares for exposure flags
  (e.g. `negative_equity_guarantee`) — a per-workflow decision, not a
  registry default.

### v2 test additions

The suite (33 tests) now additionally proves: valid role / temporality /
comparability values on every entry; the 10 snapshot-pair fields are
`derived_input` and `not_comparable`; all `cumulative_*` fields and
`allocated_losses` are `cumulative`; all `original_*` / securitisation
baselines are `static_baseline` and carry no `period_change` tag; every
weighted average names an existing canonical weight field (and only
weighted averages do); every share metric has an explicit basis (`count`
or a canonical weight field); internal lender scales are
`requires_scale_alignment`; no entry uses `concentration` as its primary
concept and every former concentration dimension has
`analytical_role: dimension`; regeneration remains byte-identical with the
committed file.
