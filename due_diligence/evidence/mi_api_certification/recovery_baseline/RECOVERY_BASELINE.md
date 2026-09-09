# RECOVERY_BASELINE — frozen 135 at harness 191fa31, application ea8c65b, live run 34323550096

| case | v | route(1st cand) | measure | agg | dataset | dims | predicates | temporal | live | failure class |
|---|---|---|---|---|---|---|---|---|---|---|
| C01 | v1 | generic | current_outstanding_balance | sum | funded | - | - | level | CORRECT | - |
| C01 | v2 | generic | - | count | funded | - | - | level | CORRECT | - |
| C01 | v3 | generic | current_outstanding_balance | sum | funded | - | - | level | CORRECT | - |
| C02 | v1 | generic | - | count | funded | - | - | level | CORRECT | - |
| C02 | v2 | generic | - | count | funded | - | - | level | CORRECT | - |
| C02 | v3 | generic | - | count | funded | - | - | level | CORRECT | - |
| C03 | v1 | generic | current_outstanding_balance | avg | funded | - | - | level | CORRECT | - |
| C03 | v2 | generic | - | count | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C03 | v3 | generic | current_outstanding_balance | avg | funded | - | - | level | CORRECT | - |
| C04 | v1 | generic | current_loan_to_value | weighted_avg | funded | - | - | level | CORRECT | - |
| C04 | v2 | generic | current_outstanding_balance | sum | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C04 | v3 | generic | current_loan_to_value | weighted_avg | funded | - | - | level | CORRECT | - |
| C05 | v1 | generic | current_outstanding_balance | sum | funded | collateral_geography | - | level | CORRECT | - |
| C05 | v2 | generic | current_outstanding_balance | sum | funded | collateral_geography | - | level | CORRECT | - |
| C05 | v3 | generic | - | count | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C06 | v1 | generic | current_outstanding_balance | sum | funded | ltv_bucket | - | level | CORRECT | - |
| C06 | v2 | generic | - | count | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C06 | v3 | generic | current_outstanding_balance | sum | funded | ltv_bucket | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C07 | v1 | generic | current_outstanding_balance | sum | funded | - | collateral_geography=Scotland | level | WRONG | WRONG |
| C07 | v2 | period_change_analysis | - | count | funded | - | - | movement | CORRECT_REFUSAL | - |
| C07 | v3 | generic | current_outstanding_balance | sum | funded | - | collateral_geography=Scotland | level | CORRECT_REFUSAL | - |
| C08 | v1 | generic | - | count | funded | erm_product_type | - | level | WRONG | WRONG |
| C08 | v2 | generic | - | count | funded | - | - | level | CORRECT_REFUSAL | - |
| C08 | v3 | generic | - | count | funded | erm_product_type | - | level | WRONG | WRONG |
| C09 | v1 | generic | current_outstanding_balance | sum | funded | - | - | level | CORRECT | - |
| C09 | v2 | generic | - | count | funded | - | - | level | CORRECT | - |
| C09 | v3 | generic | current_outstanding_balance | sum | funded | - | - | level | CORRECT | - |
| C10 | v1 | generic | current_outstanding_balance | sum | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C10 | v2 | generic | - | count | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C10 | v3 | generic | current_outstanding_balance | sum | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C11 | v1 | generic | current_loan_to_value | weighted_avg | funded | - | - | level | CORRECT | - |
| C11 | v2 | generic | current_loan_to_value | weighted_avg | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C11 | v3 | generic | current_interest_rate | weighted_avg | funded | - | - | level | WRONG | WRONG |
| C12 | v1 | geo_exposure | current_outstanding_balance | sum | funded | collateral_geography,geographic_region_collateral_itl3 | - | level | WRONG | HARNESS_ARTEFACT(E7/E8) |
| C12 | v2 | geo_exposure | current_outstanding_balance | sum | funded | geographic_region_collateral_itl3 | - | level | WRONG | HARNESS_ARTEFACT(E7/E8) |
| C12 | v3 | geo_exposure | current_outstanding_balance | sum | funded | geographic_region_collateral_itl3 | - | level | WRONG | HARNESS_ARTEFACT(E7/E8) |
| C13 | v1 | generic | current_outstanding_balance | sum | funded | geographic_region_obligor | - | level | UNSCOREABLE | UNSCOREABLE |
| C13 | v2 | generic | current_outstanding_balance | sum | funded | number_of_borrowers | - | level | UNSCOREABLE | UNSCOREABLE |
| C13 | v3 | generic | - | count | funded | - | - | level | UNSCOREABLE | UNSCOREABLE |
| C14 | v1 | geo_exposure | current_outstanding_balance | sum | funded | collateral_geography | - | level | CORRECT | - |
| C14 | v2 | geo_exposure | current_outstanding_balance | sum | funded | collateral_geography | - | level | CORRECT | - |
| C14 | v3 | generic | current_outstanding_balance | sum | funded | collateral_geography | - | level | CORRECT | - |
| C15 | v1 | portfolio_summary | - | count | funded | - | - | level | CORRECT | - |
| C15 | v2 | portfolio_summary | - | count | funded | - | - | level | CORRECT | - |
| C15 | v3 | portfolio_summary | - | count | funded | - | - | level | CORRECT | - |
| C16 | v1 | pipeline_summary | - | count | pipeline | - | - | level | CORRECT | - |
| C16 | v2 | pipeline_summary | - | count | pipeline | - | - | level | CORRECT | - |
| C16 | v3 | pipeline_summary | - | count | pipeline | - | - | level | CORRECT | - |
| C17 | v1 | generic | - | count | pipeline | - | - | level | CORRECT | - |
| C17 | v2 | generic | - | count | pipeline | - | - ‼live | level | INCORRECT_REFUSAL | FALSE_UNKNOWN_CATEGORY |
| C17 | v3 | generic | - | count | pipeline | - | - | level | CORRECT | - |
| C18 | v1 | portfolio_risk_comparison | - | count | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C18 | v2 | generic | - | count | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C18 | v3 | portfolio_risk_comparison | - | count | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C19 | v1 | generic | current_loan_to_value | weighted_avg | funded | source_portfolio_id | - | level | WRONG | WRONG |
| C19 | v2 | generic | - | count | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C19 | v3 | generic | - | count | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C20 | v1 | evolution | current_outstanding_balance | sum | funded | - | - | axis='over time' | CORRECT | - |
| C20 | v2 | period_change_analysis | current_outstanding_balance | sum | funded | - | - | axis='across reporting periods' movement | CORRECT | - |
| C20 | v3 | evolution | current_outstanding_balance | sum | funded | - | - | level | CORRECT | - |
| C21 | v1 | evolution | - | count | funded | - | - | axis='over time' | CORRECT | - |
| C21 | v2 | evolution | - | count | funded | - | - | axis='month by month' movement | CORRECT | - |
| C21 | v3 | evolution | - | count | funded | - | - | axis='across reporting periods' | CORRECT | - |
| C22 | v1 | evolution | current_loan_to_value | weighted_avg | funded | - | - | axis='over time' | WRONG | HARNESS_ARTEFACT(E7/E8) |
| C22 | v2 | evolution | - | count | funded | - | - | axis='across reporting periods' | WRONG | WRONG |
| C22 | v3 | evolution | current_loan_to_value | weighted_avg | funded | - | - | level | WRONG | HARNESS_ARTEFACT(E7/E8) |
| C23 | v1 | evolution | current_outstanding_balance | sum | funded | - | - | axis='over time' | CORRECT | - |
| C23 | v2 | period_change_analysis | current_outstanding_balance | sum | funded | - | - | axis='across reporting periods' movement | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C23 | v3 | evolution | current_outstanding_balance | sum | funded | - | - | axis='by month' | CORRECT | - |
| C24 | v1 | generic | current_outstanding_balance | sum | funded | - | - ‼prior | rel=current_vs_previous | INCORRECT_REFUSAL | FALSE_UNKNOWN_CATEGORY |
| C24 | v2 | generic | - | count | funded | - | - | span=the month/1 rel=month_on_month | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C24 | v3 | generic | current_outstanding_balance | sum | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C25 | v1 | evolution | current_outstanding_balance | sum | funded | - | collateral_geography=Scotland | axis='over time' | WRONG | WRONG |
| C25 | v2 | period_change_analysis | current_outstanding_balance | sum | funded | - | collateral_geography=Scotland | axis='across reporting periods' movement | WRONG | WRONG |
| C25 | v3 | evolution | current_outstanding_balance | sum | funded | - | collateral_geography=Scotland | level | WRONG | WRONG |
| C26 | v1 | period_change_analysis | current_outstanding_balance | sum | funded | - | - ‼prior | rel=current_vs_previous movement | INCORRECT_REFUSAL | FALSE_UNKNOWN_CATEGORY |
| C26 | v2 | generic | - | count | funded | - | - | span=the month/1 rel=month_on_month | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C26 | v3 | period_change_analysis | current_outstanding_balance | sum | funded | - | - | span=the month/1 rel=month_on_month axis='Month-on-month' movement | CORRECT | - |
| C27 | v1 | period_change_analysis | - | count | funded | - | - ‼prior | rel=current_vs_previous movement | INCORRECT_REFUSAL | FALSE_UNKNOWN_CATEGORY |
| C27 | v2 | analytical_composition | redemptions_received_in_period | sum | funded | - | - ‼new,churn | span=the month/1 rel=month_on_month | INCORRECT_REFUSAL | FALSE_UNKNOWN_CATEGORY |
| C27 | v3 | period_change_analysis | - | count | funded | - | - ‼exited | span=the month/1 rel=month_on_month axis='month on month' movement | INCORRECT_REFUSAL | FALSE_UNKNOWN_CATEGORY |
| C28 | v1 | funded_bridge | - | sum | funded | - | - | movement | WRONG | WRONG |
| C28 | v2 | funded_bridge | - | sum | funded | - | - | span=the month/1 rel=month_on_month movement | CORRECT | - |
| C28 | v3 | funded_bridge | - | sum | funded | - | - | movement | WRONG | WRONG |
| C29 | v1 | period_change_analysis | - | count | funded | - | - | movement | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C29 | v2 | period_change_analysis | current_outstanding_balance | sum | funded | collateral_geography | - ‼two | span=the last 3 months/3 movement | INCORRECT_REFUSAL | FALSE_UNKNOWN_CATEGORY |
| C29 | v3 | portfolio_summary | - | count | funded | - | - | movement | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C30 | v1 | generic | current_outstanding_balance | sum | funded | - | - | rel=current_vs_previous | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C30 | v2 | period_change_analysis | current_outstanding_balance | sum | funded | - | - | span=the month/1 rel=month_on_month movement cmp=['last month', 'latest'] | CORRECT | - |
| C30 | v3 | period_change_analysis | current_outstanding_balance | sum | funded | - | - | rel=current_vs_previous movement cmp=['prior period', 'latest'] | CORRECT | - |
| C31 | v1 | forecast_extrapolation | - | sum | funded | - | - | level | CORRECT | - |
| C31 | v2 | generic | - | count | funded | - | - | level | CORRECT | - |
| C31 | v3 | forecast_extrapolation | - | sum | funded | - | - | level | CORRECT | - |
| C32 | v1 | analytical_composition | forecast_funded_balance | sum | forecast | - | - | level | WRONG | WRONG |
| C32 | v2 | analytical_composition | current_outstanding_balance | sum | funded | - | - | level | WRONG | WRONG |
| C32 | v3 | generic | current_outstanding_balance | sum | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C33 | v1 | cohort_conversion | - | sum | funded | - | - | level | CORRECT | - |
| C33 | v2 | generic | - | count | pipeline | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C33 | v3 | cohort_conversion | - | sum | pipeline | - | - | level | CORRECT | - |
| C34 | v1 | evolution | current_outstanding_balance | sum | funded | - | - | axis='over time' | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C34 | v2 | evolution | - | count | funded | - | - | axis='by the year' | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C34 | v3 | generic | - | count | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C35 | v1 | scenario | - | sum | funded | - | - | movement | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C35 | v2 | generic | - | count | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C35 | v3 | forecast_extrapolation | - | sum | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C36 | v1 | risk_limits | - | count | funded | - | - | level | CORRECT_REFUSAL | - |
| C36 | v2 | risk_limits | - | count | funded | - | - | level | CORRECT_REFUSAL | - |
| C36 | v3 | risk_limits | - | count | funded | - | - | level | CORRECT_REFUSAL | - |
| C37 | v1 | risk_limits | - | count | funded | - | - | level | CORRECT_REFUSAL | - |
| C37 | v2 | generic | - | count | funded | - | - | level | CORRECT_REFUSAL | - |
| C37 | v3 | risk_limits | - | count | funded | - | - | level | CORRECT_REFUSAL | - |
| C38 | v1 | generic | current_outstanding_balance | sum | funded | collateral_geography | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C38 | v2 | geo_exposure | current_outstanding_balance | sum | funded | collateral_geography | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C38 | v3 | geo_exposure | current_outstanding_balance | sum | funded | collateral_geography | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C39 | v1 | generic | - | count | pipeline | pipeline_stage | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C39 | v2 | generic | current_interest_rate_margin | weighted_avg | pipeline | pipeline_stage | - ‼live | level | INCORRECT_REFUSAL | FALSE_UNKNOWN_CATEGORY |
| C39 | v3 | generic | - | count | pipeline | pipeline_stage | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C40 | v1 | pipeline_stage_movement | - | count | pipeline | - | - ‼kfi into application | movement | INCORRECT_REFUSAL | FALSE_UNKNOWN_CATEGORY |
| C40 | v2 | pipeline_stage_movement | - | count | pipeline | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C40 | v3 | pipeline_stage_movement | - | count | pipeline | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C41 | v1 | borrowing_base | - | count | funded | - | - | level | CORRECT_REFUSAL | - |
| C41 | v2 | generic | current_valuation_amount | sum | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C41 | v3 | borrowing_base | - | count | funded | - | - | level | CORRECT_REFUSAL | - |
| C42 | v1 | borrowing_base | - | count | funded | - | - | level | CORRECT_REFUSAL | - |
| C42 | v2 | generic | - | count | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C42 | v3 | borrowing_base | - | count | funded | - | - | level | CORRECT_REFUSAL | - |
| C43 | v1 | borrowing_base | - | count | funded | - | - | level | CORRECT_REFUSAL | - |
| C43 | v2 | generic | - | count | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C43 | v3 | borrowing_base | - | count | funded | - | - | level | CORRECT_REFUSAL | - |
| C44 | v1 | borrowing_base | - | count | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C44 | v2 | borrowing_base | - | count | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C44 | v3 | borrowing_base | - | count | funded | - | - | level | CORRECT_REFUSAL | - |
| C45 | v1 | generic | current_outstanding_balance | sum | funded | - | - | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
| C45 | v2 | generic | current_outstanding_balance | sum | funded | - | - ‼answer | level | INCORRECT_REFUSAL | FALSE_UNKNOWN_CATEGORY |
| C45 | v3 | generic | current_outstanding_balance | sum | funded | - | current_outstanding_balance=1000000000.0 | level | INCORRECT_REFUSAL | INCORRECT_REFUSAL |
