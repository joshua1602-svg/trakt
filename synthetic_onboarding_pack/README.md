# Synthetic Onboarding Pack

This folder is a **synthetic lender data room** used to exercise the Trakt
Onboarding Agent v2 end-to-end. It contains no real counterparty data.

## Contents

| File | Artefact | Purpose |
|------|----------|---------|
| `monthly_loan_report.csv` | Current loan report | Loan-level balances, rates, dates |
| `monthly_cashflow_report.csv` | Cashflow report | Principal / interest / fee cashflows |
| `monthly_collateral_report.csv` | Collateral report | Property valuation, postcode, LTV |
| `monthly_pipeline_report.csv` | Pipeline report | Applications, brokers, expected funding |
| `warehouse_funding_agreement.md` | Warehouse agreement | Facility terms + eligibility (unstructured) |
| `synthetic_securitisation_summary.md` | Securitisation summary | Indicative term deal (unstructured) |
| `synthetic_data_dictionary.csv` | Data dictionary | Field descriptions |

## Deliberate test signals

The pack is intentionally constructed to exercise each onboarding stage:

- **Overlap:** `current_balance` (loan report) and `principal_outstanding`
  (cashflow report) are the same business field across two sources, keyed on
  `loan_id`.
- **Conflicting reporting dates:** loan/collateral reports use `2026-01-31`
  while the cashflow report uses `2026-02-01`.
- **Unresolved enum:** `employment_status` contains the value `manual`, which is
  not a valid canonical enum.
- **Warehouse config:** advance rate, margin, limit and lender name are present
  in the (unstructured) warehouse agreement only.

Run the onboarding agent against this pack:

```bash
python -m engine.onboarding_agent.cli \
  --input-dir synthetic_onboarding_pack \
  --client-name SYNTHETIC_ONBOARDING_TEST \
  --output-dir onboarding_output/synthetic_onboarding_test \
  --registry config/system/fields_registry.yaml \
  --aliases-dir config/system
```
