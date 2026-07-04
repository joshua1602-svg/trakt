# Unsecured Consumer — synthetic funded loan tape

Fully synthetic UK **unsecured consumer** portfolio (fixed-rate amortising
personal loans, debt-consolidation loans and retail point-of-sale finance).
There is **no collateral** — recourse is to the obligor only. Regulated-
securitisation home would be **ESMA Annex 6 (Consumer)**. It exists to test how
the Trakt registries and onboarding infrastructure — built for Equity Release /
ESMA Annex 2 — behave for an unsecured, collateral-free asset class.

- Originator: *Northwind Consumer Finance Ltd* (synthetic)
- Reporting / data cut-off date: 2026-01-31
- 26 funded loans spanning IFRS 9 Stages 1–3, current / delinquent / charged-off.

Files (all synthetic):

| File | Onboarding role | Notes |
|------|-----------------|-------|
| `unsecured_consumer_funded_loan_tape.csv` | `current_loan_report` | The raw funded loan tape (39 columns). |
| `unsecured_consumer_cashflow_report.csv` | `cashflow_report` | Current-period scheduled vs actual cashflows. |
| `warehouse_funding_agreement.md` | `warehouse_agreement` | Facility terms (out of MI scope). |

## Coverage — funded loan tape columns

**Core loan economics (resolve to the `common` registry block):**
`Loan Identifier`, `Original Obligor Identifier`, `Data Cut Off Date`,
`Origination Date`, `Maturity Date`, `Original Term`, `Product Type`,
`Original Principal Balance`, `Current Principal Balance`,
`Current Interest Rate`, `Interest Rate Type`, `Payment Frequency`, `Purpose`,
`Origination Channel`, `Currency`.

**Performance / arrears / default (category `regulatory` — in scope only under
`regulatory_mi`):** `Account Status`, `Days Past Due`, `Arrears Balance`,
`Default Date`, `Default Amount`, `Allocated Losses`, `Cumulative Recoveries`.

**Credit-risk metrics (resolve to the `common` analytics risk block):**
`IFRS9 Stage`, `Internal Risk Grade`, `Credit Score` → `internal_risk_score`,
`Probability of Default`, `Loss Given Default`, `Exposure at Default`,
`Debt To Income Ratio` → `debt_to_income_ratio`.

**Borrower:** `Employment Status`, `Primary Income`, `Geographic Region`,
`Postcode` map; **`Borrower Age` leaks onto the equity-release field
`borrower_1_age`.**

**Unsecured / consumer-specific:** **`Secured / Unsecured`,
`Number of Dependents`, `Residential Status`, `Affordability Assessment Result`
and `Monthly Instalment` have no canonical home and remain unmapped.** There is
no collateral, valuation or LTV — which itself exercises the platform's
collateral/LTV-centric assumptions.

See [`../HARDENING_FINDINGS.md`](../HARDENING_FINDINGS.md) for the full analysis.
