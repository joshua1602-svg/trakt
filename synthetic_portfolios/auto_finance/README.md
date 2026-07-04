# Auto Finance — synthetic funded loan tape

Fully synthetic UK **auto-finance** portfolio (Hire Purchase / Personal
Contract Purchase), secured on motor vehicles. Regulated-securitisation home
would be **ESMA Annex 5 (Automobile)**. It exists to test how the Trakt
registries and onboarding infrastructure — built for Equity Release / ESMA
Annex 2 — behave for an asset class they were not designed for.

- Originator: *Meridian Auto Finance Ltd* (synthetic)
- Reporting / data cut-off date: 2026-01-31
- 26 funded agreements spanning IFRS 9 Stages 1–3, performing / arrears /
  defaulted, HP and PCP (PCP carries a balloon / GFV).

Files (all synthetic):

| File | Onboarding role | Notes |
|------|-----------------|-------|
| `auto_finance_funded_loan_tape.csv` | `current_loan_report` | The raw funded loan tape (48 columns). |
| `auto_finance_cashflow_report.csv` | `cashflow_report` | Current-period scheduled vs actual cashflows. |
| `warehouse_funding_agreement.md` | `warehouse_agreement` | Facility terms (out of MI scope). |

## Coverage — funded loan tape columns

**Core loan economics (resolve to the `common` registry block):**
`Loan Identifier`, `Original Obligor Identifier`, `Data Cut Off Date`,
`Origination Date`, `Maturity Date`, `Original Term`,
`Original Principal Balance`, `Current Principal Balance`,
`Current Interest Rate`, `Interest Rate Type`, `Deposit Amount`,
`Payment Frequency`, `Purpose`, `Origination Channel`, `Currency`.

**Performance / arrears / default (category `regulatory` — in scope only under
`regulatory_mi`):** `Account Status`, `Days Past Due`, `Arrears Balance`,
`Default Date`, `Default Amount`, `Allocated Losses`, `Cumulative Recoveries`.

**Credit-risk metrics (resolve to the `common` analytics risk block):**
`IFRS9 Stage` → `ifrs9_stage`, `Internal Risk Grade` → `internal_risk_grade`,
`Credit Score` → `internal_risk_score`,
`Probability of Default` → `probability_of_default`,
`Loss Given Default` → `loss_given_default`,
`Exposure at Default` → `exposure_at_default`.

**Borrower:** `Employment Status`, `Primary Income`, `Geographic Region`,
`Postcode` map; **`Borrower Age` leaks onto the equity-release field
`borrower_1_age`** (there is no asset-neutral age field).

**Vehicle collateral:** `Collateral Type`, `New or Used`, valuations and LTV
map — but **`Vehicle Make`, `Vehicle Model`, `Vehicle Registration Year`,
`Mileage`, `Vehicle Identification Number`, `Fuel Type`, `Agreement Type`
(HP/PCP) and `Balloon Payment`/GFV have no canonical home and remain unmapped.**

See [`../HARDENING_FINDINGS.md`](../HARDENING_FINDINGS.md) for the full analysis.
