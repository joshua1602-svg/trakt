# synthetic_portfolios — non-equity-release asset-class test packs

Two raw synthetic **funded loan tapes** for asset classes Trakt was **not** built
for, used to measure how hardened the existing registries and onboarding
infrastructure are outside equity release. Trakt is an Equity Release Mortgage /
ESMA Annex 2 (residential real estate) platform.

| Pack | Asset class | ESMA home | Secured? |
|------|-------------|-----------|----------|
| [`auto_finance/`](auto_finance/) | UK auto HP/PCP | Annex 5 (Automobile) | Yes — motor vehicle |
| [`unsecured_consumer/`](unsecured_consumer/) | UK unsecured personal loans | Annex 6 (Consumer) | No |

Each pack is a raw lender extract: a funded loan tape, a current-period cashflow
report, and a (synthetic) warehouse funding agreement. Both tapes carry full
credit-risk coverage (PD / LGD / EAD / IFRS 9 stage / internal grade / internal
score / arrears / default / loss / recovery) and — for auto — vehicle collateral
(value, LTV, make/model/mileage/VIN). All data is fictional and deterministic.

## Files

```
synthetic_portfolios/
  generate_portfolios.py      # deterministic generator (fixed seed) for the CSVs
  HARDENING_FINDINGS.md       # the analysis / report
  auto_finance/
    auto_finance_funded_loan_tape.csv
    auto_finance_cashflow_report.csv
    warehouse_funding_agreement.md
    README.md                 # data dictionary + coverage
  unsecured_consumer/
    unsecured_consumer_funded_loan_tape.csv
    unsecured_consumer_cashflow_report.csv
    warehouse_funding_agreement.md
    README.md                 # data dictionary + coverage
```

## Reproduce

```bash
# (re)generate the tapes — deterministic, so output is stable
python synthetic_portfolios/generate_portfolios.py

# run each tape through the Agentic onboarding model and check the findings
python -m pytest tests/test_synthetic_nonequity_portfolios.py -v
```

To drive a tape through the model by hand:

```bash
python -m engine.onboarding_agent.cli \
  --input-dir synthetic_portfolios/auto_finance \
  --client-name MERIDIAN_AUTO \
  --output-dir out_auto \
  --registry config/system/fields_registry.yaml \
  --aliases-dir config/system \
  --mode mi_only          # or regulatory_mi
```

Inspect `out_auto/05c_mapping_trace.csv` (per-column mapped / out_of_scope /
unmapped), `out_auto/09_onboarding_run_summary.json` (review status +
`product_profile_resolution`), and `out_auto/17_domain_coverage.json`.

## Headline result

The base pipeline ingests, classifies and maps the **core loan economics and the
whole credit-risk block** for both never-seen asset classes with **no code
changes** — that part is hardened. The gaps are concentrated in asset
identification (auto is silently classified as *equity release* with confidence
1.0 — fail-open), the collateral model (no vehicle attributes), and the
regime/enum layer (no ESMA Annex 5/6; RRE codes are reached for on a vehicle
book). Full detail and recommendations in
[`HARDENING_FINDINGS.md`](HARDENING_FINDINGS.md).
