# Test results and remaining recommendations

## Backend

```
$ python -m pytest tests/operations_control/ -q
2 failed, 279 passed
```

The 279 include **59 new onboarding tests**. All 59 pass.

The 2 failures are **pre-existing on this branch and unrelated to this work** —
verified by stashing every change and re-running:

```
tests/operations_control/test_annex2_delivery.py::TestRealComponentsMiniGolden::test_normaliser_and_builder_pass_xsd
tests/operations_control/test_annex2_delivery.py::TestRealComponentsMiniGolden::test_builder_interventions_are_captured
```

Both fail with "The generated file did not pass the regulator's format check"
in the Annex 2 XML builder against the real XSD. They fail identically before
and after this change.

### What the new tests cover

| Area | Tests | Notable assertions |
|---|---|---|
| Governed vocabularies | 5 | Every option list matches its existing owner exactly; MI is derived, never chosen; Annex 12 enums resolve from the regime template; each product declares what it cannot represent |
| Validation | 6 | LEI shape; a pipeline book cannot carry regime scope (the intake's rule); a source for an unknown portfolio is refused; messages carry no file paths |
| Derivation | 3 | Sources follow from portfolios and reporting; no pipeline cadence means no pipeline registration; regime scope is not set when the asset cannot support it |
| Adoption | 6 | Current values arrive populated; cadences come from the source registry; products are read from what is configured; gaps are reported not guessed; every value carries its origin; the legacy files are byte-identical afterwards |
| Generation | 10 | The client configuration keeps its existing shape including the legacy `regime` key; generated files say they are generated; enrichment/transformations/IVSR triggers survive adoption; a new client inherits nothing; the UK geography override is written in its governed shape; an acquired book without a supplied curve gets no forecast treatment; **an approved mapping and pinned fingerprint are never reset**; planning writes nothing |
| The workflow | 5 | End-to-end creation; source registrations rebuild when portfolios change; an incomplete profile cannot be approved; approval requires a reason; review reports what cannot be represented |
| Versioning and audit | 6 | A change creates a new version and the old one is intact and marked superseded; history records who/when/what/before/after/why; the hash-chained audit stays intact; an unchanged artefact reports as unchanged |
| Client views | 5 | A legacy client is listed as such; the six tabs are populated; the source list is the live registry, not a copy |
| Resolver seam | 3 | A client without a profile still uses the repository file; an onboarded client resolves with its generated configuration; **the generated configuration satisfies the Annex 2 preflight with no blockers** |
| API and tenancy | 7 | Unauthenticated is refused; a full wizard runs over HTTP; an ordinary operator may draft but not approve; another tenant's client and draft are 404, not 403; a draft cannot be named after another tenant's client, and the refusal happens before the write |

## Frontend

```
$ npm test
Test Files  17 passed (17)
     Tests  93 passed (93)

$ npx tsc --noEmit
(clean)
```

93 include **13 new onboarding tests**: the home screen's adoption offer, the
seven-step rail, adopted values with their origins, MI not being offered as a
choice, regime questions in business terms with the reported code named, derived
source registrations, the review screen showing everything before anything is
written, approval refused until answers are complete, approval recording its
reason, the six editor tabs, immutable history with before and after, and an
ordinary operator being refused approval by the same rule the server applies.

## Screenshots

`docs/screenshots/client_onboarding/`, captured from the running UI:

| File | Screen |
|---|---|
| `01_client_onboarding_home.png` | Client Onboarding home |
| `02_new_client_wizard.png` | Wizard step 1, adopting an existing client |
| `03_regime_configuration.png` | Regime configuration, including "not held here" |
| `04_portfolios_step.png` | Portfolios — governed values, no free text |
| `05_source_registration.png` | Source registrations, derived |
| `06_review_and_confirmation.png` | Review, before answers are complete |
| `07_review_ready_to_approve.png` | Review, ready to approve |
| `08_existing_client_editor.png` | Existing-client editor |
| `09_existing_client_portfolios.png` | Portfolios tab |
| `10_existing_client_regimes.png` | Regimes tab |
| `11_existing_client_history.png` | History tab |

---

## Remaining architectural recommendations

Ordered by how much they matter.

### 1. The committed ERE LEI is invalid and blocks Annex 2 today

`config/client/config_client_ERM_UK.yaml:67` holds
`213800ABCDE123456701N202501` — 27 characters where `preflight.py:25` requires
20. Today's code blocks an ERE Annex 2 delivery on its own configuration.
Correct it as the first governed change through onboarding, so the fix carries
a reason and an author.

### 2. The legal entity name should stop doubling as a regime field

`defaults.originator_name` is simultaneously the client's legal name and Annex 2
RREL82, and the Annex 12 projector falls back to it for IVSS3/IVSS4. Onboarding
now asks for the legal entity name once and writes it to both places, which
solves the operator's problem but not the model's. The clean shape is a client
identity block that regime projections read *from*, with RREL82 declared as
"sourced from client identity" in the delivery rules.

### 3. `ASSET_MODEL` should move from Python into the governed asset package

`operations_control/configuration/packages.py:58` hard-codes the asset-to-regime
support table. Onboarding reads it rather than restating it, so a new asset
class is a one-line change — but it is still a code change and a deploy.
`config/asset/product_profiles.yaml` is already governed and already the right
home. Moving it makes "support a new asset class" an administrator action.

### 4. The source registry has no history

`SourceRegistry.upsert()` replaces a record outright. Onboarding compensates by
never clearing the fields a delivery has earned, but a wrong upsert from any
path is still unrecoverable. The registry deserves the same
version-and-supersede treatment the onboarding profiles and configuration
packages already have.

### 5. `portfolio.static_reporting_date` is delivery data in a standing file

`config/client/config_client_ERM_UK.yaml:13` carries a reporting date in the
master client configuration, and `loan_engine.reporting_date` repeats it. Both
are delivery-specific; the workflow already supplies the reporting period.
Onboarding does not write either. They should be removed once nothing reads
them, which needs a check across `trakt_run.py` and the loan engine.

### 6. Annex 12 trigger and cashflow blocks need a governed model

IVSR1-IVSR10 and IVSF1-IVSF6 are currently hard-coded per deal as forced
XML-structure values. Generation preserves them because they are deal structure,
not client standing information — but "preserved because we do not understand
them yet" is a holding position, not a design. They need either a deal-structure
model or a genuine derivation from the tape.

### 7. A second client would expose the single-portfolio `portfolio:` block

The legacy `portfolio:` block is single-valued and describes the client's
dominant book. Onboarding populates it from the first portfolio and puts the
real per-portfolio detail in the portfolio registry, which is correct and
compatible. But a client whose books differ in jurisdiction or currency has no
way to express that in the legacy shape. The portfolio registry should become
the authority for those two fields, with the client block retained as a default.

### 8. Onboarding-time contact details are not yet used

The reporting and operational contacts onboarding now collects are written to
the client configuration but nothing reads them. Wiring them into publication
notification would close the loop, and is the obvious next increment.
