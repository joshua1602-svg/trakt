# Test results

## Backend

```
$ python -m pytest tests/operations_control/ -q
2 failed, 316 passed
```

**96 onboarding tests, all passing.** The 2 failures are pre-existing and
unrelated — `test_annex2_delivery.py::TestRealComponentsMiniGolden`, which fails
in the Annex 2 XML builder against the real XSD, identically before and after
this work.

### What the new tests prove

| Area | Tests | Notable |
|---|---|---|
| Catalogue | 6 | No client values anywhere in it; vocabularies match the modules that own them; every field declares who supplies it and what it belongs to; delivery-specific values are declared and not collected; **a future regime added to the governed declaration reaches the wizard with no code change** |
| Conditional requirements | 3 | A field is required only when its condition holds; an entity role drives its own requirements; an unparseable condition degrades to optional |
| **New-client onboarding** | 14 | A case starts blank with a generated reference; **no existing client needs selecting**; **no legacy configuration is read — proved by pointing the legacy paths at nothing**; a client is created only on activation; the configuration is built from the answers; **nothing is copied from any other client**; several portfolios; **source registrations exist before the first delivery**; a pipeline book is deliberate not assumed; regulatory scope is derived and never on a pipeline book; standing regime information is collected and generated; delivery-specific values are never written; the client index is generated; **activation is idempotent**; **generation is deterministic** |
| Entities | 4 | One entity holds several roles without duplication; references are generated; regulatory reporting requires an originator; a portfolio cannot point at a removed entity |
| Validation | 7 | Invalid identifier; **duplicate client identifier**; **duplicate portfolio identifier**; a regime the asset cannot support; **a regime that does not apply asks for nothing**; selecting a regime activates its requirements; no file paths in any message |
| Information requests | 5 | The checklist asks only what the client can answer; a request moves the case through its lifecycle; an outstanding request blocks approval; a response can supply the answers; unresolved questions are visible |
| Governance | 8 | **Approval writes no configuration**; activation requires approval; approval requires a reason; **withdrawal writes nothing and keeps the record**; a withdrawn case cannot be edited; an illegal transition is refused in operator wording; the case history records who/when/what/before/after/why; activation appends to the hash-chained audit; the preview shows generated identifiers and defaults; the preview reports what cannot be represented |
| Migration | 8 | **Not required for new-client onboarding**; populates the same generic model; legacy values become entities with roles; **an invalid legacy identifier surfaces as a migration issue**; **changes no active configuration**; every adopted value carries its origin; blocks onboarding does not own survive; an approved mapping is never reset |
| Amendments | 6 | Start from the version in force; a client with no configuration cannot be amended; approval creates a new version and keeps the old one; history shows before and after; an invalid amendment cannot activate; an amendment does not trip the collision check |
| Home queues | 3 | Cases appear in the queue matching their status; an activated client appears as active; a legacy client is offered for migration only |
| Resolver seam | 3 | A client without configuration uses the repository file; an activated client resolves with its generated configuration; **the generated configuration satisfies the regulatory preflight with no blockers** |
| **Inference and defaults** | 19 | The jurisdiction supplies the currency and the clock; **an unlisted jurisdiction does not guess a currency**; an answer the operator gave is never overwritten; the client identifier is proposed, whole words only, and avoids one already taken; portfolio identifiers follow the platform convention; a single entity owns every portfolio without asking; **a sample pack answers format, files and asset class**; **an ambiguous tape is left as a question**; the Annex 2 originator block is never asked for and tracks the entity it names; investor-report contacts come from the contacts step; the retention holder code comes from the role; the exposure type comes from the asset class; a UK book reports GBZZZ without being asked; conventions come from the asset class; **a declared default fills what nobody answered — including in the regime blocks**; **a default never displaces something better known**; **a case nobody has touched holds no defaults**; **a minimal case reaches a complete configuration from six answered steps** |
| API and tenancy | 6 | The information model is served; a case starts blank over HTTP; unauthenticated refused; an operator may work a case but not approve; **a case cannot be named after another tenant's client**; another tenant's case is 404; an illegal transition returns an operator-safe message |

## Frontend

```
$ npx tsc --noEmit
(clean)

$ npm test
Test Files  17 passed (17)
     Tests  111 passed (111)
```

**31 onboarding browser tests**, including:

- the home screen **leads with starting a new client, not importing one** — and
  the migration section is asserted to appear *after* the primary action in the
  document;
- a blank case opens with a generated reference and no client;
- **the wizard's questions come from the governed catalogue**, and
  system-generated values are not among them;
- one entity can hold several roles;
- management information is not offered as a choice;
- deliveries are derived and their expected location shown;
- **the client checklist excludes the operator's own decisions**;
- an information request can be raised from the checklist;
- everything that will be created is shown before anything is created — and the
  client is asserted still not onboarded at that point;
- **approve then activate**, with the client asserted *not* configured between
  the two;
- legacy migration pre-populates and flags the invalid identifier;
- amendments start from the version in force and produce a second version with
  the first still readable;
- an ordinary operator is refused approval;
- **an onboarding can be cancelled from any step** — the action is
  offered before the wizard is finished, asks why, says what is *not*
  being removed, and disappears once the client is live.

### Mobile at 390px

Two browser tests assert the layout properties, and the screenshot script
asserts the real thing:

```
13_mobile_home:  horizontal overflow = 0px
14_mobile_case:  horizontal overflow = 0px
15b_cancel_dialog: horizontal overflow = 0px
```

Measured as `documentElement.scrollWidth - clientWidth` in a real 390×844
viewport. The step rail scrolls rather than overflowing; long generated values
(storage paths, identifiers) wrap with `break-all`.

## Whole repository — baseline comparison

The repository carries a large pre-existing failure population, so a pass count
proves nothing on its own. Both suites were run and the failure **sets** compared:

```
$ python -m pytest tests/ -q --tb=no          # baseline a945c0f (before this work)
165 failed, 3021 passed, 32 skipped, 10 errors

$ python -m pytest tests/ -q --tb=no          # the first implementation
165 failed, 3080 passed, 32 skipped, 10 errors

$ comm -13 baseline_failures branch_failures  # regressions
(empty)
```

The 175 failing/erroring tests were **identical sets**. That comparison covered
the first implementation; the corrected implementation changes the same
subsystem and its own suite is green apart from the two pre-existing Annex 2 XSD
failures, which are in the same state on the baseline.

**The repository is not green, and this work does not make it green.** The 165
pre-existing failures are outside this subsystem — a traced example is
`tests/test_delivery_xml_agent_review.py`, which errors with
`KeyError: 'preview_policy'` because `config/delivery/xml_preview_policy.yaml`
does not carry that key. No commit in this work touches `config/delivery/`.

## Screenshots

`docs/screenshots/client_onboarding/`, captured from the running UI.

| File | Screen |
|---|---|
| `01_onboarding_home.png` | Onboarding home — blank start leads |
| `02_blank_new_client_start.png` | A blank new-client case |
| `02b_inferred_from_jurisdiction.png` | Currency, time zone and identifier filled in from the country, each showing where it came from |
| `03_entities_step.png` | Entities, one entity with three roles |
| `04_portfolios_step.png` | Portfolios |
| `05_reporting_and_regime.png` | Reporting products, with eligibility explained |
| `05b_expected_deliveries.png` | Expected deliveries |
| `05c_regime_configuration.png` | Regulatory information |
| `06_information_request_checklist.png` | The client information checklist |
| `07_validation_readiness.png` | Readiness, ready to approve |
| `08_configuration_preview.png` | Full configuration preview |
| `09_approval_confirmation.png` | Approved, awaiting activation |
| `10_active_client.png` | The activated client |
| `11_legacy_migration_entry.png` | Legacy migration entry |
| `12_amendment_entry.png` | Amendment entry |
| `15_cancel_onboarding.png` | Cancelling a case, with the reason required |
| `15b_cancel_onboarding_mobile.png` | The same at 390px |
| `13_mobile_home.png` | 390px home |
| `14_mobile_case.png` | 390px case |
