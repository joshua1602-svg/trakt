# Legacy configuration ↔ onboarding model

Every onboarding answer lands in an artefact that already exists. The mapping is
declared once, in `config/regime/onboarding_standing_fields.yaml`
(`writes_to:`), and used in **both directions** — generation writes through it,
adoption reads back through it — so a field can never be adopted from one place
and written to another.

---

## Step 1 — Client

| Onboarding field | Legacy home | Direction |
|---|---|---|
| Client identifier | `client.client_id` | Both |
| Client name | `client.display_name` | Both |
| Legal entity name | `client.legal_entity_name`, falling back to `defaults.originator_name` (RREL82) | Both |
| LEI | `defaults.originator_legal_entity_identifier` (RREL83) | Both |
| Jurisdiction | `portfolio.country` | Both |
| Reporting currency | `portfolio.base_currency` | Both |
| Time zone | `client.time_zone` | **New key** in the existing `client:` block |
| Environment | `client.environment` | Both |
| Primary reporting contact / email | `client.reporting_contact.{name,email}` | **New keys** |
| Operational contact / email | `client.operational_contact.{name,email}` | **New keys** |

The four contact keys are new. Nothing in the platform carried contacts before
(see findings §4); they extend the existing `client:` block rather than
introducing a file.

## Step 2 — Portfolio

| Onboarding field | Legacy home |
|---|---|
| Portfolio identifier | `SourceRecord.source_portfolio_id`; portfolio registry `source_portfolio_id` |
| Display name | portfolio registry `source_portfolio_label` |
| Portfolio type (direct / acquired) | `SourceRecord.source_portfolio_type` |
| Asset class | `portfolio.asset_class` (client config), validated against `ASSET_MODEL` |
| Warehouse / SPV / managed | **No legacy home.** Written to portfolio metadata as a new key. |
| Funded reporting cadence | `SourceRecord.frequency` where `dataset == funded` |
| Pipeline reporting cadence | `SourceRecord.frequency` where `dataset == pipeline` |
| Originates | portfolio registry `originates` (defaulted by type) |
| Pipeline data available | portfolio registry `pipeline_data_available` — **derived**, not asked |

Governed value lists, all resolved from existing owners
(`operations_control/onboarding/model.py:governed_vocabularies`):

| List | Owner |
|---|---|
| Asset classes and the regimes each supports | `operations_control.configuration.packages.ASSET_MODEL` |
| Portfolio types | `apps.blob_trigger_app.source_registry.SourceRecord` |
| Datasets | `operations_control.contracts.BATCH_DATASETS` |
| Cadences | `apps.blob_trigger_app.path_parser.VALID_FREQUENCIES` |
| Reporting products | `config/regime/onboarding_standing_fields.yaml` |

## Step 3 — Reporting

Derived, not re-decided. `derive_reporting()` reads the **same** asset support
model the effective-configuration resolver uses to accept or refuse a regime
outcome (`resolver.py:117-123`). MI is not offered as a choice at all, because
the outcome vocabulary has no MI-less option. The operator's answer sets
`pipeline.esma_enabled`, `supported_regimes`, `default_regime` and `regime` in
the client configuration.

## Step 4 — Regime configuration

| Onboarding field | Regime code | Writes to |
|---|---|---|
| Originator name | RREL82 | `client_config:defaults.originator_name` |
| Originator LEI | RREL83 | `client_config:defaults.originator_legal_entity_identifier` |
| Originator country | RREL84 | `client_config:defaults.originator_establishment_country` |
| Original lender LEI | RREL80 | `client_config:defaults.original_lender_legal_entity_identifier` |
| Original lender country | RREL81 | `client_config:defaults.original_lender_establishment_country` |
| NUTS classification year | — | `client_config:nuts_classification_year` |
| UK geography as GBZZZ | RREL11 / RREC6 | `client_config:regime_overrides.ESMA_Annex2.uk_geography.enabled` |
| Securitisation name | IVSS3 | `annex12_config:annex12.deal.IVSS3` |
| Reporting entity name | IVSS4 | `annex12_config:annex12.deal.IVSS4` |
| Contact person / telephone / email | IVSS5/6/7 | `annex12_config:annex12.deal.IVSS5/6/7` |
| Risk retention method | IVSS8 | `annex12_config:annex12.deal.IVSS8` |
| Risk retention holder | IVSS9 | `annex12_config:annex12.deal.IVSS9` |
| Underlying exposure type | IVSS10 | `annex12_config:annex12.deal.IVSS10` |
| Risk transfer method | IVSS11 | `annex12_config:annex12.deal.IVSS11` |
| Revolving / ramp-up end date | IVSS13 | `annex12_config:annex12.deal.IVSS13` |
| Excess spread trapping | IVSS20 | `annex12_config:annex12.deal.IVSS20` |
| Risk weight approach | IVSS30 | `annex12_config:annex12.deal.IVSS30` |

Annex 12 enumerations are read from `config/regime/annex12_template.yaml` at
render time, so the wizard cannot offer a value the projector would refuse.

## Step 5 — Investor / static reporting

| Onboarding field | Legacy home |
|---|---|
| Report title | `mi.branding.app_title` |
| Brand colour | `mi.branding.theme.primary_color` |
| Logo | `mi.branding.logo_uri` — **new key** |
| Disclaimer | `mi.branding.disclaimer` — **new key** |
| Payment convention | `loan_engine.interest_payment_frequency` |
| Day-count convention | `loan_engine.day_count_convention` |
| Investor contact / email, trustee, warehouse lender, reporting convention | **No legacy home.** Written to a new `reporting_parties:` block in the client configuration. |

## Step 6 — Source registration

Generated, never typed. `derive_sources()` produces one funded registration per
portfolio plus one pipeline registration where a pipeline cadence was given.
`regime_required` is set from the reporting answers, and only on a
regime-capable dataset — the same rule
(`contracts.REGIME_CAPABLE_DATASETS`) the intake enforces.

Generation is **additive on an existing record**: approved mapping id, pinned
schema fingerprint, role schemas, role aliases and last-successful markers are
carried through untouched, and a source already proven by a delivery is never
demoted back to `pending_review`. Onboarding owns the business facts (source
system, cadence, regime scope); the registry owns the delivery evidence.

The expected blob location is shown, derived from the layout the path parser
already enforces. It is never typed in.

---

## Fields that cannot currently be represented

Captured in the profile and shown at review, but **not written to any artefact**
— because inventing a home for them would be inventing configuration.

| Field | Why not |
|---|---|
| **Sponsor** | Annex 2 is the underlying-exposure report and carries no sponsor field. Sponsor exists only as an Annex 12 risk-retention holder *value* (IVSS9 = SPON). |
| **SSPE** | No SSPE field in the Annex 2 field universe. The securitisation entity is named in Annex 12 (IVSS3/IVSS4). |
| **National competent authority** | Trakt produces the report; it does not file it. No NCA field exists in the field universe or the delivery rules. |
| **STS status / notification identifier** | No STS fields in the current Annex 2 model. Adding them means extending `annex2_field_universe.yaml`, which is an administrator change to a governed package. |
| **Servicer (regulatory)** | `SourceRecord.source_system` records who supplies the data, which onboarding does capture. There is no regulatory servicer field in the Annex 2 model. |
| **Trustee** | Captured as static reporting information (it appears on investor report covers) and written to `reporting_parties:`, but Annex 12 has no trustee field, so nothing regulatory reads it. |
| **Issuer** | Same position as SSPE. |
| **Trigger definitions (IVSR1-IVSR10)** | Currently hard-coded per deal in the Annex 12 client config as forced XML-structure values. They are deal-structure information, not client standing information, and are left exactly where they are until a governed trigger model exists. Generation preserves them. |
| **Calculation conventions (as regime fields)** | Day-count and payment frequency are captured and written under `loan_engine:`, but they are not Annex 2 fields. |
| **Static pool cohort definitions** | Asset configuration, administered through the governed asset package, not per client. |

Each is declared in `config/regime/onboarding_standing_fields.yaml` under the
product's `unrepresented:` list with its reason, surfaced on the regime step and
again at review. Nothing is silently dropped.

---

## Why one new file was needed

`config/regime/onboarding_standing_fields.yaml` is the only configuration file
this work adds. It defines **no new fields**. Every field it names already
exists in `annex2_field_universe.yaml`, `annex2_delivery_rules.yaml`,
`annex12_template.yaml` or the client configuration.

What did not exist anywhere is the *declaration of which of those fields are
standing* rather than delivery-specific. Onboarding needs exactly that
distinction — only standing information belongs in a client's profile — and the
delivery rules do not carry it.

It is registered in `LAYER_FILES[LAYER_REGIME]`
(`operations_control/configuration/packages.py`), so it is versioned,
validated, activated and rolled back through the existing administrator area
like every other regime file. Adding a future regime to onboarding is a
configuration change, not a code change.
