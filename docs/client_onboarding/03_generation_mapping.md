# From approved answers to Trakt configuration

Activation is the only place active configuration is written, and it writes only
the artefacts Trakt already reads.

## What a brand-new client gets

Verified end to end against a client Trakt had never seen, with no legacy files
present:

| Artefact | Path | Contents |
|---|---|---|
| Client configuration | `config/client/config_client_{CLIENT}.yaml` | `client:` identity and contacts, `portfolio:` jurisdiction and currency, `defaults:` originator values, `supported_regimes` / `default_regime` / `regime`, `pipeline:` switches, `mi.branding`, `loan_engine` conventions, `reporting_parties` |
| Investor report configuration | `config/client/config_client_{CLIENT}_annex12.yaml` | `annex12.deal` standing fields. Written only when investor reporting is selected. |
| Portfolio metadata | `config/client/portfolio_registry_{CLIENT}.yaml` | One entry per portfolio: type, label, owning entity, origination, pipeline availability, period convention |
| Client index | `config/client/client_index_{CLIENT}.yaml` | Tenant record: display name, default portfolio, readable portfolios |
| Source registrations | the durable source registry | One record per portfolio and dataset, `status: pending_review`, no fingerprint |

## Routing

Each catalogue field declares where its answer goes:

| `writes_to` | Meaning |
|---|---|
| `client_config:<dotted>` | Direct key in the client configuration |
| `annex12_config:<dotted>` | Direct key in the investor report overlay |
| `portfolio_registry:<key>` | Key on the portfolio's metadata entry |
| `source_registry:<field>` | Field on the source registration record |
| `tenancy:<dotted>` | The client index |
| `derived:<artefact>.<block>` | Shapes that artefact through a rule the generator owns |
| `onboarding_record` | No existing artefact represents it |

`artefacts.py::_route` walks the catalogue and applies the direct writes; the
rule-driven ones are named and implemented explicitly, because they must stay
consistent with each other:

- **Reporting products** → `pipeline.esma_enabled`, `supported_regimes`,
  `default_regime` and the legacy `regime` key together.
- **Entity roles** → the originator's name, identifier and country become
  `defaults.originator_*` (RREL82/83/84); the reporting entity's name becomes
  IVSS3/IVSS4; every other assigned role appears under `reporting_parties`.
- **UK geography override** → written in its full governed shape (value plus
  target fields), never as a bare flag.

## Field-by-field mapping

### Client and contacts

| Answer | Artefact key |
|---|---|
| Client name | `client_config:client.display_name` |
| Client identifier | `client_config:client.client_id` |
| Jurisdiction | `client_config:portfolio.country` |
| Reporting currency | `client_config:portfolio.base_currency` |
| Reporting time zone | `client_config:client.time_zone` |
| Environment | `client_config:client.environment` |
| Reporting contact name / email | `client_config:client.reporting_contact.{name,email}` |
| Operational contact name / email | `client_config:client.operational_contact.{name,email}` |
| Reporting contact telephone | `annex12_config:annex12.deal.IVSS6` |

### Entities

| Answer | Where it lands |
|---|---|
| Legal name | `defaults.originator_name` (originator) / IVSS3–4 (reporting entity) / `reporting_parties` (other roles) |
| Identifier | `defaults.originator_legal_entity_identifier` / IVSS1 |
| Country of establishment | `defaults.originator_establishment_country` |
| Roles | Decide which of the above apply |
| Registration number, registered address | **Onboarding record only** |

### Portfolios

| Answer | Artefact key |
|---|---|
| Portfolio identifier | `portfolio_registry:source_portfolio_id` |
| Portfolio name | `portfolio_registry:source_portfolio_label` |
| How the book was acquired | `portfolio_registry:source_portfolio_type` |
| Asset class | `client_config:portfolio.asset_class` |
| How the book is held | `portfolio_registry:structure` |
| Owning legal entity | `portfolio_registry:owning_entity` (+ resolved name) |
| Portfolio reporting currency | `portfolio_registry:reporting_currency` |
| Still originating | `portfolio_registry:originates` |
| Reporting period convention | `portfolio_registry:period_convention` |

### Sources

| Answer | Registry field |
|---|---|
| Portfolio | `source_portfolio_id` |
| Book | `dataset` |
| Expected cadence | `frequency` |
| Who sends the data | `source_system` |
| Expected files | `expected_columns` |
| Included in regulatory reporting | `regime_required` (derived) |
| Mapping approved | Sets `status` to active rather than pending review |
| How files arrive, file format, cut-off, sample provided | **Onboarding record only** |

### Regime

Routed by `config/regime/onboarding_standing_fields.yaml`: RREL82/83/84 and
RREL80/81 to `defaults.*`, NUTS year and the UK geography override to their
governed keys, and every Annex 12 `IVSS*` to `annex12.deal.*`.

### Presentation

| Answer | Artefact key |
|---|---|
| Report title | `client_config:mi.branding.app_title` |
| Brand colour | `client_config:mi.branding.theme.primary_color` |
| Logo, disclaimer | `client_config:mi.branding.{logo_uri,disclaimer}` |
| Day-count convention | `client_config:loan_engine.day_count_convention` |
| Payment frequency | `client_config:loan_engine.interest_payment_frequency` |
| Reporting calendar note | **Onboarding record only** |

## Determinism and idempotency

Both are tested.

**Deterministic** — nothing in rendering reads the clock, the filesystem or a
random source. `plan()` called twice on the same case produces byte-identical
documents.

**Idempotent** — applying an approved case again reports every document as
`unchanged` and rewrites nothing; source records are upserted by their stable
`client/portfolio/dataset/frequency` key; and `store.commit` returns the version
already in force when the answers hash matches, so a repeated activation cannot
fork a client's history.

## What generation will not do

- **Copy another client.** A new client's documents are built from its own
  answers. `base_documents` exists only so a *migration* preserves blocks
  onboarding does not own.
- **Clear what a delivery earned.** An existing source record keeps its approved
  mapping, pinned fingerprint, role schemas and last-successful markers, and is
  never demoted from active. Onboarding owns the business facts; the registry
  owns the delivery evidence.
- **Invent a forecast treatment.** An acquired book with no supplied runoff curve
  is left to the platform default rather than given one here.

## Fields still unsupported by existing configuration

Collected, versioned and shown at review — but written to no artefact, because
inventing a home for them would be inventing configuration.

| Field | Why |
|---|---|
| Sponsor, SSPE / issuer | No such field in the Annex 2 universe. Annex 12 names the securitisation entity only. |
| National competent authority | Trakt produces the report; it does not file it. |
| STS status and notification identifier | Extending the field universe is an administrator change to a governed package. |
| Regulatory servicer | `source_system` records who supplies data; there is no regulatory servicer field. |
| Risk-retention percentage | Annex 12 carries method (IVSS8) and holder (IVSS9), not percentage. |
| Transaction identifiers and dates | No standing transaction model exists. |
| Company registration number, registered address, authorised approver | No artefact holds them. |
| Delivery channel, file format, cut-off convention, sample provided | Operational facts the intake does not read from configuration. |
| Investor report recipients, reporting calendar note | No distribution model exists yet. |
| Annex 12 triggers (IVSR*) and cashflows (IVSF*) | Deal structure, not client standing information. Preserved where they are. |

The smallest governed extension that would close most of this gap is a
**standing transaction block** in the client configuration — owned by the client
layer, consumed by regime projections — which would hold sponsor, SSPE, STS
status and transaction identifiers. It is deliberately not added here: it needs
a consumer before it needs a schema.
