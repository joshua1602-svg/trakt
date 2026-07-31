# The revised domain model and the onboarding field catalogue

## The model

Three concepts, and one of them is new.

### The onboarding case

`operations_control/onboarding/case.py`. A case is one attempt to bring a client
into Trakt, or to change what Trakt holds about one. It is opened **before
anyone knows what the client will be called**, so it lives under its own
reference — `ONB-2026-0001` — and adopts a client id only when step 1 supplies
one.

Three kinds share the model:

| Kind | Where the answers start | Status |
|---|---|---|
| `new_client` | Blank | **The product** |
| `migration` | A legacy client's existing files | Secondary |
| `amendment` | The version currently in force | Ongoing change |

Only the entry point differs. Validation, information requests, generation and
approval are identical, so a migrated client and a newly onboarded one are the
same client afterwards.

### Statuses

```
draft ──▶ information_requested ──▶ awaiting_client ──▶ in_review
  │                                                        │
  │                                              changes_required
  ▼                                                        │
ready_for_approval ──▶ approved ──▶ activated              │
  └──────────────── withdrawn ◀─────────────────────────────┘
```

Transitions are enforced in `case.py::TRANSITIONS`, not by the browser. The
distinction that matters: **approval records the decision; activation performs
it.** Every status except `activated` writes no active configuration.

### Legal entities

The corrected model's central addition. Previously a client had one
`legal_entity_name` and one `lei`. Now a case holds a list of entities, each
with its own identity and **a list of roles** drawn from a governed vocabulary:

```
originator · sponsor · sspe · reporting_entity · original_lender · servicer
risk_retention_holder · trustee · warehouse_lender · investment_manager
calculation_agent · seller
```

The same company commonly originates and services. It is captured once and
given both roles; generation reads roles to decide where it appears. Portfolios
point at an entity by reference, so an entity cannot be silently orphaned.

### Information requests

`case.py::InformationRequest`. A governed record of what was asked, of whom,
when, what came back, what evidence arrived, and who accepted it. A full client
portal is not required for this to be useful — and if one is built later, it
posts into the same record.

The **client checklist** is derived, not maintained: the catalogue knows which
fields are `client_supplied` and which are conditionally required, so the list
of what a client still owes falls out of the answers so far. Asking a client for
an identifier Trakt mints, or for something inferred from their own upload,
never happens.

---

## The catalogue

`config/onboarding/field_catalogue.yaml`, loaded by
`operations_control/onboarding/catalogue.py`.

It is the single declaration of what Trakt needs to know. The wizard renders it,
validation reads it, the client checklist derives from it, and generation routes
through it. **Adding a field is a change to this file, not to a form component.**
A test asserts it contains no client values.

### The two classifications

Every field declares **who supplies it**, which is the axis that decides whether
it can appear on a client information request at all:

| `source` | Meaning | Asked of the client? |
|---|---|---|
| `client_supplied` | Only the client knows it | **Yes** |
| `operator_supplied` | Trakt's operator decides it | No — asked of the operator |
| `inferred` | Read from data the client uploads | Never asked |
| `derived` | Computed from another answer | Never asked |
| `trakt_default` | A governed default applies | Only if overridden |
| `system_generated` | Trakt mints it | Never asked |
| `delivery_specific` | Changes every period | **Excluded from onboarding** |

…and **what it belongs to**: `client`, `legal_entity`, `portfolio`, `source`,
`transaction`, `regime`, `output`, `contact`, `branding`, `operational`.

### Sections

| Section | Repeatable | Fields | Notes |
|---|---|---|---|
| `client` | | 6 | Identity, jurisdiction, currency, time zone |
| `entities` | ✓ | 7 | Legal entities with reusable roles |
| `contacts` | | 7 | Reporting, operational, investor, approver |
| `portfolios` | ✓ | 9 | Books, with owning entity and period convention |
| `sources` | ✓ (derived) | 11 | Expected deliveries per book and dataset |
| `reporting` | | 1 | Which products apply |
| `regime` | | 21 | From the governed regime declaration |
| `presentation` | | 7 | Branding and calculation conventions |

### Governed vocabularies

Lists owned elsewhere are **read from their owner** at load time, not restated:

| Vocabulary | Owner |
|---|---|
| Asset classes | `operations_control.configuration.packages.ASSET_MODEL` |
| Datasets | `operations_control.contracts.BATCH_DATASETS` |
| Cadences | `apps.blob_trigger_app.path_parser.VALID_FREQUENCIES` |
| Annex 12 enumerations | `config/regime/annex12_template.yaml` |

A catalogue that restated the dataset list would eventually disagree with the
intake. Lists with no owner elsewhere — entity roles, delivery channels, file
formats, period conventions — are declared in the catalogue itself.

### Conditional requirements

`required_when` carries a tiny total grammar: `<path> == <value>`,
`!=`, `in [a, b]`, `contains`, `always`. Evaluated by `catalogue.evaluate`.

```yaml
- key: lei
  required_when: "roles contains originator"     # per-entity

- key: reporting_contact_phone
  required_when: "reporting.products contains investor_reporting"
```

An expression Trakt cannot parse makes the field **optional**, so a malformed
catalogue degrades to permissive rather than making onboarding impossible.

### Regime fields

The `regime` section declares `fields_from: regime_products` and is filled from
`config/regime/onboarding_standing_fields.yaml`. A future regime added there
appears in the wizard with no code change — a test proves it, by loading a
synthetic `future_regime` and asserting its field arrives.

### What is deliberately not collected

Declared under `not_collected`, so the catalogue is a complete account of the
information model rather than only of the questions:

| Field | Why |
|---|---|
| Reporting period | Delivery-specific. Supplied with each delivery. |
| Static reporting date | Delivery-specific, despite living in the legacy client file. |
| Expected schema fingerprint | Learned from the first real delivery. Asking would be asking the operator to guess. |
| File role signatures | Learned at mapping approval. |
| Delivery location | System-generated from the storage layout. Shown, never typed. |
| No-data policy | The governed regime rules carry defaults; an override is an administrator change. |
