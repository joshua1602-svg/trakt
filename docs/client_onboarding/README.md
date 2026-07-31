# Client Onboarding

A governed capability inside the Operations Control Centre for bringing a client
Trakt has never met into Trakt, and for maintaining what Trakt holds about them
afterwards.

## The product

```
Start new client onboarding      ← blank. No client selected. Nothing read.
        ↓
Ask the business questions       ← rendered from the governed field catalogue
        ↓
Request what the client owes     ← derived, not maintained by hand
        ↓
Validate continuously            ← as answers arrive, not only at the end
        ↓
Preview the configuration        ← exactly what will be written
        ↓
Approve                          ← records the decision. Writes nothing.
        ↓
Activate                         ← the only place configuration is created
```

It works for a client with no existing YAML, no source registrations, no
portfolio metadata, no prior delivery and no prior workflow.

## Three workflows, one model

| | Where answers start | Status |
|---|---|---|
| **New client** | Blank | **The product** |
| **Legacy migration** | An existing client's files | Secondary and optional |
| **Amendment** | The version in force | Ongoing change |

Only the entry point differs. Validation, requests, generation and approval are
identical, so a migrated client and a newly onboarded one are the same client
afterwards.

## Documents

| Document | Contents |
|---|---|
| [01 — Gap analysis](01_gap_analysis.md) | The first implementation traced against the corrected requirement: what supported blank onboarding, what assumed an existing client, and what was missing entirely. |
| [02 — Domain model and catalogue](02_domain_model_and_catalogue.md) | The case, its statuses, entities with reusable roles, information requests; and the client-agnostic field catalogue with its two classifications. |
| [03 — Generation mapping](03_generation_mapping.md) | Every answer and the artefact it becomes. Determinism, idempotency, and the fields no existing artefact can represent. |
| [04 — Implementation ledger](04_implementation_ledger.md) | File by file, plus the API surface and storage layout. |
| [06 — Field classification](06_field_classification.md) | Every field in one of five buckets, and the questions removed: 33 human-required fields down to 15. |
| [05 — Test results](05_test_results.md) | What the tests prove, the 390px measurements, and the baseline comparison. |

Screenshots: [`docs/screenshots/client_onboarding/`](../screenshots/client_onboarding/).

## What activation generates

| Artefact | Existing home |
|---|---|
| Client configuration | `config/client/config_client_{CLIENT}.yaml` |
| Investor report configuration | `config/client/config_client_{CLIENT}_annex12.yaml` |
| Portfolio metadata | `config/client/portfolio_registry_{CLIENT}.yaml` |
| Client index | `config/client/client_index_{CLIENT}.yaml` |
| Source registrations | the durable source registry |

Same formats, same layers, same readers. No parallel configuration stack.

## Design commitments

- **It starts blank.** The existing ERE configuration informed the schema; it is
  not required to run onboarding, and is only ever read by the optional
  migration path.
- **The catalogue is the model.** Adding a field is a change to
  `config/onboarding/field_catalogue.yaml`, not to a form component. Vocabularies
  are read from the modules that own them; regime fields come from the governed
  regime declaration, so a future regime needs no code.
- **Approval and activation are different acts.** Approval records a decision.
  Activation writes configuration. Nothing active exists before it.
- **A case can always be abandoned.** Cancelling is offered at every step, not
  only at the end, because the moment someone decides to stop is exactly the
  moment they should not have to finish the wizard to say so. It asks why, keeps
  the record, and creates nothing.
- **Portfolios and sources exist before the first delivery.** A portfolio is not
  created by a publication; it is created by an approved onboarding.
- **Generation is deterministic and idempotent**, merges rather than replaces,
  and never clears what a real delivery earned a source record.
- **Ask as little as possible.** Every field is classified by who can actually
  answer it. Trakt infers the currency and time zone from the jurisdiction, the
  asset class and file format from a sample pack, the identifiers from the
  client's name, and every regime block from answers already given — 15
  required human answers, not 33. Nothing is silent: each value shows where it
  came from and can be overruled.
- **Migration is secondary.** It is offered at the foot of the home page, it
  changes nothing before approval, and a legacy value today's rules refuse is
  raised as an issue rather than carried across quietly.

## Relationship to Operations

Operations processes deliveries. Onboarding creates the configuration those
deliveries resolve against. After a client is activated, the delivery workflow
asks only:

```
Client → Portfolio → Reporting period → Files
```

Everything else is derived from the approved onboarding: which reporting
products apply, which book carries regulatory scope, and every standing
regulatory value.
