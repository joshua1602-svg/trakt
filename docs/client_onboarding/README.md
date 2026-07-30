# Client Onboarding

A governed capability inside the Operations Control Centre for creating and
maintaining a client's **standing configuration** — the facts that stay true
between deliveries.

It sits alongside Operations. Operations processes deliveries; Client Onboarding
creates the configuration those deliveries resolve with. It is not part of the
monthly reporting cycle.

## The idea in one line

The operator answers business questions; Trakt writes the technical
configuration — into the artefacts it already reads, not into a parallel set.

## Documents

| Document | Contents |
|---|---|
| [01 — Current architecture findings](01_current_architecture_findings.md) | How client configuration is assembled today, traced to file and function. Mandatory vs optional fields, operational vs reporting metadata, the source registry, every Annex/regime standing field, and everything else maintained by hand. |
| [02 — Mapping and gaps](02_mapping_and_gaps.md) | Every onboarding answer and the legacy artefact it reads from and writes to. The fields that cannot currently be represented, with reasons. Why one new configuration file was genuinely required. |
| [03 — Implementation ledger](03_implementation_ledger.md) | File by file: what is new, what changed, what was deliberately left alone. API surface and storage layout. |
| [04 — Migration strategy](04_migration_strategy.md) | How an existing client is adopted without re-entering anything, what adoption preserves, and how to roll back. |
| [05 — Test results and recommendations](05_test_results_and_recommendations.md) | What the tests cover and what they prove. Eight remaining architectural recommendations. |

Screenshots: [`docs/screenshots/client_onboarding/`](../screenshots/client_onboarding/).

## What it generates

| Artefact | Existing home |
|---|---|
| Client configuration | `config/client/config_client_{CLIENT}.yaml` |
| Investor report configuration | `config/client/config_client_{CLIENT}_annex12.yaml` |
| Portfolio metadata | `config/client/portfolio_registry_{CLIENT}.yaml` |
| Source registrations | the durable source registry |

Same formats, same layers, same readers. What onboarding adds is authorship,
versioning and an audit trail.

## Design commitments

- **No duplicate configuration.** Every answer lands in an artefact that already
  exists. The mapping is declared once, in
  `config/regime/onboarding_standing_fields.yaml`, and used in both directions.
- **Generation is a merge.** Blocks onboarding does not own — enrichment,
  transformations, deal-structure triggers — are carried through untouched. An
  existing source registration keeps the mapping and fingerprint a real delivery
  earned it.
- **Nothing is written before approval.** The review screen shows every artefact
  that will be created or changed, with the current content alongside.
- **Versioning over overwrite.** Every approval creates a new immutable version
  carrying who, when, what changed, before, after and why, and appends to the
  existing hash-chained audit trail.
- **No hard-coded regime or asset class.** The wizard renders whatever the
  governed standing-field declaration contains, and offers whatever the asset
  support model allows. Adding a regime is a configuration change.
- **Shipping this adopts no one.** A client without an approved profile resolves
  the repository file exactly as before.

## Relationship to Operations

After a client is onboarded the manual delivery workflow asks only:

```
Client → Portfolio → Reporting period → Files
```

Everything else is derived: the reporting products from the client's profile,
the regime scope from the source registration, the standing regime values from
the generated client configuration.
