# Migration strategy

## The principle

Nothing changes for an existing client until someone decides it should. Shipping
this code adopts no one.

`EffectiveConfigResolver.client_config_for()` returns the onboarding-generated
configuration **only** when the client has an approved profile. Otherwise it
returns the repository file, byte for byte as before. There is no flag day and
no big-bang cutover.

## Adopting a client

1. **Open Client Onboarding.** A client configured before onboarding existed is
   listed as *Not yet onboarded*, with its current values described rather than
   hidden.
2. **Adopt current configuration.** `migration.adopt()` reads what already
   exists:

   | Source | What it yields |
   |---|---|
   | `config/client/config_client_ERM_UK.yaml` | Identity, jurisdiction, currency, regimes, Annex 2 standing values, branding, calculation conventions |
   | `config/client/config_client_annex12.yaml` | Standing investor-report fields (IVSS5-IVSS10 …) |
   | `config/client/portfolio_registry.yaml` | Portfolio labels, origination, pipeline availability |
   | `config/tenancy.yaml` | Display name, portfolio list |
   | The durable source registry | Portfolios, datasets, cadences, source systems, regime scope |

3. **The first screen shows current values, ready for editing.** Every adopted
   value carries where it came from — "Taken from
   `defaults.originator_legal_entity_identifier` (RREL83) in the client
   configuration" — so nothing appears from nowhere.
4. **What the legacy files cannot answer is named, not guessed.** For ERE that
   is the reporting and operational contacts, which have never existed in any
   configuration file. They appear as an explicit "still needed" note rather
   than as blank fields the operator has to notice.
5. **Review, then approve.** The review screen shows every artefact that will be
   created or changed, with the current content alongside. Approval requires a
   reason and an administrator.

## What adoption preserves

Adoption is a merge onto the legacy documents, not a replacement:

- `enrichment`, `transformations`, `pipeline_persistence` and any administrator
  additions to the client configuration are carried through untouched.
- Annex 12 deal-structure blocks — the IVSR trigger definitions and IVSF
  cashflow items — survive intact. They are not standing client information and
  onboarding does not claim them.
- An existing source record keeps its approved mapping id, pinned schema
  fingerprint, role schemas, role aliases and last-successful markers, and a
  source already proven by a real delivery is never demoted to
  `pending_review`. Resetting those would send a recognised recurring pack back
  through source onboarding — a real operational regression, and the reason
  generation merges rather than overwrites.

A genuinely new client starts from an empty document. It never inherits another
client's enrichment rules, which is the silent coupling this capability exists
to remove.

## After adoption

The client's configuration is generated from its profile on every approval.
Editing the repository YAML by hand no longer takes effect for that client —
the resolver reads the generated artefact — and the generated file says so in
its header. This is the intended end state: the YAML becomes an implementation
artefact, not a user interface.

## Rollback

Three levels, in increasing order of severity:

1. **Re-approve.** Open the client, change what is wrong, approve with a reason.
   That is a new version; the old one stays readable.
2. **Restore an earlier version.** Every version is addressable
   (`GET /ops/onboarding/clients/{id}/versions/{n}`) and carries the full
   profile, so an earlier state can be re-entered and re-approved.
3. **Fall back to the repository file.** Deleting a client's generated artefact
   returns `client_config_for()` to the repository path, and the client behaves
   exactly as it did before adoption. No code change, no redeploy.

## Suggested order

1. Adopt ERE in a non-production environment and compare the generated
   configuration against the committed file. The tests already assert that a
   generated configuration satisfies the Annex 2 preflight with no blockers.
2. Fix the invalid committed LEI (findings §1) as the first governed change, so
   the audit trail records the correction with a reason.
3. Adopt ERE in production.
4. Onboard the next new client through the wizard rather than through a pull
   request.
