# Legacy client documents — fixtures, not configuration

These are the pre-onboarding client documents, kept because the adoption path
(`operations_control.onboarding.migration.adopt`) genuinely reads one and its
regression tests need a real example to read.

They are **not** production configuration and nothing at runtime resolves to
them. A client's configuration is created by `OnboardingService.activate()`,
versioned and hashed per client in the operations store, and reached only
through `EffectiveConfigResolver.client_config_for(client_id)`. That resolver
has no default and no fallback: a client Trakt cannot resolve is a client Trakt
does not deliver for.

They previously lived at `config/client/`, where every runtime consumer that
lacked a client used them as its default — so a second client's delivery could
be projected under ERE Funding's identity, LEI and reporting date. Retiring
that fallback is what moved them here.

`config_client_annex12.yaml` also carries `IVSS2` (a reporting period end date)
for which no governed owner exists yet; see the residual note in the retirement
commit. A reporting date must come from run context, never from standing client
configuration, so it was not migrated into the catalogue to make this move
possible.
