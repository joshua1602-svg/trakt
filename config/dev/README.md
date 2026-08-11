# Development identity and entitlement configuration

Real, loadable versions of the five governance registries, for the Sprint 1
external-agent proof and for local development.

## Why these live under `config/dev/` and not at `config/`

Each registry has a **default path at the repository root** (`config/tenancy.yaml`,
`config/organisations.yaml`, …), and for `organisations.yaml` in particular the
mere existence of the file is a posture change:

> **organisation mode** (file present) — an unregistered or disabled directory is
> refused (`ORGANISATION_NOT_REGISTERED` / `ORGANISATION_DISABLED`, both 403) on
> any path carrying a validated directory. There is no permissive fallback.
> — `docs/governed_capability_architecture.md` §3

Committing these at the default paths would therefore switch the live ERE Copilot
deployment into organisation mode against a directory GUID that is a placeholder,
and every Copilot request would start failing. That is precisely the fail-closed
behaviour the design intends; the mistake would be triggering it from a
development fixture.

So they live here and are wired by the documented environment variables. Nothing
loads them unless a deployment or a test says so.

## Wiring

```bash
export TRAKT_TENANCY_CONFIG=config/dev/tenancy.yaml
export TRAKT_ORGANISATIONS_CONFIG=config/dev/organisations.yaml
export TRAKT_RESOURCES_CONFIG=config/dev/resources.yaml
export TRAKT_ENTITLEMENTS_CONFIG=config/dev/entitlements.yaml
export TRAKT_PRINCIPALS_CONFIG=config/dev/principals.yaml

# Whose data this deployment serves.
export MI_AGENT_CLIENT_ID=ERE

# The agent API is off unless switched on.
export TRAKT_AGENT_API_ENABLED=true

# Local development identity. Refused outright in production.
export TRAKT_AGENT_AUTH_MODE=disabled
export TRAKT_AGENT_DEV_DIRECTORY=00000000-0000-4000-8000-00000000a2a7
```

`scripts/agent_dev_env.sh` sets exactly this; source it rather than copying.

## What is configured

| Registry | Contents |
|---|---|
| `tenancy.yaml` | One tenant, `ERE`, with an explicit portfolio allow-list — so tenancy rule 3 is ON, not the "any well-formed selector" default. |
| `organisations.yaml` | `ere` (the originator) and `a2a_test_agent` (the Sprint 1 machine identity), each on its own directory. |
| `resources.yaml` | `ERE/source_portfolio/direct_001` (**Portfolio A**), `ERE/source_portfolio/acquired_001` (**Portfolio B**), `ERE/portfolio/ere_total`, and `ERE/spv/spv_ii` — registered but unpartitionable, so it can be named and refused. |
| `entitlements.yaml` | `a2a_test_agent` holds `risk:read` on **Portfolio A only**. Nothing on Portfolio B. |
| `principals.yaml` | Empty. A service principal is an application, not an individual — `context_from_agent_principal` does not consult this registry, and no directory here is principal-gated. |

## The exit test this configuration exists to make true

| Call | Expected |
|---|---|
| `evaluate_covenants` on `ERE/source_portfolio/direct_001` | authorised |
| `evaluate_covenants` on `ERE/source_portfolio/acquired_001` | `RESOURCE_NOT_AUTHORISED` |
| `evaluate_covenants` on `ERE/source_portfolio/does_not_exist` | `RESOURCE_NOT_AUTHORISED` — the *same* code and message |
| `evaluate_covenants` on `ERE/spv/spv_ii` | `RESOURCE_NOT_AUTHORISED` (it is not granted; and it could not be, because it is unpartitionable) |
| a tool needing `mi:query` | `SCOPE_MISSING` — the agent's scopes are derived from its grants, and it holds none |

The third row is the one worth re-reading: an unauthorised resource and a
nonexistent one must be indistinguishable, or a caller can enumerate another
organisation's books by comparing error codes.

## Going to a real deployment

Replace the placeholder directory GUIDs with real Entra directory ids, move the
files to the default `config/` paths (or keep the environment variables), set
`TRAKT_AGENT_AUTH_MODE=entra`, `TRAKT_AGENT_ENTRA_AUDIENCE`, and assign the
`Trakt.Agent` app role to the calling application's service principal.

Registering an organisation is enough for its directory to be accepted — the
directory allow-list is the union of `TRAKT_COPILOT_ENTRA_TENANT_ID` and every
enabled organisation's directories, so no GUID has to be duplicated into an app
setting.
