# Migrating the React dashboard to native multitenant Entra authentication

## What this replaces, and why

The dashboard signs users in through Azure Static Web Apps' custom Entra
provider. That provider is pinned to one directory (`openIdIssuer` in
`frontend/mi-agent-ui/staticwebapp.config.json`), so the only way to admit a
customer's staff is to invite them as **B2B guests** into the Trakt directory and
then hand each of them a Static Web Apps role invitation. That is not a SaaS
access model: it makes every customer's user an object in our directory, and it
ties authorisation to an invitation list the product cannot see.

Pointing the same provider at `/organizations` does not fix it. Consent succeeds
and the callback then fails, because Easy Auth validates the token's `iss`
against the single issuer it was configured with, and a multitenant authority
issues tenant-specific issuers. The provider has no "any tenant" mode to
configure — which is why the dashboard moves off it rather than being retuned.

The target is the shape the Copilot and agent surfaces already use:

```
external work account
  → https://login.microsoftonline.com/organizations
  → React (MSAL, authorization code + PKCE, no client secret)
  → access token for api://<trakt-mi-api-app-id>/MI.Access
  → trakt-mi-api validates signature, issuer, audience, expiry, scope
  → Trakt resolves organisation (tid) and named user (oid)
  → authorised user gets MI access
```

Azure Static Web Apps ends up hosting the React bundle and nothing else.

**Explicitly not used:** B2B guest accounts; SWA `client` invitations as the
long-term access model; an email address as a security identifier; the SPA's ID
token as an API credential; a client secret in the SPA.

## Two questions, two answers

The distinction this migration must not blur:

| Question | Answered by | Where |
|---|---|---|
| Who are you? | Microsoft Entra, via a validated access token | `mi_agent_api/react_auth.py` |
| Which company is asking? | `config/organisations.yaml` (verified `tid`) | `trakt_core/organisation.py` |
| Which person? | `config/principals.yaml` (verified `tid` + `oid`) | `trakt_core/principal.py` |
| May they, and to what? | `config/entitlements.yaml` × `config/resources.yaml` | `trakt_core/entitlement.py` |
| Whose data is served? | deployment configuration (`MI_AGENT_CLIENT_ID`) | `mi_agent_api/dependencies.py` |

A successful Microsoft sign-in answers only the first row. It is not, and must
never become, an entitlement to Trakt MI.

## Status

| Phase | State |
|---|---|
| 1 — Entra app registrations (`MI.Access`, SPA redirect URI) | Azure-side, not in this repo |
| 2 — backend bearer validation + dual-auth mode | **implemented** (this change) |
| 3 — MSAL in the React SPA | not started |
| 4 — SPA feature flag / preview deployment | not started |
| 5 — populate the Trakt identity registries | not started |
| 6 — Becquerel pilot | not started |
| 7 — security test pass | backend half implemented (`test_react_bearer_auth.py`) |
| 8 — production cutover | not started |
| 9 — retire the SWA auth path | not started |

Phase 2 is deliberately **additive**: with `TRAKT_MI_REACT_AUTH_MODE` unset the
API behaves exactly as it did before, and a bearer token is not looked at.

## Phase 1 — Entra (Azure portal, no code)

1. **`trakt-react-dashboard`** (existing, already multitenant). Add a
   **Single-page application** platform with redirect URIs for the SWA host and
   `http://localhost:5173`; add `https://app.traktinfra.io` when the custom
   domain lands. Leave the existing Web platform and the SWA settings in place
   until Phase 9 — both platform types coexist on one registration, and that is
   what keeps the rollback open.
2. **`trakt-mi-api`** (new Entra App Registration — not the App Service of the
   same name). Supported account types: multiple tenants. Expose an API with
   Application ID URI `api://<TRAKT_MI_API_CLIENT_ID>` and a **delegated** scope
   `MI.Access`.
3. **Permission**: on `trakt-react-dashboard`, add `MI.Access` as a delegated
   permission and grant admin consent in the Trakt tenant.

> **The detail that usually bites.** For an external tenant to consent at all,
> either the API registration is itself multitenant, or the SPA is listed in the
> API's `knownClientApplications` so one consent covers both. Miss it and
> external sign-in fails at consent, looking like a broken login rather than a
> missing registration.

## Phase 2 — the backend (implemented here)

### `mi_agent_api/react_auth.py`

Validates the token and applies the dashboard's policy. It **reuses**
`copilot_auth.allowed_directories`, `_select_directory`, `_jwks_client` and
`_allowed_issuers` rather than restating them — the same seam `agent_auth` uses,
so registering an organisation opens exactly one door.
`test_react_bearer_auth.py::test_react_auth_reuses_the_copilot_directory_allow_list`
pins those names.

What must hold for a token to be accepted:

| Check | Failure |
|---|---|
| RS256 signature against the issuing directory's published JWKS | 401 |
| `iss` is `…/{tid}/v2.0` or `sts.windows.net/{tid}/` for that directory | 401 |
| `aud` ∈ `TRAKT_MI_ENTRA_AUDIENCE` | 401 |
| `exp` present and in the future (`require: exp, iss, aud`) | 401 |
| `tid` present, on the allow-list, and equal to the verifying directory | 401 |
| `oid` present (`sub` is *not* accepted — it is pairwise per application) | 401 |
| `scp` present — an app-only token cannot act as a dashboard user | 403 |
| `scp` contains `TRAKT_MI_REQUIRED_SCOPE` (default `MI.Access`) | 403 |

401s are deliberately indistinguishable from one another: saying which check
failed would let a caller enumerate which organisations this deployment serves.
403 is used only where the token is valid and the *grant* is missing, which is
actionable for the caller's own administrator and reveals nothing.

### `TRAKT_MI_REACT_AUTH_MODE`

Read per request by `auth.auth_guard`:

- `swa` (default) — header only. A bearer token is ignored. Deploying this code
  with the default changes nothing.
- `both` — a request carrying `Authorization: Bearer` is authenticated by
  `react_auth`; anything else falls through to the header path. A token that
  fails to validate is **never** retried as a header request.
- `bearer` — token only. `X-MS-CLIENT-PRINCIPAL` stops being a credential. Until
  this is set, anything that can reach the API directly can present one.

An unrecognised value is 503, not a silent fallback to `swa`.

### Identity

`identity.context_from_react_principal` resolves organisation → principal →
entitlements from the verified `tid`/`oid`, exactly as the Copilot path does
(the shared ordering now lives in `_resolve_who_is_asking`, used by both). It
reports `channel=react`, `actor_type=user`, `DEFAULT_MI_SCOPES`, and an
`actor_id` that is the **object id** — never the display name or email, both of
which are mutable.

`TRAKT_MI_REACT_REQUIRE_PRINCIPAL` turns named-user licensing fail-closed: with
it set, a user with no row in `config/principals.yaml` is refused even in a
directory nobody has enumerated. It defaults **off** because turning it on before
the registry is populated refuses everyone, including whoever is performing the
migration. Turn it on in Phase 8, once every dashboard user is registered.

## Phase 3 — MSAL in the SPA (landed, flag off)

| File | Role |
|---|---|
| `src/auth/msalConfig.ts` | env → config, the flag, the `/organizations` authority, the scope requests |
| `src/auth/msalTokenProvider.ts` | `acquireTokenSilent` → interactive fallback; account selection |
| `src/auth/tokenProvider.ts` | the one place a credential is attached to a request |
| `src/auth/bootstrap.ts` | `initialize()` → `handleRedirectPromise()` → active account → register provider |
| `src/auth/AuthBoundary.tsx` | sign-in gate; renders children untouched when the flag is off |
| `src/main.tsx` | renders after bootstrap when the flag is on, immediately when off |
| `src/api/HttpAgentClient.ts` | `authHeaders()` on every fetch, plus a credentialed deck download |
| `scripts/check-bundle-leaks.mjs` | run by `npm run build`; fails on credential material in `dist/` |

**Two things worth knowing.**

*The deck download changed shape under bearer auth.* `<a href>` is a browser
navigation: it carries cookies, so it works behind Easy Auth, and it carries no
`Authorization` header, so under bearer auth `/mi/decks/download` answers 401.
With the flag on, the menu fetches the bytes with the token attached and hands
the browser a blob instead. With the flag off it is the same navigation it
always was.

*MSAL ships in both builds.* The imports are static, so the library is in the
bundle even when the flag is off — no MSAL object is constructed and no code
path runs, but the bytes are there (index chunk ≈396 KB gzipped). Making the
imports dynamic would remove them from the flag-off build; it was not done here
because it adds lazy-loading machinery to the sign-in gate, which is the one
path Phase 4 exists to prove.

### One behaviour change to plan for in Phase 3

`/me` returns `isOperator: false` for every bearer-authenticated caller, and the
SPA gates operator-only features on that flag (`HeaderBar`, `EvolutionPanel`).
So at cutover an operator signing in with MSAL sees the *client* view until
operator status has a Trakt-side source.

This is deliberate. The obvious implementation — an `operator` app role read from
the token's `roles` claim — does not survive multitenancy: once the API is
provisioned in a customer's directory, that customer's administrator can assign
their own users any app role the application defines. A privilege an external
tenant can grant itself is not a privilege. Operator status must come from a fact
only Trakt controls: an `organisation_type: operator` registration in
`config/organisations.yaml`, resolved onto the context and surfaced from there.
Wire that in Phase 3 or Phase 5, before the cutover in Phase 8.

### Deploy settings

```
TRAKT_MI_REACT_AUTH_MODE=swa            # then 'both' once the code is deployed
TRAKT_MI_ENTRA_AUDIENCE=api://<TRAKT_MI_API_CLIENT_ID>
TRAKT_MI_REQUIRED_SCOPE=MI.Access
TRAKT_MI_REACT_REQUIRE_PRINCIPAL=false  # Phase 8 flips this
```

The accepted directories come from `TRAKT_COPILOT_ENTRA_TENANT_ID` (historical
name, shared allow-list) plus every enabled organisation in
`config/organisations.yaml`. Before that file exists the allow-list is exactly
the app setting, so enabling `both` does not widen who may sign in.

### Verifying the deploy

`GET /health` reports the posture without an attempted login:

```json
"governance": { "dashboardAuth": { "mode": "both",
                                   "bearerConfigured": true,
                                   "requiresRegisteredPrincipal": false } }
```

`bearerConfigured: false` means every token will be answered 503 — a state
otherwise indistinguishable from a bad token in the browser.

Then confirm zero regression before going further: the existing dashboard login,
the Copilot path, and MI Query must all behave exactly as they did.

## Phase 5 — the registries (read this before creating the files)

Creating `config/organisations.yaml` flips the deployment from compatibility mode
to **organisation mode**, in which an unregistered or disabled directory is
refused on *every* path that carries a verified directory — including the live
Copilot surface. So:

1. Register the **current production directory** and the **current admin
   account** first, in the same change.
2. Deploy, then re-test React, Copilot, the agent API and MI Query.
3. Only then add an external organisation.

## Phase 6 — the pilot, and the test that matters most

With Becquerel's directory registered but nobody from it in
`config/principals.yaml`, `josh.hall@becquerelventures.com` should sign in to
Microsoft **successfully** and still be refused by Trakt. That refusal is the
proof that the two questions stayed separate.

To register him afterwards, call `GET /me` with his token: on the bearer path it
echoes `microsoftTenantId` and `microsoftObjectId`, which is the exact
`(tid, oid)` pair a `principals.yaml` row is written against. His email address
belongs in that row as human-readable metadata and is never consulted.

## Security tests

`mi_agent_api/tests/test_react_bearer_auth.py` covers the backend half: mode
behaviour (including that a failed token never falls back to the header, and that
`bearer` mode makes the SWA header inert), every token-validation failure above,
forged-`tid`, an ID token used as an API credential, an app-only token, a token
minted for another Trakt surface — and, separately, the Trakt authorisation
layer: registered/unregistered/disabled users, unregistered directories, two
organisations reaching the same client dataset distinguishably, and the
fail-closed named-user switch.

Two things it does **not** cover, because they are not backend behaviour: MSAL
token acquisition in the SPA (Phase 3), and end-to-end sign-in from a real
external tenant (Phase 6).

## Known gap, deliberately out of scope

`trakt_core.entitlement.authorise_resource_access` is not called by any
`mi_agent_api` route today. Entitlements are resolved and frozen onto the
context, but MI routes do not yet narrow per resource — dataset isolation still
comes from deployment-per-tenant. This migration therefore delivers tenant-level
and named-user-level control; "organisation A sees SPV I but not SPV II" needs an
enforcement call added to the MI routes and is separate work.
