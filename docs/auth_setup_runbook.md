# MI Agent authentication — setup runbook

Adds authentication + role-based access to the **client-facing MI Agent** so only
invited users can reach the UI or the API. Scope: one client, ~5 client users +
operators. Model chosen: **Entra ID B2B guest invites · platform Easy Auth + Static
Web Apps config · client role + operator role**.

The application code and config are already in this repo (`mi_agent_api/auth.py`,
the guard/CORS/exception changes in `mi_agent_api/app.py`, and
`frontend/mi-agent-ui/staticwebapp.config.json`). This runbook is the **Azure-side
work you apply**; no further code changes are required to turn auth on.

---

## Architecture

```
client user ──login (Entra ID)──▶ Azure Static Web App (Standard)
                                    │  serves the React MI Agent UI
                                    │  enforces staticwebapp.config.json (auth + roles)
                                    ▼
                            linked backend  ── injects X-MS-CLIENT-PRINCIPAL ─▶ trakt-mi-api (App Service, FastAPI)
                                                                                  auth.py reads the principal,
                                                                                  requires client|operator role
```

- The SWA does the login and **forwards the verified identity** to the API as the
  `X-MS-CLIENT-PRINCIPAL` header. The API trusts that header (no token validation
  in-app) — the supported linked-backend / Easy Auth pattern.
- Same-origin: the UI calls `/api/*`, which the SWA proxies to the App Service, so
  there is no cross-origin/token handling in the browser.

---

## Values you need to supply (fill these in before starting)

| Placeholder | Where used | Value |
|---|---|---|
| `AAD_TENANT_ID` | staticwebapp.config.json issuer | _your Entra tenant ID_ |
| `AAD_CLIENT_ID` | SWA app setting | _app registration (client) ID_ |
| `AAD_CLIENT_SECRET` | SWA app setting | _app registration client secret_ |
| SWA name | Azure portal | _e.g. the existing `nice-smoke-…` SWA_ |
| App Service name | Azure portal | `trakt-mi-api` |
| Client email domain | B2B invites | _e.g. `@client-domain.com`_ |
| Operator emails | role assignment | _your team's accounts_ |
| Client tenant label | `MI_AGENT_CLIENT_ID` | _the single client id this deployment serves_ |

> If the SWA is currently on the **Free** tier, it must be upgraded to **Standard**
> — custom Entra providers and linked backends require Standard.

---

## Step 1 — Entra app registration

1. Entra ID → App registrations → New registration. Name e.g. `trakt-mi-agent`.
2. Redirect URI (Web): `https://<swa-hostname>/.auth/login/aad/callback`.
3. Certificates & secrets → new client secret → record the value → this is
   `AAD_CLIENT_SECRET`. The Application (client) ID is `AAD_CLIENT_ID`; the
   Directory (tenant) ID is `AAD_TENANT_ID`.
4. (Recommended) App roles → create two app roles, values **`client`** and
   **`operator`** (Allowed member types: Users/Groups). These are surfaced to the
   API. *Alternatively* you can assign the same role names via SWA role invitations
   in Step 5 without app roles — either works; app roles scale better.
5. Token configuration (only if using app roles): ensure the `roles` claim is
   emitted (it is by default for assigned app roles).

## Step 2 — Static Web App (Standard) + config

1. Upgrade the SWA to **Standard** if needed.
2. Configuration → Application settings → add:
   - `AAD_CLIENT_ID` = _app (client) ID_
   - `AAD_CLIENT_SECRET` = _client secret_
3. Deploy `frontend/mi-agent-ui/staticwebapp.config.json` (it ships with the app;
   replace `AAD_TENANT_ID` in the `openIdIssuer` with your tenant ID — or template
   it in the build).

## Step 3 — Link the backend

1. SWA → APIs → Link an existing backend → select the `trakt-mi-api` App Service.
2. This makes the SWA proxy `/api/*` to the App Service and forward the principal.

## Step 4 — App Service (`trakt-mi-api`) settings

Add these application settings and restart:

| Setting | Value |
|---|---|
| `MI_AGENT_AUTH_ENABLED` | `true` |
| `MI_AGENT_CLIENT_ROLE` | `client` |
| `MI_AGENT_OPERATOR_ROLE` | `operator` |
| `MI_AGENT_CLIENT_ID` | _the single client id_ |
| `MI_AGENT_CORS_ORIGINS` | _the SWA origin, e.g. `https://<swa-hostname>`_ |

> With the linked backend the UI is same-origin, so CORS is defensive only. Never
> set `MI_AGENT_CORS_ORIGINS` to `*` — the app no longer has a wildcard fallback,
> and an unset value denies cross-origin browser calls.

## Step 5 — Frontend build target + invite users

1. Build the SWA with `VITE_AGENT_API_URL=/api` so the UI calls the linked backend
   same-origin. **Update `.github/workflows/azure-static-web-apps-*.yml`** — the
   current `VITE_AGENT_API_URL` points at the Streamlit host and must change to
   `/api` **after** the linked backend exists (Step 3).
2. Invite the ~5 client users as **B2B guests** (Entra → Users → Invite external
   user) using their `@client-domain.com` addresses.
3. Assign roles:
   - *App-role path:* Enterprise applications → `trakt-mi-agent` → Users and groups
     → assign each client guest the **client** role and each operator the
     **operator** role.
   - *SWA-invitation path (alternative):* SWA → Role management → invite each user
     and grant role `client` or `operator`.

## Step 6 — Verify

```bash
# health is open (probe) even with auth on:
curl -i https://<swa-hostname>/api/health           # 200

# protected without login → redirected to Entra login:
curl -i https://<swa-hostname>/api/me               # 302 to /.auth/login/aad

# after logging in through the browser, /api/me returns identity + roles:
#   {"authenticated": true, "user": "...", "roles": ["client"], "isOperator": false}
```

Also confirm: an authenticated user with **no** client/operator role gets 403; the
UI is unreachable when logged out; `/api/health` contains no server file path.

## Rollback / disable

Set `MI_AGENT_AUTH_ENABLED=false` on the App Service to bypass API enforcement
(the SWA route rules still gate the UI). To fully revert, remove the
`staticwebapp.config.json` auth block and unlink the backend. **Do not expose the
UI to the client with auth disabled.**

---

## What the code already does (no action needed)

- `mi_agent_api/auth.py` — parses `X-MS-CLIENT-PRINCIPAL` (SWA and Easy Auth
  shapes), global `auth_guard` requiring an MI role, `client`/`operator`
  distinction. Fails closed: no principal → 401, no MI role → 403.
- `mi_agent_api/app.py` — guard wired as a global dependency; CORS wildcard
  fallback removed; `/health` no longer leaks the dataset path; unhandled errors
  return a generic 500 (no stack/paths); `GET /me` reports the resolved identity.
- `frontend/mi-agent-ui/staticwebapp.config.json` — Entra login, `/api/*` and the
  SPA restricted to `client`/`operator`, health left open, login/logout redirects,
  hardening headers.
- Tests: `mi_agent_api/tests/test_auth.py` (guard + parsing); the existing suite is
  unaffected (`mi_agent_api/tests/conftest.py` disables enforcement for it).

## Known follow-ups (not blockers for one client)

- **Per-request tenant entitlement.** With one client, data isolation is achieved
  by deploying exactly one client's dataset; the caller-supplied `client_id` only
  selects cohorts *within* that dataset. True multi-client entitlement (mapping a
  principal → allowed client_ids) is a later change when a second client is hosted.
- **UI identity chip / logout link.** The platform enforces access already; adding
  a signed-in-user chip + logout in the React header is UX polish (use `/.auth/me`
  and `/logout`).
