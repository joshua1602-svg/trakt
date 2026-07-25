# Trakt Copilot agent — client deployment runbook

Deployment model: **private organisational Microsoft 365 Copilot agent** (not
Marketplace). Trakt owns the Entra application and the packaging; the client
administrator only uploads the finished package, grants admin consent, and
assigns users. The client never edits manifests, never registers anything in
the Teams Developer Portal, never builds a zip, and never configures Azure.

Verified against the July 2026 Microsoft schemas: Teams app manifest 1.19
(`copilotAgents.declarativeAgents`), declarative agent **v1.7**, API plugin
**v2.4**. Entra SSO for API plugins uses an **auth config in the Microsoft
Enterprise token store** referenced from the plugin manifest as
`{"type": "OAuthPluginVault", "reference_id": "<auth config ID>"}` — there is
no separate "MicrosoftEntra" auth type. `webApplicationInfo` is not required
for API-plugin SSO.

---

## Part 1 — One-time Trakt setup (once, in Trakt's tenant)

1. **Entra app registration** (e.g. `trakt-copilot-api`), account type
   **"Accounts in any organizational directory" (multi-tenant)**:
   - **Expose an API** → add scope `access_as_user` (admin + user consentable).
   - **Expose an API → Add a client application** → pre-authorise the Microsoft
     Enterprise token store client id `ab3be6b7-f5df-413d-ac2d-abf1e3fd9c0b`
     for that scope.
   - **Authentication** → Web platform → add redirect URI
     `https://teams.microsoft.com/api/platform/v1.0/oAuthConsentRedirect`.
   - Manifest → `accessTokenAcceptedVersion: 2`.
   - Record the **application (client) id** → `entra_app_id` in client configs.
2. **Entra SSO auth config** (Teams developer portal in **Trakt's** tenant →
   Tools → *Microsoft Entra SSO client ID registration* → Register client ID):
   - Base URL: the client's Trakt MI API base URL.
   - Client ID: the app id from step 1. Scope: `access_as_user`.
   - **Restrict usage by org**: the client's Microsoft 365 organisation.
   - **Restrict usage by app**: "Any Teams app" initially; after first upload,
     bind to the client's app id (printed by the packager).
   - Record the generated **auth config ID** (portal label: *Microsoft Entra
     SSO registration ID*) → `sso_auth_config_id`, and the generated
     **Application ID URI** → `app_id_uri`.
3. **Back-fill the Entra app**: add the generated Application ID URI to the app
   registration's `identifierUris` (manifest editor supports multiple URIs).
4. Nothing in Parts 1–2 is per-user; repeat step 2 (and the per-client parts of
   step 3) once per client organisation.

## Part 2 — Per-client Trakt packaging and API configuration

1. **API deployment** (existing dedicated deployment-per-client model,
   unchanged). Set on the client's `trakt-mi-api` App Service:
   - `TRAKT_COPILOT_AUTH_MODE=entra`
   - `TRAKT_COPILOT_ENTRA_TENANT_ID=<the CLIENT's Entra tenant GUID>`
     (tokens are issued by the user's home tenant)
   - `TRAKT_COPILOT_ENTRA_AUDIENCE=<app_id_uri>,<entra_app_id>`
     (v1 tokens carry the URI as audience; v2 tokens carry the app GUID)
   - `TRAKT_COPILOT_REQUIRED_SCOPE=access_as_user`
   - `TRAKT_COPILOT_DOWNLOAD_SIGNING_KEY=<random ≥32 chars>`
   - `TRAKT_COPILOT_PUBLIC_BASE_URL=<the client's API base URL>`
2. **Client configuration file** (kept out of the repo; see
   `deploy/copilot-agent/client-config.sample.json`): `client_id`,
   `api_base_url`, `entra_app_id`, `app_id_uri`, `sso_auth_config_id`,
   `scope_name`, `package_version` (bump on every re-issue).
3. **Build the package**:
   ```bash
   python deploy/copilot-agent/package_agent.py --config <client>.json
   # → deploy/copilot-agent/dist/trakt-copilot-agent-<client_id>.zip
   ```
   The build **fails** if any placeholder survives substitution. The Teams app
   id is a deterministic UUIDv5 of `client_id`, so re-issued packages update
   the client's existing app instead of creating a duplicate.
4. **Validate** (Microsoft official tooling):
   ```bash
   npx -y @microsoft/m365agentstoolkit-cli validate \
       --manifest-file <unzipped>/manifest.json --telemetry false
   ```
   plus `python -m pytest mi_agent_api/tests/test_copilot_package.py -q`
   (asserts the packaged declarativeAgent/ai-plugin against schema versions,
   the three-action surface, and placeholder absence). A signed-in
   `atk validate --package-file <zip>` run against the Developer Portal
   service is recommended from any Trakt M365 account before hand-over.
5. Send the client administrator: the zip, the admin-consent URL
   (`https://login.microsoftonline.com/<clientTenant>/adminconsent?client_id=<entra_app_id>`),
   and Part 3 below.

## Part 3 — Client administrator actions (the complete list)

1. **Upload the app**: Microsoft 365 admin center → **Settings → Integrated
   apps → Upload custom apps** → upload `trakt-copilot-agent-<client>.zip`.
2. **Assign users**: choose the users/groups permitted to use the Trakt agent.
3. **Grant admin consent**: open the admin-consent URL provided by Trakt and
   accept (this consents Trakt's multi-tenant API app — scope
   `access_as_user` — for the organisation).
4. Done. Users open **Microsoft 365 Copilot → Agents → Trakt** (may take a few
   minutes to appear) and can immediately ask portfolio questions or fetch the
   latest investor deck / canonical loan tape. On first use each user clicks
   one sign-in confirmation; subsequent use is silent SSO.

The client administrator does **not**: edit any manifest, open the Teams
Developer Portal, register OAuth clients, build packages, touch Azure
resources, or handle any Trakt secret.

---

## Acceptance gate before merge/rollout

Structural validation (Agents Toolkit manifest validation + official v1.7/v2.4
schema validation + the repo test suite) is automated and passing. The final
gate — uploading the generated sample package to a test Microsoft 365 tenant
and confirming it is recognised as a declarative agent — **requires a live
tenant and must be performed by a Trakt operator** (Part 2 step 4's signed-in
`atk validate --package-file`, then Part 3 steps in the test tenant). Do not
merge to `main` until that upload has succeeded.
