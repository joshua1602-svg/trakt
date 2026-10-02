# Trakt Teams app — Copilot agent + proactive notifications

**One Teams application, two capabilities.** The declarative agent (Microsoft
365 Copilot) and the notification bot ship in the same package, under the same
app id and the same branding. Teams manifest v1.19 carries `copilotAgents` and
`bots` side by side, so no second application is required — and
`package_agent.py` refuses to build a package that has lost either one.

```
manifest.json
├── copilotAgents.declarativeAgents  →  declarativeAgent.json  →  ai-plugin.json
│                                                              →  trakt-copilot-openapi.yaml
└── bots[0]                          →  personal scope, notification-only
```

---

## Building the package

```bash
# Development / sideload — ${{...}} tokens are substituted by the toolchain:
python deploy/copilot-agent/package_agent.py

# Release build — insists every token has already been substituted:
python deploy/copilot-agent/package_agent.py --require-resolved

# Copilot only — leave the notification bot out of the archive (the repository
# manifest keeps it). Needs only the plugin's OAuth registration id:
python deploy/copilot-agent/package_agent.py --copilot-only --require-resolved \
    --oauth-config-id <registration id from the Teams developer portal>
```

The build always fails if the declarative agent is missing, the bot is missing,
or the bot declares a scope other than `personal`. `--require-resolved`
additionally rejects a literal `${{TEAMS_BOT_APP_ID}}`, because a manifest
shipped with the token installs cleanly and then fails every proactive send,
days later, in production. `${{TEAMS_BOT_APP_ID}}` is a per-deployment value and
lives in the repository unresolved, exactly as `${{OAUTH2_CONFIGURATION_ID}}`
already does in `ai-plugin.json`.

---

## Azure resources

| Resource | Why | New? |
| --- | --- | --- |
| Entra app registration (multi-tenant) | Bot identity; external-tenant consent | Yes |
| Azure Bot resource | Registers the Teams channel + messaging endpoint | Yes |
| App setting `TRAKT_TEAMS_BOT_APP_ID` | Bot app (client) id | Yes |
| App setting `TRAKT_TEAMS_BOT_APP_PASSWORD` | Bot secret — **Key Vault reference** | Yes |
| Blob container `trakt-state` | Recipients, batches, outbox | Existing |
| App Service `trakt-mi-api` | Hosts `/v1/teams/bot/messages` | Existing |
| Function App timer | Delivery worker | Existing app |

The messaging endpoint to register on the Azure Bot resource is:

```
https://<trakt-mi-api-host>/v1/teams/bot/messages
```

---

## App settings

| Setting | Purpose |
| --- | --- |
| `TRAKT_TEAMS_NOTIFICATIONS` | Master kill switch. Overrides the config file without a redeploy — reach for this in an incident. |
| `TRAKT_TEAMS_BOT_ENABLED` | Mounts the messaging endpoint. Off ⇒ the route does not exist. |
| `TRAKT_TEAMS_BOT_APP_ID` | Bot app id; also the expected inbound token audience. |
| `TRAKT_TEAMS_BOT_APP_PASSWORD` | Bot secret. Key Vault reference; never in a config file, never logged. |
| `TRAKT_TEAMS_BOT_AUTH_MODE` | `botframework` (default, fail closed) or `disabled` (local dev only). |
| `TRAKT_TEAMS_TRAKT_TENANT` | The Trakt tenant this deployment serves. |
| `TRAKT_COPILOT_WORKSPACE_BASE_URL` | Deep-link base. Shared with Copilot Workspace promotion, so both cannot point at different environments. |

Delivery behaviour (scope, message toggles, recommendation level, item caps,
recipients, retries) lives in `config/mi/teams_notifications.yaml`. **No
threshold belongs there** — materiality stays in `config/mi/insights.yaml`, and
limits stay in the governed concentration-test configuration.

---

## Pilot onboarding

1. **Admin installs** the Trakt app for the named pilot user (Teams admin
   centre, or sideload during the pilot).
2. **Teams sends an activity**; the endpoint captures the conversation
   reference. The user is now *addressable* — and deliberately **not**
   authorised: no portfolio contexts, notifications off.
3. **An operator authorises** the mapping:

   ```bash
   python -m trakt_notifications.cli recipients ERE
   python -m trakt_notifications.cli authorise ERE <recipient_id> \
       --contexts total --by <operator>
   ```

4. **Enable delivery** — set `enabled: true`, or `TRAKT_TEAMS_NOTIFICATIONS=1`.

Installing the app can never, by itself, start a feed of portfolio data to
whoever installed it. Step 3 is not optional.

---

## Operating it

```bash
python -m trakt_notifications.cli outbox ERE --failures   # what needs attention
python -m trakt_notifications.cli diagnose ERE            # structured report
python -m trakt_notifications.cli show ERE <batch_id>     # what was said
python -m trakt_notifications.cli deliver ERE             # run a pass by hand
```

**Run one delivery worker.** Blob writes are last-writer-wins, so two workers
racing for the same outbox item would both believe they hold it. The send-time
idempotency check makes that harmless rather than duplicating a message, but
single-worker is the supported configuration.

---

## External-tenant installation

The client's Teams administrator must:

1. accept the app package (custom app upload, or Teams admin centre);
2. consent to the bot for their tenant — the Entra app is multi-tenant, and the
   bot requests no Graph permissions in v1;
3. allow the app for the pilot users under their app-permission policy.

The existing Copilot OAuth registration is unchanged, so an existing Copilot
installation continues to work through the upgrade.

---

## Not in v1

Teams channels, group chats, email, SMS, self-service subscriptions,
interactive mitigation actions, Graph-based mass installation, editing a
delivered card on correction (a clearly labelled correction message is sent
instead), and any message type beyond the two required ones.

---

## Production prerequisites (learned on the first client go-live)

These are **Azure / Entra settings that live outside the repository**. A new
environment, or a rebuilt `trakt-mi-api`, needs every one of them or Copilot
fails in ways that look like application errors.

1. **Exempt the Copilot paths from App Service platform authentication.**
   `trakt-mi-api` runs Easy Auth with *Require authentication → HTTP 401*. That
   layer rejects Copilot's call before Trakt sees it (an **empty-body 401** with
   `WWW-Authenticate: Bearer realm=…` and no application log line), because
   Copilot's token is for the Trakt Copilot API app, not the dashboard app. The
   Copilot routes validate their own bearer token, so only these paths are
   exempted and everything else stays protected:

   ```bash
   SUB=$(az account show --query id -o tsv)
   URL="/subscriptions/$SUB/resourceGroups/<rg>/providers/Microsoft.Web/sites/trakt-mi-api/config/authsettingsV2?api-version=2022-03-01"
   az rest --method get --url "$URL" > auth-backup.json          # keep this
   jq '{properties: .properties} | .properties.globalValidation.excludedPaths =
       ["/v1/copilot/mi/query","/v1/copilot/artifacts/latest","/v1/copilot/artifacts/download"]' \
       auth-backup.json > auth-new.json
   az rest --method put --url "$URL" --body @auth-new.json
   ```

   Saving restarts the app. **Verify:** an unauthenticated
   `curl -i -X POST https://<host>/v1/copilot/mi/query …` must answer with
   `Server: uvicorn` and a **JSON** 401 (`A bearer token is required.`). An empty
   401 means the platform layer is still blocking it. Never switch platform
   authentication off or to "allow anonymous": the dashboard relies on it.

2. **App settings** on `trakt-mi-api` (see `deploy/trakt-mi-api/app_settings.example.json`):
   `TRAKT_COPILOT_AUTH_MODE=entra`, `TRAKT_COPILOT_ENTRA_AUDIENCE` (the bare app
   id **and** `api://<app-id>`), `TRAKT_COPILOT_REQUIRED_SCOPE=Trakt.Copilot`,
   `TRAKT_COPILOT_DOWNLOAD_SIGNING_KEY` (generate a fresh one),
   `TRAKT_COPILOT_PUBLIC_BASE_URL`, `TRAKT_COPILOT_WORKSPACE_BASE_URL`, and the
   client's directory in `TRAKT_COPILOT_ENTRA_TENANT_ID` (shared with the
   dashboard allow-list). **Never** set `TRAKT_COPILOT_AUTH_MODE=disabled` in
   production; to switch Copilot off quickly, clear
   `TRAKT_COPILOT_ENTRA_AUDIENCE` (the routes answer 503; the dashboard is
   unaffected).

3. **Teams developer portal → OAuth client registration**: client id = the
   Trakt Copilot API app, scope `api://<app-id>/Trakt.Copilot`,
   authorization/token/refresh endpoints on `/common/oauth2/v2.0/…`, PKCE off,
   *request body parameters*. Add the redirect URL the portal shows to the
   Entra app. The registration id goes in at **build** time
   (`package_agent.py --oauth-config-id`), never into the repository.

4. **Consent.** The `Trakt.Copilot` scope is *admins only*. The first sign-in in a
   tenant must be approved by an administrator ("consent on behalf of the
   organisation"). Note the portal does not offer an app's own API under
   *API permissions → My APIs*, so a static consent link may not cover it.

5. **Publishing.** An organisation catalogue that already holds a version will
   not accept a changed package under the same version: bump `version` (manifest,
   OpenAPI `info.version` and header default, `copilot_package.py`) every time.
   Installation by admin may be unavailable for an agent-only app; users add it
   from the agent store ("Built by your org").

**Diagnosing a Copilot failure:** open `trakt-mi-api → Monitoring → Log stream`
and ask the question. No `POST /v1/copilot/mi/query` line means the call never
reached Trakt (platform authentication, or the Teams OAuth sign-in); a line with
401/403/503 means Trakt answered, and the status says why. In Copilot, `-developer on`
shows the raw request and response of each plugin call.
