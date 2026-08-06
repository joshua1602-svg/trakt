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
