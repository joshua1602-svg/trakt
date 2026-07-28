# Deployment checklist (Phase 8 §27)

Concise pre-onboarding configuration checklist. No secret values are recorded
here. Derived from the verified runtime, not from documentation.

## Identity / authentication (release blocker)

- [ ] **React**: deploy behind Azure Easy Auth / SWA so the platform injects and
      overwrites `X-MS-CLIENT-PRINCIPAL`. Set `WEBSITE_AUTH_ENABLED`. Without it,
      in Azure production, `require_trustworthy_platform_auth` fails closed
      (correct) — but the deployment then answers nothing, so this must be on.
- [ ] **Never** set `MI_AGENT_AUTH_ENABLED=false` in production — it injects a
      synthetic operator principal and bypasses auth. Leave unset/true.
- [ ] The `deploy/trakt-mi-api/app_settings.example.json` does **not** contain
      `WEBSITE_AUTH_ENABLED` / `MI_AGENT_CLIENT_ID` — the example is not a
      working authenticated production config. Supply both.

## Tenant → dataset binding (release blocker)

- [ ] Set `MI_AGENT_CLIENT_ID` to the client tenant, OR embed the client in
      `MI_AGENT_PLATFORM_URI` (`blob://…/platform/{client}/latest/…`). This is
      the tenant of record; it is never taken from a request.
- [ ] **Single-tenant launch condition (ASSURE-002 containment):** the
      onboarding/central-tape root (`MI_AGENT_ONBOARDING_OUTPUT_ROOT` or the blob
      platform root) must contain **only this client's data**. Dashboard GET
      routes other than `/mi/snapshot` are not yet tenant-bound; a shared root
      would expose them. Do not co-locate clients under one root until all GET
      routes are bound.
- [ ] `config/tenancy.yaml` is optional and absent by default (open namespace).
      For multi-tenant, provide it and verify `authorise_portfolio_access`
      denies cross-tenant selectors.

## Data source (release blocker)

- [ ] Point the MI agent at the promoted platform canonical or central tape via
      `MI_AGENT_PLATFORM_URI` / `MI_AGENT_PLATFORM_CANONICAL` /
      `MI_AGENT_CENTRAL_TAPE`. Do not rely on the `synthetic_demo` glob fallback.
- [ ] Confirm `TRAKT_RUNTIME_MODE` is unset or `production` (default is
      `production`, which refuses synthetic/fixture sources). Verify synthetic
      data cannot be selected in production (policy enforces this; the demo
      glob resolves to a **prior-period** file — do not use it as the source).
- [ ] Ensure the active dataset carries a **single reporting date** (a `…/latest`
      pointer). A combined multi-date tape now resolves as-of-latest with a
      disclosure (ASSURE-004), but a single-date cut is the intended shape.

## Currency (launch restriction)

- [ ] **Restrict the launch to a single currency.** Monetary aggregation is only
      guarded on the governed `/mi/query` point-in-time route (ASSURE-006);
      dashboard GET / geo / cohort sum sites do not suppress mixed currency.
      Confirm the client portfolio is single-currency (GBP for ER-UK).

## Copilot (defer)

- [ ] `TRAKT_COPILOT_AUTH_MODE=entra` (default) with
      `TRAKT_COPILOT_ENTRA_TENANT_ID` / `TRAKT_COPILOT_ENTRA_AUDIENCE`. Unset ⇒
      503 on all Copilot routes (fail closed).
- [ ] `TRAKT_COPILOT_DOWNLOAD_SIGNING_KEY` **must be set** for multi-worker
      deployments — unset generates an ephemeral per-process key, breaking
      signed-download redemption across workers and on restart.
- [ ] **Recommended: disable Copilot at launch** — 9 pre-existing Copilot
      artifact tests fail on the branch base and the signed-download redemption
      route carries no auth dependency (HMAC + TTL only). Validate separately.

## CORS / origins

- [ ] Set explicit permitted origins (no wildcard). The operator console adds
      CORS only when origins are configured; the MI API CORS is not a wildcard.

## Logging / audit sink

- [ ] Route the `trakt.audit` logger to a durable sink. Governed `/mi/query`
      emits audit for success/controlled-failure/blocked; dashboard GETs do not.

## Secrets

- [ ] No secrets are committed (verified: examples only). Provide signing keys,
      Entra config and storage credentials via app settings / Key Vault.
