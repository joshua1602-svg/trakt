# Trakt Microsoft 365 Copilot — v2 release

Builds on [`copilot_v1_implementation.md`](./copilot_v1_implementation.md). Three
workstreams, all preserving the existing MI query architecture, analytical
engine, business logic, React Workspace and Copilot interaction model. This is
**not** a generic capability/plugin/execution framework — it is the specific
changes below.

Package version bumped **1.0.0 → 2.0.0**. This release changes the shipped
package (manifests, OpenAPI, declarative-agent instructions), so it **requires a
one-time re-upload at every client tenant**. See "Re-upload" at the end.

---

## Workstream 1 — Generic artifact registry

The downloadable-artifact axis is now a single generic action over a server-side
registry, replacing the two per-artifact functions.

- **Package surface:** `getLatestInvestorDeck` + `getLatestCanonicalTape` →
  **one** `getArtifact(artifactType)` function (`ai-plugin.json`), one generic
  OpenAPI path `GET /v1/copilot/artifacts/latest?artifactType=…`
  (`trakt-copilot-openapi.yaml`). `artifactType` is an **open string with no
  enum**, so new types need no package change.
- **Server registry:** `mi_agent_api/copilot_artifacts.py` maps
  `artifact_type → resolver`. `copilot_actions.get_artifact` looks up the type,
  resolves it, and returns the same `CopilotArtifactInfo` as before (behaviour
  preserved for `investor_deck` and `canonical_tape`). The signed-download route
  resolves the file through the same registry (no more `if deck / if tape`).
- **Types shipped:** `investor_deck`, `canonical_tape` (migrated), plus
  `mapping_report`, `validation_report`, `esma_xml` (new). Each new resolver
  reuses the existing latest-output conventions: an explicit per-artifact env
  override (`MI_AGENT_MAPPING_REPORT` / `MI_AGENT_VALIDATION_REPORT` /
  `MI_AGENT_ESMA_XML`), else the conventional filename in the platform `latest/`
  directory.
- **Adding an artifact type is now one `register(ArtifactSpec(...))` call** in
  `copilot_artifacts.py` — a server deploy, no package artefact touched. The
  rule-10 denylist that named ESMA XML / mapping / validation reports as
  "not available" is removed from the declarative-agent instructions.

**Remaining manifest-level hardcoding after this release:** none that blocks a
new artifact type. The only per-type text left in the package is *advisory* —
example type names in the `getArtifact` description and instructions to help the
model pick a good `artifactType`. A new type resolves server-side without
editing them; refreshing those examples is optional and only improves routing.

## Workstream 2 — Intelligent Workspace promotion

`askTraktMi` answers now carry a `classification` and, when it helps, a
`workspaceUrl` deep link into the React MI Workspace.

- **Classifier:** `mi_agent_api/copilot_workspace.py` — a pure, deterministic
  read over signals the engine already emitted (no new analysis):
  - `unmet` — `ok == false`, or a chart was requested but none produced, or the
    result was truncated.
  - `fully-served-complex` — a chart was produced, ≥ 15 result rows, multiple
    dimensions, multiple metrics, an exploratory intent, or a filtered result.
  - `fully-served-simple` — everything else.
- **Link policy:** only `fully-served-complex` and `unmet` carry a link. The URL
  is built from `TRAKT_COPILOT_WORKSPACE_BASE_URL` plus `client`, `run`, `q`
  (question) and `filters`, so the Workspace **restores context**. When the env
  var is unset, classification still runs and the link is simply suppressed.
- **React hydration:** `readUrlContext()` (`src/state/persistence.ts`) parses
  those params; `useWorkspace` prefers them over persisted localStorage for
  client/run selection and exposes the question as `initialQuestion`, which
  `AppShell` pre-fills into the composer. No router added; normal visits keep the
  existing localStorage behaviour.
- **Model guidance:** the declarative-agent instructions now tell the model to
  surface `workspaceUrl` when present (as the full chart / drill-through /
  filtering for complex, or the way to complete the request for unmet) and never
  to invent one when absent. `ai-plugin.json` maps the card `url` to
  `$.workspaceUrl`.

### Open risk — session hand-off (NOT implemented this release)

Opening a `workspaceUrl` restores query context but does **not** carry the
Copilot session/auth across. The React Workspace is served behind the platform's
Entra Easy Auth. The assumption is that a user already signed into the same
tenant in their browser passes Easy Auth's Entra SSO **silently**, so the link
lands in-context rather than at a sign-in screen. This is **unverified in
code**. **Manual tenant testing is required** — from Copilot (desktop, web and
mobile), same tenant and cross-device — before relying on seamless hand-off in
production. If silent SSO does not hold on a surface, users will hit an Entra
sign-in before the Workspace loads (context is still preserved through it).

## Workstream 3 — Package version telemetry

Every Copilot request now identifies which package version the tenant is running.

- The shipped package sends an `X-Trakt-Package-Version` header on every
  operation, defaulted in the OpenAPI to the version the package was built from
  (`2.0.0`).
- `mi_agent_api/copilot_package.py` holds the single source of truth
  (`COPILOT_PACKAGE_VERSION`, asserted equal to the manifest `version` and the
  OpenAPI `info.version` by the package test). `log_request` emits one line per
  request — `op=… package_version=… backend_package_version=…` — and appends
  `STALE_TENANT_PACKAGE` when the tenant's reported version differs from the
  backend's expected version. `askTraktMi` also echoes `packageVersion` in its
  response body.

---

## Files changed

**Server**
- `mi_agent_api/copilot_artifacts.py` *(new)* — the artifact registry + resolvers.
- `mi_agent_api/copilot_workspace.py` *(new)* — Workspace-promotion classifier + URL builder.
- `mi_agent_api/copilot_package.py` *(new)* — package-version telemetry.
- `mi_agent_api/copilot_actions.py` — generic `getArtifact` route, registry-backed download, WS2/WS3 wiring on `askTraktMi`; removed the two per-artifact routes and the tape helpers (moved into the registry).
- `mi_agent_api/app.py` — comment refresh only.

**Package (forces the re-upload)**
- `deploy/copilot-agent/ai-plugin.json` — two functions (`askTraktMi`, `getArtifact`); card `url` → `$.workspaceUrl`.
- `deploy/copilot-agent/trakt-copilot-openapi.yaml` — one generic artifact path; `X-Trakt-Package-Version` header parameter; `workspaceUrl`/`classification`/`packageVersion` on the answer schema; `info.version` 2.0.0.
- `deploy/copilot-agent/declarativeAgent.json` — generic `getArtifact` rule, Workspace-link rule, rule-10 denylist removed.
- `deploy/copilot-agent/manifest.json` — `version` 2.0.0; description refreshed.

**React**
- `src/state/persistence.ts` — `readUrlContext()`.
- `src/state/useWorkspace.ts` — URL-context hydration + `initialQuestion`.
- `src/components/AppShell.tsx`, `src/components/AgentChatPanel.tsx` — pre-fill composer from `initialQuestion`.

**Tests**
- `mi_agent_api/tests/test_copilot_package.py` — validates the two generic actions, registry completeness, open-string `artifactType`, no per-artifact hardcoding, and version-telemetry consistency.
- `mi_agent_api/tests/test_copilot_actions.py` — migrated to `getArtifact`; unknown-type 404; new-type resolution; classification + telemetry.
- `mi_agent_api/tests/test_copilot_artifacts.py` *(new)*, `mi_agent_api/tests/test_copilot_workspace.py` *(new)*, `frontend/mi-agent-ui/src/state/persistence.test.ts` *(new)*.

## Re-upload

This release changes the package, so **each client tenant's IT admin must
re-upload the v2 package once**. After that:

1. **Is this the final required re-upload for future artifact additions?** For
   *artifact additions*, yes — this is the last re-upload they require.
2. **Can new artifact types then be added server-side only?** Yes — register a
   resolver in `copilot_artifacts.py` and deploy; the package does not move.
3. **What would still require a future package update?** Only changes to the
   package itself: a new *action* (beyond `askTraktMi`/`getArtifact`), changes to
   OAuth scopes / `validDomains` / the API host, editing the advisory example
   type names or instruction wording, new conversation starters, or a manifest
   schema-version bump. None of these is triggered by adding an artifact type.
