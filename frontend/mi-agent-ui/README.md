# Trakt · MI Agent — React Front End

The production-oriented React **presentation & orchestration layer** for the
existing Python MI platform. A two-pane analytical workspace: an analyst asks
natural-language MI questions on the left; the agent returns typed **artifacts**
(KPIs, charts, tables, validation, risk, scenario) rendered on the right.

```
User Question → AgentClient → MI Agent → Analytics Engine → Artifact Schema → React Renderer
```

This is a **parallel front end**. It does not touch or replace the Streamlit MI
dashboard (`mi_agent/streamlit_mi_agent.py`, `analytics/streamlit_app_erm.py`).
Responses are currently served by a `MockAgentClient`; the contract mirrors
`mi_agent.mi_agent_workflow.run_mi_agent_query`, so a real backend is a drop-in.

See [`docs/react_mi_alignment_review.md`](../../docs/react_mi_alignment_review.md)
for the full Streamlit→React capability/chart/dimension/measure/artifact mapping
and the artifact-schema rationale.

## Running it

Requires Node 18+ (developed on Node 22).

```bash
cd frontend/mi-agent-ui
npm install
npm run dev      # http://localhost:5173
npm run build    # type-check (tsc) + production build
npm test         # vitest suite (intent routing, guards, response, rendering)
npm run lint     # tsc --noEmit
```

## Architecture

The app has clear, backend-ready boundaries:

| Layer | Path | Responsibility |
| --- | --- | --- |
| **Domain** | `src/domain/` | Strong types + guards. `mi.ts` (states/aggregations/chart types mirroring `MIQuerySpec`), `artifacts.ts` (the artifact union + payloads), `agent.ts` (`AgentRequest`/`AgentResponse`/contexts), `guards.ts`. |
| **API** | `src/api/` | `AgentClient` interface; `MockAgentClient` (latency + simulated failures). UI talks **only** to `AgentClient`. `createAgentClient()` is the single swap point for a future `HttpAgentClient`. |
| **Data** | `src/data/` | `catalog.ts` (dimensions/measures/portfolios), `mockArtifacts.ts` (lineage-carrying artifact builders), `mockResponses.ts` (intent routing + narratives). |
| **State** | `src/state/` | `useWorkspace` (orchestration: context, chat, canvas, loading/error/empty) + versioned `localStorage` persistence. |
| **Components** | `src/components/` | `AppShell` → `HeaderBar`/`AgentChatPanel`/`ArtifactCanvas`. `artifacts/ArtifactRenderer` dispatches by type to per-type views. `states/` holds empty/error/loading. |

### Artifact-driven UI

The canvas is driven by artifact **type**, not hardcoded dashboards. Each
artifact carries lineage (`source`: engine, MI state, spec, `asOf`, portfolio)
and a `mock` flag (surfaced as a disclosure badge). Adding an artifact type means
adding a payload type + guard + view and registering it in `ArtifactRenderer` —
nothing in the canvas/card layer changes.

Supported types: `kpi`, `chart` (bar/line/area/scatter/bubble/waterfall),
`table`, `validation`, `risk` (RAG concentration limits + grade-migration
matrix), `scenario` (balance run-off / LTV / NNEG projection).

### Mocked agent intents

`classifyIntent` (keyword routing, mirroring
`mi_agent/interpreter/deterministic.py`) maps a question to one of:
`portfolio_overview`, `concentration_risk`, `pipeline`, `static_pools`,
`risk_monitoring`, `scenario`, `validation`, or `unknown` (default dashboard).

## Connecting to the MI Agent API

A FastAPI backend that wraps the real MI Agent lives at
[`mi_agent_api/`](../../mi_agent_api). The React app talks to it via
`HttpAgentClient` (same `AgentClient` interface as the mock).

```bash
# 1. start the backend (from repo root), pointing it at the governed datasets
pip install -r requirements.txt -r mi_agent_api/requirements.txt

export MI_AGENT_CLIENT_ID=ERE                                  # the TENANT
export MI_AGENT_PLATFORM_CANONICAL=platform/ERE/latest/platform_canonical_typed.csv
export MI_AGENT_PIPELINE_ROOT=pipeline                         # funded + pipeline
uvicorn mi_agent_api.app:app --reload --port 8000

# 2. start the UI — it proxies to the API same-origin (see vite.config.ts)
cd frontend/mi-agent-ui
npm run dev
```

`.env.development` already sets `VITE_AGENT_API_URL=/`, so `npm run dev` is live
against the backend with no extra configuration.

> **Do not set an absolute `VITE_AGENT_API_URL` in Codespaces or behind any
> forwarded port.** `http://localhost:8000` resolves in the *browser*, which is
> not on the API host, so the request never reaches uvicorn and the forwarding
> proxy answers instead — surfacing as `MI Agent API returned 404`. Keep the
> relative `/` base and let the dev-server proxy forward. Point the proxy
> somewhere else with `VITE_PROXY_TARGET=http://host:port` if the API is not on
> `:8000`.

Client selection (`src/api/index.ts`):

- `VITE_AGENT_API_URL` set → **HttpAgentClient** (live Python MI Agent).
- unset (or `VITE_AGENT_MODE=mock`) → **MockAgentClient** (demo mode).

The app **always builds and runs without the backend** (mock fallback). API and
network errors surface as a retryable error message in chat (and the canvas
keeps the last good artifacts). The API returns artifacts already in this
schema, so `HttpAgentClient` is a thin transport + envelope translation — no
component changes were needed to go live.

## Notes / assumptions

- Figures are **illustrative** of a UK Equity Release Mortgage (ERM) portfolio.
- Brand/chart palette (`src/lib/theme.ts`) mirrors `analytics/charts_plotly.py`
  and `mi_agent/mi_chart_factory.DEFAULT_THEME`.
- Pinned artifacts persist across turns and float to the top; per-turn artifacts
  replace the previous turn's (except pinned).
- Workspace state (context, chat, artifacts) persists in `localStorage` under a
  versioned key.
