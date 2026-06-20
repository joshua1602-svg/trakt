# Trakt · MI Agent — React Front End (First-Pass Prototype)

An enterprise-grade React front end for the **MI Agent**: a two-pane analytical
workspace where an analyst asks natural-language MI questions on the left and
the agent generates charts, tables, KPI strips and validation summaries on the
right.

This is a **parallel prototype**. It does not touch or replace the existing
Streamlit MI dashboard (`mi_agent/streamlit_mi_agent.py`). All data and agent
responses are **mocked** — the component and data shapes are deliberately close
to a future MI Agent API so wiring a live backend is mechanical.

> Reference for the analytical concepts: the Python MI stack under
> [`mi_agent/`](../../mi_agent) and [`analytics/`](../../analytics)
> (portfolio stratifications, pipeline / forward exposure, static pools,
> validation / governance). The brand palette (navy `#232D55`, periwinkle
> `#919DD1`) is taken from `analytics/charts_plotly.py`.

## Running it

Requires Node 18+ (developed on Node 22).

```bash
cd frontend/mi-agent-ui
npm install
npm run dev      # http://localhost:5173
```

Other scripts:

```bash
npm run build    # type-check (tsc) + production build to dist/
npm run preview  # serve the production build
npm run lint     # tsc --noEmit type check
```

## What it does (first pass)

- **Left — MI Agent chat panel** (fixed ~400px): chat history, starter prompt
  suggestions, streaming-style loading state, and assistant responses that carry
  a concise narrative, surfaced **assumptions**, and clickable links to the
  artifacts they produced.
- **Right — Artifact workspace**: a vertical **stack** or **tabbed** view of
  artifact cards. Each card has a title, description, source/timestamp context
  and actions: **pin**, **copy** (JSON), **download** (JSON), **collapse/expand**.
  Pinned artifacts persist across turns and float to the top.
- **Top header**: brand, portfolio selector, reporting-date selector,
  environment/status indicator (`Staging · Mock Data`) and a user area.
- **Landing state**: an executive KPI strip, balance-by-region chart, and a
  data-quality / governance summary.

### Mocked agent intents

The mock engine (`src/data/agentEngine.ts`) classifies a question via keyword
matching and returns a narrative + relevant artifacts:

| Intent | Example prompt | Artifacts |
| --- | --- | --- |
| `portfolio_overview` | "Show portfolio movement since last period" | KPI strip, funded vs. pipeline area chart |
| `concentration_risk` | "Explain top concentration risks" | Regional bar chart + concentration table |
| `pipeline` | "Generate pipeline bridge to £100MM securitisation size" | Waterfall bridge + funded/pipeline line |
| `static_pools` | "Show static pool performance by vintage" | Cumulative-redemption-by-vintage line |
| `validation` | "Summarise validation issues blocking reporting" | Validation summary (blockers/warnings/passes) |
| `unknown` | anything else | Generic response + default dashboard artifacts |

## Component hierarchy

```
App
└── AppShell                      # state + orchestration (mocked agent turns)
    ├── HeaderBar
    │   └── PortfolioSelector
    ├── AgentChatPanel
    │   ├── ChatMessage
    │   └── PromptSuggestions
    └── ArtifactCanvas
        └── ArtifactCard          # pin / copy / download / collapse
            ├── KPIGrid
            ├── ChartArtifact     # bar | line | area | waterfall (Recharts)
            ├── TableArtifact
            └── ValidationSummaryArtifact
```

Shared: `src/types/index.ts` (domain types), `src/data/mockData.ts`
(representative ERM UK figures), `src/lib/utils.ts` (formatting helpers),
`src/components/ui.tsx` (Card / Badge / IconButton primitives).

## Stack

- **Vite 6** + **React 18** + **TypeScript** (strict)
- **Tailwind CSS v4** (`@tailwindcss/vite`) with design tokens in `src/index.css`
- **Recharts** for charts, **lucide-react** for icons

## Wiring to a real backend

Replace the body of `runAgent()` in `src/data/agentEngine.ts` with a `fetch`
to the MI Agent API. The function already returns the `AgentResponse` shape
(`intent`, `narrative`, `assumptions`, `artifacts`) consumed by `AppShell`, and
the `Artifact` / `ArtifactData` union in `src/types` maps directly onto the
chart/table/KPI/validation outputs the Python stack already produces.

## Notes / assumptions

- All figures are **illustrative** and representative of a UK Equity Release
  Mortgage (ERM) portfolio; they are not real.
- Prompt suggestions hide after the first message to keep the chat compact.
- Per-turn artifacts replace the previous turn's (except pinned ones); this
  keeps the canvas focused on the current question.
