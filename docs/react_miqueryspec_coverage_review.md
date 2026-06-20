# React ↔ MIQuerySpec Coverage Review

**Question:** Can the current React architecture render, preserve, and explain
**every valid MIQuerySpec response** produced by `MIQuerySpec`,
`run_mi_agent_query`, the semantic registry, the chart factory, and the
`/mi/query` API?

**Scope:** architecture/compatibility review. Source verified against code, not
mock data: `mi_agent/mi_query_spec.py`, `mi_query_validator.py`,
`mi_query_executor.py`, `mi_chart_factory.py`, `mi_agent_workflow.py`,
`mi_agent_api/{app,catalogue,adapters}.py`, and the React `domain/`, `api/`,
`components/artifacts/` modules.

---

## 0. Update — Plotly rendering implemented (2026-06-20)

The highest-priority gap below ("native heatmap/treemap via a lazy Plotly
renderer") is now **closed**:

- The API adapter emits a **chart artifact carrying `source.figure`** (the raw
  Plotly figure) for **every** chart type, including `heatmap`/`treemap`, plus
  the result table. The figure is included only when it has traces; empty
  results stay table-only.
- React routes chart artifacts **Plotly-first**: `source.figure` present →
  lazy-loaded `PlotlyArtifactView`; else Recharts (bar/line/area/scatter/bubble/
  waterfall); else an explicit `unsupported` state. Plotly ships in a **separate
  async chunk** (`plotly.js-dist-min`, ~1.42 MB gzip) so the initial bundle is
  unchanged (~188 KB gzip).
- **heatmap and treemap are now faithfully rendered** whenever the backend
  provides a figure. The matrices in §2/§4 marked them "degraded → table"; they
  are now "✅ via Plotly" when a figure is present (table still emitted alongside).

Remaining: a slimmer custom Plotly partial bundle, and the deterministic parser
needs materialised bucket columns (e.g. `age_bucket`) in the demo dataset to
exercise heatmap live.

---

## 1. Executive conclusion

**Partially — and now closer.** After the minimal fixes in this change set, the
React layer correctly handles the **common, end-to-end paths** (bar, line,
scatter, bubble, table-only, summary→KPI, validation failures, runtime/network
errors) and **never silently drops** an output it cannot draw. It is **not yet**
a universal MIQuerySpec renderer: `heatmap` and `treemap` are **degraded**
(rendered as their underlying table, with the native type + raw Plotly figure
preserved), and there is no native Plotly path, so any chart whose meaning lives
in the Plotly figure rather than the result table is not drawn faithfully.

Two structural truths make "partial" the honest answer:

1. The agent's chart fidelity lives in a **Plotly figure**; React renders from
   the **result table** via Recharts. For `bar/line/scatter/bubble` the table is
   sufficient (lossless on data). For `heatmap/treemap` it is not a faithful
   visual without a Plotly renderer.
2. The MI Agent flow only emits **chart + table (+ validation)**. `risk` and
   `scenario` artifacts in the React app are **demo-only** (separate Python
   engines, not wrapped by `/mi/query`). They are not part of MIQuerySpec
   coverage and are excluded from this matrix.

Before this change set, `scatter`/`bubble` were **broken** (the adapter built a
single categorical series instead of x/y/size). That is now fixed.

---

## 2. MIQuerySpec capability matrix

Validity rules are taken from `mi_query_validator._check_chart_structure` and
`_check_slot`. "Renderable" = produces a faithful visual in React today.

| chart_type | Required slots (valid spec) | Aggregation | top_n | Executor `result_type` | React render |
| --- | --- | --- | --- | --- | --- |
| `bar` | `dimension` or `x` (dimension/date) **and** (`metric` or count agg) | any allowed on metric; `count`/`count_distinct` always ok | ✅ (grouped) | `table` | ✅ native |
| `line` | `x` = date/cohort/trend **and** `metric` | any | ❌ | `table` | ✅ native |
| `scatter` | numeric `x` **and** numeric `y` | n/a (loan-level) | ❌ | `loan_level` | ✅ native (fixed) |
| `bubble` | numeric `x`, `y`, **and** `size` | n/a (loan-level) | ❌ | `loan_level` | ✅ native (fixed) |
| `heatmap` | ≥2 dimensions **and** (numeric intensity or count agg) | count or metric | ❌ | `table` | ⚠️ degraded → table |
| `treemap` | ≥1 hierarchy/dimension **and** (numeric size or count agg) | count or metric | ✅ | `table` | ⚠️ degraded → table |
| `none` | intent must be `summary`/`table` (not `chart`) | any | (table only) | `summary`/`table` | ✅ KPI / table |

Cross-cutting spec rules that the matrix depends on:

- **weighted_avg** requires a `weight_field` (spec or registry default
  `current_outstanding_balance`). Validator error otherwise.
- **top_n** is valid only for grouped outputs (`bar`, `table`, `treemap`).
- **filters** (`spec.filters`) are applied server-side in the executor; they are
  not yet surfaced as a React control (NL-driven only).
- **Ambiguous bare dimensions** (`stage, portfolio, region, rate, balance`) are
  rejected by the validator and must be resolved to concrete fields.

---

## 3. API response shape inventory (`/mi/query` → adapter)

Every shape `run_mi_agent_query` can return, and how the API/adapter expresses it:

| Response shape | Origin | API envelope | Artifacts emitted |
| --- | --- | --- | --- |
| Chart + table (bar/line) | grouped/line executor + chart factory | `ok:true` | `chart` + `table` |
| Chart + table (scatter/bubble) | loan-level executor + chart factory | `ok:true` | `chart` (x/y/size) + `table` |
| Table-only (heatmap/treemap) | grouped, chart not renderable | `ok:true` + warning | `table` (carries `nativeChartType` + `figure`) |
| Summary (`chart_type=none`/`intent=summary`) | `_execute_summary` | `ok:true` | `kpi` |
| Table intent (`intent=table`) | grouped/summary | `ok:true` | `table` |
| Empty result | executor returns 0 rows | `ok:true` | `table` (no rows); no chart |
| Invalid spec (bad dimension/metric/structure) | validator fails | `ok:false` | `validation` (blockers) |
| Aggregation error (e.g. weighted_avg w/o weight) | validator | `ok:false` | `validation` |
| Parser ambiguity | deterministic parser / interpreter | `ok:false` or clarification text | `validation` / narrative |
| Unsupported intent/chart combo | executor raises | `ok:false`, `error` set | none (error surfaced in `answer`) |
| No data available | API data source missing | `ok:false` | none; validation-style error |
| Backend/network down | transport | thrown `AgentError` | — (chat error + retry) |
| Plotly figure metadata | chart factory `to_json().figure` | carried in `source.figure` | on `chart` and on fallback `table` |
| Warnings (percent-scale, sampling, fallback) | executor/chart/adapter | `warnings[]` | shown on card + chat |
| Assumptions | **not emitted by the agent** | `assumptions: []` | (schema parity only) |

Notes:
- `interpreted` is a **dict** from `describe_spec`; the adapter flattens it to a
  human string for `answer`/`interpreted`.
- The MI Agent does **not** produce narrative "assumptions"; the field is kept
  empty for schema parity (the mock client populates it for demo realism).

---

## 4. Renderer coverage matrix

| chart_type | Native React renderer | Preserves Plotly figure | Shows query_result table | Empty state | Warnings/errors | Source/lineage | Native vs render type distinguished |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bar` | ✅ | ✅ (`source.figure`) | ✅ | ✅ | ✅ | ✅ | ✅ (`nativeChartType`) |
| `line` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `scatter` | ✅ (fixed) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `bubble` | ✅ (fixed) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `heatmap` | ❌ → table | ✅ (on table) | ✅ | ✅ | ✅ | ✅ | ✅ |
| `treemap` | ❌ → table | ✅ (on table) | ✅ | ✅ | ✅ | ✅ | ✅ |
| `none` | ✅ (KPI/table) | n/a | ✅ | ✅ | ✅ | ✅ | n/a |

Render-only variants (`area`, `waterfall`) are **not** MIQuerySpec chart types;
they exist solely for mock/demo visuals and are explicitly segregated in
`domain/mi.ts` (`MI_CHART_TYPES` vs `CHART_TYPES`). The adapter never emits them.

---

## 5. Artifact contract assessment

The frontend already uses a **canonical artifact union**, not raw
`chart_result`/`query_result`. The Python adapter performs the
normalization, so React is **not coupled to Plotly or pandas internals** — it
consumes `{type, …, source}` artifacts. Strengths:

- Discriminated union with type guards; `ArtifactRenderer` dispatches by type.
- Lineage on every artifact (`source.engine/state/spec/asOf/portfolio`).
- `mock` disclosure flag distinguishes demo vs live.

Gaps the contract should close (addressed or recommended below):

- **Unsupported outputs** had no first-class representation (silently became a
  table or a generic div). → **Fixed:** added an `unsupported` artifact type +
  guard + renderer; the `ArtifactRenderer` default now routes unknown types to
  it instead of a bare string.
- **Native vs render chart type** was not captured. → **Fixed:**
  `source.nativeChartType` + `source.figure` now carried on charts and on
  fallback tables.
- **Errors/warnings/assumptions** are response-level (and validation is an
  artifact). A dedicated `error`/`warning`/`narrative` artifact type is **not**
  required today but is included in the recommended schema for symmetry.

Conclusion: the contract is **adequate and correctly decoupled**; it needs
enrichment (below), not replacement.

---

## 6. Recommended canonical artifact schema

A superset that can represent every MIQuerySpec output without leaking backend
internals. (Current code implements the core; ⊕ marks recommended additions.)

```ts
type ArtifactType =
  | "kpi" | "chart" | "table" | "validation"
  | "unsupported"            // ✅ added
  | "risk" | "scenario"      // demo-only (separate engines)
  | "narrative";             // ⊕ optional, for answer/explanation blocks

interface ArtifactSource {
  engine: string;                 // "mi_agent.workflow" | ...
  label: string;
  state?: MIState;
  spec?: Partial<MIQuerySpec>;    // resolved MIQuerySpec echo
  resolvedFields?: Record<string, ResolvedField>;
  asOf?: string;                  // reporting date
  portfolio?: string;             // portfolio context
  nativeChartType?: MIChartType;  // ✅ backend-native chart type
  figure?: unknown;               // ✅ raw Plotly payload
  inputQuestion?: string;         // ⊕ original NL question
}

interface ArtifactBase {
  id: string;
  type: ArtifactType;
  title: string;
  description?: string;
  source: ArtifactSource;         // lineage + portfolio + reporting date + spec
  createdAt: string;              // generated timestamp
  mock: boolean;
  warnings?: string[];
  assumptions?: string[];         // ⊕ per-artifact (today: response-level)
  errors?: string[];             // ⊕ per-artifact errors
  pinned?: boolean;
}

interface ChartArtifact extends ArtifactBase {
  type: "chart";
  chartType: ChartType;           // frontend renderer type
  // source.nativeChartType = backend-native MIQuerySpec type
  // source.figure          = raw Plotly payload
  xKey: string;                   // normalized chart data ...
  series: ChartSeries[];
  rows: Array<Record<string, string | number>>;
  valueFormat?: ValueFormat;
}

interface TableArtifact extends ArtifactBase { type: "table"; columns: TableColumn[]; rows: Row[]; }
interface KPIArtifact extends ArtifactBase { type: "kpi"; kpis: KPI[]; }
interface ValidationArtifact extends ArtifactBase { type: "validation"; summary: {...}; issues: ValidationIssue[]; }
interface UnsupportedArtifact extends ArtifactBase { type: "unsupported"; reason: string; } // ✅
```

Key distinctions the schema makes explicit:
- **backend-native chart type** (`source.nativeChartType`) vs **frontend
  renderer type** (`chartType`).
- **raw Plotly payload** (`source.figure`) vs **normalized chart data**
  (`xKey/series/rows`) vs **table data** (`columns/rows`).

---

## 7. Gap analysis

**Supported today (end-to-end, faithful):**
- `bar`, `line`, `scatter`, `bubble` charts (+ result table).
- `none`/summary → KPI; `intent=table` → table; empty result → empty table.
- Invalid spec / aggregation error / ambiguity → `validation` artifact, `ok:false`.
- Runtime/network/backend-down → retryable error in chat; canvas keeps last good artifacts.
- Lineage, warnings, mock disclosure, Plotly-figure preservation.

**Partially supported (degraded but lossless on data):**
- `heatmap`, `treemap` → shown as the underlying table with `nativeChartType` +
  `figure` preserved and a warning. Data is complete; the *visual* is not.

**Not yet supported:**
- Native `heatmap`/`treemap` rendering (no Recharts equivalent wired; no Plotly
  path).
- Faithful re-render of arbitrary Plotly figures (the figure is carried but not
  drawn).
- Explicit `filters`/sort controls in the UI (server applies NL-derived filters
  only).
- Multi-metric grouped charts (adapter emits a single value series for grouped
  results; a bar with two metrics would show one).

**Response/error states fully handled:** invalid spec, no-data, network error,
unsupported type (now explicit), warnings. **Thin:** parser *clarification*
(returned as `ok:false`/narrative; no dedicated clarification UI).

**Chart types needing native renderer work:** `heatmap`, `treemap` (highest
value); multi-series grouped `bar`.

**Where Plotly rendering may be preferable to Recharts translation:** `heatmap`
and `treemap` — their layout/encoding is awkward to reconstruct losslessly from
the table; a lazy-loaded Plotly artifact renderer reading `source.figure` would
be more faithful than a Recharts re-implementation.

**Where API response normalization should happen:** in the **Python adapter**
(`mi_agent_api/adapters.py`) — it already owns Plotly→artifact translation; keep
all backend-shape knowledge there so React stays renderer-only.

**Where frontend state may break down:**
- `useWorkspace` replaces non-pinned artifacts each turn; large `loan_level`
  tables (sampled, but still up to the executor cap) are persisted to
  `localStorage` and could exceed quota — persistence should cap/trim payloads.
- No pagination/virtualization for large tables.

---

## 8. Recommended next implementation steps (priority order)

1. **Native heatmap/treemap** — add a lazy-loaded Plotly artifact renderer that
   draws `source.figure` (keeps bundle light via dynamic import); promotes
   `heatmap`/`treemap` from degraded → faithful. *(Highest priority.)*
2. **Multi-metric grouped charts** — let the adapter emit multiple series when a
   grouped result has >1 value column.
3. **Catalogue-driven validation in the UI** — use `/mi/catalogue` to pre-validate
   dimension/measure/aggregation choices before submit (fewer round-trips).
4. **Persistence guardrails** — cap/trim `loan_level` rows before writing to
   `localStorage`; add table virtualization.
5. **Filters/sort controls** — surface `spec.filters` + `top_n` as UI controls,
   sent through `AgentRequest`.
6. **Clarification UX** — render parser clarification questions as an
   interactive prompt rather than a plain `ok:false` message.

---

## Appendix — changes made in this review (minimal, low-risk)

- `mi_agent_api/adapters.py`: fixed `scatter`/`bubble` to emit x/y/size series
  from the resolved spec; added `source.nativeChartType`; preserved
  `source.figure` on `heatmap`/`treemap` fallback tables.
- `frontend/.../domain`: added `unsupported` artifact type + `isUnsupportedArtifact`
  guard; added `MIChartType`, `source.nativeChartType`, `source.figure`.
- `frontend/.../components/artifacts`: added `UnsupportedArtifactView`; routed the
  `ArtifactRenderer` default through it; added the card icon + canvas label.
- Tests: backend `test_adapters.py` (8 cases across all chart types + empty);
  frontend guard + unsupported/scatter renderer tests.

Validation: backend **13 passed**; frontend **35 passed**; `npm run build` ✅.
