# Current-vs-forward concentration routing — capability proof

Base `73653f2` (code identical to `9d24d7c`), tree clean. **Nothing built.**

> ## STOP
>
> **The forward calculation cannot be constructed from existing governed
> capability.** The primitive the frozen bank names is real and deterministic,
> but it does not compute the requested answer, and the route that owns the
> approved limits never claims the question. Reporting rather than building, per
> the brief's own gate.

---

## 1. The named primitive: `mi_agent.risk_monitor.run_funded_vs_forecast`

Traced from public callable to deterministic output.

```
mi_agent/risk_monitor/__init__.py      exports run_funded_vs_forecast
mi_agent/risk_monitor/monitor.py:202   run_funded_vs_forecast(store, client_id, dimension, *, route, …)
   ├─ validate_risk_monitor_route(route)
   ├─ _frame_for(store, …, "total_funded",          …)   -> funded frame
   ├─ _frame_for(store, …, "total_forecast_funded", …)   -> forecast frame
   └─ concentration_movement(funded, forecast, dimension,
                             balance_col="forecast_contribution",
                             baseline_balance_col=…, kind="funded_vs_forecast")
mi_agent/risk_monitor/concentration.py:94   -> RiskMonitorResult(frame, issues, metadata)
```

**Inputs consumed:** two assembled state frames — `total_funded` and
`total_forecast_funded` — plus a dimension, thresholds and optional stage
probabilities. Forecast contribution comes from the `forecast_contribution`
column.

**Run on the live MI frames** (funded 640 rows, pipeline 8, forecast 645 via
`workspace.build_forecast_view_frame`), dimension `geographic_region_obligor`:

```
geographic_region_obligor  baseline_share  current_share  share_change  increasing  status_current
                 Scotland        0.167687       0.169871      0.002184        True           green
                    North        0.155729       0.154543     -0.001186       False           green
                 Midlands        0.146085       0.144972     -0.001112       False           green
                    …
               North West        0.000000       0.001298      0.001298        True           green
```

It computes, deterministically, with no issues raised.

### What it does and does not answer

| the brief asks | the primitive returns |
|---|---|
| current limit position | `baseline_share` — a **share**, not a position against any limit |
| projected/forecast position | `current_share` — the forecast share ✔ |
| expected breaches or worsening headroom | `share_change` / `increasing` ✔ for *worsening share*; `status_current` is RAG **against generic thresholds**, and there is **no headroom and no test** |

**It does not know the concentration tests.** `status_current` comes from
`get_concentration_thresholds` — `{"amber": 0.20, "red": 0.30}` by default, a
generic share band. The approved tests `risk_limits` evaluates are a different
artefact, and `config/mi/risk_monitor.yaml` says so in terms:

> *"Thresholds below are PLACEHOLDER defaults to be tuned per client; **client
> concentration limits will live in a separate `config/client/` file**."*

That separate file is `config/client/risk_limits_config.py::ALL_LIMITS` — what
`risk_limits` reads. Compare the two outputs on the same book:

```
risk_limits (current)  "5 passed, 0 warning(s), 6 breach(es), 1 need review,
                        3 unavailable. Nearest to limit: Top 3 brokers
                        (-31.5 pp headroom)."          <- NAMED TESTS + HEADROOM
run_funded_vs_forecast  eight regional shares, all green, no test named
```

A question naming *"which concentration **tests**"* cannot be answered by a
primitive that has no tests in it. Presenting regional share drift as the answer
would be a substitution of a different kind, not a fix.

### Three further blockers, each independently sufficient

1. **No `SnapshotStore` on the MI path.** `run_funded_vs_forecast(store, …)`
   requires one. `SnapshotStore` is abstract with a single implementation,
   `snapshot/adapters/local_fs.py::LocalFsSnapshotStore`. **No module under
   `mi_agent_api/` references `SnapshotStore` at all**, no environment variable
   configures a root, and the MI fixture contains no registered snapshots.
   Reaching it needs a snapshot registration pipeline, not routing.
2. **It requires a caller-chosen `dimension`.** Nothing in the question supplies
   one. Choosing would be an assumption, which the brief forbids.
3. **Its thresholds are not the approved limits**, per §1 above.

---

## 2. The route that owns the approved limits never claims the question

`mi_workflows.analytical.registry.CAPABILITIES` holds both halves:

```
concentration_limits          datasets=('funded', 'limits')
funded_balance_forecast       datasets=('funded', 'pipeline')
```

and `analytical_composition` is a live MI route that derives its reconciliation
from the capabilities that ran. So a composition is the obvious home. **It never
gets the chance:**

```
plan_for("Are any concentration limits currently breached?")        -> None
plan_for("Which limits are closest to breach today?")               -> None
plan_for("Do we expect to breach any concentration limits?")        -> None
plan_for("Which concentration tests are we at risk of breaching?")  -> None
plan_for("Are any limits projected to breach?")                     -> None
plan_for("Which limits could breach based on the current forecast?")-> None
```

`mi_workflows/analytical/planner.py::plan_for` returns `None` for **every**
concentration question, current or forward. No plan is built, so
`analytical_composition` never claims one and the two capabilities are
unreachable for this intent.

---

## 3. The governed intent owner already carries the distinction — almost

`mi_workflows.analytical.intent.classify` separates current from forward
**today**, with no new vocabulary:

| question | families | operations | requirements |
|---|---|---|---|
| Are any concentration limits currently breached? | `LIMITS_CONCENTRATION` | STATUS, CONCENTRATION | `limit_evidence` |
| Which limits are closest to breach today? | `LIMITS_CONCENTRATION` | STATUS, HEADROOM, RANKING | `limit_evidence` |
| Do we expect to breach any concentration limits? | `LIMITS_CONCENTRATION`, **`FORECAST_PROJECTION`** | STATUS, **FORECAST_BREACH**, PROJECT_VALUE | `limit_evidence`, **`forecast`** |
| Are any limits projected to breach? | `LIMITS_CONCENTRATION`, **`FORECAST_PROJECTION`** | **FORECAST_BREACH** | `limit_evidence`, **`forecast`** |
| Which limits could breach based on the current forecast? | `LIMITS_CONCENTRATION`, **`FORECAST_PROJECTION`** | **FORECAST_BREACH** | `limit_evidence`, **`forecast`** |
| **Which concentration tests are we at risk of breaching?** | `LIMITS_CONCENTRATION` *(current only)* | STATUS, HEADROOM, RANKING | `limit_evidence` |

Two findings:

- **`FORECAST_BREACH` is declared and consumed by nothing.** It is defined at
  `intent.py:93` and appended at `:750`. The only other occurrence in the estate
  is `CONCENTRATION_FORECAST_BREACH`, a separately-named constant in
  `mi_agent_pptx/watchlist.py` — the deck builder, not the MI query path. The
  operation is a label with no consumer.
- **One required acceptance case is classified current-only.** *"Which
  concentration tests are we at risk of breaching?"* matches only `('breaching',)`
  and carries no `FORECAST_PROJECTION`. The existing owner nearly carries the
  distinction; it is one phrase short.

---

## 4. Where the acceptance cases stand today

question → intent → subject dataset → capability → datasets read → receipt → answer

| question | families | route | read | outcome |
|---|---|---|---|---|
| Are any concentration limits currently breached? | LIMITS_CONCENTRATION | `risk_limits` | funded | ✔ **correct** — named tests, headroom |
| Which limits are closest to breach today? | LIMITS_CONCENTRATION | `risk_limits` | funded | ✔ **correct** |
| Do we expect to breach any concentration limits? | +FORECAST_PROJECTION | `risk_limits` | funded | ✗ **current-state substitution** |
| Which concentration tests are we at risk of breaching? | LIMITS_CONCENTRATION | `risk_limits` | funded | ✗ **current-state substitution** |
| Are any limits projected to breach? | +FORECAST_PROJECTION | `risk_limits` | funded | ✔ already **refuses**: *"I understood that you asked for a forward projection, but that could not be applied…"* |
| Which limits could breach based on the current forecast? | +FORECAST_PROJECTION | `risk_limits` | funded | ✔ already **refuses**, same facet |
| Summarise the current pipeline. | PIPELINE | `portfolio_summary` | funded | ✗ (the separate dataset finding, unchanged) |

**Two of the four forward phrasings already refuse correctly.** The estate has a
forward-projection facet that fires and declines. Only two substitute.

---

## 5. What a refusal would cost — measured, not proposed

Simulated only: `risk_limits` declines when the **existing** intent owner reports
`FORECAST_PROJECTION`. Nothing shipped.

| question | today | gated |
|---|---|---|
| Are any concentration limits currently breached? | risk_limits ✔ | **unchanged** ✔ |
| Which limits are closest to breach today? | risk_limits ✔ | **unchanged** ✔ |
| Do we expect to breach any concentration limits? | substitutes | **governed refusal**: *"I understood this as a limits concentration, forecast projection question, but I have not answered it…"* |
| Are any limits projected to breach? | refuses | refuses |
| Which limits could breach based on the current forecast? | refuses | refuses, reconciled `forecast` |
| **Which concentration tests are we at risk of breaching?** | substitutes | **still substitutes** — the intent owner classifies it current-only |
| Summarise the current pipeline. | unchanged | **unchanged** ✔ |

So gating on the existing owner satisfies three of the four hard behavioural
rules and **fails the fourth acceptance case**, because the classification is one
phrase short of carrying it.

---

## 6. What I did not do, and what I need

**Not built.** The brief's gate is explicit: *"If the capability does not already
deterministically calculate the requested forward answer, STOP and report. Do not
create a new forecast methodology."* It does not, on three independent grounds.

Nothing was added to `risk_limits`, no `pipeline_root`, no capability broadened,
no limit configuration, forecast assumption or reconciliation methodology
touched. No new vocabulary added.

**Two rulings would unblock, and they are different sizes:**

1. **Refuse-only** — gate `risk_limits` on the existing `FORECAST_PROJECTION`
   family. Measured above: converts one substitution into a governed refusal,
   leaves the current-state cases untouched, leaves the pipeline case untouched.
   It uses the existing governed intent owner and adds no vocabulary. It does
   **not** satisfy *"Which concentration tests are we at risk of breaching?"*.
2. **Refuse-only, plus one phrase in the intent owner** so *"at risk of
   breaching"* carries `FORECAST_PROJECTION`. That is a change to the existing
   governed owner rather than new vocabulary elsewhere, and it would satisfy all
   four forward acceptance cases as refusals.

Neither produces a forward *answer*. Producing one needs a snapshot store on the
MI path, a dimension policy, and a mapping from the approved tests to the
forward frame — which is the forecast methodology the brief forbids inventing.

### Environment
`MI_AGENT_LLM_PARSER=off` (F2), repository root (F6).
**Successful model responses: 0.**
