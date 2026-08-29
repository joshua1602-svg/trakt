# The narrowing decision — evidence

Base `9d24d7c`, tree clean. **No code changes.** Every claim below comes from a
live call path: files intercepted at `pandas.read_csv`/`read_parquet`, scoped to
what happens *inside* `try_route`, each question in a fresh process with
`TRAKT_CONFIG_CACHE=off` so no cache can hide a read.

**Decision, per case:**

| # | question | decision |
|---|---|---|
| 1 | Summarise the current pipeline. | **(a) narrow eligibility, route elsewhere** — an existing route already answers it correctly |
| 2 | …which concentration tests are we at risk of breaching? | **(b) narrow eligibility, refuse** — no route supports it, and the refusal already exists and is honest |
| 3 | Show funded vs pipeline contribution. | **(b) refuse — BLOCKED.** Narrowing today yields an internal validation error, not a refusal |
| 4 | What is the weighted expected pipeline contribution? | **(b) refuse — BLOCKED**, same cause |

**No route gains `pipeline_root` and no capability is broadened.** The evidence
does not support it for any of the three routes.

---

## 1. Method, and one correction it forced

Intercepting every file read *during a request* is wrong, twice over:

- `mi_service._resolve_frame` loads the frame **before** `try_route`, so
  whole-request interception credits the route with the loader's read;
- **a read that ENUMERATES what exists is not a read that FEEDS THE ANSWER.**

The second cost me a wrong conclusion before I caught it. `risk_limits` opens
every weekly pipeline extract:

```
mi_agent_api/portfolio_context.py:194   discovered_pipeline_portfolios
mi_agent_api/pipeline_contract.py:204   discover_pipeline_sources
mi_agent_api/pipeline_contract.py:83    _read_source
   -> tests/fixtures/pipeline_history_5w/2026-05-01/M2L_KFI_and_Pipeline_2026_05_01.csv
```

That is **portfolio discovery**. It lists what exists on the platform and
computes nothing. Counting it as "reads the pipeline" would have credited
`risk_limits` with a capability it does not have — the exact mistake this
diagnosis exists to avoid making in reverse. Reads are therefore classified by
call stack into *answer* and *discovery*, and only *answer* reads count.

---

## 2. Route × supported dataset

From reads that feed the answer, one fresh process per representative, 28
representatives across 19 live routes.

| route | funded | pipeline | evidence |
|---|:--:|:--:|---|
| `(point-in-time)` | **Y** | **Y** | frame is `_resolve_frame(view)` — the named dataset by construction; Q10B answers 8 loans/5 stages reconciled `pipeline` |
| `analytical_composition` | **Y** | **Y** | reads `funded+pipeline` for a forecast question |
| `forecast_extrapolation` | **Y** | **Y** | reads `funded+pipeline` |
| `scenario` | **Y** | **Y** | reads `funded+pipeline` |
| `evolution` | **Y** | **Y** | reads `funded` or `pipeline` per question |
| `temporal_compare` | **Y** | **Y** | reads `funded` or `pipeline` per question |
| `cohort_conversion` | – | **Y** | reads `pipeline` only |
| `evolution_funnel` | – | **Y** | reads `pipeline` only |
| `evolution_pipeline_stage` | – | **Y** | reads `pipeline` only |
| **`portfolio_summary`** | **Y** | **–** | reads `funded` for both a funded and a *pipeline*-named question |
| **`risk_limits`** | **Y** | **–** | reads `funded`; pipeline touch is **discovery only** |
| **`funded_bridge`** | **Y** | **–** | reads `funded` for both |
| `period_movement` | **Y** | – | |
| `cohort_progression` | **Y** | – | pipeline touch is discovery only |
| `period_change` / `period_change_analysis` | **Y** | – | |
| `concentration_analysis`, `geo_exposure`, `portfolio_risk_comparison` | – | – | compute from the pre-loaded frame; no route-scoped read |

**Nine of nineteen routes genuinely support the pipeline.** The three that
substitute are not among them, and none of the three takes a `pipeline_root`
parameter — so they could not honour a pipeline request even if they asked.

---

## 3. The four traced: interpretation → claim → selection → root read

All four are identical up to the route boundary, and all four diverge there.

```
                      Q1 summary        Q2 limits         Q3/Q4 bridge
interpretation        pipeline          pipeline          pipeline
  resolve_dataset     'pipeline'        'pipeline'        'pipeline'
dataset claim
  datasetContext      pipeline          pipeline          pipeline
  _resolve_frame      8 rows loaded     8 rows loaded     8 rows loaded
route selection
  try_route(view=)    'pipeline'        'pipeline'        'pipeline'
  recogniser wins     portfolio_summary risk_limits       funded_bridge
root read (answer)    funded            funded            funded
  discovery only      –                 pipeline          –
reconciliation        funded            funded            funded
```

**The dataset is decided correctly, the pipeline frame is loaded, and the route
is told — and then the route reads `output_root` regardless.** The failure is
entirely at the last step, and it is the same step for all four.

---

## 4. The decisive experiment: what happens if they decline

Simulated only — the three recognisers' `recognise` gated on
`resolve_dataset(question) == funded`, the live `REGISTRY` rebuilt with
`dataclasses.replace` (the recognisers are frozen and registered at import).
Nothing shipped.

| question | today | if the route declines |
|---|---|---|
| **Summarise the current pipeline.** | `portfolio_summary` · funded · *"640 loans … £172.1m"* | **`(point-in-time)` · pipeline · "8 loans · £3.6MM"** |
| **…concentration tests at risk?** | `risk_limits` · funded · *"5 passed, 6 breaches…"* | `(point-in-time)` · pipeline · **governed refusal**: *"I understood this as a pipeline, limits concentration question, but I have not answered it: this asks about concentration limits, which are governed…"* |
| **Show funded vs pipeline contribution.** | `funded_bridge` · funded · a funded bridge | ✗ **`"I could not build a governed query for this question: chart_type 'none' is not valid for intent 'chart'."`** |
| **Weighted expected pipeline contribution?** | `funded_bridge` · funded · a funded bridge | ✗ same internal error |

### Case 1 → (a), route elsewhere

`8 loans · £3.6MM` **is** the constructible truth (8 rows, £3,600,000, 5 stages),
and the frozen bank's rationale for grading this `WRONG / SILENT` was precisely
*"answered from the FUNDED book; the question named the pipeline dataset"*.
Declining fixes exactly what was recorded as wrong, via a route that already
works. No new capability.

### Case 2 → (b), refuse

No route computes concentration limits over the pipeline — `risk_limits` is the
only limits route and it reads funded. The fall-through **already produces a
governed refusal that names the capability boundary** and reconciles `pipeline`.
Declining converts a confident current-state answer into an honest refusal, using
machinery that exists. This is the `CURRENT-STATE SUBSTITUTION` the frozen bank
recorded for Q25A/B/C.

### Cases 3 & 4 → (b) in principle, BLOCKED in practice

No route computes a pipeline contribution, so refusing is right. But the
fall-through does not refuse — it fails validation:

```
_deterministic_parse("show funded vs pipeline contribution.")
    intent='chart'   chart_type='none'   metric=None   dims=[]
```

**The parser emits an internally inconsistent spec**, with no routing involved.
It is pre-existing and currently *masked* by `funded_bridge` claiming the
question before the spec is validated. Narrowing eligibility without addressing
it trades a wrong answer for an internal error message — a different failure, not
a better one.

A truth exists for case 4: `weighted_expected_funded_amount` sums to
**£1,320,000** over the 8 pipeline rows. Nothing computes it, which is why the
answer is a refusal and not a route.

---

## 5. What the evidence does NOT support

**Do not add `pipeline_root` to any of the three.** No evidence suggests they
have pipeline semantics:

- `portfolio_summary` computes a *funded* headline — loan count, funded balance,
  WA LTV, WA rate, borrower age. The pipeline frame has no funded balance; its
  measure is `weighted_expected_funded_amount`. The metrics are not the same
  metrics.
- `risk_limits` evaluates an operator-approved concentration configuration over
  the funded book. Its only pipeline contact is enumerating portfolios.
- `funded_bridge` attributes movement in *funded* balance between two reporting
  periods. A pipeline case has no funded balance to attribute.

Broadening any of them would be inventing a capability, not honouring a request.

---

## 6. Recommended order, and what each step needs

1. **Case 1 — narrow `portfolio_summary`.** Lowest risk: the replacement route
   exists, answers correctly, and reconciles the named dataset.
2. **Case 2 — narrow `risk_limits`.** The refusal exists and is governed.
3. **Cases 3 & 4 — fix the spec defect first**, then narrow `funded_bridge`.
   Narrowing first is the wrong order: it exposes an internal error to a reader.

The `intent='chart'` / `chart_type='none'` inconsistency is its own defect,
independent of datasets, and is worth a look regardless of this decision — any
question that reaches point-in-time with that spec fails the same way.

**Regression for all of it:** the six registered pipeline answers, which must
stay byte-identical, three of which are already served by the point-in-time path
that cases 1 and 2 would route to.

### Environment
`MI_AGENT_LLM_PARSER=off` (F2), repository root (F6), `TRAKT_CONFIG_CACHE=off`
for the read tracing. **Successful model responses: 0.**
