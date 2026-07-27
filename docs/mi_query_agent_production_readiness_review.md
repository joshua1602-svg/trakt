# MI Query Agent — Production Readiness Review

**Scope.** The MI Query chatbot only: `POST /mi/query` and the query
infrastructure directly beneath it. Dashboard endpoints, PPTX, onboarding,
canonical transformation, the platform assembler, regime reporting, React
visualisation components, blob orchestration and Copilot are **not** in scope and
were not modified.

**Method.** Static review of every stage of the request path, plus an empirical
sweep: 48 questions across 13 capability families, and 10 questions × 7 portfolio
lenses (`None`, `total`, `direct`, `acquired`, `direct_001`, `acquired_001`, and
an unknown `direct_999`), executed end-to-end through the real FastAPI app
against a three-portfolio provenance frame (`direct_001`, `direct_002`,
`acquired_001`). Everything below labelled **Proven** was observed in that sweep,
not inferred from reading.

---

## 1. Architectural review

### 1.1 The stages, and who owns each

| # | Stage | Module | Verdict |
|---|---|---|---|
| 1 | API routing | `mi_agent_api/app.py::query` | Thin, correct — but its route surface is coupled to one deployment topology (§2) |
| 2 | Governance | `mi_agent_api/mi_service.py` | Strong. Scope → tenancy → source approval, all before any dataframe is touched |
| 3 | Capability routing | `mi_agent_api/chat_routing.py::try_route` | **Weak.** Re-parses the question, ignores the portfolio lens on 7 of 11 routes, never consults `resolve_capabilities` |
| 4 | Intent / semantic parse | `mi_agent/llm_query_parser.py::_deterministic_parse` | Broad but ordered-if-chain; several collisions (§5) |
| 5 | Lens resolution | `mi_agent/portfolio_lens.py` + `portfolio_scope.py` + `trakt_core/portfolio.py` | Registry-driven and dynamic on the point-in-time path; **not applied at all** on routed intents (§3) |
| 6 | Query planning / validation | `mi_query_validator.py`, `mi_dataset_profile.py`, `mi_query_contract.py` | Excellent. Fail-closed dimension and filter invariants are a genuine strength |
| 7 | Executor | `mi_agent/mi_query_executor.py` | Mathematically correct; per-group Python loop is the scaling limit (§10) |
| 8 | Response generation | `mi_agent_api/adapters.py`, `presenters.py` | Clean, additive, stable contract |
| 9 | Drill-through | `extra_filters` → spec → executor | Works; filters are applied to the full dataset before aggregation, correctly |
| 10 | Governed envelope | `trakt_core/envelope.py` | Well-designed — but is currently populated with claims the analysis does not support on routed intents (§3.3) |

### 1.2 What is genuinely good

* **Governance ordering.** `execute_governed_mi_query` runs scope → tenant →
  portfolio authorisation → source approval before resolving a dataframe.
  A caller who is not entitled never causes data to be read.
* **Fail-closed invariants.** `check_dimension_invariant` and
  `check_filter_invariant` refuse an answer where a parsed dimension or filter
  was silently dropped. This is the strongest single control in the agent.
* **Error containment.** No analytical fault escapes as a 500. Every failure path
  returns a controlled envelope with `ok:false` and a reason.
* **Dynamic group membership.** `resolve_scope` computes `direct` from the
  registry every call. Onboarding `direct_003` widens Direct with no code change.
  There is no fixed id list anywhere in the scope resolver.
* **Coverage disclosure.** `ScopeCoverage` reports which portfolios answered and
  which could not, per field.

### 1.3 The structural problem

There are **two intent parsers and two intent taxonomies**:

1. `chat_routing.try_route` calls `_deterministic_parse`, then dispatches on the
   resulting spec through its own ordered `if` chain to eleven route handlers.
2. `run_mi_agent_query` calls `parse_with_repair` — **parsing the same question a
   second time** — then executes the point-in-time path.

Consequences:

* The routing decision can be made on a *different spec* from the one executed
  (chat routing is always deterministic; the workflow may use the LLM parser).
* Every request pays the parser cost twice.
* The portfolio lens is applied **only** in `run_mi_agent_query`, so anything
  that routes never gets one (§3).
* `try_route`'s handlers each re-derive their own answer shape, source notes and
  reconciliation block, so 11 near-duplicate envelope constructions exist.

This is the single biggest obstacle to answering thousands of questions: adding a
capability today means adding a route handler, a recogniser, an envelope builder
and a lens application — four places, none of which the type system connects.

---

## 2. Root cause of the 404

### 2.1 What is eliminated

| Candidate | Verdict | Evidence |
|---|---|---|
| Incorrect frontend endpoint path | **Eliminated** | `HttpAgentClient.ask` posts to `${baseUrl}/mi/query`; every one of the 15 paths the client calls exists on the app |
| FastAPI routing / router registration | **Eliminated** | `app.routes` enumerates `POST /mi/query`; a live `TestClient` call returns **200** with a correct governed envelope |
| Dependency injection | **Eliminated** | `build_dependencies()` resolves; the sweep executed 48 questions successfully |
| Capability registration | **Eliminated** | `CAPABILITY = "mi.question.answer"` is a constant, not a lookup that can miss |
| Request payload shape | **Eliminated** | Pydantic `QueryRequest`; a bad body is 422, not 404 |
| Response contract mismatch | **Eliminated** | A mismatch is a client-side parse error, not an HTTP status |
| Authentication | **Eliminated as a 404** | `auth_guard` raises 401/403; it can never produce 404 |
| Middleware | **Eliminated** | Only `CORSMiddleware` is installed; it does not 404 |
| Tenant / portfolio resolution | **Eliminated as a 404** | `PORTFOLIO_NOT_AUTHORISED` → 403, `TENANT_MISMATCH` → 403 (`trakt_core/errors.py`) |
| Environment configuration | **Contributing** | See below |

Decisive: **`trakt_core.errors._CODES` maps exactly one code to 404 —
`ARTEFACT_NOT_FOUND` — and the query capability never raises it.** The MI Query
Agent is *incapable* of returning 404. Therefore the 404 is produced **before the
request reaches FastAPI**, by whatever is in front of it.

### 2.2 The architectural cause

The API's route surface is **hard-coupled to one deployment topology, and the
repository contains two contradictory statements about which topology is
deployed** — with nothing that tests either.

* `mi_agent_api/auth.py` (module docstring) and `docs/auth_setup_runbook.md`
  §4–§6 describe the design: React on Azure Static Web Apps with `trakt-mi-api`
  as a **linked backend**, so the browser calls the API **same-origin under
  `/api`**, and Easy Auth injects `X-MS-CLIENT-PRINCIPAL`. The runbook is
  explicit: *"The website needs to send report questions to `/api`
  (`VITE_AGENT_API_URL` = `/api`)."*
* `.github/workflows/azure-static-web-apps-nice-smoke-067ac7603.yml` builds the
  SPA with `VITE_AGENT_API_URL: https://trakt-mi-api.azurewebsites.net` — an
  **absolute, cross-origin** base that bypasses the linked backend entirely.

Both configurations fail, for different reasons, and **neither failure is
detectable from the code**:

* **Linked-backend topology (`/api`).** The browser posts `/api/mi/query`. Azure
  Static Web Apps reverse-proxies linked-backend requests to the App Service
  **with the path intact**. FastAPI registers `/mi/query` and has **no `/api`
  prefix, no `root_path`, and no router prefix** — confirmed by grep across the
  whole repository: the string `/api/mi` appears nowhere in any Python module.
  The App Service therefore answers **404**. This is the exact reported symptom.
* **Absolute-URL topology.** The browser calls the App Service directly, so the
  SWA never injects the principal header and `MI_AGENT_CORS_ORIGINS` (default:
  `localhost:5173,localhost:4173`) does not include the SWA origin. That fails as
  401/CORS rather than 404 — but it is equally broken.
* **Either topology, unmatched path.** `staticwebapp.config.json` sets
  `navigationFallback.rewrite: /index.html` and excludes only static assets. Any
  API path SWA does not forward is swallowed by the SPA fallback, which for a
  `POST` is a **404**.

So the architectural cause is:

> **The MI Query Agent's HTTP route surface is a deployment-topology assumption
> rather than a contract.** There is no prefix normalisation, no CORS
> configuration derived from the deployed front end, and — critically — **no test
> anywhere that asserts the set of paths the React client calls is servable by
> the FastAPI app.** A base-URL change in a YAML file silently breaks every
> question the chatbot can answer, and nothing in CI notices.

### 2.3 The correction (not a symptom patch)

Implemented in this change (§7, §8):

1. **Gateway-prefix normalisation.** A small ASGI-level middleware strips a
   configured gateway prefix (`MI_AGENT_API_PREFIX`, default `/api`) from the
   inbound path **only when the remainder resolves to a route the app actually
   serves**. The same app now answers `/mi/query` and `/api/mi/query`, so the API
   is correct under both documented topologies and under a future gateway that
   mounts it elsewhere.
2. **CORS derived from the deployed front end.** `MI_AGENT_ALLOWED_ORIGIN` (or
   the SWA-provided hostname) is added to the allow-list, so the absolute-URL
   topology stops failing preflight.
3. **A route-surface contract test.** `test_mi_query_route_contract.py` reads the
   literal paths out of `HttpAgentClient.ts` and asserts every one is servable —
   bare **and** under the gateway prefix. A future base-URL or route change that
   would reproduce this 404 now fails CI.
4. **SPA fallback hardening.** `staticwebapp.config.json` excludes `/api/*` and
   `/mi/*` from `navigationFallback`, so an API call can never be answered with
   `index.html`.
5. **A self-describing probe.** `GET /health` now reports `apiPrefix` and the
   accepted path forms, so a deployment can be diagnosed with one curl instead of
   by inspecting a build log.

---

## 3. Lens resolution assessment

### 3.1 What is correct

The point-in-time path is genuinely dynamic and registry-driven. Proven across
all seven lenses:

| Lens | Portfolios resolved | Region groups returned |
|---|---|---|
| `None` / `total` | `direct_001, direct_002, acquired_001` | 10 |
| `direct` | `direct_001, direct_002` | 9 |
| `acquired` | `acquired_001` | 7 |
| `direct_001` | `direct_001` | 6 |
| `acquired_001` | `acquired_001` | 7 |

`direct_002` was picked up by the `direct` group with **no code change** — the
registry, not a hard-coded list, defines membership. `PortfolioScope.filters`
emits the resolved **id list**, never a type string, so a group is exactly the sum
of its registered members. There is **no `direct_001` / `acquired_001` hard-coding
in any resolver**.

### 3.2 Defect L1 — routed intents ignore the lens entirely (**Proven, critical**)

`try_route` builds its keyword arguments as:

```python
kw = dict(client_id=…, run_id=…, output_root=…, pipeline_root=…,
          portfolio_id=…, as_of=…)          # no source_lens
```

Only `_route_portfolio_summary`, `_route_period_movement`, `_route_bridge` and
`_route_cohort_progression` accept and apply `source_lens`. The other seven —
`_route_geo`, `_route_evolution`, `_route_compare`, `_route_forecast`,
`_route_scenario`, `_route_risk`, `_route_conversion` — **do not declare the
parameter at all**, so the workspace lens is silently discarded.

Proven for geographic concentration, the most-asked MI question there is:

```
lens=total         → Largest geographic concentration: Nottingham at £831k (15.4% of the book)
lens=direct        → Largest geographic concentration: Nottingham at £831k (15.4% of the book)
lens=acquired      → Largest geographic concentration: Nottingham at £831k (15.4% of the book)
lens=acquired_001  → Largest geographic concentration: Nottingham at £831k (15.4% of the book)
```

Byte-identical across every lens. The same holds for `risk_limits`. A user working
in the Acquired workspace is shown the **Total book** and told it is theirs.

### 3.3 Defect L2 — the governed envelope asserts a scope the answer does not have (**Proven, critical**)

`mi_service._stamp_routed_scope` stamps the *resolved lens scope* onto every
routed answer, unconditionally — including answers that were never narrowed. For
the geographic question above at `lens=acquired_001`:

```json
"portfolioScope":  {"context_id": "acquired_001", "portfolio_ids": ["acquired_001"]},
"governance.scope": {"portfolios_used": ["acquired_001"],
                     "is_fully_consolidated": true, "disclosure": null}
```

The numbers are Total; the governance block certifies them as `acquired_001`,
fully consolidated, with no disclosure. **This is worse than omitting the stamp:**
the control that exists to prevent misattribution is the thing performing it.

### 3.4 Defect L3 — an unknown portfolio silently widens to Total (**Proven, high**)

`lens=direct_999` (not in the registry) resolves to Total. `resolve_scope`
correctly records `fell_back_to_total: true, requested_context_id: "direct_999"`
— but `warnings` is `[]` and the disclosure reads *"Fully consolidated across all
3 portfolios in Total"*. The user is never told their selection was discarded. A
stale UI selection, a renamed portfolio or a typo silently changes the scope of
the answer.

### 3.5 Defect L4 — natural language cannot name a non-conventional portfolio (**High**)

```python
_COHORT_ID_RE = re.compile(r"\b((?:direct|acquired)_\d+)\b", re.IGNORECASE)
```

Only `direct_<n>` / `acquired_<n>` are recognised in a question. The explicit
selection path already accepts any provenance-valid slug via
`_SELECTABLE_COHORT_ID_RE`, and `/mi/source-portfolios` will list such a
portfolio — but the user cannot name it in the chat. A managed-service client
onboarded as `alp_origination` is selectable in the dropdown and invisible to the
chatbot. This is the one place where the *naming convention*, not the registry,
decides what exists. **Future `Direct_x` / `Acquired_x` portfolios work; anything
outside the convention does not.**

### 3.6 Defect L5 — capability is inferred from portfolio type, not metadata (**Medium**)

`portfolio_lens.available_lenses` sets `funded_only: ptype == "acquired"`, i.e.
"acquired books have no pipeline". The module's own closing comment says this
reasoning was deliberately removed from `is_acquired_only()` and moved to
`trakt_core.portfolio.resolve_capabilities` precisely so an acquired vehicle that
*does* originate can be configured rather than coded around. The assumption
survives here.

---

## 4. Capability audit

Empirical, 48 questions, whole-book lens. `route=None` means the point-in-time
executor answered.

| Capability | Status | Evidence / defect |
|---|---|---|
| Portfolio overview | **Implemented** | KPI summary artifact |
| Balances | **Implemented** | `sum(current_outstanding_balance)` |
| Geography (breakdown) | **Implemented** | bar + table, lens-correct |
| Geography (concentration) | **Broken (lens)** | `geo_exposure` route; ignores lens (§3.2) |
| LTV (KPI and by dimension) | **Implemented** | weighted-average, correct weight field |
| Age (KPI and buckets) | **Implemented** | |
| Interest rates | **Implemented** | weighted-average |
| Property values | **Implemented** | |
| Borrowers | **Partial** | joint/sole resolves to a filter; `borrower_type` absent from the pack, correctly disclosed |
| Protected Equity | **Correctly unsupported** | governed controlled-unsupported message; no value fabricated |
| Risk | **Broken (lens)** | `risk_limits` route; ignores lens |
| Time series | **Partial** | monthly/weekly work; **quarterly fails validation**; **year-on-year silently returns a single KPI** |
| Tables | **Partial** | `"… as a table"` is ignored; the request became a heatmap and failed validation |
| Breakdowns | **Implemented** | |
| Comparisons (temporal) | **Broken** | `"compare October and November"` returned a whole-book count KPI, `ok:true`. Silently answered a different question |
| Comparisons (portfolio) | **Broken** | `"compare direct and acquired"` returned a whole-book count KPI. `resolve_comparison_lenses()` exists and is **never called from the chat path — dead code** |
| Rankings — Top N | **Broken (route hijack)** | `"show top 5 regions by balance"` hijacked into `geo_exposure`, which ignores `top_n`, the metric and the lens, and returns top-15 ITL3 areas |
| Rankings — Bottom N | **Implemented** | `_RANK_ASC` handles smallest/lowest/bottom |
| Filters | **Implemented** | numeric + categorical, applied pre-aggregation, fail-closed invariant |
| Multi-dimensional (2-D) | **Implemented** | heatmap/matrix |
| Multi-dimensional (3-D) | **Broken** | 3 dimensions fail validation. `_build_multi_dim_table_spec` exists to handle exactly this and is **unreachable in practice** |
| Percentages / share-of-book | **Missing** | `"what percentage of the book is in the South East?"` → whole-book count, filter dropped, no share computed, `ok:true` |
| Weighted averages | **Implemented** | correct, single-pass |
| Drill-through | **Implemented** | loan-level filtered table |
| Trend analysis | **Partial** | renders a line; no trend *statement* |
| Cohorts | **Broken here** | cohort progression failed validation on this pack |
| Forecast | **Route unreachable here** | needs a run-scoped artefact; degrades to a KPI rather than a governed explanation |
| Pipeline | **Ungoverned failure** | returns `ok:false, "No governed pipeline data is available"` instead of the governed `CAP_PIPELINE` explanation that already exists |
| Count distinct | **Unreachable** | `"how many distinct brokers"` → bar chart of balance by broker, validation failed. `aggregate_series` supports `count_distinct`; nothing routes to it |
| Averages (loan size) | **Wrong answer, silent** | `"what is the average loan size?"` → **sum** of balance by `ticket_bucket`. Neither an average nor a KPI, `ok:true` |
| Drawdowns | **Missing** | unmapped (correctly refused) |

**Dead / unreachable code found:** `resolve_comparison_lenses` (never called),
`_build_multi_dim_table_spec` (unreachable behind validation),
`count_distinct` aggregation (no route), and a literal no-op branch in
`try_route`:

```python
if _detect_unsupported_concept(question, semantics, set(semantics.get("fields", {}))) is not None:
    pass
```

---

## 5. Semantic weaknesses

| # | Weakness | Severity | Proof |
|---|---|---|---|
| S1 | **Portfolio/metric vocabulary collision.** `_DIRECT_TERMS` contains `"origination"`, `"originated"`, `"new lending"`, `"current book"`, `"own book"` — ordinary *measure* words. `"show origination volumes by month"` silently applied `source_portfolio_id ∈ [direct_001, direct_002]` | **Critical** | Proven — filter visible in the returned spec |
| S2 | **Route hijack by superlative.** Any question containing a geo term *and* `"top "`/`"largest"` is captured by the ITL3 engine, discarding `top_n`, the metric, the filters and the lens | **High** | Proven |
| S3 | **Unknown filter field binds to the wrong column.** `_filter_field_of` scans the *whole question*, not the comparator clause. `"balance by region where flurb is above 3"` produced `current_outstanding_balance > 3` — a predicate the user never asked for, applied silently | **High** | Proven |
| S4 | **Unknown metric silently replaced.** `"show me the unicorn ratio by region"` answered as balance by region, `ok:true`. Directly contradicts the agent's stated "nothing was guessed" contract | **High** | Proven |
| S5 | **Filter precedence.** In `"what percentage of the book is in the South East?"` the categorical value was dropped entirely rather than becoming a filter | High | Proven |
| S6 | **Aggregation collision.** `"average loan size"` — `"size"` matched the `ticket_bucket` *dimension*, which outranked `"average"`, producing a summed bar chart | High | Proven |
| S7 | **Temporal grain is a substring list.** `is_line` tests literal phrases; `"quarter"`, `"quarterly"`, `"year on year"`, `"yoy"`, `"annually"`, `"by year"` are absent | Medium | Proven (quarterly fails, YoY returns a KPI) |
| S8 | **Output-format intent ignored.** `"as a table"` / `"as a chart"` never reach `output_format`; chart type is chosen purely by dimension arity | Medium | Proven |
| S9 | **Ordered-if-chain has no confidence arbitration.** The first recogniser to match wins. `_forecast_scale_recognizer` → `_bridge_recognizer` → `_cohort_progression_recognizer` → `_compare_recognizer` → `_risk_limit_recognizer` are tried in a fixed order with no score, so a question matching two recognisers is resolved by source-code line number | Medium | Structural |
| S10 | **Comparison logic is unreachable.** `spec.temporal_mode == "compare"` only routes when `chat_routing` succeeds; when it does not, the point-in-time path ignores `compare_periods` and answers a whole-book KPI **without disclosing that no comparison was performed** | High | Proven |
| S11 | **Hard-coded dimension preferences.** `("broker_channel", "erm_product_type", "account_status")` and `_NUMERIC_AXIS_BUCKET` embed one client's field names in the parser | Medium | Static |
| S12 | **Bubble axes are heuristic.** `x = age if "age" in q else balance; y = ltv if "ltv" in q else balance` — two literal substring tests decide the chart's axes | Medium | Static |
| S13 | **No follow-up / conversational state.** Every request is stateless; `"and by region?"` cannot resolve against the previous answer. `analysisContext.ts` builds a standalone question client-side, so the *backend* has no follow-up semantics at all | Medium | Structural |

---

## 6. Intent coverage assessment

| Question form | Covered? | Notes |
|---|---|---|
| "What is …" | Yes | KPI path |
| "Show …" | Yes | chart/table path |
| "Breakdown …" | Yes | |
| "Top N …" | Partially | hijacked when the dimension is geographic (S2) |
| "Largest / biggest …" | Partially | same |
| "Average …" | Partially | collides with dimension tokens (S6) |
| "How many …" | Yes | filtered count path |
| "Where …" | Yes | resolves to a regional breakdown |
| "Trend …" | Partially | monthly/weekly only (S7) |
| "Compare …" | **No** | silently degrades to a KPI (S10) |
| "What changed …" | Partially | `_route_period_movement` exists but needs a run-scoped artefact; otherwise unmapped |
| "What explains …" | **No** | returned a bare balance KPI, `ok:true` |
| "Why …" | **No** | returned a bare balance KPI, `ok:true` |
| "What are the main drivers …" | **No** | returned a bare balance KPI, `ok:true` |
| "What percentage / share …" | **No** | filter dropped, no share computed |
| "How many distinct …" | **No** | `count_distinct` unreachable |

**Assessment.** Descriptive intents ("what is", "show", "breakdown") are well
covered. **Explanatory intents ("why", "what explains", "what drives") are the
largest coverage gap, and they fail in the worst possible way** — not with a
refusal but with a confident, plausible, unrelated number. The
`_route_period_movement` attribution engine already computes driver
decomposition; it is simply not reachable from an explanatory question.

**Recommendation.** Add an explanatory-intent recogniser that either routes to the
existing attribution engine or, when it cannot, returns the controlled
"I couldn't map this question" refusal. Never a KPI.

---

## 7. Direct vs Acquired behaviour

The governed capability model (`trakt_core.portfolio.resolve_capabilities`) is
well designed and does exactly what Part 7 asks: it decides Pipeline,
origination-forecast, runoff-forecast and consolidated-forecast applicability
from **portfolio metadata (`originates`, `has_runoff_profile`,
`pipeline_data_available`)**, never from the name, and produces a typed reason
code (`NON_ORIGINATING`, `NO_PIPELINE_DATA`, `NO_RUNOFF_PROFILE`) with prose.

**It is not called anywhere in the chat path.** `grep` shows
`resolve_capabilities` used only by `portfolio_context.py`, consumed only by the
dashboard REST endpoints in `app.py`. `chat_routing.try_route` never asks whether
a capability applies to the scope.

Consequence, proven: asking the chatbot about the pipeline returns
`ok:false, "No governed pipeline data is available for the pipeline view."` —
a data-availability error — for **every** lens, where the governed layer would
have said *"No portfolio in this scope originates new lending, so there is no
origination pipeline to report."* for an acquired scope, and *"No governed
pipeline data has been supplied for direct_001, direct_002"* for a direct one.

**Recommendation (architectural).** `try_route` should resolve the scope's
capability state once and gate each route on it, returning the governed
`CapabilityState.detail` as the answer when a capability is unavailable. That
single change makes every current and future capability behave correctly for
Direct, Acquired and Total without any per-route special-casing.

---

## 8. Total portfolio behaviour

Verified correct on the point-in-time path:

* **Aggregation.** Total applies **no** filter and aggregates the raw row set once
  (`PortfolioScope.filters` returns `{}` for Total). There is no per-portfolio
  pre-aggregation and therefore **no double counting**.
* **Weighting.** `aggregate_series` computes `Σ(v·w)/Σ(w)` over the whole scope in
  a single pass with a shared non-null mask. It is **not** an average of
  per-portfolio averages, so Total weighting is mathematically correct.
* **Filtering / drilling.** Drill-through filters are merged into the spec and
  applied to the full frame before aggregation.
* **Reconciliation.** `_build_reconciliation` reports included vs excluded balance
  and coverage %, and surfaces exclusions as a user-visible warning.

Not correct: **Total comparison and Total explanation** — see §4 (comparisons) and
§6 (explanatory intents). Total *drilling* and *aggregating* are sound; Total
*comparing* and *explaining* are not implemented.

---

## 9. Error handling

| Case | Behaviour | Verdict |
|---|---|---|
| Invalid metric | Silently substituted with the default balance metric | **Fails** — should refuse |
| Invalid filter field | Threshold silently bound to another column | **Fails** — should refuse |
| Missing fields | Controlled unsupported message naming the missing fields | **Correct** |
| Unsupported combinations (3-D) | Controlled validation failure | Correct, though the capability exists |
| Missing capabilities | Data-availability error, not a governed explanation | **Fails** (§7) |
| Unsupported comparisons | Silent KPI | **Fails** |
| Empty results | Controlled `no_values_after_preparation` | **Correct** |
| Future / unknown portfolios | Silent widening to Total | **Fails** (§3.4) |
| Exceptions | Fully contained; no 500 observed in 384 executions | **Correct** |

One further robustness issue: `mi_service._run_analysis` wraps the entire routing
call in `except Exception: routed = None`. In the sweep this swallowed
`"expected str, bytes or os.PathLike object, not NoneType"` — a configuration
fault (no onboarding output root) that silently disables **every routed
capability** with only a log line. The user sees a degraded point-in-time answer
and is told nothing.

---

## 10. Performance review

| Area | Finding |
|---|---|
| **Double parse** | Every request parses the question twice — once in `try_route`, once in `run_mi_agent_query`. Pure waste, and a correctness hazard when the two parsers disagree |
| **Grouped aggregation** | `_grouped_aggregate` iterates groups in **Python** (`for keys, g in work.groupby(...)`) calling `aggregate_series` per group and building a list of dicts. Vectorised `groupby().agg()` would be 1–2 orders of magnitude faster on a high-cardinality dimension (postcode district, broker). **This is the binding constraint when larger portfolios are onboarded** |
| **Full-frame copies** | `execute_mi_query` does `df.copy()` per request; `_apply_filters`, `_bucket_missing`, `_stringify` and `_group_sum` each copy again. Five full materialisations of the tape per query |
| **Filter ordering** | Filters are applied before grouping — correct. But they are applied *after* the full copy, so the copy is always of the unfiltered frame |
| **Repeated scans** | `profile_dataset` runs a full column profile on every request; `_detect_percent_scale` rescans; `coverage_for_frame` scans per portfolio per field |
| **Caching** | Dataset and semantics are cached (signature / mtime) and warmed at startup — good. There is **no result cache** server-side; the React client caches per session |

**Recommendation, in priority order:** (1) parse once and pass the spec into
routing; (2) vectorise `_grouped_aggregate`; (3) filter before copying; (4) cache
`profile_dataset` on the dataset signature.

---

## 11. Production robustness

* **Dead code:** `resolve_comparison_lenses` (never called from any live path);
  the `if …: pass` no-op in `try_route`; `MiQueryRequest.client_id` (documented
  deprecated).
* **Duplicate parsers:** two parse calls per request (§1.3).
* **Duplicate intent logic:** `_is_line` in the parser vs `_EVOLUTION_MARKERS` in
  `chat_routing` are two independent, non-identical definitions of "this is a
  trend question". `_is_geo_exposure` similarly duplicates the parser's
  concentration handling.
* **Duplicate envelope construction:** 11 route handlers each build their own
  answer/artifact/reconciliation block.
* **Hard-coded assumptions:** `_COHORT_ID_RE` naming convention (§3.5);
  `funded_only = acquired` (§3.6); client-specific field preferences (S11);
  bubble-axis heuristics (S12); `DEFAULT_CLIENT_ID = "client_001"`.
* **Unreachable branches:** `_build_multi_dim_table_spec`, `count_distinct`.

Nothing was removed in this change except the literal no-op branch: the rest is
reachable-in-principle capability that should be *wired up*, not deleted.

---

## 12. Architectural improvements — recommended order

1. **Parse once.** Move the parse above routing and pass the spec down. Removes
   the double cost and the two-taxonomy divergence.
2. **Make the lens a routing input, not a route implementation detail.** Every
   route receives the resolved scope; a route that cannot honour it must say so.
   *(Partially delivered here — see §8 of the change list.)*
3. **Gate routes on `resolve_capabilities`.** One call, applied uniformly, makes
   Direct/Acquired/Total behaviour correct for every present and future
   capability without per-route code.
4. **Replace the ordered-if chain with a scored recogniser registry.** Each
   capability declares a recogniser, a confidence and a handler; the router picks
   the highest score and can report the runner-up as an ambiguity. This is what
   makes "thousands of questions" tractable.
5. **A single envelope builder.** One `GovernedAnswer` construction shared by all
   routes.
6. **Refuse rather than substitute.** An unresolved metric or filter field must
   fail closed, exactly as an unresolved *dimension* already does.
7. **Vectorise the executor** and cache the dataset profile.
8. **Add explanatory intents** wired to the existing attribution engine.
9. **Server-side follow-up context**, so conversational refinement stops being a
   client-side string concatenation.

---

---

## 13. Changes made

Scope discipline: nothing outside the MI Query Agent and its supporting query
infrastructure was touched. No dashboard endpoint, PPTX, onboarding, canonical
transformation, assembler, regime, React visualisation, blob or Copilot module
was modified.

### Files changed

| File | Change |
|---|---|
| **`mi_agent_api/gateway.py`** *(new)* | Gateway-prefix normalisation. The API now serves its routes bare **and** under `MI_AGENT_API_PREFIX` (default `/api`), stripping only when the remainder resolves to a route it actually serves. Plus `MI_AGENT_ALLOWED_ORIGIN` for the cross-origin topology. **The 404 fix.** |
| `mi_agent_api/app.py` | Installs the normaliser outermost; adds the deployed front-end origin to CORS; `/health` now reports `routing.apiPrefix` and the exact query paths it answers on |
| `mi_agent_api/chat_routing.py` | `_route_geo` narrows its frame to the resolved portfolio scope (share-of-book becomes share-of-scope, and the scope is named in the answer); every route now leaves through `_disclose_lens_scope`, which sets `metadata.lensApplied` and discloses in plain words when a route could not honour the lens; `_is_geo_exposure` no longer captures explicit grouped rankings; the `if …: pass` no-op removed |
| `mi_agent_api/mi_service.py` | `_stamp_routed_scope` stamps the scope the **answer has**, not the scope requested — an un-narrowed answer is stamped Total |
| `mi_agent/portfolio_lens.py` | `_DIRECT_TERMS` restricted to portfolio-qualified phrases, so measure vocabulary ("origination", "new lending", "current book") no longer silently narrows a Total workspace to Direct |
| `mi_agent/llm_query_parser.py` | `where`/`whose`/`having` added as predicate clause boundaries; an unresolvable threshold is recorded and surfaced instead of binding to another column; a question that is *only* an unknown predicate is refused; `_detect_top_n` now recognises "bottom N" / "smallest N" |
| `mi_agent/mi_agent_workflow.py` | An unknown portfolio selection that widened to Total is now disclosed to the user, not just recorded in a dict |
| `frontend/mi-agent-ui/staticwebapp.config.json` | `/api/*`, `/mi/*`, `/health`, `/me` excluded from the SPA navigation fallback so an API call can never be answered with `index.html` |
| `deploy/trakt-mi-api/README.md`, `docs/auth_setup_runbook.md` | Both deployment topologies documented, with the one-curl diagnostic |
| **`mi_agent_api/scripts/validate_mi_query_lenses.py`** *(new)* | Reproducible manual validation across Direct / Acquired / Total, over both path forms |

### Tests added

| File | Tests | Covers |
|---|---|---|
| `mi_agent_api/tests/test_mi_query_route_contract.py` | 19 | The 404 root cause. Reads the literal paths out of the shipped `HttpAgentClient.ts` and requires the API to serve every one, bare and prefixed; prefix normalisation must not widen the route surface; `/health` states the topology; the SPA fallback cannot swallow an API call |
| `mi_agent_api/tests/test_mi_query_lens_matrix.py` | 116 | Five portfolios (3 direct, 2 acquired) through the real endpoint: lens resolution, dynamic group membership, Total = Σ parts, weighted-average correctness, routed-intent lens application, un-narrowable-route disclosure, unknown-portfolio disclosure, empty datasets, drill-through composition, question-overrides-dropdown, capability-unavailable, and a 12 × 7 question/lens matrix that must never raise |
| `mi_agent/tests/test_mi_query_capability_matrix.py` | 51 + 7 tracked gaps | Semantic contract: measure vocabulary is not portfolio vocabulary, predicates resolve to their own clause or are refused, rankings stay rankings, the implemented capability set, and every known gap as a `strict` xfail so the documented matrix cannot drift |

**186 tests added.** 13 of them fail on the pre-change code and pass after —
verified by stashing the source changes and re-running, so they are genuine
regression guards rather than assertions written around current behaviour.

### Test results

| Suite | Before | After |
|---|---|---|
| `mi_agent/tests` + `mi_agent_api/tests` | 1205 passed, **11 failed** | 1391 passed, **11 failed** |
| Affected repo-wide suites (governed portfolio, lens wiring, provenance, dependency direction, envelope, render, packaging) | — | 219 passed |

The 11 failures are **pre-existing and out of scope** — Copilot artefact
resolution (`test_copilot_actions.py`), the funded central tape and funded
enrichment. Confirmed unchanged by re-running one on stashed code: it fails
identically with and without this change. No regression was introduced.

---

## 14. Manual validation

`python -m mi_agent_api.scripts.validate_mi_query_lenses` — five portfolios,
11 questions, 6 lenses, each asked on **both** `/mi/query` and `/api/mi/query`.

**Path parity: 0 mismatches.** Every question returns the identical governed
answer through both front doors — the 404 is fixed without the gateway being
able to change what the agent says.

**Direct / Acquired / Total resolve and narrow correctly:**

```
Q: show balance by region
  total         scope=(direct_001, direct_002, direct_003, acquired_001, acquired_002)   10 groups
  direct        scope=(direct_001, direct_002, direct_003)                                9 groups
  acquired      scope=(acquired_001, acquired_002)                                        8 groups
  direct_001    scope=(direct_001)                                                        6 groups
  acquired_002  scope=(acquired_002)                                                      6 groups
```

`direct_003` appears in the Direct answer without existing in any source file —
membership comes from the registry, not from code.

**Routed intents now honour the lens (the proven defect, fixed):**

```
Q: where is the largest geographic concentration?
  total         lensApplied=True   Nottingham       £831k (15.4% of the book)
  direct        lensApplied=True   Berkshire West   £627k (20.1% of Direct)
  acquired      lensApplied=True   Leeds            £594k (26.3% of Acquired)
  direct_001    lensApplied=True   Nottingham       £298k (26.7% of direct_001)
  acquired_002  lensApplied=True   Bristol, City of £458k (36.2% of acquired_002)
```

Five distinct answers where there were previously five identical ones, each
naming its own scope and reporting share **of that scope**.

**A route that cannot narrow now says so, and is stamped with the scope it
actually has:**

```
Q: are we within our risk limits?
  total         lensApplied=True   scope=(all five)
  acquired      lensApplied=False  scope=(all five)
                · Scope not narrowed: this risk-limit answer is computed across the whole
                  platform book … these figures are NOT Acquired-only.
```

Previously this answer was stamped `portfolios_used: ["acquired_001"],
is_fully_consolidated: true, disclosure: null` while carrying whole-book numbers.

**An unknown portfolio is disclosed rather than silently widened:**

```
  direct_999    scope=(all five)
                · Requested portfolio 'direct_999' is not in the governed portfolio
                  registry, so this answer covers the TOTAL book (…) rather than the
                  requested scope.
```

**Unavailable capabilities refuse honestly, on every lens:** protected equity
returns the governed controlled-unsupported message naming the missing field, and
no lens produced an exception across the full 66-cell matrix.

### Residual, not fixed here

* The unknown-portfolio fallback is disclosed on the point-in-time path but not
  on routed answers (a routed `direct_999` resolves to Total and is treated as a
  legitimate Total request). Worth unifying with recommendation §12.2.
* Everything in §12 items 1, 3–9 remains open by design — each is a structural
  change larger than a production-readiness fix, and each is now covered by a
  failing-or-tracked test rather than by nothing.

---

*Findings marked **Proven** were reproduced end-to-end through the FastAPI
application; the sweep harness and its raw output are reproducible from
`mi_agent_api/scripts/validate_mi_query_lenses.py`,
`mi_agent_api/tests/test_mi_query_lens_matrix.py` and
`mi_agent/tests/test_mi_query_capability_matrix.py`.*
