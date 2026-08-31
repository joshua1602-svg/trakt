# MI Cross-Channel Economic Ownership Audit

**Branch** `claude/mi-cross-channel-ownership-audit` (audit-only child of `claude/mi-pack-engine-hardening`)
**Starting SHA** `14305acb6519ae7fa82583e4789ef706eb1f0c30`
**Phase 3 regression, confirmed** 107 failed / 7432 passed / 434 skipped — **zero new failures** against the 107-failure baseline; +75 passing.
**Prior reports** `docs/reports/mi_pack_engine_audit.md`, `docs/reports/mi_pack_engine_phase3.md`

No production code was modified. Two temporary provenance probes were written, run and deleted.

---

## 1. Executive verdict

### **B — TARGETED MIGRATION REQUIRED**

The architecture largely holds. The MI engine owns portfolio economics; the MI
Query Agent's LLM is confined to parsing; PPTX after Phase 3 consumes engine
values for composition, forecast accuracy and limit direction. There is **no
evidence of systemic incorrectness** on any surface, and **no publication
assurance layer is warranted**.

But the audit found **one P0**, and it is not in PPTX — it is in **React**, on
the live MI Query path, and it was proved end-to-end on the real
`POST /mi/query` route:

> **React and the engine hold two different definitions of whether a measure is
> additive.** The engine decides by aggregation (`_sum` / `count` / `_total`);
> React decides by *display format* (`gbp` / `number`). A format cannot tell a
> sum from an average.
>
> Asked *"What is the average balance by broker channel?"*, the engine correctly
> classifies the measure as non-additive, caps the chart to ten of thirteen
> brokers and **drops the tail**. React then marks the same column additive,
> **sums ten averages into a "total" of £3,060,094** — the real book is
> £38,646,184 — and the Insight Panel tells the user:
>
> *"Broker E has the largest current outstanding balance avg, at **11% of the
> total**."*
> *"The top 3 account for **31%** of current outstanding balance avg."*

The remaining findings are ordinary duplicated ownership: three formulas
implemented twice, and roughly fifteen residual PPTX derivations already named
and ratcheted by Phase 3.

**The concern is NOT principally PPTX-specific.** Phase 3 left PPTX in better
shape than React.

---

## 2. Production paths

All three proved by execution, not inferred from filenames.

### React Dashboard

```
main.tsx → App → useWorkspace.runQuery
  → HttpAgentClient.ask()  POST /mi/query
  → ApiResponse {ok, answer, artifacts[], spec, …}
  → HttpAgentClient maps  narrative = body.answer ?? "Query executed."
  → presentAnswer()                     ← lib/responsePresenter.ts
  → mergeArtifacts → ArtifactCard
       ├── ArtifactRenderer  → chart/table/KPI/risk views
       └── InsightPanel      → computeInsights()   ← lib/insights.ts
```

Workspace panels (Geography, Evolution, Forecast, Risk Limits) call their own
`/mi/*` endpoints through the same client.

### PPTX

```
POST /mi/decks/generate → deck_generation._run_generation
  → pptx_stage.generate_investor_pptx → mi_agent_pptx.cli.run
  → mi_api.build_dashboard_data (calls the same /mi/* compute functions)
  → composition.build_facts → DeckBuilder handlers → render.py → .pptx
  → preflight (24 gates) → sidecar
```

### MI Query Agent

```
POST /mi/query → run_mi_agent_query
  → ParsedQuestion.parse   ← llm_query_parser (LLM HERE, AND ONLY HERE)
  → MIQuerySpec
  → mi_query_executor      ← pandas groupby/sum/mean — DETERMINISTIC
  → answer text composed deterministically in mi_agent_workflow
  → adapters.adapt_workflow_result → artifacts
```

**Excluded as not production-reachable:** legacy Streamlit,
`mi_agent_pptx.pptx_builder` / `insight_resolver` / `MetricResolver`'s
aggregation half (test-only), `mi_agent.states` assembler (declared but unread),
`snapshot.store` (not on any MI/PPTX path).

---

## 3. React findings

**Eight production-reachable economically meaningful operations.**

| # | Site | Operation | Class | Note |
|---|---|---|---|---|
| R1 | `lib/drill.ts:39,89,104` | **additivity decided from display format** (`gbp`/`number`) | **C — P0** | The engine decides the same thing by aggregation name and uses it to drop a tail. See §3.1. |
| R2 | `lib/insights.ts:180-212` | `total = Σ shown rows`; `mean`, `median`, `stdev`, `spread`, `topShare`, `top3Share` | **C** | Re-aggregates engine output. Correct for additive measures; wrong for R1's misclassification. |
| R3 | `lib/drill.ts:132-168` | per-value aggregation of loan-level rows | **B** | Legitimate drill composition over rows the engine returned. |
| R4 | `components/GeographyPanel.tsx:65-70` | `top5 = Σ top 5 balances`; `top5Pct = top5/total×100` | **D** | **Exact duplicate of PPTX `insights.py:429`.** Engine publishes per-area `sharePct` and `total`, not top-5. |
| R5 | `components/EvolutionPanel.tsx:91` | `changePct = (cur−prev)/|prev|×100`, worst-move selection | **D** | Same shape as PPTX `insights._pct_change` and `deck._stock_strap`. |
| R6 | `components/artifacts/RiskArtifactView.tsx:27` | `usage = share/limit×100`, **capped at 100** | **D** | Utilisation is a governed concentration concept (`utilization`). This artifact comes from the legacy `risk_monitor` and carries no utilisation, so React fills a gap — but under a name the engine owns. The 100% cap makes a 150%-of-limit breach render identically to one exactly at limit. |
| R7 | `components/DrillThroughPanel.tsx:93` | `selected/total×100` | **B** | A share of the current selection, not of the book. |
| R8 | `lib/responsePresenter.ts:47-78` | composes a sentence from R2 (`topLabel`, `topValue`, `topShare`, `total`) | **B / conditional C** | Fires only when the backend narrative is absent, debug-shaped, or **under 70 characters**. |

Genuine rendering (excluded, per the brief): currency symbols, MM/K, percent
display, bar widths, choropleth heat scale, label ordering.

### 3.1 The P0, proved

`_cap_bar_rows` (`mi_agent_api/adapters.py:725`) — the **engine's** definition:

```python
additive = (value_key.endswith("_sum") or value_key == "count"
            or value_key.endswith("_total"))
if additive:   head + [aggregated "Other" row]      # total preserved
else:          ordered[:n]                          # TAIL DROPPED
```

`drill.ts:39,89` — **React's** definition:

```ts
const ADDITIVE_FORMATS = new Set<ValueFormat>(["gbp", "number"]);
additive: !!format && ADDITIVE_FORMATS.has(format)
```

`_infer_col_format` strips `_avg` / `_weighted_avg` and returns the base field's
format, so `current_outstanding_balance_avg` is published with
`format: "gbp"` — correct for rendering, and read by React as "sum-able".

**Observed on the real route** (13 brokers, 130 loans):

```
engine  : answer "Here is the bar … Calculated: Average Balance · grouped by Broker · 13 groups"
          chart rows = 10 of 13, valueFormat gbp, otherCategories present = False
React   : primary = current_outstanding_balance_avg, additive = TRUE
          total    = 3,060,094      (Σ of 10 averages)
          topShare = 0.1069
          top3Share= 0.3144
InsightPanel renders:
   info  | Broker E has the largest current outstanding balance avg, at 11% of the total.
   info  | The top 3 account for 31% of current outstanding balance avg.
TRUE portfolio total   = 38,646,184
TRUE portfolio average =    297,278
```

The `isSnapshotTimeSeries` guard in `insights.ts` shows the double-counting trap
was anticipated for *time series*. The **average** case was not.

The engine already knows the answer — it used it to decide whether to fold an
"Other" bucket. **It simply does not publish it.**

---

## 4. PPTX findings — residual after Phase 3

Phase 3 migrated 9, kept 4 presentation semantics, kept 7 formatting, identified
3 redundant and named 11 unmigrated. Re-auditing the current branch confirms
that position. **Fifteen** economic operations remain (the 11 Class A plus
Class B/D items):

| Site | Operation | Class | Another owner? | React does it too? |
|---|---|---|---|---|
| `deck.py:350` `_has_spread` | `max(values)/total` vs 99.5% | B (suppression) | share is engine-computable | no |
| `deck.py:550` | executive pipeline week-on-week diff | A | `pipeline_contract` | **yes — R5 shape** |
| `deck.py:1396` `_movement_finding` | leg / Σlegs | A | `funded_balance_movement` | no |
| `deck.py:1518,1524,1531` | stock delta, per-book move, largest share | A | `funded_evolution` | partially (R5) |
| `deck.py:2033` | pipeline week-on-week diff | A | `pipeline_contract` | **yes — R5** |
| `deck.py:2041` | `avg_case = amount / cases` | A | `pipeline_contract` | no |
| `deck.py:2512` | scenario band spread | B | — | no |
| `insights.py:89,401` | pct change, WA LTV delta pp | A | `snapshots.monthly_change`, `funded_evolution` | **yes — R5** |
| `insights.py:429,430` | geo top-5 and top-area share | A | `geo.exposure_by_itl3` | **YES — R4, exact duplicate** |
| `insights.py:286` | contributor share of movement | A | `funded_bridge` | no |
| `movement.py:84` | `closing − opening` | A | `funded_bridge` | no |
| `movement.py:95` | Σ contributors vs total (re-check) | D | engine already asserts | no |
| `movement.py:105` | materiality floor `|opening| × 0.5%` | B | governed config | no |
| `watchlist.py:255` | WA LTV delta pp | A | `funded_evolution` | no |
| `cohorts.py:95` | `balance / loan_count` | A | cohorts service | no |
| `composition.py:222,260` | `pipeline_share` | A | composition fact | no |

**Economic value vs narrative wording.** Phase 3's boundary holds and is
structurally enforced: `forecast_accuracy.py` contains no division and no
aggregation; `concentration.travel` / `stress_note` contain no comparison; no
presentation module divides by a `total_bal`. Those three ownership tests
already exist and pass.

---

## 5. MI Query findings — **the assumption holds**

**Proved, not assumed:**

- **The LLM is confined to parsing.** The only LLM reference in
  `mi_agent_workflow` is `llm_query_parser`, producing an `MIQuerySpec`.
  There is no LLM call in the execution or answer path.
- **Execution is deterministic.** `mi_query_executor` does the groupby/sum/mean.
- **The answer text is composed deterministically.** Observed:
  *"Here is the bar for your query, covering 13 groups. Calculated: Average
  Balance · grouped by Broker · 13 groups · 130 loans · as at 30 June 2026."*
- **No post-engine recomputation** was found in the Query answer path: no
  re-summing of rows, no ratio, no delta, no independent limit application, no
  measure substitution.

**Zero post-engine economic calculations in MI Query.**

Two boundary notes:

1. **The Query Agent's answer is safe; its *client* is not.** Everything in §3
   sits downstream of a correct Query answer. The engine's narrative is
   currently long enough (146 chars) to defeat `RICH_MIN = 70`, so
   `groundedSentence` does not fire on this route today — but the **Insight
   Panel** consumes `computeInsights` unconditionally, so the P0 reaches the
   user regardless.
2. **M365 Copilot (a fourth surface) hands rows to an LLM.**
   `copilot_actions.CopilotSupportingArtifact` passes result rows with the
   instruction *"Compose any narrative ONLY from this and supportingValues."*
   That is a prompt-level control, not an enforced one — an LLM holding rows can
   arithmetically combine them. Out of scope here; flagged.

---

## 6. Cross-channel ownership matrix

| Concept | Engine owner | React | PPTX | Query | Multiple owners? | Risk |
|---|---|---|---|---|---|---|
| Funded balance | `snapshots.compute_funded_snapshot` | render | render | executor | no | — |
| Loan count | same | render | render | executor | no | — |
| Average loan balance | `snapshots` (`avg_balance`, basis declared) | render | render | executor | no | — |
| WA current LTV | `snapshots._weighted_average` | render | render | executor | no | — |
| WA original LTV | same | render | render | — | no | — |
| WA property value | same | render | render | — | no | — |
| WA interest rate | same | render | render | — | no | — |
| Aggregate gearing | `snapshots` (Phase 3) | render | render | — | no | — |
| Pipeline live amount | `pipeline_prep` (Phase 3 split) | render | render | — | no | — |
| Pipeline live count | same | render | render | — | no | — |
| Weighted expected pipeline | `pipeline_prep` | render | render | — | no | — |
| Forecast funded balance | `forecast_bridge` | render | render | — | no | — |
| Funded movement | `period_change.bridge` | render | render | — | no | — |
| Constituent-book share | `portfolio_context.balance_share` | render | **consumes** | — | no | — |
| Constituent-book movement | `funded_evolution` breakdowns | render | **derives** (`deck:1524`) | — | **yes** | P1 |
| Stratification shares | `snapshots` stratifications | render | **derives** (`_has_spread`) | executor | **yes** | P2 |
| Concentration utilisation | `concentration_tests_api` | **derives** (R6, capped) | render | — | **yes** | P1 |
| Concentration headroom | `concentration_tests_api` | render | render | render | no | — |
| Concentration direction | `concentration_tests_api` (Phase 3) | — | **consumes** | — | no | — |
| Forecast error / bias | `evolution.forecast_evolution` (Phase 3) | — | **consumes** | — | no | — |
| Cohort balance vs formation | cohorts service | render | render | — | no | — |
| Cohort survival | cohorts service | render | render | — | no | — |
| NNEG headroom | `evolution._nneg_metrics` | render | render | — | no | — |
| **Geographic top-5 concentration** | **none** | **derives (R4)** | **derives** | — | **yes — no engine owner** | **P1** |
| **Period-over-period % change** | partial | **derives (R5)** | **derives** | — | **yes** | P1 |
| **Measure additivity** | `adapters._cap_bar_rows` | **derives (R1)** | n/a | — | **yes — conflicting** | **P0** |
| Insight statistics (total/top share) | none | **derives (R2)** | `insights.py` | — | **yes** | P0 via R1 |

---

## 7. Population ownership

The pipeline defect showed population matters more than arithmetic. Auditing
who can change membership:

| Population | Owner | Consumer receives | Can consumer alter membership? |
|---|---|---|---|
| Live vs terminal pipeline | `pipeline_prep.live_mask` (Phase 3) | resolved totals + counts | **No.** Split in the engine; consumers get scalars. |
| Funded vs pipeline lens | `data_resolver` / route | separate payloads | No |
| Reporting period | run resolution | one period per payload | No |
| Constituent portfolio | `portfolio_context.resolve_context` | scoped frame | No — scope is a request parameter, resolved server-side |
| Cohort membership | `evolution.funded_cohort_progression` (static pool) | frozen id set | No |
| Concentration population | `concentration_tests.evaluation` | evaluated tests | No |
| Forecast population | `forecast_bridge` | bridge payload | No |
| **Chart category population** | `adapters._cap_bar_rows` | **top-N, tail sometimes dropped** | **React cannot alter it — but it cannot see that it happened.** |

**No consumer can independently redefine an analytical population.** Every
filtered population is resolved server-side and delivered as a result.

The one exposure is not membership but **awareness of membership**: when the
engine drops a non-additive tail, `otherCategories` is `None` and nothing on the
artifact says "this is a partial population". React then treats the ten shown
rows as the whole. That is the mechanism by which R1 becomes user-visible, and
it is a **payload completeness gap, not a consumer-owned population**.

---

## 8. Duplicate formula findings — exact only

**D1 — Geographic top-5 concentration**

```python
# mi_agent_pptx/insights.py:429
top5 = sum(a.get("balance", 0) for a in areas[:5]) / total * 100.0
```
```ts
// GeographyPanel.tsx:67,70
const top5 = sorted.slice(0, 5).reduce((acc, a) => acc + a.balance, 0);
top5Pct: t ? (top5 / t) * 100 : 0
```
Same concept, same formula, two implementations, one payload, **no engine
owner**. They agree today.

**D2 — Period-over-period percentage change**

```python
# mi_agent_pptx/insights.py:92
return change / abs(prior) * 100.0
```
```ts
// EvolutionPanel.tsx:91
const changePct = ((cur - prev) / Math.abs(prev)) * 100;
```
Same formula. React applies it to `pipeline_amount`; PPTX to funded balance.

**D3 — Measure additivity** (the P0). Two definitions, in §3.1, which **do not
agree**.

**D4 — Concentration utilisation.** `concentration_tests_api` computes
`utilization`; `RiskArtifactView.tsx:27` recomputes `share/limit` for the legacy
risk artifact. Different payloads, same named concept.

Explicitly **not** counted as duplication: `ratio → \`${ratio*100}%\``,
`compact_currency`, percent-scale normalisation, axis formatting.

---

## 9. Test / assurance coverage

| Surface / value | Existing protection |
|---|---|
| Live pipeline stock | **Strong** — 13 tests, fixture truth read off the Status column, downstream consumers proved |
| Stage reconciliation | **Strong** — 8 tests, zero residual, amount-amendment identity, suppression |
| Composition share / opening | **Strong** — behaviour + AST ownership ratchet |
| Forecast accuracy | **Strong** — behaviour + "renderer may not divide or aggregate" |
| Concentration direction | **Strong** — max/min parametrised + "renderer may not compare" |
| Measure basis | **Strong** — 12 tests incl. "WA LTV must not become a ratio of aggregates" |
| Funded bridge, exits | **Strong** — independent identity checks, not implementation echoes |
| React↔PPTX presentation parity | **Medium** — `test_presentation_parity.py` drives both real routes, but covers currency and bucket order only |
| **React `insights.ts` / `drill.ts`** | **Weak** — `insights.test.ts` exists and asserts observation *shape*; **nothing asserts the additivity contract or that a share is over the right denominator** |
| **React `GeographyPanel` top-5** | **None** |
| **React `EvolutionPanel` changePct** | **None** |
| **`RiskArtifactView` utilisation** | **None** |
| PPTX residual 15 | Ratchet + deck-level gates; no per-value truth tests |

Engine calculations with **independent** truth tests (not implementation
echoes): funded bridge identity, exit classification, live pipeline stock, stage
reconciliation, measure basis, forecast accuracy, concentration direction.

---

## 10. Risk register

### P0 — can silently publish materially wrong portfolio economics

| ID | Finding |
|---|---|
| **P0-1** | **R1/D3 — conflicting additivity definitions.** React sums non-additive measures into a "total" and states shares of it in the Insight Panel. Proved on the real `/mi/query` route: 11% and 31% shares of a £3.06m "total" of averages, against a real book of £38.6m. Compounded by the engine dropping the tail without saying so. |

### P1 — duplicated analytical ownership capable of channel divergence

| ID | Finding |
|---|---|
| P1-1 | D1 — geographic top-5 concentration implemented twice, no engine owner |
| P1-2 | D2 — period-over-period % change implemented in React and PPTX |
| P1-3 | D4 — utilisation recomputed in `RiskArtifactView`, **capped at 100%**, hiding breach magnitude |
| P1-4 | PPTX per-book movement and constituent share derived at `deck.py:1518-1531` |
| P1-5 | Pipeline week-on-week deltas and average case amount derived in PPTX (`deck.py:550, 2033, 2041`) |

### P2 — legitimate downstream semantic, insufficiently pinned

| ID | Finding |
|---|---|
| P2-1 | `_has_spread` suppression threshold — a share computed to make a display decision |
| P2-2 | `movement.py` materiality floor as a module constant, not governed config |
| P2-3 | React drill aggregation (R3) — legitimate, untested for denominator correctness |
| P2-4 | `responsePresenter` 70-character substitution threshold — an undocumented cliff |
| P2-5 | M365 Copilot receives rows with a prompt-level-only constraint |

### P3 — rendering only

Currency symbols, MM/K, percent display, bar widths, choropleth heat, tick
formatting, label ordering, text wrapping. **Not inventoried.**

---

## 11. Required actions

### MUST FIX BEFORE CLIENT 1

1. **Publish additivity on the artifact contract.** The engine already computes
   it in `_cap_bar_rows`; emit it per measure (e.g. `displayHints[key].additive`
   or a `measures[].additive` field) and have `drill.ts` consume it instead of
   inferring from format. **One field, one consumer change.**
2. **Say when a population is partial.** When `_cap_bar_rows` drops a
   non-additive tail, mark the artifact (a `truncated` / `shownCategories` flag).
   React must not compute any part-to-whole statistic over a partial population.

Both are small, and together they close P0-1 at its root rather than patching
the Insight Panel.

### SHOULD FIX

3. Give geographic top-5 concentration an engine owner (`geo.exposure_by_itl3`);
   both consumers read it. (P1-1)
4. Publish period-over-period change on `funded_evolution` /
   `pipeline_evolution`; retire both derivations. (P1-2)
5. Remove the `Math.min(…, 100)` cap on utilisation, or label it. (P1-3)
6. Continue the Phase 3 ratchet: the 11 named Class A PPTX values. (P1-4, P1-5)

### ACCEPTABLE AS-IS

- React drill aggregation over returned rows (R3, R7).
- PPTX narrative wording over engine-owned structured values.
- Presentation thresholds (leg dominance 0.45, bias floor 0.5pp, spread 99.5%) —
  documented as presentation decisions in code.
- All P3 formatting.

---

## 12. Publication assurance decision

### **NEW PUBLICATION VALIDATION LAYER REQUIRED: NO**

The five conditions in the decision principle:

1. **One authoritative owner per economic definition** — holds for every
   headline concept in the matrix. The exceptions are three named formulas, two
   of which simply lack an owner rather than having a conflicting one.
2. **Consumers cannot silently redefine analytical populations** — holds
   absolutely. Every population is resolved server-side. The one gap is that a
   consumer cannot *see* that a population was truncated; that is a payload
   field, not a control layer.
3. **Surfaces cannot independently invent conflicting answers** — holds for
   MI Query (proved: LLM parses only) and now largely for PPTX. It does **not**
   hold for React's insight statistics, which is P0-1.
4. **Remaining downstream operations are deliberate presentation semantics** —
   holds after the six actions above.
5. **Existing tests give proportionate assurance** — strong on the engine, weak
   on React's insight layer specifically.

A publication-assurance gate would be **the wrong instrument**: it would inspect
outputs for a defect whose cause is one missing boolean on a payload contract.
The correct fix is to publish what the engine already knows. Adding a validation
layer instead would leave the conflicting definition in place and add a second
thing to keep in step with it.
