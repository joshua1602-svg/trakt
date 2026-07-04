# MI Agent Chatbot — Critical Production-Readiness Review

> **Status update:** the P0 items in §6 are now implemented on this branch —
> honest refusal path for unmapped questions, removal of the `" vs "` scatter
> trap, borrower-type vocabulary, forecast bridge/MM/conversion vocabulary, LLM
> wiring into `/mi/query` (high-confidence-only zero-cost gate), registry
> `borrower_type` drift fix with generator-equality + synonym-uniqueness
> guards, and a golden-question regression suite
> (`mi_agent/tests/test_chat_golden_regressions.py`,
> `mi_agent/tests/test_registry_governance.py`). The P1/P2 items remain open.

**Scope:** the live MI Agent chat functionality only — `POST /mi/query` end to end:
deterministic parser (`mi_agent/llm_query_parser.py`), semantic registry
(`mi_agent/mi_semantics_field_registry.yaml`), governed-intent routing
(`mi_agent_api/chat_routing.py`), workflow/executor
(`mi_agent/mi_agent_workflow.py`, `mi_agent/mi_query_executor.py`), API wiring
(`mi_agent_api/app.py`, `adapters.py`) and the React chat client
(`frontend/mi-agent-ui/src/api/*`, `components/ChatMessage.tsx`).

**Trigger:** three live questions produced wrong or nonsensical answers:

1. "Generate pipeline bridge to £100MM securitisation size" → whole-book KPI
   summary (2,780 loans / £562.9MM).
2. "Ticket size by borrower type (i.e., single vs joint)" → a **scatter of
   Current LTV vs Borrower Age** over 73 records.
3. "increase in completion conversion rates" → point-in-time summary
   (73 loans / £8.9MM).

All three failure modes were **reproduced locally** against the checked-in
registry and parser (see §2) — they are deterministic code behaviour, not data
or environment flukes.

**Verdict: not production-ready.** The chatbot has no "I don't understand"
path, silently answers a *different* question than asked with full confidence,
contains a hard-coded chart trap triggered by the token `" vs "`, and the LLM
path — the thing that would have rescued these questions — is not wired into
the API at all (and, as configured, would not even be consulted for one of the
three failures if it were). Details and priorities below.

---

## 1. The headline architectural facts

### 1.1 The LLM is not "off" — it is unreachable from the live API

- `mi_agent_api/app.py:1366` hard-codes `parser_mode="deterministic"` and never
  passes `llm_enabled` or `model` into `run_mi_agent_query`. In
  `mi_agent_workflow.py:248`, `effective_llm = bool(llm_enabled) and
  parser_mode == "llm"` — permanently `False` for the API.
- `mi_agent/mi_agent_config.py` (`ENABLE_LLM_MI_AGENT`,
  `MI_AGENT_LLM_PROVIDER`, `MI_AGENT_LLM_MODEL`, key-missing fallback with
  human-readable warnings) is **never imported by `app.py`**. Setting the env
  vars and adding an API key changes nothing. The config module, its
  warning/status machinery, and `/health` LLM status are all dead code from the
  chat endpoint's perspective.
- The governed-intent router (`chat_routing.py:637`) also calls
  `_deterministic_parse` directly, so compare/forecast/risk/evolution intent
  detection is regex-only regardless of any LLM setting.

**Consequence for the user's hypothesis** ("the LLM is off because I have no
tokens, so the deterministic route is weak"): partially right, but buying
tokens will not fix this. Even after wiring the LLM in:

- Question 2 (**the scatter hallucination**) would *still never reach the
  LLM*: the deterministic parse is rated `parser_confidence="medium"` and
  passes validation, and `parse_with_repair`'s `zero_cost_first` gate
  (`llm_query_parser.py:1829-1831`) accepts any valid high/medium-confidence
  deterministic parse without consulting the LLM. Confirmed with a mock LLM:
  `llm calls: 0`, scatter spec returned.
- Questions 1 and 3 parse at `low` confidence, so they *would* go to the LLM —
  but only once the API actually enables it.

### 1.2 There is no intelligibility gate — unrecognized questions silently answer "the whole book"

When nothing matches, the parser's terminal fallback
(`llm_query_parser.py:1396-1402`) emits
`MIQuerySpec(intent="summary", chart_type="none", aggregation="count")` with
`parser_confidence="low"` and explanation *"Could not map question to a chart
deterministically."* That spec:

- **passes validation** (`mi_query_validator.py:308` — a metric-less summary is
  explicitly valid);
- executes as `{"loan_count": N, "<balance>_sum": Σ}` over the whole active
  frame (`mi_query_executor.py:685-704`);
- is rendered as a confident KPI card with the canned answer *"Here is the
  result for your query, covering 1 group(s)."* (`adapters.py:466-479`).

`parser_confidence` is computed on every branch and then **thrown away**: in
the deterministic-only path it never gates anything
(`llm_query_parser.py:1824-1826`), `describe_spec` omits it
(`mi_agent_workflow.py:83-104`), `adapt_workflow_result` hard-codes
`"assumptions": []` (`adapters.py:617-618`), and the frontend renders the
backend narrative verbatim for KPI results
(`lib/responsePresenter.ts:99`). A low-confidence guess and a high-confidence
exact parse are indistinguishable on screen.

The only refusal mechanism, `_detect_unsupported_concept`
(`mi_agent_workflow.py:116-157`), is a denylist of 9 known-missing concepts
(arrears, NNEG, credit score, …). It is not a positive "did I understand
this?" check and fires for none of the three failing questions.

**This is the single biggest go-live risk.** In front of an IC or an investor,
a question the bot cannot parse returns a real-looking, reconciled,
wrong-question answer with zero disclosure.

---

## 2. Root-cause traces for the three failing questions (reproduced)

Reproduced by running `_deterministic_parse` against the checked-in registry.

### Q1: "Generate pipeline bridge to £100MM securitisation size"

- `_FORECAST_SCALE_RE` (`llm_query_parser.py:573-584`) does **not** fire:
  the vocabulary knows "run-rate", "scale-up", "reach £N", "securitisation
  **scale**", "milestone" — but not "**bridge**", "securitisation **size**",
  or "pipeline bridge". Confirmed: `regex fires: False`.
- Even if it fired, the target parser `_forecast_target_value`
  (`llm_query_parser.py:587-597`) **cannot read "£100MM"**: the `m(illion)`
  regex requires a word boundary after a single `m`, so `100mm` fails; the
  fallback needs 4+ raw digits. Confirmed: `£100MM → None`, `100mm → None`,
  and **`£0.1bn → None`** (no bn multiplier either). "MM" is standard
  securitisation notation — this will recur constantly.
- Result: terminal low-confidence summary fallback → whole funded book
  (2,780 / £562.9MM) presented as the answer. The intended analytic — a
  bridge from current funded balance through weighted pipeline to a £100MM
  target — is exactly what `forecast_bridge.py` / `_route_forecast`'s
  `pipeline_needed` / `reach_threshold` kinds already compute. The data and
  service exist; the question can't reach them.

### Q2: "Ticket size by borrower type (i.e., single vs joint)"

Confirmed parse:
`chart=scatter, x=current_loan_to_value, y=youngest_borrower_age, aggregation=loan_level, confidence=medium`.

- "ticket size" resolves to `ticket_bucket` and is then **discarded**.
- "borrower type" is not in the parser's hard-coded
  `EXPLICIT_DIMENSION_TERMS` dict (`llm_query_parser.py:234-280`), even though
  the registry **has** a `borrower_type` dimension with exactly the right
  synonyms ("single or joint", "joint borrower", …) and
  `find_field(sem, role="dimension", keywords=("borrower type",))` resolves it
  correctly. The deterministic parser's main paths bypass the registry's
  synonym vocabulary in favour of a small hard-coded dict — the registry
  governance is decorative for dimension detection.
- The phrase "single **vs** joint" contains `" vs "`, which trips the scatter
  branch (`llm_query_parser.py:1315-1327`). That branch **hard-codes the
  axes**: unless the question mentions ltv+rate, any `" vs "` question becomes
  `x = LTV, y = borrower age`, loan-level. "Ticket size" and "borrower type"
  play no part. Any "X vs Y" phrasing — "north vs south", "direct vs
  acquired", "single vs joint" — is converted into the same LTV-vs-Age
  scatter.
- Rated `medium` confidence → would bypass the LLM even if wired (§1.1).
- The 73 records / £8.9MM indicates the query ran over the 73-row pipeline (or
  lens-filtered) frame rather than the funded book, with no on-screen
  disclosure of which dataset answered (§4.3).

Adjacent confirmed bug: `"balance by borrower type"` → bar of balance by
**`amortisation_type`** (medium confidence). The post-`by` strict keyword
match (`llm_query_parser.py:1373-1377`) tokenises "borrower type" and the
token "type" hits `amortisation_type` first in registry iteration order —
despite the module docstring's promise that explicitly requested dimensions
are "never substituted".

### Q3: "increase in completion conversion rates"

- No recognizer covers a conversion-rate question phrased this way:
  `_FORECAST_SCALE_RE` only knows the literal strings "what conversion rate" /
  "completion rate is assumed" / "annualised completion"; the evolution router
  requires both a line-chart parse and an explicit trend marker
  (`chat_routing.py:610-623`) — "increase in" is not in `_EVOLUTION_MARKERS`.
- No metric term matches ("rates" only resolves via the phrase "interest
  rate"). Terminal low-confidence summary fallback again.
- View selection is lexical (`workspace.py:40-54`: substring "forecast" >
  "pipeline" > "funded" > active tab). "completion"/"conversion" select
  nothing, so the active tab's frame answered: the whole 73-case pipeline
  frame summarized (73 / £8.9MM) — **not** completed cases; no stage filter is
  implied by "completion" anywhere.
- The correct analytic exists twice over: KFI→completion conversion is
  computed by `forecast_extrapolation.build_extrapolation`
  (`kfiConversionForecast`) and the weekly funnel by
  `evolution.pipeline_funnel_evolution` — both already surfaced on the
  dashboard. The question simply cannot reach them.

**Common pattern:** for all three questions the analytics already exist in the
dashboard/services; the failure is entirely in NL → intent mapping plus the
absence of an honest refusal path. This should be an explicit go-live
invariant: *any analytic visible on the dashboard must be reachable by at
least one natural phrasing, and any question that can't be mapped must say
so.*

---

## 3. Semantic registry findings

### 3.1 Registry drift landmine — regeneration will delete `borrower_type` (P0)

`mi_semantics_field_registry.yaml` is stamped "AUTO-GENERATED … Do NOT edit by
hand", but the `borrower_type` entry (yaml:339) was **hand-spliced in**:

- it does not exist in `build_mi_semantics_registry.py`'s `CURATION` dict;
- the metadata block still says `field_count: 104 / core: 71 / derived: 14`
  while the file actually contains 105 / 72 / 15 — the delta is exactly
  `borrower_type`;
- it sits out of the generator's sorted order.

Running the documented refresh (`python -m
mi_agent.build_mi_semantics_registry`) will silently delete the one
single-vs-joint dimension the funded prep actually materialises
(`funded_prep.py:198-217`). Fix: add it to `CURATION`, regenerate, and add a
CI check that generator output == checked-in YAML.

### 3.2 Duplicate/dead single-vs-joint concepts

- `borrower_type` (materialised, working) and `borrower_structure` (derived
  from `number_of_borrowers`, which is **virtual and never derived by
  funded_prep** → effectively always empty) both exist, sharing synonyms
  ("single or joint", "joint borrower", "single borrower"). Resolution between
  them currently depends on dict iteration order — i.e., on the hand-edited
  file ordering. The parser's `_borrower_structure_filter`
  (`llm_query_parser.py:958-987`) prefers the dead `borrower_structure` field
  first. Keep one concept.
- Other ungoverned synonym collisions: "vintage" → both `origination_date` and
  `vintage_year`. No synonym-uniqueness test exists.

### 3.3 Registry advertises fields the tape cannot answer

Curated first-class fields with no source column and no prep derivation for
the ERE tape: `ifrs9_stage`, `probability_of_default`/`lgd`/`ead` (+ buckets),
`internal_risk_*`, `equity`, `indexed_*`, DSCR, rental/NOI, `postcode`,
`tenure`, `recoveries_in_period`, `redemptions_received_in_period`,
principal/interest arrears amounts. Queries resolving to them validate against
semantics and then fail (or silently drop) at execution. A
registry-vs-active-columns reachability report should gate onboarding.

### 3.4 Parser vocabulary vs registry vocabulary (systemic)

Dimension detection in the main deterministic paths uses the hard-coded
`EXPLICIT_DIMENSION_TERMS` dict and `_NUMERIC_AXIS_BUCKET`, not the registry's
governed synonyms — so adding a synonym to the registry does **not** make the
chatbot understand it. Meanwhile `semantic_resolver.py` and `funded_prep`
materialise bucket dimensions (`interest_rate_bucket`, `time_on_book_bucket`,
`balance_band`) that have **no registry entry** at all, while the registry's
`ticket_bucket` is unknown to the quantile layer. Three partially-overlapping
vocabularies (registry, parser dicts, prep/quantile) is how "ticket size by
borrower type" falls through every crack.

---

## 4. API / frontend wiring findings

### 4.1 Dataset & as-of context (correctness)

- **`asOfDate` is decorative.** For the funded view,
  `_resolve_query_frame` returns the single env-configured active dataset
  (`app.py:1288-1289`), ignoring both `portfolio_id` run selection and
  `asOfDate`; the response is then *labelled* with the requested asOf. A user
  who selects an earlier run gets latest-snapshot answers under the selected
  date. `/mi/snapshot` **does** resolve the specific run — so dashboard tiles
  and chat can disagree on identical questions.
- **Two independent keyword heuristics pick the dataset** —
  `resolve_active_view` (question wording overrides tab) and chat_routing's
  `_dataset_for` — and they can disagree with each other and with the tab.
  Which dataset actually answered is only in `metadata.datasetContext`, which
  the UI never renders. Q2/Q3 answering from a 73-row frame with no dataset
  disclosure is this gap made visible.
- Default `client_001` fallbacks throughout (`app.py:1282`,
  `chat_routing.py:176`, etc.): a missing/malformed `portfolioId` silently
  answers for a fictitious client instead of erroring.

### 4.2 Mock client can ship to production

`createAgentClient` (`api/index.ts:21-28`): if `VITE_AGENT_API_URL` is unset
at build time, the production bundle silently falls back to
`MockAgentClient` with canned answers, distinguished only by a small "Demo
data" badge. Fail the build instead.

### 4.3 Presentation

- The generic *"Here is the result for your query, covering N group(s)."*
  string (`adapters.py:466-479`) is shown for every KPI/summary answer because
  `presentAnswer` only builds grounded sentences for chart/table artifacts
  (`responsePresenter.ts:92-99`).
- `interpreted` (the "what I understood" text) is transported but never
  rendered; `assumptions` is always `[]` from the backend. The user has no
  window into interpretation — which is precisely what made these three
  failures look like data bugs instead of parse bugs.

### 4.4 Robustness/security (noting; not chat-specific)

- `auth.py` fail-open risk: `MI_AGENT_AUTH_ENABLED` unset/empty → synthetic
  operator principal, all checks bypassed. Make it fail-closed.
- Pervasive `except Exception → logger.warning → degraded payload` (incl. the
  chat router swallow at `app.py:1349-1351`): systemic failures present as
  "no data" answers rather than errors/alerts.
- `/tmp/trakt/mi_platform` scratch mirror for client data; module-level caches
  can serve stale snapshots across requests.

---

## 5. Executor correctness landmines (affect "correct-looking" answers)

1. **`top_n` ranks by balance, not the requested metric**
   (`mi_query_executor.py:620-647`, priority `("balance","count","concentration")`).
   "Top 10 brokers by average LTV" returns the biggest-balance brokers, with
   the disclosure buried as a hidden technical diagnostic.
2. **`between` percent-rescale bug** (`_apply_filters`,
   `mi_query_executor.py:451-455`): rescaling reads a non-existent
   `value["between"]` key, so "LTV between 60 and 80" against fraction-stored
   LTV matches ~nothing, silently.
3. **`concentration_pct` basis mismatch** (`:594-617`): for avg/weighted-avg
   metrics the "% of total" column is share-of-*balance*, unlabelled.
4. **Weighted-average weight silently defaults to balance**
   (`resolve_weight_field`, `:201-226`) with no user-facing disclosure;
   zero-denominator → NaN flows to display.
5. **Percent display correctness hinges entirely on display hints** being
   populated by dataset profiling; a missing hint renders fraction LTV 0.36 as
   "0.4%".

---

## 6. Prioritised recommendations

### P0 — go-live blockers

1. **Honest refusal path.** When the parse hits the terminal fallback or
   `parser_confidence == "low"`, return a controlled "I couldn't map this
   question — here's what I can answer" response (optionally with suggested
   rephrasings drawn from the registry catalogue). Never execute the
   whole-book summary as an answer to an unparsed question.
   (`mi_agent_workflow.py` after L263; `llm_query_parser.py:1396`.)
2. **Kill the `" vs "` scatter trap.** Remove the hard-coded LTV-vs-Age
   default (`llm_query_parser.py:1315-1327`); only emit a scatter when both
   axes are actually resolved from the question; treat categorical "X vs Y"
   (single vs joint, direct vs acquired) as dimension/filter phrasing.
3. **Wire the LLM into `/mi/query`** via `get_llm_config()` (respect
   `ENABLE_LLM_MI_AGENT`), and make `zero_cost_first` accept only
   **high**-confidence deterministic parses — medium-confidence guesses (the
   scatter trap's rating) must go to the LLM when it is available. Surface
   LLM/deterministic status in `/health` and in response metadata.
4. **Fix registry drift**: add `borrower_type` to `CURATION`, regenerate,
   retire or repair the dead `borrower_structure`, add a CI generator-equality
   check and a synonym-uniqueness test.
5. **Golden-question regression suite.** Codify the demo/IC question bank —
   including these three verbatim — asserting route, dataset, spec fields, and
   answer shape. The repo's e2e tests cover the happy governed intents; none
   cover "question the parser cannot understand".

### P1 — before broad user exposure

6. Route dimension detection through the registry synonyms (one vocabulary):
   `EXPLICIT_DIMENSION_TERMS` should be generated from, or replaced by,
   registry lookups so registry edits actually change chat behaviour; fix the
   post-`by` token substitution (`borrower type` → `amortisation_type`).
7. Extend the forecast/bridge vocabulary: "bridge", "securitisation size",
   "MM"/"bn" magnitudes (`_forecast_target_value`), "conversion rate"
   phrasings routed to `kfiConversionForecast` / funnel evolution.
8. Render interpretation in the UI: "Interpreted as: <metric> by <dimension>
   over <dataset> (<asOf>)", including parser confidence and the dataset that
   actually answered (`metadata.datasetContext`), plus real `assumptions`
   (weight field, ranking basis, lens).
9. Make funded point-in-time queries honour the selected run/asOf, or label
   the answer with the actual snapshot used.
10. Fail production builds when `VITE_AGENT_API_URL` is unset (no silent mock).

### P2 — hardening

11. Executor fixes: metric-aware `top_n` ranking, `between` rescale bug,
    concentration basis labelling, weighted-avg disclosure.
12. Fail-closed auth default; reduce blanket exception swallowing on the chat
    path (distinguish "no data" from "something broke").
13. Registry-vs-tape reachability report at onboarding; prune or tier-gate the
    advertised-but-unbacked fields.

---

## 7. Direct answers to the review questions

- **"Is it just because the LLM is off?"** No. The LLM is not merely off — it
  is unwired (`app.py:1366` hardcodes deterministic). And even wired+funded,
  the zero-cost-first gate would keep the worst failure (the `" vs "` scatter)
  away from the LLM because the wrong parse self-rates as medium confidence.
  The deterministic route additionally lacks a refusal path, so its failures
  present as confident answers.
- **"These questions should not be difficult — the analytics are already in
  the dashboard."** Correct, and confirmed: pipeline-bridge/threshold maths
  (`forecast_extrapolation`, `forecast_bridge`), single-vs-joint stratification
  (`borrower_type` materialised by `funded_prep`), and KFI→completion
  conversion (funnel evolution + `kfiConversionForecast`) all exist and are
  dashboard-reachable. Every one of the three failures is a
  question-understanding failure, not an analytics gap — which is why the
  P0 list is about parsing, refusal, and vocabulary rather than new analytics.
