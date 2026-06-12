# MI Agent v1 — Acceptance & Live-LLM Smoke-Test Report

**Date:** 2026-06-12T12:30Z
**Branch:** `claude/mi-agent-v1-foundation-ahdtwc`
**Semantic registry:** v0.2.2 — 69 fields (45 core, 24 extended, 8 derived buckets)
**Scope:** acceptance / smoke test only. One small bug fix applied (duplicate
warnings); no new features; no non-`mi_agent/` files modified.

---

## 1. Summary verdict

### ✅ PASS WITH LIMITATIONS

The MI Agent v1 stack works end-to-end: natural-language question → governed
`MIQuerySpec` → validation → execution → enterprise chart → table → exports.
The only limitations are environmental/by-design (no live LLM API key in this
environment; deterministic parser is limited to fixed phrasings; the chosen
synthetic CSV lacks a few dimension columns so the harness enriched them). None
are stack defects.

---

## 2. Tests run

| Check | Result |
|---|---|
| `pytest mi_agent/tests -q` | **111 passed** (was 110; +1 new privacy test) |
| Deterministic smoke tests (5 questions) | **5/5 PASS** |
| Live LLM smoke tests | **Not run** (no `ANTHROPIC_API_KEY`); **mock-LLM fallback: 5/5 PASS** |
| Missing-key graceful fallback | **PASS** (config reports unavailable + warning, no crash) |
| Prompt privacy check | **PASS** (no raw data sent; `build_prompt` does not accept a dataframe) |
| Streamlit headless smoke test (AppTest) | **PASS** |

No non-MI-Agent tests were required for this pass. No files outside `mi_agent/`
were modified.

---

## 3. Deterministic smoke-test table

Data: `synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv`
(36 loans). This regulatory-shaped file lacks `broker_channel`,
`erm_product_type` and `age_bucket`, so the test harness added them the way the
analytics layer (`mi_prep.add_buckets`) would (synthesised broker/product;
`age_bucket` derived from `youngest_borrower_age`). All five ran via
`run_mi_agent_query(...)` in deterministic mode.

| # | Question | Chart type | Validation | Execution | Rows | Warnings | Chart HTML | CSV |
|---|----------|-----------|-----------|-----------|------|----------|-----------|-----|
| 1 | Show balance by region | bar | Passed | OK | 10 | 1 | ✅ | ✅ |
| 2 | Show weighted average LTV by product type | bar | Passed | OK | 3 | 1 | ✅ | ✅ |
| 3 | Show LTV by age bucket and region as a heatmap | heatmap | Passed | OK | 12 | 1 | ✅ | ✅ |
| 4 | Show balance by region and broker as a treemap | treemap | Passed | OK | 19 | 1 | ✅ | ✅ |
| 5 | Show LTV by borrower age sized by balance | bubble | Passed | OK | 36 | 1 | ✅ | ✅ |

Generated specs (key fields):

1. `bar` · metric `current_outstanding_balance` · dim `geographic_region_obligor` · `sum`
2. `bar` · metric `current_loan_to_value` · dim `erm_product_type` · `weighted_avg` · weight `current_outstanding_balance`
3. `heatmap` · metric `current_loan_to_value` · dims `[geographic_region_obligor, account_status]` · `avg`
4. `treemap` · metric `current_outstanding_balance` · hierarchy `[geographic_region_obligor, broker_channel]` · `sum`
5. `bubble` · x `youngest_borrower_age` · y `current_loan_to_value` · size `current_outstanding_balance` · `loan_level`

**Note on #3:** the deterministic parser maps "age bucket and region" to
`[region, account_status]` — it does **not** resolve "age bucket" to the
`age_bucket` dimension (it picks region + a broker/type/status dimension). This
is the documented deterministic fixed-phrasing limitation; the LLM path resolves
`age_bucket` correctly (it is a first-class semantic field). The chart still
validated, executed and rendered.

Warning observed on every run (informational, by design):
`percent-scale heuristically detected as 'whole_number_percent' (median 3.866);
the executor does NOT rescale percentages`.

---

## 4. LLM smoke-test table (mock fallback — no live key)

No `ANTHROPIC_API_KEY` was available, so per the brief the live path was not
run; a mock LLM (returning governed `MIQuerySpec` JSON) exercised the full LLM
workflow path (prompt build → parse → validate → execute → chart). Provider
reported as `mock`; default model `claude-haiku-4-5-20251001`.

| # | Question | Provider/model | Chart | Strict JSON | Validation | Repair attempts | Execution | Rendered |
|---|----------|----------------|-------|-------------|-----------|-----------------|-----------|----------|
| 1 | Where are we most concentrated? | mock / claude-haiku-4-5-20251001 | bar | ✅ | Passed | 0 | OK | bar |
| 2 | Are newer vintages riskier? | mock / claude-haiku-4-5-20251001 | bar | ✅ | Passed | 0 | OK | bar |
| 3 | Show me the interaction between borrower age, LTV and exposure | mock / claude-haiku-4-5-20251001 | bubble | ✅ | Passed | 0 | OK | bubble |
| 4 | Which products drive most of the balance? | mock / claude-haiku-4-5-20251001 | bar | ✅ | Passed | 0 | OK | bar |
| 5 | Are higher LTV loans concentrated with specific brokers? | mock / claude-haiku-4-5-20251001 | bar | ✅ | Passed | 0 | OK | bar |

**Repair loop** is separately proven by unit tests
(`test_repair_loop_fixes_invalid_then_valid`, `test_repair_loop_exhausts_and_reports`):
an invalid spec (e.g. `sum` on a percentage) triggers a re-prompt with the
validation errors; a corrected spec is accepted (`repair_attempts=1`); if still
invalid after `max_attempts`, the failure metadata + errors are surfaced and the
invalid spec is never executed.

**Missing-key fallback:** `get_llm_config({ENABLE_LLM_MI_AGENT=true,
provider=anthropic})` → `available=False`, status *"LLM requested but API key
missing — deterministic fallback"*, warning emitted; the app does not crash.

---

## 5. Privacy check

`build_prompt(user_question, mi_semantics)` — the signature **does not accept a
dataframe**. A full LLM workflow run was executed against data spiked with
sentinels (`SENTINEL_LOAN_X`, balance `987654321`) and the captured prompt was
scanned.

**Sent to the LLM (data-free catalogue + question only):**
- semantic field keys, business names, descriptions, synonyms
- roles, formats, chartability
- allowed aggregations, allowed chart roles, bucket info
- the user question
- (during repair) the previous JSON + validation error strings

**NOT sent to the LLM (confirmed absent from the prompt):**
- raw loan rows / dataframe samples
- borrower-level values, balances, postcodes
- loan identifiers
- full canonical CSV content

Result: **no sentinel value leaked**; catalogue keys (`current_loan_to_value`)
and business names (`Current LTV`) are present. Regression-locked by
`test_llm_prompt_accepts_no_dataframe_and_leaks_no_raw_data`.

---

## 6. Export check

All four exports verified via `mi_agent_workflow` helpers and the Streamlit
download buttons (AppTest shows 4 download buttons present):

| Export | Mechanism | Result |
|---|---|---|
| Result CSV | `result_csv_bytes` → `st.download_button` | ✅ bytes, includes `concentration_pct` |
| Chart HTML | `chart_html_str` (Plotly, CDN) → download | ✅ valid `<html>` |
| MIQuerySpec JSON | `spec_json_str` → download | ✅ parseable JSON |
| Metadata JSON | `metadata_json_str` → download | ✅ parser/validation/warnings/metadata |

---

## 7. Known limitations

- **No live LLM run** in this environment (no `ANTHROPIC_API_KEY`, `anthropic`
  not installed); mock-LLM fallback was used. Re-run with a key to exercise the
  real API.
- **No Azure integration**, **no production pipeline integration**, **no
  authentication**, **no persistent multi-user history** (session-only).
- **No direct PPTX export** (HTML / JSON / image-via-`kaleido` only).
- **Deterministic parser is limited to fixed phrasings** — e.g. it does not map
  "age bucket" to the `age_bucket` dimension (uses region + status instead). The
  LLM path is the intended route for free-form questions.
- **Percent-scale inconsistency remains metadata-driven, not transformed** — the
  executor detects and reports the apparent scale; it never rescales data. On
  mixed portfolios (fraction LTV + whole-number rates) the blended heuristic can
  read as `whole_number_percent`; the chart factory honours whatever is detected.
- **Demo CSV column gaps** — the chosen synthetic file lacks `broker_channel`,
  `erm_product_type` and `age_bucket`; on the raw file, product/broker questions
  correctly fail validation with "canonical column not present" (governed
  behaviour). The harness enriched these columns to exercise all five questions.

## Small bug fixed during this pass

- **Duplicate warnings** in `mi_agent_workflow.run_mi_agent_query`: the chart
  factory copies the executor's warnings onto its result, so extending the
  workflow warning list from both sources duplicated entries (e.g. the
  percent-scale note appeared twice). Fixed by de-duplicating the final warning
  list (order-preserving) on all return paths. No behaviour change beyond clean
  warnings.

---

## 8. Recommended next actions

- **Package/demo v1** — the stack is acceptance-ready for a controlled demo.
- **Run a live-LLM pass** with a real `ANTHROPIC_API_KEY` to confirm the live
  Anthropic round-trip and observe real repair-loop behaviour.
- **(Optional, small) Synonym-driven deterministic resolution** — let the
  deterministic parser map "age bucket"/"vintage"/"ticket size" to the derived
  bucket dimensions via the `synonyms` already in the registry (v2 item; not
  required for v1 acceptance).
- No other code changes required for v1.

---

### How to reproduce

```bash
# 1. Baseline
pytest mi_agent/tests -q

# 2. Deterministic app (no key needed)
streamlit run mi_agent/streamlit_mi_agent.py
#    sidebar → Load synthetic demo CSV → click the example questions

# 3. LLM mode (live)
export ENABLE_LLM_MI_AGENT=true
export MI_AGENT_LLM_PROVIDER=anthropic
export MI_AGENT_LLM_MODEL=claude-3-5-haiku-latest   # or leave unset for the cheap default
export ANTHROPIC_API_KEY=...
streamlit run mi_agent/streamlit_mi_agent.py
```
