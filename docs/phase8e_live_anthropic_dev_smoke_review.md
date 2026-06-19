# Phase 8E — live Anthropic dev smoke review

*Status: dev-only, manual evaluation harness + review. This is **not** production
integration and has **no** CI dependency on Anthropic.*

---

## Purpose

Phases 8A–8C proved the governed pipeline with deterministic interpreters and
**fake** Anthropic clients. Phase 8E adds a controlled, manual way to ask a real
Claude model the same fixed set of MI questions and observe whether it interprets
them into MIQuerySpec v2 accurately enough to run against the existing
synthetic/local MI runtime.

The model is held to exactly the same boundary as everywhere else:

* Claude **proposes a spec only** — it never computes analytics.
* Output is routed through `MIQuerySpec.normalized()` and `validate_query_spec()`.
* Execution happens **only** via `interpret_and_run_mi_query` → `run_mi_query`.
* Invalid or ambiguous interpretations are **never executed**.

---

## How to run the smoke locally

```bash
# From the repo root, with the optional anthropic SDK installed in your env:
ANTHROPIC_API_KEY=sk-... python scripts/phase8e_live_anthropic_smoke.py

# Optional: choose a different (gitignored) output path
ANTHROPIC_API_KEY=sk-... python scripts/phase8e_live_anthropic_smoke.py --out artifacts/my_run.json
```

The script prints a per-question `PASS/FAIL` line and writes a full JSON artefact
(see below). It exits `0` if every question matched its expected safe behaviour,
`1` if any question failed grading, and `2` if the API key is missing.

### Required environment

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | **Yes** | Authenticates the real Anthropic client. The script refuses to run (exit 2) without it. No key is read from any other source. |

The `anthropic` Python SDK must be installed in the local environment. It is
imported **lazily** by `AnthropicClient`, so the normal test suite and repo do
**not** depend on it.

---

## What data is used

* Three small, **synthetic, in-memory** MI snapshot frames (the same shape as the
  Phase 6B/8C fixtures) registered in a `LocalFsSnapshotStore` under a temporary
  directory.
* No files outside the temp directory are read; the store is discarded on exit.

## What is sent to Anthropic

* Only the **constrained prompt** built by `build_mi_spec_prompt` (Phase 8B): the
  hard rules, the allowed fields/enums/route ids, the known semantic field keys,
  the interpretation rules, the synthetic context anchors (client id `phase8e_devsmoke`,
  route `mi`, the three fabricated dates), and the **question text** from the
  fixed list below.
* **No client data, no PII, no row-level loan data** is ever sent. The snapshot
  frames stay local; only the question and the governed vocabulary go to the
  model.

### Why no real client data should be used

This is an uncontrolled, exploratory call to an external provider. Sending real
client or borrower data would (a) transmit confidential data to a third party,
(b) risk it being logged or retained outside our control, and (c) provide no
analytical benefit — the model only needs the *question* and the *governed
vocabulary*, never the data, because it does not compute results. The harness is
hard-wired to synthetic data for this reason.

---

## The controlled question set

Twelve questions expected to **execute**, five expected to **clarify** (not
execute). This mirrors the Phase 8A golden set and the Phase 8D capability map.

| Question | Expected safe behaviour |
|---|---|
| show total funded | execute (state) |
| show total pipeline | execute (state) |
| show forecast funded | execute (state) |
| trend funded balance over the last three months | execute (temporal trend) |
| compare funded balance to last month | execute (temporal compare) |
| show funded balance by portfolio | execute (risk concentration) |
| show funded balance by region | execute (risk concentration) |
| show pipeline by stage | execute (risk concentration) |
| show concentration by region | execute (risk concentration) |
| show risk grade migration | execute (risk migration) |
| show IFRS stage migration | execute (risk migration) |
| show PD bucket migration | execute (risk migration) |
| show risk | clarify (vague risk view) |
| show changes | clarify (no period) |
| show stage | clarify (ambiguous stage) |
| show portfolio | clarify (ambiguous portfolio measure) |
| show rate | clarify (metric vs buckets) |

---

## Evaluation output

For each question the harness records, into
`artifacts/phase8e_live_anthropic_smoke_results.json`:

* `raw_question`;
* `expected_behaviour` (`execute` / `clarify`);
* `raw_claude_output` (the exact text the model returned);
* `parsed_candidate_spec` (the parsed JSON object, pre-normalisation);
* `normalized_spec` (after `MIQuerySpec.normalized()`);
* `validation_result` (`ok` + issue `codes`);
* `executed` (true/false);
* `runtime_result_summary` (mode, row_count, ok, chart_instruction,
  metadata keys) when executed;
* `issue_codes`;
* `clarification_question` (if any);
* `passed` (graded against expected safe behaviour).

**The artefact is gitignored** (`artifacts/*` with a tracked `.gitkeep`) because
it can contain raw provider responses. Do not commit it.

### Pass/fail criteria

* **execute** questions pass iff the interpretation produced a valid spec that
  executed and the runtime result was `ok`.
* **clarify** questions pass iff nothing executed — either the model returned a
  clarification, or the proposed spec was blocked by validation. Both are safe
  outcomes; the key property is that an ambiguous question never silently runs.

### How to interpret results

* A high execute-pass rate means Claude maps clear MI questions onto the governed
  vocabulary reliably.
* Any **execute → fail** is an interpreter/prompt gap (wrong field, wrong mode,
  hallucinated value, or missing date anchor). Inspect `raw_claude_output` and
  `validation_result.codes` to see why.
* A **clarify → fail** would be the serious case — it means an ambiguous question
  produced a spec that executed. The governed gates make this unlikely, but it is
  exactly what this smoke is designed to catch.
* Because LLM output is non-deterministic, treat a single run as indicative, not
  definitive; re-run and compare.

---

## Observed results

**The live smoke was NOT run as part of this phase.** No `ANTHROPIC_API_KEY` is
available in the development/CI environment, and this phase explicitly must not
add any CI dependency on Anthropic.

What **was** verified here, without any provider access:

* The script **fails safely without a key** — `run_smoke()` / `main([])` return
  exit code `2`, write nothing, and import no SDK (verified by
  `tests/test_phase8e_live_anthropic_dev_smoke.py`).
* Importing the script does **not** pull in the `anthropic` SDK or touch the
  network.
* The controlled question set matches the documented Phase 8A/8D set.
* The **governed path the live smoke uses** behaves correctly with **fake**
  clients: a valid spec executes (3 funded rows), malformed output does not
  execute, an invalid enum is blocked (`invalid_enum_value`), and a clarification
  object does not execute. The grading helper grades execute/clarify correctly.

When a key is available, run the command above and paste the resulting
`passed/total` and any failing rows into this section.

> _Results table to be filled in after a live run:_
>
> | Question | executed | passed | issue codes / clarification |
> |---|---|---|---|
> | _(populate from artifacts/phase8e_live_anthropic_smoke_results.json)_ | | | |

---

## Prompt / interpreter weaknesses to watch for

These are the most likely failure modes to look for in a live run (informed by
the deterministic baseline and the Phase 8D review):

* **Date anchoring** — Claude must use only the supplied context anchors for
  trend/compare. If it invents dates, the spec may still validate but answer the
  wrong window; if it omits them, expect `temporal_selector_incomplete`.
* **"portfolio" resolution** — should map to `portfolio_id` only when a portfolio
  reference is available, else clarify. Watch for it guessing a dimension.
* **Bare ambiguous terms** — "stage"/"rate"/"risk"/"changes" must clarify, not be
  emitted as a bare dimension (`ambiguous_dimension`).
* **Bucket dimensions** — even if Claude proposes `balance_band` etc. with a
  quantile strategy, the runtime does not currently materialise the bucket column
  (Phase 8D caveat); such a query may validate but not return a meaningful
  grouping.
* **Over-helpful prose / code** — the adapter strips fences and rejects code, but
  a verbose model may wrap JSON in commentary; confirm `parse_spec_json` recovers
  the object.

---

## Recommended follow-ups

* **Run the live smoke** in a controlled dev environment with a key and record
  the results here.
* If execute-pass rate is low, **harden the prompt/interpreter** (Phase 8B/8A) —
  expand interpretation rules and golden coverage — rather than loosening
  validation.
* Consider a small **repeat-run harness** (e.g. N runs per question) to quantify
  non-determinism before trusting the model for a demo.
* Keep this strictly **dev-only** until production controls exist (key
  management, rate limits, logging, cost controls) — see Phase 8D's production
  gaps.

---

## Scope / exclusions honoured

Dev-only live-Anthropic smoke harness + review and safe-behaviour tests only. No
production LLM integration, no external calls in tests, no API key required for
tests, no real client data, no Azure, no onboarding orchestration, no
Streamlit/UI, no M&A runtime, no Annex 2/XML/regulatory changes, no new chart
types. The LLM never computes analytics and never bypasses MIQuerySpec
validation.
