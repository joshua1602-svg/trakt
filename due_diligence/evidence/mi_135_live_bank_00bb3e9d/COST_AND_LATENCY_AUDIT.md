# Cost and latency audit — MI 135 bank at frozen product `00bb3e9d`

Read-only. No live call, no deployment, no product change. Every figure below is
from evidence already collected, except where it is explicitly an ESTIMATE and
the estimator is named.

## 1. Model calls per `/mi/query` request

| Arm | Calls per request | Evidence |
|---|---|---|
| Legacy free-form parser | **0** | `_mi_llm_config()` → `enabled=False, available=False, requested=False, status=disabled`. The arm is withdrawn in code (`_FREE_FORM_ARM_STATUS = "withdrawn_unsafe_boundary"`), and `enabled` is documented as always false. |
| Concept-merge arm | **1**, on 82 of 104 (79%) | Envelope `conceptMerge` block: `source: "model"`, `model: "claude-opus-5"` on all 82. Outcome: `no_change` 71, `applied` 11. |
| interpretation_v2 governed Opus | **2 – 6** (a tool-use loop) | `AnthropicInterpreterClient.emit_intent` issues one `messages.create` per round, `max_rounds = 6`, the final round forcing the intent tool. 578 metadata tool invocations over 104 questions (median 6, p90 7). |
| Retries / repair | **0 model retries** | The 32 harness retries were transport-level (30×HTTP 401, 2×HTTP 500). A 401 is rejected at authentication and never reaches the provider. |

## 2. Total provider API calls across the 104 completed

**104 HTTP requests ≠ 104 provider calls.**

```
interpretation rounds   between 208 (2/question floor) and 624 (max_rounds ceiling)
concept-merge calls     82
-------------------------------------------------------------------
TOTAL PROVIDER CALLS    ~290 – 706   for 104 acceptance questions
```

The range cannot be narrowed further from the evidence: `metadata_calls` records
tool INVOCATIONS (578), and a single round may carry several tool_use blocks, so
rounds cannot be derived from it exactly. Nothing in the production record
states the round count.

## 3. Tokens and cost

**Production records NO token telemetry.** All 104 governed records carry
`model.usage = {}`, while `model_id` and `raw_payload` are present and the
redaction list is empty on every one. This is not the redactor — `redact` was
run against the usage keys and they survive intact — and every return path in
`emit_intent` carries the accumulator. The cause is not determinable by reading;
the fact is that **cost cannot be attributed from production evidence today.**

The concept-merge arm DOES record cost, and it is measured, not estimated here:

```
concept-merge, 82 calls      $0.6653 total    median $0.0082/call
```

Interpretation cost is ESTIMATED from `run8_135_signoff` — the same bank, same
model (`claude-opus-5`), same interpreter — over the 102 of 104 question ids it
covers, priced by the product's OWN estimator (`llm_query_parser.estimate_cost`,
opus $5.00/$25.00 per Mtok, cache reads 0.1x, writes 1.25x):

```
input               766,105 tokens
output              128,430 tokens
cache read        3,127,824 tokens
cache write           8,988 tokens
estimated cost         $8.66      median $0.0835/question   p90 $0.1175
```

```
TOTAL ESTIMATED COST OF THE 104 COMPLETED   ~$9.33      (~$0.090 per question)
IMPLIED COST OF THE REMAINING 31            ~$2.8
```

Interpretation is **93%** of spend; concept-merge is 7%.

## 4. Latency

```
interpreter alone (run8 telemetry)   median 16.9s   p90 24.9s   max 39.0s
end-to-end question spacing          median 28.6s   p90 36.8s   max 140.6s
```

Model time is roughly 60% of each cycle; deterministic execution and rendering
are the remainder. No production record carries a latency field — the spacing
above is derived from consecutive `recorded_at` stamps, and includes the
harness's own polling.

**Wall clock is dominated by the harness, not the product:**

```
104 completed questions   08:02:26 -> 08:55:03    52.6 min
31 failed questions       08:55    -> 10:31       ~96 min
total                                             149 min
```

The 31 failures took nearly twice as long as the 104 successes, because each
burned a full 180s evidence poll waiting for a record that a 401 can never
produce. `31 x 180s = 93 min` accounts for it precisely.

## 5. Harness concurrency and waits

- **Strictly serial.** One `for` loop over the cases; no threads, no pool.
- `time.sleep(5.0)` once per transport retry (32 occurrences).
- The evidence poller sleeps `interval` (3.0s) between attempts and
  **re-downloads the entire Kudu evidence log on every attempt**, so each poll
  grows more expensive as the log grows. This is the single largest non-model
  cost in the run and it is entirely harness-side.

## 6. What `LIVE_MODEL_CALLS = 135` meant

**Confirmed: HTTP acceptance requests, not provider API calls.** The collector
increments the counter once per POST to `/mi/query`. The true provider call
count for the same work was roughly 3–7x higher. The field is misnamed; it
counts acceptance cases.

## The finding that decides the options

**None of the 31 outstanding questions targets a migrated capability.**

```
forecast            6      not migrated
limit_assessment    6      not migrated
borrowing_base      6      not migrated
(capability unbound) 13    n/a
---------------------------------------
would exercise a migrated runtime:  0 of 31
can only ever be LEGACY_FALLBACK:  31 of 31
```

Running them through the canary is guaranteed to produce 31 legacy fallbacks. It
would add **no** governed-path information at all.

## Options to finish commercial acceptance

### A — all 31 through the current full canary path
- ~90–210 provider calls, **~$2.8**, ~15 min with the poll fixed (~95 min without).
- Yields: live legacy user-outcome for all 31, plus 31 records confirming the
  governed path declines — which is already known structurally.
- **Buys almost nothing on the migration axis.**

### B — end-to-end only the migrated capabilities, reuse interpretation evidence
- The migrated subset of the 31 is **empty**, so this degenerates to running
  nothing and taking all 31 from `run8`.
- **~$0**, but `run8` is vocabulary 2.0.0 on a different build and measures
  interpretation only — never serving, never the user-facing answer. It cannot
  tell you whether a lender gets a correct answer today.
- Weaker claim than the bank is meant to support.

### C — legacy-path measurement with the canary OFF (strictly better than A)
- For questions the governed path provably cannot serve, the user-visible answer
  is identical whether the canary is on or off — it is the legacy envelope
  either way. With `MI_AGENT_PLAN_SERVE=off` (and shadow off), **no
  interpretation_v2 call is made at all**.
- Cost falls to the concept-merge arm alone: **~$0.25 for all 31**, an ~11x
  saving over A, for the same user-facing evidence.
- What is given up: the governed record's `fallback_reason` for those 31. That
  is predictable from capability and already evidenced 82 times in this bank.
- Requires one operator config change and no product change.

**Recommendation: C**, then treat the bank as complete for user-facing
correctness, with the migration axis measured — as it already is — on the 104
questions where a governed runtime could actually engage.

Two harness fixes should land before any further run, both free:
1. Never poll for an evidence record after a transport failure (saves 93 min).
2. Rename `LIVE_MODEL_CALLS` to `ACCEPTANCE_REQUESTS`, and record provider round
   counts if the product ever exposes them.

And one product observation, recorded not fixed: **token usage is not being
persisted in governed evidence**, so cost cannot be attributed per request in
production. That is a commercial-observability gap independent of this bank.
