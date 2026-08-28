# MI Query Agent — API cost, from retained records

No model call was made to produce this measurement. Every figure comes from
`metadata.conceptMerge.usage` retained on envelopes captured during acceptance,
priced by the repository's own `mi_agent.llm_query_parser.estimate_cost`.

    opus family    $5.00 / 1M input     $25.00 / 1M output
    cache read     0.1x input           cache write   1.25x input

---

## 1 · What survived, and what did not

| retained record | rows | usage present |
|---|---:|---:|
| 166 bank, final (`23804de`) | 166 | **166** |
| 166 bank, recalibrated start | 166 | 165 |
| 166 bank, previous sprint end | 166 | 165 |
| 166 bank, previous sprint start | 166 | 165 |
| 216-call stability run | 216 | **0** |
| 1,446 surface, both sweeps | 2,892 | **0** |
| 24 CR4, Q16B ×20, controls ×180 | 224 | **0** |

**The stability, surface, CR4 and control harnesses discarded the usage
metadata.** They recorded `conceptMerge.status` and the model id and dropped the
rest, so roughly 3,300 model calls cannot be priced and never will be. Only the
four bank captures kept the whole `conceptMerge` object.

That gap is the reason for §5.

---

## 2 · Headline, from the final run at the production SHA

166 questions, 166 successful API calls, all `claude-opus-5`.

| | |
|---|---:|
| Successful API calls | **166** |
| Total estimated cost | **$0.7958** |
| Mean per question | **$0.004794** |
| Median | $0.004188 |
| p95 | $0.008860 |
| Maximum | $0.018013 (Q01A) |
| Minimum | $0.001740 |
| **Estimated cost per 1,000 questions** | **$4.79** |

Mean tokens per call:

| input | output | cache read | cache write |
|---:|---:|---:|---:|
| 22.6 | 138.8 | 2,093.6 | 26.4 |

Cache hit rate: **98.8% by call** (164 of 166), **98.8% by token**
(347,543 read of 351,930 read + written).

Cost per question equals cost per call here because the arm makes at most one
call per question and made one on every question in this run.

### Corroboration across four independent runs

| run | calls | total | mean | per 1,000 |
|---|---:|---:|---:|---:|
| final (`23804de`) | 166 | $0.7958 | $0.004794 | **$4.79** |
| recalibrated start | 165 | $0.8038 | $0.004871 | $4.87 |
| previous sprint end | 165 | $0.7792 | $0.004723 | $4.72 |
| previous sprint start | 165 | $0.7893 | $0.004784 | $4.78 |

Spread of 3% across ~660 calls.

---

## 3 · The warm cache flatters this, and by how much

**$4.79 per 1,000 is a lower bound, not a production forecast.** A 166-question
sweep runs back-to-back, so the 2,110-token vocabulary prompt stays hot and 98.8%
of calls billed it at 0.1x. Production traffic arriving after the cache TTL pays
1.25x for the same tokens.

The same 166 calls, repriced:

| regime | total | mean | per 1,000 |
|---|---:|---:|---:|
| as measured — warm cache | $0.7958 | $0.004794 | **$4.79** |
| no prompt caching at all | $2.3542 | $0.014182 | $14.18 |
| every call a cold cache write | $2.7942 | $0.016832 | **$16.83** |

A single cold call made while wiring the telemetry below billed
`cache_creation_input_tokens: 2110` and cost **$0.016293** — within 3% of the
cold-regime estimate, which corroborates it.

**Plan on $5–$17 per 1,000 questions depending on traffic density.** Sparse
interactive use sits near the top; a batch or a busy period sits near the bottom.

Two further caveats, both widening the range downward:

* the deterministic configuration makes **no calls at all** — $0 per 1,000;
* a question that reaches no arm (arm disabled, or provider unavailable) costs
  nothing, so blended cost falls with the unavailability rate.

---

## 4 · What drives the spread

Cost is dominated by output tokens: input averages 22.6 tokens against 138.8
output, and output bills at 5x. The maximum, Q01A at $0.018013, is the first
question of the sweep — it paid the cache write for the whole vocabulary prompt.
Every later question rode that write. In production the write recurs whenever the
cache goes cold, which is what §3 prices.

---

## 5 · Per-query cost telemetry, added for go-live

The arm published raw token counts and nothing priced them, so cost per question
could only be reconstructed by whoever still held the envelopes — and most
harnesses had already discarded them.

`mi_agent_api/concept_merge_arm.py` now publishes `metadata.conceptMerge.cost`
beside the existing `usage`, priced by `estimate_cost` — the estate's existing
owner of the pricing table, so there is no second opinion about what a token
costs. A replayed proposal is not a call and is not priced.

```json
"cost": {
  "input_tokens": 16, "output_tokens": 121, "total_tokens": 2247,
  "cache_read_tokens": 0, "cache_write_tokens": 2110,
  "estimated_input_cost": 0.013268, "estimated_output_cost": 0.003025,
  "estimated_total_cost": 0.016293, "cost_estimate_status": "estimated"
}
```

`cost_estimate_status` is `"unknown"` rather than a silent $0 when a model has no
pricing entry, so an overridden model surfaces as unpriced instead of free.

**This is a production change and it invalidates the `23804de` freeze.**
Re-verified:

* frozen regression manifest — **85, name for name**;
* the code is inside `apply()`, reached only when the arm is enabled, so the
  deterministic configuration cannot touch it;
* arm, cost-hardening and availability suites — 61 passed, 1 failed, and that
  failure (`test_layered_question_routes_to_llm_even_when_deterministic_parses`)
  is one of the frozen 85.

The freeze tag stays at `23804de` and is **not** moved. The SHA carrying
telemetry is a separate deployment candidate.
