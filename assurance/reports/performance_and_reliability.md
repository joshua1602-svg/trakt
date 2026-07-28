# Performance and reliability (Phase 8)

## Performance (§28)

Measured on the base synthetic portfolio (118 loans, 3 source portfolios),
deterministic parser (no LLM), governed `/mi/query` end-to-end, 40 executions
across 5 representative questions (KPI, distribution, comparison, concentration,
scoped weighted average).

| Path | p50 | p95 | max |
|---|---|---|---|
| Cold (dataset cache reset each call) | 150 ms | 212 ms | 546 ms |
| Warm (dataset cached, 30 s TTL) | 60 ms | 109 ms | 111 ms |

Failure rate: 0/80 (no exceptions).

* No path is close to a React or Copilot timeout (both budget seconds, not
  hundreds of ms). The cold-start max (546 ms) is a first-load dataset read +
  prep; the 30 s dataset cache amortises it.
* The LLM parser is off in this measurement (deterministic path). With
  `ANTHROPIC_API_KEY` set the parser can add network latency on the repair path;
  the deterministic-first design means the common case does not call it. A live
  LLM latency measurement was **not run** (no API key in the assurance
  environment) — flagged as not-tested.

Repeated-query cache behaviour: warm p50 is ~2.5× faster than cold, confirming
the signature-keyed dataset cache is effective. No premature optimisation
undertaken.

## Reliability / failure injection (§29)

Controlled behaviour verified through the governed capability:

| Injected fault | Result | Client sees a stack trace? |
|---|---|---|
| Configured dataset file missing | Falls back to bundled synthetic (test mode); production policy refuses synthetic → fail closed | No |
| Malformed CSV (unterminated quote) | `status=error`, `STORAGE_UNAVAILABLE` | No (trace logged server-side only) |
| Empty portfolio (0 rows) | `status=error`, `CALCULATION_FAILED`, no fabricated KPI | No |
| Unsupported field (NNEG) | `status=error`, `UNSUPPORTED_QUESTION`, "no value was fabricated" | No |
| Foreign `client_id` | `status=blocked`, `TENANT_MISMATCH` | No |

* No raw stack trace reaches the client on any path; failures are categorised
  onto stable error codes and remain auditable.
* Empty and malformed inputs fail closed rather than fabricating a zero.

**Not tested (recorded honestly):** audit-sink failure, cache backend failure,
blob storage unavailability, artefact-generation failure, BSR-load failure, and
registry-file corruption were not injected end-to-end in this pass (the code
paths are written to degrade, per discovery, but were not exercised under
fault). These belong to a follow-up reliability sweep before a multi-tenant or
Copilot/artefact launch.
