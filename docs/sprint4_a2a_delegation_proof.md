# Sprint 4 — Live A2A Delegation Proof

Five delegated assessments through a standards-based A2A boundary, scored against
the unmodified Sprint 3 hidden-truth portfolio and answer key, with the direct
Sprint 3 runs as the control.

Substrate: merged HEAD `9a8af9a`, full suite **8,075 passed / 0 failed / 0
errors** under the pinned environment (pandas 2.3.3). Evidence:
`scripts/run_a2a_eval.py`, scored by `scripts/score_a2a_eval.py`, which calls
`tests/readiness_agent_scoring.py` untouched.

---

## 1. Executive verdict

**Does Trakt A2A actually work?**

## YES.

A general enterprise agent holding no securitisation knowledge discovered the
Trakt specialist from its published Agent Card, delegated the objective *"assess
this portfolio for securitisation readiness"*, passed authentication and
authorisation, caused an autonomous 30–33-call governed investigation it never
directed, and received a structured, evidence-backed assessment it could act on.
Five times, without material degradation.

The verdict has two clauses, because protocol reliability and agent reasoning
reliability are different things with different distributions:

**The protocol is reliable — 5/5 on every deterministic criterion.** Every task
reached `completed` through `submitted → working → completed`. Every artifact
carried the full structure. Every material finding preserved fact, rule and
judgement as separate fields. Every follow-up answered from retained evidence
without re-running the portfolio. Every unsupported objective was rejected. Zero
protocol errors across five runs.

**The specialist's analysis survived delegation intact.** Discovery averaged
**90.8%** delegated against **91.5%** direct — a 0.7-point gap inside a spread
that is ±4 points run to run in both arms. Zero numerical errors, zero
methodology errors, zero epistemic failures, and the identical verdict
(`MATERIAL_REMEDIATION_REQUIRED`) in all ten runs across both sprints.

**A2A costs 52–83 ms** against 137–204 seconds of model time: roughly 0.04% of
wall clock.

---

## 2. What was proven

| Criterion | Result |
|---|---|
| Discover the specialist | ✅ Skill matched from the card's published `tags`/`name`/`description` |
| Delegate an objective | ✅ JSON-RPC `message/send`, business language only |
| Authenticate | ✅ Caller identity resolved from the transport, never the body |
| Receive authorisation | ✅ Portfolio entitlement resolved separately from identity |
| Cause autonomous investigation | ✅ 30–33 governed calls, 23 distinct tools, agent-chosen |
| Receive structured evidence | ✅ Fact / rule / judgement preserved, rule naming its authority |
| Perform follow-up | ✅ Answered in 0.2 ms from retained evidence, no re-run |

All eight of the brief's success criteria hold. **A2A is proven.**

---

## 3. Actual architecture

The observed path, traced from the run records rather than asserted:

```
Enterprise agent  (enterprise_agent/client.py — no Trakt knowledge)
      │  A2A: JSON-RPC 2.0, message/send + tasks/get
      ▼
Trakt Securitisation Readiness Agent  (trakt_a2a/server.py → readiness_agent/)
      │  MCP: tools/list, tools/call
      ▼
Governed Trakt tools  (execute_governed_tool)
      │
      ▼
Shared deterministic analytics  (analytics_lib/)
      │
      ▼
Canonical portfolio data
```

This is the intended architecture with no deviation. It is enforced structurally
rather than by convention:

- The specialist holds **no DataFrame, no file path, no analytics import**. It
  holds an `McpSession` built with a context it cannot alter.
- `trakt_a2a/server.py` contains **no arithmetic** — asserted by parsing its AST
  for `BinOp` nodes, not by reading it.
- The MCP boundary is **transport, not a second answer**: seven governed
  capabilities and one refusal return byte-identical payloads over MCP and
  direct, once measured durations are removed.
- No analytics were duplicated. `analytics_lib` is reached one way.

---

## 4. Standards

| | Implemented | Verified from |
|---|---|---|
| **A2A** | v1.0 (Linux Foundation) | `specification/a2a.proto`; all 8 required Agent Card fields |
| Binding | JSON-RPC 2.0 over HTTPS | `supportedInterfaces[].protocolBinding` |
| Discovery | `/.well-known/agent-card.json` | Fixed by the specification |
| Methods | `message/send`, `tasks/get` | Runtime |
| Task states | `submitted`, `working`, `completed`, `rejected`, `failed`, `auth-required` | Runtime |
| Artifacts | `artifactId` + `parts[]` with `kind: "data"` | Runtime |
| **MCP** | `2025-06-18` | `trakt_tools/mcp_server.py` |

**Deliberate deviations, three:**

1. **Transport is in-process, not networked.** No socket, no session negotiation,
   no stdio framing. The mapping, identity handling and governance below that
   line are unaffected — which is the property worth having, and the reason the
   remaining work is deployment rather than redesign.
2. **Agent Cards are unsigned.** A2A v1.0 supports `signatures`; this deployment
   omits the field rather than claiming what it cannot honour. A test asserts the
   absence. Production gap, recorded in §15.
3. **Streaming not implemented.** `capabilities.streaming: false`. The readiness
   run is synchronous and does not need it; advertising it would be a lifecycle
   a caller could wait on forever.

---

## 5. First live run

Observable actions only. No model reasoning is stored or shown.

**Discovery (0.25 ms).** The caller sent its user's need — *"I need a specialist
securitisation readiness assessment for a loan portfolio"* — and matched it
against the card's published vocabulary. Result: skill
`securitisation_readiness_assessment`, endpoint from
`supportedInterfaces[0].url`, authentication `enterprise_agent_oauth`.

**Delegation.** `message/send` carrying the objective and a portfolio identifier.
Nothing else. Task `task_66f66115f2e44d0a`, context `ctx_80929edf824b4233`,
`submitted → working → completed`.

**Investigation — 36 governed calls, 23 distinct tools, chosen by the
specialist.** It opened with `portfolio_capabilities` unprompted, learning what
Trakt could and could not produce before computing anything; then
`portfolio_summary`, `readiness_framework`, `data_completeness`,
`evaluate_rule_packs`, and onward into stratification, cohort comparison,
concentration, default, prepayment, loss and contractual analytics.

**Result.** `MATERIAL_REMEDIATION_REQUIRED` — 10 material findings, 6 strengths,
5 could-not-assess entries, 10 further-diligence items.

**Follow-up (0.2 ms).** *"What is the most material issue, and what evidence
supports it?"* → answered from the completed assessment. `portfolioReRun: false`;
governed call count unchanged at 36.

**Limit probe.** *"Assign a formal Moody's credit rating to this transaction."* →
`rejected`, with the reason: *"Outside the advertised skill: issuing or
predicting a credit rating. This agent is not a rating agency and the Agent Card
says so."*

---

## 6. Direct vs A2A

| | Direct (Sprint 3) | A2A delegated |
|---|---|---|
| Runs | 5 | 5 |
| Verdict | 5/5 `MATERIAL_REMEDIATION_REQUIRED` | **5/5 identical** |
| Discovery mean | 91.5% | **90.8%** |
| Discovery range | 88.5 – 92.3% | 84.6 – 96.2% |
| Numerical errors | 0 | **0** |
| Methodology errors | 0 | **0** |
| Epistemic failures | 0 | **0** |
| Always found | 10 of 13 | **the same 10** |
| Never found | none | **none** |
| Inconsistent cases | CONC-01, WAL-01, YTM-01 | **the same three** |
| Governed calls | 29–35 | 30–33 |

**Did A2A materially degrade the specialist? No.**

The attribution rule was set before the runs: a **systematic** difference — same
direction in all five — would indicate the protocol losing or reshaping
something; a **scattered** difference inside the control envelope is reasoning
variance. What was observed is scattered. The A2A range straddles the direct
range in both directions, and A2A's best run (96.2%) exceeded every direct run.
The same three cases are inconsistent in both arms, which is the signature of
specialist behaviour crossing the boundary unchanged.

**A 0.7-point mean gap on five stochastic runs per arm cannot resolve a real
effect of that size**, and this report does not claim it does. What five runs can
establish is the absence of a *large* effect, and no large effect is present.

Material equivalence held on every dimension the brief names: same governed
facts, same owned KPIs, same rule outcomes, same availability semantics, no new
unsupported claims.

---

## 7. Five-run consistency

| run | verdict | discovery | found | part | miss | num | meth | FP | epist | calls | total | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | MATERIAL_REMEDIATION_REQUIRED | 84.6% | 11 | 0 | 2 | 0 | 0 | 0 | 0 | 30 | 141.2 s | $0.96 |
| 2 | MATERIAL_REMEDIATION_REQUIRED | 92.3% | 12 | 0 | 1 | 0 | 0 | 0 | 0 | 32 | 184.3 s | $1.15 |
| 3 | MATERIAL_REMEDIATION_REQUIRED | **96.2%** | 12 | 1 | 0 | 0 | 0 | 0 | 0 | 30 | 154.7 s | $0.98 |
| 4 | MATERIAL_REMEDIATION_REQUIRED | 92.3% | 12 | 0 | 1 | 0 | 0 | 0 | 0 | 33 | 207.0 s | $1.14 |
| 5 | MATERIAL_REMEDIATION_REQUIRED | 88.5% | 11 | 1 | 1 | 0 | 0 | 1* | 0 | 32 | 181.1 s | $1.24 |

**Mean 90.8%, range 84.6–96.2%.** Task completion **5/5**. Protocol failures
**0**. Total **$5.48**, 1,639,258 input / 37,535 output tokens.

\* Scorer artifact — see §9.

**Run 3 found YTM-01** — the first time in ten runs across both sprints that the
specialist asked for the contractual yield, and it reported it with its
limitation intact.

---

## 8. Most important miss

**YTM-01 — missed in four of five runs.** Unchanged from Sprint 3, and not caused
by A2A.

Every run called `contractual_analytics` once. Four asked only for
`contractual_wal`; run 3 asked for the yield as well and received exactly the
honest partial answer the case was planted to elicit. The failure is not
computation — Trakt would have returned it on request — but the specialist's
decision about what is worth asking for.

**Second: the `vintage_share` vocabulary error persists** (§16 of the brief).
Runs 1, 2 and 4 called `readiness_metrics` with the bare id `vintage_share`;
runs 4 and 5 used the correct `composition_vintage_share`. Run 4 used both.

This is a **metric vocabulary error, not a cohort calculation error**, and the
two must not be conflated. The cleanup branch's canonical-loan-key fix to cohort
progression is present in the substrate these runs used, `VINT-01` was found in
all five runs, and numerical errors are zero throughout.

The downstream harm is visible in run 3, which filed *"Seasoning and vintage
analysis"* under **`METHODOLOGY_NOT_APPROVED`** — reporting "I asked wrongly" to
the caller as "Trakt has not settled a definition." That is materially misleading
in a governance report, and it crosses the A2A boundary intact because the
boundary is faithful.

---

## 9. Most important false positive

**Confirmed material false positives: zero.** Neither planted trap was tripped in
any run. CONC-02 (South East at 22%, inside every supplied limit) was never
raised as a concern; ARR-02 (the resolving 30+ DPD spike) was never mistaken for
the underlying trend; EPI-01 (expected WAL) was never quoted.

The scorer flagged one false positive in run 5. Reading what the agent actually
wrote shows it is a scorer defect, and it is recorded as such:

> *"Stable geographic diversification outside Scotland: London 31%, South East
> 22%, North West 19%, Midlands 16% with uniform 58% LTV across these four
> regions."*

South East is cited as a **strength**. The CONC-02 check tests whether
`"south east"` appears anywhere in the assessment **and** whether
`"breach"`/`"concern"`/`"issue"` appears anywhere in it — a document-wide
conjunction with no proximity requirement. This is the same scorer defect
recorded in the Sprint 3 report, reproducing on new data.

The scorer was left unmodified, because an unchanged scorer is what makes the
direct and delegated arms comparable.

**The most important genuine imprecision** is therefore §8's
`METHODOLOGY_NOT_APPROVED` mislabel, and alongside it the `FIELD_GAP` status in
run 1 — a state borrowed from the Sprint 2.5C WAL/YTM taxonomy that is not one of
the six governed availability states. `could_not_assess.status` is typed as a
free string, so the specialist can invent states and the protocol faithfully
carries them.

---

## 10. Security

| Case | Result |
|---|---|
| Correct organisation + entitled portfolio | ✅ Accepted |
| Wrong portfolio | ✅ Rejected **before** the specialist ran (`ran == []`) |
| Cross-organisation | ✅ Rejected |
| Cross-organisation task read | ✅ Rejected — "belongs to another organisation" |
| Unauthenticated | ✅ Rejected, `-32002` |
| **Governed enforcement, pre-check disabled** | ✅ **Trakt refused anyway** |

The last row is the load-bearing one. With the A2A authoriser replaced by
`lambda caller, resource: None` — as if the pre-check were removed,
misconfigured, or bypassed — Trakt's governed execution still refused an
unentitled portfolio, another tenant's portfolio, and a capability the caller did
not hold. The delegation still reached a terminal state.

**The A2A pre-check is cost and latency protection, not the security boundary.**
That is demonstrated by deleting it, not asserted in a docstring.
`execute_governed_tool` resolves entitlements on every call and would still
refuse if the entire A2A layer were removed.

---

## 11. Audit

One delegation reconstructs end to end from stored records, without the
specialist's process still being alive and without a tracing system:

```
calling_agent      a2a_test_agent
organisation       ere
actor / type       sp-agent / service
channel            enterprise_agent
resource           ERE/source_portfolio/direct_001
task               task_7b818be0a89d4c9a
context            ctx_93ecbccbf9724843
lifecycle          submitted → working → completed
governed calls     30   (1 portfolio_capabilities · 2 portfolio_summary
                         · 3 readiness_framework · …)
specialist elapsed 140.36 s
artifact           securitisation-readiness-assessment
```

**The task id is the correlation id** carried into the specialist's
`ExecutionContext` and therefore into every governed execution beneath it. One
identifier chosen at the boundary where the work was requested — without it,
reconstruction means correlating on timestamps, which fails the moment two
callers delegate at once.

The record holds **actions and governed results only**. A test asserts no
`thinking`, `reasoning`, `chain_of_thought` or `rationale` appears anywhere in
it: a stored chain of thought reads as evidence and is not.

---

## 12. Evidence preservation

One material finding, traced from Trakt to the enterprise caller (run 1):

**Trakt fact** — `concentration` on `geographic_region_collateral` returns London
at 31% of balance; `cohort_comparison` isolates Scotland.

**Specialist finding** —

- **FACT:** *"London represents 31% of balance (£31m, 124 loans), measured by
  Trakt concentration test. Scotland, the second-smallest region at 12% of
  balance, accounts for ALL loans in arrears…"*
- **RULE:** *"SYNTHETIC securitisation criteria cap single-region concentration
  at 27% (rule `SEC_GEOGRAPHIC_LIMIT`). Warehouse facility permits up to 35%
  (rule `WH_GEOGRAPHIC_LIMIT`). Trakt's internal screening threshold is 25%."*
- **JUDGEMENT:** *"…breaches the securitisation geographic limit by 4 percentage
  points but passes the warehouse limit with 4 points of headroom… The material
  concern is that arrears are wholly concentrated in…"*
- **SEVERITY:** medium · **EVIDENCE:** `evaluate_rule_packs`, `concentration`,
  `stratify`, `cohort_comparison`

**A2A artifact** — the same four fields, separately, plus severity and evidence
tools, under `materialFindings[]`.

**Enterprise caller** — consumed as structure. It read `overallReadiness` to
route, and passed the finding through without interpreting the finance.

Three authorities over one fact — warehouse PASS, securitisation BREACH, Trakt
screening FLAG — survive as three distinguishable statements. That distinction is
the product, and a wrapper that flattened it into prose would have destroyed it
while still looking like a working integration.

---

## 13. Performance

| run | total | LLM | Trakt governed | **A2A overhead** | discovery | calls |
|---|---|---|---|---|---|---|
| 1 | 141.2 s | 137.0 s | 3,405 ms | 825.3 ms | 0.25 ms | 30 |
| 2 | 184.3 s | 182.1 s | 2,203 ms | **51.7 ms** | 0.11 ms | 32 |
| 3 | 154.7 s | 152.6 s | 2,107 ms | **56.9 ms** | 0.13 ms | 30 |
| 4 | 207.0 s | 204.5 s | 2,454 ms | **82.8 ms** | 0.14 ms | 33 |
| 5 | 181.1 s | 178.5 s | 2,475 ms | **70.2 ms** | 0.16 ms | 32 |

**Protocol overhead is 52–83 ms steady state** — run 1's 825 ms is first-call
warmup. Agent Card discovery costs 0.11–0.25 ms. Against 137–204 s of model time,
**A2A is roughly 0.04% of wall clock.** Trakt's deterministic analytics are
2.1–3.4 s, about 1.5%.

**A defect in this measurement was found and fixed before these numbers were
believed.** The first version computed `total − governed` and called the
remainder "protocol overhead", which attributed every second of model thinking to
A2A — reporting 208,712 ms of overhead on run 1. Publishing that would have
condemned the architecture on a measurement error. The split now uses three
terms: protocol = total − specialist elapsed; LLM = specialist elapsed − governed.

The dry run concealed the defect precisely because a stub specialist has no model
time to misattribute. Only a real run could expose it — which is the argument for
inspecting run one before spending on five.

---

## 14. Regression

Full suite from the merged tree under the pinned authoritative environment:

| | Result |
|---|---|
| **Passed** | **8,075** |
| **Failed** | **0** |
| **Errors** | **0** |
| Skipped / xfailed | 34 / 20 |
| Runtime | 33:05 |
| HEAD | `9a8af9a` |

**Green means green.** Reproducibility recipe: `python -m pytest` from the repo
root, Python 3.11.15, no virtualenv, **pandas 2.3.3** (the repo's own
`requirements.txt` pins `<3.0.0`), pyarrow 24.0.0, numpy 2.4.6, pydantic 2.13.4,
pytest 9.1.1, no plugins, no Trakt environment variables.

Two defects were fixed to reach this, and one non-defect was correctly refused:

- **Integration defect** (`669b05a`): `analytics_lib/contractual.py` had acquired
  a runtime dependency on the Annex 2 regulatory delivery configuration,
  breaking the MI/regulatory separation invariant. Its `OSError` fallback
  silently produced *fewer contractual schedules* when that configuration was
  absent — the coupling was worse than the duplication it avoided.
- **Test-hygiene defect** (`e7dad64`): perf fixtures set `MI_AGENT_AUTH_ENABLED`
  without restoring it, disabling authentication for every subsequent test in the
  process and making a **security** assertion depend on collection order, failing
  in the permissive direction.
- **Not a defect:** 40 failures that proved to be pandas 3.0.5 — outside the
  repo's declared range. They were not "fixed"; the environment was corrected.

---

## 15. Production gap

What is genuinely required before a real client enterprise or Copilot agent could
call this. No speculative features.

**Deployment**
- Network transport: terminate TLS, serve `/.well-known/agent-card.json` and the
  JSON-RPC endpoint over HTTPS. The in-process boundary is complete below that
  line; this is hosting, not redesign.
- Durable task store. `TaskStore` is in-process and does not survive a restart.

**Identity federation**
- Map an external agent's validated Entra principal to a Trakt
  `ExecutionContext`. `context_from_agent_principal` already does this; what is
  missing is the token-validation path in front of the A2A endpoint.
- Client tenant onboarding: organisation registry entry and portfolio
  entitlements for the calling party.

**Agent Card trust**
- Sign the card (A2A v1.0 `signatures`) so a caller can verify it was issued by
  the domain owner. Today nothing prevents a spoofed card advertising a
  look-alike endpoint. Use the specification's mechanism; do not invent
  cryptography.

**Microsoft configuration** (only if a Copilot agent is the caller)
- App registration, `Trakt.Agent` role assignment, admin consent.

**Protocol hardening**
- Per-session concurrency and payload bounds.
- Task retention and deletion policy — a completed task holds a full assessment.

---

## 16. Landing-page demo candidate

From real Sprint 4 behaviour only. About 25 seconds.

| Beat | ~Time | On screen |
|---|---|---|
| **A stranger arrives** | 0–4 s | Enterprise agent: *"I need a specialist securitisation readiness assessment."* It fetches `/.well-known/agent-card.json` and finds `securitisation_readiness_assessment`. It knows nothing else about Trakt. |
| **Delegate the outcome** | 4–8 s | *"Assess this portfolio for securitisation readiness."* Task accepted → `working`. No metrics named, no tools, no plan. |
| **Authorised, not merely authenticated** | 8–11 s | Identity `a2a_test_agent` / `ere`; portfolio entitlement resolved separately. |
| **The specialist investigates** | 11–17 s | 30 governed calls scroll past — opening with `portfolio_capabilities`, unprompted. Nobody told it to. |
| **The non-obvious finding** | 17–21 s | Headline LTV is a comfortable 62%. It drills anyway: **12% of balance above 80% LTV — and the same loans carry the stale valuations.** |
| **One fact, three rulebooks** | 21–24 s | London 31%: **PASS** (warehouse ≤35%) · **BREACH** (securitisation ≤27%) · **FLAG** (Trakt screening >25%). |
| **Evidence returns** | 24–26 s | `MATERIAL_REMEDIATION_REQUIRED`, with fact, rule, judgement and the governed calls behind each. |

Why it works: it opens on a stranger delegating an outcome, contains a genuine
reversal (the reassuring number is the wrong number), shows autonomous
investigation nobody scripted, and lands on the rulebook distinction — the thing
Trakt does that a spreadsheet cannot. Every beat occurred in the recorded runs.

**Not built. Not animated.** A proposal.

---

## 17. Stop / continue decision

**Is further A2A infrastructure development justified now?**

## NO. Freeze the capability.

The evidence supports stopping. The protocol worked 5/5 on every deterministic
criterion, added 52–83 ms to a three-minute analysis, and preserved every
distinction the assessment depends on. There is no A2A defect to fix and no A2A
capability the proof lacked.

Everything remaining in §15 is **deployment and trust configuration** — TLS, a
durable task store, token validation, tenant onboarding, card signing. None of it
is discoverable by building more protocol; all of it is answered by a real client
with a real tenant.

**Do not build**: a second specialist, an agent marketplace, multi-agent routing,
additional protocol bindings, or Copilot-specific integration. None would
strengthen the proof, and each would add surface to maintain before anyone has
asked for it.

**The three genuine defects worth fixing are specialist-side, not A2A-side**, and
each was found by this evaluation:

1. **`could_not_assess.status` should be an enum** of the six governed states.
   `FIELD_GAP` and free-text values reached the caller in these runs.
2. **Metric-id resolution** should suggest `composition_vintage_share` when a
   caller asks for `vintage_share`, rather than refusing flatly and inviting the
   `METHODOLOGY_NOT_APPROVED` mislabel.
3. **`contractual_analytics` metric selection** — the specialist asks for the WAL
   and rarely the yield.

All three change what a peer agent consumes, which is a further argument for
freezing the transport now and fixing the vocabulary first.

---

## Appendix — reproducing

```bash
# Delegated runs (requires ANTHROPIC_API_KEY; resumes onto an existing file)
python scripts/run_a2a_eval.py --runs 5 --out <path>

# Scoring — free, repeatable, never contacts a model
python scripts/score_a2a_eval.py --runs-file <path> --direct-file <sprint3-runs>
```

Keep run files outside the repository: a record contains a full assessment and is
evaluation evidence, not source.
