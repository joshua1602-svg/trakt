# Review C/D — Risk register and decision recommendation

## C. Risk register

| # | Severity | Finding | Evidence | Owner action |
|---|---|---|---|---|
| R1 | **High** | **Expectation gap on "MI + ESMA Annex 2":** no current path — OCC *or* blob-trigger production — produces Annex 2 XML or XSD validation. Both end at the projected CSV. The only complete XML+XSD chain is the legacy gate chain, demo-only and documented as "retire the runtime". An operator choosing the Annex 2 outcome could reasonably believe a submittable report results. | Report A §1–2; `orchestrator.py:293-309`; `router.py:749-759`; `docs/annex2_path_map_promotion_policy.md:7` | Decide the Phase 3 XML strategy (E4); until then adjust OCC copy (E2) |
| R2 | **High** | **Regulatory onboarding is gated shut for every client:** 38 of 107 universe codes (31 Mandatory-priority) lack `field_rules`, so every `regulatory_mi` onboarding parks `NEEDS_CONFIGURATION` — in OCC *and* in production blob routing alike. Proven XML capability exists for 36 of the 38 regardless. | Report A §4; reconciliation CSV | Complete or defer rules (E1) |
| R3 | Medium | Design-doc/code divergence: doc 05 claims API responses ride in `trakt_core.GovernedResult`; implementation returns ad-hoc `{ok,…}` envelopes and never imports it. | Report B §3 | E3 (adopt envelope or amend doc) |
| R4 | Medium | Third authentication mechanism in the repo (ops token+tenancy vs `mi_agent_api` Easy Auth vs operator-console token). | Report B §3 | Phase 3 convergence on the `mi_agent_api` seam |
| R5 | Medium | Dual persistence of the same fact: rule store (record) + projected client memory (agent-readable). Safe only while projection stays one-directional and idempotent. | Report B §3 | Document invariant; no independent memory writes from OCC |
| R6 | Medium | Complexity hotspots: `translate_run_state` F(48), `extract_mapping_decisions` F(47), `resolve_decision` D(25); `engine.py` maintainability C. Coverage gaps: `api/app.py` 61%, `adapters.py` 68%. | Report B §2 | E3 consolidation batch |
| R7 | Medium | Dual state (workflow doc + orchestrator `run_state.json`) — currently correctly derived in one place; a future writer bypassing `translate_run_state` would desynchronise them. | Report B §3 | Document invariant (E3) |
| R8 | Low | 851-line `MockOpsClient` statically imported → bundled in production builds (inert: build-time gate). | Report B §3 | E3 (dynamic import / slim fixture) |
| R9 | Low | `retire()` implemented + tested but no API route; `result_history` written, never read; one unused import. | Report B §2 | E3 |
| R10 | Low | Pre-existing pipeline debris (not OCC): duplicate XSD copies; `annex2_xml_structure_contract.yaml` loaded by nothing; enum truth duplicated between `enum_mapping.yaml` and delivery rules; stale `trakt_run.py:1163` stage label; production runs the regime projector twice per delivery. | Report A §2 | Housekeeping backlog (E5) |
| R11 | Low | Phase 2 report imprecision: "31 codes lack rules" (actual: 38 lack rules, 31 of them Mandatory-priority) and an unqualified "genuine pre-existing gap" without noting the proven path populates 36/38. "≈106/107" itself has no repository evidence — real prior results are 105 and 104 fields. | Report A §3–4 | Corrected by this review |
| R12 | Informational | The "13,478 lines" headline is 36% npm lockfile and 8% design docs; hand-written production logic ≈6,300 lines. | Report B §1 | — |
| R13 | Informational | `demo_platform.run_demo` stage 6 fails on `ask_trakt_mi()` signature drift (pre-existing, unrelated to OCC; regulatory stage completes first). | Report A §5 | Separate fix ticket |

## D. Decision recommendation

**Approve Phase 2 for merge, subject to targeted consolidation (scenario B) —
no hold is warranted on the Annex 2 path.**

Rationale against each alternative:
- *Hold pending Annex 2 path correction* — not applicable: instrumented
  execution proves the OCC drives the **same current production conductor,
  agents, configs and projector** as the blob-trigger path; there is no
  legacy-path wiring to correct. The two genuine Annex 2 issues (R1, R2) are
  properties of the existing pipeline that the OCC exposed; both require
  approval-gated changes to existing files, which the OCC correctly did not
  make.
- *Hold pending material simplification* — the size concern dissolves on
  measurement (R12): ≈6,300 hand-written production lines for twelve agreed
  capabilities, zero new backend dependencies, no import cycles, 0.12%
  duplication, 80% coverage. Scenario B (−~1,000 lines, −4 files, zero
  behaviour change) is worth doing but is not merge-blocking.
- *Reject and redesign* — no finding supports it.

Sequencing: merge Phase 2; land E2 (copy honesty) and E3 (consolidation)
immediately after; take E1 (delivery rules) and E4 (XML strategy) as the first
Phase 3 decisions since R1+R2 gate any real regulatory client.

### Appendix — governing question, answered

The Operations Control Centre **has genuinely wrapped the current production
Annex 2 workflow** (proven by runtime instrumentation and production-parity
adapter construction) and added an operational layer whose true size
(≈6.3k lines) is proportionate to the mandated capability list, with
~1,000 lines of identified trim. It did **not** wrap a legacy path. The
surprising Annex 2 findings are real properties of the current pipeline:
XML+XSD only exists on a retired demo-only chain, and the new agentic chain's
own configuration gate blocks 38 codes that the proven chain populates anyway.
