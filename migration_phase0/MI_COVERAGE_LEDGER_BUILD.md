# Semantic coverage ledger — build result

Start `a0a8f32`. End `6ac01c4` + this report.

**Stages 1 and 2 are built, measured and committed. Stage 3 was NOT started** —
its own entry gate ("do not begin unless Stage 2 proves the disclosure estate
is clean enough to enforce") is not met, and a second, independent blocker
appeared mid-build.

| commit | stage |
|---|---|
| `21f1938` | Stage 1 — the ledger, disclose only |
| `6ac01c4` | Stage 2 — route disclosure + two central equivalences |
| this | report |

**Production files changed:** `question_interpretation/completeness.py` (+128),
`mi_agent_api/mi_service.py` (+40), `mi_agent_api/chat_routing.py` (+41).
**≈209 LOC.** No new module, no ontology, no prompt change, no extra model call.

---

## 1–6. The ledger

**Authoritative owner:** `completeness.coverage_report`. Routes publish what
they executed; this alone decides whether that covers what was asked.

**Representation:** concept-founded. Each stated concept gets one of
`RESOLVED` / `UNSUPPORTED` (the estate declined it and said so) / `UNACCOUNTED`.
There is deliberately **no `NON_SEMANTIC` state**: ordinary words never enter
the ledger, because it is built from what governed owners *name*, not from
tokenising the sentence. That is why the false-positive rate on correct answers
is zero rather than tuned.

Never span-founded — `schema.Slot` records that 170 of 690 measured claims carry
no recoverable span and that a consumer "must never require it".

**Independence (1A audit, all four conditions):** `stated_concepts` takes no
spec, no contract and no envelope; its internal `detect_requested_facets` call
deliberately omits `resolved_filters`; it owns no phrase bank; its surface grows
from the registry and value catalogue. **Proven, not asserted:** for Q16B the
facet ledger reports nothing for `drawdown` while `stated_concepts` reports
`('value','erm_product_type','drawdown')`.

## 7–8. Stage 1

Deterministic 166 **byte-identical**; frozen 278-module manifest **exact at 85**.
Initial census: **46 of 980 answering questions** carried an unaccounted concept.

**Q16B, concept omitted → `unaccounted: erm_product_type`, 20/20 runs.** The
gate can see the omission. That was Stage 1's whole purpose.

## 9–12. Stage 2

| class | n | action |
|---|---|---|
| **D2** applied but undisclosed | 7 closed | `portfolio_summary`, `evolution` now declare scope |
| **D3** governed equivalence | 2 rules | `scope_applied` satisfies `scoped`; the band relation read inversely |
| **D1** genuine omission | left visible | this is what Stage 3 must refuse |
| **D4** detector false positive | **none found** | no concept required suppression |

Both D3 rules are central and registry-read, not per-route. Closed: the four
`Summarise the … book` cases, both `tickets larger than £150k` cases, and
`Show the acquired book balance by month`.

**Stage 2 movement: 166/166 byte-identical, frozen exact at 85.** No value, row,
route or figure changed.

## 13. Remaining 39 — and why Stage 3 did not start

| route | kind | n | genuine omission? |
|---|---|---|---|
| (point-in-time) | value | 9 | **yes** — drawdown, lump_sum, source_portfolio_type |
| (point-in-time) | measure | 6 | mostly yes |
| analytical_composition | value | 6 | mixed — pipeline_stage populations are declared in findings, not filters |
| **analytical_composition** | **dataset** | **8** | **no — disclosure** |
| **forecast_extrapolation** | **dataset** | **2** | **no — disclosure** |
| **funded_bridge** | **dataset** | **2** | **no — disclosure** |
| **funded_bridge** | **scope** | **2** | **no — disclosure** |
| **cohort_progression** | **scope** | **2** | **no — disclosure** |
| portfolio_summary | value | 1 | yes — "Summarise the Acquired" answers over 640 |

**At least 16 of 39 are still disclosure gaps.** Enforcing now would refuse that
many correct answers. Stage 3's entry gate is explicit, and it is not met.

The `dataset` group is not a disclosure fix I was willing to make blind:
`forecast` is a governed view (`VIEWS = funded, pipeline, forecast`) with **no
readable tape** — a forecast is computed from funded and pipeline data — so
`reconciliation.dataset` can never equal `forecast` under current semantics.
Closing those 12 requires a decision about what "reconciled against" means for
a derived view. That is a semantics change, and the brief's stop condition
covers it: *"if fixing disclosures requires calculation/routing changes, stop"*.

`cohort_progression` (2) has no `interpretation` parameter, so the shared
primitive cannot reach it without a signature change; and both cases are Q19A,
a known-wrong answer where a coverage refusal would be the desired outcome
anyway.

## 14–18. Stage 3 — NOT BUILT

The rule was specified and is unchanged from `a0a8f32`:

> Answering is prohibited where `stated_concepts` names a governed concept and
> the executed contract records no disposition for it. `UNACCOUNTED` → refusal;
> `RESOLVED` → answer; `UNSUPPORTED` → existing governed behaviour.

It was not implemented, for two independent reasons.

**Blocker 1 — the entry gate.** 16+ of 39 remaining flags are disclosure, not
omission (§13).

**Blocker 2 — the API credit balance was exhausted mid-build.**
`models.list` succeeds; every `messages.create` returns
`400 … Your credit balance is too low`. So:

* the 20-run Q16B measurement above is valid evidence **about the ledger** —
  every run was `proposal_unavailable`, which *is* the omission state, and the
  ledger reported `unaccounted` in all 20 — but it is **no evidence about the
  model's proposal rate**;
* the entire stochastic acceptance protocol (Q16B ≥20 with **both** states
  observed, Q17C ≥10, Unknown/Missing age ≥10, seven recoveries ≥6, five
  regressions ≥6, must-refuse ≥6) **cannot be run**. Every one of those
  criteria is UNMEASURED, not passed.

Flipping to fail-closed without that protocol would be exactly what this
programme has refused to do at every prior stage.

## 19–25. Acceptance status

| gate | status |
|---|---|
| deterministic 166 byte-identical | **PASS** (both stages) |
| frozen 278-module manifest = 85 | **PASS** (both stages) |
| ledger present on every answer | **PASS** — 1,612/1,612 |
| Q16B omission visible as UNACCOUNTED | **PASS** — 20/20 |
| answer/value movement in Stage 2 | **PASS** — zero |
| Opus-arm 166 byte-identical | **NOT MEASURED** — no credits |
| stochastic controls (Q16B/Q17C/recoveries/regressions) | **NOT MEASURED** — no credits |
| full 1,446 Opus sweep | **NOT MEASURED** — no credits |
| new controlled refusals from genuine omission | **0** — Stage 3 not built |
| false coverage refusals | **0** — nothing is enforced yet |

## 26–28. Known wrong, limits, residual risk

Outside coverage's stated scope and unchanged: **Q04C** (right population,
wrong grain — not an omission) and **Q19A** (the intent owner cannot separate a
two-period delta from a window progression).

**Explicit semantic-estate limit, unchanged:** Q10B's bare "size" has no owner
that can name it, so coverage cannot guarantee it. The guarantee is bounded to
*material user semantics the governed estate can independently name*, and no
broader claim is made.

**Residual risks:** `_carried` is now load-bearing and needs adversarial review
before it gates anything; the `dataset`-for-derived-views question is unresolved
and blocks 12 of the remaining flags; and the ledger currently proves a concept
*reached* the contract, not that it was applied to the right rows.

---

# DO NOT FREEZE

The infrastructure that closes the omission hole is built, measured and safe:
the ledger sees Q16B's omission every time, costs nothing, and moved not one
answer across 1,612 questions in either stage. Stage 2 closed 7 false flags
through one shared primitive and two registry-read equivalences rather than
per-route patches.

But the invariant is **not yet enforcing**, so the hole it was built to close is
still open in production terms — an omission still widens the answer today. It
cannot responsibly be switched on until the remaining ~16 disclosure gaps close,
and it cannot be *accepted* at all until API credits allow the stochastic
protocol to run, since a single deterministic pass cannot validate a control
whose entire purpose is guarding a stochastic failure.

Both blockers are bounded and named. Neither is a design failure. The next
sprint should close the `dataset`-for-derived-views question and the four
remaining route disclosures, then run Stage 3 behind the full repeated-run
protocol.
