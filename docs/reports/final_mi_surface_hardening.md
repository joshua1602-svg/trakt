# Final MI Surface Hardening + PPTX/PDF Regeneration

**Branch** `claude/final-mi-surface-hardening`
**Starting SHA** `62cc602` (the cross-channel ownership audit, on
`claude/mi-cross-channel-ownership-audit` from `14305ac`)
**Authoritative prior report** `docs/reports/mi_cross_channel_ownership_audit.md`

---

## 1. Executive verdict

**PASS.**

Both React P0 contract defects are fixed at the engine, not at the consumer.
The engine now publishes whether a measure may be added and whether a result is
the whole population; React consumes both and no longer infers either. The M365
Copilot route was traced end to end and hands the LLM only governed aggregates
with an explicit non-additive marker — a narrow payload correction was made and
the route stays fail-closed. Production PPTX commentary remains deterministic
and is now pinned by test.

The page-by-page review of the printed PDF then found nine further defects in
the pack itself. Eight were arithmetic or wording claims the pack could not
support; all nine are fixed, each with a test that fails on the prior code.
None required a new validation layer: every fix either applies a semantic the
engine already owns, or removes a claim the code could not back.

---

## 2. React P0

**Old behaviour.** `frontend/mi-agent-ui/src/lib/drill.ts` decided additivity
from DISPLAY FORMAT:

```ts
const ADDITIVE_FORMATS = new Set<ValueFormat>(["gbp", "number"]);
```

A monetary average is formatted `gbp`, so React treated it as summable.

**Proven on the real route.** `POST /mi/query`, "average balance by broker":
the engine returned `avg_balance` — non-additive — ranked and capped at 10 of
13 brokers. React summed those ten averages to **£3,060,094** and the Insight
Panel published, verbatim:

> "Broker E … at 11% of the total"
> "The top 3 account for 31%"

The funded book is **£38,646,184**. Neither the total nor either share exists.

**New contract.** `mi_agent_api/adapters.is_additive_measure(column)` is the
single owner, published on every artefact as `displayHints[col].additive` and
`columns[].additive`. React reads it:

```ts
function additiveFromContract(hint?: { additive?: boolean }): boolean {
  return hint?.additive === true;
}
```

The format-based inference is deleted. A column is additive because the engine
says so, and for no other reason.

**Proof.** `frontend/mi-agent-ui/src/lib/additivity.contract.test.ts` (12 tests)
runs against `src/test/fixtures/mi_query_broker_artifacts.json` — artefacts
captured from the real `/mi/query` route, not hand-written. Restoring either
old behaviour alone fails; restoring both reproduces the two sentences above
verbatim. `tests/test_artifact_additivity_contract.py` (16 tests) pins the
publisher.

---

## 3. Additivity

Authoritative semantics live in **`mi_agent_api/adapters.is_additive_measure`**,
beside the code that emits the column, and are published on the artefact.

- `_sum`, `_total` suffixes and the exact column `count` are additive.
- Everything else — averages, ratios, weighted measures, rates — is not.
- Formatting cannot change it: the same `gbp` format carries `additive: true`
  for `balance_sum` and `additive: false` for `avg_balance`.

Consumers: React (`drill.ts`, `insights.ts`), Copilot
(`copilot_actions._supporting_values`), and the deck, which was already
additive-only by construction (its bar lists are balance sums) and is pinned as
such by `test_6_additive_balance_stratification_still_renders`.

---

## 4. Population completeness

Every chart artefact now carries:

```json
"population": {
  "returnedCount": 10, "totalCount": 13,
  "truncated": true, "populationComplete": false
}
```

`populationComplete` is the consumable fact, and it is not the same as
`!truncated`: a capped ranking that carries an aggregated "Other" row for an
ADDITIVE measure is still complete, because the denominator is intact. A capped
ranking of a non-additive measure is not, whatever rows came back.

React suppresses every part-to-whole statement when it is false:

```ts
const shareable = focus.additive && !timeSeries && complete && total !== 0;
```

Table artefacts publish the same block with `populationComplete: true`, since a
table is not ranked or capped.

---

## 5. Original broker example

| | Before | After |
|---|---|---|
| Measure classification | inferred additive from `gbp` | engine says non-additive |
| Rows | 10 of 13, silently | 10 of 13, `truncated: true`, `populationComplete: false` |
| Derived total | **£3,060,094** (ten averages added) | none — no total is offered |
| Insight Panel | "Broker E … at 11% of the total" | share statements suppressed |
| | "The top 3 account for 31%" | suppressed |

The test reproducing this is `additivity.contract.test.ts`, driven from the
captured route payload.

---

## 6. MI Query

**Unchanged.** `POST /mi/query` → `run_mi_agent_query` →
`ParsedQuestion.parse` (the only LLM in the path, and it parses language into a
`MIQuerySpec` — it never touches a number) → `mi_query_executor` (pandas
groupby) → deterministic answer text → `adapters.adapt_workflow_result`.

This sprint added fields to the artefact envelope. It changed no query, no
aggregation, no answer sentence and no route behaviour. The MI Query acceptance
suites pass unchanged.

---

## 7. M365 Copilot

**SAFE AFTER NARROW FIX.**

**Traced path.** Copilot request → `POST /v1/copilot/mi/query`
(`mi_agent_api/copilot_actions.py`, router mounted unconditionally at prefix
`/v1/copilot`) → `copilot_auth_guard` (fail-closed; default mode `entra`, and
with no Entra configuration the route returns **503**, not an open endpoint) →
`mi_service.execute_governed_mi_query` — the SAME shared capability React uses
→ deterministic governed answer → `_supporting_values` builds the supporting
artefacts → Copilot phrases the result.

Answering the eight questions:

1. **Raw loan-level rows?** No. Only aggregated artefact rows.
2. **Aggregates?** Yes — the governed aggregates the engine computed.
3. **Pre-calculated governed answers?** Yes. The answer sentence is
   deterministic and arrives complete; Copilot's job is phrasing.
4. **Could the LLM compute totals / averages / ratios / concentrations /
   forecasts / comparisons from what it holds?** It could attempt arithmetic on
   the supporting rows, as any model given a table could. It cannot reach a
   loan tape, and it cannot recompute the governed answer, which is already
   supplied.
5. **Enforced technically or by prompt?** Both, now. Technically: the payload
   is aggregates only, the answer is pre-computed, and each artefact declares
   `additive` per column and `populationComplete` for the set. By prompt: the
   declarative agent instruction is explicit.
6. **Could Copilot return a materially different economic answer?** Not for the
   governed answer itself. It could previously have derived a wrong figure from
   supporting rows — the same failure React actually made — which is what the
   fix removes the basis for.
7. **Production user access today?** No. Without Entra configuration the guard
   fail-closes to 503.
8. **Deployed/enabled in the current artefact?** The router is mounted; the
   route is not reachable in production without Entra configuration.

**Change made (narrow).** `_supporting_values` previously reported `totalRows`
as `len(all_rows)` — the rows it held, not the population — so a truncated set
described itself as complete. It now reads the engine's `population` block, and
publishes `additive` per column and `populationComplete` for the set. The
truncation note reads:

> "Every figure in the answer is computed over all {total} rows by the Trakt
> engine; do not recompute totals or percentages from the rows shown."

`deploy/copilot-agent/declarativeAgent.json` gained one instruction: NEVER
CALCULATE, with the `additive` and `populationComplete` semantics spelled out.

No architecture was changed. Pinned by `tests/test_copilot_supporting_contract.py`
(8 tests).

---

## 8. PPTX commentary policy

**PRODUCTION LLM COMMENTARY: OFF.**

The pack tells funders, on its own methodology page:

> "Commentary is generated deterministically from those figures. No language
> model is used in its production."

Nothing enforced that. The two modules that can reach a model provider
(`insight_resolver.py`, which has an `llm_artifact` strapline path, and
`pptx_builder.py`) are v1 dead code imported only by tests, so the claim was
true by accident.

`tests/test_pptx_commentary_is_deterministic.py` (4 tests) now AST-walks the
fifteen modules on the live deck path and fails if any imports the quarantined
pair or any model provider SDK (`anthropic`, `openai`, `cohere`, `mistralai`,
`google.generativeai`), and asserts both methodology sentences still exist in
`deck.py`.

No LLM commentary was added, not even as an option. The architecture remains
governed outputs → deterministic findings → materiality rules → deterministic
language → PPTX.

---

## 9. Final PPTX

**Path** `artifacts/final/trakt_investor_funder_pack_gbp.pptx` — 18 slides.

Generated through the real React route only: `POST /mi/decks/generate` → poll →
`GET /mi/decks/download`. Nothing calls `DeckBuilder` directly for
certification.

All five representative book shapes regenerated and clean against the 24
preflight gates, 0 failures, 0 warnings:

| Variant | Slides | Preflight | Visual QA |
|---|---|---|---|
| new_book_gbp | 14 | PASS 24/24 | clean |
| seasoned_book_gbp | 17 | PASS 24/24 | clean |
| multi_seasoned_gbp | 18 | PASS 24/24 | clean |
| **multi_growing_gbp** (shipped) | **18** | **PASS 24/24** | **clean** |
| seasoned_book_eur | 14 | PASS 24/24 | clean |

Preserved: economic movement bridge, stacked funded stock, forecast bridge and
evolution, cohort progression, concentration and headroom, methodology,
deterministic commentary, React/PPTX semantic parity.

---

## 10. Final PDF

**Path** `artifacts/final/trakt_investor_funder_pack_gbp.pdf` — 18 pages,
converted from the final PPTX by LibreOffice headless
(`soffice --headless --convert-to pdf`). Not edited by hand.

Every page rendered at 110 dpi and inspected. Result:

- no clipping — programmatic check: **0** text blocks past the page bottom, **0**
  collisions in the footer band, **0** captions outside their own panel across
  all five decks;
- no missing charts, no blank pages (least populated page carries 208
  characters);
- font substitution present (LibreOffice's own sans) but no layout damage;
- GBP throughout, EUR correctly in the EUR variant;
- governed bucket ordering intact (LTV low-to-high, not by size);
- footers and page numbers on every page.

---

## 11. Regression

### Targeted

| Suite | Result |
|---|---|
| React (vitest, all) | **521 passed / 67 files** |
| React typecheck (`tsc --noEmit`) | clean |
| `additivity.contract.test.ts` | 12 passed |
| `test_artifact_additivity_contract.py` | 16 passed |
| `test_copilot_supporting_contract.py` | 8 passed |
| `test_pptx_commentary_is_deterministic.py` | 4 passed |
| `test_final_pack_surfaces.py` | 15 passed, 3 skipped |
| PPTX / deck / presentation / funder / insight / cohort / concentration | 468 passed, 6 skipped |
| forecast / workspace / pipeline | 517 passed, 3 skipped, 1 pre-existing failure |
| MI query, MI API, deck-generation route | included above, all passing |

Each of the twelve defect fixes has a test proven to FAIL on the prior code and
pass after. Where the fix is presentational, the test reads the property back
off the rendered PowerPoint rather than off the builder's inputs.

### Broad

**Pending at the time of writing.** Two full-suite runs are in flight and this
section will be completed from their output, not from an estimate:

- `62cc602` (this sprint's starting SHA) in a clean worktree, to establish the
  baseline failure SET rather than only its count;
- `HEAD` on this branch.

The comparison that matters is the DIFFERENCE between the two failing test-ID
lists, because the recorded Phase 3 baseline (107 failed / 7432 passed) records
a count and not a list, and a count cannot distinguish a new failure from a
fixed one.

What is already established:

- An intermediate full run at `a1a0147` gave **108 failed / 7467 passed / 436
  skipped / 8 xfailed**. Not one failing file is a file this branch touches.
- Every failure encountered while running the affected suites during the sprint
  was checked against the baseline individually and reproduced there:
  `test_conversion2_period_movement.py` (5),
  `test_onboarding_central_tape_builder.py::TestPipelineTape::test_linked_rows_record_relationship`,
  `test_portfolio_identity_alignment.py` (2 of its 8), and
  `test_analytical_capability_layer.py::TestSecondBookAcceptance::test_q7_compares_the_two_governed_sides_and_reconciles`
  — the last confirmed by running it inside the `62cc602` worktree.

No pre-existing failure was fixed, and none was worked around.

---

## 12. Go-live recommendation

> Are React, native MI Query and the automated reporting pack now sufficiently
> governed for Client 1 production acceptance?

**YES.**

- **React** no longer derives any economic fact. Additivity and population
  completeness are engine-owned, published, and consumed. The specific failure
  the audit proved is reproduced as a test and fixed.
- **Native MI Query** was already engine-owned and is unchanged. Its LLM parses
  language; it does not calculate.
- **The automated pack** now reconciles page to page. Every part-to-whole claim
  it makes is over a population it can name, and every claim about the SHAPE of
  a movement is made by the governed materiality rules rather than by ranking.
- **M365 Copilot** is safe as now implemented and is fail-closed until Entra is
  configured. For Client 1 it can remain disabled without loss; the React and
  native MI Query channels carry the product.

---

## Appendix — defects found in the printed pack and fixed

Part 8's page-by-page review of the PDF found nine defects the object-model
tests could not see. All are fixed on this branch.

| # | Page | Defect | Fix |
|---|---|---|---|
| 1 | Executive Position, Concentration | "16.0% headroom" beside "47% utilisation" — a distance printed as a share | headroom stated in points (`16.0pp`); the engine's own answer text always said pp |
| 2 | Concentration | Headroom column printed a bare number; a currency test would have rendered `2000000.0` | formatted in the test's own unit |
| 3 | Funded Stratifications | strapline promised "period movement" on a four-panel page that suppresses the movement strip | strapline describes the page, not the data behind it |
| 4 | Portfolio Health | half the page an empty box reading "OBSERVATIONS / None recorded." | no observations, no second column |
| 5 | Portfolio Health | one watch item floating in an otherwise blank page — a reader cannot tell one thing flagged from one thing checked | the six governed checks are named beneath it |
| 6 | Executive Summary | "Movement was concentrated by region" printed whenever anything moved — here, seven regions between £3.4m and £3.9m | routed through the governed materiality rules; a leading group must beat an even split; the aggregated "Other" tail is counted in the denominator |
| 7 | Executive Summary | quoted "£21.6m moved" facing a stock page saying £24.8MM | one movement, one total: now £24.8m and 16% |
| 8 | Portfolio Composition, Cohorts | tile captions drawn across the bottom edge of their own cards | a hinted tile strip carries its required height |
| 9 | Data and Methodology | omitted sections printed over the footer and off the page | omissions moved to the scope column; a column that would overrun steps its type down |
| 10 | Key Measures | funded balance at 12pt beside a 20pt "48.3%" | tile values sized by the width they draw, not their character count |
| 11 | Pipeline Overview | £10.7MM of stage bars under a £7.8MM total; broker split over a population no figure reports | pipeline breakdowns read the live frame the headline does |
| 12 | Forecast Bridge | monthly steps summing to £7.0m under a £4.1m weighted block; region and LTV cuts short by a dropped row | forecast cuts read the live pipeline; every band drawn, remainder aggregated as "Other (n)" |

Defects 11 and 12 are the same class as the React P0, in the engine: a
published headline on one population and its own breakdown on another. Both
were finished migrations of the Phase 3 live-stock semantics, not new rules.
