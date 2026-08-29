# Hardening and capability recovery

Start `adbeb5fbd497d632ae83cb0bac7711c927df78a6`, tree clean. Three production
files changed, all in phases 1 and 2. **Phase 4 implemented no further
recoveries, and `funded_bridge` was not narrowed** — the evidence says its
replacement path is not reachable for the shadowed wordings.

---

## Phase 1 — `portfolio_summary` narrowed

`question → resolve_dataset → frame loaded → route → answer reads → reconciliation → answer`

```
BEFORE
  Summarise the current pipeline.
  resolve_dataset  pipeline
  frames loaded    [('pipeline', 8), ('funded', 640)]
  route            portfolio_summary
  answer reads     ['funded']
  reconciliation   funded
  answer           "the portfolio holds 640 loans with a funded balance of £172.1m"

AFTER
  route            (point-in-time)
  reconciliation   pipeline
  answer           "8 loans · Current Outstanding Balance: £3.6MM."
```

£3.6MM over 8 rows is the pipeline tape's own total, and the path is the one the
sibling *"Give me an overview of the pipeline by size and stage"* already proved.
The gate asks `workspace.resolve_dataset` — the single governed dataset owner —
fails open, adds no `pipeline_root`, and alters no calculation. Population is a
different axis: *"Summarise the acquired book"* still routes here and still
answers 199 loans / £54.7m.

**Movement: Q10A only, both arms.** WRONG → NO_COMPUTABLE_TRUTH, the label
following from the grader's fidelity gate (the frozen WRONG verdict was recorded
against the funded answer and is correctly not carried onto a different one).

---

## Phase 2 — the masked intent/output-shape contradiction

**Root cause.** `_bridge_recognizer` built `intent="chart"` beside
`chart_type="none"` — a spec claiming an output shape it does not name, because
the waterfall is built by the `funded_bridge` route rather than the chart
factory. Surveyed across 1,446 questions: **23 specs carried the invalid pair,
all 23 from that one builder**, all `bridge_query=True`.

**Invariant.** *A parser must not emit an internally invalid combination of
intent and output shape.* Enforced at the owner: the bridge spec now declares
`intent="table"` (the pair the validator already permits), and
`_deterministic_parse` is a thin wrapper that **fails closed** rather than
emitting a contradictory spec.

**Deliberately only the self-contradiction.** The validator's other chart rules
ask whether a chart can be RENDERED from the fields — 22 corpus questions carry
such a spec and are answered correctly by a route today. Failing those closed
would convert working answers into refusals.

**Movement: none.** Zero answers changed on either arm. Proven without touching
`funded_bridge` eligibility: bridge questions still answer, and simulating the
route declining shows the internal message replaced by a governed refusal.

---

## Phase 3 — capability-recovery inventory

### The known family, traced first

`weighted_expected_funded_amount` on the MI pipeline frame:

```
stage         gross        p      weighted
COMPLETED    100,000    1.00        NaN
OFFER        200,000    0.75    150,000
OFFER        300,000    0.75    225,000
WITHDRAWN    400,000     NaN        NaN
APPLICATION  500,000    0.45    225,000
KFI          600,000    0.20    120,000
COMPLETED    700,000    1.00        NaN
OFFER        800,000    0.75    600,000
                              ---------
all 8 rows                    1,320,000
open only (5)                 1,320,000
COMPLETED/WITHDRAWN (3)   gross 1,200,000   weighted 0
```

**Double-count prevention is in the column itself**: `weighted_expected_funded_amount`
is NaN for COMPLETED and WITHDRAWN, so the open-pipeline sum and the all-rows sum
are the same figure. Funded £172,055,547 + £1,320,000 = **£173.38m**, which is the
`funded_balance_forecast` executor's £173.4m.

**The two questions are not the same analytic.** *"What is the weighted expected
pipeline contribution?"* asks for one figure (£1.32m). *"Show funded vs pipeline
contribution."* asks for two components side by side. Both components are
produced by `funded_balance_forecast` (executors.py:959), with the exclusion
disclosed in its own prose.

**But neither reaches it.** `plan_for` returns `None` for both, while returning
`['funded_balance_forecast', 'pipeline_completion_forecast']` for three sibling
wordings that classify into the *same* families. The capability is proven; the
wording fails before capability selection. **CR4, not CR2.**

`mi_agent_api/forecast_bridge.py::compute_forecast_bridge` is a public
deterministic callable whose only live callers are in `mi_agent_pptx` — a genuine
CR2-shaped find — but wiring it would not recover these two questions, because
they never reach a planner at all.

### The table

| id | outcome (off/merge) | first divergence | CR | evidence | sibling | route | recovery | safe now? |
|---|---|---|---|---|---|---|---|---|
| Q01B | FALSE_REFUSAL / FALSE_REFUSAL | wording fails before capability selection | **CR4** | sibling Q01A answers the same case correctly | Q01A | (point-in-time) | semantic-compiler layer | no |
| Q01C | FALSE_REFUSAL / CORRECT | wording fails before capability selection | **CR4** | the merge arm answers it; the deterministic wording fails before capab | Q01A | (point-in-time) | semantic-compiler layer | no |
| Q02B | FALSE_REFUSAL / CORRECT | wording fails before capability selection | **CR4** | the merge arm answers it; the deterministic wording fails before capab | Q02A | (point-in-time) | semantic-compiler layer | no |
| Q03A | WRONG / CORRECT | wording fails before capability selection | **CR4** | the merge arm answers it; the deterministic wording fails before capab | Q03B | (point-in-time) | semantic-compiler layer | no |
| Q03C | FALSE_REFUSAL / CORRECT | wording fails before capability selection | **CR4** | the merge arm answers it; the deterministic wording fails before capab | Q03B | (point-in-time) | semantic-compiler layer | no |
| Q04A | FALSE_REFUSAL / FALSE_REFUSAL | wording fails before capability selection | **CR4** | sibling Q04B answers the same case correctly | Q04B | (point-in-time) | semantic-compiler layer | no |
| Q04C | WRONG / WRONG | wording fails before capability selection | **CR4** | sibling Q04B answers the same case correctly | Q04B | (point-in-time) | semantic-compiler layer | no |
| Q05B | FALSE_REFUSAL / FALSE_REFUSAL | wording fails before capability selection | **CR4** | sibling Q05A answers the same case correctly | Q05A | (point-in-time) | semantic-compiler layer | no |
| Q05C | WRONG / CORRECT | wording fails before capability selection | **CR4** | the merge arm answers it; the deterministic wording fails before capab | Q05A | (point-in-time) | semantic-compiler layer | no |
| Q07B | WRONG / FALSE_REFUSAL | wording fails before capability selection | **CR4** | sibling Q07A answers the same case correctly | Q07A | (point-in-time) | semantic-compiler layer | no |
| Q10C | FALSE_REFUSAL / FALSE_REFUSAL | wording fails before capability selection | **CR4** | sibling Q10B answers the same case correctly | Q10B | (point-in-time) | semantic-compiler layer | no |
| Q12C | FALSE_REFUSAL / FALSE_REFUSAL | wording fails before capability selection | **CR4** | sibling Q12A answers the same case correctly | Q12A | (point-in-time) | semantic-compiler layer | no |
| Q15B | FALSE_REFUSAL / FALSE_REFUSAL | wording fails before capability selection | **CR4** | sibling Q15A answers the same case correctly | Q15A | (point-in-time) | semantic-compiler layer | no |
| Q15C | FALSE_REFUSAL / FALSE_REFUSAL | wording fails before capability selection | **CR4** | sibling Q15A answers the same case correctly | Q15A | (point-in-time) | semantic-compiler layer | no |
| Q16B | WRONG / CORRECT | wording fails before capability selection | **CR4** | the merge arm answers it; the deterministic wording fails before capab | Q16A | (point-in-time) | semantic-compiler layer | no |
| Q17B | FALSE_REFUSAL / FALSE_REFUSAL | wording fails before capability selection | **CR4** | sibling Q17A answers the same case correctly | Q17A | (point-in-time) | semantic-compiler layer | no |
| Q17C | WRONG / CORRECT | wording fails before capability selection | **CR4** | the merge arm answers it; the deterministic wording fails before capab | Q17A | (point-in-time) | semantic-compiler layer | no |
| Q19A | WRONG / WRONG | wording fails before capability selection | **CR4** | sibling Q19B answers the same case correctly | Q19B | cohort_progression | semantic-compiler layer | no |
| Q20B | FALSE_REFUSAL / FALSE_REFUSAL | wording fails before capability selection | **CR4** | sibling Q20A answers the same case correctly | Q20A | period_movement | semantic-compiler layer | no |
| Q21B | FALSE_REFUSAL / FALSE_REFUSAL | wording fails before capability selection | **CR4** | sibling Q21A answers the same case correctly | Q21A | period_change_analysis | semantic-compiler layer | no |
| Q21C | FALSE_REFUSAL / FALSE_REFUSAL | wording fails before capability selection | **CR4** | sibling Q21A answers the same case correctly | Q21A | period_change_analysis | semantic-compiler layer | no |
| Q23A | CORRECT / FALSE_REFUSAL | wording fails before capability selection | **CR4** | sibling Q23C answers the same case correctly | Q23C | forecast_extrapolation | semantic-compiler layer | no |
| Q23B | FALSE_REFUSAL / FALSE_REFUSAL | wording fails before capability selection | **CR4** | sibling Q23A answers the same case correctly | Q23A | (point-in-time) | semantic-compiler layer | no |
| Q24B | FALSE_REFUSAL / FALSE_REFUSAL | wording fails before capability selection | **CR4** | sibling Q24A answers the same case correctly | Q24A | (point-in-time) | semantic-compiler layer | no |
| CFO60 | FALSE_REFUSAL / FALSE_REFUSAL | no governed capability | **CR5** | the concentration route measures a GOVERNED dimension set; the sibling | — | concentration_analysis | none — refusal correct | n/a |
| CFO61 | FALSE_REFUSAL / FALSE_REFUSAL | no governed capability | **CR5** | as CFO60: 'broker' is not in the governed concentration dimension set. | — | concentration_analysis | none — refusal correct | n/a |
| CFO71 | FALSE_REFUSAL / FALSE_REFUSAL | no governed capability | **CR5** | not a capability gap — a CORRECT ambiguity refusal: "'value' could mea | — | (point-in-time) | none — refusal correct | n/a |
| Q25A | FALSE_REFUSAL / FALSE_REFUSAL | no governed capability | **CR5** | forward approved-limit methodology does not exist (protected) | — | (point-in-time) | none — refusal correct | n/a |
| Q25B | FALSE_REFUSAL / FALSE_REFUSAL | no governed capability | **CR5** | forward approved-limit methodology does not exist (protected) | — | (point-in-time) | none — refusal correct | n/a |
| Q25C | FALSE_REFUSAL / FALSE_REFUSAL | no governed capability | **CR5** | forward approved-limit methodology does not exist (protected) | — | (point-in-time) | none — refusal correct | n/a |
| Q22B | FALSE_REFUSAL / FALSE_REFUSAL | plan builds, guard refuses | **CR6** | a plan builds; a downstream guard refuses | — | analytical_composition | guard defect, separate | no |
| Q22C | FALSE_REFUSAL / FALSE_REFUSAL | plan builds, guard refuses | **CR6** | a plan builds; a downstream guard refuses | — | analytical_composition | guard defect, separate | no |

### Counts

| bucket | n | disposition |
|---|---:|---|
| **CR1** capability reachable, wrong ownership | **1** | **recovered in Phase 1** (Q10A) |
| **CR2** capability exists, not exposed to MI | 0 actionable | `compute_forecast_bridge` is CR2-shaped but wiring it recovers nothing — its candidate questions are CR4 |
| **CR3** composition of existing primitives | 0 | none proven |
| **CR4** recognition failure, capability proven | **24** | out of scope by instruction; recorded for the semantic layer |
| **CR5** genuine capability/config gap | 6 | refusals remain correct |
| **CR6** downstream guard defect | 2 | recorded separately, not folded into recovery |

### Capabilities discovered but previously unreachable

- `mi_agent_api/forecast_bridge.py::compute_forecast_bridge` — public and
  deterministic; live callers only in `mi_agent_pptx`. No MI route reaches it.
- `mi_workflows/analytical/executors.py::funded_balance_forecast` — reachable,
  and already produces both components of the contribution family; the two
  shadowed wordings simply never arrive.

### CR5, with evidence rather than assumption

- **CFO60 / CFO61** — `concentration_analysis` measures a **governed dimension
  set**. The sibling *"Show origination channel concentration."* answers; `product`
  and `broker` are not in the set. Extending it is governed-configuration work.
- **CFO71** — not a capability gap at all: a **correct ambiguity refusal**
  (*"'value' could mean Balance or Valuation"*). The sibling *"Show the pipeline
  by stage."* proves the analytic works.
- **Q25A/B/C** — forward approved-limit methodology, protected and out of scope.

---

## Phase 4 — recoveries implemented

**One: the Phase 1 CR1.** Nothing further met all six conditions.

**`funded_bridge` was NOT narrowed.** Its rule requires the replacement path to
be proven first. It is not: both shadowed questions are CR4 and never reach the
capability that computes them. Narrowing today would leave one answering a
pipeline total that is not a funded-vs-pipeline contribution, and the other
refusing. Stopping and reporting, as the rule directs.

## Questions converted

| | |
|---|---|
| refusal/wrong → correct | **Q10A** — funded-book substitution → the pipeline answer, from the proven path |
| internal error → governed refusal | *"What is the weighted expected pipeline contribution?"*, latent until a route declines |

## Deliberately left as refusals

CR5×6 (governed dimension set, ambiguity, forward-limit methodology) and CR4×24,
which the instruction excludes from phrase-level repair in this task.

## Regression

| surface | result |
|---|---|
| 75-question bank + CFO 91, both arms | 1 movement total (Q10A), Phase 1 |
| CFO 91 | byte-identical, both arms |
| six registered pipeline answers | **6/6 byte-identical** |
| 1,446-question spec census | invalid intent/shape pairs 23 → **0** |
| frozen 278-module manifest | **85 failures, unchanged** |
| new WRONG / current-state substitution | none |
| previously correct regressed | none |
| internal validation error user-visible | none |

## Remaining blockers

1. **CR4 × 24** — the largest recoverable population, all with a proven
   capability, all blocked on recognition. Needs the semantic layer, not grammar.
2. **CR6 × 2** (Q22B/Q22C) — a plan builds (`period_movement` ×2) and a guard
   refuses.
3. **`funded_bridge` shadowing** — cannot be narrowed until a reachable
   replacement exists for its two shadowed wordings.
