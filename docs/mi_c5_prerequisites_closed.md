# C5 prerequisites — dataset ownership finalised, `case` aligned, comparison periods carried

Base `83da6a7` → `ad67ee9`. C5 was **not** started.

---

## 1. Dataset-remediation regression, by name

Full estate, same environment, against the 214-name baseline at `d927963`:

```
186 failed, 10229 passed, 35 skipped, 16 xfailed, 28 errors   =  214 names
```

| | |
|---|---|
| introduced | **1** |
| gone from baseline | **1** |

**Gone** is a rename: `test_unqualified_amount_routes_to_active_dataset` →
`test_an_unqualified_question_means_the_same_thing_on_every_tab`.

**Introduced was mine, and is recorded as a real failing name rather than waved
through:**
`test_the_second_owners_wider_vocabulary_survived_its_retirement` still asserted
`resolve_dataset("How many cases completed?") == "pipeline"`, written when the
artefact vocabulary had four words and not updated when `case` was dropped two
commits later. A stale assertion of mine, not a product regression. Fixed at
`a8296df`, and not merely deleted — the pipeline half of the can-fail moved to
`"How many applications completed?"` so that half still *can* fail, and the
`case` case was kept with the opposite expectation so the word's absence is
stated rather than inferred from a gap.

Also confirmed at that point: C1–C4 guards and the silent-drop guards **221
passed**; C5 verifier 0 dataset / 0 measure disagreements.

## 2. Status

**DATASET OWNERSHIP REMEDIATED — ZERO UNEXPLAINED BLAST.**

## 3. The `case` house rule

| shape | resolves | why |
|---|---|---|
| bare `case` / `cases` | **dataset-neutral → FUNDED/default** | in this estate it names a funded loan at least as often as a pipeline case |
| `pipeline cases`, `open pipeline cases`, `cases … in the pipeline` | **PIPELINE** | independent pipeline evidence in the same sentence |
| `application`, `kfi`, `offer` | **PIPELINE** | unambiguous pre-funding artefacts |
| `forecast …` | **FORECAST** | forecast precedence, unchanged |

Neither route identity nor the active tab is used to disambiguate `case`.

The evidence, not the intuition: `"Which region gained the most cases since last
month?"` sits in the P1C golden bank under `# -- loan count --`, beside
`"Which region added the most loans month-on-month?"`, and expects a ranked
**funded** movement. Across the 882 corpus questions, **not one** reaches the
pipeline through a bare `case` — every artefact-driven movement comes from
`application`, `kfi` or `offer`.

## 4. The intent vocabulary change

`mi_workflows/analytical/intent.py`, `_PIPELINE_TERMS`: `" case "` and
`" cases "` removed. **Nothing else.**

`" caseload "` stays — a different word, unambiguously the pipeline.
`_COMPLETION_TERMS`, the strong artefacts, explicit-pipeline and stage
semantics, and forecast precedence are untouched.

Why it mattered: `pipelining` (line 617) is the only reader of that vocabulary,
and it sets `FAMILY_PIPELINE` + `REQ_PIPELINE_DATASET`. `unmet_requirements`
then checks that requirement against `dataset != "pipeline"`. So the intent
layer asserted a pipeline dataset for sentences the authoritative owner called
funded, and the refusal was the symptom.

## 5. Blast radius, 882 corpus questions + 10 probes

The corpus alone would have been a **blind census**: all 7 of its bare-`case`
questions also say *"pipeline"* outright, so none of them can move. Ten probes
carry the shape the change exists for.

```
intent FAMILY changed       : 3
intent REQUIREMENT changed  : 3
DATASET changed             : 0
ROUTE changed               : 0
ok/refusal changed          : 1
```

The three static movers are the bare-case shape and nothing else:
`"How many cases are there?"`, `"Which region gained the most cases since last
month?"`, `"cases by region"`. **Zero dataset movement** — the owner already
called them funded, so this closes a disagreement rather than moving an answer.

**The one behavioural movement, stated plainly:** `"How many cases are there?"`
was a controlled refusal (*"I understood this as a pipeline question…"*) and now
answers **11,035 loans** from the funded book. That is the house rule.

### The limit, asserted rather than left implicit

`"How many cases completed?"` keeps `dataset=funded` **and** keeps its PIPELINE
family, because that reading comes from `completed`, not `case`. It still
refuses, and should: a completion is the pipeline→funded transition event, and
its sibling *"How many loans are we completing at the moment?"* is one of the
four measured defects the fail-closed rule exists to stop — where it returned
11,035 loans with a green guard. `_COMPLETION_TERMS` was deliberately not
touched.

## 6. Worked-example matrix

Every row asserted across **six** tab values (`None`, `""`, `funded`,
`pipeline`, `forecast`, `nonsense`) — the tab is semantically irrelevant.

| question | dataset |
|---|---|
| Which region gained the most cases since last month? | FUNDED |
| How many cases are there? | FUNDED |
| cases by region | FUNDED |
| How many cases completed? | FUNDED |
| How many pipeline cases are there? | PIPELINE |
| pipeline cases by stage | PIPELINE |
| open pipeline cases | PIPELINE |
| How many applications are there? | PIPELINE |
| How many KFIs are there? | PIPELINE |
| How many offers are outstanding? | PIPELINE |
| Forecast application volumes next month | FORECAST |
| Forecast case completions next quarter | FORECAST |

Plus can-fail tests that the alignment did **not** remove the PIPELINE family
from explicit pipeline cases or from the strong artefacts, and a structural test
that a bare `case` cannot reappear in `_PIPELINE_TERMS`.

## 7. `comparison_period` — before and after

**Before:** `TimeClaim` carried `comparison_period: Slot` whose `raw_text` was
`", ".join(compare_periods)`. `Slot` has no list field. Recovering the pair
meant splitting a display join — re-parsing a serialisation, which breaks on any
period label containing the separator. **0 of 26** readings of the
`temporal_compare` surface carried it structurally.

**After:**

```python
TimeClaim.comparison_periods: Tuple[str, ...] = ()      # the values
TimeClaim.comparison_period: Slot                       # the wording, unchanged
analytical_plan.comparison_periods(interpretation)      # the structural read
```

The same closure `window_periods` already made for `trend_window`.

**Additive, proved.** All 882 corpus interpretations dumped before and after and
compared field by field: **5 gain a populated `comparison_periods`, 0 other
differences.**

**What was backed out.** The first cut had `comparison_period()` itself return
the pair's first element. That is a change of MEANING, not representation — on
all five corpus questions carrying a comparison it turns `"October, November"`
into `"October"` — which is the registered STOP. Reverted; the accessor keeps
its wording semantics and the one caller wanting structure asks by name.

**The planner consumes the structure.** `build_funded_bridge_plan` takes the
first period from the pair rather than the rendered string; a two-period
question would otherwise hand `STACK_PERIODS` a start period no tape has.
Proved representation-only **before** the switch: across all **12** cases
executed routing shows `funded_bridge` owns, the structural first period and the
join are identical.

**Payload/receipt, asserted denominators, before-halves taken with the change
stashed out:**

| route | pairs | differences |
|---|---|---|
| `funded_bridge` | 36 of expected 36 | **0** |
| `geo_exposure` | 36 of expected 36 | **0** |
| `period_movement` | 36 of expected 36 | **0** |

## 8. Production lines, per prerequisite

| | executable | comment/docstring |
|---|---|---|
| `case` vocabulary alignment | **+2 −2 (net 0)** | +21 |
| `comparison_period` closure | **+22 −3 (net +19)** | +40 |

## 9. C5 dependency verifier

```
readings                                   : 26
dataset disagreements, contract AS BUILT   :  0   (was 10)
dataset disagreements, WITH the view wired :  0   (was  3)
measure disagreements (at the same dataset):  0
readings whose periods are STRUCTURAL      : 26 of an EXPECTED 26   (was 0)
```

The denominator is **asserted**, not reported: every owned case is a two-period
comparison — that is what makes it a `temporal_compare` question — so every
reading must carry the pair. The instrument now exits non-zero on a gap.

**The C5 dependency model contains no known prerequisite gap.**

## 10. C5 thresholds — unchanged, and a caveat that matters

`docs/mi_conversion5_stop_conditions.md` has **one commit**, `7d9f4c6`, and has
not been touched since. Shared ≤ 75, route-specific 90–150, total ≤ 225 stand.
`docs/mi_c5_cost_regime_rebaseline.md` and the C1–C4 reports are likewise
untouched; no historical cost was moved.

Both prerequisites are **product/contract hardening outside C5 cost**:

* **Dataset ownership** fixed a live defect — the same sentence meant different
  things on different tabs — and consolidated two owners into one. Its lines do
  not count against C5. It does **not** consume C5's budgeted `dataset`
  accessor either: `analytical_plan` still has no dataset accessor, so C5 still
  writes one. The ~20-line estimate stands.

* **`comparison_period`** was **not** anticipated by the cost model. The
  re-baseline recorded `comparison_period` as "already bridged in C4" with "no
  unread-field surprises identified", and that was wrong at field level. The C5
  STOP report classified it as an unplanned Regime B item that, absorbed, would
  have taken predicted shared to **62 + 20 = 82** against a threshold of 75.

**So C5 is now easier to pass than it would have been, and that should be read
into its verdict rather than discovered afterwards.** Roughly **19 executable
lines of shared work** that C5 would have had to carry have been spent outside
it. When C5 runs, its report should state measured shared cost *and* "plus 19
lines closed as prerequisites beforehand", or a REGIME MODEL SUPPORTED verdict
will be comparing C5 against C1–C4 on different terms.

The thresholds are **not** being raised or lowered to account for this. Moving
them would be the rationalisation the programme exists to avoid; disclosing the
shift is the honest alternative.

## 11. Final regression, by name

Full estate at `4c62a0f`, same environment, against the 214-name baseline:

```
185 failed, 10279 passed, 35 skipped, 16 xfailed, 28 errors   =  213 names
```

| | |
|---|---|
| **introduced failing names** | **0** |
| gone from baseline | 1 — the rename in §1 |

Re-confirmed at the same commit:

```
C5 dependency verifier   0 dataset / 0 measure / 26 structural of an EXPECTED 26   (exit 0)
882 dataset census       M1 2302, M2 5, M3 96, UNEXPLAINED 0
P1C golden bank + dataset ownership + comparison periods   161 passed
```

Silent drops remain 0; no unexplained route movement; no unexplained
answer/refusal movement. C1–C4 unchanged.

## 12. Status

**C5 PREREQUISITES CLOSED — READY TO RESTART**

## 13. Recommended next task

**Run Conversion 5 on `temporal_compare` against the thresholds already
committed at `7d9f4c6`** — shared ≤ 75, route-specific 90–150, total ≤ 225 —
and report its measured shared cost **alongside the 19 executable lines closed
as prerequisites beforehand**, per §10.

## 14. Scope held

C5 not started. No route converted. No subject/operation accessors beyond the
prerequisite. No threshold change. No `evolution` or `period_change` migration.
No T3–T7, no LLM logic, no broad cleanup.
