# Period-change capability agreement — STOP at the Phase 2 gate

```
PRODUCT_BASELINE = 00bb3e9d8175a395b6643772379866d6bc6169eb
PRODUCT_HEAD     = 00bb3e9d8175a395b6643772379866d6bc6169eb  (product unchanged)
PRODUCT_FILES_CHANGED = 0
STRUCTURED_NORMALISATION_POSSIBLE = NO
STOP_CONDITION = #11 (and #8) — the clean target state adds a missing semantic
                 slot rather than inferring around it
```

## ROOT_CAUSE — it is not a capability disagreement

The capability labels are **already derived deterministically and correctly**.
`normalise.py::_owning_capability` binds the owner from measure ownership, and
the governed vocabulary says:

```
current_outstanding_balance   owning_capability = None          (generic field)
portfolio_overview            owning_capability = portfolio_summary
funded_balance_movement       owning_capability = funded_bridge
bridge_component              owning_capability = funded_bridge
```

So the three Q18 capabilities are the faithful consequence of three different
**measures**:

| | capability | operation | measure | period |
|---|---|---|---|---|
| Q18A | generic_analysis | compare | current_outstanding_balance | relative_pair |
| Q18B | portfolio_summary | compare | **portfolio_overview** | relative_pair |
| Q18C | funded_bridge | movement | **funded_balance_movement** | relative_pair |

Everything else is identical across all nine intents in the family:
`comparison.kind=none`, `dimensions=[]`, `outputs=[]`, `target=null`,
`grain=monthly`, `statistic=null`. Population and filters differ only where the
questions genuinely differ (Q19 = Direct lens, Q20 = drawdown filter).

**The root cause is a MEASURE disagreement.** One business question — "how did
the book change?" — was resolved to three different governed measures.

## Why it cannot be normalised from structured state

To make the paraphrases agree, the **measure** would have to be rewritten. Three
independent reasons say no:

1. **The governed contract forbids it.** Every specialist measure's own
   description reads: *"Owned by the `<capability>` capability. Its methodology
   is deterministic and **is not composed by the interpreter**."* Rewriting
   `funded_balance_movement` to `current_outstanding_balance` asserts that a
   capability-owned measure with an undisclosed methodology equals a generic
   field delta. Nothing in the structured contract supports that equality.

2. **The repository already drew this line, deliberately.**
   `normalise.py::BOUNDED` names this exact residual:

   > `movement_measure_choice`: "period_movement/current_outstanding_balance
   > against funded_bridge/funded_balance_movement is a choice between two
   > governed MEASURES, not between two owners of one measure. Collapsing it
   > would decide whether a period change is a net movement or a bridge figure,
   > which is deterministic analytical behaviour this sprint may not change."

   The normalisation owner exists, is wired into the compiler
   (`compiler.py:298`), records every rewrite in plan provenance
   (`compiler.py:1010`), and names Q18/Q19/Q20 in its own docstring. It already
   normalised what could be normalised: period spelling
   (`previous_reporting_period` → `relative_pair`), identity labels, and
   measure→owner binding. What remains it declined on stated grounds.

3. **No structured field can adjudicate.** Every other slot is identical, so
   there is no deterministic discriminator that says the reader wanted a net
   delta rather than a decomposition or an overview. Making `operation`
   authoritative does not help: Q18A and Q18B both say `compare`, so the
   measure would still have to be rewritten for Q18B.

Stop condition #8 also applies: collapsing `funded_balance_movement` would not
preserve explicit bridge intent as a class, because the only structural signal
separating "bridge" from "movement" is the measure itself.

## THE MISSING SEMANTIC SLOT

**There is no governed representation of the FORM OF CHANGE, independent of the
measure.** Today it is smuggled into the measure choice:

```
net delta            current_outstanding_balance over a period pair
decomposition        funded_balance_movement / bridge_component
multi-metric view    portfolio_overview
```

and `operation` *also* carries `movement` / `bridge` / `summary` / `compare`, so
one distinction is expressible in two slots that can disagree — which is exactly
what Q18 shows. The target-state change is to make one slot authoritative for
form-of-change and derive the other, so that "how did the book change" resolves
to one contract regardless of which slot the model reaches for.

**Not implemented in this sprint, by instruction.**

## A second, narrower residual — Q20 is a genuine owner choice

Q20 is different from Q18/Q19 and worth separating:

```
Q20A  generic_analysis  movement  current_outstanding_balance  relative_pair
Q20B  period_movement   movement  current_outstanding_balance  relative_pair
Q20C  period_movement   movement  current_outstanding_balance  relative_pair
```

Identical measure, operation and period — **a pure capability disagreement with
no measure difference.** This is redundancy 3's own shape, and normalisation
cannot reach it only because `_owning_capability` returns `None` for a measure
with no owning capability.

It is still not a normalisation, because the two owners are not
interchangeable: `CAPABILITY_OPERATIONS["generic_analysis"]` lists `movement`,
yet no generic runtime implements it — Q19A and Q20A both stopped at
`OPERATION_NOT_TEMPORAL`. So the vocabulary says generic_analysis supports
movement while only `period_change` actually does. Choosing between them decides
which arithmetic runs, which normalisation may not do.

**This is a capability-ownership decision:** does generic-measure movement
belong to `generic_analysis` or to `period_movement`? Resolving it in the
vocabulary would make Q20 paraphrase-stable and is strictly smaller than the
form-of-change slot. Reported, not implemented.

## Paraphrase invariance, measured offline over all 45 canonicals

```
CANONICAL_GROUPS (>=2 compiled variants)            34 of 45
CANONICALS_WITH_CAPABILITY_DISAGREEMENT_BEFORE       3   Q18, Q19, Q20
CANONICALS_WITH_OPERATION_DISAGREEMENT_BEFORE        8
CANONICALS_WITH_MEASURE_DISAGREEMENT_BEFORE          8   Q08 Q09 Q10 Q18 Q19 Q25 SM03 SM09
CANONICALS_WITH_MATERIAL_PLAN_DISAGREEMENT_BEFORE   10   + Q20, Q21
```

AFTER figures are identical to BEFORE: nothing was changed.

Note that capability disagreement is the *narrow* problem (3 canonicals) and
measure disagreement is the wider one (8). Fixing capability labels would not
have addressed most of the instability.

## Q18 / Q19 / Q20 status

```
Q18  STILL_DEFECTIVE  — measure disagreement; needs the form-of-change slot
Q19  STILL_DEFECTIVE  — same shape as Q18
Q20  STILL_DEFECTIVE  — capability-ownership decision, narrower and separable
```

No variant was changed, so BEFORE and AFTER are the same for all nine; the
recorded state is in the Phase 1 table above and in the 135-bank evidence.

## Controls

Not run, and deliberately so: with zero product change there is nothing to
control against. The explicit-bridge, level-comparison, point-in-time, pipeline
and scope controls belong to the sprint that implements a change, and asserting
them here would report a gate that guarded nothing.
