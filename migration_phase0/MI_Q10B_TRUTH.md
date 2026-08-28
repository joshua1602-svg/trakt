# Q10B — independent truth

> *"Give me an overview of the pipeline by size and stage."*

Computed from the governed pipeline extract with pandas, **using neither
implementation as the oracle**, before either answer was classified.

## What "size" is, under governed semantics

A governed size band exists: **`ticket_bucket`**, declared in the semantics
registry with `derived_from: current_outstanding_balance`. The registered
dimension vocabulary maps `ticket size`, `loan size`, `ticket`, `ticket bucket`,
`balance band`, `balance bucket`, `exposure band` to it.

**The bare word "size" is not in that vocabulary.** That is why the
deterministic parser recognised `stage` and dropped `size`: not a routing or
execution fault, a vocabulary gap (standing finding F1).

## The population and the calculation

Population: the governed pipeline extract, **8 rows, £3.6m total**.
Aggregation: total balance. Grouping: `pipeline_stage` × `ticket_bucket`.

Stage values: APPLICATION, COMPLETED, KFI, OFFER, WITHDRAWN.
Ticket values: 100-150k, 200-300k, 300-500k, 500k-1m.

| pipeline_stage | ticket_bucket | balance | loans |
|---|---|---|---|
| APPLICATION | 500k-1m | £500,000 | 1 |
| COMPLETED | 100-150k | £100,000 | 1 |
| COMPLETED | 500k-1m | £700,000 | 1 |
| KFI | 500k-1m | £600,000 | 1 |
| OFFER | 200-300k | £200,000 | 1 |
| OFFER | 300-500k | £300,000 | 1 |
| OFFER | 500k-1m | £800,000 | 1 |
| WITHDRAWN | 300-500k | £400,000 | 1 |

**Non-empty stage × size groups: 8.** By stage alone: 5.

The question names both axes, so stage × size is required. Recorded as
`independent_truth: {"cells": 8}` — the group-structure claim the grader can
check against artefact rows.

## Classification

| implementation | output | verdict |
|---|---|---|
| deterministic | "covering 5 groups … grouped by Pipeline Stage" | **WRONG / SILENT** — a requested axis dropped, with nothing disclosing the loss |
| Opus arm | "covering 8 groups … grouped by Ticket Size and Pipeline Stage" | **CORRECT** — matches truth |

The frozen human grade recorded CORRECT against the deterministic answer with
`independent_truth: null`. That grade was wrong, and it is the second time in
this programme a frozen human verdict has covered a silent loss.

**No production code was changed for Q10B.** Closing the deterministic gap
would mean adding "size" to the dimension vocabulary — grammar, and out of
scope. The oracle is corrected; the product gap is recorded.
