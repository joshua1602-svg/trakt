# Pre-registration — KIND_THRESHOLD receipt defect

Base `470a92a`. Written and committed **before** the production change.

## The defect

`reconcile_routed_facets` stamps every `KIND_THRESHOLD` facet:

```python
elif facet.kind == KIND_THRESHOLD:
    facet.status = LOST
    facet.reason = ("this governed capability does not apply a value "
                    "threshold, so the figure is not restricted to it")
```

Unconditional. No evidence consulted — on a path whose sibling `KIND_POPULATION`
facets are stamped from `metadata.populationApplied` by `reconcile_population`
before ever reaching this function.

Reproduced on the live book (`alderbridge`, 3 governed periods): the route
narrows correctly per period, publishes the narrowing, and the answer refuses.

## What the evidence actually supports

| | carries field | carries operator | carries value |
|---|---|---|---|
| `KIND_THRESHOLD` facet (comparator form) | **no** (`field_key=None`) | no — only inside `label` | no — only inside `label` |
| `populationApplied.applied` | yes (`"<field> (applied within each period)"`) | **no** | **no** |
| `spec.filters` → `material_predicates` | yes | yes | yes |

The ledger alone cannot prove a specific field/operator/value threshold, and the
threshold facet names none of the three structurally. **Certification therefore
cannot come from matching those two objects.**

It can come from a different, sound route. `_apply_filters` applies **every**
entry in `spec.filters` or raises `_require_column`, and appends each field it
applied to the ledger. So when every governed material predicate the spec carries
is present in the ledger's `applied` list and none is `unavailable`, every
narrowing the question expressed did run — and the threshold facet, which is the
lexical twin of one of those predicates, is proven.

The codebase already records that these are twins: *"raising every spec.filters
entry on the point-in-time path would add a population facet to 81 corpus
questions, all of them numeric bounds — 'LTV above 50%' — which KIND_THRESHOLD
already represents. Two facets for one predicate is the duplicate-claim defect."*

Measured 1:1 on every probe, including a two-threshold question.

## Authorised to move

Only cases satisfying **all four**:

1. a governed threshold was requested (a `KIND_THRESHOLD` facet exists);
2. execution genuinely applied it (`_apply_filters` ran the predicate);
3. matching evidence was published (`populationApplied.applied` names every
   material predicate's field, `unavailable` is empty);
4. the receipt previously stamped it LOST solely because the branch ignored that
   evidence.

Expected movement: **REFUSED → DELIVERED**, on funded filtered evolution.

## Must not move

- Filtering economics — values, row counts, periods, the deterministic result.
- `_apply_filters`, `_filtered_funded_evo`, the route, the plan layer.
- Interpretation, dataset, measure, route ownership.
- Unfiltered evolution.
- Pipeline filtered evolution (the route still returns `None` for it).
- Any threshold not proven by execution evidence.
- Geographic-scope refusals (a separate facet and a separate owner).

## Not authorised

- No route-name whitelist, no raw-question words, no label parsing.
- No "thresholds are supported now" blanket rule.
- No inference from interpretation alone — spec presence is not evidence.
- No new filter semantics, no Pipeline-filter expansion.

## Fail-closed conditions (must remain LOST)

| control | why |
|---|---|
| no `populationApplied` at all | nothing executed, nothing proven |
| ledger names a different field than the spec predicate | the narrowing that ran is not the one asked for |
| `unavailable` non-empty | a requested narrowing could not be applied |
| spec carries no material predicate | the threshold never resolved into a predicate |
| fewer predicates than threshold facets | one threshold resolved, another did not |

## STOP conditions

`STOP — POPULATION LEDGER TOO WEAK` · `STOP — RECEIPT EVIDENCE INSUFFICIENT` ·
`STOP — THRESHOLD RECEIPT BLAST` · `STOP — CALCULATION MOVEMENT`
