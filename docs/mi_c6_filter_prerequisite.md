# C6 filter prerequisite — measured, and not closed

Base `ee776aa`. **No production code changed.** C6 not executed.

The task was to make `FilterClaim` authoritative for `evolution` and reproduce
per-period narrowing through `analytical_plan.lens_filters`. That remediation is
**not buildable as specified**, for reasons established from executable evidence
below. Three of the task's own STOP conditions are factually met.

---

## 1. Three things called "filters", and what each can actually express

The brief warned not to assume `_apply_filters` and `FilterClaim` mean the same
thing because they share a word. Measured, the warning understates it — the third
name is further away still.

```
FilterClaim fields  : state, raw_text, span, reason, source,
                      operator, value, categorical_value, provides, clause_id
...carries a FIELD key? : NO

lens_filters returns    : source_portfolio_id only
lens_filters keys on    : plan step with kind == "source_portfolio_lens"

_apply_filters handles  : semantic-key resolution=True, numeric ops=True,
                          percent scale=True
_scope_frame_lens on an absent column : NO-OP (continue)
_apply_filters  on an absent column : RAISES (_require_column)
```

`FilterClaim` says so itself, by design:

> *"The FIELD the condition bears on is not here: every `threshold` facet in the
> release candidate carries `field_key=None` for the same reason — identifying
> that a clause exists is a different job from resolving what it binds."*

And `clause_id` — the mechanism that would join a field-half to a bound-half —
"stays None until an interpreter supplies a basis for it".

`lens_filters` is not a row-predicate engine at all. It is the **source-portfolio
lens**: it finds a `SELECT_POPULATION` step of kind `source_portfolio_lens` and
returns `{"source_portfolio_id": [...]}`. It cannot express `LTV > 50`.

## 2. The current ownership chain, traced

```
question
  → ParsedQuestion.parse(question, semantics).spec.filters      {'current_loan_to_value': {'op':'gt','value':50.0}}
  → _route_evolution: filtered = bool(spec.filters)
  → if filtered and dataset != "funded": return None            ← filtered PIPELINE evolution is not handled at all
  → _filtered_funded_evo(...) per governed period
       → mi_query_executor._apply_filters(df, spec, semantics, [])
            → resolve_semantic_field → _require_column → mask
  → envelope.metadata.populationApplied  (execution evidence)
```

Per-period narrowing is **already correct**: `_filtered_funded_evo` loops the
governed frames and applies the filter inside each one. It does not filter the
latest population and project backwards.

The contract runs alongside, and cannot describe it:

| probe | `spec.filters` | `FilterClaim(op, val, cat)` | `lens_filters` |
|---|---|---|---|
| F-NUM `above 50% LTV` | `{'current_loan_to_value': {'op':'gt','value':50.0}}` | `[(None,None,None), ('gt','50.0',None)]` | `None` |
| F-CAT `for London` | `{'collateral_geography': 'London'}` | `[(None,None,'London')]` | `None` |
| F-STAGE `offer-stage cases` | `{}` | `[(None,None,'OFFER')]` | `None` |
| F-NONE (control) | `{}` | `[]` | `None` |

The claim carries the **bound** and never the **field**. `lens_filters` returns
`None` for every one.

## 3. What evolution actually filters on, across the corpus

```
corpus questions carrying spec.filters: 119 of 882
   current_loan_to_value          56
   youngest_borrower_age          20
   current_outstanding_balance    15
   borrower_type                  15
   collateral_geography            9
   months_on_book                  4
   current_interest_rate           2
expressible by lens_filters (source_portfolio_id only): 0
```

**Zero of 119.** Routing evolution's filters through `lens_filters` would express
none of them, and `_scope_frame_lens` no-ops on an absent column — so the
narrowing would silently vanish and a whole-book figure would answer a narrowed
question. That is the exact defect class this programme has removed twice.

## 4. Delivered filtered coverage — there is none, and the reason is a third owner

On the real demo book (`alderbridge`, 3 governed periods, ~1.9bn):

| probe | ok | per-period narrowing actually performed | outcome |
|---|---|---|---|
| `evolution by month for loans above 50% LTV` | **False** | `current_loan_to_value` applied per period, `rowsAfter: 1889` | refused |
| `evolution by month for London` | **False** | `collateral_geography` applied per period, `rowsAfter: 1380` | refused |
| `evolution by month for borrowers over 75` | **False** | `youngest_borrower_age` applied per period, `rowsAfter: 2722` | refused |
| `evolution by month` (control) | True | n/a | 3 periods, £1.93bn → £1.96bn |

The route narrows correctly, **declares** it in `populationApplied`, and the
answer is refused anyway. The refusal owner is `reconcile_routed_facets`:

```python
mi_agent/execution_receipt.py:3249
        elif facet.kind == KIND_THRESHOLD:
            facet.status = LOST
            facet.reason = ("this governed capability does not apply a value "
                            "threshold, so the figure is not restricted to it")
```

**Unconditional.** No evidence consulted, no route exception, no ledger check —
on a path where the sibling `KIND_POPULATION` branch two blocks below *does*
consult `population_applied(facet, ledger=population_ledger(envelope), ...)`.
The geographic branch falls through to the same shape (`LOST`, "the geographic
scope was not applied to the calculation").

So a fourth semantic owner asserts that no routed capability applies a threshold,
while the route is applying one and saying so. **This — not evolution's filter
machinery — is why no filtered evolution question can deliver.** The fix pattern
already exists ten lines away in the same function.

## 5. Why the specified remediation cannot be built

| step the brief specifies | status |
|---|---|
| make `FilterClaim` authoritative | **impossible today** — the claim cannot name the field |
| reproduce per-period narrowing via `lens_filters` | **impossible today** — expresses only `source_portfolio_id`, 0 of 119 |
| establish delivered filtered coverage | **impossible today** — an unconditional facet refuses every case |

Building it would require, in order:

1. a sound field↔bound join on `FilterClaim` (the `clause_id` work Stage 1
   deliberately deferred for want of an interpreter basis);
2. a **general** filter primitive in the plan layer — `lens_filters` is a
   portfolio lens and must not be widened into a row-predicate engine by
   accident;
3. the threshold/geographic facet owners taught to read execution evidence.

Each is a prerequisite in its own right. None is plumbing, and none can honestly
be called "the smallest generic production change".

## 6. Dataset-guard scrutiny (§13)

The guard re-keyed in the previous task, audited as asked:

| question | answer |
|---|---|
| what authorises the allowance? | a companion test asserting the stage reader returns only canonical stages, never a view, and **consumes** `resolve_dataset` rather than deriving one |
| merely a function-name whitelist? | it is a **module-path** whitelist, with a semantic condition attached to one named module |
| does an undisclosed third caller still fail? | **yes** — verified by mutation, a new `undisclaimed_mention` caller in `chat_routing.py` fails it |
| can a caller self-declare? | **no** — the allowance list lives in the test file; production cannot grant itself entry |

**Material weakness, reported rather than redesigned:** a second dataset reader
added *inside* an already-allowed module passes silently. Verified — appending a
rogue `undisclaimed_mention`-based dataset reader to `question_interpretation/
lexical.py` leaves all 96 tests green. The guard catches a new *module*, not a
new *reader*. With only two allowed modules, both small single-purpose owners,
that is a bounded exposure, but it is weaker than the guard's own docstring
implies.

## 7. Stage/Funnel superset recheck (§15)

The ten superset disagreements remain inert. Nothing consumes the stage claim:
sub-route selection is still `_FUNNEL_KEYWORDS` and the three phrases. The claim
is produced and read by no one, so no previously-unanswerable question became
answerable. Confirmed by the zero-movement census already on record, and by there
being **no production change in this task at all**.

## 8. C6 four-part dependency matrix

| dependency | represented | owner agreement | plan consumable | delivered coverage | status |
|---|---|---|---|---|---|
| dataset | yes | 0 disagreements | `dataset_of` | ✓ | **READY** |
| measure | yes | 0 disagreements | `measure_request` | ✓ | **READY** |
| historical series | yes | n/a | `span_periods` | ✓ 5 weekly / 3 monthly | **READY** |
| time / grain | yes | declaration wins | n/a | ✓ | **READY** |
| population | yes | route does not narrow | `_whole_dataset_step` | n/a | **READY** |
| ordinary evolution | yes | ✓ | dispatch | ✓ | **READY** |
| Pipeline Stage | yes | 894/904, superset only | **no** | 4 of 5 stage-scoped | REPRESENTED, NOT CONSUMED |
| Stage evolution | yes | ✓ | **no** | ✓ | REPRESENTED, NOT CONSUMED |
| Funnel | yes (a composition) | ✓ | **no** | ✓ | REPRESENTED, NOT CONSUMED |
| **filters** | **field NOT representable** | **not measurable** | **NO** | **0 delivered of 119** | **BLOCKED** |

Not green. Thresholds are therefore **not** pre-registered and
`docs/mi_conversion6_stop_conditions.md` is **not** created.

## 9. Cost

| bucket | raw lines |
|---|---|
| shared filter infrastructure | **0** |
| product hardening | **0** |
| cleanup | **0** |
| **total production** | **0** |
| instrument (`migration_phase0/filter_ownership_trace.py`) | reported separately, not production |

No production code was changed, because no change could be made honestly. The
prerequisite cost stands at the 171 raw lines already spent on the Pipeline Stage
contract.

## 10. Status

# STOP — FILTER CONTRACT INCOMPLETE

`evolution` legitimately applies filters on seven governed fields across 119
corpus questions. `FilterClaim` cannot express **which field** any of them binds
to, and `lens_filters` can express none of them. Two further STOP conditions from
the same brief are also met and are recorded above rather than chosen as the
headline: no filtered evolution case delivers (§5), and `lens_filters` would
change filtering semantics from *raise* to *silent no-op* on an absent column
(§7).

**Recommended next task — and it is not the filter migration.** Close the
unconditional `KIND_THRESHOLD` refusal in `reconcile_routed_facets` first: teach
it to read the `populationApplied` ledger the route already publishes, exactly as
the neighbouring `KIND_POPULATION` branch does. That is a bounded, single-owner
product-hardening fix with a pattern already in the same function; it converts
119 corpus questions from *narrowed-then-refused* to delivered, and it is the
only way any filtered evolution case can become non-vacuous. The contract work
(field↔bound join, general plan filter primitive) should be scoped only after
there is a delivered case to measure equivalence against.
