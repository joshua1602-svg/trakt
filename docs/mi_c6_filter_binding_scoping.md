# C6 filter architecture scoping — governed field ↔ predicate binding

Base `4e27e4b`. **Scoping and design only.** No production code changed, no
`FilterClaim` modified, no plan primitive added, C6 not executed.

---

## 1. The premise needs one correction

The brief states:

> the existing route works out that binding itself
> `contract says ">50"` → `route looks at question again` → `route figures out "LTV"`

**That is not what production does.** Measured:

```
route calls _filter_field_of            : no
route calls _resolve_subject            : no
route calls THRESHOLD_SUBJECT_PATTERNS  : no
```

`_route_evolution` never binds a field. It reads `spec.filters`, which the
**parser** has *already* keyed by governed field, and hands it to
`_apply_filters`. The binding is resolved **once**, upstream, before any route
sees it.

So the gap is not a route re-reading English. It is narrower: the binding is
computed and then **carried in a provenance string instead of a structure**.
`projection._filters` writes:

```python
source="parser.filters[%s]" % key      # key == "current_loan_to_value"
```

The governed field is right there, in the `source` text, on the claim.

## 2. The real binding chain

```
question
  → detect_measure_set(q, semantics, available_columns, with_spans=True)
  → remainder = _mask_spans(q, measure_spans)        # blanks measures, PRESERVES offsets
  → _parse_filters(remainder, semantics, available_columns, unresolved=…, spans=…)
       → clause_spans() splits into independent clauses
       → _filter_field_of(clause, semantics, available_columns, anchor, value_end)
            → lexical.THRESHOLD_SUBJECT_PATTERNS      # the single lexical owner
            → _resolve_subject(kind, semantics, available_columns)
                 → _ltv_metric / _age_metric / _rate_metric / _balance_metric / find_field
  → spec.filters = {governed_field: {op, value}}
  → population.material_predicates(spec.filters, semantics) → Predicate(field, op, value)
  → _apply_filters / apply_population
```

**Order matters, and this is easy to get wrong.** An earlier cut of the
instrument called `_parse_filters` directly and bound *"balance evolution … above
50% LTV"* to `current_outstanding_balance`, reporting **no LTV family at all**.
Production masks the measure first, so "funded balance" cannot compete as a
filter subject and LTV wins. Any relocated binder must run **after** measure
extraction or receive the masked remainder.

## 3. Filter families — verified from the corpus, not assumed

119 questions carry filters; **121 predicates**; seven families:

| family | count | example wording | resolved field | resolver |
|---|---|---|---|---|
| LTV | **56** | "above 50% LTV" | `current_loan_to_value` | `_ltv_metric` |
| borrower age | 20 | "borrowers over 75" | `youngest_borrower_age` | `_age_metric` |
| balance | 15 | "loans above 200000" | `current_outstanding_balance` | `_balance_metric` |
| borrower type | 15 | "joint borrowers" | `borrower_type` | categorical resolver |
| geography | 9 | "in London" | `collateral_geography` | `_parse_categorical_filter` |
| months on book | 4 | "the back book" | `months_on_book` | categorical/seasoning |
| interest rate | 2 | "rate above 6%" | `current_interest_rate` | `_rate_metric` |

**97 numeric · 24 categorical.**

## 4. Numeric and categorical are two mechanisms today

They are **not** one path:

- numeric → `_filter_field_of` + `_FILTER_COMPARATORS`, emits `{op, value}`
- categorical → `_parse_categorical_filter` / `_borrower_structure_filter`, emits a
  bare value

They converge only at `material_predicates`, which normalises both into
`Predicate(field, op, value)` — mapping a bare value to `op="eq"` and a list to
`op="in"`. **That convergence point already exists and is already generic.**

## 5. What `FilterClaim` owns today, and why the field was deferred

| responsibility | owned? |
|---|---|
| clause recognition | **yes** |
| operator | yes (when the parser supplied it) |
| value | yes |
| normalised/canonical value | partly — `categorical_value` |
| **field** | **no, deliberately** |
| field family | no |
| clause identity / provenance | `source`, `span`, `provides`; `clause_id` always `None` |

Its docstring gives the reason, and it is sound:

> *"The FIELD the condition bears on is not here: every `threshold` facet …
> carries `field_key=None` for the same reason — identifying that a clause exists
> is a different job from resolving what it binds."*

A `FilterClaim` may be **half** of a clause: the facet layer supplies the wording,
the parser supplies field+bound, and `clause_id` "stays None until an interpreter
supplies a basis for it". Adding `field` to *that* object would put a resolved
fact onto a claim that is sometimes only a fragment of the question's wording.

**So no — adding `field_key` to `FilterClaim` is not automatically correct.**

## 6. The reusable mechanism already exists

`mi_agent/population.py`:

```python
@dataclass(frozen=True)
class Predicate:
    field: str
    op: str
    value: Any

def material_predicates(filters, semantics) -> List[Predicate]
def apply_population(frame, predicates, semantics) -> (frame, PopulationEvidence)
```

- **`Predicate` is exactly `field + operator + value`** — the structure §13 asks for.
- `material_predicates` is generic across all seven families and both shapes.
- `apply_population` "reuses the executor's own comparison semantics rather than
  reimplementing them, so a route and the point-in-time path cannot disagree" —
  i.e. **the deterministic per-snapshot filter primitive already exists.**
- Scope is already excluded by name:

  > *"This module deliberately does NOT treat `source_portfolio_id` as a row
  > predicate: P1I-A governs that phrase family as SCOPE, and collapsing it into a
  > filter here to make one common mechanism would regress that ruling."*

  That is §14's population-lens separation, already enforced with a ruling behind
  it — not something the new design must invent.

It is already consumed by `population_facets`, `reconcile_population`, and the
threshold receipt fix.

## 7. Candidate architectures

### Option A — enrich `FilterClaim` with `field`
Add `field: Optional[str]`, populated from the key `projection._filters` already
iterates.
*Smallest diff. But it puts a resolved field on an object explicitly designed to
hold only what the question SAID, and on claims that may be a clause half.*

### Option B — join `FilterClaim` to the field by SPAN
The parser already computes `spans[field]` — and **no caller passes it**. Revive
it and join each claim to its field by offset overlap.
**Measured, and it does not work:**

```
JOINABLE (exactly one claim overlaps): 90 of 121
AMBIGUOUS (0 or >1 claims overlap)   : 12
parser produced no span              : 19
```

Broken down: **90 of 97 numeric joinable, 0 of 24 categorical** — the categorical
resolvers emit no spans at all. A mechanism that fails closed on a quarter of the
corpus, and on an entire shape, is not the binding.

### Option C — carry the existing `Predicate` on the contract  ← **RECOMMENDED**
`FilterClaim` is unchanged and keeps its job. The interpretation gains a
**resolved** channel alongside it, built by the call `population_facets` already
makes:

```
FilterClaim            "what the question said"   (clause, operator, value, span)
        +
PopulationClaim[]      "what it resolved to"      (Predicate: field, op, value, state)
```

`projection` populates it from `spec.filters` via `material_predicates` — no new
resolver, no new vocabulary, no second registry. The plan reads the resolved
channel; the route reads nothing.

## 8. Comparison

| criterion | A: enrich FilterClaim | B: span join | **C: carry Predicate** |
|---|---|---|---|
| one semantic owner | yes | yes | **yes** |
| no downstream reread | yes | yes | **yes** |
| generic across 7 families | yes | **no** | **yes** |
| numeric **and** categorical | yes | **no — 0/24** | **yes** |
| preserves provenance | dilutes claim's role | yes | **yes — claim keeps wording, predicate keeps resolution** |
| multiple filters | yes | partial | **yes** |
| fail-closed on ambiguity | needs new state | inherent | **yes — claim state already models it** |
| reusable beyond `evolution` | yes | yes | **yes — already used by 3 receipt paths** |
| compatible with registries | yes | yes | **yes — already is** |
| new structures invented | 1 field | 0 | **0 — reuses `Predicate`** |
| blast radius | contract + every claim consumer | projection only | **projection + plan + one route** |
| migration risk | medium — changes a shared claim's meaning | high — silently partial | **low — additive, relocation only** |

**Recommendation: Option C.**

## 9. Ambiguity and fail-closed — what production does today

The parser has an `unresolved` channel: *"A threshold was stated but its field is
not a governed field in this dataset. Refuse it visibly rather than binding it to
some other column."* It is live (passed at one call site).

But measured, it is **narrower than the brief assumes**:

| probe | result |
|---|---|
| "funded balance **over 50**" | binds `current_outstanding_balance`, `unresolved=[]` |
| "loans above 50 **zorkmids**" | binds `current_outstanding_balance`, `unresolved=[]` |

A bare bound does **not** refuse — it falls back to the balance metric. That is a
pre-existing product behaviour, not a C6 question. **The target must reproduce
it**, per "do not invent new filtering behaviour". What the contract must add is
only the *ability to represent* an unresolvable binding (`state=unresolvable`
with a reason), so that when the resolver does decline, the plan fails closed
rather than seeing an empty list. Changing when it declines is a separate product
decision and is out of scope here.

## 10. Multiple-filter semantics — reproduce, do not redesign

| case | today |
|---|---|
| two different fields | both kept — **AND** (successive narrowing) |
| same field twice ("LTV above 50 **and** LTV below 80") | **last clause wins, first silently lost** |
| "joint borrowers **in London**" | `borrower_type` kept, **London lost** |
| unavailable field | `_require_column` raises → controlled refusal |
| OR | **no representation exists anywhere** |

The dict keying of `spec.filters` is what collapses a repeated field. `Predicate`
is a **list**, so Option C can carry both halves of a range without changing
today's behaviour — but doing so *would* change behaviour and is therefore
explicitly out of scope. C6 must preserve the collapse.

## 11. Delivered cases — the representation expresses them exactly

| case | field | op | value | per-period rows | per-period economics |
|---|---|---|---|---|---|
| balance, LTV>50 | `current_loan_to_value` | gt | 50.0 | 1721 · 1799 · 1889 | £432.4m · £451.0m · £472.5m |
| balance, age>75 | `youngest_borrower_age` | gt | 75.0 | 2648 · 2682 · 2722 | £565.5m · £575.3m · £588.4m |
| balance, loan>200k | `current_outstanding_balance` | gt | 200000.0 | 3555 · 3610 · 3666 | £1031.3m · £1047.5m · £1064.9m |
| count, LTV>50 | `current_loan_to_value` | gt | 50.0 | 1721 · 1799 · 1889 | 1721 · 1799 · 1889 |

Every one is a single `Predicate`. No English is reread downstream.

## 12. The plan primitive

`analytical_plan` has seven primitives; `SELECT_POPULATION` exists but is used
**only** with `kind="source_portfolio_lens"`.

Proposed: a **second kind on the existing primitive**, not a new one —

```
SELECT_POPULATION(kind="row_predicates", predicates=[Predicate, …])
SELECT_POPULATION(kind="source_portfolio_lens", portfolio_ids=[…])   # unchanged
```

so §14's separation is structural rather than conventional, and `lens_filters`
keeps meaning exactly what it means now. Execution is `apply_population`, which
already exists and already shares the executor's comparison semantics. Applied
per snapshot inside `stack_periods`, it reproduces today's per-period narrowing
by construction.

Not named `evolution_filter`: nothing about it is evolution-specific.

## 13. Pipeline Stage

Keep it as the **dimension** it already is. `pipeline_stage` is declared
`role: dimension` in the field contract and categorical over `total_pipeline` in
the stratification catalogue, and the stage claim already produces a
`DimensionClaim` + `FilterClaim` pair. When a specific stage is named it can
generate an ordinary `Predicate("pipeline_stage", "eq", "OFFER")` — the same
generic representation, reached through the dimension concept.

**Do not** give Pipeline Stage its own filter path, and do not reshape the filter
design around it.

## 14. Blast radius

**Ownership relocation** (same meaning, moved upstream) — the overwhelming
majority:

| consumer | change |
|---|---|
| `question_interpretation/schema.py` | add a resolved population channel |
| `question_interpretation/projection.py` | populate it from `material_predicates` |
| `mi_agent_api/analytical_plan.py` | second `SELECT_POPULATION` kind + reader |
| `mi_agent_api/chat_routing.py` (`_route_evolution`) | consume the plan instead of `spec.filters` |
| `mi_agent_api/evolution.py` | per-snapshot `apply_population` |
| `mi_agent/population.py` | **none** — reused as-is |
| `mi_agent/execution_receipt.py` | **none** — already reads `material_predicates` |
| `mi_agent/mi_query_executor._apply_filters` | **none** — point-in-time path unchanged |

**Semantic change:** none intended. The one risk is that `apply_population` and
`_apply_filters` must agree; `apply_population`'s docstring says it reuses the
executor's comparison semantics precisely so they cannot diverge, and that claim
must be *measured*, not trusted.

## 15. Prerequisite cost — ranges, not false precision

Canonical unit: raw added + deleted production lines.

| item | raw lines |
|---|---|
| contract: resolved population channel | 30–50 |
| projection binding | 20–35 |
| plan primitive (second kind + reader) | 25–45 |
| executor adapter (per-snapshot) | 15–30 |
| duplicate-owner removal in `evolution` | 20–40 |
| **total prerequisite** | **110–200** |
| tests | 150–250, separate |

No C6 thresholds derived — the matrix is not green yet.

## 16. Refreshed C6 filter dependency

```
representation     : WOULD BE SOLVED  - Predicate already carries field+op+value;
                                        the contract must carry it (not invent it)
owner agreement    : MEASURABLE       - the shipped resolver is the truth, and the
                                        instrument built here observes it directly
plan consumable    : WOULD BE SOLVED  - second SELECT_POPULATION kind + apply_population
delivered coverage : PASS             - four cases, real per-period movement
```

The first three stay **amber until implemented and proved**.

## 17. Implementation sequence

1. Add the resolved population channel to the contract (additive, no consumer).
2. Populate it in `projection` from `material_predicates`; prove 121/121 agreement
   against the shipped resolver with the instrument built here.
3. Add the second `SELECT_POPULATION` kind and its reader; prove
   `apply_population` ≡ `_apply_filters` per period on the four delivered cases.
4. Switch `_route_evolution` to the plan; prove zero blast on 882.
5. Remove the route-local filter machinery only once 1–4 are green.

Steps 1–3 are additive and independently provable; only step 4 moves behaviour.
