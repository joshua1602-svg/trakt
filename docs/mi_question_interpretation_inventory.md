# Where clause interpretation happens today — an inventory

Not a design. A map of what reads the raw question text, what each one decides,
and where they disagree.

| | |
|---|---|
| Tree | `28ece25` (`mi-query-agent-rc`), via `4e051f3` |
| Corpus | 602 natural-language questions — 252 calibration bank, 350 ERE golden library |
| Book | real funded tape, 11,035 loans, £1,964.89m |
| Reproduce | `python -m clause_splitting_phase1.run_interpreter_inventory` |

**Headline: there are not two interpreters. There are at least eleven distinct
entry points that read the raw question, across nine modules, and the two the
brief names are only the largest two.**

---

## 1. Every place that reads the raw question to decide a concept or a slot

A mechanical sweep — functions taking a question/text parameter and performing
lexical operations on it — finds **86 functions across 13 modules**. Most are
helpers inside a larger decision. The **distinct decision points** are these.

### The interpreters proper

| # | Entry point | What it decides | What it reads |
|---|---|---|---|
| 1 | `llm_query_parser._deterministic_parse` | the whole `MIQuerySpec` — metric, aggregation, dimensions, filters, ranking, forecast, bridge, cohort, risk | the raw question, plus the registry; 29 sub-functions, ~46 lexical ops in the entry function alone |
| 2 | `execution_receipt.detect_requested_facets` | every material facet stated in the question, "before any parsing decision" | the raw question, the registry, and the **frame** (for geographic values) |
| 3 | `answer_type.asked` | the answer TYPE — currency / count / rate / age / date / mixed / any — "from its wording alone" | the raw question only. Has **its own subject-side clause split** |
| 4 | `interpreter/deterministic.interpret` | a validated `MIQuerySpec v2` — a **second, separate parser** | the raw question, its own `_has` / `_any` token helpers |
| 5 | `period_request.requested_span` / `requested_unit` | the reporting window, and the finest time UNIT named | the raw question; own `_SPANS`, `_UNIT_PATTERNS`, `_VAGUE_RECENCY_RE` |
| 6 | `portfolio_lens.resolve_lens` / `resolve_comparison_lenses` / `names_total_scope` / `mentions_portfolio` | which source-portfolio scope the question names | the raw question |
| 7 | `population.fabricated_bounds` / `fabricated_concepts` / `_requests_concept` | which governed population concepts and numeric bounds the question actually stated | the raw question, checked against the spec |
| 8 | `mi_agent_workflow._detect_unsupported_concept` | whether the question names a concept no field supports | the raw question |
| 9 | `mi_agent_api/chat_routing` — 15 functions (`_is_portfolio_summary`, `_is_period_movement`, `_is_scenario`, `_scenario_multiplier`, `_is_geo_exposure`, `_is_conversion`, `_is_evolution`, `_is_aggregate_contribution_question`, `_prog_metric_key`, `_dataset_for`, …) | which route answers the question, and route-specific parameters | the raw question |
| 10 | `mi_agent_api/concentration_query.detect_intent` | whether this is a limit/concentration question and which test | the raw question |
| 11 | `period_change/recognition.recognise` | whether this is a Period Change Analysis question, and its interpretation | the raw question |

A twelfth, `concentration_tests/matching.extract_from_text`, reads covenant
documents rather than user questions and is out of scope here.

### The same clause split, implemented three times

The most direct duplication. All three decide *where the subject clause ends*,
and two of them say in comments that they are the same rule as another:

| Implementation | Openers | Method |
|---|---|---|
| `llm_query_parser._metric_slot` | `_FILTER_OPENER_RE` — 13 terms | truncate at first opener, only where a digit follows |
| `answer_type.subject_side` | `_CONDITION_OPENERS` — **the same 13 terms**, declared separately | split at `by` first, then truncate at first opener, only where a digit follows |
| `execution_receipt._is_filter_subject` | `_FILTER_AFTER_RE` / `_FILTER_BEFORE_RE` — **a different set** | 32-character windows either side of the measure word |

The first two vocabularies are **identical and duplicated rather than shared**:
`above, at least, at most, below, fewer than, for, greater than, less than,
more than, over, under, where, with`.

The third differs: it **adds** `between`, `exceeding`, `in excess of` and the
operators `< > =`, and **drops** `where`, `with`, `for`, `fewer than`. So the
facet layer will not recognise `where` as a filter opener in this test, and the
other two will not recognise `between`.

`answer_type.py`'s own comment states the intent: *"The same rule the parser
applies in `llm_query_parser._metric_slot`, and for the same reason."* It is the
same rule, written twice, from two vocabularies that are equal today by
coincidence of maintenance rather than by construction.

---

## 2. Where the two known interpreters agree and disagree

602 questions through both. The facet layer is read **before reconciliation**,
so this is interpreter against interpreter, not interpreter against execution.

### Headline counts

| Slot | Disagreements |
|---|---:|
| filter | **87** |
| grouping | **57** |
| answer type (`answer_type.asked` vs the type the parser's result would have) | **36** |

### Filter — 87, and most are not disagreements about the sentence

| Classification | n | Meaning |
|---|---:|---|
| both see the clause; only the parser resolves the field | **69** | **not a disagreement** — a division of labour |
| parser binds a filter, facet layer raises no filter facet | 14 | of these, **9 raise `grouping_dimension` instead** — a slot disagreement |
| different fields | 2 | |
| facet only | 2 | |

The 69 are the important row. **Every `threshold` facet carries
`field_key=None`.** For *how many loans have LTV above 50%* the facet layer
emits `threshold("LTV over 50")` with no field, while the parser resolves
`current_loan_to_value`.

So the facet layer already identifies **that a filter clause exists and what it
says**, and deliberately does not resolve **which field it binds**. That is the
domain-blind/domain-aware split, already present in the code, on 69 of 87 cases.

The two "different fields" cases show the same split from the other side:

> *How many South East loans have LTV above 50%* — parser
> `['current_loan_to_value']`, facet `['collateral_geography']`.

Neither is wrong. The parser resolved the threshold's field; the facet layer
resolved the geography. They are describing different clauses of the same
sentence and there is no shared structure in which both belong.

### Grouping — 57, and 45 are the facet layer seeing what the parser dropped

| Classification | n |
|---|---:|
| facet only | **45** |
| parser only | 7 |
| different fields | 5 |

Of the 45, **41 are cases where the parser assigns the dimension to neither
slot** — *pipeline by broker*, *show NNEG by region*, *missing region count*.
These are routed elsewhere and the parser drops the dimension entirely; the
facet layer records it because `requested_dimension_terms` resolves *without*
dataset-availability filtering, precisely so a dropped dimension can be
disclosed. That is the facet layer working as designed.

The other 4, and the 9 from the filter section, are one structural fact:

> **`KIND_GROUPING` is not a grouping claim. It is "a dimension the question
> named", with no grouping-versus-filter distinction.**

*how many joint borrowers are there* → parser: filter on `borrower_type`; facet:
`grouping_dimension(borrower_type)`. *balance by region for joint borrowers* →
parser: grouping `collateral_geography`, filter `borrower_type`; facet:
`grouping_dimension` for **both**, indistinguishable.

### Answer type — 36

| asked → result | n |
|---|---:|
| currency → count | 23 |
| rate → currency | 7 |
| rate → count | 5 |
| count → currency | 1 |

The 23 are dominated by pipeline questions — *pipeline amount by stage*,
*expected funded by stage* — where `answer_type.asked` reads currency from
"amount"/"funded" and the parser returns `metric=None, aggregation=count`.

### Questions raising no facet at all — 171

Not a gap. They are overwhelmingly whole-book KPIs — *what is our total funded
balance*, *loan count*, *how many mortgages do we have* — which state no
material facet to disclose. The facet layer is a **disclosure** mechanism: it
raises something only when there is something that could go missing.

That is also the reason it cannot serve as a complete interpretation on its own
today: **absence of a facet is not a statement about the sentence.**

---

## 3. Are `applied` / `unavailable` / `lost` already the span states?

**Close, and not the same thing.** The mapping is not one-to-one in either
direction.

| Facet status | Meaning | Span-model equivalent |
|---|---|---|
| `applied` | reached execution and demonstrably shaped the result | `filled` — **plus** an execution guarantee the span model does not have |
| `unavailable` | the dataset does not carry the field | `unresolvable`, with a reason |
| `unsupported` | no governed capability expresses it | `unresolvable`, with a reason |
| `rejected` | execution considered it and declined, with a reason | `unresolvable`, with a reason |
| `lost` | requested, material, absent from execution **with no reason** — fails closed | **no equivalent** |
| *(no facet raised)* | nothing detected | ambiguous between `empty` and "no detector covers this" |

Two asymmetries matter.

**`applied` is stronger than `filled`.** The comment on `KIND_POPULATION` states
the principle: *"Presence of the filter on the spec is NOT evidence it ran; only
execution evidence is."* It exists because twelve of thirteen specialist routes
ignored `spec.filters`, so a back-book question was answered across the whole
book with `ok=True` and a spec that still claimed the filter. A span asserted at
the top of the pipeline carries no such evidence.

**`lost` has no span equivalent**, for the same reason: it is defined by
comparison against execution, and a span is asserted before execution.

### What a span carries that a facet does not

Six things, all structural:

1. **Position and extent.** A facet has a `label` — a rendered phrase — not
   offsets into the sentence. Leftmost-outermost precedence and
   subject-as-residue are not computable from facets.
2. **A subject slot.** There is no facet kind for "the thing being measured".
   The measure appears only as a *problem*: `multi_measure`,
   `unresolved_measure`, `relationship`. A correctly-understood single measure
   raises nothing.
3. **Grouping versus filter for a named dimension.** One kind covers both, as
   §2 shows.
4. **A period-filter slot.** Of the 15 kinds, the only temporal one is
   `comparison_period`. *The last three months* has no facet kind.
5. **A target slot.** `threshold` covers a filter bound; `projection` marks a
   forward question. Nothing distinguishes *loans over £500k* from *when do we
   reach £500m*.
6. **Exhaustiveness.** Facets are raised by detectors, so unrecognised wording
   produces silence rather than a residue. There is no "everything else" channel.

### What a facet carries that a span does not

Worth recording, because it is not in the proposed design:

* `status` reconciled against **execution evidence**, and the `lost` fail-closed
  state built on it.
* `alt_keys` — legitimate alternative resolutions of the same request, so a
  re-resolution ("region" → readable geography or the NUTS3 code field) is never
  mistaken for a substitution.
* `reason` and `disclosure()` — user-facing text naming both the wording and the
  field, so a refusal can be acted on.

### The 15 kinds, against the six spans

| Span | Facet kinds covering it |
|---|---|
| operation | `ranking`, `requested_statistic`, `projection`, `aggregate_contribution`, `share` |
| subject | `multi_measure`, `unresolved_measure`, `relationship` — **problems only** |
| grouping | `grouping_dimension` — **conflated with filter** |
| filter | `geographic_scope`, `threshold` (fieldless), `row_population`, `stress_scenario` |
| period | `comparison_period` only — **no period filter** |
| target | **none** |

---

## 4. Could a time-axis grouping be carried by the facet layer as it stands?

**The reading already exists and is correct. It is the carriage that is missing.**

### The grain is already interpreted correctly today

`period_request.requested_unit` — *"The finest time UNIT the question names"* —
run over all 24 time-series probes:

| Question | `requested_unit` | parser `chart_type` | parser `trend_grain` | facet kinds |
|---|---|---|---|---|
| funded balance by month | **month** | line | `None` | — |
| funded balance by quarter | **quarter** | bar | `None` | — |
| how many loans each month | **month** | none | `None` | — |
| total exposure each quarter | **quarter** | none | `None` | — |
| average LTV by month | **month** | line | `None` | `requested_statistic` |
| funded balance over time by region | *(none)* | line | `None` | `grouping_dimension` |
| balance by month by broker | **month** | line | `None` | `grouping_dimension` |

`requested_unit` gets **every grain right**, including *by quarter* and *each
month*, which the parser does not recognise as a time axis at all. The facet
layer raises **no facet for the time axis in any of the 24** — where
`grouping_dimension` fires it is for the non-time axis (region, broker, arrears
bucket).

### The disclosure shape also already exists

`execution_receipt.granularity_disclosure` emits exactly the required object:

```python
RequestedFacet(kind=KIND_GROUPING, label=asked, status=UNAVAILABLE,
               reason=f"this answer is reported at {reported} level, not by {asked}")
```

That is a granularity a route could not honour, expressed as a grouping facet
with a disclosable status and a reason — the structure a *by week* question
against a month-end book needs. But `_ROUTE_GRANULARITY` contains **one entry**,
`{"geo_exposure": ("postcode", "ITL3 area")}` — geographic, not temporal — and
it is called from one place, `mi_agent_api/mi_service.py:541`.

`period_request` has the temporal twin already written — `finer_than` and
`granularity_clarification` — and it is called from **one** place,
`chat_routing.py:1020`, only to clarify when a request is finer than month.

### What would have to change

Stated as facts, not as a proposal:

1. **`requested_unit`'s reading would have to reach a facet.** It is currently
   consumed at one call site and discarded everywhere else.
2. **A time axis would need to be distinguishable from a dimension axis.**
   `KIND_GROUPING` carries a `field_key`; a time grain is not a registry field,
   so either it needs a distinct kind or `field_key` needs to admit a non-field
   axis.
3. **The grouping/filter conflation would have to be resolved first**, otherwise
   *balance by month over the last 6 months* — a time grouping and a time period
   filter in one sentence — cannot be represented at all.
4. **`trend_grain` is never set from the question.** The only assignment in the
   codebase is `interpreter/deterministic.py:153`, hard-coded to `"monthly"`.
   Whatever the facet layer concluded would still have nowhere to land on the
   spec.
5. **A facet would have to be raisable for a correct request, not only a
   problem.** Today a facet is a disclosure; a time axis that *can* be honoured
   raises nothing, so nothing downstream can act on it.

---

## Summary of the inventory

* **Eleven distinct entry points** read the raw question, not two.
* **The subject-side clause split is implemented three times**, from two
  identical-but-duplicated vocabularies and one different one.
* **The facet layer and the parser mostly do not disagree about the sentence.**
  69 of 87 filter disagreements are the facet layer identifying a clause and
  declining to resolve its field — a division of labour, not a conflict.
* **The one real structural conflict is `KIND_GROUPING`**, which conflates a
  grouping axis with a filter on a named dimension, and accounts for the 9
  filter and 4 grouping slot disagreements.
* **The facet states are close to span states but not identical.** `applied`
  carries an execution guarantee a span does not; `lost` has no span equivalent;
  "no facet" is ambiguous where `empty` is explicit.
* **A facet has no position, no subject slot, no period-filter slot and no
  target slot**, and no residue channel — 171 questions raise nothing at all.
* **The time grain is already read correctly** by `period_request`, and the
  disclosure shape already exists in `granularity_disclosure`. Neither reaches
  a grouping slot, and `trend_grain` is never set from the question.
