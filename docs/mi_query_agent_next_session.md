# MI Query Agent — next session brief (from 2026-09-03)

Everything below was established against the **live deployed build** and a
115-question replay of what real users actually asked. Nothing here is
inferred from reading code alone — four diagnoses made that way on 2026-09-03
were wrong, and each was caught only by live evidence.

## Where things stand

Baseline: **83 of 115 questions answered**, measured by
`migration_phase0/replay_probe.py --from-log queries.json`. The corpus holds
heavy paraphrase — roughly 37 distinct analytics — so the count understates
capability: about 30 of 37 analytics work.

Of the 32 failures: ~11 genuine defects, 4 vocabulary gaps, 13 correct
governance refusals, 3 the region filter defect, 1 non-determinism.

## The three things to fix, in order

### 1. A value carried by several aliased fields resolves to nothing

**Reproduction, deterministic, no fixtures needed:**

```python
from mi_agent.categorical_spans import value_field
value_field("london", {"collateral_geography": ["London"]})            # ('collateral_geography', 'London')
value_field("london", {"collateral_geography": ["London"],
                       "geographic_region_obligor": ["London"]})       # None   <-- the defect
```

`value_field` returns `None` when more than one field claims a value, and its
docstring gives the reason: *"an ambiguous narrowing must be disclosed, never
resolved by preference."* That rule is right — it stops a product type
("lump sum") being bound to geography.

But this tape carries identical content in `collateral_geography` and
`geographic_region_obligor`. Two claimants, so every region FILTER resolves to
nothing and reports `unknown category: 'london'`. Grouping is unaffected
because it picks a field by preference and never binds a value — which is
exactly why "Show balance by region." answers while "balance in London" does
not, and why the `direct` book fails despite a perfectly clean vocabulary.

**This predates the 2026-09-03 region work** (two claimants already; the
registry change made it four).

**The seam:** every region field declares `value_domain: uk_region`. Hits that
all share one `value_domain` are ALIASES of one concept, not competing
concepts — resolve to the preferred one. Hits spanning different domains stay
ambiguous and are disclosed, so the "lump sum" protection is untouched.

**Care needed:** `value_field(value, available_values)` cannot see
`value_domain` today. Passing semantics in is a signature change on a shared
owner with several callers, and this function sits under EVERY categorical
filter, not just region. Read the callers before changing the signature.

Fixes 3 failures directly, and removes a whole class of filter failure.

### 2. A calendar time expression steals stage questions from the route that can answer them

```
"...moved into Offer in the last reporting period"  -> pipeline_stage_movement -> ANSWERED
"...moved into Offer stage in the last month"       -> temporal_compare        -> refused
"...moved into Offer in the last week?"             -> temporal_compare        -> refused
```

`pipeline_stage_movement` narrows by stage and answers 45 questions.
`temporal_compare` compares a whole-population metric across two periods and
CANNOT narrow to a stage, so the receipt correctly refuses with "stage — this
answer covers the whole population".

**It is a routing-precedence defect, not a capability gap.** The analytic
exists and works; a relative time expression outranks the stage recogniser and
hands the question to a route that cannot honour it.

Covers the NARROWING_LOST (6) and UNCONFIRMED (3) clusters — 9 failures, the
largest group.

**Do not fix this by teaching `temporal_compare` to narrow.** The question
belongs to the route that already answers it; the fix is which route claims it.

### 3. Non-determinism

"How many cases left KFI in the last week" ANSWERED in one replay and failed
the language-understanding step in the next, on byte-identical deployed code.
For a client-facing assessment this matters more than the count, because it is
the failure a reader cannot reason about or work around.

## What was done on 2026-09-03 and is already live

- A bucket name is not a threshold (`60-70% LTV bucket` answers).
- The comparison period defaults and DISCLOSES, in the `metric_defaulted`
  shape; the bare case still refuses and `must_refuse_both_arms.py` passes
  unchanged.
- `temporal_compare` declares the grain it ran at; a weekly series answering a
  monthly question delivers with the grain disclosed rather than refusing.
- The harmonised region columns are registered and preferred WHERE PRESENT,
  and the funded bridge knows them.
- `clean()` spells an ampersand as "and"; three approved synonyms take both
  books to 100% governed region vocabulary.

## Tools built, and what each refuses to do

- `migration_phase0/replay_probe.py` — replays the telemetry corpus and
  compares each question against its prior outcome. Stops on a 401 rather than
  scoring authentication failures as regressions.
- `migration_phase0/alignment_probe.py` — tests two hypotheses about the books
  and can EXONERATE either.
- `migration_phase0/region_vocabulary_audit.py` — how much of a book's
  geography is governed. Exits non-zero while anything is ungoverned; refuses
  to run against a copy of the rule it cannot prove is the runtime's.

## Discipline that earned its place

1. **Read the interface before writing against it.** Every defect introduced
   on 2026-09-03 came from asserting a field or function name from memory:
   `governance.errorCode`, `start_reporting_date`, `data_source.active_frame`,
   `\b` before a timestamp's "T". Everything read first held up.
2. **An empty result is not a clean result.** Three "zero blast" diffs that
   day were empty files from dead runs. Check for a summary line before
   believing a count, and compare two-way with explicit counts.
3. **Falsify every test against the unfixed code.** A test that passes both
   ways proves nothing.
4. **Diff failure SETS, not counts** — and confirm both runs covered the same
   files.
5. **Fix upstream, never fit the agent over bad data.** The MI layer's job is
   to refuse to launder a defect, not to tolerate it.
6. The full `tests/` suite (7,988 tests) does NOT complete in the web
   environment's window. Scope every blast claim, or use CI.

## Open, lower priority

- 4 UNMAPPED phrasings with working equivalents — cheapest wins available.
- No data cut-off is surfaced anywhere: answers say "as at 2026-06-30" while
  funded data was last updated 2025-11-30. Not blocking, but the platform
  asserts a currency it cannot evidence.
- Two independent geography systems: the React map aggregates ITL3 areas from
  postcodes; MI queries use ITL1 regions from the taxonomy. Nothing reconciles
  them.
- `tests/test_business_semantics_registry.py::test_committed_registry_matches_regeneration`
  fails on `main` and predates this work.

---

# Second session, same day — what was done and what remains

Three commits on `claude/mi-query-agent-defects-27s4ys`, one per defect above,
each with its failing test written first and falsified against the unfixed code.

## 1. The aliased value — FIXED

`categorical_spans.preferred_field` is the rule: several fields that declare one
`value_domain` are ALIASES of one concept and resolve to the field the GROUPING
owner already prefers (`llm_query_parser.domain_field_preference`, which returns
`_REGION_PREFERENCE` for `uk_region`). Hits spanning different domains, and any
field declaring no domain, stay ambiguous and are disclosed — the "lump sum"
protection is untouched and is pinned two ways.

`value_field` grew a third parameter, `semantics`. All nine call sites were read
and pass the registry they already hold; omitting it keeps the strict rule.
`concept_proposal.vocabulary` was counting claimants itself — a second copy of
the same rule — and now asks `preferred_field`, or it would have withheld as
ambiguous the very terms the parser binds.

## 2. The stolen stage question — FIXED

`temporal_compare` yields a sentence carrying a governed stage-movement
construction (`stage_movement_query.names_a_stage_movement`, the same reading
its own recogniser uses). It was NOT taught to narrow, and that boundary is
pinned: "Compare KFI balance this month vs last month" NAMES a stage without
putting it in motion, still reaches `temporal_compare`, and is still refused.

## 3. The non-determinism — DIAGNOSED, NOT FIXED

See `docs/mi_query_non_determinism.md`. One outbound model call decides it: the
concept-merge arm reports `proposal_unavailable` on any exception, and
`_enforce_model_availability` converts an otherwise-successful envelope into the
language-understanding refusal. The deterministic reading does not vary.

Second finding, on the RECORD: that refusal is coded `CALCULATION_FAILED` /
`capability` / `retryable: false`, which both the telemetry and `replay_probe`
count as an ERROR — so a transient model outage is recorded as the system having
broken, and the code contradicts the sentence ("Please try again"). Fixing it
needs a governed decision: no existing error code means "an upstream model was
unavailable; ask again", and code values are part of the external contract.

## THE VERIFICATION THAT COULD NOT BE RUN, AND WHAT WAS RUN INSTEAD

`replay_probe.py --from-log queries.json` **was not run**. Three separate
blockers, none of them worked around:

* `queries.json` is not in the tree — it is a saved `/ops/mi-queries` response.
* No `MI_BEARER` was available.
* This environment's network policy denies `app.traktinfra.io` outright
  (the proxy answers 403 to CONNECT). The deployed build cannot be reached at
  all from here.

So the **83/115 baseline was neither reproduced nor beaten, and nothing here
claims it was.** What was measured instead, all offline, all reproducible from
this tree:

| measurement | result |
|---|---|
| `scripts/run_mi_query_stage_movement_banks.py`, base vs HEAD, per question | **0 of 215 moved** (166-question bank, stage bank 36/36, near-neighbours 13/13) |
| the same three banks on a LIVE-SHAPED book (second region column added) | **0 of 215 moved** |
| `migration_phase0/live_shape_probe.py`, base vs HEAD (25 questions) | **10 FIXED, 0 REGRESSED**, 10 unchanged-ok, 5 still failing — all five the must-refuse set |
| `mi_agent/tests` + `question_interpretation/tests` failure node sets | 28 → 21, no new failure |
| `mi_agent_api/tests` + 17 routing files in `tests/` | 66 → 48, no new failure |

`must_refuse_both_arms.py` **could not run**: it needs `/tmp/cfo_env`, an
ephemeral fixture no committed script rebuilds. Its three questions are in
`live_shape_probe` instead, and all three still refuse.

The full 7,988-test suite does not complete in this environment's window and was
not run. Every blast claim above is scoped to the files named.

## The replay, against the deployed build (2026-09-04)

PR #398 merged and deployed (`deploy-mi-api.yml` is `workflow_dispatch` only —
merging does NOT ship the MI API), then `replay_probe.py` re-run over SSH
against the 115-question telemetry corpus:

| | baseline | after |
|---|---|---|
| answered | 83 / 115 | **87 / 115** |
| regressed | 2 | **1** |
| still failing | 30 | 27 |

FIXED 8, UNCHANGED_OK 77, WAS_MIXED 2. `pipeline_stage_movement` answered 49 of
49. No model calls occurred on any question (`metadata.llm` shows `calls: 0`
wherever it appears), and every parse was `deterministic`.

**The one regression was not the diff, and is now fixed.** "Where was the
greatest pipeline attrition?" came back *"parsed dimension(s) neither applied
nor rejected: pipeline_stage"*. The parse did not move — the same question at
both commits over 48 column/value combinations gives byte-identical specs, and
neither builds that shape. The CONCEPT-MERGE ARM proposed `pipeline stage` as a
dimension and `_apply_to_spec` filled the empty slot on a **loan-level** spec,
which has no group columns for an axis to land in. Reproduced with the arm
stubbed, offline.

`OperationProfile.accepts_grouping_axis` was already the rule and
`_AGGREGATIONS_WITHOUT_AN_AXIS` was one entry short: `loan_level` now sits
beside `share`, held to the same measured bar — across the 882-question corpus
the deterministic parser builds `loan_level` 29 times and carries a dimension in
NONE of them.

**So the arm IS live in production**, and this is its second measured instance of
changing what an answer is rather than whether there is one — the first being
the language-understanding refusal in `docs/mi_query_non_determinism.md`. Worth
deciding deliberately whether it should be on at all.

## Three words the reader owns (2026-09-04, from the product owner)

Definitions given, and where each was fixed — none by widening the metric-residue
guard, which exists to stop "show me the unicorn ratio by region" answering as
balance by region and still does:

| word | means | fixed at |
|---|---|---|
| `funded` | the PORTFOLIO, i.e. not the pipeline | `_ANALYTICAL_FRAMING_WORDS`, beside its twin `pipeline` |
| `withdrawals` | the pipeline stage WITHDRAWN | `pipeline_prep._STAGE_CANON`, the ONE map the question vocabulary is derived from |
| `amount` | defaults to current outstanding balance, count as fallback | a governed default with `metric_defaulted` disclosure |

A contract test caught the first attempt at `withdrawals`: the question-side
vocabulary must be a SUBSET of `_STAGE_CANON`, so extending the derived
vocabulary is forbidden and the authoritative map is where a spelling goes. The
singular `withdrawal` is then dropped from the question vocabulary by the
fragment rule (it is a prefix of the plural) — the same rule that stops
`complete` becoming a COMPLETED stage — and that is pinned rather than worked
around.

`_metric_side_residue` now also asks `pipeline_stage_vocabulary`, on the
principle it already applies to the book's own values: a word a governed owner
claims is not a measure this dataset lacks. This reaches the case book values
cannot — a stage the loaded frame carries no column for.

## The region double-bind — CLOSED on the live build; the journey is audited

At `3acd6d0` three region questions carried TWO filters for one region and
refused "'Region' is not available in this dataset". On `df0a1c5`, asked live,
all three answer with exactly one filter and nothing not-applied, the Acquired
lens applying correctly alongside the region on the third.

**The frame did NOT move** — the portfolio owner confirms 958 loans (885
acquired, 73 direct), unchanged. The 640-loan/30-June book this document
previously cited as the comparison is from `MI_FINAL_LIVE_DATA_READINESS.json`
and describes a different environment; reading it as this portfolio was an
error. So the cause remains unattributed: two of the three questions contain
`funded`, which #400 stopped reading as a missing measure, but the third
contains nothing #399 or #400 touched.

Because the symptom is gone and the journey is not, see
**`docs/mi_query_region_end_to_end_audit.md`** — five owners, three field
families, one word — and `tests/test_region_topology.py`, which pins the
topology so any of them moving is loud. The material finding: **Risk Limits
evaluates geographic concentration on `geographic_region_obligor` (NUTS3) while
MI answers on `canonical_region_reporting`.** Two dashboard surfaces, two
groupings of one book, nothing reconciling them.

## Newly measured, still open

* **A restriction on the axis being grouped — FIXED.** "Show balance by region
  for loans in Wales and Scotland." now answers with two rows, and so does the
  single-region form with one. Three separate things were wrong and each was
  fixed at its own owner:

  - `_grouping_segments` split EVERY "and" as an axis separator, so the axes
    read as ["region for loans in wales", "scotland"]. An "and" inside a
    segment's own qualifier coordinates VALUES. `_AXIS_QUALIFIER_RE` is now the
    one owner of that boundary, shared with the reader below.
  - Only the first value bound. The clause splitter leaves a bare " scotland",
    and a clause that is nothing but a governed value resolved to nothing.
    `_whole_clause_value` reads it and `_with_value` widens the field's
    condition to `{"op": "in", ...}` — the shape the executor and the
    drill-through already use. This also fixed the same loss one axis over:
    "by broker for loans in Wales and Scotland" bound Wales and dropped
    Scotland (the gate caught it, so nothing wrong was published).
  - `_grouped_value_filters` dropped any filter whose field was the grouping
    dimension. It now keeps one that RESTRICTS the axis, and only when the
    values are the book's own and are not the axis phrase's own words —
    "show balance by lump sum" still drops, which is the case that rule exists
    for.
  - `execution_receipt._filter_values` could not read a condition dict, so the
    geographic-scope facet could not see a narrowing the receipt itself had
    already described as "Region in Wales, Scotland". `not_in`/`ne` are
    deliberately excluded: their operands are what an answer leaves out.

  The single-value case, recorded yesterday as defensible to refuse, now
  answers. It is not defensible to refuse it while answering the two-value
  one, and a reader who wants the figure alone can ask for it directly.

* **The place-resolver fallback — FIXED.** With no value catalogue,
  `_parse_categorical_filter` bound ANY captured phrase to a region field in
  title case. The 882-question corpus carried five: `collateral_geography` =
  'Concentration Versus Limit', 'Equity Release Supermarket Limited' (twice — a
  BROKER), 'October' (a MONTH) and 'Weighting'. It was invisible until the
  grouped-filter rule above was narrowed to the case it was written for.

  `region_resolution.looks_like_region_term` — the governed ITL ladder the
  canonical transformation itself uses, written for exactly this question and
  called from nowhere — now decides. A term it knows still binds with no
  catalogue, so every real place keeps working; a term it does not know is
  recorded as an unresolved narrowing instead, using the SAME note the
  catalogued ending writes, because a warning would not refuse and removing the
  invented binding would otherwise have answered over the whole book for the
  first time.

  No served answer moves: serving always has a frame, so the catalogued ending
  owns these questions, and all five were checked through the app on base and
  HEAD — identical. `filter_ownership_trace` goes 119 → 115 of 882, and its
  assertion in `tests/test_assurance_measurement_failure.py` was updated with
  the four losses named.

* **The `limit` noun, and the category that went with it — FIXED.** "What is
  the largest geographic concentration versus limit?" was refused with
  `unknown category: 'concentration versus limit'`. The question names no
  category at all: `_CATEGORICAL_FILTER_RE` reads "geographic X" as "the place
  X", and `_claimed_by_an_owner` — the guard that stops an unclaimed candidate
  being RECORDED as a category the book lacks — claimed "concentration"
  (analytical framing) and "versus" (a grouping marker) and not "limit".

  `_RISK_LIMIT_NOUNS` is the risk-limit vocabulary's own nouns, read only by
  that guard, with every entry asserted to appear in `_RISK_LIMIT_RE` so the two
  cannot drift. Deliberately NOT added to `_ANALYTICAL_FRAMING_WORDS`, where
  "limit" reads like kin to "concentration" and "exposure": a word in that set
  is not metric residue, so "Show the limit by region" would stop refusing with
  *"'limit' is not a governed measure in this dataset"* and answer with a
  balance breakdown. That refusal is pinned.

  Removing the refusal exposed a second defect underneath it, fixed with it
  because the first makes it reachable: `_RISK_LIMIT_RE` never matches this
  sentence, so the ANALYTICAL INTENT BOUNDARY claims it — and the boundary set
  `risk_limit_query` while leaving `risk_limit_category` open. The route
  answered every limit category: *"5 passed … Nearest to limit: Top 3 brokers"*
  for a question about geography. The boundary now settles the category from the
  parser's own reader (`llm_query_parser.risk_limit_category`), never
  overriding a settled parse, and the answer is scoped: *"geographic
  concentration: 2 passed, 0 warning(s), 2 breach(es) … Nearest to limit:
  Scotland"*. `_route_risk` already narrowed to the category and already
  refuses honestly when one has no configured tests; nothing there changed.

* **Two grouping axes are fine, and are pinned so this is not re-opened.**
  "Balance by region by broker" answers as a 5x3 heatmap — either order, and
  with "and" as well as a second "by". Measured on the base commit too, so it
  never depended on any of this work.
* **`mi_agent_api/tests` is order-dependent elsewhere too.** Several modules set
  and pop `MI_AGENT_PIPELINE_ROOT` globally in setUp/tearDown.
  `test_stage_movement_query` was fixed by re-asserting its environment per ask
  (15 nodes recovered); the same pattern is still live in its neighbours.
## Region lineage and go-live defensibility — 2026-09-04

Two surfaces answer "region". Until today neither said which region it meant,
and one of them was measuring the wrong one.

### 1. The limit was tested on the raw column, and read a breach as compliant

`engine.region_taxonomy` harmonises every book onto `canonical_region_reporting`
and records, per row, how each raw value was mapped (`region_mapping_method` =
exact / synonym / unresolved / absent). The MI Query Agent has preferred that
column for every region question since the aliasing work. The RISK-LIMIT
evaluator never learned about it: `risk_limits._REGION_COLUMNS` began at the raw
`collateral_geography`.

Reproduced in `mi_agent_api/tests/test_a_limit_is_tested_on_the_governed_region.py`
against the unfixed code — a tape spelling one region three ways:

    "South West" · "south-west" · "SOUTH WEST"        3 x 25% of the book

    limit  "South West must not exceed 40%"
    before actual 25.00  status GREEN     <- 75% of the book, reported compliant
    after  actual 75.00  status RED

and the MI Agent, asked the same thing in words, already answered 75%. Two
dashboard surfaces, one word, contradictory numbers, with the limit erring in
the unsafe direction.

`_REGION_COLUMNS` now leads with the harmonised reporting family and keeps the
old order behind it, so a deployment with no taxonomy configured has no
harmonised column and behaves exactly as before (pinned by
`test_a_tape_with_no_harmonised_column_still_falls_back`).

### 2. Nothing read the provenance the taxonomy was written to record

`region_mapping_method` had zero consumers. An answer grouped by region was
computed from the rows that carried a governed value and said nothing about the
rest, and said nothing about WHICH of the three families produced it —
reporting, NUTS3, or ITL3, which cover different populations.

`mi_agent/region_basis.py` is the one reader. It decides nothing, resolves no
value and changes no figure; it answers two questions about an executed query:

    which region field did it measure, and at which level?
    what share of the frame carries a governed value at that level?

It now backs both surfaces:

* `ExecutionReceipt.region_basis` — published as `regionBasis` on every
  `execution_receipt`, and, when coverage is PARTIAL, said out loud on the
  receipt line the reader is shown. Full coverage stays silent: a caveat
  printed on every answer is a caveat nobody reads. Unknown coverage (no frame)
  is stated as unknown, never as complete.
* `risk_limits.region_basis_block(df)` — published as `regionBasis` on the
  limits envelope, with the same partial-coverage sentence carried on each
  geographic test, and each test's `dimensionKey` now recording the column its
  actual was ACTUALLY measured on rather than the Schedule 8 keyword hint.

### A correction to the 2026-09-03 audit

That audit recorded "Risk Limits hardcoded to NUTS3, no fallback", read from
`schedule8_extractor._CATEGORY_RULES`. That tuple is an extraction-time
DIMENSION HINT. The evaluator already fell back across three columns — it just
led with the raw one rather than the governed one, which is the defect above.
`tests/test_region_topology.py` records the corrected coupling.
