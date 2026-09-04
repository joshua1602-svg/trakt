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
| `migration_phase0/live_shape_probe.py`, base vs HEAD (24 questions) | **9 FIXED, 0 REGRESSED**, 10 unchanged-ok, 5 still failing — all five the must-refuse set |
| `mi_agent/tests` + `question_interpretation/tests` failure node sets | 28 → 21, no new failure |
| `mi_agent_api/tests` + 17 routing files in `tests/` | 66 → 48, no new failure |

`must_refuse_both_arms.py` **could not run**: it needs `/tmp/cfo_env`, an
ephemeral fixture no committed script rebuilds. Its three questions are in
`live_shape_probe` instead, and all three still refuse.

The full 7,988-test suite does not complete in this environment's window and was
not run. Every blast claim above is scoped to the files named.

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

* **`limit` is claimed by no owner, so a risk question refuses.** "What is the
  largest geographic concentration versus limit?" reaches `risk_limits` and is
  then refused with `unknown category: 'concentration versus limit'`, because
  `_claimed_by_an_owner` recognises "concentration" and "versus" and not
  "limit" — the risk-limit vocabulary is not one of the owners it consults.
  Measured on base and unchanged by any of this work; it is the next cheap win
  in that area.

* **Two grouping axes are fine, and are pinned so this is not re-opened.**
  "Balance by region by broker" answers as a 5x3 heatmap — either order, and
  with "and" as well as a second "by". Measured on the base commit too, so it
  never depended on any of this work.
* **`mi_agent_api/tests` is order-dependent elsewhere too.** Several modules set
  and pop `MI_AGENT_PIPELINE_ROOT` globally in setUp/tearDown.
  `test_stage_movement_query` was fixed by re-asserting its environment per ask
  (15 nodes recovered); the same pattern is still live in its neighbours.
