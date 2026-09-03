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
