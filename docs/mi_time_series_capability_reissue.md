# Time-series capability — REISSUED, rated from artifact contents

**This supersedes the table in `docs/mi_time_series_capability_report.md`.**
That report is not withdrawn — its measurements were correct for the instrument
and the product version it ran against. It is superseded because both have since
moved.

> **The previously published ratings were FLOORS set by the instrument, not
> ceilings set by the product.** A shape rated ABSENT meant *"the surface could
> not see it"*. That is a strictly weaker claim than *"the product cannot do
> it"*, and on two shapes the difference is the whole rating.

**58 runs — 29 phrasings × 2 books**, deterministic arm, through
`execute_governed_mi_query` with routing as shipped. Re-rated by
`question_interpretation/mi_capability_recontent.py`. **The LLM arm was also run
on both books and returns identical ratings**, so nothing below depends on which
parser answers.

---

## Why the earlier table understated what ships

`time_series_surface` decides both limbs of every rating by matching **column
names**:

* a time axis is a column whose *name* contains `period`, `month`, `quarter`, …;
* a requested dimension is a column whose *name* contains `region`, `seasoning`,
  `source_portfolio`, ….

Neither is what the rating rule says. A time axis is also proven by **a column
pair naming the two ends of a movement** (`prior`/`current`, `start`/`end`,
`opening`/`closing`, `previous`/`latest`), and a dimension is its **values**, not
its column name. Three artifacts carry both limbs under names that match neither
list, and were rated ABSENT with the reason *"neither a time axis nor the
requested breakdown"* — wrong on both counts.

This reissue resolves dimension domains **from the dataset itself**
(`collateral_geography` → the UK regions; `source_portfolio_type` →
direct/acquired; `seasoning_segment` → front/back book) and asks whether any
artifact column *holds those values*, whatever it is called.

**A second reason the old table is stale:** it predates P0. It records *11
answered, 18 refused, 3 silent drops*. Post-P0 the three silent drops are
refusals, giving *8 answered, 21 refused, **0 silent drops*** — the same 29 runs,
with the dangerous cases converted to honest ones.

---

## The reissued table

Identical on **both books** and on **both arms**.

| # | capability | was | **now** | what the artifact carries |
|---|---|---|---|---|
| 1 | metric × time | PROVEN | **PROVEN** | 4 of 4. `period` with 3 distinct points. |
| 2 | metric × time × filter | PARTIAL | **PARTIAL** | 1 of 4. Only a seasoning-population scope works; all three numeric/threshold filters refuse. |
| 3 | metric × time × dimension | ABSENT | **ABSENT** | 0 of 4. All refused, naming the loss. |
| 4 | metric × time × dimension × filter | ABSENT | **ABSENT** | 0 of 3. All refused. |
| 5 | metric × time × two dimensions | ABSENT | **ABSENT** | 0 of 3. All refused (post-P0; two were silent drops before). |
| 6 | period-over-period movement by segment | ABSENT | **ABSENT** | 0 of 4. Routes to `period_change_analysis`; the segment is not carried. |
| 7 | ranked historical movement | ABSENT | **PARTIAL** | **1 of 4.** `Which region has grown fastest?` returns rank + `category` (the 12 UK regions) + `start_value`/`end_value`. |
| 8 | comparison of two historical segments | ABSENT | **PARTIAL** | **2 of 3.** Both `front/back book` and `direct/acquired` return `population` carrying the two cohorts with `prior`/`current`. |

**Two shapes move ABSENT → PARTIAL. Nothing moves down.**

The old report's line for shape 7 — *"No artifact combines a ranking with a time
axis"* — is false as published. The artifact does exactly that; the surface could
not see it because the ranking's dimension column is called `category` and its
period ends are called `start_value`/`end_value`.

---

## The three ratings that changed, with the evidence

### T7 — `Which region has grown fastest?`

```
columns      : rank, category, start_value, end_value, movement,
               percent_movement, presence
time axis    : movement pair (start_value, end_value)
category     : 12 distinct -> East Midlands, East of England, London,
                              North East, North West, Scotland, ...
```

A ranked movement of balance **by region across two period ends** — T7's
definition, delivered. `category` holds the region domain resolved from
`collateral_geography`.

### T8 — `How has the front book moved over time compared with the back book?`

```
columns      : measure, population, period, prior, current, change
time axis    : movement pair (prior, current)
population   : 2 distinct -> Front Book (0-12 months), Back Book (13+ months)
```

### T8 — `Compare balance over time for direct and acquired`

```
population   : 2 distinct -> Direct, Acquired
```

> Across 2026-04-30 → 2026-06-30, **Direct**, 7,126 loans: Current Outstanding
> Balance £1.36bn → £1.39bn (+£21.5m). Across 2026-04-30 → 2026-06-30,
> **Acquired**, 3,909 loans: £568.3m → £579.4m (+£11.1m).

Both cohorts, both period ends, the movement between them. The domain match is
tolerant of presentation labelling in one direction only: the book stores
`front book`, the artifact renders `Front Book (0-12 months)`, and that counts —
while an unrelated column holding two values does not. A test pins both halves.

---

## What did NOT change, and why that matters

**Shapes 3, 4, 5 and 6 remain ABSENT on contents, not on column names.** Every
phrasing of all four refuses — 14 phrasings, no exceptions. These are not
measurement artefacts; the product genuinely does not carry a per-period
breakdown by region, by LTV band, by two dimensions, or a period-over-period
movement by segment. Across all 29, **21 refuse and name what was lost** — the
honour-or-clarify contract working.

**No silent drops on either book.** Every absence above is disclosed.

---

## Client-facing consequence

The summary that says *ranked historical movement* and *comparison of two
historical segments* are absent understates what ships. Both are **partial**:
regional growth ranking answers, and cohort comparison answers for both the
seasoning split and the direct/acquired split. What is genuinely absent is the
**per-period breakdown family** (shapes 3–6), which is a coherent and
straightforward thing to say.

**Reachability of the remaining absences is a separate question with a separate
answer** — see `docs/mi_phrasing_reachability.md`. Three of them are reachable
today under a different wording, which changes what P1 is building.

---

## Reproducing

The two environment traps that silently rate everything ABSENT are recorded in
`docs/mi_measurement_environment_traps.md`. The instrument now **refuses to
measure** when either is present rather than producing a clean, quotable, wrong
table.

```
TRAKT_RUNTIME_MODE=development \
  python -m question_interpretation.mi_capability_recontent --book alderbridge
TRAKT_RUNTIME_MODE=development \
  python -m question_interpretation.mi_capability_recontent --book kestrelmoor
```

`tests/test_mi_capability_recontent.py` — 25 tests. Mutating the matcher to
credit any column with two values breaks three of them; collapsing the
same-request grouping to same-shape breaks a fourth.
