# The Gate — LLM arm over the 29 time-series phrasings

**A measurement, not a stage.** No product code was changed. Nothing found here
was fixed. P1 was not begun.

Repo `joshua1602-svg/trakt`, branch `claude/clause-splitting-phase-1-cft1wx`.
**Base commit confirmed before anything below was written:** branch head
`285c4a1` at the time of the run, with `00fdef6` an ancestor — at or after the
required base.

Run in full: **29 phrasings x 2 books x 5 repeats, LLM parser ON**, against
Haiku 4.5. 145 LLM-arm runs per book, 290 in total, 130 counted model calls
per book.

---

## 1. The Gate's own question, answered flat

**The two arms agree everywhere. Both books, all 29 phrasings, every repeat.**

```
                                                            alderbridge  kestrelmoor
  same outcome                                                  29           29
  the model proposed something the route does not reach          0            0
  the guard refused a proposal the route would have handled      0            0
  NEITHER CATEGORY FITS                                          0            0

  self-disagreement across 5 repeats                       0 of 29      0 of 29
  silent drops under the LLM arm                                 0            0
  honest refusals                                        105 of 145   105 of 145
```

No divergence to attribute, in either direction, on either book. The two books
are identical row for row — same ratings, modal shares, distinct counts, call
counts and parser modes. `105 = 21 x 5`: the deterministic arm's 21-of-29
refusals, reproduced five times over.

**That is the result, and it is reported as flat rather than mined for a
difference to justify the run.** Turning the LLM parser on changes nothing on
this surface.

Shape ratings, unchanged by the arm and identical on both books:

| shape | | det | LLM |
|---|---|---|---|
| T1 | metric x time | PROVEN | PROVEN |
| T2 | metric x time x filter | PARTIAL | PARTIAL |
| T3 | metric x time x dimension | ABSENT | ABSENT |
| T4 | metric x time x dimension x filter | ABSENT | ABSENT |
| T5 | metric x time x two dimensions | ABSENT | ABSENT |
| T6 | period-over-period movement by segment | ABSENT | ABSENT |
| T7 | ranked historical movement | ABSENT | ABSENT |
| T8 | comparison of two historical segments | ABSENT | ABSENT |

**Section 5 is the reason those T7 and T8 rows should not be read at face
value.** It is the most consequential thing this run found, and it is not about
the LLM arm at all.

---

## 2. First — did the model actually run?

Asserted by **counting calls** at `mi_agent/llm_query_parser.py::_invoke`,
before any comparison above is trusted.

```
  total counted model calls        : 130   (per book)
  questions that reached the model : 26 of 29
```

**The arm observed something.** This is not the earlier "15 of 15 identical"
failure, where a harness read `parser_mode` (snake_case), got `None` on both
arms, and reported agreement with no evidence the model had run at all.

### 2.1 `parserMode` could not have established this, even read correctly

The metadata key is `parserMode` — camelCase, and this instrument reads it. That
is still not enough:

```
  counted call?   parserMode        runs
  -------------   ---------------   ----
  yes             None               115
  yes             deterministic       15
  no              None                15
```

**`parserMode = None` covers both cases.** It appears on all 15 runs where no
call happened *and* on 115 runs where a call did. The field cannot separate "the
model ran" from "the model did not", so no reading of it establishes that the
arm observed anything. Only the counter does.

The 15 runs stamped `deterministic` are the three T5 phrasings, and all 15 **did**
call the model; `parserModeDetail` is `deterministic_fallback` there. The model
was invoked, its proposal was validated, and the deterministic spec is what
executed — the documented mechanism, confirmed on this surface.

### 2.2 The three zero-call questions are a real result

Identical on both books:

* `How has balance changed since last month by region?`
* `month-on-month change in balance by region`
* `Which LTV band moved most between periods?`

`zero_cost_first` hands off only on validation failure, non-`high` confidence, or
a layered question. These three passed the gate outright, so the model was never
asked. **Not a bug and not a hole in the measurement** — the user reaches the
deterministic answer on these three regardless of the arm.

---

## 3. Per question

`calls` is per repeat; `distinct` is distinct outcomes across the 5 repeats — 1
everywhere, so there is no self-disagreement anywhere. **Identical on both
books.**

| shape | question | det | LLM | modal | distinct | calls | parserMode |
|---|---|---|---|---|---|---|---|
| T1 | Show me balance by month | PROVEN | PROVEN | 5/5 | 1 | 1 | None |
| T1 | balance over time | PROVEN | PROVEN | 5/5 | 1 | 1 | None |
| T1 | How has the funded balance moved over time? | PROVEN | PROVEN | 5/5 | 1 | 1 | None |
| T1 | total balance by reporting period | PROVEN | PROVEN | 5/5 | 1 | 1 | None |
| T2 | Show me balance by month for loans over £150k | ABSENT | ABSENT | 5/5 | 1 | 1 | None |
| T2 | balance over time for loans with LTV above 50% | ABSENT | ABSENT | 5/5 | 1 | 1 | None |
| T2 | How has balance moved over time for the front book? | PROVEN | PROVEN | 5/5 | 1 | 1 | None |
| T2 | balance by month where balance is above £150,000 | ABSENT | ABSENT | 5/5 | 1 | 1 | None |
| T3 | Show me balance by month by region | ABSENT | ABSENT | 5/5 | 1 | 1 | None |
| T3 | balance over time by region | ABSENT | ABSENT | 5/5 | 1 | 1 | None |
| T3 | How has balance moved over time, broken down by region? | ABSENT | ABSENT | 5/5 | 1 | 1 | None |
| T3 | balance by month split by LTV band | ABSENT | ABSENT | 5/5 | 1 | 1 | None |
| T4 | Show me balance by month by region for loans over £150k | ABSENT | ABSENT | 5/5 | 1 | 1 | None |
| T4 | balance over time by region for the front book | ABSENT | ABSENT | 5/5 | 1 | 1 | None |
| T4 | balance by month by LTV band for loans above £150,000 | ABSENT | ABSENT | 5/5 | 1 | 1 | None |
| T5 | Show me balance by month by region and LTV band | ABSENT | ABSENT | 5/5 | 1 | 1 | deterministic |
| T5 | balance over time by region and ticket size | ABSENT | ABSENT | 5/5 | 1 | 1 | deterministic |
| T5 | balance by month broken down by LTV band and region | ABSENT | ABSENT | 5/5 | 1 | 1 | deterministic |
| T6 | How has balance changed since last month by region? | ABSENT | ABSENT | 5/5 | 1 | **0** | None |
| T6 | What moved between periods by region? | ABSENT | ABSENT | 5/5 | 1 | 1 | None |
| T6 | month-on-month change in balance by region | ABSENT | ABSENT | 5/5 | 1 | **0** | None |
| T6 | period on period movement by LTV band | ABSENT | ABSENT | 5/5 | 1 | 1 | None |
| T7 | Which region grew the most over the last three months? | ABSENT | ABSENT | 5/5 | 1 | 1 | None |
| T7 | Which LTV band moved most between periods? | ABSENT | ABSENT | 5/5 | 1 | **0** | None |
| T7 | Rank regions by balance growth over time | ABSENT | ABSENT | 5/5 | 1 | 1 | None |
| T7 | Which region has grown fastest? | ABSENT* | ABSENT* | 5/5 | 1 | 1 | None |
| T8 | How has the front book moved over time compared with the back book? | ABSENT* | ABSENT* | 5/5 | 1 | 1 | None |
| T8 | Compare balance over time for direct and acquired | ABSENT* | ABSENT* | 5/5 | 1 | 1 | None |
| T8 | How have direct and acquired balances moved over the periods? | ABSENT | ABSENT | 5/5 | 1 | 1 | None |

Eight of the 29 answer rather than refuse. Five are PROVEN. **The three marked
`ABSENT*` answer, and their ABSENT rating does not survive inspection of the
artifact — section 5.**

---

## 4. The three P0 refusals — by name, on both books

| question | limb | alderbridge | kestrelmoor |
|---|---|---|---|
| `Show me balance by month by region and LTV band` | time axis | **HELD** — refused 5/5, 0 silent drops | **HELD** — refused 5/5, 0 silent drops |
| `balance by month broken down by LTV band and region` | time axis | **HELD** — refused 5/5, 0 silent drops | **HELD** — refused 5/5, 0 silent drops |
| `How have direct and acquired balances moved over the periods?` | segments | **HELD** — refused 5/5, 0 silent drops | **HELD** — refused 5/5, 0 silent drops |

**All three hold under the LLM arm, on both books, in every repeat**, each
refusing with the honour-or-clarify message naming the limb it could not apply.

The first two are among the 15 runs where the model was called and
`deterministic_fallback` was stamped: **the model was asked, proposed something,
and the refusal held anyway.** That is the P0 guard doing what it exists to do,
measured rather than assumed.

The third one holds, and section 5.2 is why that is not straightforwardly good
news.

---

## 5. The finding: three ABSENT ratings that the artifact contradicts

This is not about the LLM arm. It affects **both arms identically**, which is
why section 1 is unaffected by it — but it changes what the T7 and T8 rows mean.

The brief's rating rule is explicit:

> A time axis is proven by the rendered rows: a time column with more than one
> distinct value, **or a column pair naming the two ends of a movement**
> (`prior`/`current`, `start`/`end`, `opening`/`closing`, `previous`/`latest`).

`time_series_surface.inspect_artifacts` implements only the first form. Its
`_TIME_HINTS` list matches column *names* (`period`, `month`, `quarter`, …) and
has no notion of the movement pair. Requested dimensions are matched the same
way, by name fragment (`region`, `geograph`, `ltv_bucket`, `seasoning`,
`source_portfolio`, `provenance`).

**Three artifacts carry both limbs under column names that neither list
matches**, and are rated ABSENT with the reason *"neither a time axis nor the
requested breakdown"* — a statement that is wrong on both counts:

### 5.1 `Which region has grown fastest?` (T7)

The response carries two tables, and **both** name the two ends of a movement:

```
the table the surface rated (11 rows):
  rank, category, start_value, end_value, movement, percent_movement, presence
  movement pair: (start_value, end_value)        <- a time axis, per the brief
  category     : the 12 UK regions

a second table (12 rows):
  dimension, canonical_field, category, start_count, end_count,
  count_share_movement, balance_share_movement, presence
  movement pair: (start_count, end_count)        <- a time axis, per the brief
  category     : 12 distinct -> East Midlands, East of England, London,
                                North East, North West, Scotland, ...
```

A ranked movement of balance **by region, across two period ends** — which is
T7's definition. The requested dimension is present with 12 distinct values, in
a column called `category`. Rated ABSENT because that column is not called
`region` and neither `start_value`/`end_value` nor `start_count`/`end_count` is
in `_TIME_HINTS`.

### 5.2 The two answered T8 comparisons

```
columns      : measure, population, period, prior, current, change
movement pair: (prior, current)                  <- a time axis, per the brief
population   : 2 distinct -> Acquired, Direct
               (and, for the other phrasing, Front Book / Back Book)
```

The prose answer for `Compare balance over time for direct and acquired`:

> Across 2026-04-30 → 2026-06-30, **Direct**, 7,126 loans: Current Outstanding
> Balance £1.36bn → £1.39bn (+£21.5m). Across 2026-04-30 → 2026-06-30,
> **Acquired**, 3,909 loans: Current Outstanding Balance £568.3m → £579.4m
> (+£11.1m).

That is T8's definition — *comparison of two historical segments* — delivered:
both segments, both period ends, the movement between them. Rated ABSENT because
the segment column is called `population` rather than `seasoning` or
`source_portfolio`, and because `prior`/`current` is not in `_TIME_HINTS`.

### 5.3 What this means, stated carefully

**On the artifact, T8 is not absent.** Two of its three phrasings return the
requested comparison. Rated on their contents rather than on column names, T8
would be **PARTIAL, not ABSENT** — some phrasings prove it, one does not, which
is exactly what `shape_rating` calls PARTIAL. T7 is the same shape of error on
one of its four phrasings.

**And the P0 refusal in section 4 sits directly on this fault line.** The system
refuses:

> `How have direct and acquired balances moved over the periods?` — *"I
> understood that you asked for Direct and Acquired tracked separately, but that
> could not be applied to the calculation"*

while answering `Compare balance over time for direct and acquired` with Direct
and Acquired tracked separately, over two periods. **The refusal is honest about
its own inability, but the route demonstrably can serve that request under
another phrasing.** That is the brief's second divergence category —
*the guard refused a proposal the route would have handled* — occurring between
phrasings rather than between arms, so the Gate's arm-vs-arm comparison does not
surface it.

**Followed up in** `docs/mi_time_series_capability_reissue.md` (shapes 7 and 8
re-rated ABSENT → PARTIAL from artifact contents),
`docs/mi_phrasing_reachability.md` (3 of the 21 absences reachable under another
wording) and `docs/mi_measurement_environment_traps.md` (the two environment
traps recorded beside the book-fallback hazard).

**Nothing here was fixed**, per the brief. Both ratings are recorded side by
side in the instrument (`rating` from the inherited surface rule, `rating_brief`
from the brief's rule) and a column census records what every artifact actually
carries. The instrument does **not** auto-credit a generically named column as
the requested breakdown — it names the column and its values and leaves the
judgement visible, because silently crediting `population` as `source_portfolio`
would be the mirror image of the receipt-trusting error the surface exists to
prevent. The two mechanised rules therefore agree on all 29 rows; the
discrepancy above is between **both** of them and the artifact's contents, and
is resolved here by reading the values.

---

## 6. The instrument, and two defects found in it

`question_interpretation/mi_gate_llm_arm.py`. Reads the 29 phrasings from
`time_series_surface.SHAPES` rather than retyping them (a test asserts the two
match exactly), reuses `inspect_artifacts` / `rate` / `shape_rating`, and reuses
the comparison shape of `llm_arm_comparison.py`. One book per process, because
the two books resolve different datasets through module-level caches.

`tests/test_mi_gate_llm_arm.py` — **29 tests, all passing.** Each feeds the
instrument a case it must catch *and* the neighbouring case it must pass, so it
is not a function that always says "bad". Mutating the runner to rate from the
receipt breaks four of them, including the P0-breach check.

Two defects were found in this instrument and are recorded rather than quietly
fixed:

1. **`total_model_calls` reported `1` instead of `130`** — `observe` reset the
   counter per question, so the field captured only the last question's count.
   No reported number depended on it; the printed total is computed
   independently from the per-run arrays.
2. **The brief's movement-pair rule was not implemented**, because
   `inspect_artifacts` was reused wholesale. This is what section 5 is about.

Both fixes were made *after* seeing output, so **both books were re-run from
scratch on the corrected instrument**, twice, and every figure in this report
comes from the final version. Regression tests pin both behaviours.

---

## 7. Reproducing this

Two environment facts, recorded because the baseline does not reproduce without
them and neither is obvious from a fresh clone:

1. **`TRAKT_RUNTIME_MODE` must be non-production.** It defaults to `production`,
   under which `trakt_core.policy` refuses both books as synthetic fixtures —
   all 29 phrasings refuse with `route=None` and every shape rates ABSENT. This
   is the sanctioned path for fixture data (`conftest.py` sets `test` for the
   suite) and cannot take effect in a deployed environment, because
   `validate_runtime_mode` refuses a non-production mode when the Azure markers
   are present.
2. **`demo_platform/workspace/` must be built** — gitignored and regenerable:
   `python -m demo_platform.run_demo --generate --onboard --orchestrate`.
   Without it the book carries a single period and T1 rates ABSENT, not PROVEN.

With both in place the deterministic arm reproduces the recorded baseline
exactly on both books — **0 silent drops, 21 of 29 honest refusals, T1 PROVEN,
T2 PARTIAL, T3–T8 ABSENT** — which is what licensed the LLM arm above.

```
TRAKT_RUNTIME_MODE=development ANTHROPIC_API_KEY=... \
  python -m question_interpretation.mi_gate_llm_arm --book alderbridge --repeats 5
TRAKT_RUNTIME_MODE=development ANTHROPIC_API_KEY=... \
  python -m question_interpretation.mi_gate_llm_arm --book kestrelmoor --repeats 5
```

---

## 8. What this does and does not establish

**Establishes.** Turning the LLM parser on changes nothing on this surface: not
one rating, not one refusal, not one artifact, on either book, across five
repeats — with the model genuinely invoked on 26 of the 29 questions and 130
calls counted per book. The three named P0 refusals hold in every repeat,
including two where the model was called and its proposal validated away. The
LLM arm is not a route to the missing time-series capability, and the
deterministic results quoted for this surface are not an artefact of having the
model switched off.

**Establishes, and was not the question asked.** Three of the ABSENT ratings on
this surface are produced by column-name matching rather than by a missing
capability, and one named P0 refusal declines a request the route serves under a
different phrasing. Both arms are affected identically.

**Does not establish.** That the arms agree on any other surface; this is the 29
time-series phrasings only. That the LLM arm is stable in general — it was stable
here, on questions where the deterministic spec is what executes, which is
weaker than stability where the model's proposal survives. Nor does a flat result
mean the model proposed nothing: it was called 130 times per book and something
came back each time. What is measured is that nothing it proposed changed the
outcome.
