# MI Query Agent — capability review

**Scope.** Can the MI Query Agent that serves the React MI Agent and Microsoft 365
Copilot answer the questions a credit / treasury / IC audience actually asks —
by narrative or by chart?

**Method.** Forty questions were run through
`mi_agent_api.mi_service.execute_governed_mi_query`, the single governed
capability (`mi.question.answer`) behind both `POST /mi/query` and the Copilot
`askTraktMi` action. No stubs, no re-implementation: the production
parse → route → execute → adapt path, including governance, ran for every
question.

**Data.** The synthetic `demo_platform` book — 11,035 loans, £1.96bn, two source
portfolios (one *direct*, one *acquired*), three month-end snapshots
(2026-04-30 / 2026-05-31 / 2026-06-30), and the client's synthetic Schedule 8
concentration limits. Chosen because it is the only synthetic asset in the repo
with genuine multi-period history and a direct-vs-acquired split, so the
temporal, forecast, bridge and risk-limit routes all have real governed inputs.

**Reproduce.**

```bash
python -m demo_platform.run_demo --generate --orchestrate   # ~90s, one-off
python scripts/run_mi_capability_review.py --out out/mi_capability_review
```

Question bank: `config/mi/golden_questions/business_semantic_questions.yaml`.

**Parser caveat — read this before acting on the findings.** No
`ANTHROPIC_API_KEY` was available in the review environment, so this measures the
**deterministic parser**. That is not a corner case: it is the configured default
whenever no key is set, it is what CI and the demo run on, and
`parse_with_repair` falls back to it whenever the LLM is unavailable or errors.
It is also *not* fully bypassed when a key is present — `zero_cost_first` returns
the deterministic spec without ever calling the LLM whenever that spec validates
and carries `parser_confidence == "high"`. Findings below are marked
**[det-only]** where an LLM parse would plausibly escalate past them, and
**[all-paths]** where the defect is downstream of parsing and an LLM cannot fix
it.

---

## 1. Headline

| Outcome | Count | Share |
|---|---:|---:|
| Answered the question asked, correctly | 5 | 12% |
| Right machinery, a stated facet dropped or re-read without saying so | 8 | 20% |
| **`ok=true` with a confident answer to a *different* question** | **19** | **48%** |
| Controlled refusal — honest and specific | 2 | 5% |
| Failed with a message that does not say what could not be resolved | 6 | 15% |

The distribution matters more than the pass rate. The agent's stated contract —
"every grouping dimension is either APPLIED or REJECTED, never silently dropped"
(`mi_agent/mi_query_contract.py`, enforced by `mi_agent/mi_query_harness.py`) —
does not hold for the shapes tested. **Nearly half of all questions returned `ok=true`, no
warning, an empty `rejected_dimensions`, and a number that answers something
else.** In a management-information tool that is worse than a refusal, because
the number is quotable.

Two concrete examples, against ground truth computed directly from the same
governed frame:

| Question | Agent answered | Truth | Error |
|---|---|---|---|
| "What is my exposure to borrowers over 85?" | £1.96bn | £31.1m (1.58% of the book) | **63×** |
| "What proportion of the book is eligible for a 75% LTV securitisation?" | 43.2% | 99.67% of balance | reads as an eligibility share; it is the WA LTV |
| "What would a 10% house price fall do to my WA LTV?" | 43.2% | 47.95% | no stress applied, not disclosed |
| "What is the average LTV in London?" (isolating probe) | 43.2%, 11,035 loans | 42.75%, 1,380 loans | whole book returned as the London answer |

What is genuinely good is also worth stating plainly: the **risk-limit,
period-change, funded-bridge, geographic-exposure and run-rate routes are real,
governed, reconciled analytics** with provenance, materiality caveats and
"unavailable" states. The failure is almost entirely in *question understanding*
and in *narrative honesty about what was actually computed* — not in the
calculation engines.

---

## 2. The nine submitted questions

### 2.1 "What is the average LTV, the average borrower age, the average borrower type in London?" — **fails**

Returned: one KPI, `Youngest Borrower Age = 71`, `Loans = 11,035`. The London
scope was dropped, two of the three requested measures were dropped, and nothing
was disclosed.

Two independent defects.

**(a) A named segment does not become a filter on an average question.**
`[det-only, but broad]` The filter branch in
`mi_agent/llm_query_parser.py:1699` is gated on:

```python
is_count_q  = bool(re.search(r"\bhow many\b|\bnumber of\b|\bcount of\b", q))
is_balance_q = bool(re.search(r"\bhow much\b|\btotal balance\b", q))
if is_count_q or is_balance_q:
    filters = _parse_filters(...)
```

An *average* / *weighted-average* question never enters it. Demonstrated:

```
"total balance in london"   -> filters {'collateral_geography': 'London'}   ✅
"balance in london"         -> filters {}                                   ❌
"average ltv in london"     -> filters {}                                   ❌
"average ltv where ltv above 50" -> filters {}                              ❌
```

This is why "exposure to borrowers over 85" returned the whole book: "exposure"
is not "how much".

**(b) Even where the filter branch is reached, the regional matcher is brittle.**
`mi_agent/llm_query_parser.py:1355`:

```python
_CATEGORICAL_FILTER_RE = re.compile(
    r"(?:geographic\s+region|geographic|geography|region|in)\s+"
    r"([a-z][a-z]*(?:\s+[a-z]+){0,2})\s*$")
```

Anchored at `$` with no punctuation class, and only the preposition "in":

```
"balance in london"            -> {'collateral_geography': 'London'}        ✅
"balance in london?"           -> {}          # a question mark defeats it  ❌
"balance for london"           -> {}          # "for" unsupported           ❌
"balance in the south east"    -> {'collateral_geography': 'The South East'} # no row matches ❌
"balance in london region"     -> {'collateral_geography': 'London Region'}  # no row matches ❌
```

Every real user types the question mark.

**(c) Multi-measure questions silently collapse to one measure.** "What is the
average LTV and the average borrower age?" returns age only. `MIQuerySpec` has a
single `metric` field; there is no multi-KPI intent, and no disclosure that two
of three were discarded.

**(d) "Average borrower type" is not a coherent request** (a mode, not a mean).
The right behaviour is to say so. It vanished instead.

### 2.2 "Which region / broker / product / borrower type has grown the most in the last month / quarter?" — **fails**

| Variant | Route taken | Result |
|---|---|---|
| region / last month | `geo_exposure` | point-in-time **concentration**, not growth |
| broker / last quarter | `period_change_analysis` | whole-book metric movements; no broker split |
| product / last quarter | `period_change_analysis` | as above |
| borrower type / last month | `period_change_analysis` | whole-book balance +£18.1m; no split |

Three distinct problems.

* **Growth intent lost to a concentration route** `[all-paths]`. "Which region has
  grown the most" is a *ranked delta* question. `geo_exposure` answered with the
  largest ITL3 exposure today. The word "grown" had no effect on route selection.
* **The ranking dimension is dropped.** `period_change_analysis` produces a
  *Composition shifts* table with 358 rows carrying exactly the per-category
  share movements needed (e.g. `Collateral Geography — South East +0.12 pp`), but
  the narrative never ranks by the dimension the user named, and the answer text
  leads with `Number Of Days In Arrears +0` — a nil movement presented as the
  headline finding.
* **"Last quarter" is silently re-read as "last month"** `[all-paths]`.
  `mi_agent/period_change/recognition.py:104` recognises `"this quarter"`; there
  is no entry for `"last quarter"` / `"in the last quarter"`, so it falls to the
  adjacent-snapshot default. The answer honestly *states* the dates it used
  ("Between 31 May 2026 and 30 June 2026") but never says the requested period
  was not honoured — and April data exists, so a real quarter comparison was
  available.

Note `broker_channel` and `erm_product_type` are genuinely absent from this book.
The right answer to the broker variant is a refusal naming that field. It
answered instead.

### 2.3 "Show me balance by LTV by borrower type" — **fails, worst case in the set**

Returned a **single-bar chart on `amortisation_type`: one row, "Interest roll-up",
100%.** Neither requested dimension appears; an unrelated third one was
substituted; `ok=true`; `rejected_dimensions` empty.

The mechanism is precise and reproducible. `_deterministic_parse` produces the
*correct* plan when it is not told which columns exist, and destroys it when it
is: `[det-only]`

```
"Show me balance by LTV by borrower type"
  available_columns=None   -> heatmap, dimensions ['ltv_bucket','borrower_type'],
                              explicit_dimension_requested=True,
                              requested_dimension_terms=['current_loan_to_value','borrower_type'],
                              parser_confidence='high'
  available_columns=<real> -> bar, dimension 'amortisation_type', dimensions [],
                              explicit_dimension_requested=False,
                              requested_dimension_terms=[],
                              dimension_substituted=False,
                              parser_confidence='medium'
```

Three things go wrong at once in the availability-aware pass:

1. `borrower_type` is absent from this book — correct to reject, but
2. rejecting it also destroys `ltv_bucket`, **which is present**; and
3. the record that anything was requested is erased — `explicit_dimension_requested`
   flips `True → False`, `requested_dimension_terms` is emptied, and
   `dimension_substituted` stays `False` **while a substitution is being made**.

Step 3 is why the fail-closed guarantee cannot fire: the invariant is enforced
against `requested_dimension_terms`, and by the time validation runs there is
nothing left to reject. This is the single most important fix in the review.

The two-dimension capability itself is sound — where both dimensions exist it
works well:

```
"Show me balance by region and LTV band"    -> heatmap, 88 groups  ✅
"Show me balance by region by occupancy type" -> heatmap, 12 groups ✅
```

### 2.4 "Show me balance by region by borrower type" — **partial**

Grouped by `collateral_geography` only, correct figures (South East £516.2m /
26.3%, London £413.8m / 21.1%). `borrower_type` was dropped, `rejectedDimensions`
was `None`, and no warning was raised. The user sees a plausible regional chart
and has no way to know half their question was discarded.

### 2.5 "Show me balance by borrower type by product" — **fails, unhelpfully**

`ok=false`, `"The proposed query failed validation."` The validation artifact
carries the detail; the answer string does not. Compare the reference refusal the
system is capable of, from B20:

> 'NNEG' is not available in this dataset. The MI book for this client does not
> include `nneg_flag`. This field is not reported, so the question cannot be
> answered from the current data (no value was fabricated).

That is the standard every failure should meet. `[all-paths]`

### 2.6 "Am I close to breaching any concentration limits?" — **correct; best answer in the review**

> 8 passed, 0 warning(s), 1 breach(es), 0 need review, 3 unavailable. Nearest to
> limit: Borrowers aged over 85 (-0.2 pp headroom). Largest concentration: WA
> current LTV at 43.2%.

Plus a 12-row table of *test / actual / limit / headroom / status / movement /
source*, with limits extracted live from the Schedule 8 document, period-on-period
movement per test, and an explicit `unavailable` state for the three tests whose
inputs the book does not carry. This is what the rest of the surface should look
like.

One follow-through gap: **"How much headroom before the *London* limit binds?"
returns the identical whole-schedule answer.** The London row is in the table
(21.1% actual vs 25.0% limit, 3.9pp headroom) but the narrative never narrows to
the limit that was named — the same scope-drop as §2.1.

### 2.7 "Based on current origination, when will funded loans be £100MM?" — **fails to route**

`ok=false`, "I couldn't map this question to a governed analytic."

The pieces all work individually. `_forecast_target_value` parses the target
correctly:

```
"…when will funded loans be £100MM?"           -> 100_000_000.0  ✅
"when will the book be £100m?"                 -> 100_000_000.0  ✅
```

and the `forecast_extrapolation` route works when triggered:

```
"What is the run rate of new lending?" ->
  "The current completion run-rate is ~£16.3m/month (£195.5m/year) based on
   2 month(s) of funded growth."  ✅
```

The recogniser simply does not fire on `"when will … be <target>"` — only on
phrasings like `"reach … at the current run rate"`. `[det-only]`

**A separate `[all-paths]` bug sits behind it.** When the route *does* fire with
a target above the milestone ladder, it reports the opposite of the truth:

```
"When will the funded book reach £2.5bn at the current run rate?"
 -> "The book has already reached £2.50bn (current funded balance £1.96bn)."
```

Self-contradictory in one sentence. `mi_agent_api/chat_routing.py:919`:

```python
def _ms(thr: float) -> Optional[Dict[str, Any]]:
    exact = next((m for m in milestones if m["threshold"] == thr), None)
    if exact: return exact
    above = [m for m in milestones if m["threshold"] >= thr]
    return above[0] if above else (milestones[-1] if milestones else None)
```

When no milestone reaches the target, it falls back to `milestones[-1]` — the
largest, which *is* reached — and line 929 then reports "already reached" for a
target the book is nowhere near. The `else` branch at line 936 ("beyond the
projection horizon") is the correct answer and is unreachable in this case.

### 2.8 "What is the average LTV of the direct book vs the acquired book?" — **partial**

Routed correctly to `portfolio_risk_comparison`. The table is right:

| metric | aggregation | Direct | Acquired | difference |
|---|---|---|---|---|
| Current Loan To Value | weighted average (wt: balance) | 43.35 | 42.68 | 0.67 |

The narrative is: *"Direct has higher observed Current Loan To Value than
Acquired."* **No numbers.** The user asked "what is the average LTV of each" and
got a direction. In React the table rescues it; in Copilot, which is text-first,
the answer as delivered does not contain the figures requested. `[all-paths]`

### 2.9 "How has the average collateral value of the book changed since inception?" — **fails**

`ok=false`: *"No governed field tagged for period-change analysis is available in
both snapshots for this portfolio."* That is not true of the underlying data —
the near-identical phrasing "…over the last three months" **does** compare it, and
reports `current_valuation_amount` £4.84bn → £4.86bn.

Three problems in one answer:

* **"Since inception" is reduced to the latest adjacent pair** (31 May → 30 June),
  discarding the April snapshot. `[all-paths]`
* **The variant that works reports `sum`, not `average`,** despite `aggregation:
  avg` on the parsed spec — £4.86bn is the total valuation of the book, not the
  £440k average collateral value.
* **The narrative contradicts its own artifact**: "0 of 1 governed metrics could be
  compared" printed above a table showing `start_value £4.84bn / end_value £4.86bn`.

For reference, the correct answer is available and simple: average
`current_valuation_amount` is £440,427 at the current cut-off, across three
snapshots.

---

## 3. Twenty-eight additional business-semantic questions

Full text, expected behaviour, observed behaviour and grade for each are in
`config/mi/golden_questions/business_semantic_questions.yaml`. These deliberately
name a business *concept* — concentration, headroom, credit quality, run-rate,
mix shift, diversification, stress, eligibility, vintage performance — and leave
the agent to choose the measure, dimension, period and route.

**Correct (4):** B01 concentration + headroom · B08 lending run-rate (though the
"is it accelerating" half goes unaddressed) · B11 regional contribution to WA LTV ·
B21 largest single-loan exposure (£841,638.96, correct).

**Controlled refusals (2):** B20 NNEG — the reference answer quoted in §2.5 — and
B03 single-broker reliance, which reaches the right outcome (`broker_channel` is
absent) but names a parsed phrase, *"'over-reliant single' is not a governed
measure"*, rather than the missing dimension.

**Partial (5):** B07 named-limit headroom (whole schedule) · B17 WA-LTV drivers
(returned a *balance* waterfall — good artifact, wrong metric, undisclosed) ·
B18 / B27 mix-shift and "relative to last month" (temporal half dropped) ·
B22 "top 10 postcodes" (answered in ITL3 areas although a `postcode` column exists
and the concentration route ranks it — substitution undisclosed).

**Silently wrong (14), the pattern that matters.** Grouped by cause:

| Cause | Questions |
|---|---|
| Scope / cohort filter dropped → whole book returned | B06 (£1.96bn for over-85s; truth £31.1m), B15 (eligibility answered with WA LTV), B14 |
| Conditional / stress premise ignored, unstressed number returned | B05, B26 (43.2% vs a true stressed 47.95%) |
| Temporal premise ignored, point-in-time answer returned | B02, B04, B24, B12 |
| Route discards the parsed spec | B10 — parsed correctly as `arrears_balance` by `collateral_geography`, then answered with generic exposure concentration ("Interest roll-up 100%") |
| Wrong comparison axis, narrative contradicting its own tables | B12, B25 — "no governed directional differences were observed" printed above tables showing clear differences |
| Wrong grain | B09 "which vintages have the highest LTV" returned ten individual loans rather than a vintage ranking — **including `loan_identifier`**. The executor's stated loan-level privacy rule ("never identifiers") is written for scatter/bubble; this table path is outside it, which is at minimum an inconsistency needing an explicit decision |
| Unrelated measure returned | B19 "what will the balance be at year end" → weighted-average interest rate 6.6% |

**Unhelpful failures (3 here, 6 across the whole bank).** B13, B16, B28 and A5 all
return the bare string `"The proposed query failed validation."` with no statement
of what could not be resolved; A7 and A9 fail with messages that are specific but
wrong about the data (§2.7, §2.9). B16 in particular is *substantively* the right
outcome — `broker_channel` is not in this book — delivered in the least useful
possible way.

---

## 4. Two defects outside the parser

**4.1 The portfolio summary states the interest rate 100× too low.** `[all-paths]`

The first thing most users ask returns:

> …Weighted-average current LTV is 43.2%, the weighted-average interest rate is
> **0.07%**…

Asked directly, the same agent returns **6.6%**, and `period_change_analysis`
also reports 6.56%. `mi_agent_api/chat_routing.py:367`:

```python
detail.append(f"the weighted-average interest rate is "
              f"{_pct_points(m['wa_interest_rate'], 2)}")
```

`movement_summary.py:122-123` converts LTV to points (`_ltv_to_points`) but passes
`wa_interest_rate` through as a **fraction** — and the same file's own
`_METRIC_DISPLAY` table classifies `wa_interest_rate` as `pct_fraction`. It is
formatted here as if it were points. A 0.07% lifetime-mortgage coupon is not a
subtle error to a credit audience.

**4.2 Low parser confidence is never surfaced.** `[all-paths]`

`_deterministic_parse` returns `parser_confidence` of `high` / `medium` / `low`
and it is used for LLM-escalation decisions, but it never reaches the user.
"What is the average LTV in London?" parses at `confidence='low'` and is presented
with the same flat certainty as a question that parsed at `high`:
*"Here is the result for your query, covering 1 group(s)."*

That default answer string is itself a problem across the whole point-in-time
path: it states the *shape* of the result and never restates **what was actually
computed**. Every silently-wrong answer in §3 would be caught by the user in one
read if the narrative said "weighted-average current LTV across all 11,035 loans
(no geographic filter applied)".

---

## 5. Recommendations, in priority order

1. **Never let the availability pass erase the request record.** In
   `_deterministic_parse`, when a requested dimension's column is missing, keep
   `requested_dimension_terms`, set `dimension_substituted=True`, and retain any
   *co-requested dimension that does exist*. Fixes §2.3, §2.4 and restores the
   fail-closed invariant the harness is meant to guarantee. **Highest value.**
2. **Extend the filter branch to every aggregation.** Move `_parse_filters` above
   the `is_count_q or is_balance_q` gate so averages, weighted averages and
   "exposure" questions carry their predicates. Fixes §2.1(a), B06, B15, B14.
3. **Harden the categorical matcher**: strip trailing punctuation, accept
   `for` / `across` / `within`, strip leading articles, and — critically —
   **validate the matched value against the dimension's actual values, refusing
   when it matches nothing** rather than filtering to zero rows or dropping it.
   Fixes §2.1(b) and turns "in Atlantis" into a refusal.
4. **Make the default narrative restate what was computed** — measure, aggregation,
   filters applied, population size, and any requested facet that was dropped.
   Surface `parser_confidence` when it is not `high`. This single change converts
   most of §3's silent-wrong answers into visibly-partial ones.
5. **Fix the two arithmetic/logic bugs**: the `milestones[-1]` fallback in
   `chat_routing.py:919` (report "beyond the projection horizon"), and the
   `wa_interest_rate` unit at `chat_routing.py:367`.
6. **Give period recognition "last quarter" / "since inception"**, and make the
   period-change narrative state when a requested period could not be honoured.
   Fixes §2.2, §2.9, B18, B27.
7. **Rank by the named dimension in growth questions.** `period_change_analysis`
   already computes the composition shifts; route "which \<dimension\> grew most"
   to a ranked delta on that dimension instead of `geo_exposure` or a generic
   metric list.
8. **Standardise every failure on the B20 refusal template** — name the concept,
   name the missing field, state that nothing was substituted. Replaces
   `"The proposed query failed validation."` everywhere.
9. **Put the numbers in the narrative** for `portfolio_risk_comparison` (§2.8) —
   Copilot has no table to fall back on.
10. **Audit loan-level output for identifier exposure** (B09) against the stated
    executor contract.
11. **Re-run this bank with the LLM parser enabled** to split the `[det-only]`
    findings into "fixed by escalation" and "still broken". Note that
    `zero_cost_first` means a *confident* wrong deterministic parse never
    escalates, so items 1–3 remain necessary regardless.
