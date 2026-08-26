# D4 — comparison direction: pre-registration

**Written and committed BEFORE the fix.** Nothing here is adjusted afterwards; a
breach is reported as a breach. D4 is a frozen-canary finding (invariant **I7**,
period order is chronological) and is **not a C7 defect** — it lives on
`temporal_compare`, converted in C5. It is corrected on its own merits, before
any ranking-contract work.

---

## 1. The single semantic owner of comparison direction

Traced end to end. The chain has exactly one place where the ORDER is decided:

```
llm_query_parser._compare_recognizer   -> spec.compare_periods       ORDER SET HERE
projection._time                       -> time.comparison_periods    copied verbatim
analytical_plan.comparison_periods     -> (str, str)                 read, not reordered
analytical_plan.build_temporal_compare_plan
      STACK_PERIODS take=named_pair · COMPARE direction="b relative to a"
temporal_compare.compare_periods       -> abs_delta = vb - va
chat_routing._route_compare            -> "moved from {periodA} to {periodB}"
```

**Everything downstream of the parser is faithful.** The plan declares the rule
it applies, the executor applies it, the receipt reports what executed, and the
prose reports the receipt. Narration, receipt and execution already agree — they
agree on a reversed pair.

So the fix belongs at the order, and **patching the narration is expressly
rejected**: it would leave `absoluteDelta`, `percentageDelta` and `direction`
inverted on the receipt while the sentence read correctly, which is a worse
defect than the one it hides.

### Root cause — one line

`mi_agent/llm_query_parser.py`, `_compare_recognizer`:

```python
periods = ([periods[0], rel] if periods else ["latest", rel])
```

When a question names no explicit month but does name a relative period
("since last month", "compared to the prior month"), the pair is built as
`["latest", <relative>]`. `latest` is by definition the **closing** period, so
the comparison **opens at the close and closes at the open**.

Two consequences, not one:

* the sign of `absoluteDelta` inverts, so a rise is narrated as a fall;
* `pct_delta = abs_delta / va` divides by the **closing** value instead of the
  opening one, so the magnitude is wrong as well.

---

## 2. Affected surface — measured, not estimated

`python -m migration_phase0.d4_comparison_direction`

13 questions produce an ordered pair (5 corpus, 8 probes). By recogniser branch:

| branch | n | reversed? | touched by this fix |
|---|---|---|---|
| `explicit_pair` | 8 | no — the reader states the order | **no** |
| `latest_plus_relative` | 4 | **yes, all of them** | **yes** |
| `explicit_plus_relative` | 1 | no | **no** |

The four reversed:

```
[corpus] ['latest', 'prior pipeline']   Compare latest pipeline with prior pipeline.
[probe ] ['latest', 'last month']       How did the book change since last month?
[probe ] ['latest', 'last month']       How did LTV change since last month?
[probe ] ['latest', 'prior month']      How did the book change compared to the prior month?
```

**`explicit_pair` is deliberately left alone.** "Compare November and October"
names its own order, and the plan's stated rule — *a comparison opens at the
first named period and closes at the second* — is honouring the reader, not
inverting them. Reordering those would substitute the system's chronology for
the reader's stated question, which is the substitution class this programme
exists to prevent.

`explicit_plus_relative` ("Compare October to last month") is also left alone:
whether "last month" falls before or after a named October is not decidable from
the question, and inventing an order there would be a new semantic decision, not
a defect fix.

---

## 3. Expected before-state — exact economics

Executed on the six-snapshot governed control book (`mi_2026_01 … mi_2026_06`).

| question | reported | table |
|---|---|---|
| How did the book change since last month? | "Funded balance moved from **£21.1m in 2026-06** to **£18.9m in 2026-05** — a change of £2.2m (**-10.57%, down**)" | `period_a=21102503.20, period_b=18872801.63, abs=-2229701.57, pct=-10.57` |
| How did LTV change since last month? | "WA current LTV moved from **38.6% in 2026-06** to **38.1% in 2026-05** — a change of 0.5% (**-1.35%, down**)" | `period_a=0.3858, period_b=0.3806, abs=-0.0052, pct=-1.35` |
| How did the book change compared to the prior month? | identical to the first | identical |
| Compare latest pipeline with prior pipeline. | refuses: *"I can't compare **latest and prior pipeline** … requested period(s) unavailable"* | — |

The book **grew** from £18.87m in May to £21.10m in June. It is reported as a
10.57% fall.

## 4. Expected corrected state — exact economics, registered before the change

| question | required after | required table |
|---|---|---|
| How did the book change since last month? | "moved from **£18.9m in 2026-05** to **£21.1m in 2026-06** — a change of £2.2m (**+11.81%, up**)" | `period_a=18872801.63, period_b=21102503.20, abs=+2229701.57, pct=+11.81` |
| How did LTV change since last month? | "moved from **38.1% in 2026-05** to **38.6% in 2026-06** — a change of 0.5% (**+1.37%, up**)" | `period_a=0.3806, period_b=0.3858, abs=+0.0052, pct=+1.37` |
| How did the book change compared to the prior month? | identical to the first | identical |
| Compare latest pipeline with prior pipeline. | still refuses; only the order in the message changes to *"prior pipeline and latest"* | — |

`pct` is derived, and both figures are stated so the change of denominator is
visible: `2229701.57 / 18872801.63 = +11.81%`, not `2229701.57 / 21102503.20 =
10.57%`. `0.0052 / 0.3806 = +1.37%`, not `0.0052 / 0.3858 = 1.35%`.

## 5. Expected canary movement

Registered exactly:

```
I7 breach on F10.a   cleared    "How did the book change since last month?"
I7 breach on F10.b   cleared    "How did LTV change since last month?"
grades changed       NONE       every element on every one of the 33 cases
new breaches         NONE
unexercised families NONE moved (F4 stays unexercised)
breach count         9 -> 7
```

The two grades of F10.a / F10.b do **not** move: `SPAN` grades UNEVIDENCED
either way (`temporal_compare` publishes no `rankedMovement`) and `MOVEMENT`
grades HONOURED either way (the prose says "moved from" before and after). Only
the chronology check moves. If any grade moves, the fix has reached further than
this registration allows and must be reported as such.

The frozen baseline is re-frozen in the same commit, with D4 recorded as an
**authorised defect correction**. The historical statement of D4 in the bank is
NOT rewritten — the bank's `known_defects_at_freeze` entry stands as the record
of what was true at freeze.

## 6. Expected regression blast

Two test assertions pin the reversed order. Both are **pinned records of the
defect**, the same shape as C6's `test_the_stage_the_shipped_route_cannot_name`,
which asserted a behaviour the product later fixed and had to be retired. The
C6 ruling applies unchanged: **an estate must not assert behaviour the product
has fixed.**

```
mi_agent/tests/test_mi_analytical_intents.py::test_compare_latest_prior_pipeline
    ["latest", "prior pipeline"]  ->  ["prior pipeline", "latest"]

mi_agent/tests/test_mi_analytical_intents.py::test_compare_change_phrasing
    "How did pipeline amount change from last week?"
    ["latest", "last week"]       ->  ["last week", "latest"]
```

Expected blast: **exactly these two assertions, and nothing else.**
`test_compare_funded_balance`, `test_compare_loan_count_is_count`,
`test_compare_wa_ltv` and the `from October to November` case assert
`explicit_pair` ordering and must be **byte-identical** after the fix — they are
the control.

## 7. Acceptance conditions

The fix is accepted only if all of these hold:

1. the four reversed pairs open at the earlier period and close at the later;
2. the eight `explicit_pair` questions are **unchanged**, order and economics;
3. `absoluteDelta`, `percentageDelta` and `direction` on the receipt agree with
   the executed pair, and the prose agrees with the receipt;
4. exact-name regression introduces **0 unexplained** failing names;
5. exactly the two registered assertions change;
6. canary movement is exactly the two I7 clearances, with **0** grade movements.

## 8. STOP condition

> **STOP — D4 IS NOT A LOCAL TEMPORAL DEFECT** if the correction requires a new
> interpretation subsystem, a chronology resolver, a change to the contract
> schema, or a change to any route.

The fix as scoped is one expression in one recogniser. If it turns out that
ordering `latest` against a relative term requires resolving both to real
reporting dates — i.e. a calendar service the parser does not have — that is a
different and larger piece of work and this task stops and reports.
