# Item 4 (A5) — the class being admitted, and the two-route finding settled

Pre-registration. No code touched. Base: HEAD `16c66b2`.

---

## 1. The two-route finding, settled first — because it is the same defect

The brief required settling it before or with A5. Measured, **it is not a
separate finding: it is A5 seen from the other side.**

```
question                                          ok     route              answer
Please provide a portfolio summary                True   portfolio_summary  457 chars, 6 KPIs, chart(8), table
portfolio overview                                True   portfolio_summary  457 chars
summarise the book                                True   portfolio_summary  457 chars
book overview                                     True   None               120 chars, 2 KPIs
key metrics                                       True   None               120 chars, 2 KPIs
What are the headline numbers for the portfolio?  True   None               120 chars, 2 KPIs
give me a snapshot of the portfolio               True   None               120 chars, 2 KPIs
How is the book doing?                            True   None               120 chars, 2 KPIs
Tell me the basics about this book                False  None               refusal
Tell me about this book                           False  None               refusal
Where do we stand?                                False  None               refusal
```

**Three outcomes for one question.** The generic path is not *deciding* these are
summaries — it is failing to recognise them and falling through to a count, which
happens to look summary-shaped. So widening the recogniser does not multiply the
inconsistency; **it removes it**, because everything in the class then reaches
the one governed answer.

That is the settlement: **one route, one answer.** Not a second summary
implementation, and not a merge of two — a recogniser that claims what it should
already have claimed.

## 2. THE CLASS, stated before implementing

A question is a request for the book's overall position when **both** hold:

**(a) It names nothing else.** No measure, no dimension, no filter, no
comparison — and the spec marks no specialist capability (risk limits, forecast,
cohort progression, bridge, temporal compare). Computed from the parser and the
spec, not enumerated.

**(b) It carries a summary-intent marker.** A vocabulary.

### Is that a class or five sentences? Honestly: a rule with a vocabulary in it.

**(a) is what makes it a rule.** It is computed, and it does all the
discriminating. Measured on phrasings I did not write the vocabulary for:

```
  tell me about brokers               EXCLUDED   (names a dimension)
  tell me about arrears               EXCLUDED   (names a measure)
  tell me about the london exposure   EXCLUDED   (names a filter)
  how is lending doing                EXCLUDED   (names a measure)
  summarise the portfolio by region   EXCLUDED   (names a dimension)
  what has changed since last month   EXCLUDED   (comparative)
  what is the CPR of this book        EXCLUDED   (names a measure the registry lacks)
  Show the risk limit pass/warn/fail summary.  EXCLUDED (spec marks risk_limit_query)
```

The CPR and WAL exclusions matter most: those keep their capability explanation,
which is a better answer than a summary nobody asked for. **The rule must not
convert an honest "I cannot compute that" into a summary**, and (a) is what
stops it.

**(b) is a list and will need extending.** It is the same kind of finite
vocabulary as `AXIS_MARKERS` or `COMPARATOR_PHRASES`, and it is stated as one
rather than dressed up: summary · summarise · overview · snapshot · basics ·
headline · key metrics · key figures · highlights · top-line · *"how is X
doing"* · *"where do we stand"* · *"how do things stand"* · *"tell me about"*.

So the honest characterisation: **a rule whose admission test is structural and
whose trigger is a vocabulary.** Not a patch — it admits sentences nobody wrote
down (*"How is the book doing?"*, *"Where do we stand?"*, *"give me a snapshot"*
were never in the examples and are admitted; *"tell me about brokers"* uses the
same trigger and is refused). But not vocabulary-free either, and the day someone
writes *"give me the top-line picture"* it will need a word added.

### One design decision, measured rather than assumed

An earlier draft also required a SUBJECT word (book/portfolio/this/we). It
excluded four true positives — `key metrics`, `summary`, `overview`, `snapshot`,
which name no subject because there is only one book — and prevented exactly one
false positive, the risk-limit question. Reading the spec marker excludes that
one directly and costs nothing, so **the subject requirement was dropped**: it
was buying one exclusion at the price of four.

## 3. Pre-registered prediction

| id | today | predicted |
|---|---|---|
| A5 *"Tell me the basics about this book"* | refusal | **the governed summary** |
| `book overview`, `key metrics` (calibration) | 2-KPI generic card | **the governed summary** |
| *"What are the headline numbers"* (A4) | 2-KPI generic card | **the governed summary** |
| `what is the CPR of this book` | capability explanation | **unchanged** |
| `Show the risk limit pass/warn/fail summary.` | risk limits | **unchanged** |
| `summarise the portfolio by region` | stratification | **unchanged** |

`shipped_shapes` predicted **15 correct · 0 wrong · 0 honest · 0 unhelpful.**

**`answer_diff`: 4 corpus answers predicted to move**, all from a thinner answer
to the governed summary, all deliberate. This is the first item in the sequence
predicted to move corpus answers, and the movement IS the settlement.

### Must not move

1. Calibration 259/259 — `kpi_029`/`kpi_030`/`kpi_031` change ANSWER but must
   still pass: they assert `expected_artifact_type: kpi`, and the summary emits a
   KPI artifact first.
2. The 44, both books `32/6/4/2`; seasoning by name Q1 4, Q7 4, Q8 12.
3. Routed surface 32/32.
4. Items 1-3's tests.

### Stop conditions

* any question naming a measure, dimension, filter or specialist capability
  becoming a summary;
* the CPR/WAL capability explanations disappearing;
* calibration falling below 259.

---

# MEASURED

## 4. Results against §3

| declared | measured | |
|---|---|---|
| A5 → the governed summary | **answered, `portfolio_summary`, 6 KPIs, chart(8), table** | ✅ |
| `book overview`, `key metrics`, headline numbers → the summary | all three, identical answer | ✅ |
| CPR / contractual WAL unchanged | capability explanations intact (435 / 389 chars) | ✅ |
| risk-limit summary unchanged | still `risk_limits` | ✅ |
| `summarise the portfolio by region` unchanged | still a stratification | ✅ |
| shipped shapes **15 / 0 / 0 / 0** | **15 / 0 / 0 / 0** | ✅ |
| calibration 259/259 | 259/259, 0 hard failures, 0 known gaps | ✅ |
| the 44, both books | `32/6/4/2`; seasoning Q1 4, Q7 4, Q8 12 | ✅ |
| routed 32/32, lexical unmoved | 32/32; 697 of 697 identical | ✅ |
| **4 corpus answers move** | **0 moved** — see §5 | ❌ |

**Eleven phrasings now return one answer.** The two-route finding is settled:
there is no longer a thin path and a rich path, because nothing falls through.

## 5. The prediction was wrong, and the reason is B7

I predicted four corpus answers would move and none did. `answer_diff` reads
**729 of 729 identical**.

The four questions this item deliberately changed are `kpi_028`–`kpi_031`, and
**all four live in the calibration bank** — the one corpus that calls
`run_mi_agent_query` directly and **bypasses routing entirely** (backlog B7).
Their differ records carry `answer: ""`; that surface grades structure, not
prose, and no route change can register in it.

So the differ's clean result says nothing whatever about this item. It is not
evidence the change was safe, and it is not evidence it worked.

**This is the fifth instance of an instrument unable to see the change it was
meant to measure — and the first found by a prediction being wrong rather than by
a defect slipping through.** The prediction assumed the corpus containing these
questions would exercise them the way production does. It does not, and that was
already written down.

What actually evidences this item: the shipped-shapes surface (15/15, which does
drive `execute_governed_mi_query`), the eleven-phrasing route probe, and the 22
tests.

## 6. Constructed coverage

> **729 of 729 identical means the differ could not reach this change at all** —
> a stronger and less comforting statement than the usual "the fix did not reach
> the corpora". The four corpus questions in the class are answered on a path
> that does not route. The claim rests on the routed probe and the tests.
