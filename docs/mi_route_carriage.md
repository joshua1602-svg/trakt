# Route carriage — tier three, the run-rate question, and where governed populations arrive

Base: merge-base `4e051f3`; `4e051f3` and `28ece25` both ancestors of HEAD
(`77b4fa0`). Deterministic arm, alderbridge, shipped service path.

---

## 1. Tier three CAN certify a wrong breakdown — measured, by construction

**Yes, and it does.** The corpus does not show it; construction does, which is
the distinction that matters after `e35a01b`.

The routed grouping check has three tiers. One proves applied (the frame names
the field), two proves lost (the frame carries no axis at all). Three stamps
APPLIED because *some* axis exists, without establishing it is the requested
one.

Across 260 corpus questions, only two routes reach tier three —
`geo_exposure` (6 stamps) and `concentration_analysis` (1) — and **no route
paired one axis with more than one requested field**. On the corpus alone the
answer would have been "no observed wrong certification", and that would have
been the `e35a01b` mistake again: *no surface reached it* read as *it cannot be
reached*.

Constructed, it takes one question:

| question | route | axes in the frame | groupings stamped |
|---|---|---|---|
| geographic exposure **by ltv bucket** | `geo_exposure` | `area`, `code` | `collateral_geography: applied`, **`ltv_bucket: applied`** |
| geographic exposure **by account status** | `geo_exposure` | `area`, `code` | `collateral_geography: applied`, **`account_status: applied`** |
| what is our geographic concentration **by borrower age** | `geo_exposure` | `area`, `code` | `collateral_geography: applied` |

The answer is broken down by ITL3 area. It is not broken down by LTV bucket or
account status. The receipt says it is.

**The mechanism, stated generally:** a route with a FIXED axis certifies every
dimension a question might name alongside it, and at most one of those can be
right. `geo_exposure` always groups by geography; any second dimension the book
carries is stamped applied on the strength of geography's own axis.

One case escapes for the wrong reason: *"geographic exposure by broker
channel"* comes back clean only because `broker_channel` is absent from this
book, so the answer refuses, no artifacts ship, and tier **two** catches it. On
a book that carries broker, it would be certified.

No test asserting tier three safe, therefore. It is not safe. Recorded as **B12**
and not fixed here — the fix is that a route declares the dimension it grouped
by, which is a change across routes rather than in the guard, and it should be
scoped rather than improvised. Until then the residue is: **`geo_exposure` will
certify any second dimension the book carries.**

---

## 2. "What is the run rate of new lending" — at HEAD

**It refuses. It did not survive both fixes.**

```
route=forecast_extrapolation   ok=False
I understood that you asked for new lending, but that could not be applied to
the calculation (new lending — this answer covers the whole population; it is
neither narrowed to nor broken down by new lending). I have not substituted a
broader figure.
```

Caught by the routed-evidence fix, not the population branch — the population
branch is point-in-time only and this is a routed answer. The forecast frame
carries `month`, `base`, `upside`, `downside`: no categorical axis, so tier two,
so LOST.

Before that fix it returned `~£16.3m/month` with `spec.filters == {}` and a
projection whose first point was £1,964,886,258 — the whole book, labelled as
new lending, with the facet stamped APPLIED.

---

## 3. But the same question in a simpler form still substitutes

Found while building the carriage list, and it is the same class on a routine
question:

```
"What is the balance of new lending?"      route=(point-in-time)  ok=True
   facets: [('grouping_dimension', 'seasoning_segment', 'applied')]
   Here is the bar for your query, covering 2 group(s).
   Total Balance · grouped by Seasoning Segment · 2 groups · 11,035 loans
```

**11,035 loans is the whole book.** The question asked for the balance of new
lending and received a two-bar breakdown of the entire book by seasoning
segment, with the facet stamped applied.

Contrast the same shape one word apart:

```
"What is the balance of the front book?"   route=(point-in-time)  ok=True
   facets: [('row_population', 'seasoning_segment', 'applied')]
   Total Balance · Seasoning Segment = Front Book · 1,177 loans
```

"Front book" resolves to a **population** and narrows to 1,177 loans. "New
lending" resolves to a **grouping** and does not narrow at all. Same governed
concept, same field, same route — different carriage, and only one of them
answers the question.

This is the fourth distinct instance of the same failure, and it is live.
Recorded as **B13**.

---

## 4. Which routes can receive a governed population

The concept resolves correctly and has since the first traces. What fails is
carriage, at a different route each time. The list, rather than the fourth
instance:

| route | does a governed population arrive? | what it does otherwise |
|---|---|---|
| point-in-time | **yes** — *"balance of the front book"* → `row_population: applied`, 1,177 loans | **but not always** — *"balance of new lending"* raises a GROUPING instead and answers over 11,035 loans (§3) |
| `analytical_composition` | **yes** — `row_population: applied` alongside the cohort comparison | — |
| `geo_exposure` | **yes** — *"geographic exposure of the front book"* → `row_population: applied`, and the answer narrows (£7.9m largest, 164 areas vs 172) | — |
| `risk_limits` | **arrives, cannot be applied** → `row_population: lost`, refuses naming the population | correct: a limit schedule is not narrowable, and it says so |
| `evolution` | **no** — *"how have the front book and the back book changed over time"* raises only a grouping, now LOST, and refuses | the population never becomes a facet; the refusal comes from the grouping |
| `forecast_extrapolation` | **no** — *"run rate of new lending"* raises only a grouping, now LOST, and refuses (§2) | as above |
| `period_movement` / `analytical_composition` movement | **no** — *"how has new lending moved since last month"* answers correctly narrowed (106 loans) but records only a grouping | **under-reports rather than mis-answers**: the route applied the population and the receipt does not say so |

Three states, and they are different problems:

* **arrives and applied** — point-in-time (sometimes), analytical composition,
  geo exposure.
* **arrives and correctly refused** — risk limits. Working as designed.
* **never arrives** — evolution, forecast extrapolation, and the movement path.
  Of these, evolution and forecast now REFUSE, which is safe but coarse: they
  refuse on the grouping's absence rather than on the population's, so the
  reader is told the answer is not *broken down by* the front book when what
  they asked was to be *narrowed to* it.

One more, unexplained and logged: *"Forecast the balance of the front book for
the next quarter"* fails with *"'Seasoning Segment' is not available in this
dataset"* — which is false; the tape carries `seasoning_segment`. **B14.**

And a contradiction worth naming: *"Show the back book balance by month"*
produces `row_population: applied` AND `grouping_dimension: lost` for the same
field in one receipt. It fails closed, so no wrong number ships, but a receipt
asserting a field was both applied and lost is not one a reader can act on.
**B15.**

---

## 5. Backlog added here

| id | finding |
|---|---|
| B12 | Tier three certifies any second dimension a fixed-axis route's book carries. Demonstrated on `geo_exposure`. The fix is a route declaring what it grouped by. |
| B13 | *"Balance of new lending"* answers over the whole book as a seasoning breakdown. "Front book" narrows; "new lending" does not. Live. |
| B14 | *"Forecast the balance of the front book"* claims `seasoning_segment` is absent from a tape that carries it. |
| B15 | One receipt asserting the same field both `applied` (population) and `lost` (grouping). |
