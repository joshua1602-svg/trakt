# PPTX Funder-Pack Enhancement — existing capabilities only

**Branch** `claude/pptx-funder-pack-enhancement`
**Starting SHA** `b1ed31d4c8e605b2bdb23bf6440e6ed4e2d6eff8` (head of `claude/pptx-ux-parity-implementation`)
**Scope** materially improve the base automated funder/investor pack using only
capability Trakt already owns.

---

## 1. Executive verdict

**PASS.**

The base pack was a catalogue of twenty-five configured slides that rendered
whatever resolved. It is now a starter pack organised around the nine questions
a funder asks in order, and it carries three pages of analysis that Trakt had
already been computing and had never drawn:

- an **economic opening-to-closing bridge**, with exits split by evidenced
  reason, that reconciles to the penny;
- **funded stock by constituent book over time**, as a stack that sums exactly
  to the period total on every period;
- a **per-book forward view**, carrying the client's approved run-off curve
  where one exists and saying plainly where one does not.

Every enhancement came from category **A** (already computed — surface it),
**B** (composition of existing production outputs) or **C** (presentation,
materiality, suppression). **No new analytical primitive, model, methodology or
asset-class engine was built.** Everything that would have needed one was
stopped and reported instead — §12 lists eight such items.

Deck lengths through the real `POST /mi/decks/generate` route: **14 / 17 / 18 /
19 / 14** slides across five book shapes, all clean against **24** publication
gates (21 before this sprint).

One scope judgment is set out in §2.5 and should be read
before merge: the economic bridge **had never rendered on any book**, and making
it render required aligning one governed constant with a contract two other
governed modules already publish.

---

## 2. What analytical capability was surfaced

### 2.1 Already computed — surfaced (category A)

| Capability | Where it already lived | What consumed it before |
|---|---|---|
| Reconciled balance bridge | `mi_agent.period_change.bridge.balance_bridge` | nothing in the pack |
| Evidence-based exit classification | `analytics_lib.history.classify_exits` | prepayment measures only |
| Per-period × per-book funded balance | `evolution.funded_evolution` breakdowns | nothing |
| Per-constituent-book projection | `forecast_bridge.portfolio_projections` | nothing |
| Concentration utilisation history | `concentration_tests_api.compute_history` | React Risk Limits tab only |
| Prior-forecast / variance series | `evolution.forecast_evolution` | one chart, unread |
| Published capability registry | `trakt_core.capability` + `config/system/mi_capability_registry.yaml` | API and agent tools only |
| Per-stage conversion lag and sufficiency | `evolution.pipeline_funnel_evolution` | drawn, mislabelled |

The deck resolves all of these through the existing `_guard` path in
`mi_agent_pptx/mi_api.py`, so a failure omits a page rather than failing a run.

### 2.2 Composition only (category B)

Two compositions were written, and both are compositions in the strict sense —
they call existing governed engines and assert the engines agree:

- **`evolution.funded_balance_movement()`** composes `balance_bridge` with
  `classify_exits`, and asserts the classified split reconciles to the bridge's
  exit leg before reporting either. Where the identity does not close, or the
  split does not sum, the caller is told so and the page is omitted with the
  engine's own reason. Exposed as `GET /mi/evolution/funded-movement`, beside
  the endpoints the dashboard already reads.
- **`mi_agent_pptx.forecast_accuracy`** summarises the forecaster's track record
  from `prior_forecast` and `funded_balance`, both already reconciled by the
  evolution service. `prior_forecast` at period *N* **is**
  `forecast_funded_balance` at period *N−1*; the module does arithmetic over
  them and refuses to summarise fewer than two observations.

### 2.3 Presentation, materiality and suppression (category C)

- **`mi_agent_api/materiality.py`** — decides whether a contribution set has a
  *driver at all*, before any commentary names one. Two independent tests must
  both pass: the leader holds ≥35% of the total magnitude, and it is separated
  from the runner-up by ≥20%. Seven regions at 4.4/4.0/4.0/3.9/3.8/3.7/3.7
  classify as `broadly_distributed` (leader share 0.16, separation 0.09);
  22.0/2.0/1.5 classify as `driven` (0.86, 0.91).
- Deck spine, conditional composition, supersession, suppression of uniform
  dimensions, axis anchoring, tick-label collision, scenario-line simplification,
  cohort and run-off terminology, funnel basis wording.

### 2.4 No new MI primitive — confirmed

No new metric, model, methodology, projection or asset-class engine was added.
`grep`-able proof: the deck config contains no asset-class name in any condition
(asserted by `test_16`), and `mi_agent_pptx/` gained no calculation — the two new
modules are a classifier of *shapes* and a summariser of *errors*, both over
figures another engine reconciled.

### 2.5 THE ONE SCOPE JUDGMENT — read this before merge

**The economic bridge had never rendered on any book, and the cause was a
disagreement between three governed modules about what a loan identity is.**

| Module | Accepted identity columns |
|---|---|
| `engine.platform_assembler.LOAN_KEY_FIELDS` | `loan_identifier`, `unique_identifier` |
| `mi_agent_api.evolution._LOAN_ID_COLS` | `loan_identifier`, `unique_identifier`, + 3 legacy |
| `mi_agent.period_change.bridge.IDENTIFIER_FIELDS` | `loan_identifier`, `original_loan_identifier`, `underlying_exposure_identifier` |
| `mi_agent_api.snapshots._loan_ids` | `loan_identifier` only |

A **regime-projected book carries the ESMA Annex 2 RREL1 name
(`unique_identifier`) instead of the analytics name, not as well as it.** On
exactly those books the bridge returned *"no stable loan identifier"* — about a
tape whose identifier the regime requires to be constant for the life of the
exposure — while the assembler that produced the tape had keyed the same loans
without complaint.

`unique_identifier` was added to `bridge.IDENTIFIER_FIELDS`, matching the
assembler. **This is not a new primitive**: no calculation changed, and the
bridge's duplicate-identifier and missing-identifier refusals are untouched, so
a tape whose identifier is not in fact stable still gets no bridge. It is an
alignment with a contract two other governed modules already publish.

`analytics_lib.history` has the same blind spot and sits on the performance-
sensitive prepayment path, so **it was not changed**. The composition layer
aliases the column before calling it (`evolution._exit_frames`): same strings,
same classification rules, no new analytic.

**`snapshots._loan_ids` still reads `loan_identifier` only.** That is outside
this sprint's scope and is reported in §12 as a divergence to close.

---

## 3. Funded stock

**Page: "Funded Stock"** — replaces the generic "Funded Balance Evolution".

- More than one constituent book → a **stacked area** over
  `funded_evolution.breakdowns["portfolio"]`, ordered largest-closing-first, with
  every period filled (a book absent in a period contributes zero rather than
  breaking the stack).
- One book → a single area. A one-colour stack of itself conveys nothing a line
  does not.
- The value axis is **anchored at zero** for stacked and currency series. A stock
  chart on a floating baseline turns a 2% move into a cliff.

### Stack reconciliation proof

New mandatory publication gate **`stack_reconciles`**, on the multi-book QA
fixture (three books, five periods):

```
stack_reconciles: True — the per-book stack sums to the period total in all 5 periods
    evidence {'offenders': []}
```

Verified to **block** when the defect is reintroduced: perturbing one book by 5
in each period produces

```
stack_reconciles: False — the per-book stack does not sum to the period total in 2 period(s)
    [{'period': '2026-05', 'total': 100.0, 'stack': 95.0, 'gap': 5.0},
     {'period': '2026-06', 'total': 110.0, 'stack': 105.0, 'gap': 5.0}]
```

### Stock and movement are one story

New mandatory gate **`stock_and_movement_agree`**. The stock series comes from
the funded-evolution loader; the bridge reconciles loan by loan. Where both pages
are in the deck they must close on the same figure:

```
stock_and_movement_agree: True — the stock series and the balance bridge close on the same figure
    evidence {'stock_closing': 104761440.0, 'bridge_closing': 104761440.0, 'gap': 0.0}
```

The stock page also *names* that shared number and hands the reader on —
but only when the two engines actually agree, because a pointer to a page that
closes on a different figure would be worse than no pointer.

---

## 4. Funded movement

**Page: "Funded Balance Movement"** — supersedes the generic "Portfolio Movement
and Drivers" wherever the loan-level bridge reconciles.

### Economic bridge reconciliation proof

Multi-book seasoned QA fixture, through the production route:

```
window            2026-05 -> 2026-06
identifier        source_portfolio_id + unique_identifier
opening           101,061,530.00
+ new loans         3,796,800.00   (12 loans)
- exited loans      2,807,840.00   (7 loans)
+ continuing        2,710,950.00   (306 loans)
= closing         104,761,440.00
identity check    104,761,440.00   residual +0.0000000000
```

### Exit decomposition

Exits are shown as one bar **per evidenced reason**, and only where the split
reconciles to the bridge's exit leg:

```
exit split    [('Redeemed', 892710.0, 2), ('Exited in default', 973370.0, 2), ('Matured', 941760.0, 3)]
split total   2,807,840.00   vs bridge exit leg 2,807,840.00
evidence      ['default_date', 'loan_redemption_flag', 'maturity_date']
```

Where the tape evidences nothing, the whole exit balance lands in
**"Exited — reason not evidenced"** and is shown as such. That is a data-quality
finding and is never quietly folded into redemptions.

### The continuing leg is not interest

The third leg is labelled **"Continuing book"** and the page states, in text:

> Continuing-book movement is the change on loans present at both dates; it is
> not split into interest, repayment or further advance.

Separating accretion from repayment needs per-loan period movement the canonical
model does not carry. Trakt owns no such attribution, so the pack does not imply
one. `test_4` asserts the words "accrued interest", "interest accrued",
"interest roll-up" and "interest rollup" appear nowhere in the deck.

---

## 5. Forward view

**Page: "Forward View by Constituent Book"** — `forecast_bridge.portfolio_projections`,
rendered for the first time. Conditional on more than one constituent book: the
forecast bridge already *is* a single book's forward view.

Columns: Book / Current / Expected additions / **Run-off retained** / Projected,
plus a total row and a projected-balance bar list.

The run-off disclosure is carried verbatim in substance and never dropped:

> Run-off applied from the client's approved curve for *…*. No approved run-off
> curve for *…*; those balances are held flat, not projected to decay. **Trakt
> generates no mortality, decay or run-off assumption.**

A projection that quietly assumes a book never redeems is worse than one that
admits it does not know.

**A defect the visual pass caught.** The page printed a total of £111.8MM over
per-book rows summing to £104.8MM. The engine was right: where the governed
pipeline cannot be attributed to an individual book it holds the whole weighted
amount outside the per-book rows, adds it to the total, and carries its own
disclosure saying so. The *slide* rendered the rows and the total and neither
the unattributed line nor the disclosure — so the first arithmetic a funder
checks came out wrong on a page about their money. The row and the engine's
words are now on the page, and a new mandatory gate **`projection_totals`**
refuses to publish a projection whose stated total is not the sum of what the
page prints.

**Headroom had vanished** from the concentration page for the same class of
reason: its column gives way when prior and expected are both present, and the
detail line meant to carry it only renders when the rows are tall enough — so on
a four-test book the one slide about headroom stated none. It is now on the
status line of every row.

### Forecast presentation

**"Forecast Evolution — Actual vs Prior Forecast"** now leads with the variance
chart and carries a track record in its subtitle and a sentence beneath:

> Across 3 periods the published forecast was typically 4.2% from the outturn,
> and overstated it on average by 4.2%; the widest miss was 4.8% at 2026-03.

Two observations are the floor. One period in which a forecast happened to be
close is luck, and reporting it as a mean error dresses a coincidence as a
property of the process. A mean signed error below 0.5% of the book is reported
as *no consistent lean* rather than given a direction.

**"Forecast Projection — Run-Rate Scale-Up"**: downside, base and upside within
3% of each other at the horizon are three lines drawn on top of one another. The
chart then carries the base case and the band is stated in words; the milestone
table still carries all three scenarios.

---

## 6. Concentration — direction of travel

The covenant table stated where each test sits and where it is expected to go,
and left the reader to guess whether it had been moving toward its limit or away
from it. `concentration_tests_api.compute_history` — which re-evaluates *today's*
approved configuration against every historical frame, so prior and current are
comparable under one definition — was already being resolved into the deck and
drawn nowhere.

The table now reads left to right as the sequence a covenant moves through:

**Prior → Current → Expected → Limit**

and each row states which way it travelled. Direction is measured **against the
limit, not against the number**, because the governed operator decides which way
is worse:

```
max test moving up     -> 'toward the limit'
max test moving down   -> 'away from the limit'
min test moving down   -> 'toward the limit'      <- a floor test falling is trouble
min test moving up     -> 'away from the limit'
noise (0.3 on a 30 limit) -> 'broadly unchanged'
no prior period        -> None (nothing is stated)
```

A move smaller than a fiftieth of the limit is the book's ordinary noise and is
reported as unchanged rather than dressed up as a trend.

### Unexplained stress results are now explained

For a test whose denominator grows faster than its numerator, converting the
whole pipeline **dilutes** the concentration — the stressed figure comes out
*below* the current one. Printed bare beside the word "stress" that reads as a
fault. It is not; it is a real property of the test, and it is now stated:

```
stress tightens  -> None (the figure is shown as usual)
stress dilutes   -> 'converting the whole pipeline would dilute this test, not stress it'
stress inert     -> 'the stress does not move this test'
```

Where prior and expected are both present the Headroom column gives way — six
columns would squeeze the test name below the width at which a governed limit
name is legible — and headroom is stated in words on the detail line instead.

---

## 7. Conditional and materiality model

### Capability-driven, never asset-class-driven

`mi_agent_pptx.mi_api` resolves every published capability against this
portfolio's canonical shape (`trakt_core.capability.describe_portfolio` +
`resolve_all`) and exposes each as a `can_<metric>` composition fact. The
registry is **asset-agnostic by construction** — a capability declares the
economic conditions it needs, and any book meeting them gets it — which is
exactly the property a conditional pack needs.

On the multi-book QA fixture: **28 capability facts, 13 available**; the
unavailable set (`arrears_stock`, `arrears_30_plus`, `default_rate`,
`contractual_wal`, …) is what that tape genuinely cannot support.

`test_16` asserts no slide condition in the shipped config contains
`equity_release`, `lifetime_mortgage`, `asset_class`, `product_type` **or
`seasoned`**.

### What now suppresses, and why

| Suppressed | Condition | Reason |
|---|---|---|
| Portfolio Composition | `type_count > 1 or portfolio_count > 1` | a table of one column |
| Forward View by Book | `constituent_books > 1` | the forecast bridge is already that book's forward view |
| Vintage Formation | `not has_cohort_progression` | progression's table already states each vintage at formation |
| Portfolio Movement and Drivers | `not has_balance_movement` | the loan-level bridge answers the same question better |
| Risk Limits | `not has_concentration` | the approved capability supersedes the legacy monitor |
| Origination Funnel | `has_pipeline_history and pipeline_share ≥ 0.05` | without weekly flow it restates the stage ladder on the previous page |
| Pipeline Stratifications, Origination Flow | `pipeline_share ≥ 0.15` | origination deep dives, for books where origination is the story |
| Run-Rate Scale-Up | `pipeline_share ≥ 0.10` | a growth page for a book that is growing |
| Secondary Stratifications | `funded_balance ≥ 100m and constituent_books ≤ 1` | a multi-book pack has already enumerated its constituents |
| Multi-Dimensional Risk | `funded_balance ≥ 25m` | thin cells read as noise |
| A uniform stratification panel | `< 99.5%` in one bucket | one full-width bar saying what its own title says |
| Indistinguishable scenario lines | band `< 3%` of base at horizon | three lines drawn on top of each other |

**Nothing is suppressed silently.** A dimension dropped for being uniform is
named on the page. Every dropped slide is recorded with its reason and rendered
in the methodology ledger. Where *every* dimension is concentrated they are all
kept: that is the finding, and an empty page is not an improvement on a flat one.

### Materiality applied to commentary

`materiality.classify` now governs every "driven by" claim in the pack:
the movement headline, the stock takeaway, the constituent-book sentence, and the
executive summary's movement attribution. Two portfolio types contributing
+£31.6m and +£30.9m are no longer described as one driving the other.

---

## 8. Terminology and credibility corrections

### Cohort terminology

**"Retention" is a survival idea, and it is only that for a count.** A balance
ratio wearing the same name exceeds 100% on a roll-up book while loans are
leaving, which tells the reader the pool grew.

- Cohort table column `Retention` → **`Balance vs formation`**, with a new
  separate **`Loan survival`** column carrying the count-based ratio.
- Card title "Retention since formation" → **"Change since formation"**.
- `CohortSeries` exposes `loan_survival` (this *is* retention) and
  `balance_vs_formation` (this is not), and the docstring says why.
- Per-book projection column `Retention` → **`Run-off retained`**: it is the
  share of balance the *client's* approved curve keeps, not a survival rate
  Trakt observed.

### Methodology-ledger contradictions

Two contradictions the pack used to print on the same document:

1. **Risk limits.** The ledger said *"no governed risk-limit artefact for this
   run"* in a pack that rendered concentration headroom two pages earlier. A
   slide config may now declare `superseded_by:`, and where the superseding page
   is in the deck the ledger says *"covered by Concentration Tests and
   Headroom"* under a distinct `superseded` category.
2. **Geography.** The ledger said *"no geographic exposure resolved"* in a pack
   whose stratification page drew a regional bar list. Geography is two things
   on this tape: the map needs area-level (ITL3) exposure; the stratification
   needs only a region field. The reason now names *which* geography is missing
   and points at the coarser cut where it rendered.

Also fixed: a one-period book was told its *bridge did not reconcile*. That page
now uses its data guard rather than a condition, so it reports the engine's own
reason — *"an opening-to-closing bridge needs two governed reporting periods; 1
available"*.

### Funnel semantics

The conversion strip worded its basis line from a deck-level lag field and every
tile from the fixed phrase *"of lagged KFI stock"* — so a deck whose rates were
computed **unlagged** said "(unlagged)" once and "of lagged KFI stock" four
times on the same page. Both now read the evaluator's own **per-stage**
`lagApplied` / `lagWeeks`, and a page with a mixed lag says so rather than
asserting one number.

### Chart honesty

- **Axis materiality**: currency and stacked series are anchored at zero.
- **Duplicate tick labels**: compact currency resolves to 0.1MM, so a series
  between 109.05m and 109.14m labelled four gridlines `£109.1MM`. The formatter
  now measures the axis range against the notation's own resolution and adds
  decimals only where ticks would collide:
  `['£109.05MM', '£109.08MM', '£109.11MM', '£109.14MM']` — four distinct labels;
  a wide series is untouched: `['£10.0MM', '£60.0MM', '£110.0MM']`.

---

## 9. Final base-pack composition

Configured: 26 slides. Rendered: whatever the book justifies.

### NEW BOOK — one constituent book, one reporting period, live pipeline (14)

```
 1. Cover                            8. Origination Funnel
 2. Executive Position               9. Pipeline Stratifications
 3. Executive Summary               10. Origination Flow — KFIs & Completions
 4. Funded Portfolio — Key Measures 11. Forecast Bridge — Funded to Forecast
 5. Funded Stratifications          12. Portfolio Health and Watch Items
 6. Multi-Dimensional Risk          13. Concentration Tests and Headroom
 7. Pipeline Overview               14. Data and Methodology
```

Target 10–14. **14.** No stock, no movement, no cohorts, no forward view — a
one-period book has no history to show, and the ledger says so for each.

### SEASONED BOOK — one book, five periods, evidenced exits (17)

```
 1. Cover                            10. Pipeline Overview
 2. Executive Position               11. Origination Funnel
 3. Executive Summary                12. Forecast Bridge — Funded to Forecast
 4. Funded Portfolio — Key Measures  13. Forecast Evolution — Actual vs Prior Forecast
 5. Funded Stock                     14. Cohort Progression
 6. Funded Stratifications           15. Portfolio Health and Watch Items
 7. Funded Stratifications — II      16. Concentration Tests and Headroom
 8. Multi-Dimensional Risk           17. Data and Methodology
 9. Funded Balance Movement
```

Target 14–18. **17.**

### MULTI-BOOK — three books, five periods, evidenced exits (18)

```
 1. Cover                            10. Pipeline Overview
 2. Executive Position               11. Origination Funnel
 3. Executive Summary                12. Forecast Bridge — Funded to Forecast
 4. Portfolio Composition            13. Forward View by Constituent Book
 5. Funded Portfolio — Key Measures  14. Forecast Evolution — Actual vs Prior Forecast
 6. Funded Stock (stacked)           15. Cohort Progression
 7. Funded Stratifications           16. Portfolio Health and Watch Items
 8. Multi-Dimensional Risk           17. Concentration Tests and Headroom
 9. Funded Balance Movement          18. Data and Methodology
```

Target 14–18. **18.**

### MULTI-BOOK GROWING — the maximal variant (19)

Everything above, **plus** Forecast Projection — Run-Rate Scale-Up, which that
book's pipeline share earns. **19, one page above the 14–18 target.** This is the
variant with three books, a live pipeline with weekly history, three periods of
forecast history and approved limits all lit at once; every page in it answers a
distinct question, and no further page could be cut on a principled rule rather
than to hit a number. Reported rather than forced.

---

## 10. Visual QA

`python scripts/pptx_visual_qa.py --out artifacts/pptx_qa` — five book shapes,
each built through `POST /mi/decks/generate` → poll → `GET /mi/decks/download`,
then the downloaded PowerPoint inspected for overlapping or clipped shapes,
unreadable type, orphaned titles, empty chart frames, duplicated titles, foreign
currency symbols and bar lists out of the governed bucket order.

```
[new_book_gbp]       14 slides — clean    Preflight: PASS — 24 gates, 0 failures, 0 warnings
[seasoned_book_gbp]  17 slides — clean    Preflight: PASS — 24 gates, 0 failures, 0 warnings
[multi_seasoned_gbp] 18 slides — clean    Preflight: PASS — 24 gates, 0 failures, 0 warnings
[multi_growing_gbp]  19 slides — clean    Preflight: PASS — 24 gates, 0 failures, 0 warnings
[seasoned_book_eur]  14 slides — clean    Preflight: PASS — 24 gates, 0 failures, 0 warnings
```

Decks: `artifacts/pptx_qa/*.pptx` (git-ignored; the report JSON is tracked).

**The visual pass found two defects no gate and no test had.** Both were pages
that were internally correct and *presented* wrongly — a total that did not sum
to its own rows, and a headroom figure that fell out of a page when a column
gave way. Both now have gates. Rendering the decks and looking at them is not
optional: neither defect is visible from a payload, a slide count, or a passing
test suite.

**The harness itself was a finding.** It generated three single-book fixtures
that only ever grew, so the bridge had no exit leg, the stack had one book, the
composition page had one column, and the per-book forward view never rendered.
It now builds a loan whose identity and static attributes are functions of the
loan number — the same loan in every period — across up to three books, with
loans leaving from the bottom of the range carrying redemption, default and
maturity evidence on the period they leave from.

### Acceptance tests

`tests/test_funder_pack_enhancement.py` — **16 named tests, 16 passing**, every
one driving the real route and reading the downloaded PowerPoint or its
production preflight sidecar. Each was verified to fail on the defect it names:

| Defect reintroduced | Tests that fail |
|---|---|
| bridge identity contract reverted | 1, 2, 3, 4, 7, 12 (6 tests) |
| exit-classifier aliasing removed | 3 |
| balance column renamed "Retention" | 14 |
| ledger supersession removed | 12 |

Two new publication gates were separately verified to **block** on a
reintroduced defect (§3).

---

## 11. Regression

Measured against the starting SHA `b1ed31d` in a `git worktree`, same
interpreter, `-p no:randomly`.

### Baseline — `b1ed31d`, whole suite

```
108 failed, 7357 passed, 433 skipped, 8 xfailed, 6 subtests passed   (28:00)
```

The 108 pre-existing failures were **not** touched, per the brief. They cluster
in the simulation and threshold-receipt suites and are unrelated to the pack
(`test_simulation_multi_source`, `test_source_scope_production_semantics`,
`test_threshold_receipt_evidence`, `test_funded_bridge_grouping_declaration`).

### Branch — `fbfaf8e`, whole suite

```
107 failed, 7373 passed, 434 skipped, 8 xfailed, 6 subtests passed   (27:51)
```

### Difference

```
new failures introduced by this sprint : 0
passed                                 : 7357 -> 7373   (+16, the 16 new acceptance tests)
failed                                 :  108 ->  107   (-1)
```

**Zero regressions.** The failure sets were compared by node id, not by count:
every one of the 107 failures on the branch is in the 108-failure baseline set.

The one baseline failure absent on the branch is
`tests/test_serving_parquet.py::test_the_serving_copy_is_materially_faster_and_smaller`,
which asserts a wall-clock speed ratio. **This sprint did not fix it** — nothing
here touches the serving copy — and it should be read as machine timing, not as
an improvement.

### Suites the sprint touches, run explicitly on the final tree

```
tests/mi_agent_pptx/                      204 passed,  2 skipped
tests/test_funder_pack_enhancement.py      16 passed
tests/test_presentation_parity.py          17 passed          (with the two above)
tests/test_deck_generation_route.py        all passed         (with the two above)
                                          ---------------------------------------
combined                                  244 passed,  3 skipped
```

Every suite the brief names — PPTX, deck route, presentation parity, funded
evolution, funded bridge, exit classification, forecast, concentration, cohort,
capability registry, conditional composition — is inside those runs or inside
the whole-suite run above.

---

## 12. Deferred capability

Stopped and reported rather than built, because each needs category **D–H**:

| Item | Needs | Why it was stopped |
|---|---|---|
| Accrued-interest decomposition of the continuing leg | **D** new primitive | needs per-loan period movement the canonical model does not carry. The page states what it measured instead. |
| Projected NNEG / HPI stress | **E** new model + **H** ER engine | no house-price path, no crossover model. Not attempted. |
| Mortality / move-to-care run-off | **E** new model | Trakt generates no decay assumption; the forward view says so on the page. |
| Restatement attribution | **D** new primitive | "why did last period's number change" needs a restatement ledger that does not exist. |
| Arrears, default rate, cure rate, CPR, loss severity | **F** methodology activation | the conventional-credit methodologies are gated behind `TRAKT_AGENT_API_ENABLED` and were **not** enabled. The registry reports them `UNAVAILABLE` for these books and the methodology page states that. |
| Contractual WAL / YTM | **F** + registry conditions | reported by the registry with its own reason, not silently absent. |
| `snapshots._loan_ids` identity divergence | out of scope | still reads `loan_identifier` only, so its new/exited loan counts remain blind on a regime-projected book. **This is a live divergence and should be closed next.** |
| Legacy `analytics/` migration | **G** | untouched, as instructed. |

---

### Commits

Logically separated, on `claude/pptx-funder-pack-enhancement`:

```
15aad79  Bring the capability audit across as this sprint's reference
1dfb02a  Decide whether a contribution set has a driver at all
93c8425  Compose the reconciled bridge and the evidenced exit split into one movement
c924488  Resolve the movement and per-book projection for the deck, and stack a series
c4e6d54  Three pages that answer the funder's questions, and honest cohort wording
549cb0e  Rebuild the pack around nine business questions, and stop the ledger
         contradicting the deck
85c1e22  Show where each limit came from, not just where it sits
b74b5aa  Make the bridge reachable on a regime-projected book, and report
         capability by registry
cb9170a  QA the five book shapes the pack actually has to be right for
287d375  Say the finding, suppress the non-finding, and never label two
         calculations alike
abaf3e8  Gate the reconciliation, and prove the pack through the route a funder uses
f68af1c  Stop the executive summary calling two equal contributors one driver
0e1c4df  Fix a total that did not sum to its rows, and the pages that clipped
         their own captions
```

---

## 13. Recommendation

> *Is this now a credible, impressive automated starter funder/investor pack that
> can be supplied to a client without pretending to be a complete asset-class
> surveillance system?*

**YES.**

It is credible because the two questions a funder asks first — *what do I own*
and *why did it change* — are now answered with an identity that reconciles to
the penny and an exit split backed by evidence on the tape, and because
everything it cannot answer it names, with the registry's own distinction
between "you have not supplied the data", "this book has no such thing" and
"that needs a model we do not run".

It is not pretending to be a surveillance system. There is no arrears page, no
default rate, no NNEG, no prepayment projection — and the methodology page says
so under **"Measures not reported for this book"**, by name and by reason.

The judgment that most deserves a reviewer's attention is §2.5: making the
economic bridge reachable required changing one governed constant. It is an
alignment with a contract two other modules already publish, the refusals that
make the bridge safe are untouched, and the change is covered by six tests — but
it is a change to a shared engine and a reviewer should agree with it, not
inherit it.

**Do not merge without a reviewer signing off §2.5.** Everything else in this
sprint is additive to the pack and gated by 24 publication checks.
