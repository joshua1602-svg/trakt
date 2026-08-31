# MI Dashboard + PPTX — Final Analytical Composition & UX Elevation

**Branch** `claude/mi-pptx-composition-elevation`
**Starting SHA** `d3b8d15` (end of `claude/final-mi-surface-hardening`)
**Gate** the prior sprint's baseline-vs-HEAD regression was re-run to completion
before any change: baseline `62cc602` reproduced the recorded Phase 3 figures
exactly (107 failed / 7432 passed) and HEAD returned 107 / 7474 with **identical
failing test-ID sets in both directions**.

---

## 1. Executive verdict

**YES.**

The pack no longer reads as a credible starter deck that happens to render. It
reads as the front end of an analytical platform, because the depth it now shows
was already in the engine and simply had no way out.

Three findings carry the sprint, and all three are the same finding:

1. **`snapshots.cross_tab` has always been generic over eleven dimensions.** The
   pack asked for a hardcoded three. It now asks for a governed selection and
   the representative book supports four crossings with no new MI.
2. **Two slide handlers existed that no slide in the deck config used.**
   `pipeline_evolution` and `funded_evolution` were written, tested and
   unreachable, so the pack could show what the pipeline and the book ARE and
   never how either MOVED.
3. **Phase 3's stage-movement reconciliation had no route, no component and no
   slide.** It now has all three, reading one computation.

**NEW MI PRIMITIVES ADDED: 0.** Every item was selection, composition,
presentation, or connecting something already built to a surface that could not
reach it.

---

## 2. Executive Position redesign

**Before.** Seven tiles laid out four and three, with a hole beside the second
row. Three of them — pipeline balance, weighted expected, forecast funded —
restated one fact that the Executive Summary restated again in words on the next
page. Two half-width trends competed for the centre under three quarters of an
inch of dead panel.

**After.** One full row of at most six tiles in priority order, so a partial row
cannot exist; below it one trajectory across the full content width, using the
whole band down to the risk strip.

| | Before | After |
|---|---|---|
| Tiles | 7 over two rows, one slot empty | ≤ 6, one row, always full |
| Measures | balance, loans, LTV, pipeline balance, pipeline cases, weighted expected, forecast, time-to-target | balance, loans, LTV, pipeline balance, forecast, **closest limit** |
| Trend | two half-width charts, capped at 2.30in | one full-width, fills the band |
| Risk | a sentence only | a tile a reader can scan, **plus** a line carrying what the tile cannot |

Two duplications went with it, both found by the tests rather than by design:

- The risk line repeated the tile's own utilisation. It now carries the distance
  to the limit.
- **A forecast equal to the funded balance is not a forecast.** On a book with no
  pipeline that tile printed the funded balance again, in a tile claiming to look
  forward. Suppressed.

The composition is assembled from `DashboardData` alone, so it remains offerable
to React as one payload.

---

## 3. Multidimensional capability

**Dimensions available: 11** — `ltv, age, region, rate, product, vintage, status,
equity, broker, borrower_type, ticket`.

**Candidate pairs declared: 12.** LTV is the primary risk axis and pairs first;
size, borrower and geography follow, because "how much of my exposure is
high-LTV AND large-ticket" is the question a credit committee asks.

**Displayed on the representative books: 4** (was 2):

```
Balance by LTV × borrower age    Balance by LTV × region
Balance by LTV × ticket size     Balance by LTV × rate
```

**Selection methodology** (`snapshots.select_multidim_pairs`). A pair is dropped
when:

| Rule | Reason published |
|---|---|
| a dimension is absent | "the tape does not supply both dimensions" |
| either axis has one category | "this is a stratification rather than a crossing" |
| density < 18% of cells | "only n% of cells carry balance, too sparse to read" |
| both dimensions already crossed above | "it repeats a story told above" |
| the page is full | "n stronger crossings already earned the page" |

Nothing branches on asset class. Every rejection carries its reason, published as
`notSelected` on `/mi/multidim` for the methodology ledger. A named pair is still
served on request: the selection decides what a PAGE shows, and a caller asking
for one crossing has already decided.

---

## 4. Funded section

| Page | Change |
|---|---|
| Funded Stock | plot area starts at 1.7in instead of 2.9in — see §11 |
| **Funded Evolution** | **new to the pack**: balance, population, weighted LTV and weighted rate over time. The handler existed and no slide used it. Gated at four periods: a four-panel trend from three points is thinner than the stock page beside it |
| Funded Stratifications | `keys:` is now a PREFERENCE — an uninformative dimension yields its slot to the next the book supports, and the page names what it did not chart |
| Multi-Dimensional | 2 crossings → 4, governed selection |
| Funded Balance Movement | geometry only; the bridge itself is unchanged |

Funded Stock and Funded Evolution are deliberately distinct: stock is the
constituent-book series, evolution is the portfolio's own measures over time,
which is where "is the book getting riskier" is answered.

---

## 5. Pipeline section

**Stratification matrix.** The pipeline had no four-panel page at all. It was
gated on the pipeline being large *relative to the funded book* — but shape is
what a funder asks about origination whatever its size, and the funnel already
carries the "is origination the story" judgement. Now earned by a pipeline large
enough for its bands to be real. Selected dimensions on the representative book:
**LTV, ticket size, borrower age, region**, with broker/channel and product
excluded and said so on the page.

**Evolution quadrant.** New to the pack: pipeline amount, live case count,
weighted expected funding, and stage composition, all weekly. Two panels showed
half of that — a reader could watch the balance move without seeing whether it
was more cases or bigger ones.

**Stage-to-stage movement.** New page, new route, new React panel, one
computation. Per live stage, on counts and money:

```
opening live + arrivals − departures ± amount change on stayers = closing live
```

On the representative book: KFI 7 + 8 − 7 = 8, Application 7 + 8 − 7 = 8,
Offer 8 + 8 − 8 = 8, residual 0.00 on every stage. Departures are split by where
the case went, because a completion is not attrition, and an amount amendment
lands in the stayer leg rather than being counted as an exit and an arrival.

**Overview second chart.** Was broker/channel, which on a direct-only book drew
one bar labelled "Direct" — the pipeline total from the tile above it, redrawn as
a chart. Now the strongest dimension the pipeline distributes across; where none
does, the panel carries the governed expected-completion profile instead.

---

## 6. Forecast

Forecast-by-region and forecast-by-LTV drew one block per category, which shows
where exposure LANDS and hides what a funder is deciding about.

```
London  ████████████████████▏██   £11.9MM
        current funded    expected additions
```

Each bar is now built from its two parts, summing to the forecast printed beside
it, in the deck and in the React Forecast view alike.

Nothing is recomputed. `workspace.forecast_breakdowns` has always emitted funded,
weighted-pipeline and forecast per category — but the parts did not survive the
top-N cap, which reshaped rows and dropped `fundedAmount`, so the first attempt
stacked to a sliver because the funded part read as zero. The cap now carries
every additive component through and aggregates them into its "Other" row the
same way it aggregates the total.

`current + incremental = forecast` is asserted per category, on capped rows, and
against the headline.

---

## 7. Cohort

**What replaced balance-only progression.** Funded balance was the hero curve
unconditionally, which on a stable book drew four nearly flat lines.

A measure now earns the curve by being **available** for these cohorts and by
**moving** as they age, judged against its own level so a 50% LTV moving one
point and a £10m balance moving £200k score the same. The governed preference
order breaks ties: NNEG headroom, weighted LTV, loan survival, weighted rate,
balance, count — risk and performance first, because "how are vintages behaving"
is a question about credit.

On the representative lifetime book the page now leads on **NNEG headroom**, four
clearly separated curves where there were four flat ones — chosen because the
tape carries valuations, not because any code knows what kind of book it is. A
book without them falls to weighted LTV. Where nothing has moved, that is stated
as the finding rather than drawn as a flat chart of it.

---

## 8. React additions

| Surface | Change |
|---|---|
| `components/pipeline/StageMovementPanel.tsx` | **new**: per-stage reconciliation, departures by destination, the identity and the identifier stated |
| `components/EvolutionPanel.tsx` | fetches and mounts the movement panel beside the pipeline series |
| `components/ForecastView.tsx` | forecast cuts drawn as current funded + expected additions, with a key |
| `components/pipeline/bits.tsx` | `BarList` accepts `parts`, sized against the same list maximum so bars stay comparable |
| `api/*`, `domain/evolution.ts`, `domain/pipeline.ts` | `getPipelineMovement`, `PipelineMovement`, `PipelineStageMovement`, `StageDeparture`, `DimensionBucket.fundedAmount` |

---

## 9. Shared analytical/presentation contract

| Fact | Owner | Consumed by |
|---|---|---|
| is a dimension worth a panel | `mi_agent_api.presentation.select_dimensions` | deck stratifications, pipeline overview |
| which crossings this book supports | `snapshots.select_multidim_pairs` | `/mi/multidim`, deck |
| per-stage case + balance movement | `evolution.pipeline_stage_movement` | `/mi/evolution/pipeline-movement`, React panel, deck |
| forecast parts per category | `workspace.forecast_breakdowns` | React ForecastView, deck |
| did this measure move | `DeckBuilder._travel` / `cohorts._travel` | evolution panels, cohort hero |

`test_pipeline_stage_movement_surface.py` proves the deck and the route read one
computation by asserting the window and identifier the route reports appear on
the rendered slide — so the movement view cannot drift into a PPTX-only visual.

---

## 10. Final pack structures

| Book | Slides | Guidance |
|---|---|---|
| New / simple GBP | **14** | 12–15 ✓ |
| Seasoned GBP | **20** | 15–19, one over |
| Multi-book seasoned GBP | **21** | 15–19, two over |
| Multi-book growing GBP | **20** | 15–19, one over |
| Seasoned EUR (no pipeline) | **15** | ✓ |

Two pages were removed on redundancy, not on length: the **Origination Funnel**
is superseded where Stage Movement reconciles the same question case by case,
and **Origination Flow** plotted the same weekly series the evolution quadrant
plots, gated on the same fact, plus two the quadrant adds.

The seasoned packs sit one to two above the guidance. Every page in them answers
a distinct question in the brief's spine, and I did not cut a page that earns its
place to hit a number — the brief's own words are that a pack is fine at length
"if all of them earn their place".

---

## 11. Visual QA

Generated through the real route (`POST /mi/decks/generate` → poll → `GET
/mi/decks/download`), then converted by LibreOffice headless.

```
artifacts/pptx_qa/{new_book_gbp,seasoned_book_gbp,multi_seasoned_gbp,
                   multi_growing_gbp,seasoned_book_eur}.{pptx,pdf}
```

All five clean on the 24 preflight gates, 0 failures, 0 warnings. Automated
sweeps across all five: **0 blank pages, 0 text past the page bottom, 0 captions
outside their own panel.** Every page of every variant rendered at 110 dpi and
inspected.

**Findings fixed during the pass:**

| Found | Fix |
|---|---|
| Full-width charts reserved 1.78in of left gutter for a 0.7in label | Margins measured in inches from the data's own formatted extremes; ~1in of slide reclaimed per full-width chart |
| Stage legend clipped along its top edge on quadrant panels | Legend band sized in inches, same defect class as the gutter |
| "WA interest rate by month" drawn as a dramatic zigzag against three axis labels all reading 6.62% | A series travelling < 0.5% of its own level states that it held; the chart magnified noise and then could not label it |
| Stage movement's fallback text rendered one character per line | `_card` returns width in inches, `_text` takes EMU |
| "−0  £0" for a zero leg | A zero leg is a dash |
| Multidim labels read "Balance By Ltv × Borrower Age" | Label built from the shared dimension names, not `.title()` |

**One QA fixture defect, fixed.** The visual-QA pipeline gave each case a stage
that was a function of its index alone, so no case ever moved. The stage-movement
page rendered "no case left a live stage" on every representative book and half
of it had never been seen. Cases now join in their own week and walk the ladder,
so the departure panel finally draws what it was built for — and the funnel and
conversion pages have something real to measure.

---

## 12. New MI check

**NEW MI PRIMITIVES ADDED: 0.**

Every item resolved to category A–D. Two items are worth naming explicitly
against §19:

- **§11 deep-dive matrices.** Delivered as a controlled library (12 candidates,
  governed selection, published rejection ledger) rendering up to four crossings
  on one page. A second `MULTIDIMENSIONAL II` page is a config entry — the
  handler already accepts a `pairs:` list — and is not in the default pack
  because splitting four crossings across two pages adds no information. The
  previously deferred region-bubble primitive was **not** built.
- **§10 funded evolution.** Delivered from the existing handler and existing
  payload. No new series was computed.

---

## 13. Regression

PLACEHOLDER_REGRESSION

---

## 14. Merge recommendation

PLACEHOLDER_MERGE
