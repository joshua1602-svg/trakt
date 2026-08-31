# MI Pack Engine Phase 3

**Branch** `claude/mi-pack-engine-hardening`
**Starting SHA** `b82a4c3b294291bfaf0e5ce5247d7dc525003e99`
**Audit used** `docs/reports/mi_pack_engine_audit.md` (Phases 1–2)

---

## 1. Executive verdict

**Pipeline correctness: FIXED.** Live pipeline stock no longer contains
completed or withdrawn cases. On the QA fixture the headline fell from
£10,700,000 / 40 cases to **£7,770,000 / 30 cases** — the 27.4% overstatement is
gone — and all four downstream consumers were proved to have moved with it, not
assumed to.

**Analytical ownership: MATERIALLY IMPROVED, not finished.** Three coherent
groups moved to engine owners with behaviour *and* ownership tests. A further
set is classified, named, and deliberately not yet migrated; the count is held
by a ratchet so it can only go down.

**No new MI primitive was required.** Everything built is a composition of
values the engine already produced, or a statement of a basis that was already
implicit. Both genuinely-new candidate builds are **DEFERRED** (§13).

**One audit finding was withdrawn on contradictory evidence** (§4). That is
reported prominently rather than buried, because the audit was mine.

---

## 2. Starting point

| | |
|---|---|
| Starting SHA | `b82a4c3b294291bfaf0e5ce5247d7dc525003e99` |
| Branch | `claude/mi-pack-engine-hardening` (child of `claude/mi-pack-engine-audit`) |
| Working tree at start | clean |
| Audit | `docs/reports/mi_pack_engine_audit.md` |

---

## 3. Pipeline stock defect

### The defect

`pipeline_prep._build_report` summed `current_outstanding_balance` over the
**whole extract** and set `row_count = len(df)`. Neither excluded terminal
cases. A completed case has funded and sits in the funded book; a withdrawn case
has gone away. Both were being counted as live pipeline.

### The correction — a SPLIT, not a filter

The governed vocabulary already existed and was simply unused by the economic
totals: `ACTIVE_STAGES` has named the live set since it was written, and
`mi_workflows.analytical` already filtered on it. The live set now has **one**
definition (`_OPEN_PIPELINE_STAGES` was a second copy) plus
`TERMINAL_SUCCESS_STAGES` / `TERMINAL_FAILURE_STAGES` for the other side.

**No row is dropped.** `live_mask()` splits; the frame keeps everything.

### Fixture difference

| | OLD | NEW |
|---|---|---|
| `total_pipeline_amount` | 10,700,000.00 | **7,770,000.00** |
| headline case count | 40 | **30** |
| overstatement | **+2,930,000 (27.4%)** | none |
| `total_extract_amount` | — | 10,700,000.00 *(new, auditable)* |
| `terminal_row_count` | — | 10 |
| `terminal_stage_counts` | — | `{'COMPLETED': 10}` |
| `live_stages` | — | `['KFI','APPLICATION','OFFER']` |

On `tests/fixtures/pipeline_history_5w` (real lender layout), week by week:

```
week          rows  live      live amt       all amt
2026-05-01       6     6     2,300,000     2,300,000
2026-05-08       7     7     2,800,000     2,800,000
2026-05-15       8     8     3,600,000     3,600,000
2026-05-22       8     5     2,400,000     3,600,000   2 Completed, 1 Withdrawn
2026-05-29       8     5     2,400,000     3,600,000   2 Completed, 1 Withdrawn
```

The first three weeks are unchanged — the fixture has no terminal case until
week four, which is what makes it worth asserting against.

### Downstream consequences — proved, not assumed

| Consumer | Before | After |
|---|---|---|
| Pipeline snapshot | 10,700,000 / 40 | **7,770,000 / 30** |
| Forecast bridge `pipelineAmount` / `pipelineCaseCount` | 10,700,000 / 40 | **7,770,000 / 30** |
| Composition fact `pipeline_amount` | 10,700,000 | **7,770,000** |
| Composition fact `pipeline_share` | ~0.093 | **0.069** |
| Weekly `pipeline_evolution` | 10,700,000 / 40 | **7,770,000 / 30** |

**A slide-inclusion decision changed as a result.** `pipeline_share` crossing
below 0.10 correctly removes the run-rate scale-up page from the maximal QA
deck, which drops from 19 to 18 slides — a page that had been included on a
wrong number.

**One assumption did not survive contact.** *Weighted* expected funding was
**already correct**: the stage-probability config assigns no probability to a
terminal stage, so those rows carried `NaN` and never reached the sum. The
*unweighted* `expected_funded_amount` **was** wrong and is now fixed. Both are
pinned — the first precisely because it is right by accident, and a future
config giving `COMPLETED` a probability of 1.0 would silently start reporting
already-funded loans as expected future funding.

### Preserved history

`stageBreakdown` still carries `COMPLETED`, the frame still carries every row,
and `test_terminal_cases_remain_available_to_history` fails if rows are dropped.

---

## 4. Pipeline identity — an audit finding WITHDRAWN

**The audit said `unique_identifier` was a missing alias for the pipeline case
key. Implementation contradicts that, and the finding is withdrawn.**

On the **real lender fixture** the contract already maps cleanly:

```
pipeline_case_identifier   <- 'Account Number'
application_identifier     <- 'KFI Number'
populated 8/8, unique 8
```

What was actually broken was **my own QA fixture**, which wrote
`unique_identifier` — the ESMA RREL1 name for a *funded underlying exposure*,
which is not a pre-funding case key and is correctly absent from the alias list.
The audit generalised from a synthetic artefact.

**The contract is unchanged.** Per the brief — *"do not add the alias merely
because names look similar"* — nothing proves a lender's regulatory loan id is
also their case key, so no alias was added. The QA fixture now writes the
contract's own alias.

### Authoritative identifier

`pipeline_case_identifier` — a **natural key carried from the source, not a
hash**. Stable across amendments by construction.

### Amount-amendment proof

`test_an_amended_amount_is_a_movement_not_an_exit_and_an_arrival`: ACC002 sits
at KFI in both weeks with its amount amended 200,000 → 220,000. It appears
**once**, as +20,000 of movement on a persisting case:

```
persistingCaseCount        1
amountChangeOnPersisting   +20,000
departureCaseCount         1   (ACC001, which moved to APPLICATION)
arrivalCaseCount           1   (ACC007, genuinely new)
```

Under `snapshot.keys.make_pipeline_opportunity_id` — which hashes `loan_amount`
— ACC002 would be a 200,000 departure **plus** a 220,000 arrival with zero
amount change, inflating both flow legs and erasing the movement. **The register
stays unwired.**

### Suppression where identity is not governed

| Condition | Behaviour |
|---|---|
| No `pipeline_case_identifier` | `available: False` with the reason naming the field. No stages returned. |
| Duplicated identifier | `available: False` — a duplicated key is not an identity. |
| Fewer than two extracts | `available: False` with the count. |

**There is no fallback.** Live stock is still reported from a valid current
extract; only case-level reconciliation suppresses.

---

## 5. 31-value classification

**A** engine analytical semantic · **B** shared presentation semantic ·
**C** formatting · **D** redundant, remove

| Value | Current owner | Class | Final owner | Migrated? | Reason |
|---|---|---|---|---|---|
| Portfolio-type balance share (×4 sites) | deck, insights | **A** | `portfolio_context.balance_share` | **YES** | Two channels ask for it; four sites could disagree |
| Opening balance = total − Σ movers (×2) | deck | **A** | `portfolio_context.opening_from_movement` | **YES** | A waterfall's base and its legs must share one definition |
| Type composition table | deck | **A** | `portfolio_context.type_composition` | **YES** | Same |
| Forecast error % per period | `forecast_accuracy` | **A** | `evolution.forecast_evolution` | **YES** | The number a funder judges the forecaster by |
| Forecast bias (mean signed) | `forecast_accuracy` | **A** | `evolution._forecast_accuracy` | **YES** | Same |
| Forecast absolute error | `forecast_accuracy` | **A** | `evolution._forecast_accuracy` | **YES** | Same |
| Widest miss + its period | `forecast_accuracy` | **A** | `evolution._forecast_accuracy` | **YES** | Same |
| Concentration direction of travel | `concentration.travel` | **A** | `concentration_tests_api.direction_of_travel` | **YES** | Which way is worse is the governed operator's, not a renderer's |
| Stress direction | `concentration.stress_note` | **A** | `concentration_tests_api.stress_effect` | **YES** | Same |
| Pipeline average case amount | deck | **A** | `pipeline_contract` | **NO** | Named; superseded in priority by the stock fix that changes its numerator |
| Pipeline week-on-week deltas | deck | **A** | `pipeline_contract` | **NO** | Named; same |
| Bridge leg shares | deck | **A** | `evolution.funded_balance_movement` | **NO** | Named in the ratchet |
| Stratification spread | deck | **A** | `snapshots` stratifications | **NO** | Named in the ratchet |
| Geographic top-5 share (×2) | deck, insights | **A** | `geo.exposure_by_itl3` | **NO** | Named in the ratchet |
| Contributor share of movement | insights | **A** | `evolution.funded_bridge` | **NO** | Named in the ratchet |
| Balance change % | insights, watchlist | **A** | `snapshots.monthly_change` | **NO** | Named |
| WA LTV change pp | insights, watchlist | **A** | `evolution.funded_evolution` | **NO** | Named |
| Share-change pp | `movement` | **A** | `evolution.funded_bridge` | **NO** | Named |
| Cohort average balance | `cohorts` | **A** | cohorts service | **NO** | Named |
| Cohort periods observed | deck | **B** | deck context | **NO** | Not economic; a count of what was drawn |
| Materiality classification | `materiality` | **B→A** | engine, structured | **NO** | See §6 — deferred with reason |
| Materiality floor (0.5%) | `movement` | **B** | governed config | **NO** | A threshold, not a measure |
| Leg dominance floor (0.45) | deck | **B** | presentation | **kept** | Decides whether to say a word |
| Bias floor (0.5pp) | `forecast_accuracy` | **B** | presentation | **kept** | Same — documented as such |
| Spread floor (99.5%) | deck | **B** | presentation | **kept** | A suppression decision |
| Retention ×100 | deck | **C** | presentation | **kept** | Formatting |
| Percent fraction→points | `cohorts`, `render` | **C** | presentation | **kept** | Formatting |
| `compact_currency` /1e9,/1e6,/1e3 | `metric_resolver` | **C** | presentation | **kept** | Formatting |
| Percent ×100 | `metric_resolver` | **C** | presentation | **kept** | Formatting |
| Axis tick scaling | `render` | **C** | presentation | **kept** | Formatting |
| Cohort retention fallback | `cohorts` | **D** | — | **NO** | Service already emits it; the fallback should be deleted |
| `movement.reconciles` | `movement` | **D** | — | **NO** | The engine already asserts this |
| Waterfall residual check | `chart_resolver` | **D** | — | **NO** | Same |

**Totals: 9 migrated to the engine · 4 confirmed as presentation semantics and
kept · 7 formatting, kept · 3 redundant, identified for removal · 11 classified
A and named but not yet migrated.**

---

## 6. Engine migrations

### Composition (`mi_agent_api/portfolio_context.py`)

`balance_share`, `opening_from_movement`, `type_composition`. A share of nothing
returns `None` rather than zero — zero asserts a measurement, `None` says one
was not made.

### Forecast (`mi_agent_api/evolution.py`)

`forecast_error_pct` per period; `forecastAccuracy` (`observations`, `biasPct`,
`errorPct`, `worstPct`, `worstPeriod`) over them. Two observations minimum: one
period in which a forecast happened to be close is luck.

### Concentration (`mi_agent_api/concentration_tests_api.py`)

`direction_of_travel` and `stress_effect`, both taking the governed `operator`.
The direction travels on the history payload as `direction`, with `priorValue`
and `priorPeriod`.

### Materiality — classified A, DEFERRED

The brief's Group 4 asks for driver classification to move upstream as
structured output. It is genuinely reused (movement, stock, insights) and the
existing `mi_agent_api/materiality.py` **already returns structured values**, not
prose — `shape`, `leader`, `leader_share`, `separation`. It already lives in
`mi_agent_api`, so it is not in the presentation layer.

What is *not* done is threading it through the governed payloads so React reads
the same classification. That is a payload-contract change across four services
and is deferred rather than half-done. **Not blocked; not started.**

### The ownership ratchet

Behaviour tests alone would let arithmetic drift back, so each migration is also
pinned structurally:

- no presentation module may divide by a `total_bal` / `total_balance`;
- `forecast_accuracy.py` may not divide or aggregate;
- `concentration.travel` / `stress_note` may not contain a comparison.

Six share-shaped divisions remain, each named with its intended owner. **The
count may only go down.**

---

## 7. Key-measure definitions

Measured on the QA fixture (318 loans):

```
avg loan balance   (unweighted mean)         329,438.49
WA property value  (balance-weighted mean)   934,923.18
WA current LTV     (balance-weighted mean)      50.4124 %

avg_balance / WA_property                       35.2370 %   -15.18 pp from the tile
ratio of aggregates  ΣB / ΣV                    41.2146 %    -9.20 pp from the tile
```

### Why they legitimately do not tie

1. **Weighting.** Average loan balance is unweighted — one vote per loan.
   Property value is balance-weighted — one vote per pound. Averages over
   different populations do not divide into one another. Worth ≈6.0pp.
2. **Average of ratios vs ratio of averages.** The LTV tile is the mean of each
   loan's own LTV — the typical *pound's* gearing. Dividing the money tiles
   gives the ratio of the aggregates — the *book's* gearing. Different economic
   statements, separated by Jensen's inequality. Worth ≈9.2pp.

### What was done

**Weighted average LTV is NOT redefined**, and a test pins the definition so
nobody later closes the gap by turning it into a ratio of aggregates.

Every derived measure now carries `basis`, `numerator` and `denominator` through
the governed KPI payload and onto the tile:

| Measure | Basis | Numerator | Denominator |
|---|---|---|---|
| Average loan balance | per loan, unweighted | Σ balance | loan count |
| WA current LTV | balance-weighted | Σ (loan LTV × balance) | Σ balance |
| WA original LTV | balance-weighted | Σ (loan original LTV × balance) | Σ balance |
| WA property value | balance-weighted | Σ (valuation × balance) | Σ balance |
| WA interest rate | balance-weighted | Σ (loan rate × balance) | Σ balance |
| WA months on book | balance-weighted | Σ (months × balance) | Σ balance |
| WA youngest age | balance-weighted | Σ (age × balance) | Σ balance |
| Single borrowers | share of loans, unweighted | single-borrower loans | loans with a known type |
| **Aggregate gearing (book LTV)** | ratio of aggregates | Σ balance | Σ valuation |

**Aggregate gearing is new as a surfaced measure, not as a primitive** — it is
Σ balance / Σ valuation over aggregates the snapshot already computes. It gives
the reader who wants the book's LTV a named place to find it, instead of
inferring it from two tiles that were never meant to divide.

The same split inside the cohort payload (`wa_ltv` a mean of ratios,
`nneg_headroom_pct` a ratio of aggregates) now declares both bases.

---

## 8. Stage reconciliation

`evolution.pipeline_stage_movement` — a composition of two prepared weekly
extracts joined on the governed case identifier.

```
opening live + arrivals - departures ± amount change on persisting = closing live
```

**Every live stage reconciles to a residual of 0.00.** Verified two ways: the
engine's own `residual` field, and a test that recomputes the stated identity
from the reported legs.

Departures are split by **where the case went** — on to another stage,
completed, withdrawn, or absent from the extract — because leaving a stage and
leaving the pipeline are different events.

**Not yet surfaced on a slide.** The engine capability is built and tested; the
presentation is deferred (§16), per the brief's own ordering.

---

## 9. Conversion

**Computable and proved by the audit (Offer → Completion 83.3%), and NOT
surfaced in this sprint.** The engine now has the case-level join that conversion
needs — `pipeline_stage_movement`'s `departuresByDestination` is the numerator —
but a conversion *rate* needs a declared window and lag treatment, and the brief
is explicit that stock ratios, flow ratios, cohort conversion and lagged
conversion must not be conflated. Choosing that definition is analytical work,
not surfacing, and it is deferred rather than approximated.

The existing funnel figure remains a **stock ratio**, and the pack already says
so on the page (added in the previous sprint).

---

## 10. Cohort capability surfaced

**None new in this sprint.** The audit established seven available measures; the
four upgrades (LTV migration, NNEG headroom, exit rate, survival) are
presentation work, which the brief orders last. What *was* done is corrective:
`nneg_headroom_pct` and `wa_ltv` now declare their differing bases, so the two
LTV-shaped figures already on the cohort page cannot be read as complements.

---

## 11. Concentration capability surfaced

**Direction of travel, now engine-owned and correct for minimum-type tests.**

The presentation layer's old rule read direction off the number, which
**inverted every `min` test**. The engine has supported minimum tests all along,
so this was a live defect waiting for a client to approve one. No new client
limits were introduced, and only the operator-approved subset is rendered.

---

## 12. Paired-dimension capability

**Not touched.** The audit established the generator is already generic over
eleven dimensions and that `MULTIDIM_PAIRS` should move to config. That is
presentation-selection work and is deferred (§16). No new grouping engine was
added, and none is needed.

---

## 13. New-build challenge

### Two limits on one exposure — **DEFERRED**

| Question | Answer |
|---|---|
| What does it uniquely answer? | "Am I inside my hard cap *and* my lower converted-pipeline threshold?" |
| Can the base pack answer adequately today? | **Yes.** Two approved tests on the same exposure already render independently, each with its own limit, direction and headroom. What is missing is only their *visual pairing*. |
| Required for Client 1? | No. No approved configuration carries a paired limit. |
| Materially improves the starter pack? | No — it improves a pack for a client who has that covenant structure. |
| New primitive? | It would change `ActiveTest`'s shape (a second threshold, or a declared relationship between two tests) — a governed model change, not a renderer convenience. |

**Deferred.** Build it when a client approves a paired limit, so the shape is
designed against a real covenant rather than an imagined one.

### Region bubble aggregate — **DEFERRED**

| Question | Answer |
|---|---|
| What does it uniquely answer? | "Which regions combine size with gearing?" |
| Can the base pack answer adequately today? | **Largely.** The regional stratification gives balance by region and the LTV × region cross-tab gives the gearing dimension. The bubble compresses two existing pages into one; it does not reveal anything neither shows. |
| Required for Client 1? | No. |
| Materially improves the starter pack? | Marginally, and it adds a chart form the pack does not otherwise use. |
| New primitive? | A one-dimensional multi-measure aggregate — modest, but genuinely a new engine output. |

**Deferred**, per the brief's stated default. The cost is a new engine output
and a new chart idiom for a question two existing pages already answer.

---

## 14. React / PPTX parity

The migrated values now live where **both** channels can read them:

| Value | Home | Reachable from React |
|---|---|---|
| Composition share / opening | `mi_agent_api.portfolio_context` | Yes — same module React's context service uses |
| Forecast error / bias / worst miss | `evolution.forecast_evolution` payload | Yes — served by `/mi/evolution/forecast` |
| Limit direction / stress effect | `concentration_tests_api` history payload | Yes — served by `/mi/concentration-tests/history` |
| Measure basis | governed KPI tile payload | Yes — every `/mi/snapshot` KPI carries it |
| Live pipeline stock | `pipeline_contract` snapshot payload | Yes |

No PPTX-only or React-only economic value was created. The presentation layer
retains layout, chart selection, titles, prose and formatting — and the
ownership tests enforce that boundary structurally.

---

## 15. Regression

<!-- REGRESSION -->

---

## 16. Deferred work

Genuine scope, deliberately not done, in the brief's own priority order:

| Item | Why |
|---|---|
| 11 remaining A-classified derivations | Each belongs to a different analytical domain and is its own migration. Named and ratcheted. |
| Materiality through the governed payloads | A payload-contract change across four services. Not started rather than half-done. |
| Stage-reconciliation **slide** | Engine built and proved; presentation is last in the brief's order. |
| Conversion **rate** definition and surfacing | Needs a declared window and lag treatment — analytical, not surfacing. |
| Cohort depth (LTV migration, NNEG, exits, survival) | Presentation. |
| Concentration approved-subset rendering breadth | Presentation. |
| `MULTIDIM_PAIRS` → config, and new pairs | Presentation selection. |
| Executive / Key Measures / Funded Stock layout items | Presentation. |
| Delete the dead aggregation engine in `metric_resolver` | Identified in the audit; unreachable, so not urgent. |

---

## 17. Merge recommendation

**YES**, with one reviewer condition.

The pipeline stock correction is a production MI defect fix with a measured
before and after, proved consumers, and thirteen targeted tests. The ownership
migrations reduce duplicate formulas and are enforced structurally. The
methodology claim is now weaker than it was and *matches the evidence*, which is
the right direction for a client-facing document.

**The condition:** §4 withdraws an audit finding that was mine. A reviewer
should confirm they agree the pipeline contract needs no new alias — the
evidence is that the real lender fixture maps cleanly and that RREL1 identifies
a funded exposure rather than a pre-funding case — because the alternative is a
one-line config change that would be wrong for the right-looking reason.
