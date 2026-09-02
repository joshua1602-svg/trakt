# Autonomous Portfolio Review / Teams Analytics — Final Production Red-Team

Adversarial stress test before Client 1 activation. The objective was to make
the proactive Teams analytics state something materially false, misleading,
overconfident or economically nonsensical while every governed calculation
behaved as designed.

**It succeeded five times.** All five are fixed with regressions that fail on the
prior commit. Production delivery remains OFF and no recipient was populated.

---

## A. Executive verdict

# CONDITIONAL GO

**GO** for the deterministic weekly/monthly Teams brief, under the operational
controls in §I.

**NO-GO** for the autonomous Portfolio Review controller, on evidence grounds
rather than on any defect found: **it could not be exercised against a real
model in this environment** (§C). Its investigation quality is therefore
unproven, and it must not drive anything a user reads until it is.

The five defects matter because none of them was an arithmetic error. Every
governed calculation was correct; the failures were in what the card *said*
about correct numbers. Four of the five were confidently wrong rather than
merely incomplete, which is the more dangerous class.

---

## B. Test evidence

Evidence classes: **A** pipeline-produced canonical · **B** existing
simulation/generated portfolios · **C** purpose-built adversarial fixture.

| # | Scenario | Ev. | Production path | Result | Issue | Fix |
|---|---|---|---|---|---|---|
| 1 | Quiet month | C | resolver→generators→card | **PASS** | none; no manufactured insight, 2 bullets | — |
| 2 | Strong organic growth | C | full | **PASS** | attributed as organic, no acquisition language | — |
| 3 | Acquisition-dominated | **A** | full | **PASS** | real multibook canonical, §F | — |
| 4 | Balance jump, no acquisition | A+C | full | **PASS** | zero acquisition attribution | — |
| 5 | Acquisition + organic deterioration | C | full | **FAIL→FIXED** | "£50.0m of the £5.0m movement"; underlying LTV +12pp unreported | D2, D1 |
| 6 | Acquisition improves headline risk | C | full | **FAIL→FIXED** | "LTV moved 30.0%→29.0%" on a book that deteriorated 8pp | D1 |
| 7 | Acquisition causes breach | C | full | **PASS** | breach leads, `concern`, acquisition named as context | — |
| 8 | Small £, major risk move | C | full | **PASS** | +0.2% balance, warning still leads | — |
| 9 | Large £, immaterial risk | C | full | **PASS** | no manufactured risk language | — |
| 10 | Missing concentration evidence | C | **real resolver** | **PASS** | P0 fix verified through `sources.resolve` | — |
| 11 | Partial analytical evidence | C | full | **PASS** | bounds what ran, names it, not an error dump | — |
| 12 | Unknown / missing product | C | full | **FAIL→FIXED** | card said "no material developments" on +40%; null rows dropped from denominator | D3, D4 |
| 13 | Pipeline grows, quality changes | C | movement→generators | **PASS** | volume not equated with performance | — |
| 14 | Pipeline shrinks — cases funded | C | full | **PASS (limited)** | not called deterioration; cause not inferred — see H3 | — |
| 15 | Pipeline shrinks — fallout | C | full | **PASS (limited)** | identical narrative to 14; governed data cannot separate them — see H3 | — |
| 16 | New portfolio, unusual chars | **A** | decomposition | **PASS** | real ids `alp_acquired`/`spv1_sponsored`, no prefix dependence | — |
| 17 | Third/fourth acquired book | C | decomposition | **PASS** | data-only, §G | — |
| 18 | Tiny portfolio denominators | C | full | **FAIL→FIXED** | "no material developments" on −45% | D3 |
| 19 | Extreme outlier | C | full | **FAIL→FIXED** | WA LTV +21.7pp beside "no material portfolio risks" | D5 |
| 20 | Reversal (breach→pass) | C | full | **PASS** | resolved breach reported, not described as current | — |
| 21 | Multiple simultaneous developments | C | full | **PASS** | ranked not dumped: 5 items, limit→acquisition→LTV→underlying | — |
| 22 | Conflicting signals | C | full | **PASS** | mixed picture preserved; no forced overall story | — |
| 23 | No prior period | C | `resolve_period` | **PASS** | no fabricated comparison; explicit unavailable | — |
| 24 | Duplicate reporting period | C | **trigger→outbox** | **PASS** | deterministic batch id suppresses second send | — |
| 25 | Wrong-period selection | C | `resolve_period` | **PASS** | 5 snapshots; latest→(07,06); pinned→(05,04), not most-recently-ingested | — |
| 26 | Multi-client isolation | C | `resolve_period` | **PASS** | A and B resolve only their own; unknown client borrows nothing | — |

### The five defects

| | Defect | Class (§12) | Root cause | Fix |
|---|---|---|---|---|
| **D1** | An acquisition masked the incumbent book. Combined-population LTV published "30.0%→29.0%" — an improvement — on a month the incumbent book deteriorated 30%→38%. | GOVERNED ANALYTIC | mix and weighted LTV measured on the combined population; an arriving book rewrites both by construction | characteristic movement read on portfolios present in **both** periods, labelled, combined stated beside it, conflict named; material if **either** population moved |
| **D2** | "£50.0m of the £5.0m movement reflects the acquisition." Share = 1000%. | NARRATIVE | `dominant_addition` divided by a net movement smaller than the addition — the ordinary shape of an acquisition alongside redemptions | share withheld rather than formatted; addition stated against the book with offsetting components named |
| **D3** | Card said "No material developments were identified" directly under a lead sentence reporting −45% / +40%. | NARRATIVE | gated on what the card *printed*; a headline-only month prints no bullet because the lead sentence already says it | gated on the insight set |
| **D4** | `_group_balance` dropped null rows from numerator **and** denominator: on £140m with £90m unset product, published Lump Sum as 100% where truth was 35.7%. **Pre-existing**; reaches `funded_bridge`, whose stated property that per-category deltas sum exactly to the net change was false wherever a dimension had nulls. | GOVERNED ANALYTIC | `astype(str)` leaves NaN as NaN, so the Unknown mask never fired | `isna()` tested before string conversion |
| **D5** | WA LTV +21.7pp in one month, beside "No material portfolio risks were identified." | MATERIALITY | funded LTV graded informational unless an underlying comparison existed | a **rise** past the already-configured floor is a risk finding, a fall is not — no new threshold, ranked below anything contractual |

All five were fixable locally. **No architectural change was required and none was
made.** Ingestion, canonical, provenance, OCC, snapshot storage, the MI Query
Agent, concentration/risk ownership, Teams delivery, readiness-agent governance,
client identity and A2A are untouched.

---

## C. Real-model agent evaluation — NOT PERFORMED

**§6 could not be executed.** This environment has no model credentials:

```
ANTHROPIC_API_KEY : not set
live call         : 401 invalid x-api-key  (api.anthropic.com)
```

The SDK installs, and the controller runs correctly against a scripted model
(20 tests), but that exercises the *controller*, not the model's judgement.
Nothing in this report should be read as evidence about the autonomous agent's
investigation quality. Specifically **unproven**:

- whether it begins with appropriate headline analyses;
- whether it drills into genuinely material developments;
- whether it recognises an acquisition from governed evidence rather than size;
- whether it separates combined from underlying performance;
- whether it wastes calls, stops appropriately, or repeats itself;
- whether it attempts arithmetic or states anything absent from tool evidence;
- whether its ranking is economically sensible;
- repeatability across runs (§11).

What **is** structurally established, and holds regardless of model:

- the session exposes three verbs and no DataFrame — the model cannot compute
  over data because it never receives any (`test_the_session_hands_the_model_no_frame`);
- the period is resolved from governed discovery and pinned in the prompt, so a
  review cannot drift to a different period;
- the opening message carries dates and **no figures**, so every number the model
  states must have come from a tool call;
- findings carry the tools they rest on; one citing a tool it never called gets
  no evidence rather than the whole transcript;
- the loop is bounded at 24 steps.

**This is the single reason the verdict is CONDITIONAL rather than GO.**

---

## D. Deterministic vs autonomous comparison — NOT PERFORMED

Requires §C. No claim of incremental analytical value from the autonomous layer
is made or supported by this exercise.

One observation that survives without a model: the deterministic brief found
every material development in scenarios 1–26 **after** the five fixes. The
autonomous layer's value proposition is therefore the drill-downs the fixed
generator set cannot anticipate — which is exactly what the shadow evidence in
§I step 6 must demonstrate before it is enabled.

---

## E. Unsupported-claim audit

Every sentence of the busy acquisition month, treated as a claim:

| Sentence | Claim type | Governed source | Supported |
|---|---|---|---|
| "Funded balance is £155.0m, +£55.0m on the month." | quantitative | `period_movement.current/delta` | yes |
| "Loan count is 4 (+1 on the month)." | quantitative | `period_movement` | yes |
| "London exposure deteriorated from pass to warning… Utilisation is 97.0%… Headroom +0.9pp." | risk status | approved concentration config: `statusTransition`, `deteriorated`, `utilization`, `headroom` | yes |
| "The acquisition of Portfolio B added £60.0m, against a net movement of +£55.0m." | attribution | `funded_composition.portfolio_additions`, identity-resolved | yes |
| "The net is smaller than the addition because £30.0m of redemptions and exits offset it." | causal-sounding | arithmetic identity of the partition, not an inference | yes |
| "Excluding portfolios added this period, LTV moved 30.0%→36.0% (+6.0pp). Including them the combined book moved 30.0%→30.6% (+0.6pp)." | quantitative, two populations | `_weighted_ltv_points` over lens-scoped frames | yes |
| "Excluding the portfolio added this period, the existing book decreased by £5.0m (−5.0%)." | quantitative | `funded_composition` under the underlying lens | yes |

**Dangerous-language scan** over the same card — hits and their justification:

- **"deteriorated"** — verbatim from the governed `deteriorated` / `statusTransition`
  fields of the approved configuration. Not a judgement added by the narrative.
- **"acquisition"** — only where `portfolio_type == acquired` from governed
  identity. An `unclassified` addition is called "the addition of the source
  portfolio" (tested).
- **"because"** — the one instance is an arithmetic identity of the
  decomposition, not a causal claim about the business.

**Not present anywhere:** "driven by", "primarily", "significant", "stable",
"no risk", "all limits", "forecast", "expected", "organic". The word
"underlying" appears only as "excluding portfolios added this period", which
states its own definition.

**Claims found unsupported before the fixes:** D1 (an improvement that did not
happen), D2 (a share that cannot exist), D3 (an absence of material developments
that contradicted the same card), D5 (an absence of risk beside a 21.7pp LTV
rise). All four are now impossible by regression.

---

## F. Acquisition-month evidence — pipeline-produced canonical

From `synthetic_demo/output/multibook/`, two consecutive platform canonicals the
pipeline actually produced. The acquisition month is built by *removing* the
acquired book from the prior period, so every loan, balance and date is the
pipeline's own output.

```
opening balance                      24,444,963.43
  + portfolio additions              11,974,544.28   alp_acquired, 37 loans
  + organic new lending                 701,227.12
  + existing-book movement              149,326.64
  - exits                                    -0.00
closing balance                      37,270,061.47
                                     ─────────────
movement                             12,825,098.04
reconciles                                    True   residual 0.0
```

- **Identity, not size:** `alp_acquired` carries no `direct_`/`acquired_` prefix,
  so `derive_portfolio_type` returns `None` and classification comes from the
  `source_portfolio_type` column. `acquisition_date` `2024-09-30` read from the
  canonical.
- **Dominance:** 93.37% of the movement — a share that is quotable here because
  the movement exceeds the addition.
- **Combined vs underlying:** headline **+52.5%**, underlying **+3.48%** over
  `alp_origination` + `spv1_sponsored`, reconciling independently.
- **Ordinary month, full precision:** the same pair without the removal
  reconciles to residual `0.0` over 115 held loans with capitalised interest,
  three arrivals and one exit.

---

## G. Scale evidence

- A third and fourth acquired book (`portfolio_gamma`, `portfolio_delta`) are
  added by changing the frame alone — no module edited, no id registered, no
  branch added; additions total £104.0m and reconcile.
- A source scan holds the five new production modules to naming **no portfolio
  id at all** (`test_no_production_module_names_a_portfolio_id`).
- `test_an_explicit_type_column_beats_the_id_prefix` covers a client whose ids
  follow their own convention: the column is the primary authority, the prefix
  only a fallback — and the real canonical in §F is exactly that case.
- Period resolution is per-client: two clients resolve only their own runs, and
  an unregistered client borrows nothing (§B.26).

---

## H. Residual risks

| | Risk | Class |
|---|---|---|
| **H1** | **The autonomous controller has never run against a real model.** Investigation quality, stopping behaviour, arithmetic abstention and repeatability are all unproven. | **GO-LIVE BLOCKER** *(for the controller only)* |
| **H2** | Board vs operational permissions are unmodelled. `portfolio_contexts` is a portfolio scope, not a role; both user types authorised for `total` receive an identical card. | **GO-LIVE BLOCKER** *(for board users only)* |
| **H3** | A pipeline reduction from completions and one from fallout produce identical narratives. Neither claims a cause, so no false statement is made — but the reader cannot tell them apart. Governed linkage does not currently support the distinction where cases leave the extract entirely. | **SHADOW-MONITOR** |
| **H4** | `completions_movement` gates on percentage change, so completions rising from zero produce `change_pct = None` and are omitted. A lumpy book's first completion week is silently dropped. Pre-existing; an omission, not a false claim. | **POST-GO-LIVE** |
| **H5** | `movement_summary.period_movement` iterates the current frame's cohorts, so a **disposed** portfolio lands in its residual rather than as a component. `funded_composition` reports disposals correctly; the older service still has the gap. | **POST-GO-LIVE** |
| **H6** | All materiality thresholds are relative by design. On a very small book a single loan is always "material". Scenario 18 now reports it correctly rather than contradicting itself, but a two-loan portfolio will generate a card every month one loan moves. | **SHADOW-MONITOR** |
| **H7** | 21 pre-existing test failures remain on `origin/main` in the MI parser, chat-routing, currency and period-movement suites, plus 15 in the funded-bridge and simulation suites (`KeyError: 'groupedBy'`) and 19 in `test_simulation_*`. None is touched by this work; all verified identical on clean `main`. | **POST-GO-LIVE** |
| **H8** | The full repository suite has never completed in this environment. The targeted regression covers every module touched and every area named, but it is not the whole suite. | **SHADOW-MONITOR** |

---

## I. Activation recommendation

1. **Fix H2 or exclude board users.** Decide whether board and operational users
   receive the same card before either is authorised. This is a policy decision,
   not a code change.
2. **Shadow one full ERE reporting cycle** with `TRAKT_MI_WEEKLY_BRIEF` enabled
   and `recipients: []`. The batch is stored even with no recipient
   (`trigger.py:228`), so the audit record of what *would* have been sent exists
   without anything being delivered.
3. **Manually reconcile every claim** on the shadow cards against the MI
   workspace — specifically the acquisition decomposition, the underlying-vs-
   combined LTV pair, and any risk transition.
4. **Enable one operational recipient** via the existing CLI. Observe one cycle.
5. **Expand to operational users** only after a cycle with no reconciliation
   exception.
6. **Autonomous controller: shadow only, and not before step 5.** Requires model
   credentials this environment does not have. Run it against at least the five
   period types in §6 with full tool traces, and read every trace for the two
   failure modes its prompt guards — adding contributor dimensions together, and
   calling a movement an acquisition. Compare against the deterministic brief per
   §7. **Do not enable it on the strength of prose quality.**
7. **Board users last**, after operational validation and after H2 is resolved.

---

## Regression

Targeted, before and after, same suites on `origin/main` and HEAD:

| | Baseline (`origin/main`) | HEAD |
|---|---|---|
| passed | 4,214 | **4,313** (+99) |
| failed | 22 | **21** |

**New failures introduced: none** — failure lists diffed test by test, `comm -13`
empty. The one difference is `test_registry_governance`, which asserts an
absolute path and therefore fails from any worktree; verified standalone on both
and **not** a fix.

The funded-bridge and simulation suites were checked separately because the
`_group_balance` fix touches existing governed grouping: **15 failures on both
`origin/main` and HEAD, identical sets.** The `_group_balance` correction
introduced none of them.

**Not completed:** the full repository suite (H8).
