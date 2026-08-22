# MI Agent — Commercial Beta Readiness Review

**Mode:** review / measure / report only. No production code, tests or vocabulary were changed.
**Baseline:** `983a755` — clean tree, full suite 8,675 passed / 0 failed.
**Fixture:** `demo_platform` / `alderbridge` (`client_001`) — 11,035 loans, £1,964,886,258.21,
snapshots 2026-04-30 / 05-31 / 06-30, cut-off 2026-06-30.
**Evidence:** 34 curated CFO questions × 2 parser paths, plus targeted probes; 36 genuine
model calls; all substantive figures reconciled independently in pandas.

---

## 1. Executive verdict

The MI Agent is **architecturally ready** for a commercial Beta and **not yet releasable**.

On the production configuration it answered **30 of 34** curated CFO questions correctly and
refused the other 4 safely. Across both parser paths it produced **zero incorrect successful
answers, zero silent semantic errors and zero hard failures** on that set. Every figure that
could be reconciled matched independently-computed truth to the penny or to 4 decimal places.

One defect blocks release. A request for a statistic the underlying field does not permit —
**"what is the median LTV?"** — is silently downgraded to the field's default aggregation and
answered as a success: **43.1562 delivered against a true median of 39.6757**, reproducible
5/5 on the production path. It has been ruled a **Beta blocker** and escalated separately
(`MI_AGENT_ESCALATION_AGGREGATION_SUBSTITUTION.md`).

The honest reading of the 30/34 score is that **the curated set never asked for a median**.
It scored clean because it did not probe the one axis the governance layer does not cover.
That is the single most important sentence in this report.

Nothing else found rises to a blocker. The remaining gaps are breadth, phrasing robustness
and parser parity — and in every observed case the agent **refused rather than guessed**.

### Executive matrix

| Area | Status | What it means commercially |
|---|---|---|
| Portfolio size, balance and loan counts | **Ready** | Headline book figures answer correctly and reconcile exactly |
| Credit and pricing averages (LTV, rate, borrower age) | **Ready** | Weighted averages are correct and consistently applied |
| Direct vs acquired analysis | **Ready** | The two sourcing channels can be compared and analysed independently |
| New lending vs seasoned book | **Ready** | Front book / back book is a governed, configurable business rule |
| Regional and geographic breakdowns | **Ready** | Balance, count and LTV by region, including within a chosen book |
| Concentration and largest exposures | **Ready** | Largest loan, top-5 share and geographic concentration all correct |
| Month-on-month movement | **Ready, with wording sensitivity** | Works; one common phrasing (“since last month”) is not recognised and refuses |
| “What share of the book is…” questions | **Not available** | The agent declines rather than answering; a top-priority gap |
| Median and percentile figures | **Blocked** | Returns the average instead of the median without saying so — must be fixed before release |
| Movement analysis within a chosen sub-book | **Not available** | Declines rather than silently reporting the whole book |
| Credit performance (arrears, defaults, impairment) | **Not possible on this data** | The client's book carries no arrears or loss data |
| Answer quality when the agent cannot help | **Ready** | It says what it could not do and never substitutes a different figure |

---

## 2. What "commercial Beta ready" means

For this review, Beta ready is not "scores well on a question bank". It is four conditions,
all of which must hold:

1. **No incorrect successful answers.** A number presented as the answer must be the answer
   to the question asked. This is the only non-negotiable condition.
2. **Every gap is a visible refusal.** Where the agent cannot answer, it must say so and must
   not substitute an adjacent figure.
3. **The supported envelope is describable to a client** in commercial language, so
   expectations can be set before the first question is typed.
4. **Enough genuine commercial coverage to be worth deploying** — the envelope must contain
   the questions a lending CFO actually asks, not merely the ones that happen to work.

Conditions 2, 3 and 4 are met. Condition 1 is met everywhere the review looked **except** the
aggregation axis, and condition 1 admits no exceptions. Hence: not yet releasable.

---

## 3. Governed architecture summary

A single governed entrypoint, `execute_governed_mi_query`, serves both the React MI Agent and
the Microsoft 365 Copilot action. It is stateless by contract, so the two surfaces cannot
drift apart in what they answer or what they refuse.

Question intent is resolved **once**, before anything is calculated:

```
question
  └─ parse (LLM, with deterministic repair and fallback)
       └─ governed scope        — which portfolios (portfolio lens)
       └─ governed population   — which rows (spec filters)
       └─ facet ledger          — every material thing the question asked for
            └─ route dispatch (one point-in-time path + 13 specialist routes)
                 └─ execution evidence: what was ACTUALLY applied
                      └─ reconciliation: requested vs applied
                           └─ refuse, or answer with a receipt
```

The load-bearing idea is the **facet ledger**. Every material element of a question —
threshold, grouping, geography, comparison period, ranking, share, cohort, row population — is
recorded as requested *before* parsing decisions are taken, then reconciled against what
execution actually did. A facet that names a number or a subject and cannot be honoured causes
a **refusal**; a facet that only shapes presentation causes a **disclosed partial**.

This is why the agent's failure mode is refusal rather than a wrong number — and it is exactly
why the aggregation defect matters: **the ledger has no facet kind for the requested
statistic**, so that one axis has no guard.

---

## 4. Primary commercial capability matrix

Evidence: the 34-question curated set, both parser paths. "Production" = genuine LLM.

| Commercial capability | Deterministic | Production | Evidence |
|---|---|---|---|
| Headline balance / AuM | Refused | **Correct** | C01, C32 |
| Loan count | **Correct** | **Correct** | C02 |
| Average loan size | Wrong shape (distribution) | **Correct** | C03 — £178,059.47 |
| Weighted-average LTV | **Correct** | **Correct** | C04 — 43.1562 |
| Average borrower age | **Correct** | **Correct** | C05 — 71.3976 |
| Weighted-average interest rate | **Correct** | **Correct** | C06 — 6.5597 |
| Direct book analysis | **Correct** | **Correct** | C07, C24 |
| Acquired book analysis | **Correct** | **Correct** | C08 — £579,377,675.23 |
| Direct vs acquired comparison | **Correct** | **Correct** | C09, C10 |
| Front book / back book | **Correct** | **Correct** | C11, C12, C13 |
| Vintage analysis | **Correct** | **Correct** | C14 — 13 vintages |
| Balance by region | **Correct** | **Correct** | C16 |
| Region ranking (static) | Chart only | Chart only | C17 — does not name the winner |
| Largest single-loan exposure | **Correct** | **Correct** | C18 — £842k / 0.043% / top-5 0.20% |
| Concentration | **Correct** | **Correct** | C19 — but see §14 on answer quality |
| Contribution to portfolio LTV | **Correct** | **Refused** | C20 — parity gap |
| Month-on-month movement | Refused on one phrasing | Refused on one phrasing | C21 vs §14 |
| Ranked regional movement | **Correct** | **Correct** | C22 — South East +£7.8m |
| Multi-measure answers | **Correct** | **Correct** | C23, C24 |
| Threshold filters (age, LTV) | **Correct** | **Correct** | C25, C27 |
| Proportion of the book | Refused | Refused | C15, C26 |
| Region × chosen book | **Correct** | **Correct** | C28, C30, C33, C34 |
| Explicit portfolio selection | **Correct** | **Correct** | C31 |
| **Median / percentile** | **Wrong answer** | **Wrong answer** | **BLOCKER — §15** |

## 5. Measure × analytic matrix

Columns are analytics; rows are the five measures the client's book supports.
✅ correct · ⚠️ works with a caveat · ❌ refuses · **🛑 unsafe**

| Measure | Scalar | Group-by | Comparison | Period movement | Contribution | Share of book | Concentration | Median/percentile |
|---|---|---|---|---|---|---|---|---|
| Balance | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ⚠️ correct on production, wrong deterministically |
| Loan count | ✅ | ✅ | ✅ | ✅ | — | ❌ | ✅ | — |
| Current LTV | ✅ | ✅ | ✅ | ✅ | ⚠️ parity | ❌ | ✅ | **🛑** |
| Borrower age | ✅ | ✅ | ✅ | ✅ | — | ❌ | ✅ | ⚠️ correct on production, wrong deterministically |
| Interest rate | ✅ | ✅ | ✅ | ✅ | — | ❌ | ✅ | **🛑** |

The two 🛑 cells are exactly the two fields whose registry entries exclude `median` from
`allowed_aggregations`. The correlation is perfect and is the root cause (see the escalation).

## 6. Population / scope composition matrix

Two independent channels: **scope** (which portfolios — the lens) and **population** (which
rows — spec filters). The commercial question is whether they compose.

| Scope × Population | Result | Evidence |
|---|---|---|
| Whole book × threshold | ✅ | C25 borrowers over 85; C27 London |
| Whole book × seasoning | ✅ | C11, C12 |
| Direct/acquired × geography | ✅ | C28 — 3,909 loans, Source Portfolio in alp_acquired |
| Direct/acquired × seasoning | ✅ | C29 — direct ∩ back book |
| Seasoning × geographic exposure | ✅ | C34 — £75.4m / 4.2%, denominator £1.795bn = the back book, **not** the whole book |
| Explicit UI selection × measure | ✅ | C31 — honours the selected portfolio |
| Explicit widening ("whole book") | ✅ | C32 — overrides a narrower selection |
| Population × specialist movement routes | ❌ **refuses** | §7 — correct behaviour, see below |

The last row is the P1L guarantee holding commercially: nine specialist routes construct their
own frames and cannot honour a row population, so they **refuse** rather than silently
returning the whole book. Verified live for period movement, period change, funded bridge,
risk limits and temporal compare — each returns *"the population seasoning_segment = Back Book
… could not be applied"* rather than a whole-book figure.

## 7. Specialist-route D1/D2/D3 assessment

Thirteen routes are reachable. Four are already population-safe (`evolution` reads filters
directly; `geo_exposure`, `concentration_analysis` and `portfolio_risk_comparison` receive a
pre-filtered frame). The remaining **nine are Class D** — they build their own frames and
currently refuse a population.

Sub-classification used here (operationalised for this review):
**D1** = commercially wanted and reachable without redesigning the analytic ·
**D2** = commercially useful but needs analytic redesign · **D3** = not commercially required
for this deployment.

| Route | Whole-book behaviour | With a population | Class | Reasoning |
|---|---|---|---|---|
| `period_movement` | ✅ ranked regional movement | Refuses | **D1** | "Which region grew most last month in the back book" is a real, frequent CFO question |
| `period_change_analysis` | ✅ +£18.1m month-on-month | Refuses | **D1** | The core movement question; a sub-book version is the natural follow-up |
| `funded_bridge` | ✅ £1.93bn → £1.96bn, +£32.6m | Refuses | **D1** | Bridges are how a CFO explains movement; per-book bridges are standard MI |
| `temporal_compare` | ✅ May → June +0.93% | Refuses | **D1** | Same family; low marginal cost once the D1 group is done |
| `risk_limits` | ✅ 8 passed, 1 breach | Refuses | **D2** | Limits are *defined* on a scope; a sub-book limit test is a governance decision, not a filter |
| `cohort_progression` | Not reachable by natural phrasing | — | **D2** | Needs vocabulary before it needs population support |
| `forecast_extrapolation` | Not reliably reachable (see §14) | Refuses | **D2** | Forecasting a sub-book requires its own governed series |
| `scenario` | Not reachable | — | **D3** | Stress/scenario analytics are out of product scope by standing ruling |
| `cohort_conversion` | "No governed pipeline data supplied" | Same | **D3** | This client supplies no pipeline data; nothing to unlock |

The four D1 routes share one seam and are the natural content of a single post-Beta increment.

## 8. Curated commercial CFO question set

34 questions, written as a lending CFO or portfolio manager would ask them, restricted to
measures and dimensions **this book actually carries**. No question requires an absent field
for the sake of difficulty. This is deliberately *not* the 40-question adversarial bank, which
is an evaluation set and not a statement of product requirements.

Families: portfolio size (3), credit / collateral (1), borrower (1), pricing (1),
provenance (4), seasoning (3), vintage (1), geography (3), exposure & concentration (3),
contribution (1), period movement (2), multi-measure (2), filtered (3), combined (5),
scope & selection (2). One question (C31) runs against an explicit UI portfolio selection.

## 9. Commercial question test results

| Outcome | Deterministic | Production (genuine LLM) |
|---|---|---|
| Correct | 29 | **30** |
| Correct with disclosed limitation | 0 | 0 |
| Safe refusal | 5 | 4 |
| **Incorrect successful** | **0** | **0** |
| **Silent semantic error** | **0** | **0** |
| **Hard failure** | **0** | **0** |

Path divergences (each reproducible 3/3):

| Q | Deterministic | Production | Reading |
|---|---|---|---|
| C01 "How large is the funded portfolio?" | Refused (unmapped) | **Correct** | Deterministic vocabulary gap on a very basic question |
| C03 "What is the average loan size?" | Distribution by ticket size | **Correct** (£178,059.47) | Deterministic drops the "average" intent |
| C30 "Balance by region for the front book" | Refused ("heatmap requires two dimensions") | **Correct** | Deterministic treats the segment as a second axis |
| C20 "Which region contributes most to portfolio LTV?" | **Correct** | **Refused** | **Production path is the weaker one here** — a parity gap in the wrong direction |

The four production-path refusals — C15 and C26 (proportion of the book), C20 (contribution)
and C21 (one phrasing of month-on-month) — are all safe and all explanatory.

## 10. Genuine-LLM acceptance evidence

Genuine model calls throughout; a deterministic fallback would not have counted. Provenance
was captured **at the parse seam**, not from the routed envelope, because the envelope carries
the route's metadata and understates model involvement.

| Metric | Value |
|---|---|
| Genuine model calls, curated set | **36** |
| `llm` (clean first-pass parse) | 31 |
| `llm_repaired` (one governed repair) | 2 |
| `validation_failed` (spec rejected, refused safely) | 1 |
| Questions needing 2 calls | 2 of 34 |

**Repeated runs on the highest-risk questions** — 5 runs each, genuine LLM:

| Question | Distinct outcomes | Result |
|---|---|---|
| "What is the median LTV?" | **1 of 5** | Identical wrong answer every run — a stable defect, not model variance |
| "What is the median LTV in the back book?" | **1 of 5** | Identical; 43.9657 against true median 40.6689 |
| "What is the median loan balance?" | **1 of 5** | Identical and **correct** (156,864.66) |
| "Which region contributes most to portfolio LTV?" | 1 of 3 | Refuses consistently |
| "How has the portfolio balance changed since last month?" | 1 of 3 | Refuses consistently |

The determinism is itself the finding: none of these are stochastic. The model is not the
source of the defect, and re-prompting will not fix it.

## 11. Independent truth reconciliation

Truth computed directly in pandas from the fixture. The production implementation was not used
to validate itself.

| Question | Delivered | Independent truth | Variance |
|---|---|---|---|
| C02 loan count | 11,035 | 11,035 | 0 |
| C04 weighted-average LTV | 43.1562462674 | 43.1562462674 | 0 |
| C05 average borrower age | 71.3975532397 | 71.3975532397 | 0 |
| C06 weighted-average rate | 6.5597233425 | 6.5597233425 | 0 |
| C08 acquired balance | £579,377,675.23 | £579,377,675.23 | 0 |
| C11 back-book LTV | 43.9656614504 | 43.9656614504 | 0 |
| C12 front-book balance | £171,736,116.72 | £171,736,116.72 | 0 |
| C18 largest exposure | £842k / 0.043% / top-5 0.20% | £841,638.96 / 0.04283% / 0.20345% | 0 (rounded display) |
| C24 direct book multi-measure | £1.39bn / 7,126 / 43.35% | £1,385,508,582.98 / 7,126 / 43.3535 | 0 |
| C32 whole book | £1.96bn / 11,035 | £1,964,886,258.21 / 11,035 | 0 |

Judgement-classified answers, reconciled by hand:

| Question | Claim made | Verified |
|---|---|---|
| C09 | "Direct has higher LTV than acquired" | ✅ 43.3535 > 42.6846 |
| C10 | Direct higher on balance and count | ✅ £1.386bn > £0.579bn; 7,126 > 3,909 |
| C20 | South East contributes 11.34 of 43.16; 26.3% of book at 43.14 | ✅ 11.3340 / 43.1562 / 26.27% / 43.1412 |
| C22 | South East £508.4m → £516.2m (+£7.8m, +1.5%) | ✅ matches the two snapshots |
| C34 | Westminster £75.4m, 4.2% "of the book" | ✅ denominator £1.795bn = **the back book**, confirming population propagation |

C34 is the commercially important one: it proves a governed sub-population survives all the way
into a specialist geographic route, with the correct denominator.

## 12. Commercial Beta scorecard

| Dimension | Score | Basis |
|---|---|---|
| Answer correctness (within envelope) | **A** | 30/34 production; every reconciled figure exact |
| Safety — no wrong numbers | **F on one axis, A elsewhere** | Aggregation substitution; nothing else |
| Refusal quality | **A** | Names what was asked, what was not done, and that nothing was substituted |
| Scope / population governance | **A** | Composes correctly; specialist routes refuse rather than widen |
| Commercial breadth | **B** | Strong on stock analysis; share-of-book absent; movement is whole-book only |
| Parser path consistency | **C** | Four divergences, one where production is the weaker path |
| Phrasing robustness | **C** | "since last month" refuses while three equivalent phrasings work |
| Answer presentation | **B−** | Two malformed labels; ranking answered as a chart; degenerate concentration output |
| Learning loop | **D** | Instrumented in-process, captured nowhere |
| Architectural readiness | **A** | One governed entrypoint, one resolution point, evidence-based reconciliation |

## 13. Supported Beta envelope

The envelope a client can be told about, in their language:

**Supported today**
- Total balance, loan count and average loan size for the book or any selected portfolio
- Weighted-average LTV, weighted-average interest rate and average borrower age
- Direct (originated) versus acquired analysis, on any of the above
- New lending versus seasoned book ("front book" / "back book"), on a governed 12-month
  boundary that is configuration-driven and can be set per client
- Origination vintage analysis
- Breakdowns by region, vintage, seasoning band and ticket size
- Combinations of the above — e.g. "balance and loan count by region for the acquired book",
  "average LTV of the back book within the direct book"
- Largest single-loan exposure, top-5 concentration and geographic concentration
- Month-on-month movement, including which region moved most
- Multiple measures in one question
- Threshold filters on borrower age and LTV
- Concentration limit testing against the client's governed limits

**Explicitly not supported in Beta** (the agent declines, it does not guess)
- Any median, percentile, quartile, standard deviation or spread
- "What proportion / how much of the book is…" questions
- Movement or bridge analysis within a chosen sub-book
- Forecasting, scenario and stress analysis
- Arrears, default, impairment or any credit-performance measure

## 14. Current limitations

Ordinary Beta limitations — none is a blocker, all fail safe.

| # | Limitation | Class | Detail |
|---|---|---|---|
| L1 | Share-of-book questions refuse | Breadth | C15, C26. The absolute figure is computed but cannot be expressed as a proportion, so the answer is withheld. Commercially the most-missed capability. |
| L2 | "Since last month" is not recognised | Phrasing | C21 refuses, while "month on month", "movement in balance last month" and "change between May and June" all correctly return +£18.1m. Capability present, phrase missing. |
| L3 | Movement within a sub-book refuses | Breadth (D1) | Correct behaviour under P1L, but a real commercial want. |
| L4 | Contribution refuses on the production path | Parity | C20 answers correctly deterministically and refuses on LLM. The weaker path is the one clients will use. |
| L5 | Static ranking returns a chart, not a winner | Presentation | C17 "which region has the highest average LTV" returns 12 groups without naming one. Contrast C22, which names South East. |
| L6 | Concentration output leads with degenerate dimensions | Presentation | C19 opens with "amortisation type … 100% of exposure" and "collateral type … 100%" — true but uninformative for a single-product book. |
| L7 | Deterministic path is materially weaker | Parity | C01, C03, C30 fail deterministically and succeed on production. Matters only if the LLM is unavailable — but that is exactly when it matters. |
| L8 | Malformed measure label | Presentation | "Calculated: Count of ·" with nothing following (deterministic, ticket-size questions). |
| L9 | Parser invents places from time phrases | Robustness | "Forecast the balance for the next three months" produced a filter on a region called "Next Three Months". It refused safely ("no loans match"), but the mechanism is wrong. Same family as a previously-recorded case. |
| L10 | Raw internal error text can reach the user | Presentation | Percentile/stddev/IQR questions surface `Execution failed: 'distribution' is not a scalar aggregation` instead of a governed refusal. |

## 15. Beta blockers

**One blocker.**

### B1 — Silent aggregation substitution (ruled BETA BLOCKER)

A request for an aggregation the field's registry does not permit is coerced to the field's
default aggregation and answered `ok=True`, with no facet, no warning and no disclosed
limitation.

| | |
|---|---|
| "What is the median LTV?" | delivers **43.1562**, true median **39.6757**, error **+8.77%** |
| "What is the median LTV in the back book?" | delivers **43.9657**, true median **40.6689** |
| "What is the median loan balance?" (deterministic) | delivers **£1,964,886,258.21**, true median **£156,864.66** |
| Reproducibility | 5/5 production, 5/5 deterministic |

Root cause: `allowed_aggregations` is enforced by **downgrading rather than declining**. The
executor computes median correctly when actually asked — the failure is in the permission
path. The facet ledger has no facet kind for the requested statistic, so the loss leaves no
evidence. Max/min *are* governed and refuse correctly; median and percentile are not.

Ruled semantics for the fix: **refuse** a disallowed aggregation using the governed refusal
template, **and widen** the deterministic parser to emit `median` where the registry already
permits it, so the two paths agree. A refusal-only change would leave the deterministic path
refusing questions the production path answers correctly.

Full detail, evidence and reproduction: `MI_AGENT_ESCALATION_AGGREGATION_SUBSTITUTION.md`.

## 16. Top five breadth unlocks

Ranked by commercial value × frequency × effort × semantic risk. Deliberately excludes
anything ruled out of product scope.

| # | Unlock | Value | Frequency | Effort | Semantic risk | Why it ranks here |
|---|---|---|---|---|---|---|
| 1 | **Share of the book** — "what proportion of the book is X" | High | Very high | Low | **Low** | The denominator is already governed (the resolved scope), the numerator is already computed. The agent refuses today only because it cannot *express* the ratio. Best value-to-risk ratio available. |
| 2 | **Governed statistic axis** (median, and percentile where meaningful) | High | Medium | Medium | Medium | Closes B1 properly rather than by refusal alone, and turns a blocker into a capability. Weighted median for LTV is a genuine product decision, not a registry edit. |
| 3 | **Population propagation into the four D1 movement routes** | High | High | Medium | **Low** | "How did the back book move last month" is a natural follow-up to questions the agent already answers. The seam exists; P1L already proved the refusal is correct, so this converts refusals into answers with no new semantics. |
| 4 | **Parser parity and phrasing robustness** | Medium | High | Low | Low | Fixes L2, L4 and L7 as one theme. Restores contribution on the production path and makes "since last month" work. Cheap, and each item is individually verifiable. |
| 5 | **Named-winner ranking for static questions** | Medium | Medium | Low | Low | "Which region has the highest LTV" should name it, as the movement route already does. Purely a presentation contract; the ranking is already computed. |

Items 1, 3 and 4 together would move the scorecard's breadth and consistency rows from B/C to
A− without introducing a single new analytic concept.

## 17. Missing-data capabilities

Capabilities that are absent because **this client's data does not carry the fields**, not
because the agent lacks the analytic. Nothing here is a product defect.

| Capability | Missing field | Consequence |
|---|---|---|
| Arrears and delinquency MI | arrears/default/impairment columns are present but **entirely zero** | No credit-performance analysis is possible or meaningful |
| Loss and impairment analysis | same | Cannot be offered |
| Borrower segmentation | no `borrower_type` | No first-time-buyer / retiree / BTL cuts |
| Product mix analysis | no `erm_product_type` | Purpose (RMRT) is the only proxy |
| Broker and channel analysis | no broker identity | No introducer performance MI |
| Pipeline and conversion MI | no pipeline data supplied | `cohort_conversion` correctly reports none was supplied |
| Reporting-date series | no `reporting_date` column | Handled via governed snapshots; one refusal message references it (C21, production path) |

These should be raised with the client as **data-supply questions**, not as product roadmap.

## 18. Out-of-product-scope analytics

Per standing ruling, and restated here so they are not re-litigated: **HPI stress testing,
Herfindahl-Hirschman concentration indices and correlation analysis are not current Trakt MI
requirements and are out of scope.** The 40-question adversarial bank contains such questions;
their presence there does not establish a requirement. `scenario` is classified D3 for the same
reason. No roadmap item in §16 or §21 depends on any of them.

## 19. Learning-loop readiness

**Verdict: NOT READY — but the hard part is already done.**

Every ingredient a learning loop needs is computed per query and returned in metadata:

- parser provenance (`llm` / `llm_repaired` / `deterministic` / `validation_failed`) and call counts
- the full facet ledger — what was asked, what was applied, what was refused and why
- population evidence — predicates applied, rows before and after
- the execution receipt — the exact calculation performed

What is missing is only **capture**. There is no question/answer persistence, no outcome
telemetry, no feedback endpoint and no user rating anywhere in the API surface. The only
telemetry present is Copilot package versioning and parquet cache counters.

Concretely, before Beta the deployment should capture, per query: question text, parser
provenance, route, facet ledger outcome, refusal reason where applicable, and a user
thumbs-up/down. That is a capture sink over data that already exists in-process — not new
instrumentation. Doing it at Beta start is what makes the Beta worth running: **refusals are
the most valuable signal the product can collect**, because each one names a real client
question the envelope does not yet cover.

## 20. UI / Beta positioning recommendation

Position it as **"MI Agent — Beta"** with a stated envelope, not as a general-purpose
analyst. Three concrete recommendations:

1. **Publish the envelope in the UI.** A short "what I can answer today" panel, in the
   commercial language of §13. The agent's greatest commercial asset is that it refuses
   rather than guesses; a client who knows the boundary reads a refusal as integrity rather
   than as failure.
2. **Make refusals actionable.** Today's refusals correctly say what was not done. They should
   also offer the nearest supported question — the vocabulary to do this already exists in the
   unmapped-question hint text.
3. **Label Beta honestly on the two known soft edges** — no median/percentile figures, and no
   share-of-book questions — until §16 items 1 and 2 land. Do not label the product Beta and
   leave the client to discover these.

Do **not** position it as replacing the existing MI pack. Position it as answering the
follow-up questions the pack provokes — which is precisely where its strengths (composition of
scope, population, provenance and seasoning) are strongest.

## 21. Recommended development roadmap

| Phase | Content | Gate |
|---|---|---|
| **P1M — Blocker fix** (mandatory before any client) | Refuse a disallowed aggregation via the governed template; widen the deterministic parser to emit `median` where the registry permits it; add a facet kind for the requested statistic so the ledger covers the axis | 0 incorrect successful answers on an aggregation-specific adversarial bank; both parser paths agree; full suite green |
| **P1N — Share of the book** | §16 item 1 | Proportion questions answer correctly with a governed denominator; no change to refusal behaviour elsewhere |
| **P1O — Parity and phrasing** | §16 item 4 — restore contribution on the production path, recognise "since last month", close C01/C03/C30 | Deterministic and production paths agree on the full curated set |
| **P1P — D1 movement population** | §16 item 3 — population propagation into `period_movement`, `period_change_analysis`, `funded_bridge`, `temporal_compare` | Sub-book movement answers correctly; routes that still cannot honour a population continue to refuse |
| **P1Q — Presentation** | §16 item 5, plus L6, L8, L10 | Ranking names a winner; no malformed labels; no raw internal error text reaches a user |
| **Beta telemetry** (do first, alongside P1M) | §19 capture sink | Every query's provenance, facets and outcome persisted; refusals queryable |

Weighted median for LTV (§16 item 2 beyond the blocker fix) should be taken as a **product
decision**, not folded into P1M. Until it is taken, median LTV is a governed refusal — which is
correct and safe.

## 22. Explicit recommendation

**Do not deploy to a client today. Deploy once P1M is complete and verified.**

The reasoning is narrow and should not be over-read. This product does the hard thing well:
it resolves scope, population, provenance and seasoning once, composes them correctly, and
refuses rather than substitutes when it cannot honour what was asked. Across 34 commercial
questions and two parser paths it produced no wrong answers, no silent errors and no crashes,
and every figure reconciled exactly against independently-computed truth. That is a strong
position from which to launch.

It is held back by one defect on one axis the governance layer never covered — and by the
uncomfortable fact that the curated set scored 30/34 precisely because it did not ask for a
median. The blocker is well understood, reproducible, root-caused, and the required semantics
have been ruled. It is a bounded piece of work, not an architectural problem.

Launch on the condition that:

1. **P1M is complete and verified** — no disallowed aggregation is ever silently substituted,
   and the ledger covers the statistic axis. **Mandatory.**
2. **Telemetry capture is live from day one**, so the Beta produces the signal that justifies it.
3. **The envelope in §13 is published in the UI**, with median/percentile and share-of-book
   named as not-yet-supported.
4. **Beta scope is limited to the stock-analysis envelope** — no movement-within-sub-book, no
   forecasting, no credit performance — with those gaps stated up front rather than discovered.

With those four conditions met, this is a product worth putting in front of a lending client.

---

MI AGENT COMMERCIAL BETA READINESS: LAUNCH WITH CONDITIONS
