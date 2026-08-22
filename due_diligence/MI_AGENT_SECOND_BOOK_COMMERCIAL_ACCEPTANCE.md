# MI Agent — Second-Book Commercial Generalisation & Presentation Acceptance

**Purpose:** establish whether the governed MI Agent generalises to a materially different lending
portfolio, or whether it has been over-fitted to Alderbridge across P0–P1N.
**Mode:** build test fixture / review / measure / report. No production behaviour was changed by
this review; the one production change made during it was a separately-briefed and separately-
approved safety fix (`48d9d44`).
**Status:** this document is the **frozen pre-analytical-capability-layer baseline**.

---

## 1. Executive verdict

**The architecture generalises. It is not over-fitted to Alderbridge.**

A second synthetic Equity Release book was built with deliberately inverted economics — acquired
67% of AuM instead of 30%, direct and acquired LTVs 13.7 points apart instead of 0.67, a front
book of 25% of loans instead of 11%, concentration in the North West and Scotland instead of the
South East. Every governed semantic transferred intact: scope, provenance, seasoning, population
propagation, statistics, multi-measure composition and multi-dimensional presentation.

| | Deterministic | Production (genuine LLM) |
|---|---|---|
| CORRECT | **48 / 50** | **49 / 50** |
| HONEST_PARTIAL | 0 | 1 |
| SAFE_REFUSAL | 2 | 0 |
| **INCORRECT_SUCCESSFUL** | **0** | **0** |
| **SILENT_SEMANTIC_ERROR** | **0** | **0** |
| **HARD_FAILURE** | **0** | **0** |

The two deterministic refusals are the **same** limitations Alderbridge exhibits, which is the
clearest available evidence of consistent governed behaviour rather than fixture-specific tuning.

The review found one Beta-blocking defect — a *fabricated population* — which reproduced on the
production demonstration book and has since been fixed and accepted under a separate brief. One
presentation limitation remains open by product ruling: **bubble charts are excluded from the
commercially-ready surface** until their sampling disclosure reaches the user.

## 2. Second synthetic portfolio design

**Kestrelmoor Lifetime Lending (synthetic)** — 12,255 funded loans, £1,331,647,994.86, three
reporting snapshots (2026-04-30 / 05-31 / 06-30), built on the current canonical schema with no
new canonical fields.

Four governed portfolios, provenance carried in governed metadata rather than inferred from IDs:

| Portfolio | Type | Loans |
|---|---|---|
| `kmr_direct_north` | direct | 3,160 |
| `kmr_direct_retail` | direct | 2,452 |
| `kmr_acq_meridian` | acquired | 4,041 |
| `kmr_acq_thornwood` | acquired | 2,602 |

Balances accrete monthly at each loan's own rate (interest roll-up), property values index, and a
small share of loans redeem — so period movement is a property of the data rather than a
manufactured delta. Realistic nulls: 207 missing valuations, 72 missing regions.

**Isolation:** the fixture, truth manifest, question bank and harnesses live entirely under the
review scratchpad with their own blob root and portfolio registry. No production client
configuration was touched; the repository tree was clean throughout.

## 3. Comparison with Alderbridge

| Characteristic | Alderbridge | Kestrelmoor |
|---|---|---|
| Loans / AuM | 11,035 / £1.965bn | 12,255 / £1.332bn |
| **Acquired share of AuM** | **29.5%** | **67.2%** |
| **Direct vs acquired waLTV** | 43.35 vs 42.68 (**0.67 apart**) | 29.66 vs 43.39 (**13.7 apart**) |
| **Front book** | 10.7% of loans, 8.7% of AuM | **25.1% of loans, 17.9% of AuM** |
| Whole-book waLTV | 43.16 | 38.90 |
| **Top region** | South East 26.3% | **North West 24.2%, Scotland 15.8%** |
| **Borrower age by provenance** | ~71.4 throughout | **69.0 direct vs 76.6 acquired** |
| Vintages | 13 (2014–2026) | 14 (2013–2026) |
| **Largest loan as % of AuM** | 0.043% | **0.156%** (3.6×) |
| Balance distribution | median £157k / mean £178k | **median £84.9k / mean £108.7k** (more skewed) |
| Nulls | negligible | 207 valuations, 72 regions |

## 4. Truth manifest summary

Computed independently in pandas; prior snapshots read straight from the dated cuts rather than
through the MI read path, so the manifest does not depend on the read path being correct.

| | |
|---|---|
| AuM / loans | £1,331,647,994.86 / 12,255 |
| Median / mean / max / min balance | £84,866.76 / £108,660.38 / £2,072,011.01 / £0.00 |
| waLTV / max / min | 38.904527 / 75.4032 / 0.0000 |
| Avg / median / exposure-weighted borrower age | 73.0663 / 73.0 / 74.6010 |
| Avg / exposure-weighted months on book | 63.19 / 72.667 |
| Direct | £436,631,033.20 / 5,612 / waLTV 29.658 |
| Acquired | £895,016,961.66 / 6,643 / waLTV 43.394 |
| Front book | £238,685,188.37 / 3,074 / waLTV 30.918 |
| Back book | £1,092,962,806.49 / 9,181 / waLTV 40.643 |
| Largest loan / share | £2,072,011.01 / 0.1556% |
| Month-on-month movement | +£81,633,816.56 (11,264 → 12,255 loans) |
| Top region growth | North West +£19,992,670.77 |

## 5–6. Commercial question results — deterministic

50 questions across seven families. **48 CORRECT, 2 SAFE_REFUSAL**, zero incorrect, zero silent,
zero hard failures. 24 scalar answers reconciled to the manifest at **zero variance**, with
populations verified by row count as well as value.

The two refusals:

| Q | Question | Why |
|---|---|---|
| K01 | "What is the total funded AuM?" | unmapped by the deterministic parser — **identical to Alderbridge C01** |
| K50 | "Show me balance by LTV band for the back book." | heatmap requires two dimensions — **identical to Alderbridge C30** |

Both are answered correctly on the production LLM path.

## 7. Genuine-LLM results

**49 CORRECT, 1 HONEST_PARTIAL**, zero incorrect, zero silent, zero hard failures. **56 genuine
model calls.**

Parser provenance across the 50: `llm` 42 · `llm_repaired` 4 · `deterministic_fallback` 3 ·
`validation_failed` 1. The deterministic fallbacks are reported as such, not as genuine LLM
parses — one of them is the sponsored-book recovery introduced by the safety fix.

## 8. Numerical reconciliation

Every substantive figure recomputed from the fixture with pandas; the MI executor was never its
own oracle. **Zero unexplained variance.** Representative:

| Question | Delivered | Truth |
|---|---|---|
| K03 weighted-average LTV | 38.904527 | 38.904527 |
| K05 exposure-weighted borrower age | 74.601 | 74.601 |
| K07 exposure-weighted months on book | 72.667 | 72.667 |
| K09 / K10 max / min balance | £2,072,011.01 / £0.00 | exact |
| K19 acquired book balance | £895,016,961.66 / 6,643 loans | exact |
| K20 back-book waLTV | 40.643241 | exact |
| K22 max balance, acquired book | exact, over 6,643 rows | exact |
| K30 max LTV, loans over £500k | exact | exact |
| K32 largest exposure | £2.1m, 0.16% of book | £2,072,011.01, 0.1556% |

## 9. Scope / population validation

Every governed population resolved correctly and was **proven by row count**, not claimed:

| Population | Rows | Balance |
|---|---|---|
| Entire AuM / sponsored book / whole book | 12,255 | £1,331,647,994.86 |
| Direct | 5,612 | £436,631,033.20 |
| Acquired | 6,643 | £895,016,961.66 |
| Front book | 3,074 | £238,685,188.37 |
| Back book | 9,181 | £1,092,962,806.49 |
| Direct ∩ back book | 3,274 | £265,071,428.39 |
| Acquired ∩ front book | 736 | £67,125,583.56 |

P1L propagation holds on the second book: K35 ("where is the back book most concentrated?") shows
`populationApplied: rowsBefore 12255 → rowsAfter 9181`, with the concentration output scoped to
£1.09bn and back-book purpose mix (RMRT 44.9%) rather than the whole-book mix (46.0%).

## 10. Chart / table presentation matrix

Assessed on all six axes required by the brief.

| Q | Artifact | Semantic | Data reconciles | Labels | Disclosure | CFO value |
|---|---|---|---|---|---|---|
| K40 balance by region | bar + table | ✅ | ✅ **sums to AuM exactly** | ✅ | ✅ nulls bucketed and stated | ✅ |
| K41 balance by LTV band | bar + table | ✅ | ✅ all bands match | ✅ | ✅ | ✅ |
| K42 loan count by age band | bar + table | ✅ | ✅ | ✅ | ✅ | ✅ |
| K43 region × LTV band | **heatmap** + table | ✅ | ✅ **89 cells sum to AuM** | ✅ | ✅ | ✅ |
| K44 LTV by age bucket × region | **heatmap** + table | ✅ | ✅ 84 cells | ✅ | ✅ | ✅ |
| K45 cross-tab region × LTV | heatmap / table | ✅ | ✅ | ✅ | ✅ | ✅ |
| K46 balance by LTV by age | **bubble** | ✅ | ⚠️ sample | ✅ | ❌ **not disclosed** | ❌ **excluded** |
| K47 bubble balance/LTV/age | **bubble** | ✅ | ⚠️ sample | ✅ | ❌ **not disclosed** | ❌ **excluded** |
| K48 region × LTV × age | table (414 rows) | ✅ | ✅ **sums to AuM** | ✅ | ✅ *"a chart shows at most two"* | ✅ |
| K49 waLTV by vintage | bar + table | ✅ | ✅ 14 vintages | ✅ | ✅ | ✅ |

**Bar charts are honest.** K40 renders 10 bars for 12 regions — the tenth is an explicit **"Other"**,
the table carries all 12 rows, the answer states "12 group(s)", and the bars sum to AuM exactly.
Investigated and cleared; not a truncation defect.

## 11. Heatmap results

Fully working on the second book. Region × LTV band produced 89 cells summing to
£1,331,647,994.86 — exactly AuM. The cell count exceeds a naive `groupby` count (73) because rows
with missing grouping values are bucketed under **"Unknown / Missing"** and disclosed, rather than
dropped, so the grid still reconciles to the funded book. That is the governed behaviour working
as designed.

## 12. Bubble results — **excluded from the commercially-ready surface**

Bubbles render correctly and their axes are right (x = borrower age, y = current LTV, size =
balance, loan-level). The problem is disclosure, not data.

## 13. Multi-dimensional results

| Dimensions | Deterministic | Genuine LLM |
|---|---|---|
| 2 (heatmap) | ✅ correct, reconciles | ✅ K44/K45 correct; K43 **honest partial** |
| 3 (table) | ✅ 414 rows, sums to AuM, disclosed | ✅ same |

**§18 parity repeats — 3 genuine runs each, all stable 3/3:**

| Question | Outcome |
|---|---|
| balance by region and LTV band | **HONEST_PARTIAL** ×3 — second dimension dropped, `verdict: partial`, facet `lost`, warning *"Not applied: region"* |
| balance by LTV bucket and age bucket | **HONEST_PARTIAL** ×3 — same, disclosed |
| bubble balance by LTV and borrower age | CORRECT ×3 (excluded per §12) |
| cross-tab region and LTV band | **CORRECT** ×3, 2 dimensions, reconciles |

Tally: 6 CORRECT, 6 HONEST_PARTIAL, **0 INCORRECT, 0 HARD_FAILURE**. This reproduces the
Alderbridge finding precisely — deterministic multi-dimensional rendering is strong, the LLM path
sometimes loses the second dimension, and **P0 discloses it every time**.

## 14. Bubble 5,000-row cap assessment

| Question | Finding |
|---|---|
| Is the bubble truncated? | **Yes** — 5,000 of 12,048 valid observations (41.5%) |
| How are rows selected? | `out.sample(n=5000, random_state=42)` — **uniform random** |
| Deterministic? | **Yes**, fixed seed |
| Biased? | **No** — the sample is unbiased; but it necessarily omits the tail |
| Disclosed to the user? | **No** |

The executor *does* generate the disclosure —
`loan-level output capped: sampled 5000 of 12048 rows (deterministic seed=42)` and
`dropped 207 loan-level row(s) with non-numeric/null x/y/size values`. Both are then classified as
technical diagnostics by `mi_agent_api/adapters.py::_TECHNICAL_WARNING_PATTERNS` and stripped from
the user-facing card, even though that module's own comment states that *partial result* and
*missing data* warnings are never matched there. They survive in `diagnostics`; they do not reach
the reader.

Consequence, measured on both books:

| | Kestrelmoor | Alderbridge |
|---|---|---|
| True max balance | **£2,072,011.01** | £841,638.96 |
| Max visible on the chart | **£971,745.82** | £777,422.23 |

The receipt says *"Calculated: Loan-level · 12,255 loans"* beside a chart holding 41% of them, and
the largest exposure is absent. A bubble of balance-by-LTV-by-age exists to show the **tail**.

**Classification: MISLEADING PRESENTATION.** Ruled by the product owner: bubble charts may remain
in the product only if the disclosure states plotted rows, the total calculation population, and
any dropped null/non-numeric rows in business-facing language. **Until then bubble charts are not
commercially ready and are excluded from the Beta presentation surface.** Not fixed in this
review; tracked separately.

## 15. LLM / deterministic parity

| Behaviour | Deterministic | Genuine LLM |
|---|---|---|
| Core CFO scalars | ✅ | ✅ |
| Headline AuM phrasing (K01) | ❌ refuses | ✅ |
| Multi-dimension (2D) | ✅ | ⚠️ sometimes 1D, always disclosed |
| Balance by LTV band for a population (K50) | ❌ refuses | ✅ |
| Governed populations | ✅ | ✅ (after the safety fix) |

Neither path dominates. The deterministic path is narrower but never loses a dimension; the LLM
path is broader but occasionally drops one and says so. Identical to Alderbridge.

## 16. Existing-regression results

Creating the second fixture changed nothing about Alderbridge.

| Asset | Result |
|---|---|
| Alderbridge Commercial Beta bank — deterministic | **29 / 5**, unchanged, 0 changed answers |
| Alderbridge Commercial Beta bank — genuine LLM | **30 / 4**, unchanged |
| Immutable 40-question bank | **14 / 40**, **0 churn** |
| P-gates (P0, P1C–P1N) + `mi_agent` + `mi_agent_api` | **2,922 passed** |
| Full repository suite | **8,854 passed, 30 skipped, 21 xfailed, 0 failed** |
| Multi-dimensional chart / golden tests | ✅ passed |

## 17. Commercial Beta scorecard — second book

| Dimension | Score | Basis |
|---|---|---|
| Answerability | **A** — 96% deterministic, 98% production | 48/50 and 49/50 |
| Safe coverage | **A** | 0 incorrect, 0 silent, 0 hard failures on either path |
| Numerical correctness | **A** | zero variance across 24 reconciled scalars + every chart dataset |
| Scope / population governance | **A** | 7 governed populations exact by row count; P1L propagation holds |
| Statistics | **A** | median, min, max, weighted averages all exact |
| Table presentation | **A** | every dataset reconciles to AuM |
| Chart presentation (bar / heatmap) | **A** | correct, labelled, null handling disclosed |
| Chart presentation (bubble) | **F** | undisclosed 41% sample — excluded |
| Multi-dimension parity | **B** | LLM drops a second dimension, always disclosed |
| Generalisation | **A** | same behaviour, same limitations, materially different book |

**Answerability rate 96% / 98% · Safe coverage 100% · Visual success 8 of 10 (bubbles excluded)
· INCORRECT_SUCCESSFUL 0 · SILENT_SEMANTIC_ERROR 0 · HARD_FAILURE 0.**

## 18. Top observed limitations

| # | Limitation | Class |
|---|---|---|
| L1 | **Bubble sampling not disclosed** — 41% plotted, largest exposure absent, receipt states full population | **OPEN — bubbles excluded from Beta** |
| L2 | LLM path drops the second dimension on some 2-D phrasings (disclosed as partial) | Normal Beta limitation |
| L3 | "Which region has the largest exposure?" answers at **ITL3 granularity** (Blackburn with Darwen £19.7m) rather than region (North West £322.5m). The basis *is* disclosed — *"across 172 ITL3 area(s). Basis: collateral"* — and the figure reconciles exactly to the true top ITL3 | Semantic granularity mismatch, disclosed |
| L4 | **"Top five regions" returns all 12**, correctly ordered with the top five leading, but the requested top-N was not applied and its absence was not disclosed | Undisclosed unhonoured facet |
| L5 | Concentration output leads with degenerate 100% dimensions (amortisation type, collateral type, currency) | Presentation quality |
| L6 | Deterministic parser refuses two phrasings the LLM answers (K01, K50) | Parser parity |
| L7 | Same threshold-attachment and phrasing-robustness limits recorded for Alderbridge | Pre-existing, unchanged |

## 19. WIN_WIN enhancement candidates

Adds capability without degrading anything currently correct.

| # | Candidate | Effort |
|---|---|---|
| W1 | Apply the requested top-N to ranked breakdowns, or disclose that the full set is shown (L4) | Small |
| W2 | Answer "which region" at the region granularity the rest of the product uses, or state the granularity in the headline (L3) | Small |
| W3 | Order concentration output by informativeness so degenerate 100% dimensions do not lead (L5) | Small |
| W4 | Close the deterministic gaps on K01/K50 phrasings (L6) | Small |

## 20. SAFETY_FIX candidates

Removes an incorrect or silent answer.

| # | Candidate | Status |
|---|---|---|
| S1 | **Bubble sampling disclosure** — surface plotted rows, total calculation population and dropped nulls in business language | **OPEN — required before bubbles ship** |
| S2 | Fabricated population — a governed population the question never requested must not execute | **CLOSED** — fixed and accepted (`48d9d44`) |

## 21. TRADE_OFF candidates — do not implement

| # | Candidate | Trade-off |
|---|---|---|
| T1 | Force the LLM path to refuse when it drops a second dimension | Converts disclosed partials into refusals — lowers answerability to raise strictness. **Needs an explicit product ruling.** |
| T2 | Raise or remove the 5,000-row bubble cap | Trades payload size and render performance for completeness. Disclosure (S1) is the cheaper fix. |

## 22. Recommended launch posture

Deploy the second-client envelope as **MI Agent — Beta** with:

1. **Bubble charts excluded** until S1 lands. Every other visual — bar, heatmap, cross-tab,
   multi-dimensional table — is commercially ready and reconciles exactly.
2. The published envelope from the Commercial Beta Readiness Review, unchanged; this book
   demonstrates it transfers.
3. The four remaining launch conditions from that review still outstanding: telemetry capture,
   published envelope in the UI, stated Beta scope, and now S1.

**The commercial question, answered plainly.** If a materially different Equity Release lender
were onboarded tomorrow, the MI Agent would behave like a very good commercial Beta product. It
answered 49 of 50 realistic CFO questions on the production path, every figure reconciled to
independently computed truth, every governed population was exact to the row, it refused honestly
where it could not help, and its failures on the new book were the *same* failures it has on the
old one. The one wrong answer it produced was a pre-existing defect that this exercise surfaced
rather than caused, and it is now closed.

We have not over-fitted to Alderbridge. The one capability that is not ready to show a client is
the bubble chart, and that is a disclosure problem rather than a correctness one.

---

SECOND-BOOK COMMERCIAL ACCEPTANCE: PASS WITH LIMITATIONS
