# Sprint 2.5 — Securitisation Readiness Framework & production readiness gate

*Baseline `584c9d1` → `2a04ec8`+. The agent is **not built**; this sprint defines
what it will assess and proves Trakt can support it.*

---

## 1. Executive summary

**What securitisation readiness means in Trakt.** Not a rating, not a prediction,
and not a replica of any agency methodology. It is the answer to a narrower and
more useful question: *is this book broadly in a state where a securitisation
process could sensibly begin, and what would an experienced reviewer look at
first?*

The whole design rests on keeping four things apart:

| | Example | Who owns it |
|---|---|---|
| **Fact** | "28.0% of balance is secured in London." | **Trakt measures it** |
| **External criterion** | "The warehouse permits 35%." → *pass* | a real agreement |
| **Trakt screening rule** | "Flag above 25%." → *flag* | **Trakt's internal threshold** |
| **Judgement** | "Within facility, but not the proposed transaction limit." | the agent, or a human |

**What Trakt measures.** 48 framework metrics across nine categories, of which
**44 (93.6%) are now deterministically measurable** through a governed tool. Every
figure comes from an existing engine: the 39 registered evaluators in the
concentration-test library, `analytics_lib`, the period-change workflow. No
securitisation-specific calculation was created.

**What rules Trakt applies.** Only its own screening thresholds, and only ever
labelled as such. Fifteen of them, each versioned, configurable, overridable, and
carrying a written rationale. Everything else is an external rulebook that
somebody supplies.

**What the future agent will judge.** Materiality, overlap between findings,
priority, what the data cannot support, and what needs a human. Trakt supplies
facts and rule outcomes; interpretation is the agent's and is attributed to it.

**Also in this sprint:** the audit lost-update race is fixed — the blocker named
at the end of Sprint 2, and a prerequisite for any autonomous concurrent work.

---

## 2. Framework v1

Full registry: `config/system/securitisation_readiness_framework.yaml`.
48 metrics. Abridged to one row per metric; `Screening` names a rule in
`TRAKT_SCREENING@v1`.

| Category | Metric | Type | Tool | Screening | External | Status |
|---|---|---|---|---|---|---|
| composition | Total balance | MEASURE_ONLY | `portfolio_summary` | — | ✓ | READY |
| composition | Loan count | MEASURE_ONLY | `portfolio_summary` | — | ✓ | READY |
| composition | Average loan size | MEASURE_ONLY | `readiness_metrics` | — | ✓ | READY |
| composition | Largest loan | HYBRID | `readiness_metrics` | — | ✓ | READY |
| composition | Seasoning | MEASURE_ONLY | `readiness_metrics` | — | ✓ | READY |
| composition | Product mix | EXTERNAL | `stratify` | — | ✓ | READY |
| composition | Fixed/variable mix | MEASURE_ONLY | `stratify` | — | ✓ | READY |
| composition | Originator mix | HYBRID | `stratify` | — | ✓ | READY |
| composition | Maturity within horizon | HYBRID | `readiness_metrics` | — | ✓ | READY |
| collateral | WA current LTV | HYBRID | `portfolio_summary` | `SCREEN_WA_LTV` | ✓ | READY |
| collateral | High-LTV share | HYBRID | `readiness_metrics` | `SCREEN_HIGH_LTV_SHARE` | ✓ | READY |
| collateral | Maximum LTV | HYBRID | `readiness_metrics` | `SCREEN_MAX_LTV` | ✓ | READY |
| collateral | Negative equity | HYBRID | `readiness_metrics` | `SCREEN_NEGATIVE_EQUITY` | ✓ | READY |
| collateral | **Valuation age** | HYBRID | `valuation_age_profile` | `SCREEN_STALE_VALUATION_SHARE` | ✓ | **READY (new)** |
| collateral | **Valuation method mix** | HYBRID | `valuation_age_profile` | `SCREEN_INDEXED_VALUATION_SHARE` | ✓ | **READY (new)** |
| collateral | **Missing valuations** | HYBRID | `valuation_age_profile` | `SCREEN_MISSING_VALUATION_SHARE` | ✓ | **READY (new)** |
| collateral | WA property value | MEASURE_ONLY | `readiness_metrics` | — | ✓ | READY |
| collateral | Largest property exposure | HYBRID | `readiness_metrics` | — | ✓ | READY |
| borrower | Largest borrower | HYBRID | `concentration` | `SCREEN_LARGEST_BORROWER_SHARE` | ✓ | READY |
| borrower | Multi-loan borrower share | MEASURE_ONLY | `readiness_metrics` | — | ✓ | READY |
| borrower | Max loans per borrower | EXTERNAL | `readiness_metrics` | — | ✓ | READY |
| borrower | Borrower geography | MEASURE_ONLY | `stratify` | — | ✓ | READY |
| borrower | Borrower age profile | EXTERNAL | `readiness_metrics` | — | ✓ | READY |
| performance | Arrears share | HYBRID | `readiness_metrics` | `SCREEN_ARREARS_SHARE` | ✓ | READY |
| performance | 90+ DPD share | HYBRID | `readiness_metrics` | `SCREEN_90_PLUS_SHARE` | ✓ | READY |
| performance | Account status mix | MEASURE_ONLY | `stratify` | — | ✓ | READY |
| performance | Defaulted share | HYBRID | `stratify` | — | ✓ | READY |
| performance | Cumulative losses | MEASURE_ONLY | — | — | ✓ | **SMALL_GAP** |
| performance | Prepayment rate | MEASURE_ONLY | — | — | ✓ | **SMALL_GAP** |
| concentration | Geographic | HYBRID | `concentration` | `SCREEN_GEOGRAPHIC_CONCENTRATION` | ✓ | READY |
| concentration | Postcode area | HYBRID | `readiness_metrics` | — | ✓ | READY |
| concentration | Distinct regions | MEASURE_ONLY | `readiness_metrics` | — | ✓ | READY |
| concentration | Top-N loans | HYBRID | `readiness_metrics` | `SCREEN_TOP_10_LOAN_SHARE` | ✓ | READY |
| concentration | Large-loan share | HYBRID | `readiness_metrics` | — | ✓ | READY |
| concentration | Property type | HYBRID | `concentration` | — | ✓ | READY |
| data_quality | Field completeness | TRAKT_SCREEN | `data_completeness` | `SCREEN_CRITICAL_FIELD_COMPLETENESS` | ✓ | READY |
| data_quality | Validation exceptions | TRAKT_SCREEN | `list_validation_exceptions` | `SCREEN_VALIDATION_FAILURES` | ✗ | READY |
| data_quality | Provenance quality | MEASURE_ONLY | `explain_values` | — | ✗ | READY |
| data_quality | Snapshot currency | MEASURE_ONLY | `portfolio_summary` | — | ✓ | READY |
| regulatory | **Mandatory field coverage** | TRAKT_SCREEN | `regulatory_readiness` | `SCREEN_MANDATORY_FIELD_COVERAGE` | ✓ | **READY (new)** |
| regulatory | **Schema validation** | TRAKT_SCREEN | `regulatory_readiness` | — | ✗ | **READY (new)** |
| regulatory | Submission acceptance | MEASURE_ONLY | — | — | ✗ | **DATA_GAP** |
| eligibility | Approved tests | EXTERNAL | `evaluate_covenants` | — | ✓ | READY |
| eligibility | Breach drill-through | EXTERNAL | `covenant_drillthrough` | — | ✓ | READY |
| eligibility | Criteria supplied? | MEASURE_ONLY | `evaluate_rule_packs` | — | ✗ | READY |
| trend | Period movement | MEASURE_ONLY | `period_change` | — | ✓ | READY |
| trend | Covenant deterioration | EXTERNAL | `evaluate_covenants` | — | ✓ | READY |
| trend | Data-quality drift | JUDGEMENT_ONLY | `data_completeness` | — | ✗ | JUDGEMENT_ONLY |

---

## 3. Coverage score

Counted honestly. A metric is *deterministic* only when its status is `READY`
**and** a governed tool serves it. `JUDGEMENT_ONLY` is excluded from the
denominator rather than scored either way — no deterministic calculation
*should* exist for it, so counting it as covered would claim credit and counting
it as a gap would imply work that should never be done.

**Overall: 44 of 47 scored metrics — 93.6%.** (Before this sprint's gap work:
39/47 = 83.0%.)

| Category | Before | After | |
|---|---|---|---|
| composition | 9/9 | **9/9** | 100% |
| collateral | 6/9 | **9/9** | 100% |
| borrower | 5/5 | **5/5** | 100% |
| performance | 4/6 | **4/6** | 66.7% |
| concentration | 6/6 | **6/6** | 100% |
| data_quality | 4/4 | **4/4** | 100% |
| regulatory | 0/3 | **2/3** | 66.7% |
| eligibility | 3/3 | **3/3** | 100% |
| trend | 2/2 | **2/2** | 100% |

The three remaining gaps are named in §4 and none is a blocker.

---

## 4. Gap register

| Gap | Importance | Type | Work required | Implemented now? | Reason |
|---|---|---|---|---|---|
| Valuation age / staleness | **High** | SMALL_GAP | Balance-weighted age profile over the governed selection policy | ✅ **Yes** | Every LTV rests on it, and stale-vs-recent is the difference between a data finding and a credit one |
| Valuation method mix | Medium | SMALL_GAP | Same profile | ✅ **Yes** | Indexed-heavy books have weaker collateral evidence at identical LTV |
| Missing valuations | **High** | SMALL_GAP | Same profile | ✅ **Yes** | These loans are silently absent from every LTV metric |
| Regulatory field coverage | **High** | SMALL_GAP | Join regime field universe to canonical `regime_mapping`; separate ND tiers | ✅ **Yes** | The highest-value gap named at the end of Sprint 2 |
| Regulatory schema validation | Medium | SMALL_GAP | Same tool | ✅ **Yes** | Falls out of the same join |
| Cumulative losses | Medium | SMALL_GAP | Register an evaluator; needs loss fields the tape may not carry | ❌ No | Library declares it with no evaluator. Needs data work first, not a wrapper — reporting an estimate would be Trakt inventing a number |
| Prepayment rate | Low | SMALL_GAP | Needs consecutive snapshots, not one | ❌ No | A reliable rate is a two-period calculation; the period-change engine is the right home, and that is a larger change |
| Submission acceptance state | Medium | DATA_GAP | External evidence ingestion | ❌ No | Trakt does not hold submission receipts. Reported as outstanding information rather than inferred from a clean projection |
| Data-quality drift | Medium | JUDGEMENT_ONLY | — | ❌ **Deliberately not** | What counts as material drift depends on which fields moved and why. The agent compares two `data_completeness` calls |
| Two-dimension stratification | Low | SMALL_GAP | Extend `analytics_lib.stratify` | ❌ No | Not needed for a first-pass review; would benefit MI equally when done |
| Multi-period trend | Low | SMALL_GAP | Period-change engine is pairwise | ❌ No | Larger change, not first-pass material |

---

## 5. New deterministic capabilities

Only what was actually added.

1. **`analytics_lib/valuation_age.py`** — balance-weighted valuation age
   distribution, physical/indexed/other method mix, stale share, missing share,
   weighted-average age. Selection runs through the **same** governed policy that
   backs `current_loan_to_value`, so a loan cannot be stale here and current in
   `explain_values`. Ages are measured against the tape's reporting date, never
   the wall clock — a profile that moved on its own could not be reproduced.

2. **`trakt_core/regulatory_regime.py`** — joins the regime's 107 loan-level
   fields to the 313 canonical fields already carrying `regime_mapping`, and
   classifies each by the regime's own ND rules: 18 fields admit **no** No-Data
   fallback (hard submission blockers), the rest permit ND1-ND4 or ND5. Nothing
   is authored; the regime says what it requires.

3. **`trakt_core/readiness.py`** — the framework registry and the rule-pack
   evaluator. Computes each distinct `(metric, parameters)` **once** and applies
   every pack to it.

4. **`operations_control/audit_chain.py`** — concurrency-safe hash-chain append.

5. **Five governed tools**, all `risk:read` because none exposes an individual
   obligation: `readiness_framework`, `readiness_metrics`,
   `valuation_age_profile`, `regulatory_readiness`, `evaluate_rule_packs`.
   **18 tools total.**

**No new metric calculation was written.** DPD bands, high-LTV cohorts, top-N
and seasoning all turned out to be existing library metrics with parameters
(`min_days`, `ltv_percent`, `n`, `years`) — the gap analysis found them rather
than assuming they were missing.

---

## 6. Rule-pack architecture

One fact. Three rulebooks. Three verdicts. **No recalculation.** Measured on
`tests/readiness_portfolio.py`, an 86-loan book built so the outcomes separate:

```
FACT   largest region share = 28.0%          ← computed once

  EXAMPLE_WAREHOUSE@v1              <= 35%   PASS     authority: warehouse_agreement
  EXAMPLE_PROPOSED_SECURITISATION@v1 <= 27%  BREACH   authority: securitisation_criteria
  TRAKT_SCREENING@v1                <= 25%   FLAG     authority: trakt_internal
```

And again on granularity: top-10 loan share 14.0% → warehouse **pass** (15%),
securitisation **breach** (12%), screening **flag** (10%).

**25 rules across three packs cost 15 fact computations.** That is not only
cheaper. If each pack measured its own London share, a parameter drift or a
rounding difference could let the warehouse pass and the criteria fail on one
portfolio for a reason no reviewer could explain. Sharing the computation makes
disagreement *impossible* rather than unlikely.

**Two vocabularies, on purpose.** Internal packs return `clear`/`flag`; external
packs return `pass`/`breach`. `summarise()` counts them in separate blocks and
carries a note that they must not be added together. A screening flag and a
contractual breach read alike in a summary and mean entirely different things.

**Adding a rulebook is a file**, not a code change: `pack_id`, `authority`,
`authority_label`, and rules naming a framework metric plus a threshold.

---

## 7. Synthetic portfolio

`tests/readiness_portfolio.py` — 86 loans, £20,000,000, truth stated as literals
worked out by hand.

| Region | Loans | Each | Balance | Share |
|---|---|---|---|---|
| London | 20 | 280,000 | 5,600,000 | **28.0%** |
| South East | 24 | 225,000 | 5,400,000 | 27.0% |
| North West | 16 | 275,000 | 4,400,000 | 22.0% |
| Wales | 14 | 200,000 | 2,800,000 | 14.0% |
| Scotland | 12 | 150,000 | 1,800,000 | 9.0% |

| Planted case | Loans | Expected |
|---|---|---|
| Clean loans | 55 | nothing to find |
| High-LTV cohort (85%) | 12 | 16.8% of balance |
| Extreme LTV (108%) | 1 | 1.4% — negative equity, one loan |
| Geographic concentration | — | **28.0%: pass / breach / flag** |
| Top-10 concentration | — | **14.0%: pass / breach / flag** |
| 30+ DPD arrears (45 days) | 6 | with 90+, 10.65% at `min_days=30` |
| 90+ DPD (120 days) | 2 | 2.25% |
| Stale valuations (2021, 61 months) | 8 | 10.0% of balance |
| Recent valuations (3 months) | 75 | the control group |
| Missing valuations | 3 | 4.125% — absent from every LTV metric |
| Multiple observations | 4 | full + indexed + purchase price |
| Strong economics / weak data | 5 | LTV 45%, no arrears, critical fields blank |
| Clean data / weak economics | 12 | the high-LTV cohort: perfect data, poor metrics |

The last two exist to test the distinction the playbook insists on: a book can
be economically weak with impeccable data, or economically sound with data that
cannot support the claim, and the remediation differs.

The Sprint 2 planted portfolio (`tests/planted_portfolio.py`, 12 loans) is
retained for valuation-selection cases; it breaches every threshold at once,
which is useful for proving a metric fires and useless for proving rulebooks
differ.

---

## 8. Investigation playbook

Full text: `docs/securitisation_readiness_investigation_playbook.md`.

Sequence: establish shape → hard breaches → screening flags by **materiality**
(affected balance, not exceedance ratio) → separate data weakness from economic
weakness → follow overlaps → compare periods → inspect evidence for material
facts → state what could not be concluded.

Stopping rules: continue on real breaches, material balance, overlapping
concerns, deteriorating trends, weak evidence. Stop on immaterial balance,
unavailable data, questions requiring human judgement, or budget — and **say the
review was truncated**, because a bounded review reported as complete is worse
than an honest partial one.

Eight prohibitions, of which the load-bearing three are: never compute a metric
yourself, never report a screening flag as a requirement, and never treat an
absent rulebook as a satisfied one.

---

## 9. Output contract

Designed, not built — full JSON shape in the playbook. The design decisions that
matter:

- `overall` is **not formulaic by default**; a colour computed from a flag count
  would imply Trakt has a view it does not have. `is_formulaic` says which.
- **`INCOMPLETE` is a first-class outcome.** A review that could not obtain what
  it needed is not AMBER.
- Every finding separates `facts` (Trakt's, with tool and snapshot) from
  `judgement` (the agent's), and `rule_source` carries `authority`.
- `affected_balance` sits beside every finding, because materiality is what a
  reviewer needs and a percentage alone does not give it.
- `outstanding_information` and `limitations` are required sections, not
  optional ones — including the standing limitation that this is not a rating.

---

## 10. Regulatory coverage

**Covered.** Loan-level Annex 2 field coverage against the regime's own universe
(107 fields), split three ways by the regime's ND rules: 18 blocking (no ND
fallback permitted), the rest ND1-ND4 or ND5 permitted. Per-field population
rates, blocking-gap list, and a submission-blocked verdict. Field-level
validation outcomes via `list_validation_exceptions`.

**Not covered.** Format and permitted-value validation of the projected output
(presence is measured, not conformity — `list_validation_exceptions` carries
validation outcomes separately). Deal-level Annex 12. Submission acceptance
receipts, which Trakt does not hold and which are reported as outstanding
information rather than inferred.

**The line the tool holds:** regulatory readiness is an *input* to the
assessment, never the assessment. A clean projection is not evidence of a sound
book, and a poor one is not evidence of a bad one. Both directions are stated in
the tool's own `limitations`.

---

## 11. MCP readiness

**Yes — every framework tool can be consumed by an independent agent through the
intended interface, with no new work.**

Verified: all **18** tools translate to MCP declarations with schemas passed
**by reference** (not copied, so they cannot drift), capability and version in
`_meta`, and agent guidance folded into the description. The five new readiness
tools all declare `resource` as required, so each is authorised the same way.

No parallel MCP-specific calculation exists, and a structural test asserts
`trakt_tools/mcp.py` references no pandas, no dataframe, no authorisation call
and no `execute_governed_tool`.

**Remaining work for Sprint 3** is unchanged and is transport, not translation:
authenticate the session and build `ExecutionContext` from the authenticated
principal (never from tool arguments — `refuse_identity_in_arguments` enforces
this), map the session to tenant and organisation as `agent_auth` does for HTTP,
call `execute_governed_tool`, translate with `tool_result`, and bound
concurrency per session.

---

## 12. Audit fix

**Old failure mode.** Both audit stores appended by read-modify-write:

```
head = read(head_uri)            # seq = 5
seq  = head.seq + 1              # 6
write(audit_uri(seq), record)    # overwrites whatever is at 6
write(head_uri, ...)
```

Two concurrent appends both read 5, both compute 6, and the second overwrites the
first. **The survivor's `prev_hash` still points at event 5, so
`verify_audit_chain` returns `True`.** The log loses a record and reports itself
as intact — which is a worse category of failure than losing one loudly.

**New protection.** Sequence allocation is an atomic exclusive create, not a
read-modify-write. On collision the loser reads the record actually in the slot,
chains onto *it*, and advances — so it converges without waiting for the winner
to update the head. The head is demoted to a hint; a stale or even backwards head
costs one probe and cannot corrupt anything.

`Storage.create_exclusive` is `O_CREAT | O_EXCL` on the filesystem and
`upload_blob(overwrite=False)` on Blob — a conditional PUT evaluated at the
storage account, so it holds across processes, hosts and scale-out instances.
Both backends already had a native answer: **no lock, lease, queue or
coordination service was introduced.**

Idempotency: `append_audit(..., idempotency_key=...)` makes a retry return the
**original** record. Without a key, behaviour is unchanged — two separate actions
that look alike are two events, and de-duplicating by content would silently drop
a real repeat. Exhausting the probe budget raises `AuditAppendConflict` rather
than writing anyway.

**Concurrency test evidence.** 16 tests running **real concurrent threads**
against a real filesystem store, not a simulated interleaving:

- 24 simultaneous appends → all 24 present, sequences exactly 1…24, no gaps or
  duplicates, chain verifies, and each `prev_hash` equals the previous record's
  hash (proving one linear chain rather than a fork that happens to verify);
- a **reproduction of the old algorithm** proving it loses an event *and* still
  returns `verify_audit_chain() == True`;
- exclusive create has exactly one winner among 24 racing threads;
- a failed write leaves no partial file;
- stale head, missing head, orphaned idempotency marker, probe exhaustion;
- the same 24-thread test against the OCC agent store, which had the same defect.

Stable across 8 consecutive runs. 1,012 existing operations-control tests pass
unchanged.

---

## 13. Regression

Full suite, both trees, same machine, same interpreter, `-p no:randomly` so
ordering is fixed and the runs are comparable. Baseline is `584c9d1` in a
separate worktree.

| | baseline `584c9d1` | current |
|---|---|---|
| passed | 5,067 | **5,221** (+154) |
| failed | 64 | **64** |
| errors | 13 | **13** |
| skipped | 33 | 33 |
| subtests passed | 6 | 6 |
| wall clock | 1,973 s | 1,981 s |

**Both complete ID sets are identical**, extracted from each run, sorted,
deduplicated and diffed in both directions:

```
failure ids: base=64  current=64
error ids:   base=13  current=13

=== NEW FAILURES ===      (empty)
=== FIXED ===             (empty)
=== NEW ERRORS ===        (empty)
=== RESOLVED ERRORS ===   (empty)

failure sets IDENTICAL
error sets IDENTICAL
```

Not "the same number" — the same failures and the same errors, by identifier.
The +154 delta is entirely new tests passing.

The 64 failures and 13 errors are the same pre-existing set carried through
Sprint 2, unrelated to this work. The 13 errors are class-level fixture errors in
`tests/test_delivery_xml_agent_review.py` (10) and
`tests/test_simulation_pipeline.py` (3).

**Sprint 2.5 added 70 tests**, in two files:

| file | tests | what it holds |
|---|---|---|
| `test_readiness_framework.py` | 54 | framework consistency, rule-pack authority separation, one-fact-three-rulebooks, valuation profile, regulatory tiers, governance across the new surface |
| `test_audit_chain_concurrency.py` | 16 | real concurrent appends, the old algorithm reproduced, idempotency, bounds |

**No existing test was changed.** Every Sprint 2.5 change is additive: two new
`trakt_core` modules, one new `analytics_lib` module, one new handler module,
one new `operations_control` module, four new configuration files, and a new
`Storage.create_exclusive` primitive. `append_audit` gained an optional
keyword-only `idempotency_key`, which no existing caller passes.

**Two behavioural changes worth naming explicitly**, both inside `append_audit`
and both invisible to its existing callers:

1. A record is now written by exclusive create rather than overwrite. Serial
   appends produce byte-identical output; only the concurrent case differs, and
   it differs by not losing records.
2. `SyntheticRunStore._hash` became a classmethod that also accepts a dict, and
   `seq` is excluded from the hashed fields — it always was, but the exclusion is
   now explicit, because chaining is by `prev_hash` and including the slot number
   would make an event's hash depend on which race it won.

---

## 14. Sprint 3 readiness

> **Is Trakt ready for an independent Securitisation Readiness Agent to receive
> "Assess this portfolio" and autonomously investigate it?**

**Yes. There are no genuine blockers.**

The agent can now: discover what a readiness review covers (`readiness_framework`,
including which metrics Trakt cannot measure, so it neither invents a checklist
nor assumes a gap); measure 44 of 47 framework metrics deterministically;
evaluate several rulebooks against the same facts with authority preserved;
assess collateral evidence and regulatory submission readiness; investigate any
finding down to the loan and the valuation observation behind it; and audit every
step concurrently without losing records.

**What was a blocker and no longer is:**

- the audit lost-update race — **fixed**;
- Annex-coverage readiness, the highest-value gap from Sprint 2 — **built**;
- no way to express external criteria separately from Trakt's own thresholds —
  **the rule-pack architecture**;
- no published definition of what a readiness review covers — **Framework v1**.

**What remains, and why none of it blocks:**

| | Why it does not block |
|---|---|
| Cumulative losses, prepayment | Reported as unavailable, which is a correct answer. Both need data or history work that a wrapper cannot substitute for |
| Submission acceptance | External evidence, correctly reported as outstanding information |
| Real criteria packs | The architecture is proven with synthetic packs; supplying real ones is deployment configuration, not code |
| Enquiry lifecycle / evidence pack | Needed for a *diligence process*, not for a single agent assessing a portfolio |

**Two cautions for Sprint 3.**

First, the temptation will be to let the agent compute something Trakt does not
expose — a ratio, a cohort, a trend — because it is faster than adding a tool.
Every time that happens the number loses its definition, its provenance and its
reproducibility. The correct response to a missing capability is a new governed
tool that every entitled organisation can call.

Second, and specific to this sprint's subject: the agent will be tempted to
report Trakt's screening flags as findings against external requirements, because
that reads more decisively. The framework, the packs and the tool contracts all
resist it, but the agent's prompt and its output must too. A screening flag means
"look at this", and nothing more.
