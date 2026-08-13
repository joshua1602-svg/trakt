# The Securitisation Readiness Agent — investigation playbook

*Sprint 2.5, Parts 9, 10 and 11. The agent is **not built**. This is the contract
it will be built against.*

---

## Why a playbook rather than a question tree

An agent handed 48 metrics and told to "assess readiness" will run all 48 and
write a summary. That is a report, not a review, and it fails in a specific way:
every finding gets equal weight, nothing is followed up, and the connections
between findings — which is where the actual analysis lives — are never made.

A hard-coded question tree fails the other way. It cannot follow a thread it was
not told about, and the moment a portfolio is unusual the tree is wrong.

So this document gives **principles and stopping rules**, not a script. The
agent's job is to investigate; Trakt's job is to make every fact it needs
measurable, evidenced and cheap to obtain.

---

## The four categories, again, because everything depends on them

| | What it is | Where it comes from | Vocabulary |
|---|---|---|---|
| **Fact** | A measured number | Trakt, from governed data | a value and a unit |
| **External criterion** | A real requirement | warehouse, term sheet, mandate, regulation | **pass** / **breach** |
| **Trakt screening rule** | An internal attention threshold | `TRAKT_SCREENING@v1` | **clear** / **flag** |
| **Judgement** | What it means | the agent, or a human | prose, always attributed |

**The agent must never present a screening flag as an external requirement.**
Every result from `evaluate_rule_packs` carries `authority`,
`authority_label` and `is_external_criterion`. Read them before writing a word.

A worked example, from the readiness test portfolio:

> **Fact.** 28.0% of balance is secured in London.
> **External.** The warehouse permits 35% — **pass**. The proposed
> securitisation criteria permit 27% — **breach**.
> **Screening.** Trakt flags above 25% — **flag**.
> **Judgement.** The book is within its current facility but would not meet the
> proposed transaction limit as drafted. Whether that is a blocker depends on
> whether the limit is negotiable, which is a human question.

Four different statements. One number.

---

## The investigation sequence

Not a script — an ordering, because the early steps determine which of the later
ones are worth doing at all.

### 1. Establish the shape before looking for problems

`readiness_framework` first: it says what a review covers and what Trakt can
measure, so the agent plans rather than guesses. Then `portfolio_summary` for
size, balance, weighted averages and composition.

A percentage means nothing until the reader knows what it is a percentage of.
Establish the denominator before quoting any share.

### 2. Hard breaches first

`evaluate_covenants` for the operator-approved tests — these carry an approver
and a configuration version and are the only results in the whole framework that
somebody has signed. Then `evaluate_rule_packs` for supplied criteria.

**A breach against a real criterion outranks every screening flag**, however
large the flag. Order the report that way.

### 3. Then screening flags, by materiality

A flag is an invitation, not a finding. Rank by affected balance, not by how far
the metric exceeded the threshold: 1% of the book at triple the threshold matters
less than 20% just over it.

### 4. Distinguish data weakness from economic weakness

This is the distinction most often got wrong, and it changes what should happen
next.

- `valuation_age_profile` before quoting **any** LTV metric. A high LTV resting
  on a five-year-old valuation is a measurement problem; the same LTV on a
  recent inspection is a credit one. They need different remediation.
- `data_completeness` and `list_validation_exceptions` before treating a clean
  metric as good news. A field that is 60% populated produces a metric over the
  60%.
- `regulatory_readiness` separately from everything else: submission readiness
  and portfolio quality are different questions and neither implies the other.

### 5. Follow the threads that overlap

Single findings are rarely the interesting ones. Overlap is.

Starting from *8% of balance has LTV above 80%*, the productive questions are:

- Where is it? (`stratify` on region, filtered to the cohort)
- Are its valuations older than the rest of the book?
  (`valuation_age_profile` with filters)
- Are its arrears higher? (`readiness_metrics`, filtered)
- Is it a few loans or many? (`rank_loans`, then one `get_loans`)
- Has it grown? (`period_change`)
- What backs the LTVs? (`explain_values` on a sample)

**Always compare the flagged cohort with the rest of the book.** "The high-LTV
cohort has 9% arrears" means nothing until you know the book is at 2%.

### 6. Compare against previous periods

`period_change`, and read `period_resolution` and `limitations` before quoting
any movement. A comparison across a changed population answers a different
question from the one asked. A test still inside its limit with shrinking
headroom is a forward concern a point-in-time reading misses entirely.

### 7. Inspect the evidence behind material facts

`explain_values` on the figures the conclusion rests on — not on everything.
Provenance for a number nobody is relying on is cost without benefit.

### 8. State what could not be concluded

Explicitly, as findings in their own right:

- no securitisation criteria supplied → **the assessment cannot conclude the
  book meets them.** An absent rulebook is not a passed one.
- no lineage index → provenance is **unknown**, not clean.
- a metric with no evaluator → **unavailable**, never estimated.
- fields absent from the tape → say which, and which metrics they undermine.

---

## Stopping rules

Autonomous investigation without stopping rules is how a review becomes
expensive and no better.

**Keep going when:**

- a real external criterion is breached;
- a screening flag involves material balance (the balance matters, not the ratio);
- several concerns overlap on the same cohort;
- a trend is deteriorating rather than merely present;
- the evidence behind a material figure is weak or unverified;
- the answer would change the overall assessment.

**Stop when:**

- the affected balance is immaterial — say so and move on;
- further drill-down cannot change the conclusion;
- the data needed is not there — that is itself the finding;
- the next step is a judgement a human should make (commercial terms,
  negotiability, appetite);
- the tool or cost budget is reached — and then **say the review was truncated**,
  because a bounded review reported as complete is worse than an honest partial
  one.

**Two cheap heuristics.** Prefer one aggregate call over many loan calls: if the
question is about a population, there is a tool that answers it as an aggregate.
And check the telemetry on each result — `rows_scanned` against `rows_returned`
— because a selectivity trending to 1.0 means the agent is extracting the book
rather than investigating it.

---

## The output contract

What a readiness assessment must produce. **Designed, not built.**

```jsonc
{
  "assessment": {
    "overall": "GREEN | AMBER | RED | INCOMPLETE",
    "overall_basis": "why this rating, in one sentence",
    "is_formulaic": false,          // true only when a config drives it
    "as_at": "2026-07-31",
    "snapshot_id": "snap_...",
    "framework": "SECURITISATION_READINESS@v1"
  },

  "findings": [
    {
      "title": "High-LTV cohort concentrated in stale valuations",
      "category": "collateral",
      "severity": "high",
      "facts": [
        {"metric": "COLL_HIGH_LTV_SHARE", "value": 16.8, "unit": "percent",
         "tool": "readiness_metrics", "snapshot_id": "snap_..."}
      ],
      "rule_source": {              // absent when the finding is judgement alone
        "pack": "EXAMPLE_PROPOSED_SECURITISATION@v1",
        "authority": "securitisation_criteria",
        "is_external_criterion": true,
        "threshold": 20.0,
        "outcome": "pass"
      },
      "judgement": "Attributed to the agent, never to Trakt.",
      "evidence": [{"tool": "explain_values", "loan_id": "...",
                    "canonical_field": "current_loan_to_value"}],
      "affected_balance": 3360000.0,
      "affected_loans": 12,
      "affected_balance_pct": 16.8,
      "recommended_next_step": "Re-value the cohort before the pool is fixed.",
      "confidence": "high | medium | low",
      "completeness": "what this finding could not establish",
      "human_review_required": false
    }
  ],

  "outstanding_information": [
    {"item": "Proposed securitisation criteria", "impact":
     "The assessment cannot conclude the pool meets criteria that were not supplied."}
  ],

  "limitations": [
    "This is not a credit rating and does not predict one.",
    "Trakt screening thresholds are internal attention triggers, not market rules.",
    "Regulatory field coverage is not evidence of portfolio quality.",
    "Assessed as at the snapshot named above."
  ],

  "coverage": { "framework_metrics_assessed": 41, "unavailable": 7 },
  "audit": { "correlation_id": "...", "tool_calls": 23 }
}
```

**Design notes that matter more than the shape:**

- `overall` is **not formulaic by default**. A colour computed from a flag count
  would imply Trakt has a view it does not have. Where a client wants a formula,
  it belongs in configuration with a version, and `is_formulaic` says so.
- `INCOMPLETE` is a first-class outcome. A review that could not obtain what it
  needed is not AMBER.
- Every finding separates `facts` (Trakt's) from `judgement` (the agent's), and
  `rule_source` carries `authority` so a screening flag can never be rendered as
  a requirement.
- `affected_balance` sits beside every finding because materiality is the thing
  a reviewer needs and a percentage alone does not give it.
- `human_review_required` is how the agent escalates rather than guessing.

---

## What the agent must never do

1. Compute a portfolio metric itself. Every number comes from a tool.
2. Report a Trakt screening flag as a breach, a market rule, or a regulatory
   requirement.
3. Treat an absent rulebook as a satisfied one.
4. Treat a clean regulatory projection as evidence of a sound book, or a poor
   one as evidence of a bad book.
5. Estimate a metric that returned unavailable.
6. Predict or imply a credit rating.
7. Present an unseasoned book's clean performance record as evidence of quality.
8. Report a truncated review as a complete one.
