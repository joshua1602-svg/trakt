# Interest-rate bucket — scope, before implementing

Base `9414b93`, tree clean. Nothing shipped: this is the scope report the fix
waits on.

---

## Your correction is right, and it changes the shape of the answer

I wrote in Stage 4 that `interest_rate_bucket` is "an undeclared field" and
that declaring it is "one registry entry". The first half was too crude. It is
a **derived banding**, defined once in `config/mi/buckets.yaml` alongside the
other bands, and materialised onto the prepared tape rather than arriving from
the source data:

```yaml
interest_rate_bucket:
  source_field: current_interest_rate
  semantic_field: interest_rate_bucket   # engine-materialised; see catalogue
  scale: percent                          # 0-100 ; engine normalises fractions
  edges:  [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0]
  labels: ["<2%","2-3%","3-4%","4-5%","5-6%","6-7%","7-8%",">=8%"]
```

Eight bands are **defined**; five are **populated** on this tape. A question
about `2-3%` would legitimately return nothing — that is data, not a defect.

## Where the derivation lives, and who computes it

**One engine, one definition.**

| | |
|---|---|
| the band definition | `config/mi/buckets.yaml` — edges and labels, one entry per band family |
| the engine | `analytics_lib/buckets.py` · `materialise_buckets(frame, config, target=...)` |
| the loader | `analytics_lib/config_loader.py` · `load_bucket_config()` |

**Nothing else computes bands.** Four call sites invoke that one engine:
`mi_agent_api/funded_prep.py:468`, `mi_agent_api/pipeline_prep.py`,
`mi_agent/states/assembler.py:440`, `mi_agent/states/temporal.py:141`.

## What computes it for the chart path

Nothing does, at chart time. `funded_prep` **materialises the column once**, at
prep, for every band family together:

```python
_DIM_SPEC = {
    "ltv_bucket":           {"kind": "bucket_ltv", "target": "current_loan_to_value"},
    "interest_rate_bucket": {"kind": "bucket",     "source": "current_interest_rate"},
    "ticket_bucket":        {"kind": "bucket",     "source": "current_outstanding_balance"},
    "age_bucket":           {"kind": "bucket",     "source": "youngest_borrower_age"},
}
```

The chart path then **reads that column**. `mi_workflows/analytical/executors.py`
lists it in `PROFILE_DIMENSIONS` as `("interest_rate_bucket", "Interest rate
band")` and groups by it; it does not band anything itself.

Measured on the acceptance tape: `interest_rate_bucket` is present with a value
on **640 of 640 rows**, five distinct bands.

## Can the query path consume the same derivation? YES — it already does, for the other three

**The query path does not compute bands at all.** `ltv_bucket`, `age_bucket`
and `ticket_bucket` all work today by *reading the materialised column*. So does
`interest_rate_bucket` — the column is there and the executor can group by it.

What stops it is one thing and one thing only: the field is **absent from the
MI semantics registry** (`mi_agent/mi_semantics_field_registry.yaml`). And
`_explicit_dimensions` honours a term only where its target key exists in that
registry, so the whole chain in front of it is inert:

| already present | |
|---|---|
| `config/mi/buckets.yaml` | the band definition |
| `funded_prep._DIM_SPEC` | the materialisation |
| `config/mi/pipeline_field_contract.yaml` | `reused: [ltv_bucket, age_bucket, ticket_bucket, interest_rate_bucket]` |
| `config/mi/stratification_catalogue.yaml` | the stratification entry |
| `config/routes/{mi,mna,regulatory_and_mi}_route.yaml` | listed on three routes |
| `llm_query_parser.EXPLICIT_DIMENSION_TERMS` | `"interest rate buckets" → interest_rate_bucket` |
| `mi_agent/semantic_resolver.py` | `"interest rate bucket"`, `"rate bucket"` |
| `mi_agent/quantile_buckets.py` | `interest_rate_bucket → current_interest_rate` |
| `mi_agent_api/workspace.py` | listed in the MI columns |
| `mi_workflows/analytical/executors.py` | `PROFILE_DIMENSIONS` |
| **absent** | **the MI semantics registry entry** |

### Would it create a second computation of the same bands? No.

Not a second computation, and **not even a second call site**. The registry
entry adds no banding code anywhere. It tells the parser that a column the one
engine already materialised is a governed dimension — exactly what the entries
for `ltv_bucket`, `age_bucket` and `ticket_bucket` say about theirs. The
registry template is identical: `source_criteria: [derived_bucket]`,
`derived: true`, `derived_from: <source field>`.

**This is the opposite of the multi-owner pattern.** The multi-owner risk would
arise only if the query path banded the rates itself. It does not, and neither
does the chart path.

### Measured, not argued

A scratch registry (`MI_AGENT_SEMANTICS` pointed at a copy with the entry
added — nothing in the repository changed):

```
Show a table of balance by LTV bucket and interest-rate bucket.  ok  dims=[LTV Bucket, Interest Rate Bucket]  30 groups
Cross-tab balance by LTV band and interest-rate band.            ok  dims=[LTV Bucket, Interest Rate Bucket]  30 groups
Break down outstanding balance by both LTV bucket and rate bucket. ok dims=[LTV Bucket, Interest Rate Bucket] 30 groups
Show balance by interest rate bucket.                            ok  dims=[Interest Rate Bucket]              5 groups
```

30 cells is the **pre-registered truth** for Q13 — `{"axes": ["ltv_bucket",
"interest_rate_bucket"], "cells": 30, "levels": {"ltv_bucket": 6,
"interest_rate_bucket": 5}}` — matched exactly. 5 groups is the five populated
bands.

### So the scope is

**One registry entry, no code, no new computation.** Larger than "one entry" in
the sense that it must be written from the same template as the other three and
declare `derived_from: current_interest_rate` so the estate knows it is a
banding — but it is a declaration, not a derivation.

**What it does NOT settle** and should be decided before shipping: whether the
registry should carry the eight DEFINED bands or the five POPULATED ones in any
disclosure. The other three bucket fields do not enumerate their bands in the
registry either, so following them means saying nothing — consistent, and worth
being deliberate about.

---

## The more urgent half — a refusal that lies about the client's data

> `'interest rate bucket' is not available in this dataset. This book does not
> report it, so the question cannot be answered from the current data (no value
> was fabricated).`

The book reports it, on every row. A refusal that is **false about the client's
data** is worse than a refusal: it is an assertion about their book they may act
on, delivered with the estate's own credibility and a parenthetical boast about
not having fabricated anything.

### Checked for elsewhere: `migration_phase0/data_claim_audit.py`

Over **226 questions** — the 75-bank, the frozen CFO 91, the simple-composition
bank, the generalisation supplement and the robustness residuals — every
refusal whose stated reason is a claim about what the book CONTAINS is
extracted, the thing it names is pulled out, and it is **checked against the
tape**.

| class | n | |
|---|---|---|
| `FALSE_about_the_book` | **3** | Q13A, Q13B, Q13C — all `interest rate bucket` |
| `TRUE_about_the_book` | 4 | `Risk Grade`, `arrears`, `NNEG`, `Highgate Mortgages book` — the tape genuinely lacks each |
| `TRUE_about_a_NAMED_FILTER` | 3 | Q21C `among`, GEN21 `offshore`, ROBUST01 `platinum` |
| `QUOTES_A_MANGLED_PHRASE` | 2 | Q15B `Break Direct- book`, Q17C `Break Direct portfolio` |

**`interest rate bucket` is the only false claim on the whole surface.** That is
the reassuring half.

Two classes are true and still worth naming:

- **`TRUE_about_a_NAMED_FILTER`** is honest for `offshore` and `platinum` — the
  reader named a category and the book has none. It is **misleading for
  Q21C**, where `among` is a preposition the parser bound as a categorical
  value: the reader said nothing about `among` and is told no loans match it.
  Separating those two would need a new reader of the sentence, which this
  programme does not add; the audit reports the class and names the members,
  and Q21C's mis-binding is already on record.
- **`QUOTES_A_MANGLED_PHRASE`** quotes `'Break Direct- book'` back as though it
  were a portfolio the registry does not hold. Nothing false is asserted about
  the data, but the reader is shown a mangled fragment of their own sentence
  presented as a name.

The audit exits non-zero on any false claim and on any change to the
pre-registered set, so this class cannot reappear silently.
