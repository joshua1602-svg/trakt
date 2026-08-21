# Bounded pre-Stage-2 check — are the bank's answer-type expectations evidence?

**Result: findings present. Stage 2 has not started.**

| | |
|---|---|
| Base | merge-base with `claude/mi-analytical-capability-layer-vlkjfw` is `4e051f3` exactly; `4e051f3` ✓ and `28ece25` ✓ ancestors of HEAD |
| Scope | 252 calibration cases, real alderbridge tape |
| Production code changed | none |
| Reproduce | `python -m question_interpretation.check_answer_type_expectations` |

---

## The question asked

All 252 calibration expectations carry `expected_answer_type`, derived from
`answer_type.asked()` — a classifier that disagrees with the parser on 46
questions and that no production path calls. Do any of the 46 correspond to
calibration cases, and do those cases pass for the reason their expectation
states?

## The direct answer

**10 of the 46 are calibration cases.** Of those:

| | n | |
|---|---:|---|
| evaluated, and **pass strictly** — `observed == expected` | **4** | `kpi_009`, `kpi_010`, `kpi_032`, `rank_172` |
| **never evaluated at all** | **6** | `pipe_183`, `pipe_184`, `pipe_188`, `pipe_194` (parse_only) · `risk_223`, `dq_231` (refuse/clarify) |
| pass only via permissiveness | 0 | |

So on the specific hypothesis — that a case in the 46 passes through
`satisfies()` leniency — **the answer is no.** The four that are graded are
graded strictly.

**But the check found the class it was looking for somewhere else**, and it
found something larger on the way.

---

## Finding 1 — four cases whose expectation cannot detect its own regression

`kpi_028` *portfolio summary* · `kpi_029` *portfolio overview* ·
`kpi_030` *book overview* · `kpi_031` *key metrics*

| | |
|---|---|
| `expected_answer_type` | `mixed` — "legitimately carries several measures at once" |
| `observed` via `of_measure` | `count` |
| Passes because | `_SATISFIES[MIXED]` contains `COUNT` |

This is not an `asked()` problem — `asked()` returns `mixed` and the stored
expectation agrees. **The defect is on the observed side.**

`portfolio summary` genuinely returns two measures:

```
result columns: ['loan_count', 'current_outstanding_balance_sum']
spec:           metric=None  aggregation='count'  measures=[]
```

`of_measure` types an answer from a **single** `metric` + `aggregation`. A
summary carries `metric=None`, so it types as `count` regardless of how many
measures the answer actually carries.

**The consequence, verified by construction:** a portfolio summary that lost the
balance column entirely and returned only `loan_count` would produce the
identical spec — `metric=None, aggregation='count'` — and pass identically.

```
satisfies('mixed', of_measure(None, 'count'))  ->  True
```

The expectation says "several measures". The check it runs cannot tell one
measure from several. **Four cases assert a property nothing verifies** — the
right-for-wrong-reason class, on the observed side rather than the expected
side.

## Finding 2 — 121 of 252 expectations are never evaluated

The answer-type check sits at the end of `evaluate_case`, after four early
returns. On this book it is reached for **131 of 252** cases:

| Never evaluated | n | Why |
|---|---:|---|
| required field absent on this book | 79 | returns at the missing-field gate |
| `execution: parse_only` | 21 | returns before the type check |
| `expected_status: refuse` / `clarify` | 21 | returns before the type check |
| **total** | **121** | |

Some of this is correct — a refusal has no answer type to check. But **the
`parse_only` cases are the ones that matter here**: `pipe_183` *pipeline amount
by stage* and `pipe_194` *pipeline amount by broker* declare
`expected_answer_type: currency` while the parser produces
`aggregation=count, metric=None`. That is precisely the disagreement, sitting in
the bank, **structurally invisible to it**.

This is not a regression and nothing is failing. It is a statement about how
much of the bank's answer-type coverage is load-bearing: **just over half.**

## Finding 3 — cleared: the 56 apparent stale derivations are not drift

56 of 252 stored expectations differ from `asked()` today. On inspection this is
the derivation working as documented, not decay:

| Pattern | n | Verdict |
|---|---:|---|
| `expected=currency`, `asked=any` | 35 | **33 carry an `expected_metric` that justifies currency** — 27 `current_outstanding_balance`, 5 `current_valuation_amount`, 1 `original_principal_balance`. The derivation's documented cross-check against the declared metric resolved them. Correct. |
| `expected=none`, `asked=*` | 21 | `none` is authored from `expected_status`, not from `asked()`. Disagreement is by design. |
| residual | 2 | `pipe_189` never evaluated; `unsup_239` passes strictly |

**No case's expectation has gone stale against the classifier that produced it.**

---

## What this means for the programme

Findings 1 and 2 are **both about the bank, not about the object**, and neither
blocks Stage 2 on its own merits — nothing is failing, nothing is unsafe, and
Stage 2 changes no behaviour. They are reported because the instruction was to
report and stop, and because two of them bear directly on later stages:

* **Finding 1 weakens `mixed` as an acceptance type.** Any stage that touches
  multi-measure answers — Stage 4's role split, Stage 5's time axis — would be
  graded on four cases that cannot detect a lost measure. Whatever else is
  decided, those four should not be counted as evidence for those stages.
* **Finding 2 bounds what "the calibration bank is green" means** for answer
  types specifically: it means 131 cases held, not 252.
* Both reinforce the standing rule already recorded: **the calibration bank must
  not be regenerated during this programme.** Regenerating would rewrite these
  expectations from a changed classifier, and a bank that moves with the code it
  grades has stopped being a control.

## Options, for decision

Stated without preference; this is a decision, not a task.

1. **Proceed to Stage 2 unchanged**, recording Findings 1 and 2 as known limits
   of the acceptance surface. Defensible: Stage 2's acceptance is
   byte-identical answers, which does not depend on answer-type expectations at
   all.
2. **Fix Finding 1 first**, by making `of_measure` read the result's actual
   measure set rather than a single spec slot. That is a production change to a
   grading path, outside the contract's stage sequence, and would need its own
   before/after.
3. **Narrow `_SATISFIES[MIXED]`** so `count` alone no longer satisfies `mixed`.
   Smaller, but it would likely fail those four cases immediately, which is
   Finding 1 becoming visible rather than fixed.

**Stage 2 has not started, and will not until this is settled.**
