# The composite boundary — scope, before the routing fix

Base `ca6323b`. **Nothing built.** The scope has produced a different rule shape
from the one anticipated, so it is reported before writing, as every previous
cycle has done when that happened.

> **The rule is not about datasets at all.** *"Read the pipeline"* and *"compose
> funded and pipeline"* do not need separating, because the estate already
> separates them — and the separation is not a dataset rule, it is a **disclosure
> rule that one route already implements and five do not.**

---

## 1. The separation already exists, and it is derivation vs assertion

`reconciliation.dataset` is written in two completely different ways in this
estate.

**Derived — `mi_workflows/analytical/route.py:312`:**

```python
def _reconciliation(result) -> Dict[str, Any]:
    """The datasets this plan actually read.

    Named from the capabilities that ran, not assumed: an offer-stage question
    reads the pipeline and nothing else, and reporting it as a full-coverage
    funded answer would misdescribe what was measured.
    """
    datasets = [...for call in result.plan.calls...]
    return {"dataset": "+".join(datasets) or "funded", ...}
```

**Asserted — a string literal at the route's own return site:**

| line | literal | route |
|---|---|---|
| 710 | `"dataset": "funded"` | `_route_portfolio_summary` |
| 881 | `"dataset": "funded"` | `_route_period_movement` |
| 1925 | `"dataset": "funded"` | `_route_risk` |
| 2070 | `"dataset": "funded"` | `_route_bridge` |
| 2199 | `"dataset": "funded"` | `_route_cohort_progression` |
| 1239 | `"dataset": "pipeline"` | `_route_evolution` |
| 1544 | `"dataset": "forecast"` | `_route_forecast` |
| 977 | `"dataset": dataset` (a variable) | `_route_compare` |

**The three substituting routes are exactly three of the hard-coded `"funded"`
sites** — `_route_portfolio_summary`, `_route_risk`, `_route_bridge`. That is not
a coincidence and it is the whole diagnosis.

### So the composite needs no carve-out

`funded+pipeline` is not a special case to protect from a dataset rule. **It is
what a derived reconciliation looks like when two capabilities ran.** The
composition is self-describing: `_reconciliation` joins the datasets the plan
actually read, so a forecast that reads funded and pipeline says so, and an
offer-stage question that reads only the pipeline says that instead — from the
same code, with no branch.

**"Read the pipeline" and "compose funded and pipeline" are the same rule
producing different answers.** Nothing distinguishes them because nothing needs
to.

### The rule, stated

> **A route must be able to say what it read, derived from what it read.**

Not *"honour the named dataset"*. A route that derives `funded` for a question
that named `pipeline` is then a **detectable, disclosable disagreement** — the
completeness check already finds exactly that shape — instead of a silent one.
Whether such a route should then narrow, refuse, or disclose is a separate
decision that this rule makes *possible* and does not make.

The three substituting routes today cannot be wrong about themselves in any
detectable way, because they assert a constant. That is the defect. They are not
reading the wrong dataset by choice — `_route_portfolio_summary` takes no dataset
parameter at all — they are **unable to say what they read, and the constant they
assert happens to be false.**

---

## 2. The enforcement gap — fail-open guard, third instance

`chat_routing.py` states the precondition twice, in comments, and calls the owner
zero times:

```
:262   "`workspace.resolve_dataset`, which is the single owner. Routes ask that owner"
:3495  "NO `view=`. The dataset is the question's, and the route asks
        `workspace.resolve_dataset` for it."
```

**All four occurrences of `resolve_dataset` in the file are comments.** `try_route`
receives `view` and spends it on the value catalogue, the interpretation
projection and the ownership re-read — never on the answering frame.

That is the same shape as Q19C's second cause, and the third instance in this
programme of **a documented precondition nothing enforces — a fail-open guard.**
The first two:

| | precondition | enforced by |
|---|---|---|
| Q19C | routes must publish the narrowing they performed | nothing, until `scopeApplied` |
| `_unknown_named_book` | a capitalised run before a book noun is a proper name *unless generic* | a hand-maintained word list, incomplete by four |
| **this** | routes ask `resolve_dataset` for the dataset | **nothing — the sentence exists only in comments** |

A precondition written in a comment is a wish. The pattern is worth naming
because all three failed the same way: the estate knew the rule, wrote it down,
and had no mechanism that could tell whether it held.

---

## 3. The regression check, registered

The five currently-correct pipeline answers, captured at `ca6323b` from the
repository root. **Any change to the routing must reproduce all six of these
exactly** (the sixth, Q10B, is the sibling that proves the capability works):

| question | route | reconciled | answer |
|---|---|---|---|
| What is the pipeline balance? | *(point-in-time)* | `pipeline` | Balance: £3.6MM · 8 loans. |
| How many cases are in the pipeline? | *(point-in-time)* | `pipeline` | 8 loans · Current Outstanding Balance: £3.6MM. |
| Show the pipeline by stage. | *(point-in-time)* | `pipeline` | 5 groups · grouped by Pipeline Stage · 8 loans |
| How has the pipeline evolved? | `evolution` | `pipeline` | Pipeline amount over 5 period(s): latest £3.6m (up over the window). |
| Show pipeline evolution by stage. | `evolution_pipeline_stage` | `pipeline` | 5 period(s): APPLICATION, COMPLETED, KFI, OFFER, WITHDRAWN |
| Give me an overview of the pipeline by size and stage. | *(point-in-time)* | `pipeline` | 5 groups · grouped by Pipeline Stage · 8 loans |

Stored at `ds_regression.json`. Note three of the six are **not routed at all** —
they fall to the point-in-time path, which honours the dataset today. A routing
change must not capture them.

Beyond these, the risk zone is the 32 delivering route-claimed questions on a
non-default dataset measured in `MI_DATASET_CLASS_SCOPE.md`, of which nine are
`funded+pipeline` compositions.

---

## 4. Why I stopped here

The brief said the rule cannot be *"honour the named dataset"*, and asked how the
two cases separate. **The answer is that they do not need separating**, which
means the change is not the one the brief anticipated: it is not a dataset-
selection rule in `try_route`, it is **five route return sites learning to derive
what they assert**, following a pattern one route already implements.

That is a different and larger surface than the four questions it recovers —
`_route_period_movement` and `_route_cohort_progression` are in the same class and
did not appear in the substitution set only because no question naming a
non-default dataset reached them. Whether to fix three sites or five, and whether
deriving is enough or the routes should then also narrow, are decisions the scope
has surfaced and not settled.

So: reported, not written. The regression check is registered and the rule is
stated. Say which and I will build it.

### Environment
`MI_AGENT_LLM_PARSER=off` (F2), run from the repository root (F6 — which cost a
measurement in this very session; see the standing findings).
**Successful model responses: 0.**
