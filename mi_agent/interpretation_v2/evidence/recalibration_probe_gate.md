# Policy recalibration — gate NOT REACHED

`POLICY_RECALIBRATION = NOT ESTABLISHED`

Not PASS. Not FAIL. The probe could not be run to completion, so there is no
behavioural verdict to report, and the two outcomes the brief asks me to choose
between both require one.

## What happened

The supplied `ANTHROPIC_API_KEY` exhausted its credit balance 21 questions into
a 57-question probe. Q18C was the last successful interpretation. Every call
from Q19A onward returned:

```
HTTP 400 invalid_request_error
Your credit balance is too low to access the Anthropic API.
```

A single cheap verification call after the run reproduced the same error, so this
is exhaustion rather than a transient — unlike Run 7's two HTTP 529s, it cannot
be retried away.

Evidence is filed as
`recal_probe_PARTIAL_21of57_api_balance_exhausted.json`, named so it cannot be
mistaken for a complete run, and carrying a `credit_exhaustion_note` that states
the same limits as this document.

## Why the surviving 21 cannot gate the change

The lost 36 readings are not a random two thirds. They are, almost exactly, the
questions this recalibration was designed to move.

| group | canonicals | completed | lost |
|---|---|---|---|
| `completeness` | Q18, Q19, Q20 | Q18 | **Q19, Q20** |
| `policy` | NL6, NL7, Q08, Q10, Q24, Q25, SM09 | Q08, Q10 | NL6, NL7, Q24, Q25, SM09 |
| `preservation` | Q16, Q21 | Q16 | Q21 |
| `concise` | Q01, Q02, Q11, SM01 | Q01, Q02, Q11 | SM01 |
| `ambiguity` | BB01, NL8 | — | **BB01, NL8** |
| `gains` | SM08 | — | **SM08** |

**Q19 is the canonical this sprint exists to fix.** Run 7 produced
`operation=movement` with an empty measure list on Q19B and Q19C, the compiler
raised `MISSING_REQUIRED_SLOT`, and two plans became clarifications. Whether the
recalibrated border repairs that is the one question the probe was built to
answer, and it is unanswered. Q20 — the same movement family, and one of Phase
2A's three improvements — is also unanswered.

Both `ambiguity` controls are gone too, so the opposite risk is equally
unmeasured: nothing here shows whether the new COMPLETENESS check has made the
interpreter clarification-heavy. That was the failure mode Phase 2A fell into,
and COMPLETENESS is the block most likely to cause it again — it tells the model
that silently omitting a required element "is not one of the options", which is
precisely the instruction that could tip a terse question into a question back.

One reading of one variant is not a canonical, so even Q18's clean 3/3 PLAN
speaks only for Q18.

## What the 21 completed readings do show

Scope: Q01, Q02, Q08, Q10, Q11, Q16, Q18 — 21 questions, scored against the same
21 in both prior full runs, so this is like for like.

| | fully | partial | PLAN | CLARIFY | measure spread | defects |
|---|---|---|---|---|---|---|
| Run 6 pre-policy | 17 | 4 | 21 | 0 | 5 | filter 1, time 2 |
| Run 7 Phase 2A | 18 | 3 | 21 | 0 | 6 | none |
| Probe recalibrated | 18 | 3 | 21 | 0 | 5 | time 1 |

Per-question movement:

* **versus Phase 2A: zero questions moved.** Every one of the 21 reached the same
  verdict and the same outcome under both policies.
* **versus pre-policy: one question better** — Q16B, which stopped dropping its
  explicit product filter. That is Phase 2A's preservation gain, surviving the
  recalibration.
* No question got worse against either baseline.

The one difference from Phase 2A is a fixture defect: Q18A carries
`DROPPED_EXPLICIT_TIME` in the probe and in Run 6, but not in Run 7. It is one
reading of one variant on a relative-period encoding, and single-run paraphrase
measurement on this bank is already known to carry roughly ±3 variance, so it is
recorded rather than interpreted.

So on the part of the cohort that completed, the recalibration held Phase 2A's
position exactly and did not reintroduce pre-policy elaboration. That is a
necessary condition, not a sufficient one. It is evidence of no collateral
damage on the controls that ran. It is not evidence that the defect is fixed,
because the questions carrying the defect never ran.

## Why the change is not reverted

The brief says to revert on FAIL. This is not a FAIL — it is an unmeasured run,
and reverting on no evidence would discard the change for a billing failure
rather than for anything it did. The 21 readings that exist point the right way
or are neutral; none points the wrong way.

The change is therefore committed (`859f685`) and explicitly **unvalidated**. The
safety position is unchanged by that, and worth stating plainly:

* `interpretation_v2` remains shadow-only. Nothing calls it from `/mi/query`,
  the React app, Teams, the dashboard or any production path.
* The change is one string constant. An AST comparison confirms that of the 14
  module-level bindings in `opus_interpreter.py`, the only one differing from
  `edceb12` is `SYSTEM_PROMPT`; the compiler, the contract, the registries and
  every production surface are byte-identical.
* 204 interpretation tests pass, including the guards that the compiler and
  contract modules did not move and that the prompt names no bank question and
  no lexical rule.

An unvalidated prompt change to a module nothing executes carries no production
risk. What it carries is an open question, and the honest record is that the
question is open.

## What would close it

A funded `ANTHROPIC_API_KEY`, then:

```
python -m mi_agent.interpretation_v2.policy_probe --live \
  --out mi_agent/interpretation_v2/evidence/recalibration_probe.json
```

57 questions. The run that exhausted the key spent roughly 183k input and 28k
output tokens across 21 successful interpretations, so budget on the order of
500k input and 75k output for the full cohort — the prompt-cache hit rate is
high (27k cached reads per call against 4k fresh input), so most of that is
cache.

The gate to apply on the result, fixed here in advance so it cannot be fitted to
the outcome:

1. **`completeness` returns to 9 PLAN.** Q19 and Q20 must carry a measure and
   compile. This is the primary condition; anything else passing without it is
   not a pass.
2. **`concise` stays 12 PLAN** and `ambiguity` stays 2 CLARIFY. If COMPLETENESS
   has made the interpreter timid, it shows here.
3. **`gains` holds** — SM08A stays fully correct.
4. **Measure spread no worse than 14**, Phase 2A's figure on this cohort, against
   pre-policy 20.
5. **Fixture defects no worse than 2.**
6. NL7 clarifying is acceptable and expected: its measure set is genuinely open,
   which on the recalibrated border makes it a required-and-open element, and
   clarifying is the correct disposition rather than a regression.

Only after that gate passes should the full 135 be spent. Contract normalisation
remains out of scope either way.
