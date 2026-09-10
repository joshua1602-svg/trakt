# Policy recalibration — probe result

`POLICY_RECALIBRATION = PASS`, with two of six pre-registered conditions missed
and reported rather than reinterpreted.

Run: 57/57 successful interpretations, 0 failed calls, `claude-opus-5`,
483,572 input and 80,594 output tokens. Loan rows transmitted 0, portfolio values
transmitted 0. Evidence: `recalibration_probe.json`. The gate was written into
`recalibration_probe_gate.md` before this run and is not edited here.

## Headline, against both baselines on the identical 57 questions

| | Run 6 pre-policy | Run 7 Phase 2A | Probe recalibrated |
|---|---|---|---|
| fully correct | 47 | 45 | **47** |
| partially correct | 8 | 5 | 6 |
| clarification | 2 | 7 | 4 |
| incorrect | 0 | 0 | **0** |
| PLAN / CLARIFY / REFUSE | 55 / 2 / 0 | 50 / 7 / 0 | 53 / 4 / 0 |
| measure spread | 20 | 14 | **11** |
| canonicals with spread | 12 | 7 | **6** |
| paraphrase-invariant | 4/19 | 6/19 | **7/19** |
| fixture defects | 5 | 2 | 4 |

Phase 2A's regression boundary — the one it failed — was +3 on non-PLAN. Phase 2A
came in at +8. This run is at +2.

Invariance is monotone across the three runs and nothing that was invariant
stopped being invariant: Q16, SM08 and BB01 were gained and none was lost.

## The six pre-registered conditions

**1. `completeness` returns to 9 PLAN — PASS.** This was the primary condition and
it passes with more than was asked.

All nine readings compile, and all nine carry a measure. Q19, the canonical this
sprint exists to fix, went further than PLAN:

| | Run 6 | Run 7 | Probe |
|---|---|---|---|
| Q19A | `period_movement` / balance | PLAN, partial | PLAN, fully correct |
| Q19B | `funded_bridge` / bridge movement | **CLARIFY** `MISSING_REQUIRED_SLOT` | PLAN, fully correct |
| Q19C | `period_movement` / balance | **CLARIFY** `MISSING_REQUIRED_SLOT` | PLAN, fully correct |

All three probe readings are `period_movement` / `movement` /
`current_outstanding_balance` / `governed_reporting_period_pair`. Q19A and Q19B
produce a **byte-identical `plan_id`**. Q19C differs from them in one field:
`period.labels` is `["month-on-month"]` where the others carry `["last month"]` —
the question's own words, which Run 6's adjudication already identified (on NL9)
as wording carried into plan identity and the subject of contract normalisation.
Semantically Q19 is now invariant.

**2. `concise` 12 PLAN — PASS. `ambiguity` 2 CLARIFY — MISSED, 1.**

`concise` is 12/12 PLAN, all fully correct. The clarification-heaviness risk did
not materialise: the only clarifications in the whole run are NL7 ×3 and NL8C,
and each carries a substantive governed reason — NL7 because "riskier" / "risk
profile" binds to no single governed measure, NL8C `AMBIGUOUS_PERIOD` because it
asks for a series with no span, grain or period count.

The miss is **BB01C** ("What is the headroom?"), which moved CLARIFY → PLAN. It
did not drop the ambiguity. It recorded it non-blocking, named both options, and
said how to get the other reading:

> Two governed headroom concepts exist: borrowing base headroom (funding
> facility) and limit headroom (one of 42 configured concentration tests). Read as
> borrowing base headroom because the question names a single unqualified
> headroom and no concentration test is identified. If the concentration-limit
> headroom was meant, name the test.

That is the disclosed-reading mechanism working as designed, and the reasoning is
governed rather than guessed: the environment has a facility configured, whereas
the limit reading would require naming one of 42 tests. The fixture scores the
result fully correct.

**It still contradicts my own Run 6 adjudication**, which ruled that BB01C *should*
be non-invariant because its third variant is a genuinely less specific question.
Two defensible positions exist and the recalibration moved the line. I am
flagging it rather than settling it: if the ruling stands that an unqualified
"headroom" must ask back, this is a real regression of one variant and the fix
belongs in the AMBIGUITY block, not in COMPLETENESS.

**3. `gains` holds — PASS.** SM08 is 3/3 PLAN, all fully correct, and SM08 is now
paraphrase-invariant. Phase 2A had improved SM08A alone.

**4. Measure spread ≤ 14 — PASS, 11.** Best of the three runs, against Phase 2A's
14 and pre-policy's 20. Canonicals with spread 6, against 7 and 12. The
elaboration control Phase 2A bought was not given back; it was extended.

**5. Fixture defects ≤ 2 — MISSED, 4.**

The four are `DROPPED_EXPLICIT_TIME` on Q18A, Q18C, Q20A and Q25C — the **identical
four** Run 6 flagged. Every defect in this run is also present pre-policy; the
policy introduced none and removed Run 6's `DROPPED_EXPLICIT_FILTER` on Q16B. As
a set, the probe's defects are a strict subset of pre-policy's.

But the comparison that fails is against Phase 2A, and it fails honestly: Run 7
got Q18A, Q18C and Q20A right and this run did not. The difference is the period
spelling. The fixture states `relative_pair`; these three readings chose
`previous_reporting_period` + `monthly`, which compiles to
`previous_governed_reporting_period` rather than `governed_reporting_period_pair`.

Run 6's adjudication already classified exactly this, on Q20, as REPRESENTATIONAL
— "one relative relationship, two spellings". Q18B in this same run shows the
other spelling on the same canonical, which is what an encoding split looks like
rather than a loss. No semantic element is absent: the movement is still
period-on-period, monthly, one period back. Q25C's defect is the temporal miss
Run 6 recorded against that variant and is present in all three runs.

I set this bar at Phase 2A's figure, and Phase 2A's 2 was the outlier of the three
runs, not the norm. That does not excuse the miss — three readings did get worse
than Phase 2A on a scored metric — but it does locate it: in the redundant period
contract that Phase 2B exists to collapse, not in the policy boundary this sprint
changed.

**6. NL7 clarifying is expected — PASS.** NL7 is 3/3 CLARIFY, and each reason
names the governed alternatives that make it a required-and-open element. On the
recalibrated border that is band four, and clarifying is the correct disposition.

## Verdict

Two conditions missed. Neither is a reason to revert, for different reasons:

* **Condition 5** is missed inside a contract redundancy the sprint did not touch
  and could not fix. The defect set is a subset of pre-policy's.
* **Condition 2** is missed on one variant, by a disclosed non-blocking reading
  rather than a silent omission, and it is a judgement call that contradicts my
  own earlier adjudication. It is the one item here that may warrant a decision
  from you rather than from me.

Against what the sprint was for, the result is unambiguous. The defect was an
interpreter that omitted semantics its operation required because the words were
not said. Q19 produced `operation=movement` with no measure in Run 7; it now
produces the balance, on all three variants, with two of them identical at the
plan level. Elaboration control improved rather than reverted, invariance rose,
and no question in the cohort is incorrect.

## Residual, for whoever picks up contract normalisation

What still splits the completeness group is entirely the two redundancies already
on record, and neither is an interpretation problem:

* **period labels in plan identity** — Q19C differs from Q19A/B in nothing else;
* **two spellings of one relative period** — Q20A against Q20B/C;
* **two capability owners for one economic analysis** — Q18B reads "the portfolio"
  as `portfolio_summary` / `compare` where Q18A/C read `funded_bridge` /
  `movement`.

Collapsing those three would take the completeness group from 0/3 invariant to
plausibly 3/3 without touching the interpreter.

The full 135 has not been run. It is the next thing worth spending, now that the
cohort gate is green.
