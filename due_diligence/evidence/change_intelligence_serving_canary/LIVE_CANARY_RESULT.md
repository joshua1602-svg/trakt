# Live serving canary — the two new arrows work; four cases need classifying

```
BANK              change_intelligence_serving_canary
BANK_SHA256       27b7d825a1b3f88e631c5931826fa0b0fdab331205ca7b4ea7d318c43aa627b1
                  asserted in CI before the first question
DEPLOYED_SHA      88cf5fde759c29f1cff29b9adbf003a2d6ebd530   PROVED (source='artefact')
RUN               https://github.com/joshua1602-svg/trakt/actions/runs/34752378644
ARTEFACT          serving-canary / serving_canary_result.json  (ID 10315767836)

QUESTIONS_ASKED   10 of 10, once each, no retries, no rephrasing
PROVIDER_FAILURES 0        AUTH 200 on the re-minted bearer
CONFIG_CHANGES    0        MI_AGENT_PLAN_SERVE set by the operator, not by CI

VERDICT           FAIL     6 / 10
```

## What the sprint built, measured live

```
S01  PASS  material_summary   route=governed_plan_material_summary
S02  PASS  material_summary   route=governed_plan_material_summary
S05  PASS  attribution        route=governed_plan_attribution
S06  PASS  attribution        route=governed_plan_attribution
```

**Both missing arrows are closed in production.** Four for four, across both temporal
routes the forms admit — S01 and S05 state an explicit pair, S02 and S06 state no
temporal relation at all and the owner's authorised `current_vs_previous` default
supplied the window. Each carries the pinned `calculation_owner` in its receipt
(`run_period_change_analysis` for material_summary, `balance_bridge` for
attribution), the pinned `composition_owner` (`insight_funded.compose`, and `None`
for attribution, which composes nothing), and arrived through the route its case
pinned. Before this sprint every one of these four was refused
`CAPABILITY_NOT_GENERIC` and answered by the legacy path.

```
S04  PASS  metric_delta       route=period_change_analysis
S10  PASS  (negative control) route=None — the ungoverned portfolio was NOT answered
```

S10 is the control that matters most for safety: a question naming a portfolio the
governed source registry does not hold was declined, not answered over the whole
book under the reader's words.

## The four failures, and how far I can honestly classify each

```
S03  FAIL  metric_delta       route=period_change
           outcome: REFUSE, pinned ANSWER
S07  FAIL  level_comparison   route=None
           temporal route 'ABSENT' is not in [relative_pair, previous_reporting_period,
                                              explicit_period]
S08  FAIL  level_comparison   route=None    (same)
S09  FAIL  material_summary   route=period_change
           outcome: REFUSE, pinned ANSWER;
           route: 'period_change', pinned 'governed_plan_material_summary';
           calculation_owner: None, pinned 'mi_agent.period_change.workflow.…'
```

### S07 and S08 — MY INSTRUMENT IS SUSPECT, and this is not a product finding

The scorer's temporal read is wrong for the oracle it reads from, and I verified
that rather than inferred it.

`CandidateIntent.to_dict()` is a dataclass `asdict`, so the recorded intent ALWAYS
carries a `time` block with a `form` key — measured:

```
states a pair    time = {"form": "relative_pair", …, "stated": true}
states nothing   time = {"form": "current",       …, "stated": false}
```

Presence is `stated`. My scorer tests KEY PRESENCE (`"form" in time`), which is the
right read for the model's RAW payload and the wrong one for the parsed dataclass —
the exact distinction the temporal presence repair exists to make, misapplied by the
instrument built to measure it. Whatever produced `'ABSENT'` here, the reader is
demonstrably not reading presence the way the repaired contract defines it.

**So S07 and S08 must not be called an interpretation defect until the artefact's
`candidate_intent` is read.** They may be a genuine miss, they may be my scorer, and
the run log alone cannot tell them apart. Both questions plainly name two reporting
states, and v4 measured `level_comparison` at 4/4 on stated pairs, which makes the
instrument the more likely of the two.

### S09 — the scoped material summary did NOT reach the new adapter

This is the most consequential of the four, and it is a real gap against the offline
control A4, which proved an Acquired scope surviving end to end on fixtures.

The route was `period_change` and the receipt carried no `calculation_owner`, so the
governed plan path did not serve it — the legacy route did, and declined. The most
likely cause is the fail-closed scope path behaving exactly as built: if the governed
portfolio registry cannot resolve the `acquired` role to portfolio ids for this
client, the lens stays a type lens, `_apply_lens_filter` RAISES rather than widening,
`_attempt_change_form` records the failure and the legacy envelope serves. Control
A4b pins that behaviour deliberately.

**The direction is the safe one** — a scoped question was refused rather than
answered over the whole book with the scope named in the receipt, which is the exact
historical defect (£22.6m reported for a book that moved £12.4m). But refusing is not
serving, and until the recorded error is read this is unclassified: registry
resolution for this client, or an adapter fault.

### S03 — the legacy route declined a metric delta

`metric_delta` is a form this sprint deliberately did not touch, and it arrived at
the path it has always arrived at. The decline is therefore not a regression from
this sprint's code, but it is unexplained: the question names both a measure and two
reporting states, which is what the route's "no implicit comparison period" guard
exists to require. S04 — the other metric delta — passed through
`period_change_analysis`, a DIFFERENT route from S03's `period_change`, so the two
were claimed by different owners. That divergence is worth understanding on its own.

## What is proved, and what is not

```
MATERIAL_SUMMARY_SERVES_LIVE      YES   S01, S02
ATTRIBUTION_SERVES_LIVE           YES   S05, S06
OWNER_DEFAULT_WINDOW_LIVE         YES   S02, S06
PINNED_CALCULATION_OWNER_RECORDED YES   4 / 4
UNGOVERNED_PORTFOLIO_REFUSED      YES   S10
SCOPED_CHANGE_QUESTION_SERVED     NO    S09 — refused, not widened
NUMERIC_PARITY_AGAINST_THE_OWNER  NOT MEASURED
```

**No independent numeric parity was established and none is claimed.** The bank says
numeric truth comes from running the deterministic owner independently over the same
governed snapshots; CI has no governed book, so that cannot be done there. Each case
records what a reader with data access needs to reconcile it — the resolved pair, the
snapshot references, the executed scope, and the bridge's own reconciliation flag and
residual. Offline parity against fixtures is the separate, passing gate. Conflating
the two would be claiming a measurement nobody took.

## Integrity

```
questions asked        10, once each; no case asked twice, none rephrased, none re-run
expectations           unchanged after seeing results — the bank hash is the one
                       asserted in CI before the first question
product code           unchanged since the tested head, asserted in CI by diffing
historical manifests   v1, v2, v3, v4 all re-hashed intact
MI_AGENT_PLAN_SERVE    not read, not written, and not inferred by any workflow
```

**The bank is now spent.** It authorised one attempt with `retries: 0` and all ten
were asked. Any further live measurement needs a new pre-registered bank.

**A guard I claimed and did not have.** I described the one-attempt protection as
"enforced by the filesystem" — the runner refuses to start if
`serving_canary_result.json` exists. Each CI run gets a fresh checkout, so that file
never pre-existed and the guard could not fire. The real protection is committing the
result, which had not happened. `serving-canary-run.yml` is push-triggered on its own
path and must not be edited again.

## The two instrument faults this run cost, both mine

1. **The provenance probe.** The first attempt read `/health`, which the app's own
   docstring warns against because it RESOLVES the governed tape; on a cold process it
   returned nothing usable while the same service answered `/mi/catalogue` with 200.
   The run stopped on a provenance failure that said nothing about which commit was
   deployed. `/` is the app's designated liveness probe and carries the same stamp.
   Cost: zero — the gate ordering meant the ten were skipped.
2. **The temporal read**, above. Cost: two cases that cannot yet be classified.
