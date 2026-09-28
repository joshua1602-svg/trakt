# Slice 1B — serving canary, local acceptance

The governed plan becomes the served answer for ONE allow-listed principal. This
directory holds the local acceptance that proves it, with **no live model call**.

## What is here

| File | What it is |
|---|---|
| `bank.json` | The pre-registered bank. Written and hashed BEFORE the harness ran. |
| `run_bank.py` | The harness. Reads `bank.json`, adjudicates, writes the report. |
| `slice1b_local_acceptance.json` | The result of the run recorded in the Slice 1B commit. |

`bank.json` sha256: `c44b34358fd2a1156052e11207250133bb02ecd1a71bb9fbc32574005b3b49e9`

## Re-running it

```
python due_diligence/evidence/slice1b_serving_canary/run_bank.py
```

Exit 0 is PASS. It needs no credentials, no network and no deployment: the
interpreter is a `ReplayClient` over the frozen run-8 sign-off payloads
(`mi_agent/interpretation_v2/evidence/run8_135_signoff_2b00172.json`), so a
question outside that file cannot be asked and no model is ever contacted.

## The bank

Six eligible cases, covering every Slice-1-expressible family present in the
frozen corpus, and two ineligible controls.

| Case | Shape | Expected |
|---|---|---|
| E1 | `count`, two numeric predicates | served NEW |
| E2 | `sum` of balance, two numeric predicates | served NEW |
| E3 | `count`, a categorical predicate and a numeric one | served NEW |
| E4 | `sum` of balance by LTV bucket × age bucket | served NEW |
| E5 | `sum` of balance by LTV bucket × ticket bucket | served NEW |
| E6 | `sum` of balance by LTV bucket × interest-rate bucket | served NEW |
| C1 | an explicit Direct/Acquired lens | `EXPLICIT_LENS` → legacy |
| C2 | a governed geography axis | `GEOGRAPHY_REQUESTED` → legacy |

## What each eligible case has to prove

1. **Plan identity.** The plan the frozen interpreter and the deterministic
   compiler produce is the plan the bank registered — same `plan_id`, same
   statistic, same measure field, same axes, same predicates with their
   comparators.
2. **Slice 1A and Slice 1B agree.** The shadow and the serving path are run on
   the same question and the same frame, and must produce the same figure. Slice
   1B changes *whether* a result ships, never what it is.
3. **The independent oracle.** The served figure, and every served cell, equals
   what `mi_agent/tests/portfolio_truth_oracle.py` computes. That module imports
   nothing from the product — a boolean mask, a `groupby` and a sum, written
   longhand — so this is not the product marking its own homework.
4. **Served provenance.** The envelope handed back carries the deterministic
   figure, read off its own KPI or table artefact, and is not the legacy
   control's. The record's `serving.response_served_from` must agree.

The two controls must be refused for the exact registered reason, and the legacy
envelope must be what serves.

## The frame

The oracle's `canonical_book(400, 20260905)`, plus two band columns the deployed
frame carries and the canonical book does not — `ticket_bucket` and
`interest_rate_bucket`. The harness derives them; the product executor and the
oracle then read the SAME column, so the banding rule is irrelevant to every
claim made here. This is about filtering, grouping and aggregation.

## Isolation

Proved before anything is served, under the same configuration:

* the allow-listed principal is handled;
* a second individual **on the same client** is not;
* a request with no identity is not;
* the membership test builds no interpreter and writes no record.

## What this acceptance does NOT prove

It is local. It does not prove the deployed App Service serves the new path, and
it spends no live interpretation. That is the next step, and it needs a
deployment and a configured principal — see the Slice 1B report.
