# Teams Portfolio Review — Message Contract, Fallback, Delivery Proof

Final sprint before commercial go-live. Three objectives: finalise the production
message behaviour, guarantee that a blocked autonomous review falls back to the
deterministic brief, and prove the existing Teams delivery mechanism works
end to end.

Production delivery remains **OFF**; no recipient was populated at any point.

---

## A. Message contract

Rendered by `scripts/show_teams_message_contract.py` — the real resolver, the
real generator, the real enrichment path and the real card, against committed
pipeline canonical. Not mock-ups.

### Quiet period

```
Monthly Funded Update — 31 July
Funded balance is £37.3m, +£0 on the month.
  - Loan count is 118 (+0 on the month).
  - Weighted-average LTV is 45.9%, +0.00pp on the month.
  - No material developments were identified in the funded book this period.

Risk Review [clear]
  - No material risks were identified from the checks that were available.
  ! Checks unavailable for this update: concentration tests.
```
**61 words.** This already satisfies §3 and §4 without change. Note the two
qualifications, which are the point of §4: *"from the checks that were
available"*, and the explicit naming of what did not run. The card never says
limits are within tolerance, because nothing tested them.

### Normal (organic growth) period

```
Monthly Funded Update — 30 June
Funded balance is £37.3m, +£554k on the month.
  - Loan count is 118 (+2 on the month).
  - The £554k movement is £701k new lending, £257k redemptions and exits,
    £111k existing-book movement.
```
**56 words.**

### Acquisition period — deterministic core + enrichment

```
Monthly Funded Update — 30 June
Funded balance is £37.3m, +£12.8m on the month.
  - Loan count is 118 (+40 on the month).
  - £12.0m of the £12.8m movement reflects the acquisition of ALP Acquired
    Back Book. The £12.8m movement is £12.0m portfolio additions, £701k new
    lending.
  - Excluding the portfolio added this period, the existing book increased by
    £851k (+3.5%) to £25.3m.

  Management observations
    • funded_composition reports £12.0m of the movement is the ALP Acquired
      Back Book (37 loans, portfolio_type acquired, acquisition_date 30
      September 2024)… This accounts for 93.4% of the £12.8m total movement
      and 32.1% of closing balance. …The reported 52.5% growth is driven by
      the addition, not organic activity.
    • With the acquisition excluded, the continuing book grew £0.9m from
      £24.4m to £25.3m… This is the true operating performance for the period.
```
**208 words** · enrichment `enriched` · gate `DEGRADED` · added 2 · dropped 1.

The §10 three-way distinction is intact: headline growth, acquired
contribution, underlying movement — stated by the deterministic core before the
agent says anything.

### Risk-warning period

```
Monthly Funded Update — 30 June
Funded balance is £42.0m, +£5.3m on the month.
  - Loan count is 118 (+2 on the month).
  - Balance-weighted current LTV moved from 45.8% to 48.7% (+2.9pp).
  - The £5.3m movement is £4.9m existing-book movement, £701k new lending.
  - 75-80 moved from 23.5% to 31.1% of the funded book (+7.6pp).
  - 2 further observations are in Trakt.

  Management observations
    • …£4.9m movement on continuing loans within a £42m equity release book…
      concentrated the book's single largest exposure to £5.7m at 71.0% LTV…
      This warrants immediate investigation of the underlying loan-level
      changes.

Risk Review [attention] — weighted-average LTV increased +2.9pp
  - Balance-weighted current LTV moved from 45.8% to 48.7% (+2.9pp).
  ! Checks unavailable for this update: concentration tests.
```
**178 words** · enrichment `enriched` · gate `DEGRADED` · added 1 · dropped 1.

The Risk Review escalates to `attention` on its own governed evidence. No
synthetic or proposed rulebook is cited anywhere.

### Blocked autonomous period — the case that used to be silence

```
Monthly Funded Update — 30 June
Funded balance is £34.7m, +£10.3m on the month.
  - Loan count is 118 (+40 on the month).
  - The acquisition of ALP Acquired Back Book added £12.0m, against a net
    movement of +£10.3m. The £10.3m movement is £12.0m portfolio additions,
    £2.3m existing-book movement.
  - Excluding the portfolio added this period, the existing book decreased by
    £1.7m (-6.9%) to £22.8m.

Risk Review [clear]
  - No material risks were identified from the checks that were available.
  ! Checks unavailable for this update: concentration tests.
```
**82 words** · enrichment `blocked` · gate `BLOCKED` · added 0.

**This is the sprint's headline result.** The gate still blocked the review —
correctly, it contained a `116%` share and two derived figures — and the reader
still receives the briefing, *including the finding that matters most*: the
underlying book fell 6.9% while the headline rose. Nothing on the card mentions
that enrichment was attempted or withheld.

---

## B. Fallback implementation

`trakt_notifications/enrichment.py`, called from `trigger._run` immediately
after `generate.build` and before contract validation.

```
generate.build(...)          # the guaranteed briefing, unchanged
    ↓
enrichment.enrich(batch, reviewer=...)   # additive; cannot raise
    ↓
validate(batch)              # an enriched batch meets the same contract
```

### Why it is a separate module

`generate.build` produces the thing that must not fail. If enrichment lived
inside it, every future edit to the optional half would be an edit to the
mandatory half. The dependency runs one way — enrichment imports the batch, the
batch knows nothing about enrichment — so there is no path by which a change to
the agent layer can alter what the deterministic layer produces.

`enrich` catches `Exception`. That is normally a smell and is correct here for
the same reason `on_publication_approved` does it: the alternative is that an
unforeseeable failure in the optional half suppresses the mandatory half.

### The four states, and the tests that hold them

| State | Delivered | Test |
|---|---|---|
| `PUBLISHABLE` | deterministic core + 2–4 observations | `test_publishable_enrichment_adds_observations_to_the_deterministic_card` |
| `DEGRADED` | core + surviving observations; dropped ones **never referenced** | `test_degraded_enrichment_delivers_what_survived_and_never_names_the_rest` |
| `BLOCKED` | deterministic brief only | `test_a_blocked_review_still_delivers_the_deterministic_briefing` |
| runtime failure | deterministic brief only | `test_any_autonomous_runtime_failure_still_delivers_the_briefing` |
| not configured | deterministic brief only | `test_no_reviewer_configured_is_not_a_failure` |

The runtime-failure test is parameterised over `RuntimeError`, `TimeoutError`,
`ValueError("credit balance is too low")` and `KeyError` — the failures actually
observed — because the invariant is not "we handled the errors we thought of".

`test_enrichment_never_removes_a_deterministic_item` asserts the additive
property directly: across all four reviewer states, the deterministic items
`generate.build` produced are still present and in order.

### What the reader never sees

A dropped finding, an error, a stack trace, a note that enrichment was
attempted, or a gap where it would have been. A briefing that apologises for its
optional half is worse than one that does not have it. The operator sees all of
it in `batch.provenance['enrichment']`.

---

## C. Deterministic results (§14)

`scripts/run_deterministic_period_suite.py` · `tests/test_deterministic_period_suite.py`.

**10/10 produced a correct brief. 423 words total, mean 42 — byte-identical to
the pre-sprint run.** The deterministic analytics freeze (§11) held.

| # | Period | Insights | Key assertion |
|---|---|---|---|
| 1 | quiet | 0 | no manufactured finding |
| 2 | organic growth | 2 info | no acquisition language |
| 3 | acquisition | 3, one attention | reconciliation + underlying stated |
| 4 | acquisition masking decline | 3, one attention | **both directions**: +£10.3m headline, −£1.7m (−6.9%) underlying |
| 5 | concentration warning | 6, one attention | LTV +2.9pp; regional +9.8pp |
| 6 | no approved limits | 2 info | silence, not reassurance |
| 7 | shrinking book | 4, one attention | reported as a fall |
| 8 | disposal | 4 | −£12.0m, −32.1pp |
| 9 | multi-portfolio (5 books, 2 arrivals) | 4, one attention | names **JV Partner Book** from the governed label |
| 10 | second client (`client9`) | 2 info | served from its own data |

No client leakage, no manufactured risk claim, no analytical drift.

---

## D. Real-model results (§15)

Five period types, one pass each, on the configured model, ~$5.

| Scenario | Gate | Verdict | Steps | Out-of-mandate | Figures rejected | Findings | Words | Prohibited terms |
|---|---|---|---|---|---|---|---|---|
| A acquisition | PUBLISHABLE | MATERIAL_DEVELOPMENTS | 14 | **0** | **0** | 2 | 292 | none |
| B organic | DEGRADED | MATERIAL_DEVELOPMENTS | 13 | **0** | 1 | 2 | 268 | none |
| C risk warning | DEGRADED | ATTENTION_REQUIRED | 15 | **0** | 1 | 1 | 222 | 1 — adjudicated below |
| D quiet | DEGRADED | INCOMPLETE_REVIEW | 10 | **0** | 1 | 3 | 315 | none |
| E mixed | **BLOCKED** | — | 9 | **0** | 3 | 0 → **brief delivered** | 0 | none |

- **Scope holds: 0 out-of-mandate calls in all five.**
- **A_acquisition is now clean** — 0 rejected figures, publishable first time.
- **E_mixed still blocks, and that is now safe.** §15's critical requirement is
  met: the deterministic briefing is produced and delivered, and the reader is
  not silent.
- **2–4 observations**: 1–3 per period, none padded to a target.

### The one prohibited-term hit, adjudicated

`C_risk_warning` used `RREL35`:

> "This product carries no contractual principal repayment (**RREL35**=OTHR), so
> movement of this magnitude on held loans is unusual…"

`RREL35` is a canonical amortisation-type field returned by
`portfolio_capabilities`, an allow-listed MI tool. The model used it to explain
why interest roll-up drives balance on an equity-release book — legitimate MI
reasoning about a governed field, not regulatory analysis. **This is a false
positive of the §17 regex, not a scope leak.** No readiness tool was called and
no Annex, submission, criterion or rulebook is mentioned.

### The quiet-period verdict — resolved, not forced (§5)

`D_quiet` still returns `INCOMPLETE_REVIEW`, and the re-run shows the reason has
changed. The model is no longer confusing an absent capability with a failed
check; it is judging that being unable to test high-LTV exposure against
approved thresholds is itself material:

> "Complete portfolio stasis with zero movement across all metrics, **but**
> covenant compliance could not be assessed due to absence of approved
> risk-limit configuration."

That reading is defensible, so it has not been argued away. **It also does not
matter commercially, because the verdict never reaches Teams**: `enrichment.attach`
takes the card's findings and nothing else.
`test_the_period_verdict_never_reaches_the_teams_card` asserts this across all
four enum values, so a future change that starts rendering it fails loudly.

Per §5's own guidance, this is the "cleaner to leave it off the card" outcome —
it already is off the card, and the uninformative enum is not holding up
deployment. What the reader gets on a still month is §3's factual baseline plus
*"No material developments were identified in the funded book this period."*

### Brevity under the revised rule (§2)

211–315 words. The hard 250-word failure is **gone**: `brief.SOFT_CARD_WORDS`
is a guide that produces a note, never a failure, and `brief.quality_flags`
now tests what actually makes a briefing bad — repetition, methodology
exposition, raw tool output, duplicate findings, hedging stacked until the claim
disappears. `test_word_count_alone_is_never_a_quality_flag` pins the instruction.

---

## E. Teams delivery proof (§16, §18)

### State this plainly: no real Teams message was delivered

**What was exercised:** the entire production path, end to end, with exactly two
seams replaced — MI resolution (supplied inputs) and the outbound HTTPS POST to
the Bot Framework.

```
approve_publication hook  →  trigger.on_publication_approved   [real]
    → sources / generate.build                                  [real]
    → enrichment.enrich                                         [real]
    → contract validate                                         [real]
    → dedupe / correction / supersession                        [real]
    → RecipientStore.select (authorisation, MS-tenant match)    [real]
    → BatchStore.save + record_reporting                        [real]
    → Outbox.enqueue                                            [real]
    → DeliveryWorker.run                                        [real]
    → cards.attachment / summary_text                           [real]
    → TeamsClient.send_card                            [RECORDED, not sent]
    → Outbox.mark_sent + teams_message_id + telemetry            [real]
```

`RecordingClient` **subclasses the production `TeamsClient`** and overrides only
`send_card`. Everything that builds the request — the card, the summary, the
conversation reference, the service URL — is production code and is captured.

**Why no live send.** This environment holds no Bot Framework app credentials
and no authorised Teams destination. §17 forbids sending portfolio information
to an unauthorised destination, and inventing one would breach that. **The wire
itself is the one link not proven, and it is a stated condition on the verdict
below**, not something glossed over.

What *is* proven about the transport: `TeamsClient` enforces a service-URL
allowlist (`_is_allowed_service_url`), classifies retryable status codes, and
the worker records `teams_message_id` and marks state. The recorded call carries
`service_url=https://smba.trafficmanager.net/emea/` and
`conversation_id=a:1conversation` — a real Bot Framework shape.

### §18 delivery tests

| Requirement | Test | Result |
|---|---|---|
| Correct recipient routing | `test_client_bs_authorised_user_never_receives_client_as_briefing` | **pass** — both lenders have real authorised recipients; A's run is approved; B receives nothing and B's outbox stays empty |
| Proactive initiation | `test_the_bot_sends_without_the_user_having_messaged_it` | **pass** — no inbound activity posted; the captured conversation reference is the whole basis for addressing |
| Deterministic-only fallback delivery | `test_a_blocked_review_still_delivers_the_deterministic_briefing` | **pass** |
| LLM/runtime failure fallback | `test_any_autonomous_runtime_failure_still_delivers_the_briefing` ×4 | **pass** |
| Publishable autonomous delivery | `test_publishable_enrichment_adds_observations_to_the_deterministic_card` | **pass** |
| Duplicate protection | `test_the_same_reporting_period_is_not_delivered_twice` · `test_a_second_delivery_pass_does_not_resend` | **pass** — suppressed at the trigger *and* idempotent in the worker |
| Disabled configuration | `test_with_delivery_disabled_nothing_is_generated_or_sent` | **pass** |
| Auditability | `test_an_operator_can_reconstruct_what_happened` | **pass** — message, period, client, recipient, enrichment outcome, delivery result |

22 tests in `tests/notifications/test_enrichment_and_delivery.py`, all passing.

---

## F. Shadow-mode proof (§19)

**Already production behaviour; now tested.** `trigger._run` saves the batch and
records the reporting position *before* it checks recipient eligibility:

> "The batch is stored and the reporting position recorded even with no
> recipient: the generated content is the audit record of what WOULD have been
> said." — `trigger.py`

`test_shadow_mode_stores_the_briefing_without_sending_it` runs an approval with
**no recipient fixture at all**, and asserts:

- `outcome.recipients == 0`, `sent_to_outbox is False`,
  suppressed `SUPPRESS_NO_RECIPIENTS`;
- nothing reaches the client;
- **the complete briefing, enrichment included, is on disk** and readable —
  the autonomous observation `"93.4% of the movement"` is in the stored message.

`test_a_shadow_period_is_not_re_notified_when_a_recipient_appears` covers the
follow-on risk: authorising a recipient after a shadow cycle must not deliver
yesterday's batch as though it were new. It is suppressed as a duplicate.

**Shadow mode is `enabled: true` + `recipients: []`.** With `enabled: false`
nothing is generated at all (suppressed at step one), which is the correct
current state and is separately tested.

---

## G. Regression (§21)

The full suite completed on both sides, **twice each**, because the first HEAD
run showed one new failure and one observation is not enough to classify it.

| | `origin/main` run 1 | `origin/main` run 2 | HEAD run 1 | HEAD run 2 |
|---|---|---|---|---|
| passed | 7,336 | 7,336 | **7,566** | **7,568** |
| failed | 171 | 171 | **172** | **172** |
| errors | 21 | 21 | 21 | 21 |
| runtime | 46m 55s | 51m 04s | 46m 51s | 50m 26s |

The two baseline runs are **identical test for test**. The two HEAD runs are
**identical test for test**. Diffing HEAD against baseline:

```
NEW on HEAD  : tests/test_occ_day1_hardening.py::TestRestartAfterInterruption
               ::test_the_operator_can_restart_and_gets_the_same_output
FIXED on HEAD: (none)
```

### The one new failure, investigated rather than dismissed

**It is not a flake in the ordinary sense** — it appeared in both HEAD runs and
in neither baseline run. It is also not a behavioural regression. The evidence:

| Experiment | Result |
|---|---|
| Full suite, HEAD | fails 2/2 |
| Full suite, `origin/main` | passes 2/2 |
| The test alone, HEAD | passes 6/6 (under CPU contention) |
| The test alone, `origin/main`, under heavy load | passes 5/5 |
| The preceding new suite + the test, HEAD, idle machine | passes 8/8 |
| The preceding new suite + the test, HEAD, **machine loaded** | **fails** — and the test aborted in 5s rather than its usual 36s |

**Mechanism.** The test starts a workflow on a background executor, deletes the
staging directory, then builds a fresh `OpsEngine` and calls
`recover_on_startup()`. The first assertion — that the workflow was recovered —
**passes**; the next one, that its stored status is `blocked`, fails with
`running`. So recovery ran, marked the run interrupted and saved it, and the
**original executor thread — which the test never stops — then wrote the run
back to `running`**. It is a race between recovery and a live executor, and it
exists on `origin/main` too.

**Why this branch makes it fire.** Nothing in the diff touches OCC, the workflow
engine, or the approval path's behaviour:

- no file under `operations_control/` is modified;
- `trigger.on_publication_approved` gained one optional keyword (`reviewer`)
  and OCC's call site does not pass it, so enrichment there is `not_attempted`
  — a no-op that adds one dict to `provenance`;
- `notification_batch_id` and `reporting_key` hash tenant, portfolio, context,
  update type, run ids and dates only — **provenance is excluded**, so
  enrichment cannot alter batch identity or the dedupe promise.

What the branch does change is the *shape of the run*: +232 tests and roughly
four extra minutes of accumulated process state before that test is reached.
That is enough to tip a load-sensitive race that was already there.

**Classification: a pre-existing test-harness race, made more likely by a longer
suite. Not a production defect** — a real OCC restart is a new process, and the
old executor is not still alive inside it to overwrite the recovery.

**It has deliberately not been "fixed".** Editing a test this branch did not
break, in a module this sprint is forbidden to touch, in order to make a
regression table look clean, is the wrong move. It is recorded in §H instead.

### Per-area check

| Area | Baseline | HEAD | |
|---|---|---|---|
| readiness agent | 3 | 3 | unchanged |
| notifications | 0 | 0 | unchanged |
| portfolio review | 0 | 0 | unchanged |
| concentration / risk | 2 | 2 | unchanged |
| simulation | 22 | 22 | unchanged |
| movement / receipt | 14 | 14 | unchanged |
| operations control | 6 | **7** | +1, the race above |

Deterministic analytics unchanged (§C: identical output, word for word).
Portfolio Review scope unchanged (22 allowed, 10 excluded).
Readiness agent unchanged (full 32-tool surface; `tool_schemas` defaults to
`None`). No client-isolation regression — routing is tested in both directions.

## H. Remaining issues

| | Issue | Class |
|---|---|---|
| **1** | **No live Teams send has been performed.** Everything up to and including the constructed Bot Framework call is exercised; the HTTPS POST itself is recorded. Requires app credentials and one authorised test destination. | **COMMERCIAL GO-LIVE BLOCKER** |
| **2** | Board vs operational permissions are not separately modelled. `portfolio_contexts` is a portfolio scope, not a role, so both user types authorised for `total` receive the same card. Client/tenant routing *is* secure and tested. | **POST-GO-LIVE** — per §20, does not block an operational-user launch |
| **3** | The agent derives figures in most runs (4 of 5 this sprint). The gate rejects every one, so nothing unsupported is published, but a blocked period yields no enrichment. | **OPERATIONAL MONITOR** — watch `enrichment.status` rates |
| **4** | `E_mixed` blocks repeatably. The deterministic brief carries the period, so the reader is served, but the most complex period gets no enrichment. | **OPERATIONAL MONITOR** |
| **5** | The internal `period_verdict` is uninformative in a deployment with no approved limits. Not Teams-facing; asserted so. | **POST-GO-LIVE** |
| **6** | Card length 178–315 words. Explicitly **not** a blocker per §23. | **POST-GO-LIVE** |
| **7** | Pre-existing full-suite failures unrelated to this work (mail, simulation, conversion suites). Identical on `origin/main`, 171 on each side. | **POST-GO-LIVE** |
| **8** | `test_occ_day1_hardening::test_the_operator_can_restart_and_gets_the_same_output` fails on HEAD's full suite (2/2) and not on baseline (2/2). Root-caused in §G to a pre-existing race between `recover_on_startup` and a live executor thread the test never stops, tipped by a longer suite. Test-harness only; a real restart is a new process. Left unfixed rather than edited to look clean. | **OPERATIONAL MONITOR** |

---

## I. Final verdict

### Deterministic Teams Brief

# CONDITIONAL GO

Every §23 criterion is met except one, and the exception is narrow and precisely
stated:

| §23 criterion | Status |
|---|---|
| deterministic 10-period suite remains correct | **met** — 10/10, unchanged |
| autonomous failure cannot suppress it | **met** — 5 failure modes tested |
| Teams mechanism proven to deliver proactively | **met to the transport boundary; the live send is not proven** |
| duplicate protection works | **met** — trigger and worker |
| client routing works | **met** — B never receives A |
| shadow mode works | **met** |
| no regression | **met with one stated exception** — zero behavioural regressions; one pre-existing test-harness race now fires (§G, §H8) |

**The exact remaining condition: perform one live proactive send to a designated
developer or test Teams recipient, and confirm the delivery receipt.** Nothing
else is outstanding. That test needs credentials and an authorised destination
this environment does not have; it is a half-day of operational work, not
engineering.

### Autonomous Portfolio Review Enrichment

# GO — as an enrichment layer

Against §23's bar for the enrichment layer:

| Requirement | Status |
|---|---|
| scope remains hard bounded | **met** — 0 out-of-mandate calls in 5 runs; refused before execution |
| numeric gate remains enforced | **met** — unrelaxed; 6 figures rejected this sprint |
| 2–4 useful insights where available | **met** — 1–3 per period, unpadded |
| quiet periods produce useful factual conclusions | **met** — via the deterministic core; §4 qualification intact |
| complex blocked periods fall back safely | **met** — the sprint's headline result |
| no readiness/regulatory leakage | **met** — one hit adjudicated as a governed field name |
| no unsupported number reaches Teams | **met** — by construction |

This is GO **specifically as an enrichment layer over a guaranteed deterministic
briefing**, and only in that shape. It is not approved to be the message. The
distinction is the whole design: the agent may improve the briefing, and it can
no longer prevent one.

Its GO is independent of the deterministic brief's remaining condition — the
live send blocks *delivery of anything*, enrichment included.

---

## Stop conditions (§22)

None triggered. No change to pipeline gates, ingestion, canonicalisation,
transformation, provenance schema, the snapshot model, or pipeline/funded
population definitions. Operations-control approval semantics are untouched: the
hook `trigger.on_publication_approved` gained one optional keyword argument
(`reviewer`) and OCC does not pass it, so approval behaviour is byte-identical.

The deterministic analytics freeze (§11) held — the 10-period output is
identical, word for word, to the pre-sprint run. The autonomous scope freeze
(§12) held — 22 allowed tools, 10 excluded, unchanged. The numeric gate (§13)
was not relaxed.

---

## Reproducing

```bash
python scripts/show_teams_message_contract.py                 # the contract, per period
python scripts/run_deterministic_period_suite.py              # §14
python -m pytest tests/notifications/test_enrichment_and_delivery.py  # §7, §18, §19
python -m pytest tests/test_deterministic_period_suite.py tests/test_portfolio_review_mandate.py

export ANTHROPIC_API_KEY=...                                  # §15, ~$1/run
python scripts/run_portfolio_review_redteam.py --set agent --runs 1 \
    --out /tmp/runs.json --data-root /tmp/rt_data
python scripts/show_teams_message_contract.py --enrich /tmp/runs.json
```
