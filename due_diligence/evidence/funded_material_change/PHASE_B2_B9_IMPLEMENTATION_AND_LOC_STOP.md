# Sprint B, Phases B2–B9 — built and verified; STOPPED at the LOC gate

```
PRODUCT_BASELINE      dcb71a20 (Sprint A) / 02e0861c (Phase B1)
LIVE_MODEL_CALLS      0
LIVE_MI_QUERY_CALLS   0
DEPLOYMENTS           0
NEW_COMPOSITION_FRAMEWORKS      0
NEW_ANALYTICAL_CALCULATION_OWNERS 0
```

## The stop

The sprint brief said: *"If Product LOC materially exceeds ~400–500 lines … stop
and explain."* It does.

```
  546  mi_agent_api/insight_funded.py          NEW
  116  mi_agent/plan_material_summary.py       NEW
   25  mi_agent_api/insight_config.py          delta
   17  mi_agent_api/insight_contract.py        delta
    7  mi_agent/interpretation_v2/compiler.py  delta
    6  mi_agent/interpretation_v2/vocabulary.py delta
  ---
  717  TOTAL PRODUCT LOC   (executable only: comments, docstrings and blank
                            lines excluded; config/mi/insights.yaml excluded)
```

**717 against a ~400–500 gate is +43%. That is material and I am stopping on it
rather than finishing B10 and B11.** B2–B9 are complete and verified; B10 (the
16 deterministic fixtures, the composition receipt gate) and B11 (the Weekly
Brief and MI regression banks) are NOT started.

### Where the estimate went wrong

Phase B1 estimated `insight_funded.py` at ~190 lines. It is 546. The error was
not in the design — no calculation owner, no second engine and no second
threshold hierarchy appeared, exactly as B1 said. The error was in counting the
**declination surface**. Each generator must return an `Omission` for every way
its input can be absent, because "a brief that silently dropped a section would
be indistinguishable from a book with nothing to report" is the existing
contract's stated rule. The funded generators have more such ways than the
pipeline ones:

| generator | declination cases |
|---|---|
| limit status transitions | no configuration; no test in it; no prior date; no transition; improvements switched off |
| measure movements | no analysis; no eligible measure; not comparable; below threshold |
| composition shifts | no analysis; no eligible dimension; below threshold |
| balance attribution | no analysis; no bridge; bridge unavailable; does not reconcile; movement immaterial |
| quiet period | switched off; something qualified; no analysis; nothing comparable |

Measured against the file it is the counterpart of, the size is proportionate
rather than bloated:

```
insight_generators.py (existing, 9 generators)  227 statements, 81 branch points
insight_funded.py     (new,      5 generators)  256 statements, 82 branch points
```

Five funded generators cost what nine pipeline ones do, because each carries
roughly twice the declination surface. I do not think there is an honest ~200
lines to cut: every reduction I could find removes a case that then fails
silently, or removes a phase that was in scope (B8's quiet period is ~45 lines;
B9's adapter is 116).

## What was built

### B2 — the funded input contract
No new pair resolver, as B1 found. `insight_funded.compose()` takes a
`PeriodChangeResult` — already pair-resolved by
`period_change.periods.resolve_periods` — and, optionally, the
`evaluate_active_tests` envelope. `compose()` is **pure**: no I/O, no snapshot
resolution, no owner call. That is what makes the whole composition reproducible
from a fixture and keeps pair resolution entirely outside the file.

### B3 — the contract, extended not replaced
Six types added to the existing `Insight` contract
(`FUNDED_BALANCE_MOVEMENT`, `FUNDED_METRIC_MOVEMENT`,
`FUNDED_COMPOSITION_SHIFT`, `FUNDED_BALANCE_ATTRIBUTION`,
`LIMIT_STATUS_TRANSITION`, `FUNDED_QUIET_PERIOD`) with their `TYPE_PRIORITY`
entries and per-type caps. Same `Insight`, same `Omission`, same
`rank_key`/`select`, same `build_brief`, same `insight_config`.
`LIMIT_STATUS_TRANSITION` deliberately shares tier 100 with
`CONCENTRATION_PROXIMITY`: both are the contractual exposure, and `rank_key`
already breaks the tie on type name and discriminator, so the order stays total.

### B4 — status transitions, with the Phase 1 correction applied
`evaluate_active_tests` takes the pair and establishes `statusTransition`
itself, so **there is no diffing layer**. The generator normalises its rows.
No size gate is applied and none would be right — the configured test already
applied the threshold, and the status changing *is* the finding.

One case is separated on purpose: a transition into or out of a non-risk status
(`unavailable`, `insufficient_data`, `expired`) is a change in whether the test
could be **evaluated**, not in exposure. Reporting "recovered" for a test that
simply stopped being measurable would be the worst available reading, so it is
reported at `info` and says so in terms.

### B5–B7 — measures, composition, attribution
All three normalise `PeriodChangeResult` into `Insight`. The gate for a measure
is chosen by its **unit**, not its name: a `percentage_point` measure is gated on
points, everything else on relative change — because 2% of a 62% weighted-average
LTV is 1.2 points and reads as a far smaller change than it is.

Attribution has **no size gate of its own**. A bridge explains a movement, so it
is emitted only where the movement it explains qualified: an immaterial
movement's decomposition is immaterial by construction. That also avoided
inventing a gross-flow ratio, which would have been new arithmetic.

Composition takes **which** categories to report from the workflow's own
`largest_increases` / `largest_decreases`, so the brief and the distribution
output can never name different leaders.

### B8 — the quiet period
One explicit statement when nothing crossed a threshold, naming what was
examined and against which thresholds. Only ever emitted when the analysis
genuinely ran — a period with no governed result produces an unavailability
omission, never a reassuring sentence.

### B9 — plan connectivity
`CHANGE_FORM_CAPABILITY["material_summary"]` was `None`; it is now
`"period_movement"` — **the same owner `metric_delta` already used**. The two
forms are separated by the governed MODE (`requested_metric` vs
`portfolio_overview`), which is a first-class concept of `period_change`, so
connecting the form needed no new capability and no new owner. The mode is
recorded in a new `CHANGE_FORM_MODE` table.

`mi_agent/plan_material_summary.py` is the adapter: `claims`,
`check_eligibility`, `period_request`, `receipt`, `execute`. Like slice 2 before
it, **it is not wired into `plan_serving_canary` or `mi_service`** — offline
capability first, serving decision later with evidence in hand. That is what
keeps `DEPLOYMENTS = 0` honest rather than nominal.

The plan now carries the form on **both sides of the provenance split**:
`intent_claims["change_form"]` is what the model said, and
`compiler_bindings["change_form"]` is the compiler's derived owner and mode.
The runtime dispatches on the **binding**, never the claim, so no serving
decision is taken from the model's raw reading.

## No arithmetic that produces a published number

Every figure comes from one of two owners. The only arithmetic in
`insight_funded.py` is `abs()` plus a comparison against a configured threshold,
and the ×100 that converts a governed fraction into the points a threshold is
expressed in — and that conversion lives **inside a formatter**, never in the
published block.

The published blocks are **projections of the owners' own `to_dict()`**, via one
`_split()` helper. There is no key this module could rename, rescale or drop,
and a field the workflow adds later arrives with no change here. The threshold
comparison for a fraction divides the threshold rather than multiplying the
governed value, for the same reason.

Where the thresholds came from: `period_change` states in its own output that it
has no materiality rule — *"No governed materiality threshold is configured for
this portfolio … No movement is described as material, significant, a breach or
high risk."* That sentence is correct about the workflow and is the exact gap
this sprint closes. The funded sections of `config/mi/insights.yaml` **are** that
configured rule. Every threshold is relative; **no absolute currency floor was
added**, as instructed. Every insight carries a `methodology.materiality` block
naming the section, the key, the value and the config source, so "why did this
appear" is answerable from the insight alone.

## Verification

All offline. No model call, no MI query, no deployment.

**Composition, end to end, from a hand-built `PeriodChangeResult`** (real
`MetricChange` / `DistributionChange` / `BalanceBridge` objects, not dicts, so
the fixture is bound to the real contract and cannot drift from it):

```
status: success | findings: 6
  [ 80] concern   LIMIT_STATUS_TRANSITION    Top-10 obligor concentration deteriorated: warning -> breach
  [ 70] attention FUNDED_METRIC_MOVEMENT     Weighted-average LTV increased +2.4pp
  [ 60] info      LIMIT_STATUS_TRANSITION    Single-region exposure recovered: warning -> pass
  [ 50] info      FUNDED_BALANCE_MOVEMENT    Current outstanding balance increased +£5.0m
  [ 40] info      FUNDED_BALANCE_ATTRIBUTION Balance movement decomposes to £100.0m → £105.0m
  [ 30] info      FUNDED_COMPOSITION_SHIFT   Product type: Bridging +9.5pp
omitted:
  immaterial   FUNDED_METRIC_MOVEMENT   1 measure moved by less than the configured threshold: Loan count
```

The same result with thresholds raised out of reach produces exactly one
finding — the quiet-period statement — plus four omissions explaining each
silence, including *"The balance movement did not cross its configured
threshold, so its decomposition is not reported."*

**Suites** (`pytest`):

| suite | result |
|---|---|
| `tests/interpretation_v2/` | 364 passed, 3 failed — all three failing identically at baseline (below) |
| `test_change_form_contract.py` + `test_change_form_semantic_bank.py` | **50 passed** (44 in Sprint A; 6 controls added) |
| `test_insight_engine.py`, `test_weekly_brief_api.py`, `test_weekly_brief_utilisation_matches_risk_limits.py` | **90 passed** — the Weekly Brief is unaffected by the contract extension |
| `test_period_change*` (mi_agent + mi_agent_api) | **308 passed** |
| concentration / risk-limit | **22 passed** |

### The Sprint A tests B9 deliberately invalidated

Eleven controls asserted that `material_summary` refuses. B9's whole purpose is
to connect it, so they could not stand — but the invariant behind them had to.
They were **re-pinned to the table rather than deleted**: an `unconnected`
fixture `monkeypatch.setitem`s whichever form it likes to `None` and asserts the
production guard, reading its production table, still refuses / states what was
understood / offers no rephrase. A future unconnected form inherits them
unchanged. A new control records that no form is unconnected today, so a
regression that disconnects one is visible.

### One guard widened, with the rationale written into it

`test_the_interpreter_policy_did_not_move` failed on *"orientation key
'capability_operations' was reworded"* — correctly: adding `summary` to
`period_movement` changes what the model is shown. It is load-bearing, not
cosmetic: `movement` and `compare` both require a named measure, and a named
measure is a metric delta, so without `summary` the form is **unreachable**. The
guard now permits exactly `{"period_movement": {"summary"}}` with that reasoning
written out; a second added operation, a removed one, or any other reworded key
still fails. This follows the file's own established pattern for the three
authorised prompt paragraphs and the one authorised reason code.

### Three failures that are not mine

Verified in a clean worktree at `02e0861c`: all three fail identically there.

| guard | baseline | now |
|---|---|---|
| `test_nothing_outside_the_new_boundary_imports_it` | fails | fails, **list unchanged** |
| `test_only_contract_modules_moved` | fails, 14 files | fails, 16 — adds `insight_config.py`, `insight_contract.py` |
| `test_production_surfaces_are_untouched` | fails, 7 files | fails, 10 — adds those two and `insight_funded.py` |

I did **not** widen these two. They are already red for unrelated prior reasons,
and widening a broken guard teaches nothing. Recording it instead: Sprint B
touches three `mi_agent_api` insight files, which both guards list.

## Findings that need a decision

**1. A bare "what changed?" read as `operation = movement` clarifies instead of
composing.** Measured:

```
summary   no measure: PLAN     capability=period_movement
movement  no measure: CLARIFY  MISSING_REQUIRED_SLOT
compare   no measure: CLARIFY  MISSING_REQUIRED_SLOT
```

Only `summary` reaches the composition without a named measure. `movement` and
`compare` state a shape that is *about* a measure, so with none named the
compiler asks which — it does not widen to every governed measure because the
form was broad. That is defensible, and it is also the likeliest reading of the
plainest phrasing of the question. Making the measure optional for this form is
a compiler change with its own evidence; **I did not slip it in**. It is pinned
as a recorded perimeter in
`test_the_broad_form_alone_does_not_excuse_a_measure_bearing_operation`.

**2. The concentration perimeter B1 flagged still stands.** `evaluate_active_tests`
needs an `ActiveConfiguration` and a `ConcentrationLibrary`. The 135 bank
measured that this book answers borrowing-base questions with *"No funding
facility is configured for this portfolio"*. Status transitions will therefore be
provable on fixtures and will report unavailability — in terms — on a book
without that configuration. A configuration fact, not a capability gap, and not
worked around.

## What remains

- **B10** — 16 deterministic fixtures and the composition receipt gate. Not started.
- **B11** — Weekly Brief and MI regression banks. The Weekly Brief suite already
  passes (90/90); the bank runs are not started.
- Wiring `plan_material_summary` to a request. Deliberately not done.

## The decision I need

1. **Accept the 717 and continue to B10–B11.** My recommendation. The overrun is
   declination surface, not scope creep: no new engine, no new calculation owner,
   no new threshold hierarchy, and the file is proportionate to the one it is the
   counterpart of.
2. **Cut scope to fit the gate.** The only honest cuts are B8 (~45 lines) and B9
   (~116), both of which were in the brief. That gets to ~556, still over.
3. **Revert and re-plan.** Everything is one commit and reverts cleanly.

Nothing further runs until you say which.
