# Semantic coverage ledger — final build result

Start `a0a8f32`. End: this commit.

| commit | stage |
|---|---|
| `21f1938` | Stage 1 — the ledger, disclose only |
| `6ac01c4` | Stage 2 — route disclosure + two central equivalences |
| `da1c27d` | Stage 2 completion — derived forecast reconciliation, bridge lens |
| `1488db5` | Stage 3 — fail-closed enforcement |
| `1af0bda` | **Stage 3 REVERTED** — the frozen manifest found five false refusals |
| this | report |

**Production files:** `question_interpretation/completeness.py` (+157),
`mi_agent_api/mi_service.py` (+40), `mi_agent_api/chat_routing.py` (+55).
No new module, no ontology, no prompt change, no additional model call.

---

## The premise changed under measurement, and it matters

This build was justified by Q16B omitting `drawdown` in 1 run of 10. **That
measurement was contaminated and I over-attributed it.**

The harness recorded the model's proposals but not whether the call reached the
model. Re-run with that asserted, under healthy API credit:

| measurement | calls reaching the model | model omissions |
|---|---|---|
| Q16B ×30 | **30 / 30** | **0** |
| seven recoveries ×6 | 42 / 42 | 0 |
| all controls ×6 | **180 / 180** | 0 |

**72 healthy runs, zero model omissions.** The original "1 in 10" was almost
certainly a `proposal_unavailable` — an API failure recorded as an empty
proposal. I cannot prove that retrospectively (the old capture has no status
field), and I am not claiming certainty; I am saying the evidence now favours
it strongly, and that my earlier attribution to model stochasticity was wrong.

**The hole is still real — and the corrected diagnosis makes it worse, not
better.** `proposal_unavailable` is not a stochastic event; it is a *systematic*
failure mode tied to availability. Whenever credits lapse, a rate limit bites or
the API is unreachable, the arm degrades silently and the answer silently
widens. Measured directly during the credit outage in this very build: **20 of
20 runs of Q16B returned a whole-book answer, and the ledger flagged
`erm_product_type` on every one.**

So the invariant's justification is stronger than when it was commissioned:
not *"the model sometimes forgets"* but *"the model is not always reachable, and
unreachability currently changes the answer instead of the answerability"*.

## Stages 1–2 — shipped and clean

**Authoritative owner:** `completeness.coverage_report`. Concept-founded, never
span-founded. Three dispositions — `RESOLVED` / `UNSUPPORTED` / `UNACCOUNTED` —
and no `NON_SEMANTIC`, because ordinary words never enter a ledger built from
what governed owners *name*.

**Independence audited before use** (1A): no spec, no contract, no envelope; the
internal facet call deliberately omits `resolved_filters`; no phrase bank;
surface grows from the registry and value catalogue.

**Disclosure closed** through one shared primitive and four central
equivalences, never per-route special cases:

| correction | kind |
|---|---|
| `portfolio_summary`, `evolution`, `funded_bridge` declare the lens they applied | D2, via the existing `_declare_scope` |
| `forecast_extrapolation` derives its reconciliation instead of asserting `"forecast"` | D2, sixth adopter of `reconciliation_for` |
| `scope_applied` satisfies `scoped` | D3 — it is the strongest scope signal, not the weakest |
| the band relation, read inversely | D3 — registry `derived_from` |
| a composite `funded+pipeline` read carries its parts | D3 — the estate's own join format |
| a forecast is carried by `projected`, not by a tape | D3 — `forecast` is a governed view with nothing to read |

**No D4 false positives were found.** Nothing required suppression.

| measure | result |
|---|---|
| flagged answering questions | **46 → 39 → 29 → 27** |
| correct answers among them (graded) | **0** |
| deterministic 166 | **166/166 byte-identical** |
| frozen 278-module manifest | **85 names, exact** |
| ledger present | 1,612 / 1,612 |

## Stage 3 — built, measured, REVERTED

On the deterministic 166 it did exactly what it was designed to do:

> 6 movements, **every one WRONG → controlled refusal** — Q03A, Q05C, Q07B,
> Q16B, Q17C, Q19A. **Zero correct answers degraded. WRONG 8 → 2**, and the two
> that remain are precisely the two the architecture said coverage cannot see.

**Then the frozen manifest moved 85 → 90.** Five correct answers became
refusals:

```
test_query_applies_drill_through_filters
test_cohort_progression_route_returns_metric_line
test_kfi_trend_by_week_e2e
test_q8_two_populations_move_independently_and_reconcile
test_a_weekly_funnel_question_is_no_longer_told_it_is_monthly
```

Every one is the **same class Stage 2 exists to close** — a route applying a
narrowing without disclosing it — on fixtures the census never reached. The
weekly funnel names its concept outright: it refuses on `KFI`, a pipeline stage
it filters by and never records.

That is the census's limit, not the ledger's. It ran over one book and 1,612
questions; those tests exercise a second book, a weekly pipeline and
drill-through filters. **Reverted; frozen manifest back to exactly 85.**

## Acceptance

| gate | result |
|---|---|
| deterministic 166 byte-identical | **PASS** |
| frozen manifest = 85 | **PASS** |
| ledger on every answer | **PASS** — 1,612/1,612 |
| arm reached the model | **PASS** — 180/180, asserted not assumed |
| five former regressions | **CORRECT 6/6 each** |
| seven CR4 recoveries | **CORRECT 6/6 each** |
| Q10B (Opus arm) | **CORRECT 6/6** |
| Q22B/C, Q10A | answered 6/6 · Q25A/B/C refuse 6/6 |
| must-refuse ×3 | **TRUE_REFUSAL 6/6 each** |
| Q16B whole-book frequency, healthy API | **0 of 30** |
| Q16B whole-book frequency, API unavailable | **20 of 20 — the hole, still open** |
| Q04C / Q19A | WRONG 6/6, outside coverage's scope, unchanged |
| new controlled refusals from genuine omission | 0 — enforcement reverted |
| false coverage refusals | 0 in production; **5 found in test fixtures**, which is why |

## Residual risks and limits

1. **The hole is open.** The ledger sees every omission and blocks none. Under
   API unavailability the answer still widens silently.
2. **The remaining backlog is five named failing tests** — a better artefact
   than the census could produce, because each says precisely which route must
   disclose what. Close those, re-run the manifest, re-apply `1488db5`.
3. 27 flagged answering questions remain; zero are correct answers on the
   graded estate, but 17 are ungraded surface questions with no oracle.
4. Unchanged limits: Q04C is not an omission; Q10B's bare "size" has no owner
   that can name it. Coverage is bounded to *material user semantics the
   governed estate can independently name*, and no broader claim is made.
5. `_carried` is now load-bearing for the ledger and would become load-bearing
   for refusals; it needs adversarial review before enforcement returns.

---

# DO NOT FREEZE

Everything shipped is safe and measured: two stages, byte-identical answers,
frozen manifest exact, and a ledger that sees every omission it was built to
see — including all 20 of the outage-driven ones.

But it does not yet *stop* one, and the corrected diagnosis makes that the
sharper problem. The risk is not a model that occasionally forgets a word; it
is that **an unreachable model currently changes the analytical meaning of an
answer rather than its availability**. That is a production risk with a known
trigger, and it is still live.

The remaining work is bounded, named and mechanical: five route disclosures,
each identified by a failing test. Close them, restore `1488db5`, re-run the
manifest and the repeated-run protocol. The invariant itself is proven — it
converted six wrong answers into controlled refusals on the deterministic arm
without touching a single correct one.
