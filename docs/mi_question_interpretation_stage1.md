# Stage 1 — schema defined, corpus projected, nothing populated in production

| | |
|---|---|
| Base | `4e051f3`; merge-base with `claude/mi-analytical-capability-layer-vlkjfw` is `4e051f3` exactly |
| Ancestry | `4e051f3` ✓ ancestor of HEAD · `28ece25` ✓ ancestor of HEAD |
| Release candidate | `28ece25` — unchanged, still shippable |
| Production code changed | **none** — `git diff 4e051f3..HEAD -- mi_agent mi_agent_api` is empty |
| Reproduce | `python -m question_interpretation.run_stage1_corpus` |

## Corpus — 939 questions across four surfaces

| Surface | n |
|---|---:|
| `ere_mi_questions` — ERE golden library | 350 |
| `ere_mi_calibration_250` — calibration bank | 252 |
| `generated_harness` — registry-driven, machine-generated | 249 |
| `nl_robustness_44` — **44 variations × 2 books** | 88 |

The 44-variation robustness bank exists on this tree at
`due_diligence/evidence/analytical_intent_v1/`, with `nl_bank.py`,
`nl_harness.py` and `nl_score.py`. Its questions are used here as a corpus
surface. **Its recorded 752-run figures are an LLM-arm measurement** —
`nl_harness.py` asserts `ANTHROPIC_API_KEY`, which is **not set in this
environment**. See "Blocker for later stages" below.

---

## What was built

**`question_interpretation/schema.py`** — data only. No parsing, no resolution,
no decisions. Slots are `filled` / `empty` / `unresolvable`; `dimensions[]`
carries a syntactic `role`; `target` is a separate slot with a `stated` /
`configured` source; `time` separates `requested_grain` from `trend_window`.

The execution vocabulary is deliberately absent. `applied` and `lost` carry
execution evidence a pre-execution object cannot have, and a test enforces that
neither they nor `field_key` can appear on the object.

**`question_interpretation/tests/test_schema.py`** — 30 tests.

**`question_interpretation/tests/mutation_check.py`** — standing condition 4,
discharged by evidence. Eight real mutations, each a defect this programme has
actually seen, applied to `schema.py` with the suite re-run against the mutant:

```
8 mutations, 8 caught, 0 undetected, 0 misapplied
```

The first run caught **two of my own mutations as unfaithful** — they did not
reproduce the defect they named, and the suite was right to stay green. Both
were rewritten to mutate the real structure before the run above.

**`question_interpretation/projection.py`** — read-only. Every slot records
which existing interpreter supplied it, so a gap is attributable to a source
rather than to the projection.

---

## What does not fit

### The four surfaces do not agree about how hard this is

| Surface | n | operation unfilled | subject unfilled | role unresolved | grain read |
|---|---:|---:|---:|---:|---:|
| `generated_harness` | 249 | **0** | **0** | **0** | 0 |
| `ere_mi_calibration_250` | 252 | 0 | 1 | 9 | 10 |
| `ere_mi_questions` | 350 | 5 | 28 | 32 | 34 |
| `nl_robustness_44` | 88 | 0 | 12 | 14 | 20 |

**The generated harness has zero gaps on every axis**, because its phrasings are
machine-generated from registry names — a rule set reading registry names back
to itself. It is a valid invariant check and it is worthless as evidence about
client wording. Every finding below comes from the other three.

### Gaps, by count

| What no existing interpreter supplies | Questions |
|---|---:|
| filter/dimension — **field known, wording not** | 119 |
| filter — **clause identified, field not** | 103 |
| time grain — **read correctly, not carried** | 64 |
| dimension role — **named, but no role available** | 55 |
| operation — **two interpreters disagree** | 46 |
| subject — no interpreter supplies it | 41 |
| operation — no interpreter supplies it | 5 |

### 1. The filter clause is seen twice and joined nowhere — 101 questions

The two readings are complementary and the projection cannot join them:

* the facet layer supplies the **clause and its wording** — every `threshold`
  facet carries `field_key=None`
* the parser supplies the **field, operator and value** — and no raw text

**101 questions carry both halves of the same clause with nothing linking
them.** For *how many loans have LTV above 50%* the object holds
`FilterClaim(raw_text="LTV over 50", source=facet.threshold)` and
`FilterClaim(operator="gt", value="50.0", source=parser.filters[current_loan_to_value])`
as two separate claims about one clause.

Neither interpreter carries offsets, so they cannot be joined positionally, and
neither carries the other's key, so they cannot be joined by identity. **This is
the largest single thing Stage 2 must resolve**, and it is a genuine gap rather
than a schema defect: the schema has one `FilterClaim` per clause; the sources
supply two halves.

### 2. 215 claims have no recoverable span

The schema asks for offsets so that precedence between competing claims is
decidable. The projection recovers a span only where a source's label happens to
be a literal substring of the question. **215 dimension and filter claims across
the corpus have no recoverable span**, because the source supplied a field key
or a rendered label rather than the words.

Consequence, stated plainly: **span-based precedence is not available from the
existing interpreters.** Either the sources start reporting offsets in Stage 2,
or the schema's span field is aspirational for a large minority of claims.

### 3. Dimension role — 55 named with no role available

The role split works where the parser assigns a slot: 663 `grouping`, 15
`filter`. It fails where the parser assigns **neither** — the facet layer names
the dimension, the parser drops it, and no source has an opinion:

> *pipeline by broker* · *show NNEG by region* · *missing region count* ·
> *show best brokers*

These are routed elsewhere, and the route does not report a role.

The 15 `filter` roles are strikingly few against 663 `grouping`, and the reason
is mechanical: a dimension is classed `filter` only when its key appears in
`spec.filters`. Categorical filters that the parser turns into a grouping — the
*account status is offer* class — are recorded as `grouping`, which is
**exactly the KIND_GROUPING conflation the role slot exists to fix, reproduced
through a different source.** The slot is right; the population of it from
today's interpreters cannot fix what the interpreters get wrong.

### 4. Operation — 46 disagreements between two live interpreters

`answer_type.asked` and the parser's aggregation disagree on 46 questions:

| parser → answer_type | n |
|---|---:|
| count → amount | 25 |
| average → amount | 8 |
| count → average | 7 |
| amount → average | 4 |
| amount → count · average → count | 1 each |

Both are shipping today, on the same sentence. *average loan balance* and *what
is the mean loan balance* are in this list.

### 5. Time grain — read correctly, carried nowhere: 64

`period_request.requested_unit` supplies a grain for 64 questions — month 43,
year 10, week 8, quarter 3. On **every one of them** `spec.trend_grain` is
`None`. The reading is correct and the carriage does not exist, exactly as the
inventory found. `trend_window` is filled on only 20.

### 6. Slots nothing supplies

* **`coverage`** — in the contract's draft operation vocabulary; **produced 0
  times in 939 questions**. No interpreter supplies it.
* **`target`** — filled on **12 of 939**, all from
  `parser.forecast_target_value`. The `configured` sense (*on target*, *versus
  plan*) has no source at all; the projection can only mark it unresolvable
  from its own regex, which is a projection artefact and not a reading.
* **`population`** — populated only from `row_population` / `cohort_comparison`
  facets, which are rare. The seasoning families that Stage 4 must leave
  unmoved are not visible here.
* **`subject`** — unfilled on 41, concentrated in pipeline and forecast
  phrasings (*run-rate forecast of funded balance*, *what is the pipeline
  conversion rate?*), where the parser sets `metric=None`.

---

## Schema corrections the corpus demands

Listed, not applied. The contract says to correct the schema from this report
before Stage 2, and that decision is yours.

1. **`FilterClaim` needs to express a half-claim, or the two halves need
   joining.** 101 questions produce two claims for one clause. Either the
   schema admits `clause` and `binding` as separate claims with an explicit
   link, or Stage 2 must make one interpreter supply both — which is a bigger
   change than Stage 2 is scoped for.
2. **`Span` is optional in practice and the schema should say so.** 215 claims
   cannot supply one. Leaving it nominally required invites a consumer to rely
   on it.
3. **`coverage` should be removed or its source named.** Nothing supplies it.
   Keeping an unsupplied member invites someone to populate it by intuition —
   the failure mode the contract warns about.
4. **`target.configured` needs a source or should be deferred.** Today only the
   projection's own regex produces it, which is not a reading.
5. **The `unresolved` dimension role needs a distinction.** "no source has an
   opinion" (55 cases, routed elsewhere) and "the sources disagree" are
   different states, and one value covers both.

No correction has been applied. `question_interpretation/stage1_corpus.json`
carries the per-question record for all 939.

---

## Blocker for later stages — standing condition 1

Standing condition 1 requires **both surfaces at every stage**. Stage 1 is
read-only so nothing is at risk, but from Stage 2:

* the calibration bank runs here — **260 passed, 0 skipped**, against the real
  alderbridge tape;
* the 44-variation bank's **deterministic** arm can be run — its 88 questions
  are data, and this Stage 1 run used them;
* the 44-variation bank's **LLM arm cannot be run** — `nl_harness.py` asserts
  `ANTHROPIC_API_KEY` and it is not set. The 752-run figures the revert commit
  records (91.0% correct/disclosed, 160 transitions) are LLM-arm measurements
  and **cannot be reproduced or re-measured in this environment.**

The 160-run regression that motivates standing condition 1 was an LLM-arm
result. **Stage 3 and Stage 4 cannot honour standing condition 1 as written
without an API key.** This needs settling before Stage 2 completes, not after.

---

## Stage 1 acceptance

* Schema authored as data with no behaviour ✓
* Tests authored, and **proven able to fail** — 8 of 8 mutations caught ✓
* Corpus projected read-only through the existing interpreters, 939 questions
  across four surfaces ✓
* Every unfillable and inconsistent slot reported ✓
* **No production code changed** ✓

Stage 2 is not started.
