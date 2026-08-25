# Conversion 6 — pre-registered stop conditions

Base `99f6009`. **Written and committed BEFORE the production switch.** Nothing
here is adjusted afterwards; a breach is reported as a breach.

Unit: **raw production diff lines added + deleted**, hunk-classified. Never
net-executable.

---

## 1. Candidate and owned surface

`evolution`, comprising three route identities: `evolution`,
`evolution_funnel`, `evolution_pipeline_stage`.

Owned surface, from executed routing over 882 distinct corpus questions:

```
32 questions owned
15 delivered      (published rows, no insufficient-data warning)
17 refused
```

**The non-vacuity rule is permanent**: `ok=True` with zero published rows, or an
`insufficient-data` warning, is NOT delivered. Applying it dropped the delivered
count from 21 to 15 and turned three matrix cells red — those three are answered
by the fixture instead.

### Fixture denominator (ruled)

The configured production discovery root carries **zero** weekly extracts, so
every pipeline, stage and funnel question answers "No weekly pipeline extracts
are available" with `ok=True`. Pipeline / Stage / Funnel temporal execution is
therefore proved against the canonical `tests/fixtures/pipeline_history_5w`
fixture, which is the pre-registered denominator, requiring:

- non-empty published rows;
- five weekly frames;
- governed stage coverage across the canonical five stages;
- row-ID and count reconciliation;
- penny-exact economics.

Every statement resting on it is **fixture-proven, production-data-unexercised**
and must be reported as such. It is not production-delivered evidence.

---

## 2. Dependency matrix — the gate that had to be green first

```
dependency                  repr    owner   plan    delivered
dataset                     GREEN   GREEN   GREEN   GREEN
measure                     GREEN   GREEN   GREEN   GREEN
historical periods          GREEN   GREEN   GREEN   GREEN
time/grain                  GREEN   GREEN   GREEN   GREEN
source scope                GREEN   GREEN   GREEN   GREEN
row predicates              GREEN   GREEN   GREEN   GREEN
ordinary evolution          GREEN   GREEN   GREEN   GREEN
Pipeline evolution          GREEN   GREEN   GREEN   GREEN*
Pipeline Stage evolution    GREEN   GREEN   GREEN   GREEN*
Funnel                      GREEN   GREEN   GREEN   GREEN*
                                                    * fixture-proven
```

---

## 3. Thresholds

| | raw lines |
|---|---|
| shared conversion ceiling | **120** |
| route-specific range | **80 – 220** |
| cleanup (duplicate owners removed) | no ceiling; recorded separately |
| **total conversion ceiling** | **340** |

Breaching the total ceiling with an *architectural* cause is **H8**. Breaching it
because a named prerequisite turned out larger is recorded, not hidden.

---

## 4. Expected semantic dependencies

The converted route consumes, and re-derives none of:

```
dataset          interpretation.dataset          (workspace.resolve_dataset)
measure          interpretation.subject          (parser.metric / aggregation)
grain            interpretation.time.grain       (period_request)
source scope     SELECT_POPULATION(source_portfolio_lens)
row predicates   SELECT_POPULATION(row_predicates)
pipeline stage   the governed stage claim        (lexical.pipeline_stage_request)
```

---

## 5. Delivered minimums

The conversion is not accepted unless the equivalence harness exercises at
least:

- **8** ordinary funded evolution questions delivering a real series;
- **1** filtered funded evolution case, penny-exact against
  `£432,425,355.79 → £450,969,362.11 → £472,527,483.38`;
- **5** weekly frames × **5** governed stages on the fixture, with identical
  case IDs and penny-exact amounts;
- **0** questions whose delivered economics move.

An equivalence measured over refusals alone is rejected.

---

## 6. Allowed movements

### AUTHORISED H4 — GOVERNED STAGE VOCABULARY ACTIVATION

Ruled: `pipeline_stage_request` is the intended product vocabulary;
`_FUNNEL_KEYWORDS` is a legacy implementation limitation, not a compatibility
contract. C6 may move questions REFUSED → DELIVERED **only** where the existing
governed stage claim already resolves them.

Measured across all 882 corpus questions, before conversion:

```
questions naming a stage : 24
governed == legacy       : 24
governed-only            :  0
legacy-only              :  0
disagree                 :  0
```

**The corpus activation set is EMPTY.** The governed vocabulary is a superset —
21 spellings against 5 substrings — but no shipped question exercises the
difference. The 16 additional spellings are:

```
KFI          illustration · kfi issued · quote
APPLICATION  applied
OFFER        offer issued · offered
COMPLETED    drawdown · drawn · funds released · live
WITHDRAWN    abandoned · cancelled · declined · lapsed · rejected · withdrawn
```

So C6 is expected to be **equivalence-preserving in fact**, with the wider
vocabulary latent. Any question that does move must be enumerated individually
and classified as AUTHORISED H4, never described as equivalence-preserved.

### Everything else

- The **17 corpus refusals stay refused.** They are grouped evolution (by
  region, broker, LTV bucket) and pipeline-by-week; enabling any of them is
  capability expansion, not migration.
- Direct / Acquired trends continue to route to `cohort_progression`. C6 does
  not take ownership of them.
- A named-portfolio trend continues to refuse.

---

## 7. Prohibited

- new stage vocabulary or stage semantics of any kind;
- new filter, dimension, ranking or time vocabulary;
- any capability beyond the governed stage activation above;
- new route-shape branching (`if grouped_filtered_trend … elif …`) — that is
  **H9** and stops the programme;
- changing what "Funded Book", "Direct" or "Acquired" mean;
- treating fixture evidence as production evidence.

---

## 8. Stop conditions

- **STOP — C6 THRESHOLD BREACH** total conversion cost over 340 raw lines with
  an architectural cause (H8).
- **STOP — C6 EQUIVALENCE FAILURE** any delivered economic movement not in the
  authorised set.
- **STOP — C6 SILENT DEGRADATION** any dropped facet, widened population,
  substituted measure, changed dataset, or evidence-free certification (H6).
- **STOP — C6 SHAPE CASCADE** conversion requires route-shape branching (H9).
- **STOP — C6 CAPABILITY EXPANSION** any refusal→delivery outside the governed
  stage activation (H4).
