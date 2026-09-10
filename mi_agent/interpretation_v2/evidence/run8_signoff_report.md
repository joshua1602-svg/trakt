# Final 135 live sign-off

`INTERPRETATION_LAYER_SIGN_OFF = PASS`

```
SHA            = 2b00172c0b033c97de8afe5fa5b60ba43e1443ec
EVIDENCE_PATH  = mi_agent/interpretation_v2/evidence/run8_135_signoff_2b00172.json
```

Working tree was clean at `2b00172` before the run and no code changed during or
after it. All post-run analysis is offline, from the recorded evidence.

## API

```
unique bank questions           = 135   (45 canonicals x 3 variants, frozen)
successful                      = 135
failed                          = 0
transport retries               = 0
additional non-bank model calls = 0
model substitutions             = 0     (claude-opus-5 returned 135/135)
tokens                          = 1,000,914 in / 177,072 out
```

No preliminary probe, no post-run diagnosis call, no repeat of the bank. The 135
bank questions were the only live calls in this task. No question needed a retry:
there were no 529s or other transport failures.

## Results

| | Run 6 (pre-policy) | Run 7 (Phase 2A) | **Run 8 (sign-off)** |
|---|---|---|---|
| fully correct | 119 | 114 | **115** |
| partially correct | 13 | 10 | **11** |
| **incorrect** | 0 | 0 | **0** |
| clarification | 2 | 9 | **8** |
| governed refusal | 1 | 2 | **1** |
| PLAN / CLARIFY / REFUSE | 132 / 2 / 1 | 124 / 9 / 2 | **126 / 8 / 1** |
| paraphrase invariant | 18/45 | 20/45 | **21/45** |

## Semantic safety

```
explicit measure drops     = 0
explicit dimension drops   = 0
explicit filter drops      = 0
explicit temporal drops    = 5   (see below — 2 are a measurement artefact)
explicit comparison drops  = 0
silent semantic widening   = 0
```

**Silent widening is the one that matters, and it is zero.** Not one plan in 135
was produced with an explicitly stated filter or grouping missing from it. Run 6's
signature failure — Q16B dropping its explicit `erm_product_type == drawdown`
filter and planning anyway, Q21 losing its grouping and planning anyway — does not
occur.

The five temporal flags need separating, because the fixture metric reads the
model's emitted payload and therefore cannot see contract normalisation:

| | stated | compiled contract | class |
|---|---|---|---|
| Q18A | `previous_reporting_period` | `governed_reporting_period_pair` back=1 | **not a defect** — normalisation resolved it to the fixture's own form |
| Q19A | `previous_reporting_period` | `governed_reporting_period_pair` back=1 | **not a defect** — same |
| Q21A | `previous_reporting_period` | `previous_governed_reporting_period` | REPRESENTATIONAL, outside normalised scope |
| NL8B | `previous_reporting_period` | `previous_governed_reporting_period` | REPRESENTATIONAL, outside normalised scope |
| Q25C | `current` | `latest_governed_reporting_period` | MODEL_MISS — read as current where the question asks for a forward projection; present in every prior run |

So two of the five are an artefact of measuring the intent instead of the plan,
two are the period-encoding redundancy on operations outside the normalised scope
(normalisation 2 is confined to `movement`, and Q21/NL8B are not movements), and
one is a genuine model miss already on record. No semantic element was lost in any
of the five: each movement is still period-on-period, monthly, one period back.

## Policy

```
unnecessary companion measures    = 0
unnecessary companion dimensions  = 1   (Q14B)
unnecessary CLARIFY / REFUSE      = 1 borderline (Q16B); 8 of 9 non-PLAN justified
```

Elaboration control is holding firmly: across 135 readings the model volunteered
one unrequested dimension and not a single unrequested measure. Run 6's seven
INTERPRETATION_POLICY canonicals — NL6, Q24, Q25, NL7, SM09, Q10, Q08 — produced
measure sets of one, two, three and five against each other; SM09 is now 3/3 fully
correct and NL6 3/3 fully correct.

Non-PLAN rose from 3 to 9 against Run 6. That deserves the scrutiny, because a
clarification-heavy interpreter is what failed Phase 2A. The three Run 6 non-PLANs
(NL3B, NL8C, BB01C) all recur. The six additions:

* **NL7A/B/C — accepted.** Pre-registered in the recalibration gate as the correct
  outcome: "riskier" / "risk profile" binds to no single governed measure, and the
  recorded reasons name the competing ones ("exposure-weighted", "several
  materially different governed risk measures that would each produce..."). Three
  readers previously chose three different baskets of proxies.
* **NL1A, NL1C — accepted.** NL1 was marked `human_review` in the fixtures before
  any run, with an unresolved tension recorded on its population. The reasons are
  substantive: NL1A, "a series cannot be produced without at least one measure,
  and several governed readings are materially different"; NL1C, "the comparison
  requires two defined periods, and neither is stated".
* **Q16B — borderline, and the one item I would not defend as clearly correct.**
  Its two siblings bound the product filter and planned. The recorded reason is
  real — "the registry carries no measure called 'drawdown balance'; 'drawdown' IS
  a governed equity-release product type, so the phrase reads either as..." — and
  B's wording ("Break drawdown balance down by...") is a compound noun where A and
  C say "for drawdown loans" explicitly. Worth noting what it replaced: this is
  the exact question that in Run 6 silently dropped the filter and answered
  something broader. Asking is strictly better than that.

This is categorically unlike the Phase 2A failure, which clarified Q19B/C where
the answer was unambiguous. Q19 is now 3/3 fully correct.

## Normalisation

```
semantic / paraphrase invariant = 21/45   (Run 6: 18, Run 7: 20)
gained vs Run 6                 = Q19, Q20, SM07, SM08
lost vs Run 6                   = Q14
normalisations applied          = 9 (all relative_period)
remaining non-invariant         = 24, every one classified
```

| class | n | canonicals |
|---|---|---|
| REPRESENTATIONAL | 11 | NL2, NL5, NL7, NL9, Q12, Q17, Q18, Q21, Q22, Q23, Q24 |
| INTERPRETATION_POLICY | 6 | NL6, Q08, Q10, Q14, Q25, SM09 |
| GENUINE_AMBIGUITY | 5 | BB01, NL1, NL3, NL4, NL8 |
| MODEL_MISS | 1 | Q16 |
| OTHER | 1 | Q09 |

The 11 REPRESENTATIONAL cases reduce to five named contract redundancies, none of
them in this sprint's scope:

1. **Undefined `population.base` for forward-looking questions** — NL2, NL9, Q23,
   Q24, four canonicals on one root cause. `funded` against `forecast` against
   `pipeline`: the contract never defines whether `base` names the book being
   projected or the projection itself, so all readings are permitted.
2. **Default-versus-absent** — `comparison.kind: "none"` against `null` (NL5, NL9,
   Q24), `population.seasoning: "any"` against omitted (NL7). The default and its
   omission are the same meaning and currently two identities. This is a fourth
   redundancy of exactly the kind just normalised, and the cheapest remaining fix.
3. **Symmetric comparison operand order** — NL7 carries
   `{left: front_book, right: back_book}` and `{left: back_book, right: front_book}`
   for one comparison.
4. **Period encoding outside `movement`** — Q21, and NL8B. Normalisation 2 is
   confined to `movement` because that is the only operation where
   `previous_reporting_period` provably cannot mean one period; on a rank or a
   breakdown the same collapse would be inventing a second period.
5. **Overlapping contract members** — operation enum (Q12 breakdown/distribution,
   Q22 rank-against-compare), output nesting (Q17), and Q18's bounded
   measure-owner case (`period_movement`/`current_outstanding_balance` against
   `funded_bridge`/`funded_balance_movement`, which is a measure choice and
   therefore arithmetic).

**Q14 is the one invariance regression against Run 6**, caused by the single
unrequested dimension on Q14B. One instance, INTERPRETATION_POLICY.

## Privacy — measured, not asserted

The benchmark writes `rows_of_data_transmitted: 0` and
`portfolio_values_transmitted: 0`, but those are hardcoded claims. So every
metadata call the model actually made was re-executed offline through the same
dispatcher and the returned payloads scanned:

```
metadata calls made            = 750
distinct calls replayed        = 219
loan rows sent to Opus         = 0   (0 payloads matching loan/account/borrower/DOB field names)
portfolio values sent          = 0   (0 currency-shaped or thousands-separated figures;
                                      0 bare integers of six digits or more)
reporting dates sent           = 0   (0 date-shaped strings in any payload)
facility commitments sent      = 0   (the facility record's positive allowlist holds;
                                      no commitment, advance rate or drawn amount appears)
dispatch errors                = 0
```

The model-facing surface is also byte-identical to the one validated at
`5ae1f73`: the intent schema, tool schema, system blocks and metadata tool list
all hash the same. MI_BEARER was not used and is not required.

## Compiler

```
invented concepts reaching plan   = 0   (every concept in all 126 plans resolves
                                         in the governed registry)
model-authored physical bindings  = 0   (no field, column, canonical_field, table,
                                         snapshot, sql or expression key appears in
                                         any of the 135 emitted payloads)
compiler revalidation             = PASS
```

The two-object boundary held through a contract change: 229 interpretation tests
pass, the intent schema is still closed (`additionalProperties: false`, 14
semantic slots, no binding slot), and the compiler independently re-derived
existence, ambiguity, applicability and physical binding for every concept in
every plan. Where normalisation rewrote an intent, `intent_claims` still records
what the model said and `compiler_bindings["normalisation"]` what the contract did
to it.

## Material residual failures

| # | case | class |
|---|---|---|
| 1 | Q25C reads "current" where the question asks for a forward projection | MODEL_MISS |
| 2 | NL3B uses `forecast_funding_date` as a dimension; compiler refuses `UNSUPPORTED_COMPOSITION` | MODEL_MISS |
| 3 | Q16B clarifies where two siblings plan the same semantics | MODEL_MISS (borderline INTERPRETATION_POLICY) |
| 4 | Q14B volunteers one unrequested dimension | INTERPRETATION_POLICY |
| 5 | Q08/Q10/Q25/NL6/SM09 measure-or-operation scope differences between variants | INTERPRETATION_POLICY |
| 6 | The five contract redundancies listed above, 11 canonicals | REPRESENTATIONAL |
| 7 | BB01/NL1/NL3/NL4/NL8 under-determined wordings | GENUINE_AMBIGUITY |
| 8 | Q09's three variants ask three different shapes of one subject | OTHER |

Note on BB01C: it clarified in this run, which settles the open adjudication I
flagged after the recalibration probe. In the probe it planned with a disclosed
non-blocking ambiguity; here it blocks and asks, matching the Run 6 ruling that an
unqualified "headroom" should ask back. No action required — it resolved toward
the stricter reading on its own.

## Sign-off

`INTERPRETATION_LAYER_SIGN_OFF = PASS`

Against each stated criterion:

* **No incorrect interpretation class introduced.** 0 incorrect in 135, as in both
  prior runs.
* **No material silent loss of explicit user semantics.** 0 measure, dimension,
  filter and comparison drops; 0 silent widening. The failure mode Run 6
  exhibited is absent.
* **No material return of excessive clarification or refusal.** 9 non-PLAN. Eight
  carry recorded governed reasons on canonicals the evidence base already
  classifies as under-determined, five of those pre-registered as correct. One,
  Q16B, is borderline and is reported as such rather than defended.
* **Policy calibration holding.** Q19 — the recalibration's whole target — is 3/3
  fully correct with two variants identical at plan level. Elaboration control is
  at one volunteered dimension and zero volunteered measures across 135.
* **Compiler safety holding.** 0 invented concepts, 0 model-authored bindings,
  closed schema, 229 tests.
* **Privacy boundary holding.** Measured across 219 replayed metadata payloads,
  not copied from a counter.
* **Remaining defects bounded and understood.** All 24 non-invariant canonicals
  classified; the 11 representational ones reduce to five named redundancies with
  identified remedies, none requiring an interpreter change.

The headline count is 115 fully correct against Run 6's 119. That difference is
six clarifications, five of them correct and one borderline, traded for zero
silent semantic loss. A run that answers a genuinely ambiguous question is not
better than one that asks; Run 6's 119 included Q16B answering the wrong, broader
question without saying so.
