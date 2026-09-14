# Targeted remediation analysis — 25 genuine partials, 24 honest refusals

    READ-ONLY. No product code, tests, bank, expectations or scores touched.
    No model call, no /mi/query, no certification case re-run.

    SOURCE   due_diligence/evidence/mi_135_case_level/mi_135_case_level_evidence.json @ 465e9305
    PRODUCT  5c436961ebe279bc0820ab006867b9f8a869bde2
    SNAPSHOT synthetic_demo/output/multibook/platform_2026-06-30_canonical_typed.csv (79 columns)

## The headline, before the detail

**Nineteen of the 25 "genuine" partials are not product defects.** They are the bank
asserting a value at the INTERPRETATION layer that the shipped contract explicitly
tells the interpreter not to state, and that the COMPILER already binds correctly.

`opus_interpreter.py` rule 7, verbatim:

> `statistic` is what the question asks for. If it does not say, leave it empty and
> the governed registry's default will apply. Do not invent one.

`vocabulary.GOVERNED_DEFAULTS["measures[].statistic"]`, verbatim:

> empty resolves to the governed registry default for that concept (a sum for
> balances, an exposure-weighted average for LTV and rates).

And the plan for Q11A, from the committed evidence:

    model  measures: [{"concept": "current_outstanding_balance", "statistic": null}]
    plan   outputs : [{"measures": [{"concept": "current_outstanding_balance",
                        "statistic": "sum", "statistic_defaulted": true, ...}]}]
    envelope: ok = true, 52 groups returned

The model did what it was told, the compiler defaulted by the registry, and the
answer came back. The fixture expects `statistic: ['sum']` on the candidate intent,
and `score_135._score_interpretation` scores the candidate intent. **That is a
BANK_OR_TEST_DEFECT of layer, not a product failure**, and it is 17 statistic cases
plus 2 weighting cases where the plan likewise binds `weight_field:
current_outstanding_balance` from the registry.

So the product-remediable population inside the 25 is **three cases**, plus two
gated on absent data and one that no deterministic rule may own.

### Part A — the 25 genuine PARTIALLY_CORRECT cases

| CASE_ID | PRIMARY | EXPECTED / CURRENT | EARLIEST_FAILURE_LAYER | OWNER_SUPPORTS | GROUP |
| --- | --- | --- | --- | --- | --- |
| Q11A | STATISTIC | statistic sum / model null (plan binds sum) | COMPILER (already correct) | YES | A1_STATISTIC_ASSERTED_AT_WRONG_LAYER |
| Q11B | STATISTIC | statistic sum / model null (plan binds sum) | COMPILER (already correct) | YES | A1_STATISTIC_ASSERTED_AT_WRONG_LAYER |
| Q11C | STATISTIC | statistic sum / model null (plan binds sum) | COMPILER (already correct) | YES | A1_STATISTIC_ASSERTED_AT_WRONG_LAYER |
| Q12A | STATISTIC | statistic sum / model null (plan binds sum) | COMPILER (already correct) | YES | A1_STATISTIC_ASSERTED_AT_WRONG_LAYER |
| Q12B | STATISTIC | statistic sum / model null (plan binds sum) | COMPILER (already correct) | YES | A1_STATISTIC_ASSERTED_AT_WRONG_LAYER |
| Q13A | STATISTIC | statistic sum / model null (plan binds sum) | COMPILER (already correct) | YES | A1_STATISTIC_ASSERTED_AT_WRONG_LAYER |
| Q13B | STATISTIC | statistic sum / model null (plan binds sum) | COMPILER (already correct) | YES | A1_STATISTIC_ASSERTED_AT_WRONG_LAYER |
| Q13C | STATISTIC | statistic sum / model null (plan binds sum) | COMPILER (already correct) | YES | A1_STATISTIC_ASSERTED_AT_WRONG_LAYER |
| Q15A | STATISTIC | statistic sum / model null (plan binds sum) | COMPILER (already correct) | YES | A1_STATISTIC_ASSERTED_AT_WRONG_LAYER |
| Q15B | STATISTIC | statistic sum / model null (plan binds sum) | COMPILER (already correct) | YES | A1_STATISTIC_ASSERTED_AT_WRONG_LAYER |
| Q15C | STATISTIC | statistic sum / model null (plan binds sum) | COMPILER (already correct) | YES | A1_STATISTIC_ASSERTED_AT_WRONG_LAYER |
| Q16A | STATISTIC | statistic sum / model null (plan binds sum) | COMPILER (already correct) | YES | A1_STATISTIC_ASSERTED_AT_WRONG_LAYER |
| Q16B | STATISTIC | statistic sum / model null (plan binds sum) | COMPILER (already correct) | YES | A1_STATISTIC_ASSERTED_AT_WRONG_LAYER |
| Q16C | STATISTIC | statistic sum / model null (plan binds sum) | COMPILER (already correct) | YES | A1_STATISTIC_ASSERTED_AT_WRONG_LAYER |
| Q17A | STATISTIC | statistic sum / model null (plan binds sum) | COMPILER (already correct) | YES | A1_STATISTIC_ASSERTED_AT_WRONG_LAYER |
| Q17B | STATISTIC | statistic sum / model null (plan binds sum) | COMPILER (already correct) | YES | A1_STATISTIC_ASSERTED_AT_WRONG_LAYER |
| Q17C | STATISTIC | statistic sum / model null (plan binds sum) | COMPILER (already correct) | YES | A1_STATISTIC_ASSERTED_AT_WRONG_LAYER |
| Q05B | WEIGHTING | weight balance / model null (plan binds balance) | COMPILER (already correct) | YES | A1_STATISTIC_ASSERTED_AT_WRONG_LAYER |
| Q05C | WEIGHTING | weight balance / model null (plan binds balance) | COMPILER (already correct) | YES | A1_STATISTIC_ASSERTED_AT_WRONG_LAYER |
| Q21A | TEMPORAL | temporal relative_pair / previous_reporting_period | NORMALIZATION | NO | A3_ATTRIBUTION_OWNER_OPERATION_SET |
| Q22C | OPERATION | operation rank / compare | NORMALIZATION | NO | A3_ATTRIBUTION_OWNER_OPERATION_SET |
| Q12C | OPERATION | operation breakdown / distribution | CAPABILITY_ROUTING | PARTIAL | A4_COMPILER_EXECUTOR_OPERATION_GAP |
| Q25C | OPERATION | operation forecast_projection / headroom | DETERMINISTIC_OWNER | NOT_ESTABLISHED | A5_DATA_GATED |
| SM03C | OPERATION | operation transition / arrivals | DETERMINISTIC_OWNER | NOT_ESTABLISHED | A5_DATA_GATED |
| Q10B | OPERATION | operation summary / breakdown | INTERPRETATION | NOT_ESTABLISHED | A6_INTERPRETATION_ONLY |

### Part A groups, with the minimum central change

**A1 — STATISTIC/WEIGHTING asserted at the wrong layer (19 cases)**
Q11A/B/C, Q12A/B, Q13A/B/C, Q15A/B/C, Q16A/B/C, Q17A/B/C, Q05B, Q05C.
Root cause: the fixture asserts on the candidate intent a value the interpreter is
instructed to omit and the compiler owns. `EXISTING_OWNER_ALREADY_SUPPORTS_REQUEST
= YES` for all 19 — the binding is already correct and already evidenced.
**MINIMUM_TARGET_STATE_CHANGE: none to product.** Test layer only, and the choice is
the operator's: either `score_135` compares the compiled `MeasureBinding.statistic`
/ `weight_field` for those two dimensions, or the fixtures record them as satisfied
by governed default. Not a product P0 under any ranking.

**A3 — the attribution owner's operation set is too narrow (2 of the 25; 3 in total)**
Q21A, Q22C — and Q22A, which sits in the already-excluded disputed 11.
`CAPABILITY_OPERATIONS['funded_bridge'] = {bridge, movement}`. `change_form =
attribution` maps to `funded_bridge`, but the natural attribution shapes are "which
X contributed most" (`rank`) and "did A or B drive more" (`compare`), neither of
which that owner admits. normalise rule 4 rebinds capability only when the implied
owner supports the stated operation, so it declines; the plan keeps
`period_movement`; eligibility then refuses `CAPABILITY_NOT_BRIDGE`.
**MINIMUM_TARGET_STATE_CHANGE:** one entry on one table —
`CHANGE_FORM_OPERATION_VARIANTS['attribution'] = {bridge, movement, rank, compare}`
with canonical `movement`, exactly the shape of the metric_delta fix this programme
already shipped. `level_comparison` runs `generic_analysis`, so nothing is taken
from another form.

**A4 — the compiler admits an operation the generic executor cannot run (1)**
Q12C. `distribution` ∈ `CAPABILITY_OPERATIONS['generic_analysis']` but ∉
`plan_runtime_adapter.ELIGIBLE_OPERATIONS = {breakdown, point_in_time}`. The plan
compiles and the executor then declines, so the question silently takes the legacy
path.
**MINIMUM_TARGET_STATE_CHANGE:** make one of the two sets derive from the other so
they cannot drift. Either the executor admits `distribution`, or the capability
stops declaring it and the compiler refuses cleanly. A refusal is the safer of the
two; a fallback that looks like success is the thing to remove.

**A5 — gated on absent data (2)** Q25C, SM03C. Both `POPULATION_NOT_EXECUTABLE`
on forecast/pipeline bases. No product change; see refusal group R6.

**A6 — interpretation-only (1)** Q10B. `breakdown` and `summary` are both declared
`pipeline` operations and the wording supports either. **NOT_ESTABLISHED, and no
central fix is proposed**: owning this would mean a wording rule after compilation,
which the target state forbids.

---

## Part B — the 24 HONEST_REFUSAL cases

| CASE_ID | REFUSAL_CODE | REQUIRED_FIELDS / CAPABILITY | FIELDS_ON_SNAPSHOT | ROOT_CAUSE | ANSWERABLE_WITH_EXISTING_DATA |
| --- | --- | --- | --- | --- | --- |
| B01 | INELIGIBLE:CAPABILITY_NOT_BRIDGE | facility bound to client_id=ERE / borrowing_base | N/A | CONFIG_SCOPE | NOT_ESTABLISHED |
| B03 | INELIGIBLE:CAPABILITY_NOT_BRIDGE | as B01 | N/A | CONFIG_SCOPE | NOT_ESTABLISHED |
| B04 | INELIGIBLE:CAPABILITY_NOT_BRIDGE | as B01 | N/A | CONFIG_SCOPE | NOT_ESTABLISHED |
| C03 | INELIGIBLE:CAPABILITY_NOT_GENERIC | as B01 | N/A | CONFIG_SCOPE | NOT_ESTABLISHED |
| C04 | INELIGIBLE:CAPABILITY_NOT_GENERIC | as B01 | N/A | CONFIG_SCOPE | NOT_ESTABLISHED |
| Q25A | INELIGIBLE:CAPABILITY_NOT_GENERIC | limit schedule for ERE / limit_assessment + forecast | N/A | CONFIG_SCOPE | NO |
| Q25B | INELIGIBLE:CAPABILITY_NOT_GENERIC | as Q25A | N/A | CONFIG_SCOPE | NO |
| Q5.1 | INELIGIBLE:CAPABILITY_NOT_GENERIC | as Q25A | N/A | CONFIG_SCOPE | NO |
| Q5.2 | INELIGIBLE:CAPABILITY_NOT_GENERIC | as Q25A | N/A | CONFIG_SCOPE | NO |
| Q07A | INELIGIBLE:CAPABILITY_NOT_GENERIC | co-dated snapshots per source portfolio | YES | GOVERNANCE | NO |
| Q07C | INELIGIBLE:CAPABILITY_NOT_GENERIC | as Q07A | YES | GOVERNANCE | NO |
| Q08A | INELIGIBLE:CAPABILITY_NOT_GENERIC | as Q07A | YES | GOVERNANCE | NO |
| Q08B | INELIGIBLE:CAPABILITY_NOT_GENERIC | as Q07A | YES | GOVERNANCE | NO |
| Q08C | INELIGIBLE:CAPABILITY_NOT_GENERIC | as Q07A | YES | GOVERNANCE | NO |
| Q04A | INELIGIBLE:GEOGRAPHY_REQUESTED | canonical_region_reporting | NO (raw region cols present) | FIELD_CONNECTIVITY | YES |
| Q04B | INELIGIBLE:GEOGRAPHY_REQUESTED | as Q04A | NO | FIELD_CONNECTIVITY | YES |
| Q04C | INELIGIBLE:GEOGRAPHY_REQUESTED | as Q04A | NO | FIELD_CONNECTIVITY | YES |
| Q07B | INELIGIBLE:CAPABILITY_NOT_GENERIC | source_portfolio_type | YES | FIELD_CONNECTIVITY | YES |
| Q05A | PLAN_RECEIPT_RECONCILIATION_FAILED | erm_product_type | NO | GENUINE_DATA_ABSENCE | NO |
| Q4.2 | INELIGIBLE:FILTERS_NOT_SUPPORTED | pipeline tape / pipeline capability | NO | GENUINE_DATA_ABSENCE | NO |
| Q9.1 | INELIGIBLE:POPULATION_NOT_EXECUTABLE | pipeline tape / forecast | NO | GENUINE_DATA_ABSENCE | NO |
| Q21B | INELIGIBLE:CAPABILITY_NOT_BRIDGE | current_loan_to_value threshold on funded_bridge | YES | CAPABILITY_CONNECTIVITY | NOT_ESTABLISHED |
| Q14C | INELIGIBLE:GEOGRAPHY_REQUESTED | — ('both' captured as a filter value) | N/A | OTHER | NOT_ESTABLISHED |
| Q21C | INELIGIBLE:CAPABILITY_NOT_BRIDGE | — ('among' captured as a filter value) | N/A | OTHER | NOT_ESTABLISHED |

### The one refusal finding that is not what the refusal says

Q04A/B/C refuse with "no loans in this book match that filter". **Eight rows on the
snapshot satisfy London + Direct + age over 75.** The plan binds geography to
`canonical_field: canonical_region_reporting`, and that column is **not on the
tape** — the tape carries `collateral_geography`, `geographic_region_collateral`,
`geographic_region_obligor` and their ITL3 variants.

`config/mi/region_taxonomy.yaml` says where the missing column comes from:

> Resolution order at onboarding … → `canonical_region_detail` +
> `canonical_region_reporting` on the tape.
> The runtime MI path reads the persisted canonical columns only.

So the resolver, the taxonomy and the capability all exist and are correct. **The
onboarding region-harmonisation step was never run for this tape.** Twelve of the
135 bind that field: Q04A/B/C, Q14A/B/C, Q16A/B/C, Q21A/B/C.

By contrast `erm_product_type` (18 cases touched) has no column, no derivation and
no onboarding step that would produce one — `amortisation_type` is a different
concept and is constant across all 118 rows. That one is genuinely absent source
data.

---

## Part C — consolidated root causes

| ROOT_CAUSE_GROUP | CASES | CASE_IDS | CURRENT_OWNER | TARGET_OWNER | CHANGE_TYPE | CONVERTIBLE | REGRESSION_RISK | SIZE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 region harmonisation absent from the tape | 12 | Q04A/B/C, Q14A/B/C, Q16A/B/C, Q21A/B/C | onboarding `engine/region_taxonomy.py`; tape lacks the persisted columns | same, run for this tape | DATA_REQUIRED | 3–6 | very low — no code changes | pipeline re-run, no LOC |
| R2 facility config scope | 5 | B01, B03, B04, C03, C04 | `config/risk/funding_facilities.yaml` @ client_id `ere_funding_uk` | a facility bound to the runtime client_id | CONFIG | 0–5 | low, but it ENABLES a refusing path — needs its own acceptance | config entry |
| R3 limit-schedule scope (+ forecast) | 4 | Q25A, Q25B, Q5.1, Q5.2 | `config/clients/client_001/risk_limits_extracted.yaml` | a schedule bound to the runtime client_id | CONFIG + DATA_REQUIRED | 0–2 | medium — forecast half still needs pipeline data | config entry |
| R4 source portfolios not co-dated | 5 | Q07A, Q07C, Q08A, Q08B, Q08C | governed rule refusing across reporting dates | unchanged rule; co-dated snapshots | DATA_REQUIRED | 0–5 | very low — the rule stays | data only |
| R5 `erm_product_type` absent | 18 touched, 1 refusal + 11 partials | Q05A + the product-type filtered set | source tape | source tape carries the field | DATA_REQUIRED | 1–3 | very low | data only |
| R6 pipeline / forecast tape absent | 4 | Q4.2, Q9.1, Q25C, SM03C | `config/mi/pipeline_field_contract.yaml` declares it; no file exists | the prepared pipeline tape | DATA_REQUIRED | 0–4 | low | data only |
| R7 attribution owner operation set | 3 | Q21A, Q22C, Q22A | `CAPABILITY_OPERATIONS['funded_bridge'] = {bridge, movement}` | `CHANGE_FORM_OPERATION_VARIANTS['attribution']` | SEMANTIC_CONTRACT | 2–3 | low — one table, same shape as the shipped metric_delta fix | ~5 LOC + controls |
| R8 compiler/executor operation gap | 1 | Q12C | `CAPABILITY_OPERATIONS` vs `plan_runtime_adapter.ELIGIBLE_OPERATIONS` | one derived from the other | SEMANTIC_CONTRACT | 0–1 | low; removes a silent fallback | ~5 LOC |
| R9 dimension parsed, neither applied nor rejected | 1 | Q07B | generic executor dimension binding | same | CONNECTIVITY | 0–1 | low | NOT_ESTABLISHED |
| R10 stopword captured as a filter value | 2 | Q14C, Q21C | interpretation | — | NOT_ESTABLISHED | 0 | — | no proposal |
| T1 statistic/weight asserted at the wrong layer | 19 | Q11A/B/C, Q12A/B, Q13A/B/C, Q15A/B/C, Q16A/B/C, Q17A/B/C, Q05B/C | `score_135._score_interpretation` reads the candidate intent | compare the compiled binding, or mark satisfied-by-default | **BANK_OR_TEST_DEFECT** | 19 on the scorecard, **0 in product behaviour** | n/a | test only |

R1 and R7 overlap on Q21A: its proximate blocker is the attribution operation set,
and it also binds the absent region column. Counted once in each group's own scope
and once only in the totals below.

### Proposals explicitly NOT made, and why

Nothing above duplicates a deterministic calculation, adds a question-specific
wording rule, rereads raw question text after compilation, weakens a refusal, or
widens scope silently. Three candidates were considered and rejected on exactly
those grounds:

- **Teaching the interpreter to emit `statistic: sum`.** Rejected: it contradicts
  `opus_interpreter` rule 7, duplicates the registry default the compiler already
  owns, and would move a governed decision back into the model. The 19 cases are a
  test-layer defect and are fixed there or not at all.
- **A wording rule for Q10B's "overview" vs "by size and stage".** Rejected: it is a
  question-specific wording rule by construction.
- **Admitting `distribution` in the executor by widening the eligible set alone.**
  Flagged: acceptable ONLY if the two sets are made to derive from one owner.
  Widening one by hand leaves the same drift that produced the silent fallback.

R2 and R3 carry a distinct risk the others do not: they turn a refusing path into an
answering one. A facility or limit schedule bound to this portfolio makes the
borrowing-base and limit capabilities execute for the first time in this estate, and
nothing in this certification measured them executing. Neither should ship on the
strength of "the refusal went away".

---

## Part D — ranked remediation plan

| RANK | GROUP | CASES | WHY HERE |
| --- | --- | --- | --- |
| **P0** | R1 region harmonisation | 12 | Largest single population, no code change, no regression surface, and it is a governed step that was simply not run. Cross-lender: every multi-book client needs it. |
| **P0** | R7 attribution operation set | 3 | One table entry, an owner that already exists, and the exact shape of a fix this programme has already shipped and controlled once. Highest confidence per line changed. |
| **P1** | R5 `erm_product_type` | 18 touched | Largest touched population, but it is source data the lender must supply; nothing in the product can recover it. |
| **P1** | R4 co-dating | 5 | Removes five refusals without touching the governed rule that produced them. Data-side, low risk. |
| **P1** | R8 compiler/executor gap | 1 | One case, but it removes a class of silent fallback rather than one symptom. |
| **P2** | R2 facility config | 5 | Config is trivial; the capability it enables is unmeasured in this estate and needs its own acceptance first. |
| **P2** | R3 limit schedule | 4 | As R2, and the forecast half is still blocked by R6. |
| **P2** | R6 pipeline tape | 4 | Unblocks R3's forecast half and four cases, but it is a whole additional governed dataset. |
| **P2** | R9 dimension connectivity | 1 | One case; root cause not yet traced to a line. |
| — | R10 stopword | 2 | No proposal. NOT_ESTABLISHED. |
| **not a product P0** | T1 statistic/weight layer | 19 | It is a fixture/scorer change. Ranking a benchmark-only change as a product P0 is exactly what the brief forbids, however large the scorecard movement. |

---

## Totals

    GENUINE_PARTIALS_ANALYSED = 25
    HONEST_REFUSALS_ANALYSED  = 24

    PARTIALS_ANSWERABLE_BY_EXISTING_OWNER = 22
      19 already bound correctly by the compiler (T1, test-layer only)
       3 reachable through an existing owner once its operation set admits the
         form's shapes (R7 x2, R8 x1)
    PARTIALS_REQUIRING_NEW_CAPABILITY     = 0
    PARTIALS_NOT_ESTABLISHED              = 3   (Q10B, Q25C, SM03C)

    REFUSALS_WITH_DATA_GENUINELY_ABSENT            = 3   Q05A, Q4.2, Q9.1
    REFUSALS_WITH_DATA_PRESENT_BUT_NOT_CONNECTED   = 4   Q04A, Q04B, Q04C, Q07B
    REFUSALS_CAPABILITY_EXISTS_NOT_CONNECTED       = 1   Q21B
    REFUSALS_CONFIG_SCOPE                          = 9   B01, B03, B04, C03, C04,
                                                         Q25A, Q25B, Q5.1, Q5.2
    REFUSALS_REQUIRING_NEW_CAPABILITY              = 0
    REFUSALS_NOT_ESTABLISHED                       = 2   Q14C, Q21C
    (also: 5 GOVERNANCE — Q07A/C, Q08A/B/C — the rule is correct; the data is not
     co-dated)

## Top shared root causes

    1. Region harmonisation never run for this tape          12 cases
    2. `erm_product_type` absent from source data            18 touched
    3. Facility / limit config bound to another client_id     9 cases
    4. Source portfolios not co-dated                         5 cases
    5. Pipeline / forecast tape absent                        4 cases
    6. Attribution owner's operation set too narrow           3 cases
    7. (test layer) statistic/weight asserted at the wrong
       layer — the single largest scorecard effect and
       NOT a product defect                                  19 cases

**Five of the top six are data or configuration, not code.** The only product-code
proposals in this analysis are R7 (~5 lines on one vocabulary table) and R8 (~5
lines reconciling two operation sets). That is the honest shape of this estate:
it is not mainly mis-built, it is mainly under-supplied.

## Recommended remediation sequence

    1. R1  run the governed onboarding region harmonisation for this tape
    2. R7  admit the attribution shapes on CHANGE_FORM_OPERATION_VARIANTS
    3. R4  co-date the source portfolio snapshots
    4. R8  make the compiler and executor operation sets derive from one owner
    5. R5  obtain erm_product_type on the source tape
    6. R6  obtain the prepared pipeline tape
    7. R2/R3  bind facility and limit config — behind their own acceptance, because
              each turns a refusing path into an answering one for the first time
    8. T1  settle the statistic/weight layer question in the test estate, separately
           and after the product work, so the two are never confused

Steps 1, 3, 5 and 6 need nothing from this repository. Steps 2 and 4 are the whole
of the proposed code change.

## Realistic upside — an estimate, not a result

The frozen certification is unchanged: FULLY_CORRECT 66 of 134 measured, 0.4925.

    of the 25 genuine partials, realistically convertible      3 - 6
      R7 2-3, R8 0-1, R1 0-3 (the Q16 set also carries the T1 layer issue)
    of the 24 refusals, realistically recoverable              9 - 17
      R1 3, R4 0-5, R2 0-5, R3 0-2, R7-adjacent 0-1, R9 0-1
    plus the already-established disputed 11 and the test-layer 19, neither of
    which is product work

    implied fully-correct range if the PRODUCT work alone landed   69 - 72   (0.51 - 0.54)
    if the DATA and CONFIG work also landed                        75 - 85   (0.56 - 0.63)
    if the test-layer question resolved the other way as well      up to ~104 (0.78)

**Read the third line with suspicion.** It is the arithmetic of a scorecard, not of a
product: those 19 cases already behave correctly and already return answers, so that
movement would record a measurement being corrected, not a system improving. The
first two lines are the ones that describe real change, and neither has been
measured — every range here is an estimate from static evidence and nothing was
re-run to test it.
