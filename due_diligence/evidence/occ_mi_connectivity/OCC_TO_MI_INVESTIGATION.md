# OCC-to-MI connectivity investigation — pipeline, product semantics, region, client identity

    READ-ONLY. No code, configuration, OCC output, canonical data or MI semantics
    changed. No certification re-run, no model call. Nothing implemented.

    MI TAPE UNDER TEST  synthetic_demo/output/multibook/platform_2026-06-30_canonical_typed.csv
                        118 rows, 79 columns
    REAL ERE TAPE       ERE_Portfolio_122025_ESMA_Annex2_canonical_ESMA_Annex2_typed.csv
                        33 rows, 130 columns
    PRODUCT_SHA         5c436961ebe279bc0820ab006867b9f8a869bde2

## The structural fact that frames everything below

The dataset the MI certification queried was **not produced by the OCC / onboarding
path**. It was produced by `synthetic_demo/run_multibook_pipeline.sh`, whose own
header calls it "the synthetic demonstration", against
`synthetic_demo/config/config_client_SYNTHETIC_MULTIBOOK.yaml` with
`client_id: alderbridge_demo`.

That script invokes Gates 1, 2, 3, 3b, 3c, 4, 4b and the exception reconciliation.
It invokes **no** OCC step: `grep` for `operations_control`, `onboarding_agent` and
`region_taxonomy` across `synthetic_demo/*.sh` and `*.py` returns nothing, and no
`onboarding_output/` directory exists in the repository at all — the path
`config/mi/pipeline_field_contract.yaml` names as the prepared-pipeline output.

So the operator-confirmation and harmonisation steps that OCC owns were never run
for this tape. Three of the four issues below are that single fact seen from
different angles. **One is not, and it is the important correction in this report.**

---

# PART A — PIPELINE CONNECTIVITY

## The headline: pipeline data is NOT absent, and my earlier reading was wrong

The remediation analysis at `2243eafe` classified Q4.2 and Q9.1 as
`GENUINE_DATA_ABSENCE` on the strength of no pipeline file existing in the
repository. **That inference was wrong, and it is exactly the error this brief
warned against — inferring absence from a refusal.** The deployed system answers
pipeline questions with real governed figures:

    Q10A  "At the weekly extract of 12 January 2026 the pipeline holds 2,780 cases
           with a total pipeline amount of £562.9m"
    SM02A "£3.1m moved from Application to Offer between 2026-01-05 and 2026-01-12,
           across 22 cases"
    SM05A "113 cases newly entered KFI between 2026-01-05 and 2026-01-12,
           carrying £23.7m"
    Q3.1  "Offer stage pipeline is £29.9m across 179 case(s) as at 2026-01-12.
           Expected completion amount … £11.1m. Expected to land: 2026-02 £11.1m."

**32 of the 34 pipeline-base cases in the 135 are FULLY_CORRECT.** The pipeline
dataset is present, discoverable, correctly identified and correctly executed. The
repository carries no pipeline CSV because the runtime discovers it from a blob
root via `MI_AGENT_PIPELINE_URI` (`mi_agent_api/datasets.py:_pipeline_root`), not
from the repo.

## The four cases, traced

| CASE_ID | SOURCE_DATA_PRESENT | GOVERNED_DATASET_PRESENT | MI_RUNTIME_CAN_DISCOVER | REQUESTED IDENTITY | AVAILABLE IDENTITY | EXACT_FAILURE_SEAM | ROOT_CAUSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q4.2 | YES | YES | YES | pipeline base, `point_in_time`, with a filter | pipeline extract 2026-01-12 | the Pipeline owner publishes a fixed report that cannot be narrowed: `FILTERS_NOT_SUPPORTED` — "read from the Pipeline owner's own report, which cannot be narrowed without recomputing them here" | CAPABILITY_CONNECTIVITY |
| Q9.1 | YES | YES | YES | `population.base = forecast` | funded 2026-06-30 + pipeline 2026-01-12 | the governed slice executes `['funded']` only, so a `forecast` base is refused before it reaches a runtime; the legacy path then could not compose funded + pipeline | CAPABILITY_CONNECTIVITY |
| Q25C | YES | YES | YES | `population.base = whole_book` + forecast + limit schedule | as above | same slice perimeter, compounded by no limit schedule bound to this runtime identity | CAPABILITY_CONNECTIVITY |
| SM03C | YES | YES | YES | `pipeline_stage_movement`, operation `arrivals` | same | **none — the data executed and the answer is correct** ("5 cases moved from Offer to Completion"). The mismatch is `arrivals` vs the fixture's `transition`, an operation-vocabulary difference | OTHER |

    MINIMUM_TARGET_STATE_FIX
      Q4.2   give the pipeline owner a governed filtered read, or keep the refusal.
             Either is defensible; what is not defensible is answering a filtered
             pipeline question with an unfiltered figure, which it correctly refuses.
      Q9.1   admit `forecast` to the governed slice's executable population bases,
             or state that cross-dataset projection stays a legacy-path capability.
      Q25C   as Q9.1, plus the limit schedule (Part D identity).
      SM03C  no product change. Operation vocabulary only.

## One data-currency observation, not a defect

The pipeline extract is dated **2026-01-12** and the funded tape **2026-06-30** —
five and a half months apart. Every pipeline answer states its own extract date, so
nothing is misrepresented, but any question composing the two (Q9.1 is exactly that)
is composing states that are not co-dated. Recorded, not adjudicated.

---

# PART B — PRODUCT SEMANTICS

## What the governed configuration actually says

`config/mi/mi_equity_release_uk_applicability.yaml` is the authority, and it is
explicit about both fields and about where their values come from:

    - field: erm_product_type
      coverage_status_if_no_source: configured_static
      default_rule: "from product config"
      configured_value_source: product_config
      operator_question: "Confirm the ERM product type (e.g. lifetime mortgage)
                          from product config."
      reason: "ERM product type is known from the product setup, not necessarily
               a source column."

    - field: erm_sub_product_type
      coverage_status_if_no_source: configured_static
      default_rule: "from product config"
      configured_value_source: product_config
      operator_question: "Confirm the ERM sub-product type (lump sum / drawdown)
                          from product config."

**Both are operator-confirmed statics that OCC is supposed to populate from product
configuration — not columns a lender extract is expected to carry.** That single
line reframes the whole issue: this is not missing source data, it is an OCC
operator-confirmation step that was never completed.

## The product taxonomy, layer by layer

| CONCEPT | SOURCE COLUMN(S) | EXAMPLE VALUES | CANONICAL FIELD | EXISTS | MI SEMANTIC FIELD | EXISTS | IN REAL ERE TAPE | IN MI TAPE | DISCOVERABLE BY MI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 ASSET / LOAN CLASS (lifetime mortgage) | operator confirmation, not a column | "lifetime mortgage" | `erm_product_type` | YES | `erm_product_type` | YES | column present, **EMPTY on all 33 rows** | NO | NO |
| 2 ADVANCE / FACILITY STRUCTURE (lump sum, drawdown) | operator confirmation; OCC alias "Lump Sum or Drawdown" → `loan_sub_type` | "lump sum", "drawdown" | **CONTESTED** — `erm_sub_product_type` per applicability, `loan_sub_type` per OCC alias | both exist in `fields_registry` | `erm_sub_product_type` only | partial | neither populated with it | NO | NO |
| 3 LOAN / ADVANCE TYPE (Initial / Further Advance) | `further_advance_flag`, `further_advance_amount` | — | `further_advance_flag` | YES | `further_advance_flag` | YES | present, **EMPTY on all 33** | NO | NO |
| 4 PRODUCT CATEGORY / FAMILY | OCC alias "product category" → `erm_product_type` | — | `erm_product_type` | YES | `erm_product_type` | YES | EMPTY | NO | NO |
| 5 NAMED PRODUCT / PLAN (OMNI 500) | `erm_sub_product_type` as ERE populated it | **"Omni 400" (20), "Omni 300" (11), "Omni 200" (2)** | `erm_sub_product_type` | YES | `erm_sub_product_type` | YES | **POPULATED** | NO | NO |
| 6 SUB-PRODUCT / VARIANT | OCC alias "product variant" → `erm_sub_product_type` | — | `erm_sub_product_type` | YES | `erm_sub_product_type` | YES | holds named plans instead | NO | NO |
| (regulatory) ESMA product type | OCC alias "Product Type" → `product_type` | — | `product_type` (ESMA Annex 8 LESL19, `category: regulatory`, `core_canonical: false`) | YES | **ABSENT from mi_semantics** | NO | column absent | NO | NO |

`EXACT_LOSS_POINT` for concepts 1, 3, 4: **OCC operator confirmation never
completed** — the canonical column exists and is empty in the real ERE tape.
For concepts 2, 5, 6: **the MI tape is a different tape** — the synthetic multibook
source extracts (62 columns) carry no product column of any kind, so nothing was
dropped downstream; it was never there.

## PRODUCT_SEMANTIC_AMBIGUITY — three governed claimants, one concept

**"Lump sum / drawdown" has three different owners asserted in governed config:**

    1. config/mi/mi_equity_release_uk_applicability.yaml
         erm_sub_product_type  operator_question: "... (lump sum / drawdown) ..."
    2. config/system/aliases_analytics.yaml
         loan_sub_type:  aliases: ["Lump Sum or Drawdown"]
    3. config/asset/product_profiles.yaml
         match.product_type: [lifetime_mortgage, equity_release, drawdown,
                              lump_sum, rio, erm]

and ERE's own data puts a fourth thing — the named plan "Omni 400" — into
`erm_sub_product_type`.

Claimant 3 is the least problematic: it is a *profile-matching signal list* for
asset-class detection, not a per-loan field, and it deliberately mixes class and
structure tokens because it is matching free text. **Claimants 1 and 2 genuinely
conflict**, and `loan_sub_type` exists in `fields_registry` but in neither the
business semantics registry nor the MI semantics registry — so today it could not be
queried even if it were populated.

**DECISION REQUIRED:** name one governed owner for advance/facility structure and
one for named product. On the evidence the cleanest split is
`erm_product_type` = asset/loan class, `erm_sub_product_type` = advance structure
(per the applicability operator question), and a distinct field for the named plan —
because ERE's "Omni 400" is demonstrably a fifth concept that currently has nowhere
else to sit. **This report does not make that decision.**

Note also that no governed VALUE vocabulary exists: `load_governed_vocabulary()`
resolves "product" → `erm_product_type`, but "lump sum", "drawdown", "lifetime
mortgage", "initial advance" and "omni" all resolve to **nothing**. The
interpreter's rule 3a tells it not to assert a filter value where
`has_governed_values` is false; the recorded plans nonetheless carry
`{"concept": "erm_product_type", "comparator": "eq", "value": "drawdown"}`. That is
worth a separate look and is **not** adjudicated here.

## Explicit answers on each term

| TERM | WHAT TRAKT MEANS TODAY | WHICH FIELD SHOULD OWN IT | BASIS |
| --- | --- | --- | --- |
| "product type" | overloaded — ESMA regulatory field AND the ERM class AND a profile signal | `erm_product_type` for MI; `product_type` stays the ESMA Annex 8 LESL19 regulatory field | fields_registry `category: regulatory`, regime_mapping LESL19; OCC alias "Product Type" |
| "loan type" | no governed meaning | NOT_ESTABLISHED — `type_of_loan` exists nowhere in this repo | absent from all three registries and both tapes |
| "lifetime mortgage" | the asset/loan class | `erm_product_type` | applicability operator question names it as the example |
| "drawdown" | advance structure | **CONTESTED** `erm_sub_product_type` vs `loan_sub_type` | the two governed claimants above |
| "lump sum" | advance structure | **CONTESTED**, as "drawdown" | as above |
| "Initial Advance" | advance sequence, implied by the further-advance pair | `further_advance_flag` (absence = initial) | only governed field in that space; no positive "initial advance" field exists |
| "product category" | family above the variant | `erm_product_type` | OCC alias "product category" → `erm_product_type` |
| "product name" | the lender's named plan | no governed owner; ERE puts it in `erm_sub_product_type` | ERE data: "Omni 400 / 300 / 200" |
| "OMNI 500" | a named plan value | as "product name" | same |
| "sub-product" | variant below the category | `erm_sub_product_type` | OCC alias "product variant" |

## OCC-to-MI product lineage

| STAGE | ASSET CLASS | ADVANCE STRUCTURE | NAMED PLAN |
| --- | --- | --- | --- |
| raw ERE column | none — operator-confirmed | none — operator-confirmed | populated in `erm_sub_product_type` |
| OCC recognised concept | YES (`erm_product_type`, aliases incl. "product category") | YES (`loan_sub_type` alias "Lump Sum or Drawdown") | partially (`erm_sub_product_type` alias "product variant") |
| approved mapping / default | `configured_static` from product config | `configured_static` from product config | — |
| **consumed?** | **NO — confirmation never completed; column empty** | **NO** | n/a |
| canonical field | exists | exists (two candidates) | exists |
| canonical transform | no drop — nothing to carry | no drop | no drop |
| promoted MI tape | **column absent** (different, synthetic tape) | absent | absent |
| MI preparation | — | — | — |
| MI semantics | field present, description empty | `erm_sub_product_type` present; `loan_sub_type` absent | present |
| query interpretation | "product" → `erm_product_type` | no value vocabulary | no value vocabulary |
| deterministic execution | field not on tape → empty population | same | same |

**Diagnosis: OCC saw the field and had a mapping for it, but the approved
operator-confirmed default was never consumed — and separately, the tape MI queried
is not an OCC-produced ERE tape at all.**

---

# PART C — REGION CROSS-CHECK

    REGION_SOURCE_PRESENT             YES
      collateral_geography (London 18, East of England 15, …),
      geographic_region_collateral, geographic_region_obligor and both ITL3
      variants are all on the MI tape.
    REGION_HARMONISATION_CONFIGURED   YES
      config/mi/region_taxonomy.yaml (version 1, default_taxonomy uk_itl1)
      engine/region_taxonomy.py, FIELD_DETAIL / FIELD_REPORTING defined
    REGION_HARMONISATION_EXECUTED     NO
      no gate in synthetic_demo/run_multibook_pipeline.sh calls it; the only
      non-test callers of engine.region_taxonomy are MI-side READERS
      (mi_agent/region_resolution.py, mi_geography.py, region_basis.py,
      interpretation_v2/metadata.py)
    CANONICAL_REGION_FIELDS_PERSISTED NO
      neither canonical_region_detail nor canonical_region_reporting is on the
      tape, and neither appears as a target in config/system/fields_registry.yaml

    IF_NOT_EXACT_LOSS_POINT
      The onboarding harmonisation step between the raw source region columns and
      the promoted tape. engine/region_taxonomy.py's own header states the route:
      "… → canonical_region_detail + canonical_region_reporting (+ provenance)
      → runtime MI execution ← reads the persisted columns only."
      The producer of this tape never ran that step, so the runtime reads a
      column that was never written.

**Same structural cause as the product fields: a governed OCC/onboarding step that
the synthetic demo route does not perform.** Twelve of the 135 bind
`canonical_region_reporting`: Q04A/B/C, Q14A/B/C, Q16A/B/C, Q21A/B/C.

---

# PART D — CLIENT / FACILITY IDENTITY

    AUTHORITATIVE_CLIENT_ID    ere_funding_uk
    SOURCE_OF_AUTHORITY        config/client/config_client_ERE.yaml
                                 client.client_id: ere_funding_uk
                                 client.environment: production
    OCC_CLIENT_ID              ere_funding_uk   (same file)
    CANONICAL_DATA_CLIENT_ID   alderbridge_demo
                                 synthetic_demo/config/config_client_SYNTHETIC_MULTIBOOK.yaml
    MI_RUNTIME_CLIENT_ID       ERE
                                 derived as portfolio_id.split("/")[0] from the
                                 harness-supplied "ERE/2026-06-30"
    FACILITY_CONFIG_CLIENT_ID  ere_funding_uk
                                 config/risk/funding_facilities.yaml

## The correction this forces

The remediation analysis at `2243eafe` called the facility "bound to another
client_id" and filed it as CONFIG_SCOPE, implying the facility config was the thing
out of place. **That reading was the wrong way round.** The facility is bound to
`ere_funding_uk`, which is exactly the client id the authoritative ERE client config
declares. The facility configuration is correct.

**Three identities are in play and no two agree:**

    governed client config     ere_funding_uk     <- authoritative
    facility register          ere_funding_uk     <- agrees with authority
    the tape MI queried        alderbridge_demo   <- a demo client
    MI runtime at query time   ERE                <- agrees with neither

`mi_agent/borrowing_base/service.py` resolves a facility with
`load_facility(client_id)`. Handed `ERE`, it finds nothing, and the borrowing-base
capability refuses correctly: **"No funding facility is configured for this
portfolio."** The refusal is accurate. The input is not.

## Does OCC already have a mechanism that should make these one identity?

**Yes.** `config/client/config_client_{client_id}.yaml` is the naming contract, the
client config declares `client_id` once, and `operations_control.configuration.packages`
composes the asset pack underneath it — the file's own comment says a restated fact
"would be a second source able to drift from the pack".

**Why the current evidence bypasses it:** the certification never went through that
resolver. The portfolio identity `ERE/2026-06-30` was supplied as a harness argument
(`--portfolio-id`) and split on `/` to produce a client id, against a tape built by
the synthetic demo under a third client id. No step in that chain consults
`config_client_ERE.yaml`.

**No YAML is changed here.** Naming the authoritative identity is a governed
decision, and the evidence says it is already made — `ere_funding_uk` — and simply
not reaching the runtime.

---

# FINAL ANSWERS

    PIPELINE_DATA_ACTUALLY_ABSENT              NO
    PIPELINE_CAPABILITY_ACTUALLY_MISSING       NO
    PIPELINE_OCC_RERUN_WOULD_RESOLVE           NO
      The pipeline is already onboarded, discoverable and answering. The three
      remaining pipeline-adjacent failures are runtime perimeter decisions
      (filtered pipeline reads; `forecast` / `whole_book` bases outside the
      governed slice), not onboarding gaps.

    PRODUCT_SOURCE_DATA_ACTUALLY_ABSENT        PARTLY
      Named plan data EXISTS (erm_sub_product_type = "Omni 400/300/200" on the
      real ERE tape). Asset class, advance structure and further-advance are
      operator-confirmed statics, not source columns, and were never confirmed.
    PRODUCT_CANONICAL_COVERAGE_SUFFICIENT      PARTLY
      Fields exist for class, category and variant. Nothing unambiguously owns
      advance structure, and nothing at all owns the named plan.
    PRODUCT_OCC_RERUN_WOULD_RESOLVE            PARTLY
      A rerun would populate the operator-confirmed statics and carry the named
      plan through, but it cannot resolve which field owns lump sum / drawdown.
      That decision must be made first or the rerun will persist the ambiguity.
    PRODUCT_MI_CAPABILITY_MISSING              NO
      The MI side resolves "product" to erm_product_type and filters correctly;
      it has no data and no governed value vocabulary to filter against.

    REGION_OCC_RERUN_WOULD_RESOLVE             YES
      The taxonomy, the resolver and the source columns are all present and
      correct. Running the step writes the two columns the runtime already reads.

    FACILITY_IDENTITY_OCC_RERUN_WOULD_RESOLVE  PARTLY
      A rerun through the governed client config would produce a tape carrying
      ere_funding_uk. It would NOT change what the MI runtime is handed at query
      time, which is a separate seam: the portfolio identity supplied to the API.

# Which issues are because ERE was not onboarded through the current OCC path?

Evidence-backed, and only these:

1. **Region — YES.** `engine/region_taxonomy.py` exists, `config/mi/region_taxonomy.yaml`
   is configured, source region columns are present, and no gate in the producing
   script calls the harmonisation. 12 cases.
2. **Product asset class / category / further advance — YES.** The applicability
   config marks them `configured_static` from product config with an operator
   question; the real ERE tape carries the columns empty; the MI tape carries no
   product column at all. 18 cases touched.
3. **Client identity on the tape — YES.** The tape declares `alderbridge_demo`
   because it was built by the synthetic demo route, not from
   `config_client_ERE.yaml`.

Explicitly **NOT** caused by the OCC path, despite looking like it:

4. **Pipeline — NO.** Already onboarded and answering; 32 of 34 cases fully correct.
5. **MI runtime client id `ERE` — NO.** That comes from the query-time portfolio
   identity, not from onboarding, and an OCC rerun would not change it.
6. **Advance-structure ownership — NO.** A governed decision that has not been made.
   Two config files claim the concept and an OCC rerun would persist whichever one
   it happened to use.
