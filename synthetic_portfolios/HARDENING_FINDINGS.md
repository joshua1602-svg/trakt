# Registry & Infrastructure Hardening — Non-Equity-Release Asset Classes

**What this is.** Two raw synthetic *funded loan tapes* — one **auto finance**
lender and one **unsecured consumer** lender — were built and run through the
Agentic onboarding model (`engine/onboarding_agent`) exactly as a new client's
tape would be. The goal was to measure how "hardened" the existing registries
and onboarding infrastructure are for asset classes the platform was not built
for. Trakt is an Equity Release Mortgage (ERM) / **ESMA Annex 2 (residential
real estate)** platform; auto finance is **ESMA Annex 5 (automobile)** and
unsecured consumer is **ESMA Annex 6 (consumer)** territory.

**No product/engine code was changed.** This exercise only *adds* the two
synthetic portfolios (`synthetic_portfolios/`) and characterisation tests
(`tests/test_synthetic_nonequity_portfolios.py`). Every claim below is asserted
by a passing test, so this document is an executable specification of current
behaviour.

> Reproduce: `python synthetic_portfolios/generate_portfolios.py` then
> `python -m pytest tests/test_synthetic_nonequity_portfolios.py -v`.

---

## Executive summary

The **base pipeline is genuinely asset-agnostic for core loan economics and
credit risk** — both never-seen tapes were ingested, classified, profiled and
mapped without a single code change, and the shared credit-risk block
(PD / LGD / EAD / IFRS 9 stage / internal grade / internal score) resolved for
both asset classes. That is the hardened part, and it is a real strength.

The **weaknesses are in asset identification, the collateral model, the regime
layer and the enum layer**, all of which are hard-wired to equity release / RRE:

| # | Finding | Severity | Evidence |
|---|---------|----------|----------|
| 1 | **Asset classification is fail-open.** Auto finance has *no* asset signal at all and is silently classified as **equity release with confidence 1.0**; the ERM product profile is applied to a motor-vehicle book. | **High** | `product_profile_resolution.applied=True`, `decision=detected_high_confidence`, evidence `asset_class=equity_release_mortgage` |
| 2 | **No product profile for auto or consumer.** Consumer is correctly detected (`consumer_loan`) but has no profile, so the engine falls back to generic (stricter) behaviour. | Medium | `decision=no_profile_generic_behaviour` |
| 3 | **Auto regime attributes are not wired to the auto annex.** Make/model/condition ARE ESMA Annex 5 regime fields and the defs already exist in the registry — but scoped to `equipment`/Annex 8, so they don't resolve for an auto tape. VIN/mileage are MI-only (not ESMA fields), so their non-mapping is expected. | **High** (auto) | registry scan; XSD `Manufacturer`/`Model`/`CurrentVehicleData1` |
| 4 | **Consumer-specific attributes have no home.** Secured/unsecured flag, dependants, residential status, affordability result, monthly instalment all **unmapped**. | Medium | 5 unmapped columns, all runs |
| 5 | **Registry & regime are RRE-centric.** No `auto`/`consumer` `portfolio_type`; no ESMA Annex 5/6 codes (though the shipped XSD defines Automobile/Consumer/Vehicle); regulatory-mode projection reaches for **RRE codes (RREL/RREC)** on a vehicle book. | **High** | registry scan + `07_gap_questions` |
| 6 | **Enum normalisation only covers ESMA Annex 2.** `collateral_type` enum is property-only (HOUSE/FLAT/OFFICE) — no vehicle code. | Medium | `enum_mapping.yaml` has one top key |
| 7 | **Borrower age leaks onto an ERM field.** A consumer/auto borrower's age maps to `borrower_1_age` (`portfolio_type: equity_release`). | Low | mapping trace |

---

## Coverage measured on the funded loan tape

Per-column mapping outcome from `05c_mapping_trace.csv` (the funded loan tape
only). `mapped` = resolved to a canonical field; `out_of_scope` = has a
canonical field but the field's category is excluded by the mode; `unmapped` =
**no canonical field exists** (the true coverage gap).

| Portfolio | Mode | Columns | mapped | out_of_scope | unmapped | review_status |
|-----------|------|--------:|-------:|-------------:|---------:|---------------|
| auto_finance | `mi_only` | 48 | 18 | 21 | 9 | blocked |
| auto_finance | `regulatory_mi` | 48 | 39 | 0 | **9** | blocked |
| unsecured_consumer | `mi_only` | 39 | 18 | 16 | 5 | blocked |
| unsecured_consumer | `regulatory_mi` | 39 | 34 | 0 | **5** | blocked |

The `out_of_scope` columns are *not* a coverage gap — they are arrears / default /
collateral fields with `category: regulatory` that `mi_only` deliberately
excludes and `regulatory_mi` pulls in (they all map under `regulatory_mi`). The
**`unmapped` count is the hard registry gap**: 9 columns for auto (all vehicle /
agreement structure), 5 for consumer (unsecured / affordability attributes).

---

## Finding 1 — Asset classification is fail-open (the headline risk)

`engine/onboarding_agent/onboarding_context.py` scores file/column/sample tokens
against a fixed asset-signal taxonomy:

```python
_ASSET_SIGNALS = {
    "equity_release_mortgage": ("equity release", "ere", "lifetime mortgage", ...),
    "residential_mortgage":    ("residential mortgage", "rmbs", "btl", ...),
    "consumer_loan":           ("personal loan", "consumer", "unsecured"),
    "sme_loan":                ("sme", "commercial loan", "business loan"),
}
...
asset_class = max(asset_scores, key=asset_scores.get) if any(asset_scores.values()) \
    else "equity_release_mortgage"     # <-- silent default
```

There is **no `auto` / `auto_finance` / `vehicle` / `motor` signal**. An auto
tape therefore fires *no* asset signal and hits the `else` branch, defaulting to
`equity_release_mortgage`. Because `product_type` also defaults to
`lifetime_mortgage`, the ERM product profile then matches with **confidence 1.0**
and is **applied** (`config/asset/product_profiles.yaml` ships only the
`equity_release_lifetime_mortgage` profile):

```
AUTO  regulatory_mi
  product_profile_resolution: applied=True, decision=detected_high_confidence,
    confidence=1.0, evidence=["asset_class=equity_release_mortgage",
                              "product_type=lifetime_mortgage"]
```

So a motor-vehicle HP/PCP book is silently treated as a lifetime mortgage — the
ERM relaxations (`maturity_date=not_applicable`, `amortisation_type` defaulted to
capitalising roll-up, `funded_status` derived, etc.) are applied to an
amortising, fixed-maturity vehicle loan. This is **fail-open**: the safe
behaviour for an unrecognised asset would be to withhold the profile and require
operator confirmation, not to assert equity release with full confidence.

Consumer behaves better only by luck of vocabulary: the tape carries the tokens
`consumer` / `unsecured` / `personal loan`, so `consumer_loan` is detected and
the ERM profile match drops to 0.25 (below the 0.55 confirm threshold) →
`no_profile_generic_behaviour`. It is correctly *not* mislabelled, but there is
still no consumer profile to apply.

**Why the LLM doesn't rescue this.** Asset detection here is deterministic
token-matching; the optional LLM context resolver is **off by default**
(`enable_context_resolver=False`, and the LLM tiers need `ANTHROPIC_API_KEY`), so
in these runs nothing ever semantically read the `Vehicle Make/Model/VIN`
columns. And even with the LLM enabled, `backstop_context()` only *accepts* an
LLM asset guess when a corroborating term exists in `_ASSET_SIGNALS`
(`asset_supported`). Because that set has no auto entry, an LLM "this is auto"
answer trips `conflict → downgraded_to_deterministic` — it reverts to
`equity_release_mortgage` and merely flags `needs_user_confirmation`. The LLM's
correct answer is vetoed by the same auto-less taxonomy.

**Recommended hardening:** add `auto_finance` and `consumer` (and their product
profiles) to the asset-signal taxonomy `_ASSET_SIGNALS` — this both fixes pure
determinism (the `Vehicle`/`Motor` tokens would match) and lets the LLM's answer
survive the backstop; and change the no-signal default from "assume equity
release" to "unknown → require confirmation" (fail-closed).

---

## Finding 3 & 4 — Collateral and asset-specific attributes have no canonical home

Two distinct things are worth separating here — a correction to an easy
overstatement:

1. **Make / model / vehicle condition ARE ESMA Annex 5 regime fields**, and the
   registry *already carries the definitions*: `manufacturer`, `model`,
   `year_of_manufacture_construction`, `new_or_used` all exist — but scoped to
   `portfolio_type: equipment` / `ESMA_Annex8` (equipment leasing). The shipped
   ESMA XSD confirms these are real regime elements (`Manufacturer`, `Model` =
   "Name of the car model", `VehicleConditionType1Code` = DEMO/NEWX/USED,
   `CurrentVehicleData1`). So this is a **wiring gap** (defs present, not mapped
   to an auto annex / portfolio_type), not a "field doesn't exist" gap.
2. **VIN and mileage are NOT ESMA regime fields** — ESMA templates avoid asset
   serial numbers / PII. They are MI / servicing / collateral-management
   attributes, so their being unmapped is *expected*, not a defect.

Result — the columns that are unmapped in every run:

- **Auto:** `Vehicle Make`, `Vehicle Model`, `Vehicle Registration Year`
  (regime attributes with defs present under Annex 8 but not wired to auto),
  and `Mileage`, `Vehicle Identification Number`, `Fuel Type`, `Agreement Type`
  (HP/PCP), `Balloon Payment` (PCP GFV) (MI / product-structure attributes).
- **Consumer:** `Secured / Unsecured`, `Number of Dependents`,
  `Residential Status`, `Affordability Assessment Result`, `Monthly Instalment`.

What *does* map is encouraging: `Collateral Type`, `New or Used`, original/current
valuation and original/current LTV all resolve — so the *value* side of auto
collateral is supported even though the *identity* of the asset is not. The
unsecured consumer tape has no collateral at all, which exercises (and passes)
the platform's tolerance for collateral-free assets.

---

## Finding 5 & 6 — Registry and regime layer are ESMA Annex 2 (RRE) centric

`portfolio_type` values in the registry: `common`, `cre`, `rre`, `sme`,
`equity_release`, `equipment`, `corporate`. There is **no `auto` or `consumer`**.
Regime code prefixes present: RREL/RREC (Annex 2), CREL/CREC (Annex 3), CRPL/CRPC
(Annex 4), LESL (Annex 8), ESTL (Annex 9). **ESMA Annex 5 (automobile, `AUTL`)
and Annex 6 (consumer, `CMRL`) are entirely absent.**

Running the auto tape in `regulatory_mi` mode makes the consequence concrete —
it blocks, and one of the blocking gates is:

```
[geography] Confirm ESMA Annex 2 UK geography policy: should RREL11/RREC6 use GBZZZ?
[core_field] core canonical field amortisation_type is missing or unmapped
[core_field] core canonical field originator_legal_entity_identifier is missing …
```

`RREL11` / `RREC6` are **residential-real-estate** loan/collateral codes — the
projector is trying to place a motor-vehicle book onto the RRE regime because
that is the only regime the registry knows. Enum normalisation is the same
story: `config/system/enum_mapping.yaml` has a single top-level key,
`ESMA_Annex2`, and its `collateral_type` enum maps HOUSE/FLAT/OFFICE/LAND — there
is no motor-vehicle collateral code.

**This is additive, not an engine rewrite.** The validation machinery is already
regime-generic: `regime_fields()` and `validate_regime_schema_and_mandatory()`
iterate *whatever* `regime_mapping` keys each field declares (that is how Annex
3/4/8/9 ride the same code path), `enum_synonyms.yaml` is regime-agnostic, and
`enum_mapping.yaml` is a regime-keyed dict where `ESMA_Annex5:` / `ESMA_Annex6:`
blocks would slot in with the identical shape. (Note enums lag further than the
registry: Annex 3/4 have field codes but no `enum_mapping` value-sets yet.)

**Recommended hardening:** extend `fields_registry.yaml` with `auto` / `consumer`
`portfolio_type` fields carrying ESMA Annex 5 (`AUTL`) / Annex 6 (`CMRL`) regime
mappings (re-using the existing `manufacturer`/`model`/`new_or_used` defs), add
the corresponding enum blocks, and register Annex 5/6 projectors — mirroring the
existing Annex 3/4/8/9 pattern.

---

## Finding 7 — Borrower age leaks onto an equity-release field

`Borrower Age` on both non-ERM tapes maps to `borrower_1_age`, which the registry
tags `portfolio_type: equity_release` (it exists because youngest-borrower age
drives the NNEG / mortality model in ERM). There is no asset-neutral borrower-age
field, so a consumer or auto borrower's age is silently recorded against an
equity-release-specific field. Low severity, but a symptom of the same
ERM-centric registry shape.

---

## What this says about "hardening"

- **Ingestion / classification / core-mapping / credit-risk: hardened.** The
  deterministic aligner, the domain-coverage model and the `common` field block
  (including the full PD/LGD/EAD/IFRS 9 risk block) carried two never-seen asset
  classes with zero code changes. The infrastructure *shape* generalises.
- **Asset identity, collateral model, regime & enum layers: not hardened.** They
  encode "this is an equity-release / RRE platform" as data and as a fail-open
  default. Presented with auto or consumer, the platform does not refuse or flag
  — it quietly assumes equity release (auto) or drops to generic handling
  (consumer), and cannot produce a compliant Annex 5/6 regulatory output.

The single highest-value fix is Finding 1: make unrecognised asset classes
**fail-closed** (require confirmation) instead of defaulting to equity release.
Everything else is additive registry/regime/enum extension along patterns the
codebase already supports.
