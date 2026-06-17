# Projection Blocker Field Review — RREL15, RREL24 & Operator/Config-Dependent Fields

**Status:** diagnostic only — no code or config changes made in this pass.
**Scope:** explain current projection-blocker classification behaviour, identify root
cause, and propose a remediation plan. Projection Agent is **not** built here and no
XML is produced.

---

## Executive summary

The latest run split 54 projection blockers into:

| subtype | count |
| --- | --- |
| `source_mapping_pending` | 29 |
| `operator_or_config_dependency` | 11 |
| `nd_or_default_rule_pending` | 10 |
| `materialised_projection_pending` | 2 |
| `not_materialised_projection_pending` | 2 |

The two `not_materialised_projection_pending` fields — **RREL15 `customer_type`** and
**RREL24 `maturity_date`** — are **misclassified**. Both are authoritative Annex 2
fields, both have a known ND envelope, and RREL24 already has an asset-config ND
default. They should have surfaced as `nd_or_default_rule_pending`.

**Root cause (single, structural):** the diagnostic (and the upstream materialisation
gate) only "knows" a field's ND/default eligibility if that field has a **full rule in
`config/regime/annex2_delivery_rules.yaml → field_rules`**. RREL15 and RREL24 are
**absent from `field_rules`**. The authoritative ND envelope actually lives in a
different file (`annex2_field_universe.yaml`, keys `nd1_4_allowed` / `nd5_allowed`) and
the ERM ND default lives in `product_defaults_ERM.yaml`, but **neither the
Transformation materialisation step nor the projection-blocker diagnostic consults
those two sources**. So ND-eligible fields without a hand-authored delivery rule fall
through to "no ND, no default, nothing related" → `not_materialised_projection_pending`.

This is not a typo and not an asset-policy bug — it is an **incomplete single-source-of-
truth + a diagnostic that reads only that incomplete source.**

No config-loading typo was found during this review.

---

## How the pipeline actually decides ND/default eligibility (verified)

```
annex2_field_universe.yaml      ← authoritative workbook ND envelope
  RREL15: nd1_4_allowed: true,  nd5_allowed: false      (ND1–4 allowed)   ✅
  RREL24: nd1_4_allowed: false, nd5_allowed: true       (ND5 only)        ✅

annex2_delivery_rules.yaml → field_rules   ← runtime "has a regime rule" gate
  RREL15: ABSENT                                          ❌
  RREL24: ABSENT                                          ❌

product_defaults_ERM.yaml → nd_defaults
  maturity_date: ND5   (annotated RREL24)                 ✅ present
  (customer_type: not present — correct; ND1–4 is asset-agnostic, not an ERM static)
```

Code path confirmed:

1. **Onboarding** (`engine/onboarding_agent/target_coverage.py:423-445`) marks any code
   **without a full rule in `field_rules`** as `pending_regime_rule`. RREL15/RREL24 hit
   this. (Onboarding *does* read the universe ND flags and asset `nd_defaults`, but only
   for the **reconciliation report** artefact `43_*`, not for the handoff classification.)
2. **Transformation** (`transformation_agent.py:528-533`) only materialises a value when
   `handoff_classification ∈ {default_downstream, nd_default_downstream,
   configured_static}`. A `pending_regime_rule`/`pending_projection_rule` field is
   **skipped**, so the ERM `maturity_date: ND5` default is **never applied**, and the
   field stays absent/blank. The field is emitted as `pending_projection_rule`.
3. **Validation** carries `pending_projection_rule → projection_required`,
   `blocking_for_projection = true`.
4. **`build_regime_index`** (`rules_adapter.py:66-81`) indexes **only `field_rules`** →
   no entry for `customer_type` / `maturity_date` → `nd_allowed = []`,
   `default_allowed = False`.
5. **Diagnostic** (`projection_blocker_diagnostics.py`) `_nd_or_default_allowed()` reads
   **only `regime_index`** (validation_agent.py:634 passes only `regime_index`; asset
   config and the universe are never handed in). Field is absent/blank, no ND, no related
   columns → `not_materialised_projection_pending`.

---

## RREL15 `customer_type` diagnosis

| # | Question | Answer |
| --- | --- | --- |
| 1 | In the authoritative universe? | **Yes** — `annex2_field_universe.yaml:182`. |
| 2 | Canonical field name | `customer_type` (`fields_registry.yaml:1084`, Annex2 code RREL15). |
| 3 | Configured as list/code/enum? | **Yes** — registry `format: list`, `allowed_values: customer_type`; enum content (CNEO/CEMO/CNRO/ENEO/EEMO/ENRO…) defined in the universe. |
| 4 | Where allowed values defined | `enum_synonyms.yaml:28` (`customer_type`), universe `content` block; registry references the enum name. |
| 5 | Where ND values defined | Universe: `nd1_4_allowed: true`, `nd5_allowed: false` → **ND1–ND4 allowed**. |
| 6 | Does the repo know ND1–ND4 are allowed? | **Yes — but only in the universe**, not in `field_rules`, and not in any source the diagnostic reads. |
| 7 | Why `not_materialised` not `nd_or_default`? | Because `build_regime_index` reads only `field_rules`, where RREL15 is absent → diagnostic sees `nd_allowed=[]`. The ND envelope in the universe is invisible to it. |
| 8 | Where is the issue? | Primarily **regime config completeness** (RREL15 missing from `field_rules`) **+ diagnostic/index data source** (doesn't read the universe). Secondary: transformation never materialises because the handoff class is `pending_regime_rule`. |
| 9 | Asset-agnostic config or operator review? | **Asset-agnostic, config/ND-driven.** ND1–ND4 ("not recorded"/"not collected") applies across asset classes. customer_type should be **enum-mapped from source when present**, else **ND-eligible (ND1–ND4)** — not an operator workbench item by default. |

## RREL24 `maturity_date` diagnosis

| # | Question | Answer |
| --- | --- | --- |
| 1 | In the authoritative universe? | **Yes** — `annex2_field_universe.yaml:323`. |
| 2 | Canonical field name | `maturity_date` (`fields_registry.yaml:2591`, Annex2 code RREL24). |
| 3 | Configured as a date field? | **Yes** — registry `format: date`. |
| 4 | Where ND values defined | Universe: `nd1_4_allowed: false`, `nd5_allowed: true` → **ND5 only**. Asset: `product_defaults_ERM.yaml:37 maturity_date: ND5`. |
| 5 | Does the repo know only ND5 is allowed? | **Yes** — universe flags it ND5-only; ERM asset config sets ND5. |
| 6 | Why `not_materialised` not `nd_or_default`? | Same structural cause: RREL24 absent from `field_rules`; the diagnostic never reads the universe ND flags nor the asset `nd_defaults`. The ERM ND5 default was **never materialised** because the handoff class was `pending_regime_rule` (transformation only materialises `*_downstream`/`configured_static`). |
| 7 | Where is the issue? | **Regime config completeness + transformation materialisation gating + diagnostic data source.** Asset config is actually *correct* here; it just isn't consulted. |
| 8 | Should ERM config set `maturity_date = ND5`? | **Yes — and it already does** (`product_defaults_ERM.yaml:37`). The problem is it isn't being applied/seen, not that it's missing. |
| 9 | How to keep this asset-specific? | The ND5 choice must remain in **asset config / asset policy** (as now), never in generic regime logic. Generic logic should only know "RREL24 permits ND5"; *whether to use it* is asset-driven. Traditional assets without `maturity_date: ND5` in their asset config must keep sourcing/validating the real date. |

### Proposed business rule — consistency check

> ERM/Lifetime Mortgage: `maturity_date` defaults to ND5 unless a valid contractual
> maturity date is supplied. Traditional amortising assets: sourced/derived, **not**
> defaulted to ND5 unless explicitly configured. Therefore RREL24 should be
> **asset-config driven, not hard-coded in generic regime logic.**

**Consistent with the architecture — and already the intended design.** The three-layer
model (regime envelope → asset class → client override) supports exactly this. ERM asset
config already encodes ND5. The only gap is plumbing: the envelope ("ND5 permitted")
isn't in the runtime gate, and the asset default isn't reaching materialisation for
`pending_regime_rule` fields. **No equity-release behaviour should move into regime
logic** — keep ND5 in `product_defaults_ERM.yaml`.

---

## Part B — selected operator/config-dependent fields (asset-agnostic)

These 8 fields are currently `operator_or_config_dependency`. That is largely an
**artefact of branch ordering**: the diagnostic checks "is this issue (or a peer issue)
`operator_required`/`config_required`?" *before* checking ND/default eligibility. Most of
these became `config_required` upstream (e.g. `source_absent` + `default_allowed` →
`config_required` in `classify_transformation_issue`), so they bucket as op/config
regardless of their true nature.

Universe ND envelope (verified): RREC9 ND5 · RREL27 ND1–4 · RREL9 ND5 · RREC17 ND1–4 ·
RREC1 **none** · RREC13 ND1–4+ND5 · RREL43 ND5. `field_rules` presence: RREC9, RREL27,
RREL40, RREL9, RREC17 **present**; RREC1, RREC13, RREL43 **absent**.

| Code | Field | Primary classification | Also | Notes |
| --- | --- | --- | --- | --- |
| RREC9 | property_type | **config_mapping_required** | operator_review_when_ambiguous | Enum-map day-1 when a property-type column exists; operator review only on conflicting candidate columns. ND5 fallback if truly absent. |
| RREL27 | purpose | **config_mapping_required** | operator_review_when_ambiguous | Enum-map from loan/product purpose; operator only on unmapped values or conflicting columns. ND1–4 eligible. |
| RREL40 | debt_to_income_ratio | **nd_defaultable_by_asset_policy** | source_required (traditional) | ERM: ND1 by asset policy (affordability/DTI not part of product) — already `ND1` in ERM config. Traditional mortgage/consumer: source-required or derivable. |
| RREL9 | redemption_date | **nd_defaultable_by_asset_policy** | projection_derivable | Only applies if redeemed/closed. Live loans → ND5. Better: projection-derive from account-status/closure-date rather than operator review. |
| RREC17 | original_valuation_amount | **source_required** | operator_review_when_ambiguous | Secured RRE asset → expect a valuation at origination. Multiple valuation columns → operator review. ND1–4 only as a last resort. |
| RREC1 | collateral_unique_identifier | **formal_client_onboarding_required** | source_required | **No ND permitted** (universe: none). Needs a stable identifier policy (derive from loan/property ID deterministically); operator review when Loan ID & Policy Number both exist. Never ND/defaulted. |
| RREC13 | current_valuation_amount | **source_required** | asset_config_defaultable / nd_defaultable | Map day-1 from latest/current valuation column. ND1–4/ND5 both permitted → ND fallback allowed if genuinely unavailable. |
| RREL43 | current_interest_rate | **source_required** | config_mapping_required | Fixed-rate ERM → map day-1 from loan interest rate. Operator review only when multiple plausible rate columns exist. ND5 eligible if absent. |

**Per-field default behaviours (asset-agnostic):**
- *Multiple candidate source columns* → do **not** auto-pick; emit
  `operator_review_when_ambiguous` (workbench item with candidate list).
- *Field absent* → consult asset policy: ND-eligible + asset opts in → ND default;
  else `source_required` (or `formal_client_onboarding_required` for identifiers).

---

## Part C — is the diagnostic logic too coarse / wrong?

The diagnostic under-uses available signals. Current vs. available:

| Signal | Used by diagnostic? | Where it actually lives |
| --- | --- | --- |
| `nd_allowed` | ✅ but only from `field_rules` | also `annex2_field_universe.yaml` (`nd1_4_allowed`/`nd5_allowed`) — **not read** |
| `default_allowed` / `default_value` | ✅ from `field_rules` only | also asset `defaults` / `nd_defaults` — **not read** |
| asset config (`product_defaults_*`) | ❌ never passed in | `product_defaults_ERM.yaml` |
| `coverage_status` / `handoff_classification` | ❌ not used in diagnostic | handoff contract / tx contract (available but ignored) |
| `transformation_status` | ❌ not used | tx contract |
| candidate source fields | ⚠️ heuristic only (name-token overlap on tape columns) | onboarding mapping proposals exist but aren't consulted |
| regime config | ✅ partial (`field_rules` only) | `annex2_delivery_rules.yaml` |

**Why RREL15 & RREL24 became `not_materialised_projection_pending`** — the precise
combination:

- ✅ **ND allowed values were not carried into the runtime regime index** (they exist in
  the *universe* file but `build_regime_index` reads only `field_rules`); **and**
- ✅ **asset config lacks/forwards no defaults to the diagnostic** (the diagnostic is
  never given asset config; RREL24's ERM ND5 default is invisible); **and**
- ✅ **regime config has no full rule for these codes**, so ND data was not accessible via
  the path the diagnostic uses.

It is **not** field-name/code misalignment (names line up across artefacts), and **not**
"present but ignored" within the source it does read. The data simply lives in files the
diagnostic and the regime index never open.

The subtype taxonomy itself is reasonable; the **inputs are too narrow**. Secondary
coarseness: the op/config branch fires before ND eligibility, so genuinely ND-eligible
fields that happen to carry a `config_required` issue are reported as
`operator_or_config_dependency` rather than the more useful `nd_or_default_rule_pending`.

---

## Part D — recommended remediation plan

### 1. Config-only changes
| Change | Priority | Risk | Owner | Why | Before Projection v1? |
| --- | --- | --- | --- | --- | --- |
| Add RREL15 & RREL24 (and other universe codes missing from `field_rules`) to `annex2_delivery_rules.yaml` with `nd_allowed` mirrored from the universe; RREL24 `default_allowed` left to asset layer | **high** | low | regime config | Makes ND envelope visible at runtime; fixes the 2 misclassifications directly | **Yes** |
| Keep ERM `maturity_date: ND5` in `product_defaults_ERM.yaml` (no change); confirm traditional asset configs do **not** set it | high | low | asset config | Preserves asset-specific ND5; prevents traditional assets inheriting it | Yes |
| Audit `field_rules` vs. universe for completeness (one-off coverage diff) | medium | low | regime config | Prevents the next "absent-from-field_rules" surprise | Recommended |

### 2. Diagnostic classification fixes
| Change | Priority | Risk | Owner | Why | Before Projection v1? |
| --- | --- | --- | --- | --- | --- |
| Make ND/default eligibility read the **universe** (`nd1_4_allowed`/`nd5_allowed`) and **asset config** (`defaults`/`nd_defaults`), not just `field_rules` | **high** | low | `projection_blocker_diagnostics.py` + `validation_agent.py` (pass asset cfg + universe in) | Root-cause fix; correct subtype even if `field_rules` is incomplete | **Yes** |
| Re-order/refine so ND-eligibility is considered even when an issue is `config_required` (or add a combined `nd_or_default_rule_pending (config-owned)` signal) | medium | low | diagnostics | Stops ND-eligible fields hiding under `operator_or_config_dependency` | Recommended |
| Use `handoff_classification` / `transformation_status` as inputs instead of re-deriving from the tape only | medium | medium | diagnostics | Higher-fidelity subtypes; fewer name-token false positives in `source_mapping_pending` | Optional |

### 3. Transformation Agent changes
| Change | Priority | Risk | Owner | Why | Before Projection v1? |
| --- | --- | --- | --- | --- | --- |
| Allow ND/default materialisation for ND-eligible fields driven by **universe envelope + asset policy**, even when the handoff class is `pending_regime_rule` (i.e. don't let an incomplete `field_rules` suppress a valid asset ND default) | **high** | medium | `transformation_agent.py` | This is why ERM `maturity_date: ND5` never landed; fixes it at the correct layer | **Yes** |
| Record the chosen ND/default **source** (universe vs asset vs regime) in the field contract for auditability | medium | low | transformation | Traceability for operators/projection | Recommended |

### 4. Validation Agent changes
| Change | Priority | Risk | Owner | Why | Before Projection v1? |
| --- | --- | --- | --- | --- | --- |
| Pass asset config + field universe into the diagnostic call (currently only `regime_index`) | **high** | low | `validation_agent.py:634` | Enables the diagnostic fix in §2 | **Yes** |
| Keep readiness booleans conservative (no auto-resolution) | high | low | validation | Diagnostics must remain advisory | Yes (already true) |

### 5. Projection Agent requirements (future PR — not now)
| Requirement | Priority | Risk | Owner | Why |
| --- | --- | --- | --- | --- |
| Consume the refined subtypes to drive ND/default emission vs. derivation vs. operator escalation | high | medium | projection | The diagnostic exists to feed projection decisions |
| Honour asset-policy ND (e.g. ERM ND5 maturity, ND1 DTI) and derive redemption_date from status | high | medium | projection + asset policy | Asset-specific behaviour stays in asset layer |

### 6. Operator workbench requirements
| Requirement | Priority | Risk | Owner | Why |
| --- | --- | --- | --- | --- |
| Surface `operator_review_when_ambiguous` items (multiple candidate columns) with the candidate list — e.g. RREC1 Loan ID vs Policy Number, RREC17/RREC13 multiple valuations, RREL43 multiple rate columns | medium | low | onboarding/workbench | Only truly ambiguous fields reach a human |
| Stable-identifier policy for RREC1 (deterministic derivation, no ND) | medium | low | onboarding policy | RREC1 permits no ND; must be a real identifier |

---

## Suggested specific changes (NOT implemented — for approval)

1. **`annex2_delivery_rules.yaml`** — add minimal rules:
   ```yaml
   RREL15:
     esma_code: RREL15
     projected_source_field: customer_type
     mandatory: true
     enforce_presence: true
     nd_allowed: [ND1, ND2, ND3, ND4]   # mirror universe nd1_4_allowed
   RREL24:
     esma_code: RREL24
     projected_source_field: maturity_date
     mandatory: true
     enforce_presence: true
     nd_allowed: [ND5]                   # mirror universe nd5_allowed; default stays in asset config
   ```
2. **`projection_blocker_diagnostics.py`** — extend `_nd_or_default_allowed()` to also
   consult the universe (`nd1_4_allowed`/`nd5_allowed`) and asset `defaults`/`nd_defaults`;
   thread those through `classify_projection_blockers(...)`.
3. **`validation_agent.py:634`** — pass `asset_cfg` and the loaded field universe into the
   diagnostic.
4. **`transformation_agent.py:528-537`** — permit asset/universe-driven ND materialisation
   for ND-eligible fields not gated behind a complete `field_rules` entry.

**Constraints honoured:** no Projection Agent, no XML, no equity-release behaviour in
generic regime logic (ND5 stays in asset config), traditional assets keep
sourcing/deriving/validating `maturity_date` normally.
