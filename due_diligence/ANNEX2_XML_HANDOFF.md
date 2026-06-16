# Onboarding → Annex 2 XML handoff contract — investigation & fix

## TL;DR
Feeding the promoted `18_central_lender_tape.csv` straight into
`engine.orchestrator.trakt_run` is **the wrong entry point**. `trakt_run`'s
Gate 1 (`semantic_alignment`) is built to map a *raw* lender tape's headers to
canonical fields. Run on an already-promoted canonical tape it re-canonicalises,
finds the Annex 2 canonical fields it expects are absent/renamed, and aborts with
a **misleading** “core canonical fields missing from client tape” error —
*before* the layers that actually supply those fields (asset defaults at gate-4
projection, ND/defaults at gate-4b normalization) ever run.

The new **handoff validation report** (`50_annex2_xml_handoff_validation.*`,
`engine/onboarding_agent/annex2_handoff_validation.py`) is the pre-flight gate:
it resolves every Annex 2 canonical field to its real delivery source and tells
you exactly what (if anything) blocks XML — instead of the gate-1 red herring.

## Answers to the investigation questions

1. **What is `18_central_lender_tape.csv`?** A *generic onboarding/MI canonical
   tape* (one row per loan, MI canonical vocabulary). It is an **intermediate
   consolidation artefact**, NOT an Annex 2 delivery tape.

2. **What should `trakt_run` run on?** Either *raw client source files* (full
   gate-1→5), or a *canonical_typed.csv* entering at gate-4. It should **not** be
   handed the promoted MI tape and asked to re-run raw-source gate-1.

3. **Is `trakt_run` wrongly re-running Gate 1 on a promoted tape?** **Yes.** That
   is the defect. The promoted tape is already canonical, so gate-1
   re-canonicalisation is both redundant and produces the misleading error.

4. **Who converts onboarding-canonical → Annex 2 delivery-canonical (ESMA
   codes)?** Gate-4 `engine/gate_4_projection/regime_projector.py`, driven by
   `fields_registry.yaml` `regime_mapping.ESMA_Annex2`, `enum_mapping.yaml`, and
   `--product-defaults`. The onboarding-side mirror is
   `engine/onboarding_agent/target_coverage.py` (28a). ND/defaults/enum
   normalisation is applied by gate-4b `annex2_delivery_normalizer.py`.

5. **Where should static/ND defaults be injected** (currency=GBP,
   interest_rate_type=Fixed, amortisation_type=Bullet, maturity_date=ND5)?
   At **Annex 2 projection/normalization**, not in generic promotion. They are
   regime/asset-specific delivery values and they **already exist** in
   `config/asset/product_defaults_ERM.yaml` (applied by the gate-4 projector via
   `--product-defaults`) and in `annex2_delivery_rules.yaml` (ND/defaults applied
   by the normalizer). The only problem is gate-1 aborts before they run.

6. **Are these alias issues?**
   - `postcode` → `property_post_code`: simple **alias** (different name, same
     thing) — surfaced; safe to add as a registry synonym / alias.
   - `Policy Completion Date` → `origination_date`: **source-column alias** —
     surfaced; add alias / approved mapping.
   - `current_outstanding_balance` → `current_principal_balance` (RREL30):
     **NOT a simple alias.** For ERM the outstanding balance includes rolled-up
     interest, while RREL30 wants principal. This is a **semantic mapping /
     derivation decision** and must be confirmed, never auto-applied.

## Why the gate-1 error is misleading (handoff validation on the real tape)

Running the new validator over the real-client promoted tape (107 Annex 2
codes) classifies delivery as:

| source_resolution | count | meaning |
|---|---:|---|
| regime/ND default | 48 | delivered from the regime rule's ND/default |
| asset default | 19 | delivered from `product_defaults_ERM.yaml` |
| promoted tape (direct) | 8 | present in the tape under the Annex 2 canonical name |
| pending regime rule | 28 | config backlog — no delivery rule yet |
| source mapping absent | 3 | required, no tape/default/ND |
| canonical alias mismatch | 1 | `current_principal_balance` ← `current_outstanding_balance` |

The four fields the gate-1 report called “missing core”
(`interest_rate_type`, `amortisation_type`, `exposure_currency_denomination`,
`maturity_date`) all resolve via **asset default** → **not blocking**. The real
work is: 1 alias decision (RREL30), the origination/post-code aliases, the 28
pending regime rules, and 3 genuinely absent sources.

## Correct end-to-end flow

```bash
# 1) Target-first onboarding (analysis + decisions) — produces 28a/28c/34/40/42-48
python -m engine.onboarding_agent.workflow --input-dir <client_input> \
  --client-name <NAME> --client-id <ID> --run-id <RUN> --mode regulatory_mi \
  --target-contract ESMA_Annex2 \
  --regime-config config/regime/annex2_delivery_rules.yaml \
  --asset-config config/asset/product_defaults_ERM.yaml \
  --registry config/system/fields_registry.yaml --aliases-dir config/system

# 2) Consolidate the source extracts into the MI central tape
python -m engine.onboarding_agent.cli promote --project-dir <PROJECT_DIR> \
  --input-dir <client_input> --client-id <ID> --run-id <RUN> \
  --registry config/system/fields_registry.yaml --enable-regulatory-reporting

# 3) HANDOFF GATE — validate the tape can actually deliver Annex 2 (NEW)
python -m engine.onboarding_agent.annex2_handoff_validation \
  --central-tape <PROJECT_DIR>/output/central/18_central_lender_tape.csv \
  --out-dir <PROJECT_DIR>
#   -> 50_annex2_xml_handoff_validation.{csv,json,md}; proceed only when xml_ready
#      (or you have accepted the alias/default resolutions).

# 4) Delivery → XML → XSD (only once the handoff gate is satisfied)
#    Do NOT feed the MI central tape to raw-source gate-1. Use the delivery view.
```

## Design decision

* **Implemented now (safe, testable):** the handoff validation gate (step 3).
  The system no longer treats `18_central_lender_tape.csv` as automatically
  XML-ready — readiness is `xml_ready` in the report.
* **Recommended next (Option A):** build an explicit
  `output/regulatory/annex2_delivery_tape.csv` that injects the asset/regime
  defaults and the accepted alias resolutions, then feed gate-4 projection
  directly (skipping raw-source gate-1). This converts the 19 asset-default and
  48 regime-default resolutions into concrete delivery columns and leaves only
  the genuine gaps. (Deferred here because it needs the real client tape to
  validate end-to-end.)
* **Option B (also acceptable):** add `--input-is-canonical` / `--skip-gate1` to
  `trakt_run` so a promoted canonical tape enters at gate-4. Note this alone does
  NOT fix the field-name/semantic gaps (e.g. RREL30) — the handoff gate still
  applies.
