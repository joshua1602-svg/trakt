# Annex 2 mapping-correction impact report (49)

Report-only before/after comparison for the 17 proposed corrected ESMA codes. Base = `claude/annex2-target-first-workflow-frnu7h`; After = `claude/annex2-mapping-corrections-proposed`. No rules changed by this report.

## XML placement independence

- **workbook_semantic affects XML placement: no** — placement is driven by `code_order` / `esma_model_structure` / ESMA code in gate 5 (`xml_builder_annex2`), not by `workbook_semantic`.
- **workbook_semantic affects value selection: yes (secondary only)** — used as a fuzzy-match synonym and display label; `projected_source_field` is the primary selector.
- **projected_source_field affects value selection: yes (primary)** — it is the `match_field` the coverage engine binds the source column to.

## XML availability

- XML value comparison not available in current workflow; impact inferred from projected source/value changes.

## Summary

- Codes compared: **17**
- Risk — low: **9**, medium: **1**, high: **7**, manual_review: **0**
- Projected-value changes (with populated source): **8**
- XML placement changes: **0**
- BEFORE bound to a WRONG source column (active data error fixed): **4**
- AFTER resolves to a valid ND/default fallback: **10**
- Appears in 28c after: **0**

## Related artefact deltas (before -> after)

- **40_summary**: annex2_semantic_mismatch_count 17->0, annex2_enum_constrained_count 16->21, annex2_enum_unconstrained_count 0->0, annex2_mapping_proposals_total 17->0
- **43_annex2_field_universe_reconciliation**: authoritative_field_count 107->107, registry_gap_count 0->0, missing_from_regime_rules_count 38->38
- **44_annex2_nd_eligibility_reconciliation**: regime_broader 0->0, divergent 5->3, regime_stricter 43->33, aligned None->None
- **45_annex2_config_alignment_review**: requires_manual_review_count 7->5
- **47_annex2_semantic_mapping_reconciliation**: semantic_mismatch 17->0, aligned 51->68
- **48_annex2_mapping_correction_proposals**: proposal_rows_total 17->0

## Per-code impact

### high (7)
- `RREL10` Resident [high]: source 'default' -> 'resident'; correct source value (now mapped to the right column). nd_allowed ND5 -> ND1; ND2; ND3; ND4. (xml_changed=True)
- `RREL13` Employment Status [high]: BEFORE bound to the wrong column 'Purpose'; source 'purpose' -> 'employment_status'; correct source value (rebound from a wrong column). nd_allowed ND5 -> ND1; ND2; ND3; ND4. (xml_changed=True)
- `RREL14` Credit Impaired Obligor [high]: BEFORE bound to the wrong column 'Litigation'; source 'litigation' -> 'credit_impaired_obligor'; correct source value (rebound from a wrong column). (xml_changed=True)
- `RREL26` Origination Channel [high]: source 'distribution_channel' -> 'origination_channel'; correct source value (now mapped to the right column). nd_allowed ND5 -> ND1; ND2; ND3; ND4; ND5. (xml_changed=True)
- `RREL72` Default Date [high]: source 'balloon_payment' -> 'default_date'; correct source value (now mapped to the right column). (xml_changed=True)
- `RREL75` Litigation [high]: BEFORE bound to the wrong column 'Credit Impaired Obligor'; source 'credit_impaired_obligor' -> 'litigation'; correct source value (rebound from a wrong column). (xml_changed=True)
- `RREL80` Original Lender Legal Entity Identifier [high]: BEFORE bound to the wrong column 'Default Amount'; source 'default_amount' -> 'original_lender_legal_entity_identifier'; valid ND/default fallback (correct source column absent). nd_allowed ND5 -> ND1; ND2; ND3; ND4; ND5. (xml_changed=True)

### medium (1)
- `RREL76` Recourse [medium]: source 'forbearance_reconcept' -> 'recourse'; default/config value changed. nd_allowed ND5 -> ND1; ND2; ND3; ND4; ND5. (xml_changed=True)

### low (9)
- `RREL17` Primary Income Type [low]: source 'secondary_income' -> 'primary_income_type'; valid ND/default fallback (unchanged output). (xml_changed=False)
- `RREL64` Cumulative Prepayments [low]: source 'prepayment_amount' -> 'cumulative_prepayments'; valid ND/default fallback (unchanged output). nd_allowed ND5 -> ND1; ND2; ND3; ND4; ND5. (xml_changed=False)
- `RREL66` Date Last In Arrears [low]: source 'repurchase_amount' -> 'date_last_in_arrears'; valid ND/default fallback (unchanged output). nd_allowed ND5 -> ND1; ND2; ND3; ND4; ND5. (xml_changed=False)
- `RREL70` Reason for Default or Foreclosure [low]: source 'interest_only_period' -> 'reason_for_default_or_foreclosure'; valid ND/default fallback (unchanged output). nd_allowed ND5 -> ND1; ND2; ND3; ND4; ND5. (xml_changed=False)
- `RREL78` Insurance Or Investment Provider [low]: source 'default_flag' -> 'insurance_or_investment_provider'; valid ND/default fallback (unchanged output). nd_allowed ND5 -> ND1; ND2; ND3; ND4; ND5. (xml_changed=False)
- `RREL79` Original Lender Name [low]: source 'date_in_default' -> 'original_lender_name'; valid ND/default fallback (unchanged output). nd_allowed ND5 -> ND1; ND2; ND3; ND4; ND5. (xml_changed=False)
- `RREL81` Original Lender Establishment Country [low]: source 'default_result_flag' -> 'original_lender_establishment_country'; valid ND/default fallback (unchanged output). nd_allowed ND5 -> ND1; ND2; ND3; ND4; ND5. (xml_changed=False)
- `RREC21` Sale Price [low]: source 'first_valuation_provider_name' -> 'sale_price'; valid ND/default fallback (unchanged output). (xml_changed=False)
- `RREC23` Guarantor Type [low]: source 'valuation_date_of_sale' -> 'guarantor_type'; valid ND/default fallback (unchanged output). nd_allowed ND1; ND2; ND3 -> ND1; ND2; ND3; ND4. (xml_changed=False)

