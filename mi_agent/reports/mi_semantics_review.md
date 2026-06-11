# MI Semantic Registry — Review of the v1 (rule-based) Generation

**Subject:** `mi_agent/mi_semantics_field_registry.yaml` as produced by the
original `build_mi_semantics_registry.py` (broad OR selection).

**Selection rule reviewed:**
`core_canonical == true` **OR** `layer in {performance, product}` **OR**
`category == analytics`.

## 1. Field count

**235 semantic fields** were generated (out of 471 canonical fields, ~50%).

This is far too large for a curated MI vocabulary. It effectively re-projects
half of the regulatory registry into the "semantic" layer rather than defining
the smaller set of concepts an analyst actually asks about in portfolio MI.

## 2. Role distribution

| Role        | Count |
|-------------|------:|
| metric      | 92    |
| dimension   | 40    |
| date        | 35    |
| flag        | 26    |
| identifier  | 18    |
| **unknown** | **24** |
| **Total**   | **235** |

Format distribution: string 82, currency 36, date 35, decimal 34, boolean 26,
integer 11, percent 11.

## 3. Unknown-role count

**24 fields (10%)** could not be classified and are tagged
`role: unknown` / "requires manual analytics classification", e.g.
`bank_internal_rating`, `borrower_basel_iii_segment`, `customer_segment`,
`fitch_public_rating_equivalent`, `moody_s_public_rating_equivalent`,
`s_p_public_rating_equivalent`, `interest_reset_period`, `payment_type`,
`prepayment_penalty`, `ranking`, `regular_interest_instalment`,
`reason_for_default_basel_ii_definition`. Carrying 24 unclassified fields into a
"semantic" layer undermines the layer's purpose.

## 4. Likely duplicate / overlapping concepts

The broad rule pulls in many fields that express the **same MI concept** at
different snapshots or granularities. An analyst means one thing by "balance",
"LTV", "valuation", or "arrears days"; the registry exposes several:

- **Balance time-buckets (8):** `outstanding_balance_period_1`,
  `_2_120`, `_121_599`, `_600` and their `_date` twins — arrears-ageing
  buckets, not a portfolio balance concept.
- **Arrears-days variants (3):** `number_of_days_in_arrears`,
  `number_of_days_in_interest_arrears`, `number_of_days_in_principal_arrears`.
- **LTV variants:** `current_loan_to_value`, `original_loan_to_value`,
  `indexed_loan_to_value`, `ltv_cap`.
- **Valuation variants:** `current_valuation_amount`, `original_valuation_amount`,
  `indexed_value`, plus method/basis/type/date satellites.
- **Borrower 1 vs Borrower 2 duplication (~8):** `borrower_2_age`,
  `borrower_2_gender`, `borrower_2_income`, `borrower_2_DOB`,
  `borrower_2_date_of_death`, `borrower_2_id`, … where an aggregate
  (e.g. `youngest_borrower_age`) is the MI-useful concept.
- **Rating-agency equivalents (6):** `fitch_/moody_s_/s_p_/dbrs_…_public_rating_equivalent`,
  `other_public_rating`, `bank_internal_rating[_prior_to_default]`.

## 5. Fields unlikely to be useful in management reporting

These are correct to hold in the canonical/regulatory registry but add noise to
an MI vocabulary:

- **Identifiers (18):** `loan_identifier`, `borrower_identifier`,
  `borrower_1_id`, `borrower_2_id`, `collateral_id`, `pool_identifier`,
  `servicer_identifier`, `group_company_identifier`, `new_obligor_identifier`, …
  (high-cardinality keys, not analysis dimensions).
- **LEIs / legal-entity codes:** `originator_legal_entity_identifier`,
  `corporate_guarantor_identifier`.
- **Industry classification codes:** `fitch_industry_code`,
  `moody_s_industry_code`, `s_p_industry_code`, `other_industry_code`.
- **Tax codes:** `obligor_tax_code`.
- **Swap / waterfall / breakage fields (6):**
  `net_periodic_payment_made_by_swap_provider`,
  `obligor_must_pay_breakage_on_swap`,
  `shortfall_in_payment_of_breakage_costs_on_swap`,
  `waterfall_a_b_pre_enforcement_scheduled_interest_payments`,
  `waterfall_a_b_pre_enforcement_scheduled_principal_payments`,
  `full_or_partial_termination_event_of_swap_for_current_period`.
- **Specialised securitisation-reporting / "at securitisation date" snapshots**
  and raw reporting-only liquidation codes.

## 6. Conclusion / direction for the redesign

The v1 generator optimises for *coverage* (a registry mirror) when MI needs
*curation* (a small, well-named, deduplicated vocabulary). Recommended redesign:

1. Replace the broad OR rule with an **explicit curated allowlist** of canonical
   fields (target **40–80**), chosen for portfolio-MI usefulness.
2. **Exclude** identifiers, LEIs, industry/tax codes, rating-agency equivalents,
   waterfall/swap fields, balance-period buckets, and duplicate
   borrower-2/guarantor fields.
3. Add **business metadata** (`business_name`, `business_description`,
   `synonyms`) to drive natural-language resolution.
4. Add an **`mi_tier`** (`core` / `extended`) so standard MI uses a tight set and
   power users can opt into the rest.
5. Pick one canonical field per concept (one balance, one current LTV, one
   valuation) and let bucketing/derivation cover the variants.
6. Drive `unknown`-role fields to **zero** by curating only classifiable fields.
