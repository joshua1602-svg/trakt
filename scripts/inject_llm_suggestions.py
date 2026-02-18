"""
Directly inject LLM-tier mapping suggestions for fields that the
deterministic engine (Tiers 1-6) could not resolve.

Run from repo root:
    python scripts/inject_llm_suggestions.py
"""

import pandas as pd
from pathlib import Path

MAPPING_REPORT = Path("out/dummy_tape_mapping_report.csv")
OUTPUT_FILE    = Path("out/dummy_tape_mapping_report_llm.csv")

# ---------------------------------------------------------------------------
# LLM suggestions for the 54 unmapped headers
# Format: raw_header -> (canonical_field or None, confidence, reasoning)
# ---------------------------------------------------------------------------
LLM_SUGGESTIONS = {
    # --- Loan identifiers / admin ---
    "Base Policy Number": (
        "underlying_exposure_identifier", 0.72,
        "Secondary policy identifier — the base/parent policy reference, maps to the "
        "underlying exposure identifier distinct from the loan-level original_underlying_exposure_identifier."
    ),
    "Policy Completion Date": (
        None, 0.0,
        "Duplicate: 'Date Of Completion' is already mapped to origination_date. "
        "Suppressing to avoid duplicate canonical mapping."
    ),
    "Agreed Facility At Outset": (
        None, 0.0,
        "Duplicate: 'Total Loan Facility' is already mapped to total_credit_limit. "
        "Same concept; suppressing to avoid collision."
    ),
    "Loan Component Number": (
        None, 0.0,
        "Loan component/tranche number for drawdown products. "
        "No matching canonical field in the equity_release registry."
    ),
    "GIF": (
        None, 0.0,
        "Acronym ambiguous in context — could be Guaranteed Interest Feature or product-level flag. "
        "No canonical field match found."
    ),
    "Mortgage Application Date": (
        None, 0.0,
        "Pre-origination application date. No canonical field for application stage exists "
        "in this registry — registry tracks from origination onwards."
    ),
    "Initial Loan": (
        None, 0.0,
        "Duplicate: 'Original Loan Amount' is already mapped to original_principal_balance. "
        "Suppressing to avoid collision."
    ),
    "Total OSBalance": (
        "current_principal_balance", 0.65,
        "Total Outstanding Balance — the total balance outstanding at reporting date. "
        "current_outstanding_balance is already occupied; current_principal_balance "
        "(regulatory, principal component) is the closest available alternative, "
        "though it excludes rolled-up interest."
    ),

    # --- Drawdown / advances ---
    "Remaining Drawdown Facility": (
        "drawdown_facility", 0.87,
        "Remaining amount available to draw down on a drawdown equity release product. "
        "drawdown_facility specifically tracks this residual facility. "
        "Note: undrawn_facility is already mapped to 'Remaining Facility'."
    ),
    "Total Advances": (
        "cumulative_drawn_amount", 0.82,
        "Sum of all advances/drawdowns taken over the life of the loan. "
        "cumulative_drawn_amount is the canonical field for total accumulated drawdowns."
    ),
    "Date Of Last Advance": (
        "further_advance_date", 0.78,
        "Date the most recent advance/drawdown was made. "
        "further_advance_date records the date of the last advance event."
    ),

    # --- Balances / interest ---
    "Balance At Anniversary Date": (
        None, 0.0,
        "Balance recorded at the annual anniversary of the policy — no direct canonical "
        "field for anniversary-date snapshots exists in this registry."
    ),
    "Interest Accrued": (
        "deferred_interest", 0.65,
        "Outstanding accrued interest not yet capitalised. In equity release, this is "
        "typically held as deferred/rolled-up interest. accrued_interest_in_period and "
        "cumulative_accrued_interest are both already mapped, so deferred_interest is "
        "the next closest available field."
    ),
    "Added Charges": (
        "total_other_amounts_outstanding", 0.70,
        "Fees and charges added to the account balance. "
        "total_other_amounts_outstanding captures miscellaneous outstanding amounts "
        "beyond principal and interest."
    ),

    # --- Repayments ---
    "Date Of Latest Repayment": (
        "payment_date", 0.75,
        "Date of the most recent voluntary repayment. payment_date records the date "
        "a payment was received/applied to the account."
    ),
    "Amount Of Repayment": (
        None, 0.0,
        "Amount of a single repayment event. No canonical field for individual "
        "repayment amount without duplication (redemptions_received_in_period is "
        "already mapped and covers periodic totals)."
    ),
    "Repayment Type": (
        "payment_type", 0.82,
        "Classification of repayment (voluntary/mandatory/forced/partial). "
        "payment_type is the canonical field for repayment/payment classification."
    ),

    # --- Status / dates ---
    "Policy Status": (
        "account_status", 0.88,
        "Status of the equity release policy (active, redeemed, in default etc.). "
        "account_status is the canonical regulatory status field for loan accounts."
    ),
    "Date Notified Of Last Death or LTC": (
        None, 0.0,
        "Date lender was notified of borrower death or long-term care event — the "
        "trigger for equity release loan redemption. No specific canonical field "
        "for notification date exists in this registry."
    ),

    # --- Valuation (duplicates) ---
    "Valuation Date": (
        None, 0.0,
        "Valuation date for the original property value. original_valuation_date is "
        "already mapped via 'Original Valuation Date'. Suppressing duplicate."
    ),
    "Valuation Date 2": (
        None, 0.0,
        "Valuation date for the latest property value. current_valuation_date is "
        "already mapped via 'Latest Valuation Date'. Suppressing duplicate."
    ),

    # --- AVM ---
    "Latest AVMAmount": (
        "collateral_value", 0.78,
        "Automated Valuation Model estimate of current property value. "
        "collateral_value is the appropriate field for current collateral estimates "
        "(current_valuation_amount is occupied by 'Latest Property Value')."
    ),
    "Latest AVMDate": (
        None, 0.0,
        "Date of the latest AVM run. No AVM-specific date field in the registry; "
        "current_valuation_date is already occupied by the surveyor valuation date."
    ),
    "Date Of Indexed Value": (
        None, 0.0,
        "Date the indexed property value was calculated. No dedicated canonical field "
        "for indexed valuation date exists in this registry."
    ),
    "LTVLast AVM": (
        None, 0.0,
        "LTV calculated using the AVM valuation. No dedicated AVM-LTV field in registry; "
        "current_loan_to_value and indexed_loan_to_value are both already mapped."
    ),

    # --- Borrower / age ---
    "Youngest Age At Completion": (
        None, 0.0,
        "Age of youngest borrower at loan completion/origination. No canonical field "
        "for age-at-origination exists; youngest_borrower_age (already mapped) is the "
        "current age at reporting date."
    ),
    "Broker Originator ID": (
        "originator", 0.70,
        "Identifier for the originating broker/intermediary. "
        "originator is the canonical field for originator reference."
    ),
    "Product Company": (
        "originator_name", 0.72,
        "Name of the product/lending company. originator_name captures the entity "
        "that originated the loan, which in equity release is the product provider."
    ),

    # --- Property identifiers / location ---
    "Post Code 2": (
        "postcode", 0.72,
        "Second postcode field — likely a duplicate entry. postcode (analytics collateral) "
        "is the alternative to property_post_code which is already occupied."
    ),

    # --- Property characteristics ---
    "Style": (
        "property_type", 0.75,
        "Architectural style of the property (detached, semi-detached, terraced, flat). "
        "In UK property data, 'Style' maps to property_type which classifies property form."
    ),
    "Acerage": (
        "residential_area", 0.62,
        "Land acreage of the property (note: likely 'Acreage' with typo). "
        "residential_area is the closest canonical field for property land area, "
        "though different units (acres vs sq metres) — requires ETL conversion."
    ),
    "Ex Local Authority": (
        None, 0.0,
        "Flag indicating former council/local authority housing. "
        "No canonical field for this UK-specific property classification exists."
    ),
    "Age Restricted": (
        "age_restricted", 0.79,
        "Flag for age-restricted property (retirement housing, sheltered accommodation). "
        "age_restricted is a direct canonical field in the rre/equity_release registry."
    ),
    "Main Residence": (
        "main_residence", 0.79,
        "Flag indicating the property is the borrower's primary/main residence. "
        "main_residence is a canonical Y/N field in the rre/equity_release registry."
    ),
    "Tenure": (
        "tenure", 0.83,
        "Property tenure type (freehold, leasehold, commonhold). "
        "tenure is a direct canonical field in the rre/equity_release registry."
    ),
    "Lease Term At Completion": (
        None, 0.0,
        "Length of the lease at loan completion for leasehold properties. "
        "No canonical field for lease term at origination; property_leasehold_expiry "
        "is CRE-only and tracks expiry date not term."
    ),
    "Remaining Lease Term": (
        None, 0.0,
        "Remaining years on a leasehold property's lease. "
        "No canonical field for remaining lease term in the equity_release registry."
    ),
    "Ground Rent": (
        "ground_rent_payable", 0.92,
        "Annual ground rent payable on a leasehold property. "
        "ground_rent_payable is the direct canonical field for this amount."
    ),
    "Service Charge": (
        None, 0.0,
        "Annual service charge for leasehold/flat properties. "
        "No canonical service_charge field exists in the registry."
    ),
    "Ground Rent Pct": (
        None, 0.0,
        "Ground rent expressed as a percentage. No canonical field for ground rent "
        "percentage in the registry."
    ),
    "Ground Rent And Service Charge Pct": (
        None, 0.0,
        "Combined ground rent and service charge as a percentage. "
        "No canonical field for this combined metric."
    ),
    "Number Of Stories": (
        None, 0.0,
        "Number of storeys/floors in the property. floor_of_property captures "
        "which floor a flat is on, not the total number of storeys — no match."
    ),
    "Listed Grade": (
        "listed_grade", 0.92,
        "UK heritage listed building grade (Grade I, Grade II*, Grade II). "
        "listed_grade is a direct canonical Y/N field in the rre registry."
    ),
    "New Build": (
        "new_build", 0.95,
        "Flag indicating a newly-built property. new_build is the direct canonical "
        "Y/N field in the rre/equity_release registry."
    ),
    "Traditional Construction": (
        None, 0.0,
        "Flag for traditionally-constructed property. No canonical field for "
        "construction method classification exists in this registry."
    ),
    "Non Traditional Construction": (
        None, 0.0,
        "Flag for non-standard construction (timber frame, prefab, etc.). "
        "No canonical field for non-traditional construction in this registry."
    ),
    "Flat Roof %": (
        None, 0.0,
        "Percentage of the property roof that is flat. No canonical field for "
        "roof type or flat roof percentage exists in this registry."
    ),
    "Flood Risk": (
        None, 0.0,
        "Flood risk rating for the property location. No canonical environmental "
        "risk field exists in the current registry."
    ),
    "Costal Erosion Risk": (
        None, 0.0,
        "Coastal erosion risk for the property. No canonical field for coastal/erosion "
        "risk exists in the registry."
    ),
    "Adjacent To Commercial": (
        None, 0.0,
        "Flag for properties adjacent to commercial premises. No canonical field "
        "for this adjacency characteristic."
    ),
    "Annexe": (
        None, 0.0,
        "Flag indicating the property has an annexe. No canonical field for "
        "annexe presence in this registry."
    ),
    "Isolated Or Rural Location": (
        None, 0.0,
        "Flag for isolated/rural property location. geographic_region_classification "
        "is already mapped and captures region; no isolated/rural flag field exists."
    ),
    "Power Of Attorney": (
        None, 0.0,
        "Flag indicating Power of Attorney is held for a borrower. "
        "No canonical field for PoA status in this registry."
    ),
    "Ported Flag": (
        None, 0.0,
        "Flag indicating the loan was ported from a previous property. "
        "No canonical field for loan porting status in this registry."
    ),
}


def main():
    df = pd.read_csv(MAPPING_REPORT)

    rows = []
    for _, row in df.iterrows():
        h = row["raw_header"]
        if row["mapping_method"] != "unmapped":
            # Already mapped deterministically — keep as-is
            rows.append({
                "raw_header": h,
                "canonical_field": row["canonical_field"],
                "mapping_method": row["mapping_method"],
                "confidence": row["confidence"],
                "reasoning": "",
            })
        elif h in LLM_SUGGESTIONS:
            canon, conf, reasoning = LLM_SUGGESTIONS[h]
            rows.append({
                "raw_header": h,
                "canonical_field": canon if canon else "",
                "mapping_method": "llm" if canon else "unmapped",
                "confidence": conf,
                "reasoning": reasoning,
            })
        else:
            rows.append({
                "raw_header": h,
                "canonical_field": "",
                "mapping_method": "unmapped",
                "confidence": 0.0,
                "reasoning": "Not in LLM suggestion set",
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Written: {OUTPUT_FILE}")

    # Summary
    mapped_llm   = out_df[out_df["mapping_method"] == "llm"]
    mapped_det   = out_df[out_df["mapping_method"].isin(["exact","normalized","alias","token_set","fuzz_token_set","fuzz_ratio_norm"])]
    still_unmapped = out_df[out_df["mapping_method"] == "unmapped"]
    total = len(out_df)

    print(f"\n{'='*60}")
    print(f"COMBINED MAPPING SUMMARY  ({total} headers)")
    print(f"{'='*60}")
    print(f"  Deterministic (Tiers 1-6) : {len(mapped_det):>3}  ({len(mapped_det)/total*100:.1f}%)")
    print(f"  LLM Tier 7                : {len(mapped_llm):>3}  ({len(mapped_llm)/total*100:.1f}%)")
    print(f"  Still unmapped            : {len(still_unmapped):>3}  ({len(still_unmapped)/total*100:.1f}%)")
    print(f"{'='*60}")

    print("\n--- LLM-MAPPED FIELDS ---")
    for _, r in mapped_llm.iterrows():
        print(f"  {r['raw_header']:<45} → {r['canonical_field']}  (conf={r['confidence']:.2f})")

    print("\n--- STILL UNMAPPED ---")
    for _, r in still_unmapped.iterrows():
        print(f"  {r['raw_header']}")



# ---------------------------------------------------------------------------
# Batch 2 — LLM-only mapping (no deterministic pre-pass)
# ---------------------------------------------------------------------------
LLM_SUGGESTIONS_BATCH_2 = {
    "Cut-Off Date": (
        "data_cut_off_date", 0.95,
        "'Cut-Off Date' is the reporting period cut-off date — the standard name for "
        "data_cut_off_date across ESMA loan-level templates."
    ),
    "Account Number": (
        "original_underlying_exposure_identifier", 0.85,
        "The primary account/loan identifier supplied by the servicer. "
        "original_underlying_exposure_identifier is the canonical field for the "
        "originator-assigned loan reference number."
    ),
    "Origination Date": (
        "origination_date", 0.98,
        "Direct semantic match — the date the loan was originated/completed."
    ),
    "Loan Type": (
        "type_of_loan", 0.92,
        "Classification of the loan product (lifetime mortgage, home reversion, etc.). "
        "type_of_loan is the canonical field for loan-type classification."
    ),
    "Last Advance Date": (
        "further_advance_date", 0.88,
        "Date of the most recent advance/drawdown taken. further_advance_date records "
        "the date of the last drawdown event on a drawdown equity release product."
    ),
    "Current Balance": (
        "current_outstanding_balance", 0.90,
        "Outstanding loan balance at the reporting date. current_outstanding_balance "
        "is the canonical analytics field for the current total balance."
    ),
    "Drawdown Remaining": (
        "drawdown_facility", 0.88,
        "Remaining drawdown capacity available to the borrower. drawdown_facility "
        "captures the residual undrawn facility on a drawdown equity release product."
    ),
    "Interest Rate": (
        "current_interest_rate", 0.92,
        "The current contractual interest rate on the loan. current_interest_rate "
        "is the standard regulatory field for the live rate."
    ),
    "Index": (
        "current_interest_rate_index", 0.78,
        "The reference/benchmark rate index to which the loan rate is linked "
        "(e.g. Bank of England Base Rate, SONIA). current_interest_rate_index "
        "holds the index type as a list value."
    ),
    "Valuation": (
        "current_valuation_amount", 0.85,
        "'Valuation' without qualifier refers to the most recent property valuation. "
        "current_valuation_amount is the canonical field for the latest collateral value."
    ),
    "Protected Equity": (
        "protected_equity_percentage", 0.80,
        "The percentage of property equity ring-fenced from the loan balance. "
        "protected_equity_percentage holds this value. If the source column contains "
        "Y/N values, remap to protected_equity_flag instead."
    ),
    "iLTV": (
        "indexed_loan_to_value", 0.92,
        "'iLTV' is a standard industry abbreviation for Indexed LTV — the loan-to-value "
        "ratio calculated using the HPI-indexed property value rather than a fresh "
        "valuation. indexed_loan_to_value is the direct canonical field."
    ),
    "Region": (
        "geographic_region_classification", 0.90,
        "Geographic region of the property (e.g. South East, North West). "
        "geographic_region_classification is the canonical regulatory field for region."
    ),
    "Borrower 1 DOB": (
        "borrower_1_DOB", 0.98,
        "DOB = Date of Birth for the primary borrower. borrower_1_DOB is the direct "
        "canonical field."
    ),
    "Borrower 1 DOD": (
        "borrower_1_date_of_death", 0.95,
        "DOD = Date of Death for the primary borrower. borrower_1_date_of_death is "
        "the direct canonical field."
    ),
    "Borrower 1 Sex": (
        "borrower_1_gender", 0.92,
        "'Sex' and 'Gender' are synonymous in this context. borrower_1_gender is the "
        "canonical field for the primary borrower's gender."
    ),
    "Borrower 2 DOB": (
        "borrower_2_DOB", 0.98,
        "DOB = Date of Birth for the second borrower. borrower_2_DOB is the direct "
        "canonical field."
    ),
    "Borrower 2 DOD": (
        "borrower_2_date_of_death", 0.95,
        "DOD = Date of Death for the second borrower. borrower_2_date_of_death is "
        "the direct canonical field."
    ),
    "Borrower 2 Sex": (
        "borrower_2_gender", 0.92,
        "'Sex' and 'Gender' are synonymous in this context. borrower_2_gender is the "
        "canonical field for the second borrower's gender."
    ),
    "original loan amount": (
        "original_principal_balance", 0.95,
        "The original loan amount at origination. original_principal_balance is the "
        "canonical regulatory field for the initial loan principal."
    ),
    "last valuation": (
        "current_valuation_amount", 0.88,
        "'Last valuation' = the most recent property valuation amount. "
        "current_valuation_amount is the canonical field for the latest collateral value."
    ),
    "last valuation dt": (
        "current_valuation_date", 0.92,
        "'dt' is a standard abbreviation for date. 'last valuation dt' = date of the "
        "most recent valuation. current_valuation_date is the direct canonical field."
    ),
    "valuation type": (
        "current_valuation_method", 0.82,
        "The methodology used for the valuation (RICS, AVM, desktop, drive-by, etc.). "
        "current_valuation_method captures the valuation approach as a list value."
    ),
    "total accrued": (
        "cumulative_accrued_interest", 0.88,
        "Total interest accrued over the life of the loan. cumulative_accrued_interest "
        "is the canonical field for the running total of rolled-up/accrued interest."
    ),
    "original valuation": (
        "original_valuation_amount", 0.95,
        "The property valuation at the time of loan origination. "
        "original_valuation_amount is the direct canonical field."
    ),
    "orig ltv": (
        "original_loan_to_value", 0.95,
        "'orig ltv' = original loan-to-value ratio at origination. "
        "original_loan_to_value is the direct canonical field."
    ),
    "channel": (
        "origination_channel", 0.88,
        "The distribution channel through which the loan was originated "
        "(broker, direct, IFA, etc.). origination_channel is the canonical "
        "regulatory field for origination channel classification."
    ),
}

BATCH_2_OUTPUT = Path("out/batch2_llm_mapping_report.csv")


def map_standalone():
    """LLM-only mapping for Batch 2 — no prior deterministic pass."""
    rows = []
    for raw_header, (canon, conf, reasoning) in LLM_SUGGESTIONS_BATCH_2.items():
        rows.append({
            "raw_header": raw_header,
            "canonical_field": canon if canon else "",
            "mapping_method": "llm" if canon else "unmapped",
            "confidence": conf,
            "reasoning": reasoning,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(BATCH_2_OUTPUT, index=False)
    print(f"Written: {BATCH_2_OUTPUT}")

    mapped   = out_df[out_df["mapping_method"] == "llm"]
    unmapped = out_df[out_df["mapping_method"] == "unmapped"]
    total    = len(out_df)

    print(f"\n{'='*60}")
    print(f"BATCH 2 — LLM-ONLY MAPPING SUMMARY  ({total} headers)")
    print(f"{'='*60}")
    print(f"  LLM Tier 7 mapped  : {len(mapped):>3}  ({len(mapped)/total*100:.1f}%)")
    print(f"  Unmapped           : {len(unmapped):>3}  ({len(unmapped)/total*100:.1f}%)")
    print(f"{'='*60}")

    print("\n--- MAPPED ---")
    for _, r in mapped.iterrows():
        print(f"  {r['raw_header']:<35} → {r['canonical_field']:<45} (conf={r['confidence']:.2f})")

    if not unmapped.empty:
        print("\n--- UNMAPPED ---")
        for _, r in unmapped.iterrows():
            print(f"  {r['raw_header']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "batch2":
        map_standalone()
    else:
        main()
