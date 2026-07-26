"""demo_platform.schemas — the two DIFFERENT raw source schemas.

Portfolio A arrives from the lender's own origination system. Portfolio B arrives
from the third-party servicer that administers the acquired back book. They
deliberately share no column names: that difference is the point of the
onboarding scene, and it is what the Onboarding Agent resolves into one canonical
model.

Each schema declares, in source-file column order:

    (source_header, model_field, note)

``model_field`` refers to the *generator's* internal model-space column, not a
canonical Trakt field — the mapping from source header to canonical field is the
job of Gate 1 plus the client's approved onboarding contract
(``demo_platform/aliases/<portfolio>/aliases_client_contract.yaml``).

Headers were chosen against what the repository already resolves: those that map
through the global alias files are marked ``global``; those that need the client
contract are marked ``contract``. See :func:`resolution_summary`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SourceColumn:
    """One column in a raw source extract."""

    header: str
    model_field: str
    resolution: str          # "global" | "contract"
    canonical_field: str     # the canonical field it resolves to
    note: str = ""


@dataclass(frozen=True)
class SourceSchema:
    """A complete raw source schema for one portfolio."""

    name: str
    system_of_record: str
    description: str
    columns: Tuple[SourceColumn, ...]

    @property
    def headers(self) -> List[str]:
        return [c.header for c in self.columns]

    def model_map(self) -> Dict[str, str]:
        """``{source_header: model_field}``."""
        return {c.header: c.model_field for c in self.columns}

    def contract_columns(self) -> List[SourceColumn]:
        return [c for c in self.columns if c.resolution == "contract"]


# --------------------------------------------------------------------------- #
# Portfolio A — lender origination system extract
# --------------------------------------------------------------------------- #
ORIGINATION_SCHEMA = SourceSchema(
    name="origination_extract",
    system_of_record="Alderbridge origination system (monthly month-end extract)",
    description=(
        "The lender's own origination and servicing system. Short, business-"
        "vocabulary headers; balances are capital plus capitalised roll-up "
        "interest; one row per loan on book at the reporting date."
    ),
    columns=(
        SourceColumn("Loan Reference", "loan_id", "contract", "loan_identifier",
                     "Primary loan key."),
        SourceColumn("Data Cut-Off Date", "reporting_date", "global", "data_cut_off_date",
                     "Month-end reporting date for the extract."),
        SourceColumn("Completion Date", "origination_date", "contract", "origination_date",
                     "Lifetime-mortgage completion date."),
        SourceColumn("Original Advance", "original_principal", "contract",
                     "original_principal_balance", "Initial lump-sum advance."),
        SourceColumn("Current Balance", "current_balance", "global",
                     "current_principal_balance",
                     "Capital plus capitalised roll-up interest at the cut-off date."),
        SourceColumn("Original Property Value", "original_valuation", "global",
                     "original_valuation_amount", "Valuation at completion."),
        SourceColumn("Property Value", "current_valuation", "contract",
                     "current_valuation_amount", "Current property valuation."),
        SourceColumn("Rate", "rate", "contract", "current_interest_rate",
                     "Fixed lifetime roll-up rate, annual percentage points."),
        SourceColumn("Customer Age", "borrower_age", "contract", "youngest_borrower_age",
                     "Age of the youngest borrower at the cut-off date."),
        SourceColumn("Region", "region", "contract", "collateral_geography",
                     "Readable ITL1 region label for the security property."),
        SourceColumn("Post Code", "postcode", "global", "postcode",
                     "Property postcode; drives ITL3 enrichment."),
        SourceColumn("Loan Status", "status", "global", "account_status",
                     "Active | Redeemed."),
        SourceColumn("Redemption Date", "redemption_date", "global", "redemption_date",
                     "Populated only for loans that redeemed in the period."),
        SourceColumn("Exposure Reference", "exposure_id", "contract",
                     "original_underlying_exposure_identifier",
                     "Exposure identifier for regulatory reporting (ESMA RREL2)."),
        SourceColumn("Borrower Reference", "obligor_id", "contract",
                     "original_obligor_identifier",
                     "Obligor identifier for regulatory reporting (ESMA RREL4)."),
        SourceColumn("Loan Purpose", "purpose", "global", "purpose",
                     "Use of proceeds. Required by the ESMA Annex 2 enum layer."),
        SourceColumn("Property Type", "property_type", "global", "property_type",
                     "Security property type."),
        SourceColumn("Interest Rate Type", "rate_type", "global", "interest_rate_type",
                     "Fixed for the life of the loan."),
        SourceColumn("Amortisation Type", "amortisation_type", "global",
                     "amortisation_type",
                     "Interest roll-up — no contractual repayment before redemption."),
        SourceColumn("Occupancy Type", "occupancy_type", "global", "occupancy_type",
                     "Owner-occupied."),
        SourceColumn("Exposure Currency Denomination", "currency", "global",
                     "exposure_currency_denomination", "GBP."),
        # --- Regulatory-reporting block (ESMA Annex 2 exposure level) -------
        SourceColumn("Long Stop Date", "maturity_date", "contract",
                     "maturity_date",
                     "Product long-stop: the date the youngest borrower reaches "
                     "110. A lifetime mortgage has no contractual repayment "
                     "date, so this is the reported maturity (RREL24)."),
        SourceColumn("Date Added To Pool", "pool_addition_date", "contract",
                     "pool_addition_date",
                     "Date the loan entered the reported pool (RREL7)."),
        SourceColumn("Approved Facility", "credit_limit", "contract",
                     "total_credit_limit",
                     "Initial advance plus the loan's reserve facility (RREL33)."),
        SourceColumn("Security Reference", "collateral_id", "contract",
                     "new_collateral_identifier",
                     "Current key for the security property (RREC4)."),
        SourceColumn("Security Reference At Completion", "collateral_id_original",
                     "contract", "original_collateral_identifier",
                     "The key as at completion. Identical on this book — the "
                     "collateral has never been re-keyed (RREC3)."),
        SourceColumn("Security Category", "collateral_type", "contract",
                     "collateral_type",
                     "Collateral class for the securitisation report (RREC5)."),
        SourceColumn("Valuation Basis", "valuation_method", "contract",
                     "current_valuation_method",
                     "How the current valuation was established (RREC14)."),
        SourceColumn("Origination Channel", "origination_channel", "global",
                     "origination_channel", "Direct on this book (RREL26)."),
        SourceColumn("Borrower Resident", "resident", "contract", "resident",
                     "Obligor resident in the collateral's country (RREL10)."),
        SourceColumn("Credit Impaired", "credit_impaired", "contract",
                     "credit_impaired_obligor",
                     "Obligor credit-impaired at origination (RREL14)."),
        SourceColumn("Litigation Flag", "litigation", "contract", "litigation",
                     "Loan subject to legal proceedings (RREL75)."),
        SourceColumn("Payment Due", "payment_due", "global", "payment_due",
                     "Amount contractually due this period — nil on a roll-up "
                     "product (RREL39)."),
        SourceColumn("Arrears Balance", "arrears_balance", "global",
                     "arrears_balance", "Arrears at the cut-off date (RREL67)."),
        SourceColumn("Days In Arrears", "days_in_arrears", "global",
                     "number_of_days_in_arrears",
                     "Days past due at the cut-off date (RREL68)."),
        SourceColumn("Default Amount", "default_amount", "global",
                     "default_amount", "Balance in default (RREL71)."),
        SourceColumn("Allocated Losses", "allocated_losses", "global",
                     "allocated_losses", "Losses allocated to date (RREL73)."),
        SourceColumn("Cumulative Recoveries", "cumulative_recoveries", "global",
                     "cumulative_recoveries", "Recoveries to date (RREL74)."),
        SourceColumn("Prepayment Fee", "prepayment_fee", "global",
                     "prepayment_fee",
                     "Early-repayment charge applied in the period (RREL61)."),
        SourceColumn("Synthetic Data Notice", "synthetic_notice", "global", "",
                     "Unmapped by design — the demonstration marker travels with "
                     "the file and is reported as an unmapped header."),
    ),
)


# --------------------------------------------------------------------------- #
# Portfolio B — third-party servicer account extract
# --------------------------------------------------------------------------- #
SERVICER_SCHEMA = SourceSchema(
    name="servicer_account_extract",
    system_of_record="Third-party servicer account extract (monthly, acquired book)",
    description=(
        "The servicer's own account vocabulary for the acquired back book: "
        "account-centric headers, an indexed valuation basis, a different status "
        "enumeration, and a separate further-advance (reserve drawdown) column."
    ),
    columns=(
        SourceColumn("Account Number", "loan_id", "contract", "loan_identifier",
                     "Servicer account number — the stable loan key."),
        SourceColumn("Reporting Cut-Off", "reporting_date", "contract",
                     "data_cut_off_date", "Servicer extract date (month-end)."),
        SourceColumn("Origination", "origination_date", "contract", "origination_date",
                     "Completion date under the servicer's naming."),
        SourceColumn("Initial Drawdown", "original_principal", "contract",
                     "original_principal_balance", "Initial advance at completion."),
        SourceColumn("Principal Outstanding", "current_balance", "contract",
                     "current_principal_balance",
                     "Capital plus capitalised roll-up interest."),
        SourceColumn("Valuation At Origination", "original_valuation", "contract",
                     "original_valuation_amount", "Vendor valuation at completion."),
        SourceColumn("Indexed Valuation", "current_valuation", "contract",
                     "current_valuation_amount",
                     "Servicer-indexed valuation at the reporting date."),
        SourceColumn("Current Interest Rate", "rate", "global", "current_interest_rate",
                     "Legacy fixed roll-up rate, annual percentage points."),
        SourceColumn("Youngest Borrower", "borrower_age", "contract",
                     "youngest_borrower_age", "Youngest life's age at the cut-off."),
        SourceColumn("Geographic Classification", "region", "contract",
                     "collateral_geography", "Readable ITL1 region label."),
        SourceColumn("Property Post Code", "postcode", "global", "property_post_code",
                     "Property postcode; drives ITL3 enrichment."),
        SourceColumn("Account State", "status", "contract", "account_status",
                     "Account status under the servicer's column naming."),
        SourceColumn("Closure Date", "redemption_date", "contract", "redemption_date",
                     "Populated only for accounts closed in the period."),
        SourceColumn("Further Advance This Period", "drawdown_amount", "global", "",
                     "Reserve-facility drawdown taken in the reporting month. "
                     "Reported for transparency; already included in Principal "
                     "Outstanding, so it is intentionally not mapped to a "
                     "canonical balance field."),
        SourceColumn("Exposure ID", "exposure_id", "contract",
                     "original_underlying_exposure_identifier",
                     "Servicer exposure identifier (ESMA RREL2)."),
        SourceColumn("Customer ID", "obligor_id", "contract",
                     "original_obligor_identifier",
                     "Servicer customer identifier (ESMA RREL4)."),
        SourceColumn("Use Of Proceeds", "purpose", "contract", "purpose",
                     "Use of proceeds under the servicer's naming. Required by the "
                     "ESMA Annex 2 enum layer."),
        SourceColumn("Dwelling Type", "property_type", "contract", "property_type",
                     "The servicer's naming for the property type — resolved by "
                     "the client contract, since 'dwelling' is not in the global "
                     "alias vocabulary."),
        SourceColumn("Rate Basis", "rate_type", "contract", "interest_rate_type",
                     "Fixed for the life of the loan. Referred for review at "
                     "onboarding — 'basis' is ambiguous — then confirmed."),
        SourceColumn("Repayment Profile", "amortisation_type", "contract",
                     "amortisation_type",
                     "Interest roll-up. The servicer's naming; referred for review "
                     "then confirmed against the canonical amortisation type."),
        SourceColumn("Occupancy", "occupancy_type", "contract", "occupancy_type",
                     "Owner-occupied. Referred for review, then confirmed."),
        SourceColumn("Currency", "currency", "global", "exposure_currency_denomination",
                     "GBP."),
        # --- Regulatory-reporting block, in the SERVICER's vocabulary --------
        SourceColumn("Contractual End Date", "maturity_date", "contract",
                     "maturity_date",
                     "The servicer's long-stop date for the account (RREL24)."),
        SourceColumn("Pool Entry Date", "pool_addition_date", "contract",
                     "pool_addition_date",
                     "For this book the acquisition completion date, or "
                     "origination where that is later (RREL7)."),
        SourceColumn("Total Facility Limit", "credit_limit", "contract",
                     "total_credit_limit",
                     "Initial drawdown plus the remaining reserve (RREL33)."),
        SourceColumn("Collateral Reference", "collateral_id", "contract",
                     "new_collateral_identifier",
                     "The servicer's current collateral key (RREC4)."),
        SourceColumn("Vendor Collateral Reference", "collateral_id_original",
                     "contract", "original_collateral_identifier",
                     "The vendor's key before the book transferred. Genuinely "
                     "different from the current key, because the servicer "
                     "re-keyed the collateral on acquisition (RREC3)."),
        SourceColumn("Collateral Class", "collateral_type", "contract",
                     "collateral_type", "Collateral class (RREC5)."),
        SourceColumn("Valuation Method", "valuation_method", "contract",
                     "current_valuation_method",
                     "How the indexed valuation was established (RREC14)."),
        SourceColumn("Introducer Type", "origination_channel", "contract",
                     "origination_channel",
                     "This book was written through intermediaries (RREL26)."),
        SourceColumn("Residency Status", "resident", "contract", "resident",
                     "Obligor residency (RREL10)."),
        SourceColumn("Impaired At Origination", "credit_impaired", "contract",
                     "credit_impaired_obligor",
                     "Credit-impaired obligor flag (RREL14)."),
        SourceColumn("Legal Proceedings", "litigation", "contract", "litigation",
                     "Account subject to legal proceedings (RREL75)."),
        SourceColumn("Amount Due", "payment_due", "contract", "payment_due",
                     "Contractually due this period — nil (RREL39)."),
        SourceColumn("Arrears Amount", "arrears_balance", "global",
                     "arrears_balance", "Arrears at the cut-off (RREL67)."),
        SourceColumn("Arrears Days", "days_in_arrears", "contract",
                     "number_of_days_in_arrears", "Days past due (RREL68)."),
        SourceColumn("Defaulted Balance", "default_amount", "contract",
                     "default_amount", "Balance in default (RREL71)."),
        SourceColumn("Losses Allocated", "allocated_losses", "global",
                     "allocated_losses", "Losses allocated to date (RREL73)."),
        SourceColumn("Recoveries To Date", "cumulative_recoveries", "contract",
                     "cumulative_recoveries", "Recoveries to date (RREL74)."),
        SourceColumn("Exit Fee Charged", "prepayment_fee", "contract",
                     "prepayment_fee",
                     "The servicer's naming for the early-repayment charge "
                     "(RREL61)."),
        SourceColumn("Synthetic Data Notice", "synthetic_notice", "global", "",
                     "Unmapped by design — the demonstration marker."),
    ),
)


# --------------------------------------------------------------------------- #
# Portfolio S — trustee deal-level extract for the sold securitisation
# --------------------------------------------------------------------------- #
#: The sponsor no longer owns SPV1, so its data does not come from the sponsor's own
#: systems at all. It arrives from the deal's trustee, in the trustee's deal vocabulary:
#: everything is a "deal asset" rather than a loan, the pool cut-off replaces the
#: reporting date, and the identifiers are the deal's, not the lender's. Three source
#: systems, three vocabularies — which is the point.
TRUSTEE_SCHEMA = SourceSchema(
    name="trustee_deal_extract",
    system_of_record="Deal trustee asset schedule (quarterly, sponsored securitisation)",
    description=(
        "The trustee's asset schedule for a securitisation the sponsor originated "
        "and sold. Deal-level vocabulary throughout: pool cut-off rather than "
        "reporting date, asset rather than loan, and the trustee's own identifiers."
    ),
    columns=(
        SourceColumn("Deal Asset ID", "loan_id", "contract", "loan_identifier",
                     "The trustee's asset key — the stable loan identifier."),
        SourceColumn("Pool Cut-Off", "reporting_date", "contract", "data_cut_off_date",
                     "The deal's pool cut-off date."),
        SourceColumn("Asset Origination Date", "origination_date", "contract",
                     "origination_date", "Completion date under the deal's naming."),
        SourceColumn("Original Balance", "original_principal", "contract",
                     "original_principal_balance", "Advance at completion."),
        SourceColumn("Current Asset Balance", "current_balance", "contract",
                     "current_principal_balance",
                     "Capital plus capitalised roll-up interest at the cut-off."),
        SourceColumn("Valuation At Origination Date", "original_valuation", "contract",
                     "original_valuation_amount", "Vendor valuation at completion."),
        SourceColumn("Latest Valuation", "current_valuation", "global",
                     "current_valuation_amount", "Latest valuation held by the trustee."),
        SourceColumn("Asset Interest Rate", "rate", "contract", "current_interest_rate",
                     "Fixed lifetime roll-up rate, annual percentage points."),
        SourceColumn("Youngest Life Age", "borrower_age", "contract",
                     "youngest_borrower_age", "Age of the youngest life at the cut-off."),
        SourceColumn("Region Of Security", "region", "contract", "collateral_geography",
                     "Readable ITL1 region label for the security property."),
        SourceColumn("Security Post Code", "postcode", "contract", "property_post_code",
                     "Property postcode; drives ITL3 enrichment."),
        SourceColumn("Asset Status", "status", "contract", "account_status",
                     "Asset status under the trustee's naming."),
        SourceColumn("Repurchase Or Redemption Date", "redemption_date", "contract",
                     "redemption_date", "Populated only for assets that left the pool."),
        SourceColumn("Underlying Exposure ID", "exposure_id", "global",
                     "original_underlying_exposure_identifier",
                     "The deal's exposure identifier (ESMA RREL2)."),
        SourceColumn("Obligor ID", "obligor_id", "contract",
                     "original_obligor_identifier",
                     "The deal's obligor identifier (ESMA RREL4)."),
        SourceColumn("Proceeds Purpose", "purpose", "contract", "purpose",
                     "Use of proceeds under the trustee's naming."),
        SourceColumn("Property Category", "property_type", "contract", "property_type",
                     "Dwelling form of the security (RREC9)."),
        SourceColumn("Rate Convention", "rate_type", "contract", "interest_rate_type",
                     "Fixed for the life of the asset."),
        SourceColumn("Amortisation Basis", "amortisation_type", "contract",
                     "amortisation_type", "Interest roll-up."),
        SourceColumn("Occupancy Basis", "occupancy_type", "contract", "occupancy_type",
                     "Owner-occupied."),
        SourceColumn("Denomination", "currency", "contract",
                     "exposure_currency_denomination", "GBP."),
        SourceColumn("Maturity Long Stop", "maturity_date", "contract", "maturity_date",
                     "Product long-stop for the asset (RREL24)."),
        SourceColumn("Pool Addition", "pool_addition_date", "contract",
                     "pool_addition_date",
                     "Date the asset entered the deal's pool (RREL7)."),
        SourceColumn("Facility Limit", "credit_limit", "contract", "total_credit_limit",
                     "Approved facility at completion (RREL33)."),
        SourceColumn("Collateral Key", "collateral_id", "contract",
                     "new_collateral_identifier",
                     "Current collateral key. Not 'Collateral ID' — that is itself a "
                     "canonical field name, which Gate 1 pins before any client "
                     "contract is consulted (RREC4)."),
        SourceColumn("Collateral ID At Closing", "collateral_id_original", "contract",
                     "original_collateral_identifier",
                     "Collateral key as at deal closing (RREC3)."),
        SourceColumn("Collateral Category", "collateral_type", "contract",
                     "collateral_type", "Collateral class (RREC5)."),
        SourceColumn("Valuation Approach", "valuation_method", "contract",
                     "current_valuation_method", "How the valuation was established."),
        SourceColumn("Introduction Route", "origination_channel", "contract",
                     "origination_channel", "How the asset was introduced (RREL26)."),
        SourceColumn("Obligor Residency", "resident", "contract", "resident",
                     "Obligor resident in the collateral's country (RREL10)."),
        SourceColumn("Impaired Flag", "credit_impaired", "contract",
                     "credit_impaired_obligor", "Credit-impaired obligor (RREL14)."),
        SourceColumn("Proceedings Flag", "litigation", "contract", "litigation",
                     "Asset subject to legal proceedings (RREL75)."),
        SourceColumn("Amount Falling Due", "payment_due", "contract", "payment_due",
                     "Contractually due this period — nil (RREL39)."),
        SourceColumn("Arrears Position", "arrears_balance", "contract",
                     "arrears_balance", "Arrears at the cut-off (RREL67)."),
        SourceColumn("Days Past Due", "days_in_arrears", "global",
                     "number_of_days_in_arrears", "Days past due (RREL68)."),
        SourceColumn("Defaulted Amount", "default_amount", "global",
                     "default_amount", "Balance in default (RREL71)."),
        SourceColumn("Loss Allocation", "allocated_losses", "contract",
                     "allocated_losses", "Losses allocated to date (RREL73)."),
        SourceColumn("Recovery Total", "cumulative_recoveries", "contract",
                     "cumulative_recoveries", "Recoveries to date (RREL74)."),
        SourceColumn("Early Redemption Charge", "prepayment_fee", "contract",
                     "prepayment_fee", "Early-repayment charge (RREL61)."),
        SourceColumn("Synthetic Data Notice", "synthetic_notice", "global", "",
                     "Unmapped by design — the demonstration marker."),
    ),
)


SCHEMAS: Dict[str, SourceSchema] = {
    ORIGINATION_SCHEMA.name: ORIGINATION_SCHEMA,
    SERVICER_SCHEMA.name: SERVICER_SCHEMA,
    TRUSTEE_SCHEMA.name: TRUSTEE_SCHEMA,
}


def schema_for(name: str) -> SourceSchema:
    try:
        return SCHEMAS[name]
    except KeyError as exc:  # pragma: no cover - configuration error
        raise KeyError(f"unknown source schema {name!r}") from exc


def shared_headers() -> List[str]:
    """Headers present in MORE THAN ONE schema.

    Used by a test that asserts the source schemas are genuinely different: only the
    deliberate synthetic marker may be shared across them.
    """
    seen: Dict[str, int] = {}
    for schema in SCHEMAS.values():
        for header in set(schema.headers):
            seen[header] = seen.get(header, 0) + 1
    return sorted(h for h, n in seen.items() if n > 1)


def resolution_summary() -> Dict[str, Dict[str, int]]:
    """Per-schema counts of headers resolved globally vs by the client contract."""
    out: Dict[str, Dict[str, int]] = {}
    for name, schema in SCHEMAS.items():
        out[name] = {
            "columns": len(schema.columns),
            "global": sum(1 for c in schema.columns if c.resolution == "global"),
            "contract": sum(1 for c in schema.columns if c.resolution == "contract"),
        }
    return out


def onboarding_mapping_rows(schema_name: str) -> List[Dict[str, Optional[str]]]:
    """The mapping table the onboarding scene shows for one portfolio."""
    schema = schema_for(schema_name)
    return [
        {
            "source_header": c.header,
            "canonical_field": c.canonical_field or None,
            "resolution": c.resolution,
            "note": c.note,
        }
        for c in schema.columns
    ]


def as_dict() -> Dict[str, object]:
    """Serialisable schema description for the demo manifest and film fixtures."""
    return {
        name: {
            "name": s.name,
            "system_of_record": s.system_of_record,
            "description": s.description,
            "headers": s.headers,
            "columns": [
                {
                    "header": c.header,
                    "canonical_field": c.canonical_field or None,
                    "resolution": c.resolution,
                    "note": c.note,
                }
                for c in s.columns
            ],
        }
        for name, s in SCHEMAS.items()
    }
