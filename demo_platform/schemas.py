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
        SourceColumn("Original Property Value", "original_valuation", "contract",
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
                     "Open | Redeemed."),
        SourceColumn("Closure Date", "redemption_date", "contract", "redemption_date",
                     "Populated only for accounts closed in the period."),
        SourceColumn("Further Advance This Period", "drawdown_amount", "global", "",
                     "Reserve-facility drawdown taken in the reporting month. "
                     "Reported for transparency; already included in Principal "
                     "Outstanding, so it is intentionally not mapped to a "
                     "canonical balance field."),
        SourceColumn("Security Type", "property_type", "global", "security_type",
                     "Security property type under the servicer's naming."),
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
        SourceColumn("Synthetic Data Notice", "synthetic_notice", "global", "",
                     "Unmapped by design — the demonstration marker."),
    ),
)


SCHEMAS: Dict[str, SourceSchema] = {
    ORIGINATION_SCHEMA.name: ORIGINATION_SCHEMA,
    SERVICER_SCHEMA.name: SERVICER_SCHEMA,
}


def schema_for(name: str) -> SourceSchema:
    try:
        return SCHEMAS[name]
    except KeyError as exc:  # pragma: no cover - configuration error
        raise KeyError(f"unknown source schema {name!r}") from exc


def shared_headers() -> List[str]:
    """Headers present in BOTH schemas.

    Used by a test that asserts the two source schemas are genuinely different:
    only the deliberate synthetic marker may be shared.
    """
    a = set(ORIGINATION_SCHEMA.headers)
    b = set(SERVICER_SCHEMA.headers)
    return sorted(a & b)


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
