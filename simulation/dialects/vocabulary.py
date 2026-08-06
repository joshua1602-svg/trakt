"""simulation.dialects.vocabulary — the lender vocabularies the dialects write in.

One table, three vocabularies:

* ``canonical`` — the header IS the canonical field name, which Gate 1 resolves
  at Tier 1 (exact) with no alias involved;
* ``lender`` — a UK lender's own origination-system vocabulary;
* ``locale`` — a continental servicer's vocabulary, used together with day-first
  dates, comma decimals and different enum synonyms.

Both non-canonical vocabularies are resolved through an APPROVED ONBOARDING
CONTRACT — a per-asset-class alias overlay under ``simulation/aliases/`` layered
at Gate 1 with ``--extra-aliases-dir``. That is the production mechanism for a
client's own header vocabulary; the global alias library is deliberately left
untouched.

``tests/test_simulation_dialects.py`` asserts that every header emitted here is
covered by the committed contract (or is an exact canonical name), so a dialect
cannot quietly rely on a fuzzy match.
"""

from __future__ import annotations

from typing import Dict, Tuple

from ..models import ASSET_BRIDGE, ASSET_EQUITY_RELEASE, ASSET_FINANCE

#: ``canonical_field -> (lender header, locale header)``.
#: Shared by every asset class.
COMMON_VOCABULARY: Dict[str, Tuple[str, str]] = {
    "loan_identifier": ("Loan Reference", "Numero Prestito"),
    "original_underlying_exposure_identifier": ("Exposure Reference", "Riferimento Esposizione"),
    "new_underlying_exposure_identifier": ("Current Exposure Reference",
                                           "Riferimento Esposizione Corrente"),
    "underlying_exposure_identifier": ("Exposure Key", "Chiave Esposizione"),
    "original_obligor_identifier": ("Borrower Reference", "Codice Debitore"),
    "new_obligor_identifier": ("Current Borrower Reference",
                               "Codice Debitore Corrente"),
    "data_cut_off_date": ("Data Cut-Off Date", "Data Riferimento"),
    "origination_date": ("Completion Date", "Data Erogazione"),
    "pool_addition_date": ("Date Added To Pool", "Data Ingresso Pool"),
    "original_principal_balance": ("Original Advance", "Importo Originario"),
    "current_principal_balance": ("Principal Outstanding", "Capitale Residuo"),
    "current_outstanding_balance": ("Current Balance", "Saldo Corrente"),
    "original_valuation_amount": ("Original Security Value", "Valore Iniziale Garanzia"),
    "current_valuation_amount": ("Latest Security Value", "Valore Attuale Garanzia"),
    "current_valuation_method": ("Valuation Basis", "Metodo Valutazione"),
    "current_interest_rate": ("Rate", "Tasso Corrente"),
    "interest_rate_type": ("Interest Rate Type", "Tipo Tasso"),
    "amortisation_type": ("Amortisation Type", "Tipo Ammortamento"),
    "account_status": ("Loan Status", "Stato Rapporto"),
    "arrears_balance": ("Arrears Balance", "Importo Arretrato"),
    "number_of_days_in_arrears": ("Days In Arrears", "Giorni Arretrato"),
    "default_amount": ("Default Amount", "Importo Default"),
    "default_date": ("Default Date", "Data Default"),
    "allocated_losses": ("Allocated Losses", "Perdite Allocate"),
    "cumulative_recoveries": ("Cumulative Recoveries", "Recuperi Cumulati"),
    "redemption_date": ("Redemption Date", "Data Estinzione"),
    "maturity_date": ("Contractual End Date", "Data Scadenza"),
    "original_term": ("Original Term Months", "Durata Originaria Mesi"),
    "total_credit_limit": ("Approved Facility", "Limite Fido"),
    "payment_due": ("Payment Due", "Rata Dovuta"),
    "exposure_currency_denomination": ("Currency", "Divisa"),
    "collateral_geography": ("Region", "Regione Garanzia"),
    "postcode": ("Post Code", "Codice Postale"),
    "purpose": ("Loan Purpose", "Finalita"),
    "originator_name": ("Originating Lender", "Ente Erogante"),
    "seller_name": ("Seller", "Cedente"),
    "servicer_name": ("Servicer", "Gestore"),
    "origination_channel": ("Origination Channel", "Canale Origine"),
    "resident": ("Borrower Resident", "Residenza Debitore"),
    "credit_impaired_obligor": ("Credit Impaired", "Debitore Deteriorato"),
    "litigation": ("Litigation Flag", "Contenzioso"),
    "new_collateral_identifier": ("Security Reference", "Riferimento Garanzia"),
    "original_collateral_identifier": ("Security Reference At Completion",
                                       "Riferimento Garanzia Iniziale"),
    "collateral_unique_identifier": ("Security Key", "Chiave Garanzia"),
    "collateral_currency": ("Security Currency", "Divisa Garanzia"),
    "collateral_type": ("Security Category", "Categoria Garanzia"),
    "prepayment_fee": ("Early Repayment Charge", "Penale Estinzione Anticipata"),
}

_EQUITY_RELEASE_VOCABULARY: Dict[str, Tuple[str, str]] = {
    "erm_product_type": ("Lifetime Product", "Prodotto Vitalizio"),
    "youngest_borrower_age": ("Customer Age", "Eta Cliente"),
    "negative_equity_guarantee": ("NNEG Applies", "Garanzia Equity Negativo"),
    "further_advance_amount": ("Further Advance Amount", "Erogazione Aggiuntiva"),
    "further_advance_flag": ("Further Advance Taken", "Flag Erogazione Aggiuntiva"),
    "loan_redemption_flag": ("Redeemed Flag", "Flag Estinzione"),
    "property_type": ("Property Type", "Tipo Immobile"),
    "occupancy_type": ("Occupancy Type", "Tipo Occupazione"),
    "current_loan_to_value": ("Current LTV", "LTV Corrente"),
    "original_loan_to_value": ("Original LTV", "LTV Originario"),
}

_BRIDGE_VOCABULARY: Dict[str, Tuple[str, str]] = {
    "charge_type": ("Charge Rank", "Rango Ipoteca"),
    "lien": ("Lien Position", "Posizione Privilegio"),
    "property_type": ("Property Type", "Tipo Immobile"),
    "occupancy_type": ("Occupancy Type", "Tipo Occupazione"),
    "loan_sub_type": ("Facility Type", "Tipo Linea"),
    "product_type": ("Product Class", "Classe Prodotto"),
    "scheduled_interest_payment_frequency": ("Interest Payment Frequency",
                                             "Frequenza Interessi"),
    "current_loan_to_value": ("Current LTV", "LTV Corrente"),
    "original_loan_to_value": ("Day One LTV", "LTV Originario"),
    "prepayment_terms_description": ("Exit Notes", "Note Uscita"),
}

_ASSET_FINANCE_VOCABULARY: Dict[str, Tuple[str, str]] = {
    "product_type": ("Product Class", "Classe Prodotto"),
    "loan_sub_type": ("Agreement Type", "Tipo Contratto"),
    "manufacturer": ("Asset Manufacturer", "Costruttore"),
    "model": ("Asset Model", "Modello"),
    "number_of_leased_objects": ("Units Financed", "Numero Beni"),
    "year_of_manufacture_construction": ("Year Of Manufacture", "Anno Costruzione"),
    "original_residual_value_of_asset": ("Original Residual Value",
                                         "Valore Residuo Iniziale"),
    "current_residual_value_of_asset": ("Current Residual Value",
                                        "Valore Residuo Attuale"),
    "date_of_lease_expiration": ("Agreement Expiry Date", "Data Fine Contratto"),
    "balloon_amount": ("Balloon Payment", "Maxi Rata Finale"),
    "nace_industry_code": ("Lessee Industry", "Settore Locatario"),
    "scheduled_principal_payment_frequency": ("Instalment Frequency",
                                              "Frequenza Rata"),
    "scheduled_interest_payment_frequency": ("Interest Payment Frequency",
                                             "Frequenza Interessi"),
    "current_loan_to_value": ("Advance To Value", "Rapporto Finanziamento Valore"),
    "original_loan_to_value": ("Original Advance To Value",
                               "Rapporto Finanziamento Iniziale"),
    "prepayment_terms_description": ("Agreement Notes", "Note Contratto"),
}

_BY_ASSET: Dict[str, Dict[str, Tuple[str, str]]] = {
    ASSET_EQUITY_RELEASE: _EQUITY_RELEASE_VOCABULARY,
    ASSET_BRIDGE: _BRIDGE_VOCABULARY,
    ASSET_FINANCE: _ASSET_FINANCE_VOCABULARY,
}

#: Columns a lender supplies that carry no canonical meaning. They must be
#: reported as unmapped headers and must never fill a canonical field.
IRRELEVANT_COLUMNS: Tuple[str, ...] = (
    "Internal Case Owner", "Servicing Team", "Source System Batch",
)

#: Optional canonical fields a dialect may legitimately omit. Dropping one must
#: degrade disclosure, never corrupt the rest of canonical truth.
OPTIONAL_OMISSIONS: Dict[str, Tuple[str, ...]] = {
    ASSET_EQUITY_RELEASE: ("further_advance_flag", "occupancy_type"),
    ASSET_BRIDGE: ("prepayment_terms_description", "occupancy_type"),
    ASSET_FINANCE: ("prepayment_terms_description", "model"),
}


#: Headers a manifest's ``change_alias`` evolution renames a column to. They are
#: part of the approved contract too — the point of that case is that a lender
#: renaming a column mid-history still resolves, not that it starts failing.
EVOLUTION_ALIASES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    ASSET_BRIDGE: {"current_outstanding_balance": ("Balance O/S",)},
}


def vocabulary(asset_class: str) -> Dict[str, Tuple[str, str]]:
    """``{canonical_field: (lender header, locale header)}`` for an asset class."""
    merged = dict(COMMON_VOCABULARY)
    merged.update(_BY_ASSET[asset_class])
    return merged


def headers_for(asset_class: str, flavour: str) -> Dict[str, str]:
    """``{canonical_field: source header}`` for one vocabulary flavour."""
    vocab = vocabulary(asset_class)
    if flavour == "canonical":
        return {field: field for field in vocab}
    index = {"lender": 0, "locale": 1}[flavour]
    return {field: pair[index] for field, pair in vocab.items()}


def all_aliases(asset_class: str) -> Dict[str, Tuple[str, ...]]:
    """``{canonical_field: aliases}`` — everything the contract must cover.

    Both non-canonical vocabularies plus any mid-history rename declared by a
    ``change_alias`` schema evolution.
    """
    out = {field: (pair[0], pair[1])
           for field, pair in vocabulary(asset_class).items()}
    for field, extra in EVOLUTION_ALIASES.get(asset_class, {}).items():
        out[field] = tuple(dict.fromkeys(out.get(field, ()) + tuple(extra)))
    return out


__all__ = ["COMMON_VOCABULARY", "EVOLUTION_ALIASES", "IRRELEVANT_COLUMNS",
           "OPTIONAL_OMISSIONS", "all_aliases", "headers_for", "vocabulary"]
