"""Regime contract — what a regulatory template requires, derived from authority.

One module builds the effective Annex 2 contract from the sources that own each
kind of fact. No stage authors regulatory truth of its own, and no hand-written
copy of the template exists.
"""

from .annex2_contract import (  # noqa: F401
    ASSET, CLIENT, DERIVATION, OPERATOR, REGISTRY, UNIVERSE, WORKBOOK, XSD,
    Annex2Contract, FieldContract, as_delivery_rules, build_contract,
    materialise_delivery_rules,
)
