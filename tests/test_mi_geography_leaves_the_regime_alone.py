#!/usr/bin/env python3
"""tests/test_mi_geography_leaves_the_regime_alone.py

MI's geography basis is an ANALYTICAL semantic. Regulatory reporting has its own
geography fields, its own sources and its own contract, and the two must not
touch. This file is the boundary, asserted rather than assumed.

WHY IT EXISTS
-------------
An earlier attempt at this work crossed the line in two ways, and both are
pinned here so they cannot recur:

  * it added an ``engine.region_taxonomy -> mi_agent`` import, inverting the one
    direction every other edge in this estate runs (MI reads the engine; the
    regulatory engine never reads MI);
  * it treated ``ND1`` in a regulatory obligor geography field as MISSING
    geography and overrode it from the collateral region. ND1 is an ESMA no-data
    code meaning "not collected" — a DECLARATION the lender makes. Overwriting a
    declaration with a different fact is not gap-filling, it is fabrication, and
    it changes what was reported to a regulator.

THE RULE
--------
ND1 stays ND1 while MI answers "Wales". Both statements are true at once because
they are about different fields, and neither field is derived from the other.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import mi_geography as geo
from mi_agent_api.funded_prep import (
    augment_platform_canonical_dimensions, prepare_funded_mi_dataset)

#: The regulatory half of the estate. Nothing here may read MI.
_REGULATORY_TREES = (
    "engine/gate_1_ingestion", "engine/gate_2_mapping", "engine/gate_3_transformation",
    "engine/gate_4_projection", "engine/gate_4b_delivery", "engine/gate_5_delivery",
    "engine/regime_contract", "engine/annex_delivery_agent", "engine/projection_agent",
    "engine/delivery_xml_agent", "engine/region_taxonomy.py",
)

#: The ESMA no-data codes. Declarations, not gaps.
_ND_CODES = ("ND1", "ND2", "ND3", "ND4", "ND5")


def _python_files():
    for entry in _REGULATORY_TREES:
        path = _REPO_ROOT / entry
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from (p for p in path.rglob("*.py"))


def _imported_modules(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):  # pragma: no cover - unreadable file
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


# --------------------------------------------------------------------------- #
# 1. Direction of dependency
# --------------------------------------------------------------------------- #
def test_no_regulatory_module_reads_the_mi_geography_owner():
    offenders = [
        (str(path.relative_to(_REPO_ROOT)), module)
        for path in _python_files()
        for module in _imported_modules(path)
        if module == "mi_agent.mi_geography"
        or module.startswith("mi_agent.mi_geography.")]
    assert offenders == [], (
        "regulatory code must not depend on the MI geography basis: " + repr(offenders))


def test_no_regulatory_module_reads_the_mi_geography_configuration():
    offenders = [str(path.relative_to(_REPO_ROOT)) for path in _python_files()
                 if "mi_geography" in path.read_text(encoding="utf-8")]
    assert offenders == []


def test_the_mi_owner_knows_nothing_about_no_data_codes():
    """A no-data code is a regulatory declaration. The MI geography owner has no
    business recognising one, let alone reinterpreting it."""
    source = (_REPO_ROOT / "mi_agent" / "mi_geography.py").read_text(encoding="utf-8")
    body = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))
    for code in _ND_CODES:
        assert code not in body


def _executable_source(path: Path) -> str:
    """``path`` with every comment and docstring removed — the code that RUNS."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    return ast.unparse(tree)


def test_the_mi_owner_names_no_geography_value():
    """Not a second mapping owner. `mi_agent.region_resolution` owns the ITL
    ladder and `engine.region_taxonomy` owns the vocabularies; this module names
    a basis and nothing else. Prose may explain them; code may not encode them."""
    body = _executable_source(_REPO_ROOT / "mi_agent" / "mi_geography.py")
    for place in ("Wales", "Scotland", "London", "TLC", "TLL", "UKC"):
        assert place not in body, place


def test_the_mi_owner_encodes_no_no_data_vocabulary():
    """It asks its own region owner whether a value is a place. It does not hold
    an opinion about what any particular non-place value MEANS — that is the
    regulatory contract's business, and restating it here would be a second copy
    of it free to drift."""
    body = _executable_source(_REPO_ROOT / "mi_agent" / "mi_geography.py")
    for code in _ND_CODES:
        assert code not in body


# --------------------------------------------------------------------------- #
# 2. ND1 stays ND1 while MI answers Wales
# --------------------------------------------------------------------------- #
def _regulatory_frame() -> pd.DataFrame:
    """Five loans. The obligor geography was never collected and the lender has
    declared so; the property region is known and readable."""
    return pd.DataFrame({
        "loan_identifier": [f"L{i}" for i in range(5)],
        "current_outstanding_balance": [100000.0] * 5,
        "current_valuation_amount": [250000.0] * 5,
        "origination_date": ["2020-06-15"] * 5,
        "reporting_date": ["2026-06-30"] * 5,
        # The REGULATORY geography fields, as Annex 2 carries them.
        "geographic_region_obligor": ["ND1"] * 5,
        "geographic_region_collateral": ["ND1"] * 5,
        # The MI geography, which the book does record.
        "collateral_geography": ["Wales", "Wales", "Scotland", "Wales", "London"],
    })


@pytest.mark.parametrize("prepare", [
    pytest.param(lambda d: augment_platform_canonical_dimensions(d)[0], id="serving"),
    pytest.param(lambda d: prepare_funded_mi_dataset(d)[0], id="historical"),
])
def test_a_no_data_declaration_survives_mi_preparation_unchanged(prepare):
    before = _regulatory_frame()
    after = prepare(before.copy())
    for field in ("geographic_region_obligor", "geographic_region_collateral"):
        assert list(after[field].astype(str)) == ["ND1"] * 5, field


@pytest.mark.parametrize("prepare", [
    pytest.param(lambda d: augment_platform_canonical_dimensions(d)[0], id="serving"),
    pytest.param(lambda d: prepare_funded_mi_dataset(d)[0], id="historical"),
])
def test_mi_still_answers_wales_on_the_same_book(prepare):
    """Both statements are true at once. That is the whole point."""
    after = prepare(_regulatory_frame())
    contract = geo.resolve_contract(asset_class="equity_release", frame=after)
    assert contract.primary_basis == geo.BASIS_COLLATERAL
    field = contract.field_for(contract.primary_basis, frame=after)
    assert field == "collateral_geography"
    assert after[field].value_counts()["Wales"] == 3


def test_the_book_does_not_claim_to_support_a_basis_it_only_declared():
    """A column holding nothing but no-data codes is not a geography. The book
    must not report the borrower basis as supported on the strength of it — that
    is what would turn a refusal into a table of five loans in "ND1"."""
    after = augment_platform_canonical_dimensions(_regulatory_frame())[0]
    contract = geo.resolve_contract(asset_class="equity_release", frame=after)
    assert not contract.supports(geo.BASIS_BORROWER)
    assert contract.supports(geo.BASIS_COLLATERAL)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
