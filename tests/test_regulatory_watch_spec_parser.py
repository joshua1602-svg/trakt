"""tests/test_regulatory_watch_spec_parser.py — the Annex 2 normalizer.

Two levels:

* **fixture level** — the committed slice of the real ESMA v1.3.1 workbook,
  asserting derivation rules, provenance and determinism;
* **artefact level** — the full vendored ESMA workbook + XSD, asserting the
  normalizer agrees with what the Trakt repository already holds. That
  agreement is the evidence that the normalizer reads the same regulation the
  live Annex 2 pathway was built against.

Run: python -m pytest tests/test_regulatory_watch_spec_parser.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from regulatory_watch import PARSER_VERSION, REGIME              # noqa: E402
from regulatory_watch.annex2_spec import (                       # noqa: E402
    parse_annex2_spec,
)
from regulatory_watch.contracts import NormalizedSpec, UNKNOWN   # noqa: E402
from tests.helpers import regulatory_watch as fx                 # noqa: E402


# --------------------------------------------------------------------------- #
# Fixture-level derivation rules
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def spec():
    return fx.build_spec()


def test_fixture_is_real_esma_content():
    data = json.loads(fx.ROWS_FIXTURE.read_text(encoding="utf-8"))
    assert data["source_artefact"].startswith("DRAFT1auth.099.001.04")
    assert data["sheet"] == "DRAFT1auth.099.001.04"
    assert data["rows"], "the fixture must carry real workbook rows"


def test_spec_identity(spec):
    assert spec.regime == REGIME
    assert spec.parser_version == PARSER_VERSION


def test_nd_permission_is_derived_from_the_schema_vocabulary(spec):
    # NoDataAllowedJustification3Code == {ND5}
    assert spec.fields["RREC9"].nd_allowed == ["ND5"]
    assert spec.fields["RREC9"].nd_justification_type == \
        "NoDataAllowedJustification3Code"
    # No NoDataOptn branch published -> ND deterministically not permitted.
    assert spec.fields["RREL1"].nd_allowed == []
    assert spec.fields["RREL1"].nd_justification_type == UNKNOWN


def test_nd_permission_unions_every_published_justification_branch(spec):
    # RREC8 publishes NoDataOptn/NoData (ND1-3, ND5) AND the nested
    # NoDataOptn/NoData4/NoData (ND4). The permitted set is the union.
    field = spec.fields["RREC8"]
    assert field.nd_allowed == ["ND1", "ND2", "ND3", "ND4", "ND5"]
    assert "NoDataAllowedJustification1Code" in field.nd_justification_type
    assert "NoDataAllowedJustification4Code" in field.nd_justification_type


def test_enum_values_come_from_the_schema(spec):
    field = spec.fields["RREC9"]
    assert field.enum_type == "PropertyType1Code"
    assert field.enum_values == sorted(field.enum_values)
    assert {"RHOS", "RFLT", "OTHR"} <= set(field.enum_values)


def test_non_coded_fields_have_no_enumeration(spec):
    assert spec.fields["RREL24"].enum_values is None
    assert spec.fields["RREL24"].enum_type == UNKNOWN


def test_mandatory_is_derived_from_multiplicity(spec):
    assert spec.fields["RREL1"].multiplicity == "[1..1]"
    assert spec.fields["RREL1"].mandatory == "mandatory"


def test_element_path_and_tag_are_derived_from_the_value_branch(spec):
    field = spec.fields["RREC9"]
    assert field.xml_path.endswith("/Coll/CollCmonData/Dtls/PrprtyTp")
    assert field.xml_tag == "PrprtyTp"
    assert any(p.endswith("/PrprtyTp/Cd") for p in field.value_paths)


def test_every_field_carries_provenance_back_to_a_source_locator(spec):
    for code, field in spec.fields.items():
        assert field.provenance, f"{code} has no provenance"
        workbook = [p for p in field.provenance
                    if p.artefact_id == fx.WORKBOOK_ARTEFACT]
        assert workbook, f"{code} has no workbook provenance"
        for entry in workbook:
            assert entry.locator.startswith("sheet=")
            assert ";row=" in entry.locator
        if field.nd_allowed:
            schema = [p for p in field.provenance
                      if p.artefact_id == fx.SCHEMA_ARTEFACT]
            assert schema, f"{code} has ND values with no schema provenance"
            assert all(p.locator.startswith("xsd:simpleType/") for p in schema)


def test_order_is_the_source_sequence(spec):
    assert spec.order == sorted(spec.order,
                                key=lambda c: spec.fields[c].order_index)
    assert spec.fields[spec.order[0]].order_index == 0


def test_spec_round_trips_through_json(spec):
    restored = NormalizedSpec.from_dict(
        json.loads(json.dumps(spec.to_dict())))
    assert restored.to_dict() == spec.to_dict()


def test_no_attribute_is_guessed(spec):
    """Every attribute is either derived or explicitly UNKNOWN/unresolved."""
    for code, field in spec.fields.items():
        for attr in ("data_type", "format_pattern", "mandatory",
                     "multiplicity", "xml_path"):
            if getattr(field, attr) == UNKNOWN:
                assert attr in field.unresolved or attr in (
                    "data_type", "format_pattern"), \
                    f"{code}.{attr} is UNKNOWN but not declared unresolved"


# --------------------------------------------------------------------------- #
# Artefact-level: the real vendored ESMA workbook + schema
# --------------------------------------------------------------------------- #

_ARTEFACTS_PRESENT = fx.REAL_WORKBOOK.exists() and fx.REAL_SCHEMA.exists()
requires_artefacts = pytest.mark.skipif(
    not _ARTEFACTS_PRESENT,
    reason="vendored ESMA workbook/schema not present in this checkout")


@pytest.fixture(scope="module")
def real_spec():
    if not _ARTEFACTS_PRESENT:
        pytest.skip("vendored ESMA artefacts not present")
    return parse_annex2_spec(fx.REAL_WORKBOOK, fx.REAL_SCHEMA,
                             spec_version="auth.099.001.04-wb1.3.1")


@requires_artefacts
def test_real_workbook_normalizes_without_unresolved_attributes(real_spec):
    unresolved = {c: f.unresolved for c, f in real_spec.fields.items()
                  if f.unresolved}
    assert unresolved == {}, unresolved
    assert len(real_spec.fields) == 104


@requires_artefacts
def test_real_parse_is_deterministic():
    a = parse_annex2_spec(fx.REAL_WORKBOOK, fx.REAL_SCHEMA, spec_version="v")
    b = parse_annex2_spec(fx.REAL_WORKBOOK, fx.REAL_SCHEMA, spec_version="v")
    assert a.to_dict() == b.to_dict()


@requires_artefacts
def test_scope_matches_the_live_annex2_pathway(real_spec):
    assert real_spec.scope["performance"] == "PRF"
    assert real_spec.scope["asset_branch"] == "ResdtlRealEsttLn/PrfrmgLn"
    assert real_spec.scope["schema_namespace"] == \
        "urn:esma:xsd:DRAFT1auth.099.001.04"


@requires_artefacts
def test_nd_permissions_agree_with_the_committed_field_universe(real_spec):
    """Independent corroboration of the repo's own Annex 2 ND configuration.

    ``config/regime/annex2_field_universe.yaml`` derives ND eligibility from the
    ESMA *template* workbook (not vendored here). This normalizer derives it
    from the message workbook's NoData branches plus the XSD justification
    vocabularies. Two independent derivations agreeing on all 104 codes is the
    evidence that the watch reads the same regulation the live pathway
    implements.
    """
    universe = yaml.safe_load(
        (_REPO / "config/regime/annex2_field_universe.yaml")
        .read_text(encoding="utf-8"))["fields"]
    disagreements = []
    compared = 0
    for code, field in real_spec.fields.items():
        entry = universe.get(code)
        if not entry or field.nd_allowed is None:
            continue
        compared += 1
        nd14 = any(v in field.nd_allowed for v in ("ND1", "ND2", "ND3", "ND4"))
        nd5 = "ND5" in field.nd_allowed
        if (nd14, nd5) != (bool(entry["nd1_4_allowed"]),
                           bool(entry["nd5_allowed"])):
            disagreements.append((code, field.nd_allowed, entry))
    assert compared >= 100
    assert disagreements == [], disagreements


@requires_artefacts
def test_currency_attribute_codes_are_absent_exactly_as_documented(real_spec):
    """RREL18 / RREL28 / RREC22 carry no XML path in the ESMA workbook.

    ``config/regime/annex2_delivery_rules.yaml`` documents these three as
    currency ATTRIBUTES rather than elements, stating they have zero paths in
    the workbook. The normalizer reaches the same conclusion independently.
    """
    for code in ("RREL18", "RREL28", "RREC22"):
        assert code not in real_spec.fields


@requires_artefacts
def test_workbook_field_labels_agree_with_the_committed_universe(real_spec):
    universe = yaml.safe_load(
        (_REPO / "config/regime/annex2_field_universe.yaml")
        .read_text(encoding="utf-8"))["fields"]
    mismatches = [
        (code, real_spec.fields[code].label, universe[code]["field_name"])
        for code in sorted(real_spec.fields)
        if code in universe
        and real_spec.fields[code].label.strip().lower()
        != str(universe[code]["field_name"]).strip().lower()
    ]
    # One known, real metadata conflict between the two ESMA workbooks. It is
    # asserted rather than tolerated so a NEW divergence fails this test.
    assert [m[0] for m in mismatches] == ["RREC2"], mismatches
