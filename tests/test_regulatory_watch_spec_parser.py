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
def rows():
    return fx.load_rows()


@pytest.fixture(scope="module")
def codelists():
    return fx.load_codelists()


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

    The ESMA mapping workbook documents these three as
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


# --------------------------------------------------------------------------- #
# ND derivation — proof against the schema itself, not against agreement
# --------------------------------------------------------------------------- #
#
# Agreement with the committed field universe shows the derivation reproduces
# what Trakt already believes. It does NOT show the METHOD is right, and it
# cannot show the method will stay right on a FUTURE artefact. These tests do
# that: they resolve every code against the real XSD element tree and prove the
# union-of-choice-branches rule is the schema's own semantics.


@requires_artefacts
def test_nd_option_is_always_an_xs_choice_of_justification_branches():
    """The union rule is the DEFINITION of the structure, not a heuristic.

    Every ``NoDataOptn`` in the schema is typed as a ``NoDataJustification*
    Choice``: an ``xs:choice`` whose branches are the alternative ways of
    populating one ND slot. The permitted ND set is therefore exactly the union
    of the branch enumerations — over a choice, the union IS the value set.
    """
    from lxml import etree
    ns = "{http://www.w3.org/2001/XMLSchema}"
    root = etree.parse(str(fx.REAL_SCHEMA)).getroot()
    complex_types = {t.get("name"): t for t in root.iter(f"{ns}complexType")
                     if t.get("name")}

    nd_types = {el.get("type")
                for t in complex_types.values()
                for el in t.iter(f"{ns}element")
                if el.get("name") == "NoDataOptn" and el.get("type")}
    assert nd_types == {"NoDataJustification1Choice",
                        "NoDataJustification3Choice",
                        "NoDataJustification4Choice"}, nd_types

    for name in sorted(nd_types):
        node = complex_types[name]
        compositors = [etree.QName(c).localname for c in node
                       if etree.QName(c).localname in ("sequence", "choice",
                                                       "all")]
        assert compositors == ["choice"], (name, compositors)

    # The ND4 branch is a SEQUENCE of (Dt, NoData) — ND4 means "data will only
    # be available from <date>", which is why it carries a date alongside the
    # justification. It is a branch of the choice, not a separate field.
    four = complex_types["NoDataFour1"]
    children = [(el.get("name"), el.get("type"))
                for el in four.iter(f"{ns}element")]
    assert children == [("Dt", "ISODate"),
                        ("NoData", "NoDataAllowedJustification4Code")]


@requires_artefacts
def test_nd_derivation_matches_the_schema_for_every_in_scope_code(real_spec):
    """The strongest available proof: every code, resolved against the XSD.

    For all 104 in-scope codes this asserts three things at once:

    * the element path the normalizer derived is a path the schema DEFINES
      (so the element-node derivation is right, not just plausible);
    * the schema's own answer to "does this field admit ND?" matches the
      normalizer's — no overstatement, no understatement;
    * where ND is admitted, ``nd_allowed`` equals the union of the branches of
      the schema's choice type, exactly.
    """
    from regulatory_watch.annex2_spec import SchemaNdModel
    model = SchemaNdModel(fx.REAL_SCHEMA)

    unresolved_paths, disagreements = [], []
    with_nd = 0
    for code in sorted(real_spec.fields):
        field = real_spec.fields[code]
        schema_nd, detail = model.nd_allowed(field.xml_path)
        if schema_nd is None:
            unresolved_paths.append((code, field.xml_path, detail))
            continue
        if schema_nd:
            with_nd += 1
        if sorted(field.nd_allowed or []) != schema_nd:
            disagreements.append((code, field.nd_allowed, schema_nd, detail))

    assert len(real_spec.fields) == 104
    assert unresolved_paths == [], unresolved_paths
    assert disagreements == [], disagreements
    assert with_nd == 87, with_nd          # 17 codes admit no ND at all


@requires_artefacts
def test_nd_permission_sets_are_exactly_the_three_published_vocabularies(
        real_spec):
    """No field invents an ND set outside what the schema's choices allow."""
    permitted = {
        ("ND1", "ND2", "ND3", "ND4", "ND5"),   # Justification1Choice
        ("ND1", "ND2", "ND3", "ND4"),          # Justification4Choice
        ("ND5",),                              # Justification3Choice
        (),                                    # no NoDataOptn branch
    }
    observed = {tuple(f.nd_allowed or []) for f in real_spec.fields.values()}
    assert observed <= permitted, observed - permitted


@requires_artefacts
def test_rrec8_and_rrel80_are_workbook_path_errors_not_nd_exceptions(
        real_spec):
    """The two anomalies are a wrong PATH cell, not a different ND structure.

    The workbook places their ND branch under a sibling of the value element
    (``.../CmonData/Lien/NoDataOptn`` while the value is
    ``.../CmonData/LienVal/Lien``). The schema disagrees: it defines
    ``NoDataOptn`` under the value element. So the generic derivation is
    correct and the workbook cell is wrong — which is why the normalizer keys
    ND rows by CODE rather than by path, and warns.
    """
    from regulatory_watch.annex2_spec import SchemaNdModel
    model = SchemaNdModel(fx.REAL_SCHEMA)
    warned = {w["code"] for w in real_spec.parse_warnings
              if w["warning"] == "ND_BRANCH_PATH_INCONSISTENT"}
    assert warned == {"RREC8", "RREL80"}, warned

    for code in sorted(warned):
        field = real_spec.fields[code]
        # The schema defines NoDataOptn under the VALUE element ...
        schema_nd, _ = model.nd_allowed(field.xml_path)
        assert schema_nd == ["ND1", "ND2", "ND3", "ND4", "ND5"]
        # ... and NOT under the sibling the workbook's PATH column names.
        assert not field.nd_path.startswith(field.xml_path + "/"), field.nd_path
        # No attribute is left unresolved: the derivation survives the error.
        assert field.unresolved == []


def test_a_workbook_schema_nd_disagreement_fails_closed(rows, codelists):
    """If the two sources disagree on ND, NEITHER is trusted.

    This is the control that makes the derivation safe on a FUTURE artefact:
    a workbook that mis-tags an ND row would otherwise import the wrong
    justification vocabulary silently.
    """
    if not _ARTEFACTS_PRESENT:
        pytest.skip("vendored ESMA artefacts not present")
    from regulatory_watch.annex2_spec import SchemaNdModel, normalize
    nd_rows = fx.nd_leaf_rows(rows, "RREC9")
    assert nd_rows
    # RREC9's schema branch is Justification3Choice ({ND5}); claim ND1-3+ND5.
    mutated = fx.edit_rows(rows, "RREC9", lambda r: r.path == nd_rows[0].path,
                           data_type="NoDataAllowedJustification1Code")
    spec = normalize(mutated, codelists, artefact_id=fx.WORKBOOK_ARTEFACT,
                     schema_artefact_id=fx.SCHEMA_ARTEFACT,
                     spec_version="disagreement",
                     schema_model=SchemaNdModel(fx.REAL_SCHEMA))
    field = spec.fields["RREC9"]
    assert field.nd_allowed is None, "a disagreement must not pick a winner"
    assert "nd_allowed" in field.unresolved
    assert any(w["warning"] == "ND_SCHEMA_DISAGREEMENT"
               for w in spec.parse_warnings)


def test_the_schema_cross_check_confirms_agreement_on_the_fixture(rows,
                                                                  codelists):
    """With the real schema wired in, the committed slice raises no ND flag."""
    if not _ARTEFACTS_PRESENT:
        pytest.skip("vendored ESMA artefacts not present")
    from regulatory_watch.annex2_spec import SchemaNdModel, normalize
    spec = normalize(rows, codelists, artefact_id=fx.WORKBOOK_ARTEFACT,
                     schema_artefact_id=fx.SCHEMA_ARTEFACT,
                     spec_version="cross-checked",
                     schema_model=SchemaNdModel(fx.REAL_SCHEMA))
    flags = {w["warning"] for w in spec.parse_warnings}
    assert "ND_SCHEMA_DISAGREEMENT" not in flags
    assert "XSD_PATH_UNRESOLVED" not in flags


# --------------------------------------------------------------------------- #
# RREL18 / RREL28 / RREC22 — currency ATTRIBUTES, not missing metadata
# --------------------------------------------------------------------------- #

@requires_artefacts
def test_currency_codes_are_absent_from_the_whole_workbook_not_just_our_scope():
    """Their absence is not an artefact of the template/branch filter.

    Parsing the entire 61k-row sheet with NO scope filter at all still finds
    zero rows for these three codes, in any template and any asset branch. So
    "no XML path" is the authority's own statement, not a scoping side effect.
    """
    import openpyxl
    import re as _re
    wb = openpyxl.load_workbook(fx.REAL_WORKBOOK, read_only=True,
                                data_only=True)
    try:
        ws = wb["DRAFT1auth.099.001.04"]
        header, hits = {}, {"RREL18": 0, "RREL28": 0, "RREC22": 0}
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 3:
                header = {str(c).strip(): j for j, c in enumerate(row) if c}
                continue
            if i < 4:
                continue
            j = header.get("RTS Field code")
            cell = "" if j is None or row[j] is None else str(row[j])
            for code in _re.split(r"[\n|,;]+", cell):
                code = code.strip()
                if code in hits:
                    hits[code] += 1
    finally:
        wb.close()
    assert hits == {"RREL18": 0, "RREL28": 0, "RREC22": 0}, hits


@requires_artefacts
def test_currency_is_a_required_xml_attribute_on_the_amount_type():
    """The positive half of the proof: the schema carries Ccy as an ATTRIBUTE.

    ``ActiveOrHistoricCurrencyAndAmount`` is xs:simpleContent extending a
    decimal with ``Ccy`` ``use="required"``. A currency therefore cannot be an
    element, which is exactly what the effective Annex 2 contract's
    representation block documents for RREL18 / RREL28 / RREC22.
    """
    from lxml import etree
    ns = "{http://www.w3.org/2001/XMLSchema}"
    root = etree.parse(str(fx.REAL_SCHEMA)).getroot()
    amount = next(t for t in root.iter(f"{ns}complexType")
                  if t.get("name") == "ActiveOrHistoricCurrencyAndAmount")
    assert amount.find(f"{ns}simpleContent") is not None
    attributes = [(a.get("name"), a.get("use"))
                  for a in amount.iter(f"{ns}attribute")]
    assert ("Ccy", "required") in attributes, attributes


# --------------------------------------------------------------------------- #
# Value leaves — a secondary leaf's type must not be invisible
# --------------------------------------------------------------------------- #

@requires_artefacts
def test_monetary_fields_capture_both_value_leaves(real_spec):
    """A monetary field publishes Val/Amt AND Val/Sgn; both are captured."""
    field = real_spec.fields["RREL73"]
    relative = [leaf["relative_path"] for leaf in field.value_leaves]
    assert relative == ["Val/Amt", "Val/Sgn"], relative
    assert field.value_leaves[1]["data_type"] == "PlusOrMinusIndicator"
    # The primary remains the first typed leaf, unchanged.
    assert field.data_type == "ActiveOrHistoricCurrencyAndAmount"


@requires_artefacts
def test_every_value_leaf_key_is_relative_to_its_element(real_spec):
    for code, field in real_spec.fields.items():
        for leaf in field.value_leaves:
            assert not leaf["relative_path"].startswith("/"), code
            assert "Document" not in leaf["relative_path"], code
