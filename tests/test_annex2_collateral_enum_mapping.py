#!/usr/bin/env python3
"""tests/test_annex2_collateral_enum_mapping.py — RREC5 collateral type.

``config/system/enum_mapping.yaml::ESMA_Annex2.collateral_type`` feeds Annex 2
field **RREC5**, which lands at
``…/Coll/CollCmonData/Dtls/CmonData/CollTp`` and is typed
``CollateralType7Code`` in ``DRAFT1auth.099.001.04_1.3.0.xsd``.

The table previously targeted ``R1``/``R2``/``C1``/``C2``, none of which are
members of that enumeration. It was not dead configuration: the projector's
synonym resolver discards a synonym whose TARGET is absent from this table, so
the real-data value "Residential property" (which ``enum_synonyms.yaml`` maps
to ``RBLD``) reached Gate 5 unmapped and the XSD rejected it. The demo platform
carried a per-run overlay to work around it.

These tests pin the correction against the XSD itself — the enumeration is read
from the schema, never hardcoded here.

Run: python -m pytest tests/test_annex2_collateral_enum_mapping.py
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import yaml
from lxml import etree

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_XSD = _REPO / "config" / "system" / "DRAFT1auth.099.001.04_1.3.0.xsd"
_MAPPING = _REPO / "config" / "system" / "enum_mapping.yaml"

#: Targets that are NOT members of CollateralType7Code and must never return.
_INVALID_LEGACY_CODES = ("R1", "R2", "C1", "C2")


def _collateral_type_enumeration() -> set:
    """The allowed CollTp values, read from the delivery XSD."""
    text = _XSD.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r'<xs:simpleType name="CollateralType7Code">(.*?)</xs:simpleType>',
        text, re.S)
    assert match, "CollateralType7Code not found in the delivery XSD"
    return set(re.findall(r'value="([^"]+)"', match.group(1)))


def _table() -> dict:
    mapping = yaml.safe_load(_MAPPING.read_text(encoding="utf-8"))
    return mapping["ESMA_Annex2"]["collateral_type"]


class TestCollateralTypeTargetsAreXsdValid(unittest.TestCase):
    def setUp(self):
        self.allowed = _collateral_type_enumeration()
        self.table = _table()

    def test_the_enumeration_was_actually_read(self):
        self.assertEqual(len(self.allowed), 25)
        self.assertIn("RBLD", self.allowed)

    def test_every_target_is_a_member_of_the_enumeration(self):
        invalid = sorted({v for v in self.table.values() if v not in self.allowed})
        self.assertEqual(invalid, [], f"targets not in CollateralType7Code: {invalid}")

    def test_the_old_invalid_codes_cannot_be_emitted(self):
        """R1/R2/C1/C2 must not be reachable as an RREC5 value."""
        for code in _INVALID_LEGACY_CODES:
            self.assertNotIn(code, set(self.table.values()),
                             f"{code} is not a CollTp member and must not be a target")
            self.assertNotIn(code, self.allowed)

    def test_each_target_validates_against_the_real_schema_type(self):
        """Not just set membership — actual XSD validation of the value."""
        text = _XSD.read_text(encoding="utf-8", errors="replace")
        simple = re.search(
            r'(<xs:simpleType name="CollateralType7Code">.*?</xs:simpleType>)',
            text, re.S).group(1)
        wrapper = ('<?xml version="1.0"?>'
                   '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">'
                   '<xs:element name="CollTp" type="CollateralType7Code"/>'
                   + simple + '</xs:schema>')
        schema = etree.XMLSchema(etree.fromstring(wrapper.encode()))
        for target in sorted(set(self.table.values())):
            doc = etree.fromstring(f"<CollTp>{target}</CollTp>".encode())
            self.assertTrue(schema.validate(doc),
                            f"{target}: {schema.error_log.last_error}")
        # ...and the legacy codes genuinely fail the same validator.
        for code in _INVALID_LEGACY_CODES:
            doc = etree.fromstring(f"<CollTp>{code}</CollTp>".encode())
            self.assertFalse(schema.validate(doc), f"{code} unexpectedly validated")


class TestCollateralTypeMappings(unittest.TestCase):
    def setUp(self):
        self.table = _table()

    def test_dwelling_forms_map_to_residential_building(self):
        for key in ("HOUSE", "DETACHED", "SEMI_DETACHED", "BUNGALOW",
                    "FLAT", "APARTMENT", "MAISONETTE"):
            self.assertEqual(self.table[key], "RBLD", key)

    def test_mixed_use_maps_to_mixd(self):
        self.assertEqual(self.table["MIXED_USE"], "MIXD")
        self.assertEqual(self.table["COMMERCIAL_RESIDENTIAL"], "MIXD")

    def test_commercial_and_industrial_split_correctly(self):
        self.assertEqual(self.table["OFFICE"], "CBLD")
        self.assertEqual(self.table["RETAIL"], "CBLD")
        self.assertEqual(self.table["WAREHOUSE"], "IBLD")
        self.assertEqual(self.table["INDUSTRIAL"], "IBLD")

    def test_land_and_development_remain_othr(self):
        """OTHR was already XSD-valid; no defect justified reclassifying it.

        Moving these to OTRE ("other real estate") is a semantic change that
        needs its own regulatory decision, not a ride-along with a validity fix.
        """
        for key in ("LAND", "AGRICULTURAL", "DEVELOPMENT"):
            self.assertEqual(self.table[key], "OTHR", key)

    def test_every_enumeration_member_has_an_identity_mapping(self):
        """A tape already speaking ESMA must survive projection unchanged."""
        for code in _collateral_type_enumeration():
            self.assertEqual(self.table.get(code), code,
                             f"{code} has no identity mapping")

    def test_labelled_real_data_forms_resolve(self):
        """Real ERE canonical data arrives as ``Label (CODE)``."""
        self.assertEqual(self.table["RESIDENTIAL BUILDING (RBLD)"], "RBLD")
        self.assertEqual(self.table["COMMERCIAL BUILDING (CBLD)"], "CBLD")
        self.assertEqual(self.table["OTHER (OTHR)"], "OTHR")

    def test_the_table_is_not_a_parenthetical_parser(self):
        """Labelled forms are EXPLICIT keys, each validated by the tests above.

        A general "extract the code in brackets" parser would need to validate
        the extracted code against this field's enumeration before it could be
        trusted; explicit keys need no such guard, and an unrecognised label
        stays unmapped rather than becoming a plausible-looking code.
        """
        self.assertNotIn("SOMETHING ELSE (ZZZZ)", self.table)
        self.assertNotIn("ZZZZ", self.table)


class TestUnsupportedValuesDoNotBecomeValidCodes(unittest.TestCase):
    """An unrecognised value must stay unrecognised, not become a code.

    ``apply_enum_mappings`` keeps the original where no mapping matches
    (``mapped.where(mapped.notna(), original)``), and ``collateral_type`` is
    NOT in the projector's ``strict_enum_fields``. So an unsupported value
    survives to Gate 5 and is rejected by the XSD — visibly. What must never
    happen is a silent substitution into a *valid-looking* CollTp.
    """

    def setUp(self):
        self.table = _table()
        self.allowed = _collateral_type_enumeration()

    def test_unknown_non_empty_values_have_no_mapping(self):
        for value in ("SPACE STATION", "TOTALLY UNKNOWN", "R1", "C2",
                      "RESIDENTIAL BUILDING (ZZZZ)"):
            self.assertIsNone(self.table.get(value.upper()),
                              f"{value!r} must not resolve to a CollTp code")

    def test_no_mapping_key_targets_a_code_it_does_not_name_or_describe(self):
        """Every identity key maps to itself; nothing is quietly re-pointed."""
        for key, target in self.table.items():
            if key in self.allowed:
                self.assertEqual(key, target,
                                 f"{key} is a valid code but maps to {target}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
