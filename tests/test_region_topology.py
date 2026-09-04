#!/usr/bin/env python3
"""What "region" means on each surface, pinned so drift is loud.

THE PROBLEM THIS GUARDS. Five owners each decide what a region is, and three
different FIELD FAMILIES wear the word:

    reporting   canonical_region_reporting / canonical_region_detail /
                collateral_geography          — business name "Region"
    NUTS3       geographic_region_obligor / geographic_region_collateral
    ITL3        geographic_region_*_itl3      — derived from the NUTS3 pair

They are not interchangeable. The reporting family is what Risk Limits and the
Stratifications mean by Region; ITL3 is a sub-geography that only resolves where
a postcode does. A question answered on one and a limit evaluated on another are
answers about different populations, and nothing in the estate reconciles them.

This file asserts no behaviour. It RECORDS the topology, so that adding a region
field, or repointing a surface, fails here and is a decision someone made rather
than one that happened. See docs/mi_query_region_end_to_end_audit.md.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_REGISTRY = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"

#: The three granularities, and every governed field in each. A field carrying
#: `value_domain: uk_region` that is in none of these fails the first test —
#: which is the point: a new region field must be classified before it can join
#: the alias pool that `categorical_spans.preferred_field` binds from.
#:
#: Read from `mi_agent.region_basis` rather than restated here, because that
#: module is what the receipt uses to say WHICH region a figure was measured on.
#: A file that recorded its own copy of the topology could agree with itself
#: while disagreeing with the answer a reader is shown.
from mi_agent.region_basis import (                    # noqa: E402
    ITL3_FIELDS as ITL3, NUTS3_FIELDS as NUTS3, REPORTING_FIELDS as REPORTING)


def _registry_fields():
    with open(_REGISTRY, "r", encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("fields") or {}


def _uk_region_fields():
    return {k for k, e in _registry_fields().items()
            if (e or {}).get("value_domain") == "uk_region"}


class TestTheFamilies(unittest.TestCase):

    def test_every_uk_region_field_is_classified(self):
        """A field may not join the alias pool unclassified.

        `value_domain: uk_region` does TWO jobs since the aliasing work: it says
        which vocabulary resolves a value (`region_resolution`), and it says
        which fields are the SAME CONCEPT for filter binding. The second claim
        is only true within a family.
        """
        self.assertEqual(_uk_region_fields(),
                         set(REPORTING) | set(NUTS3) | set(ITL3))

    def test_the_reporting_family_is_what_a_reader_calls_region(self):
        fields = _registry_fields()
        for key in REPORTING:
            with self.subTest(field=key):
                self.assertEqual((fields[key] or {}).get("business_name"),
                                 "Region" if key != "canonical_region_detail"
                                 else "Region Detail")


class TestWhatEachSurfaceReads(unittest.TestCase):
    """The five owners. Each line here is a real coupling, not a preference."""

    def test_mi_prefers_the_harmonised_reporting_region(self):
        from mi_agent.llm_query_parser import _REGION_DEFAULT, _REGION_PREFERENCE

        self.assertEqual(_REGION_PREFERENCE[0], "canonical_region_reporting")
        self.assertEqual(_REGION_DEFAULT, "collateral_geography")

    def test_mi_never_offers_a_sub_geography_as_the_region_axis(self):
        """ITL3 is a finer geography, not another spelling of Region."""
        from mi_agent.llm_query_parser import _REGION_PREFERENCE

        self.assertFalse(set(_REGION_PREFERENCE) & set(ITL3))

    def test_an_itl3_only_claim_is_disclosed_rather_than_bound(self):
        """The safety that keeps the pooling honest — and it is IMPLICIT, held
        only by ITL3's absence from the preference order. Asserted so it cannot
        be lost by editing that order."""
        from mi_agent.categorical_spans import preferred_field

        with open(_REGISTRY, "r", encoding="utf-8") as fh:
            semantics = yaml.safe_load(fh)
        self.assertIsNone(preferred_field(list(ITL3), semantics))

    def test_risk_limits_evaluates_geography_on_the_reporting_family(self):
        """FIXED 2026-09-04, and this is the line that used to record the
        disagreement. The evaluator's preference order now LEADS with the
        harmonised reporting family, so a concentration limit and an MI answer
        about the same region are measured on the same column. Before this, a
        tape spelling one region three ways had a 75% concentration tested as
        three 25% bars and a 40% limit reported compliant.

        The Schedule 8 keyword rule still carries the NUTS3 field as its
        extraction-time DIMENSION HINT, which is a different thing from the
        basis: each test now records the column its actual was measured on."""
        from mi_agent_api.risk_limits import _REGION_COLUMNS

        self.assertEqual(_REGION_COLUMNS[:len(REPORTING)], tuple(REPORTING))
        self.assertIn("geographic_region_obligor", _REGION_COLUMNS)

    def test_the_schedule_8_hint_is_a_hint_and_not_the_basis(self):
        from mi_agent.risk_monitor.schedule8_extractor import _CATEGORY_RULES

        geographic = [r for r in _CATEGORY_RULES
                      if r[1] == "geographic_concentration"]
        self.assertTrue(geographic)
        self.assertIn(geographic[0][2], NUTS3)

    def test_the_exposure_map_reads_only_itl3(self):
        from mi_agent_api.geo import _ITL3_FIELDS

        self.assertEqual(set(_ITL3_FIELDS), set(ITL3))

    def test_the_funded_bridge_keeps_its_own_region_family(self):
        """A second list, fixed on 2026-09-03 after the harmonised columns were
        registered and it was not updated. Recorded because it is the shape of
        defect this whole file exists to make loud."""
        from mi_agent_api.chat_routing import _REGION_FAMILY

        self.assertIn("canonical_region_reporting", _REGION_FAMILY)
        self.assertTrue(set(_REGION_FAMILY) & set(REPORTING))


class TestTheReceiptCanNameTheLevel(unittest.TestCase):
    """Lineage: the answer a reader is shown must be able to say which of the
    three families produced it, and how much of the book resolved there."""

    def test_every_classified_field_has_a_level(self):
        from mi_agent.region_basis import level_of

        for key in REPORTING + NUTS3 + ITL3:
            with self.subTest(field=key):
                self.assertIsNotNone(level_of(key))

    def test_a_field_that_is_not_a_region_has_no_level(self):
        from mi_agent.region_basis import level_of

        self.assertIsNone(level_of("product_type"))
        self.assertIsNone(level_of(None))

    def test_each_level_has_a_word_a_reader_understands(self):
        from mi_agent.region_basis import LEVEL_LABELS, level_of

        for key in REPORTING + NUTS3 + ITL3:
            with self.subTest(field=key):
                self.assertIn(level_of(key), LEVEL_LABELS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
