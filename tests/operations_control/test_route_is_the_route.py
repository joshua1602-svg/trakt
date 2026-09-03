#!/usr/bin/env python3
"""The MI Query log must say WHICH capability answered.

MEASURED 2026-09-03. All 954 records carried ``route: "mi"`` -- answered,
refused and errored alike. The field that exists to say where the query model
is weak said the same thing about every question in the corpus, so a whole day
of live refusals could not be attributed to any capability, and four
successive diagnoses were made by reading code instead of reading the log.

The cause: `build_record` read ``spec.route_id`` first, and `mi_query_spec`
declares that field ``route_id: str = "mi"  # mi | mna | regulatory_annex2``.
It is the DOMAIN, constant for every MI question, so reading it first meant
``metadata.route`` -- where the routed capability's own name sits -- was never
reached.

It also contradicted the field's own documented intent. The comment said the
workflow path leaves the route unset so that "no named route stays visible";
a constant default is exactly the back-fill that stops it being visible.
"""
from __future__ import annotations

import unittest

from operations_control.mi_query_telemetry import SCHEMA_VERSION, build_record


class _Result:
    tenant_id = "ERE"
    portfolio_id = "ERE/2026-06-30"
    request_id = "req"
    correlation_id = "corr"
    capability = "mi.question.answer"
    status = "ok"
    error_code = None
    warnings = ()
    snapshot = None
    audit = None

    def __init__(self, metadata, spec=None):
        self.result = {"metadata": metadata,
                       "spec": spec if spec is not None else {"route_id": "mi"},
                       "answer": "x", "ok": True}


def _record(metadata, spec=None):
    return build_record(_Result(metadata, spec), question="q")


class TestTheRouteIsTheRoutedCapability(unittest.TestCase):

    def test_it_records_the_route_that_answered(self):
        self.assertEqual(_record({"route": "pipeline_summary"})["route"],
                         "pipeline_summary")

    def test_the_domain_no_longer_masks_it(self):
        """THE LIVE DEFECT. `route_id` is "mi" on every MI question; read
        first, the route name is unreachable."""
        rec = _record({"route": "stage_movement"}, spec={"route_id": "mi"})
        self.assertEqual(rec["route"], "stage_movement")
        self.assertNotEqual(rec["route"], "mi")

    def test_no_named_route_records_None_rather_than_a_constant(self):
        """The workflow path runs no route. Recording "mi" there asserts one
        did, which is the back-fill this field's own comment forbids."""
        self.assertIsNone(_record({}, spec={"route_id": "mi"})["route"])

    def test_the_domain_is_kept_rather_than_discarded(self):
        """A different fact, not a redundant one. Dropping it to fix the
        conflation would lose what the record already carried."""
        rec = _record({"route": "geo_exposure"}, spec={"route_id": "mi"})
        self.assertEqual(rec["domain"], "mi")

    def test_every_outcome_is_attributable(self):
        for route in ("pipeline_summary", "stage_movement", "portfolio_summary"):
            with self.subTest(route=route):
                self.assertEqual(_record({"route": route})["route"], route)

    def test_the_schema_version_marks_the_change(self):
        """Readers use `.get`, so 1.0.0 records still load. The version is how
        an operator knows which records can be trusted to name a route -- every
        record written before this one cannot."""
        self.assertEqual(SCHEMA_VERSION, "1.1.0")


class TestTheLogRowCarriesBoth(unittest.TestCase):

    def test_the_row_shows_route_and_domain(self):
        from operations_control.api.mi_query_routes import _row
        row = _row({"route": "stage_movement", "domain": "mi"})
        self.assertEqual(row["route"], "stage_movement")
        self.assertEqual(row["domain"], "mi")

    def test_a_pre_change_record_still_renders(self):
        """1.0.0 records carry no `domain`. The log must not break on the very
        history it exists to help read."""
        from operations_control.api.mi_query_routes import _row
        self.assertIsNone(_row({"route": "mi"})["domain"])


if __name__ == "__main__":
    unittest.main()
