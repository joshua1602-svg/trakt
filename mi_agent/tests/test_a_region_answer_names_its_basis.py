#!/usr/bin/env python3
"""A region answer says WHICH region it measured, and how much of it resolved.

THE DEFECT. `engine.region_taxonomy` writes `region_mapping_method` onto every
prepared row — exact / synonym / unresolved / absent — precisely so a
consolidated answer "can always explain where each category came from". Nothing
read it. A "balance by region" answer over a tape where part of the book has no
governed region returned a number computed from the rows that DID resolve, with
no statement anywhere in the response that the others were dropped, and no
statement of which of the three region granularities (reporting / NUTS3 / ITL3)
produced it. Two surfaces answering "region" on different fields over different
covered populations both presented as "Region".

The number is not necessarily wrong. Standing silent about its basis is what
makes it indefensible, so the receipt now names:

    field    the governed column the figure was grouped or filtered on
    level    which of the three families that column belongs to
    coverage how many of the frame's rows carry a governed value at that level

and, when coverage is PARTIAL, says so in the rendered receipt line. Full
coverage stays silent: a caveat printed on every answer is a caveat nobody
reads.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import execution_receipt as rc
from mi_agent.mi_query_spec import MIQuerySpec

REPORTING = "canonical_region_reporting"
NUTS3 = "geographic_region_obligor"
ITL3 = "geographic_region_obligor_itl3"
METHOD = "region_mapping_method"
_LTV = "current_loan_to_value"

_SEMANTICS = {
    "fields": {
        REPORTING: {"business_name": "Region", "value_domain": "uk_region"},
        NUTS3: {"business_name": "Region", "value_domain": "uk_region"},
        ITL3: {"business_name": "ITL3 Region", "value_domain": "uk_region"},
        _LTV: {"business_name": "Current LTV"},
        "product_type": {"business_name": "Product Type"},
    }
}


class _Grouped:
    """A grouped result over `field`, carrying the executor's own metadata."""

    result_type = "table"
    row_count = 3

    def __init__(self, field=REPORTING):
        self.metadata = {"group_field_keys": [field] if field else [],
                         "reconciliation": {"total_records": 10,
                                            "records_after_filters": 10,
                                            "records_included": 10}}


def _book(resolved: int, unresolved: int, field: str = REPORTING):
    """A frame where `unresolved` of the rows carry no governed region."""
    rows = resolved + unresolved
    return pd.DataFrame({
        field: (["London"] * resolved) + ([None] * unresolved),
        METHOD: (["exact"] * resolved) + (["unresolved"] * unresolved),
        _LTV: [0.5] * rows,
    })


def _receipt(field=REPORTING, frame=None, filters=None):
    spec = MIQuerySpec(intent="chart", chart_type="bar", metric=_LTV,
                       dimension=field, aggregation="sum",
                       filters=filters or {})
    return rc.build_receipt(spec=spec, query_result=_Grouped(field),
                            semantics=_SEMANTICS, facets=[], frame=frame)


# --------------------------------------------------------------------- basis #
def test_the_receipt_names_the_field_and_the_level_it_measured():
    basis = _receipt(frame=_book(8, 2)).to_dict()["regionBasis"]
    assert basis["field"] == REPORTING
    assert basis["level"] == "reporting"


def test_a_nuts3_answer_is_not_recorded_as_a_reporting_answer():
    """The double-bind made visible: Risk Limits evaluates geography on the
    NUTS3 field while MI answers on the reporting family. Two answers about
    different populations must be distinguishable in the record."""
    basis = _receipt(NUTS3, frame=_book(6, 4, NUTS3)).to_dict()["regionBasis"]
    assert basis["field"] == NUTS3
    assert basis["level"] == "nuts3"


def test_a_sub_geography_is_recorded_as_a_sub_geography():
    basis = _receipt(ITL3, frame=_book(5, 5, ITL3)).to_dict()["regionBasis"]
    assert basis["level"] == "itl3"


def test_a_question_that_never_touched_region_gets_no_basis():
    spec = MIQuerySpec(intent="chart", chart_type="bar", metric=_LTV,
                       dimension="product_type", aggregation="sum")
    receipt = rc.build_receipt(spec=spec, query_result=_Grouped("product_type"),
                               semantics=_SEMANTICS, facets=[],
                               frame=_book(8, 2))
    assert receipt.to_dict()["regionBasis"] is None


def test_a_region_FILTER_is_a_region_basis_too():
    """"Balance in London" is measured on the region field just as much as
    "balance by region" is, and drops the same unresolved rows."""
    basis = _receipt(field=None, frame=_book(8, 2),
                     filters={REPORTING: "London"}).to_dict()["regionBasis"]
    assert basis["field"] == REPORTING


# ------------------------------------------------------------------ coverage #
def test_coverage_is_counted_from_the_governed_provenance_column():
    basis = _receipt(frame=_book(8, 2)).to_dict()["regionBasis"]
    assert basis["rows"] == 10
    assert basis["resolved"] == 8
    assert basis["share"] == 0.8
    assert basis["methods"]["exact"] == 8
    assert basis["methods"]["unresolved"] == 2


def test_a_frame_the_receipt_never_saw_leaves_coverage_unstated():
    """Degrade honestly: an unknown share is None, never 100%."""
    basis = _receipt(frame=None).to_dict()["regionBasis"]
    assert basis["field"] == REPORTING
    assert basis["level"] == "reporting"
    assert basis["rows"] is None and basis["share"] is None


# --------------------------------------------------------------- disclosure #
def test_partial_coverage_is_said_out_loud():
    line = _receipt(frame=_book(8, 2)).render()
    assert "Region basis" in line
    assert "80" in line


def test_full_coverage_says_nothing():
    assert "Region basis" not in _receipt(frame=_book(10, 0)).render()
