#!/usr/bin/env python3
"""The Annex 12 arrears buckets, against the regulator's own definition.

Three defects lived here with no test of any kind on them, and all three are
the sprint's headline rule in different disguises — *do not call an amount a
rate*, and *do not let two definitions of one band coexist*:

1. Every one of IVSS38–IVSS44 summed ``current_outstanding_balance`` into a
   field the ESMA schema types ``PercentageRate``. A pool with £10m in the
   30–59 band reported "10000000" into a percentage element, in a field the
   constraint file marks as admitting **no** ND fallback.
2. IVSS38, named "1 to 29 days", used ``min: 0`` — so every performing loan in
   the book was reported as being in arrears. On a healthy pool that is the
   largest misstatement of the set.
3. IVSS40, named "60 to 89 days", used ``min: 50``. Balances at 50–59 days
   were counted twice, in IVSS39 and again in IVSS40, and the bands no longer
   partitioned the book.

The authority is in the repository and was read, not assumed —
``config/system/DRAFT1auth.098.001.04_1.3.0.xsd``, ``ArrearsData2``:

    "The percentage of exposures in arrears on principal and/or interest
    payments due for a period between 30 and 59 days (inclusive) as at the
    data cut-off date. The percentage is calculated as the total outstanding
    principal amount ... of the exposures in this category of arrears,
    relative to the total outstanding principal amount of all exposures as at
    the data cut-off date."

Three things fall out of that sentence, and each is asserted below: it is a
percentage; the denominator is the *whole pool*, not the arrears book; and the
bands are inclusive at both ends.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.gate_4_projection.annex12_projector import bucket_share

TEMPLATE = _REPO_ROOT / "config" / "regime" / "annex12_template.yaml"
CONSTRAINTS = _REPO_ROOT / "config" / "regime" / "annex12_field_constraints.yaml"

#: The bands the schema declares, and the codes Trakt maps them to.
BANDS = {
    "IVSS38": (1, 29),
    "IVSS39": (30, 59),
    "IVSS40": (60, 89),
    "IVSS41": (90, 119),
    "IVSS42": (120, 149),
    "IVSS43": (150, 179),
}

# A pool whose arithmetic can be done in the head. £1,000,000 total, and one
# loan sitting exactly on each boundary that the old bounds got wrong.
POOL = pd.DataFrame([
    # days in arrears, balance
    {"number_of_days_in_arrears": 0,   "current_outstanding_balance": 600_000},
    {"number_of_days_in_arrears": 1,   "current_outstanding_balance": 50_000},
    {"number_of_days_in_arrears": 29,  "current_outstanding_balance": 50_000},
    {"number_of_days_in_arrears": 30,  "current_outstanding_balance": 80_000},
    {"number_of_days_in_arrears": 55,  "current_outstanding_balance": 20_000},
    {"number_of_days_in_arrears": 59,  "current_outstanding_balance": 20_000},
    {"number_of_days_in_arrears": 60,  "current_outstanding_balance": 40_000},
    {"number_of_days_in_arrears": 90,  "current_outstanding_balance": 100_000},
    {"number_of_days_in_arrears": 200, "current_outstanding_balance": 40_000},
])
POOL_TOTAL = 1_000_000.0


def _share(minimum, maximum):
    return bucket_share(POOL, "current_outstanding_balance",
                        "number_of_days_in_arrears", minimum, maximum)


def _computations():
    doc = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    return {spec.get("output_alias"): spec
            for spec in (doc["annex12"].get("computations") or {}).values()
            if spec.get("output_alias")}


def test_a_bucket_reports_a_percentage_of_the_pool_not_a_balance():
    """£80k + £20k + £20k = £120k of a £1,000,000 pool. The schema wants
    12.0, not 120000."""
    assert _share(30, 59) == pytest.approx(12.0)


def test_the_denominator_is_the_whole_pool_not_the_arrears_book():
    """£400,000 of the pool is in arrears at all. If the denominator were the
    arrears book, 30–59 would report 120/400 = 30.0%. The regulator says
    "relative to the total outstanding principal amount of all exposures", so
    the performing £600k stays in the denominator."""
    assert _share(30, 59) == pytest.approx(12.0)
    assert _share(30, 59) != pytest.approx(30.0)


def test_the_bands_are_inclusive_at_both_ends():
    """"between 30 and 59 days (inclusive)". The loans at exactly 30 and
    exactly 59 are both in — 80k + 20k + 20k. Excluding either boundary would
    give 4.0% or 10.0%."""
    assert _share(30, 59) == pytest.approx(12.0)
    assert _share(60, 89) == pytest.approx(4.0)     # the loan at exactly 60


def test_a_performing_loan_is_not_reported_as_one_to_twenty_nine_days():
    """The IVSS38 defect. £600k of this pool is current. The band is 1–29, so
    the answer is 100k/1m = 10.0%. With the old ``min: 0`` it was 70.0% — the
    whole performing book declared as being in arrears."""
    assert _share(1, 29) == pytest.approx(10.0)
    assert _share(0, 29) == pytest.approx(70.0), (
        "sanity: the old bound really did sweep in the performing book")


def test_the_bands_do_not_overlap_and_sum_to_the_arrears_share():
    """The IVSS40 defect. With ``min: 50`` the £20k at 55 days and the £20k at
    59 days were counted in IVSS39 *and* IVSS40, so the bands summed to more
    arrears than the book contained."""
    banded = sum(_share(lo, hi) for lo, hi in BANDS.values())
    tail = _share(180, 10_000)
    assert banded + tail == pytest.approx(40.0), (
        "£400k of a £1m pool is in arrears; the bands must partition it")


def test_the_template_declares_the_bands_the_schema_declares():
    """The bounds live in configuration, so the configuration is what has to
    agree with the schema — asserting only the function would let the template
    drift back."""
    computations = _computations()
    for code, (lo, hi) in BANDS.items():
        spec = computations[code]
        assert spec["method"] == "BUCKET_SHARE", (
            f"{code} is typed PercentageRate by the schema; BUCKET_SUM would "
            "put a balance in it")
        assert (spec["min"], spec["max"]) == (lo, hi), (
            f"{code} declares {spec['min']}–{spec['max']}, schema says {lo}–{hi}")
    assert computations["IVSS44"]["min"] == 180


def test_every_arrears_code_the_constraints_call_a_percentage_produces_one():
    """The constraint file and the computation disagreed for seven codes, and
    since these admit no ND fallback the disagreement could not be absorbed —
    it would have gone into a submission."""
    constraints = yaml.safe_load(CONSTRAINTS.read_text(encoding="utf-8"))
    fields = constraints["annex12_field_constraints"]["fields"]
    computations = _computations()
    for code in list(BANDS) + ["IVSS44"]:
        declared = str(fields[code].get("format", "")).upper()
        assert "PERCENTAGE" in declared
        assert computations[code]["method"] == "BUCKET_SHARE"


def test_an_empty_pool_reports_nothing_rather_than_dividing_by_zero():
    empty = POOL.iloc[0:0]
    assert bucket_share(empty, "current_outstanding_balance",
                        "number_of_days_in_arrears", 30, 59) is None
