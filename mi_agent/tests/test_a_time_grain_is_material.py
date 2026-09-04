#!/usr/bin/env python3
"""A grain the answer could not express is a different question, not a footnote.

THE RULING, 2026-09-04. Time grain is a MATERIAL semantic facet. A response may
remain PARTIAL only where the omitted element cannot change population, measure,
comparison basis, period, or economic interpretation. A grain changes two of
those: "month on month" and "week on week" compare different spans over
different boundaries, and a monthly improvement is not a weekly one.

WHAT SHIPPED BEFORE. From the live bank:

    Has pipeline progression improved month on month?
        verdict     partial
        notApplied  month — this answer is reported at week level, not by month

The disclosure was real, so this was never SILENT — and it was still an answer to
a question the reader did not ask, with the correction printed underneath the
figure rather than in place of it. `KIND_TIME_GRAIN` sat in `SHAPE_FACETS`, whose
own comment defended it: refusing would deny "a correct weekly movement for want
of a monthly series that does not exist". That trade is now settled the other
way, and the reasoning it replaces is recorded beside it.

WHAT MUST NOT CHANGE. A question that genuinely asks for the grain the series
publishes still answers. Refusing weekly questions to make monthly ones honest
would trade one wrong answer for many missing ones, and the paired tests below
exist so that trade cannot be made by accident.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import execution_receipt as rc


def _receipt(*facets):
    return rc.ExecutionReceipt(measure="Balance", aggregation="sum",
                               facets=list(facets))


def _grain(label, status):
    return rc.RequestedFacet(kind=rc.KIND_TIME_GRAIN, label=label, status=status,
                             reason=("this answer is reported at week level, "
                                     "not by %s" % label))


# --------------------------------------------- the grain that cannot be given #
def test_a_month_on_month_question_cannot_answer_from_a_weekly_series():
    verdict, message = rc.assess(_receipt(_grain("month", rc.UNSUPPORTED)))
    assert verdict == rc.VERDICT_REFUSE
    assert "month" in message
    assert "not substituted a broader figure" in message


def test_the_refusal_names_the_grain_rather_than_failing_vaguely():
    _, message = rc.assess(_receipt(_grain("quarter", rc.UNSUPPORTED)))
    assert "quarter" in message
    assert "week level" in message


def test_every_unhonoured_grain_status_blocks_not_only_unsupported():
    """A grain the series could not express refuses however it failed —
    unavailable, rejected or unsupported are all "you asked for month and got
    week"."""
    for status in (rc.UNSUPPORTED, rc.UNAVAILABLE, rc.REJECTED, rc.LOST):
        verdict, _ = rc.assess(_receipt(_grain("month", status)))
        assert verdict == rc.VERDICT_REFUSE, status


# ------------------------------------- the grain the reader actually asked for #
def test_a_weekly_question_answered_weekly_still_answers():
    """THE PAIR. The whole risk of this change is refusing sound weekly
    questions to make monthly ones honest."""
    verdict, message = rc.assess(_receipt(_grain("week", rc.APPLIED)))
    assert verdict == rc.VERDICT_OK
    assert message is None


def test_a_question_naming_no_grain_is_untouched():
    verdict, _ = rc.assess(_receipt())
    assert verdict == rc.VERDICT_OK


def test_an_applied_grain_beside_an_applied_comparison_still_answers():
    verdict, _ = rc.assess(_receipt(
        _grain("week", rc.APPLIED),
        rc.RequestedFacet(kind=rc.KIND_COMPARISON_PERIOD, label="last week",
                          status=rc.APPLIED)))
    assert verdict == rc.VERDICT_OK


# ------------------------------------------------- the invariant, stated once #
def test_a_grain_is_a_material_facet_not_a_shape_one():
    """The classification itself, so the trade cannot be quietly reversed."""
    assert rc.KIND_TIME_GRAIN in rc.NUMBER_OR_SUBJECT_FACETS
    assert rc.KIND_TIME_GRAIN not in rc.SHAPE_FACETS


def test_partial_remains_available_for_a_genuinely_cosmetic_facet():
    """PARTIAL is not abolished — it is reserved. A grouping the answer could
    not draw leaves the population, the measure and the period untouched."""
    assert rc.SHAPE_FACETS, "no facet may remain disclosable at all"
