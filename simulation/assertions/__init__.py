"""simulation.assertions — what each production stage must have produced.

One module per verified stage. Every module returns a list of
:class:`~simulation.models.Assertion` records rather than raising, so a run
produces a complete evidence file instead of stopping at the first
disagreement — and so a failure can be CLASSIFIED (see
:class:`~simulation.models.FailureClass`) rather than surfacing as a traceback.

The assertions compare production output against
:mod:`simulation.reference_truth`, which never imports the production
calculation it is checking.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..models import Assertion

#: Relative tolerance for a money figure that has been through a CSV round-trip
#: and a float sum. Two-decimal currency over ~100k loans cannot be compared
#: exactly, and pretending otherwise would make the suite flap.
MONEY_RTOL = 1e-6
#: Absolute floor, so a near-zero expected value does not demand exactness.
MONEY_ATOL = 0.05
#: Percentage points, for weighted averages and shares.
PERCENT_ATOL = 0.01


def approx(actual: Optional[float], expected: Optional[float], *,
           rtol: float = MONEY_RTOL, atol: float = MONEY_ATOL) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    return abs(float(actual) - float(expected)) <= max(
        atol, rtol * abs(float(expected)))


def check(stage: str, name: str, passed: bool, *, expected: Any = None,
          actual: Any = None, detail: str = "",
          failure_class: Optional[str] = None) -> Assertion:
    return Assertion(stage=stage, name=name, passed=bool(passed),
                     expected=expected, actual=actual, detail=detail,
                     failure_class=None if passed else failure_class)


def compare(stage: str, name: str, actual: Optional[float],
            expected: Optional[float], *, rtol: float = MONEY_RTOL,
            atol: float = MONEY_ATOL, detail: str = "",
            failure_class: Optional[str] = None) -> Assertion:
    return check(stage, name, approx(actual, expected, rtol=rtol, atol=atol),
                 expected=expected, actual=actual, detail=detail,
                 failure_class=failure_class)


def failures(assertions: List[Assertion]) -> List[Assertion]:
    return [a for a in assertions if not a.passed]


__all__ = ["MONEY_ATOL", "MONEY_RTOL", "PERCENT_ATOL", "approx", "check",
           "compare", "failures"]
