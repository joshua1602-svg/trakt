"""Change 1 — the grain reading reaching a facet, once, in the facet layer.

The inventory found eleven entry points reading the raw question and the
subject-side clause split implemented three times from vocabularies that agreed
by maintenance rather than construction. The obvious short path to making
time-series questions work was to call `requested_unit` inside `evolution`. That
would have been a twelfth reader and would have defeated the premise, so the
call lives in `granularity_facets` and every route inherits the result through
the facet it produces.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from mi_agent import execution_receipt as R

_REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# One reader
# --------------------------------------------------------------------------- #
def test_the_grain_is_read_in_exactly_one_place_on_the_serving_path():
    """`requested_unit` must not gain a second serving-path caller.

    Counted from the source rather than trusted, because the failure mode is a
    well-meant copy added later. `chat_routing`'s forecast branch is the one
    pre-existing caller and is grandfathered by name — that call is what change
    1 generalises, and removing it is a separate change with its own
    measurement.
    """
    class _Calls(ast.NodeVisitor):
        def __init__(self):
            self.hits = []

        def visit_Call(self, node):
            fn = node.func
            name = (fn.attr if isinstance(fn, ast.Attribute)
                    else fn.id if isinstance(fn, ast.Name) else None)
            if name == "requested_unit":
                self.hits.append(node.lineno)
            self.generic_visit(node)

    callers = []
    for root in ("mi_agent", "mi_agent_api", "mi_workflows"):
        for path in sorted((_REPO_ROOT / root).rglob("*.py")):
            if "test" in path.name:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:                              # pragma: no cover
                continue
            visitor = _Calls()
            visitor.visit(tree)
            for line in visitor.hits:
                callers.append("%s:%d" % (path.relative_to(_REPO_ROOT), line))

    # Parsed, not grepped. The first draft of this matched any line containing
    # the name and failed on its own docstring — the same defect a source-check
    # test in Stage 3 had, where it matched a COMMENT describing the code it was
    # asserting about. A test that reads prose is not reading the program.
    callers = [c for c in callers if "period_request.py" not in c]
    grandfathered = [c for c in callers if "chat_routing.py" in c]
    fresh = [c for c in callers if "chat_routing.py" not in c]
    assert len(grandfathered) <= 1, grandfathered
    assert len(fresh) == 1, (
        "the grain must be read in ONE place on the serving path; found %d: %s"
        % (len(fresh), fresh))
    assert "execution_receipt.py" in fresh[0], fresh


def test_the_facet_layer_is_the_place_it_is_read():
    src = inspect.getsource(R.granularity_facets)
    assert "requested_unit" in src


# --------------------------------------------------------------------------- #
# What the reader produces
# --------------------------------------------------------------------------- #
def test_it_returns_both_grains_a_question_names():
    """Spatial and temporal are separate claims and must not displace one
    another — a question can get one wrong and the other right."""
    facets = R.granularity_facets("exposure by postcode", "geo_exposure")
    assert [f.label for f in facets] == ["postcode"]

    facets = R.granularity_facets("balance by week", "evolution")
    assert [f.label for f in facets] == ["week"]


def test_a_question_with_no_grain_produces_nothing():
    """Can-fail: a reader that always produced a facet would pass the above."""
    assert R.granularity_facets("balance by region", "evolution") == []
    assert R.granularity_facets("what is the total balance", None) == []


# --------------------------------------------------------------------------- #
# `day`, which the owner did not recognise
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question", [
    "show me daily funded balance", "balance day by day", "balance per day",
    "balance by day",
])
def test_a_daily_grain_is_read(question):
    from question_interpretation.lexical import requested_unit
    assert requested_unit(question) == "day"


@pytest.mark.parametrize("question", [
    "Show loans more than 90 days in arrears.",
    "Show number of days in arrears by broker.",
    "what is the balance today",
])
def test_arrears_vocabulary_is_not_a_grain(question):
    """The can-fail half, and the reason `day` matches only constructions.

    Drafted as the bare noun it moved five corpus questions and every one was
    arrears, where "days" is part of a MEASURE. Zero were daily-series requests.
    In this domain the bare noun is overwhelmingly a duration — which is not
    true of the other units' bare nouns, so `day` needs the construction and
    they do not.
    """
    from question_interpretation.lexical import requested_unit
    assert requested_unit(question) != "day"


def test_day_is_the_finest_unit():
    from question_interpretation.lexical import finer_than
    assert finer_than("day", "week")
    assert finer_than("day", "month")
    assert not finer_than("month", "day")


# --------------------------------------------------------------------------- #
# The window, separately from the grain
# --------------------------------------------------------------------------- #
def _envelope(periods):
    return {"artifacts": [{"rows": [{"period": "2026-%02d" % (i + 1),
                                     "value": 1} for i in range(periods)]}]}


def test_a_window_longer_than_the_book_downgrades_the_period_facet():
    applied = R.RequestedFacet(kind=R.KIND_COMPARISON_PERIOD,
                               label="comparison period", status=R.APPLIED)
    out = R.check_window_coverage([applied], _envelope(3),
                                  "balance by month over the last 12 months",
                                  "evolution")
    assert out[0].status == R.UNSUPPORTED
    assert "the last 12 months" in out[0].reason
    assert "covers 3" in out[0].reason


def test_a_window_the_book_covers_is_left_alone():
    """Can-fail: downgrading unconditionally would pass the test above."""
    applied = R.RequestedFacet(kind=R.KIND_COMPARISON_PERIOD,
                               label="comparison period", status=R.APPLIED)
    out = R.check_window_coverage([applied], _envelope(3),
                                  "balance by month over the last 3 months",
                                  "evolution")
    assert out[0].status == R.APPLIED


def test_a_route_declaring_no_series_is_never_downgraded():
    """Never refused on an unverifiable basis."""
    applied = R.RequestedFacet(kind=R.KIND_COMPARISON_PERIOD,
                               label="comparison period", status=R.APPLIED)
    out = R.check_window_coverage([applied], {"artifacts": []},
                                  "balance over the last 12 months", "evolution")
    assert out[0].status == R.APPLIED


def test_the_window_shortfall_is_raised_even_with_no_period_facet():
    """Otherwise a question naming a window nothing represents goes unsaid."""
    out = R.check_window_coverage([], _envelope(3),
                                  "funded balance over the last 6 months",
                                  "evolution")
    assert [f.kind for f in out] == [R.KIND_COMPARISON_PERIOD]
    assert out[0].status == R.UNSUPPORTED


# --------------------------------------------------------------------------- #
# The two defects stay apart
# --------------------------------------------------------------------------- #
def test_a_grain_substitution_and_a_coverage_limit_say_different_things():
    """They are owed different sentences. Telling a reader their weekly request
    was answered monthly does not tell them their twelve-month window was
    three."""
    grain = R.time_axis_disclosure("week", "evolution")
    grain = R.reconcile_routed_facets([grain], route="evolution",
                                      semantics={"fields": {}},
                                      available_columns=None, envelope={})[0]
    window = R.check_window_coverage([], _envelope(3),
                                     "balance over the last 12 months",
                                     "evolution")[0]
    assert "reported at month level" in grain.reason
    assert "the last 12 months" in window.reason
    assert grain.reason != window.reason
