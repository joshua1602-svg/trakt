"""A vague recency resolves to the governed window — never to the widest one.

The defect this pins
--------------------
"How has the profile of new originations changed in the last few months?" was
answered over TWELVE months on a thirteen-snapshot book. ``requested_span``
returned None for a vague span and the analytical route fell through to
WINDOW_RETAINED — earliest snapshot to latest.

It read as correct for the whole of V1 only because the fixture had three
snapshots, so the widest window happened to be two months, which passes for
"a few". That is the general hazard: a fallback whose output coincides with the
right answer on the fixture you test against.

The ruling implemented here: resolve vague recency to the governed
``seasoning.lending_windows.recent_max_months`` and disclose it. Not clarify —
the governed configuration exists to settle exactly this, and clarifying on a
phrase this common makes the product tiresome. Clarification stays reserved for
spans with no governed convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mi_agent import period_request as pr  # noqa: E402

#: Near-identical phrasings. Handling these differently from one another would
#: be worse than either choice, so they are asserted together.
VAGUE = [
    "How has the profile of new originations changed in the last few months?",
    "Are we originating different types of loans now compared with a few months ago?",
    "How does recent lending compare with what we were originating earlier?",
    "Has the risk and borrower profile of new business changed recently?",
    "what has changed lately",
    "how has the book moved in recent months",
    "what has happened over the couple of months just gone",
]


@pytest.mark.parametrize("question", VAGUE)
def test_vague_recency_resolves_to_the_governed_window(question):
    span = pr.requested_span(question)
    assert span is not None, "a vague recency must not fall through to no span"
    assert span.governed is True
    assert span.periods == pr.governed_recent_months()


@pytest.mark.parametrize("question", VAGUE)
def test_a_governed_resolution_is_disclosed(question):
    """Resolving is allowed; resolving silently is not."""
    span = pr.requested_span(question)
    text = span.disclosure()
    assert text and "governed recent window" in text
    assert f"last {span.periods} months" in text


def test_the_governed_window_comes_from_config_not_from_code():
    class _Cfg:
        recent_max_months = 5
    span = pr.requested_span("how is recent lending doing", config=_Cfg())
    assert span.periods == 5, "the window must follow the client's configuration"
    assert "last 5 months" in span.disclosure()


@pytest.mark.parametrize("question,periods", [
    ("How much has the book grown this year?", 12),
    ("over the last three months", 3),
    ("what changed over the last 6 months", 6),
    ("month on month movement", 1),
])
def test_a_counted_span_is_unaffected_and_is_not_marked_governed(question, periods):
    span = pr.requested_span(question)
    assert span is not None and span.periods == periods
    assert span.governed is False, "the question counted it; no convention was applied"
    assert span.disclosure() is None


def test_last_three_months_is_read_as_counted_not_as_vague():
    """Ordering matters: the vague scan runs last, or it would swallow this."""
    span = pr.requested_span("how did we do over the last three months")
    assert span.periods == 3 and span.governed is False


@pytest.mark.parametrize("question", [
    "Based on the last few weeks, what level of completions are we achieving?",
    "how has the balance from direct lending changed relative to acquired lending?",
])
def test_no_governed_convention_means_no_span_is_invented(question):
    """A vague WEEK span and a question with no period both stay at None.

    Weeks have no governed window, so they belong to the granularity
    clarification; a question naming no period at all has nothing to honour.
    """
    assert pr.requested_span(question) is None
