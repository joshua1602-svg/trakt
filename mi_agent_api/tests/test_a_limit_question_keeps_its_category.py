#!/usr/bin/env python3
"""A limits question the BOUNDARY hands over keeps the category it named.

WHY THIS EXISTS NOW. "What is the largest geographic concentration versus
limit?" used to be refused for an invented category, and the vocabulary fix that
removed that refusal is what makes this reachable: the question now reaches the
`risk_limits` route, which answers.

But the route is handed `risk_limit_category = None`, so it answers with EVERY
limit category — measured: *"5 passed, 0 warning(s), 6 breach(es) … Nearest to
limit: Top 3 brokers"* for a question about the GEOGRAPHIC one. An answer
covering a broader population than the question named is the substitution this
estate refuses everywhere else, and it is only reachable because the refusal in
front of it was removed. Fixing the first without the second would trade a wrong
refusal for a quiet widening.

WHERE THE GAP IS. `_risk_limit_recognizer` sets the flag AND the category from
`_risk_limit_category`. The analytical intent boundary — which is what claims
this question, because `_RISK_LIMIT_RE` never matches it — sets the flag alone.
`_route_risk` already scopes to the category and already refuses honestly when
a category has no configured tests; nothing there needed changing.

So the boundary now settles the category from the PARSER'S OWN reader, under
the same three rules it already keeps: never override a settled parse, never
invent an analytic, never widen a measure.
"""
from __future__ import annotations

import unittest

from mi_agent import llm_query_parser as P
from mi_agent.mi_query_spec import MIQuerySpec
from mi_workflows.analytical import intent as AI

GEOGRAPHIC = "What is the largest geographic concentration versus limit?"


def _spec(**kw) -> MIQuerySpec:
    return MIQuerySpec(intent="summary", chart_type="none", aggregation="count",
                       title="t", **kw)


class TestTheBoundarySettlesTheCategory(unittest.TestCase):

    def test_the_reader_is_the_parser_s_own(self):
        """One reader. A second opinion about what category a question names is
        how two phrasings of one question reach two different answers."""
        self.assertEqual(P.risk_limit_category(GEOGRAPHIC),
                         "geographic_concentration")
        self.assertEqual(P.risk_limit_category(GEOGRAPHIC),
                         P._risk_limit_category(GEOGRAPHIC.lower()))

    def test_the_question_the_recogniser_never_claimed(self):
        """The premise of all of this: this sentence does not match
        `_RISK_LIMIT_RE`, so the parser sets neither flag nor category."""
        self.assertIsNone(P._RISK_LIMIT_RE.search(GEOGRAPHIC.lower()))

    def test_the_boundary_sets_flag_and_category_together(self):
        spec = _spec()
        _reading, flags = AI.settle(GEOGRAPHIC, spec)
        self.assertTrue(flags.get("risk_limit_query"))
        self.assertEqual(flags.get("risk_limit_category"),
                         "geographic_concentration")
        self.assertEqual(getattr(spec, "risk_limit_category"),
                         "geographic_concentration")

    def test_a_settled_category_is_never_overridden(self):
        spec = _spec(risk_limit_query=True,
                     risk_limit_category="broker_concentration")
        _reading, flags = AI.settle(GEOGRAPHIC, spec)
        self.assertNotIn("risk_limit_category", flags)
        self.assertEqual(getattr(spec, "risk_limit_category"),
                         "broker_concentration")

    def test_a_limits_question_naming_no_category_settles_none(self):
        """Nothing is invented. A question that scopes to no category leaves the
        route to summarise every one of them, as it does today."""
        spec = _spec()
        _reading, flags = AI.settle("Are we within our limits?", spec)
        self.assertTrue(flags.get("risk_limit_query")
                        or getattr(spec, "risk_limit_query", False))
        self.assertIsNone(getattr(spec, "risk_limit_category", None))

    def test_a_question_of_another_family_is_untouched(self):
        spec = _spec()
        _reading, flags = AI.settle("Show balance by region.", spec)
        self.assertNotIn("risk_limit_category", flags)
        self.assertIsNone(getattr(spec, "risk_limit_category", None))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
