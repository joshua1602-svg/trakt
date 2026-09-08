#!/usr/bin/env python3
"""A governed view name is a dataset, not something to calculate.

MEASURED LIVE, 2026-09-03. "Show me the funded loan book summary by region"
was refused:

    'funded' is not a governed measure in this dataset; no substitute was
    used. [...] Ask for a governed measure — e.g. balance, LTV, interest
    rate, borrower age or property value

The deterministic parser emits no measure at all for that sentence, so the
slot came from the language layer: it read the word naming the DATASET as the
thing to calculate. `mi_query_executor` then reported it faithfully, and a
reader saw the product decline an ordinary question with a lecture about
measures.

The executor is not wrong — it reports what it was handed. The rejection
belongs where a measure list is normalised, which is one place and already
carries a notes channel.

THE SAME COLLISION, ALREADY SOLVED ONCE. `lexical.pipeline_stage_vocabulary`
drops every stage spelling that collides with a governed view name, reading
the registry so a newly registered view cannot silently reintroduce the
clash — because ``funded`` names a stage in a tape cell and a dataset in a
sentence. This is that rule, applied to the measure slot.
"""
from __future__ import annotations

import unittest

from mi_agent.mi_query_spec import MIQuerySpec, normalise_measures


def _fields(measures):
    return [m["field"] for m in measures]


class TestAViewNameIsDropped(unittest.TestCase):

    def test_the_live_failure(self):
        measures, _ = normalise_measures(["funded", "balance"])
        self.assertEqual(_fields(measures), ["balance"])

    def test_every_governed_view_name(self):
        from mi_agent_api.workspace import VIEWS
        self.assertTrue(VIEWS, "the view registry is empty; this test proves nothing")
        for view in VIEWS:
            with self.subTest(view=view):
                measures, _ = normalise_measures([view, "balance"])
                self.assertEqual(_fields(measures), ["balance"])

    def test_it_reads_the_registry_rather_than_a_hand_list(self):
        """A newly registered view must be covered without editing this rule —
        the reason `pipeline_stage_vocabulary` reads the registry too."""
        import mi_agent.mi_query_spec as S
        from unittest import mock
        with mock.patch.object(S, "_governed_view_names",
                               return_value=frozenset({"newview"})):
            measures, _ = normalise_measures(["newview", "balance"])
        self.assertEqual(_fields(measures), ["balance"])

    def test_the_drop_is_silent_because_nothing_was_lost(self):
        """The view name has already done its work selecting the dataset.
        "funded was not applied" about an answer computed from the funded book
        would disclose something that did not happen."""
        _, notes = normalise_measures(["funded", "balance"])
        self.assertEqual(notes, [])


class TestItDoesNotSwallowRealMeasures(unittest.TestCase):
    """The boundary. A rule that quietly ate a governed measure would turn a
    visible refusal into a silently narrower answer, which is worse."""

    def test_governed_measures_survive(self):
        for name in ("balance", "current_outstanding_balance",
                     "current_loan_to_value", "loan_count"):
            with self.subTest(name=name):
                measures, _ = normalise_measures([name])
                self.assertEqual(_fields(measures), [name])

    def test_an_unknown_measure_still_reaches_the_executor(self):
        """"gross margin" is not a view name. It must still travel, so the
        executor can decline it by name rather than the parser hiding it."""
        measures, _ = normalise_measures(["gross margin"])
        self.assertEqual(_fields(measures), ["gross margin"])

    def test_a_view_name_alone_leaves_no_measure_rather_than_a_wrong_one(self):
        measures, _ = normalise_measures(["funded"])
        self.assertEqual(measures, [])

    def test_the_spec_path_applies_it_too(self):
        """`from_dict` is what the language layer's response goes through."""
        spec = MIQuerySpec.from_dict({"measures": ["funded", "balance"],
                                      "dimension": "collateral_geography"})
        self.assertEqual(_fields(spec.measures), ["balance"])


if __name__ == "__main__":
    unittest.main()
