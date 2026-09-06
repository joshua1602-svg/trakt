#!/usr/bin/env python3
"""The census is an instrument, so this file tests the instrument.

Two jobs, and the second is the one that matters.

1. THE BASELINE HOLDS. Every question in the corpus means what the checked-in
   artifact says it means. A semantic change is expected to move some of them —
   the failure prints exactly which, and the fix is to look at each movement,
   decide it is intended, and rewrite the artifact in the SAME commit, so the
   movement appears in that commit's diff where a reviewer will see it.

2. THE INSTRUMENT CAN STILL SEE. A census is only worth its runtime while it
   records what decides an answer, and this one has been too narrow twice:

     * recording FILTERS only, it reported zero movements for a change to what
       "by" means — a change that moved DIMENSIONS.
     * resolved with no value catalogue, every categorical narrowing resolved to
       nothing, so a change to how populations are read could not move it.

   Both were silent. The tests below pin the facet list and pin the resolution
   context, so narrowing the instrument fails here rather than in six months.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.tests import semantic_census as census_mod                # noqa: E402


class TestTheBaselineHolds(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.current = census_mod.census()
        cls.baseline = json.loads(
            census_mod.BASELINE.read_text(encoding="utf-8"))

    def test_the_corpus_has_not_silently_changed_size(self):
        self.assertEqual(len(self.current), len(self.baseline))
        self.assertEqual(set(self.current), set(self.baseline))

    def test_no_question_has_moved(self):
        moved = census_mod.movements(self.baseline, self.current)
        self.assertEqual(
            moved, [],
            "the census moved. Read every line below, decide each movement is "
            "intended, then regenerate the artifact IN THIS COMMIT with\n"
            "    python -m mi_agent.tests.semantic_census --write\n\n"
            + "\n".join(moved))

    def test_nothing_in_the_corpus_raises(self):
        raised = {q: v["__error__"] for q, v in self.current.items()
                  if "__error__" in v}
        self.assertEqual(raised, {}, "the parser raised on a corpus question")


class TestTheInstrumentCanStillSee(unittest.TestCase):
    """What the census must record, pinned so it cannot be narrowed silently."""

    def test_every_answer_deciding_facet_is_recorded(self):
        sample = json.loads(census_mod.BASELINE.read_text(encoding="utf-8"))
        recorded = set()
        for interpretation in sample.values():
            recorded |= set(interpretation)
        missing = set(census_mod.FACETS) - recorded
        self.assertEqual(missing, set(),
                         "the census stopped recording a facet that decides an "
                         "answer")

    def test_it_records_dimensions_and_not_only_filters(self):
        """The exact narrowing that made an earlier census report a meaningless
        zero: a change to what "by" means moves the AXIS, not the population."""
        with_axes = [q for q, i in json.loads(
            census_mod.BASELINE.read_text(encoding="utf-8")).items()
            if i.get("dimensions")]
        self.assertGreater(len(with_axes), 50,
                           "the census sees almost no grouped questions, so it "
                           "cannot detect a change to grouping")

    def test_it_resolves_against_the_books_own_values(self):
        """Without a value catalogue no categorical narrowing resolves, and a
        change to how populations are read cannot move the census at all."""
        _semantics, columns, values = census_mod._book_context()
        self.assertGreater(len(columns), 50)
        self.assertGreater(len(values), 5,
                           "no field has a value catalogue; the census is blind "
                           "to categorical narrowings")
        with_filters = [q for q, i in json.loads(
            census_mod.BASELINE.read_text(encoding="utf-8")).items()
            if i.get("filters")]
        self.assertGreater(len(with_filters), 50,
                           "the census sees almost no narrowed questions")

    def test_the_persisted_form_is_stable(self):
        """Canonical ordering, so a diff shows a meaning that moved and never an
        iteration order that did."""
        data = json.loads(census_mod.BASELINE.read_text(encoding="utf-8"))
        self.assertEqual(census_mod.canonical_json(data),
                         census_mod.canonical_json(json.loads(
                             census_mod.canonical_json(data))))


if __name__ == "__main__":
    unittest.main()
