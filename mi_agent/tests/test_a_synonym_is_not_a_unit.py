#!/usr/bin/env python3
"""A field's synonyms say what a reader calls it, not what it measures.

THE DEFECT THIS LOCKS. "How many loans are over 80 years old" bound
`probability_of_default > 80` instead of the borrower's age. PD carries the
synonym "one year pd", the unit resolver matched "year" as a SUBSTRING of it,
and a question about people became a question about default probability —
silently, with a plausible number.

It survived three full regressions of ~3,600 tests. Nothing in the estate asks
about an age in years, so nothing went red. It was found only by re-running the
882-question corpus census, which is why that census is now a development
instrument rather than a closing artefact.

THE DISTINCTION THE FIX RESTS ON. A field's KEY and BUSINESS NAME say what it
measures; its SYNONYMS say what a reader might call it. Those are different
claims, and `find_field` in this module already ranks them that way — a name hit
is documented there as "the strong signal" and a synonym hit ranks below it. The
unit test now makes the same distinction, on word boundaries.

    pipeline_case_age_days      days     the key says so
    number_of_days_in_arrears   days     the key says so
    months_on_book              months   the key says so
    probability_of_default      —        "one year pd" is a synonym
    youngest_borrower_age       —        an age in years, named in neither

This is the third substring-where-a-boundary-was-meant defect found in this
programme, which is why the last class below tests the PROPERTY across every
governed field rather than the one field that failed.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.llm_query_parser import (                              # noqa: E402
    _deterministic_parse, _field_names_unit, _fields, _parse_filters)
from mi_agent.mi_query_validator import load_mi_semantics            # noqa: E402

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))

AGE = "youngest_borrower_age"
CASE_AGE = "pipeline_case_age_days"
PD = "probability_of_default"

PIPELINE_COLUMNS = ["pipeline_case_age_days", "current_outstanding_balance",
                    "youngest_borrower_age", "pipeline_stage"]


class TestBorrowerAgeBindsToBorrowerAge(unittest.TestCase):
    """The regression itself, in every phrasing that states an age in years."""

    QUESTIONS = (
        "how many loans are over 80 years old",
        "how many loans are over 75 years old",
        "how many borrowers are over 70 years old",
        "balance for borrowers over 80 years old",
        "how many funded loans have a borrower aged 85 or older",
        "how many loans have a borrower older than 70",
    )

    def test_the_age_bound_lands_on_the_borrower(self):
        for question in self.QUESTIONS:
            with self.subTest(question=question):
                filters = _parse_filters(question, _SEMANTICS)
                self.assertIn(AGE, filters, filters)
                self.assertNotIn(PD, filters)

    def test_the_spec_agrees(self):
        spec, _meta = _deterministic_parse(
            "how many loans are over 80 years old", _SEMANTICS)
        self.assertEqual(spec.filters,
                         {AGE: {"op": "gt", "value": 80.0}})


class TestCaseAgeStillBindsToTheCase(unittest.TestCase):
    """The capability the unit resolver was built for must survive the repair."""

    def test_a_bound_in_days_lands_on_the_case(self):
        filters = _parse_filters(
            "how many pipeline cases are older than 30 days?", _SEMANTICS,
            PIPELINE_COLUMNS)
        self.assertIn(CASE_AGE, filters)
        self.assertNotIn(AGE, filters)


class TestASynonymCannotClaimAUnit(unittest.TestCase):
    """The property, over every governed field — not the one that failed."""

    #: Units whose word can plausibly appear inside an unrelated synonym.
    UNITS = ("days", "weeks", "months", "years")

    def test_no_field_claims_a_unit_it_is_not_named_for(self):
        fields = _fields(_SEMANTICS)
        for key, entry in fields.items():
            for unit in self.UNITS:
                if not _field_names_unit(key, entry or {}, unit):
                    continue
                with self.subTest(field=key, unit=unit):
                    # The claim must be visible in the field's own NAME.
                    haystack = " ".join([
                        key, str((entry or {}).get("business_name") or ""),
                        str((entry or {}).get("display_name") or ""),
                    ]).lower().replace("_", " ")
                    self.assertIn(unit.rstrip("s"), haystack)

    def test_probability_of_default_does_not_claim_years(self):
        """Named because it is the field that did, via 'one year pd'."""
        entry = _fields(_SEMANTICS).get(PD) or {}
        self.assertIn("one year pd",
                      [s.lower() for s in (entry.get("synonyms") or ())])
        self.assertFalse(_field_names_unit(PD, entry, "years"))

    def test_a_synonym_containing_a_unit_word_does_not_confer_it(self):
        """Stated generally: for every field, a unit word appearing ONLY in its
        synonyms must not make it that unit's owner."""
        fields = _fields(_SEMANTICS)
        for key, entry in fields.items():
            synonyms = " ".join((entry or {}).get("synonyms") or ()).lower()
            name = " ".join([key, str((entry or {}).get("business_name") or "")
                             ]).lower().replace("_", " ")
            for unit in self.UNITS:
                stem = unit.rstrip("s")
                if stem in synonyms and stem not in name:
                    with self.subTest(field=key, unit=unit):
                        self.assertFalse(
                            _field_names_unit(key, entry or {}, unit),
                            f"{key} claims {unit} from a synonym alone")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
