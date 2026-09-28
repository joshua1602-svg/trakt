"""`confidence: float = 0.0` is a promise, and a dataclass does not keep it.

The live delivery's third halt, recorded in `09_onboarding_run_summary.json`
and nowhere else::

    "mapping_review_summary": {
        "error": "'<=' not supported between instances of 'str' and 'float'"
    }

`llm_mapping_reviewer._unresolved_candidates` asks::

    low_conf = m.confidence <= policy.deterministic_confidence_above

and candidates are rebuilt straight from artefacts::

    MappingCandidate(**{k: v for k, v in m.items()
                        if k in MappingCandidate.__dataclass_fields__})

A dataclass annotation converts nothing, so whatever the JSON holds arrives as
it was written — and this codebase writes a confidence THREE ways:

  * a number, 0.0–1.0, in `05_mapping_candidates`;
  * a word — `"high"`, `"medium"`, `"low"`, `"no_match"` — in
    `mapping_backstop_validator`, which grades the number into a band;
  * a deliberate blank for "not applicable": `mapping_trace` records
    ``semantic_alignment_confidence: ""`` for every column semantic alignment
    did not decide. The live pack had 112 of them.

THE COST WAS FOUR ARTEFACTS AND A NIGHT. That comparison is inside the stage
that writes `28a_target_coverage_matrix`, `28c_human_decision_queue` and
`34_target_first_decisions` — three of the four artefacts
`build_workflow_summary` requires. Its caller catches everything ("never break
the onboarding run on review failure"), so the run continued, finished, and
reported FAILED with an EMPTY `run_error` and three required artefacts missing.
The operator was told "The first step did not finish cleanly. Try running
again", which is the least informative state this system can produce: a real
fault, no message, no file, no line.

So the type is made true at the boundary, once, for every producer.
"""

from __future__ import annotations

import json

import pytest

from engine.onboarding_agent.llm_mapping_reviewer import _unresolved_candidates
from engine.onboarding_agent.onboarding_models import MappingCandidate


class _Policy:
    deterministic_confidence_above = 0.9
    zero_cost_first = True


class TestTheFailureItself:

    def test_a_blank_confidence_no_longer_stops_the_mapping_review(self):
        """`mapping_trace` writes "" for a column semantic alignment skipped."""
        candidates = [MappingCandidate(source_column="Loan Policy Number",
                                       confidence="", requires_review=True)]
        unresolved, _skips = _unresolved_candidates(candidates, _Policy())
        assert len(unresolved) == 1

    def test_the_comparison_is_between_two_numbers(self):
        c = MappingCandidate(source_column="Month Run", confidence="")
        assert isinstance(c.confidence, float)
        assert (c.confidence <= _Policy.deterministic_confidence_above) is True


class TestEveryWayThisCodebaseWritesAConfidence:

    @pytest.mark.parametrize("written,read", [
        # the number
        (0.95, 0.95), (0.0, 0.0), (1, 1.0), ("0.42", 0.42),
        # the band, as `mapping_backstop_validator` grades it
        ("high", 0.95), ("medium", 0.85), ("low", 0.5),
        ("no_match", 0.0), ("none", 0.0), ("unmapped", 0.0),
        # "not applicable"
        ("", 0.0), (None, 0.0), ("   ", 0.0),
        # a percentage, as a spreadsheet hands it over
        ("95%", 0.95),
    ])
    def test_it_is_read_as_the_number_it_means(self, written, read):
        assert MappingCandidate(confidence=written).confidence == pytest.approx(read)

    def test_a_word_keeps_its_ordering_against_the_threshold(self):
        """The bands must still sort the way their names do."""
        conf = lambda w: MappingCandidate(confidence=w).confidence  # noqa: E731
        assert conf("no_match") < conf("low") < conf("medium") < conf("high")

    def test_something_unreadable_is_not_confident(self):
        """Never invent confidence out of a value nobody can read."""
        assert MappingCandidate(confidence="rubbish").confidence == 0.0

    def test_a_flag_is_not_a_confidence(self):
        """`True` is an int in Python; 1.0 would be total confidence."""
        assert MappingCandidate(confidence=True).confidence == 0.0


class TestRebuildingFromAnArtefactIsTheRealPath:
    """How the string got in: the constructor nobody type-checks."""

    def test_a_candidate_rebuilt_from_json_holds_a_float(self):
        written = json.dumps([
            {"source_column": "Loan Policy Number", "confidence": ""},
            {"source_column": "Current Balance", "confidence": "high"},
            {"source_column": "Month Run", "confidence": 0.8},
        ])
        rebuilt = [
            MappingCandidate(**{k: v for k, v in m.items()
                                if k in MappingCandidate.__dataclass_fields__})
            for m in json.loads(written)]
        assert all(isinstance(c.confidence, float) for c in rebuilt)

    def test_the_whole_rebuilt_set_survives_the_reviewer(self):
        """The live shape: a pack whose columns were not all decided."""
        rebuilt = [MappingCandidate(source_column=f"col_{i}", confidence=v,
                                    requires_review=True)
                   for i, v in enumerate(("", "high", "low", 0.5, None, "no_match"))]
        unresolved, skips = _unresolved_candidates(rebuilt, _Policy())
        assert len(unresolved) + skips == len(rebuilt)

    def test_a_round_trip_through_to_dict_is_stable(self):
        once = MappingCandidate(source_column="x", confidence="high")
        twice = MappingCandidate(**once.to_dict())
        assert twice.confidence == once.confidence == 0.95
