"""A finding that does not block still has to say what it found.

The validate stage said two different things depending on its verdict. A
refusal named every blocking check, in the operator's words, with what became
of the mapping behind it. A pass said:

    "1 finding(s) need review but do not block."

— a number, on a stage that had gone green, with no way to tell one rounding
disagreement from every loan in the delivery contradicting its own stated
figure.

That was survivable only while REVIEW was the quiet outcome. It is not any
more. Volume no longer promotes a warning-severity check to BLOCKING, because
a cross-check that disagrees across a whole book is a case for reading it, not
for refusing the book — so this stage is now exactly where a systematic
disagreement arrives. A count would bury it.

Both verdicts are now said in the same words, differing only in the
materiality they carry, which is the only thing that actually differs: what
the operator has to do about it.
"""

from __future__ import annotations

from operations_control.occ_agent.execution import (
    MATERIALITY_MARK,
    SyntheticOnboardingAdapters,
)


class _Stub:
    """The two things the sentence reads besides the finding itself."""
    resolved_by_file: dict = {}
    consolidation: dict = {}

    def say(self, finding):
        return SyntheticOnboardingAdapters._finding_sentence(self, finding)


LTV = {
    "field_name": "PORTFOLIO",
    "issue_type": "LTV002",
    "affected_rows": 358,
    "error_rate": 63.03,
    "materiality": "REVIEW",
}


class TestAReviewFindingSaysWhatItFound:

    def test_it_names_the_check_and_the_count(self):
        said = _Stub().say(LTV)
        assert "LTV002" in said
        assert "358 record(s)" in said
        assert "63.03%" in said

    def test_it_carries_its_own_materiality(self):
        assert "materiality REVIEW" in _Stub().say(LTV)
        assert "materiality BLOCKING" in _Stub().say(
            dict(LTV, materiality="BLOCKING"))

    def test_it_says_what_the_check_actually_tests(self):
        """The rule's own description, not the rule id on its own."""
        assert "The check:" in _Stub().say(LTV)

    def test_a_finding_with_no_stated_materiality_reads_as_review(self):
        """A missing verdict must not silently print as a blocking one."""
        said = _Stub().say({k: v for k, v in LTV.items() if k != "materiality"})
        assert "materiality REVIEW" in said

    def test_both_verdicts_are_said_the_same_way(self):
        blocking = _Stub().say(dict(LTV, materiality="BLOCKING"))
        review = _Stub().say(LTV)
        assert blocking.replace("BLOCKING", "REVIEW") == review


class TestARerunReplacesItsOwnFiguresRatherThanStackingThem:
    """Observations only ever accumulated, which a changing count cannot do.

    They were built for artefact notes — a file was recognised, and it stays
    recognised. A validation figure is the opposite: a rerun exists to change
    it. An operator who fixed a mapping and reran would otherwise be shown the
    number they had just fixed, sitting above the number they fixed it to, with
    nothing on either line to say which was current.
    """

    def _observations(self, existing, this_run):
        """The replacement the service performs, in isolation."""
        kept = [o for o in existing if MATERIALITY_MARK not in o]
        return kept + list(this_run)

    def test_last_runs_figure_does_not_survive_this_run(self):
        before = [_Stub().say(dict(LTV, affected_rows=567, error_rate=99.82))]
        after = self._observations(before, [_Stub().say(LTV)])
        assert len(after) == 1
        assert "358 record(s)" in after[0]
        assert "567 record(s)" not in " ".join(after)

    def test_an_artefact_note_is_not_a_finding_and_is_kept(self):
        note = "PropertyExtract - Omni 2026_09_01.xlsx recognised as collateral."
        after = self._observations([note], [_Stub().say(LTV)])
        assert note in after

    def test_a_run_that_finds_nothing_clears_the_last_one(self):
        before = [_Stub().say(LTV)]
        assert self._observations(before, []) == []

    def test_every_finding_carries_the_mark_that_makes_this_work(self):
        for materiality in ("REVIEW", "BLOCKING"):
            assert MATERIALITY_MARK in _Stub().say(
                dict(LTV, materiality=materiality))
