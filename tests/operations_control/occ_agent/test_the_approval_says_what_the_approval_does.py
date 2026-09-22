"""The consent record has to describe the consent being given.

Three things the review package said about activating a live case were not
what activating it does. None of them changed what the platform did — all
three are the record of what an approver agreed to, which on a governance
platform is the product.

1.  "Approving this activation does not approve any mapping." It does.
    ``confirm_activation`` carries the run's settled decisions into the
    governed store, where they become the standing reading of every future
    delivery for that client. The sentence was true when a mapping was first
    proposed during the first live ingestion, and became false when the
    rehearsal started settling them — so it told an approver they were signing
    a SMALLER thing than they were signing, which is the one direction a
    consent record must never be wrong in.

2.  "What activation would do" listed four things and did five or six.
    ``build_intent`` was never passed the decisions or the field requests, so
    it could not have named either even in principle: not the mappings written
    into the governed rules, and not the draft configuration version the
    requested fields are raised as. Its fourth line also said the agent would
    "map" the delivery, where the mapping had already been decided.

3.  A sample that changed could not change what it had settled. ``_set``
    fills a blank and never overwrites, which is right for a default meeting
    an answer somebody gave and wrong for a value derived from the sample: it
    cannot tell "a human decided this" from "Trakt worked this out earlier,
    from less". A client who supplied one file and then supplied three went on
    being expected to send one — the inference read all three and declined to
    write them down — and no operator action could shift it.
"""

from __future__ import annotations

from operations_control.occ_agent.adapters import build_intent
from operations_control.occ_agent.review import (
    MAPPING_NOTE_NONE,
    mapping_note,
)
from operations_control.onboarding import inference


class _Facts:
    client_id = "ERE"
    client_name = "ERE Funding Limited"
    portfolio_id = "direct_001"
    dataset = "funded"
    outcome = "mi"
    cadence = "monthly"


class _Case:
    """Only what build_intent reads off a case (nothing, today)."""


FILES = [{"name": "LoanExtract One - OMNI 2026_09_01.xlsx", "target": "raw/a"},
         {"name": "Principal And Interest - OMNI 2026_09_01.xlsx",
          "target": "raw/b"},
         {"name": "PropertyExtract - Omni 2026_09_01.xlsx", "target": "raw/c"}]


def _intent(**kw):
    return build_intent(_Case(), _Facts(), reporting_period="2026-08",
                        files=FILES, configuration_artefacts=["a", "b", "c", "d"],
                        **kw)


class TestTheMappingNoteMatchesWhatIsBeingApproved:

    def test_settled_mappings_are_named_as_part_of_the_approval(self):
        said = mapping_note(71, "ERE Funding Limited")
        assert "ARE part of this approval" in said
        assert "ERE Funding Limited's governed rules" in said

    def test_it_never_claims_a_mapping_is_not_approved_when_one_is(self):
        assert "does not approve any mapping" not in mapping_note(1, "ERE")
        assert "were not collected" not in mapping_note(1, "ERE")

    def test_a_case_with_no_settled_mappings_reads_as_it_always_did(self):
        assert mapping_note(0, "ERE") == MAPPING_NOTE_NONE
        assert "none are approved here" in mapping_note(0, "ERE")

    def test_a_client_with_no_name_still_reads(self):
        assert "this client's governed rules" in mapping_note(3, "")


class TestTheActionListNamesEveryThingActivationDoes:

    def test_it_says_the_mappings_would_be_written(self):
        actions = " ".join(_intent(mappings=71).actions)
        assert "71 confirmed field mapping(s)" in actions
        assert "ERE's governed rules" in actions

    def test_it_says_the_field_requests_become_a_draft(self):
        actions = " ".join(_intent(field_requests=2).actions)
        assert "2 requested field(s)" in actions
        assert "DRAFT system configuration version" in actions
        assert "Nothing is added to the registry by activating." in actions

    def test_it_stops_claiming_the_agent_will_map_what_is_already_mapped(self):
        assert "profile, map," not in " ".join(_intent(mappings=71).actions)
        assert "against those mappings" in " ".join(_intent(mappings=71).actions)

    def test_a_case_with_nothing_settled_reads_exactly_as_before(self):
        actions = _intent().actions
        assert len(actions) == 4
        assert "profile, map, transform, validate and assemble" in actions[3]

    def test_the_counts_are_on_the_record_not_only_in_the_prose(self):
        """The note and the action list must be counted once, not twice."""
        intent = _intent(mappings=71, field_requests=2)
        assert intent.mappings == 71
        assert intent.field_requests == 2
        assert intent.to_dict()["mappings"] == 71

    def test_the_other_four_actions_are_untouched(self):
        actions = _intent(mappings=71, field_requests=2).actions
        joined = " ".join(actions)
        assert "Write 4 configuration artefact(s) for ERE" in joined
        assert "Register the expected source deliveries" in joined
        assert "Place 3 file(s) in the production raw location" in joined


class _SampleCase:
    def __init__(self, sample, sources):
        self.answers = {"sample": sample, "sources": sources}
        self.provenance: dict = {}

    def items(self, key):
        return self.answers.get(key) or []

    def block(self, key):
        return self.answers.setdefault(key, {})


def _sample(*names):
    return {"files": [{"name": n, "headers": ["Loan Policy Number"]}
                      for n in names]}


ONE = "LoanExtract One - OMNI 2026_09_01.xlsx"
TWO = "Principal And Interest - OMNI 2026_09_01.xlsx"
THREE = "PropertyExtract - Omni 2026_09_01.xlsx"


class TestANewSampleChangesWhatTheSampleSettled:

    def _refresh(self, sample, sources):
        case = _SampleCase(sample, sources)
        inference.refresh_from_sample(case)
        return case

    def test_three_files_replace_a_one_file_expectation(self):
        sources = [{"portfolio_id": "direct_001", "dataset": "funded",
                    "expected_files": [ONE]}]
        self._refresh(_sample(ONE, TWO, THREE), sources)
        assert sources[0]["expected_files"] == [ONE, TWO, THREE], (
            "a delivery that sends three files was still registered as one")

    def test_the_record_says_where_the_new_list_came_from(self):
        sources = [{"portfolio_id": "direct_001", "dataset": "funded",
                    "expected_files": [ONE]}]
        case = self._refresh(_sample(ONE, TWO), sources)
        assert case.provenance["sources[0].expected_files"] == (
            "the files in the sample the client supplied")

    def test_a_changed_file_format_is_taken_too(self):
        sources = [{"portfolio_id": "direct_001", "dataset": "funded",
                    "file_format": "csv"}]
        self._refresh(_sample("book.xlsx"), sources)
        assert sources[0]["file_format"] == "xlsx"

    def test_a_pipeline_book_is_not_spoken_for_by_the_funded_sample(self):
        """It carries different files; this pack says nothing about it."""
        sources = [{"portfolio_id": "direct_001", "dataset": "pipeline",
                    "expected_files": ["pipeline.xlsx"]}]
        self._refresh(_sample(ONE, TWO, THREE), sources)
        assert sources[0]["expected_files"] == ["pipeline.xlsx"]

    def test_no_sample_changes_nothing(self):
        sources = [{"portfolio_id": "direct_001", "dataset": "funded",
                    "expected_files": [ONE]}]
        self._refresh({}, sources)
        assert sources[0]["expected_files"] == [ONE]

    def test_the_ordinary_inference_pass_still_never_overwrites(self):
        """Only registering a sample refreshes. Every other save is unchanged."""
        sources = [{"portfolio_id": "direct_001", "dataset": "funded",
                    "expected_files": [ONE]}]
        case = _SampleCase(_sample(ONE, TWO, THREE), sources)
        inference._sources(case, {"cadence": {"monthly": "calendar_month_end"}},
                           {})
        assert sources[0]["expected_files"] == [ONE]
