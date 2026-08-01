"""Pipeline versus funded: two operational data streams, never one file list.

The instruction the OCC Agent must understand verbatim:

    Onboard Capital 123. They will provide a pipeline and a funded book.

must produce TWO separate source registrations on the onboarding case — a
funded stream and a pipeline stream — through Client Onboarding's own
``sources`` rows, with the platform's regime rule intact: pipeline never feeds
a regime, funded may where product, asset class and configuration permit.
"""

from __future__ import annotations

from operations_control.contracts import REGIME_CAPABLE_DATASETS
from operations_control.occ_agent import derive as _derive

from .conftest import ACTOR, TENANT_A

INSTRUCTION = ("Onboard Capital 123. They will provide a pipeline and a "
               "funded book.")


def _case(service, instruction=INSTRUCTION):
    return service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction=instruction)


# --------------------------------------------------------------------------- #
# The exact required instruction
# --------------------------------------------------------------------------- #

def test_capital_123_keeps_its_full_name(service):
    agent_case = _case(service)
    assert agent_case.case.client_name == "Capital 123"


def test_capital_123_creates_two_separate_stream_registrations(service):
    agent_case = _case(service)
    sources = agent_case.case.items("sources")
    datasets = sorted(str(s.get("dataset")) for s in sources)
    assert datasets == ["funded", "pipeline"]
    # Both registrations belong to the same book, separately keyed.
    keys = {str(s.get("source_key")) for s in sources}
    assert len(keys) == 2


def test_the_two_streams_are_not_collapsed_into_one_delivery_answer(service):
    agent_case = _case(service)
    pipeline = next(s for s in agent_case.case.items("sources")
                    if s.get("dataset") == "pipeline")
    funded = next(s for s in agent_case.case.items("sources")
                  if s.get("dataset") == "funded")
    assert pipeline.get("dataset") == "pipeline"
    assert funded.get("dataset") == "funded"


def test_pipeline_is_never_regime_required(service):
    agent_case = _case(service)
    pipeline = next(s for s in agent_case.case.items("sources")
                    if s.get("dataset") == "pipeline")
    assert pipeline.get("regime_required") is False
    assert "pipeline" not in REGIME_CAPABLE_DATASETS


def test_a_blanket_delivery_answer_does_not_overwrite_stream_identity(service):
    agent_case = _case(service)
    agent_case = service.answer_from_instruction(
        agent_case, instruction="They deliver weekly.", actor=ACTOR)
    sources = agent_case.case.items("sources")
    assert sorted(str(s.get("dataset")) for s in sources) == \
        ["funded", "pipeline"]
    assert all(s.get("cadence") == "weekly" for s in sources)


# --------------------------------------------------------------------------- #
# The stream summary projection
# --------------------------------------------------------------------------- #

def test_stream_summaries_report_pipeline_as_mi_only(service):
    agent_case = _case(service)
    streams = _derive.stream_summaries(agent_case.case,
                                       service.onboarding.catalogue)
    by_dataset = {s["dataset"]: s for s in streams}
    assert set(by_dataset) == {"funded", "pipeline"}
    pipeline = by_dataset["pipeline"]
    assert pipeline["purpose"] == "Pipeline MI only"
    assert pipeline["regime_capable"] is False
    assert pipeline["regime_status"] == "not_applicable"
    funded = by_dataset["funded"]
    assert funded["regime_capable"] is True
    assert funded["regime_status"] in ("potential", "configured",
                                       "not_eligible")


def test_stream_summaries_travel_on_the_status_projection(service):
    agent_case = _case(service)
    status = service.status(agent_case)
    assert {s["dataset"] for s in status["streams"]} == {"funded", "pipeline"}


def test_the_conversation_reports_the_streams_separately(service):
    agent_case = _case(service)
    replies = [m["text"] for m in agent_case.run.messages
               if m["role"] == "agent"]
    assert any("pipeline" in text.lower() and "funded" in text.lower()
               for text in replies)


# --------------------------------------------------------------------------- #
# The pack projection the tab renders (white-screen regression)
# --------------------------------------------------------------------------- #

def test_status_pack_carries_the_full_drafted_document(service):
    agent_case = _case(service)
    agent_case = service.draft_pack(agent_case, actor=ACTOR)
    pack = service.status(agent_case)["pack"]
    # Every field the tab dereferences must be present, populated or empty.
    for key in ("sections", "confirmations", "not_asked", "summary",
                "email", "artefacts", "history", "status", "receipt", "sent"):
        assert key in pack, f"status().pack is missing {key!r}"
    assert pack["sections"], "a drafted pack must carry its sections"


def test_status_pack_defaults_are_safe_before_any_draft(service):
    agent_case = _case(service)
    pack = service.status(agent_case)["pack"]
    assert pack["sections"] == []
    assert pack["confirmations"] == []
    assert pack["not_asked"] == []
    assert pack["summary"] == {}
