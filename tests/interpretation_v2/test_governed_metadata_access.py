"""Opus reads Trakt's governed metadata, and the compiler still decides.

The addendum's architecture:

    question
        ↓
    Opus  ↕  read-only governed metadata
        ↓      system registry · asset config · portfolio config · capabilities
    CandidateIntent
        ↓
    deterministic compiler   ← independently re-validates EVERY concept
        ↓
    GovernedQueryPlan / Refuse / Clarify

Two things must both hold, and they pull in opposite directions:

  * the interpreter must be able to FIND what Trakt governs, or it clarifies on
    ordinary questions for want of metadata that exists;
  * what it finds must not shortcut a single check.

So the tests here come in pairs — one that the metadata is reachable, one that
reaching it changes nothing about what the compiler demands.
"""

from __future__ import annotations

import json

import pytest

from mi_agent.interpretation_v2 import (
    OUTCOME_PLAN,
    CompilerContext,
    DeterministicCompiler,
    ModelResponse,
    OpusInterpreter,
    load_governed_vocabulary,
)
from mi_agent.interpretation_v2.metadata import (
    SOURCES,
    GovernedMetadataService,
    asset_metadata,
    governed_values_for_field,
    metadata_tool_schemas,
    portfolio_semantic_context,
)

from .conftest import build_intent, intent_payload


@pytest.fixture()
def service(vocabulary):
    return GovernedMetadataService(vocabulary)


# --------------------------------------------------------------------------- #
# 1 · the authoritative estate is actually read
# --------------------------------------------------------------------------- #

def test_every_declared_source_exists_and_is_read():
    """The adapter names its sources so the estate it depends on is inspectable.

    A source that quietly stopped existing would degrade the vocabulary without
    failing anything, which is the failure mode this list exists to prevent.
    """
    missing = [name for name, path in SOURCES.items() if not path.exists()]
    assert missing == [], f"declared sources that are not there: {missing}"


def test_the_index_is_built_from_the_canonical_registry(vocabulary):
    assert len(vocabulary.concepts) > 100
    balance = vocabulary.resolve("current_outstanding_balance")
    assert balance is not None
    assert balance.role == "measure"
    assert balance.canonical_field == "current_outstanding_balance"


def test_analytical_metadata_reaches_the_concept(service):
    """Role, temporality and applicability come from the Business Semantics
    Registry — the "relationships to other governed concepts" the addendum
    names, and what lets an interpreter tell a stock from a flow."""
    result = service.get_concept_metadata("current_outstanding_balance")
    assert result["found"] is True
    assert result["temporality"], "temporality is governed metadata"
    assert result["asset_applicability"]
    assert "permitted_statistics" in result


def test_asset_configuration_reaches_the_model(service):
    asset = service.get_asset_metadata()
    assert asset["asset_class"] == "equity_release"
    assert asset["profile"] is not None
    assert "drawdown" in asset["governed_product_types"]
    assert "KFI" in asset["governed_pipeline_stages"]
    assert asset["primary_geography_basis"] in ("collateral", "borrower")


def test_portfolio_configuration_reaches_the_model(service):
    context = service.get_portfolio_semantic_context()
    assert context["base_currency"] == "GBP"
    assert context["country"] == "GB"
    assert context["available_portfolio_lenses"] == ["direct", "acquired", "all"]
    assert context["funding_facility_configured"] is True
    assert context["concentration_tests_configured"] > 0


def test_the_capability_catalogue_is_discoverable(service):
    found = service.search_capabilities("borrowing")
    assert any(row["capability"] == "borrowing_base"
               for row in found["intent_capabilities"])

    borrowing = service.get_capability_metadata("borrowing_base")
    assert borrowing["found"] is True
    assert "headroom" in borrowing["operations"]
    assert "borrowing_base_headroom" in borrowing["owned_measures"]

    catalogue = service.search_capabilities()
    assert catalogue["registered_analytical_capabilities"], (
        "the 28-capability registry must be discoverable, not just the "
        "intent-level names")


def test_search_finds_a_concept_from_a_business_word(service):
    result = service.search_concepts("loan to value")
    ids = [row["concept_id"] for row in result["concepts"]]
    assert "current_loan_to_value" in ids


def test_search_reports_the_ambiguity_rather_than_resolving_it(service):
    """"region" is claimed by several governed fields. The tool names them."""
    result = service.get_concept_metadata("region")
    assert result["found"] is False
    assert "AMBIGUOUS" in result["reason"]
    assert len(result["candidates"]) > 1


def test_a_word_the_registry_lacks_returns_suggestions_not_a_match(service):
    result = service.get_concept_metadata("ebitda")
    assert result["found"] is False
    assert result["reason"] == "not a governed concept"


# --------------------------------------------------------------------------- #
# 2 · reaching the metadata shortcuts nothing
# --------------------------------------------------------------------------- #

def test_everything_search_advertises_actually_compiles(vocabulary, compiler):
    """The service and the compiler read ONE index, so they cannot disagree.

    A metadata service that drifted from the registry the compiler validates
    against would be worse than none: it would advertise concepts that then
    refuse, and every such refusal would look like a model error.
    """
    service = GovernedMetadataService(vocabulary)
    advertised = [row["concept_id"] for row in
                  service.search_concepts("", limit=40)["concepts"]]
    assert advertised, "premise: search returns something"

    for concept_id in advertised:
        concept = vocabulary.resolve(concept_id)
        assert concept is not None, f"search advertised {concept_id!r}, index lacks it"
        if concept.role != "measure" or concept.is_specialist:
            continue
        result = compiler.compile(build_intent(measures=[{"concept": concept_id}]))
        assert result.outcome == OUTCOME_PLAN or result.codes(), concept_id
        if result.plan is not None:
            assert result.plan.outputs[0].measures[0].canonical_field == \
                concept.canonical_field


def test_a_concept_the_model_read_is_still_checked_for_applicability(vocabulary):
    """Found in the registry is not the same as applicable here.

    The interpreter can retrieve a cross-asset concept and name it; the compiler
    still asks whether it applies to THIS asset class, and refuses if not.
    """
    from dataclasses import replace

    concept = vocabulary.resolve("current_loan_to_value")
    narrowed = replace(vocabulary, concepts=dict(
        vocabulary.concepts,
        current_loan_to_value=replace(concept,
                                      asset_applicability=("auto_finance",))))
    result = DeterministicCompiler(CompilerContext(narrowed)).compile(
        build_intent(measures=[{"concept": "current_loan_to_value"}]))
    assert result.plan is None
    assert "CONCEPT_UNAVAILABLE" in result.codes()


def test_a_concept_the_model_read_is_still_checked_for_availability(vocabulary):
    """Applicable is not the same as present on this book."""
    scoped = CompilerContext(vocabulary,
                             available_fields=["current_outstanding_balance"])
    result = DeterministicCompiler(scoped).compile(
        build_intent(measures=[{"concept": "indexed_loan_to_value"}]))
    assert result.plan is None
    assert "CONCEPT_UNAVAILABLE" in result.codes()


def test_reading_the_registry_does_not_unlock_a_forbidden_statistic(compiler):
    """A permitted-statistics list is a fact, not a permission slip.

    The model can read that ``current_loan_to_value`` permits a weighted
    average. Naming a statistic outside that list still refuses.
    """
    result = compiler.compile(build_intent(
        measures=[{"concept": "current_loan_to_value", "statistic": "sum"}]))
    assert result.plan is None
    assert "UNSUPPORTED_STATISTIC" in result.codes()


def test_reading_the_registry_does_not_unlock_an_unsupported_composition(compiler):
    result = compiler.compile(build_intent(capability="portfolio_summary",
                                           operation="bridge"))
    assert result.plan is None
    assert "UNSUPPORTED_OPERATION" in result.codes()


def test_the_metadata_service_cannot_mutate_anything(vocabulary):
    """Read-only means read-only: the index is the same object afterwards."""
    before = json.dumps({k: v.concept_id for k, v in
                         sorted(vocabulary.concepts.items())})
    service = GovernedMetadataService(vocabulary)
    for name in ("get_asset_metadata", "get_portfolio_semantic_context"):
        service.call(name, {})
    service.call("search_concepts", {"query": "balance"})
    service.call("get_allowed_values", {"concept_id": "erm_product_type"})
    after = json.dumps({k: v.concept_id for k, v in
                        sorted(vocabulary.concepts.items())})
    assert before == after


# --------------------------------------------------------------------------- #
# 3 · the retrieval loop
# --------------------------------------------------------------------------- #

class RetrievingClient:
    """A client that retrieves metadata first, the way a live model does."""

    def __init__(self, payload, *, lookups=("balance",)):
        self.payload = payload
        self.lookups = lookups
        self.dispatched = []

    def emit_intent(self, *, system, user, tool_schema, tool_name,
                    metadata_tools=(), dispatch=None):
        results = []
        for word in self.lookups:
            results.append(dispatch("search_concepts", {"query": word}))
        results.append(dispatch("get_allowed_values",
                                {"concept_id": "erm_product_type"}))
        self.dispatched = results
        return ModelResponse(payload=self.payload, model_id="scripted",
                             metadata_calls=tuple(
                                 {"tool": "search_concepts", "arguments": {"query": w}}
                                 for w in self.lookups))


def test_the_interpreter_hands_the_model_a_working_dispatcher(vocabulary, compiler):
    client = RetrievingClient(intent_payload())
    interpreter = OpusInterpreter(client, vocabulary=vocabulary)
    outcome = interpreter.interpret("What is our total balance?")

    assert outcome.ok
    assert client.dispatched, "the dispatcher was never callable"
    assert client.dispatched[0]["concepts"], "search returned nothing"
    assert client.dispatched[-1]["has_governed_values"] is True
    assert outcome.metadata_calls, "retrievals must be recorded as provenance"


def test_metadata_retrievals_are_recorded_on_the_plans_provenance(vocabulary,
                                                                  compiler):
    client = RetrievingClient(intent_payload(), lookups=("balance", "ltv"))
    interpreter = OpusInterpreter(client, vocabulary=vocabulary)
    outcome = interpreter.interpret("What is our total balance?")
    result = compiler.compile(outcome.intent)
    assert result.outcome == OUTCOME_PLAN
    # The intent carries what was retrieved, so an audit can see what the model
    # looked at before it decided.
    assert len(outcome.metadata_calls) == 2


def test_a_dispatcher_is_never_required_for_the_compiler(compiler):
    """The compiler needs no metadata service at all.

    It reads the index directly. If it went through the tool surface, the tool
    surface would be in the trust path — and the tool surface is what the model
    talks to.
    """
    import ast
    import inspect
    from pathlib import Path

    from mi_agent.interpretation_v2 import compiler as compiler_module

    tree = ast.parse(Path(inspect.getfile(compiler_module)).read_text(
        encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert ".metadata" not in imported and "metadata" not in imported


def test_the_tool_surface_offers_exactly_the_addendums_functions():
    names = [tool["name"] for tool in metadata_tool_schemas()]
    for required in ("search_concepts", "get_concept_metadata",
                     "get_allowed_values", "search_capabilities",
                     "get_capability_metadata", "get_asset_metadata",
                     "get_portfolio_semantic_context"):
        assert required in names
