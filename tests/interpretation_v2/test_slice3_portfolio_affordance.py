"""Slice 3: the model-facing portfolio contract moved. Nothing under it did.

THE FINDING THIS ANSWERS. A live boundary run of twelve fresh interpretations
measured the role axis working 4/4 and the NAMED source axis 0/4. The cause was
not the model: `population.source_reference` reached it as an undescribed
string, the system prompt mentioned neither the field nor the concept, and
`get_allowed_values` told it in terms to record a blocking ambiguity for exactly
the thing the named axis exists to express. It also read a book's proper name as
two other axes, because `back_book` is a governed seasoning token and nothing
said which axis a bare "back book" belongs to.

SO THE CONTRACT MOVED AND THE SEMANTICS DID NOT. Everything asserted below is
either (a) the affordance that is new, or (b) a governed behaviour that must be
exactly what it was. The eight non-regression proofs the brief requires are
named in the test names so a reader can find each one.
"""

from __future__ import annotations

import json

import pytest

from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import (candidate_intent_json_schema,
                                               parse_candidate_intent)
from mi_agent.interpretation_v2.metadata import GovernedMetadataService
from mi_agent.interpretation_v2.opus_interpreter import SYSTEM_PROMPT
from mi_agent.interpretation_v2.vocabulary import (CAPABILITY_OPERATIONS,
                                                   PORTFOLIO_SCOPE_AXES,
                                                   POPULATION_BASES,
                                                   POPULATION_LENSES,
                                                   SEASONING_SEGMENTS,
                                                   load_governed_vocabulary)

_EVIDENCE = "mi_agent/interpretation_v2/evidence/run8_135_signoff_2b00172.json"


@pytest.fixture(scope="module")
def vocabulary():
    return load_governed_vocabulary()


@pytest.fixture(scope="module")
def registry():
    """Two books of one client, named the way a client actually names them.

    `alp_acquired`'s label contains the words of two OTHER governed axes, which
    is the whole difficulty: "Acquired" is a role and "Back Book" is a seasoning
    token, and neither is what the name means.
    """
    from trakt_core.portfolio import PortfolioRecord, PortfolioRegistry

    return PortfolioRegistry(client_id="ere_funding_uk", portfolios=(
        PortfolioRecord(portfolio_id="alp_acquired", portfolio_type="acquired",
                        label="ALP Acquired Back Book",
                        aliases=("ALP back book",)),
        PortfolioRecord(portfolio_id="alp_origination", portfolio_type="direct",
                        label="ALP Originations"),
    ))


def _payload(population, **over):
    body = {"schema_version": "candidate_intent/1.0",
            "capability": "generic_analysis", "operation": "point_in_time",
            "population": population,
            "measures": [{"concept": "current_outstanding_balance"}],
            "time": {"form": "current"}}
    body.update(over)
    return body


def _compile(population, registry=None, **over):
    compiler = DeterministicCompiler(CompilerContext(source_registry=registry))
    return compiler.compile(parse_candidate_intent(_payload(population, **over)))


def _scope(result):
    plan = getattr(result, "plan", None)
    if plan is None:
        return []
    return [(p.canonical_field, p.value) for p in plan.population.scope_predicates]


# --------------------------------------------------------------------------- #
# PROOF 1 — an old payload, with no source_reference, parses as it always did
# --------------------------------------------------------------------------- #

def test_proof_1_payloads_without_a_source_reference_are_unchanged():
    """The new slot is optional and absent means absent — not empty string."""
    intent = parse_candidate_intent(_payload({"base": "funded", "lens": "direct"}))
    assert intent.population.source_reference is None
    assert intent.population.key() == ("funded", "direct", "any", None)

    bare = parse_candidate_intent(_payload({}))
    assert bare.population.key() == ("funded", "all", "any", None)
    assert _scope(_compile({})) == []


# --------------------------------------------------------------------------- #
# PROOF 2 — the 135-case signed-off corpus replays byte-identically
# --------------------------------------------------------------------------- #

def test_proof_2_the_signed_off_135_replays_identically():
    """Every recorded payload, re-parsed and re-compiled by this code.

    Outcome AND plan identity, not a sample: if the contract edit had disturbed
    parsing or binding anywhere in the signed-off corpus, one of 135 would move.
    Zero model calls — the payloads have been on disk since the sign-off run.
    """
    with open(_EVIDENCE, "r", encoding="utf-8") as handle:
        recorded = json.load(handle)["results"]
    assert len(recorded) == 135

    compiler = DeterministicCompiler(CompilerContext())
    moved = []
    for row in recorded:
        result = compiler.compile(parse_candidate_intent(row["raw_payload"]))
        plan = getattr(result, "plan", None)
        now = (str(result.outcome), getattr(plan, "plan_id", "") or "")
        was = (str(row["outcome"]), row.get("plan_id") or "")
        if now != was:
            moved.append((row["question_id"], was, now))
    assert moved == [], f"{len(moved)} of 135 signed-off cases moved: {moved[:5]}"


# --------------------------------------------------------------------------- #
# PROOF 3 — Direct / Acquired binds exactly as before
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("role", ["direct", "acquired"])
def test_proof_3_the_role_axis_binds_one_predicate(role):
    result = _compile({"base": "funded", "lens": role})
    assert str(result.outcome) == "PLAN"
    assert _scope(result) == [("source_portfolio_type", role)]


def test_proof_3_the_role_axis_still_composes_with_filters_and_dimensions():
    result = _compile({"base": "funded", "lens": "acquired"},
                      operation="breakdown", dimensions=["ltv_bucket"],
                      filters=[{"concept": "erm_product_type", "comparator": "eq",
                                "value": "drawdown"}])
    assert str(result.outcome) == "PLAN"
    assert _scope(result) == [("source_portfolio_type", "acquired")]


# --------------------------------------------------------------------------- #
# PROOF 4 — total funded is still the default, and still unnarrowed
# --------------------------------------------------------------------------- #

def test_proof_4_the_total_funded_default_is_unchanged():
    assert parse_candidate_intent(_payload({})).population.base == "funded"
    for population in ({}, {"base": "funded"}, {"base": "funded", "lens": "all"}):
        result = _compile(population)
        assert str(result.outcome) == "PLAN"
        assert _scope(result) == [], f"{population} narrowed the default"


# --------------------------------------------------------------------------- #
# PROOF 5 — the named source binding is the one that was accepted
# --------------------------------------------------------------------------- #

def test_proof_5_a_governed_name_binds_to_the_identity_column(registry):
    for reference in ("ALP Acquired Back Book", "ALP back book",
                      "alp acquired back book"):
        result = _compile({"base": "funded", "source_reference": reference},
                          registry=registry)
        assert str(result.outcome) == "PLAN"
        assert _scope(result) == [("source_portfolio_id", "alp_acquired")]


def test_proof_5_an_unknown_or_unregistered_name_still_refuses(registry):
    unknown = _compile({"base": "funded", "source_reference": "Portfolio Phoenix"},
                       registry=registry)
    assert str(unknown.outcome) == "REFUSE"
    assert [r.code for r in unknown.reasons] == ["CONCEPT_UNAVAILABLE"]

    # No registry at all is the fail-closed state, not a widening.
    none = _compile({"base": "funded", "source_reference": "ALP back book"})
    assert str(none.outcome) == "REFUSE"
    assert [r.code for r in none.reasons] == ["CONCEPT_UNAVAILABLE"]


def test_proof_5_a_role_and_a_name_remain_separate_axes(registry):
    """Both stated means both applied. The registry decides whether they agree."""
    result = _compile({"base": "funded", "lens": "acquired",
                       "source_reference": "ALP Acquired Back Book"},
                      registry=registry)
    assert sorted(_scope(result)) == [("source_portfolio_id", "alp_acquired"),
                                      ("source_portfolio_type", "acquired")]


# --------------------------------------------------------------------------- #
# PROOF 6 — slice 2 temporal behaviour is untouched
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("time_block", [
    {"form": "current"},
    {"form": "series", "grain": "monthly"},
    {"form": "relative_pair", "labels": ["this month"], "periods_back": 1},
    {"form": "range", "labels": ["October 2025", "June 2026"]},
    {"form": "explicit_period", "labels": ["June 2026"]},
])
def test_proof_6_temporal_forms_survive_a_portfolio_scope(time_block, registry):
    scoped = _compile({"base": "funded", "source_reference": "ALP back book"},
                      registry=registry, time=time_block)
    plain = _compile({"base": "funded"}, time=time_block)
    assert str(scoped.outcome) == str(plain.outcome)
    if getattr(plain, "plan", None) is not None:
        from dataclasses import asdict
        assert asdict(scoped.plan.period) == asdict(plain.plan.period), (
            "naming a book changed how the period was read")


# --------------------------------------------------------------------------- #
# PROOF 7 — attribution is still its own capability
# --------------------------------------------------------------------------- #

def test_proof_7_attribution_stays_separate_from_generic_analysis():
    assert "movement" in CAPABILITY_OPERATIONS["period_movement"]
    assert "movement" in CAPABILITY_OPERATIONS["generic_analysis"]
    result = _compile({"base": "funded", "lens": "acquired"},
                      capability="period_movement", operation="movement",
                      time={"form": "relative_pair", "labels": ["this month"],
                            "periods_back": 1})
    assert str(result.outcome) == "PLAN"
    assert result.plan.capability == "period_movement"
    assert _scope(result) == [("source_portfolio_type", "acquired")]


# --------------------------------------------------------------------------- #
# PROOF 8 — seasoning is intact; the generic phrase is no longer its property
# --------------------------------------------------------------------------- #

def test_proof_8_seasoning_still_binds_when_it_is_explicitly_asked_for():
    """The axis is not removed, disabled or narrowed. Only its lexical claim is."""
    assert SEASONING_SEGMENTS == frozenset({"front_book", "back_book", "any"})
    for segment in ("front_book", "back_book"):
        result = _compile({"base": "funded", "seasoning": segment})
        assert str(result.outcome) == "PLAN"
        assert _scope(result) == [("seasoning_segment", segment)]


def test_proof_8_seasoning_is_still_a_dimension_and_still_discoverable(vocabulary):
    service = GovernedMetadataService(vocabulary)
    found = service.search_concepts("seasoning")
    assert "seasoning_segment" in {c["concept_id"] for c in found["concepts"]}
    metadata = service.get_concept_metadata("seasoning_segment")
    assert metadata["found"] is True and metadata["role"] == "dimension"


def test_proof_8_the_lifecycle_axis_now_owns_the_generic_phrase(vocabulary):
    """Where the ownership is stated, and that it is stated ONCE.

    In the governed vocabulary, which is versioned and recorded in every plan's
    provenance — not in the hand-written prompt, which is policy and which
    `test_the_policy_names_no_bank_question_and_no_lexical_rule` keeps free of
    business vocabulary.
    """
    lifecycle = next(a for a in PORTFOLIO_SCOPE_AXES["axes"]
                     if a["slot"] == "population.base")
    assert "BACK BOOK" in lifecycle["values"]["funded"]
    assert "FRONT BOOK" in lifecycle["values"]["pipeline"]

    boundary = vocabulary.concepts["seasoning_segment"]
    view = GovernedMetadataService(vocabulary).get_concept_metadata(
        boundary.concept_id)
    assert "NOT the funded/pipeline lifecycle axis" in view["axis_boundary"]

    base = candidate_intent_json_schema()["properties"]["population"][
        "properties"]["base"]["description"]
    assert "BACK BOOK" in base and "FRONT BOOK" in base

    lowered = SYSTEM_PROMPT.lower()
    for phrase in ("back book", "front book", "acquired book", "drawdown"):
        assert phrase not in lowered, f"the prompt hard-codes {phrase!r}"


# --------------------------------------------------------------------------- #
# The affordance itself
# --------------------------------------------------------------------------- #

def test_every_population_slot_now_tells_the_model_what_it_is_for():
    slots = candidate_intent_json_schema()["properties"]["population"]["properties"]
    assert set(slots) == {"base", "lens", "seasoning", "source_reference"}
    for name, slot in slots.items():
        assert slot.get("description"), f"population.{name} is undescribed"
    assert set(slots["base"]["enum"]) == set(POPULATION_BASES)
    assert set(slots["lens"]["enum"]) == set(POPULATION_LENSES)
    assert "ATOMIC" in slots["source_reference"]["description"]
    assert "get_source_portfolios" in slots["source_reference"]["description"]


def test_the_value_list_guidance_no_longer_dead_ends_a_named_book(vocabulary):
    """The exact sentence that turned a resolvable name into a refusal."""
    service = GovernedMetadataService(vocabulary)
    row = service.get_allowed_values("source_portfolio_id")
    assert row["has_governed_values"] is False
    assert row["named_portfolio_route"] == "population.source_reference"
    assert "get_source_portfolios" in row["guidance"]

    # And an ordinary concept without governed values is NOT given the route.
    ordinary = service.get_allowed_values("current_loan_to_value")
    assert "named_portfolio_route" not in ordinary


def test_the_axes_are_declared_as_four_and_as_independent():
    slots = [a["slot"] for a in PORTFOLIO_SCOPE_AXES["axes"]]
    assert slots == ["population.base", "population.lens",
                     "population.source_reference", "population.seasoning"]
    assert "on its own terms" in PORTFOLIO_SCOPE_AXES["these_are_independent"]
    assert "NOT a seasoning restriction" in (
        PORTFOLIO_SCOPE_AXES["these_are_independent"])
    atomic = PORTFOLIO_SCOPE_AXES["a_governed_name_is_atomic"]
    assert "the WHOLE phrase is that book's identity" in atomic
    assert "Do NOT also read those words as other axes" in atomic
    # The SPV axis is named as absent rather than silently conflated.
    blob = json.dumps(PORTFOLIO_SCOPE_AXES).lower()
    assert "spv" not in blob, "an ungoverned axis was offered to the model"


def test_the_registry_is_per_call_and_never_remembered(vocabulary, registry):
    """A cached interpreter must not carry one client's books into the next."""
    import inspect

    from mi_agent.interpretation_v2.opus_interpreter import OpusInterpreter

    from .conftest import ScriptedClient, intent_payload

    assert "source_registry" not in inspect.signature(
        OpusInterpreter.__init__).parameters

    seen = []

    class Recording(ScriptedClient):
        def emit_intent(self, **kwargs):
            dispatch = kwargs.get("dispatch")
            seen.append(dispatch.__self__.source_registry)
            return super().emit_intent(**kwargs)

    interpreter = OpusInterpreter(Recording(intent_payload()), vocabulary=vocabulary)
    interpreter.interpret("total balance", source_registry=registry)
    interpreter.interpret("total balance")
    assert seen == [registry, None], (
        "the interpreter carried a registry between questions")
