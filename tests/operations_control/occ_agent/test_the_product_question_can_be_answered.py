"""The question that clears six blockers has to be on the screen.

REPORTED FROM THE LIVE CASE, and it took three exchanges to find because
everything looked right from the server's side:

    "There is no button / tile / card for 'Decisions waiting for you' - this is
     getting critical and very frustrating now"

Seven required fields blocking a delivery, six of them answerable by confirming
the product, and no way to confirm it. The decision WAS being raised and WAS on
the run document. It could not be seen, and could not have been clicked if it
had been.

TWO DECISION SHAPES, AND THIS ONE WORE THE WRONG ONE. A stage writes a RAW row
into ``34_target_first_decisions.yaml`` — ``decision_type`` and ``issue`` at the
top level, ``status: "pending"`` — and ``_decisions_from_run`` converts it into
the operator-facing CARD that ``run.open_decisions`` holds. The product question
is raised by the GATE rather than by the artefact, and was prepended to the list
after that conversion had already run:

    decisions = self._decisions_from_run(run, facts, run_root)
    if adapters.product_profile_decision is not None:
        decisions = [adapters.product_profile_decision, *decisions]

So it sat in a list of cards wearing a raw row's shape, and was invisible twice:

  * EVERY reader in this system treats a missing status as open —
    ``d.get("status", "open")`` — and this was the one decision that set one
    explicitly, to ``"pending"``. The screen's ``d.status === "open"`` dropped
    it, ``open_now`` did not count it, ``blocking_decisions`` did not see it.
  * Had it rendered, ``DecisionCardView`` does ``decision.evidence.map(...)``
    and ``decision.options.map(...)`` on arrays the raw shape does not carry.
    That throws, and takes down the whole decisions panel with it.

Either alone hides the question. These tests hold the card to the contract the
screen actually reads, field by field, because "it is on the run document" was
true the whole time and was not the same as "an operator can answer it".
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent import base_mi_gate as gate

ASSET = "equity_release"
BLOCKED = ["maturity_date", "amortisation_type", "interest_rate_type",
           "originator_legal_entity_identifier", "originator_name",
           "exposure_currency_denomination", "current_principal_balance"]


@pytest.fixture()
def card():
    resolved = gate.needs_confirmation(ASSET, "")
    assert resolved is not None, \
        "equity release no longer asks for confirmation; this fixture is stale"
    return gate.confirmation_decision(resolved, BLOCKED)


class TestItReadsAsOpen:
    def test_it_sets_no_explicit_status(self, card):
        """THE BUG. A missing status reads as open everywhere in this system;
        an explicit "pending" reads as open NOWHERE."""
        assert "status" not in card
        assert card.get("status", "open") == "open"

    def test_the_screen_s_own_filter_keeps_it(self, card):
        """`AgentCase.tsx` renders `status.open_decisions.filter(d => d.status
        === "open" && !MAPPING_DECISION_TYPES.has(d.subject?.decision_type))`.
        Both halves are asserted here because both dropped it."""
        assert card.get("status", "open") == "open"
        assert (card.get("subject") or {}).get("decision_type") not in (
            "mapping_proposal", "mapping_confirmation", "mapping_ambiguity")

    def test_it_counts_as_an_open_decision_on_the_run(self, card):
        """`service.run_synthetic_onboarding` moves the run to
        EXCEPTIONS_REQUIRE_INPUT on `d.get("status", "open") == "open"`. A
        question nobody is sent to answer is not a question."""
        assert card.get("status", "open") == "open"
        assert card["blocking"] is True


class TestTheCardCanBeRendered:
    """Every field `DecisionCardView` touches. A missing array is not a blank
    card — it is a TypeError that removes the panel."""

    def test_the_arrays_it_maps_over_are_present(self, card):
        assert isinstance(card["evidence"], list)
        assert isinstance(card["options"], list)
        for item in card["evidence"]:
            assert "data" in item

    def test_the_prose_it_prints_is_present(self, card):
        for key in ("title", "question", "issue", "materiality",
                    "downstream_consequence"):
            assert str(card.get(key) or "").strip(), f"{key} is empty"

    def test_the_approve_button_sends_a_usable_value(self, card):
        """The primary action is `onAnswer("approve", decision.recommendation)`,
        and `resolve_decision` writes that value to
        `run.confirmed_product_profile`. An empty recommendation would confirm
        the product as nothing at all."""
        assert card["recommendation"] == "equity_release_lifetime_mortgage"

    def test_every_option_carries_a_value_and_a_label(self, card):
        for option in card["options"]:
            assert option.get("value")
            assert option.get("label")


class TestItStillCarriesWhatTheServerReads:
    """`resolve_decision` reads these from the TOP level to record the
    confirmation. Adding the card shape must not move them."""

    def test_the_type_and_the_profile_stay_where_resolve_looks(self, card):
        assert card["decision_type"] == "product_confirmation"
        assert card["profile_id"] == "equity_release_lifetime_mortgage"

    def test_it_names_what_confirming_would_clear(self, card):
        """"Confirm the product" with no consequence attached is a question
        nobody can weigh."""
        assert "maturity_date" in card["would_clear"]
        assert "current_principal_balance" not in card["would_clear"], \
            "a field base MI genuinely requires must not read as clearable"
        assert "maturity date" in card["downstream_consequence"]
        assert "regulatory return" in card["downstream_consequence"]
