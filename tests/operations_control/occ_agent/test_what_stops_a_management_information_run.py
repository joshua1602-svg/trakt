"""What stops a management-information run, and what only stops the return.

A live equity release onboarding was blocked on eight required fields, among
them ``maturity_date`` — and a lifetime mortgage has no contractual maturity
date. The same client's files, loaded through the platform's own ingestion
route, went through: nothing there gates on materiality. That gate is the
Agent's alone, and it stopped on every BLOCKING finding without ever asking
what the product needs.

``config/asset/product_profiles.yaml`` has answered this per field, per
product, all along. For the equity release lifetime mortgage:

    maturity_date                       not_applicable
    amortisation_type                   defaulted
    interest_rate_type                  defaulted
    exposure_currency_denomination      defaulted
    originator_legal_entity_identifier  optional
    originator_name                     optional
    current_principal_balance           required
    data_cut_off_date                   required

Six excused, two required. Management information without a balance or a
cut-off date is not a report — so the two that remain are the right two, and
that is the configuration's judgement rather than this module's.

THE CURRENCY IS DEFAULTED, NOT DEMANDED. A UK lifetime-mortgage lender does not
vary the denomination loan by loan, and the platform already holds the answer
in three governed places: the client's approved reporting currency captured at
onboarding, the asset pack's own default, and the central tape builder's
static default and country inference. Blocking MI on a currency column asks the
client to restate, per row, what they told us once at onboarding. It was
``required`` only because nothing had declared it and an undeclared field falls
through to ``required``.

THE GUARD THAT IS NOT WORKED AROUND. On the asset class alone the platform
scores the profile at 0.6, inside its confirm band, and PROPOSES rather than
applies it. Relaxing a check on the strength of a guess is exactly what that
band exists to prevent. So nothing is excused until an operator confirms the
product — and the question is raised in front of the blockers it would clear,
naming them, because "confirm the product" with no consequence attached is a
question nobody can weigh.
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent import base_mi_gate as gate

ASSET = "equity_release"
PROFILE = "equity_release_lifetime_mortgage"

EXCUSED_FOR_MI = ("maturity_date", "amortisation_type", "interest_rate_type",
                  "exposure_currency_denomination",
                  "originator_legal_entity_identifier", "originator_name")
REQUIRED_FOR_MI = ("current_principal_balance", "data_cut_off_date")


def findings(*fields, materiality="BLOCKING"):
    return [{"field_name": f, "materiality": materiality} for f in fields]


ALL_EIGHT = findings(*EXCUSED_FOR_MI, *REQUIRED_FOR_MI)
N = len(ALL_EIGHT)


def names(rows):
    return {r["field_name"] for r in rows}


# --------------------------------------------------------------------------- #
# Until the product is confirmed, nothing is excused
# --------------------------------------------------------------------------- #

class TestTheProductMustBeConfirmedFirst:

    def test_asset_class_alone_excuses_nothing(self):
        """The platform proposes on 0.6 and does not apply. Acting on that
        would be relaxing a required-field check on a guess."""
        blocking, excused = gate.split(ALL_EIGHT, asset_class=ASSET)
        assert excused == []
        assert len(blocking) == N

    def test_the_question_is_raised_instead(self):
        pending = gate.needs_confirmation(ASSET)
        assert pending is not None
        assert pending.profile_id == PROFILE
        assert not pending.applied

    def test_the_question_names_what_confirming_would_clear(self):
        """"Confirm the product" with no consequence attached is a question
        nobody can weigh."""
        pending = gate.needs_confirmation(ASSET)
        decision = gate.confirmation_decision(
            pending, [r["field_name"] for r in ALL_EIGHT])

        assert set(decision["would_clear"]) == set(EXCUSED_FOR_MI)
        assert "maturity date" in decision["proposed_mapping"]
        assert "regulatory return" in decision["proposed_mapping"]

    def test_the_question_reads_as_a_question(self):
        pending = gate.needs_confirmation(ASSET)
        decision = gate.confirmation_decision(pending, ["maturity_date"])
        assert decision["issue"] == ("Is this book an equity release lifetime "
                                     "mortgage?")
        assert decision["blocking"] is True

    def test_once_confirmed_there_is_nothing_left_to_ask(self):
        assert gate.needs_confirmation(ASSET, PROFILE) is None


# --------------------------------------------------------------------------- #
# Once confirmed, the profile decides
# --------------------------------------------------------------------------- #

class TestOnceTheProductIsConfirmed:

    def test_the_six_the_product_does_not_need_stop_blocking(self):
        blocking, excused = gate.split(ALL_EIGHT, asset_class=ASSET,
                                       confirmed_profile_id=PROFILE)
        assert names(excused) == set(EXCUSED_FOR_MI)
        assert names(blocking) == set(REQUIRED_FOR_MI)

    def test_a_report_still_needs_a_balance_and_a_cut_off_date(self):
        """Not a judgement made here — the profile marks these required."""
        blocking, _excused = gate.split(ALL_EIGHT, asset_class=ASSET,
                                        confirmed_profile_id=PROFILE)
        assert names(blocking) == set(REQUIRED_FOR_MI)

    def test_an_excused_finding_carries_the_policy_that_excused_it(self):
        _blocking, excused = gate.split(ALL_EIGHT, asset_class=ASSET,
                                        confirmed_profile_id=PROFILE)
        by_field = {r["field_name"]: r for r in excused}
        assert by_field["maturity_date"]["base_mi_policy"] == "not_applicable"
        assert by_field["originator_name"]["base_mi_policy"] == "optional"
        assert by_field["amortisation_type"]["base_mi_policy"] == "defaulted"

    def test_an_excused_finding_is_reported_not_hidden(self):
        """"Not applicable to this product" is an answer, not an absence. An
        operator who cannot see it cannot question it."""
        _blocking, excused = gate.split(ALL_EIGHT, asset_class=ASSET,
                                        confirmed_profile_id=PROFILE)
        assert len(excused) == len(EXCUSED_FOR_MI)
        for row in excused:
            assert row["excused_reason"] == gate.EXCUSED_NOTE

    def test_the_sentence_says_which_answer_let_it_through(self):
        _blocking, excused = gate.split(ALL_EIGHT, asset_class=ASSET,
                                        confirmed_profile_id=PROFILE)
        said = gate.sentence(
            next(r for r in excused if r["field_name"] == "maturity_date"))
        assert "not applicable for this product" in said
        assert "regulatory return" in said


# --------------------------------------------------------------------------- #
# The regime is not excused
# --------------------------------------------------------------------------- #

class TestTheRegulatoryReturnIsItsOwnVerdict:
    """Two questions, not one.

        "It should be the case the Operator invokes an MI + Regime run AND
         that MI can run without Regime being fully validated."

    ``base_mi`` used to be emptied the moment a regime was in play, so a
    regulatory run excused nothing and stopped on every field Annex 2 wants.
    That conflated "is the management information sound?" with "is the
    regulatory return complete?" — and the cost was concrete: a lender waiting
    on one Legal Entity Identifier could not see their own book.

    The pipeline was already built for two answers. The validation manifest
    carries ``ready_for_validation_complete`` and ``ready_for_projection`` as
    separate flags; only this gate had folded them together.
    """

    def test_mi_is_excused_on_its_own_terms_even_under_a_regime(self):
        blocking, excused = gate.split(ALL_EIGHT, asset_class=ASSET,
                                       regime="ESMA_Annex2",
                                       confirmed_profile_id=PROFILE)
        names = {r["field_name"] for r in excused}
        assert "maturity_date" in names
        assert len(blocking) < N

    def test_the_asset_pack_answers_before_the_lender_is_asked(self):
        """THE OPERATOR'S RULE, in their words:

            "Any core_canonical: true fields that are not met for MI purposes
             must first consult the asset and client configuration to assess
             whether there are any rules. For example, maturity date is not
             relevant for an equity release portfolio."

        ``product_defaults_ERM.yaml`` answers ``maturity_date: ND5`` — no fixed
        term — so it is not outstanding and must never reach an operator as
        something the lender still owes us.
        """
        _, excused = gate.split(ALL_EIGHT, asset_class=ASSET,
                                regime="ESMA_Annex2",
                                confirmed_profile_id=PROFILE)
        pending = gate.regime_outstanding(
            excused, regime="ESMA_Annex2", asset_class=ASSET)
        assert "maturity_date" not in {r["field_name"] for r in pending}

    def test_the_client_configuration_answers_before_the_lender_is_asked(self):
        """The originator's name is a standing client field, captured once on
        the entity holding the role — not a column in a monthly extract."""
        _, excused = gate.split(ALL_EIGHT, asset_class=ASSET,
                                regime="ESMA_Annex2",
                                confirmed_profile_id=PROFILE)
        pending = gate.regime_outstanding(
            excused, regime="ESMA_Annex2", asset_class=ASSET,
            client_defaults={"originator_name": "ERE Funding Limited"})
        assert "originator_name" not in {r["field_name"] for r in pending}

    def test_what_nobody_can_supply_is_still_reported(self):
        """RREL83 permits no ND code and must match GLEIF, so an LEI nobody
        holds is a real ask — and it is ONE value in client config rather than
        a column in every monthly extract."""
        _, excused = gate.split(ALL_EIGHT, asset_class=ASSET,
                                regime="ESMA_Annex2",
                                confirmed_profile_id=PROFILE)
        pending = gate.regime_outstanding(
            excused, regime="ESMA_Annex2", asset_class=ASSET)
        lei = next((r for r in pending
                    if r["field_name"] == "originator_legal_entity_identifier"),
                   None)
        assert lei is not None
        assert lei["regime_code"] == "RREL83"
        assert "RREL83" in gate.regime_sentence(lei)

    def test_nothing_is_outstanding_when_no_regime_is_prepared(self):
        """An MI-only delivery owes the regulator nothing."""
        _, excused = gate.split(ALL_EIGHT, asset_class=ASSET,
                                confirmed_profile_id=PROFILE)
        assert gate.regime_outstanding(
            excused, regime="", asset_class=ASSET) == []


# --------------------------------------------------------------------------- #
# It does not widen
# --------------------------------------------------------------------------- #

class TestItDoesNotWiden:

    def test_nothing_below_blocking_is_touched(self):
        """Only rows the aggregator already called BLOCKING are considered;
        nothing here promotes a finding."""
        rows = findings("maturity_date", materiality="REVIEW")
        blocking, excused = gate.split(rows, asset_class=ASSET,
                                       confirmed_profile_id=PROFILE)
        assert (blocking, excused) == ([], [])

    def test_an_unknown_asset_class_excuses_nothing(self):
        blocking, excused = gate.split(ALL_EIGHT, asset_class="not_a_product")
        assert excused == []
        assert len(blocking) == N

    def test_a_field_the_profile_does_not_mention_keeps_blocking(self):
        rows = findings("some_field_nobody_declared")
        blocking, excused = gate.split(rows, asset_class=ASSET,
                                       confirmed_profile_id=PROFILE)
        assert excused == []
        assert len(blocking) == 1
