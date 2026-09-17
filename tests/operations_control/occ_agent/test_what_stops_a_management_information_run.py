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

class TestTheRegulatoryReturnIsNotExcused:

    def test_every_field_blocks_again_once_a_regime_is_prepared(self):
        """``base_mi`` speaks for management information. A field the profile
        marks optional for MI can still be mandatory for the regulator."""
        blocking, excused = gate.split(ALL_EIGHT, asset_class=ASSET,
                                       regime="ESMA_Annex2",
                                       confirmed_profile_id=PROFILE)
        assert excused == []
        assert len(blocking) == N

    def test_the_product_question_is_not_raised_on_a_regime_run(self):
        """There is nothing it could clear, so asking it would be noise."""
        blocking, _ = gate.split(ALL_EIGHT, asset_class=ASSET,
                                 regime="ESMA_Annex2")
        assert len(blocking) == N


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
