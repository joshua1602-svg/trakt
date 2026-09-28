"""A saved answer stays visible, and stays correctable.

WHAT HAPPENED TO AN OPERATOR

They pasted a client's concentration limits into the box on the case, pressed
save, and the screen looked unchanged. The server had returned 200. There was
no way to tell, by looking, whether the answer had landed — because an answered
client question left the form entirely, so the box either showed an empty
unanswered question or showed nothing at all, and those two look the same.

Two defects, one cause.

1. AN ANSWER COULD NOT BE SEEN.

   ``build`` served only ``client_facing`` rows, and a field stops being
   client-facing the moment it has a value. So the one thing an operator needs
   after saving — what got saved — was the one thing the form stopped carrying.

2. AN ANSWER COULD NOT BE CORRECTED.

   ``plan_response`` checks every submitted key against the form actually
   served, which is right: it is what stops a free-text lane into the case. But
   with answered questions off the form, a correction to one was refused as
   "not a question Trakt puts to a client". A typo in a client's answer could
   be fixed through the conversation, and not through the form it was typed
   into.

WHAT THE FIX IS NOT

It is NOT putting answered questions back among the ones a client is asked.
``steps`` is what a client is put in front of and what the pack is built from,
and "an answered question is not asked again" is a rule with its own tests —
the first attempt at this fix broke five of them, which is how the design was
found to be wrong.

Answered questions sit BESIDE the steps, in ``ClientForm.answered``. The
client's own form is unchanged; ``ClientForm.field`` looks in both, so a
correction to an answer is accepted against the key it was saved under.

Deliberately narrow: only a field the catalogue declares ``asked_of_client``
qualifies, so this cannot promote a value Trakt derived, a default it applied
or an identifier it minted into a client question.
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent import client_form as _client_form

from .conftest import ACTOR, TENANT_A

OPENING = ("Onboard Northstar Lending. UK equity release. Monthly portfolio "
           "MI. Portfolio id direct_101.")

LIMITS = ("Maximum 10% of the portfolio to any one postcode district.\n"
          "Maximum LTV 55% at origination.\n"
          "No more than 5% of loans above GBP 750,000.")

KEY = "risk_limits.concentration_tests"


@pytest.fixture()
def asked(service):
    """A case where the concentration limits are an outstanding question."""
    case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction=OPENING)
    service.instruct(case, text="Set the concentration tests status to "
                                "supplied.", actor=ACTOR, confirm=True)
    return case


def served(service, case, key=KEY):
    """One field, whether it is still being asked or has come back.

    Two places on purpose. `steps` is what a CLIENT is put in front of, and a
    question they have answered must never appear there again — that rule has
    its own tests and this change does not touch it. `answered` sits beside the
    steps for the operator's benefit.
    """
    form = service.client_form(case).to_dict()
    for step in form["steps"]:
        for group in step["groups"]:
            for field in group["fields"]:
                if field["key"] == key:
                    return field
    return next((f for f in form["answered"] if f["key"] == key), None)


def in_steps(service, case, key=KEY):
    """True when the key is among the questions a client is still asked."""
    form = service.client_form(case).to_dict()
    return any(f["key"] == key for step in form["steps"]
               for g in step["groups"] for f in g["fields"])


class TestAnAnswerCanBeSeen:
    def test_the_question_is_asked_before_it_is_answered(self, service, asked):
        field = served(service, asked)
        assert field is not None
        assert field["answered"] is False
        assert field["value"] in (None, "")

    def test_it_is_still_there_afterwards(self, service, asked):
        """The whole defect: the box vanished, so a save could not be checked."""
        service.submit_client_response(asked, actor=ACTOR,
                                       response={KEY: LIMITS})
        field = served(service, asked)
        assert field is not None, "an answered question left the form"
        assert field["answered"] is True
        # And it is NOT back among the questions the client is asked.
        assert not in_steps(service, asked)

    def test_it_shows_what_was_actually_saved(self, service, asked):
        service.submit_client_response(asked, actor=ACTOR,
                                       response={KEY: LIMITS})
        assert served(service, asked)["value"] == LIMITS

    def test_every_line_survives(self, service, asked):
        """Three limits went in. An operator has to be able to see three."""
        service.submit_client_response(asked, actor=ACTOR,
                                       response={KEY: LIMITS})
        assert served(service, asked)["value"].count("\n") == 2

    def test_an_unanswered_question_is_not_marked_answered(self, service,
                                                           asked):
        service.submit_client_response(asked, actor=ACTOR,
                                       response={KEY: LIMITS})
        others = [f for step in service.client_form(asked).to_dict()["steps"]
                  for g in step["groups"] for f in g["fields"]]
        assert all(not f["answered"] for f in others)
        assert all(f["value"] in (None, "", [], {}) for f in others)


class TestAnAnswerCanBeCorrected:
    def test_a_typo_can_be_fixed_through_the_same_box(self, service, asked):
        service.submit_client_response(asked, actor=ACTOR,
                                       response={KEY: "Frist version."})
        service.submit_client_response(asked, actor=ACTOR,
                                       response={KEY: "First version."})
        assert served(service, asked)["value"] == "First version."

    def test_the_correction_is_what_the_case_holds(self, service, asked):
        service.submit_client_response(asked, actor=ACTOR,
                                       response={KEY: "First."})
        service.submit_client_response(asked, actor=ACTOR,
                                       response={KEY: "Second."})
        answers = service.onboarding.load_case(asked.case_ref).answers
        assert answers["risk_limits"]["concentration_tests"] == "Second."


class TestTheCheckItReplacesStillHolds:
    """Widening the form must not open a lane into the case."""

    def test_a_key_the_catalogue_does_not_declare_is_still_refused(
            self, service, asked):
        with pytest.raises(_client_form.UnknownAnswerKey):
            service.submit_client_response(
                asked, actor=ACTOR, response={"risk_limits.made_up": "x"})

    def test_a_field_that_is_not_the_clients_is_still_refused(self, service,
                                                              asked):
        """`concentration_tests_status` is the operator's decision, and sits in
        the same section as the question that is the client's."""
        with pytest.raises(_client_form.NotAClientQuestion):
            service.submit_client_response(
                asked, actor=ACTOR,
                response={"risk_limits.concentration_tests_status": "supplied"})

    def test_a_value_trakt_derived_is_not_promoted_to_a_client_question(
            self, service, asked):
        """`file_format` is inferred from the sample. It has a value, so it is
        KNOWN — and widening on KNOWN alone would have served it."""
        service.instruct(asked, text="The file format is xlsx.", actor=ACTOR,
                         confirm=True)
        assert served(service, asked, "sources[0].file_format") is None
        assert served(service, asked, "sources.file_format") is None

    def test_trakt_does_not_make_up_an_answer_over_a_real_one(self, service,
                                                              asked):
        """`generate_answers` reads the served form. An answered question on it
        would otherwise be overwritten with a fixture."""
        service.submit_client_response(asked, actor=ACTOR,
                                       response={KEY: LIMITS})
        service.generate_synthetic_answers(asked, actor=ACTOR)
        assert served(service, asked)["value"] == LIMITS


# --------------------------------------------------------------------------- #
# The whole decision, in one place
# --------------------------------------------------------------------------- #

class TestTheConcentrationDecisionIsOneAct:
    """`record_concentration_outcome` existed, with its controls, and nothing
    reached it.

    The status was set through the conversation and the limits pasted into the
    client form: two acts, nothing tying them together, and an operator who did
    one and not the other had a case that said "supplied" with no answer behind
    it — or an answer nobody had recorded a decision about.
    """

    def test_one_call_records_the_status_and_the_answer(self, service, asked):
        service.record_concentration_outcome(
            asked, actor=ACTOR, status="supplied", response_text=LIMITS)
        risk = service.onboarding.load_case(asked.case_ref).answers["risk_limits"]
        assert risk["concentration_tests_status"] == "supplied"
        assert risk["concentration_tests"] == LIMITS

    def test_a_blank_answer_cannot_be_recorded_as_supplied(self, service,
                                                           asked):
        """The control this route exists to enforce."""
        from operations_control.engine import OpsError
        with pytest.raises(OpsError) as exc:
            service.record_concentration_outcome(
                asked, actor=ACTOR, status="supplied", response_text="   ")
        assert exc.value.code == "OCC_AGENT_BLANK_SUPPLIED"

    def test_a_status_trakt_does_not_know_is_refused(self, service, asked):
        from operations_control.engine import OpsError
        with pytest.raises(OpsError):
            service.record_concentration_outcome(
                asked, actor=ACTOR, status="probably_fine")

    def test_deferring_records_the_reason_with_it(self, service, asked):
        service.record_concentration_outcome(
            asked, actor=ACTOR, status="deferred_with_reason",
            reason="The client has not yet provided its limits.")
        risk = service.onboarding.load_case(asked.case_ref).answers["risk_limits"]
        assert risk["concentration_tests_status"] == "deferred_with_reason"
        assert risk["concentration_tests_status_reason"] == \
            "The client has not yet provided its limits."

    def test_it_clears_the_blocker(self, service):
        """What an operator is actually trying to achieve.

        A fresh case, not the `asked` fixture: that one has already had the
        status set, so it would be asserting against a blocker already gone.
        """
        case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                   instruction=OPENING)
        before = [p["field"]
                  for p in service.onboarding_readiness(case)["blocking"]]
        assert "concentration_tests_status" in before
        service.record_concentration_outcome(
            case, actor=ACTOR, status="supplied", response_text=LIMITS)
        after = [p["field"]
                 for p in service.onboarding_readiness(case)["blocking"]]
        assert "concentration_tests_status" not in after

    def test_the_text_survives_whole(self, service, asked):
        """Not through the conversation, which truncates at the first clause."""
        service.record_concentration_outcome(
            asked, actor=ACTOR, status="supplied", response_text=LIMITS)
        stored = service.onboarding.load_case(
            asked.case_ref).answers["risk_limits"]["concentration_tests"]
        assert stored.count("\n") == 2
        assert "750,000" in stored


class TestTheRouteTheScreenCalls:
    def test_it_records_over_the_api(self, api_client, case_ref, headers):
        body = api_client.post(f"/ops/agent/cases/{case_ref}/concentration",
                               headers=headers,
                               json={"status": "supplied",
                                     "response_text": LIMITS})
        assert body.status_code == 200
        risk = body.json()["onboarding"]["answers"]["risk_limits"]
        assert risk["concentration_tests"] == LIMITS

    def test_a_blank_supplied_is_refused_over_the_api(self, api_client,
                                                      case_ref, headers):
        refused = api_client.post(f"/ops/agent/cases/{case_ref}/concentration",
                                  headers=headers,
                                  json={"status": "supplied",
                                        "response_text": ""})
        assert refused.status_code == 400

    def test_another_tenant_cannot_record_it(self, api_client, case_ref):
        denied = api_client.post(f"/ops/agent/cases/{case_ref}/concentration",
                                 headers={"X-Operator-Token": "tok-b"},
                                 json={"status": "not_applicable",
                                       "reason": "no facility"})
        assert denied.status_code == 404


@pytest.fixture()
def headers():
    return {"X-Operator-Token": "tok-a"}


@pytest.fixture()
def api_client(agent_env):
    import importlib

    from fastapi.testclient import TestClient

    from apps.blob_trigger_app.storage import Storage
    import operations_control.api.app as app_module
    from operations_control.occ_agent import api as agent_api
    from operations_control.occ_agent.service import OccAgentService
    importlib.reload(app_module)
    agent_api.configure(OccAgentService(
        Storage(agent_env["blob_root"]),
        container="operations-control-synthetic",
        sandbox=agent_env["sandbox"]))
    return TestClient(app_module.app)


@pytest.fixture()
def case_ref(api_client, headers):
    created = api_client.post("/ops/agent/cases", headers=headers,
                              json={"instruction": OPENING})
    return created.json()["case_ref"]
