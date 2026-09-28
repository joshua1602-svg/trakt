"""An Agent case is real work, and the OCC must not pretend otherwise.

WHAT AN OPERATOR SAW

A real client onboarding was issued to a client and was waiting on their reply.
Meanwhile Client Onboarding showed "Nothing in progress", "Nothing is waiting on
a client", and Home showed five zeros under "Nothing needs your attention".

None of that was wrong about the STORE. An Agent case lives in
``operations-control-synthetic`` and reaches the governed container only at
activation, through the promotion doorway — and ``promote`` refuses anything not
already APPROVED, so a case appears governed-side for the first time already
approved, and is activated immediately after. It can therefore never appear in
the drafts, awaiting-client or in-review queues. It goes from absent to Active
in one step.

That is the isolation boundary doing its job at the storage layer and failing at
the product layer. The doorway governs what may be WRITTEN. It was never meant
to govern what an operator may SEE.

THE LINK THAT COULD NOT WORK

The Agent case screen offered "Client onboarding — the onboarding case itself,
in the screens an operator normally works it in", pointing at
``/onboarding/{case_id}``. Two things were wrong with it:

* the case is not in the governed store until it activates, so the screen would
  404 on it; and
* ``/onboarding/{id}`` is not a route at all. The case wizard is at
  ``/onboarding/cases/{id}``.

So the link was broken for every case anyone would ever click it on, and broken
twice.
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent import states as _states

from .conftest import ACTOR, TENANT_A

OPENING = ("Onboard Northstar Lending. UK equity release. Monthly portfolio "
           "MI. Portfolio id direct_101.")


def links_of(service, agent_case):
    return {row["label"]: row for row in service.status(agent_case)["occ_links"]}


@pytest.fixture()
def opened(service):
    return service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction=OPENING)


class TestTheClientOnboardingLinkIsNotOfferedBeforeItWorks:
    def test_it_is_absent_while_the_case_is_only_synthetic(self, service,
                                                           opened):
        """Nothing to link to: the governed store has never heard of it."""
        assert "Client onboarding" not in links_of(service, opened)

    def test_the_other_links_are_untouched(self, service, opened):
        """They point at views that exist regardless of this case."""
        labels = links_of(service, opened)
        assert "Platform configuration" in labels
        assert "Rules" in labels

    def test_when_it_is_offered_it_points_at_a_real_route(self, service,
                                                          opened):
        """``/onboarding/{id}`` was never a route. The wizard is at
        ``/onboarding/cases/{id}``, which is what Client Onboarding's own home
        links to."""
        from operations_control.occ_agent.service import _occ_links
        from operations_control.onboarding.case import ACTIVATED

        case = opened.case
        case.status = ACTIVATED
        row = {r["label"]: r for r in _occ_links(case, opened.run)}
        assert "Client onboarding" in row
        assert row["Client onboarding"]["to"] == \
            f"/onboarding/cases/{case.case_id}"


class TestTheCaseIsListedAsWork:
    """The list an operator can find it in, whatever screen they start from."""

    def test_an_open_case_is_listed_with_what_a_queue_row_needs(self, service,
                                                               opened):
        rows = service.list_cases(TENANT_A)
        found = [r for r in rows if r["case_ref"] == opened.case_ref]
        assert found, "an open case is listed"
        row = found[0]
        # Everything a governed queue row renders from, so the OCC can show
        # the case without reaching into the synthetic store itself.
        assert row["client_name"]
        assert row["onboarding_status"]
        assert row["state_label"]
        assert "synthetic" in row

    def test_the_onboarding_status_is_the_one_the_queues_sort_on(self, service,
                                                                opened):
        """A queue maps on onboarding status, not the run's own state."""
        from operations_control.onboarding.case import DRAFT
        row = [r for r in service.list_cases(TENANT_A)
               if r["case_ref"] == opened.case_ref][0]
        assert row["onboarding_status"] == DRAFT
        assert row["state"] != DRAFT      # the run has its own vocabulary


class TestNothingAboutTheDoorwayMoves:
    """Visibility widened; the boundary did not."""

    def test_the_case_is_still_written_only_to_the_synthetic_container(
            self, service, opened, blob_root):
        from .conftest import live_container_paths
        assert live_container_paths(blob_root) == []

    def test_the_agents_store_still_refuses_the_live_container(self, service):
        """The guard that makes the isolation real, not a convention."""
        from operations_control.occ_agent.store import (
            StoreMisconfigured, assert_isolated,
        )
        assert_isolated(service.store.container)      # must not raise
        with pytest.raises(StoreMisconfigured):
            assert_isolated("operations-control")
