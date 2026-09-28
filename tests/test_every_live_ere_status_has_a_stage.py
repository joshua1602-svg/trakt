"""Every status ERE's pipeline report uses reads as a funnel stage.

A census of ERE's 90 weekly pipeline snapshots (Sep 2025 - Sep 2026) found
two live statuses the stage map did not know: "Funds Requested" (after the
offer, before the money is released) and "Decision Refer" (an application
referred for an underwriting decision). Both read UNKNOWN, and UNKNOWN is left
out of the expected-funding forecast — so the cases nearest to completing
counted for nothing.
"""
from __future__ import annotations

import pytest

from mi_agent_api.pipeline_prep import canonical_stage


@pytest.mark.parametrize("label, stage", [
    ("KFI", "KFI"), ("Application", "APPLICATION"), ("Decision Refer", "APPLICATION"),
    ("Offer", "OFFER"), ("Funds Requested", "OFFER"),
    ("Completed", "COMPLETED"), ("Withdrawn", "WITHDRAWN"),
])
def test_an_ere_status_reads_as_its_stage(label, stage):
    assert canonical_stage(label) == stage


def test_the_new_statuses_are_forecast():
    from mi_agent_api.pipeline_prep import ACTIVE_STAGES
    assert canonical_stage("Funds Requested") in ACTIVE_STAGES
    assert canonical_stage("Decision Refer") in ACTIVE_STAGES
