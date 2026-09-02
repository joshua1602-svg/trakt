#!/usr/bin/env python3
"""The deterministic brief across ten governed periods, as a regression.

``scripts/run_deterministic_period_suite.py`` prints these for a human. This
asserts the handful of things that must not change, so a future edit to the
generator set cannot quietly break one of them. Every period is derived from
the committed multibook canonical by deleting rows, scaling an existing balance
column or re-keying a source portfolio — no row is authored.

The assertions are deliberately about BEHAVIOUR rather than wording: that a
quiet month says nothing, that a disposal is reported as a fall, that an
arrival is separated from the book underneath it, that an unclassified arrival
is never called an acquisition, and that none of it is pinned to a client id or
a portfolio name. Pinning the sentences would make every future improvement to
the prose look like a regression.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tests.portfolio_review_redteam as rt

pytestmark = pytest.mark.skipif(
    not (rt.PRIOR_CSV.exists() and rt.CURRENT_CSV.exists()),
    reason="the committed multibook canonical pair is not present")

CLIENT = "client2"


@pytest.fixture(scope="module")
def periods(tmp_path_factory) -> Dict[str, Dict[str, Any]]:
    """Every §12 period, with its brief. Built once — the frames are large."""
    import warnings

    warnings.filterwarnings("ignore")
    from mi_agent_api import insight_engine as ie

    root = tmp_path_factory.mktemp("periods")
    out: Dict[str, Dict[str, Any]] = {}
    for scenario in rt.build_periods(root, CLIENT):
        client = scenario.traps.get("client_id", CLIENT)
        brief = ie.build_funded(str(scenario.root), client, tenant_id=client)
        out[scenario.key] = {
            "scenario": scenario, "brief": brief,
            "insights": brief.get("insights") or [],
            "text": " ".join(str(i.get("summary") or "")
                             for i in (brief.get("insights") or ())),
        }
    return out


def _severities(entry: Dict[str, Any]) -> List[str]:
    return [str(i.get("severity")) for i in entry["insights"]]


# --------------------------------------------------------------------------- #
# A quiet period is a real answer
# --------------------------------------------------------------------------- #
def test_a_quiet_period_says_nothing(periods):
    """Two identical frames must not manufacture a finding.

    The most common way an automated brief loses a reader is by sending one
    every period whether or not anything happened.
    """
    entry = periods["quiet"]
    assert entry["insights"] == []
    assert entry["text"] == ""


def test_an_ordinary_month_is_reported_without_alarm(periods):
    entry = periods["organic_growth"]
    assert entry["insights"]
    assert set(_severities(entry)) == {"info"}
    assert "acquisition" not in entry["text"].lower()


# --------------------------------------------------------------------------- #
# Acquisition attribution
# --------------------------------------------------------------------------- #
def test_an_arrival_is_attributed_and_the_book_beneath_it_reported(periods):
    entry = periods["acquisition"]

    assert "attention" in _severities(entry)
    assert "acquisition of" in entry["text"]
    assert "Excluding the portfolio added this period" in entry["text"]


def test_an_arrival_masking_a_decline_reports_both_directions(periods):
    """The case the headline exists to hide.

    The book grew £10.3m and the business underneath it SHRANK. A brief that
    reported only the movement would be true and useless.
    """
    entry = periods["acquisition_masking_decline"]
    text = entry["text"]

    assert "increased by £10.3m" in text
    assert "decreased by £1.7m" in text
    assert "-6.9%" in text
    # The addition exceeds the net movement, so it must not be stated as a
    # share of it — that would read as more than 100%.
    assert "against a net movement of" in text


def test_a_disposal_is_reported_as_a_fall(periods):
    entry = periods["disposal"]
    text = entry["text"]

    assert "decreased by £12.0m" in text
    assert "-32.1pp" in text


# --------------------------------------------------------------------------- #
# Movement direction and risk characteristics
# --------------------------------------------------------------------------- #
def test_a_contracting_book_is_reported_as_contracting(periods):
    entry = periods["book_shrinks"]
    assert "decreased by" in entry["text"]
    assert "increased by £" not in entry["text"]


def test_a_concentrating_book_raises_its_ltv_movement_to_attention(periods):
    """One exposure grown sixfold moves the weighted LTV, and that is a finding."""
    entry = periods["concentration_warning"]

    assert "attention" in _severities(entry)
    assert "Balance-weighted current LTV moved" in entry["text"]


def test_no_concentration_claim_is_made_when_none_was_resolved(periods):
    """Silence, not reassurance.

    The guard that stops an unevaluated risk domain reading as a clean result
    lives in ``trakt_notifications.sources`` and is covered by
    ``test_risk_evidence``. What belongs here is the other half: with no
    governed concentration snapshot, the funded brief must not imply one was
    checked.
    """
    text = periods["no_concentration_config"]["text"].lower()

    for phrase in ("within limit", "no breach", "concentration is", "headroom"):
        assert phrase not in text


# --------------------------------------------------------------------------- #
# Scale and isolation
# --------------------------------------------------------------------------- #
def test_five_source_portfolios_and_two_simultaneous_arrivals(periods):
    """Nothing is pinned to three books, or to any book's name.

    The two arriving portfolios are clones re-keyed to ids and labels that
    appear nowhere in the codebase, and the brief names one of them from the
    governed label column.
    """
    entry = periods["multi_portfolio"]
    text = entry["text"]

    assert "Excluding the 2 portfolios added this period" in text
    assert ("JV Partner Book" in text or "SPV2 Sponsored Securitisation" in text)
    # A cloned DIRECT book is an addition, never an acquisition.
    assert "acquisition of" not in text


def test_a_second_client_is_served_from_its_own_data(periods):
    """No client id appears in production code, so a second one just works."""
    entry = periods["second_client"]
    assert entry["scenario"].traps["client_id"] == "client9"
    assert entry["insights"]
    assert entry["text"] == periods["organic_growth"]["text"]


def test_no_client_or_portfolio_name_is_hard_coded_in_the_generators():
    """The scaling claim, checked at the source rather than inferred.

    Looks for these names as STRING LITERALS rather than as substrings: a
    tenant id branched on in production code is written ``"ERE"``, and a bare
    substring search finds "ERE" inside "WHERE" in a docstring and fails for
    the wrong reason.
    """
    import re

    from mi_agent_api import funded_composition, insight_generators_funded

    for module in (insight_generators_funded, funded_composition):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for name in ("alp_acquired", "alp_origination", "spv1_sponsored",
                     "ERE", "client2", "client9"):
            literal = re.compile(rf"""['"]{re.escape(name)}['"]""")
            assert not literal.search(source), (
                f"{name!r} appears as a literal in {module.__name__}")


# --------------------------------------------------------------------------- #
# Brevity
# --------------------------------------------------------------------------- #
def test_every_period_stays_card_sized(periods):
    """The deterministic layer's whole point of comparison with the agent."""
    for key, entry in periods.items():
        assert len(entry["text"].split()) <= 120, f"{key} is too long"
