#!/usr/bin/env python3
"""tests/test_pipeline_stage_movement_surface.py

Pipeline stage movement, on the route and in the pack.

Phase 3 built the reconciliation and nothing rendered it: no HTTP route, no
React component, no slide. This pins that both surfaces read ONE payload — the
engine supplies the reconciled movement, the renderer decides presentation —
and that the identity holds on counts and on money.
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pd = pytest.importorskip("pandas")
pptx = pytest.importorskip("pptx")


@pytest.fixture(scope="module")
def deployed(tmp_path_factory):
    """A book with a governed weekly pipeline, deployed the way the app runs."""
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    import pptx_visual_qa as Q

    tmp = tmp_path_factory.mktemp("stagemove")
    client = "stagemove1"
    root = Q.write_book(tmp / "runs", client, "multi_growing")
    Q.write_pipeline(root, client)
    config = tmp / "client.yaml"
    config.write_text("portfolio:\n  base_currency: GBP\n", encoding="utf-8")
    saved = dict(os.environ)
    os.environ.update({
        "MI_AGENT_ONBOARDING_OUTPUT_ROOT": str(root),
        "MI_AGENT_CLIENT_ID": client,
        "MI_AGENT_PIPELINE_ROOT": str(root),
        "TRAKT_LOCAL_BLOB_ROOT": str(tmp / "blob"),
        "TRAKT_INVESTOR_PPTX_PERSIST": "true",
        "TRAKT_INVESTOR_PPTX_ON_DEMAND": "true",
        "MI_AGENT_AUTH_ENABLED": "false",
        "TRAKT_MI_CLIENT_CONFIG": str(config),
        "TRAKT_STORAGE_BACKEND": "file",
        "TRAKT_RUNTIME_MODE": "test",
    })
    for key in ("MI_AGENT_DECK_ROOT", "AZURE_STORAGE_CONNECTION_STRING",
                "TRAKT_BLOB_CONNECTION"):
        os.environ.pop(key, None)
    from mi_agent_api import currency as currency_mod, data_source, datasets
    from mi_agent_api import deck_generation
    currency_mod._load_client_config.cache_clear()
    datasets._CLIENT_CURRENCY_CACHE.clear()
    data_source.reset_cache()
    deck_generation.reset_jobs()
    yield client
    os.environ.clear()
    os.environ.update(saved)
    currency_mod._load_client_config.cache_clear()
    datasets._CLIENT_CURRENCY_CACHE.clear()
    data_source.reset_cache()
    deck_generation.reset_jobs()


@pytest.fixture(scope="module")
def payload(deployed):
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app

    got = TestClient(app).get("/mi/evolution/pipeline-movement",
                              params={"client_id": deployed})
    assert got.status_code == 200, got.text
    return got.json()


# --------------------------------------------------------------------------- #
# The route.
# --------------------------------------------------------------------------- #

def test_the_route_exists_and_serves_the_reconciliation(payload):
    """Catches: an engine capability with no way for any surface to reach it."""
    assert payload["dataset"] == "pipeline_movement"
    assert payload["available"] is True, payload.get("reason")
    assert payload["stages"], payload


def test_case_movements_reconcile(payload):
    """opening + arrivals - departures = closing, on CASES."""
    for stage in payload["stages"]:
        expected = (stage["openingCaseCount"] + stage["arrivalCaseCount"]
                    - stage["departureCaseCount"])
        assert expected == stage["closingCaseCount"], stage["stage"]


def test_balance_movements_reconcile(payload):
    """...and on MONEY, with the amendment leg carrying the change on stayers."""
    for stage in payload["stages"]:
        identity = (stage["openingAmount"] + stage["arrivalAmount"]
                    - stage["departureAmount"]
                    + stage["amountChangeOnPersisting"])
        assert identity == pytest.approx(stage["closingAmount"], abs=0.01), stage
        assert stage["reconciles"] is True
    assert payload["reconciles"] is True


def test_an_amount_amendment_keeps_the_case_identity(payload):
    """Catches: a re-priced case counted as an exit and an arrival.

    A case present in both extracts is a stayer whatever its amount did, so its
    movement lands in the amendment leg — never in departures and arrivals.
    """
    for stage in payload["stages"]:
        assert (stage["persistingCaseCount"]
                == stage["openingCaseCount"] - stage["departureCaseCount"])


def test_departures_are_split_by_where_the_case_went(payload):
    """"Left the stage" and "left the pipeline" are different events."""
    for stage in payload["stages"]:
        destinations = stage.get("departuresByDestination") or []
        assert sum(d["caseCount"] for d in destinations) == \
            stage["departureCaseCount"], stage["stage"]


def test_the_route_names_the_identifier_it_joined_on(payload):
    assert payload.get("identifierField")
    assert payload.get("openingWeek") and payload.get("closingWeek")


def test_missing_stable_identity_suppresses_the_analysis_rather_than_guessing():
    """There is deliberately no fallback: without a stable case key the only
    honest answer is that this cannot be reported."""
    from mi_agent_api import evolution

    out = evolution.pipeline_stage_movement("/nonexistent-root", "nobody")
    assert out["available"] is False
    assert out["reason"]


# --------------------------------------------------------------------------- #
# One payload, two surfaces.
# --------------------------------------------------------------------------- #

# THE DECK NO LONGER DRAWS THIS RECONCILIATION. Two "Pipeline Stage Movement"
# slides existed — this one and the governed stage-transition payload — and one
# title cannot mean two pages, so the deck carries the transition payload and
# its parity is pinned by tests/mi_agent_pptx/test_stage_transition_parity.py.
#
# Everything above still applies: /mi/evolution/pipeline-movement is live, the
# dashboard reads it, and the identity it reconciles is worth holding to.
