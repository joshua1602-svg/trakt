"""Defect A — `funded_bridge` declares the axis it ACTUALLY attributed by.

The route computed a bridge by the requested dimension correctly, but never
published `metadata.groupedBy`. `declared_group_fields` reads declarations only
— "a route that declares nothing gets no certification" — so `grouping_proven`
returned False, the requested grouping was marked LOST, and the route refused an
answer it had already computed.

THE RULE IS "DECLARE WHAT WAS EXECUTED, NOT WHAT WAS REQUESTED." The declaration
is `evolution.funded_bridge`'s own `dimensionCol` — the candidate it found
present in the data. That is the safety property: a question naming a dimension
the bridge could not use leaves its request correctly unproven, so Defect B's
unavailable results cannot become deliverable through this channel.
"""
from __future__ import annotations

import os
import warnings

import pytest

from mi_agent import execution_receipt as R


@pytest.fixture(scope="module")
def live():
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    before = dict(os.environ)
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext
    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)

    def ask(question, scope=None):
        return execute_governed_mi_query(
            MiQueryRequest(question=question,
                           source_portfolio_lens=scope), ctx).result or {}
    try:
        yield ask
    finally:
        os.environ.clear()
        os.environ.update(before)


def _meta(env):
    return env.get("metadata") or {}


def _facets(env):
    es = (env.get("executionSummary") or _meta(env).get("executionSummary") or {})
    return {(f.get("label"), f.get("status")) for f in (es.get("facets") or ())}


# --------------------------------------------------------------------------- #
# Positive: the executed axis is declared and the generic proof succeeds
# --------------------------------------------------------------------------- #
class TestTheExecutedGroupingIsDeclared:
    def test_bridge_by_region_declares_and_delivers(self, live):
        e = live("Funded balance bridge by region")
        assert _meta(e)["route"] == "funded_bridge"
        assert _meta(e)["groupedBy"] == ["collateral_geography"]
        assert ("region", "applied") in _facets(e)
        assert e["ok"] is True
        # the already-correct calculation, unchanged
        assert "£1.93bn" in e["answer"] and "£1.96bn" in e["answer"]

    def test_bridge_by_ltv_band_declares_the_dimension_it_used(self, live):
        """A different governed bridge dimension: the EXACT executed axis is
        declared, not a fixed route axis."""
        e = live("balance bridge by LTV band")
        assert _meta(e)["groupedBy"] == ["ltv_bucket"]
        assert ("ltv band", "applied") in _facets(e)
        assert e["ok"] is True

    def test_a_scoped_bridge_declares_the_same_axis(self, live):
        e = live("Funded balance bridge by region", "acquired")
        assert _meta(e)["groupedBy"] == ["collateral_geography"]
        assert ("region", "applied") in _facets(e)
        assert e["ok"] is True
        assert "£568.3m" in e["answer"]

    def test_the_generic_proof_is_what_certifies_it(self, live):
        """No route allowlist: `funded_bridge` is absent from
        ROUTE_DECLARED_AXES, so the certification stands on the declaration."""
        assert "funded_bridge" not in R.ROUTE_DECLARED_AXES
        e = live("Funded balance bridge by region")
        assert "collateral_geography" in R.declared_group_fields(e, "funded_bridge")


class TestGroupingAndFilterRolesDoNotCollapse:
    def test_a_companion_dimension_is_judged_on_its_own(self, live):
        """'by region for joint borrowers' — region is the executed axis and is
        applied; borrower type is judged separately and is NOT swept into the
        grouping declaration."""
        e = live("Bridge the funded balance by region for joint borrowers")
        assert _meta(e)["groupedBy"] == ["collateral_geography"]
        assert ("region", "applied") in _facets(e)
        assert ("joint borrower", "applied") not in _facets(e)


# --------------------------------------------------------------------------- #
# Negative: never declare what execution did not do
# --------------------------------------------------------------------------- #
class TestNeverOverDeclares:
    def test_a_missing_dimension_declares_nothing_and_still_refuses(self, live):
        """DEFECT B'S SAFETY GATE. The tape carries no product type, so the
        bridge is unavailable — it must not acquire a grouping declaration and
        must not become deliverable."""
        e = live("Bridge the funded balance by product")
        assert _meta(e).get("groupedBy") is None
        assert e["ok"] is False
        assert ("product", "applied") not in _facets(e)
        assert "£0" not in (e.get("answer") or "")

    def test_a_requested_dimension_alone_never_certifies(self, live):
        """The declaration is execution evidence: 'by product' REQUESTS a
        grouping, and no grouping field is declared for it."""
        e = live("Bridge the funded balance by product")
        assert "erm_product_type" not in R.declared_group_fields(e, "funded_bridge")

    def test_an_unavailable_bridge_declares_nothing(self):
        """Below the route: an unavailable calculation carries no dimensionCol,
        so there is nothing to declare even before the envelope is built."""
        from mi_agent_api import platform_snapshots_blob as blob
        import apps.blob_trigger_app.storage as storage_mod
        import pandas as pd
        from mi_agent_api import evolution as evo

        frame = pd.DataFrame({"current_outstanding_balance": [100_000, 100_000],
                              "geographic_region_obligor": ["A", "B"],
                              "source_portfolio_type": ["direct", "direct"]})
        orig_open, orig_frames = storage_mod.open_storage, blob.build_funded_evolution_frames
        try:
            storage_mod.open_storage = lambda: object()
            blob.build_funded_evolution_frames = lambda *a, **k: [
                {"run_id": "2026-03-31", "reporting_date": "2026-03-31",
                 "df": frame, "source": "x"}]
            br = evo.funded_bridge("blob://x", "client_001",
                                   ["geographic_region_obligor"])
        finally:
            storage_mod.open_storage, blob.build_funded_evolution_frames = orig_open, orig_frames
        assert br["available"] is False
        assert br.get("dimensionCol") is None


class TestTheWholeBookBridgeIsUnchanged:
    def test_a_bridge_naming_no_dimension_still_delivers(self, live):
        e = live("funded balance bridge")
        assert e["ok"] is True
        assert "£1.93bn" in e["answer"] and "£1.96bn" in e["answer"]

    def test_a_scoped_bridge_naming_no_dimension_still_delivers(self, live):
        e = live("Funded balance bridge for the acquired book")
        assert e["ok"] is True
        assert "£568.3m" in e["answer"]

    def test_an_unheld_portfolio_name_still_refuses(self, live):
        """A scope refusal is not a grouping refusal, and this fix must not
        touch it."""
        e = live("Funded balance bridge for the Highgate Mortgages Book")
        assert e["ok"] is False
        assert "Highgate Mortgages Book" in (e.get("answer") or "")
