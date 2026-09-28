#!/usr/bin/env python3
"""The production catalogue, as a `SnapshotStore`. Scope, shape and fail-closed.

The end-to-end reconciliation through `serve` lives in
`due_diligence/evidence/plan_temporal_slice2/temporal_production_binding.py`.
What is asserted here is the adapter's own contract: that it reads production's
index shape, that it creates no catalogue of its own, and that the client
boundary is a property of the object rather than of how a caller happens to
call it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from mi_agent_api import governed_snapshot_store as store_mod
from snapshot.model import SnapshotNotFoundError

CLIENT = "ERE"
OTHER = "OTHER_TENANT"

INDEX = {
    "portfolios": [
        {"client_id": CLIENT, "label": "ERE", "runs": [
            {"run_id": "run_20260531", "reporting_date": "2026-05-31",
             "loan_count": 368},
            {"run_id": "run_20260630", "reporting_date": "2026-06-30",
             "loan_count": 386},
            # A run with no resolvable period. A period that cannot be named
            # cannot be selected, and guessing one would be a substitution.
            {"run_id": "run_unknown", "reporting_date": None, "loan_count": 1},
        ]},
        {"client_id": OTHER, "label": "Other", "runs": [
            {"run_id": "run_20250331", "reporting_date": "2025-03-31",
             "loan_count": 9}]},
    ],
    "source": "test",
}


def loader(client_id, run_id, root):
    return pd.DataFrame({"loan_identifier": ["L1"]}), {"run_id": run_id}


def bound(client_id=CLIENT):
    return store_mod.GovernedFundedSnapshotStore(
        index_provider=lambda: dict(INDEX), frame_loader=loader,
        client_id=client_id)


class TestShape:
    def test_it_reads_the_production_index_contract(self):
        headers = bound().list_snapshots(CLIENT,
                                         route=store_mod.FUNDED_ROUTE)
        assert [h.reporting_date for h in headers] == ["2026-05-31", "2026-06-30"]
        assert [h.snapshot_id for h in headers] == [
            f"{CLIENT}/run_20260531", f"{CLIENT}/run_20260630"]
        assert {h.route for h in headers} == {store_mod.FUNDED_ROUTE}
        assert {h.cadence for h in headers} == {"monthly"}

    def test_a_run_with_no_reporting_date_is_not_selectable(self):
        headers = bound().list_snapshots(CLIENT, route=store_mod.FUNDED_ROUTE)
        assert all(h.reporting_date for h in headers)
        assert len(headers) == 2                 # the third run is dropped

    def test_the_identity_is_the_production_portfolio_id_form(self):
        assert store_mod.snapshot_id_for("ERE", "run_1") == "ERE/run_1"
        assert store_mod.split_snapshot_id("ERE/run_1") == ("ERE", "run_1")


class TestScope:
    def test_a_bound_store_speaks_for_one_client_only(self):
        store = bound()
        assert store.list_snapshots(OTHER, route=store_mod.FUNDED_ROUTE) == []
        with pytest.raises(SnapshotNotFoundError):
            store.get_snapshot(f"{OTHER}/run_20250331")

    def test_another_clients_run_cannot_be_loaded_by_id(self):
        """The hole the cross-portfolio control found, asserted so it stays shut.

        `get_snapshot` once derived the client from the snapshot id, so a store
        built for one tenant would resolve and load another's run if handed its
        id. Nothing on the temporal path does that; a boundary that depends on
        no caller trying is not a boundary.
        """
        with pytest.raises(SnapshotNotFoundError):
            bound().load_loans(f"{OTHER}/run_20250331")

    def test_a_run_the_catalogue_does_not_list_is_not_loadable(self):
        """The only approval semantics the estate keeps today."""
        with pytest.raises(SnapshotNotFoundError):
            bound().load_loans(f"{CLIENT}/run_19990101")

    def test_the_pipeline_route_is_not_this_catalogue(self):
        assert bound().list_snapshots(CLIENT, route="pipeline") == []


class TestFailClosed:
    def test_an_unreadable_catalogue_yields_no_history(self):
        def boom():
            raise RuntimeError("blob unavailable")
        store = store_mod.GovernedFundedSnapshotStore(
            index_provider=boom, frame_loader=loader, client_id=CLIENT)
        assert store.list_snapshots(CLIENT, route=store_mod.FUNDED_ROUTE) == []

    def test_an_empty_run_is_a_missing_snapshot_not_an_empty_answer(self):
        store = store_mod.GovernedFundedSnapshotStore(
            index_provider=lambda: dict(INDEX),
            frame_loader=lambda *a: (pd.DataFrame(), {}), client_id=CLIENT)
        with pytest.raises(SnapshotNotFoundError):
            store.load_loans(f"{CLIENT}/run_20260630")

    def test_a_loader_fault_is_a_missing_snapshot(self):
        def angry(*_):
            raise IOError("storage down")
        store = store_mod.GovernedFundedSnapshotStore(
            index_provider=lambda: dict(INDEX), frame_loader=angry,
            client_id=CLIENT)
        with pytest.raises(SnapshotNotFoundError):
            store.load_loans(f"{CLIENT}/run_20260630")

    def test_the_query_path_may_not_publish_history(self):
        with pytest.raises(NotImplementedError):
            bound().register_snapshot(None, pd.DataFrame())

    def test_build_store_never_raises(self):
        class Broken:
            snapshot_index = None
            def _onboarding_output_root(self):
                raise RuntimeError("no root")
        assert store_mod.build_store(Broken()) is None


class TestNoSecondCatalogue:
    def test_the_adapter_invents_no_dates_and_no_settings(self):
        """It adapts a catalogue; it does not become one.

        No reporting date literal, no environment variable, and no filesystem
        walk of its own. The on-disk walk that exists is reached THROUGH
        `datasets.snapshot_index`, where production's resolution order already
        lives.
        """
        source = Path(store_mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value
                if len(text) == 10 and text[4] == "-" and text[7] == "-" \
                        and text[:4].isdigit():
                    raise AssertionError(f"a reporting date literal: {text!r}")
        for forbidden in ("os.environ", "getenv", "glob(", "listdir", "walk(",
                          "open("):
            assert forbidden not in source, f"the adapter reaches {forbidden}"

    def test_it_reads_the_catalogue_production_already_owns(self):
        import inspect
        source = inspect.getsource(store_mod.build_store)
        assert "snapshot_index" in source
        assert "_resolve_run_dataframe" in source
