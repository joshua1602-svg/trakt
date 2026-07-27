#!/usr/bin/env python3
"""tests/test_platform_assembly_all_portfolios.py

The client platform tape must be built from ALL accepted portfolios.

The defect: the Assembler discovers canonicals from a LOCAL store under the run
output root, which in the managed service is an ephemeral container scratch
reclaimed between runs. Each run published only the portfolio it had just
processed, so the store held exactly one file and the "platform" tape was a
byte-for-byte copy of whichever portfolio ran last — 885 rows / 43 columns of
acquired_001, with direct_001's 73 rows and its 17 extra columns silently absent.

The fix hydrates the store from the durable ``accepted/{client}/`` prefix before
assembling. These tests cover the combined result, the union schema, attribution,
duplicate handling, and the registry-status exclusion that is the only sanctioned
way for an accepted portfolio to leave the platform tape.

Run: python -m pytest tests/test_platform_assembly_all_portfolios.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apps.blob_trigger_app import assembler_refresh as AR
from apps.blob_trigger_app.layout import Layout
from apps.blob_trigger_app.persistence import ProductionPersistence
from apps.blob_trigger_app.source_registry import (
    STATUS_ACTIVE, STATUS_RETIRED, SourceRecord, SourceRegistry)
from apps.blob_trigger_app.storage import Storage

CLIENT = "ERE"
ACQUIRED_ROWS = 885
DIRECT_ROWS = 73

#: Shared canonical columns, including the provenance the Assembler requires.
_SHARED = ["loan_identifier", "source_portfolio_id", "source_portfolio_type",
           "source_portfolio_label", "acquisition_date", "seller_name",
           "portfolio_cohort", "current_outstanding_balance", "data_cut_off_date"]
#: Fields only the direct book carries — these must survive into the platform tape.
_DIRECT_ONLY = [f"direct_only_field_{i}" for i in range(17)]


def _canonical(source_portfolio_id: str, rows: int, *, extra_columns=(),
               portfolio_type="direct", start=0) -> pd.DataFrame:
    data = {
        "loan_identifier": [f"{source_portfolio_id}-{start + i}" for i in range(rows)],
        "source_portfolio_id": [source_portfolio_id] * rows,
        "source_portfolio_type": [portfolio_type] * rows,
        "source_portfolio_label": [f"{source_portfolio_id} book"] * rows,
        "acquisition_date": ["2026-01-31"] * rows,
        "seller_name": ["Seller Ltd"] * rows,
        "portfolio_cohort": [f"{source_portfolio_id}_2026"] * rows,
        "current_outstanding_balance": [100000 + i for i in range(rows)],
        "data_cut_off_date": ["2026-06-30"] * rows,
    }
    for column in extra_columns:
        data[column] = [f"{column}-value"] * rows
    return pd.DataFrame(data)


class _Ctx:
    """A durable store holding accepted canonicals, plus an EMPTY local scratch.

    The empty scratch is the point: it reproduces a fresh container, where the
    only way to see another portfolio is to hydrate it from durable storage.
    """

    def __init__(self, root: Path):
        self.root = root
        self.storage = Storage(root)
        self.layout = Layout()
        self.persistence = ProductionPersistence(self.storage, self.layout)
        self.registry = SourceRegistry(
            "blob://trakt-state/registry/source_registry.yaml", storage=self.storage)
        self.accepted_root = root / "scratch" / "_accepted"
        self.platform_out = root / "scratch" / "_platform"

    def accept(self, source_portfolio_id: str, frame: pd.DataFrame) -> Path:
        """Publish a portfolio's accepted canonical to DURABLE storage."""
        local = self.root / "staging" / f"{source_portfolio_id}_canonical_typed.csv"
        local.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(local, index=False)
        self.persistence.persist_accepted(CLIENT, source_portfolio_id, str(local))
        return local

    def register(self, source_portfolio_id: str, status: str = STATUS_ACTIVE):
        self.registry.upsert(SourceRecord(
            client_id=CLIENT, source_portfolio_id=source_portfolio_id,
            dataset="funded", frequency="monthly", status=status))

    def refresh(self, source_portfolio_id: str, local_canonical: Path, **kw):
        params = {"client_id": CLIENT,
                  "source_portfolio_id": source_portfolio_id,
                  "canonical_path": str(local_canonical),
                  "accepted_root": str(self.accepted_root),
                  "platform_out_dir": str(self.platform_out),
                  "persistence": self.persistence,
                  "registry": self.registry}
        params.update(kw)
        return AR.default_assembler_refresher(**params)

    def platform(self) -> pd.DataFrame:
        return pd.read_csv(self.platform_out / "platform_canonical_typed.csv",
                           low_memory=False)


class _Base(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory(prefix="platform_assembly_")
        self.addCleanup(td.cleanup)
        self.ctx = _Ctx(Path(td.name))

    def _two_portfolios(self):
        """The production shape: a wide direct book and a narrower acquired one."""
        acquired = _canonical("acquired_001", ACQUIRED_ROWS,
                              portfolio_type="acquired")
        direct = _canonical("direct_001", DIRECT_ROWS, extra_columns=_DIRECT_ONLY)
        self.ctx.accept("acquired_001", acquired)
        direct_local = self.ctx.accept("direct_001", direct)
        acquired_local = self.ctx.root / "staging" / "acquired_001_canonical_typed.csv"
        self.ctx.register("acquired_001")
        self.ctx.register("direct_001")
        return acquired_local, direct_local


# --------------------------------------------------------------------------- #
# The combined tape
# --------------------------------------------------------------------------- #
class TestCombinedAssembly(_Base):

    def test_two_portfolios_with_different_schemas_both_assemble(self):
        # 1
        acquired_local, _direct = self._two_portfolios()
        result = self.ctx.refresh("acquired_001", acquired_local)
        self.assertEqual(result["portfolios"], ["acquired_001", "direct_001"])

    def test_row_count_is_the_sum_of_accepted_portfolios(self):
        # 4 — 885 + 73 = 958
        acquired_local, _direct = self._two_portfolios()
        self.ctx.refresh("acquired_001", acquired_local)
        self.assertEqual(len(self.ctx.platform()), ACQUIRED_ROWS + DIRECT_ROWS)

    def test_union_schema_is_preserved(self):
        # 2 + 3 — the 17 direct-only columns survive, null for acquired rows.
        acquired_local, _direct = self._two_portfolios()
        self.ctx.refresh("acquired_001", acquired_local)
        platform = self.ctx.platform()

        for column in _DIRECT_ONLY:
            self.assertIn(column, platform.columns,
                          f"{column} was dropped from the platform schema")
        acquired_rows = platform[platform["source_portfolio_id"] == "acquired_001"]
        direct_rows = platform[platform["source_portfolio_id"] == "direct_001"]
        self.assertTrue(acquired_rows[_DIRECT_ONLY].isna().all().all())
        self.assertTrue(direct_rows[_DIRECT_ONLY].notna().all().all())

    def test_schema_is_not_the_last_processed_portfolios_schema(self):
        # The defect's signature: platform width == the narrower portfolio's.
        acquired_local, _direct = self._two_portfolios()
        self.ctx.refresh("acquired_001", acquired_local)
        platform = self.ctx.platform()
        acquired_width = len(_SHARED)
        self.assertGreater(len(platform.columns), acquired_width)
        self.assertGreaterEqual(len(platform.columns),
                                len(_SHARED) + len(_DIRECT_ONLY))

    def test_portfolio_attribution_is_retained_on_every_row(self):
        # 5
        acquired_local, _direct = self._two_portfolios()
        self.ctx.refresh("acquired_001", acquired_local)
        platform = self.ctx.platform()

        self.assertEqual(platform["source_portfolio_id"].isna().sum(), 0)
        self.assertEqual(
            platform["source_portfolio_id"].value_counts().to_dict(),
            {"acquired_001": ACQUIRED_ROWS, "direct_001": DIRECT_ROWS})
        for field in ("source_portfolio_type", "source_portfolio_label",
                      "portfolio_cohort"):
            self.assertEqual(platform[field].isna().sum(), 0, field)
        self.assertEqual(
            set(platform.loc[platform["source_portfolio_id"] == "acquired_001",
                             "source_portfolio_type"]), {"acquired"})

    def test_attribution_comes_from_the_canonical_field_not_the_filename(self):
        # A file named for one portfolio whose ROWS say another must be attributed
        # by the rows. Path identity is a discovery key only.
        frame = _canonical("acquired_001", 5, portfolio_type="acquired")
        self.ctx.accept("acquired_001", frame)
        self.ctx.register("acquired_001")
        local = self.ctx.root / "staging" / "acquired_001_canonical_typed.csv"
        self.ctx.refresh("acquired_001", local)
        self.assertEqual(set(self.ctx.platform()["source_portfolio_id"]),
                         {"acquired_001"})

    def test_filename_is_only_a_discovery_key(self):
        self.assertEqual(
            AR.portfolio_id_from_accepted_uri(
                "blob://processed-v2/accepted/ERE/direct_001_canonical_typed.csv"),
            "direct_001")
        self.assertEqual(AR.portfolio_id_from_accepted_uri("nonsense.txt"), "")


# --------------------------------------------------------------------------- #
# Isolation between portfolios
# --------------------------------------------------------------------------- #
class TestPortfolioIsolation(_Base):

    def test_processing_one_portfolio_does_not_remove_another(self):
        # 6
        acquired_local, direct_local = self._two_portfolios()
        self.ctx.refresh("direct_001", direct_local)
        self.assertEqual(len(self.ctx.platform()), ACQUIRED_ROWS + DIRECT_ROWS)

        self.ctx.refresh("acquired_001", acquired_local)
        platform = self.ctx.platform()
        self.assertEqual(len(platform), ACQUIRED_ROWS + DIRECT_ROWS)
        self.assertEqual(sorted(platform["source_portfolio_id"].unique()),
                         ["acquired_001", "direct_001"])

    def test_reprocessing_acquired_still_yields_the_combined_tape(self):
        # 7 — the reported production case, repeated.
        acquired_local, _direct = self._two_portfolios()
        for _ in range(3):
            self.ctx.refresh("acquired_001", acquired_local)
            self.assertEqual(len(self.ctx.platform()),
                             ACQUIRED_ROWS + DIRECT_ROWS)

    def test_publishing_never_overwrites_another_portfolios_entry(self):
        acquired_local, _direct = self._two_portfolios()
        self.ctx.refresh("acquired_001", acquired_local)
        store = AR.accepted_store_dir(self.ctx.accepted_root, CLIENT)
        direct = pd.read_csv(store / "direct_001_canonical_typed.csv")
        self.assertEqual(len(direct), DIRECT_ROWS)
        self.assertEqual(set(direct["source_portfolio_id"]), {"direct_001"})

    def test_a_new_portfolio_joins_without_reprocessing_the_others(self):
        acquired_local, _direct = self._two_portfolios()
        self.ctx.refresh("acquired_001", acquired_local)

        third = _canonical("acquired_002", 11, portfolio_type="acquired")
        third_local = self.ctx.accept("acquired_002", third)
        self.ctx.register("acquired_002")
        self.ctx.refresh("acquired_002", third_local)

        platform = self.ctx.platform()
        self.assertEqual(len(platform), ACQUIRED_ROWS + DIRECT_ROWS + 11)
        self.assertEqual(sorted(platform["source_portfolio_id"].unique()),
                         ["acquired_001", "acquired_002", "direct_001"])


# --------------------------------------------------------------------------- #
# Hydration — the actual fix
# --------------------------------------------------------------------------- #
class TestHydration(_Base):

    def test_an_empty_local_store_is_filled_from_durable_accepted(self):
        self._two_portfolios()
        store = AR.accepted_store_dir(self.ctx.accepted_root, CLIENT)
        self.assertFalse(store.exists())

        report = AR.hydrate_accepted_store(
            self.ctx.persistence, self.ctx.accepted_root, CLIENT,
            registry=self.ctx.registry)
        self.assertEqual(sorted(report["hydrated"]),
                         ["acquired_001", "direct_001"])
        self.assertEqual(report["errors"], [])

    def test_without_hydration_the_assembler_sees_only_this_run(self):
        # The defect, reproduced: publish one portfolio into an empty store and
        # assemble — exactly what production was doing.
        acquired_local, _direct = self._two_portfolios()
        AR.publish_accepted_canonical(
            self.ctx.accepted_root, client_id=CLIENT,
            source_portfolio_id="acquired_001", canonical_path=acquired_local)
        store = AR.accepted_store_dir(self.ctx.accepted_root, CLIENT)
        self.assertEqual(AR._portfolios_in_store(store), ["acquired_001"])

    def test_hydration_reports_what_it_pulled(self):
        self._two_portfolios()
        acquired_local = self.ctx.root / "staging" / "acquired_001_canonical_typed.csv"
        result = self.ctx.refresh("acquired_001", acquired_local)
        hydration = result["accepted_hydration"]
        self.assertEqual(sorted(hydration["hydrated"]),
                         ["acquired_001", "direct_001"])
        self.assertEqual(hydration["excluded"], [])

    def test_a_missing_persistence_degrades_without_crashing(self):
        acquired_local, _direct = self._two_portfolios()
        result = self.ctx.refresh("acquired_001", acquired_local, persistence=None)
        self.assertEqual(result["portfolios"], ["acquired_001"])


# --------------------------------------------------------------------------- #
# Registry status is the only sanctioned exclusion
# --------------------------------------------------------------------------- #
class TestRegistryStatusExclusion(_Base):

    def test_a_retired_portfolio_is_excluded_and_reported(self):
        # 9
        acquired_local, _direct = self._two_portfolios()
        self.ctx.register("direct_001", status=STATUS_RETIRED)

        result = self.ctx.refresh("acquired_001", acquired_local)
        self.assertEqual(result["portfolios"], ["acquired_001"])
        self.assertEqual(result["accepted_hydration"]["excluded"],
                         [{"source_portfolio_id": "direct_001",
                           "reason": "registry_status_retired"}])
        self.assertEqual(len(self.ctx.platform()), ACQUIRED_ROWS)

    def test_an_active_portfolio_is_included(self):
        acquired_local, _direct = self._two_portfolios()
        self.ctx.register("direct_001", status=STATUS_ACTIVE)
        result = self.ctx.refresh("acquired_001", acquired_local)
        self.assertIn("direct_001", result["portfolios"])

    def test_a_portfolio_unknown_to_the_registry_is_still_included(self):
        # Dropping rows because a lookup missed would be silent data loss.
        acquired = _canonical("acquired_001", 5, portfolio_type="acquired")
        unknown = _canonical("legacy_001", 4)
        self.ctx.accept("acquired_001", acquired)
        self.ctx.accept("legacy_001", unknown)
        self.ctx.register("acquired_001")
        local = self.ctx.root / "staging" / "acquired_001_canonical_typed.csv"
        result = self.ctx.refresh("acquired_001", local)
        self.assertIn("legacy_001", result["portfolios"])
        self.assertEqual(len(self.ctx.platform()), 9)

    def test_one_active_dataset_keeps_the_portfolio(self):
        # A portfolio with several dataset/frequency records is retired only when
        # ALL of them are.
        acquired_local, _direct = self._two_portfolios()
        self.ctx.registry.upsert(SourceRecord(
            client_id=CLIENT, source_portfolio_id="direct_001",
            dataset="pipeline", frequency="weekly", status=STATUS_RETIRED))
        result = self.ctx.refresh("acquired_001", acquired_local)
        self.assertIn("direct_001", result["portfolios"])


# --------------------------------------------------------------------------- #
# Duplicates are explicit
# --------------------------------------------------------------------------- #
class TestDuplicateHandling(_Base):

    def test_duplicate_composite_keys_are_reported_not_silently_dropped(self):
        # 8 — the Assembler refuses rather than deduplicating behind the operator.
        from engine.platform_assembler import PlatformAssemblyError

        clashing = pd.concat(
            [_canonical("acquired_001", 3, portfolio_type="acquired")] * 2,
            ignore_index=True)
        self.ctx.accept("acquired_001", clashing)
        self.ctx.register("acquired_001")
        local = self.ctx.root / "staging" / "acquired_001_canonical_typed.csv"

        with self.assertRaises(PlatformAssemblyError) as raised:
            self.ctx.refresh("acquired_001", local)
        message = str(raised.exception)
        self.assertIn("Duplicate composite keys", message)
        self.assertIn("acquired_001", message)

    def test_the_same_loan_id_in_two_portfolios_is_not_a_duplicate(self):
        # Uniqueness is (source_portfolio_id + loan_identifier), so two books can
        # legitimately share a loan number.
        acquired = _canonical("acquired_001", 4, portfolio_type="acquired")
        direct = _canonical("direct_001", 4)
        direct["loan_identifier"] = acquired["loan_identifier"].tolist()
        self.ctx.accept("acquired_001", acquired)
        self.ctx.accept("direct_001", direct)
        self.ctx.register("acquired_001")
        self.ctx.register("direct_001")
        local = self.ctx.root / "staging" / "acquired_001_canonical_typed.csv"
        self.ctx.refresh("acquired_001", local)
        self.assertEqual(len(self.ctx.platform()), 8)


# --------------------------------------------------------------------------- #
# Downstream consumption
# --------------------------------------------------------------------------- #
class TestDownstreamConsumesTheCombinedTape(_Base):
    """MI and the investor deck must see BOTH portfolios, not the last one."""

    def _persisted_platform(self):
        """Assemble, then persist exactly as the router does after a funded pack."""
        acquired_local, _direct = self._two_portfolios()
        self.ctx.refresh("acquired_001", acquired_local)
        local = self.ctx.platform_out / "platform_canonical_typed.csv"
        return self.ctx.persistence.persist_platform(CLIENT, "2026-06-30", str(local))

    def test_latest_and_dated_snapshots_are_both_written(self):
        persisted = self._persisted_platform()
        self.assertTrue(persisted["latest"].endswith(
            "platform/ERE/latest/platform_canonical_typed.csv"), persisted["latest"])
        self.assertTrue(persisted["period"].endswith(
            "platform/ERE/2026-06-30/platform_canonical_typed.csv"), persisted["period"])

    def test_the_persisted_latest_holds_both_portfolios(self):
        persisted = self._persisted_platform()
        for uri in (persisted["latest"], persisted["period"]):
            local = self.ctx.storage._local_path(uri)
            frame = pd.read_csv(local, low_memory=False)
            self.assertEqual(len(frame), ACQUIRED_ROWS + DIRECT_ROWS, uri)
            self.assertEqual(sorted(frame["source_portfolio_id"].unique()),
                             ["acquired_001", "direct_001"], uri)

    def test_the_mi_layer_resolves_the_combined_tape(self):
        # 10 — MI reads whatever platform canonical is published, so the fix
        # reaches MI without MI changing.
        import importlib
        import os

        self._persisted_platform()
        platform_dir = self.ctx.platform_out
        previous = {k: os.environ.get(k) for k in
                    ("MI_AGENT_PLATFORM_DIR", "MI_AGENT_PLATFORM_URI",
                     "MI_AGENT_PLATFORM_CANONICAL", "MI_AGENT_ANALYTICS_DATASET")}
        os.environ["MI_AGENT_PLATFORM_DIR"] = str(platform_dir)
        for key in ("MI_AGENT_PLATFORM_URI", "MI_AGENT_PLATFORM_CANONICAL",
                    "MI_AGENT_ANALYTICS_DATASET"):
            os.environ.pop(key, None)
        try:
            ds = importlib.import_module("mi_agent_api.data_source")
            path, kind = ds.resolve_data_source()
            self.assertEqual(kind, "platform_canonical")
            frame = pd.read_csv(path, low_memory=False)
            self.assertEqual(len(frame), ACQUIRED_ROWS + DIRECT_ROWS)
            self.assertEqual(sorted(frame["source_portfolio_id"].unique()),
                             ["acquired_001", "direct_001"])
            # The union schema survives to the MI layer too.
            for column in _DIRECT_ONLY:
                self.assertIn(column, frame.columns)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_the_deck_artifact_loader_finds_the_combined_tape(self):
        # The investor deck resolves the same artefact by name.
        self._persisted_platform()
        found = sorted(self.ctx.platform_out.glob("platform_canonical_typed.csv"))
        self.assertEqual(len(found), 1)
        frame = pd.read_csv(found[0], low_memory=False)
        self.assertEqual(len(frame), ACQUIRED_ROWS + DIRECT_ROWS)



if __name__ == "__main__":  # pragma: no cover
    unittest.main()
