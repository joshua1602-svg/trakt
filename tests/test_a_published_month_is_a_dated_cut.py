"""Every month the Operations Control Centre published was invisible to MI.

The OCC publishes a delivery under its reporting period, and a monthly period
is ``2026-08``: ``platform/ERE/2026-08/platform_canonical_typed.csv``. The MI
reader recognised only ``YYYY-MM-DD`` folders as dated cuts, so ERE's August and
every backfilled month since October 2025 were skipped. Evolution showed "No
periods available" with the history published, and the reporting-date control
offered only the ``latest`` copy.

A month folder is now a dated cut for that month's last day. A full-date folder
for the same day still wins, and a selected month-end loads its month folder.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from tests.test_platform_blob_snapshots import (_BlobPlatformFixture,
                                                _canonical_csv,
                                                _reset_read_cache)


class _MonthFolders(_BlobPlatformFixture):
    """What the OCC writes: month folders, plus the ``latest`` copy."""
    CUTS = {"2025-10": (5_000_000, 2_000_000),
            "2025-12": (6_000_000, 2_500_000),
            "2026-08": (7_000_000, 0)}

    def __init__(self, td: str):
        self.local_blob_root = Path(td) / "blobstore"
        base = self.local_blob_root / "processed-v2" / "platform" / "ERE"
        for cut, (d, a) in self.CUTS.items():
            (base / cut).mkdir(parents=True, exist_ok=True)
            (base / cut / "platform_canonical_typed.csv").write_text(
                _canonical_csv(d, a, cut + "-28"))
        (base / "latest").mkdir(parents=True, exist_ok=True)
        (base / "latest" / "platform_canonical_typed.csv").write_text(
            _canonical_csv(7_000_000, 0, "2026-08-31"))
        self.base = base


def _listing(fx):
    import mi_agent_api.platform_snapshots_blob as pb
    from apps.blob_trigger_app.storage import open_storage
    with fx.env():
        return pb.list_dated_platform_canonicals(fx.ROOT, open_storage())


def test_a_month_folder_is_a_cut_on_the_month_s_last_day():
    with tempfile.TemporaryDirectory() as td:
        dated = _listing(_MonthFolders(td))
    assert [d["date"] for d in dated] == ["2025-10-31", "2025-12-31", "2026-08-31"]
    assert all("/latest/" not in d["uri"] for d in dated)


def test_a_full_date_folder_for_the_same_day_wins():
    with tempfile.TemporaryDirectory() as td:
        fx = _MonthFolders(td)
        (fx.base / "2026-08-31").mkdir()
        (fx.base / "2026-08-31" / "platform_canonical_typed.csv").write_text(
            _canonical_csv(1, 1, "2026-08-31"))
        dated = _listing(fx)
    aug = [d for d in dated if d["date"] == "2026-08-31"]
    assert len(aug) == 1 and "/2026-08-31/" in aug[0]["uri"]


def test_a_selected_month_end_loads_its_month_folder():
    import mi_agent_api.platform_snapshots_blob as pb
    from apps.blob_trigger_app.storage import open_storage
    with tempfile.TemporaryDirectory() as td:
        fx = _MonthFolders(td)
        _reset_read_cache()
        with fx.env():
            df = pb.resolve_run_frame(fx.ROOT, open_storage(), None, "2025-12-31")
            etag = pb.canonical_etag(fx.ROOT, open_storage(), "2025-12-31")
    assert df is not None and len(df) == 3
    assert float(df["current_outstanding_balance"].sum()) == 6_000_000 * 2 + 2_500_000
    assert etag is not None


def test_a_date_that_is_not_a_month_end_does_not_borrow_the_month():
    import mi_agent_api.platform_snapshots_blob as pb
    from apps.blob_trigger_app.storage import open_storage
    with tempfile.TemporaryDirectory() as td:
        fx = _MonthFolders(td)
        _reset_read_cache()
        with fx.env():
            assert pb.resolve_run_frame(fx.ROOT, open_storage(), None,
                                        "2025-12-15") is None


def test_funded_evolution_sees_every_published_month():
    import mi_agent_api.app as app
    with tempfile.TemporaryDirectory() as td:
        fx = _MonthFolders(td)
        _reset_read_cache()
        with fx.env():
            resp = app.funded_evolution(portfolioId="ERE")
    periods = [p.get("reporting_date") or p.get("period") or p.get("run_id")
               for p in resp["periods"]]
    assert len(periods) == 3, resp.get("error")
    assert resp["singlePeriod"] is False
