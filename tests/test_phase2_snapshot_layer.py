"""Phase 2 — snapshot/history layer tests.

Use temporary directories and small pandas DataFrames. Prove registration,
idempotency, conflict-refusal, listing/filtering, temporal resolution,
loan round-trip, deterministic keys, strict date separation, and graceful
handling of optional-field gaps. No legacy ``analytics/`` or Azure imports.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from snapshot import (
    SnapshotConflictError,
    SnapshotHeader,
    SnapshotNotFoundError,
    SnapshotValidationError,
    compute_source_file_id,
    make_pipeline_opportunity_id,
    make_snapshot_id,
    select_stable_loan_key,
)
from snapshot.adapters import LocalFsSnapshotStore
from snapshot import model as M

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_frame(ids=("A1", "A2", "A3"), with_segmentation=True):
    data = {
        "loan_identifier": list(ids),
        "current_outstanding_balance": [100_000.0 + i for i in range(len(ids))],
        "origination_date": ["2019-05-01"] * len(ids),
    }
    if with_segmentation:
        data["portfolio_id"] = ["P1"] * len(ids)
        data["spv_id"] = ["S1"] * len(ids)
        data["acquired_portfolio_id"] = ["AP1"] * len(ids)
    return pd.DataFrame(data)


def make_header(source_file_id="sha256:aaa", reporting_date="2024-01-31",
                route="mi", cadence="monthly",
                upload_timestamp="2024-02-03T09:30:00", **over):
    kwargs = dict(
        client_id="clientA", route=route, reporting_date=reporting_date,
        source_file_id=source_file_id, cadence=cadence,
        upload_timestamp=upload_timestamp, source_file_name="tape.csv",
    )
    kwargs.update(over)
    return SnapshotHeader(**kwargs)


@pytest.fixture
def store(tmp_path):
    return LocalFsSnapshotStore(root=tmp_path / "snaps")


# --------------------------------------------------------------------------- #
# 1. Registration writes header, loans, manifest, latest pointer
# --------------------------------------------------------------------------- #


def test_register_writes_all_artifacts(store, tmp_path):
    res = store.register_snapshot(make_header(), make_frame())
    assert res.created and not res.idempotent
    sid = res.snapshot_id

    # Manifest + header + loans on disk.
    assert (store.root / "manifest.json").exists()
    snap_dir = store.root / "clienta" / "mi" / sid
    assert (snap_dir / "header.json").exists()
    assert (snap_dir / "loans.csv").exists()

    # Latest pointer.
    pointer = store.get_latest_pointer("clientA", "mi")
    assert pointer and pointer["snapshot_id"] == sid

    # get_snapshot round-trips the header.
    hdr = store.get_snapshot(sid)
    assert hdr.client_id == "clientA" and hdr.row_count == 3


# --------------------------------------------------------------------------- #
# 2. Idempotency vs conflict
# --------------------------------------------------------------------------- #


def test_duplicate_same_source_is_idempotent(store):
    r1 = store.register_snapshot(make_header(), make_frame())
    r2 = store.register_snapshot(make_header(), make_frame())
    assert r2.snapshot_id == r1.snapshot_id
    assert r2.idempotent and not r2.created
    assert any(i["code"] == M.DUPLICATE_SNAPSHOT_SAME_SOURCE for i in r2.issues)
    # Only one snapshot persisted.
    assert len(store.list_snapshots("clientA")) == 1


def test_same_id_conflicting_content_refused(store):
    store.register_snapshot(make_header(), make_frame())
    # Same source_file_id (=> same snapshot_id) but different content.
    with pytest.raises(SnapshotConflictError):
        store.register_snapshot(make_header(), make_frame(ids=("X1", "X2")))


def test_same_slot_different_source_refused(store):
    store.register_snapshot(make_header(source_file_id="sha256:aaa"),
                            make_frame())
    # Same client/route/reporting_date/cadence, different source content.
    with pytest.raises(SnapshotConflictError):
        store.register_snapshot(make_header(source_file_id="sha256:bbb"),
                                make_frame(ids=("Z1",)))


# --------------------------------------------------------------------------- #
# 3. Listing / filtering
# --------------------------------------------------------------------------- #


def _seed_multi(store):
    store.register_snapshot(
        make_header(source_file_id="sha256:1", reporting_date="2024-01-31"),
        make_frame())
    store.register_snapshot(
        make_header(source_file_id="sha256:2", reporting_date="2024-02-29"),
        make_frame())
    store.register_snapshot(
        make_header(source_file_id="sha256:3", reporting_date="2024-03-31",
                    cadence="weekly"),
        make_frame())
    store.register_snapshot(
        make_header(source_file_id="sha256:4", reporting_date="2024-02-15",
                    route="mna"),
        make_frame())


def test_list_filters(store):
    _seed_multi(store)
    assert len(store.list_snapshots("clientA")) == 4
    assert len(store.list_snapshots("clientA", route="mi")) == 3
    assert len(store.list_snapshots("clientA", route="mna")) == 1
    assert len(store.list_snapshots("clientA", cadence="weekly")) == 1
    rng = store.list_snapshots("clientA", route="mi",
                               since="2024-02-01", until="2024-03-31")
    assert [h.reporting_date for h in rng] == ["2024-02-29", "2024-03-31"]
    assert store.list_snapshots("otherclient") == []


# --------------------------------------------------------------------------- #
# 4. Temporal resolution
# --------------------------------------------------------------------------- #


def test_resolve_latest(store):
    _seed_multi(store)
    latest = store.resolve_latest("clientA", route="mi")
    assert latest.reporting_date == "2024-03-31"


def test_resolve_as_of(store):
    _seed_multi(store)
    hdr = store.resolve_as_of("clientA", "2024-02-10", route="mi")
    assert hdr.reporting_date == "2024-01-31"  # latest on/before 2024-02-10
    hdr2 = store.resolve_as_of("clientA", "2024-02-29", route="mi")
    assert hdr2.reporting_date == "2024-02-29"


def test_resolve_range_ordered(store):
    _seed_multi(store)
    rng = store.resolve_range("clientA", "2024-01-01", "2024-03-01", route="mi")
    assert [h.reporting_date for h in rng] == ["2024-01-31", "2024-02-29"]


def test_resolve_compare(store):
    _seed_multi(store)
    baseline, current = store.resolve_compare(
        "clientA", "2024-01-31", "2024-03-31", route="mi")
    assert baseline.reporting_date == "2024-01-31"
    assert current.reporting_date == "2024-03-31"


def test_resolve_latest_missing_raises(store):
    with pytest.raises(SnapshotNotFoundError):
        store.resolve_latest("nobody")


# --------------------------------------------------------------------------- #
# 5. Loan round-trip + reserved columns
# --------------------------------------------------------------------------- #


def test_load_loans_round_trip(store):
    res = store.register_snapshot(make_header(), make_frame(ids=("A1", "A2")))
    loans = store.load_loans(res.snapshot_id)
    assert list(loans["loan_id"]) == ["A1", "A2"]  # stable funded ids
    assert set(loans["loan_identifier"]) == {"A1", "A2"}
    assert list(loans["current_outstanding_balance"]) == [100_000.0, 100_001.0]
    for col in ("snapshot_id", "client_id", "reporting_date",
                "cut_off_date", "upload_timestamp"):
        assert col in loans.columns
    assert set(loans["snapshot_id"]) == {res.snapshot_id}
    assert set(loans["client_id"]) == {"clientA"}


# --------------------------------------------------------------------------- #
# 6. Date separation (the key invariant)
# --------------------------------------------------------------------------- #


def test_upload_timestamp_not_conflated_with_reporting_date(store):
    res = store.register_snapshot(
        make_header(reporting_date="2024-01-31",
                    upload_timestamp="2024-02-03T09:30:00"),
        make_frame())
    hdr = store.get_snapshot(res.snapshot_id)
    assert hdr.reporting_date == "2024-01-31"
    assert hdr.upload_timestamp == "2024-02-03T09:30:00"
    assert hdr.reporting_date != hdr.upload_timestamp[:10]
    loans = store.load_loans(res.snapshot_id)
    assert set(loans["reporting_date"]) == {"2024-01-31"}
    assert set(loans["upload_timestamp"]) == {"2024-02-03T09:30:00"}


def test_missing_reporting_date_fails_clearly(store):
    with pytest.raises(SnapshotValidationError):
        store.register_snapshot(make_header(reporting_date=None), make_frame())


def test_missing_reporting_date_allowed_for_tests(tmp_path):
    store = LocalFsSnapshotStore(root=tmp_path / "s",
                                 allow_missing_reporting_date=True)
    res = store.register_snapshot(make_header(reporting_date=None), make_frame())
    assert res.created
    assert any(i["code"] == M.MISSING_REPORTING_DATE for i in res.issues)


def test_cut_off_defaulting_configurable(tmp_path):
    # Default off: cut_off stays empty, no default issue.
    off = LocalFsSnapshotStore(root=tmp_path / "off")
    r1 = off.register_snapshot(make_header(cut_off_date=None), make_frame())
    assert off.get_snapshot(r1.snapshot_id).cut_off_date is None

    # Default on: cut_off becomes reporting_date with an issue recorded.
    on = LocalFsSnapshotStore(root=tmp_path / "on",
                              default_cut_off_to_reporting=True)
    r2 = on.register_snapshot(make_header(cut_off_date=None), make_frame())
    hdr = on.get_snapshot(r2.snapshot_id)
    assert hdr.cut_off_date == hdr.reporting_date
    assert any(i["code"] == M.CUT_OFF_DATE_DEFAULTED_TO_REPORTING_DATE
               for i in r2.issues)


def test_dates_stored_iso(store):
    res = store.register_snapshot(
        make_header(reporting_date="2024/01/31"), make_frame())
    assert store.get_snapshot(res.snapshot_id).reporting_date == "2024-01-31"


# --------------------------------------------------------------------------- #
# 7. Optional-field gaps produce issues, not crashes
# --------------------------------------------------------------------------- #


def test_missing_segmentation_fields_produce_issues(store):
    res = store.register_snapshot(
        make_header(), make_frame(with_segmentation=False))
    codes = [i["code"] for i in res.issues]
    seg_issues = [i for i in res.issues
                  if i["code"] == M.MISSING_OPTIONAL_SEGMENTATION_FIELD]
    assert {i["field"] for i in seg_issues} == {"portfolio_id", "spv_id",
                                                "acquired_portfolio_id"}
    assert res.created  # did not crash


def test_pipeline_rows_get_opportunity_id(store):
    # No funded loan key columns -> opportunity id is derived; loan_id is in the
    # OPP_ namespace, kept distinct from funded ids.
    frame = pd.DataFrame({
        "kfi_number": ["K1", "K2"],
        "broker": ["B", "B"],
        "loan_amount": [50_000, 60_000],
        "current_outstanding_balance": [0, 0],
    })
    res = store.register_snapshot(make_header(), frame)
    loans = store.load_loans(res.snapshot_id)
    assert all(str(x).startswith("OPP_") for x in loans["loan_id"])
    assert loans["opportunity_id"].notna().all()
    assert loans["stable_entity_id"].isna().all()


def test_rows_with_no_key_recorded_not_crashed(store):
    frame = pd.DataFrame({"current_outstanding_balance": [1.0, 2.0]})
    res = store.register_snapshot(make_header(), frame)
    assert any(i["code"] == M.MISSING_STABLE_LOAN_KEY for i in res.issues)
    assert res.created


# --------------------------------------------------------------------------- #
# 8. Deterministic keys
# --------------------------------------------------------------------------- #


def test_compute_source_file_id_deterministic():
    a = compute_source_file_id(b"hello world")
    b = compute_source_file_id(b"hello world")
    c = compute_source_file_id(b"different")
    assert a == b and a != c and a.startswith("sha256:")


def test_compute_source_file_id_from_path(tmp_path):
    p = tmp_path / "tape.csv"
    p.write_bytes(b"col\n1\n")
    assert compute_source_file_id(p) == compute_source_file_id(b"col\n1\n")


def test_make_snapshot_id_deterministic():
    a = make_snapshot_id("clientA", "mi", "2024-01-31", "sha256:x")
    b = make_snapshot_id("clientA", "mi", "2024-01-31", "sha256:x")
    c = make_snapshot_id("clientA", "mi", "2024-02-29", "sha256:x")
    assert a == b and a != c and a.startswith("snap_")


def test_opportunity_id_deterministic_and_distinct_from_loan_key():
    row1 = {"kfi_number": "K1", "broker": "B", "loan_amount": 50_000}
    row2 = {"kfi_number": "K1", "broker": "B", "loan_amount": 50_000}
    row3 = {"kfi_number": "K9", "broker": "B", "loan_amount": 50_000}
    assert make_pipeline_opportunity_id(row1) == make_pipeline_opportunity_id(row2)
    assert make_pipeline_opportunity_id(row1) != make_pipeline_opportunity_id(row3)
    assert make_pipeline_opportunity_id({"unrelated": "x"}) is None
    # Funded loan key is a real id, never an opp_ hash.
    assert select_stable_loan_key({"loan_identifier": "L100"}) == "L100"
    assert make_pipeline_opportunity_id(row1).startswith("opp_")


# --------------------------------------------------------------------------- #
# 9. No forbidden imports
# --------------------------------------------------------------------------- #


def test_no_legacy_or_azure_imports():
    pkg = REPO_ROOT / "snapshot"
    banned = ("from analytics", "import streamlit", "import plotly",
              "import azure", "from azure", "azure.storage", "BlobServiceClient")
    for py in pkg.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{py} contains forbidden {token!r}"
        for line in text.splitlines():
            s = line.strip()
            assert s != "import analytics" and not s.startswith("import analytics."), s
