#!/usr/bin/env python3
"""A governed snapshot catalogue built from the independent truth oracle.

The frames come from `portfolio_truth_oracle.canonical_history`, which imports
nothing from the product; they are registered through the product's OWN
`LocalFsSnapshotStore`, because snapshot registration and resolution are part of
what slice 2 is being measured on. So the data is independent and the catalogue
is real.

Every header declares `cadence="monthly"`. A catalogue that declares no cadence
cannot honour a stated grain — `plan_temporal_runtime._cadence_check` refuses
one — and a fixture that omitted it would be testing that refusal instead of the
series.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from snapshot.adapters import LocalFsSnapshotStore
from snapshot.model import SnapshotHeader

from . import portfolio_truth_oracle as truth

CLIENT_ID = "oracle_client"
ROUTE = "funded"


def build_store(root: Any, history: Sequence[Tuple[str, Any]], *,
                client_id: str = CLIENT_ID, route: str = ROUTE,
                cadence: Optional[str] = "monthly") -> LocalFsSnapshotStore:
    """Register `history` into a fresh store under `root` and return it."""
    store = LocalFsSnapshotStore(root=Path(root))
    for reporting_date, frame in history:
        header = SnapshotHeader(
            client_id=client_id, route=route, reporting_date=reporting_date,
            source_file_id=f"{client_id}-{route}-{reporting_date}",
            cadence=cadence, cut_off_date=reporting_date,
            source_file_name=f"{reporting_date}.csv", row_count=int(len(frame)))
        store.register_snapshot(header, frame)
    return store


def default_history(periods: int = 8) -> List[Tuple[str, Any]]:
    """The standard eight-month fixture: 2025-11-30 … 2026-06-30.

    Eight months so every month name occurs at most ONCE, which is what lets
    "since March" resolve to a single governed period. The thirteen-month
    variant below exists to prove the opposite case.
    """
    return truth.canonical_history(periods=periods)


def repeated_month_history() -> List[Tuple[str, Any]]:
    """Thirteen months, so March occurs twice and a bare "March" is ambiguous."""
    return truth.canonical_history(periods=13, start_year=2025, start_month=3)
