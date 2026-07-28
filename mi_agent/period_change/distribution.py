"""mi_agent.period_change.distribution — composition shift for BSR dimensions.

§13. A field whose governed role is ``dimension`` and whose governed aggregation
is ``distribution`` does not have a portfolio-level value that moves; its
*composition* moves. This module reports that movement per category:

    category · start count · end count · start count share · end count share ·
    count-share movement · balance and balance share where the governed balance
    field is available · balance-share movement

and ranks the largest positive and negative shifts.

What it deliberately does not do
--------------------------------
* it does not compute a concentration index — that is
  ``analytics_lib.concentration`` / ``mi_agent.risk_monitor.concentration``
  territory and inventing a second one here would create two answers;
* it does not call a category increase good or bad. Only a governed ordering
  (the risk monitor's configured orderings) can say that, and this workflow has
  none; the shift is reported and left as an observation.

Category shares are computed against each snapshot's OWN denominator, so a
changing portfolio size is handled correctly: shares always sum to 1 within a
snapshot, and the movement is the difference between two well-formed shares.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from analytics_lib.numeric import coerce_numeric
from analytics_lib.stratify import UNKNOWN_LABEL

from ..business_semantics import BusinessSemanticsEntry
from .calculations import BALANCE_FIELD
from .models import (
    STATUS_AVAILABLE,
    STATUS_NOT_AVAILABLE,
    STATUS_NOT_COMPARABLE_DUE_TO_AVAILABILITY,
    STATUS_PARTIALLY_AVAILABLE,
    CategoryShift,
    DistributionChange,
    SnapshotFrame,
    evidence_ref,
)

#: How many categories each ranking reports. A distribution with fifty
#: categories is not made clearer by listing all fifty as "notable".
TOP_SHIFTS = 5

PRESENCE_BOTH = "both"
PRESENCE_START_ONLY = "start_only"
PRESENCE_END_ONLY = "end_only"


def normalise_categories(series: pd.Series) -> pd.Series:
    """Canonical category labels for a dimension column.

    Trims, and buckets every missing / blank / placeholder value into the single
    ``Unknown`` label ``analytics_lib.stratify`` already uses — so a
    period-change distribution names its missing bucket exactly as every other
    stratification in the platform does, and a value that is blank at one date
    and null at the other does not appear as two different categories.
    """
    text = series.astype("string").str.strip()
    blank = text.isna() | text.eq("") | text.str.lower().isin(
        ["nan", "nat", "none", "<na>", "null"])
    return text.where(~blank, UNKNOWN_LABEL)


def _snapshot_table(df: Any, field: str, *, with_balance: bool
                    ) -> Tuple[Dict[str, Dict[str, float]], int, Optional[float]]:
    """``({category: {count, balance}}, total_count, total_balance)``."""
    categories = normalise_categories(df[field])
    total_count = int(len(categories))
    balances: Optional[pd.Series] = None
    if with_balance:
        balances = coerce_numeric(df[BALANCE_FIELD]).fillna(0.0)

    table: Dict[str, Dict[str, float]] = {}
    for category, index in categories.groupby(categories).groups.items():
        entry = {"count": float(len(index))}
        if balances is not None:
            entry["balance"] = float(balances.loc[index].sum())
        table[str(category)] = entry

    total_balance = float(balances.sum()) if balances is not None else None
    return table, total_count, total_balance


def _share(value: Optional[float], total: Optional[float]) -> Optional[float]:
    if value is None or not total:
        return None
    return value / total


def distribution_change(entry: BusinessSemanticsEntry,
                        start_snapshot: SnapshotFrame,
                        end_snapshot: SnapshotFrame) -> DistributionChange:
    """The composition shift of one governed dimension between two snapshots."""
    field = entry.field
    start_df, end_df = start_snapshot.frame, end_snapshot.frame
    in_start = start_df is not None and field in getattr(start_df, "columns", [])
    in_end = end_df is not None and field in getattr(end_df, "columns", [])

    if not in_start and not in_end:
        return DistributionChange(
            field=field, display_name=entry.display_name,
            analytical_concept=entry.analytical_concept,
            status=STATUS_NOT_AVAILABLE, categories=(), balance_field=None,
            start_total_count=0, end_total_count=0,
            start_total_balance=None, end_total_balance=None,
            confidence=entry.confidence, rationale=entry.rationale,
            notes=(f"Field {field!r} is present in neither snapshot.",))

    if not in_start or not in_end:
        return DistributionChange(
            field=field, display_name=entry.display_name,
            analytical_concept=entry.analytical_concept,
            status=STATUS_NOT_COMPARABLE_DUE_TO_AVAILABILITY, categories=(),
            balance_field=None,
            start_total_count=int(len(start_df)) if in_start else 0,
            end_total_count=int(len(end_df)) if in_end else 0,
            start_total_balance=None, end_total_balance=None,
            confidence=entry.confidence, rationale=entry.rationale,
            notes=(f"Field {field!r} is reported at only one of the two "
                   f"reporting dates, so its composition shift cannot be "
                   f"calculated.",))

    with_balance = all(BALANCE_FIELD in getattr(f, "columns", [])
                       for f in (start_df, end_df))
    start_table, start_count, start_balance = _snapshot_table(
        start_df, field, with_balance=with_balance)
    end_table, end_count, end_balance = _snapshot_table(
        end_df, field, with_balance=with_balance)

    shifts: List[CategoryShift] = []
    for category in sorted(set(start_table) | set(end_table)):
        s = start_table.get(category)
        e = end_table.get(category)
        presence = (PRESENCE_BOTH if s and e
                    else PRESENCE_START_ONLY if s else PRESENCE_END_ONLY)
        s_count = int(s["count"]) if s else 0
        e_count = int(e["count"]) if e else 0
        s_share = _share(float(s_count), float(start_count))
        e_share = _share(float(e_count), float(end_count))
        s_balance = (s or {}).get("balance") if with_balance else None
        e_balance = (e or {}).get("balance") if with_balance else None
        if with_balance:
            s_balance = float(s_balance or 0.0)
            e_balance = float(e_balance or 0.0)
        s_bshare = _share(s_balance, start_balance) if with_balance else None
        e_bshare = _share(e_balance, end_balance) if with_balance else None

        shifts.append(CategoryShift(
            category=category, start_count=s_count, end_count=e_count,
            start_count_share=s_share, end_count_share=e_share,
            count_share_movement=(None if s_share is None or e_share is None
                                  else e_share - s_share),
            start_balance=s_balance, end_balance=e_balance,
            start_balance_share=s_bshare, end_balance_share=e_bshare,
            balance_share_movement=(None if s_bshare is None or e_bshare is None
                                    else e_bshare - s_bshare),
            presence=presence))

    # Each ranking is sorted in its own direction, with the category name as a
    # total tie-break, so two equal shifts always order the same way.
    moved = [c for c in shifts if c.count_share_movement is not None]
    increases = tuple(
        c.category for c in sorted(
            (c for c in moved if c.count_share_movement > 0),
            key=lambda c: (-c.count_share_movement, c.category)))[:TOP_SHIFTS]
    decreases = tuple(
        c.category for c in sorted(
            (c for c in moved if c.count_share_movement < 0),
            key=lambda c: (c.count_share_movement, c.category)))[:TOP_SHIFTS]

    notes: List[str] = []
    new_categories = [c.category for c in shifts if c.presence == PRESENCE_END_ONLY]
    gone_categories = [c.category for c in shifts if c.presence == PRESENCE_START_ONLY]
    if new_categories:
        notes.append(f"{len(new_categories)} category value(s) appear only in "
                     f"the closing snapshot: {', '.join(new_categories[:5])}.")
    if gone_categories:
        notes.append(f"{len(gone_categories)} category value(s) appear only in "
                     f"the opening snapshot: {', '.join(gone_categories[:5])}.")
    if start_count != end_count:
        notes.append(
            f"The population changed from {start_count:,} to {end_count:,} "
            f"records; shares are calculated against each snapshot's own total.")
    if not with_balance:
        notes.append(
            f"Balance shares are not reported: the governed balance field "
            f"{BALANCE_FIELD!r} is not available in both snapshots.")

    unknown = next((c for c in shifts if c.category == UNKNOWN_LABEL), None)
    status = STATUS_AVAILABLE
    if unknown and (unknown.start_count or unknown.end_count):
        status = STATUS_PARTIALLY_AVAILABLE
        notes.append(
            f"{unknown.start_count} opening and {unknown.end_count} closing "
            f"records carry no value for this dimension and are reported in the "
            f"'{UNKNOWN_LABEL}' category rather than dropped.")

    return DistributionChange(
        field=field, display_name=entry.display_name,
        analytical_concept=entry.analytical_concept, status=status,
        categories=tuple(shifts),
        balance_field=BALANCE_FIELD if with_balance else None,
        start_total_count=start_count, end_total_count=end_count,
        start_total_balance=start_balance, end_total_balance=end_balance,
        largest_increases=increases, largest_decreases=decreases,
        confidence=entry.confidence, rationale=entry.rationale,
        evidence=(
            evidence_ref("distribution", start_snapshot, field_name=field,
                         detail={"role": "start", "total_count": start_count}),
            evidence_ref("distribution", end_snapshot, field_name=field,
                         detail={"role": "end", "total_count": end_count}),
        ),
        notes=tuple(notes))
