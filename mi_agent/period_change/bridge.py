"""mi_agent.period_change.bridge — the deterministic balance bridge.

§14. Where a stable loan identifier and the governed balance field exist in both
snapshots, the opening balance is reconciled to the closing balance across four
movements that are facts about loan identity, not estimates:

    opening balance
      + closing balances of loans that are new in the end snapshot
      − opening balances of loans that exited before the end snapshot
      + balance movement on loans present at both dates
      = closing balance

The reconciliation is checked, not asserted, and a documented rounding tolerance
absorbs float accumulation. Anything that would require inference — a missing or
duplicated identifier, an absent balance field — produces an explicit limitation
and NO bridge. An estimated bridge is worse than no bridge: it looks like a
reconciliation and is not one.

Deliberately out of scope: a cashflow waterfall, an attribution model, or an
inferred transaction ledger. Those require data this workflow does not have.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import pandas as pd

from analytics_lib.numeric import coerce_numeric

from .calculations import BALANCE_FIELD, mixed_currency
from .models import (
    BRIDGE_ROUNDING_TOLERANCE,
    BRIDGE_STATUS_UNAVAILABLE_MIXED_CURRENCY,
    MIXED_CURRENCY_NOTE,
    BRIDGE_STATUS_AVAILABLE,
    BRIDGE_STATUS_DOES_NOT_RECONCILE,
    BRIDGE_STATUS_UNAVAILABLE_DUPLICATE_IDENTIFIERS,
    BRIDGE_STATUS_UNAVAILABLE_MISSING_IDENTIFIERS,
    BRIDGE_STATUS_UNAVAILABLE_NO_BALANCE_FIELD,
    BRIDGE_STATUS_UNAVAILABLE_NO_IDENTIFIER,
    BalanceBridge,
    SnapshotFrame,
    evidence_ref,
)

#: Canonical loan identity columns, most canonical first. ``loan_identifier`` is
#: the canonical field the rest of the platform keys loans on
#: (``mi_agent_api.snapshots._loan_ids``); the others are accepted only as the
#: same canonical concept under an alternative canonical name.
IDENTIFIER_FIELDS: Tuple[str, ...] = (
    "loan_identifier", "original_loan_identifier", "underlying_exposure_identifier")

#: Provenance column that qualifies a loan identifier. Originators reuse simple
#: sequences ("0001"), so on a consolidated book the identifier alone is not a
#: loan identity. Where this column exists in both snapshots the key becomes
#: composite, which is what stops originator A's loan 0001 exiting and
#: originator B's loan 0001 arriving from reading as one continuing loan.
QUALIFIER_FIELD = "source_portfolio_id"


def _resolve_identifier(start_df: Any, end_df: Any) -> Optional[str]:
    for column in IDENTIFIER_FIELDS:
        if (column in getattr(start_df, "columns", [])
                and column in getattr(end_df, "columns", [])):
            return column
    return None


def resolve_key_fields(start_df: Any, end_df: Any) -> Tuple[str, ...]:
    """The loan-identity key: composite where provenance allows, else bare.

    Returns ``()`` when no canonical identifier is present in both snapshots.
    """
    identifier = _resolve_identifier(start_df, end_df)
    if identifier is None:
        return ()
    if all(QUALIFIER_FIELD in getattr(f, "columns", [])
           for f in (start_df, end_df)):
        return (QUALIFIER_FIELD, identifier)
    return (identifier,)


def _key_series(df: Any, key_fields: Tuple[str, ...]) -> pd.Series:
    """One normalised identity string per row, from one or more columns."""
    parts = []
    for column in key_fields:
        parts.append(df[column].astype("string").str.strip().str.lower())
    key = parts[0]
    for extra in parts[1:]:
        key = key.str.cat(extra, sep="␟", na_rep="")
    return key


def _keyed(df: Any, key_fields: Tuple[str, ...]) -> Tuple[pd.Series, int, int]:
    """``(balance_by_key, missing_identifier_count, duplicate_key_count)``.

    A row is "missing" when ANY component of the key is blank: a composite key
    with a blank half identifies nothing.
    """
    blank = None
    for column in key_fields:
        col = df[column].astype("string").str.strip()
        col_blank = col.isna() | col.eq("") | col.str.lower().isin(
            ["nan", "none", "<na>", "null"])
        blank = col_blank if blank is None else (blank | col_blank)

    missing = int(blank.sum())
    keys = _key_series(df, key_fields)[~blank]
    duplicates = int(len(keys) - keys.nunique())
    balances = coerce_numeric(df[BALANCE_FIELD]).fillna(0.0)[~blank]
    return pd.Series(balances.values, index=keys.values), missing, duplicates


def balance_bridge(start_snapshot: SnapshotFrame, end_snapshot: SnapshotFrame
                   ) -> BalanceBridge:
    """Reconcile the opening balance to the closing balance, or explain why not."""
    start_df, end_df = start_snapshot.frame, end_snapshot.frame
    if any(BALANCE_FIELD not in getattr(f, "columns", [])
           for f in (start_df, end_df)):
        return BalanceBridge(
            status=BRIDGE_STATUS_UNAVAILABLE_NO_BALANCE_FIELD,
            limitation=(f"A balance bridge needs the governed balance field "
                        f"{BALANCE_FIELD!r} in both snapshots. It is not "
                        f"available, so no bridge is reported."))

    if any(mixed_currency(f) for f in (start_df, end_df)):
        return BalanceBridge(
            status=BRIDGE_STATUS_UNAVAILABLE_MIXED_CURRENCY,
            balance_field=BALANCE_FIELD,
            limitation=(MIXED_CURRENCY_NOTE + " A bridge that added them would "
                        "reconcile arithmetically while meaning nothing, so no "
                        "bridge is reported."))

    key_fields = resolve_key_fields(start_df, end_df)
    if not key_fields:
        return BalanceBridge(
            status=BRIDGE_STATUS_UNAVAILABLE_NO_IDENTIFIER,
            balance_field=BALANCE_FIELD,
            limitation=("A balance bridge needs a stable loan identifier "
                        "present in both snapshots. None of the canonical "
                        f"identifier fields ({', '.join(IDENTIFIER_FIELDS)}) is "
                        "available, so no bridge is reported and none is "
                        "estimated."))
    identifier = " + ".join(key_fields)

    start_balances, start_missing, start_dupes = _keyed(start_df, key_fields)
    end_balances, end_missing, end_dupes = _keyed(end_df, key_fields)

    if start_dupes or end_dupes:
        return BalanceBridge(
            status=BRIDGE_STATUS_UNAVAILABLE_DUPLICATE_IDENTIFIERS,
            balance_field=BALANCE_FIELD, identifier_field=identifier,
            identifier_fields=key_fields,
            limitation=(f"{start_dupes} opening and {end_dupes} closing records "
                        f"share a duplicate {identifier!r}. Loan identity cannot "
                        f"be established, so no bridge is reported."))
    if start_missing or end_missing:
        return BalanceBridge(
            status=BRIDGE_STATUS_UNAVAILABLE_MISSING_IDENTIFIERS,
            balance_field=BALANCE_FIELD, identifier_field=identifier,
            identifier_fields=key_fields,
            limitation=(f"{start_missing} opening and {end_missing} closing "
                        f"records carry no {identifier!r}. A bridge over a "
                        f"partial population would not reconcile to the book, "
                        f"so no bridge is reported."))

    start_ids = set(start_balances.index)
    end_ids = set(end_balances.index)
    continuing = start_ids & end_ids
    new_ids = end_ids - start_ids
    exited_ids = start_ids - end_ids

    opening = float(start_balances.sum())
    closing = float(end_balances.sum())
    new_balance = float(end_balances.loc[sorted(new_ids)].sum()) if new_ids else 0.0
    exited_balance = (float(start_balances.loc[sorted(exited_ids)].sum())
                      if exited_ids else 0.0)
    continuing_movement = (
        float(end_balances.loc[sorted(continuing)].sum()
              - start_balances.loc[sorted(continuing)].sum())
        if continuing else 0.0)

    reconstructed = opening + new_balance - exited_balance + continuing_movement
    residual = closing - reconstructed
    reconciles = abs(residual) <= BRIDGE_ROUNDING_TOLERANCE

    return BalanceBridge(
        status=(BRIDGE_STATUS_AVAILABLE if reconciles
                else BRIDGE_STATUS_DOES_NOT_RECONCILE),
        balance_field=BALANCE_FIELD, identifier_field=identifier,
        identifier_fields=key_fields,
        opening_balance=opening, closing_balance=closing,
        new_loan_balance=new_balance, exited_loan_balance=exited_balance,
        continuing_movement=continuing_movement,
        new_loan_count=len(new_ids), exited_loan_count=len(exited_ids),
        continuing_loan_count=len(continuing),
        reconciles=reconciles, residual=residual,
        limitation=(None if reconciles else
                    (f"The bridge does not reconcile: a residual of "
                     f"{residual:,.2f} remains outside the documented "
                     f"{BRIDGE_ROUNDING_TOLERANCE} tolerance.")),
        evidence=(
            evidence_ref("balance_bridge", start_snapshot,
                         field_name=BALANCE_FIELD,
                         detail={"role": "opening", "loan_count": len(start_ids)}),
            evidence_ref("balance_bridge", end_snapshot,
                         field_name=BALANCE_FIELD,
                         detail={"role": "closing", "loan_count": len(end_ids)}),
        ))
