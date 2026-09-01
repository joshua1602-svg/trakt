"""mi_agent_api/funded_composition — why the funded book moved, not just by how much.

A reporting month in which a book was acquired must not be described as ordinary
organic growth. The two are different facts about the business and a reader who
cannot tell them apart is being misinformed by a true number: "funded assets
increased 46%" is arithmetically correct and, on an acquisition month, useless.

THE DECOMPOSITION
-----------------
One partition of both frames, so the components sum EXACTLY to the movement::

    opening funded balance            (prior reporting period)
      + portfolio additions           source portfolios present now, absent prior
      - portfolio disposals           source portfolios present prior, absent now
      + organic new lending           new loans inside CONTINUING portfolios
      - exits                         departed loans inside CONTINUING portfolios
      +/- existing-book movement      balance change on loans present in both
    = closing funded balance

The identity holds by construction rather than by tolerance. Each frame is
partitioned twice — first by whether the loan's source portfolio continues, then,
within the continuing portfolios, by whether the loan itself continues — so every
pound of both frames lands in exactly one component. ``reconciles`` reports the
residual anyway: a partition that stopped partitioning should say so rather than
be trusted because it once did.

WHERE "ACQUISITION" COMES FROM, AND WHERE IT DOES NOT
----------------------------------------------------
It comes from governed portfolio identity. A source portfolio appearing for the
first time is an observed **portfolio addition** — a fact about identity, not
about size. Whether that addition is an ACQUISITION is a separate question,
answered only by:

  1. an explicit ``source_portfolio_type`` on the rows, or
  2. ``engine.provenance.derive_portfolio_type`` over the governed id.

When neither resolves, the addition is reported as ``unclassified`` and is
described as a new source portfolio — never as an acquisition. A large balance
movement is never, anywhere in this module, evidence that a book was bought.
That rule exists because the inference is so easy to make and so damaging when
wrong: a bulk re-boarding of the existing book onto a new id would present as an
acquisition to anything reading the number instead of the identity.

SCALE
-----
No portfolio id, prefix, count or client appears here. A second, fifth or
twentieth acquired book is data: it arrives with a governed
``source_portfolio_id``, is absent from the prior frame, and decomposes. Nothing
below needs to be edited for it.

DEGRADATION
-----------
The two granularities fail independently. Without ``source_portfolio_id`` the
portfolio-level split is not derivable and is reported unavailable — while the
loan-level components still are. Without a loan identifier the reverse holds.
Each missing piece is named; neither is substituted by the other.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

import pandas as pd

from analytics_lib.numeric import coerce_numeric

from . import evolution as evolution_mod
from .movement_summary import _cohorts

_BALANCE = "current_outstanding_balance"
_PORTFOLIO_ID = "source_portfolio_id"
_PORTFOLIO_TYPE = "source_portfolio_type"
_PORTFOLIO_LABEL = "source_portfolio_label"
_ACQUISITION_DATE = "acquisition_date"

#: Governed portfolio classifications. ``unclassified`` is a first-class value,
#: not a failure: it is what an addition is called when identity does not say.
TYPE_ACQUIRED = "acquired"
TYPE_DIRECT = "direct"
TYPE_UNCLASSIFIED = "unclassified"

#: Component keys, in statement order. The order is the reading order of the
#: reconciliation, so a renderer never has to know one.
COMPONENTS: Sequence[str] = (
    "portfolio_additions",
    "portfolio_disposals",
    "organic_new_lending",
    "exits",
    "existing_book_movement",
)

#: A component smaller than this share of the gross movement is not named in a
#: narrative. Presentation only — it never changes a number, and the full
#: decomposition is always returned.
NARRATIVE_SHARE_FLOOR = 0.05


def _num(series: pd.Series) -> pd.Series:
    return coerce_numeric(series)


def _balance(df: Optional[pd.DataFrame], mask: Optional[pd.Series] = None) -> float:
    if df is None or df.empty or _BALANCE not in df.columns:
        return 0.0
    frame = df if mask is None else df[mask]
    return round(float(_num(frame[_BALANCE]).fillna(0.0).sum()), 2)


def _portfolio_ids(df: Optional[pd.DataFrame]) -> Set[str]:
    if df is None or df.empty or _PORTFOLIO_ID not in df.columns:
        return set()
    ids = df[_PORTFOLIO_ID].dropna().astype(str).str.strip()
    return {i for i in ids.unique() if i and i.lower() != "nan"}


def _in_portfolios(df: pd.DataFrame, ids: Set[str]) -> pd.Series:
    if _PORTFOLIO_ID not in df.columns:
        return pd.Series(False, index=df.index)
    return df[_PORTFOLIO_ID].astype(str).str.strip().isin(ids)


def _loan_ids(df: Optional[pd.DataFrame], col: Optional[str]) -> Set[str]:
    if df is None or df.empty or not col or col not in df.columns:
        return set()
    ids = df[col].dropna().astype(str).str.strip()
    return {i for i in ids.unique() if i and i.lower() not in ("nan", "none")}


def classify_portfolio(df: Optional[pd.DataFrame], portfolio_id: str) -> str:
    """``acquired`` / ``direct`` / ``unclassified`` for one governed id.

    Identity only, in the governed order of authority: what the rows assert,
    then what the id itself encodes. Never the balance, the row count, or the
    fact that the portfolio is new — none of those is evidence of provenance.
    """
    if df is not None and not df.empty and _PORTFOLIO_TYPE in df.columns:
        rows = df[df[_PORTFOLIO_ID].astype(str).str.strip() == portfolio_id]
        values = rows[_PORTFOLIO_TYPE].dropna().astype(str).str.strip().str.lower()
        values = values[~values.isin(("", "nan", "none"))]
        if not values.empty:
            stated = values.iloc[0]
            if stated in (TYPE_ACQUIRED, TYPE_DIRECT):
                return stated

    try:
        from engine.provenance import derive_portfolio_type
    except Exception:  # noqa: BLE001 - the contract module is optional at runtime
        return TYPE_UNCLASSIFIED
    return derive_portfolio_type(portfolio_id) or TYPE_UNCLASSIFIED


def _portfolio_detail(df: Optional[pd.DataFrame], portfolio_id: str,
                      label_by_id: Dict[str, str]) -> Dict[str, Any]:
    """One added or disposed portfolio, as identity describes it."""
    detail: Dict[str, Any] = {
        "source_portfolio_id": portfolio_id,
        "label": label_by_id.get(portfolio_id, portfolio_id),
        "portfolio_type": classify_portfolio(df, portfolio_id),
        "balance": 0.0,
        "loan_count": 0,
    }
    if df is not None and not df.empty and _PORTFOLIO_ID in df.columns:
        mask = df[_PORTFOLIO_ID].astype(str).str.strip() == portfolio_id
        detail["balance"] = _balance(df, mask)
        detail["loan_count"] = int(mask.sum())
        if _ACQUISITION_DATE in df.columns:
            dates = df.loc[mask, _ACQUISITION_DATE].dropna().astype(str)
            dates = dates[~dates.str.strip().str.lower().isin(("", "nan", "nat"))]
            if not dates.empty:
                detail["acquisition_date"] = str(dates.iloc[0]).strip()[:10]
    return detail


def _labels(df: Optional[pd.DataFrame]) -> Dict[str, str]:
    return {c["id"]: c["label"] for c in _cohorts(df)}


def decompose(current: Optional[pd.DataFrame],
              prior: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """The governed movement decomposition between two prepared funded frames.

    Pure: two frames in, one structure out. No discovery, no I/O, no service
    call — so the whole rule is testable against frames a test states outright,
    and the reconciliation is checkable by hand.
    """
    opening = _balance(prior)
    closing = _balance(current)
    movement = round(closing - opening, 2)

    result: Dict[str, Any] = {
        "available": True,
        "opening_balance": opening,
        "closing_balance": closing,
        "movement": movement,
        "components": {k: 0.0 for k in COMPONENTS},
        "portfolio_additions": [],
        "portfolio_disposals": [],
        "continuing_portfolio_ids": [],
        "unavailable": {},
    }

    if current is None or prior is None:
        result["available"] = False
        result["unavailable"]["comparison"] = (
            "two governed funded reporting periods are required")
        return result

    # ---- portfolio-level: identity, never size -------------------------- #
    cur_pf, pri_pf = _portfolio_ids(current), _portfolio_ids(prior)
    portfolio_level = bool(cur_pf or pri_pf)
    if not portfolio_level:
        # Fail closed on the acquisition claim specifically. Everything below
        # still runs; what is lost is only the split this column would give.
        result["unavailable"][_PORTFOLIO_ID] = (
            "the governed funded tape carries no source_portfolio_id, so a "
            "portfolio addition cannot be distinguished from organic growth")
        added: Set[str] = set()
        removed: Set[str] = set()
        continuing = set()
    else:
        added = cur_pf - pri_pf
        removed = pri_pf - cur_pf
        continuing = cur_pf & pri_pf

    cur_labels, pri_labels = _labels(current), _labels(prior)
    result["portfolio_additions"] = [
        _portfolio_detail(current, pid, cur_labels) for pid in sorted(added)]
    result["portfolio_disposals"] = [
        _portfolio_detail(prior, pid, pri_labels) for pid in sorted(removed)]
    result["continuing_portfolio_ids"] = sorted(continuing)

    additions_total = round(
        sum(p["balance"] for p in result["portfolio_additions"]), 2)
    disposals_total = round(
        -sum(p["balance"] for p in result["portfolio_disposals"]), 2)

    # ---- loan-level, WITHIN the continuing portfolios -------------------- #
    if portfolio_level:
        cur_cont = current[_in_portfolios(current, continuing)]
        pri_cont = prior[_in_portfolios(prior, continuing)]
    else:
        cur_cont, pri_cont = current, prior

    loan_col = (evolution_mod._loan_id_col(current)
                or evolution_mod._loan_id_col(prior))
    continuing_movement = round(_balance(cur_cont) - _balance(pri_cont), 2)

    if not loan_col:
        # Without a loan key the continuing book cannot be split into new,
        # departed and held. It is reported whole rather than apportioned by a
        # guess, and the reason is named.
        result["unavailable"]["loan_identifier"] = (
            "the governed funded tape carries no loan identifier, so new "
            "lending, exits and existing-book movement cannot be separated")
        result["components"] = {
            "portfolio_additions": additions_total,
            "portfolio_disposals": disposals_total,
            "organic_new_lending": None,
            "exits": None,
            "existing_book_movement": continuing_movement,
        }
    else:
        cur_ids = _loan_ids(cur_cont, loan_col)
        pri_ids = _loan_ids(pri_cont, loan_col)
        cur_key = cur_cont[loan_col].astype(str).str.strip()
        pri_key = pri_cont[loan_col].astype(str).str.strip()

        held = cur_ids & pri_ids
        organic = _balance(cur_cont, ~cur_key.isin(pri_ids))
        departed = -_balance(pri_cont, ~pri_key.isin(cur_ids))
        existing = round(_balance(cur_cont, cur_key.isin(held))
                         - _balance(pri_cont, pri_key.isin(held)), 2)

        result["components"] = {
            "portfolio_additions": additions_total,
            "portfolio_disposals": disposals_total,
            "organic_new_lending": organic,
            "exits": departed,
            "existing_book_movement": existing,
        }
        result["counts"] = {
            "new_loans": len(cur_ids - pri_ids),
            "exited_loans": len(pri_ids - cur_ids),
            "held_loans": len(held),
        }

    stated = [v for v in result["components"].values() if v is not None]
    total = round(sum(stated), 2)
    result["reconciliation"] = {
        "sum_of_components": total,
        "movement": movement,
        "residual": round(total - movement, 2),
        "reconciles": abs(total - movement) < 0.01,
        "basis": ("one partition of both frames: portfolio identity first, then "
                  "loan identity within the continuing portfolios"),
    }
    result["loan_identifier_field"] = loan_col
    return result


# --------------------------------------------------------------------------- #
# The two lenses
# --------------------------------------------------------------------------- #
def underlying_lens_filters(decomposition: Dict[str, Any]
                            ) -> Optional[Dict[str, Any]]:
    """Lens filters selecting the EXISTING book — the continuing portfolios.

    Returns the filters in the shape ``evolution._scope_frame_lens`` already
    consumes, so "excluding the acquisition" is the existing population
    mechanism pointed at a different id list. Nothing here defines a second
    population: an underlying-book answer and a Direct-lens answer are narrowed
    by one function over one column.

    ``None`` when there is nothing to exclude — no additions, or no governed
    portfolio identity — so a caller cannot present a Total answer as an
    underlying one.
    """
    if not decomposition.get("portfolio_additions"):
        return None
    continuing = decomposition.get("continuing_portfolio_ids") or []
    if not continuing:
        return None
    return {_PORTFOLIO_ID: list(continuing)}


def dominant_addition(decomposition: Dict[str, Any], *,
                      share_floor: float = 0.5) -> Optional[Dict[str, Any]]:
    """The added portfolio that accounts for most of the movement, if one does.

    Deterministic, and deliberately a share of the MOVEMENT rather than of the
    book: the question a monthly review has to settle is whether the period is
    explained by an addition, not whether the addition is large.
    """
    additions = decomposition.get("portfolio_additions") or []
    movement = decomposition.get("movement")
    if not additions or not movement:
        return None
    lead = max(additions, key=lambda p: p["balance"])
    share = lead["balance"] / movement if movement else None
    if share is None or share < share_floor:
        return None
    return {**lead, "share_of_movement": round(share, 4)}


# --------------------------------------------------------------------------- #
# Service entry — the same frames every other funded surface reads
# --------------------------------------------------------------------------- #
def composition_movement(output_root, client_id: str, *,
                         to_run_id: Optional[str] = None,
                         lens_filters: Optional[Dict[str, Any]] = None,
                         lens_label: str = "Total",
                         span_periods: int = 1,
                         scope=None) -> Dict[str, Any]:
    """Decompose the funded movement for one client over ``span_periods``.

    Resolves through ``evolution.funded_frames`` — the SAME period discovery,
    the same prepared frames and the same lens the evolution series, the bridge
    and ``period_movement`` use. Two surfaces disagreeing about which two months
    they compared is the failure this shares a resolver to avoid.
    """
    frames = evolution_mod.funded_frames(output_root, client_id, to_run_id,
                                         scope=scope)
    scoped = []
    for f in frames:
        d = evolution_mod._scope_frame_lens(f.get("df"), lens_filters)
        if d is not None and len(d):
            scoped.append({**f, "df": d})

    span = max(1, int(span_periods or 1))
    if len(scoped) <= span:
        return {"available": False, "lens": lens_label,
                "spanRequested": span, "periodsAvailable": len(scoped),
                "reason": ("at least two governed funded reporting periods are "
                           "needed to decompose a movement")}

    cur, pri = scoped[-1], scoped[-1 - span]
    out = decompose(cur["df"], pri["df"])
    out.update({
        "lens": lens_label,
        "currentPeriod": cur.get("run_id"),
        "priorPeriod": pri.get("run_id"),
        "currentReportingDate": cur.get("reporting_date"),
        "priorReportingDate": pri.get("reporting_date"),
        "sourceFiles": [pri.get("source"), cur.get("source")],
    })
    return out


def narrative_components(decomposition: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Components worth naming, largest absolute first.

    Presentation only. Every component is always present in ``components``; this
    decides which are big enough to put in a sentence, so a £4k exit does not
    appear beside a £68m addition as though the two were comparable.
    """
    components = decomposition.get("components") or {}
    gross = sum(abs(v) for v in components.values() if v)
    if not gross:
        return []
    rows = [
        {"component": key, "amount": value,
         "share_of_gross": round(abs(value) / gross, 4)}
        for key, value in components.items()
        if value and abs(value) / gross >= NARRATIVE_SHARE_FLOOR
    ]
    rows.sort(key=lambda r: abs(r["amount"]), reverse=True)
    return rows
