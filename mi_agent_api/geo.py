"""Geographic exposure aggregation by UK ITL3 area.

Answers "where is the book concentrated?" at ITL3 granularity (e.g. Bristol),
from the loan tape only:

  * the ITL3 area per loan comes from the tape's ``geographic_region_*_itl3``
    field when present, otherwise it is derived from the property postcode via
    the in-repo ``uk_itl_master_lookup_v2.csv`` (postcode district -> ITL3);
  * exposure (balance) and loan count are summed per ITL3 area, with each
    area's share of the total.

This is the DATA layer for the UK exposure view. It needs no map geometry — a
ranked "heat" (treemap/bar) renders from this directly; a shaded choropleth
additionally needs ITL3 boundary polygons, which are not derivable from a tape.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from analytics_lib.numeric import coerce_numeric

_BALANCE = "current_outstanding_balance"
_LTV = "current_loan_to_value"
_AGE = "youngest_borrower_age"
# Prefer collateral (property) location; fall back to obligor.
_ITL3_FIELDS = ("geographic_region_collateral_itl3", "geographic_region_obligor_itl3")
_POSTCODE_FIELDS = ("property_post_code", "postcode", "post_code")
_LOOKUP_NAME = "uk_itl_master_lookup_v2.csv"


def _lookup_path() -> Optional[Path]:
    """Locate the postcode->ITL3 master lookup. Overridable via MI_AGENT_ITL_LOOKUP."""
    override = os.environ.get("MI_AGENT_ITL_LOOKUP")
    if override and Path(override).is_file():
        return Path(override)
    here = Path(__file__).resolve()
    for base in (here.parent, *here.parents):
        cand = base / _LOOKUP_NAME
        if cand.is_file():
            return cand
        cand = base / "config" / "system" / _LOOKUP_NAME
        if cand.is_file():
            return cand
    return None


@lru_cache(maxsize=1)
def _load_lookup() -> Dict[str, Dict[str, Any]]:
    """``{postcode_prefix -> {itl3_code, itl3_name}}`` and ``{itl3_code -> name}``,
    both keyed for O(1) resolution. Cached; empty when the file is absent."""
    path = _lookup_path()
    if not path:
        return {"by_prefix": {}, "name_by_code": {}}
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception:  # noqa: BLE001 - a bad lookup must not break a query
        return {"by_prefix": {}, "name_by_code": {}}
    by_prefix: Dict[str, Dict[str, str]] = {}
    name_by_code: Dict[str, str] = {}
    for _, r in df.iterrows():
        pfx = str(r.get("postcode_prefix", "")).strip().upper()
        code = str(r.get("itl3_code", "")).strip().upper()
        name = str(r.get("itl3_name", "")).strip()
        if pfx and code:
            by_prefix[pfx] = {"itl3_code": code, "itl3_name": name}
        if code and name:
            name_by_code.setdefault(code, name)
    return {"by_prefix": by_prefix, "name_by_code": name_by_code}


def _outward_code(postcode: str) -> str:
    """The outward (district) code of a UK postcode: the part before the space,
    or all-but-the-last-3 chars when unspaced. ``'BS1 4DJ' -> 'BS1'``."""
    pc = str(postcode or "").strip().upper()
    if not pc:
        return ""
    if " " in pc:
        return pc.split()[0]
    return pc[:-3] if len(pc) > 3 else pc


def _prefix_to_itl3(outward: str, by_prefix: Dict[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Resolve a district code to ITL3, trimming trailing digits on a miss
    (``'BS16' -> 'BS1'``) so a finer district still maps to its area."""
    o = outward
    while o:
        hit = by_prefix.get(o)
        if hit:
            return hit
        if o[-1].isdigit():
            o = o[:-1]
        else:
            break
    return None


def exposure_by_itl3(df: pd.DataFrame) -> Dict[str, Any]:
    """Exposure and loan count per ITL3 area, tape-driven. Never raises."""
    lookup = _load_lookup()
    by_prefix = lookup["by_prefix"]
    name_by_code = lookup["name_by_code"]
    columns = list(getattr(df, "columns", []))

    itl3_field = next((f for f in _ITL3_FIELDS if f in columns), None)
    pc_field = next((f for f in _POSTCODE_FIELDS if f in columns), None)
    bal = coerce_numeric(df[_BALANCE]) if _BALANCE in columns else None
    if bal is None:
        return {"available": False, "reason": "no balance column", "areas": [],
                "total": 0.0, "coveragePct": 0.0}

    n = len(df)
    # Vectorised two-source resolution (replaces a per-row Python loop):
    #   1. the tape's ITL3 field, normalised (uppercase, sentinel nulls -> "");
    #   2. for rows still unresolved, the property postcode -> ITL3, resolving each
    #      UNIQUE outward code once (there are few distinct districts) rather than
    #      per row.
    empty = pd.Series([""] * n, index=df.index, dtype="object")
    if itl3_field:
        raw = df[itl3_field].astype("string").str.strip().str.upper()
        code_series = raw.where(~raw.isin(["", "NAN", "NONE", "NULL"]), "").fillna("")
        code_series = code_series.astype("object")
    else:
        code_series = empty.copy()
    resolved_from_itl3 = int((code_series != "").sum())
    name_series = empty.copy()

    resolved_from_postcode = 0
    if pc_field is not None:
        need = code_series == ""
        if bool(need.any()):
            outward = df.loc[need, pc_field].map(_outward_code)
            resolved = {o: _prefix_to_itl3(o, by_prefix) for o in outward.unique()}
            pc_code = outward.map(lambda o: (resolved.get(o) or {}).get("itl3_code", ""))
            pc_name = outward.map(lambda o: (resolved.get(o) or {}).get("itl3_name", ""))
            code_series.loc[need] = pc_code
            name_series.loc[need] = pc_name
            resolved_from_postcode = int((pc_code != "").sum())

    # Per-area analytics (surfaced on hover): average ticket, weighted-average
    # LTV and average borrower age. Each is optional — computed only where its
    # column is on the tape.
    ltv = coerce_numeric(df[_LTV]) if _LTV in columns else None
    age = coerce_numeric(df[_AGE]) if _AGE in columns else None
    work_cols: Dict[str, Any] = {
        "itl3_code": code_series.to_numpy(),
        "itl3_name": name_series.to_numpy(),
        "balance": bal.fillna(0.0).to_numpy(),
    }
    if ltv is not None:
        # Balance-weighted LTV: Σ(ltv×balance) ÷ Σ(balance over rows with an LTV).
        work_cols["ltv_x_bal"] = (ltv * bal).to_numpy()
        work_cols["bal_for_ltv"] = bal.where(ltv.notna(), other=0.0).fillna(0.0).to_numpy()
    if age is not None:
        work_cols["age"] = age.to_numpy()
    work = pd.DataFrame(work_cols)
    work = work[work["itl3_code"] != ""]
    resolved_rows = int(len(work))
    if resolved_rows == 0:
        reason = ("no ITL3 field and no property postcode on the tape"
                  if not (itl3_field or pc_field)
                  else "no loan resolved to an ITL3 area (postcodes unmatched)")
        return {"available": False, "reason": reason, "areas": [],
                "total": 0.0, "coveragePct": 0.0}

    agg_spec: Dict[str, Any] = {
        "balance": ("balance", "sum"), "count": ("itl3_code", "size"),
        "itl3_name": ("itl3_name", "first")}
    if ltv is not None:
        agg_spec["ltv_x_bal"] = ("ltv_x_bal", "sum")
        agg_spec["bal_for_ltv"] = ("bal_for_ltv", "sum")
    if age is not None:
        agg_spec["age"] = ("age", "mean")
    grouped = work.groupby("itl3_code", as_index=False).agg(**agg_spec)
    total = float(grouped["balance"].sum())

    def _area(r) -> Dict[str, Any]:
        bal_sum = float(r["balance"])
        cnt = int(r["count"])
        a: Dict[str, Any] = {
            "itl3_code": r["itl3_code"],
            "itl3_name": r["itl3_name"] or name_by_code.get(r["itl3_code"], r["itl3_code"]),
            "balance": round(bal_sum, 2),
            "count": cnt,
            "sharePct": round(bal_sum / total * 100.0, 2) if total else None,
            # Average ticket = mean loan size in the area.
            "avgTicket": round(bal_sum / cnt, 2) if cnt else None,
        }
        if ltv is not None:
            bfl = float(r["bal_for_ltv"])
            a["avgLtv"] = round(float(r["ltv_x_bal"]) / bfl, 2) if bfl else None
        if age is not None:
            av = r["age"]
            a["avgAge"] = round(float(av), 1) if pd.notna(av) else None
        return a

    areas = [_area(r) for _, r in grouped.iterrows()]
    areas.sort(key=lambda a: a["balance"], reverse=True)

    return {
        "available": True,
        "areas": areas,
        "areaCount": len(areas),
        "total": round(total, 2),
        "coveragePct": round(resolved_rows / n * 100.0, 2) if n else 0.0,
        "resolvedFromItl3Field": resolved_from_itl3,
        "resolvedFromPostcode": resolved_from_postcode,
        "lookupAvailable": bool(by_prefix),
        "basis": ("collateral" if itl3_field == _ITL3_FIELDS[0]
                  else "obligor" if itl3_field == _ITL3_FIELDS[1]
                  else "postcode_derived"),
    }
