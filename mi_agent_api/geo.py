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
from typing import Any, Dict, List, Optional

import pandas as pd

from analytics_lib.numeric import coerce_numeric

_BALANCE = "current_outstanding_balance"
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
    codes: List[str] = []
    names: List[str] = []
    resolved_from_itl3 = 0
    resolved_from_postcode = 0
    for i in range(n):
        code = ""
        name = ""
        if itl3_field:
            v = str(df[itl3_field].iloc[i]).strip().upper()
            if v and v not in ("", "NAN", "NONE", "NULL"):
                code = v
                resolved_from_itl3 += 1
        if not code and pc_field:
            hit = _prefix_to_itl3(_outward_code(df[pc_field].iloc[i]), by_prefix)
            if hit:
                code = hit["itl3_code"]
                name = hit["itl3_name"]
                resolved_from_postcode += 1
        if code and not name:
            name = name_by_code.get(code, code)
        codes.append(code)
        names.append(name)

    work = pd.DataFrame({"itl3_code": codes, "itl3_name": names,
                         "balance": bal.fillna(0.0).to_numpy()})
    work = work[work["itl3_code"] != ""]
    resolved_rows = int(len(work))
    if resolved_rows == 0:
        reason = ("no ITL3 field and no property postcode on the tape"
                  if not (itl3_field or pc_field)
                  else "no loan resolved to an ITL3 area (postcodes unmatched)")
        return {"available": False, "reason": reason, "areas": [],
                "total": 0.0, "coveragePct": 0.0}

    grouped = work.groupby("itl3_code", as_index=False).agg(
        balance=("balance", "sum"), count=("itl3_code", "size"),
        itl3_name=("itl3_name", "first"))
    total = float(grouped["balance"].sum())
    areas = [
        {"itl3_code": r["itl3_code"],
         "itl3_name": r["itl3_name"] or name_by_code.get(r["itl3_code"], r["itl3_code"]),
         "balance": round(float(r["balance"]), 2),
         "count": int(r["count"]),
         "sharePct": round(float(r["balance"]) / total * 100.0, 2) if total else None}
        for _, r in grouped.iterrows()]
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
