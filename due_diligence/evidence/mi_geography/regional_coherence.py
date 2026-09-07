#!/usr/bin/env python3
"""Regional coherence across every surface that answers a region question.

WHAT THIS PROVES. On the same book, at the same period, four independent paths
produce the SAME regional breakdown, cell for cell:

    serving        mi_agent_api.funded_prep.augment_platform_canonical_dimensions
    historical     mi_agent_api.funded_prep.prepare_funded_mi_dataset
    dashboard      mi_agent_api.evolution._region_breakdown_column + _breakdown
    oracle         a plain pandas groupby over the raw tape, written here and
                   depending on nothing under test

The oracle is the point of the exercise. The first three share code; if they
agreed with each other and were all wrong together, nothing above would notice.
The fourth reads the tape directly and groups it, so a disagreement between it
and the estate is a defect in the estate.

WHAT IT MEASURED BEFORE THE GEOGRAPHY CONTRACT EXISTED, on the demo platform
book (11,035 loans, 2026-06-30):

    serving      grouped `canonical_region_reporting` — PRESENT on every row,
                 POPULATED on none; `region_mapping_method: unresolved` 11,035
                 times, because the harmonisation read the obligor column first
                 and that column holds ITL3 codes no ITL1 taxonomy resolves
    historical   grouped `collateral_geography` — 11 readable regions
    dashboard    grouped `geographic_region_obligor` — 172 ITL3 codes
    oracle       11 readable regions

Three surfaces, three different answers to one question, and the one a reader
actually got was the empty one.

Run:  python -m due_diligence.evidence.mi_geography.regional_coherence
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import mi_geography as geo

DEFAULT_TAPE = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
                / "platform" / "alderbridge" / "latest" / "platform_canonical_typed.csv")
BALANCE = "current_outstanding_balance"


# --------------------------------------------------------------------------- #
# The oracle. Deliberately naive, deliberately independent.
# --------------------------------------------------------------------------- #
def oracle_breakdown(raw: pd.DataFrame, column: str) -> Dict[str, Dict[str, Any]]:
    """Balance and loan count per region, straight from the tape.

    No preparation, no registry, no parser. Blank values are excluded rather
    than bucketed, which is what every surface under test also claims to do.
    """
    frame = raw[[column, BALANCE]].copy()
    frame[column] = frame[column].astype(str).str.strip()
    frame = frame[~frame[column].str.lower().isin(
        ["", "nan", "none", "nat", "<na>", "null"])]
    frame[BALANCE] = pd.to_numeric(frame[BALANCE], errors="coerce")
    grouped = frame.groupby(column, dropna=True)[BALANCE].agg(["sum", "count"])
    return {str(k): {"balance": round(float(v["sum"]), 2), "loans": int(v["count"])}
            for k, v in grouped.iterrows()}


def _from_prepared(prepared: pd.DataFrame, column: str) -> Dict[str, Dict[str, Any]]:
    return oracle_breakdown(prepared, column)


def _dashboard_breakdown(prepared: pd.DataFrame, contract) -> Dict[str, Any]:
    """The dashboard's own series, in its own shape.

    ``evolution._breakdown`` emits ``[{key, value}]`` — a balance per region, no
    loan count — and it routes rows with no region into an explicit
    ``Unknown / Missing`` bucket so the breakdown reconciles to the book total.
    The oracle EXCLUDES those rows instead. Both are defensible and they are not
    the same convention, so the comparison below is on the regions themselves and
    the bucket is reconciled separately rather than counted as a disagreement.
    """
    from mi_agent_api import evolution as ev

    column = ev._region_breakdown_column(prepared, contract)
    rows = ev._breakdown(prepared, column) if column else []
    cells = {str(r["key"]): round(float(r["value"]), 2) for r in rows}
    missing = cells.pop(ev.MISSING_BUCKET, None)
    return {"column": column, "cells": cells, "unknownBucket": missing}


def _compare_balances(left: Dict[str, float], right: Dict[str, Dict[str, Any]],
                      *, tolerance: float = 0.01) -> List[str]:
    """Balance-only comparison, for a surface that publishes no loan count."""
    problems: List[str] = []
    for key in sorted(set(left) | set(right)):
        a = left.get(key)
        b = (right.get(key) or {}).get("balance")
        if a is None or b is None:
            problems.append(f"{key}: present in only one")
            continue
        if abs(a - b) > tolerance:
            problems.append(f"{key}: balance {a} vs {b}")
    return problems


def _compare(left: Dict[str, Dict[str, Any]], right: Dict[str, Dict[str, Any]],
             *, tolerance: float = 0.01) -> List[str]:
    """Every cell that differs, named. Empty when the two agree."""
    problems: List[str] = []
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key), right.get(key)
        if a is None or b is None:
            problems.append(f"{key}: present in only one ({'left' if a else 'right'})")
            continue
        if a["loans"] != b["loans"]:
            problems.append(f"{key}: loans {a['loans']} vs {b['loans']}")
        if abs((a["balance"] or 0) - (b["balance"] or 0)) > tolerance:
            problems.append(f"{key}: balance {a['balance']} vs {b['balance']}")
    return problems


def run(tape: Path = DEFAULT_TAPE, *, asset_class: str = "equity_release"
        ) -> Dict[str, Any]:
    from mi_agent_api.funded_prep import (
        augment_platform_canonical_dimensions, prepare_funded_mi_dataset)

    raw = pd.read_csv(tape, low_memory=False)
    contract = geo.resolve_contract(asset_class=asset_class, frame=raw)
    basis_column = geo.field_for_basis(contract.primary_basis, frame=raw)

    serving, _ = augment_platform_canonical_dimensions(raw.copy(), geography=contract)
    historical, _ = prepare_funded_mi_dataset(raw.copy())

    oracle = oracle_breakdown(raw, basis_column)
    serving_cells = _from_prepared(serving, basis_column)
    historical_cells = _from_prepared(historical, basis_column)
    dashboard = _dashboard_breakdown(serving, contract)

    report: Dict[str, Any] = {
        "tape": str(tape),
        "rows": int(len(raw)),
        "contract": contract.to_dict(),
        "regionColumn": basis_column,
        "dashboardColumn": dashboard["column"],
        "cells": {"oracle": len(oracle), "serving": len(serving_cells),
                  "historical": len(historical_cells),
                  "dashboard": len(dashboard["cells"])},
        "dashboardUnknownBucket": dashboard["unknownBucket"],
        "disagreements": {
            "serving_vs_oracle": _compare(serving_cells, oracle),
            "historical_vs_oracle": _compare(historical_cells, oracle),
            "dashboard_vs_oracle": _compare_balances(dashboard["cells"], oracle),
            "serving_vs_historical": _compare(serving_cells, historical_cells),
        },
        "oracleCells": oracle,
    }
    # The harmonised column, measured rather than assumed: it is no longer what
    # a region question binds to, but it must still resolve.
    if "canonical_region_reporting" in serving.columns:
        column = serving["canonical_region_reporting"]
        populated = column.notna() & (column.astype(str).str.strip() != "")
        report["harmonised"] = {
            "populated": int(populated.sum()),
            "distinct": int(column[populated].nunique()),
            "methods": (serving["region_mapping_method"].value_counts().to_dict()
                        if "region_mapping_method" in serving.columns else {}),
        }
    # The dashboard's Unknown / Missing bucket must be exactly the balance the
    # oracle excluded: the two conventions differ, and this is what makes that a
    # stated difference rather than a discrepancy.
    total = round(float(pd.to_numeric(raw[BALANCE], errors="coerce").sum()), 2)
    covered = round(sum(c["balance"] for c in oracle.values()), 2)
    excluded = round(total - covered, 2)
    report["reconciliation"] = {
        "bookBalance": total,
        "coveredByRegions": covered,
        "excludedByOracle": excluded,
        "dashboardUnknownBucket": dashboard["unknownBucket"],
        "agrees": abs((dashboard["unknownBucket"] or 0.0) - excluded) <= 0.01,
    }
    report["coherent"] = (not any(report["disagreements"].values())
                          and report["reconciliation"]["agrees"])
    return report


def main(argv: Optional[List[str]] = None) -> int:
    tape = Path(argv[0]) if argv else DEFAULT_TAPE
    if not tape.is_file():
        print(json.dumps({"skipped": f"tape not present: {tape}"}, indent=2))
        return 0
    report = run(tape)
    printable = {k: v for k, v in report.items() if k != "oracleCells"}
    print(json.dumps(printable, indent=2))
    print("\nOracle cells:")
    for name, cell in sorted(report["oracleCells"].items()):
        print(f"  {name:<28} {cell['loans']:>7,} loans  £{cell['balance']:>18,.2f}")
    print(f"\nResult: {'COHERENT' if report['coherent'] else 'DIVERGENT'}")
    return 0 if report["coherent"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
