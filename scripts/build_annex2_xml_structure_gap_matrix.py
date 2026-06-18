#!/usr/bin/env python3
"""
build_annex2_xml_structure_gap_matrix.py
========================================

Derive the **production** Annex 2 XML structure gap matrix from the authoritative
artefacts in the repo:

  * config/regime/annex2_field_universe.yaml   (107 Annex 2 codes, format, ND)
  * config/regime/annex2_delivery_rules.yaml   (workbook_semantic leaf tokens,
                                                mandatory, projected_source_field)

It writes ``output/config_review/annex2_xml_structure_gap_matrix.csv`` — one row
per ESMA code — classifying, for production XML, the proposed XML level /
cardinality, whether a real XML path is known, and the remaining structure /
data gap.

This script DOES NOT generate production XML, does not touch production gates,
and does not use preview placeholders. It is a read-only analysis over config +
the vendored ESMA XSD (DRAFT1auth.099.001.04_1.3.0.xsd) and its sample message.
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
_UNIVERSE = _REPO / "config" / "regime" / "annex2_field_universe.yaml"
_RULES = _REPO / "config" / "regime" / "annex2_delivery_rules.yaml"
_OUT = _REPO / "output" / "config_review" / "annex2_xml_structure_gap_matrix.csv"

# Report/header-level codes (one-per-report) and their confirmed XSD paths,
# read directly from the vendored XSD + sample message (Securitisation1).
_HEADER_PATHS = {
    "RREL1": "Document/ScrtstnNonAsstBckdComrclPprUndrlygXpsrRpt/NewCrrctn/ScrtstnRpt/ScrtstnIdr",
    "RREL6": "Document/ScrtstnNonAsstBckdComrclPprUndrlygXpsrRpt/NewCrrctn/ScrtstnRpt/CutOffDt",
}
# Exposure-record identification codes (one-per-loan), confirmed under UndrlygXpsrId.
_EXPOSURE_ID_BASE = ("Document/ScrtstnNonAsstBckdComrclPprUndrlygXpsrRpt/NewCrrctn/"
                     "ScrtstnRpt/UndrlygXpsrRcrd/UndrlygXpsrId")
_EXPOSURE_ID_PATHS = {
    "RREL2": f"{_EXPOSURE_ID_BASE}/OrgnlUndrlygXpsrIdr",
    "RREL3": f"{_EXPOSURE_ID_BASE}/NewUndrlygXpsrIdr",
    "RREL4": f"{_EXPOSURE_ID_BASE}/OrgnlOblgrIdr",
    "RREL5": f"{_EXPOSURE_ID_BASE}/NewOblgrIdr",
}
# Collateral identifiers (one-per-collateral) confirmed under Coll/CollIdr in the
# residential-performing sample. Only the two CollIdr children present in the
# sample are asserted; the universe field names disambiguate them:
#   RREC3 "Original Collateral Identifier" -> OrgnlIdr
#   RREC4 "New Collateral Identifier"      -> NewIdr
# RREC1/RREC2 have NO confirmed CollIdr child and are left unmapped (not guessed).
_COLL_BASE = ("Document/ScrtstnNonAsstBckdComrclPprUndrlygXpsrRpt/NewCrrctn/ScrtstnRpt/"
              "UndrlygXpsrRcrd/UndrlygXpsrData/ResdtlRealEsttLn/PrfrmgLn/Coll")
_COLL_ID_PATHS = {
    "RREC3": f"{_COLL_BASE}/CollIdr/OrgnlIdr",
    "RREC4": f"{_COLL_BASE}/CollIdr/NewIdr",
}

# Resolved-for-data fields (delivery-valid; structure still needs the XSD path).
_RESOLVED_FOR_DATA = {"RREL35"}

# Gap taxonomy.
GAP_MISSING_FIELD_PATH = "missing_field_xml_path"
GAP_MISSING_CARDINALITY = "missing_cardinality_rule"
GAP_DATA_DEPENDENCY = "data_dependency"
GAP_RESOLVED_DATA_NOT_STRUCTURE = "resolved_for_data_not_structure"
GAP_EXPOSURE_MAP = "missing_exposure_record_mapping"
GAP_COLLATERAL_MAP = "missing_collateral_record_mapping"


def _load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _classify(code: str, rule: dict, entry: dict):
    rg = "RREC" if code.upper().startswith("RREC") else "RREL"
    semantic = str(rule.get("workbook_semantic") or "").strip()
    mandatory = bool(rule.get("mandatory", False))

    # XML level + cardinality.
    if code in ("RREL1", "RREL6"):
        level, cardinality = "report_header", "one_per_report"
    elif code in _EXPOSURE_ID_PATHS:
        level, cardinality = "exposure_identification", "one_per_loan"
    elif rg == "RREC":
        level, cardinality = "collateral", "one_per_collateral"
    else:
        level, cardinality = "exposure", "one_per_loan"

    # Best-known full XML path (only where confirmed against XSD + sample).
    if code in _HEADER_PATHS:
        xml_path = _HEADER_PATHS[code]
    elif code in _EXPOSURE_ID_PATHS:
        xml_path = _EXPOSURE_ID_PATHS[code]
    elif code in _COLL_ID_PATHS:
        xml_path = _COLL_ID_PATHS[code]
    else:
        xml_path = ""  # leaf token alone is not a full path → not "known".

    path_known = bool(xml_path)
    leaf_hint = semantic  # may be "" when the rule has a TBC/mismapped path.

    # Gap class.
    if path_known and code in _RESOLVED_FOR_DATA:
        gap = GAP_RESOLVED_DATA_NOT_STRUCTURE
    elif path_known:
        gap = GAP_DATA_DEPENDENCY  # structure confirmed; only data work remains.
    elif code in _RESOLVED_FOR_DATA:
        gap = GAP_RESOLVED_DATA_NOT_STRUCTURE
    elif rg == "RREC":
        gap = GAP_COLLATERAL_MAP if not leaf_hint else GAP_MISSING_FIELD_PATH
    else:
        gap = GAP_EXPOSURE_MAP if not leaf_hint else GAP_MISSING_FIELD_PATH

    # Owner + risk + action.
    if path_known:
        owner = "data_pipeline"
        action = "confirm value provenance; XSD path already resolved"
        risk = "low"
    elif leaf_hint:
        owner = "delivery_xml_schema_mapping"
        action = f"resolve full XSD path for leaf '{leaf_hint}' (intermediate wrappers + ND option)"
        risk = "high" if mandatory else "medium"
    else:
        owner = "delivery_xml_schema_mapping"
        action = "no XSD path mapped (workbook_semantic TBC/mismapped) — map from XSD sequence"
        risk = "high" if mandatory else "medium"

    blocks_production = gap != GAP_RESOLVED_DATA_NOT_STRUCTURE or not path_known
    notes = []
    if leaf_hint and not path_known:
        notes.append(f"leaf token hint: {leaf_hint}")
    if not leaf_hint and not path_known:
        notes.append("workbook_semantic absent (TBC/mismapped in delivery rules)")
    if code in _RESOLVED_FOR_DATA:
        notes.append("delivery-valid for data; XSD path still to be confirmed")
    nd = []
    if entry.get("nd1_4_allowed"):
        nd.append("ND1-4")
    if entry.get("nd5_allowed"):
        nd.append("ND5")
    if nd:
        notes.append("ND allowed: " + ",".join(nd) + " (needs NoDataOptn wrapper)")

    return {
        "esma_code": code,
        "canonical_field": rule.get("projected_source_field") or entry.get("field_name", ""),
        "record_group": rg,
        "current_delivery_status": "mandatory" if mandatory else "optional",
        "proposed_xml_level": level,
        "proposed_cardinality": cardinality,
        "xml_path_known": str(path_known).lower(),
        "xml_path": xml_path or (leaf_hint and f"(leaf-only) {leaf_hint}") or "",
        "xsd_required": "true",
        "blocks_preview_xml": "false",
        "blocks_production_xml": str(bool(blocks_production)).lower(),
        "gap_class": gap,
        "recommended_action": action,
        "owner": owner,
        "risk_level": risk,
        "notes": "; ".join(notes),
    }


def main() -> int:
    universe = (_load_yaml(_UNIVERSE).get("fields") or {})
    rules = (_load_yaml(_RULES).get("field_rules") or {})

    rows = []
    for code in sorted(universe, key=lambda c: (c[:4], int("".join(ch for ch in c if ch.isdigit()) or 0))):
        rows.append(_classify(code, rules.get(code, {}) or {}, universe.get(code, {}) or {}))

    cols = ["esma_code", "canonical_field", "record_group", "current_delivery_status",
            "proposed_xml_level", "proposed_cardinality", "xml_path_known", "xml_path",
            "xsd_required", "blocks_preview_xml", "blocks_production_xml", "gap_class",
            "recommended_action", "owner", "risk_level", "notes"]
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # Console summary.
    from collections import Counter
    gaps = Counter(r["gap_class"] for r in rows)
    known = sum(1 for r in rows if r["xml_path_known"] == "true")
    print(f"Wrote {_OUT} ({len(rows)} codes)")
    print(f"  XML paths confirmed against XSD: {known}/{len(rows)}")
    for g, n in gaps.most_common():
        print(f"  {g}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
