#!/usr/bin/env python3
"""
build_annex2_path_map_promotion_checklist.py
============================================

Production-readiness PROMOTION checklist for the Annex 2 field-to-XSD path map.

Reads the committed path map (config/delivery/annex2_field_xsd_path_map.yaml) and
emits, for every one of the 107 fields, a reviewable promotion row that SEPARATES
two independent axes:

  * PATH readiness  — is the XML path production-eligible / acceptable?
  * DATA readiness  — is the real client/operator/config/source value available?

A field can have a production-eligible XML path and STILL be blocked by missing
data. Nothing here makes a field production-ready, generates production XML, or
changes a production gate.

Output: output/config_review/annex2_path_map_promotion_checklist.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_PATHMAP = _REPO / "config" / "delivery" / "annex2_field_xsd_path_map.yaml"
_OUT = _REPO / "output" / "config_review" / "annex2_path_map_promotion_checklist.csv"

# promotion-status -> (recommendation, risk, path-blocks-after-review)
_RULES = {
    "confirmed_by_xsd_sample": (
        "PROMOTE path: production-eligible XML path (XSD+sample). Production XML "
        "still requires DATA readiness (separate axis).", "low", False),
    "workbook_xsd_validated": (
        "ACCEPT for builder behind a structure gate: ESMA workbook path, XSD-validated. "
        "NOT production-eligible until formally accepted by the path-map review policy.",
        "medium", False),
    "manual_review_required": (
        "DO NOT PROMOTE: needs a manual ESMA-code -> XSD-element crosswalk.",
        "high", True),
    "unresolved": (
        "DO NOT PROMOTE: no XSD/workbook/sample evidence; production-blocking.",
        "high", True),
    "conflict": (
        "DO NOT PROMOTE: collision / multi-code-cell pollution (e.g. RREC mapped "
        "outside Coll). Resolve manually; XSD wins.", "high", True),
}

_COLS = ["esma_code", "canonical_field", "current_mapping_status", "proposed_mapping_status",
         "xml_path", "xsd_validated", "workbook_evidenced", "sample_evidenced",
         "manual_review_required", "blocks_production_xml_before_review",
         "blocks_production_xml_after_review", "promotion_recommendation",
         "risk_level", "notes"]


def main():
    pm = yaml.safe_load(_PATHMAP.read_text())["field_xsd_path_map"]["fields"]
    rows = []
    for f in pm:
        ps = f.get("promotion_status", "unresolved")
        es = f.get("evidence_source", "")
        has_path = bool(f.get("xml_path"))
        xsd_validated = has_path and ps in ("confirmed_by_xsd_sample", "workbook_xsd_validated")
        workbook_evidenced = "workbook" in es
        sample_evidenced = es == "sample_xml"
        rec, risk, path_blocks_after = _RULES.get(ps, _RULES["unresolved"])
        # PATH-dimension blocking. NOTE: production XML ALSO requires data readiness,
        # which is a SEPARATE axis (data_readiness) and currently pending for ALL fields.
        path_blocks_before = ps != "confirmed_by_xsd_sample"
        manual = ps in ("manual_review_required", "unresolved", "conflict")
        note = ("PATH axis only. Production XML also requires DATA readiness "
                f"(data_readiness={f.get('data_readiness')}); production_ready="
                f"{f.get('production_ready')} for all fields. ")
        if ps == "conflict":
            note += "Multi-code-cell pollution / collision — collateral must stay nested under Coll."
        elif ps == "workbook_xsd_validated":
            note += "Workbook path re-validated against XSD; awaiting formal acceptance."
        rows.append({
            "esma_code": f["esma_code"],
            "canonical_field": f.get("canonical_field", ""),
            "current_mapping_status": f.get("mapping_status", ""),
            "proposed_mapping_status": ps,
            "xml_path": f.get("xml_path") or "",
            "xsd_validated": xsd_validated,
            "workbook_evidenced": workbook_evidenced,
            "sample_evidenced": sample_evidenced,
            "manual_review_required": manual,
            "blocks_production_xml_before_review": path_blocks_before,
            "blocks_production_xml_after_review": path_blocks_after,
            "promotion_recommendation": rec,
            "risk_level": risk,
            "notes": note.strip(),
        })

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLS)
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    pc = Counter(r["proposed_mapping_status"] for r in rows)
    before = sum(1 for r in rows if r["blocks_production_xml_before_review"])
    after = sum(1 for r in rows if r["blocks_production_xml_after_review"])
    print(f"Wrote {_OUT} ({len(rows)} fields)")
    for ps, n in pc.most_common():
        print(f"  {ps}: {n}")
    print(f"  PATH blocks before review: {before}")
    print(f"  PATH blocks after review:  {after}")
    print("  (production XML still also requires DATA readiness — separate axis)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
