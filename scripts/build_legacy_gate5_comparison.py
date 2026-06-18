#!/usr/bin/env python3
"""
build_legacy_gate5_comparison.py
================================

Reconcile the legacy Gate 5 Annex 2 XML builder against the new XSD-based field
path map. Produces:

    output/config_review/legacy_gate5_vs_xsd_path_map.csv

For every Annex 2 code, it compares the path the legacy builder would use (the
ESMA mapping-workbook PATH column, which the legacy builder reads at runtime)
against the committed XSD path map, re-validating each legacy path against the
actual XSD tree.

This is REVIEW/RECONCILIATION ONLY. It generates no production XML, changes no
production gate, and does not wire the legacy builder into any path.
"""

from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.build_annex2_field_xsd_path_map import (  # noqa: E402
    _WORKBOOK, _WB_SHEET, _WB_VALUE_TAGS, xsd_path_validator,
)

_UNIVERSE = _REPO / "config" / "regime" / "annex2_field_universe.yaml"
_PATHMAP = _REPO / "config" / "delivery" / "annex2_field_xsd_path_map.yaml"
_OUT = _REPO / "output" / "config_review" / "legacy_gate5_vs_xsd_path_map.csv"

# Statuses.
MATCHES = "matches_xsd"
UNCONFIRMED = "legacy_path_unconfirmed"
CONFLICTS = "legacy_path_conflicts_with_xsd"
FLAT_BUT_NESTED = "legacy_flat_but_xsd_nested"
DEFAULT_RISK = "legacy_default_injection_risk"
ND_RISK = "legacy_nd_injection_risk"
NOT_USED = "not_used_by_legacy_builder"
COULD_UPGRADE = "could_upgrade_path_map_after_review"
DISCARD = "discard_legacy_assumption"

# Codes the legacy builder force-injects / coerces (unsafe silent fill).
_VALUE_COERCION_CODES = {"RREL12"}   # _coerce_record_value_for_branch -> "2026"


def _raw_legacy_element_path(specs, valid):
    """The element-level workbook path the legacy builder would target (may be a
    multi-code-cell pollution), plus its leaf tag and whether it validates."""
    clean = [s for s in specs if "Cxl" not in s.path.strip("/").split("/")]
    if not clean:
        # only a Cxl row -> legacy would treat it under cancellation (not reporting)
        return (specs[0].path.strip("/"), specs[0].tag, False) if specs else ("", "", False)
    elem_rows = [s for s in clean
                 if s.path.strip("/").split("/")[-1] == s.tag and s.tag not in _WB_VALUE_TAGS]
    elem_rows.sort(key=lambda s: len(s.path.strip("/").split("/")))
    chosen = elem_rows[0] if elem_rows else sorted(clean, key=lambda s: len(s.path))[0]
    p = chosen.path.strip("/")
    return p, chosen.tag, valid(p)


def _has_nd_injection(specs):
    return any("ScndryOblgrIncm" in s.path.split("/") for s in specs)


def main():
    valid = xsd_path_validator()
    universe = yaml.safe_load(_UNIVERSE.read_text())["fields"]
    pm = {f["esma_code"]: f
          for f in yaml.safe_load(_PATHMAP.read_text())["field_xsd_path_map"]["fields"]}

    # legacy workbook specs (residential / performing), as the legacy builder loads them.
    from engine.gate_5_delivery.xml_builder_annex2 import load_mapping_specs
    specs_by_code = {}
    if _WORKBOOK.exists():
        try:
            specs_by_code = load_mapping_specs(str(_WORKBOOK), _WB_SHEET, "PRF")
        except Exception:
            specs_by_code = {}

    rows = []
    for code in sorted(universe, key=lambda c: (c[:4], int("".join(ch for ch in c if ch.isdigit()) or 0))):
        rg = "RREC" if code.startswith("RREC") else "RREL"
        m = pm.get(code, {})
        new_path = m.get("xml_path") or ""
        canonical = m.get("canonical_field", "")
        specs = specs_by_code.get(code, [])

        if not specs:
            rows.append(dict(
                esma_code=code, canonical_field=canonical,
                legacy_gate5_xml_path_or_tag="", new_xsd_path_map_xml_path=new_path,
                legacy_assumption="code not present in the workbook RRE/PRF branch",
                xsd_evidence="n/a",
                status=NOT_USED, recommended_action="map manually from XSD/sample",
                risk_level="low",
                notes="legacy builder would not emit this field for residential performing"))
            continue

        leg_path, leg_tag, leg_valid = _raw_legacy_element_path(specs, valid)
        parts = leg_path.split("/")
        under_coll = "Coll" in parts
        nd_inject = _has_nd_injection(specs)

        legacy_assumption = (f"workbook PATH -> <{leg_tag}>; legacy reads full path incl. leaf; "
                             "wide one-row-per-loan; singleton Coll (no repeating collateral)")
        xsd_evidence = ("legacy path validates against XSD tree" if leg_valid
                        else "legacy path does NOT validate against XSD tree")

        # classify
        if code in _VALUE_COERCION_CODES:
            status = DEFAULT_RISK
            action = "RETIRE coercion: legacy fabricates a value (e.g. RREL12 -> '2026'); never fabricate"
            risk = "high"
            note = "legacy _coerce_record_value_for_branch injects a fabricated value"
        elif rg == "RREC" and leg_valid and not under_coll:
            status = CONFLICTS
            action = ("DISCARD legacy placement: collateral code mapped outside Coll via a "
                      "multi-code-cell; XSD requires it nested under .../PrfrmgLn/Coll")
            risk = "high"
            note = "multi-code-cell pollution (shared RTS-code cell) — XSD wins; keep unresolved"
        elif not leg_valid:
            status = UNCONFIRMED
            action = "do not adopt; legacy path not confirmable against the XSD"
            risk = "medium"
            note = "legacy/workbook path could not be validated against the XSD tree"
        elif new_path and leg_path == new_path:
            status = MATCHES
            action = ("retained: workbook path matches the XSD path map" if nd_inject is False
                      else "retain path; but RETIRE legacy ND5 auto-injection for this branch")
            risk = "low" if not nd_inject else "medium"
            note = ("workbook + XSD agree" if not nd_inject
                    else "path agrees, but legacy injects ND5 defaults under ScndryOblgrIncm")
            if nd_inject:
                status = ND_RISK
        elif new_path and leg_path != new_path:
            status = CONFLICTS
            action = "XSD path map wins; review the workbook discrepancy"
            risk = "high"
            note = f"legacy path differs from path map ({new_path.split('/')[-1]})"
        else:
            # legacy validates but path map has no path -> candidate upgrade.
            status = COULD_UPGRADE
            action = "upgrade path map to inferred_high_confidence after manual review"
            risk = "medium"
            note = "legacy/workbook offers an XSD-valid candidate the path map lacks"

        rows.append(dict(
            esma_code=code, canonical_field=canonical,
            legacy_gate5_xml_path_or_tag=leg_path or f"<{leg_tag}>",
            new_xsd_path_map_xml_path=new_path,
            legacy_assumption=legacy_assumption, xsd_evidence=xsd_evidence,
            status=status, recommended_action=action, risk_level=risk, notes=note))

    cols = ["esma_code", "canonical_field", "legacy_gate5_xml_path_or_tag",
            "new_xsd_path_map_xml_path", "legacy_assumption", "xsd_evidence",
            "status", "recommended_action", "risk_level", "notes"]
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    c = Counter(r["status"] for r in rows)
    print(f"Wrote {_OUT} ({len(rows)} codes)")
    for k, n in c.most_common():
        print(f"  {k}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
