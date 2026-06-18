#!/usr/bin/env python3
"""
build_annex2_path_acceptance_decisions.py
=========================================

Formal acceptance-gate decisions for the Annex 2 field-to-XSD path map.

For every one of the 107 fields, applies the acceptance criteria (re-validating
each path against the vendored XSD) and records the per-criterion evidence plus
the decision:

    sample_confirmed | accepted_for_builder | needs_manual_review | rejected

The decision logic is the single source in
``scripts/build_annex2_field_xsd_path_map.evaluate_acceptance`` (also used to
stamp ``builder_acceptance_status`` into the path-map YAML), so the YAML and this
CSV cannot drift.

Output: output/config_review/annex2_path_acceptance_decisions.csv

This is a PATH-acceptance gate only. It does NOT make any field production-ready,
generate production XML, change a production gate, or import legacy Gate 5
runtime behaviour / silent defaults / fabricated values.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.build_annex2_field_xsd_path_map import (  # noqa: E402
    evaluate_acceptance, xsd_path_validator,
)

_PATHMAP = _REPO / "config" / "delivery" / "annex2_field_xsd_path_map.yaml"
_OUT = _REPO / "output" / "config_review" / "annex2_path_acceptance_decisions.csv"

_COLS = ["esma_code", "canonical_field", "record_group", "promotion_status",
         "builder_acceptance_status", "xml_path", "from_workbook_path", "xsd_validated",
         "code_label_type_consistent", "not_polluted_multicode",
         "respects_rrel_rrec_hierarchy", "rrec_nested_under_coll", "nodataoptn_handling",
         "sample_confirmed", "decision_reason", "risk_level", "production_ready", "notes"]


def main():
    fields = yaml.safe_load(_PATHMAP.read_text())["field_xsd_path_map"]["fields"]
    valid = xsd_path_validator()

    rows = []
    for f in fields:
        d = evaluate_acceptance(f, valid)
        rows.append({
            "esma_code": f["esma_code"],
            "canonical_field": f.get("canonical_field", ""),
            "record_group": f.get("record_group", ""),
            "promotion_status": f.get("promotion_status", ""),
            "builder_acceptance_status": d["builder_acceptance_status"],
            "xml_path": f.get("xml_path") or "",
            "from_workbook_path": d["from_workbook_path"],
            "xsd_validated": d["xsd_validated"],
            "code_label_type_consistent": d["code_label_type_consistent"],
            "not_polluted_multicode": d["not_polluted_multicode"],
            "respects_rrel_rrec_hierarchy": d["respects_rrel_rrec_hierarchy"],
            "rrec_nested_under_coll": d["rrec_nested_under_coll"],
            "nodataoptn_handling": d["nodataoptn_handling"],
            "sample_confirmed": d["sample_confirmed"],
            "decision_reason": d["decision_reason"],
            "risk_level": d["risk_level"],
            "production_ready": f.get("production_ready", False),
            "notes": ("PATH-acceptance only. Production XML still requires DATA readiness "
                      f"(data_readiness={f.get('data_readiness')}) and final-schema validation; "
                      "production_ready=False for all fields."),
        })

    # sanity: YAML status must equal the freshly-evaluated status (no drift).
    by_code = {f["esma_code"]: f for f in fields}
    for r in rows:
        assert r["builder_acceptance_status"] == by_code[r["esma_code"]]["builder_acceptance_status"], \
            f"acceptance drift for {r['esma_code']}"

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLS)
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    c = Counter(r["builder_acceptance_status"] for r in rows)
    print(f"Wrote {_OUT} ({len(rows)} fields)")
    for k, n in c.most_common():
        print(f"  {k}: {n}")
    print(f"  production_ready: {sum(1 for r in rows if r['production_ready'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
