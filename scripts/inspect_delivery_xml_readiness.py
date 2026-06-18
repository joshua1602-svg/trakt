#!/usr/bin/env python3
"""
inspect_delivery_xml_readiness.py
=================================

Small, read-only diagnostic for a Delivery/XML Agent output directory.

Given an ``output/delivery_xml`` directory (artefacts 60..64), it prints:

  * delivery readiness flags;
  * delivery issue mix (by blocker type) + remediation grouping;
  * blocked field list (ESMA codes);
  * top affected ESMA codes;
  * whether any XML file was generated.

Usage::

    python scripts/inspect_delivery_xml_readiness.py \\
      onboarding_output/client_001/run_pre_xml_final_check_3/output/delivery_xml

It never writes, never mutates, and never generates XML.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.delivery_xml_agent.remediation import group_delivery_issues  # noqa: E402


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def inspect(delivery_dir: str | Path) -> Dict[str, Any]:
    """Return a structured summary of a delivery_xml directory (no printing)."""
    d = Path(delivery_dir)
    manifest = _read_json(d / "60_delivery_manifest.json")
    readiness = _read_json(d / "61_delivery_readiness.json")
    frame = _read_csv(d / "62_delivery_normalised_frame.csv")
    issues = _read_csv(d / "63_delivery_issues.csv")

    blocked_rows = [r for r in frame if r.get("delivery_status") == "blocked"]
    blocked_codes = sorted({r.get("esma_code", "") for r in blocked_rows if r.get("esma_code")})
    status_mix = Counter(r.get("delivery_status", "") for r in frame)
    blocker_mix = Counter(i.get("delivery_blocker_type", "") for i in issues)
    affected = Counter(r.get("esma_code", "") for r in blocked_rows if r.get("esma_code"))
    xml_files = [p.name for p in d.glob("*.xml")]

    flags = {
        k: manifest.get(k, readiness.get(k))
        for k in ("delivery_xml_ran", "delivery_normalisation_complete",
                  "xml_generation_allowed", "xml_generated", "ready_for_xml_delivery",
                  "next_agent")
    }
    return {
        "delivery_dir": str(d),
        "exists": (d / "60_delivery_manifest.json").exists(),
        "flags": flags,
        "status_mix": dict(status_mix),
        "blocker_mix": dict(blocker_mix),
        "blocked_codes": blocked_codes,
        "top_affected_codes": affected.most_common(10),
        "xml_files": xml_files,
        "groups": group_delivery_issues(issues),
    }


def _print_report(summary: Dict[str, Any]) -> None:
    if not summary["exists"]:
        print(f"No delivery package found at: {summary['delivery_dir']}")
        print("(expected 60_delivery_manifest.json — run the Delivery/XML Agent first)")
        return

    print(f"Delivery/XML readiness — {summary['delivery_dir']}")
    print("\n## Readiness flags")
    for k, v in summary["flags"].items():
        print(f"  {k:32s} = {v}")

    print("\n## Delivery status mix")
    for k, v in sorted(summary["status_mix"].items()):
        print(f"  {k:24s} {v}")

    print("\n## Delivery issue mix (by blocker type)")
    if summary["blocker_mix"]:
        for k, v in sorted(summary["blocker_mix"].items()):
            print(f"  {k:32s} {v}")
    else:
        print("  (none)")

    print("\n## Remediation groups")
    for key, g in summary["groups"].items():
        if g["issue_count"] == 0 and not g["codes"]:
            continue
        codes = ", ".join(g["codes"]) if g["codes"] else "-"
        print(f"  [{key}] {g['title']}: {g['issue_count']} issue(s); codes: {codes}")
        print(f"      owner={g['owner']} preview={g['needed_before_preview']} "
              f"production={g['needed_before_production']}")

    print(f"\n## Blocked ESMA codes ({len(summary['blocked_codes'])})")
    print("  " + (", ".join(summary["blocked_codes"]) if summary["blocked_codes"] else "(none)"))

    print("\n## Top affected ESMA codes")
    for code, n in summary["top_affected_codes"]:
        print(f"  {code:10s} {n}")

    print("\n## XML generated?")
    print(f"  xml files present: {summary['xml_files'] or 'NONE'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inspect a Delivery/XML Agent output directory.")
    ap.add_argument("delivery_dir",
                    help="Path to output/delivery_xml (artefacts 60..64).")
    args = ap.parse_args(argv)
    _print_report(inspect(args.delivery_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
