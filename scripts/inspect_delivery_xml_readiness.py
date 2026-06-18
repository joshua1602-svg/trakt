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

With ``--format-invalid`` it additionally drills into the ``delivery_invalid``
rows and prints, per ESMA code, the failing value sample(s) and the regime
regex / enum the value violated.

Usage::

    python scripts/inspect_delivery_xml_readiness.py \\
      onboarding_output/client_001/run_pre_xml_final_check_3/output/delivery_xml

    python scripts/inspect_delivery_xml_readiness.py --format-invalid \\
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

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.delivery_xml_agent.remediation import group_delivery_issues  # noqa: E402

_DEFAULT_REGIME = _REPO_ROOT / "config" / "regime" / "annex2_delivery_rules.yaml"


def _falsey(v: Any) -> bool:
    return str(v).strip().lower() in ("false", "0", "no", "")


def _load_field_rules(regime_config_path: str | Path) -> Dict[str, Dict[str, Any]]:
    try:
        cfg = yaml.safe_load(Path(regime_config_path).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    rules = cfg.get("field_rules")
    return {str(k): (v or {}) for k, v in rules.items()} if isinstance(rules, dict) else {}


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


def format_invalid_detail(
    delivery_dir: str | Path,
    regime_config_path: str | Path | None = None,
    max_samples: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """Per-code detail of the ``delivery_invalid`` rows in 62.

    Returns ``{esma_code: {canonical_field, count, format_fail, enum_fail,
    samples, regex, enum_map_keys}}``. The regex / enum come from the regime
    ``field_rules`` (the same contract the agent validated against). Read-only.
    """
    d = Path(delivery_dir)
    frame = _read_csv(d / "62_delivery_normalised_frame.csv")
    # default to the regime config the agent actually used (from 60), else repo default.
    if regime_config_path is None:
        manifest = _read_json(d / "60_delivery_manifest.json")
        regime_config_path = manifest.get("regime_config_path") or _DEFAULT_REGIME
    rules = _load_field_rules(regime_config_path)

    out: Dict[str, Dict[str, Any]] = {}
    for r in frame:
        if r.get("delivery_status") != "delivery_invalid":
            continue
        code = r.get("esma_code", "")
        b = out.setdefault(code, {
            "esma_code": code, "canonical_field": r.get("canonical_field", ""),
            "count": 0, "format_fail": 0, "enum_fail": 0, "samples": [],
            "regex": "", "enum_map_keys": [],
        })
        b["count"] += 1
        if _falsey(r.get("format_valid", "")):
            b["format_fail"] += 1
        if _falsey(r.get("enum_valid", "")):
            b["enum_fail"] += 1
        val = r.get("projected_value", "")
        if val and val not in b["samples"] and len(b["samples"]) < max_samples:
            b["samples"].append(val)

    for code, b in out.items():
        rule = rules.get(code, {})
        validators = rule.get("validators") if isinstance(rule.get("validators"), dict) else {}
        transform = rule.get("transform") if isinstance(rule.get("transform"), dict) else {}
        b["regex"] = str(validators.get("regex", ""))
        enum_map = transform.get("enum_map") if isinstance(transform.get("enum_map"), dict) else {}
        b["enum_map_keys"] = sorted(str(k) for k in enum_map)[:10]
    return out


def _print_format_invalid(detail: Dict[str, Dict[str, Any]]) -> None:
    print("\n## Format-invalid drill-down (delivery_invalid rows)")
    if not detail:
        print("  (none — no delivery_invalid rows)")
        return
    for code in sorted(detail):
        b = detail[code]
        print(f"  [{code}] {b['canonical_field']}: {b['count']} invalid row(s) "
              f"(format_fail={b['format_fail']}, enum_fail={b['enum_fail']})")
        print(f"      sample value(s): {b['samples'] or '(blank)'}")
        if b["regex"]:
            print(f"      violated regex: {b['regex']}")
        if b["enum_map_keys"]:
            print(f"      allowed enum keys (first 10): {b['enum_map_keys']}")
        if not b["regex"] and not b["enum_map_keys"]:
            print("      (no regime regex/enum_map found for this code — check the rule)")


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
    ap.add_argument("--format-invalid", action="store_true",
                    help="Drill into delivery_invalid rows: per-code failing value "
                    "sample(s) and the violated regime regex/enum.")
    ap.add_argument("--regime-config", default="",
                    help="Regime rules YAML for the regex/enum lookup "
                    "(defaults to the path recorded in 60_delivery_manifest.json).")
    ap.add_argument("--samples", type=int, default=3,
                    help="Max distinct failing value samples per code (default 3).")
    args = ap.parse_args(argv)
    summary = inspect(args.delivery_dir)
    _print_report(summary)
    if args.format_invalid and summary["exists"]:
        _print_format_invalid(format_invalid_detail(
            args.delivery_dir,
            regime_config_path=args.regime_config or None,
            max_samples=args.samples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
