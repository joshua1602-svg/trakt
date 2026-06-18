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


def inspect_preview(delivery_dir: str | Path) -> Dict[str, Any]:
    """Read-only summary of the non-production preview artefacts under
    ``<delivery_dir>/preview/`` (70..77 readiness + any emitted 80../90.. XML).

    Reports the client-preview and synthetic-schema-test readiness, whether any
    preview XML exists, the production XML state, and the placeholder / exclusion
    / synthetic value counts. Never writes."""
    d = Path(delivery_dir)
    preview = d / "preview"
    client = _read_json(preview / "70_xml_preview_readiness.json")
    synth = _read_json(preview / "75_synthetic_schema_test_readiness.json")

    client_xml = preview / "client_preview" / "85_client_preview.xml"
    synth_xml = preview / "synthetic_schema_test" / "94_synthetic_schema_test.xml"
    synth_catalog = _read_csv(preview / "synthetic_schema_test" /
                              "92_synthetic_values_catalog.csv")
    synthetic_value_count = sum(
        1 for r in synth_catalog if r.get("source") == "synthetic_schema_test")

    # production XML state (from the production manifest — never modified here).
    manifest = _read_json(d / "60_delivery_manifest.json")
    prod_xml_files = [p.name for p in d.glob("*.xml")]

    return {
        "preview_dir": str(preview),
        "preview_exists": (preview / "70_xml_preview_readiness.json").exists(),
        "production_xml": {
            "xml_generation_allowed": manifest.get("xml_generation_allowed"),
            "ready_for_xml_delivery": manifest.get("ready_for_xml_delivery"),
            "xml_generated": manifest.get("xml_generated"),
            "production_xml_files": prod_xml_files,
        },
        "client_preview": {
            "enabled": client.get("enabled"),
            "xml_preview_allowed": client.get("xml_preview_allowed"),
            "ready_for_xml_preview": client.get("ready_for_xml_preview"),
            "placeholder_count": len(client.get("placeholder_codes", []) or []),
            "exclusion_count": len(client.get("excluded_codes", []) or []),
            "must_resolve_count": len(client.get("must_resolve_codes", []) or []),
            "xml_exists": client_xml.exists(),
        },
        "synthetic_schema_test": {
            "enabled": synth.get("enabled"),
            "synthetic_schema_test_allowed": synth.get("synthetic_schema_test_allowed"),
            "ready_for_synthetic_schema_test": synth.get("ready_for_synthetic_schema_test"),
            "field_universe_count": synth.get("field_universe_count"),
            "synthetic_value_count": synthetic_value_count or synth.get("synthetic_value_count"),
            "xml_exists": synth_xml.exists(),
        },
        "remaining_production_blockers": client.get("remaining_production_blockers", []),
    }


def _print_preview_report(p: Dict[str, Any]) -> None:
    print("\n=== Non-production preview / synthetic schema-test readiness ===")
    if not p["preview_exists"]:
        print(f"No preview package at: {p['preview_dir']}")
        print("(run engine.delivery_xml_agent.preview_readiness.evaluate_and_emit first)")
        return

    prod = p["production_xml"]
    print("\n## Production XML readiness (UNCHANGED, read-only)")
    print(f"  xml_generation_allowed = {prod['xml_generation_allowed']}")
    print(f"  ready_for_xml_delivery = {prod['ready_for_xml_delivery']}")
    print(f"  xml_generated          = {prod['xml_generated']}")
    print(f"  production XML present  = {prod['production_xml_files'] or 'NONE'}")

    c = p["client_preview"]
    print("\n## Client preview readiness")
    print(f"  enabled                = {c['enabled']}")
    print(f"  xml_preview_allowed    = {c['xml_preview_allowed']}")
    print(f"  ready_for_xml_preview  = {c['ready_for_xml_preview']}")
    print(f"  placeholder count      = {c['placeholder_count']}")
    print(f"  exclusion count        = {c['exclusion_count']}")
    print(f"  must-resolve count     = {c['must_resolve_count']}")
    print(f"  client preview XML     = {'present' if c['xml_exists'] else 'NONE'}")

    s = p["synthetic_schema_test"]
    print("\n## Synthetic schema-test readiness")
    print(f"  enabled                     = {s['enabled']}")
    print(f"  synthetic_schema_test_allowed = {s['synthetic_schema_test_allowed']}")
    print(f"  ready_for_synthetic_schema_test = {s['ready_for_synthetic_schema_test']}")
    print(f"  field universe count        = {s['field_universe_count']}")
    print(f"  synthetic value count       = {s['synthetic_value_count']}")
    print(f"  synthetic schema XML        = {'present' if s['xml_exists'] else 'NONE'}")

    print("\n## Remaining production blockers")
    print("  " + (", ".join(p["remaining_production_blockers"]) or "(none)"))


def inspect_xsd_structured_preview(delivery_dir: str | Path) -> Dict[str, Any]:
    """Read-only summary of the XSD-structured (third) preview mode under
    ``<delivery_dir>/preview/``. Never writes."""
    d = Path(delivery_dir)
    preview = d / "preview"
    readiness = _read_json(preview / "78_xsd_structured_preview_readiness.json")
    sp = preview / "xsd_structured_preview"
    xml_file = sp / "105_xsd_structured_preview.xml"
    frame = _read_csv(sp / "100_xsd_structured_preview_frame.csv")
    excl = _read_csv(sp / "103_xsd_structured_preview_exclusions.csv")
    validation = _read_json(sp / "107_xsd_structured_preview_xsd_validation.json")
    manifest = _read_json(d / "60_delivery_manifest.json")

    records = sorted({r.get("loan_identifier", "") for r in frame if r.get("loan_identifier")})
    placeholders = sum(1 for r in frame if r.get("value_source") == "preview_only_placeholder")
    return {
        "preview_dir": str(preview),
        "readiness_exists": (preview / "78_xsd_structured_preview_readiness.json").exists(),
        "enabled": readiness.get("enabled"),
        "xsd_structured_preview_allowed": readiness.get("xsd_structured_preview_allowed"),
        "xml_generated": xml_file.exists(),
        "xsd_validation_attempted": validation.get("xsd_validation_attempted"),
        "xsd_validation_passed": validation.get("xsd_validation_passed"),
        "records_emitted": len(records),
        "fields_emitted": len(frame),
        "placeholder_fields": placeholders,
        "excluded_fields": len(excl),
        "rrec_all_nested_under_coll": readiness.get("rrec_all_nested_under_coll"),
        "rejected_or_manual_skipped": readiness.get("rejected_or_manual_skipped", []),
        "production_gates_false": not any([
            manifest.get("xml_generation_allowed"), manifest.get("ready_for_xml_delivery"),
            manifest.get("xml_generated")]),
        "known_limitations": validation.get("known_limitations", []),
    }


def _print_xsd_structured_report(p: Dict[str, Any]) -> None:
    print("\n=== XSD-structured (nested) preview readiness ===")
    if not p["readiness_exists"]:
        print(f"No XSD-structured preview readiness at: {p['preview_dir']}")
        return
    print(f"  mode enabled                  = {p['enabled']}")
    print(f"  xsd_structured_preview_allowed= {p['xsd_structured_preview_allowed']}")
    print(f"  XML generated                 = {p['xml_generated']}")
    print(f"  XSD validation attempted      = {p['xsd_validation_attempted']}")
    print(f"  XSD validation passed         = {p['xsd_validation_passed']}")
    print(f"  records emitted               = {p['records_emitted']}")
    print(f"  fields emitted                = {p['fields_emitted']}")
    print(f"  placeholder fields            = {p['placeholder_fields']}")
    print(f"  excluded fields               = {p['excluded_fields']}")
    print(f"  RREC nested under Coll        = {p['rrec_all_nested_under_coll']}")
    print(f"  rejected/manual paths skipped = {p['rejected_or_manual_skipped'] or 'none'}")
    print(f"  production gates remain false = {p['production_gates_false']}")
    if not p["xsd_validation_passed"] and p["known_limitations"]:
        print("  known limitations (why XSD validation does not pass yet):")
        for k in p["known_limitations"]:
            print(f"    - {k}")


def inspect_xsd_structured_synthetic(delivery_dir: str | Path) -> Dict[str, Any]:
    """Read-only summary of the engineering-only XSD-structured synthetic schema
    test (4th mode) under ``<delivery_dir>/preview/``. Never writes."""
    d = Path(delivery_dir)
    sp = d / "preview" / "xsd_structured_synthetic_schema_test"
    xml_file = sp / "114_xsd_structured_synthetic.xml"
    validation = _read_json(sp / "116_xsd_structured_synthetic_xsd_validation.json")
    catalog = _read_csv(sp / "112_xsd_structured_synthetic_values_catalog.csv")
    manifest = _read_json(d / "60_delivery_manifest.json")
    synth_count = sum(1 for r in catalog if r.get("source") == "synthetic_schema_test")
    return {
        "synthetic_dir": str(sp),
        "exists": (sp / "116_xsd_structured_synthetic_xsd_validation.json").exists(),
        "xml_generated": xml_file.exists(),
        "xsd_validation_attempted": validation.get("xsd_validation_attempted"),
        "xsd_validation_passed": validation.get("xsd_validation_passed"),
        "error_count": validation.get("error_count"),
        "validation_errors": validation.get("validation_errors", []),
        "records_generated": validation.get("records_generated"),
        "fields_generated": validation.get("fields_generated"),
        "synthetic_values_count": synth_count or validation.get("synthetic_values_count"),
        "production_gates_false": not any([
            manifest.get("xml_generation_allowed"), manifest.get("ready_for_xml_delivery"),
            manifest.get("xml_generated")]),
    }


def _print_xsd_synthetic_report(p: Dict[str, Any]) -> None:
    print("\n=== XSD-structured SYNTHETIC schema test (engineering only) ===")
    if not p["exists"]:
        print(f"No synthetic schema test at: {p['synthetic_dir']}")
        print("(enable xsd_structured_synthetic_schema_test in the preview policy)")
        return
    print(f"  XML generated                = {p['xml_generated']}")
    print(f"  XSD validation attempted     = {p['xsd_validation_attempted']}")
    print(f"  XSD validation passed        = {p['xsd_validation_passed']}")
    print(f"  error count                  = {p['error_count']}")
    print(f"  records generated            = {p['records_generated']}")
    print(f"  fields generated             = {p['fields_generated']}")
    print(f"  synthetic values             = {p['synthetic_values_count']}")
    print(f"  production gates remain false = {p['production_gates_false']}")
    if p["error_count"] and p["validation_errors"]:
        print("  remaining validation errors (first few):")
        for e in p["validation_errors"][:5]:
            print(f"    - {e}")


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
    ap.add_argument("--preview", action="store_true",
                    help="Also report the non-production client-preview readiness "
                    "(70..74) under <delivery_dir>/preview/.")
    ap.add_argument("--synthetic-schema-test", action="store_true",
                    help="Also report the engineering-only synthetic full-coverage "
                    "schema-test readiness (75..77) under <delivery_dir>/preview/.")
    ap.add_argument("--xsd-structured-preview", action="store_true",
                    help="Also report the nested XSD-structured preview readiness "
                    "(78..79 + 100..107) under <delivery_dir>/preview/.")
    ap.add_argument("--xsd-structured-synthetic", action="store_true",
                    help="Also report the engineering-only XSD-structured synthetic "
                    "schema test (110..116) under <delivery_dir>/preview/.")
    args = ap.parse_args(argv)
    summary = inspect(args.delivery_dir)
    _print_report(summary)
    if args.format_invalid and summary["exists"]:
        _print_format_invalid(format_invalid_detail(
            args.delivery_dir,
            regime_config_path=args.regime_config or None,
            max_samples=args.samples))
    if args.preview or args.synthetic_schema_test:
        _print_preview_report(inspect_preview(args.delivery_dir))
    if args.xsd_structured_preview:
        _print_xsd_structured_report(inspect_xsd_structured_preview(args.delivery_dir))
    if args.xsd_structured_synthetic:
        _print_xsd_synthetic_report(inspect_xsd_structured_synthetic(args.delivery_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
