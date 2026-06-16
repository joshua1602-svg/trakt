"""annex2_handoff_validation.py
================================

Validate the onboarding -> Annex 2 XML handoff contract BEFORE invoking the
delivery pipeline. Report-only.

The promoted ``18_central_lender_tape.csv`` is a *generic onboarding/MI canonical
tape*, NOT an Annex 2 delivery tape. The Annex 2 delivery path (gate-4 projector
-> gate-4b normalizer -> gate-5 xml_builder) needs the Annex 2 *canonical* fields
(the registry fields that carry an ``ESMA_Annex2`` code), resolved from one of:

  * the promoted tape (direct column, or a registry synonym),
  * an asset default (``product_defaults_ERM.yaml``),
  * a regime default / ND default (``annex2_delivery_rules.yaml``),

or surfaced as a problem:

  * a *canonical alias mismatch* (a related MI column exists under a different
    name, e.g. ``current_outstanding_balance`` vs ``current_principal_balance``),
  * a *pending regime rule* (code has no rule yet — config backlog),
  * a *source mapping absent* (no tape column, default or ND).

This module produces ``50_annex2_xml_handoff_validation.{csv,json,md}`` so the
operator knows EXACTLY why a field would not deliver — instead of the misleading
gate-1 "core canonical fields missing from client tape" error that fires when an
already-promoted canonical tape is wrongly fed back through raw-source gate-1.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_REPO = Path(__file__).resolve().parents[2]
_REGIME_DEFAULT = _REPO / "config" / "regime" / "annex2_delivery_rules.yaml"
_UNIVERSE_DEFAULT = _REPO / "config" / "regime" / "annex2_field_universe.yaml"
_REGISTRY_DEFAULT = _REPO / "config" / "system" / "fields_registry.yaml"
_ASSET_DEFAULT = _REPO / "config" / "asset" / "product_defaults_ERM.yaml"

# Known MI-canonical -> Annex2-canonical candidates. These are SURFACED for an
# explicit operator decision, never auto-applied (the values differ in meaning).
_ALIAS_CANDIDATES: Dict[str, List[str]] = {
    "current_principal_balance": ["current_outstanding_balance", "original_principal_balance"],
    "property_post_code": ["postcode", "post_code"],
    "origination_date": ["policy_completion_date", "completion_date", "origination_date_of_policy"],
    "maturity_date": ["policy_maturity_date", "redemption_date"],
}

COLUMNS = [
    "esma_code", "annex2_required_field", "current_promoted_field_present",
    "matched_promoted_field", "source_resolution", "asset_default_available",
    "regime_default_available", "nd_allowed", "delivery_value_status",
    "blocking", "recommended_fix",
]


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _is_nd(v: Any) -> bool:
    return str(v).strip().upper().startswith("ND")


def _load_asset_default_keys(asset_path: Path) -> set:
    try:
        cfg = yaml.safe_load(asset_path.read_text()) or {}
    except Exception:
        return set()
    keys = set((cfg.get("defaults") or {}).keys())
    keys |= set((cfg.get("nd_defaults") or {}).keys())
    return keys


def _registry_code_to_canonical(registry_path: Path) -> Dict[str, Dict[str, Any]]:
    reg = (yaml.safe_load(registry_path.read_text()) or {}).get("fields", {}) or {}
    out: Dict[str, Dict[str, Any]] = {}
    for fname, meta in reg.items():
        code = ((meta or {}).get("regime_mapping") or {}).get("ESMA_Annex2", {}) or {}
        c = code.get("code")
        if c:
            out[c] = {"canonical": fname, "synonyms": (meta or {}).get("synonyms") or []}
    return out


def build_annex2_xml_handoff_validation(
    central_tape_path: str | Path,
    regime_config_path: Optional[str | Path] = None,
    registry_path: Optional[str | Path] = None,
    asset_config_path: Optional[str | Path] = None,
    universe_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    regime_config_path = Path(regime_config_path or _REGIME_DEFAULT)
    registry_path = Path(registry_path or _REGISTRY_DEFAULT)
    asset_config_path = Path(asset_config_path or _ASSET_DEFAULT)
    universe_path = Path(universe_path or _UNIVERSE_DEFAULT)

    rules = (yaml.safe_load(regime_config_path.read_text()) or {}).get("field_rules", {}) or {}
    universe = (yaml.safe_load(universe_path.read_text()) or {}).get("fields", {}) or {}
    code2canon = _registry_code_to_canonical(registry_path)
    asset_keys = _load_asset_default_keys(asset_config_path)

    # Promoted tape columns (normalised for matching).
    tape_cols: List[str] = []
    tp = Path(central_tape_path)
    if tp.exists():
        with tp.open(newline="") as f:
            header = next(csv.reader(f), [])
        tape_cols = [c.strip() for c in header]
    tape_norm = {_norm(c): c for c in tape_cols}

    def _tape_has(name: str) -> Optional[str]:
        return tape_norm.get(_norm(name))

    def _code_sort(c: str):
        import re
        m = re.match(r"([A-Za-z]+)(\d+)", c)
        return (m.group(1), int(m.group(2))) if m else (c, 0)

    rows: List[Dict[str, Any]] = []
    for code in sorted(set(universe) | set(rules) | set(code2canon), key=_code_sort):
        rule = rules.get(code)
        canon = code2canon.get(code, {}).get("canonical", "")
        required = (rule or {}).get("projected_source_field") or canon
        synonyms = code2canon.get(code, {}).get("synonyms") or []

        direct = _tape_has(required) if required else None
        synonym_hit = next((_tape_has(s) for s in synonyms if _tape_has(s)), None)
        alias_hit = next((_tape_has(a) for a in _ALIAS_CANDIDATES.get(required, [])
                          if _tape_has(a)), None)
        asset_def = required in asset_keys
        default_value = (rule or {}).get("default_value")
        default_allowed = bool((rule or {}).get("default_allowed"))
        regime_value_default = default_allowed and default_value is not None and not _is_nd(default_value)
        nd_list = [str(x).upper() for x in ((rule or {}).get("nd_allowed") or [])]
        nd_default = default_allowed and _is_nd(default_value) if default_value is not None else bool(nd_list)
        mandatory = bool((rule or {}).get("mandatory")) and bool((rule or {}).get("enforce_presence"))

        # Data resolution takes priority over rule presence: a value can deliver
        # via the tape or an asset default even when the delivery rule is still
        # pending. Pending-rule is surfaced as a secondary note / its own status
        # only when nothing else resolves the value.
        matched = ""
        if direct:
            matched, res, status, blocking = direct, "promoted_tape_direct", "resolved_from_tape", False
            fix = "none"
        elif synonym_hit:
            matched, res, status, blocking = synonym_hit, "promoted_tape_synonym", "resolved_by_synonym", False
            fix = f"delivered from tape synonym '{synonym_hit}' -> '{required}'"
        elif asset_def:
            res, status, blocking = "asset_default", "resolved_by_asset_default", False
            fix = "resolved from asset default (product_defaults_ERM.yaml)"
            if alias_hit:
                fix += (f"; NOTE tape has '{alias_hit}' — map to '{required}' to deliver "
                        "real data instead of the default")
        elif rule is not None and regime_value_default:
            res, status, blocking = "regime_default", "resolved_by_regime_default", False
            fix = f"resolved from regime default '{default_value}'"
        elif rule is not None and nd_default:
            res, status, blocking = "regime_nd_default", "resolved_by_nd_default", False
            fix = "resolved to a valid ND/default per the regime rule"
            if alias_hit:
                fix += (f"; tape has '{alias_hit}' — map to '{required}' to deliver real "
                        "data instead of ND")
        elif alias_hit:
            matched, res, status = alias_hit, "canonical_alias_mismatch", "needs_alias_decision"
            blocking = mandatory or rule is None
            fix = (f"map/derive tape column '{alias_hit}' -> Annex2 canonical "
                   f"'{required}' ({code}); values differ in meaning, so confirm "
                   "(add alias / approved mapping override / derivation)")
        elif rule is None:
            res, status, blocking = "pending_regime_rule", "pending_regime_rule", False
            fix = (f"no regime rule for {code} and no tape/asset source — add a field "
                   "rule in annex2_delivery_rules.yaml (config backlog) and/or map a source")
        else:
            res, status = "source_mapping_absent", "unresolved"
            blocking = mandatory
            fix = (f"no tape column, asset/regime default or ND for '{required}' ({code}); "
                   "obtain source data or confirm an ND value is permitted")

        if rule is None and res not in ("pending_regime_rule",):
            fix += " (note: no regime delivery rule yet — pending config)"

        rows.append({
            "esma_code": code,
            "annex2_required_field": required,
            "current_promoted_field_present": bool(direct),
            "matched_promoted_field": matched,
            "source_resolution": res,
            "asset_default_available": asset_def,
            "regime_default_available": bool(regime_value_default or nd_default),
            "nd_allowed": "; ".join(nd_list),
            "delivery_value_status": status,
            "blocking": blocking,
            "recommended_fix": fix,
        })
    return rows


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    res: Dict[str, int] = {}
    status: Dict[str, int] = {}
    for r in rows:
        res[r["source_resolution"]] = res.get(r["source_resolution"], 0) + 1
        status[r["delivery_value_status"]] = status.get(r["delivery_value_status"], 0) + 1
    blocking = [r for r in rows if r["blocking"]]
    return {
        "fields_total": len(rows),
        "resolved_from_tape": res.get("promoted_tape_direct", 0) + res.get("promoted_tape_synonym", 0),
        "resolved_by_asset_default": res.get("asset_default", 0),
        "resolved_by_regime_default": res.get("regime_default", 0) + res.get("regime_nd_default", 0),
        "canonical_alias_mismatch": res.get("canonical_alias_mismatch", 0),
        "pending_regime_rule": res.get("pending_regime_rule", 0),
        "source_mapping_absent": res.get("source_mapping_absent", 0),
        "blocking_total": len(blocking),
        "blocking_codes": [r["esma_code"] for r in blocking],
        "source_resolution_counts": res,
        "delivery_value_status_counts": status,
        # XML-ready ONLY when nothing blocking and nothing pending.
        "xml_ready": len(blocking) == 0 and res.get("pending_regime_rule", 0) == 0,
    }


def write(out_dir: str | Path, rows: List[Dict[str, Any]], summary: Dict[str, Any],
          *, central_tape_path: str = "") -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "50_annex2_xml_handoff_validation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    (out / "50_annex2_xml_handoff_validation.json").write_text(
        json.dumps({"central_tape": str(central_tape_path), "summary": summary,
                    "rows": rows}, indent=2, default=str))
    md = [
        "# Annex 2 XML handoff validation (50)", "",
        "Checks whether the promoted onboarding tape can actually deliver the "
        "Annex 2 canonical fields, BEFORE running the XML/XSD pipeline. Report-only.",
        "",
        f"- Promoted tape: `{central_tape_path}`",
        f"- Fields checked: **{summary['fields_total']}**",
        f"- **XML-ready: {summary['xml_ready']}**", "",
        "## Resolution",
        f"- Resolved from tape: **{summary['resolved_from_tape']}**",
        f"- Resolved by asset default: **{summary['resolved_by_asset_default']}**",
        f"- Resolved by regime/ND default: **{summary['resolved_by_regime_default']}**",
        f"- Canonical alias mismatch (decision needed): **{summary['canonical_alias_mismatch']}**",
        f"- Pending regime rule (config backlog): **{summary['pending_regime_rule']}**",
        f"- Source mapping absent: **{summary['source_mapping_absent']}**",
        f"- **Blocking: {summary['blocking_total']}**", "",
    ]
    blk = [r for r in rows if r["blocking"]]
    md += ["## Blocking items (must resolve before XML)", ""]
    if blk:
        for r in blk:
            md.append(f"- `{r['esma_code']}` {r['annex2_required_field']} "
                      f"({r['source_resolution']}): {r['recommended_fix']}")
    else:
        md.append("_None blocking._")
    pend = [r for r in rows if r["source_resolution"] == "pending_regime_rule"]
    md += ["", f"## Pending regime rules ({len(pend)})", "",
           "These are a config backlog (no field rule yet) — NOT 'core fields "
           "missing from the client tape'.", ""]
    md.append(", ".join(f"`{r['esma_code']}`" for r in pend) or "_None._")
    (out / "50_annex2_xml_handoff_validation_summary.md").write_text("\n".join(md) + "\n")
    return {
        "csv": str(out / "50_annex2_xml_handoff_validation.csv"),
        "json": str(out / "50_annex2_xml_handoff_validation.json"),
        "summary_md": str(out / "50_annex2_xml_handoff_validation_summary.md"),
    }


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Validate the onboarding -> Annex 2 XML handoff.")
    p.add_argument("--central-tape", required=True, help="Promoted 18_central_lender_tape.csv")
    p.add_argument("--regime-config", default=str(_REGIME_DEFAULT))
    p.add_argument("--registry", default=str(_REGISTRY_DEFAULT))
    p.add_argument("--asset-config", default=str(_ASSET_DEFAULT))
    p.add_argument("--universe", default=str(_UNIVERSE_DEFAULT))
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    rows = build_annex2_xml_handoff_validation(
        a.central_tape, a.regime_config, a.registry, a.asset_config, a.universe)
    summary = summarise(rows)
    paths = write(a.out_dir, rows, summary, central_tape_path=a.central_tape)
    print(json.dumps({"summary": summary, "paths": paths}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
