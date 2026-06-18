"""
preview_readiness.py
====================

Two SEPARATE non-production XML artefact modes for the Delivery/XML Agent.

Production XML stays blocked. This module never reads, writes or flips the
production gates (``xml_generation_allowed`` / ``ready_for_xml_delivery`` /
``xml_generated``). It only computes and emits two *additional*, clearly
non-production artefact families, each behind its own flag:

  * **Client Preview XML** (``client_safe_preview``) — a watermarked,
    client-facing preview built from real delivery-valid values, safe
    ``PREVIEW_ONLY_*`` placeholders for approved identifier fields, and explicit
    exclusions for operator-ambiguous / optional fields. It NEVER fabricates
    valuations, current rates or economic balances. Flags:
    ``xml_preview_allowed`` / ``ready_for_xml_preview`` / ``xml_preview_generated``.

  * **Synthetic Full-Coverage Schema Test XML**
    (``synthetic_full_coverage_schema_test``) — an engineering-only artefact
    that populates EVERY Annex 2 field with a dummy value (preferring real
    delivery-valid values where present) so the XML structure and full field
    coverage can be exercised. Every synthetic value is labelled
    ``source = synthetic_schema_test``. Flags: ``synthetic_schema_test_allowed`` /
    ``ready_for_synthetic_schema_test`` / ``synthetic_schema_test_generated``.

The field sets (placeholders, exclusions, must-resolve, never-fabricate) are
read from ``config/delivery/xml_preview_policy.yaml`` — they are NOT hard-coded
here. Both modes are disabled by default.

Reads a delivery package (60..63) produced by ``delivery_xml_agent`` and writes
the preview artefacts under ``<delivery_dir>/preview/``:

    70_xml_preview_readiness.json
    71_xml_preview_readiness.md
    72_xml_preview_policy_application.csv
    73_xml_preview_assumptions.csv
    74_xml_preview_blockers.csv
    75_synthetic_schema_test_readiness.json
    76_synthetic_schema_test_readiness.md
    77_synthetic_schema_field_plan.csv

    preview/client_preview/      80..86   (only if client mode enabled + allowed)
    preview/synthetic_schema_test/ 90..95 (only if synthetic mode enabled + allowed)

The production ESMA XSD path/cardinality mapping is NOT yet configured, so both
preview XML files are emitted under an internal, clearly non-production preview
namespace and are NOT XSD-valid. Production XSD mapping remains a documented
blocker (see ``docs/xml_preview_policy_spec.md``).
"""

from __future__ import annotations

import csv
import json
import xml.sax.saxutils as sax
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from engine.delivery_xml_agent import gate5_adapter as g5

AGENT = "delivery_xml_preview_evaluator"
AGENT_VERSION = "1.0"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_POLICY = _REPO_ROOT / "config" / "delivery" / "xml_preview_policy.yaml"
_DEFAULT_UNIVERSE = _REPO_ROOT / "config" / "regime" / "annex2_field_universe.yaml"
_DEFAULT_REGIME = _REPO_ROOT / "config" / "regime" / "annex2_delivery_rules.yaml"

# Disposition tokens used in the client-preview policy application.
DISP_REAL = "real_value"
DISP_PLACEHOLDER = "preview_placeholder"
DISP_EXCLUDED = "excluded"
DISP_MUST_RESOLVE = "must_resolve"
DISP_NOT_REQUIRED = "not_required_blank"

# Value-source labels.
SRC_REAL = "real_delivery_valid"
SRC_PLACEHOLDER = "preview_only_placeholder"
SRC_SYNTHETIC = "synthetic_schema_test"

# Dummy values per Annex 2 format token. Chosen to pass typical type/format
# validation. enum_map fields prefer the first authoritative code (see
# _synthetic_value_for_code). This is a value *generator*, not a field list.
_DUMMY_BY_TOKEN = {
    "{ALPHANUM-28}": "SYNTHDUMMY0000000000000000XX",
    "{ALPHANUM-100}": "SYNTH_SCHEMA_TEST_DUMMY",
    "{ALPHANUM-1000}": "SYNTH_SCHEMA_TEST_DUMMY",
    "{ALPHANUM-10000}": "SYNTH_SCHEMA_TEST_DUMMY",
    "{ALPHANUM-10}": "SYNTHDUMMY",
    "{MONETARY}": "1000.00",
    "{PERCENTAGE}": "0.0500",
    "{DATEFORMAT}": "2025-01-01",
    "{YEAR}": "2025",
    "{INTEGER-9999}": "100",
    "{Y/N}": "Y",
    "{CURRENCYCODE_3}": "EUR",
    "{COUNTRYCODE_2}": "ES",
    "{NUTS}": "ES30",
    "{LEI}": "5493000000000000DUMY",
    "{LIST}": "OTHR",
}
_DUMMY_FALLBACK = "SYNTH_DUMMY"


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "y")


# --------------------------------------------------------------------------- #
# Policy access
# --------------------------------------------------------------------------- #

def load_policy(policy_path: str | Path | None = None) -> Dict[str, Any]:
    """Load the preview policy YAML (source of truth for the field sets)."""
    return _read_yaml(Path(policy_path) if policy_path else _DEFAULT_POLICY)


def _mode_cfg(policy: Dict[str, Any], mode: str) -> Dict[str, Any]:
    modes = policy.get("preview_modes") or {}
    return modes.get(mode) or {}


# --------------------------------------------------------------------------- #
# Field-universe / regime access for the synthetic plan + dummy values
# --------------------------------------------------------------------------- #

def _load_field_rules(regime_config_path: str | Path) -> Dict[str, Dict[str, Any]]:
    cfg = _read_yaml(Path(regime_config_path))
    rules = cfg.get("field_rules")
    return {str(k): (v or {}) for k, v in rules.items()} if isinstance(rules, dict) else {}


def _enum_first_target(rule: Dict[str, Any]) -> str:
    transform = rule.get("transform") if isinstance(rule.get("transform"), dict) else {}
    enum_map = transform.get("enum_map") if isinstance(transform.get("enum_map"), dict) else {}
    if not enum_map:
        return ""
    # Prefer a stable authoritative target: first mapped *value*.
    return str(next(iter(enum_map.values())))


def _synthetic_value_for_code(
    code: str,
    universe_entry: Dict[str, Any],
    rule: Dict[str, Any],
) -> Tuple[str, str]:
    """Return ``(dummy_value, format_token)`` for a synthetic field.

    Prefers an enum_map authoritative code, else a per-token dummy, else a
    generic fallback. Always returns a non-empty value (full coverage).
    """
    token = str(universe_entry.get("format", "")).strip()
    enum_target = _enum_first_target(rule)
    if enum_target:
        return enum_target, token or "{LIST}"
    value = _DUMMY_BY_TOKEN.get(token, _DUMMY_FALLBACK)
    return value, token


def _xml_record_group(code: str, record_group: str) -> str:
    rg = (record_group or "").strip().upper()
    if not rg:
        rg = "RREC" if str(code).upper().startswith("RREC") else "RREL"
    return g5.record_group_to_xml_group(rg)


# --------------------------------------------------------------------------- #
# Client preview verdict
# --------------------------------------------------------------------------- #

def _client_disposition(
    row: Dict[str, Any],
    policy: Dict[str, Any],
) -> Tuple[str, str]:
    """Return ``(disposition, reason)`` for one delivery-frame code row under
    the client-preview policy."""
    cp = policy.get("client_preview_field_policy") or {}
    placeholder_fields = set(cp.get("placeholder_fields") or [])
    exclusion_fields = set(cp.get("exclusion_fields") or [])
    exclusion_types = set(cp.get("exclusion_blocker_types") or [])
    must_fields = set(cp.get("must_resolve_before_preview_fields") or [])
    must_types = set(cp.get("must_resolve_before_preview_blocker_types") or [])
    never_fields = set(cp.get("never_fabricate_fields") or [])
    never_tokens = set(cp.get("never_fabricate_format_tokens") or [])
    resolved = set(policy.get("resolved_fields") or [])

    code = row["esma_code"]
    status = row["delivery_status"]
    blocker = row.get("delivery_blocker_type", "")
    fmt_token = row.get("_format_token", "")

    if status == "deliverable":
        return DISP_REAL, "real delivery-valid value"
    if status == "not_required_blank":
        return DISP_NOT_REQUIRED, "field not required / blank, not shown"

    # Blocked / invalid below. Resolved fields should never be blocked, but be
    # explicit: a resolved field is never placeholdered/excluded by policy.
    if code in resolved:
        return DISP_MUST_RESOLVE, "resolved field unexpectedly blocked — investigate"

    # Economic fields are never fabricated/placeholdered in the client preview.
    if code in never_fields or fmt_token in never_tokens:
        if blocker in exclusion_types:
            return DISP_EXCLUDED, "economic field excluded (never fabricated)"
        return DISP_MUST_RESOLVE, "economic field blocked — never fabricated, must resolve"

    if code in must_fields:
        return DISP_MUST_RESOLVE, f"{code} must be resolved before preview"
    if blocker in must_types:
        return DISP_MUST_RESOLVE, f"blocker {blocker} must be resolved before preview"
    if code in placeholder_fields:
        return DISP_PLACEHOLDER, "approved identifier/reference placeholder"
    if code in exclusion_fields or blocker in exclusion_types:
        return DISP_EXCLUDED, f"excluded ({blocker or 'optional'})"
    # Conservative default: anything else blocking must be resolved.
    return DISP_MUST_RESOLVE, f"unclassified blocker {blocker or status}"


def evaluate_client_preview(
    *,
    code_rows: List[Dict[str, Any]],
    policy: Dict[str, Any],
    mode_cfg: Dict[str, Any],
    production_blockers: List[str],
) -> Dict[str, Any]:
    """Compute the client-preview verdict + per-code policy application."""
    applications: List[Dict[str, Any]] = []
    counts = {DISP_REAL: 0, DISP_PLACEHOLDER: 0, DISP_EXCLUDED: 0,
              DISP_MUST_RESOLVE: 0, DISP_NOT_REQUIRED: 0}
    must_resolve_codes: List[str] = []
    placeholder_codes: List[str] = []
    excluded_codes: List[str] = []

    prefix = (policy.get("client_preview_field_policy") or {}).get(
        "placeholder_prefix", "PREVIEW_ONLY_")

    for row in code_rows:
        disp, reason = _client_disposition(row, policy)
        counts[disp] = counts.get(disp, 0) + 1
        preview_value = ""
        source = ""
        if disp == DISP_REAL:
            preview_value = row.get("delivery_value", "") or row.get("projected_value", "")
            source = SRC_REAL
        elif disp == DISP_PLACEHOLDER:
            preview_value = f"{prefix}{row['esma_code']}"
            source = SRC_PLACEHOLDER
            placeholder_codes.append(row["esma_code"])
        elif disp == DISP_EXCLUDED:
            excluded_codes.append(row["esma_code"])
        elif disp == DISP_MUST_RESOLVE:
            must_resolve_codes.append(row["esma_code"])
        applications.append({
            "esma_code": row["esma_code"],
            "canonical_field": row.get("canonical_field", ""),
            "record_group": row.get("record_group", ""),
            "delivery_status": row["delivery_status"],
            "delivery_blocker_type": row.get("delivery_blocker_type", ""),
            "disposition": disp,
            "preview_value": preview_value,
            "value_source": source,
            "reason": reason,
        })

    allowed = len(must_resolve_codes) == 0
    enabled = bool(mode_cfg.get("enabled", False))
    verdict = {
        "mode": "client_safe_preview",
        "enabled": enabled,
        "allowed": allowed,
        "ready": bool(allowed and enabled),
        "reasons": (
            ["all blocking fields are covered by approved placeholders or exclusions"]
            if allowed else
            [f"{len(set(must_resolve_codes))} field(s) must be resolved before preview: "
             + ", ".join(sorted(set(must_resolve_codes))[:20])]
        ),
        "watermark": mode_cfg.get("watermark", ""),
        "disposition_counts": counts,
        "placeholder_codes": sorted(set(placeholder_codes)),
        "excluded_codes": sorted(set(excluded_codes)),
        "must_resolve_codes": sorted(set(must_resolve_codes)),
        "remaining_production_blockers": production_blockers,
        "applications": applications,
    }
    return verdict


# --------------------------------------------------------------------------- #
# Synthetic full-coverage verdict
# --------------------------------------------------------------------------- #

def evaluate_synthetic_full_coverage(
    *,
    universe: Dict[str, Dict[str, Any]],
    field_rules: Dict[str, Dict[str, Any]],
    real_values: Dict[str, str],
    policy: Dict[str, Any],
    mode_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Plan a synthetic value for every Annex 2 field in the universe.

    Real delivery-valid values are preferred (labelled real); every other field
    is populated with a labelled synthetic dummy. Engineering-only.
    """
    sp = policy.get("synthetic_schema_test_policy") or {}
    synth_label = sp.get("synthetic_source_label", SRC_SYNTHETIC)
    prefer_real = bool(sp.get("prefer_real_values", True))

    plan: List[Dict[str, Any]] = []
    synthetic_count = 0
    real_count = 0
    validated_count = 0
    for code in universe:
        entry = universe.get(code) or {}
        rule = field_rules.get(code, {})
        record_group = "RREC" if code.upper().startswith("RREC") else "RREL"
        real = real_values.get(code, "") if prefer_real else ""
        if real:
            value, source = real, SRC_REAL
            real_count += 1
        else:
            value, _token = _synthetic_value_for_code(code, entry, rule)
            source = synth_label
            synthetic_count += 1
        fmt_ok = g5.format_valid(value, rule)
        enum_ok = g5.enum_valid(value, rule)
        validated = bool(fmt_ok and enum_ok)
        if validated:
            validated_count += 1
        plan.append({
            "esma_code": code,
            "field_name": entry.get("field_name", ""),
            "record_group": record_group,
            "xml_record_group": _xml_record_group(code, record_group),
            "format_token": entry.get("format", ""),
            "value": value,
            "value_source": source,
            "format_valid": fmt_ok,
            "enum_valid": enum_ok,
            "validated": validated,
        })

    allowed = len(plan) > 0
    enabled = bool(mode_cfg.get("enabled", False))
    verdict = {
        "mode": "synthetic_full_coverage_schema_test",
        "enabled": enabled,
        "allowed": allowed,
        "ready": bool(allowed and enabled),
        "engineering_only": True,
        "client_facing": False,
        "reportable": False,
        "watermark": mode_cfg.get("watermark", ""),
        "field_universe_count": len(universe),
        "planned_field_count": len(plan),
        "synthetic_value_count": synthetic_count,
        "real_value_count": real_count,
        "validated_value_count": validated_count,
        "synthetic_source_label": synth_label,
        "plan": plan,
    }
    return verdict


# --------------------------------------------------------------------------- #
# XML emission (flat, non-production, internal preview namespace)
# --------------------------------------------------------------------------- #

def _xml_header(mode: str, watermark: str, namespace: str, extra: Dict[str, str]) -> List[str]:
    attrs = " ".join(f'{k}="{sax.quoteattr(v)[1:-1]}"' for k, v in extra.items())
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<!-- {watermark} -->",
        "<!-- THIS IS NOT A REGULATORY SUBMISSION. Production XML remains blocked. -->",
        f'<TraktNonProductionPreview xmlns="{namespace}" mode="{mode}" '
        f'watermark="{sax.quoteattr(watermark)[1:-1]}" {attrs}>',
    ]


def _xml_field(code: str, name: str, value: str, source: str, indent: str = "    ") -> str:
    return (f'{indent}<Field code="{code}" name="{sax.quoteattr(name)[1:-1]}" '
            f'source="{source}">{sax.escape(value)}</Field>')


def _build_client_preview_xml(
    *, applications: List[Dict[str, Any]], frame_rows: List[Dict[str, Any]],
    watermark: str, namespace: str, meta: Dict[str, str],
) -> str:
    """Flat, internally-consistent client preview XML. Fields with disposition
    real/placeholder are emitted; excluded/must-resolve are omitted (and listed
    in the exclusions/blockers artefacts)."""
    # disposition + preview value per code.
    disp_by_code = {a["esma_code"]: a for a in applications}
    emit_codes = {c for c, a in disp_by_code.items()
                  if a["disposition"] in (DISP_REAL, DISP_PLACEHOLDER)}

    lines = _xml_header("client_safe_preview", watermark, namespace, meta)
    lines.append("  <Meta>")
    for k, v in meta.items():
        lines.append(f"    <{k}>{sax.escape(v)}</{k}>")
    lines.append("    <ProductionXmlStatus>BLOCKED</ProductionXmlStatus>")
    lines.append("  </Meta>")

    # group emitted rows by loan + record group, preserving frame order.
    seen: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    order: List[Tuple[str, str]] = []
    for r in frame_rows:
        code = r["esma_code"]
        if code not in emit_codes:
            continue
        key = (r.get("loan_identifier", ""), r.get("record_group", ""))
        if key not in seen:
            seen[key] = []
            order.append(key)
        seen[key].append(r)

    for (loan, rg) in order:
        xml_group = _xml_record_group("", rg)
        elem = "Collateral" if xml_group == "collateral" else "UnderlyingExposure"
        lines.append(f'  <{elem} recordGroup="{rg}" loanIdentifier="{sax.escape(loan)}">')
        for r in seen[(loan, rg)]:
            a = disp_by_code[r["esma_code"]]
            lines.append(_xml_field(r["esma_code"], r.get("canonical_field", ""),
                                    a["preview_value"], a["value_source"]))
        lines.append(f"  </{elem}>")

    lines.append("</TraktNonProductionPreview>")
    return "\n".join(lines) + "\n"


def _build_synthetic_xml(
    *, plan: List[Dict[str, Any]], watermark: str, namespace: str, meta: Dict[str, str],
) -> str:
    """Flat full-coverage synthetic XML — one record per record group covering
    every Annex 2 field. Every value is labelled by source."""
    lines = _xml_header("synthetic_full_coverage_schema_test", watermark, namespace, meta)
    lines.append("  <Meta>")
    for k, v in meta.items():
        lines.append(f"    <{k}>{sax.escape(v)}</{k}>")
    lines.append("    <EngineeringOnly>true</EngineeringOnly>")
    lines.append("    <Reportable>false</Reportable>")
    lines.append("    <ProductionXmlStatus>BLOCKED</ProductionXmlStatus>")
    lines.append("  </Meta>")

    for rg, elem in (("RREL", "UnderlyingExposure"), ("RREC", "Collateral")):
        rows = [p for p in plan if p["record_group"] == rg]
        if not rows:
            continue
        lines.append(f'  <{elem} recordGroup="{rg}" loanIdentifier="SYNTH_LN0001">')
        for p in rows:
            lines.append(_xml_field(p["esma_code"], p.get("field_name", ""),
                                    p["value"], p["value_source"]))
        lines.append(f"  </{elem}>")

    lines.append("</TraktNonProductionPreview>")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def evaluate_and_emit(
    delivery_dir: str | Path,
    *,
    policy_path: str | Path | None = None,
    field_universe_path: str | Path | None = None,
    regime_config_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Evaluate both non-production preview modes for a delivery package and
    emit the readiness artefacts (70..77). Builds the client preview (80..86)
    and/or synthetic schema-test (90..95) artefacts ONLY when the corresponding
    mode is ``enabled`` in the policy AND its verdict allows it.

    Never reads, writes or flips production XML gates. Returns a dict of the two
    verdicts, the new preview/synthetic flags and the artefact paths.
    """
    d = Path(delivery_dir)
    if not (d / "60_delivery_manifest.json").exists():
        raise FileNotFoundError(
            f"No delivery package at {d} (expected 60_delivery_manifest.json).")

    policy = load_policy(policy_path)
    universe = g5.field_universe_index(
        field_universe_path or
        (policy.get("synthetic_schema_test_policy") or {}).get("field_universe_path")
        or _DEFAULT_UNIVERSE)
    field_rules = _load_field_rules(regime_config_path or _DEFAULT_REGIME)

    manifest = _read_json(d / "60_delivery_manifest.json")
    readiness = _read_json(d / "61_delivery_readiness.json")
    frame_rows = _read_csv(d / "62_delivery_normalised_frame.csv")
    issues = _read_csv(d / "63_delivery_issues.csv")

    # Production gates are READ-ONLY here — echoed for audit, never changed.
    production_flags = {
        "xml_generation_allowed": bool(manifest.get("xml_generation_allowed", False)),
        "ready_for_xml_delivery": bool(manifest.get("ready_for_xml_delivery", False)),
        "xml_generated": bool(manifest.get("xml_generated", False)),
    }

    # remaining production blockers (distinct blocker types in 63).
    production_blockers = sorted({
        i.get("delivery_blocker_type", "") for i in issues
        if i.get("delivery_blocker_type")
    })

    # Collapse the long frame to one row per code (a code's worst disposition).
    code_rows = _collapse_frame_to_codes(frame_rows, universe)
    real_values = {
        c["esma_code"]: c.get("delivery_value", "")
        for c in code_rows
        if c["delivery_status"] == "deliverable" and c.get("delivery_value", "")
    }

    client_cfg = _mode_cfg(policy, "client_safe_preview")
    synth_cfg = _mode_cfg(policy, "synthetic_full_coverage_schema_test")

    client_verdict = evaluate_client_preview(
        code_rows=code_rows, policy=policy, mode_cfg=client_cfg,
        production_blockers=production_blockers)
    synth_verdict = evaluate_synthetic_full_coverage(
        universe=universe, field_rules=field_rules, real_values=real_values,
        policy=policy, mode_cfg=synth_cfg)

    preview_dir = d / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    meta_common = {
        "client_id": str(manifest.get("client_id", "")),
        "run_id": str(manifest.get("run_id", "")),
        "target_contract_id": str(manifest.get("target_contract_id", "")),
    }

    # ---- 70..74 client preview readiness artefacts ----
    _write_client_readiness_artefacts(
        preview_dir, client_verdict, production_flags, policy, meta_common)

    # ---- 75..77 synthetic readiness artefacts ----
    _write_synthetic_readiness_artefacts(
        preview_dir, synth_verdict, production_flags, meta_common)

    # ---- emit client preview XML (80..86) only if enabled AND allowed ----
    client_generated = False
    client_paths: Dict[str, str] = {}
    if client_cfg.get("enabled") and client_verdict["allowed"]:
        client_paths = _emit_client_preview(
            preview_dir, client_cfg, client_verdict, frame_rows, policy, meta_common,
            production_blockers)
        client_generated = True

    # ---- emit synthetic schema-test XML (90..95) only if enabled AND allowed ----
    synth_generated = False
    synth_paths: Dict[str, str] = {}
    if synth_cfg.get("enabled") and synth_verdict["allowed"]:
        synth_paths = _emit_synthetic_schema_test(
            preview_dir, synth_cfg, synth_verdict, policy, meta_common)
        synth_generated = True

    # New, SEPARATE flags — never the production gates.
    flags = {
        "xml_preview_allowed": client_verdict["allowed"],
        "ready_for_xml_preview": client_verdict["ready"],
        "xml_preview_generated": client_generated,
        "synthetic_schema_test_allowed": synth_verdict["allowed"],
        "ready_for_synthetic_schema_test": synth_verdict["ready"],
        "synthetic_schema_test_generated": synth_generated,
        # production gates echoed read-only and asserted unchanged.
        "production_flags_unchanged": production_flags,
    }

    return {
        "preview_dir": str(preview_dir),
        "client_preview_verdict": client_verdict,
        "synthetic_full_coverage_verdict": synth_verdict,
        "flags": flags,
        "client_preview_paths": client_paths,
        "synthetic_schema_test_paths": synth_paths,
    }


def _collapse_frame_to_codes(
    frame_rows: List[Dict[str, Any]], universe: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """One representative row per ESMA code. A code is treated as blocked/invalid
    if ANY of its rows is, so the preview is conservative."""
    severity = {"delivery_invalid": 4, "blocked": 3, "not_required_blank": 2,
                "deliverable": 1, "": 0}
    by_code: Dict[str, Dict[str, Any]] = {}
    for r in frame_rows:
        code = r.get("esma_code", "")
        if not code:
            continue
        cur = by_code.get(code)
        if cur is None or severity.get(r.get("delivery_status", ""), 0) > \
                severity.get(cur.get("delivery_status", ""), 0):
            by_code[code] = dict(r)
    out = []
    for code, r in by_code.items():
        r["_format_token"] = str((universe.get(code) or {}).get("format", ""))
        out.append(r)
    return sorted(out, key=lambda x: x["esma_code"])


# --------------------------------------------------------------------------- #
# Readiness artefact writers
# --------------------------------------------------------------------------- #

def _write_client_readiness_artefacts(
    preview_dir: Path, verdict: Dict[str, Any], production_flags: Dict[str, bool],
    policy: Dict[str, Any], meta: Dict[str, str],
) -> None:
    # 70 json
    doc = {
        "agent": AGENT, "agent_version": AGENT_VERSION, "created_at": _now(),
        **meta,
        "mode": "client_safe_preview",
        "xml_preview_allowed": verdict["allowed"],
        "ready_for_xml_preview": verdict["ready"],
        "enabled": verdict["enabled"],
        "watermark": verdict["watermark"],
        "reasons": verdict["reasons"],
        "disposition_counts": verdict["disposition_counts"],
        "placeholder_codes": verdict["placeholder_codes"],
        "excluded_codes": verdict["excluded_codes"],
        "must_resolve_codes": verdict["must_resolve_codes"],
        "remaining_production_blockers": verdict["remaining_production_blockers"],
        # production gates are echoed read-only and explicitly unchanged.
        "production_flags_unchanged": production_flags,
        "production_xml_remains_blocked": True,
    }
    (preview_dir / "70_xml_preview_readiness.json").write_text(
        json.dumps(doc, indent=2, default=str), encoding="utf-8")

    # 71 md
    (preview_dir / "71_xml_preview_readiness.md").write_text(
        _client_readiness_md(verdict, production_flags, meta), encoding="utf-8")

    # 72 policy application
    _write_csv(preview_dir / "72_xml_preview_policy_application.csv",
               ["esma_code", "canonical_field", "record_group", "delivery_status",
                "delivery_blocker_type", "disposition", "preview_value",
                "value_source", "reason"],
               verdict["applications"])

    # 73 assumptions (placeholders)
    assumptions = [
        {"esma_code": a["esma_code"], "canonical_field": a["canonical_field"],
         "assumption": "preview-only placeholder substituted for blocked identifier",
         "preview_value": a["preview_value"], "value_source": a["value_source"]}
        for a in verdict["applications"] if a["disposition"] == DISP_PLACEHOLDER
    ]
    _write_csv(preview_dir / "73_xml_preview_assumptions.csv",
               ["esma_code", "canonical_field", "assumption", "preview_value",
                "value_source"], assumptions)

    # 74 blockers (must-resolve + remaining production blockers)
    blockers = [
        {"esma_code": a["esma_code"], "canonical_field": a["canonical_field"],
         "delivery_blocker_type": a["delivery_blocker_type"],
         "kind": "must_resolve_before_preview", "detail": a["reason"]}
        for a in verdict["applications"] if a["disposition"] == DISP_MUST_RESOLVE
    ]
    for bt in verdict["remaining_production_blockers"]:
        blockers.append({"esma_code": "", "canonical_field": "",
                         "delivery_blocker_type": bt, "kind": "remaining_production_blocker",
                         "detail": "production XML remains blocked by this category"})
    _write_csv(preview_dir / "74_xml_preview_blockers.csv",
               ["esma_code", "canonical_field", "delivery_blocker_type", "kind",
                "detail"], blockers)


def _write_synthetic_readiness_artefacts(
    preview_dir: Path, verdict: Dict[str, Any], production_flags: Dict[str, bool],
    meta: Dict[str, str],
) -> None:
    doc = {
        "agent": AGENT, "agent_version": AGENT_VERSION, "created_at": _now(),
        **meta,
        "mode": "synthetic_full_coverage_schema_test",
        "synthetic_schema_test_allowed": verdict["allowed"],
        "ready_for_synthetic_schema_test": verdict["ready"],
        "enabled": verdict["enabled"],
        "engineering_only": True,
        "client_facing": False,
        "reportable": False,
        "watermark": verdict["watermark"],
        "field_universe_count": verdict["field_universe_count"],
        "planned_field_count": verdict["planned_field_count"],
        "synthetic_value_count": verdict["synthetic_value_count"],
        "real_value_count": verdict["real_value_count"],
        "validated_value_count": verdict["validated_value_count"],
        "production_flags_unchanged": production_flags,
        "production_xml_remains_blocked": True,
    }
    (preview_dir / "75_synthetic_schema_test_readiness.json").write_text(
        json.dumps(doc, indent=2, default=str), encoding="utf-8")
    (preview_dir / "76_synthetic_schema_test_readiness.md").write_text(
        _synthetic_readiness_md(verdict, production_flags, meta), encoding="utf-8")
    _write_csv(preview_dir / "77_synthetic_schema_field_plan.csv",
               ["esma_code", "field_name", "record_group", "xml_record_group",
                "format_token", "value", "value_source", "format_valid",
                "enum_valid", "validated"],
               verdict["plan"])


def _client_readiness_md(verdict, production_flags, meta) -> str:
    def yn(v):
        return "✅ yes" if v else "❌ no"
    c = verdict["disposition_counts"]
    lines = [
        "# Client Preview XML readiness", "",
        f"Client: {meta.get('client_id','')}  ",
        f"Run: {meta.get('run_id','')}  ",
        f"Mode: **client_safe_preview** (non-production)", "",
        "> This is not a reportable regulatory XML. This is a non-production "
        "preview. Production XML remains blocked.", "",
        "## Verdict", "",
        f"- mode enabled: {yn(verdict['enabled'])}",
        f"- xml_preview_allowed: {yn(verdict['allowed'])}",
        f"- ready_for_xml_preview: {yn(verdict['ready'])}",
        "",
        "## Production gates (UNCHANGED, read-only)", "",
        f"- xml_generation_allowed: {yn(production_flags['xml_generation_allowed'])}",
        f"- ready_for_xml_delivery: {yn(production_flags['ready_for_xml_delivery'])}",
        f"- xml_generated: {yn(production_flags['xml_generated'])}",
        "",
        "## Disposition", "",
        f"- real values: {c.get(DISP_REAL,0)}",
        f"- preview-only placeholders: {c.get(DISP_PLACEHOLDER,0)}",
        f"- excluded: {c.get(DISP_EXCLUDED,0)}",
        f"- must-resolve-before-preview: {c.get(DISP_MUST_RESOLVE,0)}",
        "",
        "The following fields use preview-only placeholders:",
        "  " + (", ".join(verdict["placeholder_codes"]) or "(none)"),
        "",
        "The following fields are excluded:",
        "  " + (", ".join(verdict["excluded_codes"]) or "(none)"),
        "",
        "The following production blockers remain:",
        "  " + (", ".join(verdict["remaining_production_blockers"]) or "(none)"),
        "",
        "## Reasons", "",
    ]
    lines += [f"- {r}" for r in verdict["reasons"]]
    lines.append("")
    return "\n".join(lines) + "\n"


def _synthetic_readiness_md(verdict, production_flags, meta) -> str:
    def yn(v):
        return "✅ yes" if v else "❌ no"
    lines = [
        "# Synthetic Full-Coverage Schema Test XML readiness", "",
        f"Client: {meta.get('client_id','')}  ",
        f"Run: {meta.get('run_id','')}  ",
        f"Mode: **synthetic_full_coverage_schema_test** (engineering only)", "",
        "> This is a synthetic engineering artefact. It is not client-facing by "
        "default. It is not reportable. It is designed to test XML structure and "
        "field coverage.", "",
        "## Verdict", "",
        f"- mode enabled: {yn(verdict['enabled'])}",
        f"- synthetic_schema_test_allowed: {yn(verdict['allowed'])}",
        f"- ready_for_synthetic_schema_test: {yn(verdict['ready'])}",
        "",
        "## Coverage", "",
        f"- Annex 2 field universe: {verdict['field_universe_count']}",
        f"- planned fields: {verdict['planned_field_count']}",
        f"- synthetic dummy values: {verdict['synthetic_value_count']}",
        f"- real values reused: {verdict['real_value_count']}",
        f"- values passing type/enum validation: {verdict['validated_value_count']}",
        "",
        "## Production gates (UNCHANGED, read-only)", "",
        f"- xml_generation_allowed: {yn(production_flags['xml_generation_allowed'])}",
        f"- ready_for_xml_delivery: {yn(production_flags['ready_for_xml_delivery'])}",
        "",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Client preview emission (80..86)
# --------------------------------------------------------------------------- #

def _emit_client_preview(
    preview_dir: Path, mode_cfg: Dict[str, Any], verdict: Dict[str, Any],
    frame_rows: List[Dict[str, Any]], policy: Dict[str, Any], meta: Dict[str, str],
    production_blockers: List[str],
) -> Dict[str, str]:
    out = preview_dir / "client_preview"
    out.mkdir(parents=True, exist_ok=True)
    watermark = mode_cfg.get("watermark", "")
    namespace = (policy.get("xml_emission") or {}).get(
        "preview_namespace", "urn:trakt:nonproduction:preview")
    apps = verdict["applications"]

    # 80 frame
    _write_csv(out / "80_client_preview_frame.csv",
               ["esma_code", "canonical_field", "record_group", "disposition",
                "preview_value", "value_source"],
               [{"esma_code": a["esma_code"], "canonical_field": a["canonical_field"],
                 "record_group": a["record_group"], "disposition": a["disposition"],
                 "preview_value": a["preview_value"], "value_source": a["value_source"]}
                for a in apps if a["disposition"] in (DISP_REAL, DISP_PLACEHOLDER)])

    # 81 lineage
    (out / "81_client_preview_lineage.json").write_text(json.dumps({
        **meta, "mode": "client_safe_preview", "created_at": _now(),
        "source_delivery_frame": "62_delivery_normalised_frame.csv",
        "non_production": True, "production_xml_remains_blocked": True,
        "placeholder_codes": verdict["placeholder_codes"],
        "excluded_codes": verdict["excluded_codes"],
    }, indent=2), encoding="utf-8")

    # 82 assumptions
    _write_csv(out / "82_client_preview_assumptions.csv",
               ["esma_code", "canonical_field", "preview_value", "value_source", "assumption"],
               [{"esma_code": a["esma_code"], "canonical_field": a["canonical_field"],
                 "preview_value": a["preview_value"], "value_source": a["value_source"],
                 "assumption": "preview-only placeholder for blocked identifier"}
                for a in apps if a["disposition"] == DISP_PLACEHOLDER])

    # 83 exclusions
    _write_csv(out / "83_client_preview_exclusions.csv",
               ["esma_code", "canonical_field", "delivery_blocker_type", "reason"],
               [{"esma_code": a["esma_code"], "canonical_field": a["canonical_field"],
                 "delivery_blocker_type": a["delivery_blocker_type"], "reason": a["reason"]}
                for a in apps if a["disposition"] == DISP_EXCLUDED])

    # 84 watermark
    (out / "84_client_preview_watermark.txt").write_text(
        watermark + "\nProduction XML remains blocked. Not for regulatory submission.\n",
        encoding="utf-8")

    # 85 xml
    xml_text = _build_client_preview_xml(
        applications=apps, frame_rows=frame_rows, watermark=watermark,
        namespace=namespace, meta=meta)
    (out / "85_client_preview.xml").write_text(xml_text, encoding="utf-8")

    # 86 summary
    (out / "86_client_preview_summary.md").write_text(
        _client_summary_md(verdict, meta, production_blockers), encoding="utf-8")

    return {k: str(out / v) for k, v in {
        "frame": "80_client_preview_frame.csv",
        "lineage": "81_client_preview_lineage.json",
        "assumptions": "82_client_preview_assumptions.csv",
        "exclusions": "83_client_preview_exclusions.csv",
        "watermark": "84_client_preview_watermark.txt",
        "xml": "85_client_preview.xml",
        "summary": "86_client_preview_summary.md",
    }.items()}


def _client_summary_md(verdict, meta, production_blockers) -> str:
    lines = [
        "# Client Preview XML — summary", "",
        "This is not a reportable regulatory XML.",
        "This is a non-production preview.",
        "Production XML remains blocked.", "",
        "The following fields use preview-only placeholders:",
        "  " + (", ".join(verdict["placeholder_codes"]) or "(none)"), "",
        "The following fields are excluded:",
        "  " + (", ".join(verdict["excluded_codes"]) or "(none)"), "",
        "The following production blockers remain:",
        "  " + (", ".join(production_blockers) or "(none)"), "",
        "> Structure note: this preview is structurally illustrative / internally "
        "consistent, emitted under an internal preview namespace. It is NOT "
        "production-XSD-valid. Production XSD path/cardinality mapping remains a "
        "blocker (see docs/xml_preview_policy_spec.md).", "",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Synthetic schema-test emission (90..95)
# --------------------------------------------------------------------------- #

def _emit_synthetic_schema_test(
    preview_dir: Path, mode_cfg: Dict[str, Any], verdict: Dict[str, Any],
    policy: Dict[str, Any], meta: Dict[str, str],
) -> Dict[str, str]:
    out = preview_dir / "synthetic_schema_test"
    out.mkdir(parents=True, exist_ok=True)
    watermark = mode_cfg.get("watermark", "")
    namespace = (policy.get("xml_emission") or {}).get(
        "preview_namespace", "urn:trakt:nonproduction:preview")
    plan = verdict["plan"]

    # 90 frame
    _write_csv(out / "90_synthetic_schema_frame.csv",
               ["esma_code", "field_name", "record_group", "xml_record_group",
                "format_token", "value", "value_source"],
               [{k: p[k] for k in ("esma_code", "field_name", "record_group",
                                   "xml_record_group", "format_token", "value",
                                   "value_source")} for p in plan])

    # 91 lineage
    (out / "91_synthetic_schema_lineage.json").write_text(json.dumps({
        **meta, "mode": "synthetic_full_coverage_schema_test", "created_at": _now(),
        "engineering_only": True, "client_facing": False, "reportable": False,
        "non_production": True, "production_xml_remains_blocked": True,
        "field_universe_count": verdict["field_universe_count"],
        "synthetic_source_label": verdict["synthetic_source_label"],
    }, indent=2), encoding="utf-8")

    # 92 synthetic values catalog (every value labelled with its source)
    _write_csv(out / "92_synthetic_values_catalog.csv",
               ["esma_code", "field_name", "value", "source", "format_token",
                "validated"],
               [{"esma_code": p["esma_code"], "field_name": p["field_name"],
                 "value": p["value"], "source": p["value_source"],
                 "format_token": p["format_token"], "validated": p["validated"]}
                for p in plan])

    # 93 watermark
    (out / "93_synthetic_schema_watermark.txt").write_text(
        watermark + "\nENGINEERING ONLY. Not client-facing. Not reportable. "
        "Production XML remains blocked.\n", encoding="utf-8")

    # 94 xml
    xml_text = _build_synthetic_xml(
        plan=plan, watermark=watermark, namespace=namespace, meta=meta)
    (out / "94_synthetic_schema_test.xml").write_text(xml_text, encoding="utf-8")

    # 95 summary
    (out / "95_synthetic_schema_summary.md").write_text(
        _synthetic_summary_md(verdict), encoding="utf-8")

    return {k: str(out / v) for k, v in {
        "frame": "90_synthetic_schema_frame.csv",
        "lineage": "91_synthetic_schema_lineage.json",
        "catalog": "92_synthetic_values_catalog.csv",
        "watermark": "93_synthetic_schema_watermark.txt",
        "xml": "94_synthetic_schema_test.xml",
        "summary": "95_synthetic_schema_summary.md",
    }.items()}


def _synthetic_summary_md(verdict) -> str:
    lines = [
        "# Synthetic Full-Coverage Schema Test XML — summary", "",
        "This is a synthetic engineering artefact.",
        "It is not client-facing by default.",
        "It is not reportable.",
        "It is designed to test XML structure and field coverage.", "",
        f"- Annex 2 fields covered: {verdict['planned_field_count']} / "
        f"{verdict['field_universe_count']}",
        f"- synthetic dummy values: {verdict['synthetic_value_count']} "
        f"(labelled source = {verdict['synthetic_source_label']})",
        f"- real values reused: {verdict['real_value_count']}",
        f"- values passing type/enum validation: {verdict['validated_value_count']}", "",
        "> Structure note: full-field coverage test emitted under an internal "
        "preview namespace. It is NOT production-XSD-valid. Production XSD "
        "path/cardinality mapping remains a blocker.", "",
    ]
    return "\n".join(lines) + "\n"
