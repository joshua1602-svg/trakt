"""mi_agent/risk_monitor/risk_limits_contract.py

Production risk-limits CONFIG CONTRACT.

The onboarding agent discovers a client's Schedule 8 concentration-limits
document, parses it deterministically (``schedule8_extractor``) and emits a
governed, machine-readable config at::

    onboarding_output/<client_id>/<run_id>/output/risk/risk_limits_config.yaml

This module is the single source of truth for that contract: it BUILDS the
config from extracted limits, WRITES it to the run path, READS it back, and
CONVERTS it to the internal limit shape the monitor computes against.

Governance rules (same as the extractor):
  * never fabricate a limit or a source — a missing/unreadable Schedule 8 yields
    ``extraction_status: not_found`` / ``failed`` with diagnostics, NOT silent
    placeholder limits;
  * every limit carries its ``source_text`` and ``source`` lineage;
  * the config is self-describing (``source_type`` / ``extraction_status`` /
    ``is_placeholder``) so the API and UI never present fallback/placeholder
    limits as if they were real Schedule 8 limits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from mi_agent.risk_monitor import schedule8_extractor as extractor

# Contract version + the controlled enums (kept stable for the API/UI contract).
CONTRACT_VERSION = "1.0"

# source_type ∈ these. schedule_8_doc = parsed from the client's Schedule 8 doc;
# onboarding_config / fallback_config = a committed config; placeholder = none.
SOURCE_SCHEDULE_8 = "schedule_8_doc"
SOURCE_ONBOARDING_CONFIG = "onboarding_config"
SOURCE_FALLBACK_CONFIG = "fallback_config"
SOURCE_PLACEHOLDER = "placeholder"

# extraction_status ∈ these.
STATUS_SUCCESS = "success"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
STATUS_NOT_FOUND = "not_found"

_AMBER_FRACTION = 0.9  # warning threshold = 90% of the hard limit (matches the monitor)

_UNIT_FOR_METRIC = {"exposure_pct": "percent", "count": "count", "amount_gbp": "gbp"}
_METRIC_FOR_UNIT = {"percent": "exposure_pct", "count": "count", "gbp": "amount_gbp"}


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #
_CATEGORY_DISPLAY = {
    "geographic_concentration": "Geographic concentration",
    "broker_concentration": "Broker / intermediary concentration",
    "large_loan_concentration": "Loan size concentration",
    "ltv_limit": "Loan-to-value",
    "interest_rate_limit": "Interest rate / WAC",
    "borrower_concentration": "Borrower concentration",
    "joint_borrower_limit": "Joint borrowers",
    "age_limit": "Borrower age",
    "property_value_concentration": "Property value",
    "other": "Other limit",
}


def _display_name(lim: Dict[str, Any]) -> str:
    cat = lim.get("category", "other")
    base = _CATEGORY_DISPLAY.get(cat, cat.replace("_", " ").title())
    region = lim.get("region")
    if region:
        return f"{base} — {region}"
    dim = lim.get("dimension")
    if dim and cat not in ("geographic_concentration",):
        return f"{base} ({dim})"
    return base


def _warning_threshold(threshold: Optional[float], operator: str) -> Optional[float]:
    if threshold is None:
        return None
    if operator == "min":
        # Warn before falling below the floor.
        return round(threshold / _AMBER_FRACTION, 4)
    return round(threshold * _AMBER_FRACTION, 4)


# --------------------------------------------------------------------------- #
# Build (extracted limits -> contract)
# --------------------------------------------------------------------------- #
def _limit_to_contract(lim: Dict[str, Any], *, source_label: str) -> Dict[str, Any]:
    operator = lim.get("direction", "max")
    threshold = lim.get("limit_value")
    metric = lim.get("metric") or _METRIC_FOR_UNIT.get(lim.get("unit", ""), "exposure_pct")
    region = lim.get("region")
    return {
        "limit_id": lim.get("limit_id"),
        "category": lim.get("category"),
        "display_name": _display_name(lim),
        "description": (lim.get("source_section") or "").strip() or None,
        "metric": metric,
        "region_codes": [region] if region else [],
        "operator": operator,
        "threshold": threshold,
        "warning_threshold": _warning_threshold(threshold, operator),
        "denominator": "funded_balance",
        "source_text": lim.get("source_snippet"),
        "source": source_label,
        # Governance fields carried through so the config round-trips losslessly.
        "unit": lim.get("unit"),
        "dimension": lim.get("dimension"),
        "confidence": lim.get("confidence"),
        "needs_review": bool(lim.get("needs_review")),
        "source_section": lim.get("source_section"),
    }


def build_config(client_id: str, *, search_roots: Optional[List[str]] = None,
                 extracted_at: str = "") -> Dict[str, Any]:
    """Discover a Schedule 8 doc for ``client_id`` and build the config contract.

    Never raises and never fabricates: an absent / unreadable / unparseable
    document is reported with the appropriate ``source_type`` /
    ``extraction_status`` and a ``diagnostics`` block, with an empty ``limits``
    list — NOT placeholder limits.
    """
    doc = extractor.locate_client_schedule8(client_id, pack_roots=search_roots)

    if doc is None:
        return {
            "contract_version": CONTRACT_VERSION,
            "client_id": client_id,
            "source_type": SOURCE_PLACEHOLDER,
            "source_file": None,
            "extraction_status": STATUS_NOT_FOUND,
            "extracted_at": extracted_at,
            "is_placeholder": True,
            "diagnostics": {
                "reason": "Schedule 8 not found in docs folder.",
                "searched_roots": list(search_roots or []),
            },
            "limits": [],
        }

    if doc.suffix.lower() not in extractor._READABLE_SUFFIXES:
        # Found but not machine-readable (PDF/DOCX) — diagnostics, not placeholders.
        return {
            "contract_version": CONTRACT_VERSION,
            "client_id": client_id,
            "source_type": SOURCE_SCHEDULE_8,
            "source_file": str(doc),
            "extraction_status": STATUS_FAILED,
            "extracted_at": extracted_at,
            "is_placeholder": False,
            "diagnostics": {
                "reason": (f"Schedule 8 document found ({doc.name}) but it is not "
                           "machine-readable (PDF/DOCX). Convert to text or add a parser."),
                "suffix": doc.suffix.lower(),
                "supported_text_suffixes": list(extractor._READABLE_SUFFIXES),
            },
            "limits": [],
        }

    extracted = extractor.extract_from_file(doc, portfolio_id=client_id)
    limits_in = extracted.get("limits", [])
    needs_review = int(extracted.get("needs_review_count", 0) or 0)
    if not limits_in:
        status = STATUS_FAILED
    elif needs_review:
        status = STATUS_PARTIAL
    else:
        status = STATUS_SUCCESS

    contract = {
        "contract_version": CONTRACT_VERSION,
        "client_id": client_id,
        "source_type": SOURCE_SCHEDULE_8,
        "source_file": str(doc),
        "extraction_status": status,
        "extracted_at": extracted_at,
        "is_placeholder": False,
        "extraction_method": extracted.get("extraction_method", "deterministic"),
        "categories": extracted.get("categories", []),
        "needs_review_count": needs_review,
        "limits": [_limit_to_contract(l, source_label="Schedule 8 document")
                   for l in limits_in],
    }
    if status != STATUS_SUCCESS:
        contract["diagnostics"] = {
            "reason": ("No numeric limits parsed from the Schedule 8 document."
                       if status == STATUS_FAILED
                       else f"{needs_review} limit(s) need manual review."),
            "needs_review_count": needs_review,
        }
    return contract


# --------------------------------------------------------------------------- #
# Convert (contract -> internal limit shape for the monitor)
# --------------------------------------------------------------------------- #
def config_to_internal_limits(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert contract ``limits`` back to the internal shape ``risk_limits``'s
    ``_compute_tests`` consumes (category / region / direction / limit_value /
    unit / source_snippet / ...)."""
    out: List[Dict[str, Any]] = []
    for c in config.get("limits", []) or []:
        regions = c.get("region_codes") or []
        unit = c.get("unit") or _UNIT_FOR_METRIC.get(c.get("metric", ""), "percent")
        out.append({
            "limit_id": c.get("limit_id"),
            "category": c.get("category"),
            "region": regions[0] if regions else None,
            "dimension": c.get("dimension"),
            "metric": c.get("metric"),
            "direction": c.get("operator", "max"),
            "limit_value": c.get("threshold"),
            "warning_value": c.get("warning_threshold"),
            "unit": unit,
            "exposure_basis": "funded",
            "confidence": c.get("confidence"),
            "needs_review": bool(c.get("needs_review")),
            "source_section": c.get("source_section") or c.get("description"),
            "source_snippet": c.get("source_text"),
        })
    return out


# --------------------------------------------------------------------------- #
# Write / read at the run path
# --------------------------------------------------------------------------- #
def run_config_path(output_root, client_id: str, run_id: str) -> Path:
    return (Path(output_root) / client_id / run_id / "output" / "risk"
            / "risk_limits_config.yaml")


def write_config(output_root, client_id: str, run_id: str,
                 config: Dict[str, Any]) -> Path:
    """Write the contract YAML to the governed run path and return it."""
    path = run_config_path(output_root, client_id, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def write_config_to_output_dir(output_dir, config: Dict[str, Any]) -> Path:
    """Write the contract YAML under an already-resolved run ``output`` dir
    (``<output>/risk/risk_limits_config.yaml``). Used by the onboarding handoff,
    whose ``output_root`` is already at the run's output level."""
    path = Path(output_dir) / "risk" / "risk_limits_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def load_config(path) -> Optional[Dict[str, Any]]:
    """Read a contract YAML back. Returns None if absent or unreadable."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def emit_for_run(output_root, client_id: str, run_id: str, *,
                 search_roots: Optional[List[str]] = None,
                 extracted_at: str = "") -> Dict[str, Any]:
    """Build the config from the client's Schedule 8 doc and write it to the run
    path. Returns the config dict (with a ``written_to`` key). Best-effort: any
    write failure is reported in ``diagnostics`` rather than raised, so it never
    breaks the onboarding handoff."""
    config = build_config(client_id, search_roots=search_roots, extracted_at=extracted_at)
    try:
        path = write_config(output_root, client_id, run_id, config)
        config = {**config, "written_to": str(path)}
    except Exception as exc:  # noqa: BLE001
        config = {**config, "write_error": str(exc)}
    return config
