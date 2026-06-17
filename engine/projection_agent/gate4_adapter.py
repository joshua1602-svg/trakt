"""
gate4_adapter.py
================

Clean, **non-raising** adapter that lets the new agentic Projection Agent reuse
the *safe* parts of the frozen Gate 4 regime projector, while reading the new
authoritative regime contract.

What it reuses from the frozen Gate 4 (``engine.gate_4_projection.regime_projector``):

  * :func:`order_fields_by_template` — deterministic ESMA-code ordering.

These are pure, deterministic functions with no enum-agent / no raising
behaviour, so they are safe to reuse verbatim.

What it deliberately does **not** reuse (deferred — see
``docs/projection_agent_v1_review.md``):

  * ``project_to_regime`` — assumes a Gate 2 ``canonical_typed.csv`` and the
    ``fields_registry.regime_mapping`` field set, and can call the ``enum_agent``
    reviewer which *raises* on unreviewed enum candidates;
  * ``apply_annex2_post_projection_guards`` — header constants, ScrtstnIdr
    generation, RREC/RREL backfill (delivery / XML-record shaping concerns);
  * the Gate 4b delivery normaliser (precision / regex / boolean-XSD / preflight)
    and the Gate 5 XML builder.

Instead this adapter builds a **rich projection index** from the new authoritative
regime contract ``config/regime/annex2_delivery_rules.yaml::field_rules`` (the
same file Gate 4b consumes), keyed by ESMA code, carrying the
``projected_source_field`` (canonical name), ND/default eligibility and the
explicit, safe value transforms (``enum_map`` / ``geography_map``).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Reuse the frozen, deterministic Gate 4 ordering primitive (single source of truth).
from engine.gate_4_projection.regime_projector import order_fields_by_template

__all__ = [
    "load_regime_rules",
    "build_projection_index",
    "order_esma_codes",
    "record_group_for_code",
    "apply_safe_transform",
    "is_nd_value",
]


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

def load_regime_rules(regime_config_path: str | Path) -> Dict[str, Any]:
    """Load the Annex 2 delivery-rules YAML (the authoritative regime contract)."""
    p = Path(regime_config_path)
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def build_projection_index(regime_cfg: Optional[dict]) -> Dict[str, Dict[str, Any]]:
    """Index regime ``field_rules`` by ESMA code into a rich projection contract.

    Each entry carries everything the Projection Agent needs to project a single
    Annex 2 field **safely** — without inventing values:

        {
          "esma_code":              "RREC9",
          "canonical_field":        "property_type",   # projected_source_field
          "mandatory":              True,
          "enforce_presence":       True,
          "nd_allowed":             ["ND5"],
          "default_allowed":        False,
          "default_value":          "",
          "enum_map":               {...},             # explicit, safe value map
          "geography_map":          {...},
          "boolean":                "xsd_lowercase_true_false" | "",
          "regex":                  "...",
          "precision":              {...},             # informational only (delivery owns it)
          "deferred":               False,             # reconciliation_scope.deferred_fields
        }
    """
    cfg = regime_cfg or {}
    deferred = set(
        (cfg.get("reconciliation_scope", {}) or {}).get("deferred_fields", []) or []
    )
    out: Dict[str, Dict[str, Any]] = {}
    for code, rule in (cfg.get("field_rules", {}) or {}).items():
        rule = rule or {}
        transform = rule.get("transform") if isinstance(rule.get("transform"), dict) else {}
        validators = rule.get("validators") if isinstance(rule.get("validators"), dict) else {}
        out[str(code)] = {
            "esma_code": str(code),
            "canonical_field": rule.get("projected_source_field", "") or "",
            "mandatory": bool(rule.get("mandatory", False)),
            "enforce_presence": bool(rule.get("enforce_presence", False)),
            "nd_allowed": [str(x).upper() for x in (rule.get("nd_allowed") or [])],
            "default_allowed": bool(rule.get("default_allowed", False)),
            "default_value": "" if rule.get("default_value") is None else str(rule.get("default_value")),
            "enum_map": transform.get("enum_map") if isinstance(transform.get("enum_map"), dict) else {},
            "geography_map": transform.get("geography_map") if isinstance(transform.get("geography_map"), dict) else {},
            "boolean": str(transform.get("boolean", "")),
            "regex": str(validators.get("regex", "")),
            "precision": rule.get("precision") if isinstance(rule.get("precision"), dict) else {},
            "deferred": str(code) in deferred,
        }
    return out


# --------------------------------------------------------------------------- #
# ESMA-code ordering (reuse frozen Gate 4)
# --------------------------------------------------------------------------- #

def order_esma_codes(codes: List[str], template_order: List[str]) -> List[str]:
    """Order ESMA codes by the ``esma_code_order.yaml`` template list.

    Reuses the frozen Gate 4 :func:`order_fields_by_template`. Codes not present
    in the template are appended (stable) at the end, exactly as Gate 4 does.
    """
    fields_list: List[Tuple[str, Dict[str, Any]]] = [(c, {"code": c}) for c in codes]
    ordered = order_fields_by_template(fields_list, template_order or [])
    return [name for name, _ in ordered]


def load_record_order(esma_code_order_path: str | Path) -> List[str]:
    """Read the Annex 2 ``Record:`` code list from ``esma_code_order.yaml``.

    The Annex 2 order lives under the ``Record:`` key (not an ``ESMA_Annex2``
    key), so the frozen ``load_template_order`` returns nothing for it. We read
    the ``Record:`` list directly and hand it to the frozen ordering primitive.
    """
    try:
        data = yaml.safe_load(Path(esma_code_order_path).read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    rec = data.get("Record")
    if isinstance(rec, list):
        return [str(x).strip() for x in rec if str(x).strip()]
    return []


def record_group_for_code(esma_code: str) -> str:
    """Map an ESMA code to its Annex 2 record group.

    RREL* → underlying-exposure / loan-level record;
    RREC* → collateral-level record. Anything else → ``other``.
    """
    c = str(esma_code).upper()
    if c.startswith("RREL"):
        return "RREL"
    if c.startswith("RREC"):
        return "RREC"
    return "other"


# --------------------------------------------------------------------------- #
# Safe value transforms (projection-stage only — NOT delivery normalisation)
# --------------------------------------------------------------------------- #

def is_nd_value(value: Any) -> bool:
    s = _to_str(value).upper()
    return len(s) in (3, 4) and s.startswith("ND") and s[2:].isdigit()


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    if s.lower() in ("nan", "<na>"):
        return ""
    return s


def apply_safe_transform(value: str, rule: Dict[str, Any]) -> Tuple[str, str, bool]:
    """Apply ONLY explicit, safe projection transforms to a materialised value.

    Projection maps canonical enum values to their ESMA code via the rule's
    explicit ``enum_map`` / ``geography_map``. It does NOT do delivery-stage
    formatting (precision, regex, boolean XSD casing) — that is deferred.

    Returns ``(projected_value, value_type, mapped)`` where:
      * ``value_type`` is ``enum_code`` | ``geography_code`` | ``nd`` | ``raw``;
      * ``mapped`` is True if an explicit map was applied.

    An unmapped enum value is returned unchanged with ``mapped=False`` and
    ``value_type='unmapped_enum'`` so the caller can raise a projection issue —
    never a guess.
    """
    v = _to_str(value)
    if v == "":
        return "", "blank", False
    if is_nd_value(v):
        return v.upper(), "nd", False

    enum_map = rule.get("enum_map") or {}
    if enum_map:
        if v in enum_map:
            return str(enum_map[v]), "enum_code", True
        lower = {str(k).lower(): str(val) for k, val in enum_map.items()}
        if v.lower() in lower:
            return lower[v.lower()], "enum_code", True
        # Already a valid target code (value present on the right-hand side)?
        if v in {str(val) for val in enum_map.values()}:
            return v, "enum_code", True
        return v, "unmapped_enum", False

    geo_map = rule.get("geography_map") or {}
    if geo_map:
        if v in geo_map:
            return str(geo_map[v]), "geography_code", True
        lower = {str(k).lower(): str(val) for k, val in geo_map.items()}
        if v.lower() in lower:
            return lower[v.lower()], "geography_code", True
        # Not a known legacy label → already a valid geography value (NUTS code /
        # year). Pass through unchanged; XSD shape is a delivery concern.
        return v, "geography_code", False

    return v, "raw", False
