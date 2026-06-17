"""
gate5_adapter.py
================

Clean, **non-raising** adapter that lets the Delivery/XML Agent reuse only the
*safe* parts of the frozen Gate 4b normaliser and Gate 5 Annex 2 XML builder,
while reading the long projected target frame.

What it reuses (pure, deterministic, no I/O, no ``sys.exit``):

  * value predicates from Gate 5 — :func:`_is_nd`, :func:`_is_date`,
    :func:`_is_iso_year`, :func:`_split_path`, :func:`load_code_order`;
  * the regex / ND / precision validation *concepts* from Gate 4b, surfaced here
    as read-only checks (never as a mutation pass).

What it deliberately does **not** do (see ``docs/delivery_xml_agent_v1_review.md``):

  * call :func:`engine.gate_5_delivery.xml_builder_annex2.build_annex2_tree`
    (assumes a wide delivery CSV, one flat record per row, and injects ND5
    NoData defaults — i.e. silent fill that would override Projection decisions);
  * run :func:`engine.gate_4b_delivery.annex2_delivery_normalizer.normalize_delivery`
    (wide-frame cell mutation + hard-gate ``sys.exit(2)``).

The adapter never raises: if a frozen import or a config read fails, it degrades
to a conservative default so the Delivery Agent can still produce its readiness
report and refuse XML cleanly.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

__all__ = [
    "is_nd_value",
    "record_group_to_xml_group",
    "xsd_type_for_code",
    "format_valid",
    "enum_valid",
    "load_record_order",
    "field_universe_index",
]

# Reuse the frozen Gate 5 pure predicates where available; fall back to local
# copies so the adapter never hard-depends on lxml being importable.
try:  # pragma: no cover - import shim
    from engine.gate_5_delivery.xml_builder_annex2 import (  # noqa: F401
        _is_nd as _g5_is_nd,
    )
except Exception:  # pragma: no cover
    _g5_is_nd = None

_ND_RE = re.compile(r"^ND[1-5]$")

# RREL → underlying-exposure record, RREC → collateral record, else header/pool.
_XML_RECORD_GROUP = {
    "RREL": "underlying_exposure",
    "RREC": "collateral",
    "other": "header_pool_report",
}


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


def is_nd_value(value: Any) -> bool:
    """True if ``value`` is an ESMA No-Data sentinel ``ND1``..``ND5``."""
    s = _to_str(value).upper()
    if _g5_is_nd is not None:
        try:
            return bool(_g5_is_nd(s))
        except Exception:  # pragma: no cover
            pass
    return bool(_ND_RE.fullmatch(s))


def record_group_to_xml_group(record_group: str) -> str:
    """Map a Projection ``record_group`` (RREL/RREC/other) to its XML record group."""
    key = str(record_group).strip().upper()
    return _XML_RECORD_GROUP.get(key, _XML_RECORD_GROUP["other"])


def load_record_order(esma_code_order_path: str | Path) -> List[str]:
    """Read the Annex 2 ``Record:`` code list from ``esma_code_order.yaml``.

    Non-raising: returns ``[]`` if the file is missing or malformed.
    """
    try:
        data = yaml.safe_load(Path(esma_code_order_path).read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    rec = data.get("Record")
    if isinstance(rec, list):
        return [str(x).strip() for x in rec if str(x).strip()]
    return []


def field_universe_index(field_universe_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Index ``annex2_field_universe.yaml`` by code (non-raising)."""
    try:
        data = yaml.safe_load(Path(field_universe_path).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    fields = data.get("fields")
    if not isinstance(fields, dict):
        return {}
    return {str(k): (v or {}) for k, v in fields.items()}


def xsd_type_for_code(code: str, universe: Dict[str, Dict[str, Any]]) -> str:
    """Best-effort XSD-ish type token for a code, from the field universe ``format``.

    Returns ``""`` when unknown — v1 treats an unknown type as *deferred*, not as
    an error, and never invents one.
    """
    entry = universe.get(str(code)) or {}
    fmt = _to_str(entry.get("format"))
    return fmt


def format_valid(value: str, regime_rule: Optional[Dict[str, Any]]) -> bool:
    """Read-only XSD-ish format check.

    ND sentinels and blank values are not format-checked here (ND allowance and
    mandatory-blank are separate gates). A non-blank, non-ND value is valid if it
    matches the regime ``validators.regex`` (when present); absent a regex, v1 is
    permissive (true XSD shape is a deferred build-time concern).
    """
    v = _to_str(value)
    if v == "" or is_nd_value(v):
        return True
    rule = regime_rule or {}
    validators = rule.get("validators") if isinstance(rule.get("validators"), dict) else {}
    regex = _to_str(validators.get("regex"))
    if not regex:
        return True
    try:
        return bool(re.fullmatch(regex, v))
    except re.error:  # pragma: no cover - bad config pattern
        return True


def enum_valid(value: str, regime_rule: Optional[Dict[str, Any]]) -> bool:
    """Read-only enum-membership check against the regime ``transform.enum_map``.

    ND sentinels and blanks pass. A non-blank value passes if there is no enum_map
    (free text / numeric) or if it is a known key **or** an already-mapped target
    value. v1 never mutates the value — it only flags membership.
    """
    v = _to_str(value)
    if v == "" or is_nd_value(v):
        return True
    rule = regime_rule or {}
    transform = rule.get("transform") if isinstance(rule.get("transform"), dict) else {}
    enum_map = transform.get("enum_map") if isinstance(transform.get("enum_map"), dict) else {}
    if not enum_map:
        return True
    keys_lower = {str(k).lower() for k in enum_map}
    vals = {str(val) for val in enum_map.values()}
    return v in vals or v.lower() in keys_lower
