"""operations_control.annex2.nd_treatments — the governed answers to
"the lender does not supply this regulatory field".

ESMA's no-data codes are the regulator's own vocabulary for that situation, and
which of them a field may carry is already stated per ESMA code in
``config/regime/annex2_delivery_rules.yaml``. Their meanings are already stated
in ``config/system/standards_library.yaml``. Nothing here decides anything: it
reads both, and offers the operator exactly the treatments the rules permit for
the field in front of them.

A no-data code is a REGULATORY statement about why a field is empty. It belongs
to the projection and never to the canonical — the projector applies it from
``defaults.nd_defaults`` while building the return, and the management canonical
keeps saying, truthfully, that the lender does not supply the field.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_REPO = Path(__file__).resolve().parents[2]
_REGIME_RULES = _REPO / "config" / "regime" / "annex2_delivery_rules.yaml"
_STANDARDS = _REPO / "config" / "system" / "standards_library.yaml"
_REGISTRY = _REPO / "config" / "system" / "fields_registry.yaml"

#: ND4 carries the date the data becomes available, so it cannot be offered as a
#: bare choice — it needs a date the operator has not been asked for. Offering it
#: would produce a code the schema rejects.
_ND4 = re.compile(r"^ND4$", re.IGNORECASE)


@lru_cache(maxsize=1)
def _nd_options() -> Dict[str, Dict[str, Any]]:
    try:
        doc = yaml.safe_load(_STANDARDS.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}
    return {str(k): (v or {}) for k, v in (doc.get("ND_OPTIONS") or {}).items()}


@lru_cache(maxsize=1)
def _rules_by_code() -> Dict[str, Dict[str, Any]]:
    try:
        doc = yaml.safe_load(_REGIME_RULES.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}
    return {str(k): (v or {}) for k, v in (doc.get("field_rules") or {}).items()}


@lru_cache(maxsize=1)
def _code_for_field() -> Dict[str, str]:
    """``{canonical field: ESMA Annex 2 code}`` from the registry."""
    out: Dict[str, str] = {}
    try:
        fields = (yaml.safe_load(_REGISTRY.read_text(encoding="utf-8")) or {}
                  ).get("fields", {}) or {}
    except Exception:  # noqa: BLE001
        return out
    for name, spec in fields.items():
        code = (((spec or {}).get("regime_mapping") or {})
                .get("ESMA_Annex2") or {}).get("code")
        if code:
            out[str(name)] = str(code)
    return out


def esma_code_for(field: str) -> str:
    """The Annex 2 code this canonical field reports as, or ``""``."""
    return _code_for_field().get(str(field or "").strip(), "")


@lru_cache(maxsize=1)
def _field_for_code() -> Dict[str, str]:
    return {code: field for field, code in _code_for_field().items()}


def field_for_esma_code(code: str) -> str:
    """The canonical field an Annex 2 code reports, or ``""``."""
    return _field_for_code().get(str(code or "").strip(), "")


def permitted_nd_codes(field: str) -> List[str]:
    """The no-data codes the regime rules allow for ``field``.

    Empty when the field has no Annex 2 code, when the rules name none, or when
    the only ones named cannot be offered as a plain choice.
    """
    code = esma_code_for(field)
    if not code:
        return []
    allowed = (_rules_by_code().get(code) or {}).get("nd_allowed") or []
    known = _nd_options()
    return [str(c).strip().upper() for c in allowed
            if str(c).strip().upper() in known
            and not _ND4.match(str(c).strip())]


def describe(nd_code: str) -> str:
    """The regulator's own words for a no-data code."""
    meta = _nd_options().get(str(nd_code).strip().upper()) or {}
    return str(meta.get("description", "") or "")


def treatment_options(field: str) -> List[Dict[str, str]]:
    """Operator-facing choices for ``field``: one per permitted no-data code."""
    return [{"value": code, "label": f"{code} — {describe(code)}"}
            for code in permitted_nd_codes(field)]


def is_permitted(field: str, nd_code: str) -> bool:
    """Is ``nd_code`` a treatment the rules allow for ``field``?"""
    return str(nd_code or "").strip().upper() in permitted_nd_codes(field)


#: Rule setting prefix. One setting per field, so an approval is a standing
#: statement about that field for this client/portfolio rather than a per-run one.
SETTING_PREFIX = "nd_default:"


def setting_for(field: str) -> str:
    return f"{SETTING_PREFIX}{field}"


def field_from_setting(setting: str) -> str:
    s = str(setting or "")
    return s[len(SETTING_PREFIX):] if s.startswith(SETTING_PREFIX) else ""


#: Rule setting prefix for a pool-level regulatory constant.
CONSTANT_PREFIX = "regulatory_constant:"


def constant_setting_for(field: str) -> str:
    return f"{CONSTANT_PREFIX}{field}"


def field_from_constant_setting(setting: str) -> str:
    s = str(setting or "")
    return s[len(CONSTANT_PREFIX):] if s.startswith(CONSTANT_PREFIX) else ""


def regulatory_constants_from_settings(settings: Dict[str, Any]) -> Dict[str, str]:
    """``{field: value}`` from approved pool-level regulatory constants."""
    out: Dict[str, str] = {}
    for setting, value in (settings or {}).items():
        field = field_from_constant_setting(setting)
        if field and str(value).strip():
            out[field] = str(value).strip()
    return out


def nd_defaults_from_settings(settings: Dict[str, Any]) -> Dict[str, str]:
    """``{field: ND code}`` from approved client-rule settings.

    Only codes the rules still permit survive: a rule approved before the regime
    rules changed must not quietly keep applying a treatment the regulator no
    longer allows for that field.
    """
    out: Dict[str, str] = {}
    for setting, value in (settings or {}).items():
        field = field_from_setting(setting)
        if field and is_permitted(field, str(value)):
            out[field] = str(value).strip().upper()
    return out


def absent_regulatory_fields(evidence: List[Dict[str, Any]]) -> List[str]:
    """Fields a projection failure reports as unmapped because they are EMPTY.

    Distinguished from a field whose values are wrong: that is a translation
    question with a different answer. Absent values are reported by the
    projector as the ``NULL`` placeholder.
    """
    absent = {"null", "none", "nan", "na", "<na>", "nat", "-", ""}
    out: List[str] = []
    for ev in evidence or []:
        data = (ev or {}).get("data")
        if not isinstance(data, dict):
            continue
        field = str(data.get("field") or "").strip()
        raw = str(data.get("values") or "")
        values = [v.strip().strip("'\"").lower() for v in raw.split(",")]
        if field and values and all(v in absent for v in values):
            out.append(field)
    return out


# --------------------------------------------------------------------------- #
# Approved regulatory enum translations
# --------------------------------------------------------------------------- #

#: Marks a rule as belonging to the REGULATORY boundary. A translation carrying
#: it changes what the regulator is told and nothing about the canonical, so it
#: must never be projected into client mapping memory, which Gate 1 applies to
#: the management data.
LAYER_REGULATORY = "regulatory"


def permitted_enum_codes(field: str) -> List[str]:
    """The codes the regime rules accept for ``field``, from its enum map."""
    code = esma_code_for(field)
    if not code:
        return []
    transform = (_rules_by_code().get(code) or {}).get("transform") or {}
    enum_map = transform.get("enum_map")
    if not isinstance(enum_map, dict):
        return []
    return sorted({str(v).strip() for v in enum_map.values() if str(v).strip()})


def is_permitted_enum(field: str, code: str) -> bool:
    return str(code or "").strip() in permitted_enum_codes(field)


def materialise_effective_delivery_rules(*, base_rules_path: Path,
                                         translations: Dict[str, Dict[str, str]],
                                         out_path: Path) -> Path:
    """Repository delivery rules + approved per-client enum translations.

    ``translations`` is ``{canonical field: {lender value: regulator code}}``.
    Each pair is merged into that field's existing ``transform.enum_map``, which
    is the table the delivery normaliser already consults, so an approved
    translation is applied while the return is built and the canonical keeps the
    lender's own wording.
    """
    doc = yaml.safe_load(Path(base_rules_path).read_text(encoding="utf-8")) or {}
    rules = doc.setdefault("field_rules", {})
    for field, pairs in (translations or {}).items():
        code = esma_code_for(field)
        if not code or code not in rules:
            continue
        transform = (rules[code].setdefault("transform", {}) or {})
        enum_map = dict(transform.get("enum_map") or {})
        if not enum_map:
            continue          # the field is not enum-mapped; nothing to extend
        for lender_value, regulator_code in (pairs or {}).items():
            if str(regulator_code).strip() in set(enum_map.values()):
                enum_map[str(lender_value)] = str(regulator_code).strip()
        transform["enum_map"] = enum_map
        rules[code]["transform"] = transform
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "# GENERATED by operations_control — effective Annex 2 delivery rules\n"
        "# (repository rules + approved regulatory enum translations).\n"
        + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return out_path
