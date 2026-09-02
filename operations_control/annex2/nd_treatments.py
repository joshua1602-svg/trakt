"""operations_control.annex2.nd_treatments — the governed answers to
"the lender does not supply this regulatory field".

ESMA's no-data codes are the regulator's own vocabulary for that situation, and
which of them a field may carry is already stated per ESMA code in the
workbook-derived field universe. Their meanings are already stated in
``config/system/standards_library.yaml``. Nothing here decides anything: it
reads the effective Annex 2 contract and offers the operator exactly the
treatments the REGULATOR permits for the field in front of them — not the
treatments someone once wrote a rule for.

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
#: The effective Annex 2 contract, derived from the authoritative sources.
#: Nothing here reads a delivery-rules file: the no-data envelope belongs to the
#: workbook-derived field universe, and the code a canonical field reports as
#: belongs to the fields registry.
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
def _contract():
    """The effective Annex 2 contract (cached by the contract module)."""
    from engine.regime_contract.annex2_contract import contract
    return contract()


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
    """The no-data codes the REGULATOR permits for ``field``.

    Read from the workbook-derived field universe by way of the effective
    contract, so a code the regulator allows ND for is offered whether or not
    anyone ever hand-wrote a rule for it. ND4 carries a date
    (``ND4-YYYY-MM-DD``) and is not offered as a bare choice.
    """
    code = esma_code_for(field)
    if not code:
        return []
    fc = _contract().get(code)
    if fc is None:
        return []
    known = _nd_options()
    return [c for c in fc.nd_allowed if c in known and not _ND4.match(c)]


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
        if not field:
            continue
        # A field the population reconciliation could not fill says so plainly;
        # a projector failure reports it as a NULL placeholder among its values.
        if str((ev or {}).get("label", "")) == "Regulatory field with no value":
            out.append(field)
            continue
        raw = str(data.get("values") or "")
        values = [v.strip().strip("'\"").lower() for v in raw.split(",")]
        if values and all(v in absent for v in values):
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
    """The regulator's own code list for ``field``, from the XSD enumeration."""
    code = esma_code_for(field)
    if not code:
        return []
    fc = _contract().get(code)
    return sorted(fc.enum_values) if fc else []


def is_permitted_enum(field: str, code: str) -> bool:
    return str(code or "").strip() in permitted_enum_codes(field)


def materialise_effective_delivery_rules(*, out_path: Path,
                                         translations: Dict[str, Dict[str, str]],
                                         base_rules_path: Optional[Path] = None,
                                         ) -> Path:
    """The effective Annex 2 contract for this run, including approved decisions.

    ``translations`` is ``{canonical field: {lender value: regulator code}}`` from
    approved decisions. Each pair is merged into the field's enum vocabulary,
    which the contract takes from the XSD enumeration — so an operator can teach
    the delivery what one of the lender's words means, and cannot invent a code
    the regulator does not define. The canonical keeps the lender's own wording.

    ``base_rules_path`` is accepted for call compatibility and ignored: the
    contract has no base file to start from.
    """
    from engine.regime_contract.annex2_contract import materialise_delivery_rules
    return materialise_delivery_rules(out_path, operator_translations=translations)


def client_regulatory_values(defaults: Dict[str, Any]) -> Dict[str, str]:
    """Client-configured values that are themselves Annex 2 regulatory targets.

    A client configuration states facts about the lender — who originated the
    book, its LEI, where it is established. Those are canonical fields with an
    Annex 2 code, so a value stated there IS the regulator's answer, and it
    reaches the return the same way any other configured regulatory value does.

    Only keys the fields registry maps to an Annex 2 code qualify, so nothing is
    listed here and a configuration key that means something else to another
    consumer is left alone.
    """
    out: Dict[str, str] = {}
    codes = _code_for_field()
    for key, value in (defaults or {}).items():
        if not isinstance(value, (str, int, float)):
            continue
        text = str(value).strip()
        if not text or not codes.get(str(key)):
            continue
        out[str(key)] = text
    return out
