"""The effective ESMA Annex 2 contract, derived from the sources that own it.

Why this exists
---------------
Annex 2 truth used to be written down twice: once in the authoritative sources,
and once in a hand-maintained delivery-rules file that five pipeline stages read
in preference to them. Where the two disagreed the hand-written copy won, so a
field could be optional at Gate 4b and mandatory at Gate 5, and a canonical field
could report as one ESMA code in validation and another in delivery.

This module builds the contract once, from the layer that owns each fact:

===================  =====================================================
UNIVERSE             ``config/regime/annex2_field_universe.yaml`` — the 107
                     codes, their names, and the ND1-4 / ND5 envelope the
                     regulator permits.
REGISTRY             ``config/system/fields_registry.yaml`` — which canonical
                     field reports as which code.
WORKBOOK             the mapping workbook — XML path, multiplicity (therefore
                     mandatory), and whether a code has an element at all.
XSD                  ``auth.099`` — the value type, its enumeration, pattern and
                     digit facets. The final arbiter of validity.
ENUM CONFIG          ``enum_mapping.yaml`` / ``enum_synonyms.yaml`` — lender
                     words that mean a regulator's code.
ASSET                the product pack — enum overrides that are true of the
                     asset class.
OPERATOR             approved decisions — translations this portfolio's operator
                     confirmed.
===================  =====================================================

What this module deliberately does NOT own
------------------------------------------
Values. A default, a no-data code or a pool constant is a decision belonging to
the product pack, the client configuration or an approved operator decision, and
all three already reach the projector through the effective client configuration.
The contract says what the regulator will accept; the layered configuration says
what this book reports. Keeping those apart is the whole point — a contract that
also carried defaults would be the legacy file again under a new name.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field as dc_field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml
from lxml import etree

from . import workbook_index as wb

_REPO = Path(__file__).resolve().parents[2]
_log = logging.getLogger(__name__)

UNIVERSE_PATH = _REPO / "config" / "regime" / "annex2_field_universe.yaml"
REGISTRY_PATH = _REPO / "config" / "system" / "fields_registry.yaml"
ENUM_MAPPING_PATH = _REPO / "config" / "system" / "enum_mapping.yaml"
ENUM_SYNONYMS_PATH = _REPO / "config" / "system" / "enum_synonyms.yaml"
ASSET_PACK_PATH = _REPO / "config" / "asset" / "product_defaults_ERM.yaml"
XSD_PATH = _REPO / "config" / "system" / "DRAFT1auth.099.001.04_1.3.0.xsd"

REGIME = "ESMA_Annex2"

#: Provenance labels. Every property records which layer decided it.
UNIVERSE = "UNIVERSE"
REGISTRY = "REGISTRY"
WORKBOOK = "WORKBOOK"
XSD = "XSD"
ASSET = "ASSET"
CLIENT = "CLIENT"
OPERATOR = "OPERATOR"
DERIVATION = "DERIVATION"
ENUM_CONFIG = "ENUM_CONFIG"

_XS = "{http://www.w3.org/2001/XMLSchema}"

#: What the XSD's own base types mean, for the codes whose simple type declares
#: no explicit pattern. Every entry is the lexical space the schema defines for
#: that type — nothing here is a house rule.
_BASE_TYPE_PATTERNS = {
    "xs:date": r"^\d{4}-\d{2}-\d{2}$",
    "xs:gYear": r"^-?\d{4}$",
    "xs:decimal": r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$",
    "xs:integer": r"^[+-]?\d+$",
}

#: Choice branches that carry the value rather than the no-data justification.
_VALUE_TAGS = ("Cd", "Val", "Rate", "Amt", "Idr", "Dt", "Nb", "Ind", "Yr",
               "Prtry", "Ctry", "LEI", "Nm", "Txt")


# --------------------------------------------------------------------------- #
# XSD: the schema's own view of a value
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _schema() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    root = etree.parse(str(XSD_PATH)).getroot()

    def facets(st) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        r = st.find(f"{_XS}restriction")
        if r is None:
            return out
        out["base"] = r.get("base")
        for f in r:
            tag = etree.QName(f).localname
            if tag == "enumeration":
                out.setdefault("enum", []).append(f.get("value"))
            elif tag in ("pattern", "minLength", "maxLength", "totalDigits",
                         "fractionDigits", "minInclusive"):
                out[tag] = f.get("value")
        return out

    simple = {st.get("name"): facets(st)
              for st in root.findall(f"{_XS}simpleType") if st.get("name")}
    complex_ = {ct.get("name"): ct
                for ct in root.findall(f"{_XS}complexType") if ct.get("name")}
    top = {e.get("name"): e
           for e in root.findall(f"{_XS}element") if e.get("name")}
    return simple, complex_, top


def _children(type_name: str) -> Dict[str, str]:
    _, complex_, _ = _schema()
    ct = complex_.get(type_name)
    if ct is None:
        return {}
    out: Dict[str, str] = {}
    for e in ct.iter(f"{_XS}element"):
        n, t = e.get("name"), e.get("type")
        if n and t:
            out.setdefault(n, t.split(":")[-1])
    return out


def _simple_content_base(type_name: str) -> Optional[str]:
    _, complex_, _ = _schema()
    ct = complex_.get(type_name)
    if ct is None:
        return None
    ext = ct.find(f"{_XS}simpleContent/{_XS}extension")
    return ext.get("base").split(":")[-1] if ext is not None and ext.get("base") else None


def _unwrap(type_name: Optional[str], depth: int = 0
            ) -> Tuple[Optional[str], Dict[str, Any]]:
    """Follow a choice or wrapper type down to the value's simple type."""
    simple, _, _ = _schema()
    if not type_name or depth > 8:
        return type_name, {}
    if type_name in simple:
        return type_name, simple[type_name]
    base = _simple_content_base(type_name)
    if base:
        return _unwrap(base, depth + 1)
    kids = _children(type_name)
    if not kids:
        return type_name, {}
    ordered = ([(n, t) for n, t in kids.items() if n in _VALUE_TAGS]
               + [(n, t) for n, t in kids.items()
                  if n not in _VALUE_TAGS and "NoData" not in n and "NoData" not in t])
    for _, t in ordered:
        resolved, f = _unwrap(t, depth + 1)
        if f:
            return resolved, f
    return (ordered[0][1] if ordered else type_name), {}


def _resolve_path(path: str) -> Optional[str]:
    _, _, top = _schema()
    parts = [p for p in path.strip("/").split("/") if p]
    if not parts:
        return None
    cur = top.get(parts[0])
    if cur is None:
        return None
    tname = (cur.get("type") or "").split(":")[-1]
    for p in parts[1:]:
        kids = _children(tname)
        if p not in kids:
            return None
        tname = kids[p]
    return tname


@lru_cache(maxsize=256)
def xsd_value_facets(code: str) -> Tuple[Optional[str], Tuple[Tuple[str, Any], ...]]:
    """``(type name, facets)`` for the value branch of ``code``'s XML element."""
    path = wb.value_path(code)
    if not path:
        return None, ()
    tname = _resolve_path(path)
    if not tname:
        return None, ()
    tname, facets = _unwrap(tname)
    return tname, tuple(sorted(
        (k, tuple(v) if isinstance(v, list) else v) for k, v in facets.items()))


def _facet_dict(code: str) -> Tuple[Optional[str], Dict[str, Any]]:
    tname, pairs = xsd_value_facets(code)
    return tname, {k: (list(v) if isinstance(v, tuple) else v) for k, v in pairs}


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FieldContract:
    """What the regulator will accept for one Annex 2 code."""

    esma_code: str
    field_name: str = ""
    canonical_field: str = ""
    mandatory: bool = False
    enforce_presence: bool = False
    nd_allowed: Tuple[str, ...] = ()
    enum_values: Tuple[str, ...] = ()
    enum_map: Mapping[str, str] = dc_field(default_factory=dict)
    boolean: bool = False
    pattern: str = ""
    precision: Mapping[str, int] = dc_field(default_factory=dict)
    emitting: bool = True
    xsd_type: str = ""
    workbook_semantic: str = ""
    derive: Mapping[str, Any] = dc_field(default_factory=dict)
    provenance: Mapping[str, str] = dc_field(default_factory=dict)

    def to_rule(self) -> Dict[str, Any]:
        """The shape the Gate 4b normaliser already consumes.

        Values are deliberately absent: a default belongs to the product pack,
        the client configuration or an approved operator decision, all of which
        reach the frame through the projector before Gate 4b sees it.
        """
        rule: Dict[str, Any] = {
            "esma_code": self.esma_code,
            "mandatory": self.mandatory,
            "enforce_presence": self.enforce_presence,
            "nd_allowed": list(self.nd_allowed),
        }
        if self.workbook_semantic:
            rule["workbook_semantic"] = self.workbook_semantic
        if self.canonical_field:
            rule["projected_source_field"] = self.canonical_field
        transform: Dict[str, Any] = {}
        if self.enum_map:
            transform["enum_map"] = dict(self.enum_map)
        if self.boolean:
            transform["boolean"] = "xsd_lowercase_true_false"
        if transform:
            rule["transform"] = transform
        if self.pattern:
            rule["validators"] = {"regex": self.pattern}
        if self.precision:
            rule["precision"] = dict(self.precision)
        if self.derive:
            rule["derive"] = dict(self.derive)
        return rule


@dataclass(frozen=True)
class Annex2Contract:
    fields: Mapping[str, FieldContract]
    performance_mode: Optional[str] = None
    sources: Tuple[str, ...] = ()

    def __getitem__(self, code: str) -> FieldContract:
        return self.fields[code]

    def get(self, code: str) -> Optional[FieldContract]:
        return self.fields.get(code)

    def codes(self) -> List[str]:
        return sorted(self.fields, key=_code_sort_key)

    def by_canonical(self) -> Dict[str, FieldContract]:
        out: Dict[str, FieldContract] = {}
        for fc in self.fields.values():
            if fc.canonical_field:
                out.setdefault(fc.canonical_field, fc)
        return out

    def mandatory_codes(self) -> List[str]:
        return [c for c in self.codes() if self.fields[c].mandatory]

    def emitting_codes(self) -> List[str]:
        return [c for c in self.codes() if self.fields[c].emitting]

    def non_emitting_codes(self) -> List[str]:
        """Codes the schema carries as an attribute rather than an element."""
        return [c for c in self.codes() if not self.fields[c].emitting]


def _code_sort_key(code: str) -> Tuple[str, int]:
    return code[:4], int(re.sub(r"\D", "", code) or 0)


# --------------------------------------------------------------------------- #
# Authoritative source readers
# --------------------------------------------------------------------------- #

def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}


@lru_cache(maxsize=1)
def universe() -> Dict[str, Dict[str, Any]]:
    return _read_yaml(UNIVERSE_PATH).get("fields") or {}


@lru_cache(maxsize=1)
def _registry_index() -> Tuple[Dict[str, str], Dict[str, str]]:
    """``(code -> canonical, code -> priority)`` from the fields registry."""
    fields = _read_yaml(REGISTRY_PATH).get("fields") or {}
    canon: Dict[str, str] = {}
    prio: Dict[str, str] = {}
    for name, spec in fields.items():
        rm = ((spec or {}).get("regime_mapping") or {}).get(REGIME) or {}
        code = rm.get("code")
        if code:
            canon.setdefault(str(code), str(name))
            prio.setdefault(str(code), str(rm.get("priority", "")))
    return canon, prio


def canonical_for(code: str) -> str:
    return _registry_index()[0].get(code, "")


def nd_envelope(code: str) -> Tuple[str, ...]:
    """The no-data codes the regulator permits for ``code``."""
    u = universe().get(code) or {}
    nd: List[str] = []
    if u.get("nd1_4_allowed"):
        nd += ["ND1", "ND2", "ND3", "ND4"]
    if u.get("nd5_allowed"):
        nd += ["ND5"]
    return tuple(nd)


@lru_cache(maxsize=1)
def _enum_config() -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    generic = (_read_yaml(ENUM_MAPPING_PATH).get(REGIME) or {})
    syn_doc = _read_yaml(ENUM_SYNONYMS_PATH) or {}
    synonyms = {k: (v or {}).get("manual") or {} for k, v in syn_doc.items()
                if isinstance(v, dict)}
    return generic, synonyms


def _asset_enum_overrides(asset_pack: Optional[Mapping[str, Any]] = None
                          ) -> Dict[str, Dict[str, str]]:
    """Asset-class enum overrides — only when an asset pack is actually given.

    An override says "for THIS product, that word means this code", which is a
    statement about an asset class and not about the regime. Reading a pack by
    default would put one product's judgement into the generic contract, which
    is how "Bullet" came to mean OTHR for every portfolio in the estate rather
    than for equity release alone. The projector applies the pack's overrides
    where a pack is in play, so nothing is lost by keeping the regime contract
    product-neutral.
    """
    if asset_pack is None:
        return {}
    return ((asset_pack.get("reporting_policy") or {}).get("enum_overrides") or {})


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def _derivations() -> Dict[str, Dict[str, Any]]:
    """Regulatory-output derivations that only make sense on the Annex frame.

    A value calculable from two other Annex codes, where the canonical model has
    no equivalent single field. Kept here rather than in the canonical derivation
    layer because the inputs are ESMA codes, not canonical columns.
    """
    return {
        # Original term is the number of months between origination and maturity.
        # Stated by the workbook itself ("number of months ... at origination"),
        # so it is a regime derivation, not a lender's choice. A supplied value
        # always wins: the normaliser only derives into a blank.
        "RREL25": {"type": "months_between_dates",
                   "start_field": "RREL23", "end_field": "RREL24"},
    }


def build_contract(*,
                   performance_mode: Optional[str] = None,
                   asset_pack: Optional[Mapping[str, Any]] = None,
                   operator_translations: Optional[Mapping[str, Mapping[str, str]]] = None,
                   ) -> Annex2Contract:
    """Materialise the effective Annex 2 contract.

    ``operator_translations`` is ``{canonical field: {lender value: ESMA code}}``
    from approved decisions; ``asset_pack`` is the product layer. Both only ever
    ADD accepted spellings — neither can widen the regulator's own vocabulary,
    because a translation whose target is not in the schema enumeration is
    dropped.
    """
    generic_enums, synonym_enums = _enum_config()
    asset_enums = _asset_enum_overrides(asset_pack)
    operator_translations = operator_translations or {}
    derivations = _derivations()

    fields: Dict[str, FieldContract] = {}
    for code in universe():
        u = universe().get(code) or {}
        canonical = canonical_for(code)
        prov: Dict[str, str] = {"field_name": UNIVERSE, "nd_allowed": UNIVERSE}

        mandatory = wb.is_mandatory(code, performance_mode)
        # A code is emitted when the schema actually defines an element at the
        # path the workbook gives it. The three that do not resolve are the
        # currency concepts the XSD carries as a Ccy attribute — derived here,
        # never declared.
        path = wb.value_path(code)
        emitting = bool(path) and _resolve_path(path) is not None
        prov["mandatory"] = WORKBOOK
        prov["emitting"] = XSD
        if canonical:
            prov["canonical_field"] = REGISTRY

        tname, facets = _facet_dict(code)
        enum_values = tuple(facets.get("enum") or ())
        enum_map: Dict[str, str] = {}
        if enum_values:
            # The regulator's own vocabulary is always accepted verbatim.
            enum_map = {v: v for v in enum_values}
            prov["enum_values"] = XSD
            for source, label in ((generic_enums.get(canonical), ENUM_CONFIG),
                                  (synonym_enums.get(canonical), ENUM_CONFIG),
                                  (asset_enums.get(canonical), ASSET),
                                  (operator_translations.get(canonical), OPERATOR)):
                for lender_value, target in (source or {}).items():
                    target = str(target).strip()
                    if target in enum_values:
                        enum_map[str(lender_value)] = target
                        prov["enum_map"] = label

        precision: Dict[str, int] = {}
        if facets.get("totalDigits"):
            precision = {"total_digits": int(facets["totalDigits"]),
                         "fraction_digits": int(facets.get("fractionDigits") or 0)}
            prov["precision"] = XSD

        pattern = ""
        if facets.get("pattern"):
            pattern = "^(?:%s)$" % facets["pattern"]
            prov["pattern"] = XSD
        elif facets.get("base") in _BASE_TYPE_PATTERNS:
            # The schema states no explicit pattern, but the base type IS a
            # constraint: an xs:date is not a region name and an xs:gYear is not
            # a number of months. Enforcing the base type at Gate 4b is what
            # stops a value Gate 5 would refuse reaching Gate 5 at all — the
            # whole point of the two stages agreeing. Still the schema's answer,
            # not a hand-written narrowing.
            pattern = _BASE_TYPE_PATTERNS[facets["base"]]
            prov["pattern"] = XSD

        boolean = facets.get("base") == "xs:boolean"
        if boolean:
            prov["boolean"] = XSD

        derive = derivations.get(code) or {}
        if derive:
            prov["derive"] = DERIVATION

        semantic = "/".join(path.rstrip("/").split("/")[-2:]) if path else ""

        fields[code] = FieldContract(
            esma_code=code,
            field_name=str(u.get("field_name", "")),
            canonical_field=canonical,
            mandatory=mandatory,
            # The BUILDER tolerates an absent column and refuses a blank one:
            # a code missing from the frame is skipped, a code present and empty
            # is refused. Gate 4b must draw the line in the same place, or it
            # blocks a delivery the schema accepts. Presence is therefore not
            # enforced; a value, where the column exists, is.
            enforce_presence=False,
            nd_allowed=nd_envelope(code),
            enum_values=enum_values,
            enum_map=enum_map,
            boolean=boolean,
            pattern=pattern,
            precision=precision,
            emitting=emitting,
            xsd_type=str(tname or ""),
            workbook_semantic=semantic,
            derive=derive,
            provenance=prov,
        )

    return Annex2Contract(
        fields=fields,
        performance_mode=performance_mode,
        sources=(
            "config/regime/annex2_field_universe.yaml",
            "config/system/fields_registry.yaml",
            wb.WORKBOOK.name,
            "config/system/DRAFT1auth.099.001.04_1.3.0.xsd",
            "config/system/enum_mapping.yaml",
            "config/system/enum_synonyms.yaml",
            "config/asset/product_defaults_ERM.yaml",
        ),
    )


@lru_cache(maxsize=4)
def _default_contract(performance_mode: Optional[str]) -> Annex2Contract:
    return build_contract(performance_mode=performance_mode)


def contract(performance_mode: Optional[str] = None) -> Annex2Contract:
    """The contract with no client or operator layer — cached."""
    return _default_contract(performance_mode)


def as_delivery_rules(c: Optional[Annex2Contract] = None) -> Dict[str, Any]:
    """The contract in the document shape the Gate 4b normaliser reads."""
    c = c or contract()
    return {
        "regime": REGIME,
        "generated_from": list(c.sources),
        "generated_by": "engine.regime_contract.annex2_contract",
        "note": ("Effective Annex 2 contract, derived from the authoritative "
                 "sources. Not hand-maintained and not a source of truth: edit "
                 "the layer that owns the fact instead."),
        "field_rules": {code: c.fields[code].to_rule() for code in c.codes()},
    }


def materialise_delivery_rules(out_path: Path, *,
                               operator_translations: Optional[Mapping[str, Mapping[str, str]]] = None,
                               asset_pack: Optional[Mapping[str, Any]] = None,
                               performance_mode: Optional[str] = None) -> Path:
    """Write the effective contract for one run, including approved decisions."""
    c = (contract(performance_mode)
         if not (operator_translations or asset_pack)
         else build_contract(performance_mode=performance_mode,
                             asset_pack=asset_pack,
                             operator_translations=operator_translations))
    doc = as_delivery_rules(c)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "# GENERATED — effective Annex 2 contract. Do not edit; do not commit as\n"
        "# a source of truth. Regenerated from the authoritative sources on every\n"
        "# run by engine.regime_contract.annex2_contract.\n"
        + yaml.safe_dump(doc, sort_keys=False, width=120), encoding="utf-8")
    return out_path


@lru_cache(maxsize=1)
def materialised_contract_path() -> str:
    """A path to the derived contract, for callers that can only take a path.

    Some components take the regime contract as a file name rather than a
    document. Rather than keep a file in the repository for them to read — which
    is exactly the second source of truth this replaced — the contract is written
    once per process to a temporary file and the path handed over. It is derived
    on every process start, so it can never drift.
    """
    import tempfile
    out = Path(tempfile.gettempdir()) / f"trakt_annex2_contract_{os.getpid()}.yaml"
    if not out.exists():
        materialise_delivery_rules(out)
    return str(out)
