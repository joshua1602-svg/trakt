"""
xsd_structured_preview_builder.py
=================================

Builds a NON-PRODUCTION, XSD-*structured* Annex 2 XML preview by placing values
inside the real ESMA hierarchy using the **builder-accepted** field-to-XSD paths
from ``config/delivery/annex2_field_xsd_path_map.yaml``:

    Document
      -> ScrtstnNonAsstBckdComrclPprUndrlygXpsrRpt -> NewCrrctn -> ScrtstnRpt
        -> ScrtstnIdr / CutOffDt                         (report header)
        -> UndrlygXpsrRcrd (one per loan)
          -> UndrlygXpsrData -> ResdtlRealEsttLn -> PrfrmgLn
            -> UndrlygXpsrCmonData ...                    (loan / RREL fields)
            -> Coll ...                                   (collateral / RREC, nested)

This is the opposite of the existing FLAT preview: it proves nested ESMA-path
construction. It is **not** production XML and never claims to be:

  * only paths with ``builder_acceptance_status`` in {sample_confirmed,
    accepted_for_builder} are used; rejected / needs_manual_review / unresolved /
    conflict paths are never used;
  * RREC collateral fields stay nested under the loan's ``Coll`` node;
  * NoDataOptn wrappers are emitted only where the path map says
    ``value_mode = value_or_nodata`` AND the value is a genuine ND sentinel;
  * no silent ND/default injection, no value fabrication (legacy Gate 5 runtime
    is NOT reused);
  * watermark + metadata are XML comments (so they never pollute XSD validation);
  * XSD validation is attempted honestly and its real result is reported — it is
    expected to FAIL today (incomplete mandatory content, leaf value-typing and
    strict sequence ordering are not yet modelled).

It writes only under the caller-supplied preview output directory.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

NS = "urn:esma:xsd:DRAFT1auth.099.001.04"
XSI = "http://www.w3.org/2001/XMLSchema-instance"
RECORD_ANCHOR = "UndrlygXpsrRcrd"
_REPORT_LEAF = "ScrtstnRpt"

_ND_RE = re.compile(r"^ND[1-5]$")
_XS = "{http://www.w3.org/2001/XMLSchema}"
# generic XSD value leaves (a choice/typed wrapper's value child).
_VALUE_TAGS = {"Cd", "Amt", "Dt", "Rate", "Nb", "Ind", "Val", "Yr", "LEI",
               "Pctg", "Lien", "NbOfMnths", "NbOfDays", "Max", "Sgn"}

ACCEPTED_FOR_BUILDER = {"sample_confirmed", "accepted_for_builder"}


def is_nd(value: Any) -> bool:
    return bool(_ND_RE.fullmatch(str(value).strip().upper()))


# --------------------------------------------------------------------------- #
# Minimal XSD resolver: given a full element path, find the value-leaf child
# (e.g. PrprtyTp -> Cd, ValtnAmt -> Val) so values nest one level deeper, the way
# the real schema expects. Non-raising; degrades to text placement.
# --------------------------------------------------------------------------- #

class XsdResolver:
    def __init__(self, xsd_path: Optional[str]):
        self.ok = False
        self._ct: Dict[str, ET.Element] = {}
        self._doc_type: Optional[str] = None
        self._cache: Dict[str, Optional[str]] = {}
        self._order_cache: Dict[str, List[str]] = {}
        if not xsd_path:
            return
        self._enums: Dict[str, List[str]] = {}
        try:
            root = ET.parse(xsd_path).getroot()
            self._ct = {c.get("name"): c for c in root.findall(f"{_XS}complexType")}
            tops = {e.get("name"): e.get("type") for e in root.findall(f"{_XS}element")}
            self._doc_type = tops.get("Document")
            for st in root.findall(f"{_XS}simpleType"):
                name = st.get("name")
                vals = [en.get("value") for en in st.iter(f"{_XS}enumeration") if en.get("value")]
                if name and vals:
                    self._enums[name] = vals
            self.ok = bool(self._doc_type)
        except Exception:
            self.ok = False

    def _children(self, type_name: Optional[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        ce = self._ct.get(type_name) if type_name else None
        if ce is None:
            return out

        def rec(node):
            for ch in node:
                tag = ch.tag.replace(_XS, "")
                if tag == "element" and ch.get("name"):
                    out[ch.get("name")] = ch.get("type")
                elif tag in ("sequence", "choice", "group"):
                    rec(ch)
        rec(ce)
        return out

    def _leaf_type(self, path: str) -> Optional[str]:
        if not self.ok:
            return None
        parts = [p for p in path.split("/") if p]
        if not parts or parts[0] != "Document":
            return None
        tname = self._doc_type
        for seg in parts[1:]:
            kids = self._children(tname)
            if seg not in kids:
                return None
            tname = kids[seg]
        return tname

    def value_child(self, path: str) -> Optional[str]:
        """Name of the leaf element's value child (non-NoDataOptn generic value
        tag), e.g. 'Cd' / 'Val' / 'Amt'. None if unknown."""
        if path in self._cache:
            return self._cache[path]
        result = None
        t = self._leaf_type(path)
        for n in self._children(t):
            if n != "NoDataOptn" and n in _VALUE_TAGS:
                result = n
                break
        self._cache[path] = result
        return result

    def nodata_codes(self, path: str) -> List[str]:
        """Allowed ND enumeration values for the element at ``path`` (via its
        NoDataOptn -> NoData type). Different fields permit different ND subsets
        (e.g. {ND1,ND2,ND3} vs {ND5} vs {ND4})."""
        t = self._leaf_type(path)
        if not t:
            return []
        nd_choice = self._children(t).get("NoDataOptn")
        if not nd_choice:
            return []
        nodata_type = self._children(nd_choice).get("NoData")
        return self._enums.get(nodata_type, []) if nodata_type else []

    def child_order(self, path: str) -> List[str]:
        """XSD-declared child element order for the element at ``path`` (i.e. the
        order of its type's sequence). Empty if unknown."""
        if path in self._order_cache:
            return self._order_cache[path]
        order = list(self._children(self._leaf_type(path)).keys())
        self._order_cache[path] = order
        return order


# --------------------------------------------------------------------------- #
# Tree construction
# --------------------------------------------------------------------------- #

def _q(tag: str) -> str:
    return f"{{{NS}}}{tag}"


def _gc(parent: ET.Element, tag: str) -> ET.Element:
    """get-or-create a singleton child (by tag)."""
    el = parent.find(_q(tag))
    if el is None:
        el = ET.SubElement(parent, _q(tag))
    return el


def _gc_ordered(parent: ET.Element, parent_path: str, tag: str,
                resolver: XsdResolver) -> ET.Element:
    """get-or-create a singleton child, INSERTED in the XSD-declared sibling
    order (so e.g. NewUndrlygXpsrIdr precedes OrgnlUndrlygXpsrIdr, ActvtyDtDtls
    precedes UndrlygXpsrDtls, CollIdr precedes CollCmonData)."""
    existing = parent.find(_q(tag))
    if existing is not None:
        return existing
    order = resolver.child_order(parent_path)
    new = ET.Element(_q(tag))
    if tag in order:
        pos = order.index(tag)
        insert_at = len(parent)
        for i, ch in enumerate(list(parent)):
            ln = ch.tag.replace(f"{{{NS}}}", "")
            if ln in order and order.index(ln) > pos:
                insert_at = i
                break
        parent.insert(insert_at, new)
    else:
        parent.append(new)
    return new


def _build_chain(start: ET.Element, start_path: str, parts: List[str],
                 resolver: XsdResolver) -> ET.Element:
    """Build/descend a singleton chain, inserting each node in XSD sibling order."""
    cur = start
    cur_path = start_path
    for p in parts:
        cur = _gc_ordered(cur, cur_path, p, resolver)
        cur_path = f"{cur_path}/{p}"
    return cur


def _place_value(leaf: ET.Element, field: Dict[str, Any], value: str,
                 resolver: XsdResolver) -> None:
    """Place a value at an accepted leaf element. ND sentinels (for
    value_or_nodata fields) go into the NoDataOptn/NoData wrapper; otherwise the
    value goes into the XSD value child (Cd/Amt/...) when known, else as text."""
    if is_nd(value) and field.get("value_mode") == "value_or_nodata":
        nd = _gc(leaf, "NoDataOptn")
        nod = _gc(nd, "NoData")
        code = str(value).upper()
        # different fields allow different ND subsets; for structural placeholders
        # pick a code the field actually permits (never invents a real value).
        allowed = resolver.nodata_codes(field["xml_path"])
        if allowed and code not in allowed:
            code = allowed[0]
        nod.text = code
        return
    child = resolver.value_child(field["xml_path"])
    if child:
        c = _gc(leaf, child)
        c.text = str(value)
        if child == "Amt":
            c.set("Ccy", "EUR")
    else:
        leaf.text = str(value)


def _seq_key(f: Dict[str, Any]) -> Tuple[int, str]:
    so = f.get("sequence_order")
    try:
        so = int(so)
    except (TypeError, ValueError):
        so = 10_000
    return (so, f["esma_code"])


def build_tree(
    *,
    emit_fields: List[Dict[str, Any]],
    loans: List[Tuple[str, Dict[str, str]]],
    max_records: int,
    watermark: str,
    meta: Dict[str, str],
    xsd_path: Optional[str],
    header_values: Optional[Dict[str, str]] = None,
) -> Tuple[ET.Element, Dict[str, Any]]:
    """Build the nested ESMA tree. ``emit_fields`` carry esma_code, xml_path,
    record_group, value_mode, value_source, sequence_order. ``loans`` is an
    ordered list of (loan_id, {esma_code: value}). ``header_values`` supplies the
    report-level (one-per-report) values keyed by esma_code; these MANDATORY
    header elements are emitted FIRST, in XSD sequence order, BEFORE any
    UndrlygXpsrRcrd — so the top-level Securitisation1 sequence is valid.
    Returns (root, stats)."""
    header_values = header_values or {}
    resolver = XsdResolver(xsd_path)
    ET.register_namespace("", NS)
    ET.register_namespace("xsi", XSI)
    root = ET.Element(_q("Document"),
                      {f"{{{XSI}}}schemaLocation": f"{NS} {Path(xsd_path).name if xsd_path else ''}"})

    header = [f for f in emit_fields if RECORD_ANCHOR not in f["xml_path"].split("/")]
    record = [f for f in emit_fields if RECORD_ANCHOR in f["xml_path"].split("/")]
    header.sort(key=_seq_key)
    record.sort(key=_seq_key)

    stats = {"records_emitted": 0, "fields_emitted": 0, "placeholder_fields": 0,
             "nodata_wrappers": 0, "rrec_fields_nested": 0, "header_fields": 0}

    def _count(field, value):
        stats["fields_emitted"] += 1
        if field.get("value_source") == "preview_only_placeholder":
            stats["placeholder_fields"] += 1
        if is_nd(value) and field.get("value_mode") == "value_or_nodata":
            stats["nodata_wrappers"] += 1
        if field.get("record_group") == "RREC":
            stats["rrec_fields_nested"] += 1

    # MANDATORY report-header singletons FIRST, in XSD sequence order. The value
    # is the report-level header value (real or a pattern-valid preview
    # placeholder); records are NEVER emitted before these exist.
    for f in header:
        val = header_values.get(f["esma_code"], "")
        if val == "":
            # fall back to any per-loan value if a report value was not supplied.
            val = next((vals.get(f["esma_code"], "") for _lid, vals in loans
                        if vals.get(f["esma_code"], "")), "")
        if val == "":
            continue
        leaf = _build_chain(root, "Document", f["xml_path"].split("/")[1:], resolver)
        _place_value(leaf, f, val, resolver)
        _count(f, val)
        stats["header_fields"] += 1

    # locate the report-parent chain (up to and incl. ScrtstnRpt) for record nodes.
    report_parts = None
    for f in record:
        parts = f["xml_path"].split("/")
        if _REPORT_LEAF in parts:
            report_parts = parts[1:parts.index(_REPORT_LEAF) + 1]  # after Document .. ScrtstnRpt
            break
    if report_parts is None and header:
        p = header[0]["xml_path"].split("/")
        if _REPORT_LEAF in p:
            report_parts = p[1:p.index(_REPORT_LEAF) + 1]

    if record and report_parts is not None:
        report_path = "/".join(["Document", *report_parts])
        report_node = _build_chain(root, "Document", report_parts, resolver)
        record_path = f"{report_path}/{RECORD_ANCHOR}"
        # process record fields in XSD sequence order too (belt-and-braces with
        # the ordered insertion in _gc_ordered).
        record_sorted = sorted(record, key=_seq_key)
        for loan_id, vals in loans[:max(0, int(max_records))]:
            # UndrlygXpsrRcrd is repeatable (maxOccurs unbounded): fresh node per
            # loan, appended AFTER the mandatory header elements already in place.
            record_node = ET.SubElement(report_node, _q(RECORD_ANCHOR))
            emitted_here = 0
            for f in record_sorted:
                val = vals.get(f["esma_code"], "")
                if val == "":
                    continue
                parts = f["xml_path"].split("/")
                rel = parts[parts.index(RECORD_ANCHOR) + 1:]
                if not rel:
                    continue
                leaf = _build_chain(record_node, record_path, rel, resolver)
                _place_value(leaf, f, val, resolver)
                _count(f, val)
                emitted_here += 1
            if emitted_here:
                stats["records_emitted"] += 1

    return root, stats


def serialize(root: ET.Element, *, watermark: str, meta: Dict[str, str]) -> str:
    """Serialize with an XML declaration and watermark/metadata as COMMENTS so
    they never interfere with XSD validation."""
    try:
        ET.indent(root, space="  ")
    except Exception:
        pass
    body = ET.tostring(root, encoding="unicode")
    meta_lines = " ".join(f"{k}={v!r}" for k, v in meta.items())
    head = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f"<!-- {watermark} -->\n"
        "<!-- NON-PRODUCTION XSD-STRUCTURED PREVIEW. NOT FOR REGULATORY SUBMISSION. "
        "Production XML remains blocked; production_ready=false. -->\n"
        f"<!-- meta: {meta_lines} ProductionXmlStatus=BLOCKED -->\n"
    )
    return head + body + "\n"


def validate_against_xsd(xml_text: str, xsd_path: Optional[str],
                         known_limitations: Optional[List[str]] = None) -> Dict[str, Any]:
    """Attempt XSD validation honestly. Returns a structured, truthful report.
    For the client-safe preview, validation is EXPECTED to fail (economic fields
    are not fabricated) — see known_limitations."""
    if known_limitations is None:
        known_limitations = [
            "Only builder-accepted fields for a small sample of loans are emitted; "
            "mandatory ESMA elements/siblings are not all present.",
            "Economically meaningful mandatory fields (valuation amounts, balances, "
            "rates, income, economic dates) are NOT fabricated in client-safe "
            "preview mode, so XSD validation may fail on missing economic values.",
            "Leaf value-typing is shallow; deep monetary wrappers (Val/Amt) are "
            "not fully modelled.",
            "Strict XSD xs:sequence ordering of intermediate containers is approximate.",
            "Asset-class/performing branch is fixed to ResdtlRealEsttLn/PrfrmgLn.",
            "Schema is the ESMA DRAFT (DRAFT1auth.099.001.04); final schema unconfirmed.",
        ]
    report: Dict[str, Any] = {
        "xsd_validation_attempted": False,
        "xsd_validation_passed": False,
        "xsd_path": xsd_path or "",
        "validation_errors": [],
        # True once the report/header sequence is correct (ScrtstnIdr, CutOffDt
        # before UndrlygXpsrRcrd) — i.e. the old top-level error is gone.
        "top_level_header_ordering_ok": False,
        "record_sequence_fixes": {},
        "remaining_error_categories": [],
        "known_limitations": known_limitations,
        "note": "Structure-proof preview; XSD validity is NOT claimed. Remaining "
                "errors are deeper record-level sequence/type gaps, reported honestly.",
    }
    if not xsd_path or not Path(xsd_path).exists():
        report["validation_errors"] = ["XSD not available for validation"]
        return report
    try:
        from lxml import etree  # type: ignore
    except Exception as exc:  # pragma: no cover
        report["validation_errors"] = [f"lxml unavailable: {exc}"]
        return report
    try:
        schema = etree.XMLSchema(etree.parse(xsd_path))
        doc = etree.fromstring(xml_text.encode("utf-8"))
        report["xsd_validation_attempted"] = True
        passed = bool(schema.validate(doc))
        report["xsd_validation_passed"] = passed
        all_errors = [f"line {e.line}: {e.message}" for e in schema.error_log]
        report["validation_errors"] = all_errors[:50]
        # the previous top-level error: UndrlygXpsrRcrd before the mandatory
        # header ScrtstnIdr. It is fixed when no such error remains.
        header_err = any(("UndrlygXpsrRcrd" in e and "ScrtstnIdr" in e and "Expected" in e)
                         for e in all_errors)
        report["top_level_header_ordering_ok"] = passed or not header_err

        def _gone(*needles):
            return passed or not any(all(n in e for n in needles) for e in all_errors)
        # The specific deeper errors targeted by this change (gone == fixed):
        report["record_sequence_fixes"] = {
            "obligor_identifiers_complete": _gone("UndrlygXpsrId", "NewOblgrIdr"),
            "oblgr_dtls_before_undrlyg_dtls": _gone("UndrlygXpsrDtls", "Expected", "OblgrDtls"),
            "valtn_present_in_collcmondata": _gone("CollCmonData", "Valtn"),
            "dtls_cmondata_first": _gone("PrprtyTp", "Expected", "CmonData"),
        }
        cats = []
        if any("UndrlygXpsrId" in e or "OblgrIdr" in e or "UndrlygXpsrIdr" in e for e in all_errors):
            cats.append("exposure_identification_sequence")
        if any("ActvtyDtDtls" in e or "UndrlygXpsrDtls" in e for e in all_errors):
            cats.append("loan_common_data_sequence")
        if any("CollIdr" in e or "CollCmonData" in e for e in all_errors):
            cats.append("collateral_sequence")
        report["remaining_error_categories"] = cats
    except Exception as exc:
        report["xsd_validation_attempted"] = True
        report["validation_errors"] = [f"{type(exc).__name__}: {exc}"]
    return report


# --------------------------------------------------------------------------- #
# Synthetic XSD-structured schema test (ENGINEERING ONLY)
# --------------------------------------------------------------------------- #
# A recursive XSD instance generator that builds the FULL mandatory
# residential-real-estate / performing-loan tree with type-valid DUMMY values,
# so that full DRAFT-XSD validation can be attempted. Every value is synthetic
# and labelled ``synthetic_schema_test``. This is never client-facing, never
# production, and never fabricates values for the client-safe preview.

SYNTHETIC_SOURCE = "synthetic_schema_test"

# Choice branches to prefer (residential performing; submit not cancel).
_CHOICE_PREFER = ("NewCrrctn", "ResdtlRealEsttLn", "PrfrmgLn")
# Optional (minOccurs=0) elements that must still be forced so the instance is
# meaningful (the report wrapper is a sequence of two optional choices).
_FORCE_INCLUDE = {"NewCrrctn"}
_SKIP_CHOICE = {"NoDataOptn", "Cxl"}


def _class_char(cls: str) -> str:
    if "A-Z" in cls:
        return "A"
    if "a-z" in cls:
        return "a"
    if "0-9" in cls:
        return "0"
    # strip ranges, take first concrete char.
    s = cls.replace("\\d", "0").replace("\\w", "A")
    return s[0] if s else "A"


def _first_alternative(p: str) -> str:
    """Return the first top-level ``|`` alternative of a regex (alternation not
    inside a character class)."""
    depth = 0
    for i, c in enumerate(p):
        if c == "[":
            depth += 1
        elif c == "]":
            depth = max(0, depth - 1)
        elif c == "|" and depth == 0:
            return p[:i]
    return p


def _sample_pattern(p: str) -> str:
    """Deterministically produce a string matching the simple ESMA regex patterns
    (character classes + quantifiers + literals; first branch of any alternation)."""
    p = _first_alternative(p)
    out: List[str] = []
    i, n = 0, len(p)
    while i < n:
        c = p[i]
        if c == "[":
            j = p.find("]", i)
            if j < 0:
                break
            cls = p[i + 1:j]
            i = j + 1
            rep = 1
            if i < n and p[i] == "{":
                k = p.find("}", i)
                q = p[i + 1:k]
                i = k + 1
                lo = q.split(",")[0]
                rep = int(lo) if lo.isdigit() else 1
            elif i < n and p[i] in "?*":
                i += 1
                rep = 0
            elif i < n and p[i] == "+":
                i += 1
                rep = 1
            out.append(_class_char(cls) * rep)
        elif c == "\\":
            ch = p[i + 1] if i + 1 < n else ""
            i += 2
            out.append("0" if ch == "d" else ("A" if ch in ("w", "S") else ch))
        elif c in "(){}|^$":
            i += 1
        else:
            lit = c
            i += 1
            if i < n and p[i] == "{":
                k = p.find("}", i)
                q = p[i + 1:k]
                i = k + 1
                lo = q.split(",")[0]
                rep = int(lo) if lo.isdigit() else 1
                out.append(lit * rep)
            else:
                out.append(lit)
    return "".join(out) or "SYNTH"


class SyntheticXsdGenerator:
    """Recursive generator of a full mandatory residential-performing instance."""

    def __init__(self, xsd_path: str):
        self.ok = False
        self.ct: Dict[str, ET.Element] = {}
        self.st: Dict[str, ET.Element] = {}
        self.tops: Dict[str, str] = {}
        self.catalog: List[Dict[str, Any]] = []
        self.code_by_path: Dict[str, str] = {}
        self.xsd_name = Path(xsd_path).name if xsd_path else ""
        try:
            root = ET.parse(xsd_path).getroot()
            self.ct = {c.get("name"): c for c in root.findall(f"{_XS}complexType")}
            self.st = {s.get("name"): s for s in root.findall(f"{_XS}simpleType")}
            self.tops = {e.get("name"): e.get("type") for e in root.findall(f"{_XS}element")}
            self.ok = "Document" in self.tops
        except Exception:
            self.ok = False

    # -- value generation -------------------------------------------------- #
    def _builtin(self, base: str) -> str:
        b = (base or "").split(":")[-1]
        if b == "date":
            return "2024-12-31"
        if b == "dateTime":
            return "2024-12-31T00:00:00"
        if b in ("gYear", "gYearMonth"):
            return "2024"
        if b in ("decimal", "double", "float"):
            return "1"
        if b in ("integer", "nonNegativeInteger", "positiveInteger", "int", "long",
                 "short", "byte"):
            return "1"
        if b == "boolean":
            return "true"
        return "SYNTH"

    def dummy(self, type_name: Optional[str], depth: int = 0) -> Tuple[str, str]:
        """Return (value, reason) for a simple/leaf type."""
        if not type_name or depth > 8:
            return "SYNTH", "fallback"
        st = self.st.get(type_name)
        if st is None:
            return self._builtin(type_name), f"builtin {type_name.split(':')[-1]}"
        restr = st.find(f"{_XS}restriction")
        if restr is None:
            return "SYNTH", "no-restriction fallback"
        enums = [e.get("value") for e in restr.findall(f"{_XS}enumeration") if e.get("value")]
        if enums:
            return enums[0], "enum first value"
        pat = restr.find(f"{_XS}pattern")
        if pat is not None and pat.get("value"):
            return _sample_pattern(pat.get("value")), "pattern sample"
        base = restr.get("base")
        if base in self.st:
            return self.dummy(base, depth + 1)
        return self._builtin(base), f"builtin base {(base or '').split(':')[-1]}"

    # -- tree construction ------------------------------------------------- #
    def generate(self, code_by_path: Optional[Dict[str, str]] = None):
        self.catalog = []
        self.code_by_path = code_by_path or {}
        ET.register_namespace("", NS)
        ET.register_namespace("xsi", XSI)
        root = ET.Element(_q("Document"),
                          {f"{{{XSI}}}schemaLocation": f"{NS} {self.xsd_name}"})
        self._fill_complex(root, self.tops.get("Document"), "Document", 0, [])
        return root, self.catalog

    def _record(self, path, value, type_name, reason, attrs=None):
        code = self.code_by_path.get(path) or self.code_by_path.get(path.rsplit("/", 1)[0], "")
        self.catalog.append({
            "xml_path": path, "esma_code": code, "value": value,
            "xsd_type": type_name or "", "value_reason": reason,
            "source_reason": SYNTHETIC_SOURCE, "source": SYNTHETIC_SOURCE,
            "attributes": ";".join(f"{k}={v}" for k, v in (attrs or {}).items()),
        })

    def _fill_complex(self, elem, type_name, path, depth, stack):
        if depth > 60:
            return
        ctype = self.ct.get(type_name)
        if ctype is None:
            return
        sc = ctype.find(f"{_XS}simpleContent")
        if sc is not None:
            ext = sc.find(f"{_XS}extension")
            base = ext.get("base") if ext is not None else None
            val, reason = self.dummy(base)
            elem.text = val
            attrs = {}
            for attr in (ext.findall(f"{_XS}attribute") if ext is not None else []):
                if attr.get("use") == "required":
                    av, _ = self.dummy(attr.get("type"))
                    elem.set(attr.get("name"), av)
                    attrs[attr.get("name")] = av
            self._record(path, val, base, "simpleContent " + reason, attrs)
            return
        for child in ctype:
            tag = child.tag.replace(_XS, "")
            if tag in ("sequence", "all"):
                self._walk(child, elem, path, depth, stack)
            elif tag == "choice":
                self._walk_choice(child, elem, path, depth, stack)

    def _walk(self, node, parent, path, depth, stack):
        for child in node:
            tag = child.tag.replace(_XS, "")
            if tag in ("sequence", "all"):
                self._walk(child, parent, path, depth, stack)
            elif tag == "choice":
                self._walk_choice(child, parent, path, depth, stack)
            elif tag == "element":
                name = child.get("name")
                if not name:
                    continue
                if child.get("minOccurs", "1") == "0" and name not in _FORCE_INCLUDE:
                    continue
                self._emit(child, parent, path, depth, stack)

    def _walk_choice(self, choice_node, parent, path, depth, stack):
        elems = [c for c in choice_node if c.tag.replace(_XS, "") == "element" and c.get("name")]
        if not elems:
            # choice of groups/sequences — take the first particle.
            for c in choice_node:
                if c.tag.replace(_XS, "") in ("sequence", "choice"):
                    self._walk(c, parent, path, depth, stack)
                    return
            return
        chosen = None
        for c in elems:
            if c.get("name") in _CHOICE_PREFER:
                chosen = c
                break
        if chosen is None:
            for c in elems:
                if c.get("name") not in _SKIP_CHOICE:
                    chosen = c
                    break
        chosen = chosen or elems[0]
        self._emit(chosen, parent, path, depth, stack)

    def _emit(self, el_node, parent, path, depth, stack):
        name = el_node.get("name")
        type_name = el_node.get("type")
        child = ET.SubElement(parent, _q(name))
        cpath = f"{path}/{name}"
        if type_name in self.ct:
            if stack.count(type_name) >= 2:   # recursion guard
                return
            self._fill_complex(child, type_name, cpath, depth + 1, stack + [type_name])
        else:
            val, reason = self.dummy(type_name)
            child.text = val
            self._record(cpath, val, type_name, reason)


def build_synthetic_document(xsd_path: str, code_by_path: Optional[Dict[str, str]] = None):
    """Build a full synthetic residential-performing instance + value catalog."""
    gen = SyntheticXsdGenerator(xsd_path)
    if not gen.ok:
        return None, []
    return gen.generate(code_by_path)


def serialize_synthetic(root: ET.Element, *, watermark: str, meta: Dict[str, str]) -> str:
    try:
        ET.indent(root, space="  ")
    except Exception:
        pass
    body = ET.tostring(root, encoding="unicode")
    meta_lines = " ".join(f"{k}={v!r}" for k, v in meta.items())
    head = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f"<!-- {watermark} -->\n"
        "<!-- ENGINEERING-ONLY SYNTHETIC SCHEMA TEST. NOT CLIENT-FACING. NOT FOR "
        "REGULATORY USE. Production XML remains blocked; production_ready=false. -->\n"
        "<!-- every value is synthetic_schema_test (dummy); never a real value. -->\n"
        f"<!-- meta: {meta_lines} ProductionXmlStatus=BLOCKED -->\n"
    )
    return head + body + "\n"
