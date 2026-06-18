#!/usr/bin/env python3
"""
build_annex2_field_xsd_path_map.py
==================================

Build the Annex 2 field-to-XSD path mapping layer by resolving the REAL ESMA XML
tree from the vendored schema and cross-checking the vendored sample message.

It is evidence-driven and deliberately conservative:

  * paths are resolved by recursively walking the residential-real-estate /
    performing-loan branch of ``DRAFT1auth.099.001.04_1.3.0.xsd``
    (ResdtlRealEsttLn/PrfrmgLn -> SecuritisationLoanData2 ->
    ExposureData1 + CollateralData22), plus the report header (ScrtstnRpt) and
    the exposure-identification block (UndrlygXpsrId);
  * a mapping is only ``confirmed`` when the delivery-rules ``workbook_semantic``
    leaf token AND the vendored sample message agree with the XSD path;
  * field labels / leaf tokens alone are treated as INFERENCE, not proof
    (inferred_high_confidence / inferred_low_confidence);
  * a token that names an element absent from the schema is a ``conflict``;
  * a field with no usable evidence is ``unresolved``.

Outputs (no production XML, no gate changes):
  * config/delivery/annex2_field_xsd_path_map.yaml   (production source of truth)
  * output/config_review/annex2_field_xsd_path_map.csv

Read-only over config + the vendored XSD/sample. Generates no XML.
"""

from __future__ import annotations

import csv
import difflib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
_XSD = _REPO / "DRAFT1auth.099.001.04_1.3.0.xsd"
_SAMPLE = _REPO / "DRAFT1auth.099.001.04_non-ABCP Underlying Exposure Report.xml"
_UNIVERSE = _REPO / "config" / "regime" / "annex2_field_universe.yaml"
_RULES = _REPO / "config" / "regime" / "annex2_delivery_rules.yaml"
_YAML_OUT = _REPO / "config" / "delivery" / "annex2_field_xsd_path_map.yaml"
_CSV_OUT = _REPO / "output" / "config_review" / "annex2_field_xsd_path_map.csv"

_XS = "{http://www.w3.org/2001/XMLSchema}"

# Confirmed structural base paths (resolved in the structure contract task).
_BASE = "Document/ScrtstnNonAsstBckdComrclPprUndrlygXpsrRpt/NewCrrctn/ScrtstnRpt"
_REC = f"{_BASE}/UndrlygXpsrRcrd"
_LOAN = f"{_REC}/UndrlygXpsrData/ResdtlRealEsttLn/PrfrmgLn"

# Generic XSD "value" tags: a complexType whose children are all generic value
# tags is a single reportable value leaf (e.g. PrprtyTp -> Cd). Domain-named
# children (OrgnlIdr, RpDt, ...) mean the type is a container -> recurse.
_VALUE_TAGS = {"Cd", "Dt", "Amt", "Rate", "Nb", "Ind", "Yr", "Lien", "NbOfMnths",
               "NbOfDays", "LEI", "Pctg", "Val", "Max"}
_NODATA = "NoDataOptn"

# Status vocabulary.
CONFIRMED = "confirmed"
HIGH = "inferred_high_confidence"
LOW = "inferred_low_confidence"
UNRESOLVED = "unresolved"
CONFLICT = "conflict"


# --------------------------------------------------------------------------- #
# XSD model
# --------------------------------------------------------------------------- #

def _load_xsd():
    root = ET.parse(_XSD).getroot()
    complex_types = {ct.get("name"): ct for ct in root.findall(f"{_XS}complexType")}
    all_element_names = {e.get("name") for e in root.iter(f"{_XS}element") if e.get("name")}
    return complex_types, all_element_names


def _content_children(ct_elem):
    """Ordered (name, type, minOccurs, maxOccurs) of the element's content model,
    descending only through sequence/choice wrappers (not into element types)."""
    out = []

    def rec(node):
        for ch in node:
            tag = ch.tag.replace(_XS, "")
            if tag == "element" and ch.get("name"):
                out.append((ch.get("name"), ch.get("type"),
                            ch.get("minOccurs", "1"), ch.get("maxOccurs", "1")))
            elif tag in ("sequence", "choice", "group"):
                rec(ch)
    rec(ct_elem)
    return out


class XsdWalker:
    def __init__(self, complex_types):
        self.ct = complex_types
        self.order = 0
        self.leaves = {}   # leaf_name -> list of leaf dicts

    def _is_complex(self, type_name):
        return type_name in self.ct

    def _add(self, name, full_path, type_name, value_mode, nd_path, level, mn, mx):
        self.order += 1
        self.leaves.setdefault(name, []).append({
            "leaf": name, "xml_path": full_path, "xsd_type": type_name,
            "value_mode": value_mode, "nd_wrapper_path": nd_path or "",
            "xml_level": level, "sequence_order": self.order,
            "min_occurs": mn, "max_occurs": mx,
        })

    def walk(self, name, type_name, parent_path, level, mn="1", mx="1", depth=0):
        full = f"{parent_path}/{name}"
        if depth > 12:
            return
        if not self._is_complex(type_name):
            self._add(name, full, type_name, "value", None, level, mn, mx)
            return
        kids = _content_children(self.ct[type_name])
        names = [k[0] for k in kids]
        if _NODATA in names:
            self._add(name, full, type_name, "value_or_nodata",
                      f"{full}/{_NODATA}/NoData", level, mn, mx)
            return
        if names and all(n in _VALUE_TAGS for n in names):
            self._add(name, full, type_name, "value", None, level, mn, mx)
            return
        # container -> recurse into each child
        for (cn, ct, cmn, cmx) in kids:
            self.walk(cn, ct, full, level, cmn, cmx, depth + 1)


def build_xsd_index():
    complex_types, all_names = _load_xsd()
    w = XsdWalker(complex_types)
    # header (report-level)
    w.walk("ScrtstnIdr", "SecuritisationIdentifier", _BASE, "header")
    w.walk("CutOffDt", "ISODate", _BASE, "header")
    # exposure identification
    w.walk("UndrlygXpsrId", "ExposureIdentificationCommonData1", _REC, "exposure")
    # loan-level common data (residential performing)
    w.walk("UndrlygXpsrCmonData", "ExposureData1", _LOAN, "exposure")
    # collateral (residential performing), nested under the loan
    w.walk("Coll", "CollateralData22", _LOAN, "collateral")
    return w.leaves, all_names


# --------------------------------------------------------------------------- #
# Sample message index
# --------------------------------------------------------------------------- #

def build_sample_index():
    root = ET.parse(_SAMPLE).getroot()
    names = set()
    paths = {}

    def rec(elem, trail):
        tag = elem.tag.split("}")[-1]
        trail2 = trail + [tag]
        names.add(tag)
        paths.setdefault(tag, []).append("/".join(trail2))
        for ch in elem:
            rec(ch, trail2)
    rec(root, [])
    return names, paths


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def _load_yaml(p):
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


# --------------------------------------------------------------------------- #
# Per-code mapping
# --------------------------------------------------------------------------- #

def map_code(code, rule, entry, xsd_index, all_xsd_names, sample_names):
    rg = "RREC" if code.upper().startswith("RREC") else "RREL"
    token = str(rule.get("workbook_semantic") or "").strip()
    leaf_key = token.split("/")[0] if token else ""
    mandatory = bool(rule.get("mandatory", False))
    nd_universe = bool(entry.get("nd1_4_allowed") or entry.get("nd5_allowed"))

    # default level by record group (corrected to nested collateral, never flat).
    level = "collateral" if rg == "RREC" else "exposure"
    xml_path = None
    xsd_element = None
    xsd_type = None
    cardinality = "unknown"
    value_mode = "unknown"
    nd_wrapper = None
    sequence_order = None
    status = UNRESOLVED
    source = "manual_review"
    note = ""

    def finalize(entry_leaf, st, src, lvl=None):
        nonlocal xml_path, xsd_element, xsd_type, value_mode, nd_wrapper, level, sequence_order
        xml_path = entry_leaf["xml_path"]
        xsd_element = entry_leaf["leaf"]
        xsd_type = entry_leaf["xsd_type"]
        value_mode = entry_leaf["value_mode"]
        nd_wrapper = entry_leaf["nd_wrapper_path"] or None
        sequence_order = entry_leaf["sequence_order"]
        level = lvl or entry_leaf["xml_level"]
        return st, src

    if leaf_key:
        matches = xsd_index.get(leaf_key, [])
        in_sample = leaf_key in sample_names
        if len(matches) == 1 and in_sample:
            status, source = finalize(matches[0], CONFIRMED, "sample_xml")
            note = f"workbook_semantic '{token}' agrees with XSD path and sample message"
        elif len(matches) == 1:
            status, source = finalize(matches[0], HIGH, "xsd")
            note = f"unique XSD element for leaf '{leaf_key}'; not present in sample message"
        elif len(matches) > 1:
            status, source = finalize(matches[0], LOW, "xsd")
            note = (f"leaf '{leaf_key}' is ambiguous: {len(matches)} XSD locations "
                    f"({', '.join(m['xml_path'].split('/')[-2] for m in matches[:4])} ...)")
        else:
            # token leaf not found by exact name. delivery_rules workbook_semantic
            # uses a workbook naming convention that often differs from the XSD
            # element names (e.g. 'AmrtstnType' vs XSD 'AmtstnTp', 'Prps' vs
            # 'Purp'), so a non-match is NOT proof of contradiction.
            if leaf_key in all_xsd_names:
                status, source = LOW, "delivery_rules"
                note = (f"leaf '{leaf_key}' exists in the XSD but not in the residential "
                        "performing branch (different asset class / sub-branch); needs review")
            else:
                near = difflib.get_close_matches(leaf_key, sorted(xsd_index), n=1, cutoff=0.82)
                if near:
                    status, source = finalize(xsd_index[near[0]][0], LOW, "delivery_rules")
                    note = (f"workbook_semantic '{token}' uses non-XSD naming; closest XSD "
                            f"element '{near[0]}' is a CANDIDATE only — needs manual confirmation")
                else:
                    status, source = UNRESOLVED, "manual_review"
                    note = (f"workbook_semantic '{token}' uses workbook (non-XSD) naming with no "
                            "close XSD element in the residential branch; needs manual XSD review")
    else:
        # no workbook_semantic token (TBC / commented out in delivery rules).
        status, source = UNRESOLVED, "manual_review"
        note = "no workbook_semantic in delivery_rules (TBC/mismapped); needs manual XSD review"

    # cardinality from level.
    if level == "header":
        cardinality = "one_per_report"
    elif level == "collateral":
        cardinality = "one_per_collateral"
    elif level == "exposure":
        cardinality = "one_per_loan"

    # value mode default when unresolved but universe says ND allowed.
    if value_mode == "unknown" and nd_universe:
        value_mode = "value_or_nodata"
        if not note.endswith("."):
            note += "; "
        note += "universe allows ND -> expect NoDataOptn wrapper"

    blocks = status != CONFIRMED
    owner = ("delivery_xml" if status == CONFIRMED
             else "config_policy" if status in (UNRESOLVED, CONFLICT) and not leaf_key
             else "manual_review")

    canonical = rule.get("projected_source_field") or _slug(entry.get("field_name", ""))
    return {
        "esma_code": code,
        "canonical_field": canonical,
        "record_group": rg,
        "xml_level": level,
        "xml_path": xml_path,
        "xsd_element": xsd_element,
        "xsd_type": xsd_type,
        "cardinality": cardinality,
        "sequence_order": sequence_order,
        "value_mode": value_mode,
        "nd_wrapper_path": nd_wrapper,
        "mapping_status": status,
        "evidence_source": source,
        "evidence_note": note,
        "blocks_production_xml": blocks,
        "owner": owner,
    }


# --------------------------------------------------------------------------- #
# Explicit, well-grounded overrides (sample-evidenced identity/header fields).
# --------------------------------------------------------------------------- #

def apply_overrides(rows_by_code, xsd_index, sample_names):
    def leaf(name):
        m = xsd_index.get(name, [])
        return m[0] if m else None

    # RREL1: report-level securitisation identifier. workbook_semantic=ScrtstnIdr,
    # XSD content/definition + 28-char pattern match the universe RREL1 content,
    # and ScrtstnIdr is present in the sample -> confirmed, with a level note.
    if "RREL1" in rows_by_code and leaf("ScrtstnIdr"):
        r = rows_by_code["RREL1"]
        e = leaf("ScrtstnIdr")
        r.update({"xml_level": "header", "xml_path": e["xml_path"],
                  "xsd_element": "ScrtstnIdr", "xsd_type": e["xsd_type"], "sequence_order": e["sequence_order"],
                  "cardinality": "one_per_report", "value_mode": e["value_mode"],
                  "mapping_status": CONFIRMED, "evidence_source": "sample_xml",
                  "owner": "delivery_xml", "blocks_production_xml": False,
                  "evidence_note": ("REPORT-LEVEL securitisation identifier (ScrtstnIdr): XSD "
                                    "definition + 28-char pattern match RREL1 content; present "
                                    "in sample. NOTE: report-level, not an exposure identifier.")})

    # RREL3/4/5: exposure identifiers present in sample under UndrlygXpsrId; the
    # field-name match is strong but not delivery_rules-proven -> high confidence.
    for code, elem in (("RREL3", "NewUndrlygXpsrIdr"), ("RREL4", "OrgnlOblgrIdr"),
                       ("RREL5", "NewOblgrIdr")):
        if code in rows_by_code and leaf(elem) and elem in sample_names:
            r = rows_by_code[code]
            e = leaf(elem)
            r.update({"xml_level": "exposure", "xml_path": e["xml_path"],
                      "xsd_element": elem, "xsd_type": e["xsd_type"], "sequence_order": e["sequence_order"],
                      "cardinality": "one_per_loan", "value_mode": "value",
                      "mapping_status": HIGH, "evidence_source": "sample_xml",
                      "owner": "manual_review", "blocks_production_xml": True,
                      "evidence_note": (f"present in sample under UndrlygXpsrId as {elem}; "
                                        "field-label match strong but not delivery_rules-proven")})

    # RREC3/RREC4: CollIdr children present in sample; field-label disambiguates
    # (Original->OrgnlIdr, New->NewIdr). High confidence, not confirmed.
    for code, elem in (("RREC3", "OrgnlIdr"), ("RREC4", "NewIdr")):
        if code in rows_by_code and leaf(elem) and elem in sample_names:
            r = rows_by_code[code]
            e = leaf(elem)
            r.update({"xml_level": "collateral", "xml_path": e["xml_path"],
                      "xsd_element": elem, "xsd_type": e["xsd_type"], "sequence_order": e["sequence_order"],
                      "cardinality": "one_per_collateral", "value_mode": "value",
                      "mapping_status": HIGH, "evidence_source": "sample_xml",
                      "owner": "manual_review", "blocks_production_xml": True,
                      "evidence_note": (f"Coll/CollIdr/{elem} present in sample; field-label "
                                        "disambiguation only — not delivery_rules-proven")})


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def build_rows():
    xsd_index, all_xsd_names = build_xsd_index()
    sample_names, _sample_paths = build_sample_index()
    universe = _load_yaml(_UNIVERSE).get("fields") or {}
    rules = _load_yaml(_RULES).get("field_rules") or {}

    def sort_key(c):
        return (c[:4], int("".join(ch for ch in c if ch.isdigit()) or 0))

    rows_by_code = {}
    for code in sorted(universe, key=sort_key):
        rows_by_code[code] = map_code(
            code, rules.get(code, {}) or {}, universe.get(code, {}) or {},
            xsd_index, all_xsd_names, sample_names)

    apply_overrides(rows_by_code, xsd_index, sample_names)
    _flag_path_collisions(rows_by_code)
    rows = [rows_by_code[c] for c in sorted(rows_by_code, key=sort_key)]
    return rows, xsd_index


def _flag_path_collisions(rows_by_code):
    """A genuine conflict: two or more codes resolve to the SAME XML path. The
    confirmed/highest-confidence claimant keeps it; the others are downgraded to
    ``conflict`` (two fields cannot occupy one element)."""
    rank = {CONFIRMED: 4, HIGH: 3, LOW: 2, UNRESOLVED: 1, CONFLICT: 0}
    by_path = {}
    for code, r in rows_by_code.items():
        p = r.get("xml_path")
        if p:
            by_path.setdefault(p, []).append(code)
    for path, codes in by_path.items():
        if len(codes) < 2:
            continue
        winner = max(codes, key=lambda c: rank.get(rows_by_code[c]["mapping_status"], 0))
        for c in codes:
            if c == winner:
                continue
            r = rows_by_code[c]
            r["mapping_status"] = CONFLICT
            r["blocks_production_xml"] = True
            r["owner"] = "manual_review"
            r["evidence_note"] = (f"path collision: resolves to the same element as {winner} "
                                  f"({path}); at most one code can map here — needs review")


_COLUMNS = ["esma_code", "canonical_field", "record_group", "xml_level", "xml_path",
            "xsd_element", "xsd_type", "cardinality", "sequence_order", "value_mode",
            "nd_wrapper_path", "mapping_status", "evidence_source", "evidence_note",
            "blocks_production_xml", "owner"]


def write_outputs(rows):
    from collections import Counter
    counts = Counter(r["mapping_status"] for r in rows)
    blocking = sum(1 for r in rows if r["blocks_production_xml"])

    # YAML — production source of truth.
    doc = {
        "field_xsd_path_map": {
            "contract_id": "ESMA_Annex2_auth099_field_xsd_path_map",
            "schema_message": "auth.099.001.04",
            "schema_version": "1.3.0",
            "schema_is_draft": True,
            "namespace": "urn:esma:xsd:DRAFT1auth.099.001.04",
            "root": "Document",
            "production_xsd_mapping_configured": False,
            "asset_branch_resolved": "ResdtlRealEsttLn/PrfrmgLn (residential performing)",
            "generated_by": "scripts/build_annex2_field_xsd_path_map.py",
            "summary": {
                "total_fields": len(rows),
                "confirmed": counts.get(CONFIRMED, 0),
                "inferred_high_confidence": counts.get(HIGH, 0),
                "inferred_low_confidence": counts.get(LOW, 0),
                "unresolved": counts.get(UNRESOLVED, 0),
                "conflict": counts.get(CONFLICT, 0),
                "production_blocking_mapping_gaps": blocking,
            },
            "production_guardrails": {
                "do_not_generate_production_xml": True,
                "do_not_change_production_gates": True,
                "do_not_use_preview_placeholders_for_mapping": True,
                "production_gates_remain": {
                    "xml_generation_allowed": False,
                    "xml_generated": False,
                    "ready_for_xml_delivery": False,
                },
            },
            "fields": rows,
        }
    }
    _YAML_OUT.parent.mkdir(parents=True, exist_ok=True)
    _YAML_OUT.write_text(yaml.safe_dump(doc, sort_keys=False, width=100), encoding="utf-8")

    # CSV.
    _CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_CSV_OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in _COLUMNS})

    return counts, blocking


def main():
    rows, _ = build_rows()
    counts, blocking = write_outputs(rows)
    print(f"Wrote {_YAML_OUT}")
    print(f"Wrote {_CSV_OUT}")
    print(f"  total fields: {len(rows)}")
    for st in (CONFIRMED, HIGH, LOW, UNRESOLVED, CONFLICT):
        print(f"  {st}: {counts.get(st, 0)}")
    print(f"  production-blocking mapping gaps: {blocking}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
