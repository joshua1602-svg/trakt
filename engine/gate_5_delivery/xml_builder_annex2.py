#!/usr/bin/env python3
"""
xml_builder_annex2.py

Annex 2 (ESMA_Annex2 / auth.099) XML builder.

Design notes specific to the Annex 2 workbook:
- Workbook PATH already contains the full XML path INCLUDING the leaf tag.
  Example observed in workbook: .../PoolAddtnDt/Dt with XML TAG <Dt>.
- XML TAG is therefore used mainly for branch selection / validation, while PATH
  remains the source of truth for node creation.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from lxml import etree

DEFAULT_NS = "urn:esma:xsd:DRAFT1auth.099.001.04"
ND_TAGS = {"NODATA", "NODATA4", "NODATAOPTN"}
RECORD_ANCHOR = "UndrlygXpsrRcrd"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


@dataclass
class MappingSpec:
    code: str
    tag: str
    path: str
    multiplicity: str
    pnp: str
    row_idx: int


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    s = str(v).strip()
    if s.lower() == "nan":
        return ""
    return s


def _clean_tag(tag: str) -> str:
    t = _safe_str(tag)
    if t.startswith("<") and t.endswith(">"):
        t = t[1:-1].strip()
    return t


def _split_codes(code_cell: str) -> List[str]:
    raw = _safe_str(code_cell)
    if not raw:
        return []
    return [c.strip() for c in raw.splitlines() if c.strip()]


def _split_path(path: str) -> List[str]:
    return [p for p in _safe_str(path).strip("/").split("/") if p]


def _path_tuple(path: str) -> Tuple[str, ...]:
    return tuple(_split_path(path))


def _is_nd(value: str) -> bool:
    return bool(re.fullmatch(r"ND[1-5]", _safe_str(value).upper()))


def _is_date(value: str) -> bool:
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", _safe_str(value)))


def _parse_multiplicity(mult: str) -> Tuple[int, Optional[int]]:
    m = _safe_str(mult).replace("[", "").replace("]", "").replace(" ", "")
    if ".." not in m:
        try:
            v = int(m)
            return v, v
        except ValueError:
            return 0, None
    a, b = m.split("..", 1)
    try:
        min_occ = int(a)
    except ValueError:
        min_occ = 0
    if b.lower() == "n":
        return min_occ, None
    try:
        return min_occ, int(b)
    except ValueError:
        return min_occ, None


def _split_multi_value(value: str) -> List[str]:
    v = _safe_str(value)
    if not v:
        return []
    if "|" in v:
        return [x.strip() for x in v.split("|") if x.strip()]
    return [v]


def get_namespace_from_xsd(xsd_path: Optional[str]) -> str:
    if not xsd_path:
        return DEFAULT_NS
    try:
        root = etree.parse(xsd_path).getroot()
        return root.get("targetNamespace") or DEFAULT_NS
    except Exception:
        return DEFAULT_NS


def load_mapping_specs(workbook_path: str, sheet_name: str, performance_mode: str) -> Dict[str, List[MappingSpec]]:
    df = pd.read_excel(workbook_path, sheet_name=sheet_name, header=3, dtype=str).fillna("")

    needed = ["RTS Field code", "XML TAG", "PATH", "MULTIPLICITY", "Template", "Performing/Non Performing"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Annex2 mapping workbook missing required columns: {missing}")

    specs_by_code: Dict[str, List[MappingSpec]] = {}
    perf = performance_mode.upper().strip()

    for i, row in df.iterrows():
        code_cell = _safe_str(row.get("RTS Field code"))
        path = _safe_str(row.get("PATH"))
        tag = _clean_tag(row.get("XML TAG"))
        template = _safe_str(row.get("Template")).upper()
        pnp = _safe_str(row.get("Performing/Non Performing")).upper()

        if not code_cell or not path or not tag:
            continue
        if "/Cxl/" in path:
            continue
        if template not in {"ALL", "RRE"}:
            continue
        if pnp and pnp not in {"PRF/NPRF", perf}:
            continue

        for code in _split_codes(code_cell):
            if not re.match(r"^RR(EL|EC)\d+$", code):
                continue
            specs_by_code.setdefault(code, []).append(
                MappingSpec(
                    code=code,
                    tag=tag,
                    path=path,
                    multiplicity=_safe_str(row.get("MULTIPLICITY")),
                    pnp=pnp,
                    row_idx=int(i),
                )
            )

    if not specs_by_code:
        raise ValueError("No Annex2 mapping specs found for templates ALL/RRE")

    for code, specs in specs_by_code.items():
        specs.sort(key=lambda s: (len(_split_path(s.path)), s.row_idx))
    return specs_by_code


def build_order_index(specs_by_code: Dict[str, List[MappingSpec]]) -> Dict[Tuple[str, ...], List[str]]:
    order: Dict[Tuple[str, ...], List[str]] = {}
    seen_paths = set()
    for specs in specs_by_code.values():
        for s in specs:
            if s.path in seen_paths:
                continue
            seen_paths.add(s.path)
            parts = _split_path(s.path)
            if len(parts) < 2:
                continue
            parent = tuple(parts[:-1])
            child = parts[-1]
            order.setdefault(parent, [])
            if child not in order[parent]:
                order[parent].append(child)
    return order


def _append_ordered_new(parent: etree._Element, child_local: str, ns: str, order_index: Dict[Tuple[str, ...], List[str]], parent_path: Tuple[str, ...]) -> etree._Element:
    """Always append a NEW node, but in workbook/XSD sequence order."""
    new_el = etree.Element(f"{{{ns}}}{child_local}")
    seq = order_index.get(parent_path, [])
    if child_local not in seq:
        parent.append(new_el)
        return new_el

    target_pos = seq.index(child_local)
    children = list(parent)
    insert_at = len(children)
    for i, ch in enumerate(children):
        ln = etree.QName(ch).localname
        if ln in seq and seq.index(ln) > target_pos:
            insert_at = i
            break
    parent.insert(insert_at, new_el)
    return new_el


def _get_or_create_singleton(parent: etree._Element, child_local: str, ns: str, order_index: Dict[Tuple[str, ...], List[str]], parent_path: Tuple[str, ...]) -> etree._Element:
    """Get existing child if present, else create one (singleton semantics)."""
    existing = parent.find(f"{{{ns}}}{child_local}")
    if existing is not None:
        return existing
    return _append_ordered_new(parent, child_local, ns, order_index, parent_path)


def get_or_create_path_singleton(root: etree._Element, full_path: str, ns: str, order_index: Dict[Tuple[str, ...], List[str]]) -> etree._Element:
    parts = _split_path(full_path)
    if not parts or parts[0] != "Document":
        raise ValueError(f"Invalid XML path (must start with /Document): {full_path}")
    cur = root
    built = ("Document",)
    for part in parts[1:]:
        cur = _get_or_create_singleton(cur, part, ns, order_index, built)
        built = (*built, part)
    return cur


def find_record_root_path(specs_by_code: Dict[str, List[MappingSpec]]) -> str:
    candidates: List[List[str]] = []
    for specs in specs_by_code.values():
        for s in specs:
            parts = _split_path(s.path)
            if RECORD_ANCHOR in parts:
                idx = parts.index(RECORD_ANCHOR)
                candidates.append(parts[: idx + 1])
    if not candidates:
        raise ValueError("Could not infer record root path containing UndrlygXpsrRcrd")
    shortest = sorted(candidates, key=len)[0]
    return "/" + "/".join(shortest)


def _relative_parts_from_record(full_path: str, record_root_path: str) -> List[str]:
    full = _split_path(full_path)
    rec = _split_path(record_root_path)
    if full[: len(rec)] != rec:
        raise ValueError(f"Path '{full_path}' not under inferred record root '{record_root_path}'")
    return full[len(rec):]


def create_new_record_node(root: etree._Element, record_root_path: str, ns: str, order_index: Dict[Tuple[str, ...], List[str]]) -> etree._Element:
    """
    Create a FRESH UndrlygXpsrRcrd branch for each row.

    We build parent singleton chain (up to ScrtstnRpt) and always append a new
    record node at the final step. This avoids reusing prior row's record node.
    """
    parts = _split_path(record_root_path)
    if not parts or parts[0] != "Document":
        raise ValueError(f"Invalid record root path: {record_root_path}")
    if parts[-1] != RECORD_ANCHOR:
        raise ValueError(f"Record root must end with {RECORD_ANCHOR}: {record_root_path}")

    parent_parts = parts[:-1]
    cur = root
    built = ("Document",)
    for part in parent_parts[1:]:
        cur = _get_or_create_singleton(cur, part, ns, order_index, built)
        built = (*built, part)

    # fresh record for this row (never reused)
    return _append_ordered_new(cur, RECORD_ANCHOR, ns, order_index, tuple(parent_parts))


def select_specs_for_value(specs: List[MappingSpec], value: str) -> List[MappingSpec]:
    """
    Deterministic branch selection for fields with multiple mapping branches.

    - ND values: prefer NoData4 for ND4; else NoData; then NoData4 fallback.
    - Non-ND values: pick deepest non-NoData branch; prefer <Dt> for date-shaped values.
    """
    value = _safe_str(value)
    is_nd = _is_nd(value)
    ordered = sorted(specs, key=lambda s: (len(_split_path(s.path)), s.row_idx), reverse=True)

    if is_nd:
        # Annex2 NoData4 branches often require additional Dt children.
        # For scalar ND values from CSV, use direct NoData under NoDataOptn.
        chosen = [s for s in ordered if s.tag.upper() == "NODATA" and "/NoDataOptn/" in s.path and "/NoData4/" not in s.path]
        if not chosen:
            chosen = [s for s in ordered if s.tag.upper() == "NODATA"]
        return chosen[:1]

    # For non-ND values, never select specs from the NoData choice branch.
    non_nd = [
        s for s in ordered
        if s.tag.upper() not in ND_TAGS
        and s.tag.upper() != "SGN"
        and "/NoDataOptn/" not in s.path
    ]
    if not non_nd:
        return []
    if _is_date(value):
        dt = [s for s in non_nd if s.tag == "Dt"]
        if dt:
            return [dt[0]]
    return [non_nd[0]]


def _apply_amount_attributes_if_needed(node: etree._Element, value: str, currency: str) -> None:
    """
    Annex 2 auth.099 amount leaves are represented as .../Amt and require Ccy.
    We set Ccy only on Amt leaves for non-ND values.
    """
    if _is_nd(value):
        return
    if etree.QName(node).localname == "Amt":
        node.set("Ccy", currency)


def apply_header_code(root: etree._Element, df: pd.DataFrame, code: str, specs_by_code: Dict[str, List[MappingSpec]], ns: str, order_index: Dict[Tuple[str, ...], List[str]], currency: str) -> None:
    if code not in df.columns or code not in specs_by_code:
        return
    specs = [s for s in specs_by_code[code] if RECORD_ANCHOR not in s.path]
    if not specs:
        return

    values = [_safe_str(v) for v in df[code].tolist()]
    non_blank = sorted({v for v in values if v != ""})
    if len(non_blank) > 1:
        raise ValueError(f"Header-level field '{code}' varies across rows: {non_blank[:5]}")

    value = values[0] if values else ""
    chosen = select_specs_for_value(specs, value)
    if not chosen:
        min_occ = min((_parse_multiplicity(s.multiplicity)[0] for s in specs), default=0)
        if min_occ > 0:
            raise ValueError(f"Mandatory header field '{code}' has no valid mapping branch for value '{value}'")
        return

    spec = chosen[0]
    min_occ, _ = _parse_multiplicity(spec.multiplicity)
    if value == "":
        if min_occ > 0:
            raise ValueError(f"Mandatory header field '{code}' is blank")
        return

    leaf = get_or_create_path_singleton(root, spec.path, ns, order_index)
    leaf.text = value
    _apply_amount_attributes_if_needed(leaf, value, currency)


def apply_record_code(record_node: etree._Element, row: pd.Series, code: str, specs_by_code: Dict[str, List[MappingSpec]], record_root_path: str, ns: str, order_index: Dict[Tuple[str, ...], List[str]], currency: str) -> None:
    if code not in row.index or code not in specs_by_code:
        return

    value = _safe_str(row.get(code))
    specs = [s for s in specs_by_code[code] if RECORD_ANCHOR in s.path]
    if not specs:
        return

    chosen_specs = select_specs_for_value(specs, value)
    if not chosen_specs:
        min_occ = min((_parse_multiplicity(s.multiplicity)[0] for s in specs), default=0)
        if min_occ > 0:
            raise ValueError(f"Mandatory record field '{code}' has no valid mapping branch for value '{value}'")
        return

    spec = chosen_specs[0]
    min_occ, max_occ = _parse_multiplicity(spec.multiplicity)

    values = _split_multi_value(value)
    if not values:
        if min_occ > 0:
            raise ValueError(f"Mandatory record field '{code}' is blank")
        return

    if max_occ is not None and len(values) > max_occ:
        raise ValueError(f"Field '{code}' has {len(values)} values > max_occ={max_occ}")

    rel_parts = _relative_parts_from_record(spec.path, record_root_path)
    if not rel_parts:
        # path is exactly record node
        if min_occ > 0:
            raise ValueError(f"Unexpected mapping for '{code}': leaf at record root not supported")
        return

    # parent path under record node (singleton chain)
    parent = record_node
    built = _path_tuple(record_root_path)
    for p in rel_parts[:-1]:
        parent = _get_or_create_singleton(parent, p, ns, order_index, built)
        built = (*built, p)

    leaf_tag = rel_parts[-1]
    for v in values:
        # if repeatable allowed, append new; otherwise singleton
        leaf = (
            _append_ordered_new(parent, leaf_tag, ns, order_index, built)
            if (max_occ is None or (max_occ is not None and max_occ > 1 and len(values) > 1))
            else _get_or_create_singleton(parent, leaf_tag, ns, order_index, built)
        )
        leaf.text = v
        _apply_amount_attributes_if_needed(leaf, v, currency)


def build_annex2_tree(df: pd.DataFrame, code_order: List[str], specs_by_code: Dict[str, List[MappingSpec]], ns: str, currency: str, xsd_path: Optional[str]) -> etree._Element:
    if df.empty:
        raise ValueError("Input projected CSV is empty")

    order_index = build_order_index(specs_by_code)
    record_root_path = find_record_root_path(specs_by_code)

    nsmap = {None: ns, "xsi": "http://www.w3.org/2001/XMLSchema-instance"}
    root = etree.Element(f"{{{ns}}}Document", nsmap=nsmap)
    if xsd_path:
        xsd_name = Path(xsd_path).name
        root.set("{http://www.w3.org/2001/XMLSchema-instance}schemaLocation", f"{ns} {xsd_name}")

    # Header/singleton codes once with cross-row consistency checks.
    # Important: Annex 2 header-required codes (e.g., RREL1/RREL6) may not be
    # listed in the Record-oriented code-order list, so include all mapped
    # non-record codes present in the CSV.
    header_codes_from_mapping = []
    for code, specs in specs_by_code.items():
        if code not in df.columns:
            continue
        if any(RECORD_ANCHOR not in s.path for s in specs):
            first_idx = min(s.row_idx for s in specs if RECORD_ANCHOR not in s.path)
            header_codes_from_mapping.append((first_idx, code))
    header_codes_from_mapping = [c for _, c in sorted(header_codes_from_mapping)]

    header_codes = []
    seen = set()
    for c in header_codes_from_mapping + code_order:
        if c in seen:
            continue
        seen.add(c)
        header_codes.append(c)

    for code in header_codes:
        apply_header_code(root, df, code, specs_by_code, ns, order_index, currency)

    # Row-level repeating record: a fresh UndrlygXpsrRcrd branch per row.
    record_codes_from_mapping = []
    for code, specs in specs_by_code.items():
        if code not in df.columns:
            continue
        if any(RECORD_ANCHOR in s.path for s in specs):
            first_idx = min(s.row_idx for s in specs if RECORD_ANCHOR in s.path)
            record_codes_from_mapping.append((first_idx, code))
    record_codes_from_mapping = [c for _, c in sorted(record_codes_from_mapping)]

    record_codes = []
    seen_rec = set()
    for c in record_codes_from_mapping + code_order:
        if c in seen_rec:
            continue
        seen_rec.add(c)
        record_codes.append(c)

    for _, row in df.iterrows():
        record_node = create_new_record_node(root, record_root_path, ns, order_index)
        for code in record_codes:
            apply_record_code(record_node, row, code, specs_by_code, record_root_path, ns, order_index, currency)

    return root


def load_code_order(path: str) -> List[str]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    # Repo currently stores Annex2 order under Record. Keep ESMA_Annex2 fallback.
    if isinstance(y.get("ESMA_Annex2"), list):
        return [str(x).strip() for x in y["ESMA_Annex2"] if str(x).strip()]
    if isinstance(y.get("Record"), list):
        return [str(x).strip() for x in y["Record"] if str(x).strip()]
    raise ValueError("code-order YAML must contain ESMA_Annex2 list or Record list")


def main() -> None:
    ap = argparse.ArgumentParser(description="Annex2 XML builder (auth.099)")
    ap.add_argument("--input", required=True, help="Projected Annex2 CSV from Gate 4")
    ap.add_argument("--output", required=True, help="Output XML path")
    ap.add_argument("--mapping-workbook", required=True, help="Annex2 mapping workbook (.xlsx)")
    ap.add_argument("--sheet", default="DRAFT1auth.099.001.04", help="Workbook sheet name")
    ap.add_argument("--code-order-yaml", required=True, help="Code order YAML")
    ap.add_argument("--xsd", default=None, help="Annex2 XSD path (optional but recommended)")
    ap.add_argument("--performance-mode", choices=["PRF", "NPRF"], default="PRF", help="Select performing/non-performing mapping branch")
    ap.add_argument("--currency", default="GBP", help="Currency code for Annex2 Amt leaves (Ccy attribute)")
    args = ap.parse_args()
    input_name = Path(args.input).name.lower()
    if "projected" in input_name and "delivery_ready" not in input_name:
        logging.warning(
            "Input appears to be projected CSV without Gate 4b normalization: %s. "
            "Recommended input is *_delivery_ready.csv.",
            args.input,
        )

    if not Path(args.input).exists():
        raise FileNotFoundError(args.input)
    if not Path(args.mapping_workbook).exists():
        raise FileNotFoundError(args.mapping_workbook)
    if not Path(args.code_order_yaml).exists():
        raise FileNotFoundError(args.code_order_yaml)

    df = pd.read_csv(args.input, dtype=str)
    code_order = load_code_order(args.code_order_yaml)
    specs_by_code = load_mapping_specs(args.mapping_workbook, args.sheet, args.performance_mode)

    ns = get_namespace_from_xsd(args.xsd)
    root = build_annex2_tree(df, code_order, specs_by_code, ns, args.currency, args.xsd)
    tree = etree.ElementTree(root)
    tree.write(args.output, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print(f"Generated: {args.output}")

    if args.xsd:
        schema = etree.XMLSchema(etree.parse(args.xsd))
        if schema.validate(root):
            print("XSD Validation: PASSED")
        else:
            print("XSD Validation: FAILED")
            for e in schema.error_log:
                print(f"  - Line {e.line}: {e.message}")
            sys.exit(1)


if __name__ == "__main__":
    main()
