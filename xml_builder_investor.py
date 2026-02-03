#!/usr/bin/env python3
"""
xml_builder_investor.py

ESMA Annex 12 XML Builder (Investor / Significant Event) — Final Version
FIXED:
 - Namespace enforcement on Root element
 - Sgn tag suppression for IVSS16/17 etc.
 - NoDataOptn wrapper handling
 - Full repeatable group logic restored
"""

import argparse
import pandas as pd
import yaml
from lxml import etree
import sys
import re
from typing import Dict, List, Tuple, Any
import math

ANNEX12_NS = "urn:esma:xsd:DRAFT1auth.098.001.04"
NSMAP = {None: ANNEX12_NS}
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------- IO ----------------

def get_namespace_from_xsd(xsd_path: str) -> str:
    """Reads the targetNamespace dynamically from the XSD file."""
    default_ns = "urn:esma:xsd:DRAFT1auth.098.001.04"
    if not xsd_path:
        return default_ns
    try:
        tree = etree.parse(xsd_path)
        root = tree.getroot()
        ns = root.get("targetNamespace")
        if ns:
            return ns
        else:
            return default_ns
    except Exception as e:
        print(f"WARNING: Could not read namespace from XSD ({e}). Using default.")
        return default_ns

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _clean_tag(tag) -> str:
    if tag is None or (isinstance(tag, float) and math.isnan(tag)):
        return ""
    t = str(tag).strip()
    if t.startswith("<") and t.endswith(">"):
        t = t[1:-1].strip()
    return t

def _split_path(path_str: str) -> List[str]:
    return [p for p in (path_str or "").strip("/").split("/") if p]

def _path_key(parts: List[str]) -> Tuple[str, ...]:
    if parts and parts[0] != "Document":
        parts = ["Document"] + parts
    return tuple(parts)

def _safe_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    s = str(val).strip()
    if s.lower() == "nan":
        return ""
    return s

def build_order_index_from_mapping_df(df: pd.DataFrame) -> Dict[Tuple[str, ...], List[str]]:
    order_index: Dict[Tuple[str, ...], List[str]] = {}
    for _, row in df.iterrows():
        path_str = _safe_str(row.get("PATH", ""))
        if not path_str:
            continue
        parts = _split_path(path_str)
        if len(parts) < 2:
            continue
        parent = parts[:-1]
        child = parts[-1]
        pkey = _path_key(parent)
        order_index.setdefault(pkey, [])
        if child not in order_index[pkey]:
            order_index[pkey].append(child)
    return order_index

def load_mapping(path: str):
    try:
        if path.endswith(".xlsx"):
            df = pd.read_excel(path, sheet_name="DRAFT1auth.098.001.04", header=3, dtype=str)
        else:
            df = pd.read_csv(path, header=3, dtype=str)
            if "RTS Field code" not in df.columns:
                df = pd.read_csv(path, header=0, dtype=str)
    except Exception as e:
        print(f"CRITICAL: Error reading mapping file '{path}': {e}")
        sys.exit(1)

    df.columns = [str(c).strip() for c in df.columns]
    order_index = build_order_index_from_mapping_df(df)
    mapping = {} 
    for _, row in df.iterrows():
        code_cell = _safe_str(row.get("RTS Field code", ""))
        if not code_cell:
            continue
        codes = [c.strip() for c in code_cell.splitlines() if c.strip()]
        tag = _clean_tag(row.get("XML TAG", ""))
        path_str = _safe_str(row.get("PATH", ""))
        mult = _safe_str(row.get("MULTIPLICITY", row.get("Multiplicity", ""))) or "1"

        if not tag or not path_str:
            continue
        spec = {"tag": tag, "path": path_str, "multiplicity": mult}
        for code in codes:
            mapping.setdefault(code, []).append(spec)
    return mapping, order_index


# -------------- Helpers --------------

def parse_multiplicity(mult_str: str):
    m = (mult_str or "").strip().lower().replace(" ", "").replace("[", "").replace("]", "")
    if ".." in m:
        a, b = m.split("..", 1)
        try:
            return int(a), (float("inf") if b == "n" else int(b))
        except ValueError:
            return 0, 1
    else:
        try:
            v = int(m)
            return v, v
        except ValueError:
            return 1, 1

def clean_bool(val: str) -> str:
    v = str(val).strip().lower()
    if v in ["y", "yes", "true", "1"]:
        return "true"
    if v in ["n", "no", "false", "0"]:
        return "false"
    return str(val)

def is_nd(val: str) -> bool:
    return str(val).strip().upper().startswith("ND")

def check_mandatory(code: str, val: str, min_occurs: int, tag: str):
    if min_occurs >= 1 and not val:
        raise ValueError(f"VALIDATION FAILED: {code} ({tag}) is Mandatory but empty.")
    if min_occurs >= 1 and "Dt" in tag and val != "ND5":
        if not DATE_PATTERN.match(val):
            raise ValueError(f"VALIDATION FAILED: Date field {code} ('{val}') must be YYYY-MM-DD.")

def _localname(el) -> str:
    return etree.QName(el).localname

def _ordered_insert(parent: etree._Element, child_tag_local: str, order_index: Dict[Tuple[str, ...], List[str]], parent_parts: List[str]) -> etree._Element:
    existing = parent.find(f"{{{ANNEX12_NS}}}{child_tag_local}")
    if existing is not None:
        return existing

    pkey = _path_key(parent_parts)
    seq = order_index.get(pkey)
    new_el = etree.Element(f"{{{ANNEX12_NS}}}{child_tag_local}")

    if not seq:
        parent.append(new_el)
        return new_el

    try:
        target_pos = seq.index(child_tag_local)
    except ValueError:
        parent.append(new_el)
        return new_el

    children = list(parent)
    insert_at = len(children)
    for i, ch in enumerate(children):
        ln = _localname(ch)
        if ln in seq:
            if seq.index(ln) > target_pos:
                insert_at = i
                break
    parent.insert(insert_at, new_el)
    return new_el

def get_or_create_path(root: etree._Element, path_str: str, order_index: Dict[Tuple[str, ...], List[str]]) -> etree._Element:
    parts = _split_path(path_str)
    current = root
    start_idx = 1 if parts and parts[0] == "Document" else 0
    built_parts = ["Document"]
    for p in parts[start_idx:]:
        current = _ordered_insert(current, p, order_index, built_parts)
        built_parts.append(p)
    return current

def set_node_value(parent_node, code, val, rules, currency):
    if code in rules.get("boolean_fields", []): val = clean_bool(val)
    
    if not is_nd(val):
        leaf_tag = _localname(parent_node)
        is_monetary = code in rules.get("monetary_fields", []) or leaf_tag.endswith("Amt") or leaf_tag in ["Dltn", "Rpchs", "PrncplShrtfll", "IntrstShrtfll"]
        if is_monetary:
            try:
                v = float(val)
                abs_v = abs(v)
                parent_node.text = str(abs_v)  # Absolute value
                parent_node.set("Ccy", currency)
                if v < 0:
                    parent_node.set("Sgn", "true")  # Negative attr
                return  # Early exit for monetary
            except ValueError:
                pass
    parent_node.text = val

    if not is_nd(val):
        leaf_tag = _localname(parent_node)
        
        # Only add Ccy if the tag implies it is a monetary amount.
        # This prevents the "Attribute Ccy not allowed" error on wrapper tags.
        is_amt_tag = leaf_tag.endswith("Amt")
        is_special_monetary = leaf_tag in ["Dltn", "Rpchs", "PrncplShrtfll", "IntrstShrtfll"]
        
        if is_amt_tag or is_special_monetary:
            parent_node.set("Ccy", currency)

def select_mapping_spec(code: str, val: str, mapping: dict, rules: dict, report_type: str = "NEWCORR") -> dict:
    specs = mapping.get(code, [])
    if not specs: return {}

    clean_code = str(code).strip()

    # --- SCALABLE FIX: Read Exclusion List from YAML ---
    # We no longer hardcode the list here. We read it from annex12_rules.yaml.
    no_sign_fields = rules.get("implicit_sign_fields", [])
    
    if clean_code in no_sign_fields:
        original_count = len(specs)
        specs = [s for s in specs if s["tag"] != "Sgn"]
        if len(specs) < original_count:
            # Optional debug print, good for verifying your YAML config is working
            # print(f"DEBUG: Removed <Sgn> tag for field {clean_code}")
            pass

    nd = is_nd(val)

    # --- Handle NoDataOptn vs NoData logic ---
    if nd:
        nd_specs = [s for s in specs if s["tag"] in ["NoData", "NoData4"]]
        if nd_specs: return nd_specs[0]
    else:
        specs = [s for s in specs if s["tag"] not in ["NoData", "NoDataOptn", "NoData4"]]

    # --- Report Type Filtering ---
    if report_type == "NEWCORR": specs = [s for s in specs if '/NewCrrctn/' in s.get('path', '')]
    elif report_type == "CXL": specs = [s for s in specs if '/Cxl/' in s.get('path', '')]
    if not specs: return {}

    # --- Leaf Selection Strategy ---
    if clean_code == "IVSR6":
        target = "Pctg" if "%" in str(val) else "Nmrcl"
        for s in specs: 
            if s["tag"] == target: return s

    # Ensure we pick 'Amt' for monetary fields (generic catch-all)
    if clean_code in rules.get("monetary_fields", []) or clean_code == "IVSF4":
        for s in specs: 
            if s["tag"] == "Amt": return s

    leaf_candidates = [s for s in specs if s["tag"] not in {"Val", "NoDataOptn"}]
    if leaf_candidates:
        return sorted(leaf_candidates, key=lambda x: len(_split_path(x["path"])))[-1]

    return sorted(specs, key=lambda x: len(_split_path(x["path"])))[-1]

# -------------- Build Tree --------------

def build_tree(row, code_order, mapping, rules, currency, order_index, report_type="NEWCORR"):
    # --- FIX 3: Force Namespace in the Element definition ---
    root = etree.Element(f"{{{ANNEX12_NS}}}Document", nsmap=NSMAP)

    repeatable_groups = rules.get("repeatable_groups", {})
    member_to_group = {}
    for g_name, g_data in repeatable_groups.items():
        for m in g_data["members"]:
            member_to_group[m] = g_name

    emitted_singletons = set()
    data_map: Dict[str, Any] = {}
    group_lengths: Dict[str, int] = {}

    for code in code_order:
        raw_val = str(row.get(code, "")).strip()
        if raw_val == "nan":
            raw_val = ""

        group_name = member_to_group.get(code)
        if group_name:
            parts = [v.strip() for v in raw_val.split("|")] if raw_val else []
            data_map[code] = parts
            group_lengths[group_name] = max(group_lengths.get(group_name, 0), len(parts))
        else:
            data_map[code] = raw_val

    # Singles
    for code in code_order:
        if code not in mapping or code in member_to_group:
            continue
        raw_val = data_map[code]
        spec = select_mapping_spec(code, raw_val, mapping, rules, report_type)
        if not spec:
            continue

        min_occ, max_occ = parse_multiplicity(spec["multiplicity"])
        is_list = max_occ > 1
        vals = [v.strip() for v in raw_val.split("|")] if (is_list and "|" in raw_val) else [raw_val]

        if not vals or (len(vals) == 1 and not vals[0]):
            check_mandatory(code, "", min_occ, spec["tag"])
            continue

        full_path = spec["path"]
        path_parts = _split_path(full_path)
        parent_path = "/" + "/".join(path_parts[:-1])
        singleton_key = (parent_path, spec["tag"])
        if max_occ == 1 and singleton_key in emitted_singletons:
            continue

        parent_node = get_or_create_path(root, parent_path, order_index)
        for val in vals:
            if not val:
                continue
            check_mandatory(code, val, min_occ, spec["tag"])
            parent_parts = _split_path(parent_path)
            if parent_parts and parent_parts[0] != "Document":
                parent_parts = ["Document"] + parent_parts
            leaf = _ordered_insert(parent_node, spec["tag"], order_index, parent_parts)
            set_node_value(leaf, code, val, rules, currency)
            if max_occ == 1:
                emitted_singletons.add(singleton_key)

    # Repeatable groups - FULL LOGIC RESTORED
    for group_name, group_def in repeatable_groups.items():
        count = group_lengths.get(group_name, 0)
        if count == 0:
            continue

        members = group_def["members"]
        container_tag = group_def["container_tag"]

        first_code = members[0]
        anchor_spec = select_mapping_spec(first_code, "DUMMY", mapping, rules, report_type)
        if not anchor_spec:
            continue

        full_path = anchor_spec["path"]
        path_parts = _split_path(full_path)

        try:
            container_idx = path_parts.index(container_tag)
            parent_path_parts = path_parts[:container_idx]
            parent_path = "/" + "/".join(parent_path_parts)
        except ValueError:
            print(f"ERROR: Container tag '{container_tag}' not found in path '{full_path}'")
            continue

        group_root = get_or_create_path(root, parent_path, order_index)

        for i in range(count):
            parent_parts = _split_path(parent_path)
            if parent_parts and parent_parts[0] != "Document":
                parent_parts = ["Document"] + parent_parts

            container_node = etree.Element(f"{{{ANNEX12_NS}}}{container_tag}")
            pkey = _path_key(parent_parts)
            seq = order_index.get(pkey, [])
            
            insert_at = len(group_root)
            if container_tag in seq:
                pos = seq.index(container_tag)
                children = list(group_root)
                for j, ch in enumerate(children):
                    ln = _localname(ch)
                    if ln in seq and seq.index(ln) > pos:
                        insert_at = j
                        break
            group_root.insert(insert_at, container_node)

            for code in members:
                if code not in mapping:
                    continue
                vals = data_map.get(code, [])
                val = vals[i] if i < len(vals) else ""
                if not val:
                    continue

                spec = select_mapping_spec(code, val, mapping, rules, report_type)
                if not spec:
                    continue

                min_occ, _ = parse_multiplicity(spec["multiplicity"])
                check_mandatory(code, val, min_occ, spec["tag"])

                field_path_parts = _split_path(spec["path"])
                try:
                    c_idx = field_path_parts.index(container_tag)
                    intermediate_parts = field_path_parts[c_idx + 1:-1]
                except ValueError:
                    intermediate_parts = []

                current_parent = container_node
                built = ["Document"] + parent_path_parts + [container_tag]

                for part in intermediate_parts:
                    current_parent = _ordered_insert(current_parent, part, order_index, built)
                    built.append(part)

                leaf = _ordered_insert(current_parent, spec["tag"], order_index, built)
                set_node_value(leaf, code, val, rules, currency)

    return root


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--code-order-yaml", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--currency", default="GBP")
    ap.add_argument("--report-type", choices=["NEWCORR", "CXL"], default="NEWCORR",
                    help="Report type: NEWCORR for new/correction, CXL for cancellation")
    ap.add_argument("--xsd", help="Path to XSD for validation")
    args = ap.parse_args()

    global ANNEX12_NS, NSMAP
    ANNEX12_NS = get_namespace_from_xsd(args.xsd)
    NSMAP = {None: ANNEX12_NS}
    print(f"INFO: Using XML Namespace: {ANNEX12_NS}")

    df = pd.read_csv(args.input)
    if df.empty:
        raise ValueError("CSV Empty")
    row = df.iloc[0]

    mapping, order_index = load_mapping(args.mapping)
    rules = load_yaml(args.rules)
    code_order = load_yaml(args.code_order_yaml).get("ESMA_Annex12", [])
    print("Loaded implicit_sign_fields:", rules.get("implicit_sign_fields", []))

    root = build_tree(row, code_order, mapping, rules, args.currency, order_index, args.report_type)
    tree = etree.ElementTree(root)
    tree.write(args.output, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print(f"Generated: {args.output}")

    if args.xsd:
        xsd_doc = etree.parse(args.xsd)
        schema = etree.XMLSchema(xsd_doc)
        if schema.validate(root):
            print("XSD Validation: PASSED")
        else:
            print("XSD Validation: FAILED")
            for e in schema.error_log:
                print(f"  - Line {e.line}: {e.message}")
            sys.exit(1)

if __name__ == "__main__":
    main()