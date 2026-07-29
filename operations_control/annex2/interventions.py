"""operations_control.annex2.interventions — builder-intervention evidence (I4).

Deterministic before/after comparison between the delivery-ready CSV (what the
governed pipeline decided) and the generated XML (what the Gate 5 builder
emitted), plus targeted scans for the builder's documented automatic
behaviours. The builder's output is NEVER altered — this module only counts,
classifies and persists what happened so no intervention is hidden from the
operator or the audit trail.

Memory-safe on multi-hundred-MB XML via lxml iterparse with element clearing.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

YEAR_RE = re.compile(r"^\d{4}$")
ND_RE = re.compile(r"^ND[1-5]$")


@dataclass
class Intervention:
    code: str                      # annex code or structural group
    intervention_type: str
    count: int
    example_source_state: str
    generated_treatment: str
    reason: str
    severity: str                  # info | warning
    review_required: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _csv_nd_count(delivery_csv: Path) -> int:
    import pandas as pd
    total = 0
    for chunk in pd.read_csv(delivery_csv, dtype=str, chunksize=5000,
                             low_memory=False):
        total += int(chunk.apply(
            lambda col: col.fillna("").str.fullmatch(r"ND[1-5]").sum()).sum())
    return total


def _rrel12_coercion_count(delivery_csv: Path) -> int:
    """Rows whose RREL12 value is not a 4-digit year — the builder coerces
    those to a constant year (documented behaviour)."""
    import pandas as pd
    total = 0
    for chunk in pd.read_csv(delivery_csv, dtype=str, usecols=lambda c: c == "RREL12",
                             chunksize=20000):
        if "RREL12" not in chunk.columns:
            return 0
        vals = chunk["RREL12"].fillna("")
        total += int((~vals.str.fullmatch(r"\d{4}") & (vals != "")).sum())
    return total


def _scan_xml(xml_path: Path) -> Dict[str, int]:
    """One pass over the XML: NoData leaves by structural group, currency
    attributes, record and NPRF counts."""
    from lxml import etree
    counts = {"records": 0, "nodata_total": 0, "nodata_hstrcl_colltn": 0,
              "nodata_scndry_incm": 0, "nodata_nonprfrmg": 0,
              "nodata_other": 0, "ccy_attributes": 0, "nonprfrmg_blocks": 0}
    # Track ancestry cheaply with a stack of localnames.
    stack: List[str] = []
    for event, elem in etree.iterparse(str(xml_path), events=("start", "end")):
        tag = etree.QName(elem).localname
        if event == "start":
            stack.append(tag)
            continue
        # end event
        if tag == "UndrlygXpsrRcrd":
            counts["records"] += 1
        elif tag == "NonPrfrmgLn":
            counts["nonprfrmg_blocks"] += 1
        if elem.get("Ccy"):
            counts["ccy_attributes"] += 1
        if len(elem) == 0 and elem.text and ND_RE.fullmatch(elem.text.strip()):
            counts["nodata_total"] += 1
            if "HstrclColltn" in stack:
                counts["nodata_hstrcl_colltn"] += 1
            elif "ScndryOblgrIncm" in stack:
                counts["nodata_scndry_incm"] += 1
            elif "NonPrfrmgLn" in stack:
                counts["nodata_nonprfrmg"] += 1
            else:
                counts["nodata_other"] += 1
        stack.pop()
        # Free memory: clear completed records.
        if tag == "UndrlygXpsrRcrd":
            elem.clear()
            parent = elem.getparent()
            if parent is not None:
                while elem.getprevious() is not None:
                    del parent[0]
    return counts


def analyse(delivery_csv: Path, xml_path: Path) -> Dict[str, Any]:
    """Produce the intervention evidence document."""
    csv_nd = _csv_nd_count(delivery_csv)
    coerced_rrel12 = _rrel12_coercion_count(delivery_csv)
    x = _scan_xml(xml_path)
    injected_nd = max(0, x["nodata_total"] - csv_nd)

    interventions: List[Intervention] = []
    if x["nodata_hstrcl_colltn"]:
        interventions.append(Intervention(
            code="HstrclColltn", intervention_type="nd_insertion",
            count=x["nodata_hstrcl_colltn"],
            example_source_state="no historical collection columns in the "
                                 "delivery data",
            generated_treatment="ND5 'no data' entries for each monthly "
                                "collection slot",
            reason="the schema requires the historical collection blocks; "
                   "the builder fills them with permitted no-data values",
            severity="warning", review_required=True))
    if x["nodata_scndry_incm"]:
        interventions.append(Intervention(
            code="RREL20/RREL21", intervention_type="nd_insertion",
            count=x["nodata_scndry_incm"],
            example_source_state="secondary income not present in the "
                                 "delivery data",
            generated_treatment="ND5 'no data' for secondary income and its "
                                "verification",
            reason="optional fields represented with permitted no-data values",
            severity="warning", review_required=True))
    if x["nodata_nonprfrmg"]:
        interventions.append(Intervention(
            code="NonPrfrmgLn", intervention_type="nd_insertion",
            count=x["nodata_nonprfrmg"],
            example_source_state="no non-performing data columns supplied",
            generated_treatment="ND5 scaffolding for the non-performing "
                                "section",
            reason="the schema branch is mandatory in shape; filled with "
                   "permitted no-data values",
            severity="warning", review_required=True))
    if coerced_rrel12:
        interventions.append(Intervention(
            code="RREL12", intervention_type="value_coercion",
            count=coerced_rrel12,
            example_source_state="geography classification year not a "
                                 "4-digit year",
            generated_treatment="replaced with a fixed year by the generator",
            reason="documented generator behaviour for non-year values",
            severity="warning", review_required=True))
    if x["ccy_attributes"]:
        interventions.append(Intervention(
            code="Amt@Ccy", intervention_type="static_attribute",
            count=x["ccy_attributes"],
            example_source_state="amounts without an explicit currency",
            generated_treatment="currency attribute stamped from the run "
                                "setting (default GBP)",
            reason="schema requires a currency on amount values",
            severity="info", review_required=False))

    review_count = sum(i.count for i in interventions if i.review_required)
    return {
        "records": x["records"],
        "nodata_in_delivery_csv": csv_nd,
        "nodata_in_xml": x["nodata_total"],
        "nodata_injected_by_builder": injected_nd,
        "nonperforming_blocks": x["nonprfrmg_blocks"],
        "interventions": [i.to_dict() for i in interventions],
        "review_required_instances": review_count,
        "summary_sentence": (
            "No automatic values were inserted by the generator."
            if review_count == 0 else
            f"Automatic values were inserted in {review_count:,} field "
            "instances. Review is required before publication."),
    }
