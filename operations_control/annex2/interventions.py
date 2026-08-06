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


def _csv_nd_count_for(delivery_csv: Path, fields: tuple) -> int:
    """ND cells the delivery CSV itself supplies for the named fields.

    Phase 2: RREL20/RREL21 are declared in
    ``config/regime/annex2_delivery_rules.yaml`` and populated by Gate 4b, so
    the ND nodes under ``ScndryOblgrIncm`` in the XML are now SOURCED, not
    builder-inserted. Counting XML nodes alone can no longer tell the two
    apart — this is what distinguishes them.
    """
    import pandas as pd
    total = 0
    for chunk in pd.read_csv(delivery_csv, dtype=str,
                             usecols=lambda c: c in fields, chunksize=20000):
        for field in fields:
            if field in chunk.columns:
                total += int(chunk[field].fillna("").str.fullmatch(r"ND[1-5]").sum())
    return total


def _rrel12_coercion_count(delivery_csv: Path) -> int:
    """Rows whose RREL12 value is not a 4-digit year.

    Phase 2: the builder no longer coerces these to a constant year — it routes
    them to the NoData branch. The count is still worth reporting, because a
    value that cannot enter its typed branch is an operator-visible event; only
    the treatment changed.
    """
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


#: Codes the XSD carries as a ``Ccy`` ATTRIBUTE on the amount they qualify
#: rather than as an element of their own — all three are currencies. A value in
#: one of these delivery-ready columns legitimately produces NO XML node, so it
#: must be excluded from the XML tie-out or the reconciliation reports a phantom
#: shortfall. Declared in
#: ``config/regime/annex2_delivery_rules.yaml::reconciliation_scope.non_emitting_fields``;
#: mirrored here because this module reads artefacts, not rules.
NON_EMITTING_CODES = ("RREL18", "RREL28", "RREC22")


def analyse(delivery_csv: Path, xml_path: Path) -> Dict[str, Any]:
    """Produce the intervention evidence document."""
    csv_nd = _csv_nd_count(delivery_csv)
    non_emitting_nd = _csv_nd_count_for(delivery_csv, NON_EMITTING_CODES)
    # The tie-out must compare like with like: only ND that CAN become a node.
    csv_nd_emitting = max(0, csv_nd - non_emitting_nd)
    sourced_scndry_nd = _csv_nd_count_for(delivery_csv, ("RREL20", "RREL21"))
    coerced_rrel12 = _rrel12_coercion_count(delivery_csv)
    x = _scan_xml(xml_path)
    injected_nd = max(0, x["nodata_total"] - csv_nd_emitting)

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
    # Only the PORTION the delivery data did not supply is a builder
    # intervention. Since Phase 2 the rules supply all of it, so this reports
    # nothing — which is the correct answer, not a missing check.
    scndry_unsourced = max(0, x["nodata_scndry_incm"] - sourced_scndry_nd)
    if scndry_unsourced:
        interventions.append(Intervention(
            code="RREL20/RREL21", intervention_type="nd_insertion",
            count=scndry_unsourced,
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
            code="RREL12", intervention_type="nodata_routing",
            count=coerced_rrel12,
            example_source_state="geography classification year not a "
                                 "4-digit year",
            generated_treatment="reported as no-data; the value is NOT "
                                "replaced with a substitute year",
            reason="the schema branch accepts a 4-digit year only, so a value "
                   "that is not one cannot be placed there",
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
        # Split so the tie-out is against what the XML can actually contain.
        "nodata_on_non_emitting_fields": non_emitting_nd,
        "nodata_in_delivery_csv_emitting": csv_nd_emitting,
        "non_emitting_codes": list(NON_EMITTING_CODES),
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
