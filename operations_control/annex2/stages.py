"""operations_control.annex2.stages — governed invocation of the proven route (I1).

Thin subprocess wrappers around the authoritative, unmodified components. Each
returns a :class:`StageOutcome` the engine translates into a Governed Agent
Result. Deterministic artefact paths + atomic XML finalisation make reruns
idempotent (a completed stage with intact artefacts is skipped, never
duplicated).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]

NORMALIZER = REPO / "engine/gate_4b_delivery/annex2_delivery_normalizer.py"
XML_BUILDER = REPO / "engine/gate_5_delivery/xml_builder_annex2.py"
RULES_PATH = REPO / "config/regime/annex2_delivery_rules.yaml"
XSD_PATH = REPO / "config/system/DRAFT1auth.099.001.04_1.3.0.xsd"
WORKBOOK = REPO / ("DRAFT1auth.099.001.04_non-ABCP Underlying Exposure "
                   "Report_Version_1.3.1.xlsx")
WORKBOOK_SHEET = "DRAFT1auth.099.001.04"
CODE_ORDER = REPO / "config/system/esma_code_order.yaml"
XML_NAME = "annex2_submission.xml"

#: Plain-English translations for the normaliser's issue categories.
_ISSUE_SENTENCES = {
    "missing mandatory delivery value":
        "A required value is missing from the data.",
    "enum / code mapping":
        "A value does not match the regulator's allowed codes.",
    "ND restriction":
        "A 'no data' code was used where the regulator does not allow it.",
    "numeric precision": "A number has more digits than the regulator allows.",
    "pattern / identifier": "An identifier is not in the required format.",
    "XML choice-branch issues": "A value cannot be represented in the report "
                                "structure.",
    "other": "A value did not pass the delivery checks.",
}


@dataclass
class StageOutcome:
    ok: bool
    blocking: bool = False
    summary: str = ""
    warnings: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    artefacts: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _file_version(path: Path) -> str:
    try:
        st = Path(path).stat()
        return f"{Path(path).name}@{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return str(path)


class Annex2Stages:
    """Real runners for the authoritative route. Tests substitute a stub with
    the same method signatures (the engine seam)."""

    def run_projection(self, *, central_canonical: str, out_dir: Path,
                       client_config: Optional[Path] = None) -> StageOutcome:
        """Invoke the EXISTING projector via the existing assembler seam."""
        from engine.assembler_agent import build_regime_command
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_regime_command(
            central_canonical, out_dir, "ESMA_Annex2",
            config=(str(client_config) if client_config else None),
            output_prefix="annex2", allow_unreviewed=True)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(REPO))
        elapsed = round(time.time() - t0, 1)
        projected = sorted(out_dir.glob("*_ESMA_Annex2_projected.csv"))
        report_p = sorted(out_dir.glob("*_ESMA_Annex2_projection_report.json"))
        if proc.returncode != 0 or not projected:
            blockers, evidence = self._translate_projection_failure(
                proc.stderr or "")
            return StageOutcome(
                ok=False, blocking=True,
                summary="The regulatory data could not be projected.",
                blockers=blockers, evidence=evidence,
                metrics={"elapsed_s": elapsed, "returncode": proc.returncode},
                report={"stderr_tail": (proc.stderr or "")[-2000:]})
        report = {}
        if report_p:
            try:
                report = json.loads(report_p[0].read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                report = {}
        return StageOutcome(
            ok=True,
            summary=(f"Regulatory data projected: "
                     f"{report.get('output_rows', '?'):,} records, "
                     f"{report.get('output_fields', '?')} fields."
                     if report.get("output_rows") else
                     "Regulatory data projected."),
            artefacts={"projected_csv": str(projected[0]),
                       "projection_report": (str(report_p[0]) if report_p else "")},
            metrics={"elapsed_s": elapsed,
                     "records": report.get("output_rows"),
                     "fields": report.get("output_fields"),
                     "missing_fields": len(report.get("missing_fields") or [])},
            report=report)

    def run_normalisation(self, *, projected_csv: str, out_dir: Path,
                          rules_path: Path = RULES_PATH) -> StageOutcome:
        """Invoke the EXISTING Gate 4b delivery normaliser, unmodified."""
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(NORMALIZER), "--input", str(projected_csv),
               "--rules", str(rules_path), "--output-dir", str(out_dir)]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(REPO))
        elapsed = round(time.time() - t0, 1)
        delivery = sorted(out_dir.glob("*_delivery_ready.csv"))
        report_p = sorted(out_dir.glob("*_delivery_report.json"))
        issues_p = sorted(out_dir.glob("*_delivery_issues.csv"))
        report: Dict[str, Any] = {}
        if report_p:
            try:
                report = json.loads(report_p[0].read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                report = {}
        warnings, blockers, evidence = self._translate_issues(issues_p)
        counts = self._frame_counts(projected_csv,
                                    str(delivery[0]) if delivery else "")
        metrics = {"elapsed_s": elapsed, "returncode": proc.returncode,
                   "rules_version": _file_version(rules_path), **counts}
        artefacts = {
            "delivery_ready_csv": str(delivery[0]) if delivery else "",
            "delivery_report": str(report_p[0]) if report_p else "",
            "delivery_issues": str(issues_p[0]) if issues_p else "",
        }
        if proc.returncode == 0 and delivery:
            return StageOutcome(
                ok=True,
                summary="Annex 2 data has been prepared for XML generation.",
                warnings=warnings, artefacts=artefacts, metrics=metrics,
                evidence=evidence, report=report)
        if not blockers:
            blockers = ["Annex 2 preparation cannot continue. The delivery "
                        "checks found problems that need attention."]
        return StageOutcome(ok=False, blocking=True,
                            summary="Annex 2 preparation found problems that "
                                    "need your review.",
                            warnings=warnings, blockers=blockers,
                            artefacts=artefacts, metrics=metrics,
                            evidence=evidence, report=report)

    def run_xml(self, *, delivery_csv: str, out_dir: Path,
                xsd_path: Path = XSD_PATH) -> StageOutcome:
        """Invoke the EXISTING Gate 5 builder + its authoritative lxml XSD
        validation, unmodified; then instrument its automatic fills (I4)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        final_xml = out_dir / XML_NAME
        tmp_xml = out_dir / (XML_NAME + ".building")
        cmd = [sys.executable, str(XML_BUILDER),
               "--input", str(delivery_csv), "--output", str(tmp_xml),
               "--mapping-workbook", str(WORKBOOK), "--sheet", WORKBOOK_SHEET,
               "--code-order-yaml", str(CODE_ORDER), "--xsd", str(xsd_path)]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(REPO))
        elapsed = round(time.time() - t0, 1)
        passed = proc.returncode == 0 and \
            "XSD Validation: PASSED" in (proc.stdout or "")
        if not (passed and tmp_xml.exists()):
            detail = [ln for ln in (proc.stdout or "").splitlines()
                      if "XSD" in ln][-3:]
            if tmp_xml.exists():
                tmp_xml.unlink()          # never leave a non-validated XML
            blockers, evidence = self._translate_builder_failure(
                (proc.stderr or "") + "\n" + (proc.stdout or ""))
            return StageOutcome(
                ok=False, blocking=True,
                summary="The Annex 2 XML could not be created or did not "
                        "pass the regulator's format check.",
                blockers=blockers, evidence=evidence,
                metrics={"elapsed_s": elapsed, "returncode": proc.returncode,
                         "xsd": _file_version(xsd_path)},
                report={"stdout_tail": (proc.stdout or "")[-1500:],
                        "stderr_tail": (proc.stderr or "")[-1500:],
                        "xsd_lines": detail})
        tmp_xml.replace(final_xml)         # atomic finalisation
        xml_hash = _sha256(final_xml)

        from . import interventions as _iv
        evidence_doc = _iv.analyse(Path(delivery_csv), final_xml)
        iv_path = out_dir / "builder_interventions.json"
        iv_path.write_text(json.dumps(evidence_doc, indent=2),
                           encoding="utf-8")

        import pandas as pd
        fields = len(pd.read_csv(delivery_csv, nrows=0).columns)
        warnings = []
        if evidence_doc["review_required_instances"]:
            warnings.append(evidence_doc["summary_sentence"])
        return StageOutcome(
            ok=True,
            summary=("Annex 2 XML was created and passed schema validation."
                     if not warnings else
                     "Annex 2 XML passed schema validation, but automatic "
                     "values require review."),
            warnings=warnings,
            artefacts={"xml": str(final_xml),
                       "interventions": str(iv_path)},
            metrics={"elapsed_s": elapsed,
                     "xml_bytes": final_xml.stat().st_size,
                     "records": evidence_doc["records"],
                     "fields": fields,
                     "xsd": _file_version(xsd_path),
                     "xsd_result": "PASSED",
                     "xml_sha256": xml_hash,
                     "nd_injected": evidence_doc["nodata_injected_by_builder"],
                     "review_required_instances":
                         evidence_doc["review_required_instances"]},
            evidence=[{"label": "Automatic values inserted by the generator",
                       "kind": "table",
                       "data": evidence_doc["interventions"]}],
            report=evidence_doc)

    # -- helpers -------------------------------------------------------- #
    @staticmethod
    def _translate_builder_failure(output: str):
        """Builder refusals -> an operator sentence plus answerable evidence.

        A mandatory field the builder finds blank is the same situation the
        projection guards report, so it is described the same way and answered
        by the same decisions: a no-data code where the rules permit one, a
        source column or a pool-level value where they do not.
        """
        import re as _re
        from . import nd_treatments as _nd
        blockers: List[str] = []
        evidence: List[Dict[str, Any]] = []
        for m in _re.finditer(
                r"Mandatory (record|header) field '([A-Z]{3,5}\d{1,3})' is blank",
                output):
            level, code = m.group(1), m.group(2)
            field = _nd.field_for_esma_code(code)
            friendly = (field or code).replace("_", " ")
            blockers.append(
                f"The regulator requires {friendly} on every "
                f"{'record' if level == 'record' else 'report'} and this "
                "delivery has no value for it.")
            evidence.append({"label": "Regulatory field with no value",
                             "kind": "text",
                             "data": {"field": field, "esma_code": code,
                                      "level": level,
                                      "found": "blank on every row"}})
        if not blockers:
            blockers.append("The generated file did not pass the regulator's "
                            "format check. This needs attention before it can "
                            "be prepared again.")
        return blockers, evidence

    @staticmethod
    def _translate_projection_failure(stderr: str):
        """Known projector refusals -> actionable operator sentences; the raw
        detail goes to collapsed evidence only."""
        import re as _re
        blockers: List[str] = []
        evidence: List[Dict[str, Any]] = []
        m = _re.search(r"Field '([^']+)' has unmapped enum values", stderr)
        if m:
            friendly = m.group(1).replace("_", " ")
            values = _re.search(r"mode: \[([^\]]*)\]", stderr)
            blockers.append(
                f"Some '{friendly}' values in the data do not match the "
                "regulator's allowed codes. They need mapping before the "
                "regulatory data can be prepared.")
            evidence.append({"label": "Values needing mapping", "kind": "text",
                             "data": {"field": m.group(1),
                                      "values": (values.group(1)
                                                 if values else "")}})
        m2 = _re.search(
            r"mandatory record-level field '([A-Z]{3,5}\d{1,3})' is blank", stderr)
        if m2 and not blockers:
            from . import nd_treatments as _nd
            code = m2.group(1)
            field = _nd.field_for_esma_code(code)
            friendly = (field or code).replace("_", " ")
            blockers.append(
                f"The regulator requires {friendly} on every record and this "
                "delivery has no value for it.")
            evidence.append({"label": "Regulatory field with no value",
                             "kind": "text",
                             "data": {"field": field, "esma_code": code,
                                      "found": "blank on every row"}})
        elif "ScrtstnIdr" in stderr or "RREL1" in stderr:
            blockers.append(
                "The securitisation identifier could not be derived. Check "
                "the regulatory details for this client.")
        if not blockers:
            blockers.append("The projection step did not finish cleanly. "
                            "Check the regulatory details and try again.")
        return blockers, evidence

    @staticmethod
    def _frame_counts(projected_csv: str, delivery_csv: str) -> Dict[str, Any]:
        import pandas as pd
        out: Dict[str, Any] = {}
        try:
            p = pd.read_csv(projected_csv, usecols=[0], dtype=str)
            out["input_records"] = len(p)
            out["input_fields"] = len(pd.read_csv(projected_csv,
                                                  nrows=0).columns)
        except Exception:  # noqa: BLE001
            pass
        if delivery_csv:
            try:
                d = pd.read_csv(delivery_csv, usecols=[0], dtype=str)
                out["output_records"] = len(d)
                out["output_fields"] = len(pd.read_csv(delivery_csv,
                                                       nrows=0).columns)
            except Exception:  # noqa: BLE001
                pass
        return out

    @staticmethod
    def _translate_issues(issues_paths: List[Path]):
        """Delivery issues CSV -> operator warnings/blockers + evidence."""
        import csv as _csv
        warnings: List[str] = []
        blockers: List[str] = []
        evidence: List[Dict[str, Any]] = []
        if not issues_paths or not issues_paths[0].exists():
            return warnings, blockers, evidence
        rows = []
        try:
            with open(issues_paths[0], newline="", encoding="utf-8") as fh:
                rows = list(_csv.DictReader(fh))
        except Exception:  # noqa: BLE001
            return warnings, blockers, evidence
        if not rows:
            return warnings, blockers, evidence
        from collections import Counter
        by_cat_sev: Counter = Counter()
        for r in rows:
            by_cat_sev[(r.get("delivery_category") or "other",
                        (r.get("severity") or "warning").lower())] += 1
        for (cat, sev), n in sorted(by_cat_sev.items()):
            sentence = _ISSUE_SENTENCES.get(cat, _ISSUE_SENTENCES["other"])
            msg = f"{sentence} ({n} instance{'s' if n != 1 else ''})"
            if sev == "error":
                blockers.append(msg)
            else:
                warnings.append(msg)
        evidence.append({"label": "Delivery check findings", "kind": "table",
                         "data": [{"category": c, "severity": s, "count": n}
                                  for (c, s), n in sorted(by_cat_sev.items())]})
        # Which values, in which field. The counts alone say a delivery cannot
        # be prepared without saying what an operator could do about it — and
        # every one of these is answerable: the lender's word needs the
        # regulator's code. Named per field, in the same shape the projection
        # failure uses, so one decision-builder serves both.
        from . import nd_treatments as _nd
        by_field: Dict[str, List[str]] = {}
        for r in rows:
            if (r.get("issue_type") or "").strip().lower() != "enum":
                continue
            if (r.get("severity") or "").strip().lower() != "error":
                continue
            code = (r.get("field") or "").strip()
            value = (r.get("input_value") or "").strip()
            if not code or not value:
                continue
            canonical = _nd.field_for_esma_code(code) or code
            seen = by_field.setdefault(canonical, [])
            if value not in seen:
                seen.append(value)
        for canonical, values in sorted(by_field.items()):
            evidence.append({"label": "Values needing mapping", "kind": "text",
                             "data": {"field": canonical,
                                      "values": ", ".join(values[:20])}})
        return warnings, blockers, evidence
