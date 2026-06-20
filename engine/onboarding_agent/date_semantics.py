"""
date_semantics.py
=================

Asset-agnostic onboarding **date semantics**: separate the funded-book reporting
date from the pipeline snapshot date.

In an MI package the loan, collateral and cashflow tapes share a single
*funded-book reporting date* (``reporting_date`` / ``funded_reporting_date``),
while the origination-pipeline file may legitimately carry a *different* snapshot
date (``pipeline_snapshot_date``) because it drives forward exposure / forecast.

This module is config/registry-driven and contains no client- or asset-specific
logic. It provides:

* alias resolution — funded-book reporting-date aliases resolve to
  ``reporting_date``; pipeline aliases resolve to ``pipeline_snapshot_date``;
* per-artefact-role date basis — loan/collateral/cashflow use the funded basis,
  pipeline uses the pipeline basis;
* date assignment from a run manifest, role/date folders, detected column dates,
  or file names (in that precedence), each with source + confidence + evidence;
* consistency validation — funded artefacts must agree on the funded reporting
  date (else a blocking ``date_basis_mismatch``); a differing
  ``pipeline_snapshot_date`` is allowed and recorded, never blocking.

Safety: it never forces ``pipeline_snapshot_date`` onto funded records, never
weakens regulatory ``data_cut_off_date`` semantics, never lets funded
loan/collateral/cashflow date mismatches pass silently, and keeps every
inferred/defaulted date auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from . import run_context as rc

# Canonical date fields.
REPORTING_DATE = "reporting_date"
PIPELINE_SNAPSHOT_DATE = "pipeline_snapshot_date"

# Date basis (which canonical date a role's artefact carries).
BASIS_FUNDED = "funded_reporting_date"
BASIS_PIPELINE = "pipeline_snapshot_date"

# Issue codes (controlled vocabulary, recorded in review/handoff artefacts).
ISSUE_BASIS_MISMATCH = "date_basis_mismatch"
ISSUE_PIPELINE_DIFFERENCE = "pipeline_vs_funded_difference"
ISSUE_FUNDED_MISSING = "funded_reporting_date_missing"


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").strip().lower()).strip()


# --------------------------------------------------------------------------- #
# Artefact-role -> date basis
# --------------------------------------------------------------------------- #
# Roles come from file_classifier (current_loan_report, collateral_report,
# cashflow_report, historical_loan_report, pipeline_report, …).
FUNDED_BOOK_ROLES = {"current_loan_report", "historical_loan_report", "loan_extract",
                     "funded_book"}
COLLATERAL_ROLES = {"collateral_report", "collateral_extract"}
CASHFLOW_ROLES = {"cashflow_report", "cashflow_extract"}
PIPELINE_ROLES = {"pipeline_report", "pipeline", "origination_pipeline"}

# All roles that must share the funded reporting date basis.
FUNDED_BASIS_ROLES = FUNDED_BOOK_ROLES | COLLATERAL_ROLES | CASHFLOW_ROLES

# Folder-token -> basis (for role/date folders like input/funded/2025-11-30).
_FOLDER_ROLE_TOKENS = {
    BASIS_FUNDED: ("funded", "loan", "loans", "loan_tape", "loantape", "book",
                   "collateral", "property", "security", "cashflow", "servicing"),
    BASIS_PIPELINE: ("pipeline", "origination", "kfi", "application", "applications"),
}


def basis_for_role(role: str) -> str:
    """Return the date basis (``funded_reporting_date`` / ``pipeline_snapshot_date``)
    for an artefact role, or ``""`` when unknown."""
    r = re.sub(r"[^a-z0-9_]+", "_", str(role or "").strip().lower())
    if r in PIPELINE_ROLES:
        return BASIS_PIPELINE
    if r in FUNDED_BASIS_ROLES:
        return BASIS_FUNDED
    return ""


def canonical_date_field_for_role(role: str) -> str:
    """Canonical date field a role's artefact carries."""
    basis = basis_for_role(role)
    if basis == BASIS_PIPELINE:
        return PIPELINE_SNAPSHOT_DATE
    if basis == BASIS_FUNDED:
        return REPORTING_DATE
    return ""


# --------------------------------------------------------------------------- #
# Alias resolution (funded reporting date vs pipeline snapshot date)
# --------------------------------------------------------------------------- #
# Funded-book reporting-date aliases -> reporting_date. These do NOT create a new
# canonical field; they all mean the funded-book reporting date.
FUNDED_REPORTING_DATE_ALIASES = (
    "reporting date", "report date", "mi reporting date", "as of date",
    "reporting period",
    "funded reporting date", "funded book reporting date", "funded as of date",
    "loan tape reporting date", "loan extract reporting date", "book date",
    "cut off date", "cut-off date", "data cut off date", "data cut-off date",
)

# Pipeline snapshot-date aliases -> pipeline_snapshot_date.
PIPELINE_SNAPSHOT_DATE_ALIASES = (
    "pipeline snapshot date", "pipeline as of date", "pipeline extract date",
    "pipeline report date", "application pipeline date", "kfi pipeline date",
    "pipeline date",
)

_FUNDED_ALIAS_SET = {_norm(a) for a in FUNDED_REPORTING_DATE_ALIASES}
_PIPELINE_ALIAS_SET = {_norm(a) for a in PIPELINE_SNAPSHOT_DATE_ALIASES}
# Generic date tokens that are basis-ambiguous and must be disambiguated by role.
_GENERIC_DATE_TOKENS = {"as of date", "as-of date", "snapshot date", "extract date",
                        "report date", "reporting date", "date"}


def resolve_date_field(column: str, role: str = "") -> Dict[str, Any]:
    """Resolve a column/synonym to its canonical date field.

    Pipeline aliases win for ``pipeline_snapshot_date``; funded aliases resolve to
    ``reporting_date``. A basis-ambiguous generic token (e.g. "as of date") is
    disambiguated by the artefact ``role`` so a date in a pipeline file maps to
    ``pipeline_snapshot_date`` and a date in a loan/collateral/cashflow file maps
    to ``reporting_date``. Returns ``{canonical_field, confidence, basis,
    matched_alias, rationale}`` (canonical_field == "" when unresolved).
    """
    token = _norm(column)
    role_basis = basis_for_role(role)

    # Explicit pipeline alias.
    if token in _PIPELINE_ALIAS_SET or "pipeline" in token:
        return {"canonical_field": PIPELINE_SNAPSHOT_DATE, "confidence": 0.95,
                "basis": BASIS_PIPELINE, "matched_alias": token,
                "rationale": "matched a pipeline snapshot-date alias"}

    # Explicit, non-generic funded alias.
    if token in _FUNDED_ALIAS_SET and token not in _GENERIC_DATE_TOKENS:
        # Even an explicit funded token defers to a pipeline artefact role, so a
        # pipeline file's "reporting date" column is a pipeline snapshot date.
        if role_basis == BASIS_PIPELINE:
            return {"canonical_field": PIPELINE_SNAPSHOT_DATE, "confidence": 0.8,
                    "basis": BASIS_PIPELINE, "matched_alias": token,
                    "rationale": "funded-style date token in a pipeline artefact "
                                 "-> pipeline snapshot date"}
        return {"canonical_field": REPORTING_DATE, "confidence": 0.95,
                "basis": BASIS_FUNDED, "matched_alias": token,
                "rationale": "matched a funded-book reporting-date alias"}

    # Basis-ambiguous generic date token: disambiguate by role.
    if token in _GENERIC_DATE_TOKENS or token in _FUNDED_ALIAS_SET:
        if role_basis == BASIS_PIPELINE:
            return {"canonical_field": PIPELINE_SNAPSHOT_DATE, "confidence": 0.7,
                    "basis": BASIS_PIPELINE, "matched_alias": token,
                    "rationale": "generic date token in a pipeline artefact"}
        if role_basis == BASIS_FUNDED:
            return {"canonical_field": REPORTING_DATE, "confidence": 0.7,
                    "basis": BASIS_FUNDED, "matched_alias": token,
                    "rationale": "generic date token in a funded artefact"}
        return {"canonical_field": REPORTING_DATE, "confidence": 0.5,
                "basis": BASIS_FUNDED, "matched_alias": token,
                "rationale": "generic date token; defaulting to funded reporting date"}

    return {"canonical_field": "", "confidence": 0.0, "basis": "",
            "matched_alias": "", "rationale": "no date alias matched"}


# --------------------------------------------------------------------------- #
# Folder / manifest conventions
# --------------------------------------------------------------------------- #

def parse_role_date_path(path: str) -> Dict[str, Any]:
    """Parse a role/date folder convention (e.g. ``input/funded/2025-11-30`` or
    ``input/pipeline/2025-12-01``). Returns ``{basis, date, role_token, evidence}``;
    ``basis``/``date`` are ``""`` when not both are present (never invented)."""
    segments = [s for s in re.split(r"[\\/]+", str(path or "")) if s]
    basis = ""
    role_token = ""
    found_date = ""
    for seg in segments:
        low = seg.lower()
        if not basis:
            for b, tokens in _FOLDER_ROLE_TOKENS.items():
                if any(tok == low or tok in low for tok in tokens):
                    basis, role_token = b, low
                    break
        # A date segment (full ISO or a period token).
        iso = rc.normalize_to_iso(seg)
        if not iso:
            per = rc.dates_from_period_token(seg)
            iso = per[0] if per else ""
        if iso and not found_date:
            found_date = iso
    return {"basis": basis if found_date else "", "date": found_date,
            "role_token": role_token, "evidence": f"folder:{path}"}


def load_run_manifest(manifest: Any) -> Dict[str, Any]:
    """Load explicit funded/pipeline dates from a run manifest (path, mapping, or
    YAML string). Recognises ``funded_reporting_date`` / ``pipeline_snapshot_date``
    at top level or under an ``mi_package`` block. Unknown -> empty strings."""
    data: Dict[str, Any] = {}
    if isinstance(manifest, dict):
        data = manifest
    elif manifest:
        p = Path(str(manifest))
        try:
            text = p.read_text(encoding="utf-8") if p.exists() else str(manifest)
            data = yaml.safe_load(text) or {}
        except Exception:
            data = {}
    block = data.get("mi_package") if isinstance(data.get("mi_package"), dict) else data
    out = {"funded_reporting_date": "", "pipeline_snapshot_date": "",
           "source": "run_manifest"}
    for key in ("funded_reporting_date", "reporting_date"):
        iso = rc.normalize_to_iso((block or {}).get(key))
        if iso:
            out["funded_reporting_date"] = iso
            break
    iso = rc.normalize_to_iso((block or {}).get("pipeline_snapshot_date"))
    if iso:
        out["pipeline_snapshot_date"] = iso
    return out


# --------------------------------------------------------------------------- #
# Per-artefact date assignment
# --------------------------------------------------------------------------- #

@dataclass
class ArtefactDate:
    file_name: str
    role: str
    basis: str
    canonical_field: str
    date: str = ""
    source: str = ""
    confidence: float = 0.0
    evidence: List[str] = dc_field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name, "role": self.role, "basis": self.basis,
            "canonical_field": self.canonical_field, "date": self.date,
            "source": self.source, "confidence": round(float(self.confidence), 4),
            "evidence": list(self.evidence),
        }


def assign_artefact_dates(
    artefacts: Sequence[Dict[str, Any]],
    manifest: Any = None,
) -> List[ArtefactDate]:
    """Assign a canonical date to each artefact by its role basis.

    Each ``artefact`` dict may carry ``file_name`` / ``role`` (or
    ``classification``) and optionally ``folder``, ``detected_date`` and a list of
    column ``date_columns``. Resolution precedence (highest first):
    run manifest (by basis) -> role/date folder -> detected column date ->
    file-name date. Every assigned date records its source + confidence + evidence.
    A pipeline date is never assigned to a funded artefact and vice versa.
    """
    man = load_run_manifest(manifest) if manifest else {
        "funded_reporting_date": "", "pipeline_snapshot_date": ""}
    results: List[ArtefactDate] = []
    for art in artefacts:
        role = str(art.get("role") or art.get("classification") or "")
        basis = basis_for_role(role)
        canonical = canonical_date_field_for_role(role)
        ad = ArtefactDate(file_name=str(art.get("file_name", "")), role=role,
                          basis=basis, canonical_field=canonical)

        # 1) explicit manifest value for this basis.
        man_val = (man.get("funded_reporting_date") if basis == BASIS_FUNDED
                   else man.get("pipeline_snapshot_date") if basis == BASIS_PIPELINE
                   else "")
        if man_val:
            ad.date, ad.source, ad.confidence = man_val, "run_manifest", 1.0
            ad.evidence.append(f"manifest:{basis}={man_val}")
            results.append(ad)
            continue

        # 2) role/date folder.
        folder = art.get("folder") or art.get("file_path") or ""
        if folder:
            fp = parse_role_date_path(str(folder))
            # Honour the artefact's own role basis; the folder only supplies a date
            # when it does not contradict the role basis.
            if fp["date"] and (not basis or not fp["basis"] or fp["basis"] == basis):
                ad.date, ad.source, ad.confidence = fp["date"], "role_date_folder", 0.85
                ad.evidence.append(fp["evidence"])
                results.append(ad)
                continue

        # 3) detected column date (already parsed upstream).
        det = rc.normalize_to_iso(art.get("detected_date"))
        if det:
            ad.date, ad.source, ad.confidence = det, "source_column", 0.9
            ad.evidence.append("detected_date_column")
            results.append(ad)
            continue

        # 4) file-name date.
        fn_dates = rc.dates_from_filename(str(art.get("file_name", "")))
        if fn_dates:
            ad.date, ad.source, ad.confidence = fn_dates[0], "filename", 0.6
            ad.evidence.append(f"filename:{art.get('file_name', '')}")
        results.append(ad)
    return results


# --------------------------------------------------------------------------- #
# Consistency validation
# --------------------------------------------------------------------------- #

def validate_date_consistency(
    artefact_dates: Sequence[ArtefactDate],
    *,
    approved_mismatch: bool = False,
) -> Dict[str, Any]:
    """Validate funded-book date consistency and record the pipeline difference.

    * Funded artefacts (loan / collateral / cashflow) must share one
      ``funded_reporting_date``; conflicting dates raise a **blocking**
      ``date_basis_mismatch`` (downgraded to a recorded, non-blocking note only
      when ``approved_mismatch`` is set).
    * A ``pipeline_snapshot_date`` that differs from the funded date is **allowed**
      and recorded as a non-blocking ``pipeline_vs_funded_difference``.

    Returns an auditable summary dict.
    """
    funded = [a for a in artefact_dates if a.basis == BASIS_FUNDED and a.date]
    pipeline = [a for a in artefact_dates if a.basis == BASIS_PIPELINE and a.date]

    funded_distinct = sorted({a.date for a in funded})
    pipeline_distinct = sorted({a.date for a in pipeline})
    issues: List[Dict[str, Any]] = []
    blocking = False

    funded_reporting_date = funded_distinct[0] if len(funded_distinct) == 1 else ""

    if len(funded_distinct) > 1:
        detail = {
            "code": ISSUE_BASIS_MISMATCH,
            "blocking": not approved_mismatch,
            "message": ("funded loan/collateral/cashflow artefacts are on "
                        f"different reporting dates: {funded_distinct}"),
            "dates_by_artefact": {a.file_name: a.date for a in funded},
            "approved": approved_mismatch,
        }
        issues.append(detail)
        blocking = blocking or detail["blocking"]

    # Pipeline difference is allowed; surface it explicitly, never blocking.
    for pdate in pipeline_distinct:
        if funded_reporting_date and pdate != funded_reporting_date:
            issues.append({
                "code": ISSUE_PIPELINE_DIFFERENCE,
                "blocking": False,
                "message": (f"pipeline_snapshot_date {pdate} differs from "
                            f"funded_reporting_date {funded_reporting_date} "
                            "(allowed: pipeline drives forward exposure)"),
                "pipeline_snapshot_date": pdate,
                "funded_reporting_date": funded_reporting_date,
            })

    if funded and not funded_reporting_date and len(funded_distinct) <= 1:
        # No funded date at all resolved.
        issues.append({"code": ISSUE_FUNDED_MISSING, "blocking": False,
                       "message": "no funded reporting date resolved from funded artefacts"})

    return {
        "funded_reporting_date": funded_reporting_date,
        "pipeline_snapshot_date": pipeline_distinct[0] if len(pipeline_distinct) == 1 else "",
        "funded_dates_distinct": funded_distinct,
        "pipeline_dates_distinct": pipeline_distinct,
        "issues": issues,
        "blocking": blocking,
        "artefact_dates": [a.as_dict() for a in artefact_dates],
    }
