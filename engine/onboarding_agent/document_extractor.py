"""
document_extractor.py
====================

PART 6/7 — read unstructured documents (markdown / text; PDF/DOCX placeholder)
to extract only config-relevant facts, under a strict minimisation policy.

Guarantees:
  * Only fields whitelisted per document type in the policy are extracted.
  * Retained evidence is a short excerpt capped by
    ``allowed_retained_evidence_chars`` — never full text, pages, clauses,
    signatures, addresses or bank details.
  * Client-specific extractions are written ONLY under the project output
    directory (:func:`assert_within_project`), never to config/system,
    config/regime or config/client.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .onboarding_models import DocumentExtraction, FileInventoryItem

_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "system" / "onboarding_agent.yaml"
# Directories that must NEVER receive client-specific extracted values.
_GLOBAL_CONFIG_DIRS = ("config/system", "config/regime", "config/client")


class PersistenceScopeError(Exception):
    """Raised when a client-specific artefact would be written outside the project."""


def load_document_policy(policy_path: Path | None = None) -> dict:
    path = Path(policy_path) if policy_path else _POLICY_PATH
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("document_extraction", {}) or {}


def assert_within_project(target: Path, project_dir: Path) -> Path:
    """Guarantee ``target`` resolves inside ``project_dir`` (PART 7 guard)."""
    target = Path(target).resolve()
    project_dir = Path(project_dir).resolve()
    try:
        target.relative_to(project_dir)
    except ValueError:
        raise PersistenceScopeError(
            f"Refusing to write client-specific artefact outside project dir: {target}"
        )
    # Belt and braces: never under a global config tree.
    s = str(target).replace("\\", "/")
    if any(f"/{d}/" in s or s.endswith(f"/{d}") for d in _GLOBAL_CONFIG_DIRS):
        raise PersistenceScopeError(f"Refusing to write into global config tree: {target}")
    return target


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_PERCENT = r"(\d+(?:\.\d+)?)\s*%"
_MONEY = r"((?:GBP|EUR|USD|£|€|\$)\s*[\d,]+(?:\.\d+)?)"


def _excerpt(text: str, match_start: int, match_end: int, cap: int) -> str:
    """Return a short, single-line evidence excerpt centred on the match, capped."""
    raw = text[max(0, match_start - 30): match_end + 30]
    raw = " ".join(raw.split())  # collapse whitespace/newlines
    return raw[:cap].strip()


def _find(pattern: str, text: str, cap: int, flags=re.IGNORECASE):
    """Return (value, evidence) for the first match of group(1), else (None, '')."""
    m = re.search(pattern, text, flags)
    if not m:
        return None, ""
    return m.group(1).strip(), _excerpt(text, m.start(), m.end(), cap)


def _extract_warehouse(text: str, doc_name: str, allowed: List[str], cap: int) -> List[DocumentExtraction]:
    low = text.lower()
    out: List[DocumentExtraction] = []

    def add(field: str, value: str, evidence: str, conf: float, status="requires_review"):
        if field in allowed:
            out.append(DocumentExtraction(
                field=field, value=value, source_document=doc_name,
                source_reference="warehouse terms", confidence=conf,
                retained_evidence=evidence[:cap], status=status,
            ))

    add("warehouse_facility_present", "true", "Warehouse funding agreement present.", 0.95, "suggested")

    m = re.search(r"warehouse lender[:\s\*|]*([A-Za-z0-9 ,&.'-]+?)(?:\n|\||$)", text, re.IGNORECASE)
    if m:
        add("warehouse_lender_name", m.group(1).strip().rstrip("|").strip(),
            _excerpt(text, m.start(), m.end(), cap), 0.7)

    v, ev = _find(rf"advance rate[^\d]*{_PERCENT}", low, cap)
    if v:
        add("advance_rate", f"{v}%", ev, 0.7)
    v, ev = _find(rf"margin[^\d]*{_PERCENT}", low, cap)
    if v:
        add("margin", f"{v}%", ev, 0.6)

    for idx in ("SONIA", "EURIBOR", "SOFR", "LIBOR", "BASE RATE"):
        if idx.lower() in low:
            pos = low.find(idx.lower())
            add("interest_index", idx, _excerpt(text, pos, pos + len(idx), cap), 0.7)
            break

    m = re.search(_MONEY, text)
    if m and ("limit" in low or "facility" in low):
        add("warehouse_limit", m.group(1).strip(), _excerpt(text, m.start(), m.end(), cap), 0.6)

    m = re.search(r"availability period[:\s\*|]*([\w ]+?months)", low)
    if m:
        add("availability_period", m.group(1).strip(), _excerpt(text, m.start(), m.end(), cap), 0.6)

    if "eligibility" in low:
        pos = low.find("eligibility")
        add("eligibility_criteria_summary", "present", _excerpt(text, pos, pos + 11, cap), 0.5)

    if "concentration" in low:
        pos = low.find("concentration")
        add("concentration_limits_summary", "present", _excerpt(text, pos, pos + 13, cap), 0.5)

    m = re.search(r"(monthly|quarterly|weekly|annually)[^.\n]{0,40}report", low)
    if m:
        add("reporting_frequency", m.group(1), _excerpt(text, m.start(), m.end(), cap), 0.5)

    return out


def _extract_securitisation(text: str, doc_name: str, allowed: List[str], cap: int) -> List[DocumentExtraction]:
    low = text.lower()
    out: List[DocumentExtraction] = []

    def add(field: str, value: str, evidence: str, conf: float, status="requires_review"):
        if field in allowed:
            out.append(DocumentExtraction(
                field=field, value=value, source_document=doc_name,
                source_reference="securitisation summary", confidence=conf,
                retained_evidence=evidence[:cap], status=status,
            ))

    add("securitisation_present", "true", "Securitisation summary present.", 0.7, "suggested")

    m = re.search(r"target pool balance[:\s\*|]*([A-Z]{3}\s*[\d,]+)", text, re.IGNORECASE)
    if m:
        add("target_pool_balance", m.group(1).strip(), _excerpt(text, m.start(), m.end(), cap), 0.6)

    for regime in ("ESMA Annex 2", "ESMA Annex 12", "ESMA_Annex2"):
        if regime.lower() in low:
            pos = low.find(regime.lower())
            add("reporting_regime", "ESMA_Annex2", _excerpt(text, pos, pos + len(regime), cap), 0.6)
            break

    return out


def extract_documents(
    inventory: List[FileInventoryItem],
    policy: Optional[dict] = None,
) -> List[DocumentExtraction]:
    """Extract config-relevant facts from warehouse / securitisation documents."""
    policy = policy if policy is not None else load_document_policy()
    cap = int(policy.get("allowed_retained_evidence_chars", 300))
    allowed_map = policy.get("allowed_extraction_fields", {}) or {}

    out: List[DocumentExtraction] = []
    for item in inventory:
        allowed = allowed_map.get(item.classification)
        if not allowed:
            continue
        if item.file_type not in ("txt", "md"):
            # PDF/DOCX: placeholder only — record presence, no text extraction.
            present_field = (
                "warehouse_facility_present"
                if item.classification == "warehouse_agreement"
                else "securitisation_present"
            )
            if present_field in allowed:
                out.append(DocumentExtraction(
                    field=present_field, value="true",
                    source_document=item.file_name,
                    source_reference="document present (no text extraction)",
                    confidence=0.5, retained_evidence="(binary document; not parsed)",
                    status="requires_review",
                ))
            continue
        try:
            text = Path(item.file_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if item.classification == "warehouse_agreement":
            out.extend(_extract_warehouse(text, item.file_name, allowed, cap))
        elif item.classification == "securitisation_document":
            out.extend(_extract_securitisation(text, item.file_name, allowed, cap))

    # Enforce evidence cap defensively (no full text ever leaves here).
    for e in out:
        if len(e.retained_evidence) > cap:
            e.retained_evidence = e.retained_evidence[:cap]
    return out


def write_document_extraction_summary(
    extractions: List[DocumentExtraction],
    project_dir: Path,
    policy: Optional[dict] = None,
) -> Path:
    """Write 17_document_extraction_summary.yaml under the project dir (guarded)."""
    project_dir = Path(project_dir)
    target = assert_within_project(project_dir / "17_document_extraction_summary.yaml", project_dir)
    policy = policy if policy is not None else load_document_policy()
    payload = {
        "_policy": {
            "persist_full_text": bool(policy.get("persist_full_text", False)),
            "client_scoped_only": bool(policy.get("client_scoped_only", True)),
            "allowed_retained_evidence_chars": int(policy.get("allowed_retained_evidence_chars", 300)),
        },
        "document_extractions": [e.to_dict() for e in extractions],
    }
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target
