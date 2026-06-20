"""
file_classifier.py
==================

PART 3 — deterministic file classification.

Given a folder of lender data-room artefacts, classify each file into one of a
small set of known report types using:

  * file-name patterns
  * column headers (structured files)
  * document text snippets (unstructured files)
  * row / column counts

Structured files (CSV/XLS/XLSX) are read for headers and shape. Unstructured
files (PDF/DOCX) get *metadata placeholder* classification only — we read text
for TXT/MD (cheap and reliable) but never attempt heavy PDF/DOCX extraction
here. Those become extraction gaps downstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from engine.gate_1_alignment.semantic_alignment import tokenize
from .onboarding_models import FileInventoryItem

# ---------------------------------------------------------------------------
# File-type buckets
# ---------------------------------------------------------------------------

STRUCTURED_SUFFIXES = {".csv", ".xlsx", ".xls"}
TEXT_SUFFIXES = {".txt", ".md"}
DOC_SUFFIXES = {".pdf", ".docx", ".doc"}

CLASSIFICATIONS = [
    "current_loan_report",
    "historical_loan_report",
    "cashflow_report",
    "collateral_report",
    "pipeline_report",
    "warehouse_agreement",
    "securitisation_document",
    "investor_report",
    "data_dictionary",
    "unknown",
]

# ---------------------------------------------------------------------------
# Signal tokens per classification.
# `headers` are matched against tokenised column names / text tokens.
# `filename` substrings are matched against the lowercased file name.
# ---------------------------------------------------------------------------

_HEADER_SIGNALS: Dict[str, List[str]] = {
    "cashflow_report": [
        "payment", "interest", "principal", "cashflow", "redemption", "fee",
        "instalment", "repayment", "arrears",
    ],
    "collateral_report": [
        "property", "valuation", "collateral", "postcode", "ltv", "region",
        "value", "epc", "occupancy",
    ],
    "current_loan_report": [
        "loan", "balance", "borrower", "origination", "maturity", "rate",
        "obligor", "account", "drawdown",
    ],
    "pipeline_report": [
        "application", "offer", "completion", "broker", "stage", "pipeline",
        "expected", "funding", "decision",
    ],
    "data_dictionary": [
        "description", "definition", "field", "dictionary", "glossary",
        "meaning", "example",
    ],
}

_FILENAME_SIGNALS: Dict[str, List[str]] = {
    "cashflow_report": ["cashflow", "cash_flow", "payment", "servicer"],
    "collateral_report": ["collateral", "property", "security", "valuation"],
    "current_loan_report": ["loan", "tape", "portfolio", "exposure", "asset"],
    "historical_loan_report": ["historic", "history", "archive"],
    "pipeline_report": ["pipeline", "application", "origination_pipeline"],
    "warehouse_agreement": ["warehouse", "facility", "funding_agreement", "fund"],
    "securitisation_document": ["securit", "abs", "rmbs", "issuance", "deal"],
    "investor_report": ["investor"],
    "data_dictionary": ["dictionary", "glossary", "data_dict"],
}

# Tokens that strongly imply a document (unstructured) classification.
_TEXT_SIGNALS: Dict[str, List[str]] = {
    "warehouse_agreement": [
        "warehouse", "facility", "advance", "margin", "eligibility", "lender",
        "drawdown", "borrowing base",
    ],
    "securitisation_document": [
        "securitisation", "securitization", "issuer", "tranche", "notes",
        "annex", "disclosure", "spv",
    ],
    "investor_report": ["investor", "noteholder", "distribution"],
}


def _detect_file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in STRUCTURED_SUFFIXES:
        return suffix.lstrip(".")
    if suffix in TEXT_SUFFIXES:
        return suffix.lstrip(".")
    if suffix in DOC_SUFFIXES:
        return suffix.lstrip(".")
    return "unknown"


def _read_structured_headers(path: Path) -> Tuple[List[str], Optional[int], Optional[int], str]:
    """Return (headers, row_count, col_count, sheet_name) for a structured file."""
    suffix = path.suffix.lower()
    try:
        if suffix in (".xlsx", ".xls"):
            xl = pd.ExcelFile(path)
            sheet = xl.sheet_names[0]
            df = xl.parse(sheet)
            return list(df.columns), int(len(df)), int(len(df.columns)), str(sheet)
        df = pd.read_csv(path, low_memory=False)
        return list(df.columns), int(len(df)), int(len(df.columns)), ""
    except Exception:
        return [], None, None, ""


def _score_structured(headers: List[str], file_name: str) -> Dict[str, float]:
    """Score each classification from header tokens + filename substrings."""
    header_tokens: set = set()
    for h in headers:
        header_tokens.update(tokenize(h))

    name_lower = file_name.lower()
    scores: Dict[str, float] = {}

    for cls, signals in _HEADER_SIGNALS.items():
        hits = sum(1 for s in signals if s in header_tokens)
        if hits:
            scores[cls] = hits / len(signals)

    for cls, parts in _FILENAME_SIGNALS.items():
        if any(p in name_lower for p in parts):
            # filename evidence is a strong prior — boost (or seed) the score
            scores[cls] = scores.get(cls, 0.0) + 0.45

    return scores


def _read_text_snippet(path: Path, max_chars: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def _score_text(text: str, file_name: str) -> Dict[str, float]:
    tokens = set(tokenize(text))
    name_lower = file_name.lower()
    scores: Dict[str, float] = {}

    for cls, signals in _TEXT_SIGNALS.items():
        hits = sum(1 for s in signals if s in tokens or s in text.lower())
        if hits:
            scores[cls] = hits / len(signals)

    for cls, parts in _FILENAME_SIGNALS.items():
        if any(p in name_lower for p in parts):
            scores[cls] = scores.get(cls, 0.0) + 0.4

    return scores


def classify_file(path: Path) -> FileInventoryItem:
    """Classify a single file into a :class:`FileInventoryItem`."""
    path = Path(path)
    file_type = _detect_file_type(path)
    item = FileInventoryItem(
        file_path=str(path),
        file_name=path.name,
        file_type=file_type,
    )

    if file_type in {"csv", "xlsx", "xls"}:
        headers, n_rows, n_cols, sheet = _read_structured_headers(path)
        item.row_count = n_rows
        item.column_count = n_cols
        item.sheet_name = sheet
        scores = _score_structured(headers, path.name)
        if not headers:
            item.classification = "unknown"
            item.confidence = 0.0
            item.notes = "could not read structured file"
            return item
    elif file_type in {"txt", "md"}:
        text = _read_text_snippet(path)
        scores = _score_text(text, path.name)
        item.notes = "text snippet classification"
    elif file_type in {"pdf", "docx", "doc"}:
        # Metadata placeholder only — no heavy extraction here.
        scores = {}
        for cls, parts in _FILENAME_SIGNALS.items():
            if any(p in path.name.lower() for p in parts):
                scores[cls] = scores.get(cls, 0.0) + 0.4
        item.notes = "document placeholder (no text extraction); classified on filename only"
    else:
        item.classification = "unknown"
        item.notes = "unsupported file type"
        return item

    if scores:
        best_cls = max(scores, key=lambda k: scores[k])
        item.classification = best_cls
        item.confidence = round(min(scores[best_cls], 1.0), 3)
    else:
        item.classification = "unknown"
        item.confidence = 0.0

    return item


def classify_directory(input_dir: Path) -> List[FileInventoryItem]:
    """Classify every supported file under ``input_dir`` (recursively, sorted).

    Discovery recurses into subdirectories so packs that use role/date folder
    conventions (e.g. ``input/funded/2025-11-30/`` and
    ``input/pipeline/2025-12-01/``) are fully ingested — files nested below the
    input root are no longer silently skipped. Hidden files/dirs (``.`` prefixed)
    and READMEs are ignored.
    """
    input_dir = Path(input_dir)
    items: List[FileInventoryItem] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(input_dir).parts
        if any(part.startswith(".") for part in rel_parts):
            # Skip hidden files and anything inside a hidden directory.
            continue
        if path.name.lower() in {"readme.md", "readme.txt"}:
            # README is documentation about the pack, not an artefact.
            continue
        items.append(classify_file(path))
    return items
