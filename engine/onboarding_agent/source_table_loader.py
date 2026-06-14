"""
source_table_loader.py
======================

Robust, EXPLICIT loader for the controlled mapping review.

The legacy ``_load_structured_dataframes`` reads only the first sheet of each
workbook and silently swallows parse errors, so a multi-file data room could end
up with column evidence for only one file and no explanation. This loader:

  * iterates EVERY inventoried file,
  * parses CSV + all sheets of XLSX/XLS (and XLSB when pyxlsb is available),
  * records an explicit parse status / error / reason per file (never silent),
  * marks documents (docx/pdf/txt/md) as document-only, and
  * marks unsupported types (e.g. xlsb without pyxlsb) explicitly.

It returns the parsed (file, sheet, frame) tables plus a per-file coverage record
that drives the ``29a_column_evidence_file_coverage`` artefact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# Parse statuses.
PARSED = "parsed"
PARSE_ERROR = "parse_error"
UNSUPPORTED = "unsupported_file_type"
DOCUMENT_ONLY = "document_only"
EMPTY = "empty"

_TABULAR = {".csv", ".xlsx", ".xls", ".xlsb"}
_DOCUMENT = {".docx", ".doc", ".pdf", ".txt", ".md"}


@dataclass
class LoadedTable:
    file_name: str
    file_path: str
    sheet_name: str
    df: pd.DataFrame


@dataclass
class FileCoverage:
    file_name: str = ""
    file_path: str = ""
    file_type: str = ""
    classification: str = ""
    domains_detected: str = ""
    parse_status: str = ""
    parse_error: str = ""
    reason_excluded: str = ""
    recommended_next_action: str = ""
    sheets_parsed: List[str] = field(default_factory=list)
    sheets_skipped: List[str] = field(default_factory=list)
    attempted_column_evidence: bool = False


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def load_source_tables(
    inventory: List[Dict[str, Any]]
) -> Tuple[List[LoadedTable], List[FileCoverage]]:
    """Parse every inventoried file. Returns (tables, coverage)."""
    tables: List[LoadedTable] = []
    coverage: List[FileCoverage] = []

    for item in inventory:
        path = item.get("file_path", "")
        name = item.get("file_name", Path(path).name)
        suffix = Path(path).suffix.lower()
        cov = FileCoverage(
            file_name=name, file_path=path,
            file_type=item.get("file_type", suffix.lstrip(".")),
            classification=item.get("classification", ""),
            domains_detected="; ".join(item.get("domains_detected", []) or []),
        )

        if suffix == ".csv":
            cov.attempted_column_evidence = True
            try:
                df = _read_csv(path)
                if df.shape[1] == 0 or df.dropna(how="all").empty:
                    cov.parse_status = EMPTY
                    cov.reason_excluded = "file has no data rows/columns"
                    cov.recommended_next_action = "check the export; resend a populated file"
                else:
                    tables.append(LoadedTable(name, path, "", df))
                    cov.parse_status = PARSED
                    cov.sheets_parsed = [""]
            except Exception as exc:
                cov.parse_status = PARSE_ERROR
                cov.parse_error = str(exc)[:300]
                cov.recommended_next_action = "fix encoding/delimiter or resend as clean CSV"

        elif suffix in (".xlsx", ".xls"):
            cov.attempted_column_evidence = True
            try:
                xl = pd.ExcelFile(path)
                sheets = list(xl.sheet_names)
                multi = len(sheets) > 1
                for sh in sheets:
                    try:
                        df = xl.parse(sh)
                        if df.shape[1] == 0 or df.dropna(how="all").empty:
                            cov.sheets_skipped.append(f"{sh} (empty)")
                            continue
                        tables.append(LoadedTable(name, path, sh if multi else "", df))
                        cov.sheets_parsed.append(sh)
                    except Exception as exc:
                        cov.sheets_skipped.append(f"{sh} ({str(exc)[:80]})")
                cov.parse_status = PARSED if cov.sheets_parsed else PARSE_ERROR
                if not cov.sheets_parsed:
                    cov.reason_excluded = "no parseable sheets: " + "; ".join(cov.sheets_skipped)
                    cov.recommended_next_action = "check the workbook; export the data sheet to CSV"
            except Exception as exc:
                cov.parse_status = PARSE_ERROR
                cov.parse_error = str(exc)[:300]
                cov.recommended_next_action = "open/repair the workbook or export to CSV"

        elif suffix == ".xlsb":
            cov.attempted_column_evidence = True
            try:
                import pyxlsb  # noqa: F401
                xl = pd.ExcelFile(path, engine="pyxlsb")
                multi = len(xl.sheet_names) > 1
                for sh in xl.sheet_names:
                    df = xl.parse(sh)
                    if df.shape[1] and not df.dropna(how="all").empty:
                        tables.append(LoadedTable(name, path, sh if multi else "", df))
                        cov.sheets_parsed.append(sh)
                cov.parse_status = PARSED if cov.sheets_parsed else PARSE_ERROR
            except ImportError:
                cov.parse_status = UNSUPPORTED
                cov.reason_excluded = "xlsb parser unavailable"
                cov.recommended_next_action = "convert to xlsx/csv or add pyxlsb support"
                cov.attempted_column_evidence = False
            except Exception as exc:
                cov.parse_status = PARSE_ERROR
                cov.parse_error = str(exc)[:300]
                cov.recommended_next_action = "convert to xlsx/csv"

        elif suffix in _DOCUMENT:
            cov.parse_status = DOCUMENT_ONLY
            cov.reason_excluded = ("not a tabular mapping source; used for "
                                   "warehouse/concentration terms if applicable")
            cov.recommended_next_action = "document extraction only"

        else:
            cov.parse_status = UNSUPPORTED
            cov.reason_excluded = f"unsupported file type '{suffix or 'unknown'}'"
            cov.recommended_next_action = "convert to xlsx/csv"

        coverage.append(cov)
    return tables, coverage
