"""
source_table_loader.py
======================

Robust, EXPLICIT loader for the controlled mapping review.

It never fails silently and never leaves a blank parse reason. For every file it
detects the real container by file SIGNATURE (not just the extension), flags
extension/signature mismatches (e.g. a ``.xlsx`` that is really a legacy OLE
compound document), attempts the appropriate parser(s), and records a full
diagnostic: declared extension, detected container, parsers attempted, engine
used, the parse error, converter availability and a concrete next action.

Real-world driver: lender files named ``*.xlsx`` whose first 8 bytes are the OLE
compound signature ``D0 CF 11 E0 A1 B1 1A E1`` (a legacy ``.xls``/OLE document),
which openpyxl (zip) cannot read and which xlrd reports as
``Can't find workbook in OLE2 compound document``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Parse statuses.
PARSED = "parsed"
PARSE_ERROR = "parse_error"
UNSUPPORTED = "unsupported_file_type"
DEPENDENCY_MISSING = "dependency_missing"
DOCUMENT_ONLY = "document_only"
EMPTY = "empty"

# Container signatures.
_OLE_MAGIC = bytes([0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1])
_ZIP_MAGIC = b"PK"

C_ZIP = "zip_ooxml_xlsx"
C_OLE = "ole_compound"
C_PLAIN = "plain_text"
C_HTML = "html_excel"
C_XML = "xml_spreadsheet"
C_CSV = "csv_tsv"
C_BINARY = "unknown_binary"

_DOCUMENT_SUFFIXES = {".docx", ".doc", ".pdf", ".txt", ".md"}

# declared extension -> expected container family (for mismatch detection).
_EXPECTED_FAMILY = {
    ".xlsx": "zip", ".xlsm": "zip", ".xlsb": "zip", ".xls": "ole",
    ".csv": "text", ".tsv": "text", ".txt": "text",
}
_CONTAINER_FAMILY = {
    C_ZIP: "zip", C_OLE: "ole", C_CSV: "text", C_PLAIN: "text",
    C_HTML: "text", C_XML: "text", C_BINARY: "binary",
}

_OLE_NEXT_ACTION = (
    "File has .xlsx extension but OLE compound signature and is not readable by "
    "openpyxl/xlrd. Open in Excel or LibreOffice locally and resave as true .xlsx "
    "or .csv, or install/configure a conversion service."
)


@dataclass
class LoadedTable:
    file_name: str
    file_path: str
    sheet_name: str
    df: pd.DataFrame


@dataclass
class SheetCoverage:
    file_name: str = ""
    declared_extension: str = ""
    detected_container_type: str = ""
    sheet_name: str = ""
    parse_status: str = ""
    rows: int = 0
    columns: int = 0
    engine_used: str = ""
    parse_error: str = ""


@dataclass
class FileCoverage:
    file_name: str = ""
    file_path: str = ""
    file_type: str = ""
    classification: str = ""
    domains_detected: str = ""
    declared_extension: str = ""
    detected_container_type: str = ""
    detected_excel_format: str = ""
    extension_mismatch_detected: bool = False
    parser_attempted: str = ""
    engine_used: str = "none"
    parse_status: str = ""
    parse_error: str = ""
    reason_excluded: str = ""
    recommended_next_action: str = ""
    conversion_available: bool = False
    conversion_tool: str = "none"
    conversion_attempted: bool = False
    conversion_status: str = ""
    conversion_error: str = ""
    converted_file_path: str = ""
    attempted_column_evidence: bool = False
    sheets_parsed: List[str] = field(default_factory=list)
    sheets_skipped: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def detect_container_type(path: str | Path) -> str:
    """Detect the real container from the file signature (not the extension)."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096)
    except OSError:
        return C_BINARY
    if head[:8] == _OLE_MAGIC:
        return C_OLE
    if head[:2] == _ZIP_MAGIC:
        return C_ZIP
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = head.decode("latin-1")
        except Exception:
            return C_BINARY
    stripped = text.lstrip().lower()
    if stripped.startswith("<?xml"):
        return C_XML
    if stripped.startswith("<html") or stripped.startswith("<!doctype html") \
            or "<table" in stripped[:2000]:
        return C_HTML
    printable = sum(1 for c in text if c in "\t\r\n" or 32 <= ord(c) < 127 or ord(c) >= 160)
    if text and printable / len(text) > 0.9:
        first_line = text.splitlines()[0] if text.splitlines() else ""
        return C_CSV if ("," in first_line or "\t" in first_line) else C_PLAIN
    return C_BINARY


def _find_converter() -> Tuple[bool, str]:
    for tool in ("libreoffice", "soffice"):
        if shutil.which(tool):
            return True, tool
    return False, "none"


def _fmt_err(exc: Exception) -> str:
    msg = str(exc).splitlines()[0].strip() if str(exc) else ""
    return f'{type(exc).__name__}("{msg[:160]}")'


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _engine_order(container: str, declared_ext: str) -> List[str]:
    if declared_ext == ".xlsb" or (container == C_ZIP and declared_ext == ".xlsb"):
        return ["pyxlsb"]
    if container == C_ZIP:
        return ["openpyxl"]
    if container == C_OLE:
        # openpyxl will fail on OLE (not a zip); xlrd is the legacy .xls reader.
        return ["openpyxl", "xlrd"]
    return ["openpyxl", "xlrd"]


def _parse_excel(path: str, container: str, declared_ext: str
                 ) -> Tuple[Optional[List[Tuple[str, pd.DataFrame]]], str, List[str], str, str]:
    """Attempt to parse an Excel-like file.

    Returns (frames | None, engine_used, parsers_attempted, error, status_hint).
    """
    attempted: List[str] = []
    errors: Dict[str, str] = {}
    dependency_missing = False
    for eng in _engine_order(container, declared_ext):
        attempted.append(eng)
        try:
            xl = pd.ExcelFile(path, engine=eng)
            frames = [(sh, xl.parse(sh)) for sh in xl.sheet_names]
            return frames, eng, attempted, "", PARSED
        except ImportError as exc:
            errors[eng] = _fmt_err(exc)
            dependency_missing = True
        except Exception as exc:
            errors[eng] = _fmt_err(exc)
    # All engines failed. Prefer the most meaningful error (xlrd for OLE).
    pref = errors.get("xlrd") or errors.get("pyxlsb") or errors.get("openpyxl") or "parse failed"
    status = DEPENDENCY_MISSING if dependency_missing and len(errors) == 1 else PARSE_ERROR
    return None, "none", attempted, pref, status


def _convert_and_reparse(path: str, tool: str) -> Tuple[Optional[List[Tuple[str, pd.DataFrame]]], str, str]:
    """Convert a file to xlsx via LibreOffice and reparse. Returns (frames, status, error/path)."""
    outdir = tempfile.mkdtemp(prefix="trakt_convert_")
    try:
        subprocess.run([tool, "--headless", "--convert-to", "xlsx", "--outdir", outdir, path],
                       capture_output=True, timeout=120, check=True)
    except Exception as exc:
        return None, "failed", _fmt_err(exc)
    out = Path(outdir) / (Path(path).stem + ".xlsx")
    if not out.exists():
        return None, "failed", "conversion produced no output file"
    try:
        xl = pd.ExcelFile(out, engine="openpyxl")
        frames = [(sh, xl.parse(sh)) for sh in xl.sheet_names]
        return frames, "converted", str(out)
    except Exception as exc:
        return None, "failed", _fmt_err(exc)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_source_tables(
    inventory: List[Dict[str, Any]],
    enable_conversion: bool = False,
    converter: Optional[Tuple[bool, str]] = None,
) -> Tuple[List[LoadedTable], List[FileCoverage], List[SheetCoverage]]:
    """Parse every inventoried file. Returns (tables, file_coverage, sheet_coverage)."""
    conv_available, conv_tool = converter if converter is not None else _find_converter()
    tables: List[LoadedTable] = []
    coverage: List[FileCoverage] = []
    sheet_cov: List[SheetCoverage] = []

    for item in inventory:
        path = item.get("file_path", "")
        name = item.get("file_name", Path(path).name)
        suffix = Path(path).suffix.lower()
        cov = FileCoverage(
            file_name=name, file_path=path, file_type=item.get("file_type", suffix.lstrip(".")),
            classification=item.get("classification", ""),
            domains_detected="; ".join(item.get("domains_detected", []) or []),
            declared_extension=suffix,
            conversion_available=conv_available, conversion_tool=conv_tool,
        )

        # Document-only types: never a tabular mapping source.
        if suffix in _DOCUMENT_SUFFIXES and suffix not in (".txt",):
            cov.detected_container_type = detect_container_type(path)
            cov.parse_status = DOCUMENT_ONLY
            cov.reason_excluded = ("not a tabular mapping source; used for "
                                   "warehouse/concentration terms if applicable")
            cov.recommended_next_action = "document extraction only"
            coverage.append(cov)
            continue

        container = detect_container_type(path)
        cov.detected_container_type = container
        fam = _CONTAINER_FAMILY.get(container, "binary")
        exp = _EXPECTED_FAMILY.get(suffix)
        cov.extension_mismatch_detected = bool(exp and exp != fam)

        # CSV / plain text / delimited.
        if container in (C_CSV, C_PLAIN) and suffix in (".csv", ".tsv", ".txt") \
                or (container == C_CSV and fam == "text"):
            cov.attempted_column_evidence = True
            cov.parser_attempted = "pandas_csv"
            try:
                sep = "\t" if suffix == ".tsv" else ","
                df = pd.read_csv(path, low_memory=False, sep=sep)
                _record_table(tables, sheet_cov, cov, name, path, "", df, "pandas_csv", suffix, container)
            except Exception as exc:
                cov.parse_status = PARSE_ERROR
                cov.engine_used = "none"
                cov.parse_error = _fmt_err(exc)
                cov.recommended_next_action = "fix encoding/delimiter or resend as clean CSV"
                sheet_cov.append(SheetCoverage(name, suffix, container, "", PARSE_ERROR, 0, 0,
                                               "none", cov.parse_error))
            coverage.append(cov)
            continue

        # Excel-like (by extension OR by detected container).
        excel_like = suffix in (".xlsx", ".xlsm", ".xlsb", ".xls") or container in (C_ZIP, C_OLE)
        if excel_like:
            cov.attempted_column_evidence = True
            frames, engine, attempted, error, status = _parse_excel(path, container, suffix)
            cov.parser_attempted = ", ".join(attempted)
            cov.engine_used = engine

            # Optional conversion fallback when parsing failed.
            if frames is None and enable_conversion:
                if conv_available:
                    cov.conversion_attempted = True
                    fr, cstatus, detail = _convert_and_reparse(path, conv_tool)
                    cov.conversion_status = cstatus
                    if fr is not None:
                        frames, engine, status = fr, f"{conv_tool}->openpyxl", PARSED
                        cov.engine_used = engine
                        cov.converted_file_path = detail
                    else:
                        cov.conversion_error = detail
                else:
                    cov.conversion_attempted = False
                    cov.conversion_status = "unavailable"
                    cov.conversion_error = "No libreoffice/soffice executable found"

            if frames is None:
                cov.parse_status = status
                cov.parse_error = error or "parse failed"
                cov.detected_excel_format = _excel_format(container, success=False)
                cov.reason_excluded = (
                    "xlsb parser unavailable" if (suffix == ".xlsb" and status == DEPENDENCY_MISSING)
                    else f"{container} not readable by {cov.parser_attempted}")
                cov.recommended_next_action = _next_action(container, suffix, status)
                sheet_cov.append(SheetCoverage(name, suffix, container, "", cov.parse_status,
                                               0, 0, "none", cov.parse_error))
            else:
                cov.detected_excel_format = _excel_format(container, success=True, engine=engine,
                                                          suffix=suffix)
                multi = len(frames) > 1
                for sh, df in frames:
                    if df.shape[1] == 0 or df.dropna(how="all").empty:
                        cov.sheets_skipped.append(f"{sh} (empty)")
                        sheet_cov.append(SheetCoverage(name, suffix, container, sh, EMPTY, 0,
                                                       df.shape[1], engine, ""))
                        continue
                    tables.append(LoadedTable(name, path, sh if multi else "", df))
                    cov.sheets_parsed.append(sh)
                    sheet_cov.append(SheetCoverage(name, suffix, container, sh, PARSED,
                                                   df.shape[0], df.shape[1], engine, ""))
                cov.parse_status = PARSED if cov.sheets_parsed else PARSE_ERROR
                if not cov.sheets_parsed:
                    cov.reason_excluded = "no non-empty sheets"
                    cov.recommended_next_action = "check the workbook; export the data sheet to CSV"
            coverage.append(cov)
            continue

        # HTML / XML disguised as a spreadsheet.
        if container in (C_HTML, C_XML):
            cov.attempted_column_evidence = True
            cov.parser_attempted = "pandas_read_html" if container == C_HTML else "none"
            if container == C_HTML:
                try:
                    dfs = pd.read_html(path)
                    if dfs:
                        _record_table(tables, sheet_cov, cov, name, path, "", dfs[0],
                                      "pandas_read_html", suffix, container)
                        coverage.append(cov)
                        continue
                except Exception as exc:
                    cov.parse_error = _fmt_err(exc)
            cov.parse_status = PARSE_ERROR
            cov.engine_used = "none"
            cov.parse_error = cov.parse_error or f"{container} not parseable as a table"
            cov.reason_excluded = f"file content is {container}, not a real workbook"
            cov.recommended_next_action = ("resave as true .xlsx or .csv from Excel/LibreOffice")
            sheet_cov.append(SheetCoverage(name, suffix, container, "", PARSE_ERROR, 0, 0,
                                           "none", cov.parse_error))
            coverage.append(cov)
            continue

        # Anything else.
        cov.parse_status = UNSUPPORTED
        cov.parse_error = ""
        cov.reason_excluded = f"unsupported/undetectable content (container={container})"
        cov.recommended_next_action = "convert to xlsx/csv from a tool that can open it"
        coverage.append(cov)

    return tables, coverage, sheet_cov


def _record_table(tables, sheet_cov, cov, name, path, sheet, df, engine, suffix, container):
    if df.shape[1] == 0 or df.dropna(how="all").empty:
        cov.parse_status = EMPTY
        cov.reason_excluded = "file has no data rows/columns"
        cov.recommended_next_action = "check the export; resend a populated file"
        cov.engine_used = engine
        sheet_cov.append(SheetCoverage(name, suffix, container, sheet, EMPTY, 0, df.shape[1],
                                       engine, ""))
        return
    tables.append(LoadedTable(name, path, sheet, df))
    cov.parse_status = PARSED
    cov.engine_used = engine
    cov.sheets_parsed.append(sheet or "(single)")
    sheet_cov.append(SheetCoverage(name, suffix, container, sheet, PARSED, df.shape[0],
                                   df.shape[1], engine, ""))


def _excel_format(container: str, success: bool, engine: str = "", suffix: str = "") -> str:
    if not success:
        if container == C_OLE:
            return "ole_compound_unknown_or_unreadable_excel"
        if container == C_ZIP:
            return "ooxml_unreadable"
        return f"{container}_unreadable"
    if engine == "xlrd":
        return "legacy_xls_biff"
    if suffix == ".xlsb" or engine == "pyxlsb":
        return "ooxml_xlsb"
    return "ooxml_xlsx"


def _next_action(container: str, suffix: str, status: str) -> str:
    if container == C_OLE:
        return _OLE_NEXT_ACTION
    if suffix == ".xlsb" and status == DEPENDENCY_MISSING:
        return "install pyxlsb or convert to xlsx/csv"
    if container == C_ZIP:
        return "workbook is a zip but unreadable; repair in Excel/LibreOffice or export to CSV"
    return "open in Excel/LibreOffice and resave as true .xlsx or .csv"
