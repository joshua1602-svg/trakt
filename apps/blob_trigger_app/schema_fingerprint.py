"""apps.blob_trigger_app.schema_fingerprint — deterministic schema fingerprint.

The fingerprint is derived from STRUCTURE (column names, order, sheet names,
file type) — never from cell values — so the same monthly tape layout produces
the same fingerprint period after period, and a real schema change flips it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SchemaInfo:
    file_type: str
    columns: List[str]                      # first/primary sheet's columns
    sheets: List[str] = field(default_factory=list)
    sheet_columns: Dict[str, List[str]] = field(default_factory=dict)
    row_count: Optional[int] = None
    fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_type": self.file_type, "columns": self.columns,
            "sheets": self.sheets, "sheet_columns": self.sheet_columns,
            "row_count": self.row_count, "fingerprint": self.fingerprint,
        }


def _hash(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def fingerprint_from_schema(
    *, file_type: str, columns: List[str],
    sheets: Optional[List[str]] = None,
    sheet_columns: Optional[Dict[str, List[str]]] = None,
    include_order: bool = True,
) -> SchemaInfo:
    """Compute a fingerprint from already-extracted schema metadata.

    ``include_order`` keeps column order significant (default). Values and row
    counts are never part of the key.
    """
    cols = list(columns) if include_order else sorted(columns)
    sc = {k: (list(v) if include_order else sorted(v))
          for k, v in (sheet_columns or {}).items()}
    payload = {
        "file_type": (file_type or "").lower().lstrip("."),
        "columns": cols,
        "sheets": sorted(sheets or []),
        "sheet_columns": sc,
    }
    info = SchemaInfo(file_type=payload["file_type"], columns=cols,
                      sheets=sorted(sheets or []), sheet_columns=sc)
    info.fingerprint = _hash(payload)
    return info


def compute_schema_fingerprint(path: str | Path, *, include_order: bool = True) -> SchemaInfo:
    """Read a tabular file (.csv / .xlsx / .xls) and compute its fingerprint.

    Only headers/sheets are read — never the full value set as the key.
    """
    p = Path(path)
    ext = p.suffix.lower().lstrip(".")
    if ext in ("xlsx", "xls", "xlsm"):
        return _fingerprint_excel(p, ext, include_order)
    return _fingerprint_csv(p, ext or "csv", include_order)


def _fingerprint_csv(p: Path, ext: str, include_order: bool) -> SchemaInfo:
    import pandas as pd
    head = pd.read_csv(p, nrows=0)
    cols = [str(c) for c in head.columns]
    info = fingerprint_from_schema(file_type=ext, columns=cols, include_order=include_order)
    return info


def _fingerprint_excel(p: Path, ext: str, include_order: bool) -> SchemaInfo:
    import pandas as pd
    xl = pd.ExcelFile(p)
    sheets = list(xl.sheet_names)
    sheet_columns: Dict[str, List[str]] = {}
    for s in sheets:
        head = xl.parse(s, nrows=0)
        sheet_columns[s] = [str(c) for c in head.columns]
    primary = sheet_columns.get(sheets[0], []) if sheets else []
    return fingerprint_from_schema(
        file_type=ext, columns=primary, sheets=sheets,
        sheet_columns=sheet_columns, include_order=include_order)
