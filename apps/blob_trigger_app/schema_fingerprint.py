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
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SchemaInfo:
    file_type: str
    columns: List[str]                      # first/primary sheet's columns
    sheets: List[str] = field(default_factory=list)
    sheet_columns: Dict[str, List[str]] = field(default_factory=dict)
    row_count: Optional[int] = None
    fingerprint: str = ""
    # Header-first role classification (populated by ``fingerprint_pack``).
    role_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    ambiguous_role_conflict: bool = False
    conflicting_roles: List[str] = field(default_factory=list)
    drift_suspected: bool = False
    drift_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_type": self.file_type, "columns": self.columns,
            "sheets": self.sheets, "sheet_columns": self.sheet_columns,
            "row_count": self.row_count, "fingerprint": self.fingerprint,
            "role_diagnostics": self.role_diagnostics,
            "ambiguous_role_conflict": self.ambiguous_role_conflict,
            "conflicting_roles": self.conflicting_roles,
            "drift_suspected": self.drift_suspected,
            "drift_files": self.drift_files,
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


def fingerprint_pack(paths, *, include_order: bool = True,
                     role_schemas: Optional[Dict[str, List[str]]] = None,
                     aliases: Optional[Dict[str, List[str]]] = None) -> SchemaInfo:
    """Combine several data files in a reporting pack into ONE fingerprint, keyed
    on **logical role → columns** (NOT exact file names).

    Roles are resolved HEADER-FIRST by :func:`file_roles.classify_pack`:

      1. header/column signature match against approved ``role_schemas``
         (``SourceRecord.file_role_schemas``) — a header set that matches an
         approved role is assigned that role *regardless of filename*;
      2. approved registry filename ``aliases``;
      3. built-in filename keyword rules;
      4. a stable normalised-name fallback.

    So a pack whose files are cosmetically renamed period-to-period (same headers)
    fingerprints IDENTICALLY and routes deterministically; a real schema change
    (column added/removed/reordered, or a new role) still flips it. The returned
    :class:`SchemaInfo` carries per-file ``role_diagnostics`` and the
    ``ambiguous_role_conflict`` / ``drift_suspected`` fail-closed signals for the
    router. Marker/non-tabular files are skipped.
    """
    from .file_roles import classify_pack

    files: List[Tuple[str, List[str]]] = []
    file_types: List[str] = []
    for path in sorted(str(p) for p in paths):
        p = Path(path)
        ext = p.suffix.lower().lstrip(".")
        if ext not in ("csv", "xlsx", "xls", "xlsm"):
            continue
        info = compute_schema_fingerprint(p, include_order=include_order)
        files.append((p.name, info.columns))
        file_types.append(ext)

    classification = classify_pack(files, role_schemas=role_schemas, aliases=aliases)
    sheet_columns = classification.role_columns()

    primary = sheet_columns.get(sorted(sheet_columns)[0], []) if sheet_columns else []
    out = fingerprint_from_schema(
        file_type="+".join(sorted(set(file_types))) or "pack",
        columns=primary, sheets=sorted(sheet_columns.keys()),
        sheet_columns=sheet_columns, include_order=include_order)
    out.role_diagnostics = classification.diagnostics()
    out.ambiguous_role_conflict = classification.ambiguous_role_conflict
    out.conflicting_roles = classification.conflicting_roles
    out.drift_suspected = classification.drift_suspected
    out.drift_files = classification.drift_files
    return out


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
