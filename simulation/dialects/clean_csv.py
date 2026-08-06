"""simulation.dialects.clean_csv — the reference source dialect.

Stable canonical-like headers, ISO dates, plain decimal numerics, canonical-like
enum values, UTF-8. Gate 1 resolves every header at Tier 1 (exact match), so
this dialect isolates the rest of the pathway from any mapping question: if a
clean-CSV case fails, the fault is not in header resolution.

It is also the equivalence REFERENCE — the other dialects are compared against
the canonical output this one produces.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..assets import get_profile
from ..models import DIALECT_CLEAN_CSV, CaseManifest, SchemaEvolution
from . import register
from . import _format as F
from ._plan import build_plan, format_switched

#: Registry ``format`` → the formatter this dialect uses.
_FORMATTERS = {
    "date": F.iso_date,
    "decimal": lambda v: F.decimal_point(v, 2),
    "integer": F.integer,
    "Y/N": lambda v: F.boolean(v, "yn"),
}


def _field_formats(asset_class: str) -> Dict[str, str]:
    from .._registry import field_formats
    return field_formats(asset_class)


def _value(field: str, raw: Any, formats: Dict[str, str]) -> str:
    fmt = formats.get(field, "")
    formatter = _FORMATTERS.get(fmt)
    if formatter is None:
        return F.text(raw)
    return formatter(raw)


def render(rows: List[Dict[str, Any]], case: CaseManifest, *, period_index: int,
           reporting_date: str, out_dir: Path,
           evolution: Sequence[SchemaEvolution] = ()) -> Path:
    profile = get_profile(case.asset_class)
    formats = _field_formats(case.asset_class)
    plan = build_plan(case, flavour="canonical", period_index=period_index,
                      fields=profile.canonical_fields, evolution=evolution)

    lines = [plan.headers]
    for row in rows:
        line = [_value(canon, row.get(canon), formats)
                for _header, canon in plan.columns]
        line += [""] * len(plan.extras)
        lines.append(line)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{case.portfolio_id}_{reporting_date.replace('-', '')}_clean"
    # A declared ``switch_format`` migrates the CONTAINER, not the vocabulary:
    # the lender starts exporting the same columns, in the same order, as XLSX.
    # Swapping in another dialect's vocabulary here would make the two
    # indistinguishable and quietly weaken the equivalence test.
    if format_switched(case, period_index, evolution) == "xlsx":
        return _write_xlsx(out_dir / f"{stem}.xlsx", lines)
    path = out_dir / f"{stem}.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(lines)
    return path


def _write_xlsx(path: Path, lines: List[List[Any]]) -> Path:
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Loan Book"
    for line in lines:
        sheet.append(list(line))
    workbook.save(path)
    return path


register(DIALECT_CLEAN_CSV, render)

__all__ = ["render"]
