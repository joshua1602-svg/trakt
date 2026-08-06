"""simulation.dialects — rendering ONE economic truth into several source formats.

A dialect receives canonical-field-space rows and writes the file a lender would
actually drop: its own header vocabulary, its own date and number conventions,
its own status words, its own column order, its own irrelevant extra columns.

Dialects never touch economics. They cannot add, remove or adjust a balance —
they only decide how the same number is written down. That is what makes
"all dialects resolve to the same canonical truth" a real test of the ingestion
and mapping path rather than a restatement of the generator.

Registered dialects:

``clean_csv``            stable canonical-like headers, ISO dates, decimal
                         numerics, UTF-8;
``lender_excel``         a UK lender's XLSX: alias headers, worksheet-name
                         variation, reordered columns, a metadata sheet,
                         percentages as formatted values, blank optional columns
                         and additional irrelevant columns;
``european_locale_csv``  a continental servicer's CSV: day-first dates, comma
                         decimals, semicolon separator, percentages as decimals
                         or percent strings, Y/N-Yes/No-1/0 booleans, enum
                         synonyms and region codes rather than names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from ..models import (
    DIALECT_CLEAN_CSV,
    DIALECT_LENDER_EXCEL,
    DIALECT_LOCALE_CSV,
    CaseManifest,
)

#: ``dialect_id -> render(rows, case, period_index, out_dir, evolution) -> Path``
_RENDERERS: Dict[str, Callable[..., Path]] = {}


def register(dialect_id: str, renderer: Callable[..., Path]) -> None:
    _RENDERERS[dialect_id] = renderer


def render(dialect_id: str, rows: List[Dict[str, Any]], case: CaseManifest, *,
           period_index: int, reporting_date: str, out_dir: Path,
           evolution: Sequence = ()) -> Path:
    """Write one period of one case in one dialect, returning the file path."""
    from . import clean_csv, lender_excel, locale_csv  # noqa: F401 - registration
    try:
        renderer = _RENDERERS[dialect_id]
    except KeyError as exc:  # pragma: no cover - closed vocabulary
        raise KeyError(f"unknown dialect {dialect_id!r}; "
                       f"known: {sorted(_RENDERERS)}") from exc
    return renderer(rows, case, period_index=period_index,
                    reporting_date=reporting_date, out_dir=Path(out_dir),
                    evolution=tuple(evolution))


def registered() -> List[str]:
    from . import clean_csv, lender_excel, locale_csv  # noqa: F401
    return sorted(_RENDERERS)


__all__ = ["DIALECT_CLEAN_CSV", "DIALECT_LENDER_EXCEL", "DIALECT_LOCALE_CSV",
           "register", "registered", "render"]
