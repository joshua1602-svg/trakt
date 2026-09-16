"""operations_control.occ_agent.workbook — reading a client's tape as it is.

WHY THIS MODULE EXISTS

A lender's extract is a workbook, not a table. Two things about a real one
break a naive ``pd.read_excel(path)``:

* **It has more than one worksheet.** ``LoanExtract One - OMNI`` opens on a
  seven-row summary tab; the five hundred and sixty-nine loans are on the
  second. ``read_excel`` with no ``sheet_name`` reads the FIRST sheet, so the
  loan book profiled as five unnamed columns and seven rows, and every
  consequence — the sample registered with Client Onboarding, the header
  mapping, the canonical tape — would have been built from a cover page.

* **The header is not in row one.** Both the loan and property extracts carry a
  title block above the real column names.

Neither is news to this platform. ``onboarding_orchestrator.
_load_structured_dataframes`` applies ``redetect_header`` and names
PropertyExtract in its comment while doing it; ``schema_fingerprint.
_fingerprint_excel`` enumerates ``xl.sheet_names`` and re-detects headers per
sheet; ``router.py`` re-detects too. The OCC Agent's own artefact path did
neither, so it was the one place in the platform that read these files wrongly.

WHICH SHEET, AND WHY IT IS SAID OUT LOUD

The sheet with the most rows, ties going to the earlier one. That is what a
person does when they open a workbook looking for the loan book, and it is the
only rule here — there is no attempt to recognise a "summary" tab by name,
because the name is the lender's and the next lender's will differ.

A rule that picks is a rule that can pick wrong, so the choice is REPORTED: the
sheet name travels on the artefact and on every mapping row it produced. An
operator who sees the loan tape mapped from "Summary" can say so; one who is
never told cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

#: How much of a sheet to read when deciding which sheet is the data.
PROBE_ROWS = 200


@dataclass
class Table:
    """One worksheet's contents, headed the way a reader would head it."""

    frame: Any = None
    #: Empty for a CSV, which has no sheets to choose between.
    sheet: str = ""
    #: Row the real header was found on, 0 when it was already row one.
    header_row: int = 0
    #: True when re-detection could not find a header that looks like one.
    header_unresolved: bool = False
    sheets: List[str] = field(default_factory=list)

    @property
    def columns(self) -> List[str]:
        return [] if self.frame is None else [str(c) for c in self.frame.columns]

    @property
    def row_count(self) -> int:
        return 0 if self.frame is None else int(len(self.frame))


def _redetect(frame):
    """The platform's own header re-detector, or the frame unchanged."""
    try:
        from engine.onboarding_agent.source_table_loader import redetect_header
    except Exception:  # noqa: BLE001 — a clean file needs no re-detection
        return frame, 0, False
    try:
        return redetect_header(frame)
    except Exception:  # noqa: BLE001
        return frame, 0, False


def read_table(path: Path, *, max_rows: Optional[int] = None) -> Table:
    """Read ``path`` the way the rest of the platform reads a client's tape.

    A CSV is read straight through. A workbook has its data sheet chosen and
    its header re-detected. ``max_rows`` caps what is loaded when only the
    shape is wanted; the sheet CHOICE always probes every sheet, because the
    whole point is that the first one may be small.

    Never raises: an unreadable file is a finding the caller reports, not a
    crash. The returned table is then simply empty.
    """
    path = Path(path)
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return Table()

    try:
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, low_memory=False,
                                **({"nrows": max_rows} if max_rows else {}))
            frame, header_row, unresolved = _redetect(frame)
            return Table(frame=frame, header_row=header_row,
                         header_unresolved=unresolved)

        book = pd.ExcelFile(path)
        sheets = [str(s) for s in book.sheet_names]
        chosen, chosen_rows = "", -1
        for name in sheets:
            try:
                probe = book.parse(name, nrows=PROBE_ROWS)
            except Exception:  # noqa: BLE001 — a sheet that will not parse
                continue
            rows = int(len(probe))
            if rows > chosen_rows:
                chosen, chosen_rows = name, rows
        if not chosen:
            return Table(sheets=sheets)

        frame = book.parse(chosen, **({"nrows": max_rows} if max_rows else {}))
        frame, header_row, unresolved = _redetect(frame)
        return Table(frame=frame, sheet=chosen, header_row=header_row,
                     header_unresolved=unresolved, sheets=sheets)
    except Exception:  # noqa: BLE001
        return Table()
