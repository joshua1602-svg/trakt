"""tests.helpers.acquired_pack — the ERE ``acquired_001`` source pack fixture.

Builds ``AcquiredLoanTape.csv``: ONE commercial portfolio delivered as ONE file,
which the seller assembled from TWO legacy rump books. The source was normalised
before delivery, so both populations share field names and schema — what differs
is legitimate row-level DATA, including the reporting cut-off date.

The fixture is generated deterministically (no randomness, no timestamps) so the
row counts, value distributions and parse outcomes a test asserts are exactly
reproducible. It deliberately carries the shapes the acquired path must handle:

  * two cut-off dates in one tape (rump A and rump B), which MI must accept as
    data rather than treat as a conflict;
  * UK ``DD/MM/YYYY`` dates of birth whose day is ``01`` by source convention
    (the provider knows only month and year);
  * blank second-borrower DOBs, which are absent values, not broken dates;
  * ``Protected Equity`` as PERCENTAGE TEXT (``0.00%`` … ``50.00%``) plus blanks —
    the value that must reach ``protected_equity_percentage`` and must never be
    handed to a boolean parser.

Nothing here is portfolio-specific production behaviour; it is test input only.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: Total rows in the acquired tape, split across the two legacy rump books.
ROW_COUNT = 885
RUMP_A_ROWS = 512
RUMP_B_ROWS = ROW_COUNT - RUMP_A_ROWS          # 373

#: Each rump book was cut on its own date. Both are legitimate.
RUMP_A_CUT_OFF = "2026-05-31"
RUMP_B_CUT_OFF = "2026-06-30"

#: The pack's blob folder period — pack CONTEXT, not a row-level replacement.
PACK_PERIOD = "2026-06-30"

PACK_KEY = "ERE_acquired_001_funded_ad_hoc_2026-06-30"
CLIENT_ID = "ERE"
SOURCE_PORTFOLIO_ID = "acquired_001"
DATASET = "funded"
FREQUENCY = "ad_hoc"
FILE_NAME = "AcquiredLoanTape.csv"
BLOB_PREFIX = (f"{CLIENT_ID}/acquired/{DATASET}/{FREQUENCY}/"
               f"{SOURCE_PORTFOLIO_ID}/{PACK_PERIOD}")

COLUMNS: List[str] = [
    "Loan Policy Number",
    "Cut Off Date",
    "Policy Completion Date",
    "Original Loan Amount",
    "Current Outstanding Balance",
    "Loan Interest Rate",
    "Post Code",
    "Borrower 1 DOB",
    "Borrower 1 Gender",
    "Borrower 2 DOB",
    "Borrower 2 Gender",
    "Protected Equity",
    "Original Property Value",
    "Latest Property Value",
    "Valuation Date",
    "Policy Status",
    "Product",
]

#: Protected-equity percentages, cycled deterministically. The empty string is a
#: genuinely absent value: null percentage AND null flag.
_PROTECTED_EQUITY = ["0.00%", "20.00%", "30.00%", "50.00%", "", "0.00%", "20.00%"]

_POSTCODES = ["NP4 8AB", "SW1A 1AA", "M1 1AE", "LS1 4AP", "BS1 4DJ",
              "EH1 1YZ", "CF10 1EP", "B1 1AA"]
_PRODUCTS = ["Legacy Lifetime 6.20%", "Legacy Drawdown 5.85%"]


def _dob(index: int, *, base_year: int) -> str:
    """A UK ``DD/MM/YYYY`` date of birth with the source's conventional day 01.

    The provider records month and year only and sets the day to ``01``; the
    fixture reproduces that faithfully so tests can prove the day is preserved
    rather than reinterpreted or re-derived.
    """
    month = (index % 12) + 1
    year = base_year + (index % 23)
    return f"01/{month:02d}/{year}"


def _rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for i in range(ROW_COUNT):
        rump_a = i < RUMP_A_ROWS
        cut_off = RUMP_A_CUT_OFF if rump_a else RUMP_B_CUT_OFF
        balance = 60000 + (i * 137) % 240000
        valuation = 180000 + (i * 911) % 520000
        # Every 7th second borrower is absent (a single-borrower loan).
        has_borrower_2 = (i % 7) != 0
        rows.append({
            "Loan Policy Number": str(80000000 + i),
            "Cut Off Date": cut_off,
            "Policy Completion Date": f"{(i % 28) + 1:02d}/{(i % 12) + 1:02d}/2019",
            "Original Loan Amount": f"{balance * 0.82:,.2f}",
            "Current Outstanding Balance": f"{balance:,.2f}",
            "Loan Interest Rate": f"{5.20 + (i % 9) * 0.15:.2f}",
            "Post Code": _POSTCODES[i % len(_POSTCODES)],
            "Borrower 1 DOB": _dob(i, base_year=1930),
            "Borrower 1 Gender": "F" if i % 2 else "M",
            "Borrower 2 DOB": _dob(i + 5, base_year=1934) if has_borrower_2 else "",
            "Borrower 2 Gender": ("M" if i % 2 else "F") if has_borrower_2 else "",
            "Protected Equity": _PROTECTED_EQUITY[i % len(_PROTECTED_EQUITY)],
            "Original Property Value": f"{valuation:,.2f}",
            "Latest Property Value": f"{valuation * 1.06:,.2f}",
            "Valuation Date": cut_off.replace("-", "/"),
            "Policy Status": "Inforce",
            "Product": _PRODUCTS[i % len(_PRODUCTS)],
        })
    return rows


def expected_protected_equity() -> Tuple[int, int, int]:
    """``(zero_rows, positive_rows, blank_rows)`` the fixture should produce."""
    zero = positive = blank = 0
    for i in range(ROW_COUNT):
        value = _PROTECTED_EQUITY[i % len(_PROTECTED_EQUITY)]
        if not value:
            blank += 1
        elif float(value.rstrip("%")) > 0:
            positive += 1
        else:
            zero += 1
    return zero, positive, blank


def write_tape(dest_dir: str | Path,
               extra_columns: Optional[Sequence[str]] = None,
               drop_columns: Optional[Sequence[str]] = None) -> Path:
    """Write ``AcquiredLoanTape.csv`` into ``dest_dir`` and return its path.

    ``extra_columns`` appends optional columns (an ADDITIVE change, which the
    approval policy treats as non-material). ``drop_columns`` removes columns —
    dropping a mandatory one is a MATERIAL change that must stop for review.
    """
    dropped = set(drop_columns or ())
    columns = [c for c in COLUMNS if c not in dropped] + list(extra_columns or ())
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / FILE_NAME
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in _rows():
            writer.writerow({**row, **{c: "" for c in (extra_columns or ())}})
    return path


def blob_prefix(period: str = PACK_PERIOD) -> str:
    return (f"{CLIENT_ID}/acquired/{DATASET}/{FREQUENCY}/"
            f"{SOURCE_PORTFOLIO_ID}/{period}")


def pack_key_for_period(period: str = PACK_PERIOD) -> str:
    return (f"{CLIENT_ID}_{SOURCE_PORTFOLIO_ID}_{DATASET}_{FREQUENCY}_{period}")


def write_blob_tree(root: str | Path, container: str = "raw-v2",
                    period: str = PACK_PERIOD,
                    extra_columns: Optional[Sequence[str]] = None,
                    drop_columns: Optional[Sequence[str]] = None) -> Path:
    """Lay the pack out under a local blob-container tree and return the folder.

    Produces ``{container}/ERE/acquired/funded/ad_hoc/acquired_001/{period}/``
    with the tape and a ``_READY.json`` marker, i.e. exactly the path convention
    the blob trigger and the backfill parse.

    ``period`` writes a later delivery of the SAME source. With no column
    overrides its schema is identical — used to prove a recurring pack does not
    need re-approval. See :func:`write_tape` for the column overrides.
    """
    folder = Path(root) / container / blob_prefix(period)
    write_tape(folder, extra_columns=extra_columns, drop_columns=drop_columns)
    (folder / "_READY.json").write_text("{}", encoding="utf-8")
    return folder
