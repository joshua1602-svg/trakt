"""tests/analytical/client_funded.py — a THIRTEEN-MONTH funded history, both books.

Why this exists
---------------
The V1 evidence set carries three monthly funded snapshots per book. Three cuts
support a month-on-month movement and nothing beyond it: a question about a
quarter, a half-year, a trend or a run-rate can only be refused or answered from
a two-point difference. Tranche C's censoring work and Tranche F's conversion
methodology both need a real series to run against.

Additive, like ``client_fixture``. The three funded tapes under
``demo_platform/workspace/`` and the three ``second_book`` produces are hashed in
the V1 MANIFEST, so neither is regenerated. This writes a longer history to a
SEPARATE root, using the same governed artefact shape — a dated
``platform_canonical_typed.csv`` plus its manifest and a ``latest/`` pointer — so
the MI read path, the preparation layer and every derived field behave exactly as
they do on the V1 books.

The generation itself is ``second_book.build``, called with a spec rather than
copied. Its keyword defaults reproduce the V1 output byte-for-byte; passing a
different portfolio set and a longer date list is the only difference here.

Pairing with the pipeline
-------------------------
The client ids match ``client_fixture``'s (``alderbridge``, ``kestrelmoor``), so
one root holds a book's funded history and the other its 52 weekly pipeline
extracts, and a question that spans both reads a consistent lender.

Deterministic: seeded per book, no wall-clock.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

from . import second_book as _sb
from .second_book import PortfolioSpec

#: Thirteen month-ends, oldest first. Ends at the V1 last reporting date: the
#: base book is generated relative to its LAST cut, so moving the end would move
#: every origination date with it, while adding earlier cuts only filters rows.
REPORTING_DATES: Tuple[str, ...] = (
    "2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30", "2025-10-31",
    "2025-11-30", "2025-12-31", "2026-01-31", "2026-02-28", "2026-03-31",
    "2026-04-30", "2026-05-31", "2026-06-30",
)

#: Shaped from the demonstration book's own composition rather than invented:
#: South East led, two portfolios, ~35% acquired by loan count, vintages 2014-.
ALDERBRIDGE_REGIONS: Dict[str, float] = {
    "South East": 0.219, "London": 0.125, "South West": 0.120,
    "East of England": 0.104, "North West": 0.093, "West Midlands": 0.081,
    "Yorkshire and The Humber": 0.074, "East Midlands": 0.073,
    "Scotland": 0.052, "Wales": 0.030, "North East": 0.029,
}

ALDERBRIDGE_PORTFOLIOS: Tuple[PortfolioSpec, ...] = (
    PortfolioSpec("alp_origination", "direct", "Alderbridge Origination",
                  loans=7126, mean_advance=150_000.0, advance_spread=80_000.0,
                  mean_origination_ltv=30.0, ltv_spread=8.0, mean_rate=6.41,
                  mean_age=70.0, front_book_share=0.28,
                  first_vintage=2014, last_vintage=2026),
    PortfolioSpec("alp_acquired", "acquired", "Alderbridge Acquired",
                  loans=3909, mean_advance=115_000.0, advance_spread=62_000.0,
                  mean_origination_ltv=33.0, ltv_spread=8.5, mean_rate=6.86,
                  mean_age=74.0, front_book_share=0.15,
                  first_vintage=2014, last_vintage=2025,
                  acquisition_date="2024-09-30"),
)

#: One entry per book: everything ``second_book.build`` needs to emit it.
BOOKS: Tuple[Dict[str, object], ...] = (
    {"client_id": "alderbridge",
     "display_name": "Alderbridge Lending Platform (synthetic)",
     "seller_name": "Coldharbour Assurance (synthetic)",
     "portfolios": ALDERBRIDGE_PORTFOLIOS,
     "region_weights": ALDERBRIDGE_REGIONS,
     "seed": 20260630},
    {"client_id": _sb.CLIENT_ID,
     "display_name": _sb.DISPLAY_NAME,
     "seller_name": _sb.SELLER_NAME,
     "portfolios": _sb.PORTFOLIOS,
     "region_weights": None,
     "seed": _sb.SEED},
)


def build(root: Path) -> Dict[str, Dict[str, object]]:
    """Write both books' thirteen-month funded history under ``root``.

    Returns ``{client_id: truth_manifest}``, each carrying the per-snapshot row
    count and total balance that an independent reconciliation checks against.
    """
    return {str(book["client_id"]): _sb.build(
        Path(root), reporting_dates=REPORTING_DATES, **book)
        for book in BOOKS}


def mi_env(root: Path, client_id: str,
           reporting_date: str = REPORTING_DATES[-1]) -> Dict[str, str]:
    """The environment that points the MI read path at one of these books."""
    return _sb.mi_env(Path(root), client_id=client_id,
                      reporting_date=reporting_date)


__all__ = ["build", "mi_env", "REPORTING_DATES", "BOOKS",
           "ALDERBRIDGE_PORTFOLIOS", "ALDERBRIDGE_REGIONS"]
