#!/usr/bin/env python3
"""assemble_multibook.py — combine per-book canonicals into a platform canonical.

SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER.

One period's books, concatenated into the single governed dataset every
downstream surface reads. This mirrors what ``engine.platform_assembler`` does
for a managed-service run: the books stay individually identifiable through the
provenance columns Gate 2 stamped, and the combined tape is what MI, reporting
and the regime projection then operate on.

Nothing is recomputed here. The assembler concatenates and orders; every figure
was produced by the gates. It fails loudly if a book's provenance is missing,
because a row that cannot say which book it belongs to would silently corrupt
every per-book total downstream.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

BOOK_ORDER = ("alp_origination", "alp_acquired", "spv1_sponsored")
PROVENANCE_COLUMNS = ("source_portfolio_id", "source_portfolio_type")


def assemble(output_dir: Path, period: str) -> Path:
    frames: List[pd.DataFrame] = []

    for book in BOOK_ORDER:
        path = output_dir / f"{book}_{period}_canonical_typed.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing canonical for {book} {period}: {path}")

        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        for column in PROVENANCE_COLUMNS:
            if column not in frame.columns:
                raise ValueError(
                    f"{path.name} carries no '{column}'. Book identity must be "
                    "stamped at Gate 2 — a combined tape whose rows cannot name "
                    "their book would corrupt every per-book total."
                )
            blank = frame[column].astype(str).str.strip() == ""
            if blank.any():
                raise ValueError(
                    f"{path.name}: {int(blank.sum())} row(s) have a blank "
                    f"'{column}'."
                )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    target = output_dir / f"platform_{period}_canonical_typed.csv"
    combined.to_csv(target, index=False)

    counts = combined["source_portfolio_id"].value_counts().to_dict()
    detail = ", ".join(f"{book}={counts.get(book, 0)}" for book in BOOK_ORDER)
    print(f"  platform_{period}: {len(combined)} rows ({detail})")
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--periods", nargs="+", required=True)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    for period in args.periods:
        assemble(output_dir, period)


if __name__ == "__main__":
    main()
