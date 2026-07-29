#!/usr/bin/env python3
"""
generate_annex2_scale_fixture.py
================================

Build a deterministic, full-scale Annex 2 *projected* CSV for the golden
regression.

Why this exists: the original 11,035-record client tape behind the historically
recorded run is not in this repository, and committing a 200 MB artefact or its
source would be the wrong trade. This script reproduces the **scale and shape**
of that run from the committed 36-record synthetic projected data, so the golden
tier genuinely exercises the route at size — memory, wall clock, artefact size —
without pretending to be the original data.

What it does and does not change:

* rows are repeated cyclically until the target record count is reached;
* per-record identifiers (``RREL2``, ``RREL3``) are made unique so the report
  does not carry 11,035 copies of one exposure identifier;
* submission-level constants (``RREL1``, LEIs, dates, enumerations, amounts) are
  left **exactly** as the projector produced them — nothing is invented, and no
  value is coerced into a shape the projector would not have emitted.

Usage::

    python scripts/generate_annex2_scale_fixture.py \\
        --source synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_ESMA_Annex2_projected.csv \\
        --output out/annex2_scale_projected.csv \\
        --records 11035
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional

#: Per-exposure identifiers that must not repeat across records.
UNIQUE_COLUMNS = ("RREL2", "RREL3")

#: How many source rows to cycle through. Kept explicit so the output is a pure
#: function of (source file, target count).
DEFAULT_TARGET = 11_035


def build_scale_fixture(source: Path, output: Path, *,
                        target_records: int = DEFAULT_TARGET) -> Path:
    """Write a projected CSV with ``target_records`` rows. Deterministic."""
    source = Path(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with source.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows: List[List[str]] = [row for row in reader if any(cell.strip() for cell in row)]

    if not rows:
        raise ValueError(f"{source} contains no data rows")

    unique_idx = [header.index(c) for c in UNIQUE_COLUMNS if c in header]
    width = len(str(target_records))

    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for n in range(target_records):
            row = list(rows[n % len(rows)])
            suffix = f"{n + 1:0{width}d}"
            for idx in unique_idx:
                base = row[idx].strip()
                # Keep the projector's identifier shape; append a record ordinal
                # so identifiers are unique without inventing a new format.
                row[idx] = f"{base}-{suffix}" if base else base
            writer.writerow(row)

    return output


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument(
        "--source",
        default="synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_ESMA_Annex2_projected.csv",
        help="Committed projected Annex 2 CSV to scale up.")
    parser.add_argument("--output", required=True, help="Destination CSV.")
    parser.add_argument("--records", type=int, default=DEFAULT_TARGET,
                        help="Target record count (default: the historical 11,035).")
    args = parser.parse_args(argv)

    path = build_scale_fixture(Path(args.source), Path(args.output),
                               target_records=args.records)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Wrote {path} — {args.records:,} records, {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
