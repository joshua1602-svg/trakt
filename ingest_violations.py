#!/usr/bin/env python3
"""
ingest_violations.py

CLI bridge: reads canonical CSV and one or more violation CSVs,
creates a snapshot, classifies findings, and inserts them into the
exception management database.

Usage:
    python ingest_violations.py \
      --canonical out/ERE_102025_canonical_typed.csv \
      --violations out_validation/ERE_102025_canonical_violations.csv \
      --violations out_validation/ERE_102025_business_violations.csv \
      --portfolio-type equity_release \
      --db trakt_exceptions.db
"""

import argparse
import hashlib
import sys
from pathlib import Path

from exception_db import init_db, create_snapshot, ingest_findings


def sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def count_csv_rows(path: str) -> int:
    """Return the number of data rows (excluding header) in a CSV."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, _ in enumerate(f):
            pass
        count = i  # last index == number of data rows (header is line 0)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest validation violations into the exception management DB."
    )
    parser.add_argument(
        "--canonical",
        required=True,
        help="Path to the canonical CSV (used for snapshot hash and row count).",
    )
    parser.add_argument(
        "--violations",
        action="append",
        required=True,
        help="Path to a violations CSV (may be specified multiple times).",
    )
    parser.add_argument(
        "--portfolio-type",
        required=True,
        help="Portfolio type, e.g. equity_release.",
    )
    parser.add_argument(
        "--db",
        default="trakt_exceptions.db",
        help="Path to SQLite database (default: trakt_exceptions.db).",
    )
    args = parser.parse_args()

    canonical_path = Path(args.canonical)
    if not canonical_path.exists():
        print(f"ERROR: Canonical file not found: {canonical_path}", file=sys.stderr)
        sys.exit(1)

    for vp in args.violations:
        if not Path(vp).exists():
            print(f"ERROR: Violations file not found: {vp}", file=sys.stderr)
            sys.exit(1)

    # Initialise DB
    conn = init_db(args.db)

    # Compute file hash and row count
    file_hash = sha256_file(str(canonical_path))
    row_count = count_csv_rows(str(canonical_path))

    # Create snapshot
    snapshot_id = create_snapshot(
        conn,
        input_file=canonical_path.name,
        file_hash=file_hash,
        portfolio_type=args.portfolio_type,
        row_count=row_count,
        canonical_path=str(canonical_path),
    )

    # Ingest each violations file
    total_findings = 0
    for vp in args.violations:
        n = ingest_findings(conn, snapshot_id, vp)
        total_findings += n

    # Compute materiality summary
    blocking = conn.execute(
        "SELECT COUNT(*) FROM findings WHERE snapshot_id = ? AND materiality = 'BLOCKING'",
        (snapshot_id,),
    ).fetchone()[0]
    review = conn.execute(
        "SELECT COUNT(*) FROM findings WHERE snapshot_id = ? AND materiality = 'REVIEW'",
        (snapshot_id,),
    ).fetchone()[0]
    info = conn.execute(
        "SELECT COUNT(*) FROM findings WHERE snapshot_id = ? AND materiality = 'INFO'",
        (snapshot_id,),
    ).fetchone()[0]

    print(
        f"Snapshot {snapshot_id} created: {total_findings} findings "
        f"({blocking} blocking, {review} review, {info} info)"
    )

    conn.close()


if __name__ == "__main__":
    main()
