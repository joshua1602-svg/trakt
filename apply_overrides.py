#!/usr/bin/env python3
"""
apply_overrides.py

CLI script: reads the canonical CSV for a snapshot, applies all approved
overrides from the remediation ledger, and writes an overridden CSV plus
an overlay manifest JSON.

Usage:
    python apply_overrides.py \
      --snapshot-id abc123 \
      --db trakt_exceptions.db \
      --output-dir out
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from exception_db import init_db, get_remediations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply override remediations to the canonical CSV."
    )
    parser.add_argument("--snapshot-id", required=True, help="Snapshot UUID.")
    parser.add_argument(
        "--db",
        default="trakt_exceptions.db",
        help="Path to SQLite database (default: trakt_exceptions.db).",
    )
    parser.add_argument(
        "--output-dir",
        default="out",
        help="Directory to write the overridden CSV and manifest.",
    )
    args = parser.parse_args()

    conn = init_db(args.db)

    # Look up the snapshot
    row = conn.execute(
        "SELECT * FROM snapshots WHERE id = ?", (args.snapshot_id,)
    ).fetchone()
    if not row:
        print(f"ERROR: Snapshot {args.snapshot_id} not found.", file=sys.stderr)
        sys.exit(1)

    snapshot = dict(row)
    canonical_path = snapshot.get("canonical_path")
    if not canonical_path or not Path(canonical_path).exists():
        print(
            f"ERROR: Canonical CSV not found at '{canonical_path}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load the canonical CSV
    df = pd.read_csv(canonical_path, low_memory=False)

    # Get all override remediations for this snapshot
    remediations = get_remediations(conn, args.snapshot_id)
    overrides = [r for r in remediations if r["action"] == "override"]

    if not overrides:
        print("No override remediations found for this snapshot. Nothing to apply.")
        conn.close()
        return

    # Apply each override
    applied = []
    for ovr in overrides:
        field = ovr["field_name"]
        row_idx = ovr["row_index"]
        new_val = ovr["override_value"]

        if field is None or row_idx is None:
            print(
                f"WARNING: Skipping override {ovr['id']} — missing field_name or row_index.",
                file=sys.stderr,
            )
            continue

        if field not in df.columns:
            print(
                f"WARNING: Field '{field}' not found in canonical CSV. Skipping override {ovr['id']}.",
                file=sys.stderr,
            )
            continue

        if row_idx < 0 or row_idx >= len(df):
            print(
                f"WARNING: Row index {row_idx} out of range. Skipping override {ovr['id']}.",
                file=sys.stderr,
            )
            continue

        old_val = df.at[row_idx, field]
        df.at[row_idx, field] = new_val

        applied.append({
            "remediation_id": ovr["id"],
            "finding_id": ovr["finding_id"],
            "rule_id": ovr["rule_id"],
            "field_name": field,
            "row_index": row_idx,
            "original_value": str(old_val) if pd.notna(old_val) else None,
            "override_value": new_val,
            "justification": ovr["justification"],
            "user_name": ovr["user_name"],
            "applied_at": ovr["created_at"],
        })

    # Write output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(canonical_path).stem
    out_csv = out_dir / f"{stem}_overridden.csv"
    df.to_csv(out_csv, index=False)
    print(f"Overridden CSV written to: {out_csv}")

    # Write overlay manifest
    manifest = {
        "snapshot_id": args.snapshot_id,
        "canonical_source": canonical_path,
        "overrides_applied": len(applied),
        "overrides": applied,
    }
    manifest_path = out_dir / f"{stem}_override_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Override manifest written to: {manifest_path}")
    print(f"Applied {len(applied)} overrides.")

    conn.close()


if __name__ == "__main__":
    main()
