#!/usr/bin/env python3
"""
export_audit_pack.py

CLI script: generates an audit-ready ZIP archive containing snapshot
metadata, findings register, remediation ledger with hash-chain
verification, and override manifest.

Usage:
    python export_audit_pack.py \
      --snapshot-id abc123 \
      --db trakt_exceptions.db \
      --output audit_pack_abc123.zip
"""

import argparse
import csv
import io
import json
import sys
import zipfile

from exception_db import (
    init_db,
    get_findings,
    get_remediations,
    compute_record_hash,
)


def verify_chain(remediations: list) -> list:
    """
    Walk the remediation ledger and verify the hash chain.

    Returns a list of per-record verification results:
        {"remediation_id": ..., "record_hash": ..., "expected_hash": ...,
         "prev_hash": ..., "expected_prev": ..., "record_ok": bool, "chain_ok": bool}
    """
    results = []
    prev_expected: str | None = None

    for r in remediations:
        # Recompute the record hash
        rec = {
            "finding_id": r["finding_id"],
            "action": r["action"],
            "field_name": r["field_name"],
            "row_index": r["row_index"],
            "original_value": r["original_value"],
            "override_value": r["override_value"],
            "rule_id": r["rule_id"],
            "justification": r["justification"],
            "user_id": r["user_id"],
            "created_at": r["created_at"],
        }
        expected_hash = compute_record_hash(rec)
        record_ok = r["record_hash"] == expected_hash

        # Verify chain link
        chain_ok = r["prev_hash"] == prev_expected

        results.append({
            "remediation_id": r["id"],
            "record_hash": r["record_hash"],
            "expected_hash": expected_hash,
            "prev_hash": r["prev_hash"],
            "expected_prev": prev_expected,
            "record_ok": record_ok,
            "chain_ok": chain_ok,
        })

        prev_expected = r["record_hash"]

    return results


def _dict_list_to_csv_bytes(rows: list, fieldnames: list | None = None) -> bytes:
    """Serialise a list of dicts to CSV bytes."""
    if not rows:
        return b""
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue().encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export an audit pack ZIP for a snapshot."
    )
    parser.add_argument("--snapshot-id", required=True, help="Snapshot UUID.")
    parser.add_argument(
        "--db",
        default="trakt_exceptions.db",
        help="Path to SQLite database (default: trakt_exceptions.db).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output ZIP path (default: audit_pack_<snapshot_id>.zip).",
    )
    args = parser.parse_args()

    conn = init_db(args.db)

    # Fetch snapshot
    row = conn.execute(
        "SELECT * FROM snapshots WHERE id = ?", (args.snapshot_id,)
    ).fetchone()
    if not row:
        print(f"ERROR: Snapshot {args.snapshot_id} not found.", file=sys.stderr)
        sys.exit(1)

    snapshot = dict(row)
    findings = get_findings(conn, args.snapshot_id)
    remediations = get_remediations(conn, args.snapshot_id)

    # Compute summary stats
    materiality_counts = {"BLOCKING": 0, "REVIEW": 0, "INFO": 0}
    status_counts = {"open": 0, "accepted": 0, "overridden": 0, "escalated": 0}
    for f in findings:
        mat = f.get("materiality", "INFO")
        if mat in materiality_counts:
            materiality_counts[mat] += 1
        st_val = f.get("status", "open")
        if st_val in status_counts:
            status_counts[st_val] += 1

    # 1. snapshot_metadata.json
    metadata = {
        "snapshot": snapshot,
        "summary": {
            "total_findings": len(findings),
            "materiality": materiality_counts,
            "status": status_counts,
            "total_remediations": len(remediations),
        },
    }

    # 2. findings_register.csv
    findings_fieldnames = [
        "id", "snapshot_id", "rule_id", "severity", "field_name",
        "row_index", "message", "classification", "materiality",
        "status", "created_at",
    ]

    # 3. remediation_ledger.csv
    remediation_fieldnames = [
        "id", "finding_id", "snapshot_id", "action", "field_name",
        "row_index", "original_value", "override_value", "rule_id",
        "justification", "user_id", "user_name", "created_at",
        "record_hash", "prev_hash",
    ]

    # 4. override_manifest.json
    overrides = [r for r in remediations if r["action"] == "override"]
    override_manifest = {
        "snapshot_id": args.snapshot_id,
        "overrides_applied": len(overrides),
        "overrides": [
            {
                "remediation_id": o["id"],
                "finding_id": o["finding_id"],
                "rule_id": o["rule_id"],
                "field_name": o["field_name"],
                "row_index": o["row_index"],
                "original_value": o["original_value"],
                "override_value": o["override_value"],
                "justification": o["justification"],
                "user_name": o["user_name"],
                "created_at": o["created_at"],
            }
            for o in overrides
        ],
    }

    # 5. chain_integrity.json
    chain_results = verify_chain(remediations)
    all_record_ok = all(r["record_ok"] for r in chain_results) if chain_results else True
    all_chain_ok = all(r["chain_ok"] for r in chain_results) if chain_results else True
    chain_integrity = {
        "snapshot_id": args.snapshot_id,
        "total_records": len(chain_results),
        "all_records_valid": all_record_ok,
        "chain_intact": all_chain_ok,
        "overall": "PASS" if (all_record_ok and all_chain_ok) else "FAIL",
        "records": chain_results,
    }

    # Build ZIP
    output_path = args.output or f"audit_pack_{args.snapshot_id[:8]}.zip"

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "snapshot_metadata.json",
            json.dumps(metadata, indent=2, default=str),
        )
        zf.writestr(
            "findings_register.csv",
            _dict_list_to_csv_bytes(findings, findings_fieldnames).decode("utf-8"),
        )
        zf.writestr(
            "remediation_ledger.csv",
            _dict_list_to_csv_bytes(remediations, remediation_fieldnames).decode("utf-8"),
        )
        zf.writestr(
            "override_manifest.json",
            json.dumps(override_manifest, indent=2, default=str),
        )
        zf.writestr(
            "chain_integrity.json",
            json.dumps(chain_integrity, indent=2, default=str),
        )

    print(f"Audit pack written to: {output_path}")
    print(f"  Findings:      {len(findings)}")
    print(f"  Remediations:  {len(remediations)}")
    print(f"  Overrides:     {len(overrides)}")
    print(f"  Chain integrity: {chain_integrity['overall']}")

    conn.close()


if __name__ == "__main__":
    main()
