#!/usr/bin/env python3
"""
exception_db.py

Immutable Exception Management Layer – Phase 1 Database Schema & CRUD.

SQLite-based storage for validation findings, remediation records with
hash-chain integrity, and audit snapshots.
"""

import hashlib
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS snapshots (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    input_file TEXT NOT NULL,
    input_file_hash TEXT NOT NULL,
    portfolio_type TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    canonical_path TEXT,
    status TEXT DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS findings (
    id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
    rule_id TEXT NOT NULL,
    severity TEXT NOT NULL,
    field_name TEXT,
    row_index INTEGER,
    message TEXT,
    classification TEXT,
    materiality TEXT,
    status TEXT DEFAULT 'open',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS remediations (
    id TEXT PRIMARY KEY,
    finding_id TEXT NOT NULL REFERENCES findings(id),
    snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
    action TEXT NOT NULL,
    field_name TEXT,
    row_index INTEGER,
    original_value TEXT,
    override_value TEXT,
    rule_id TEXT NOT NULL,
    justification TEXT NOT NULL,
    user_id TEXT NOT NULL,
    user_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    record_hash TEXT NOT NULL,
    prev_hash TEXT
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT,
    role TEXT DEFAULT 'analyst'
);
"""


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    """Create tables if they don't exist and return a connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def compute_record_hash(remediation_dict: dict) -> str:
    """SHA-256 of concatenated remediation fields (deterministic ordering)."""
    parts = [
        str(remediation_dict.get("finding_id", "")),
        str(remediation_dict.get("action", "")),
        str(remediation_dict.get("field_name", "")),
        str(remediation_dict.get("row_index", "")),
        str(remediation_dict.get("original_value", "")),
        str(remediation_dict.get("override_value", "")),
        str(remediation_dict.get("rule_id", "")),
        str(remediation_dict.get("justification", "")),
        str(remediation_dict.get("user_id", "")),
        str(remediation_dict.get("created_at", "")),
    ]
    payload = "".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_chain_head(conn: sqlite3.Connection, snapshot_id: str) -> Optional[str]:
    """Return the record_hash of the most recent remediation for *snapshot_id*."""
    row = conn.execute(
        "SELECT record_hash FROM remediations "
        "WHERE snapshot_id = ? ORDER BY created_at DESC LIMIT 1",
        (snapshot_id,),
    ).fetchone()
    return row["record_hash"] if row else None


# ---------------------------------------------------------------------------
# Snapshot CRUD
# ---------------------------------------------------------------------------

def create_snapshot(
    conn: sqlite3.Connection,
    input_file: str,
    file_hash: str,
    portfolio_type: str,
    row_count: int,
    canonical_path: Optional[str] = None,
) -> str:
    """Insert a new snapshot and return its UUID."""
    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO snapshots (id, created_at, input_file, input_file_hash, "
        "portfolio_type, row_count, canonical_path, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')",
        (sid, now, input_file, file_hash, portfolio_type, row_count, canonical_path),
    )
    conn.commit()
    return sid


# ---------------------------------------------------------------------------
# Findings CRUD
# ---------------------------------------------------------------------------

def _classify_rule(rule_id: str) -> str:
    """Map a rule_id to a classification category."""
    mapping = {
        "CORE001": "mandatory_null",
        "CORE002": "mandatory_null",
        "CORE003": "nd_not_permitted",
        "FMT_DATE": "value_format_error",
        "FMT_CCY_CODE": "value_format_error",
        "FMT_DEC": "value_format_error",
        "FMT_INT": "value_format_error",
        "FMT_BOOL": "value_format_error",
        "ENUM_INVALID": "enum_violation",
        "REG001": "schema_missing",
        "REG002": "mandatory_null",
    }
    return mapping.get(rule_id, "business_logic_violation")


def _determine_materiality(severity: str, classification: str) -> str:
    """Derive materiality from severity + classification."""
    sev = (severity or "").strip().lower()
    if sev == "error" and classification in ("mandatory_null", "schema_missing"):
        return "BLOCKING"
    if sev == "error":
        return "REVIEW"
    # warn / info / anything else
    return "INFO"


def ingest_findings(
    conn: sqlite3.Connection,
    snapshot_id: str,
    violations_csv_path: str,
    classification_fn=None,
) -> int:
    """
    Read a violations CSV, normalise its schema, classify each finding,
    and insert rows into the findings table.

    Supports both validator output formats:
      - validate_canonical:       rule_id, severity, field, row, message
      - validate_business_rules:  rule_id, severity, description, message, row_index

    Returns the number of findings inserted.
    """
    import csv as _csv

    if classification_fn is None:
        classification_fn = _classify_rule

    rows_inserted = 0
    now = datetime.now(timezone.utc).isoformat()

    with open(violations_csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            rule_id = row.get("rule_id", "UNKNOWN")
            severity = (row.get("severity") or "error").strip().lower()
            if severity in ("warning",):
                severity = "warn"

            # Normalise field_name: canonical uses "field", business uses no field
            field_name = row.get("field") or row.get("field_name") or None

            # Normalise row_index: canonical uses "row", business uses "row_index"
            raw_row = row.get("row") if row.get("row") not in (None, "") else row.get("row_index")
            try:
                row_index = int(raw_row) if raw_row not in (None, "") else None
            except (ValueError, TypeError):
                row_index = None

            message = row.get("message") or row.get("description") or ""
            classification = classification_fn(rule_id)
            materiality = _determine_materiality(severity, classification)

            fid = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO findings "
                "(id, snapshot_id, rule_id, severity, field_name, row_index, "
                "message, classification, materiality, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)",
                (
                    fid,
                    snapshot_id,
                    rule_id,
                    severity,
                    field_name,
                    row_index,
                    message,
                    classification,
                    materiality,
                    now,
                ),
            )
            rows_inserted += 1

    conn.commit()
    return rows_inserted


# ---------------------------------------------------------------------------
# Remediation CRUD
# ---------------------------------------------------------------------------

def create_remediation(
    conn: sqlite3.Connection,
    finding_id: str,
    snapshot_id: str,
    action: str,
    field_name: Optional[str],
    row_index: Optional[int],
    original_value: Optional[str],
    override_value: Optional[str],
    rule_id: str,
    justification: str,
    user_id: str,
    user_name: str,
) -> str:
    """
    Create an immutable remediation record with hash-chain linking.

    Returns the new remediation UUID.
    """
    rid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    rec = {
        "finding_id": finding_id,
        "action": action,
        "field_name": field_name,
        "row_index": row_index,
        "original_value": original_value,
        "override_value": override_value,
        "rule_id": rule_id,
        "justification": justification,
        "user_id": user_id,
        "created_at": now,
    }
    record_hash = compute_record_hash(rec)
    prev_hash = get_chain_head(conn, snapshot_id)

    conn.execute(
        "INSERT INTO remediations "
        "(id, finding_id, snapshot_id, action, field_name, row_index, "
        "original_value, override_value, rule_id, justification, "
        "user_id, user_name, created_at, record_hash, prev_hash) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            rid,
            finding_id,
            snapshot_id,
            action,
            field_name,
            row_index,
            original_value,
            override_value,
            rule_id,
            justification,
            user_id,
            user_name,
            now,
            record_hash,
            prev_hash,
        ),
    )

    # Update the finding status based on action
    status_map = {"override": "overridden", "accept": "accepted", "escalate": "escalated"}
    new_status = status_map.get(action, "open")
    conn.execute(
        "UPDATE findings SET status = ? WHERE id = ?",
        (new_status, finding_id),
    )

    conn.commit()
    return rid


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_findings(
    conn: sqlite3.Connection,
    snapshot_id: str,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    materiality: Optional[str] = None,
) -> list:
    """Return findings as a list of dicts, with optional filters."""
    query = "SELECT * FROM findings WHERE snapshot_id = ?"
    params: list = [snapshot_id]

    if status is not None:
        query += " AND status = ?"
        params.append(status)
    if severity is not None:
        query += " AND severity = ?"
        params.append(severity)
    if materiality is not None:
        query += " AND materiality = ?"
        params.append(materiality)

    query += " ORDER BY created_at"
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_remediations(conn: sqlite3.Connection, snapshot_id: str) -> list:
    """Return all remediations for a snapshot as a list of dicts."""
    rows = conn.execute(
        "SELECT * FROM remediations WHERE snapshot_id = ? ORDER BY created_at",
        (snapshot_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_snapshots(conn: sqlite3.Connection) -> list:
    """Return all snapshots ordered by creation time (newest first)."""
    rows = conn.execute(
        "SELECT * FROM snapshots ORDER BY created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]
