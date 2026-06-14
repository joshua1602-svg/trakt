"""
mapping_memory.py
=================

PART 9 / PART 10 — client-specific mapping memory for the Onboarding Agent.

Persists *approved* client-specific decisions so future onboarding runs for the
same client do not re-ask the same questions. Memory is **client-scoped** (never
global by default) and is stored as plain YAML under a per-client folder::

    {memory_root}/{client_id}/client_memory/
        mapping_memory.yaml          mapping_override | validation_only |
                                     out_of_scope | mark_unavailable
        source_precedence_memory.yaml source_precedence
        enum_memory.yaml             enum_mapping
        ignored_columns.yaml         ignore_column

For project-local tests / demos the memory root is the onboarding output dir, so
memory lands under ``onboarding_output/{client_id}/client_memory/``. Nothing here
ever writes into production config folders.

Design rules (PART 10):
  * Application order is: project/run-approved overrides -> CLIENT MEMORY ->
    alias/registry deterministic mapping -> ambiguity rules -> gaps/LLM. This
    module only handles the client-memory step; it mutates mapping candidates in
    place and reports what it applied.
  * Memory is **mode-aware and field-scope-safe**: a remembered mapping whose
    canonical target is out of scope for the current mode is *rejected*, never
    silently applied.
  * Memory never silently overrides a **material conflict**: when a remembered
    mapping still applies but the new data no longer matches the evidence that
    justified it, the mapping is kept but flagged for review and a warning gap is
    raised.

This module deliberately has **no pandas / streamlit dependency** so it imports
cheaply (workbench smoke tests, memory unit tests).
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

# Decision types a memory entry can carry.
DECISION_MAPPING_OVERRIDE = "mapping_override"
DECISION_SOURCE_PRECEDENCE = "source_precedence"
DECISION_ENUM_MAPPING = "enum_mapping"
DECISION_IGNORE_COLUMN = "ignore_column"
DECISION_VALIDATION_ONLY = "validation_only"
DECISION_OUT_OF_SCOPE = "out_of_scope"
DECISION_MARK_UNAVAILABLE = "mark_unavailable"

VALID_DECISION_TYPES = (
    DECISION_MAPPING_OVERRIDE,
    DECISION_SOURCE_PRECEDENCE,
    DECISION_ENUM_MAPPING,
    DECISION_IGNORE_COLUMN,
    DECISION_VALIDATION_ONLY,
    DECISION_OUT_OF_SCOPE,
    DECISION_MARK_UNAVAILABLE,
)

# decision_type -> memory file it is persisted in.
_DECISION_FILES = {
    DECISION_MAPPING_OVERRIDE: "mapping_memory.yaml",
    DECISION_VALIDATION_ONLY: "mapping_memory.yaml",
    DECISION_OUT_OF_SCOPE: "mapping_memory.yaml",
    DECISION_MARK_UNAVAILABLE: "mapping_memory.yaml",
    DECISION_SOURCE_PRECEDENCE: "source_precedence_memory.yaml",
    DECISION_ENUM_MAPPING: "enum_memory.yaml",
    DECISION_IGNORE_COLUMN: "ignored_columns.yaml",
}

_MEMORY_FILES = (
    "mapping_memory.yaml",
    "source_precedence_memory.yaml",
    "enum_memory.yaml",
    "ignored_columns.yaml",
)

# Below this value-match rate a remembered mapping is treated as MATERIALLY
# conflicting with the new data (and is flagged rather than silently applied).
MATERIAL_CONFLICT_THRESHOLD = 0.98


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def normalize_column(col: str) -> str:
    """Normalise a source column name for stable matching.

    ``loan_amount``, ``Loan Amount`` and ``loan  amount`` all normalise to the
    same token (``loan amount``) so memory recorded against one spelling matches
    the others on a future run.
    """
    return re.sub(r"[\s_]+", " ", str(col or "").strip().lower()).strip()


def file_matches(pattern: str, file_name: str) -> bool:
    """Glob-match a ``source_file_pattern`` against a file name (case-insensitive)."""
    if not pattern:
        return True
    fn = str(file_name or "")
    return fnmatch.fnmatch(fn.lower(), str(pattern).lower())


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Memory entry
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """One remembered, approved client decision (PART 9 schema)."""

    client_id: str = ""
    decision_type: str = DECISION_MAPPING_OVERRIDE
    source_file_pattern: str = "*"
    source_column: str = ""
    normalized_source_column: str = ""
    canonical_field: str = ""
    mode: str = ""
    domain: str = ""
    confidence: float = 1.0
    approved_by: str = ""
    approved_at: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    applies_to_future_runs: bool = True
    notes: str = ""
    # enum-specific qualifier (the raw source enum value being remembered).
    source_value: str = ""

    def __post_init__(self):
        if not self.normalized_source_column and self.source_column:
            self.normalized_source_column = normalize_column(self.source_column)
        if not self.approved_at:
            self.approved_at = _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryEntry":
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def resolve_memory_dir(
    memory_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    client_id: str = "",
) -> Path:
    """Resolve the client-scoped ``client_memory`` directory.

    Precedence: an explicit ``memory_dir`` wins; otherwise it is derived as
    ``{output_dir}/{client_id}/client_memory`` (project-local convention).
    """
    if memory_dir:
        return Path(memory_dir)
    if not client_id:
        raise ValueError("resolve_memory_dir requires either memory_dir or client_id")
    base = Path(output_dir) if output_dir else Path("onboarding_output")
    return base / client_id / "client_memory"


class MappingMemoryStore:
    """File-backed, client-scoped store of :class:`MemoryEntry` records."""

    def __init__(self, memory_dir: str | Path, client_id: str = ""):
        self.memory_dir = Path(memory_dir)
        self.client_id = client_id
        self._entries: List[MemoryEntry] = []
        self.load()

    # -- IO ------------------------------------------------------------
    def load(self) -> "MappingMemoryStore":
        self._entries = []
        for fname in _MEMORY_FILES:
            path = self.memory_dir / fname
            if not path.exists():
                continue
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
            if isinstance(raw, dict):  # tolerate {entries: [...]} form
                raw = raw.get("entries", []) or []
            for d in raw:
                if isinstance(d, dict):
                    self._entries.append(MemoryEntry.from_dict(d))
        return self

    def _write_file(self, fname: str, entries: List[MemoryEntry]) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        path = self.memory_dir / fname
        payload = [e.to_dict() for e in entries]
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def _flush(self) -> None:
        """Persist all in-memory entries back to their decision-type files."""
        by_file: Dict[str, List[MemoryEntry]] = {f: [] for f in _MEMORY_FILES}
        for e in self._entries:
            fname = _DECISION_FILES.get(e.decision_type, "mapping_memory.yaml")
            by_file[fname].append(e)
        for fname, entries in by_file.items():
            if entries or (self.memory_dir / fname).exists():
                self._write_file(fname, entries)

    # -- mutation ------------------------------------------------------
    def _dedupe_key(self, e: MemoryEntry) -> Tuple:
        return (
            e.client_id, e.decision_type, e.source_file_pattern,
            e.normalized_source_column or normalize_column(e.source_column),
            e.canonical_field, e.source_value,
        )

    def save_entry(self, entry: MemoryEntry) -> MemoryEntry:
        """Add (or replace a matching) entry and persist. Returns the entry."""
        if entry.decision_type not in VALID_DECISION_TYPES:
            raise ValueError(f"unknown decision_type: {entry.decision_type!r}")
        if not entry.client_id:
            entry.client_id = self.client_id
        if not entry.normalized_source_column and entry.source_column:
            entry.normalized_source_column = normalize_column(entry.source_column)
        key = self._dedupe_key(entry)
        self._entries = [e for e in self._entries if self._dedupe_key(e) != key]
        self._entries.append(entry)
        self._flush()
        return entry

    def save_entries(self, entries: List[MemoryEntry]) -> int:
        n = 0
        for e in entries:
            self.save_entry(e)
            n += 1
        return n

    # -- queries -------------------------------------------------------
    @property
    def entries(self) -> List[MemoryEntry]:
        return list(self._entries)

    def by_type(self, decision_type: str) -> List[MemoryEntry]:
        return [e for e in self._entries if e.decision_type == decision_type]

    def for_client(self, client_id: str) -> List[MemoryEntry]:
        return [e for e in self._entries if e.client_id == client_id]

    @property
    def is_empty(self) -> bool:
        return not self._entries

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self._entries:
            out[e.decision_type] = out.get(e.decision_type, 0) + 1
        out["total"] = len(self._entries)
        return out


# ---------------------------------------------------------------------------
# Application to a new run (PART 10)
# ---------------------------------------------------------------------------


def _entry_applies_to(entry: MemoryEntry, file_name: str, source_column: str) -> bool:
    if not entry.applies_to_future_runs:
        return False
    if not file_matches(entry.source_file_pattern, file_name):
        return False
    return normalize_column(source_column) == (
        entry.normalized_source_column or normalize_column(entry.source_column)
    )


def _scope_allows(field_scope: Any, canonical_field: str) -> bool:
    """True when ``canonical_field`` is in scope for the run (mode/field-scope-safe)."""
    if field_scope is None or not canonical_field:
        return True
    included = getattr(field_scope, "included_fields", None)
    if included:
        return canonical_field in included
    # Fall back to the explicit exclusion check.
    return not getattr(field_scope, "is_excluded", lambda _f: False)(canonical_field)


def apply_mapping_memory(
    mapping_candidates: List[Any],
    store: MappingMemoryStore,
    field_scope: Any = None,
    mode: str = "",
    conflict_signals: Optional[Dict[str, float]] = None,
    gap_start_index: int = 1000,
) -> Dict[str, Any]:
    """Apply remembered mapping decisions to ``mapping_candidates`` in place.

    Handles the mapping-affecting decision types (mapping_override,
    validation_only, out_of_scope, mark_unavailable, ignore_column). Returns a
    dict with::

        applied    - count of remembered mappings applied
        warned     - count applied but flagged due to a material conflict
        rejected   - count rejected (out of scope for the mode)
        ignored_columns - set of (file_name, source_column) marked ignored
        gap_questions   - warning GapQuestion-like dicts for material conflicts
        details    - per-entry application records (audit)

    ``conflict_signals`` maps a canonical field to the *current* value-match rate
    observed in the new data (e.g. from 04_source_overlap_analysis). When a
    remembered mapping's evidence claimed a high match but the current signal is
    materially lower, the mapping is kept but flagged and a warning is emitted.
    """
    conflict_signals = conflict_signals or {}
    overrides = [
        e for e in store.entries
        if e.decision_type in (
            DECISION_MAPPING_OVERRIDE, DECISION_VALIDATION_ONLY,
            DECISION_OUT_OF_SCOPE, DECISION_MARK_UNAVAILABLE,
        )
    ]
    ignores = store.by_type(DECISION_IGNORE_COLUMN)

    applied = warned = rejected = 0
    ignored_columns: Set[Tuple[str, str]] = set()
    gap_questions: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    gap_seq = gap_start_index

    for cand in mapping_candidates:
        file_name = getattr(cand, "source_file", "")
        source_column = getattr(cand, "source_column", "")

        # ignore_column wins: drop the mapping and remember it is ignored.
        ign = next((e for e in ignores if _entry_applies_to(e, file_name, source_column)), None)
        if ign is not None:
            ignored_columns.add((file_name, source_column))
            cand.candidate_canonical_field = ""
            cand.method = "ignored_by_client_memory"
            cand.requires_review = False
            cand.reason = "Source column ignored per client mapping memory."
            applied += 1
            details.append({"file": file_name, "column": source_column,
                            "decision": DECISION_IGNORE_COLUMN, "status": "applied"})
            continue

        entry = next((e for e in overrides if _entry_applies_to(e, file_name, source_column)), None)
        if entry is None:
            continue

        target = entry.canonical_field
        # Field-scope / mode safety: never apply an out-of-scope target.
        if entry.decision_type == DECISION_MAPPING_OVERRIDE and not _scope_allows(field_scope, target):
            rejected += 1
            details.append({"file": file_name, "column": source_column,
                            "canonical_field": target, "decision": entry.decision_type,
                            "status": "rejected_out_of_scope"})
            continue

        # Material-conflict guard: do not silently override a changed value.
        claimed = float((entry.evidence or {}).get("value_match_rate", 0) or 0)
        current = conflict_signals.get(target)
        material_conflict = (
            current is not None
            and claimed >= 0.99
            and current < MATERIAL_CONFLICT_THRESHOLD
        )

        if entry.decision_type == DECISION_MAPPING_OVERRIDE:
            cand.candidate_canonical_field = target
            cand.method = "client_memory"
            cand.confidence = max(float(getattr(cand, "confidence", 0) or 0), float(entry.confidence or 0))
            if material_conflict:
                cand.requires_review = True
                cand.reason = (
                    f"Client memory maps '{source_column}' to '{target}', but the new "
                    f"data no longer matches the evidence that justified it "
                    f"(was {claimed:.0%}, now {current:.0%}). Needs review."
                )
                warned += 1
                gap_seq += 1
                gap_questions.append({
                    "question_id": f"M{gap_seq}",
                    "category": "memory_conflict",
                    "severity": "high",
                    "question": (
                        f"Client memory says {source_column} maps to {target}, but values "
                        f"no longer match expected validation sources. Keep, re-map or ignore?"
                    ),
                    "reason": (
                        f"Remembered mapping evidence claimed a {claimed:.0%} value match; "
                        f"current data shows {current:.0%}."
                    ),
                    "candidate_answers": ["keep_memory_mapping", "provide_mapping_override",
                                           "mark_unavailable", "ignore"],
                    "default_recommendation": "provide_mapping_override",
                    "blocking_for": [],
                    "source_evidence": f"{file_name}:{source_column}",
                    "subject": target,
                    "subject_value": source_column,
                })
                details.append({"file": file_name, "column": source_column,
                                "canonical_field": target, "decision": entry.decision_type,
                                "status": "applied_with_warning"})
            else:
                cand.requires_review = False
                cand.reason = "Applied from client mapping memory (approved decision)."
                details.append({"file": file_name, "column": source_column,
                                "canonical_field": target, "decision": entry.decision_type,
                                "status": "applied"})
            applied += 1

        elif entry.decision_type == DECISION_VALIDATION_ONLY:
            cand.requires_review = False
            cand.reason = "Marked validation-only per client mapping memory."
            cand.method = "client_memory_validation_only"
            applied += 1
            details.append({"file": file_name, "column": source_column,
                            "decision": entry.decision_type, "status": "applied"})

        elif entry.decision_type in (DECISION_OUT_OF_SCOPE, DECISION_MARK_UNAVAILABLE):
            cand.candidate_canonical_field = ""
            cand.method = f"client_memory_{entry.decision_type}"
            cand.requires_review = False
            cand.reason = f"{entry.decision_type} per client mapping memory."
            ignored_columns.add((file_name, source_column))
            applied += 1
            details.append({"file": file_name, "column": source_column,
                            "decision": entry.decision_type, "status": "applied"})

    return {
        "applied": applied,
        "warned": warned,
        "rejected": rejected,
        "ignored_columns": ignored_columns,
        "gap_questions": gap_questions,
        "details": details,
    }


def resolved_enum_keys(store: MappingMemoryStore) -> Set[Tuple[str, str]]:
    """(normalized_field, source_value) pairs remembered as enum decisions."""
    out: Set[Tuple[str, str]] = set()
    for e in store.by_type(DECISION_ENUM_MAPPING):
        field_norm = normalize_column(e.canonical_field or e.source_column).replace(" ", "_")
        out.add((field_norm, e.source_value))
    return out


def resolved_precedence_fields(store: MappingMemoryStore) -> Set[str]:
    """Canonical fields with a remembered source-precedence decision."""
    return {e.canonical_field for e in store.by_type(DECISION_SOURCE_PRECEDENCE) if e.canonical_field}


def ignored_column_keys(
    store: MappingMemoryStore, inventory: List[Dict[str, Any]]
) -> Set[Tuple[str, str]]:
    """(file_name, column) pairs that ignore/out-of-scope memory suppresses.

    Resolves remembered ignore/out-of-scope/mark-unavailable entries against the
    actual file inventory + their columns so callers can suppress downstream gaps.
    """
    keys: Set[Tuple[str, str]] = set()
    suppress_types = (DECISION_IGNORE_COLUMN, DECISION_OUT_OF_SCOPE, DECISION_MARK_UNAVAILABLE)
    entries = [e for e in store.entries if e.decision_type in suppress_types]
    if not entries:
        return keys
    for inv in inventory or []:
        fname = inv.get("file_name", "")
        for e in entries:
            if not file_matches(e.source_file_pattern, fname):
                continue
            # We do not have the column list here; match on the remembered column.
            if e.source_column:
                keys.add((fname, e.source_column))
    return keys


def precedence_rules_from_memory(store: MappingMemoryStore) -> Dict[str, Any]:
    """Build a 13_source_precedence_rules-style dict from precedence memory."""
    rules: Dict[str, Any] = {}
    for e in store.by_type(DECISION_SOURCE_PRECEDENCE):
        if not e.canonical_field:
            continue
        ev = e.evidence or {}
        rules[e.canonical_field] = {
            "primary_source_file": ev.get("primary_source_file", ""),
            "primary_source_column": ev.get("primary_source_column", ""),
            "secondary_source_file": ev.get("secondary_source_file", ""),
            "secondary_source_column": ev.get("secondary_source_column", ""),
            "reconciliation_status": ev.get("reconciliation_status", "review_required"),
            "approved_by": e.approved_by,
            "from_client_memory": True,
        }
    return rules


def summarize_application(result: Dict[str, Any], store: MappingMemoryStore) -> Dict[str, Any]:
    """A compact summary for the run summary / review pack (PART 10)."""
    return {
        "client_mapping_memory_loaded": not store.is_empty,
        "memory_entries_total": store.counts().get("total", 0),
        "memory_entries_applied": int(result.get("applied", 0)),
        "memory_entries_warned": int(result.get("warned", 0)),
        "memory_entries_rejected": int(result.get("rejected", 0)),
        "memory_entry_counts_by_type": {
            k: v for k, v in store.counts().items() if k != "total"
        },
    }
