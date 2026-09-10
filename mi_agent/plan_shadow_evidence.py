#!/usr/bin/env python3
"""What a shadowed request leaves behind. Written before anything uses it.

WHY THIS EXISTS AT ALL, AND WHY FIRST. The live integration sign-off recorded
only a projection of each governed plan, not the model's raw payload — so one
case (I04) could not be re-adjudicated afterwards without buying the
interpretation again. That is the mistake this module exists to prevent: a
shadowed request has to leave behind enough to settle any question about it
OFFLINE, months later, with no model and no deployment.

So the record carries every stage whole — the raw model response, the
CandidateIntent with its ambiguities, the complete CompileResult including the
plan's own provenance (intent claims, compiler bindings, normalisation), every
eligibility reason, the bound executable contract, the deterministic result with
its grouped cells and receipt, and the legacy control's disposition.

WHAT IT MUST NEVER CARRY. Credentials and borrower rows, and the guard is
structural rather than a promise: `redact` drops any key whose name looks like a
credential, masks any value carrying a known secret prefix, and drops any value
that is a DataFrame or Series outright — a frame is the one shape a borrower row
could arrive in. Every drop is recorded as a redaction note, so a reader can see
that something was removed rather than wonder.

IT CANNOT BREAK A REQUEST. Every function here returns rather than raises, and a
sink that cannot be written increments a counter and logs — a shadow that could
take down a served answer would be worse than no shadow, and that applies to its
evidence as much as to its arithmetic.

AND IT WILL NOT WRITE INTO THE REPOSITORY. A sink path inside a git working tree
is refused, because production evidence accumulating in a checkout is how a
secret ends up in a commit.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger("mi_agent.plan_shadow_evidence")

SCHEMA_VERSION = "plan_shadow_evidence/1.0"

#: Where records go. A path ending `.jsonl` is appended to; anything else is
#: treated as a directory and gets one file per record. Absent means "record
#: nothing", so a deployment can run the wiring without producing evidence it has
#: nowhere to put.
SINK_ENV_VAR = "MI_AGENT_PLAN_SHADOW_EVIDENCE"

# Dispositions. Every shadowed request ends in exactly one of these, and each is
# a recorded outcome rather than an absence — "nothing was written" must never be
# the way a failure shows up.
OUTSIDE_CANARY = "OUTSIDE_CANARY"
SHADOW_SKIPPED_BUSY = "SHADOW_SKIPPED_BUSY"
INTERPRETER_FAILURE = "INTERPRETER_FAILURE"
CLARIFY = "CLARIFY"
REFUSE = "REFUSE"
INELIGIBLE = "INELIGIBLE"
EXECUTED = "EXECUTED"
EXECUTION_ERROR = "EXECUTION_ERROR"
ORCHESTRATION_ERROR = "ORCHESTRATION_ERROR"

DISPOSITIONS = frozenset({
    OUTSIDE_CANARY, SHADOW_SKIPPED_BUSY, INTERPRETER_FAILURE, CLARIFY, REFUSE,
    INELIGIBLE, EXECUTED, EXECUTION_ERROR, ORCHESTRATION_ERROR})

#: A key whose name contains any of these is dropped, whatever it holds.
FORBIDDEN_KEY_SUBSTRINGS = ("authorization", "bearer", "api_key", "apikey",
                            "access_token", "id_token", "refresh_token",
                            "secret", "password", "credential", "cookie")

#: A string value containing any of these is masked rather than dropped, so the
#: record still shows that a field was present.
SECRET_VALUE_MARKERS = ("sk-ant-", "bearer ")

REDACTED = "[REDACTED]"

#: An upper bound on recorded grouped cells. A breakdown wider than this is
#: recorded truncated WITH a note, because an unbounded record is its own risk.
MAX_CELLS = 5000

_failures = 0
_written = 0


def evidence_failures() -> int:
    """How many records could not be persisted. The observable failure signal."""
    return _failures


def evidence_written() -> int:
    return _written


def reset_counters() -> None:
    """For tests. Production never calls this."""
    global _failures, _written
    _failures = 0
    _written = 0


def correlation_id() -> str:
    """One opaque id per shadowed request. Not derived from any user value."""
    return f"shadow_{uuid.uuid4().hex[:20]}"


def new_record(*, correlation_id: str, question: str,
               client_id: Optional[str] = None, run_id: Optional[str] = None,
               view: Optional[str] = None,
               portfolio_id: Optional[str] = None) -> Dict[str, Any]:
    """The skeleton every shadowed request fills in, stage by stage.

    The stage keys exist from the start and hold None until a stage runs, so an
    abandoned record says WHICH stage it got to rather than merely being short.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "correlation_id": correlation_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "request": {
            "question": question,
            "client_id": client_id,
            "run_id": run_id,
            "dataset_view": view,
            "portfolio_id": portfolio_id,
        },
        "disposition": None,
        "model": None,
        "interpretation": None,
        "compiler": None,
        "eligibility": None,
        "execution": None,
        "legacy_control": None,
        "redactions": [],
        "evidence_persisted": None,
    }


def _is_frame(value: Any) -> bool:
    """A DataFrame or Series, without importing pandas to find out."""
    return type(value).__name__ in ("DataFrame", "Series", "Index")


def redact(node: Any, *, notes: Optional[List[str]] = None,
           path: str = "") -> Tuple[Any, List[str]]:
    """Strip credentials and frames, recording every removal.

    Returns a new structure; the input is not modified, so a caller cannot end up
    having mutated a live result envelope by asking for it to be recorded.
    """
    notes = [] if notes is None else notes

    if _is_frame(node):
        notes.append(f"{path or 'root'}: dropped a {type(node).__name__} — a "
                     f"frame is how a borrower row would arrive")
        return None, notes

    if isinstance(node, Mapping):
        clean: Dict[str, Any] = {}
        for key, value in node.items():
            name = str(key)
            where = f"{path}.{name}" if path else name
            if any(token in name.lower() for token in FORBIDDEN_KEY_SUBSTRINGS):
                notes.append(f"{where}: dropped — the key name reads as a "
                             f"credential")
                continue
            child, notes = redact(value, notes=notes, path=where)
            clean[name] = child
        return clean, notes

    if isinstance(node, (list, tuple)):
        out = []
        for index, item in enumerate(node):
            child, notes = redact(item, notes=notes, path=f"{path}[{index}]")
            out.append(child)
        return out, notes

    if isinstance(node, str):
        low = node.lower()
        if any(marker in low for marker in SECRET_VALUE_MARKERS):
            notes.append(f"{path or 'root'}: masked — the value carries a "
                         f"credential marker")
            return REDACTED, notes
        return node, notes

    if isinstance(node, (int, float, bool)) or node is None:
        return node, notes

    # Anything else is recorded as its repr rather than silently dropped: an
    # unexpected type in the evidence is itself worth seeing.
    return f"<{type(node).__name__}> {str(node)[:200]}", notes


def _sink_path() -> str:
    return str(os.environ.get(SINK_ENV_VAR) or "").strip()


def _inside_a_git_working_tree(path: str) -> bool:
    """Whether `path` sits under a directory containing `.git`."""
    current = os.path.abspath(path if os.path.isdir(path)
                              else os.path.dirname(path) or ".")
    while True:
        if os.path.exists(os.path.join(current, ".git")):
            return True
        parent = os.path.dirname(current)
        if parent == current:
            return False
        current = parent


def write(record: Dict[str, Any]) -> bool:
    """Persist one record. Returns whether it landed; never raises.

    Redaction happens HERE rather than at the call sites, so there is exactly one
    place where a record can reach a disk and exactly one place that has to be
    right about credentials.
    """
    global _failures, _written
    try:
        clean, notes = redact(record)
        if notes:
            clean["redactions"] = list(clean.get("redactions") or ()) + notes

        path = _sink_path()
        if not path:
            record["evidence_persisted"] = False
            return False

        if _inside_a_git_working_tree(path):
            _failures += 1
            logger.warning(
                "shadow evidence sink %r is inside a git working tree; refusing "
                "to write production evidence into a checkout", path)
            record["evidence_persisted"] = False
            return False

        clean["evidence_persisted"] = True
        body = json.dumps(clean, default=str)
        if path.endswith(".jsonl"):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(body + "\n")
        else:
            os.makedirs(path, exist_ok=True)
            name = f"{record.get('correlation_id') or correlation_id()}.json"
            with open(os.path.join(path, name), "w", encoding="utf-8") as handle:
                handle.write(json.dumps(clean, indent=2, default=str) + "\n")
        _written += 1
        record["evidence_persisted"] = True
        return True
    except Exception:                                                # noqa: BLE001
        _failures += 1
        logger.warning("shadow evidence could not be persisted", exc_info=True)
        try:
            record["evidence_persisted"] = False
        except Exception:                                            # noqa: BLE001
            pass
        return False


def cells_of(frame: Any, dimensions: List[str], value_column: str
             ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Grouped cells as `[{dimension: key, ..., "value": figure}]`, bounded.

    Aggregates and group KEYS only — a group key is a governed category like a
    region or an LTV band, never a row. Returns `(cells, note)`; the note is set
    when the grid was truncated or could not be read.
    """
    try:
        if frame is None or getattr(frame, "empty", True):
            return [], "the execution produced no rows"
        # NOT `getattr(...) or ()`: a pandas Index raises on a truth test, so the
        # `or` made every grouped capture fail with "truth value is ambiguous"
        # and record an empty grid. Caught by this module's own tests.
        columns = getattr(frame, "columns", None)
        columns = [] if columns is None else list(columns)
        if value_column not in columns:
            candidates = [c for c in columns if c not in dimensions]
            if not candidates:
                return [], "no value column in the result"
            value_column = candidates[0]
        cells: List[Dict[str, Any]] = []
        note = None
        for position, (_, row) in enumerate(frame.iterrows()):
            if position >= MAX_CELLS:
                note = (f"truncated at {MAX_CELLS} cells of "
                        f"{len(frame)} — recorded bounded on purpose")
                break
            cell = {dimension: str(row[dimension]) for dimension in dimensions}
            try:
                cell["value"] = float(row[value_column])
            except (TypeError, ValueError):
                cell["value"] = None
            cells.append(cell)
        return cells, note
    except Exception as exc:                                         # noqa: BLE001
        return [], f"cells unreadable: {type(exc).__name__}: {exc}"[:200]
