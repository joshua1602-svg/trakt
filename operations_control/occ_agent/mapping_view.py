"""operations_control.occ_agent.mapping_view — every column, and what became of it.

WHY THIS MODULE EXISTS

``SyntheticRun.mapping_report`` has always held one row per source column: the
column, the canonical field the header mapper matched it to, the tier it
matched at, the confidence, and a note. It travelled in the readiness package
and in the review package, it was typed in the frontend — and no screen ever
rendered it.

The consequence is not cosmetic. An operator saw only the columns the mapper
could NOT settle, raised as decisions. Everything it settled on its own — the
large majority, and the part nobody ever checks — was invisible. On a hundred-
column Annex 2 tape that is the difference between answering the twenty-nine
questions asked and being able to see all hundred and seven answers.

WHY THE CLASSIFICATION IS DONE HERE AND NOT IN THE BROWSER

A row is "accepted automatically" when its tier is trusted OR its confidence
clears the threshold — the same test :mod:`execution` applies when deciding
whether to use a mapping without asking. Re-expressing that test in TypeScript
would make a screen that can disagree with the engine about which mappings were
checked by a human, which is exactly the kind of quiet divergence this
platform has been bitten by before. The constants are imported, not copied.

WHAT THIS IS NOT

Not a second mapping path, and not a place a mapping can be changed. It reads
what the run already recorded and says it out loud. Changing a mapping is
answering its decision, which has its own governed route.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .execution import LOW_CONFIDENCE, _TRUSTED_TIERS

#: What became of one source column. The order is the order of a reader's
#: interest, worst first — a column nobody has looked at matters more than one
#: that matched its own name exactly.
ROW_NEEDS_YOU = "needs_you"
ROW_UNREADABLE = "unreadable"
ROW_UNCHECKED = "unchecked"
ROW_UNUSED = "unused"
ROW_CONFIRMED = "confirmed"
ROW_AUTOMATIC = "automatic"

STATE_ORDER = (ROW_NEEDS_YOU, ROW_UNREADABLE, ROW_UNCHECKED, ROW_UNUSED,
               ROW_CONFIRMED, ROW_AUTOMATIC)

STATE_LABELS = {
    ROW_NEEDS_YOU: "Needs you",
    ROW_UNREADABLE: "Could not be read",
    #: A weak match in a file the canonical tape is NOT built from. Worth an
    #: operator's eye — the real orchestrator consolidates loan-domain fields
    #: from these files — but no decision was raised against it, so calling it
    #: "Needs you" would point at a question that does not exist.
    ROW_UNCHECKED: "Weak match, nothing asked",
    ROW_UNUSED: "Not used",
    ROW_CONFIRMED: "You confirmed it",
    ROW_AUTOMATIC: "Matched automatically",
}

#: How each tier reads to somebody who did not write the mapper. The tier names
#: are the mapper's own vocabulary (``semantic_alignment.HeaderMapper.map_one``)
#: and mean nothing to an operator; what they need is how firm the evidence is.
TIER_LABELS = {
    "exact": "The column is named exactly as the field is",
    "normalized": "The names match once case and punctuation are ignored",
    "alias": "A known alias for this field",
    "token_set": "The words overlap, but the names are not the same",
    "fuzz_token_set": "The words are similar, not the same",
    "fuzz_ratio_norm": "The names are spelled similarly",
    "unmapped": "Nothing Trakt reports on resembles this column",
    "empty": "The column has no name",
    "operator_approved": "You said so",
    "unreadable": "The file could not be read",
}


def classify(row: Dict[str, Any], *, has_open_decision: bool = False) -> str:
    """What became of one column, from the row the mapper wrote.

    An open decision wins over the tier. Two columns claiming one canonical
    field is an ambiguity, and ``execution`` raises it as a decision AFTER both
    rows have already been written to the report at whatever tier they matched
    at — often an exact or alias match, because that is how both came to claim
    the same field. Reading the tier alone, the table would say "matched
    automatically" about a column the run is blocked on.

    A weak match outside the primary tape is ``ROW_UNCHECKED`` rather than
    ``ROW_NEEDS_YOU``: the canonical tape is not built from that file, so no
    decision was raised and there is nothing for an operator to answer. It is
    still reported, because the real orchestrator consolidates loan-domain
    fields from those files whatever this adapter does with them.
    """
    if has_open_decision:
        return ROW_NEEDS_YOU
    tier = str(row.get("tier") or "")
    if tier == "operator_approved":
        return ROW_CONFIRMED
    if tier == "unreadable":
        return ROW_UNREADABLE
    if not str(row.get("canonical_field") or ""):
        return ROW_UNUSED
    confidence = row.get("confidence")
    trusted = (tier in _TRUSTED_TIERS
               or (confidence is not None and float(confidence) >= LOW_CONFIDENCE))
    if trusted:
        return ROW_AUTOMATIC
    # Absent on rows written before the report covered every file, and those
    # were all primary-tape rows.
    return ROW_NEEDS_YOU if row.get("primary", True) else ROW_UNCHECKED


def _label(canonical_field: str) -> str:
    return str(canonical_field or "").replace("_", " ").strip()


def _open_decision_by_column(decisions: Iterable[Dict[str, Any]]
                             ) -> Dict[str, str]:
    """Which open decision, if any, belongs to each source column.

    So a row reading "Needs you" can be the thing you click, rather than
    sending an operator to hunt for the matching question in another list.
    """
    out: Dict[str, str] = {}
    for decision in decisions or []:
        if str(decision.get("status", "open")) != "open":
            continue
        subject = decision.get("subject") or {}
        column = str(subject.get("source_column") or "").strip()
        if column:
            out.setdefault(column.lower(), str(decision.get("decision_id") or ""))
        for column in (subject.get("source_columns") or []):
            name = str(column or "").strip()
            if name:
                out.setdefault(name.lower(),
                               str(decision.get("decision_id") or ""))
    return out


def overview(run: Any) -> Dict[str, Any]:
    """Every source column and what became of it, ready to render.

    Ordered worst-first within each file, so the columns that need a person are
    at the top of the table rather than wherever the tape happened to put them.
    """
    decisions = _open_decision_by_column(getattr(run, "open_decisions", []))
    rows: List[Dict[str, Any]] = []
    for raw in getattr(run, "mapping_report", []) or []:
        column = str(raw.get("source_column") or "")
        decision_id = decisions.get(column.lower(), "")
        state = classify(raw, has_open_decision=bool(decision_id))
        confidence: Optional[float] = None
        if raw.get("confidence") is not None:
            try:
                confidence = round(float(raw["confidence"]), 4)
            except (TypeError, ValueError):
                confidence = None
        tier = str(raw.get("tier") or "")
        rows.append({
            "source_file": str(raw.get("source_file") or ""),
            "source_column": column,
            "canonical_field": str(raw.get("canonical_field") or ""),
            "field_label": _label(raw.get("canonical_field")),
            "tier": tier,
            "tier_label": TIER_LABELS.get(tier, tier.replace("_", " ")),
            "confidence": confidence,
            "note": str(raw.get("note") or ""),
            "state": state,
            "state_label": STATE_LABELS[state],
            "decision_id": decision_id,
            # Whether this column's file is the one the canonical tape is built
            # from. The table groups by file and says which is which, so an
            # operator is not left to infer it from the filename.
            "primary": bool(raw.get("primary", True)),
        })

    # The primary tape first — it is the one the canonical tape is built from,
    # so it is what an operator checks first — then the rest alphabetically,
    # and within each file the rows that want attention at the top.
    rows.sort(key=lambda r: (not r["primary"], r["source_file"],
                             STATE_ORDER.index(r["state"]),
                             r["source_column"].lower()))
    counts = {state: sum(1 for r in rows if r["state"] == state)
              for state in STATE_ORDER}
    # "Mapped" is what is FEEDING a canonical field, so a proposal waiting on
    # an operator does not count. Counting it would make a blocked run read as
    # more complete than a finished one.
    in_use = counts[ROW_CONFIRMED] + counts[ROW_AUTOMATIC]
    files: List[Dict[str, Any]] = []
    for row in rows:
        name = row["source_file"]
        if not name or any(f["name"] == name for f in files):
            continue
        files.append({"name": name, "primary": row["primary"],
                      "columns": sum(1 for r in rows
                                     if r["source_file"] == name)})
    return {
        "rows": rows,
        "counts": {**counts, "columns": len(rows), "mapped": in_use},
        "files": files,
    }
