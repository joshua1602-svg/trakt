"""operations_control.occ_agent.field_registry — somewhere for a column that
matched nothing to go.

THE DEAD END THIS REMOVES. A delivery's columns came back in three states: one
Trakt matched, one it half-matched and asked about, and one it could not place
at all. The third was labelled "Not used" and that was the end of the screen. On
a first delivery that is most of the tape — eighty-nine of a hundred and fifty
three for the first client through this — and an operator looking at a column
they KNOW the meaning of had nothing to do about it. The knowledge that would
have fixed the mapping was in the room and there was no way to write it down.

THE TWO THINGS A PERSON ACTUALLY KNOWS, AND WHY THEY ARE DIFFERENT ACTS

  * "This is our name for a field Trakt already has." A lender calls the
    outstanding balance ``CurBalGBP``; the platform calls it
    ``current_principal_balance``. Nothing new exists — a name does. That is an
    ALIAS, and it is this client's. It is staged like every other answer on the
    mapping table (:mod:`.staging`), applied when the operator commits the set,
    and promoted at activation into a governed rule scoped to the portfolio.

  * "Trakt has no field for this." That is not a mapping at all. It is a change
    to the platform's canonical vocabulary, which every client, every regime
    projection and every validation rule reads. :mod:`operations_control.rules`
    is explicit that "the core field registry is never written" from the
    operations container, and the registry has its own governed route — a
    versioned system config package, drafted, validated and activated by an
    administrator (:mod:`operations_control.configuration.packages`).

So the second one is a REQUEST, recorded with the column that prompted it, the
sample values behind it, who asked, and why. It does not map anything and it
does not block the delivery: the column stays out, visibly, and the request
travels with the case for a configuration owner to action. An onboarding
operator adding a canonical field with one click would be a client's first
delivery changing what every other client's report means.

WHAT IS *NOT* HERE. No fuzzy matching, no suggestion, no second mapper. The
operator has already read the column and the table has already told them what
Trakt made of it. This module takes their answer, checks it against the
registry, and records it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..engine import OpsError
from .execution import REGISTRY_PATH, mapping_key

#: The decision type an alias is recorded as. Deliberately the SAME type a
#: confirmed mapping uses: the operator is answering "what field is this
#: column?", which is the confirmation question, and
#: :mod:`.mapping_promotion` already knows how to turn that answer into a
#: governed rule. A type of its own would need a second promotion path to say
#: exactly the same thing.
ALIAS_DECISION_TYPE = "mapping_confirmation"

#: Marks a decision this module wrote rather than one the mapper raised, so the
#: run can always say a mapping came from an operator naming a field outright
#: rather than from confirming something Trakt proposed.
BASIS_OPERATOR = "operator"

#: Where a requested field stands. It never becomes ``registered`` from here —
#: only an activated system config version does that — but the run records the
#: ask and its withdrawal, which are the parts an operator controls.
REQUEST_OPEN = "requested"
REQUEST_WITHDRAWN = "withdrawn"


class FieldRegistryError(OpsError):
    """An unmapped column could not be given somewhere to go."""

    def __init__(self, code: str, message: str, http_status: int = 409):
        super().__init__(code, message, http_status=http_status)


def catalogue(asset_type: str, *, registry_path: Path = REGISTRY_PATH,
              regime: str = "") -> List[Dict[str, Any]]:
    """Every canonical field this book may map to, ready for a picker.

    The SAME selection the mapper itself works from
    (:func:`engine.gate_1_alignment.semantic_alignment.select_registry_fields`),
    so an operator can never choose a field the run would then refuse — the
    list on the screen and the list the engine matches against are one list.
    """
    from engine.gate_1_alignment.semantic_alignment import (
        load_field_registry,
        select_registry_fields,
    )

    registry = load_field_registry(Path(registry_path))
    fields = registry.get("fields") or {}
    out: List[Dict[str, Any]] = []
    for name in select_registry_fields(registry, asset_type or ""):
        meta = fields.get(name) or {}
        codes = meta.get("regime_mapping") or {}
        entry = {
            "name": name,
            "label": str(name).replace("_", " "),
            "category": str(meta.get("category") or ""),
            "format": str(meta.get("format") or ""),
            "layer": str(meta.get("layer") or ""),
            "core_canonical": bool(meta.get("core_canonical")),
            # Which regulatory code this field answers, where it answers one.
            # An operator choosing between two plausible fields is choosing
            # between two regulatory obligations, and the screen should say so.
            "regimes": sorted(str(k) for k in codes),
        }
        if regime and regime in codes:
            entry["regime_code"] = str((codes.get(regime) or {}).get("code")
                                       or "")
        out.append(entry)
    return out


def known_field(name: str, asset_type: str, *,
                registry_path: Path = REGISTRY_PATH) -> bool:
    """Is this a canonical field this book may map to?"""
    return any(f["name"] == str(name or "")
               for f in catalogue(asset_type, registry_path=registry_path))


def registered_anywhere(name: str, *,
                        registry_path: Path = REGISTRY_PATH) -> bool:
    """Is this a canonical field AT ALL, for any book?

    Asked separately from :func:`known_field` because the two have different
    remedies. A field that exists but belongs to another portfolio type is a
    configuration question; a field that exists for THIS book is simply the
    alias action; a name that exists nowhere is a genuine request.
    """
    from engine.gate_1_alignment.semantic_alignment import load_field_registry
    registry = load_field_registry(Path(registry_path))
    return str(name or "") in (registry.get("fields") or {})


def alias_decision(*, source_file: str, source_column: str,
                   target_field: str, actor: str, reason: str,
                   at: str) -> Dict[str, Any]:
    """An operator naming the field a column feeds, recorded as settled.

    It arrives already approved, because there was never a question: nothing
    asked the operator anything about this column — they volunteered what it
    is. The record still carries an approver and a timestamp, because that is
    what promotion reads and what an auditor asking "who said ``Pool Ref`` was
    the portfolio identifier?" has to be able to answer.

    An EMPTY ``target_field`` is the operator saying this column feeds nothing,
    which is an answer and not an absence of one. It is recorded the same way,
    so the next delivery does not ask about the column again.
    """
    words = str(target_field).replace("_", " ")
    title = (f"'{source_column}' is {words}" if target_field
             else f"'{source_column}' is not used")
    return {
        "decision_id": _alias_id(source_file, source_column),
        "kind": "field_mapping",
        "title": title,
        "question": (f"Trakt could not place '{source_column}'. What field "
                     "does it feed?"),
        "blocking": False,
        "status": "approved",
        "issue": f"'{source_column}' is not a column Trakt recognised.",
        "evidence": [{"label": "What the operator said", "kind": "text",
                      "data": {"issue": (f"{source_column} → {target_field}"
                                         if target_field
                                         else f"{source_column} → not used"),
                               "detail": reason}}],
        "recommendation": "",
        "recommendation_source": BASIS_OPERATOR,
        "confidence": None,
        "materiality": "REVIEW",
        "downstream_consequence": (
            "This column feeds the report from now on, and the mapping is "
            "promoted into the client's governed rules when the case is "
            "activated."),
        "options": [],
        "resolution": "approve",
        "resolved_value": target_field,
        "resolved_by": actor,
        "resolved_at": at,
        "reason": reason or "an operator named the field this column feeds",
        "subject": {
            "artefact": "unmapped_column",
            "decision_id": _alias_id(source_file, source_column),
            "decision_type": ALIAS_DECISION_TYPE,
            "source_file": source_file,
            "source_column": source_column,
            "source_columns": [],
            "target_field": target_field,
            "proposed_mapping": (f"{source_column} → {target_field}"
                                 if target_field else ""),
            "basis": BASIS_OPERATOR,
        },
    }


def field_request(*, source_file: str, source_column: str, field_name: str,
                  label: str, description: str, data_type: str,
                  actor: str, at: str, samples: Optional[List[str]] = None
                  ) -> Dict[str, Any]:
    """An ask for a canonical field the platform does not have.

    Everything a configuration owner needs to decide it without coming back to
    ask: the name, what it means, the column and file that prompted it, what
    the values look like, and who wants it.
    """
    return {
        "request_id": _request_id(source_file, source_column),
        "status": REQUEST_OPEN,
        "field_name": field_name,
        "label": label or str(field_name).replace("_", " "),
        "description": description,
        "data_type": data_type,
        "source_file": source_file,
        "source_column": source_column,
        "sample_values": list(samples or [])[:5],
        "requested_by": actor,
        "requested_at": at,
        "route": ("config package, system layer: "
                  "config/system/fields_registry.yaml"),
    }


def validate_field_name(name: str) -> str:
    """The registry's own naming, checked before the ask is recorded.

    A request for ``Curr Bal (GBP)`` is a request a configuration owner cannot
    action, and finding that out a week later helps nobody.
    """
    candidate = str(name or "").strip()
    if not candidate:
        raise FieldRegistryError(
            "OCC_AGENT_FIELD_NAME_REQUIRED",
            "A new field needs a name before it can be requested.",
            http_status=400)
    normalised = candidate.strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in normalised:
        normalised = normalised.replace("__", "_")
    if not all(ch.isalnum() or ch == "_" for ch in normalised):
        raise FieldRegistryError(
            "OCC_AGENT_FIELD_NAME_INVALID",
            "A canonical field name is lower-case words joined by "
            "underscores — letters, numbers and underscores only.",
            http_status=400)
    if not normalised[0].isalpha():
        raise FieldRegistryError(
            "OCC_AGENT_FIELD_NAME_INVALID",
            "A canonical field name starts with a letter.",
            http_status=400)
    return normalised


def _scoped_id(prefix: str, source_file: str, source_column: str) -> str:
    """An id that is unique across the pack AND survives a long filename.

    The obvious spelling — slug the combined ``file::column`` — truncates at 48
    characters, and a real delivery spends most of that on the filename:
    ``ERE_LoanExtract_One_202608.xlsx::Current Interest Rate`` lands at 47, one
    character from eating the column name entirely. Two columns of that file
    sharing a prefix would then collapse onto one id, and the rerun collapses
    decisions by id — so one operator's answer would be dropped on the floor
    with nothing saying so.

    The COLUMN leads, because that is what a person reads in an audit trail,
    and a digest of the exact pair follows, because that is what makes it
    unique whatever the names are.
    """
    import hashlib

    from .execution import _slug
    key = mapping_key(source_file, source_column)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{_slug(source_column)[:40]}_{digest}"


def _alias_id(source_file: str, source_column: str) -> str:
    return _scoped_id("alias", source_file, source_column)


def _request_id(source_file: str, source_column: str) -> str:
    return _scoped_id("fieldreq", source_file, source_column)
