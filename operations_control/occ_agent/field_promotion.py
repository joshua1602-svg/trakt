"""operations_control.occ_agent.field_promotion — an operator's ask for a
canonical field, carried into the governed route that can grant it.

THE ASK WAS A NOTE FOR SOMEBODY. An operator who met a column the platform has
no field for could record what it is, and the record went into the readiness
package and stopped there. Somebody had to read it, understand it, and hand-edit
a versioned configuration file — which is how a real need becomes a thing that
never happens.

WHAT THIS DOES INSTEAD. At activation, and only at activation, every standing
request becomes one **draft** system configuration version carrying the new
fields. A configuration owner opens it in Platform configuration, reads the
diff against what is in force, validates it and activates it — the route that
already exists for changing ``config/system/fields_registry.yaml``, entered at
the right place with the work already done.

WHY A DRAFT AND NEVER AN ACTIVATION. The field registry is the vocabulary every
client's report is written in: every regime projection, every validation rule
and every other client's mapping reads it. ``operations_control.rules`` states
the invariant — "the core field registry is never written" from this container
— and this module does not break it. It writes a PROPOSAL into the package
store, which is a different thing: nothing in force changes, no delivery sees
the new field, and the second pair of eyes that the config lifecycle exists to
require is still required. One client's first delivery must not be able to
change what every other client's report means.

WHY AT ACTIVATION AND NOT WHEN THE OPERATOR ASKS. The same reason mappings
promote there and not sooner: a rehearsal that is never activated must leave no
trace outside its own sandbox. A draft config version is a global artefact. An
operator exploring a practice case would otherwise litter the platform's
configuration history with proposals for clients that never went live.
"""

from __future__ import annotations

from typing import Any, Dict, List

import yaml

#: The layer the field registry belongs to, and the file within it.
LAYER = "system"
REGISTRY_FILE = "config/system/fields_registry.yaml"

#: A requested field is never core-canonical and never sits in a regime
#: mapping. Both are claims about the platform's obligations that an onboarding
#: operator is not in a position to make, and a reviewer can add either.
_DEFAULTS = {"allowed_values": None, "category": "analytics",
             "layer": "core", "core_canonical": False}

#: What the operator said the values look like -> the registry's own word for
#: it. An unrecognised or absent answer leaves the format for the reviewer to
#: fill in rather than guessing at one.
_FORMATS = {"string": "string", "decimal": "decimal", "date": "date",
            "list": "list", "Y/N": "Y/N"}


def open_requests(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The asks still standing. A withdrawn one is kept on the run as part of
    the record and must not reach a configuration owner."""
    return [r for r in (requests or [])
            if str(r.get("status") or "") == "requested"
            and str(r.get("field_name") or "")]


def field_entry(request: Dict[str, Any], *, asset_type: str
                ) -> Dict[str, Any]:
    """One registry entry, as narrow as the ask justifies.

    ``portfolio_type`` is THIS book's asset class rather than ``common``: the
    operator met the column in an equity release tape and said what it means
    there. Whether every asset class means the same by it is a second claim
    they did not make, and widening it later is a governed act with its own
    record — the safe direction to travel in.
    """
    entry = dict(_DEFAULTS)
    entry["portfolio_type"] = asset_type or "common"
    fmt = _FORMATS.get(str(request.get("data_type") or ""), "")
    if fmt:
        entry["format"] = fmt
    return entry


def draft_registry(current: str, requests: List[Dict[str, Any]], *,
                   asset_type: str) -> str:
    """The registry file as it would read with the requested fields in it.

    Built from the version in FORCE rather than from the repository file, so a
    draft never quietly reverts a change somebody else activated in between.
    A name already present is left exactly as it is: the ask is satisfied and
    overwriting a field in use would be the worst possible outcome of asking
    for a new one.
    """
    doc = yaml.safe_load(current) or {}
    fields = doc.setdefault("fields", {})
    for request in open_requests(requests):
        name = str(request["field_name"])
        if name in fields:
            continue
        fields[name] = field_entry(request, asset_type=asset_type)
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


def notes(requests: List[Dict[str, Any]], *, case_ref: str) -> str:
    """What a configuration owner needs to decide it, in their words.

    The column, the file, what the operator says it means and who asked — so
    the reviewer can judge the proposal without coming back to the case.
    """
    lines = [f"Fields requested during onboarding {case_ref}."]
    for request in open_requests(requests):
        detail = str(request.get("description") or "").strip()
        samples = ", ".join(str(v) for v in
                            (request.get("sample_values") or [])[:3])
        lines.append(
            f"- {request['field_name']}: from '{request.get('source_column')}' "
            f"in {request.get('source_file')}, asked for by "
            f"{request.get('requested_by')}."
            + (f" {detail}" if detail else "")
            + (f" Values look like: {samples}." if samples else ""))
    return "\n".join(lines)


def propose(packages: Any, requests: List[Dict[str, Any]], *, by: str,
            case_ref: str, asset_type: str) -> List[Dict[str, Any]]:
    """One draft system config version carrying every standing request.

    Never raises: a draft that could not be written is reported on the
    activation result, because an activation that succeeded in every other
    respect must not be reported as failed over a proposal — and a proposal
    silently lost is worse than one that says it was lost.
    """
    standing = open_requests(requests)
    if not standing:
        return []
    try:
        active = packages.ensure_seeded(LAYER, by=by)
        current = (active.get("files") or {}).get(REGISTRY_FILE) or {}
        draft = packages.create_draft(
            LAYER, by=by,
            edits={REGISTRY_FILE: draft_registry(
                str(current.get("content") or ""), standing,
                asset_type=asset_type)},
            notes=notes(standing, case_ref=case_ref))
    except Exception as exc:                # noqa: BLE001 — reported, not lost
        return [{"field_name": str(r.get("field_name") or ""),
                 "status": "not_proposed",
                 "error": f"{type(exc).__name__}: {exc}"} for r in standing]
    return [{"field_name": str(r.get("field_name") or ""),
             "status": "proposed",
             "layer": LAYER,
             "version": draft.get("version"),
             "source_file": str(r.get("source_file") or ""),
             "source_column": str(r.get("source_column") or "")}
            for r in standing]
