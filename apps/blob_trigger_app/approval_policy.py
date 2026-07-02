"""apps.blob_trigger_app.approval_policy — materiality / evidence classifier.

Implements the one-click-vs-auto-approve decision for a RECURRING source whose
schema fingerprint has changed. Deterministic + LLM evidence in; a
:class:`MaterialityResult` out. Pure logic — no I/O, no Azure — so the router can
call it inline and tests can exercise every branch without a live orchestrator.

APPROVAL POLICY (thresholds are config-driven — see :func:`load_thresholds`):

  NON-MATERIAL (eligible for AUTO-APPROVE) = ALL of:
    * the logical role set is unchanged;
    * no previously-mapped mandatory field becomes unmapped;
    * no NEW mandatory field appears unmapped;
    * deterministic mapping confidence ≥ auto_conf AND value-match rate ≥
      value_match for changed columns, OR LLM mapping confidence ≥ llm_conf
      (canonical-only nulling stays enforced downstream);
    * changes are limited to cosmetic header text, column reorder, or additive
      optional columns.

  MATERIAL (force one-click) = ANY of:
    * a new / removed mandatory field;
    * a logical role change / conflict;
    * mapping confidence below the thresholds;
    * a book-type / target contradiction.

A NON-MATERIAL, structurally-cosmetic recurring change (reorder / additive /
header rename) carries FULL deterministic evidence (all previously-mapped columns
still present) and auto-approves without any LLM call. Genuinely ambiguous cases
fall back to the LLM confidence gate, and failing that, to one-click review.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# --------------------------------------------------------------------------- #
# Config-driven thresholds
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Thresholds:
    #: deterministic mapping confidence floor for the deterministic evidence path.
    auto_conf: float = 0.85
    #: value-match rate floor for CHANGED columns on the deterministic path.
    value_match: float = 0.98
    #: LLM mapping confidence floor for the LLM evidence path.
    llm_conf: float = 0.95


def _f(env: Dict[str, str], name: str, default: float) -> float:
    try:
        v = env.get(name)
        return float(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


def load_thresholds(env: Optional[Dict[str, str]] = None) -> Thresholds:
    """Resolve auto-approve thresholds from the environment (safe defaults).

    ``TRAKT_APPROVAL_AUTO_CONF`` / ``TRAKT_APPROVAL_AUTO_VALUE_MATCH`` /
    ``TRAKT_APPROVAL_AUTO_LLM_CONF``.
    """
    env = env if env is not None else os.environ
    return Thresholds(
        auto_conf=_f(env, "TRAKT_APPROVAL_AUTO_CONF", 0.85),
        value_match=_f(env, "TRAKT_APPROVAL_AUTO_VALUE_MATCH", 0.98),
        llm_conf=_f(env, "TRAKT_APPROVAL_AUTO_LLM_CONF", 0.95),
    )


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #

@dataclass
class MaterialityResult:
    material: bool
    auto_approvable: bool
    reasons: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Role/column diffing (header-normalised, order-aware)
# --------------------------------------------------------------------------- #

def _norm(col: str) -> str:
    """Normalise a header for cosmetic comparison (matches file_roles rule)."""
    return re.sub(r"[^a-z0-9]+", "", str(col).lower())


def _base_role(role: str) -> str:
    """Collapse the ``role#1`` disambiguation suffix to the base logical role."""
    return role.split("#", 1)[0]


def _role_sets(schemas: Dict[str, Sequence[str]]) -> set:
    return {_base_role(r) for r in (schemas or {})}


def diff_role_schemas(old: Dict[str, Sequence[str]],
                      new: Dict[str, Sequence[str]]) -> Dict[str, Any]:
    """Structural diff of old vs new ``{role: [columns]}`` header signatures.

    Returns roles added/removed, and per shared role the columns removed / added /
    reordered / cosmetically renamed (normalised-equal but different exact text).
    """
    old = old or {}
    new = new or {}
    old_roles, new_roles = _role_sets(old), _role_sets(new)
    # Map base role -> columns (first signature wins for a base with #-dupes).
    def _by_base(schemas):
        out: Dict[str, List[str]] = {}
        for r, cols in schemas.items():
            out.setdefault(_base_role(r), list(cols))
        return out
    old_b, new_b = _by_base(old), _by_base(new)

    per_role: Dict[str, Any] = {}
    for role in sorted(old_roles & new_roles):
        oc, nc = list(old_b[role]), list(new_b[role])
        on, nn = [_norm(c) for c in oc], [_norm(c) for c in nc]
        removed = [c for c in oc if _norm(c) not in set(nn)]
        added = [c for c in nc if _norm(c) not in set(on)]
        reordered = (set(on) == set(nn)) and (on != nn)
        # cosmetic header rename: same normalised set but different exact text.
        cosmetic = (set(on) == set(nn)) and (set(oc) != set(nc))
        per_role[role] = {
            "removed_columns": removed, "added_columns": added,
            "reordered": reordered, "cosmetic_header_change": cosmetic}
    return {
        "roles_added": sorted(new_roles - old_roles),
        "roles_removed": sorted(old_roles - new_roles),
        "role_set_changed": old_roles != new_roles,
        "per_role": per_role,
    }


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #

def classify(
    *,
    old_role_schemas: Dict[str, Sequence[str]],
    new_role_schemas: Dict[str, Sequence[str]],
    old_fingerprint: Optional[str] = None,
    new_fingerprint: Optional[str] = None,
    mandatory_columns: Optional[Sequence[str]] = None,
    det_conf: Optional[float] = None,
    value_match_rate: Optional[float] = None,
    llm_conf: Optional[float] = None,
    book_type_conflict: bool = False,
    thresholds: Optional[Thresholds] = None,
) -> MaterialityResult:
    """Classify a recurring source's fingerprint change as MATERIAL vs NON-MATERIAL.

    ``old_role_schemas`` are the pinned registry signatures; ``new_role_schemas``
    are this pack's ``{role: columns}``. ``mandatory_columns`` (optional) is the
    normalised set of mandatory source columns for the source's contract — when
    supplied, a removed column is only material if it is mandatory; without it,
    ANY removed previously-mapped column is treated as material (conservative).

    ``det_conf`` / ``value_match_rate`` / ``llm_conf`` are OPTIONAL evidence. When
    the structural change is provably cosmetic (reorder / additive / header
    rename with no removals and no role change), the deterministic evidence is
    FULL (all previously-mapped columns still present) and defaults to 1.0/1.0, so
    the common recurring case auto-approves with no LLM call.
    """
    t = thresholds or load_thresholds()
    # No pinned baseline to compare against → cannot prove "significantly the same".
    # Fail closed to MATERIAL (one-click) rather than auto-approving on no evidence.
    if not old_role_schemas or not new_role_schemas:
        return MaterialityResult(
            material=True, auto_approvable=False,
            reasons=["no_pinned_header_signatures_to_compare"],
            evidence={"old_fingerprint": old_fingerprint,
                      "new_fingerprint": new_fingerprint,
                      "old_role_schemas_present": bool(old_role_schemas),
                      "new_role_schemas_present": bool(new_role_schemas)})
    diff = diff_role_schemas(old_role_schemas, new_role_schemas)
    reasons: List[str] = []
    mand = {_norm(c) for c in (mandatory_columns or [])}

    structurally_material = False
    if book_type_conflict:
        reasons.append("book_type_or_target_contradiction")
        structurally_material = True
    if diff["role_set_changed"]:
        reasons.append(
            f"logical_role_set_changed added={diff['roles_added']} removed={diff['roles_removed']}")
        structurally_material = True

    removed_any: List[str] = []
    removed_mandatory: List[str] = []
    added_any: List[str] = []
    for role, d in diff["per_role"].items():
        for c in d["removed_columns"]:
            removed_any.append(f"{role}:{c}")
            if not mand or _norm(c) in mand:
                removed_mandatory.append(f"{role}:{c}")
        added_any.extend(f"{role}:{c}" for c in d["added_columns"])
    if removed_mandatory:
        # A previously-mapped (mandatory, or unknown-so-assume-mandatory) field
        # became unmapped → material.
        reasons.append(f"mandatory_field_removed {removed_mandatory}")
        structurally_material = True

    # Evidence path. For a provably-cosmetic change the deterministic evidence is
    # complete; otherwise use whatever the caller measured.
    cosmetic_only = (not structurally_material and not removed_any)
    if det_conf is None and cosmetic_only:
        det_conf = 1.0
    if value_match_rate is None and cosmetic_only:
        value_match_rate = 1.0

    det_ok = (det_conf is not None and value_match_rate is not None
              and det_conf >= t.auto_conf and value_match_rate >= t.value_match)
    llm_ok = (llm_conf is not None and llm_conf >= t.llm_conf)
    evidence_ok = det_ok or llm_ok

    material = structurally_material or not evidence_ok
    if not structurally_material and not evidence_ok:
        reasons.append(
            f"mapping_confidence_below_threshold det_conf={det_conf} "
            f"value_match={value_match_rate} llm_conf={llm_conf} "
            f"(need det≥{t.auto_conf}&vm≥{t.value_match} OR llm≥{t.llm_conf})")

    evidence = {
        "role_set_diff": diff,
        "removed_columns": removed_any,
        "removed_mandatory_columns": removed_mandatory,
        "added_columns": added_any,
        "det_conf": det_conf,
        "value_match_rate": value_match_rate,
        "llm_conf": llm_conf,
        "det_evidence_ok": det_ok,
        "llm_evidence_ok": llm_ok,
        "cosmetic_only": cosmetic_only,
        "old_fingerprint": old_fingerprint,
        "new_fingerprint": new_fingerprint,
        "thresholds": asdict(t),
    }
    if not reasons:
        reasons.append("non_material: role set unchanged, no mandatory field lost, "
                       "changes limited to reorder/additive/cosmetic, evidence ≥ thresholds")
    return MaterialityResult(material=material, auto_approvable=not material,
                             reasons=reasons, evidence=evidence)
