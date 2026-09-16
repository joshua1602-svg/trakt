"""operations_control.occ_agent.mapping_promotion — a rehearsal's mapping
decisions, carried into the governed rules that production reads.

THE GAP THIS CLOSES. A rehearsal reads the client's real columns, raises the
ones it cannot settle, and an operator answers them. Until now
``resolve_decision`` wrote that answer onto the RUN and nowhere else. Nothing
carried it across the doorway: ``activate()`` builds the configuration from the
CASE and hands over the file bytes, so the production ingest re-derived every
mapping from scratch and re-raised the same questions. The most valuable thing
a rehearsal produces — a human's reading of this lender's column names — was
thrown away at the moment it became usable.

WHAT IS PROMOTED, AND WHAT IS NOT

Only a mapping decision a human actually settled: ``status == "approved"`` with
a resolution of ``approve`` or ``amend``. A rejected decision promotes nothing,
because "no" is not a mapping. An unresolved one cannot be here at all — a run
with blocking decisions never reaches activation.

Only in LIVE mode. A rehearsal that is never activated must leave no trace in
the governed store, which is the property the whole synthetic boundary exists
to hold. Promotion therefore happens at activation, beside the case promotion,
and never at the moment the operator answers.

SCOPE IS DELIBERATELY NARROW. A promoted rule is scoped to the PORTFOLIO, not
the client. The operator answered a question about the columns in this book's
tape; whether the client's other books use the same names is a second claim
they did not make. Widening a rule later is a governed act with its own record,
and it is the safe direction to travel in.

THE TWO DIRECTIONS A MAPPING QUESTION IS ASKED

They are not symmetrical, and reading them as one is how a promoted rule would
come out backwards:

* ``mapping_confirmation`` — "'Curr Bal' looks like current balance; is it?"
  The SOURCE COLUMN is fixed and the answer names the canonical field.
* ``mapping_ambiguity`` — "two columns could be current balance; which?"
  The CANONICAL FIELD is fixed and the answer names the source column.

``Engine._persist_rule`` makes the same distinction for decisions answered on a
live workflow. This module matches it rather than inventing a second reading,
so a rule promoted from a rehearsal is indistinguishable in shape from one an
operator approved in production.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..contracts import KIND_FIELD_MAPPING, now_iso
from ..rules import RuleRecord

#: The decision types this module knows how to read. A decision type it has
#: never seen is left alone rather than guessed at — an unpromoted mapping
#: costs one repeated question, and a wrong one corrupts a delivery.
CONFIRMATION = "mapping_confirmation"
AMBIGUITY = "mapping_ambiguity"
PROMOTABLE_TYPES = (CONFIRMATION, AMBIGUITY)

#: Resolutions that assert a mapping. "reject" asserts the absence of one and
#: promotes nothing.
PROMOTABLE_RESOLUTIONS = ("approve", "amend")

#: Where a promoted rule applies. See the module note: the operator answered
#: about THIS book's tape.
PROMOTED_SCOPE = "portfolio"

#: Recorded on every promoted rule, so the governed record can always say a
#: rule came from a rehearsal rather than from a live delivery.
SUGGESTED_BY = "occ_agent_rehearsal"


def mapping_of(decision: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """``{source_column, canonical_field}`` a settled decision asserts.

    ``None`` when the decision asserts no mapping — it was rejected, is not a
    mapping decision, was never resolved, or came out incomplete.
    """
    if str(decision.get("decision_type") or "") not in PROMOTABLE_TYPES:
        return None
    if str(decision.get("status") or "") != "approved":
        return None
    resolution = str(decision.get("resolution") or "")
    if resolution not in PROMOTABLE_RESOLUTIONS:
        return None

    source = str(decision.get("source_column") or "").strip()
    canonical = str(decision.get("target_field") or "").strip()
    answer = str(decision.get("resolved_value") or "").strip()

    if resolution == "amend" and answer:
        # Which side the operator's answer belongs on depends on which side the
        # question fixed. See the module note.
        if str(decision.get("decision_type")) == AMBIGUITY:
            source = answer
        else:
            canonical = answer

    if not source or not canonical:
        return None
    return {"source_column": source, "canonical_field": canonical}


def rules_from(decisions: List[Dict[str, Any]], *, client_id: str,
               portfolio_id: str, workflow_id: str) -> List[RuleRecord]:
    """The governed rules a run's settled mapping decisions amount to.

    Builds records; persists nothing. Assembling and writing are separate so a
    caller can show an operator what activation would add to the client's
    standing rules without adding it.
    """
    out: List[RuleRecord] = []
    seen: set = set()
    for decision in decisions or []:
        mapping = mapping_of(decision)
        if mapping is None:
            continue
        # One rule per source column. A run that asked about the same column
        # twice states one fact about it, and the last answer is the one that
        # stood when the run passed.
        key = mapping["source_column"].strip().lower()
        if key in seen:
            out = [r for r in out
                   if str(r.payload.get("source_column", "")).strip().lower()
                   != key]
        seen.add(key)
        canonical = mapping["canonical_field"]
        out.append(RuleRecord(
            rule_id="", version=0, kind=KIND_FIELD_MAPPING,
            scope=PROMOTED_SCOPE, client_id=client_id,
            portfolio_id=portfolio_id,
            payload=dict(mapping),
            description=(f"Treat '{mapping['source_column']}' as "
                         f"'{canonical.replace('_', ' ')}'."),
            suggested_by=SUGGESTED_BY,
            confidence=_confidence(decision),
            decision_id=str(decision.get("decision_id") or ""),
            workflow_id=workflow_id,
            approved_by=str(decision.get("resolved_by") or ""),
            # WHEN THE HUMAN ANSWERED, not when activation got round to
            # writing it. The two can be days apart, and the approval date is
            # what an auditor asks about.
            approved_at=str(decision.get("resolved_at") or "") or now_iso(),
            reason=str(decision.get("reason") or "")
            or "settled during the onboarding rehearsal"))
    return out


def _confidence(decision: Dict[str, Any]) -> Optional[float]:
    """The mapper's own confidence, kept as evidence.

    A human approved the rule, so the rule is not held at the mapper's
    confidence — but WHY the question was asked is part of the record, and a
    column confirmed at 0.41 reads differently in six months' time from one
    confirmed at 0.95.
    """
    try:
        return round(float(decision.get("confidence")), 4)
    except (TypeError, ValueError):
        return None


def promote(rules_store: Any, decisions: List[Dict[str, Any]], *,
            client_id: str, portfolio_id: str,
            workflow_id: str) -> List[Dict[str, str]]:
    """Persist the settled mappings as governed rules. Returns what was added.

    Each goes through ``RuleStore.approve``, so an existing active rule for the
    same column becomes version n+1 and supersedes its predecessor rather than
    being overwritten — the same path a live approval takes. Nothing is
    deleted, and the prior version stays readable in the rule's history.
    """
    added: List[Dict[str, str]] = []
    for rule in rules_from(decisions, client_id=client_id,
                           portfolio_id=portfolio_id,
                           workflow_id=workflow_id):
        stored = rules_store.approve(rule)
        added.append({"rule_id": stored.rule_id,
                      "version": str(stored.version),
                      "source_column": str(rule.payload.get("source_column")),
                      "canonical_field": str(rule.payload.get("canonical_field"))})
    return added
