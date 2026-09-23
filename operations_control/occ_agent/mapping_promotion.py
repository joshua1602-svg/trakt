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
from .staging import NOT_USED_VALUE as NOT_USED

#: The decision types this module knows how to read. A decision type it has
#: never seen is left alone rather than guessed at — an unpromoted mapping
#: costs one repeated question, and a wrong one corrupts a delivery.
CONFIRMATION = "mapping_confirmation"
AMBIGUITY = "mapping_ambiguity"
#: A confident match on a FIRST onboarding, which is proposed rather than
#: applied because nobody has yet said this client means what the platform's
#: alias registry says they mean. It reads exactly like a confirmation — the
#: source column is fixed and the answer names the field — and it is the type
#: that carries the bulk of a first delivery. Leaving it unpromotable would
#: make the one approval an operator gives produce no governed rule at all,
#: and the second month would ask the whole tape again.
PROPOSAL = "mapping_proposal"
PROMOTABLE_TYPES = (CONFIRMATION, AMBIGUITY, PROPOSAL)

#: Resolutions that assert a mapping. "reject" asserts the absence of one and
#: promotes nothing.
PROMOTABLE_RESOLUTIONS = ("approve", "amend")

#: Where a promoted rule applies. See the module note: the operator answered
#: about THIS book's tape.
PROMOTED_SCOPE = "portfolio"

#: Recorded on every promoted rule, so the governed record can always say a
#: rule came from a rehearsal rather than from a live delivery.
SUGGESTED_BY = "occ_agent_rehearsal"

#: Recorded as the author of a withdrawal, so the rule's history says a
#: settled operator decision took it out of force rather than a person.
WITHDRAWN_BY = "occ_agent_withdrawal"


def _key(source_column: str) -> str:
    """One column, however the pack and the decision each spelled it."""
    return " ".join(str(source_column or "").split()).strip().lower()


def _read(decision: Dict[str, Any], key: str) -> str:
    """One field of a decision, from wherever that decision keeps it.

    A mapping decision exists in two shapes and this module is handed both. The
    adapter writes the RAW artefact row, with everything at the top level. The
    run holds the operator-facing CARD, which
    ``OccAgentService._decision_card`` assembles — and which keeps the
    source column and the target field under ``subject``.

    Reading only the top level is how promotion came to do nothing at all:
    activation passes ``run.open_decisions``, every card returned "" for
    ``decision_type``, every decision failed the first test, and the rehearsal's
    settled mappings were dropped on the floor at the one moment they became
    usable. The unit tests did not catch it because they built the raw shape,
    which is the shape this function was written against and not the shape it
    is called with.
    """
    value = decision.get(key)
    if value in (None, ""):
        value = (decision.get("subject") or {}).get(key)
    return str(value or "").strip()


def mapping_of(decision: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """``{source_column, canonical_field}`` a settled decision asserts.

    ``None`` when the decision asserts no mapping — it was rejected, is not a
    mapping decision, was never resolved, or came out incomplete.
    """
    if _read(decision, "decision_type") not in PROMOTABLE_TYPES:
        return None
    if str(decision.get("status") or "") != "approved":
        return None
    resolution = str(decision.get("resolution") or "")
    if resolution not in PROMOTABLE_RESOLUTIONS:
        return None

    source = _read(decision, "source_column")
    canonical = _read(decision, "target_field")
    answer = str(decision.get("resolved_value") or "").strip()

    if resolution == "amend" and answer:
        # Which side the operator's answer belongs on depends on which side the
        # question fixed. See the module note.
        if _read(decision, "decision_type") == AMBIGUITY:
            source = answer
        else:
            canonical = answer

    if not source or not canonical or canonical == NOT_USED:
        return None
    return {"source_column": source, "canonical_field": canonical}


def withdrawal_of(decision: Dict[str, Any]) -> str:
    """The source column a settled decision says feeds NOTHING, if it says so.

    ``mapping_of`` returns ``None`` for four different situations — not a
    mapping decision, never resolved, rejected, or came out incomplete — and
    the caller treated all four the same way: write no rule. For three of them
    that is right. For the fourth it is the bug.

    An operator setting a column aside is a POSITIVE STATEMENT that it feeds
    nothing, and it usually lands on a column that was mapped before. Writing
    no rule leaves the OLD rule current, because `RuleStore.approve` is what
    supersedes and nothing called it. So the Operations Control Centre showed
    the column with no target while the governed store still mapped it, the
    engine read the store, and the operator's removal did nothing at all —
    `Latest Property Value` was still feeding `current_valuation_amount` on
    the live delivery, at v1, days after it was taken away.

    Returns "" unless the decision is settled AND asserts no mapping, so a
    question nobody answered never withdraws anything.
    """
    if _read(decision, "decision_type") not in PROMOTABLE_TYPES:
        return ""
    if str(decision.get("status") or "") != "approved":
        return ""
    if str(decision.get("resolution") or "") not in PROMOTABLE_RESOLUTIONS:
        return ""
    source = _read(decision, "source_column")
    if not source:
        return ""
    return "" if mapping_of(decision) else source


def rules_from(decisions: List[Dict[str, Any]], *, client_id: str,
               portfolio_id: str, workflow_id: str) -> List[RuleRecord]:
    """The governed rules a run's settled mapping decisions amount to.

    Builds records; persists nothing. Assembling and writing are separate so a
    caller can show an operator what activation would add to the client's
    standing rules without adding it.
    """
    out: List[RuleRecord] = []
    seen: set = set()
    # A decision about SEVERAL columns is read first, so a per-column answer
    # written later supersedes it — the same ordering ``_approved_mappings``
    # uses, and for the same reason: an ambiguity says which column won, and
    # that column's own record says what it feeds.
    for decision in sorted(
            decisions or [],
            key=lambda d: len((d.get("subject") or {}).get("source_columns")
                              or d.get("source_columns") or []) < 2):
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

    # AND WHAT THE OPERATOR TOOK AWAY. Settling a mapping wrote a rule; setting
    # the same column aside wrote nothing, so the rule it had already written
    # stayed in force and the removal was cosmetic. Both directions now reach
    # the store, by the same governed path: `retire` marks the rule withdrawn
    # and keeps its history, exactly as `approve` supersedes rather than
    # overwrites. A column that was never mapped has nothing to withdraw.
    withdrawn = {w for w in (withdrawal_of(d) for d in decisions or []) if w}
    if not withdrawn:
        return added
    keyed = {_key(w) for w in withdrawn}
    for rule in rules_store.list_current(client_id):
        if rule.kind != KIND_FIELD_MAPPING:
            continue
        column = str((rule.payload or {}).get("source_column") or "")
        if _key(column) not in keyed:
            continue
        retired = rules_store.retire(
            client_id, rule.rule_id, by=WITHDRAWN_BY,
            reason=f"'{column}' was set aside by an operator; it feeds nothing.")
        if retired is not None:
            added.append({"rule_id": rule.rule_id,
                          "version": str(retired.version),
                          "source_column": column,
                          "canonical_field": "",
                          "withdrawn": "true"})
    return added
