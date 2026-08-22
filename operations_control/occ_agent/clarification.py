"""operations_control.occ_agent.clarification — phase two, once a file exists.

The onboarding pack asks a client only what genuinely blocks the first
ingestion. Everything about what their numbers MEAN is deferred by the
catalogue (``deferred_until``), because asking it cold makes a client write a
data dictionary for a file Trakt has not looked at — and Trakt can read most of
the answer for itself.

This module is the other half of that promise. Deferring a question is only
honest if the question is actually asked later, so once a representative file
has arrived this reports what is still outstanding, **with the evidence to put
it against**:

    not  "What does current balance mean?"
    but  "We found Current Balance and Accrued Interest as separate columns."

It builds information-request ITEMS. It does not send anything, create a
request, or move a case: the caller hands them to the existing governed path
(:meth:`OccAgentService.request_client_information` →
``OnboardingService.create_request``), which is the same route an operator's
own request takes and carries the same approval, audit and status transitions.
Nothing here is a second channel to the client.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..onboarding.case import OnboardingCase
from ..onboarding.catalogue import Catalogue, catalogue as _catalogue
from . import classification as _classification

#: The categories a deferred question lands in. ``first_delivery`` is the one
#: this module exists for; a question that is ``not_applicable`` to this
#: client's products is not asked later either, because it was never relevant.
DEFERRED = _classification.FIRST_DELIVERY


def _observed_columns(artefacts: Sequence[Dict[str, Any]]) -> List[str]:
    """Every column name seen across the delivered files, in first-seen order.

    This is the evidence. It comes from the artefact registration that already
    profiled each file, so nothing is re-read and no new analysis is invented
    to support a question.
    """
    seen: List[str] = []
    for artefact in artefacts or []:
        for column in (artefact.get("columns") or []):
            name = str(column).strip()
            if name and name not in seen:
                seen.append(name)
    return seen


def outstanding(case: OnboardingCase,
                artefacts: Sequence[Dict[str, Any]], *,
                cat: Optional[Catalogue] = None) -> List[Dict[str, Any]]:
    """The deferred questions a delivered file has now made askable.

    Empty until at least one file has arrived — which is the whole point. A
    clarification with no file behind it is the generic question the pack
    deliberately does not ask, and issuing one here would put it back.
    """
    if not artefacts:
        return []
    cat = cat or _catalogue()
    evidence = _observed_columns(artefacts)
    files = [str(a.get("source_file") or "") for a in artefacts
             if a.get("source_file")]

    items: List[Dict[str, Any]] = []
    for row in _classification.classify_all(case.answers, cat=cat):
        if row.category != DEFERRED:
            continue
        declared = cat.field(row.section, row.field)
        # Only what a client can actually answer. A deferred field Trakt reads
        # for itself (an inferred asset class, a schema fingerprint) is not a
        # question — it is work the pipeline has already done.
        if declared is None or declared.source != "client_supplied":
            continue
        items.append({
            "section": row.section,
            "field": row.field,
            "index": row.index,
            "label": row.label,
            "help": declared.help,
            # What makes the question specific rather than generic. The caller
            # renders it; this states it.
            "evidence": {"files": files, "columns": evidence},
            "deferred_reason": row.reason,
        })
    return items


def summary(items: Sequence[Dict[str, Any]],
            artefacts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """A short account of what was analysed and what is left.

    Reported so a clarification can open with what Trakt worked out rather than
    with what it wants, which is the difference between "we have mapped 47 of
    49 fields, please confirm these two" and "please provide a field
    dictionary".
    """
    columns = _observed_columns(artefacts)
    return {"files_analysed": len([a for a in artefacts if a]),
            "columns_observed": len(columns),
            "questions_remaining": len(items)}
