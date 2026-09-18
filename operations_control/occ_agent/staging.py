"""operations_control.occ_agent.staging — the operator's working copy of the
mapping table, and the one act that commits it.

WHAT WAS WRONG WITH APPROVING AS YOU GO

Every answer applied itself. Confirming one column resolved its decision,
promoted nothing yet but wrote to the run, and — the moment the last blocking
decision cleared — reran the whole onboarding. On a hundred-and-fifty-column
tape that makes reading the table an irreversible walk: an operator who
confirms forty columns and then realises the fortieth was wrong has already
committed thirty-nine, and the run may already have moved on.

    "Workflow should be operator confirms each field and then there is a final
    'confirm' all changes button which is what persists. Before that point the
    user can change their confirmed fields."

So this module holds the two halves apart.

  * STAGING is reading. An operator says what each column is — confirm what
    Trakt proposed, change it to another field, or set it aside — and nothing
    happens. No decision is resolved, no rule is promoted, no control is rerun.
    Each entry can be replaced or withdrawn, as many times as they like.

  * COMMITTING is the governed act. One call applies every staged answer AND
    the proposals nobody touched, resolves each decision in its own right with
    its own approver and timestamp, and lets the run go on.

THE DRAFT IS PERSISTED, WHICH IS NOT THE SAME AS APPLIED. It lives on the run
rather than in the browser because the reading IS the work, and a hard refresh
or a closed tab must not cost an afternoon of it. Nothing downstream reads it:
``_approved_mappings`` reads resolved decisions, promotion reads resolved
decisions, and a case abandoned mid-draft leaves no mapping behind.

WHY AN UNTOUCHED PROPOSAL COMMITS AS PROPOSED. A proposal already carries a
field — Trakt's reading of the column — and the operator's act is to accept or
correct it. Requiring a click on all hundred and fifty to say "yes, as shown"
would be the friction the one-act approval removed, and would make the button
unreachable rather than deliberate. A QUESTION is different: a weak match or an
ambiguity has no answer yet, so the commit refuses while one is unanswered
rather than inventing one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#: What an operator can say about one column.
ACTION_CONFIRM = "confirm"        # the field Trakt read it as is right
ACTION_AMEND = "amend"            # it is a different field, named here
ACTION_NOT_USED = "not_used"      # this column feeds nothing
ACTION_CLEAR = "clear"            # take the staged answer back
ACTIONS = (ACTION_CONFIRM, ACTION_AMEND, ACTION_NOT_USED, ACTION_CLEAR)

#: How each staged answer resolves the decision it answers, in the vocabulary
#: ``_approved_mappings`` and ``mapping_promotion`` already read. Changed to a
#: different field is an AMENDMENT, and promotion reads that word to know the
#: answer names the field rather than the column.
RESOLUTION = {ACTION_CONFIRM: "approve",
              ACTION_AMEND: "amend",
              ACTION_NOT_USED: "approve"}

#: The resolved value that means "do not use this column". The same string
#: ``OccAgentService._approved_mappings`` already treats as an empty target, so
#: a set-aside column reaches the adapter as an answered column with no field
#: rather than as a question nobody answered.
NOT_USED_VALUE = "mark_unavailable"


def key(source_file: str, source_column: str) -> Tuple[str, str]:
    return (str(source_file or ""), str(source_column or ""))


def find(staged: List[Dict[str, Any]], source_file: str,
         source_column: str) -> Optional[Dict[str, Any]]:
    """The staged answer for one column of one file, if there is one."""
    return by_column(staged).get(key(source_file, source_column))


def by_column(staged: List[Dict[str, Any]]
              ) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {key(e.get("source_file"), e.get("source_column")): e
            for e in staged or []}


def entry(*, source_file: str, source_column: str, action: str,
          target_field: str, decision_id: str, actor: str, at: str,
          reason: str = "") -> Dict[str, Any]:
    """One staged answer, carrying who said it and when.

    The approver is recorded HERE rather than at the commit, because it is the
    reading that is the judgement: a second operator pressing the button has
    not read the column, and a record that credited them with it would be
    wrong. The commit carries its own actor beside these.
    """
    return {"source_file": source_file, "source_column": source_column,
            "action": action, "target_field": target_field,
            "decision_id": decision_id, "staged_by": actor, "staged_at": at,
            "reason": reason}


def resolved_value(staged: Dict[str, Any]) -> str:
    """What this staged answer writes as the decision's resolution value."""
    if staged.get("action") == ACTION_NOT_USED:
        return NOT_USED_VALUE
    return str(staged.get("target_field") or "")


def summary(staged: List[Dict[str, Any]]) -> Dict[str, int]:
    """How many of each kind are waiting, for the button to name itself."""
    out = {ACTION_CONFIRM: 0, ACTION_AMEND: 0, ACTION_NOT_USED: 0}
    for item in staged or []:
        action = str(item.get("action") or "")
        if action in out:
            out[action] += 1
    return out
