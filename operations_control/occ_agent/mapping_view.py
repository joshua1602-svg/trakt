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

WHAT THIS IS, AND WHAT IT IS NOT

It is now the surface a mapping is ANSWERED on: every question about a source
column — a proposal, a weak match, an ambiguity — is settled from its row, as a
draft the operator commits in one act (:mod:`.staging`). That replaced a
decision card per column, which applied its answer the moment it was clicked
and so could not be taken back.

It is still not a second mapping path. Nothing here decides anything: it reads
what the run recorded and what the operator staged, and says both out loud. The
applying is ``OccAgentService.confirm_mappings``, the promoting is
:mod:`.mapping_promotion`, and the classification below is the ENGINE's own
test rather than a second opinion about it.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from .execution import LOW_CONFIDENCE, _TRUSTED_TIERS

#: What became of one source column. The order is the order of a reader's
#: interest, worst first — a column nobody has looked at matters more than one
#: that matched its own name exactly.
ROW_NEEDS_YOU = "needs_you"
#: A confident match on a FIRST onboarding, waiting on the approval that makes
#: it this client's mapping. Ranked below "needs you" — a proposal is read and
#: approved in bulk, a question has to be answered one at a time — and above
#: everything settled, because until it is approved the run does not move.
ROW_PROPOSED = "proposed"
ROW_UNREADABLE = "unreadable"
ROW_UNCHECKED = "unchecked"
ROW_UNUSED = "unused"
#: The operator has been through this column and said what it is — and has not
#: committed the set yet. A DRAFT: it is reversible, nothing downstream reads
#: it, and it becomes ``ROW_CONFIRMED`` when the set is confirmed. Sorted below
#: everything still wanting attention, because it is work already done.
ROW_STAGED = "staged"
ROW_CONFIRMED = "confirmed"
ROW_AUTOMATIC = "automatic"

STATE_ORDER = (ROW_NEEDS_YOU, ROW_PROPOSED, ROW_UNREADABLE, ROW_UNCHECKED,
               ROW_UNUSED, ROW_STAGED, ROW_CONFIRMED, ROW_AUTOMATIC)

STATE_LABELS = {
    ROW_NEEDS_YOU: "Needs you",
    ROW_PROPOSED: "Proposed",
    ROW_UNREADABLE: "Could not be read",
    #: A weak match in a file the canonical tape is NOT built from, left
    #: unasked. No run raises one any more — every file's columns are put to a
    #: person now — but a case that was rehearsed before that change still
    #: holds rows in this state, and re-labelling them "Needs you" would point
    #: at a question that was never raised.
    ROW_UNCHECKED: "Weak match, nothing asked",
    ROW_UNUSED: "Not used",
    ROW_STAGED: "Ready to confirm",
    ROW_CONFIRMED: "You confirmed it",
    ROW_AUTOMATIC: "Matched automatically",
}

#: How each tier reads to somebody who did not write the mapper. The tier names
#: are the mapper's own vocabulary (``semantic_alignment.HeaderMapper.map_one``)
#: and mean nothing to an operator; what they need is how firm the evidence is.
#: On what evidence a column reads as it does — the "mapping basis" an
#: operator needs to weigh a row. A proposal from a model and a contract-backed
#: alias are both "a suggested mapping" on the screen and are not remotely the
#: same claim, so the table has to distinguish them.
BASIS_OPERATOR = "you"
BASIS_DETERMINISTIC = "deterministic"
BASIS_MODEL = "model"

BASIS_LABELS = {
    BASIS_OPERATOR: "You confirmed it",
    BASIS_DETERMINISTIC: "Trakt's own matching",
    BASIS_MODEL: "Suggested by a model, not yet confirmed",
}

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

#: WHAT KIND OF READING THIS IS, IN TWO WORDS.
#:
#: The tier sentence explains the evidence and is the right length to READ; it
#: is the wrong length to SCAN a hundred and fifty rows by. An operator working
#: down the table is asking one question first — is this an alias, a model's
#: guess, or something else? — and the answer has to be the same width and in
#: the same place on every row.
#:
#: It is also the one place a model's suggestion is described honestly. Such a
#: row's TIER is ``unmapped``, because the deterministic mapper is what failed;
#: rendering the tier alone put "Nothing Trakt reports on resembles this
#: column" beside a violet "From a model" chip, which reads as the screen
#: contradicting itself.
KIND_OPERATOR = "operator"
KIND_ALIAS = "alias"
KIND_NAME = "name"
KIND_SIMILAR = "similar"
KIND_MODEL = "model"
KIND_NONE = "none"
KIND_UNREADABLE = "unreadable"

KIND_LABELS = {
    KIND_OPERATOR: "You said so",
    KIND_ALIAS: "Known alias",
    KIND_NAME: "Same name",
    KIND_SIMILAR: "Similar name",
    KIND_MODEL: "A model's suggestion",
    KIND_NONE: "Nothing matched",
    KIND_UNREADABLE: "Could not be read",
}

#: Tier -> kind, for the tiers the deterministic mapper reports.
_TIER_KINDS = {
    "operator_approved": KIND_OPERATOR,
    "alias": KIND_ALIAS,
    "exact": KIND_NAME,
    "normalized": KIND_NAME,
    "token_set": KIND_SIMILAR,
    "fuzz_token_set": KIND_SIMILAR,
    "fuzz_ratio_norm": KIND_SIMILAR,
    "unreadable": KIND_UNREADABLE,
}


def match_kind(tier: str, basis: str) -> str:
    """Which of the seven kinds this row's reading is.

    The BASIS wins where the two disagree: a column the mapper could not place
    and a model proposed a field for has tier ``unmapped``, and calling it
    "Nothing matched" would hide the proposal the operator is being asked
    about.
    """
    if basis == BASIS_MODEL:
        return KIND_MODEL
    return _TIER_KINDS.get(str(tier or ""), KIND_NONE)


def classify(row: Dict[str, Any], *, has_open_decision: bool = False,
             is_proposal: bool = False) -> str:
    """What became of one column, from the row the mapper wrote.

    An open decision wins over the tier. Two columns claiming one canonical
    field is an ambiguity, and ``execution`` raises it as a decision AFTER both
    rows have already been written to the report at whatever tier they matched
    at — often an exact or alias match, because that is how both came to claim
    the same field. Reading the tier alone, the table would say "matched
    automatically" about a column the run is blocked on.

    A weak match outside the primary tape reads ``ROW_UNCHECKED`` only on a run
    rehearsed before every file's columns were put to a person. Nothing raises
    that state now; it is kept so an older case still says what happened to it
    rather than claiming a question was asked that never was.
    """
    if has_open_decision:
        # A proposal is open and blocking, like any other — but it is answered
        # by approving the set, not by working through a list of questions, so
        # calling it "Needs you" would put seventy clean columns in the same
        # queue as the three that are genuinely unresolved.
        return ROW_PROPOSED if is_proposal else ROW_NEEDS_YOU
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
                             ) -> Dict[Tuple[str, str], Tuple[str, str, str]]:
    """``{(file, column): (decision_id, decision_type, evidence)}`` per open
    decision.

    So a row reading "Needs you" can be the thing you click, rather than
    sending an operator to hunt for the matching question in another list —
    and so the table can tell a PROPOSAL, which is approved with the set, from
    a question that has to be answered on its own.

    KEYED ON THE PAIR. Keyed on the column NAME alone, one proposal raised
    against the tape's 'Pool' marked the property extract's 'Pool' and the
    cashflow extract's 'Pool' as proposed too — three rows pointing at one
    question, two of which that question does not answer. A decision that
    names no file is still matched on the name alone, so a case part-way
    through its onboarding does not lose the link to answers it already has.
    """
    out: Dict[Tuple[str, str], Tuple[str, str, str]] = {}
    for decision in decisions or []:
        if str(decision.get("status", "open")) != "open":
            continue
        subject = decision.get("subject") or {}
        entry = (str(decision.get("decision_id") or ""),
                 str(subject.get("decision_type") or ""),
                 _evidence_of(decision))
        source_file = str(subject.get("source_file") or "").strip()
        names = [str(subject.get("source_column") or "").strip()]
        names += [str(c or "").strip()
                  for c in (subject.get("source_columns") or [])]
        for name in names:
            if name:
                out.setdefault((source_file, name.lower()), entry)
    return out


def _evidence_of(decision: Dict[str, Any]) -> str:
    """What the question said, in the words it said it in.

    THIS USED TO LIVE ONLY ON A CARD. An ambiguity's card carried the one thing
    that actually settles it — "'Current Balance' carries values for 24 of 24
    records; 'Principal Balance' for 21 of 24" — and when every mapping
    question moved onto the table, deleting the card would have deleted the
    evidence with it. A row that asks an operator to choose and does not say
    what the choice turns on is a row that gets guessed at.
    """
    for item in decision.get("evidence") or []:
        data = item.get("data") if isinstance(item, dict) else None
        if isinstance(data, dict):
            detail = str(data.get("detail") or "").strip()
            if detail:
                return detail
    return str(decision.get("question") or "").strip()


def _decision_for(decisions: Dict[Tuple[str, str], Tuple[str, str, str]],
                  source_file: str, column: str) -> Tuple[str, str, str]:
    """This row's open decision: its own file's, or an unscoped one."""
    key = column.lower()
    return (decisions.get((source_file, key))
            or decisions.get(("", key))
            or ("", "", ""))


def decision_for_column(decisions: Iterable[Dict[str, Any]], source_file: str,
                        column: str) -> Tuple[str, str, str]:
    """``(decision_id, decision_type)`` for one column of one file.

    The same lookup the table uses, exported so a staged answer records the
    decision it will resolve rather than the service re-deriving the match and
    the two disagreeing about which question a row belongs to.
    """
    return _decision_for(_open_decision_by_column(decisions), source_file,
                         column)


def _mark_contested(rows: List[Dict[str, Any]]) -> None:
    """Name, on every row, the other columns claiming the same field.

    WHY THIS APPEARED WHEN EVERY FILE STARTED BEING PROPOSED. Two columns of
    ONE file claiming one canonical field has always been an ambiguity the
    engine cannot resolve, and ``execution`` raises it as a blocking question.
    Two FILES carrying the same fact is not that: it is the ordinary shape of a
    delivery, the cross-file step reconciles it, and blocking on it would stop
    almost every pack — a loan identifier legitimately appears in every
    extract.

    But it was invisible, and it is no longer harmless. Every file's columns
    are now proposed, so an operator approving the set approves four columns
    onto one field and four governed rules follow. They are entitled to see
    that set before they approve it, and to see it on each row rather than
    having to hold the whole table in their head.

    Reported, never blocking. Which file to believe is a source-precedence
    rule, and that is not an artefact this screen can write.
    """
    claims: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        # A STAGED ANSWER IS WHAT THIS COLUMN CLAIMS NOW. Reading the report
        # instead, an operator who had just set the losing column aside would
        # still see "2 columns claim this" on both — and the one piece of
        # feedback they need is whether they have resolved the clash.
        field = (row["staged_field"] if row["staged_action"]
                 else row["canonical_field"])
        # A row with no field claims nothing, and one that could not be read
        # claims nothing either.
        if field and row["state"] != ROW_UNREADABLE:
            claims.setdefault(field, []).append(row)
    for row in rows:
        row["also_claimed_by"] = []
    for field, claimants in claims.items():
        if len(claimants) < 2:
            continue
        for row in claimants:
            row["also_claimed_by"] = [
                {"source_file": other["source_file"],
                 "source_column": other["source_column"],
                 # Whether the clash is inside one file, which IS a blocking
                 # ambiguity, or across the pack, which is not. The two read
                 # differently to an operator and must not be conflated.
                 "same_file": other["source_file"] == row["source_file"]}
                for other in claimants if other is not row]


def overview(run: Any) -> Dict[str, Any]:
    """Every source column and what became of it, ready to render.

    Ordered worst-first within each file, so the columns that need a person are
    at the top of the table rather than wherever the tape happened to put them.
    """
    decisions = _open_decision_by_column(getattr(run, "open_decisions", []))
    # Columns an operator asked for a new canonical field for. Still unmapped —
    # a request is not a field — but the row has to say the ask was made, or an
    # operator scanning eighty-nine unused columns cannot tell the ones they
    # have already dealt with from the ones they have not.
    requests = {(str(r.get("source_file") or ""),
                 str(r.get("source_column") or "").lower()): r
                for r in (getattr(run, "field_requests", None) or [])
                if str(r.get("status") or "") == "requested"}
    # The operator's working copy: what they have said each column is, held as
    # a draft until the set is committed. See :mod:`.staging`.
    staged = {(str(e.get("source_file") or ""),
               str(e.get("source_column") or "")): e
              for e in (getattr(run, "staged_mappings", None) or [])}
    rows: List[Dict[str, Any]] = []
    for raw in getattr(run, "mapping_report", []) or []:
        column = str(raw.get("source_column") or "")
        source_file = str(raw.get("source_file") or "")
        decision_id, decision_type, evidence = _decision_for(
            decisions, source_file, column)
        answer = staged.get((source_file, column))
        state = classify(raw, has_open_decision=bool(decision_id),
                         is_proposal=(decision_type == "mapping_proposal"))
        if answer is not None:
            # A staged answer wins over everything the mapper said: the
            # operator has read the column and named it, and a row still
            # reading "Needs you" after they answered would send them back to
            # answer it twice.
            state = ROW_STAGED
        confidence: Optional[float] = None
        if raw.get("confidence") is not None:
            try:
                confidence = round(float(raw["confidence"]), 4)
            except (TypeError, ValueError):
                confidence = None
        tier = str(raw.get("tier") or "")
        # What a model proposed for a column Trakt could not place. Never a
        # mapping — it is carried beside the row so the table can show what was
        # suggested AND where the suggestion came from, because "suggested
        # mapping" with no basis beside it invites an operator to accept a
        # proposal on the same footing as a contract-backed match.
        suggested = str(raw.get("llm_field") or "")
        if tier == "operator_approved":
            basis = BASIS_OPERATOR
        elif str(raw.get("canonical_field") or ""):
            basis = BASIS_DETERMINISTIC
        elif suggested:
            basis = BASIS_MODEL
        else:
            basis = ""
        rows.append({
            "source_file": source_file,
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
            # What the question this row is waiting on said. Carried onto the
            # row because the row is where it is now answered.
            "decision_detail": evidence,
            "basis": basis,
            "basis_label": BASIS_LABELS.get(basis, ""),
            "match_kind": match_kind(tier, basis),
            "match_kind_label": KIND_LABELS.get(match_kind(tier, basis), ""),
            # What the operator staged, and nothing at all when they have not
            # been through this column yet.
            "staged_action": str((answer or {}).get("action") or ""),
            "staged_field": str((answer or {}).get("target_field") or ""),
            "staged_label": _label((answer or {}).get("target_field")),
            "staged_by": str((answer or {}).get("staged_by") or ""),
            # Whether the operator set this column aside themselves or a field
            # request set it aside for them. The two undo differently — see
            # :func:`staging.is_request_driven` — so the row says which rather
            # than leaving the screen to infer it from the request beside it.
            "staged_origin": str((answer or {}).get("origin") or ""),
            "suggested_field": suggested,
            "suggested_label": _label(suggested),
            "suggested_reason": str(raw.get("llm_reasoning") or ""),
            "requested_field": str(
                (requests.get((source_file, column.lower())) or {}
                 ).get("field_name") or ""),
            # Whether this column's file is the one the canonical tape is built
            # from. The table groups by file and says which is which, so an
            # operator is not left to infer it from the filename.
            "primary": bool(raw.get("primary", True)),
        })

    _mark_contested(rows)

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
    contested = sum(1 for r in rows if r["also_claimed_by"])
    # A question the operator has not staged an answer to. `classify` puts a
    # staged row in ROW_STAGED, so what is left in ROW_NEEDS_YOU is exactly
    # the set that has no answer — which is what the commit refuses on.
    unanswered = counts.get(ROW_NEEDS_YOU, 0)
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
        # WHAT THE APPROVAL ACT NEEDS TO SAY FOR ITSELF.
        #
        # `proposed` is how many columns one approval would settle, so the
        # button can name its own consequence instead of reading "Approve".
        # `blocking_questions` is how many are NOT approvable that way: a
        # genuine ambiguity or a weak match has to be answered on its own, and
        # an operator told "approve 71" while three questions wait would be
        # told the run is one click from moving when it is not.
        "proposed": counts.get(ROW_PROPOSED, 0),
        "blocking_questions": unanswered,
        # WHAT THE COMMIT WOULD DO, so the button can name its own consequence
        # rather than reading "Confirm". `staged` is what the operator changed
        # or confirmed by hand; `proposed` is what they left as Trakt read it
        # and which commits with the rest; `unanswered_questions` is what has
        # no answer at all and is why the button is refused.
        "staged": counts.get(ROW_STAGED, 0),
        "to_confirm": counts.get(ROW_STAGED, 0) + counts.get(ROW_PROPOSED, 0),
        "unanswered_questions": unanswered,
        # How many rows are one of two or more claiming a single field. Not a
        # blocker — see `_mark_contested` — but the operator asked to be able
        # to find them, and a count they cannot filter to is a number.
        "contested": contested,
    }
