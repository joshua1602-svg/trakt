"""
review_interpreter.py
====================

PART 4/5 — LLM-assisted answer interpretation, governed exactly like the MI
Agent parser:

    natural-language answer
      -> interpreted structured ReviewAnswerSpec      (this module)
      -> deterministic validation                     (answer_ingestion rules)
      -> proposed updates                             (validation block)
      -> HUMAN CONFIRMATION required                  (cli --confirm)
      -> approved artefacts written                   (answer_ingestion)

Two modes, like ``llm_query_parser.parse_user_question``:
  * llm_enabled=False (default) -> deterministic, offline keyword interpreter
    (safe for unit tests; no network / API key).
  * llm_enabled=True            -> optional, mockable ``llm_callable`` that maps
    text -> {qid: answer}. Output is treated as data and re-validated; it is
    NEVER executed and NEVER writes approved artefacts.

The interpreter only ever proposes; it does not write approved artefacts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from .answer_ingestion import ProjectContext, _validate_answer
from .gap_analyzer import ENUM_DECISION_ACTIONS
from .review_models import InterpretedAnswer, ReviewAnswerSpec

_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b")

# Phrase -> source file matching for source-of-truth answers.
_SOURCE_PHRASES = {
    "loan report": "loan_report",
    "loan tape": "loan_report",
    "cashflow report": "cashflow_report",
    "cash flow report": "cashflow_report",
    "collateral report": "collateral_report",
    "pipeline report": "pipeline_report",
}

# Enum action keyword detection.
_ENUM_PHRASES = [
    ("treat_as_missing", ("treat", "missing")),
    ("treat_as_missing", ("as missing",)),
    ("map_to_othr", ("othr",)),
    ("map_to_nd1", ("nd1",)),
    ("exclude_field_from_regulatory_delivery", ("exclude",)),
    ("provide_custom_mapping", ("custom",)),
]


def _match_source_file(text_l: str, candidates: List[str]) -> Optional[str]:
    # Direct filename mention wins.
    for c in candidates:
        if c.lower() in text_l:
            return c
    # Phrase -> classification token, then candidate whose name contains it.
    for phrase, token in _SOURCE_PHRASES.items():
        if phrase in text_l:
            key = token.replace("_report", "")
            for c in candidates:
                if key in c.lower():
                    return c
    return None


# Keywords that tie a source-of-truth question's subject to a text clause.
_SUBJECT_KEYWORDS = {
    "geography": ("region", "geography", "collateral"),
    "balance": ("balance", "principal"),
    "funding": ("funding", "pipeline"),
    "valuation": ("valuation", "value"),
}


def _subject_keywords(subject: str) -> set:
    kws = set(subject.lower().split("_"))
    for token, extra in _SUBJECT_KEYWORDS.items():
        if token in subject.lower():
            kws.update(extra)
    return {k for k in kws if len(k) > 2}


def _match_source_clause(text_l: str, candidates: List[str], subject: str) -> Optional[str]:
    """Match the source named in the SAME clause as the question's subject.

    Handles answers naming several reports, e.g. 'use the loan report for
    balance, use the collateral report for region'.
    """
    kws = _subject_keywords(subject)
    for clause in re.split(r",|\band\b|;", text_l):
        if any(k in clause for k in kws):
            m = _match_source_file(clause, candidates)
            if m:
                return m
    return _match_source_file(text_l, candidates)


def _interpret_one(question: Dict[str, Any], text: str, text_l: str) -> Optional[InterpretedAnswer]:
    category = question.get("category", "")
    candidates = question.get("candidate_answers", []) or []

    if category == "date":
        # Prefer a date that is an offered candidate; else any date in the text.
        for c in candidates:
            if c in text:
                return InterpretedAnswer(c, 0.9, "Matched an offered reporting date.")
        m = _DATE_RE.search(text)
        if m:
            return InterpretedAnswer(m.group(1), 0.7, "Parsed a date from free text.")
        return None

    if category == "source_of_truth":
        match = _match_source_clause(text_l, candidates, question.get("subject", ""))
        if match:
            return InterpretedAnswer(match, 0.9, "Matched source report phrase.")
        return None

    if category == "enum":
        for action, needles in _ENUM_PHRASES:
            if all(n in text_l for n in needles):
                canonical = action if action in ENUM_DECISION_ACTIONS else action
                # normalise case to the canonical action token
                canonical = {a.lower(): a for a in ENUM_DECISION_ACTIONS}.get(action.lower(), action)
                return InterpretedAnswer(canonical, 0.9, "Detected enum decision phrase.")
        return None

    if category == "geography":
        if "gbzzz" in text_l:
            return InterpretedAnswer("GBZZZ", 0.9, "Confirmed GBZZZ for ESMA.")
        if "itl3" in text_l or "itl" in text_l:
            return InterpretedAnswer("ITL3", 0.8, "Selected ITL3.")
        return None

    # Generic / config: pick a candidate mentioned in the text.
    for c in candidates:
        if c and c.lower() in text_l:
            return InterpretedAnswer(c, 0.7, "Matched a candidate answer.")
    return None


def interpret_answers(
    text: str,
    questions: List[Dict[str, Any]],
    llm_enabled: bool = False,
    llm_callable: Optional[Callable[[str, List[Dict[str, Any]]], Dict[str, str]]] = None,
) -> ReviewAnswerSpec:
    """Interpret a natural-language answer into a :class:`ReviewAnswerSpec`.

    Deterministic by default. If ``llm_enabled`` / ``llm_callable`` is provided,
    the callable returns a ``{question_id: answer_text}`` mapping which is then
    wrapped and (later) deterministically validated.
    """
    text_l = text.lower()

    if llm_enabled or llm_callable is not None:
        raw: Dict[str, str] = (llm_callable or (lambda t, q: {}))(text, questions)
        answers = {
            qid: InterpretedAnswer(str(val), 0.8, "LLM-interpreted answer.")
            for qid, val in raw.items()
        }
        return ReviewAnswerSpec(answers=answers, requires_confirmation=True, interpreter="llm")

    answers: Dict[str, InterpretedAnswer] = {}
    for q in questions:
        ia = _interpret_one(q, text, text_l)
        if ia is not None:
            answers[q.get("question_id")] = ia
    return ReviewAnswerSpec(answers=answers, requires_confirmation=True, interpreter="deterministic")


# ---------------------------------------------------------------------------
# Validation + dry-run file output (PART 5)
# ---------------------------------------------------------------------------


def validate_spec(spec: ReviewAnswerSpec, ctx: ProjectContext) -> Dict[str, Any]:
    """Run the SAME deterministic validation as answer ingestion."""
    q_by_id = ctx.question_by_id()
    blocking_ids = [q["question_id"] for q in ctx.questions if q.get("severity") == "blocking"]

    invalid: List[Dict[str, str]] = []
    proposed_updates: List[str] = []
    for qid, ia in spec.answers.items():
        q = q_by_id.get(qid)
        if not q:
            invalid.append({"question_id": qid, "reason": "unknown question id"})
            continue
        ok, reason = _validate_answer(q, {"answer": ia.answer})
        if not ok:
            invalid.append({"question_id": qid, "reason": reason})
            continue
        cat = q.get("category")
        if cat == "date":
            proposed_updates.append("config/reporting_date")
        elif cat == "source_of_truth":
            proposed_updates.append(f"source_precedence/{q.get('subject')}")
        elif cat == "enum":
            proposed_updates.append(f"enum/{q.get('subject')}.{q.get('subject_value')}")
        elif cat == "geography":
            proposed_updates.append("config/geography_policy")
        elif cat == "config":
            proposed_updates.append(f"config/{q.get('subject')}")

    blocking_unanswered = [qid for qid in blocking_ids if qid not in spec.answers]
    if invalid:
        status = "invalid"
    elif blocking_unanswered:
        status = "requires_review"
    else:
        status = "valid"

    return {
        "status": status,
        "invalid_answers": invalid,
        "blocking_unanswered": blocking_unanswered,
        "proposed_updates": sorted(set(proposed_updates)),
    }


def interpret_answers_to_file(project_dir: str | Path, text: str) -> Path:
    """Dry-run: interpret + validate, write 16_interpreted_answer_spec.yaml only.

    Never writes approved artefacts (10..15).
    """
    project_dir = Path(project_dir)
    ctx = ProjectContext(project_dir)
    spec = interpret_answers(text, ctx.questions)
    spec.validation = validate_spec(spec, ctx)

    payload = spec.to_ingestible_dict()
    out = project_dir / "16_interpreted_answer_spec.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out
