#!/usr/bin/env python3
"""clause_splitting_phase1/reference_splitter.py

A reference implementation of the rules in `rules/clause_rules.yaml`, built for
ONE purpose: to run them read-only over the question corpus so §4.2 can report
what they cannot classify.

It is an analysis tool, not the Phase 2 layer. It is not imported by product
code, not wired into anything, and its output is a report. Phase 2 builds the
real thing as a pure function with its own tests; nothing here is that.

It reads the rules file and does nothing that is not written there. Where a
rule is `withheld` it is not implemented, so the wording that rule would have
claimed falls through to rule 35 and is reported as unresolved — which is the
finding, not a bug to be papered over.

DOMAIN BLINDNESS. The splitter decides only that a clause of some kind starts
here and stops there. It reads no registry and knows no field meanings. It
judges operands by SHAPE alone — does this look like a date, a quantity, a
duration — which is what rules 31 and 32 turn on. Concept resolution is a
separate pass in `resolve.py`, standing in for business semantics, and it runs
strictly INSIDE the spans this file produces.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_RULES_PATH = Path(__file__).resolve().parent / "rules" / "clause_rules.yaml"

_TIME_UNITS = ("day", "days", "week", "weeks", "month", "months",
               "quarter", "quarters", "year", "years")

_GRAIN_OF = {
    "day": "day", "daily": "day", "days": "day",
    "week": "week", "weekly": "week", "weeks": "week",
    "month": "month", "monthly": "month", "months": "month",
    "quarter": "quarter", "quarterly": "quarter", "quarters": "quarter",
    "year": "year", "yearly": "year", "annual": "year", "years": "year",
}

_COMPARATORS = ("above", "below", "over", "under", "at least", "of at least",
                "greater than", "less than", "more than", "exceeding",
                "exceed", "exceeds", "in excess of")

_QUANTITY = re.compile(
    r"(?:£|\$|€)\s*\d[\d,.]*\s*(?:k|m|bn|b|mm)?|"      # £250k, £100m
    r"\b\d[\d,.]*\s*(?:%|per cent|percent)|"            # 50%
    r"\b\d[\d,.]*\s*(?:k|m|bn|mm)\b|"                   # 250k
    r"\b\d[\d,.]*\b", re.I)

_DATE_LIKE = re.compile(
    r"\b(19|20)\d{2}\b|"
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|"
    r"\bq[1-4]\b", re.I)

_DURATION = re.compile(r"\b\d+\s*(?:%s)\b" % "|".join(_TIME_UNITS), re.I)


@dataclass
class Opener:
    """One opener match that survived rule filtering."""
    rule_id: int
    section: str
    span: str
    start: int
    end: int
    text: str
    value: Optional[str] = None
    kind: Optional[str] = None
    #: Set when this opener lost a competition, with the rule that beat it.
    beaten_by: Optional[Tuple[int, str]] = None


@dataclass
class Split:
    """A question cut into labelled spans of TEXT. No concepts, by design."""
    question: str
    operation: Optional[str] = None
    operation_text: Optional[str] = None
    subject_text: Optional[str] = None
    grouping_texts: List[str] = field(default_factory=list)
    grouping_grains: List[str] = field(default_factory=list)
    filter_texts: List[str] = field(default_factory=list)
    period_texts: List[str] = field(default_factory=list)
    period_kinds: List[str] = field(default_factory=list)
    target_texts: List[str] = field(default_factory=list)
    target_kind: Optional[str] = None
    second_population_text: Optional[str] = None
    unresolved: List[str] = field(default_factory=list)
    #: (losing_rule, winning_rule, wording, precedence_rule) — §4.2 item 2.
    competitions: List[Tuple[int, int, str, int]] = field(default_factory=list)
    fired: List[int] = field(default_factory=list)


class ReferenceSplitter:
    def __init__(self, rules_path: Path = _RULES_PATH):
        doc = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        self.rules: Dict[int, dict] = {r["id"]: r for r in doc["rules"]}
        self._compiled: List[Tuple[re.Pattern, dict, str]] = []
        for r in doc["rules"]:
            if r.get("status") == "withheld":
                continue
            for op in (r.get("openers") or []):
                pat = re.compile(r"(?<![a-z])%s(?![a-z])" % re.escape(op.strip()), re.I)
                self._compiled.append((pat, r, op.strip()))
        # Longest opener first, so 'by month' beats 'by' at the same offset.
        self._compiled.sort(key=lambda t: -len(t[2]))

    # -- opener discovery -------------------------------------------------
    def _candidates(self, q: str) -> List[Opener]:
        out: List[Opener] = []
        for pat, rule, op in self._compiled:
            for m in pat.finditer(q):
                out.append(Opener(
                    rule_id=rule["id"], section=rule["section"],
                    span=rule.get("span") or "", start=m.start(), end=m.end(),
                    text=op, value=rule.get("value"), kind=rule.get("kind")))
        return out

    def _admissible(self, cand: Opener, q: str) -> bool:
        """Rule-specific conditions that must hold for an opener to open."""
        rule = self.rules[cand.rule_id]
        tail = q[cand.end:]
        if rule.get("requires_comparator"):
            # Rules 11 and 12: 'with' / 'for' open a filter only alongside a
            # comparator. Otherwise 'for each region' and 'with a broker' would
            # open filter spans on nothing.
            window = tail[:60]
            return any(c in window for c in _COMPARATORS)
        if cand.rule_id == 5:
            # 'which' alone is a listing. A direction word must be present.
            d = rule.get("direction") or {}
            words = list(d.get("descending", [])) + list(d.get("ascending", []))
            if cand.text == "which":
                return any(re.search(r"(?<![a-z])%s(?![a-z])" % w, q) for w in words)
        if cand.rule_id == 20 and cand.text == "each":
            # Rule 26 owns 'each <time unit>'.
            return not re.match(r"\s*(%s)\b" % "|".join(_TIME_UNITS), tail, re.I)
        if cand.rule_id == 39:
            return False  # 'at <value>' needs a forward operation; applied later
        if cand.rule_id == 22 and cand.text in ("this", "current"):
            # 'this|current <unit>'. Without a time unit, 'the current balance'
            # is not a period — it is the subject.
            return bool(re.match(r"\s*(%s)\b" % "|".join(_TIME_UNITS), tail, re.I))
        if cand.section == "F" and rule.get("kind") != "configured":
            # Rule 42: a target requires a forward operation. Without one,
            # 'exceed' is a bare comparator (rule 13), not a target opener.
            return self._has_forward_opener(q)
        return True

    def _has_forward_opener(self, q: str) -> bool:
        for rid in (6, 7, 8):
            for op in (self.rules[rid].get("openers") or []):
                if re.search(r"(?<![a-z])%s(?![a-z])" % re.escape(op), q, re.I):
                    return True
        return False

    # -- competition ------------------------------------------------------
    def _resolve_competitions(self, cands: List[Opener], split: Split) -> List[Opener]:
        """Leftmost-outermost (rule 33), plus the specificity preference that
        lets rule 26 beat rule 17 on the same words."""
        cands.sort(key=lambda c: (c.start, -(c.end - c.start)))
        kept: List[Opener] = []
        claimed_to = -1
        for c in cands:
            overlapping = [k for k in kept if not (c.end <= k.start or c.start >= k.end)]
            if overlapping:
                k = overlapping[0]
                # A time-axis opener beats a plain grouping opener on the same
                # words: 'by month' is rule 26, never rule 17 on the noun month.
                if c.rule_id == 26 and k.rule_id in (17, 20):
                    kept.remove(k)
                    split.competitions.append((k.rule_id, 26, c.text, 26))
                    kept.append(c)
                    claimed_to = max(claimed_to, c.end)
                else:
                    split.competitions.append((c.rule_id, k.rule_id, c.text, 33))
                continue
            if c.start < claimed_to:
                split.competitions.append((c.rule_id, kept[-1].rule_id if kept else 0,
                                           c.text, 33))
                continue
            kept.append(c)
            claimed_to = max(claimed_to, c.end)
        return kept

    def _suppress_reopeners(self, kept: List[Opener], s: Split) -> List[Opener]:
        """Rule 33, applied to bare comparators.

        `where LTV above 50%` is ONE filter clause. Rule 10 opened it and it is
        still open, so the bare comparator of rule 13 falls inside a span
        already consumed and does not reopen. Without this the clause splits in
        two and the field name is stranded between the halves.
        """
        out: List[Opener] = []
        open_filter = False
        for c in sorted(kept, key=lambda c: c.start):
            if c.rule_id == 13 and open_filter:
                s.competitions.append((13, out[-1].rule_id, c.text, 33))
                continue
            if c.section in ("B",) and c.rule_id in (10, 11, 12):
                open_filter = True
            elif c.section in ("C", "D", "F") or c.rule_id in (14, 15):
                open_filter = False
            out.append(c)
        return out

    # -- shape tests (domain-blind) ---------------------------------------
    @staticmethod
    def _looks_like_date(s: str) -> bool:
        return bool(_DATE_LIKE.search(s))

    @staticmethod
    def _looks_like_quantity(s: str) -> bool:
        return bool(_QUANTITY.search(s))

    @staticmethod
    def _looks_like_duration(s: str) -> bool:
        return bool(_DURATION.search(s)) or bool(
            re.search(r"\b(%s)\b" % "|".join(_TIME_UNITS), s, re.I))

    # -- main -------------------------------------------------------------
    def split(self, question: str) -> Split:
        q = question.strip()
        s = Split(question=q)
        cands = [c for c in self._candidates(q) if self._admissible(c, q)]
        kept = self._resolve_competitions(cands, s)

        kept = self._suppress_reopeners(kept, s)
        content = [c for c in kept if c.section in ("B", "C", "D", "F")]
        operations = [c for c in kept if c.section == "A"]

        # The operation span takes the earliest operation opener only. A second
        # one inside a consumed span does not reopen (rule 33).
        if operations:
            op = min(operations, key=lambda c: c.start)
            s.operation = op.value
            s.operation_text = op.text
            s.fired.append(op.rule_id)
            for other in operations:
                if other is not op:
                    s.competitions.append((other.rule_id, op.rule_id, other.text, 33))

        # Each content opener claims from its own start to the next one's start.
        content.sort(key=lambda c: c.start)
        bounds = []
        for i, c in enumerate(content):
            stop = content[i + 1].start if i + 1 < len(content) else len(q)
            bounds.append((c, q[c.start:stop].strip(" ,.?")))

        consumed: List[Tuple[int, int]] = []
        prev_end = 0
        for i, (c, text) in enumerate(bounds):
            s.fired.append(c.rule_id)
            operand = text[len(c.text):].strip(" ,.?")
            start = c.start
            stop = content[i + 1].start if i + 1 < len(content) else len(q)
            reclassified = (c.rule_id == 13 and c.text == "over"
                            and self._looks_like_duration(q[c.end:]))
            if c.rule_id == 13 and not reclassified:
                # Rule 13 as amended: a BARE comparator names no field, so the
                # field it constrains sits to its LEFT. Without extending the
                # span leftwards the field name is left in the residue and
                # becomes the subject — which is defect occurrence 2 exactly.
                start, text = self._extend_left(q, c, prev_end, text)
            consumed.append((start, stop))
            self._place(s, c, text, operand)
            prev_end = stop

        # Subject is the residue (rule 34), never computed by scanning.
        s.subject_text = self._residue(q, operations, consumed)

        # Post-passes that need the whole split in view.
        self._apply_target_precedence(s)
        self._apply_estimate_discard(s)
        self._apply_rule_35(s)
        return s

    # -- placement --------------------------------------------------------
    def _place(self, s: Split, c: Opener, text: str, operand: str) -> None:
        rid = c.rule_id
        if rid == 26:  # time axis
            grain = None
            for word, g in _GRAIN_OF.items():
                if re.search(r"(?<![a-z])%s(?![a-z])" % word, c.text, re.I):
                    grain = g
                    break
            s.grouping_texts.append(text)
            s.grouping_grains.append(grain or "unspecified")
            return
        if c.span == "grouping":
            s.grouping_texts.append(text)
            s.grouping_grains.append(None)
            return
        if c.span == "target":
            s.target_texts.append(text)
            s.target_kind = self.rules[rid].get("kind") or "value"
            return
        if c.span == "period":
            if rid == 25:  # rule 27: period or second population
                if self._looks_like_date(operand) or self._looks_like_duration(operand):
                    s.period_texts.append(text)
                    s.period_kinds.append("comparison")
                    s.competitions.append((25, 27, c.text, 27))
                else:
                    s.second_population_text = operand
                    s.competitions.append((25, 27, c.text, 27))
                return
            if rid == 23:  # rule 28: qualifies a time unit, or a population
                if self._looks_like_duration(operand):
                    s.period_texts.append(text)
                    s.period_kinds.append("prior")
                else:
                    s.second_population_text = operand
                s.competitions.append((23, 28, c.text, 28))
                return
            s.period_texts.append(text)
            s.period_kinds.append(self.rules[rid].get("kind") or "period")
            return
        if c.span == "filter":
            if rid == 14:  # rule 31: between
                dates = len(_DATE_LIKE.findall(operand))
                if dates >= 2:
                    s.period_texts.append(text)
                    s.period_kinds.append("explicit_range")
                    s.competitions.append((14, 31, c.text, 31))
                    return
                s.competitions.append((14, 31, c.text, 31))
            if rid == 13 and c.text == "over":  # rule 32: duration or quantity
                lookahead = s.question[c.end:]
                if self._looks_like_duration(lookahead):
                    s.period_texts.append(text)
                    s.period_kinds.append("window")
                    s.competitions.append((13, 32, c.text, 32))
                    return
                s.competitions.append((13, 32, c.text, 32))
            s.filter_texts.append(text)
            return

    #: Words that are never the field a bare comparator constrains.
    _LEFT_STOP = {"have", "has", "had", "is", "are", "was", "were", "with",
                  "of", "a", "an", "the", "that", "do", "we", "i", "and", "or",
                  "loans", "loan", "cases", "case", "accounts", "borrowers",
                  "any", "there", "which", "whose", "s"}

    def _extend_left(self, q: str, c: Opener, floor: int,
                     text: str) -> Tuple[int, str]:
        segment = q[floor:c.start]
        words = re.findall(r"\S+", segment)
        take: List[str] = []
        for w in reversed(words[-4:]):
            if w.strip(" ,.?'").lower() in self._LEFT_STOP:
                break
            take.insert(0, w)
        if not take:
            return c.start, text
        phrase = " ".join(take)
        start = c.start - (len(phrase) + 1)
        return max(floor, start), (phrase + " " + text).strip()

    def _residue(self, q: str, operations: List[Opener],
                 consumed: List[Tuple[int, int]]) -> str:
        cuts = sorted(consumed + [(o.start, o.end) for o in operations])
        out, pos = [], 0
        for a, b in cuts:
            if a > pos:
                out.append(q[pos:a])
            pos = max(pos, b)
        if pos < len(q):
            out.append(q[pos:])
        return re.sub(r"\s+", " ", " ".join(out)).strip(" ,.?")

    # -- post-passes ------------------------------------------------------
    def _apply_target_precedence(self, s: Split) -> None:
        """Rules 41 and 42. A quantity in a forward question is a target."""
        forward = s.operation in ("forward", "forward_probability")
        if forward and s.filter_texts:
            moved = [t for t in s.filter_texts if self._looks_like_quantity(t)]
            for t in moved:
                s.filter_texts.remove(t)
                s.target_texts.append(t)
                s.target_kind = s.target_kind or "value"
                s.competitions.append((13, 41, t, 41))
                s.fired.append(41)
        if s.target_texts and not forward:
            # Rule 42: a target on a historical operation is a contradiction.
            for t in s.target_texts:
                s.unresolved.append("target on a non-forward operation: %s" % t)
            s.target_texts = []
            s.target_kind = None
            s.fired.append(42)

    def _apply_estimate_discard(self, s: Split) -> None:
        """Rule 30's exception: 'estimate the current balance' is not a forecast."""
        if s.operation != "forward" or s.operation_text not in (
                "estimate", "roughly when", "approximately when",
                "what's your estimate of"):
            return
        if s.target_texts or s.period_texts:
            return
        rest = " ".join(filter(None, [s.subject_text or ""] + s.grouping_texts))
        if re.search(r"\b(current|currently|today|now|to date|so far|is there|"
                     r"do we have|in the book)\b", rest, re.I):
            s.operation = None
            s.fired.append(30)
            s.competitions.append((8, 30, s.operation_text or "estimate", 30))

    def _apply_rule_35(self, s: Split) -> None:
        """An unrecognised trailing prepositional phrase is unresolved, not subject."""
        sub = s.subject_text or ""
        m = re.search(r"(?<![a-z])(in|to|for|on|against|with|as per|of the)\s+(.+)$",
                      sub, re.I)
        if not m:
            return
        trailing = m.group(0).strip()
        # A bare 'of the book' style tail with no opener that claimed it.
        if len(trailing.split()) >= 2:
            s.unresolved.append(trailing)
            s.subject_text = sub[:m.start()].strip(" ,.?") or None
            s.fired.append(35)
