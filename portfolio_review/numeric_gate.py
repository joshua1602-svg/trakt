"""The publication gate: no figure reaches a reader unless Trakt returned it.

THE RULE
--------
    No numeric value may be published unless that exact numeric value
    originates from a governed tool result available to the agent.

THE EVIDENCE THIS EXISTS ON
---------------------------
The real-model red-team put the agent on real canonical and it published:

    "ORIGINATION-0043 at 70.99% (£954k) and SPV1-0022 at 70.79% (£926k).
     Combined they are £1.88m."

£1,880,000 appears in no payload of that session. The model added two numbers
and narrated the operation while doing it. It also published `93%` of a movement
(a division), `5.14 percentage points` (a subtraction) and `£7.51m` (a
percentage of a total). The system prompt forbade every one of these in its
first and most emphatic rule.

So prompting is not a control here, and this module is not a stronger warning —
it is a deterministic post-condition. A finding whose figures cannot be traced
does not reach the card.

WHAT THIS IS NOT
----------------
Not an analytics engine. It computes no measure, corrects no number and answers
no question. It has exactly one operation — *does this stated figure appear in
what Trakt returned* — and one lever, which is what to do when it does not.

WHY IT CANNOT SIMPLY RECOMPUTE
------------------------------
A gate that worked out what the model meant and substituted the right number
would be a second source of financial truth, which is the thing the whole estate
is arranged to prevent. It refuses; it never repairs.

THE SCALINGS, AND WHY NO ARITHMETIC IS AMONG THEM
-------------------------------------------------
A writer legitimately says "£11.97m" for 11,974,544.28, and that has to pass.
So a stated figure is matched against every governed number under unit scalings
(m, bn, k, percent) at the precision the writer chose. It is never matched
against a *combination* of two governed numbers, because combining is precisely
the act being detected. That asymmetry is the whole design: presentation is
allowed, derivation is not.

WHAT ESCAPES IT, STATED PLAINLY
-------------------------------
A derived figure that coincides with some unrelated governed number passes. With
several hundred numbers in a session this is not negligible, and the gate is
therefore a floor and not a proof. It catches every instance the red-team found;
it cannot promise there is no instance it would miss. The claim ledger it emits
(§16) exists so a human can see which governed field each figure was matched to
and notice when the match is nonsense.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

#: A figure as a writer states it: optional currency, digits, optional unit.
FIGURE = re.compile(
    r"(?<![\w.])(£|\$|€)?\s?(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)"
    r"\s?(m|bn|k|%|pp|ppt|bps|percentage points?|basis points?)?(?![\w])", re.I)

#: A date is not a measurement.
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

#: Digits inside a field code or a loan id — ``RREC17/18/19``,
#: ``ORIGINATION-0043`` — name a thing, not a quantity.
CODE_TOKEN = re.compile(r"[A-Za-z]+[\d/–\-]*\d")

#: Standing terms whose digits are a bucket the industry defines: "90+",
#: "90 days", "3 months".
TERM_AFTER = re.compile(r"^\s*(\+|days?\b|months?\b|years?\b)")

#: Scalings a writer applies to ONE governed number. No entry combines two.
_SCALES: Tuple[float, ...] = (1.0, 1e3, 1e6, 1e9, 1e-2, 1e2)
_UNIT_SCALE = {"m": 1e6, "bn": 1e9, "k": 1e3, "bps": 1e-4}

#: Small integers a review may use to count its own sentences — "the top 3",
#: "two of the five". Not measurements, and requiring a tool call for them
#: would make the gate absurd rather than strict.
FREE_INTEGERS = frozenset(range(0, 13))

#: Narrative fields a reader sees, in the order they see them. The first two are
#: the card's face: an unsupported figure there is not survivable by dropping a
#: finding, because the headline IS the message.
SURFACE_FIELDS = ("headline", "summary")


# --------------------------------------------------------------------------- #
# What Trakt returned
# --------------------------------------------------------------------------- #
@dataclass
class GovernedIndex:
    """Every number a session's governed results contained, and where from.

    Keyed by value so a lookup is a comparison rather than a search, and each
    value carries its origins so the claim ledger can name the field a figure
    was matched to instead of asserting a bare "found".
    """

    #: value -> [(tool, dotted path)]
    origins: Dict[float, List[Tuple[str, str]]] = field(default_factory=dict)

    def absorb(self, tool: str, payload: Any) -> None:
        self._walk(tool, payload, "")

    def values(self) -> Iterable[float]:
        return self.origins.keys()

    def sources_for(self, value: float) -> List[Tuple[str, str]]:
        return self.origins.get(value, [])

    def _add(self, tool: str, path: str, value: float) -> None:
        self.origins.setdefault(float(value), []).append((tool, path))

    def _walk(self, tool: str, node: Any, path: str) -> None:
        if isinstance(node, bool) or node is None:
            return
        if isinstance(node, (int, float)):
            self._add(tool, path or "(root)", node)
        elif isinstance(node, dict):
            for key, value in node.items():
                self._walk(tool, value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, (list, tuple)):
            for i, value in enumerate(node):
                self._walk(tool, value, f"{path}[{i}]")
        elif isinstance(node, str):
            # Governed strings carry figures too — a formatted summary line, a
            # warning quoting a threshold. A number the agent could legitimately
            # read off a result must count as governed however it was rendered.
            for match in FIGURE.finditer(node):
                try:
                    self._add(tool, path or "(root)",
                              float(match.group(2).replace(",", "")))
                except ValueError:
                    pass


# --------------------------------------------------------------------------- #
# One stated figure
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Claim:
    """One figure a review states, and what it was matched against.

    Carries §7's required record: value, unit, the field it was stated in, the
    source tool and field, and whether the match is exact.
    """

    field_path: str
    stated: str
    value: float
    unit: str
    decimals: int
    grounded: bool
    source_tool: Optional[str] = None
    source_field: Optional[str] = None
    matched_value: Optional[float] = None
    excerpt: str = ""

    def to_row(self) -> Dict[str, Any]:
        return {
            "output_number": self.stated + (self.unit or ""),
            "in": self.field_path,
            "governed_source_tool": self.source_tool or "—",
            "source_field": self.source_field or "—",
            "source_value": self.matched_value,
            "exact_match": "yes" if self.grounded else "NO",
        }


def _match(value: float, decimals: int, unit: str, index: GovernedIndex
           ) -> Optional[Tuple[float, str, str]]:
    """The governed number a stated figure is a correct rendering of.

    Tolerance is half of the last stated decimal place at the stated magnitude,
    so "£11.97m" matches 11,974,544.28 and "£11.98m" does not. A figure written
    without decimals gets a half-unit tolerance at its own scale, which is what
    makes "£12m" a fair statement of that number and "£13m" not.

    **Truncation is accepted as well as rounding.** Writers routinely write
    "£954k" for 954,513.89, which rounds to 955k — strictly a misstatement, and
    one that would be rejected by a rounding-only window. Rejecting it would
    make the gate fire constantly on ordinary presentation, and a control that
    cries wolf is a control somebody turns off. Widening to truncation costs
    nothing in safety: both renderings come from ONE governed number, and no
    combination of two numbers becomes reachable by allowing it.
    """
    scales = (_UNIT_SCALE[unit],) if unit in _UNIT_SCALE else _SCALES
    best: Optional[Tuple[float, str, str]] = None
    for scale in scales:
        target = value * scale
        tol = 0.5 * (10 ** -decimals) * scale + 1e-9
        unit_step = 2 * tol
        low, high = target - tol, target + tol
        if target >= 0:
            high = target + unit_step      # truncated toward zero
        else:
            low = target - unit_step
        for candidate in index.values():
            if low <= candidate <= high:
                tool, path = index.sources_for(candidate)[0]
                # An exact hit beats a rounded one, so stop only at zero
                # distance and otherwise keep the first plausible source.
                if candidate == target:
                    return candidate, tool, path
                best = best or (candidate, tool, path)
    return best


def _narrative_fields(review: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Every field a reader sees, labelled."""
    out: List[Tuple[str, str]] = []
    for key in ("headline", "summary", "period_explained_by"):
        value = review.get(key)
        if isinstance(value, str) and value.strip():
            out.append((key, value))
    for i, finding in enumerate(review.get("findings") or ()):
        for key in ("title", "observation", "why_it_matters"):
            value = finding.get(key)
            if isinstance(value, str) and value.strip():
                out.append((f"findings[{i}].{key}", value))
    for i, gap in enumerate(review.get("could_not_assess") or ()):
        for key in ("check", "reason", "implication"):
            value = gap.get(key)
            if isinstance(value, str) and value.strip():
                out.append((f"could_not_assess[{i}].{key}", value))
    return out


def _excerpt(text: str, needle: str, width: int = 90) -> str:
    at = text.find(needle)
    if at < 0:
        return text[:width]
    start = max(0, at - width // 2)
    return ("..." if start else "") + text[start:at + width // 2].strip() + "..."


def claims_in(text: str, field_path: str, index: GovernedIndex) -> List[Claim]:
    """Every figure one narrative field states, each resolved against Trakt."""
    out: List[Claim] = []
    cleaned = DATE.sub(" ", text)
    for match in FIGURE.finditer(cleaned):
        _currency, digits, unit = match.groups()
        raw = digits.replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        before = cleaned[max(0, match.start() - 14):match.start()]
        after = cleaned[match.end():match.end() + 8]
        if CODE_TOKEN.search(before) and not before.endswith(" "):
            continue
        if TERM_AFTER.match(after):
            continue

        unit = (unit or "").lower()
        if unit.startswith(("percentage", "pp")) or unit == "ppt":
            unit = "%"
        elif unit.startswith("basis"):
            unit = "bps"
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if not unit and decimals == 0 and abs(value) in FREE_INTEGERS:
            continue

        hit = _match(value, decimals, unit, index)
        out.append(Claim(
            field_path=field_path, stated=raw, value=value, unit=unit,
            decimals=decimals, grounded=hit is not None,
            matched_value=hit[0] if hit else None,
            source_tool=hit[1] if hit else None,
            source_field=hit[2] if hit else None,
            excerpt=_excerpt(text, raw)))
    return out


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
#: What the gate did to a review.
PUBLISHABLE = "PUBLISHABLE"          # nothing was dropped
DEGRADED = "DEGRADED"                # a finding was dropped; the rest stands
BLOCKED = "BLOCKED"                  # the card's face was unsupported


@dataclass
class GateResult:
    """The reviewed narrative, what was removed, and the ledger for §16."""

    status: str
    review: Optional[Dict[str, Any]]
    claims: List[Claim] = field(default_factory=list)
    dropped_findings: List[Dict[str, Any]] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    @property
    def unsupported(self) -> List[Claim]:
        return [c for c in self.claims if not c.grounded]

    @property
    def publishable(self) -> bool:
        return self.status in (PUBLISHABLE, DEGRADED) and bool(self.review)

    def ledger(self) -> List[Dict[str, Any]]:
        """§16's audit table: every published number and its governed source."""
        published = self._published_fields()
        return [c.to_row() for c in self.claims if c.field_path in published]

    def _published_fields(self) -> Set[str]:
        if not self.review:
            return set()
        return {path for path, _ in _narrative_fields(self.review)}


def apply(review: Optional[Dict[str, Any]], index: GovernedIndex) -> GateResult:
    """Gate one review. Refuses; never repairs.

    * An unsupported figure in the **headline or summary** blocks the review.
      The face of the card is the message, and there is no honest way to publish
      a headline with a number nobody can source.
    * An unsupported figure in a **finding** drops that finding. The rest of the
      review is unaffected, because the findings are independent and one bad
      figure is not a reason to withhold four good ones.
    * A dropped finding is recorded, never silently removed.
    """
    if not review:
        return GateResult(status=BLOCKED, review=None,
                          reasons=["no review was submitted"])

    claims = [c for path, text in _narrative_fields(review)
              for c in claims_in(text, path, index)]
    bad = [c for c in claims if not c.grounded]
    if not bad:
        return GateResult(status=PUBLISHABLE, review=review, claims=claims)

    surface = [c for c in bad
               if c.field_path.split("[")[0].split(".")[0] in SURFACE_FIELDS]
    if surface:
        return GateResult(
            status=BLOCKED, review=None, claims=claims,
            reasons=[f"{c.field_path} states {c.stated}{c.unit}, which no "
                     f"governed result contains" for c in surface])

    drop = {int(c.field_path.split("[")[1].split("]")[0])
            for c in bad if c.field_path.startswith("findings[")}
    kept = [f for i, f in enumerate(review.get("findings") or ())
            if i not in drop]
    dropped = [{"finding": f, "reason": "; ".join(
                    f"{c.stated}{c.unit} is not a governed value"
                    for c in bad
                    if c.field_path == f"findings[{i}].title"
                    or c.field_path.startswith(f"findings[{i}]."))}
               for i, f in enumerate(review.get("findings") or ())
               if i in drop]

    # Every claim is kept, including the ones that caused a drop: the rejected
    # figure is the evidence that the gate fired, and a result that discarded it
    # would report a mysteriously shortened review. `ledger()` narrows to what
    # was actually published; `unsupported` is deliberately the full list.
    return GateResult(
        status=DEGRADED, review={**review, "findings": kept}, claims=claims,
        dropped_findings=dropped,
        reasons=[f"dropped {len(dropped)} finding(s) stating figures no "
                 f"governed result contains"])
