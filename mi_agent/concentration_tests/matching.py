"""mi_agent.concentration_tests.matching — extraction and library mapping.

Turns client-supplied covenant wording, limits tables or monitoring rows into
structured :class:`~mi_agent.concentration_tests.models.TestProposal` records
and maps each one onto the governed library:

* ``matched``     — a direct library metric with resolved parameters;
* ``composed``    — representable via approved primitives / list parameters
                    (London plus South East; balances above £1m; borrowers
                    with at least three loans);
* ``ambiguous``   — a plausible metric with an unresolved contractual
                    definition, carrying concern codes and short, specific
                    confirmation questions;
* ``unsupported`` — not safely representable: a structured implementation
                    request, never an improvised formula.

The default extractor is deterministic (regex + library aliases + a standard
UK region vocabulary) so the whole workflow runs offline and reproducibly. A
model may be plugged in through :class:`ExtractionAssist` to PROPOSE additional
candidate extractions; every candidate — model-proposed or not — flows through
the same validation, the same concern codes and the same human approval. The
model is never the authority.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

from .library import ConcentrationLibrary, MetricDefinition
from .models import (
    ConcernCode,
    MATCH_AMBIGUOUS,
    MATCH_COMPOSED,
    MATCH_MATCHED,
    MATCH_UNSUPPORTED,
    OPERATOR_MAX,
    OPERATOR_MIN,
    PROPOSAL_PENDING_APPROVAL,
    PROPOSAL_PENDING_CONFIRMATION,
    PROPOSAL_UNSUPPORTED,
    SourceEvidence,
    TestProposal,
)

# --------------------------------------------------------------------------- #
# Vocabularies
# --------------------------------------------------------------------------- #
#: Standard UK ITL1 region labels plus common contractual synonyms. This is a
#: national statistical vocabulary, not client configuration.
UK_REGIONS = (
    "london", "greater london", "south east", "south west", "east anglia",
    "east of england", "east midlands", "west midlands", "north east",
    "north west", "yorkshire and the humber", "yorkshire", "wales",
    "scotland", "northern ireland",
)

#: ITL1 (formerly NUTS1) statistical codes, as used in limits tables that name
#: the region by code as well as by label. Every value is a member of
#: ``UK_REGIONS`` so a code and a label collapse to one canonical region.
_ITL1_CODES = {
    "ukc": "north east", "ukd": "north west", "uke": "yorkshire and the humber",
    "ukf": "east midlands", "ukg": "west midlands", "ukh": "east of england",
    "uki": "london", "ukj": "south east", "ukk": "south west",
    "ukl": "wales", "ukm": "scotland", "ukn": "northern ireland",
}
_ITL1_CODE_RE = re.compile(r"\bUK([C-N])\b", re.I)

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|per\s*cent\.?|percent)", re.I)
_MONEY_RE = re.compile(
    r"(?:£|GBP|EUR|€|USD|\$)\s*([\d,]+(?:\.\d+)?)\s*(million|m\b|k\b|thousand|bn|billion)?",
    re.I)
_AGE_RE = re.compile(
    r"(?:aged?|age)\s+(?:is\s+)?(?:below|under|over|above)?\s*(\d{2})\b"
    r"|\b(?:below|under|over|above)\s+(?:the\s+age\s+of\s+)?(\d{2})\b", re.I)
_LOANS_COUNT_RE = re.compile(
    r"(?:more than|at least|exceed(?:ing)?|over)\s+(\w+)\s+loans", re.I)
_TOP_N_RE = re.compile(r"top\s+(\d+)", re.I)
_LTV_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%?\s*(?:ltv|loan[- ]to[- ]value)"
                     r"|(?:ltv|loan[- ]to[- ]value)\D{0,20}?(\d+(?:\.\d+)?)\s*%", re.I)
_DPD_RE = re.compile(r"(\d+)\s*(?:\+|or more)?\s*days", re.I)
_EFFECTIVE_RE = re.compile(
    r"(?:with effect from|effective(?: date)?(?: of| from)?)\s+(\d{4}-\d{2}-\d{2}|"
    r"\d{1,2}\s+\w+\s+\d{4})", re.I)

#: Modal verbs vary by drafter ("must", "shall", "will", "may") so the markers
#: below are written without them wherever the rest of the phrase is decisive.
_MAX_MARKERS = (
    "must not exceed", "may not exceed", "shall not exceed", "not to exceed",
    "not exceed",
    "no more than", "not more than", "less than or equal", "not be more",
    "capped at", "maximum of", "up to", "<=", "≤", "not be greater",
    "cannot exceed",
)
_MIN_MARKERS = (
    "at least", "no less than", "not less than", "not be less",
    "not fall below", "minimum of", ">=", "≥",
    "greater than or equal", "must remain above", "must exceed",
)
_ZERO_MARKERS = ("no such loans", "none permitted", "not permitted",
                 "must be zero", "= 0", "of 0%", "0% of")

_CLAUSE_SPLIT_RE = re.compile(r"\n+|(?<=\.)\s+(?=\d+\.\d+\s)")
_CLAUSE_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+")

#: Facility documents bracket a negotiated number until the document is
#: agreed — "[50]%", "£[1,500,000]", "[3.75]% per annum". The brackets are a
#: drafting convention, not arithmetic, so they are removed before any number
#: is read. Only digit-bearing brackets are touched: "[Reserved]", "[sic]" and
#: cross-references such as "[Schedule 8]" are left exactly as drafted.
_BRACKET_NUMBER_RE = re.compile(r"\[\s*([\d,]+(?:\.\d+)?)\s*\]")
_AMPERSAND_RE = re.compile(r"\s*&\s*")

#: A defined term introduced as: "Concentration Limit Denominator" means …
_DEFINED_TERM_RE = re.compile(
    r"[“”‘’\"']\s*([A-Za-z][A-Za-z0-9 ()/-]{2,79}?)\s*"
    r"[“”‘’\"']\s+means\b",
    re.I)
#: … an amount equal to the greater of: (a) £33,000,000 and (b) the Current
#: Balance of the portfolio. A floored denominator, not a test in its own right.
_GREATER_OF_RE = re.compile(r"\bgreater of\b", re.I)
_BALANCE_LIMB_RE = re.compile(r"\b(?:current|principal|outstanding)\s+balance\b",
                              re.I)

#: A sentence that introduces a limits table rather than stating a limit
#: itself: the comparison lives here, the numbers live in the rows below.
_TABLE_LEAD_IN_MARKERS = ("set out below", "set out in the table", "as follows",
                          "listed below", "specified below")


@dataclass
class ExtractedTest:
    """A raw extracted test before library mapping."""

    source_text: str = ""
    locator: str = ""
    name: str = ""
    threshold_percent: Optional[float] = None
    threshold_amount: Optional[float] = None
    threshold_count: Optional[float] = None
    operator: Optional[str] = None
    effective_date: str = ""
    extraction_confidence: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)
    #: The sentence that governs this row when the limit came from a table —
    #: it carries the comparison and the denominator the row omits. Read for
    #: interpretation and shown to the reviewer alongside the row.
    context_text: str = ""

    @property
    def interpretation_text(self) -> str:
        """Everything the mapper is entitled to read for this test."""
        return " ".join(p for p in (self.context_text, self.source_text) if p)

    @property
    def evidence_text(self) -> str:
        """What the reviewer is shown: the row never appears without the
        sentence that gives it its comparison."""
        return "\n".join(p for p in (self.context_text, self.source_text) if p)


@dataclass(frozen=True)
class DefinedDenominator:
    """A contractually defined denominator carrying a floor, e.g. the
    "Concentration Limit Denominator" — the greater of a fixed amount and the
    portfolio balance. Applies to every clause that references the term."""

    term: str
    floor_amount: float
    source_text: str


class ExtractionAssist(Protocol):
    """Optional model seam. Implementations may PROPOSE candidate extractions
    for text the deterministic pass could not read; every candidate is
    validated and mapped by the same deterministic code path."""

    def propose(self, text: str) -> List[Dict[str, Any]]: ...


# --------------------------------------------------------------------------- #
# Number helpers
# --------------------------------------------------------------------------- #
def _parse_money(text: str) -> Optional[float]:
    m = _MONEY_RE.search(text)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    scale = (m.group(2) or "").lower()
    if scale in ("million", "m"):
        value *= 1_000_000
    elif scale in ("k", "thousand"):
        value *= 1_000
    elif scale in ("bn", "billion"):
        value *= 1_000_000_000
    return value


def _parse_percent(text: str) -> Optional[float]:
    m = _PCT_RE.search(text)
    return float(m.group(1)) if m else None


def _parse_operator(text: str) -> Optional[str]:
    lowered = text.lower()
    if any(marker in lowered for marker in _ZERO_MARKERS):
        return OPERATOR_MAX
    for marker in _MIN_MARKERS:
        if marker in lowered:
            return OPERATOR_MIN
    for marker in _MAX_MARKERS:
        if marker in lowered:
            return OPERATOR_MAX
    return None


def _parse_count(text: str) -> Optional[float]:
    m = _LOANS_COUNT_RE.search(text)
    if not m:
        return None
    token = m.group(1).lower()
    if token.isdigit():
        return float(token)
    return float(_WORD_NUMBERS[token]) if token in _WORD_NUMBERS else None


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def normalise_source_text(text: str) -> str:
    """Remove drafting conventions that hide numbers from the reader.

    Strips the square brackets around negotiated figures and expands "&" to
    "and" so region labels match the statistical vocabulary. Nothing is
    reworded and no number is changed — the evidence a reviewer sees is still
    the client's own wording, only readable.
    """
    out = str(text or "")
    out = _BRACKET_NUMBER_RE.sub(r"\1", out)
    out = _AMPERSAND_RE.sub(" and ", out)
    return out


def _definition_match(clause: str) -> Optional[str]:
    m = _DEFINED_TERM_RE.search(clause)
    return m.group(1).strip() if m else None


def _is_definition_only(clause: str) -> bool:
    """A clause that defines a term and states no comparison is a definition,
    not a test. A clause that does both (a limit that defines its own basis
    inline) stays a test."""
    return _definition_match(clause) is not None and _parse_operator(clause) is None


def extract_definitions(text: str) -> Dict[str, DefinedDenominator]:
    """Find denominators defined as a floor — "the greater of £X and the
    Current Balance …" — keyed by lower-cased term.

    A floored denominator changes every percentage in the schedule, so it is
    read once from the definitions and applied wherever the term appears,
    rather than being re-guessed clause by clause.
    """
    found: Dict[str, DefinedDenominator] = {}
    for clause in split_clauses(normalise_source_text(text)):
        term = _definition_match(clause)
        if not term or not _GREATER_OF_RE.search(clause):
            continue
        if not _BALANCE_LIMB_RE.search(clause):
            continue
        amount = _parse_money(clause)
        if amount is None:
            continue
        found[term.lower()] = DefinedDenominator(
            term=term, floor_amount=amount, source_text=clause)
    return found


def _is_table_lead_in(clause: str) -> bool:
    """A sentence that states the comparison and defers the numbers to a table
    below it."""
    lowered = clause.lower()
    return clause.rstrip().endswith(":") or any(
        marker in lowered for marker in _TABLE_LEAD_IN_MARKERS)


def split_clauses(text: str) -> List[str]:
    clauses: List[str] = []
    for part in _CLAUSE_SPLIT_RE.split(str(text or "")):
        part = " ".join(part.split())
        if len(part) >= 8:
            clauses.append(part)
    return clauses


def _build_extraction(clause: str, *, operator: Optional[str],
                      pct: Optional[float], amount: Optional[float],
                      count: Optional[float], context: str = "",
                      confidence: float) -> ExtractedTest:
    num_m = _CLAUSE_NUM_RE.match(clause)
    eff = _EFFECTIVE_RE.search(clause)
    return ExtractedTest(
        source_text=clause,
        locator=f"clause {num_m.group(1)}" if num_m else "",
        name=clause[:80],
        threshold_percent=pct,
        threshold_amount=amount,
        threshold_count=count,
        operator=operator,
        effective_date=eff.group(1) if eff else "",
        extraction_confidence=confidence,
        context_text=context,
    )


def extract_from_text(text: str) -> List[ExtractedTest]:
    """Deterministically extract candidate tests from free text / covenant
    wording. A clause qualifies when it carries a comparison marker together
    with a number; everything else is left for the operator, not guessed.

    Two document structures are handled explicitly, because both are ordinary
    drafting rather than anything a reader would call ambiguous:

    * a **definition** ("X" means …) states no limit and is not proposed as a
      test — but a floored denominator it defines is collected separately;
    * a **limits table** puts the comparison in a lead-in sentence and the
      numbers in the rows beneath it, so each row inherits that sentence. The
      lead-in is only consumed when rows actually follow it; otherwise it is
      emitted on its own so nothing is silently dropped.
    """
    out: List[ExtractedTest] = []
    lead_in: Optional[str] = None
    lead_in_operator: Optional[str] = None
    lead_in_rows = 0

    def _flush_lead_in() -> None:
        nonlocal lead_in, lead_in_operator, lead_in_rows
        if lead_in is not None and lead_in_rows == 0:
            out.append(_build_extraction(
                lead_in, operator=lead_in_operator, pct=None, amount=None,
                count=None, confidence=0.6))
        lead_in, lead_in_operator, lead_in_rows = None, None, 0

    for clause in split_clauses(normalise_source_text(text)):
        if _is_definition_only(clause):
            continue
        operator = _parse_operator(clause)
        pct = _parse_percent(clause)
        amount = _parse_money(clause)
        count = _parse_count(clause)
        lowered = clause.lower()
        if operator is None and any(m in lowered for m in _ZERO_MARKERS):
            operator = OPERATOR_MAX
        if pct is None and any(m in lowered for m in _ZERO_MARKERS):
            pct = 0.0

        has_number = pct is not None or amount is not None
        if operator is not None and not has_number and _is_table_lead_in(clause):
            _flush_lead_in()
            lead_in, lead_in_operator, lead_in_rows = clause, operator, 0
            continue
        if operator is not None:
            # A clause that states its own comparison ends the table above it.
            _flush_lead_in()
        elif lead_in is not None and has_number:
            lead_in_rows += 1
            row = _build_extraction(
                clause, operator=lead_in_operator, pct=pct, amount=amount,
                count=count, context=lead_in, confidence=0.85)
            if not row.locator:
                # A table row is located by the clause that introduces it.
                lead_num = _CLAUSE_NUM_RE.match(lead_in)
                if lead_num:
                    row.locator = f"clause {lead_num.group(1)}"
            out.append(row)
            continue

        if operator is None and not has_number:
            continue
        confidence = 0.9 if (operator and (has_number or count is not None)) else 0.6
        out.append(_build_extraction(
            clause, operator=operator, pct=pct, amount=amount, count=count,
            confidence=confidence))
    _flush_lead_in()
    return out


_ROW_NAME_KEYS = ("test", "test_name", "name", "covenant", "limit", "description")
_ROW_THRESHOLD_KEYS = ("threshold", "limit_value", "value", "max", "cap")
_ROW_OPERATOR_KEYS = ("operator", "direction", "comparison")
_ROW_UNIT_KEYS = ("unit", "units")
_ROW_DEFINITION_KEYS = ("definition", "calculation", "calculation_definition",
                        "numerator", "denominator", "basis", "notes")


def extract_from_rows(rows: Sequence[Dict[str, Any]]) -> List[ExtractedTest]:
    """Extract from a structured limits table (spreadsheet rows / completed
    limits template). A malformed row becomes a low-confidence extraction the
    mapping stage marks up with concerns — it is never dropped silently."""
    out: List[ExtractedTest] = []
    for i, row in enumerate(rows or []):
        if not isinstance(row, dict):
            out.append(ExtractedTest(
                source_text=str(row), locator=f"row {i + 1}",
                name=str(row)[:80], extraction_confidence=0.2))
            continue
        norm = {str(k).strip().lower().replace(" ", "_"): v
                for k, v in row.items()}
        name = next((str(norm[k]) for k in _ROW_NAME_KEYS
                     if norm.get(k) not in (None, "")), "")
        threshold_raw = next((norm[k] for k in _ROW_THRESHOLD_KEYS
                              if norm.get(k) not in (None, "")), None)
        op_raw = str(next((norm[k] for k in _ROW_OPERATOR_KEYS
                           if norm.get(k) not in (None, "")), "")).lower()
        unit = str(next((norm[k] for k in _ROW_UNIT_KEYS
                         if norm.get(k) not in (None, "")), "")).lower()
        definition = " ".join(
            str(norm[k]) for k in _ROW_DEFINITION_KEYS
            if norm.get(k) not in (None, ""))
        text = normalise_source_text(" — ".join(p for p in (name, definition) if p))

        pct = amount = None
        if threshold_raw is not None:
            token = normalise_source_text(str(threshold_raw)).strip()
            if token.endswith("%") or unit in ("percent", "%", "pct"):
                pct = _parse_percent(token) or _parse_percent(token + "%")
                if pct is None:
                    try:
                        pct = float(token.rstrip("%").replace(",", ""))
                    except ValueError:
                        pct = None
            else:
                amount = _parse_money(token)
                if amount is None:
                    try:
                        amount = float(token.replace(",", ""))
                    except ValueError:
                        amount = None
                if unit in ("percent", "pct") and amount is not None:
                    pct, amount = amount, None
        operator = None
        if op_raw:
            if any(t in op_raw for t in ("max", "<=", "≤", "not exceed", "cap")):
                operator = OPERATOR_MAX
            elif any(t in op_raw for t in ("min", ">=", "≥", "at least", "floor")):
                operator = OPERATOR_MIN
        if operator is None:
            operator = _parse_operator(text)
        confidence = 0.9 if (name and operator and (pct is not None or amount is not None)) else 0.5
        out.append(ExtractedTest(
            source_text=text or str(row), locator=f"row {i + 1}", name=name or f"row {i + 1}",
            threshold_percent=pct, threshold_amount=amount,
            operator=operator, extraction_confidence=confidence,
            raw=dict(norm)))
    return out


# --------------------------------------------------------------------------- #
# Follow-up questions (short, specific, only for unresolved items)
# --------------------------------------------------------------------------- #
QUESTION_FOR_CONCERN = {
    ConcernCode.DENOMINATOR_UNDEFINED: (
        "Should the denominator be current principal, current balance "
        "including accrued interest, or borrowing-base eligible balance?"),
    ConcernCode.BALANCE_BASIS_UNDEFINED: (
        "Should this test read the original (initial) balance or the current "
        "balance?"),
    ConcernCode.NET_WAC_DEFINITION_UNCERTAIN: (
        "Does 'Net WAC' mean customer coupon net of servicing fee only, or "
        "net of servicing, hedging and funding costs? Please give the "
        "deduction in percentage points."),
    ConcernCode.JOINT_DEFINITION_UNCERTAIN: (
        "Should the joint-borrower test use the joint/single classification "
        "on each loan, or the number of borrowers? And is the limit measured "
        "on number of loans or current balance?"),
    ConcernCode.BORROWER_DEFINITION_UNCERTAIN: (
        "Is the age test on the youngest or the oldest borrower on each "
        "loan?"),
    ConcernCode.INDEX_SERIES_MISSING: (
        "Which house-price index source and series govern this test?"),
    ConcernCode.INDEX_BASE_DATE_MISSING: (
        "Which reference date does the index ratio measure against?"),
    ConcernCode.REGION_MAPPING_UNCERTAIN: (
        "Which region classification does this test use (e.g. ITL1 regions), "
        "and how should combined regions be read?"),
    ConcernCode.THRESHOLD_UNIT_UNCERTAIN: (
        "Is the threshold a percentage of the portfolio or a monetary "
        "amount?"),
    ConcernCode.COMPARISON_OPERATOR_UNCERTAIN: (
        "Is this a maximum (must not exceed) or a minimum (must not fall "
        "below)?"),
    ConcernCode.WEIGHTING_BASIS_UNDEFINED: (
        "Is the average weighted by current balance, original balance, or "
        "unweighted?"),
}


def _question(codes: Iterable[str]) -> List[str]:
    seen = []
    for code in codes:
        q = QUESTION_FOR_CONCERN.get(code)
        if q and q not in seen:
            seen.append(q)
    return seen


# --------------------------------------------------------------------------- #
# Mapping
# --------------------------------------------------------------------------- #
def _regions_in(text: str) -> List[str]:
    """Regions named in the text, by label or by ITL1 code. A row that gives
    both ("UKI + UKJ | London and South East") yields one entry per region,
    not one per spelling."""
    lowered = " ".join(str(text).lower().split())
    found: List[str] = []

    def _add(region: str) -> None:
        # "london" inside "greater london" — keep the most specific only.
        if any(region != other and region in other for other in found):
            return
        found[:] = [f for f in found if not (f != region and f in region)]
        if region not in found:
            found.append(region)

    for region in UK_REGIONS:
        if region in lowered:
            _add(region)
    for m in _ITL1_CODE_RE.finditer(lowered):
        region = _ITL1_CODES.get("uk" + m.group(1).lower())
        if region:
            _add(region)
    return [r.title() for r in found]


@dataclass
class _MappingContext:
    lib: ConcentrationLibrary
    client_id: str
    portfolio_id: str
    onboarding_run_id: str
    source_reference: str
    #: Floored denominators defined elsewhere in the same document, keyed by
    #: lower-cased term.
    definitions: Dict[str, DefinedDenominator] = field(default_factory=dict)

    def denominator_for(self, text: str) -> Optional[DefinedDenominator]:
        lowered = " ".join(str(text).lower().split())
        for term, definition in self.definitions.items():
            if term in lowered:
                return definition
        return None


def map_extracted_test(ext: ExtractedTest, ctx: _MappingContext) -> TestProposal:
    """Map one extracted test onto the library. Deterministic and total: every
    input yields a proposal; anything unclear degrades to ambiguous or
    unsupported with named concerns."""
    text = ext.interpretation_text
    lowered = text.lower()
    concerns: List[str] = []
    params: Dict[str, Any] = {}
    outcome = MATCH_UNSUPPORTED
    metric: Optional[MetricDefinition] = None
    unit = ""
    threshold: Optional[float] = None

    hits = ctx.lib.find_by_alias(text)
    if hits:
        metric = hits[0][0]

    regions = _regions_in(text)
    is_regional = bool(regions) and ("region" in lowered or "located in" in lowered
                                     or any(r.lower() in lowered for r in regions))

    # --- family-specific parameterisation ---------------------------------- #
    if metric is None and is_regional:
        metric = ctx.lib.get("geo_region_share")
    if metric is not None:
        mid = metric.metric_id
        if mid == "geo_region_share":
            if regions:
                params["regions"] = regions
                outcome = MATCH_COMPOSED if len(regions) > 1 else MATCH_MATCHED
            else:
                concerns.append(ConcernCode.REGION_MAPPING_UNCERTAIN)
                outcome = MATCH_AMBIGUOUS
            threshold, unit = ext.threshold_percent, "percent"
        elif mid == "geo_largest_region_share":
            threshold, unit = ext.threshold_percent, "percent"
            outcome = MATCH_MATCHED
        elif mid in ("property_value_below_share", "property_value_above_share"):
            amount = ext.threshold_amount
            comparison = "below" if mid.endswith("below_share") else "above"
            if amount is None:
                concerns.append(ConcernCode.THRESHOLD_UNIT_UNCERTAIN)
                outcome = MATCH_AMBIGUOUS
            else:
                params.update({"amount": amount, "comparison": comparison})
                outcome = MATCH_COMPOSED
            if "original" in lowered:
                params["value_basis"] = "original"
            elif "indexed" in lowered:
                params["value_basis"] = "indexed"
            elif "current" in lowered:
                params["value_basis"] = "current"
            else:
                concerns.append(ConcernCode.BALANCE_BASIS_UNDEFINED)
                outcome = MATCH_AMBIGUOUS
            threshold, unit = ext.threshold_percent, "percent"
        elif mid == "balance_above_share":
            amount = ext.threshold_amount
            if amount is None:
                concerns.append(ConcernCode.THRESHOLD_UNIT_UNCERTAIN)
                outcome = MATCH_AMBIGUOUS
            else:
                params["amount"] = amount
                outcome = MATCH_COMPOSED
            if "initial" in lowered or "original" in lowered:
                params["balance_basis"] = "original"
            elif "current" in lowered:
                params["balance_basis"] = "current"
            else:
                concerns.append(ConcernCode.BALANCE_BASIS_UNDEFINED)
                outcome = MATCH_AMBIGUOUS
            threshold, unit = ext.threshold_percent, "percent"
        elif mid == "balance_average":
            if "initial" in lowered or "original" in lowered:
                params["balance_basis"] = "original"
            elif "current" in lowered:
                params["balance_basis"] = "current"
            else:
                concerns.append(ConcernCode.BALANCE_BASIS_UNDEFINED)
            threshold, unit = ext.threshold_amount, "currency"
            outcome = MATCH_MATCHED
        elif mid == "borrower_age_share":
            m = _AGE_RE.search(text)
            age = float(m.group(1) or m.group(2)) if m else None
            if age is None:
                concerns.append(ConcernCode.THRESHOLD_UNIT_UNCERTAIN)
                outcome = MATCH_AMBIGUOUS
            else:
                params["age"] = age
            if any(t in lowered for t in ("below", "under", "younger")):
                params["comparison"] = "below"
            elif any(t in lowered for t in ("above", "over", "older")):
                params["comparison"] = "above"
            else:
                concerns.append(ConcernCode.COMPARISON_OPERATOR_UNCERTAIN)
            if "youngest" in lowered:
                params["age_basis"] = "youngest"
            elif "oldest" in lowered:
                params["age_basis"] = "oldest"
            else:
                concerns.append(ConcernCode.BORROWER_DEFINITION_UNCERTAIN)
            outcome = outcome if outcome == MATCH_AMBIGUOUS else (
                MATCH_AMBIGUOUS if concerns else MATCH_COMPOSED)
            threshold, unit = ext.threshold_percent, "percent"
        elif mid == "rate_net_wac":
            concerns.append(ConcernCode.NET_WAC_DEFINITION_UNCERTAIN)
            outcome = MATCH_AMBIGUOUS
            threshold, unit = ext.threshold_percent, "percent"
        elif mid == "rate_gross_wac":
            threshold, unit = ext.threshold_percent, "percent"
            outcome = MATCH_MATCHED
        elif mid == "rate_type_share":
            values = []
            if "variable" in lowered or "floating" in lowered:
                values = ["variable", "floating"]
            elif "fixed" in lowered:
                values = ["fixed"]
            if values:
                params["values"] = values
                outcome = MATCH_MATCHED
            else:
                concerns.append(ConcernCode.AGGREGATION_BASIS_UNDEFINED)
                outcome = MATCH_AMBIGUOUS
            threshold, unit = ext.threshold_percent, "percent"
        elif mid == "borrower_joint_share":
            # "loans which have two Borrowers" states the basis outright — the
            # covenant counts borrowers, so there is nothing to ask. Wording
            # that names the CONCEPT ("joint borrowers") still asks: it says
            # nothing about whether the book encodes jointness as a
            # classification flag or as a borrower count, and the two can
            # disagree on the same loan.
            if any(t in lowered for t in ("two borrowers", "2 borrowers",
                                          "more than one borrower",
                                          "number of borrowers",
                                          "multiple borrowers")):
                params["joint_basis"] = "borrower_count"
                outcome = MATCH_COMPOSED
            else:
                concerns.append(ConcernCode.JOINT_DEFINITION_UNCERTAIN)
                outcome = MATCH_AMBIGUOUS
            if any(t in lowered for t in ("single borrowers", "sole borrowers")):
                params["invert"] = True
            threshold, unit = ext.threshold_percent, "percent"
        elif mid == "borrower_aggregate_balance_share":
            amount = ext.threshold_amount
            if amount is None:
                concerns.append(ConcernCode.THRESHOLD_UNIT_UNCERTAIN)
                outcome = MATCH_AMBIGUOUS
            else:
                params["amount"] = amount
                outcome = MATCH_COMPOSED
            # Which balance the per-borrower aggregate is struck on. "initial"
            # and "original" are the same thing in facility drafting.
            if "initial" in lowered or "original" in lowered:
                params["aggregate_basis"] = "original"
            elif "current" in lowered:
                params["aggregate_basis"] = "current"
            else:
                concerns.append(ConcernCode.BALANCE_BASIS_UNDEFINED)
                outcome = MATCH_AMBIGUOUS
            if any(t in lowered for t in ("less than", "below", "under")):
                params["comparison"] = "below"
            else:
                params["comparison"] = "above"
            threshold, unit = ext.threshold_percent, "percent"
        elif mid == "borrower_multi_loan_share":
            count = ext.threshold_count or _parse_count(text)
            if count:
                params["min_loans"] = int(count)
                outcome = MATCH_COMPOSED
            else:
                concerns.append(ConcernCode.BORROWER_DEFINITION_UNCERTAIN)
                outcome = MATCH_AMBIGUOUS
            threshold, unit = ext.threshold_percent, "percent"
        elif mid == "borrower_max_loans":
            threshold = ext.threshold_count or ext.threshold_percent
            unit = "count"
            outcome = MATCH_MATCHED
        elif mid == "top_n_loans_share":
            m = _TOP_N_RE.search(text)
            params["n"] = int(m.group(1)) if m else 1
            threshold, unit = ext.threshold_percent, "percent"
            outcome = MATCH_MATCHED
        elif mid == "ltv_above_share":
            m = _LTV_RE.search(text)
            band = float(m.group(1) or m.group(2)) if m else None
            if band is None:
                concerns.append(ConcernCode.THRESHOLD_UNIT_UNCERTAIN)
                outcome = MATCH_AMBIGUOUS
            else:
                params["ltv_percent"] = band
                outcome = MATCH_COMPOSED
            if "original" in lowered:
                params["ltv_basis"] = "original"
            elif "indexed" in lowered:
                params["ltv_basis"] = "indexed"
            elif "current" in lowered:
                params["ltv_basis"] = "current"
            threshold, unit = ext.threshold_percent, "percent"
        elif mid == "ltv_weighted_average":
            if "original" in lowered:
                params["ltv_basis"] = "original"
            elif "indexed" in lowered:
                params["ltv_basis"] = "indexed"
            elif "current" in lowered:
                params["ltv_basis"] = "current"
            threshold, unit = ext.threshold_percent, "percent"
            outcome = MATCH_MATCHED
        elif mid == "index_hpi_ratio":
            concerns.extend([ConcernCode.INDEX_SERIES_MISSING,
                             ConcernCode.INDEX_BASE_DATE_MISSING])
            outcome = MATCH_AMBIGUOUS
            threshold, unit = ext.threshold_percent, "percent"
        elif mid == "perf_arrears_share":
            m = _DPD_RE.search(text)
            if m:
                params["min_days"] = float(m.group(1))
            threshold, unit = ext.threshold_percent, "percent"
            outcome = MATCH_MATCHED
        elif metric.implementation_status == "not_implemented":
            outcome = MATCH_UNSUPPORTED
        else:
            threshold = (ext.threshold_percent if metric.unit == "percent"
                         else ext.threshold_amount)
            unit = metric.unit
            outcome = MATCH_MATCHED

    # --- contractual denominator -------------------------------------------- #
    # A term such as "Concentration Limit Denominator" — the greater of a fixed
    # amount and the portfolio balance — is read once from the definitions and
    # applied to every clause that references it. A metric that cannot carry
    # the floor says so rather than quietly measuring against the wrong base.
    definition = ctx.denominator_for(text)
    if definition is not None and metric is not None:
        if "denominator_floor" in metric.parameters:
            params["denominator_floor"] = definition.floor_amount
            spec = metric.parameters.get("denominator") or {}
            if "current_balance" in (spec.get("values") or []):
                params["denominator"] = "current_balance"
        elif ConcernCode.DENOMINATOR_UNDEFINED not in concerns:
            concerns.append(ConcernCode.DENOMINATOR_UNDEFINED)

    # --- shared checks ------------------------------------------------------ #
    operator = ext.operator or ""
    if metric is not None and not operator:
        concerns.append(ConcernCode.COMPARISON_OPERATOR_UNCERTAIN)
    if metric is not None and threshold is None:
        # A percent metric quoted a money amount (or vice versa) or no number.
        if metric.unit == "percent" and ext.threshold_amount is not None:
            concerns.append(ConcernCode.THRESHOLD_UNIT_UNCERTAIN)
        elif metric.unit in ("currency",) and ext.threshold_percent is not None:
            threshold, unit = ext.threshold_percent, "percent"
            concerns.append(ConcernCode.THRESHOLD_UNIT_UNCERTAIN)
        elif ext.threshold_count is not None and metric.unit == "count":
            threshold, unit = ext.threshold_count, "count"
    if metric is not None and metric.implementation_status == "not_implemented":
        outcome = MATCH_UNSUPPORTED
        concerns.append(ConcernCode.UNSUPPORTED_METRIC)
    if not ext.source_text.strip():
        concerns.append(ConcernCode.SOURCE_EVIDENCE_MISSING)
    if metric is None:
        outcome = MATCH_UNSUPPORTED
        concerns.append(ConcernCode.UNSUPPORTED_METRIC)

    if metric is not None and outcome in (MATCH_MATCHED, MATCH_COMPOSED):
        problems = ctx.lib.validate_parameters(metric.metric_id, params)
        required_missing = [p for p in problems if "is missing" in p]
        if required_missing:
            outcome = MATCH_AMBIGUOUS
            if ConcernCode.AGGREGATION_BASIS_UNDEFINED not in concerns:
                concerns.append(ConcernCode.AGGREGATION_BASIS_UNDEFINED)
    if concerns and outcome in (MATCH_MATCHED, MATCH_COMPOSED):
        outcome = MATCH_AMBIGUOUS

    if outcome == MATCH_UNSUPPORTED:
        status = PROPOSAL_UNSUPPORTED
    elif outcome == MATCH_AMBIGUOUS or concerns:
        status = PROPOSAL_PENDING_CONFIRMATION
    else:
        status = PROPOSAL_PENDING_APPROVAL

    mapping_confidence = {MATCH_MATCHED: 0.9, MATCH_COMPOSED: 0.85,
                          MATCH_AMBIGUOUS: 0.5, MATCH_UNSUPPORTED: 0.0}[outcome]

    proposal = TestProposal(
        onboarding_run_id=ctx.onboarding_run_id,
        client_id=ctx.client_id,
        portfolio_id=ctx.portfolio_id,
        evidence=SourceEvidence(source_reference=ctx.source_reference,
                                source_text=ext.evidence_text,
                                locator=ext.locator),
        extracted_test_name=ext.name,
        extracted_category=metric.category if metric else "",
        proposed_metric_id=metric.metric_id if metric else "",
        match_outcome=outcome,
        extracted_operator=operator,
        extracted_threshold=threshold,
        extracted_unit=unit or (metric.unit if metric else ""),
        parameters=params,
        effective_date=ext.effective_date,
        extraction_confidence=ext.extraction_confidence,
        mapping_confidence=mapping_confidence,
        concern_codes=list(dict.fromkeys(concerns)),
        concern_messages=[QUESTION_FOR_CONCERN.get(c, c.replace("_", " "))
                          for c in dict.fromkeys(concerns)],
        confirmation_questions=_question(dict.fromkeys(concerns)),
        status=status,
    )
    if outcome == MATCH_UNSUPPORTED:
        proposal.implementation_request = {
            "requested_capability": ext.name,
            "source_text": ext.source_text,
            "reason": ("No governed library metric or approved composition "
                       "represents this test."),
            "suggested_category": metric.category if metric else "unknown",
        }
    return proposal


def _flag_duplicates(proposals: List[TestProposal]) -> None:
    """Same metric + same parameters twice → duplicated_test on the later one;
    same metric + parameters with a different threshold → conflicting."""
    seen: Dict[str, TestProposal] = {}
    for p in proposals:
        if not p.proposed_metric_id:
            continue
        key = p.proposed_metric_id + "|" + repr(sorted(p.parameters.items()))
        earlier = seen.get(key)
        if earlier is None:
            seen[key] = p
            continue
        if earlier.extracted_threshold == p.extracted_threshold and \
                earlier.extracted_operator == p.extracted_operator:
            code = ConcernCode.DUPLICATED_TEST
        else:
            code = ConcernCode.CONFLICTING_THRESHOLDS
        for target in (p, earlier) if code == ConcernCode.CONFLICTING_THRESHOLDS else (p,):
            if code not in target.concern_codes:
                target.concern_codes.append(code)
                target.concern_messages.append(code.replace("_", " "))
                if target.status == PROPOSAL_PENDING_APPROVAL:
                    target.status = PROPOSAL_PENDING_CONFIRMATION


def build_proposals(*, lib: ConcentrationLibrary, client_id: str,
                    portfolio_id: str = "", onboarding_run_id: str = "",
                    source_reference: str = "", text: str = "",
                    rows: Optional[Sequence[Dict[str, Any]]] = None,
                    assist: Optional[ExtractionAssist] = None
                    ) -> List[TestProposal]:
    """The governed extraction + mapping workflow.

    ``text`` and/or ``rows`` are extracted deterministically; an optional
    ``assist`` may add candidates, which are normalised into the SAME
    :class:`ExtractedTest` shape and mapped by the same code. Returns the full
    proposal list including duplicates/conflicts flagged across the batch.
    """
    ctx = _MappingContext(lib=lib, client_id=client_id,
                          portfolio_id=portfolio_id,
                          onboarding_run_id=onboarding_run_id,
                          source_reference=source_reference,
                          definitions=extract_definitions(text) if text else {})
    extracted: List[ExtractedTest] = []
    if text:
        extracted.extend(extract_from_text(text))
    if rows:
        extracted.extend(extract_from_rows(rows))
    if assist is not None and text:
        covered = {e.source_text for e in extracted}
        for raw in assist.propose(text) or []:
            if not isinstance(raw, dict):
                continue
            candidate = ExtractedTest(
                source_text=str(raw.get("source_text") or "").strip()[:2000],
                locator=str(raw.get("locator") or "model_proposed"),
                name=str(raw.get("name") or "")[:80],
                threshold_percent=raw.get("threshold_percent"),
                threshold_amount=raw.get("threshold_amount"),
                operator=(raw.get("operator")
                          if raw.get("operator") in (OPERATOR_MAX, OPERATOR_MIN)
                          else None),
                extraction_confidence=min(
                    max(float(raw.get("confidence") or 0.0), 0.0), 1.0),
            )
            if candidate.source_text and candidate.source_text not in covered:
                extracted.append(candidate)
    proposals = [map_extracted_test(e, ctx) for e in extracted]
    _flag_duplicates(proposals)
    return proposals
