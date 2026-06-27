"""mi_agent/risk_monitor/schedule8_extractor.py

Deterministic Schedule 8 concentration-limit extraction.

Reads a securitisation "Schedule 8" (concentration limits / eligibility
criteria) document as TEXT and extracts STRUCTURED, governed risk limits:

  * limit category (geographic / broker / large-loan / LTV / WAC / borrower /
    age / property-value / other);
  * a numeric limit value + direction (max / min) + unit (percent / count / gbp);
  * the SOURCE SNIPPET (the sentence) and the SECTION heading it came from;
  * a confidence flag and a ``needs_review`` flag for ambiguous statements.

Design rules (governance):
  * Deterministic regex/keyword parsing only — no LLM, fully testable.
  * NEVER fabricate a limit: a sentence that names no numeric threshold (e.g.
    "should be monitored ... subject to policy") is recorded as ``needs_review``
    with ``limit_value: null`` rather than dropped or invented.
  * Limits whose category is recognised but whose mapping to a portfolio metric
    is uncertain are flagged ``needs_review`` so an analyst confirms them.

The output is a plain dict that serialises cleanly to YAML/JSON.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# A "section" is a numbered/blank-line-separated block headed by an ALL-CAPS or
# "N. Heading" line. Sentences inside inherit that heading.
_SECTION_HEADING_RE = re.compile(r"^\s*(?:\d+\.\s*)?([A-Z][A-Z /&-]{3,})\s*$")
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_GBP_RE = re.compile(r"(?:GBP|£)\s*([\d,]+(?:\.\d+)?)")
_COUNT_RE = re.compile(r"more than\s+(\d+)\s+(?:loans?|borrowers?)", re.IGNORECASE)
_TOPN_RE = re.compile(r"top\s+(\d+)\s+brokers?", re.IGNORECASE)

# Category detection keywords (checked in priority order). Each maps to a
# (category, suggested_dimension, basis) tuple used downstream by the monitor.
_CATEGORY_RULES: List[Tuple[str, str, Optional[str], str]] = [
    # (keyword regex, category, dimension hint, exposure basis)
    (r"\btop\s+\d+\s+brokers?\b", "broker_concentration", "broker_channel", "funded"),
    (r"\bbroker\b|\bintermediary\b", "broker_concentration", "broker_channel", "funded"),
    (r"\blondon\b|\bsouth east\b|\bscotland\b|\bregion\b|\bgeograph",
     "geographic_concentration", "geographic_region_obligor", "funded"),
    (r"\bsingle loan\b|any single loan|loan size|greater than .* 1,?000,?000",
     "large_loan_concentration", None, "funded"),
    (r"\bloan to value\b|\bltv\b", "ltv_limit", "current_loan_to_value", "funded"),
    (r"variable rate|interest rate|\bwac\b|weighted average .* rate",
     "interest_rate_limit", "current_interest_rate", "funded"),
    (r"joint borrower", "joint_borrower_limit", "borrower_structure", "funded"),
    (r"single borrower|connected borrower|per borrower|borrower\b",
     "borrower_concentration", None, "funded"),
    (r"aged over|age over|over 85|borrowers aged", "age_limit", "youngest_borrower_age",
     "funded"),
    (r"property|valuation", "property_value_concentration", "current_valuation_amount",
     "funded"),
]

_MAX_RE = re.compile(r"must not exceed|no .* greater than|may not|must be (?:no|not) (?:more|greater)|"
                     r"not exceed|maximum|no more than|no such loans", re.IGNORECASE)
_MIN_RE = re.compile(r"at least|must be (?:no|not) less than|minimum|must exceed", re.IGNORECASE)
# Hedge language → the statement is not a hard numeric limit.
_HEDGE_RE = re.compile(r"should be monitored|subject to|prevailing|where appropriate|"
                       r"as applicable|policy", re.IGNORECASE)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48]


def _split_sentences(block: str) -> List[str]:
    # Numbered clauses ("1.1 ...") and full stops both delimit statements.
    parts = re.split(r"(?<=[.])\s+(?=[A-Z0-9])", block.strip())
    return [p.strip() for p in parts if p.strip()]


def _detect_category(sentence: str) -> Tuple[str, Optional[str], str]:
    low = sentence.lower()
    for kw, cat, dim, basis in _CATEGORY_RULES:
        if re.search(kw, low):
            return cat, dim, basis
    return "other", None, "funded"


def _detect_direction(sentence: str) -> str:
    if _MIN_RE.search(sentence) and not _MAX_RE.search(sentence):
        return "min"
    return "max"  # concentration limits are overwhelmingly upper bounds


def _detect_value(sentence: str) -> Tuple[Optional[float], str]:
    """Return (value, unit). Percentages preferred, then counts, then GBP."""
    pm = _PERCENT_RE.search(sentence)
    if pm:
        return float(pm.group(1)), "percent"
    cm = _COUNT_RE.search(sentence)
    if cm:
        return float(cm.group(1)), "count"
    gm = _GBP_RE.search(sentence)
    if gm:
        return float(gm.group(1).replace(",", "")), "gbp"
    return None, "unknown"


def _region_for(sentence: str) -> Optional[str]:
    low = sentence.lower()
    for name in ("london", "south east", "scotland", "north east", "north west",
                 "yorkshire", "east midlands", "west midlands", "wales",
                 "northern ireland", "east anglia", "south west"):
        if name in low:
            return name.title()
    if "any other single region" in low or "single region" in low:
        return "Any single region"
    return None


def extract_schedule8_limits(text: str, *, source_name: str = "schedule_8") -> Dict[str, Any]:
    """Extract structured concentration limits from Schedule 8 ``text``."""
    lines = text.splitlines()
    current_section = "Unsectioned"
    blocks: List[Tuple[str, str]] = []  # (section, sentence)
    buf: List[str] = []

    def _flush(section: str):
        if buf:
            joined = " ".join(buf)
            for s in _split_sentences(joined):
                blocks.append((section, s))
            buf.clear()

    for raw in lines:
        line = raw.rstrip()
        heading = _SECTION_HEADING_RE.match(line)
        if heading and len(line.split()) <= 8:
            _flush(current_section)
            current_section = heading.group(1).title().strip()
            continue
        if not line.strip():
            _flush(current_section)
            continue
        buf.append(line.strip())
    _flush(current_section)

    limits: List[Dict[str, Any]] = []
    review_count = 0
    seen_ids: Dict[str, int] = {}
    for section, sentence in blocks:
        # Only statements that look like a rule (contain a limit verb or a number).
        has_rule_verb = bool(_MAX_RE.search(sentence) or _MIN_RE.search(sentence)
                             or _HEDGE_RE.search(sentence))
        value, unit = _detect_value(sentence)
        if not has_rule_verb and value is None:
            continue
        category, dim, basis = _detect_category(sentence)
        direction = _detect_direction(sentence)
        hedged = bool(_HEDGE_RE.search(sentence))
        needs_review = value is None or hedged or category == "other"
        confidence = "high"
        if needs_review:
            confidence = "low" if value is None else "medium"
        if needs_review:
            review_count += 1

        region = _region_for(sentence) if category == "geographic_concentration" else None
        base_id = _slug(f"{category}_{region or dim or section}")
        seen_ids[base_id] = seen_ids.get(base_id, 0) + 1
        limit_id = base_id if seen_ids[base_id] == 1 else f"{base_id}_{seen_ids[base_id]}"

        limits.append({
            "limit_id": limit_id,
            "category": category,
            "region": region,
            "dimension": dim,
            "metric": "exposure_pct" if unit == "percent" else (
                "count" if unit == "count" else "amount_gbp"),
            "direction": direction,
            "limit_value": value,
            "unit": unit,
            "exposure_basis": basis,
            "confidence": confidence,
            "needs_review": needs_review,
            "source_section": section,
            "source_snippet": sentence,
        })

    return {
        "portfolio_id": None,
        "source_document": source_name,
        "extraction_method": "deterministic",
        "limit_count": len(limits),
        "needs_review_count": review_count,
        "categories": sorted({l["category"] for l in limits}),
        "limits": limits,
    }


def extract_from_file(path: str | Path, *, portfolio_id: Optional[str] = None
                      ) -> Dict[str, Any]:
    """Extract from a Schedule 8 file. Returns a controlled UNAVAILABLE block when
    the file does not exist or is not machine-readable text — never a fabrication."""
    p = Path(path)
    if not p.exists():
        return {
            "portfolio_id": portfolio_id, "source_document": str(p),
            "extraction_method": "deterministic", "available": False,
            "status": "unavailable",
            "reason": "Schedule 8 document not found.",
            "limit_count": 0, "needs_review_count": 0, "categories": [], "limits": [],
        }
    try:
        text = p.read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, OSError):
        return {
            "portfolio_id": portfolio_id, "source_document": str(p),
            "extraction_method": "deterministic", "available": False,
            "status": "needs_review",
            "reason": ("Schedule 8 document is not machine-readable text "
                       "(e.g. scanned PDF / binary). Manual extraction required."),
            "limit_count": 0, "needs_review_count": 0, "categories": [], "limits": [],
        }
    out = extract_schedule8_limits(text, source_name=p.name)
    out["portfolio_id"] = portfolio_id
    out["available"] = True
    out["status"] = "needs_review" if out["needs_review_count"] else "ok"
    return out


def locate_schedule8(*search_roots: str | Path) -> Optional[Path]:
    """Find a Schedule 8 document under the given roots (and common fixtures)."""
    roots = list(search_roots) + [
        "tests/fixtures/client_001_mi_pack", "tests/fixtures", "portfolio_test",
        "config/clients/client_001",
    ]
    patterns = ("*schedule*8*", "*schedule_8*", "*Schedule 8*", "*concentration*")
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue
        for pat in patterns:
            hits = sorted(rp.glob(f"**/{pat}"))
            hits = [h for h in hits if h.suffix.lower() in (".txt", ".md", ".csv")]
            if hits:
                return hits[0]
    return None
