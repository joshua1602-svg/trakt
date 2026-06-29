"""mi_agent.portfolio_lens — source-portfolio lens resolver for the MI Agent.

Maps natural-language portfolio scope ("the acquired book", "direct only",
"acquired_001") onto deterministic filters over the source-provenance fields
stamped at onboarding (``source_portfolio_type`` / ``source_portfolio_id`` —
see engine/provenance.py). The MI Agent answers through three lenses:

  * total     — all rows (direct + acquired), no source filter;
  * direct    — source_portfolio_type == direct;
  * acquired  — source_portfolio_type == acquired;
  * cohort    — source_portfolio_id == <exact id> (e.g. acquired_001).

This module is pure: it resolves a lens from text and applies it to an
:class:`~mi_agent.mi_query_spec.MIQuerySpec` (merging filters + recording the
lens label). It never touches data, so it is trivially unit-testable and works
regardless of which MI entrypoint builds the spec.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SOURCE_TYPE_FIELD = "source_portfolio_type"
SOURCE_ID_FIELD = "source_portfolio_id"

LENS_TOTAL = "total"
LENS_DIRECT = "direct"
LENS_ACQUIRED = "acquired"
LENS_COHORT = "cohort"

# An exact source-cohort id, e.g. direct_001 / acquired_002.
_COHORT_ID_RE = re.compile(r"\b((?:direct|acquired)_\d+)\b", re.IGNORECASE)

# Phrase → lens. Order matters only within a family; matching is keyword-based.
_DIRECT_TERMS = (
    "direct", "directly originated", "originated", "origination",
    "organic", "current book", "own book", "in-house", "new origination",
    "new lending", "newly originated",
)
_ACQUIRED_TERMS = (
    "acquired", "acquisition", "back book", "backbook", "purchased book",
    "purchased", "bought book", "inorganic", "legacy book", "m&a",
)
_TOTAL_TERMS = (
    "total", "whole book", "whole portfolio", "all loans", "all portfolios",
    "entire portfolio", "entire book", "combined", "overall", "group",
    "across the book", "across all",
)

_COMPARISON_TERMS = (" vs ", " vs. ", " versus ", "compare", "comparison",
                     "side by side", "side-by-side", "against")


@dataclass
class PortfolioLens:
    """A resolved portfolio lens: a label + the filters that realise it."""

    name: str
    label: str
    filters: Dict[str, Any] = field(default_factory=dict)
    cohort_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "filters": dict(self.filters),
            "cohort_id": self.cohort_id,
        }


def total_lens() -> PortfolioLens:
    return PortfolioLens(LENS_TOTAL, "Total", {})


def _cohort_lens(cohort_id: str) -> PortfolioLens:
    cid = cohort_id.strip().lower()
    return PortfolioLens(LENS_COHORT, cid, {SOURCE_ID_FIELD: cid}, cohort_id=cid)


def _type_lens(ptype: str) -> PortfolioLens:
    label = "Direct" if ptype == LENS_DIRECT else "Acquired"
    return PortfolioLens(ptype, label, {SOURCE_TYPE_FIELD: ptype})


def _contains_any(text: str, terms) -> bool:
    return any(t in text for t in terms)


def resolve_lens(text: Optional[str]) -> PortfolioLens:
    """Resolve a single portfolio lens from free-text. Defaults to *total*.

    Precedence: explicit cohort id > acquired/direct keyword > total. An
    explicit "total/whole book" phrase forces *total* even if other words
    appear.
    """
    if not text:
        return total_lens()
    low = " " + str(text).strip().lower() + " "

    # Exact cohort id always wins (most specific).
    m = _COHORT_ID_RE.search(low)
    if m:
        return _cohort_lens(m.group(1))

    has_direct = _contains_any(low, _DIRECT_TERMS)
    has_acquired = _contains_any(low, _ACQUIRED_TERMS)

    # If both families are mentioned (e.g. "direct vs acquired") this is a
    # comparison, not a single lens — fall back to total for the single-lens
    # view; callers wanting both should use resolve_comparison_lenses().
    if has_direct and has_acquired:
        return total_lens()
    if has_acquired:
        return _type_lens(LENS_ACQUIRED)
    if has_direct:
        return _type_lens(LENS_DIRECT)
    # Explicit total phrasing or nothing recognised → whole book.
    return total_lens()


def is_comparison(text: Optional[str]) -> bool:
    if not text:
        return False
    return _contains_any(" " + str(text).lower() + " ", _COMPARISON_TERMS)


def resolve_comparison_lenses(text: Optional[str]) -> List[PortfolioLens]:
    """Resolve a side-by-side comparison into 2+ lenses, else ``[]``.

    Handles the common securitisation cuts:
      * direct vs acquired                  → [Direct, Acquired]
      * direct_001 vs acquired_001          → [direct_001, acquired_001]
      * acquired_001 vs acquired_002        → [acquired_001, acquired_002]
    """
    if not text:
        return []
    low = " " + str(text).strip().lower() + " "

    ids = [c.lower() for c in _COHORT_ID_RE.findall(low)]
    # De-duplicate while preserving order.
    seen: set = set()
    uniq_ids = [c for c in ids if not (c in seen or seen.add(c))]
    if len(uniq_ids) >= 2:
        return [_cohort_lens(c) for c in uniq_ids]

    if not is_comparison(low):
        return []

    has_direct = _contains_any(low, _DIRECT_TERMS)
    has_acquired = _contains_any(low, _ACQUIRED_TERMS)
    if has_direct and has_acquired:
        return [_type_lens(LENS_DIRECT), _type_lens(LENS_ACQUIRED)]
    return []


def lens_title_suffix(lens: PortfolioLens) -> str:
    """Human-readable suffix for a chart/table/card title, e.g. ' — Direct'."""
    return f" — {lens.label}"


def apply_lens(spec, lens: PortfolioLens):
    """Merge a lens onto an MIQuerySpec (in place) and return the spec.

    The lens filters are merged into ``spec.filters`` (lens wins on conflict),
    the resolved lens is recorded on ``spec.portfolio_lens`` for output
    metadata, and a title suffix is appended when the spec has no explicit one.
    """
    if lens is None:
        return spec
    merged = dict(getattr(spec, "filters", {}) or {})
    merged.update(lens.filters)
    spec.filters = merged
    # Record lens metadata (the field is added to MIQuerySpec).
    try:
        spec.portfolio_lens = lens.to_dict()
    except Exception:  # pragma: no cover - spec without the attribute
        pass
    if lens.name != LENS_TOTAL and getattr(spec, "title", None):
        if lens.label not in str(spec.title):
            spec.title = f"{spec.title}{lens_title_suffix(lens)}"
    return spec


def resolve_and_apply(spec, text: Optional[str]):
    """Convenience: resolve the lens from text and apply it to the spec."""
    return apply_lens(spec, resolve_lens(text))
