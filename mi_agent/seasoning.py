"""mi_agent.seasoning — the governed VINTAGE / SEASONING axis.

Seasoning answers *when a loan was originated*. It is a different axis from
PROVENANCE (direct / acquired), which answers *where the loan came from*, and
the two are independent:

    a directly-originated loan written five years ago
        -> DIRECT     by provenance
        -> BACK BOOK  by seasoning

so "back book" must never resolve to *acquired*, and "new origination" must
never resolve to *direct*. Both mappings were live defects before P1J-1; the
lens vocabulary in :mod:`mi_agent.portfolio_lens` owns provenance and this
module owns seasoning, with no overlap between their vocabularies.

The chain is derived, never stored::

    origination_date -> months_on_book -> seasoning_bucket   (analytical bands)
                                       -> seasoning_segment  (binary front/back)

``months_on_book`` is measured against the **governed reporting / cut-off date
of the frame being queried**, never wall-clock "today", so the same loan queried
against an older snapshot carries the seasoning it had at that reporting date.

One config block (``seasoning:`` in ``config/mi/buckets.yaml``) drives BOTH
outputs, so the bands and the front/back split can never disagree. The binary
split is layered on the same model rather than being a second mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

#: Derived column names. These are DERIVED read-time dimensions, not source data.
MONTHS_ON_BOOK_FIELD = "months_on_book"
VINTAGE_YEAR_FIELD = "vintage_year"
SEASONING_BUCKET_FIELD = "seasoning_bucket"
SEASONING_SEGMENT_FIELD = "seasoning_segment"

#: The two values of the binary segment. Human-readable because they are shown
#: in answers and receipts, where "Back Book" reads better than "back_book".
FRONT_BOOK = "Front Book"
BACK_BOOK = "Back Book"

_DEFAULT_CONFIG_PATH = (Path(__file__).resolve().parent.parent
                        / "config" / "mi" / "buckets.yaml")


@dataclass(frozen=True)
class SeasoningBand:
    """One inclusive integer month band. ``max_months`` None ⇒ open-ended."""

    name: str
    min_months: int
    max_months: Optional[int] = None

    def contains(self, months: float) -> bool:
        if months < self.min_months:
            return False
        return self.max_months is None or months <= self.max_months


@dataclass(frozen=True)
class SeasoningConfig:
    """The governed seasoning model: one boundary plus the analytical bands."""

    front_book_max_months: int
    bands: Tuple[SeasoningBand, ...]

    def segment_for(self, months: Optional[float]) -> Optional[str]:
        """Binary front/back for a months-on-book value (None ⇒ unknown)."""
        if months is None or pd.isna(months):
            return None
        return FRONT_BOOK if months <= self.front_book_max_months else BACK_BOOK

    def band_for(self, months: Optional[float]) -> Optional[str]:
        """Analytical seasoning band for a months-on-book value."""
        if months is None or pd.isna(months):
            return None
        for band in self.bands:
            if band.contains(months):
                return band.name
        return None

    @property
    def band_names(self) -> List[str]:
        return [b.name for b in self.bands]

    def describe_segment(self, segment: str) -> str:
        """Receipt-grade label, e.g. ``Front Book (0-12 months)``.

        The boundary is spelled out because it is configurable: a reader must be
        able to see WHICH cutoff produced the population they are looking at.
        """
        if segment == FRONT_BOOK:
            return f"{FRONT_BOOK} (0-{self.front_book_max_months} months)"
        if segment == BACK_BOOK:
            return f"{BACK_BOOK} ({self.front_book_max_months + 1}+ months)"
        return str(segment)


#: Phrases that name ONE SPECIFIC seasoning population, with the segment each
#: selects. Every entry is anchored on a population noun ("book", "loans") or is
#: unambiguously a cohort of originations, because these force a POPULATION
#: FILTER and a filter must never be created from a word that might have meant
#: something else.
#:
#: Deliberately ABSENT — these stay dimension synonyms but never select a
#: population, because they are genuinely ambiguous and must fail safe:
#:   * "new lending"    — reads as a FLOW measure ("the run rate of new lending")
#:                        at least as naturally as a population of loans;
#:   * "seasoning"      — names the axis, not a side of it;
#:   * "older vintages" — asks for analysis ACROSS vintage cohorts rather than
#:                        for back_book=true.
_SEGMENT_PHRASES: Tuple[Tuple[str, str], ...] = (
    (r"\bfront book\b", FRONT_BOOK),
    (r"\bnew origination(?:s)?\b", FRONT_BOOK),
    (r"\brecent origination(?:s)?\b", FRONT_BOOK),
    (r"\bnewly originated\b", FRONT_BOOK),
    (r"\brecently originated\b", FRONT_BOOK),
    (r"\bback book\b", BACK_BOOK),
    (r"\bbackbook\b", BACK_BOOK),
    (r"\blegacy book\b", BACK_BOOK),
    (r"\bseasoned book\b", BACK_BOOK),
    (r"\bseasoned loans\b", BACK_BOOK),
)

_SEGMENT_RES: Tuple[Tuple[Any, str], ...] = ()


def _segment_res() -> Tuple[Tuple[Any, str], ...]:
    global _SEGMENT_RES
    if not _SEGMENT_RES:
        import re
        _SEGMENT_RES = tuple((re.compile(p, re.I), seg)
                             for p, seg in _SEGMENT_PHRASES)
    return _SEGMENT_RES


def segments_named(text: Optional[str]) -> List[str]:
    """The distinct seasoning segments a question names, in first-seen order."""
    if not text:
        return []
    found: List[str] = []
    for rx, segment in _segment_res():
        if rx.search(str(text)) and segment not in found:
            found.append(segment)
    return found


def mask_segment_phrases(text: Optional[str]) -> str:
    """``text`` with seasoning-segment phrases blanked, preserving offsets.

    The same discipline P1I-A applies to governed SCOPE phrases, extended to the
    seasoning vocabulary P1J-1 introduced. Without it "how many acquired loans
    are in the front book?" had the place-resolver read "front" as a region,
    producing a filter on a collateral geography called "Front" that matched no
    rows — so a governed population of 250 loans was unreachable.

    Blanking rather than deleting keeps every other offset valid.
    """
    if not text:
        return text or ""
    out = list(str(text))
    for rx, _segment in _segment_res():
        for match in rx.finditer(str(text)):
            for i in range(match.start(), match.end()):
                out[i] = " "
    return "".join(out)


def resolve_segment_population(text: Optional[str]) -> Optional[str]:
    """The single seasoning segment a question SELECTS, or None.

    Returns a value only when the question names exactly one side of the split.
    Naming BOTH ("new origination ... vs the back book", "compare the front book
    with the back book") is a COMPARISON: the answer must group by the segment,
    not narrow to one of them, so this returns None and the dimension stands.

    This is a role decision, taken once, before the spec is validated — the same
    discipline as the governed scope resolution: decide whether a phrase names a
    population or an axis *before* anything is built from it, so a filter that
    would narrow the answer is never created by accident.
    """
    named = segments_named(text)
    return named[0] if len(named) == 1 else None


def _coerce_bands(raw: Any) -> Tuple[SeasoningBand, ...]:
    bands: List[SeasoningBand] = []
    for entry in (raw or []):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        mx = entry.get("max_months")
        bands.append(SeasoningBand(
            name=name,
            min_months=int(entry.get("min_months") or 0),
            max_months=None if mx is None else int(mx),
        ))
    return tuple(bands)


def load_seasoning_config(path: Optional[Path | str] = None) -> SeasoningConfig:
    """Load the governed ``seasoning:`` block.

    Configuration-driven by contract: a client moves the front/back boundary or
    re-cuts the bands by editing config, never by changing code.
    """
    import yaml

    target = Path(path) if path else _DEFAULT_CONFIG_PATH
    data: Dict[str, Any] = {}
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - a missing/broken config must not break MI
        data = {}
    block = (data.get("seasoning") or {}) if isinstance(data, dict) else {}
    bands = _coerce_bands(block.get("buckets"))
    front = block.get("front_book_max_months")
    return SeasoningConfig(
        front_book_max_months=int(front) if front is not None else 12,
        bands=bands,
    )


def months_between(origination: pd.Series, reporting: pd.Series) -> pd.Series:
    """Whole months from origination to the reporting date, per row.

    Same arithmetic the funded-tape prep already used for ``months_on_book`` —
    kept in one place so the two paths can never drift apart.
    """
    return ((reporting.dt.year - origination.dt.year) * 12
            + (reporting.dt.month - origination.dt.month))


def derive_seasoning(frame: pd.DataFrame,
                     months: Optional[pd.Series] = None,
                     config: Optional[SeasoningConfig] = None) -> List[str]:
    """Add ``seasoning_bucket`` / ``seasoning_segment`` to *frame* in place.

    ``months`` defaults to the frame's own ``months_on_book``. Returns the names
    of the columns actually added, for derived-field disclosure. Additive and
    idempotent: a column already present is never overwritten.
    """
    cfg = config or load_seasoning_config()
    if months is None:
        if MONTHS_ON_BOOK_FIELD not in frame.columns:
            return []
        months = frame[MONTHS_ON_BOOK_FIELD]
    numeric = pd.to_numeric(months, errors="coerce")
    if not numeric.notna().any():
        return []

    added: List[str] = []
    if SEASONING_BUCKET_FIELD not in frame.columns and cfg.bands:
        frame[SEASONING_BUCKET_FIELD] = numeric.map(cfg.band_for)
        added.append(SEASONING_BUCKET_FIELD)
    if SEASONING_SEGMENT_FIELD not in frame.columns:
        frame[SEASONING_SEGMENT_FIELD] = numeric.map(cfg.segment_for)
        added.append(SEASONING_SEGMENT_FIELD)
    return added
