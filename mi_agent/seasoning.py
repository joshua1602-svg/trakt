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
    #: The two finer lending windows of the governed ruling. Defaults match the
    #: ruling; both are config-driven for the same reason the front/back boundary
    #: is — a client re-cuts "new" by editing config, never by changing code.
    new_max_months: int = 1
    recent_max_months: int = 3

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

    def lending_windows(self) -> Tuple["LendingWindow", ...]:
        """The four governed windows, most specific first.

        Built from THIS config, so a client that moves the front/back boundary
        moves the front and back windows with it and nothing can disagree.
        """
        return (
            LendingWindow(LENDING_NEW,
                          f"New lending (last {self.new_max_months} month"
                          f"{'s' if self.new_max_months != 1 else ''})",
                          max_months=self.new_max_months),
            LendingWindow(LENDING_RECENT,
                          f"Recent lending (last {self.recent_max_months} months)",
                          max_months=self.recent_max_months),
            LendingWindow(LENDING_FRONT_BOOK,
                          self.describe_segment(FRONT_BOOK),
                          max_months=self.front_book_max_months,
                          segment=FRONT_BOOK),
            LendingWindow(LENDING_BACK_BOOK,
                          self.describe_segment(BACK_BOOK),
                          after_months=self.front_book_max_months,
                          segment=BACK_BOOK),
        )

    def lending_window(self, key: str) -> Optional["LendingWindow"]:
        """One governed window by key, or ``None`` for a key we do not govern."""
        for window in self.lending_windows():
            if window.key == key:
                return window
        return None

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

#: ---------------------------------------------------------------------------
#: GOVERNED LENDING WINDOWS — the business ruling of 2026-08.
#: ---------------------------------------------------------------------------
#: A second, FINER reading of the same axis. The binary front/back partition
#: above answers "which side of the book is this loan on"; a lending window
#: answers "how recently was it written", and the ruling names four:
#:
#:     NEW         originated in the last 1 month     (L1M)
#:     RECENT      originated in the last 3 months     (L3M)
#:     FRONT BOOK  originated in the last 12 months    (L12M)
#:     BACK BOOK   older than 12 months                (>L12M)
#:
#: They are NESTED, not a partition: every NEW loan is also RECENT and also
#: FRONT BOOK. That is deliberate — a CFO asking about "new lending" and a CFO
#: asking about "the front book" are asking two different questions, and
#: collapsing them into one segment is the defect this ruling exists to fix.
#:
#: All four are expressed on ``months_on_book``, the SAME derived axis the
#: segment and the bands already use, so there is one model and no second
#: mechanism. FRONT/BACK keep their existing ``seasoning_segment`` predicate so
#: nothing that already resolves them changes.
#:
#: WHAT THIS DELIBERATELY DOES NOT DO: it does not add "new lending" to
#: ``_SEGMENT_PHRASES``. Those phrases select a population EVERYWHERE in the
#: stack, and the ruling is explicit that "lending" carries a role that depends
#: on analytical context — a POPULATION of loans in a profile/mix/risk question,
#: an ORIGINATION FLOW in a run-rate/volume question. The role decision is taken
#: by the analytical intent layer, which is the only place that context exists.
#: This module supplies the windows and the vocabulary; it takes no view on role.
LENDING_NEW = "new"
LENDING_RECENT = "recent"
LENDING_FRONT_BOOK = "front_book"
LENDING_BACK_BOOK = "back_book"

#: Fixed evaluation order, most specific first. A question naming both "new
#: lending" and "the back book" is a comparison, and the caller sees both.
LENDING_WINDOW_KEYS: Tuple[str, ...] = (
    LENDING_NEW, LENDING_RECENT, LENDING_FRONT_BOOK, LENDING_BACK_BOOK)


@dataclass(frozen=True)
class LendingWindow:
    """One governed origination window, as a row predicate on months on book."""

    key: str
    label: str
    #: Inclusive upper bound in months. ``None`` ⇒ open-ended (back book).
    max_months: Optional[int] = None
    #: EXCLUSIVE lower bound in months. ``None`` ⇒ from origination.
    after_months: Optional[int] = None
    #: Set when the window is exactly one side of the binary segment, so those
    #: two keep the predicate they already had rather than gaining a second one.
    segment: Optional[str] = None

    def predicate(self) -> Dict[str, Any]:
        """The governed row predicate, in ``spec.filters`` form."""
        if self.segment is not None:
            return {SEASONING_SEGMENT_FIELD: self.segment}
        if self.after_months is not None:
            return {MONTHS_ON_BOOK_FIELD: {"op": "gt", "value": self.after_months}}
        return {MONTHS_ON_BOOK_FIELD: {"op": "le", "value": self.max_months}}


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


#: Vocabulary that names a governed LENDING WINDOW. Every entry is anchored on a
#: lending / origination / book noun, because a bare "new" or "recent" is a time
#: adverb about the QUESTION ("what is the balance recently?") rather than a
#: cohort of loans, and a population must never be created from a word that
#: might have meant something else.
#:
#: The front/back entries are NOT repeated here — they are taken from
#: ``_SEGMENT_PHRASES`` at lookup time, so the two vocabularies cannot drift.
#:
#: This is a CONCEPT vocabulary, not a question template list: it names the ways
#: a book is described, not the ways a question is asked.
_LENDING_PHRASES: Tuple[Tuple[str, str], ...] = (
    # NEW — the most recent origination flow.
    (r"\bnew lending\b", LENDING_NEW),
    (r"\bnew business\b", LENDING_NEW),
    (r"\bnew loans?\b", LENDING_NEW),
    (r"\bnew advances?\b", LENDING_NEW),
    (r"\bnewly written\b", LENDING_NEW),
    (r"\bnew(?:ly)? funded\b", LENDING_NEW),
    # RECENT — a slightly wider window than NEW, and explicitly so.
    (r"\brecent lending\b", LENDING_RECENT),
    (r"\brecent business\b", LENDING_RECENT),
    (r"\brecent loans?\b", LENDING_RECENT),
    (r"\brecent advances?\b", LENDING_RECENT),
    (r"\blending recently\b", LENDING_RECENT),
    # The VERB of origination, in the present. "Are we originating different
    # loans now?" names the loans currently being written; the past tense ("what
    # we were originating earlier") names a prior period, which is a comparand
    # rather than a second population and is handled as one.
    (r"\boriginating\b", LENDING_RECENT),
    (r"\bwe(?:'ve| have) originated recently\b", LENDING_RECENT),
    (r"\boriginated recently\b", LENDING_RECENT),
    # BACK BOOK — the same side the segment vocabulary already names, reached by
    # the "older" family of words the segment vocabulary deliberately excludes
    # because they read as an axis rather than a side. In a LENDING context they
    # are unambiguous: "older lending" is a population of older loans.
    (r"\bolder lending\b", LENDING_BACK_BOOK),
    (r"\bolder loans?\b", LENDING_BACK_BOOK),
    (r"\bolder business\b", LENDING_BACK_BOOK),
    (r"\bolder vintages?\b", LENDING_BACK_BOOK),
    (r"\bearlier lending\b", LENDING_BACK_BOOK),
    (r"\bearlier originations?\b", LENDING_BACK_BOOK),
    (r"\blegacy lending\b", LENDING_BACK_BOOK),
    (r"\bseasoned lending\b", LENDING_BACK_BOOK),
)

#: The segment phrases, expressed as lending windows. One mapping, one owner.
_SEGMENT_TO_WINDOW = {FRONT_BOOK: LENDING_FRONT_BOOK, BACK_BOOK: LENDING_BACK_BOOK}

_LENDING_RES: Tuple[Tuple[Any, str], ...] = ()


def _lending_res() -> Tuple[Tuple[Any, str], ...]:
    global _LENDING_RES
    if not _LENDING_RES:
        import re
        _LENDING_RES = tuple((re.compile(p, re.I), key)
                             for p, key in _LENDING_PHRASES)
    return _LENDING_RES


def lending_windows_named(text: Optional[str]) -> List[str]:
    """The governed lending windows a question names, in first-seen order.

    Both vocabularies are consulted — the segment phrases this module already
    owns AND the lending phrases above — so "compare new lending with the back
    book" yields ``["new", "back_book"]`` and a caller sees a genuine pair.

    Order is by POSITION IN THE QUESTION, not by vocabulary, because the first
    population a comparative question names is its subject and the second its
    comparand, and reversing them reverses the sign of every delta reported.

    This function decides WHICH windows are named. It does NOT decide whether
    they should be executed as a population — that is a role decision, and it
    depends on analytical context this module cannot see. See the ruling note
    above ``LENDING_NEW``.
    """
    if not text:
        return []
    hits: List[Tuple[int, str]] = []
    for rx, key in _lending_res():
        match = rx.search(str(text))
        if match:
            hits.append((match.start(), key))
    for rx, segment in _segment_res():
        match = rx.search(str(text))
        if match:
            hits.append((match.start(), _SEGMENT_TO_WINDOW[segment]))
    found: List[str] = []
    for _position, key in sorted(hits):
        if key not in found:
            found.append(key)
    return found


def names_lending_window(text: Optional[str]) -> bool:
    """Whether a question names any governed lending window at all.

    Used by the fabricated-population guard: a question that names one has
    REQUESTED the seasoning concept, and may therefore be answered with a
    seasoning population. One that names none may not.
    """
    return bool(lending_windows_named(text))


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
    windows = (block.get("lending_windows") or {}) if isinstance(block, dict) else {}
    new_max = windows.get("new_max_months")
    recent_max = windows.get("recent_max_months")
    return SeasoningConfig(
        front_book_max_months=int(front) if front is not None else 12,
        bands=bands,
        new_max_months=int(new_max) if new_max is not None else 1,
        recent_max_months=int(recent_max) if recent_max is not None else 3,
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
