"""mi_agent_api.presentation — the presentation semantics both surfaces share.

The analysis is defined once and already shared: the React dashboard and the
investor PPTX call the same compute functions. What was *not* shared was
everything decided AFTER the payload — the order categories are drawn in, how a
bucket label is tidied, whether a dimension is ordinal at all. React decided it
in TypeScript (``lib/stratOrder.ts``); the deck decided it, differently, in
matplotlib. The same LTV stratification therefore read in bucket order on screen
and in balance order in the pack.

This module is the single owner of those decisions. It answers three questions,
and deliberately nothing else:

    what ORDER are these categories presented in?
    what does this category LABEL read as?
    is this dimension ORDINAL (a band ladder) or nominal (a set of names)?

The order is not a heuristic here. ``config/mi/buckets.yaml`` already declares
the governed ladder for every banded dimension — LTV, borrower age, ticket size,
interest rate, PD/LGD/EAD, time on book — in the order the business reads them.
That declaration is the authority; parsing a numeric bound out of a label is only
the fallback for dimensions the registry does not band (vintage years, say).

WHAT THIS MODULE IS NOT
-----------------------
It computes no economic value, selects no population and owns no threshold. It
receives categories that a governed compute function already produced and says
what order to draw them in. Anything that changes a NUMBER belongs upstream in
the MI layer, not here.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: Labels that mean "no data". They sort last in every mode and are never parsed
#: for an ordinal bound. Mirrors ``lib/stratOrder.ts`` UNKNOWN_RE, which this
#: module replaces as the authority.
_UNKNOWN_RE = re.compile(
    r"^(unknown|not supplied|not available|none|n/?a|other|-)$", re.IGNORECASE)

#: A bare number carrying a redundant ``.0`` fraction — a vintage year that made
#: the trip through a float column ("2008.0"). Display-only tidy-up.
_FLOAT_YEAR_RE = re.compile(r"^(\d{1,4})\.0$")

#: Stratification / dimension keys as the API and the deck name them, mapped to
#: the bucket key in ``config/mi/buckets.yaml`` that declares their ladder. Keys
#: are matched case-insensitively and after stripping a ``_bucket``/``_band``
#: suffix, so ``ltv``, ``ltv_bucket`` and ``ltv_band`` all resolve.
_DIMENSION_TO_BUCKET: Dict[str, str] = {
    "ltv": "ltv_bucket",
    "current_ltv": "ltv_bucket",
    "original_ltv": "original_ltv_bucket",
    "age": "borrower_age_bucket",
    "borrower_age": "borrower_age_bucket",
    "youngest_borrower_age": "youngest_borrower_age_bucket",
    "rate": "interest_rate_bucket",
    "interest_rate": "interest_rate_bucket",
    "ticket": "balance_band",
    "balance": "balance_band",
    "loan_size": "balance_band",
    "pd": "pd_bucket",
    "lgd": "lgd_bucket",
    "ead": "ead_bucket",
    "time_on_book": "time_on_book_bucket",
    "seasoning": "time_on_book_bucket",
}

#: Dimensions that are ordinal by nature but are not banded by the registry, so
#: their order comes from the numeric-bound fallback rather than a declared
#: ladder. Vintage is the case that matters: "2023" < "2024" < "2025".
_ORDINAL_WITHOUT_BUCKET = frozenset({"vintage", "vintage_year", "cohort",
                                     "origination_year", "month", "period"})


def _norm_key(dimension: Optional[str]) -> str:
    key = str(dimension or "").strip().lower()
    for suffix in ("_bucket", "_band", "_bands", "_buckets"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    return key


@lru_cache(maxsize=1)
def _bucket_labels() -> Dict[str, Tuple[str, ...]]:
    """``{bucket_key: (label, ...)}`` from the governed bucket registry.

    Cached because it is read once per bar list and the registry is static for
    the life of the process. Never raises: a registry that cannot be read leaves
    every dimension on the ordinal-bound fallback, which is the behaviour that
    existed before this module.
    """
    try:
        from analytics_lib.buckets import load_bucket_config
        config = load_bucket_config() or {}
    except Exception:  # noqa: BLE001 - presentation must never break MI
        return {}
    out: Dict[str, Tuple[str, ...]] = {}
    for key, spec in config.items():
        if not isinstance(spec, dict):
            continue
        labels = spec.get("labels")
        if isinstance(labels, (list, tuple)) and labels:
            out[str(key)] = tuple(str(v) for v in labels)
            # A bucket also answers to the semantic field it materialises, so a
            # caller naming the COLUMN (``ticket_bucket``) resolves to the ladder
            # declared under the bucket key (``balance_band``).
            semantic = spec.get("semantic_field")
            if semantic:
                out.setdefault(_norm_key(semantic), tuple(str(v) for v in labels))
    return out


def governed_ladder(dimension: Optional[str]) -> Optional[Tuple[str, ...]]:
    """The declared label ladder for *dimension*, or ``None`` if it has none."""
    key = _norm_key(dimension)
    if not key:
        return None
    labels = _bucket_labels()
    bucket = _DIMENSION_TO_BUCKET.get(key)
    if bucket and bucket in labels:
        return labels[bucket]
    return labels.get(key)


def clean_label(label: Any) -> str:
    """Tidy a category label for display. ``"2008.0"`` -> ``"2008"``.

    Display-only: the value behind the label is never touched, and a label that
    is already clean is returned unchanged.
    """
    text = str(label if label is not None else "").strip()
    match = _FLOAT_YEAR_RE.match(text)
    return match.group(1) if match else text


def is_unknown(label: Any) -> bool:
    """Whether *label* is a "no data" bucket, which always sorts last."""
    text = clean_label(label)
    return not text or bool(_UNKNOWN_RE.match(text))


def ordinal_bound(label: Any) -> Optional[float]:
    """The numeric sort bound parsed from a band label, or ``None``.

    Handles the shapes the registry and the tape produce::

        "40-50%" / "40–50%"  -> 40        "<20%" / "<=20"  -> 19.5
        ">=100%" / "100%+"   -> 100       "85+"            -> 85
        "2008"               -> 2008      "£100K-£150K"    -> 100
        "50-100k"            -> 50000     "500k-1m"        -> 500000

    A ``<``-style label sorts just below its boundary so it precedes the band
    that starts there.
    """
    text = clean_label(label)
    if not text or is_unknown(text):
        return None
    below = bool(re.match(r"^\s*[<≤]", text))
    stripped = re.sub(r"[£$€,\s]", "", text)
    match = re.search(r"-?\d+(\.\d+)?", stripped)
    if not match:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    # A magnitude suffix attached to the FIRST number ("50-100k" -> 50k).
    tail = stripped[match.end():match.end() + 1].lower()
    if tail == "k":
        value *= 1_000
    elif tail == "m":
        value *= 1_000_000
    elif tail == "b":
        value *= 1_000_000_000
    return value - 0.5 if below else value


def is_ordinal(dimension: Optional[str], labels: Sequence[Any]) -> bool:
    """Whether these categories form a ladder rather than a set of names.

    A governed ladder settles it outright. Without one, a dimension is ordinal
    when at least half its real (non-unknown) labels parse to a numeric bound —
    the same test React applied, kept as the fallback for dimensions the registry
    does not band.
    """
    if governed_ladder(dimension):
        return True
    if _norm_key(dimension) in _ORDINAL_WITHOUT_BUCKET:
        return True
    real = [l for l in labels if not is_unknown(l)]
    if not real:
        return False
    numeric = sum(1 for l in real if ordinal_bound(l) is not None)
    return numeric * 2 >= len(real)


def order_key(dimension: Optional[str]):
    """A sort key over CATEGORY LABELS for *dimension*.

    Three tiers, in this order: the governed ladder position where the registry
    declares one; the parsed numeric bound where it does not but the dimension is
    ordinal; case-insensitive alphabetical otherwise. Unknown-style buckets sink
    below all three.
    """
    ladder = governed_ladder(dimension)
    positions = {label: i for i, label in enumerate(ladder or ())}

    def key(label: Any):
        text = clean_label(label)
        if is_unknown(text):
            return (2, 0.0, "")
        if text in positions:
            return (0, float(positions[text]), "")
        bound = ordinal_bound(text)
        if bound is not None:
            # A label the ladder does not carry still sorts by its own bound,
            # after every label the ladder does — a band the registry never
            # declared is real data and must not vanish or lead.
            return (1 if ladder else 0, bound, text.lower())
        return (1 if ladder else 0, float("inf"), text.lower())

    return key


def order_categories(labels: Iterable[Any], *, dimension: Optional[str] = None
                     ) -> List[str]:
    """The display order for *labels*, cleaned. Input is not mutated."""
    items = [clean_label(l) for l in labels]
    if not is_ordinal(dimension, items):
        return sorted(items, key=lambda t: (is_unknown(t), t.lower()))
    return sorted(items, key=order_key(dimension))


#: Marks a payload whose bar order was decided here. A renderer that sees it
#: must draw the bars in the order given and must not re-sort — that is the
#: whole point. Its absence means an older or mock payload, where a renderer may
#: still fall back to its own heuristic.
DISPLAY_ORDER_GOVERNED = "governed"


def order_bars(bars: Sequence[Dict[str, Any]], *, dimension: Optional[str] = None,
               label_key: str = "label") -> List[Dict[str, Any]]:
    """Reorder already-computed bars for display, cleaning their labels.

    The bars arrive from a governed compute function ranked by materiality
    (``analytics_lib.stratify`` sorts by balance, descending) and any truncation
    to a top-N has ALREADY happened against that ranking. This only re-sequences
    what survived, so the most material bands are still the ones shown — they are
    simply drawn in the order a reader expects to see them.

    No value is touched.
    """
    key = order_key(dimension)
    ordinal = is_ordinal(dimension, [b.get(label_key) for b in bars])

    def sort_key(bar: Dict[str, Any]):
        label = clean_label(bar.get(label_key))
        return key(label) if ordinal else (is_unknown(label), label.lower())

    out: List[Dict[str, Any]] = []
    for bar in sorted(bars, key=sort_key):
        row = dict(bar)
        row[label_key] = clean_label(bar.get(label_key))
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# Informativeness — is this dimension worth a panel?
#
# A breakdown with one meaningful category is not insight. "Broker / channel:
# Direct 100%" spends a panel restating a fact the reader already had, and on a
# four-panel matrix it displaces a dimension that would have said something.
#
# This decides, once, whether a distribution has anything to show, and ranks
# the candidates that do. It reads bars a governed compute function already
# produced and classifies their SHAPE — no economic value is derived here, and
# no dimension is named. A book whose exposure genuinely sits in one bucket
# reports that dimension as uninformative for THIS book, not as unavailable.
# --------------------------------------------------------------------------- #

#: A dimension needs at least this many categories carrying real weight before a
#: panel can show a distribution rather than a single bar.
MIN_MEANINGFUL_CATEGORIES = 2

#: A category is "meaningful" at or above this share of the dimension's total.
#: Below it, a handful of loans in a second band does not make a distribution:
#: the panel would draw one full-width bar and a sliver.
MEANINGFUL_SHARE = 0.02

#: Above this share in a single category the distribution is effectively
#: degenerate — 99% in one band tells the reader the same thing a single bar
#: does, at the cost of a panel.
DEGENERATE_SHARE = 0.98


#: Reason codes for a dimension that did not reach the page. Each is
#: mechanically derivable from the selector's own inputs, which is the whole
#: point: a pack that prints "not informative" about a dimension whose data
#: says otherwise has told the reader something false in the one place it
#: explains itself.
REASON_NOT_SUPPLIED = "NOT_SUPPLIED"
REASON_ONE_CATEGORY = "ONE_CATEGORY"
REASON_LOW_INFORMATION = "LOW_INFORMATION"
REASON_LOW_COVERAGE = "LOW_COVERAGE"
REASON_LOWER_RANKED = "LOWER_RANKED_THAN_SELECTED_DIMENSIONS"

#: Investor-facing wording for each code. The code is what a test asserts; the
#: sentence is what the methodology page prints.
REASON_WORDING: Dict[str, str] = {
    REASON_NOT_SUPPLIED: "this book does not report it",
    REASON_ONE_CATEGORY: ("the whole balance sits in one category, so there is "
                          "no distribution to show"),
    REASON_LOW_INFORMATION: ("one category carries almost the whole balance, so "
                             "the distribution restates the total"),
    REASON_LOW_COVERAGE: ("too little of the book can be classified on this "
                          "dimension to read a distribution"),
    REASON_LOWER_RANKED: ("other dimensions carry more information for this "
                          "book"),
}

#: Below this share of the book classified, a distribution is drawn over so
#: little of the portfolio that its shape is not the portfolio's shape.
MIN_COVERAGE = 0.60

#: Scores are compared at this granularity. Two dimensions whose information
#: content differs by less than a rounding step are equivalent for a reader,
#: and THAT is where presentation preference — the economic relevance of the
#: cut — is allowed to decide. It is a tie-breaker and nothing more.
SCORE_PRECISION = 2

#: Categories at which granularity stops adding much. The factor is
#: k / (k + REFERENCE), which rises with category count but with diminishing
#: returns, so a two-category split is discounted without being excluded — a
#: single-versus-joint borrower cut can be economically important precisely
#: because it is binary.
GRANULARITY_REFERENCE = 4.0


def _bar_values(bars: Sequence[Dict[str, Any]], value_key: str) -> List[float]:
    out: List[float] = []
    for bar in bars or ():
        try:
            value = float(bar.get(value_key) or 0.0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            out.append(value)
    return out


def _evenness(shares: Sequence[float]) -> float:
    """How evenly the balance is spread, from 0 (all in one) to 1 (uniform).

    Normalised Shannon entropy. It is the standard measure of "how much does
    knowing this dimension tell me", it is computable from the bars alone, and
    it can be recomputed by hand from the published shares — which is what
    makes the ledger auditable rather than an opaque model output.
    """
    real = [s for s in shares if s > 0]
    if len(real) < 2:
        return 0.0
    entropy = -sum(s * math.log(s) for s in real)
    return entropy / math.log(len(real))


def dispersion(bars: Sequence[Dict[str, Any]], *, value_key: str = "balance",
               coverage: Optional[float] = None) -> Dict[str, Any]:
    """How much this distribution has to say, and whether it can be read.

    Returns the shape AND an information ``score``. The score is
    ``evenness x granularity``:

      * EVENNESS is normalised entropy — a distribution concentrated in one
        band tells a reader less than one spread across its bands;
      * GRANULARITY is ``k / (k + 4)`` over the categories carrying real
        weight — more bands resolve more, with diminishing returns, so seven
        LTV bands beat two but two are not ruled out.

    Deliberately NOT "most categories wins": that rule would make a binary
    single-versus-joint split permanently ineligible against a seven-band cut,
    and it is not true that the binary split says less about a lifetime book.
    """
    values = _bar_values(bars, value_key)
    total = sum(values)
    if not values or total <= 0:
        return {"categories": len(bars or ()), "effectiveCategories": 0,
                "topShare": None, "evenness": 0.0, "granularity": 0.0,
                "score": 0.0, "coverage": coverage, "informative": False,
                "reasonCode": REASON_NOT_SUPPLIED,
                "reason": REASON_WORDING[REASON_NOT_SUPPLIED]}

    shares = sorted((v / total for v in values), reverse=True)
    meaningful = [s for s in shares if s >= MEANINGFUL_SHARE]
    effective = len(meaningful)
    top = shares[0]
    evenness = _evenness(meaningful)
    granularity = (effective / (effective + GRANULARITY_REFERENCE)
                   if effective else 0.0)
    score = round(evenness * granularity, 6)

    code: Optional[str] = None
    if effective < MIN_MEANINGFUL_CATEGORIES:
        code = REASON_ONE_CATEGORY
    elif top >= DEGENERATE_SHARE:
        code = REASON_LOW_INFORMATION
    elif coverage is not None and coverage < MIN_COVERAGE:
        code = REASON_LOW_COVERAGE

    return {"categories": len(values), "effectiveCategories": effective,
            "topShare": round(top, 6), "evenness": round(evenness, 6),
            "granularity": round(granularity, 6), "score": score,
            "coverage": coverage, "informative": code is None,
            "reasonCode": code,
            "reason": REASON_WORDING[code] if code else None}


def is_informative(bars: Sequence[Dict[str, Any]], *, value_key: str = "balance"
                   ) -> bool:
    """Whether this distribution earns a panel at all."""
    return bool(dispersion(bars, value_key=value_key)["informative"])


def _coverage_of(entry: Mapping[str, Any]) -> Optional[float]:
    """The share of the book this dimension can classify, where the governed
    stratification payload reports it."""
    for key in ("coveragePct", "coverage_pct"):
        value = entry.get(key)
        if value is not None:
            try:
                pct = float(value)
            except (TypeError, ValueError):
                continue
            return pct / 100.0 if pct > 1.0 else pct
    return None


def select_dimensions(candidates: Sequence[Dict[str, Any]], *, want: int,
                      value_key: str = "balance", bars_key: str = "bars",
                      key_key: str = "key",
                      preferred: Sequence[str] = ()) -> Dict[str, Any]:
    """Pick up to ``want`` dimensions BY INFORMATION CONTENT.

    The rule, in order:

      1. drop what cannot be read — nothing supplied, one category, a
         degenerate split, or too little of the book classified;
      2. rank what survives by its information score;
      3. where scores are equivalent at the published precision, order by
         ``preferred`` — the governed economic relevance of the cut.

    Preference is a TIE-BREAKER. It used to be the first sort key, which meant
    a preferred dimension beat a more informative one whatever the data said:
    on a representative book, ticket size (five bands, 57% in one) displaced
    origination vintage (five bands, 20% in the largest) purely because
    "ticket" appeared earlier in a list. Worse, the ledger then reported that
    more informative dimensions had been available, which the same numbers
    contradicted.

    Returns ``{"selected": [...], "rejected": [{key, label, reasonCode,
    reason, score, ...}]}``. Every rejection carries the code and the numbers
    it was derived from, so the methodology page can be checked against the
    selector's own inputs.
    """
    order = {k: i for i, k in enumerate(preferred)}
    scored, rejected = [], []
    for entry in candidates or ():
        key = str(entry.get(key_key) or "")
        shape = dispersion(entry.get(bars_key) or (), value_key=value_key,
                           coverage=_coverage_of(entry))
        if not shape["informative"]:
            rejected.append({"key": key, "label": entry.get("label"),
                             "reasonCode": shape["reasonCode"],
                             "reason": shape["reason"],
                             "score": shape["score"],
                             "effectiveCategories": shape["effectiveCategories"],
                             "topShare": shape["topShare"]})
            continue
        scored.append((entry, shape, key))

    def rank(item):
        _entry, shape, key = item
        # INFORMATION FIRST. The score is rounded so that two dimensions a
        # reader could not tell apart are a genuine tie — and only there does
        # the governed preference order decide.
        return (-round(shape["score"], SCORE_PRECISION),
                order.get(key, len(order)),
                key)

    ranked = sorted(scored, key=rank)
    selected = [entry for entry, _shape, _key in ranked[:max(0, want)]]
    # The score at the CUT is the lowest selected score — which is the last
    # entry actually selected, not the want-th, because a book may not have
    # ``want`` usable dimensions to give.
    cut = (round(ranked[len(selected) - 1][1]["score"], SCORE_PRECISION)
           if selected else None)
    for entry, shape, key in ranked[max(0, want):]:
        rejected.append({
            "key": key, "label": entry.get("label"),
            "reasonCode": REASON_LOWER_RANKED,
            # The truthful form of what this used to claim: it is a RANKING
            # statement, and the scores that produced it travel with it.
            "reason": (f"{len(selected)} dimensions scored higher for this "
                       f"book (this {shape['score']:.2f} against {cut:.2f} at "
                       f"the cut)"),
            "score": shape["score"],
            "effectiveCategories": shape["effectiveCategories"],
            "topShare": shape["topShare"]})
    return {"selected": selected, "rejected": rejected}
