"""Which geography a book reports on, decided once as a property of the ASSET.

THE PROBLEM THIS SETTLES
------------------------
A loan book carries more than one geography, and they are different facts:

    borrower / obligor      where the obligor is
    collateral / property   where the security is

For a lifetime mortgage the two usually coincide. For an auto loan they do not
coincide at all. For a buy-to-let they routinely disagree. So "balance by
region", asked with no qualifier, has no answer until somebody has decided
which geography this book reports on.

Until this module existed, nothing had. Three separate places each answered the
question by taking whichever geography column happened to be POPULATED FIRST:

  * ``mi_agent_api.funded_prep._coalesce_group_dimensions`` gap-filled the
    BORROWER column from the COLLATERAL column row by row, so a book that never
    collected obligor geography reported the property's region as the
    borrower's — a silent wrong answer with no way for a reader to tell;
  * ``engine.region_taxonomy`` harmonises from the first populated of its source
    fields, so on a book whose obligor column holds ITL3 codes and whose
    collateral column holds readable region names, the harmonised column
    resolved for nobody: 11,035 rows, ``region_mapping_method: unresolved``,
    zero populated — while the readable names sat untouched beside it;
  * ``llm_query_parser._preferred_region`` then bound generic "region" to the
    head of a fixed preference order whose first entry was that empty column.

"First populated wins" is not a semantics. It is the absence of one. A geography
basis is a decision about what the book IS, so it belongs where the book's other
established facts live: the asset class onboarding settled, and the governed
portfolio registry that carries it.

WHAT THIS MODULE OWNS, AND WHAT IT DOES NOT
-------------------------------------------
It owns exactly one thing: **which basis a question is measured on, and which
column carries that basis in a given book.**

It resolves no geography VALUE. No place name, postcode, ITL or NUTS code
appears here or in its configuration. ``mi_agent.region_resolution`` remains the
only owner of the ITL ladder that maps a reader's term onto a book's value, and
``engine.region_taxonomy`` remains the only harmoniser of region vocabularies.
Adding a second table of places is how the last attempt at this went wrong.

It is MI-side, and it stays MI-side. The regulatory pipeline projects Annex
geography fields from their own sources under their own contract; an ND code in
a regulatory geography field is a DECLARATION ("not collected"), never an
absence for MI to fill. Nothing under ``engine/gate_*``, ``engine/regime_*`` or
``engine/annex_delivery_agent`` reads this module, and nothing here reads them.

THE CONTRACT
------------
============================= ==========================================
question says                 measured on
============================= ==========================================
"region", "by region",        the portfolio's CONFIGURED PRIMARY BASIS
"geography", "area",
"the Scottish balance"
"borrower region",            the BORROWER basis, always
"obligor region"
"property region",            the COLLATERAL basis, always
"collateral region",
"asset region"
============================= ==========================================

Explicit language always wins over the configured default — a reader who says
"borrower region" has stated the basis and is owed that basis or nothing.

And when the book cannot support an explicitly requested basis, the answer is a
REFUSAL. Never a silent substitution of the primary basis: an answer labelled
"borrower region" that was measured on collateral is worse than no answer,
because the reader cannot tell.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger("mi_agent.mi_geography")

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: The governed per-asset-class default table. Configuration, not code: adding
#: an asset class must not require a release.
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "asset" / "mi_geography.yaml"

#: Environment override, for a deployment that keeps its asset configuration
#: somewhere else.
ENV_CONFIG_PATH = "TRAKT_MI_GEOGRAPHY_CONFIG"

#: The registry key a portfolio entry uses to override its asset class default.
REGISTRY_KEY = "mi_geography"
REGISTRY_BASIS_KEY = "primary_basis"

BASIS_BORROWER = "borrower"
BASIS_COLLATERAL = "collateral"
BASES: Tuple[str, ...] = (BASIS_BORROWER, BASIS_COLLATERAL)

#: How a reader may name a basis. The MI semantics registry already carries
#: these words as field synonyms; this table is what turns the word into the
#: BASIS rather than into one particular column, so that "collateral region"
#: means the collateral basis whichever column a given book carries it in.
_BASIS_TERMS: Dict[str, str] = {
    "borrower": BASIS_BORROWER, "borrowers": BASIS_BORROWER,
    "obligor": BASIS_BORROWER, "obligors": BASIS_BORROWER,
    "customer": BASIS_BORROWER, "client": BASIS_BORROWER,
    "collateral": BASIS_COLLATERAL, "property": BASIS_COLLATERAL,
    "properties": BASIS_COLLATERAL, "asset": BASIS_COLLATERAL,
    "security": BASIS_COLLATERAL,
}

#: The columns that carry each basis, in GRANULARITY TIERS, most readable first.
#:
#: Two columns in the same tier are the same fact at the same granularity,
#: delivered under different names — ``collateral_geography`` and
#: ``property_region`` are one book saying "region" and another saying the same
#: thing. Which of them a source system supplies is a delivery detail, so they
#: gap-fill each other freely.
#:
#: Two columns in DIFFERENT tiers are the same fact at different granularities,
#: and they do not gap-fill each other. Filling a column of readable region
#: names from a column of ITL3 codes leaves one column speaking two vocabularies
#: — "London", "South East", "TLC31" — which splits a breakdown into categories
#: no reader can reconcile. The tier is instead chosen once, per book, by
#: :func:`field_for_basis`: the most readable tier that this book actually
#: populates wins, and a book that only ever carried codes is answered in codes
#: rather than in a mixture.
#:
#: ``canonical_region_reporting`` / ``canonical_region_detail`` are deliberately
#: absent. They are harmonised OUTPUTS derived from whichever source field was
#: populated first, so they belong to no basis and cannot answer a question that
#: is about one. A reader who wants them asks for them by name.
TIER_REPORTING = "reporting"
TIER_CODE = "code"

BASIS_FIELD_TIERS: Dict[str, Tuple[Tuple[str, Tuple[str, ...]], ...]] = {
    BASIS_COLLATERAL: (
        (TIER_REPORTING, ("collateral_geography", "property_region")),
        (TIER_CODE, ("geographic_region_collateral",
                     "geographic_region_collateral_itl3")),
    ),
    BASIS_BORROWER: (
        (TIER_CODE, ("geographic_region_obligor",
                     "geographic_region_obligor_itl3")),
    ),
}

#: Flattened, most readable first. The order is a granularity preference only:
#: every field here carries the same basis, so choosing among them cannot change
#: what an answer MEANS, which is the whole reason it is safe to let data
#: presence decide here and nowhere else.
BASIS_FIELDS: Dict[str, Tuple[str, ...]] = {
    basis: tuple(field for _tier, fields in tiers for field in fields)
    for basis, tiers in BASIS_FIELD_TIERS.items()}


#: The columns that may be OFFERED as the region axis: the head of each tier.
#:
#: A finer geography is not another spelling of Region. ``*_itl3`` carries a
#: basis and gap-fills its tier head, but binding a breakdown to it turns eleven
#: regions into a hundred and seventy sub-regions the reader did not ask for —
#: so it is never the axis. Because coalescing has already filled the tier head
#: from it, a book whose only borrower geography arrived as ITL3 still answers:
#: on the head, at the granularity the head carries.
AXIS_FIELDS: Dict[str, Tuple[str, ...]] = {
    basis: tuple(fields[0] for _tier, fields in tiers)
    for basis, tiers in BASIS_FIELD_TIERS.items()}


def axis_fields(basis: Optional[str]) -> Tuple[str, ...]:
    """The columns ``basis`` may be MEASURED on, most readable first."""
    return AXIS_FIELDS.get(str(normalise_basis(basis) or ""), ())


def taxonomy_source_fields(contract: Any = None,
                           extra: Sequence[str] = ()) -> Tuple[str, ...]:
    """Which columns the region HARMONISATION should read, and in what order.

    ``engine.region_taxonomy.apply`` takes its source columns from the caller and
    fills, row by row, from the first populated one. Its own default order leads
    with the BORROWER column, which is how the live platform book came to have
    ``canonical_region_reporting`` unresolved on all 11,035 of its rows: the
    obligor column holds ITL3 codes, which no ITL1 taxonomy can resolve, while
    the readable collateral names sat in the column behind it.

    So MI states the order instead of accepting one. With a contract in hand the
    book's own basis leads, which is what makes the harmonised column coherent
    with every other regional answer. Without one, the collateral axis leads the
    borrower axis — a stated order over BASES, applied to a column that belongs
    to no basis and is only ever a last-resort axis.

    ``extra`` keeps any additional sources the caller's taxonomy declares (a bare
    ``region`` column, say) at the tail, so nothing that resolved before stops
    resolving.
    """
    lead = axis_fields(getattr(contract, "primary_basis", None) or contract)
    order: list = list(lead)
    for basis in (BASIS_COLLATERAL, BASIS_BORROWER):
        for field in axis_fields(basis):
            if field not in order:
                order.append(field)
    for field in extra:
        if field and field not in order:
            order.append(field)
    return tuple(order)


def tiers_for_basis(basis: Optional[str]
                    ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """``((tier, fields), ...)`` for ``basis``, most readable tier first."""
    return BASIS_FIELD_TIERS.get(str(normalise_basis(basis) or ""), ())


def interchangeable_field_groups() -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """``((primary, sources), ...)`` — every set of columns that may gap-fill
    one another, one per (basis, tier). This is what ``funded_prep`` coalesces
    on, so the preparation and the query layer can never disagree about which
    region columns are the same fact.
    """
    out = []
    for _basis, tiers in BASIS_FIELD_TIERS.items():
        for _tier, fields in tiers:
            if len(fields) > 1:
                out.append((fields[0], tuple(fields)))
    return tuple(out)


#: Reverse index: which basis a column belongs to. A region column absent from
#: this map belongs to no basis — which is a statement, not an oversight.
_BASIS_BY_FIELD: Dict[str, str] = {
    field: basis for basis, fields in BASIS_FIELDS.items() for field in fields}

#: How the effective basis was arrived at, for the receipt.
SOURCE_QUESTION = "stated_in_question"
SOURCE_PORTFOLIO = "portfolio_registry"
SOURCE_ASSET_DEFAULT = "asset_class_default"
SOURCE_NONE = "unconfigured"

_BLANK = {"", "nan", "none", "nat", "<na>", "null"}

#: How many DISTINCT values to test before concluding a column carries no
#: geography. A tape has thousands of rows and a handful of regions, and the
#: answer is "yes" on the first recognised one — the cap only bounds the "no".
_DISTINCT_PROBE_LIMIT = 200

#: How many rows to look at before falling back to the whole column. A book's
#: regions are all present within the first few hundred loans in every tape this
#: estate has seen; the full scan behind it is what keeps that an optimisation
#: rather than an assumption.
_HEAD_SAMPLE_ROWS = 1000


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
def normalise_basis(value: Any) -> Optional[str]:
    """The governed basis a word names, or None if it names no basis."""
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return None
    if text in BASES:
        return text
    return _BASIS_TERMS.get(text)


#: Region phrases that STATE their basis. A reader who writes one of these has
#: said which geography they mean and is owed that geography or a refusal —
#: never the book's primary basis wearing the label they asked for.
#:
#: Kept here rather than in the parser so that the layer that BINDS the phrase
#: and the layer that REFUSES when the book cannot support it read one table.
BASIS_TERMS: Dict[str, str] = {
    "obligor region": BASIS_BORROWER,
    "obligor regions": BASIS_BORROWER,
    "obligor geography": BASIS_BORROWER,
    "borrower region": BASIS_BORROWER,
    "borrower regions": BASIS_BORROWER,
    "borrower geography": BASIS_BORROWER,
    "borrower location": BASIS_BORROWER,
    "collateral region": BASIS_COLLATERAL,
    "collateral regions": BASIS_COLLATERAL,
    "collateral geography": BASIS_COLLATERAL,
    "collateral location": BASIS_COLLATERAL,
    "property region": BASIS_COLLATERAL,
    "property regions": BASIS_COLLATERAL,
    "property geography": BASIS_COLLATERAL,
    "property location": BASIS_COLLATERAL,
    "asset region": BASIS_COLLATERAL,
    "security region": BASIS_COLLATERAL,
}

_STATED_BASIS_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(t) for t in BASIS_TERMS),
                             key=len, reverse=True)) + r")\b", re.IGNORECASE)


def stated_basis(question: Optional[str]) -> Optional[str]:
    """The basis a question NAMES, or None if it names none.

    Returns None when a question names both — "borrower region versus property
    region" states two and is not a case for the single-basis refusal below; the
    ordinary dimension machinery answers it or declines to.
    """
    found = {BASIS_TERMS[m.group(1).lower()]
             for m in _STATED_BASIS_RE.finditer(str(question or ""))}
    return next(iter(found)) if len(found) == 1 else None


def basis_of_field(field: Optional[str]) -> Optional[str]:
    """Which basis ``field`` carries, or None if it carries none."""
    return _BASIS_BY_FIELD.get(str(field or ""))


def basis_fields(basis: Optional[str]) -> Tuple[str, ...]:
    """The columns carrying ``basis``, most readable first."""
    return BASIS_FIELDS.get(str(basis or ""), ())


def other_basis(basis: Optional[str]) -> Optional[str]:
    """The basis that is not ``basis``. Used to explain a refusal."""
    resolved = normalise_basis(basis)
    if resolved == BASIS_BORROWER:
        return BASIS_COLLATERAL
    if resolved == BASIS_COLLATERAL:
        return BASIS_BORROWER
    return None


# --------------------------------------------------------------------------- #
# The governed default table
# --------------------------------------------------------------------------- #
def config_path() -> Optional[str]:
    configured = os.environ.get(ENV_CONFIG_PATH)
    if configured:
        return configured
    return str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None


@lru_cache(maxsize=8)
def _load_defaults(location: Optional[str] = None) -> Dict[str, str]:
    """``{asset_class: basis}``. Never raises: a broken table degrades to "no
    governed default", which is refused rather than guessed."""
    loc = location or config_path()
    if not loc:
        return {}
    try:
        import yaml
        doc = yaml.safe_load(Path(loc).read_text(encoding="utf-8")) or {}
    except Exception as exc:                                     # noqa: BLE001
        logger.warning("MI geography defaults unreadable (%s): %s", loc, exc)
        return {}
    if not isinstance(doc, Mapping):
        return {}
    table = doc.get("primary_basis_by_asset_class")
    if not isinstance(table, Mapping):
        return {}
    out: Dict[str, str] = {}
    for key, value in table.items():
        asset = str(key or "").strip().lower()
        basis = normalise_basis(value)
        if asset and basis:
            out[asset] = basis
    return out


def default_primary_basis(asset_class: Any,
                          *, location: Optional[str] = None) -> Optional[str]:
    """The governed default basis for ``asset_class``, or None when the table
    declares none. An unlisted asset class is NOT given a basis: MI would rather
    say it does not know than assume one."""
    from mi_agent.portfolio_metadata import normalise_asset_class

    table = _load_defaults(location)
    raw = str(asset_class or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw and raw in table:
        return table[raw]
    normalised = normalise_asset_class(asset_class)
    if normalised and normalised in table:
        return table[normalised]
    return None


def configured_basis(entry: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The basis a governed portfolio-registry entry declares, if any.

    Accepts both the nested shape written by onboarding
    (``mi_geography: {primary_basis: collateral}``) and a flat
    ``mi_geography: collateral`` a human may hand-author.
    """
    if not isinstance(entry, Mapping):
        return None
    block = entry.get(REGISTRY_KEY)
    if isinstance(block, Mapping):
        return normalise_basis(block.get(REGISTRY_BASIS_KEY))
    return normalise_basis(block)


# --------------------------------------------------------------------------- #
# What a given book can actually support
# --------------------------------------------------------------------------- #
def _carries_geography(series) -> bool:
    """Does this column actually carry a geography?

    Not "is it non-blank". A regulatory geography field may be populated on every
    row with a value that DECLARES the geography was never collected, and reading
    that as a category produces a regional breakdown whose largest row is a
    declaration. So the question is put to the estate's own region owner instead:
    ``mi_agent.region_resolution`` maps a term through the governed ITL ladder —
    postcode, ITL3 code, ITL3/ITL2/ITL1 name, alias — and a column with no value
    the ladder recognises is not a geography this book can be measured on.

    Delegating the judgement is the point. MI states no opinion about any
    regulatory vocabulary; it asks whether a value is a PLACE, using the one
    table that already answers that question for every other part of MI. A book
    whose regions lie outside the governed ladder is one MI cannot answer
    regional questions on, and saying so is the honest result.
    """
    try:
        from mi_agent.region_resolution import looks_like_region_term

        # A HEAD SAMPLE FIRST, then the whole column only if it found nothing.
        #
        # This runs once per request over every region column of an 11,000-row
        # tape, so the common case has to be cheap — and the common case is a
        # column whose very first value is a place. Scanning the whole column is
        # reserved for the answer that costs something to get wrong: concluding
        # a column carries NO geography, which refuses a question.
        def _probe(values) -> bool:
            for value in values:
                if value and value.lower() not in _BLANK \
                        and looks_like_region_term(value):
                    return True
            return False

        text = series.dropna().astype(str)
        sample = text.head(_HEAD_SAMPLE_ROWS).str.strip().drop_duplicates()
        if _probe(sample.head(_DISTINCT_PROBE_LIMIT)):
            return True
        if len(text) <= _HEAD_SAMPLE_ROWS:
            return False
        distinct = text.str.strip().drop_duplicates()
        return _probe(distinct.head(_DISTINCT_PROBE_LIMIT))
    except Exception:                                            # noqa: BLE001
        return False


def _carrying_columns(frame) -> Optional[frozenset]:
    """The basis columns of ``frame`` that carry a geography, or None when no
    frame was supplied."""
    if frame is None:
        return None
    try:
        return frozenset(
            str(column) for column in frame.columns
            if column in _BASIS_BY_FIELD and _carries_geography(frame[column]))
    except Exception:                                            # noqa: BLE001
        return None


def field_for_basis(basis: Optional[str], *, available_columns=None,
                    frame=None, carrying: Optional[frozenset] = None
                    ) -> Optional[str]:
    """The column this book carries ``basis`` in, most readable first.

    With a ``frame`` in hand a column must be POPULATED to be chosen — the
    lesson of ``canonical_region_reporting``, which was present on every serving
    frame, empty on every row of them, and chosen anyway. With only column names
    the best available test is presence.
    """
    fields = axis_fields(basis)
    if not fields:
        return None
    if carrying is None:
        carrying = _carrying_columns(frame)
    if carrying is not None:
        for field in fields:
            if field in carrying:
                return field
        return None
    if available_columns is None:
        return fields[0]
    columns = {str(c) for c in available_columns}
    for field in fields:
        if field in columns:
            return field
    return None


def supported_bases(*, available_columns=None, frame=None) -> Tuple[str, ...]:
    """The bases this book can answer on, in governed order.

    The frame is scanned ONCE and the result shared across the bases: reading
    every region column twice per request is a cost with nothing to show for it.
    """
    carrying = _carrying_columns(frame)
    return tuple(b for b in BASES
                 if field_for_basis(b, available_columns=available_columns,
                                    frame=frame, carrying=carrying))


# --------------------------------------------------------------------------- #
# The resolved contract for one request
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GeographyContract:
    """What geography this book reports on, and how that was decided.

    ``primary_basis`` is None for a book whose asset class carries no governed
    default and whose registry entry declares none. That is a real state and it
    is stated, not papered over: generic region language on such a book is
    unresolved, exactly as an unknown dimension would be.
    """

    primary_basis: Optional[str] = None
    source: str = SOURCE_NONE
    asset_class: Optional[str] = None
    supported: Tuple[str, ...] = ()

    def field_for(self, basis: Optional[str], *, available_columns=None,
                  frame=None) -> Optional[str]:
        return field_for_basis(basis, available_columns=available_columns,
                               frame=frame)

    def supports(self, basis: Optional[str]) -> bool:
        resolved = normalise_basis(basis)
        return bool(resolved) and resolved in self.supported

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primaryBasis": self.primary_basis,
            "basisSource": self.source,
            "assetClass": self.asset_class,
            "supportedBases": list(self.supported),
        }


def resolve_contract(*, asset_class: Any = None,
                     registry_entry: Optional[Mapping[str, Any]] = None,
                     available_columns=None, frame=None,
                     config_location: Optional[str] = None) -> GeographyContract:
    """The effective geography contract for one request.

    Precedence: the portfolio's own declaration, then the asset class default,
    then nothing. A portfolio override exists so that a book whose asset class
    default does not fit can be represented without a second mechanism — and so
    that the override is visible in the registry rather than inferred from data.
    """
    from mi_agent.portfolio_metadata import normalise_asset_class

    resolved_class = normalise_asset_class(asset_class)
    declared = configured_basis(registry_entry)
    if declared:
        basis, source = declared, SOURCE_PORTFOLIO
    else:
        basis = default_primary_basis(asset_class, location=config_location)
        source = SOURCE_ASSET_DEFAULT if basis else SOURCE_NONE
    return GeographyContract(
        primary_basis=basis, source=source, asset_class=resolved_class,
        supported=supported_bases(available_columns=available_columns,
                                  frame=frame))


def contract_for_scope(*, client_id: Optional[str] = None,
                      portfolio_ids: Sequence[str] = (),
                      asset_class: Any = None,
                      available_columns=None, frame=None) -> GeographyContract:
    """The contract for a request that may span several portfolios.

    Every portfolio in scope resolves its own basis, from its own declaration or
    its own asset class. They then have to AGREE: a mixed-asset scope containing
    a mortgage book (collateral) and an auto book (borrower) has no single
    meaning for "balance by region", and inventing one would put two different
    facts in the same column of the same table. Disagreement therefore yields no
    primary basis, and generic region language on that scope is unresolved —
    which a reader can act on, unlike a silently mixed answer.
    """
    entries: Dict[str, Mapping[str, Any]] = {}
    try:
        from mi_agent.portfolio_metadata import load_portfolio_metadata
        entries = dict(load_portfolio_metadata(client_id))
    except Exception as exc:                                     # noqa: BLE001
        logger.info("portfolio geography overlay unavailable: %s", exc)

    # Narrow to the portfolios named, when any of them is one this registry
    # knows. A caller that names none — or names a client rather than a book —
    # gets every entry the client has, which is the same reading of an
    # unqualified scope the rest of the service uses.
    wanted = [str(p).strip().lower() for p in (portfolio_ids or ()) if str(p).strip()]
    matched = [entries[p] for p in wanted if p in entries]
    selected = matched or list(entries.values())

    bases: set = set()
    sources: set = set()
    classes: set = set()
    for entry in selected:
        declared = configured_basis(entry)
        if declared:
            bases.add(declared)
            sources.add(SOURCE_PORTFOLIO)
        else:
            fallback = default_primary_basis(entry.get("asset_class"))
            if fallback:
                bases.add(fallback)
                sources.add(SOURCE_ASSET_DEFAULT)
        resolved_class = entry.get("asset_class")
        if resolved_class:
            classes.add(str(resolved_class))

    if not bases and asset_class is not None:
        fallback = default_primary_basis(asset_class)
        if fallback:
            bases.add(fallback)
            sources.add(SOURCE_ASSET_DEFAULT)
        classes.add(str(asset_class))

    supported = supported_bases(available_columns=available_columns, frame=frame)
    if len(bases) != 1:
        if len(bases) > 1:
            logger.info("portfolios in scope report on different geography bases "
                        "(%s); generic region language is unresolved for this scope",
                        ", ".join(sorted(bases)))
        return GeographyContract(primary_basis=None, source=SOURCE_NONE,
                                 asset_class=(sorted(classes)[0] if len(classes) == 1
                                              else None),
                                 supported=supported)
    return GeographyContract(
        primary_basis=next(iter(bases)),
        source=(SOURCE_PORTFOLIO if SOURCE_PORTFOLIO in sources
                else SOURCE_ASSET_DEFAULT),
        asset_class=(sorted(classes)[0] if len(classes) == 1 else None),
        supported=supported)


def contract_for_portfolio(portfolio_id: Optional[str] = None, *,
                           client_id: Optional[str] = None,
                           asset_class: Any = None,
                           available_columns=None, frame=None
                           ) -> GeographyContract:
    """The contract for a named portfolio, read from the governed registry.

    Degrades to the asset-class default (and then to no basis) rather than
    failing: geography configuration must never be the reason a question that
    does not depend on it goes unanswered.
    """
    entry: Optional[Mapping[str, Any]] = None
    resolved_class = asset_class
    try:
        from mi_agent.portfolio_metadata import load_portfolio_metadata
        overlay = load_portfolio_metadata(client_id)
        if portfolio_id:
            entry = overlay.get(str(portfolio_id).strip().lower())
        if entry and resolved_class is None:
            resolved_class = entry.get("asset_class")
    except Exception as exc:                                     # noqa: BLE001
        logger.info("portfolio geography overlay unavailable: %s", exc)
    return resolve_contract(asset_class=resolved_class, registry_entry=entry,
                            available_columns=available_columns, frame=frame)
