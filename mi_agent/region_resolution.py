"""Resolve whatever a user calls a region to whatever the book actually stores.

Why this exists
---------------
"exposure to London" could not be answered on a real book. Not because the book
has no London exposure — the demonstration book carries 1,380 London loans — but
because the governed region filter targets ``geographic_region_obligor``, which
holds ``TLI43``. The value ``London`` matched nothing, the filter was dropped,
and 50 calibration cases plus the last remaining xfail failed on that one cause.

The framing that made this look like a choice between two fields was wrong.
**Raw loan tapes do not generally carry ITL codes at all.** They carry a partial
postcode, a free-text county, or the lender's own regional grouping. The ITL code
is something the canonical transformation DERIVES
(``engine/gate_2_transform/canonical_transform.py``) from a postcode outcode via
``uk_itl_master_lookup_v2.csv``. So the question is not which representation the
dimension resolves to. It is that the dimension must accept whatever the user
says and resolve it through the mapping the transformation already produces.

That mapping carries the whole ladder — postcode prefix, ITL3 code, ITL3 name,
ITL2 code and name, ITL1 code and name — so this is a lookup over a
transformation already performed, not new capability.

What resolves
-------------
Every one of these reaches the same rows:

    London            ITL1 name              -> the 21 TLI3x/TLI4x-shaped codes
    Greater London    common alias
    TLI43             ITL3 code, verbatim
    Inner London East ITL3 name
    SE1 / SE          postcode district or area
    South East        ITL1 name (distinct from the SE postcode area)

Resolution is against the VALUES PRESENT in the column being filtered, so the
same term works whether the book stores codes, names, postcodes, or the lender's
own grouping — and a term that reaches nothing returns nothing rather than a
silently dropped filter.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOOKUP_NAME = "uk_itl_master_lookup_v2.csv"

#: A UK ITL/NUTS code: TLI43, UKI32.
_CODE_RE = re.compile(r"^(?:TL|UK)[A-Z]\d{0,2}$", re.I)
#: A UK postcode district (SE1, AB10) or area (SE, AB).
_POSTCODE_RE = re.compile(r"^[A-Z]{1,2}\d{0,2}[A-Z]?$", re.I)

#: Common ways people name a region that the ITL vocabulary spells differently.
#: The ITL1 names carry parenthetical qualifiers no one types.
_ALIASES: Dict[str, str] = {
    "greater london": "london",
    "east of england": "east (england)",
    "east anglia": "east (england)",
    "the east": "east (england)",
    "east midlands": "east midlands (england)",
    "west midlands": "west midlands (england)",
    "north east": "north east (england)",
    "the north east": "north east (england)",
    "north west": "north west (england)",
    "the north west": "north west (england)",
    "south east": "south east (england)",
    "the south east": "south east (england)",
    "south west": "south west (england)",
    "the south west": "south west (england)",
    "yorkshire": "yorkshire and the humber",
    "yorks": "yorkshire and the humber",
    "the humber": "yorkshire and the humber",
    "ni": "northern ireland",
    "ulster": "northern ireland",
}


def _norm(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _variants(value) -> Set[str]:
    """Every spelling one name can be stored or typed as.

    The ITL1 names carry parenthetical qualifiers ("East (England)") that no one
    types, and people put a definite article in front ("the south east"). Both
    forms have to reach the same rows, or a near-identical phrasing fails.
    """
    base = _norm(value)
    if not base:
        return set()
    out = {base}
    stripped = re.sub(r"\s*\([^)]*\)\s*", " ", base).strip()
    if stripped:
        out.add(stripped)
    for form in list(out):
        if form.startswith("the "):
            out.add(form[4:])
        else:
            out.add(f"the {form}")
    return {f for f in out if f}


@lru_cache(maxsize=1)
def _table() -> Tuple[Tuple[str, ...], ...]:
    """The governed lookup, as rows of
    ``(prefix, itl3_code, itl3_name, itl2_code, itl2_name, itl1_code, itl1_name)``.

    Loaded from the same reference file the canonical transformation uses. If it
    is absent this returns empty and every caller degrades to exact matching —
    never to a fabricated mapping.
    """
    for candidate in (_REPO_ROOT / _LOOKUP_NAME,
                      _REPO_ROOT / "reference_data" / _LOOKUP_NAME):
        if not candidate.exists():
            continue
        try:
            import csv
            with candidate.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                return tuple(
                    (str(r.get("postcode_prefix") or ""), str(r.get("itl3_code") or ""),
                     str(r.get("itl3_name") or ""), str(r.get("itl2_code") or ""),
                     str(r.get("itl2_name") or ""), str(r.get("itl1_code") or ""),
                     str(r.get("itl1_name") or ""))
                    for r in reader)
        except Exception:                        # pragma: no cover - unreadable file
            return ()
    return ()


@lru_cache(maxsize=1)
def _index() -> Dict[str, Set[str]]:
    """term -> every ITL3 code it denotes.

    One index over every rung of the ladder. A postcode district, an ITL3 code,
    an ITL3 name, an ITL2 name and an ITL1 name are all just keys onto the same
    set of granular codes.
    """
    index: Dict[str, Set[str]] = {}

    def add(term: str, code: str) -> None:
        key = _norm(term)
        if key and code:
            index.setdefault(key, set()).add(code.strip().upper())

    for prefix, itl3, itl3_name, itl2, itl2_name, itl1, itl1_name in _table():
        add(prefix, itl3)
        # The postcode AREA (the letters) as well as the district.
        area = re.match(r"^([A-Z]{1,2})", (prefix or "").upper())
        if area:
            add(area.group(1), itl3)
        add(itl3, itl3)
        add(itl3_name, itl3)
        add(itl2, itl3)
        add(itl2_name, itl3)
        add(itl1, itl3)
        add(itl1_name, itl3)
    return index


def codes_for(term: str) -> Set[str]:
    """Every ITL3 code the term denotes, or an empty set."""
    key = _norm(term)
    if not key:
        return set()
    index = _index()
    return set(index.get(_ALIASES.get(key, key)) or index.get(key) or ())


def looks_like_region_term(term: str) -> bool:
    """Is this a term the governed mapping knows, in any representation?"""
    return bool(codes_for(term))


def _names_for(code: str) -> Set[str]:
    """Every readable name a code can be stored under."""
    code = (code or "").strip().upper()
    out: Set[str] = set()
    for _prefix, itl3, itl3_name, _itl2, itl2_name, _itl1, itl1_name in _table():
        if itl3.strip().upper() == code:
            out.update(_norm(n) for n in (itl3_name, itl2_name, itl1_name) if n)
    return out


def _canonical_matches(term: str, present: List) -> Set:
    """The present values that mean the SAME GOVERNED REGION as ``term``.

    Delegated to `engine.region_taxonomy`, which is the estate's answer to "are
    these the same region": it cleans punctuation, separators and `&`/`and`, and
    applies the client's approved synonym table. Re-implementing that here is
    what left `Yorkshire & Humberside` selecting five of sixty-five loans while
    the taxonomy resolved all three spellings to one region.

    Empty when no taxonomy is configured, when the term resolves to nothing
    governed, or when the owner cannot be reached — in every one of which the
    caller keeps exactly the behaviour it had before this existed.
    """
    try:
        from engine.region_taxonomy import METHOD_UNRESOLVED, resolve_taxonomy

        taxonomy = resolve_taxonomy(None)
        if taxonomy is None:
            return set()
        wanted, method = taxonomy.resolve_detail(term)
        if not wanted or method == METHOD_UNRESOLVED:
            return set()
        # Resolved once per DISTINCT spelling: a tape has thousands of rows and
        # a handful of regions.
        same = set()
        seen = {}
        for value in present:
            key = str(value)
            if key not in seen:
                got, _ = taxonomy.resolve_detail(key)
                seen[key] = got
            if seen[key] and seen[key] == wanted:
                same.add(value)
        return same
    except Exception:  # noqa: BLE001 - no owner reachable, no canonical rule
        return set()


def resolve(term: str, present_values: Iterable) -> List:
    """The values IN THIS COLUMN that ``term`` denotes.

    ``present_values`` is what the column actually holds — codes on one book,
    region names on another, the lender's own grouping on a third. The return is
    a subset of it, so the caller filters on values that exist rather than on a
    representation this module preferred.

    An empty return means the term reaches nothing here. That is a real answer —
    "no exposure to X" or "this book does not report X" — and is never the same
    as dropping the filter.
    """
    present = [v for v in present_values if v is not None and str(v).strip() != ""]
    key = _norm(term)
    if not key:
        return []

    # 1. Exact match on the stored representation, whatever it is.
    key_forms = _variants(key) | _variants(_ALIASES.get(key, key))
    exact = [v for v in present if _variants(v) & key_forms]
    # THE GOVERNED CANONICAL SET, unioned rather than consulted only on a miss.
    # A PARTIAL exact match is exactly the case that must not short-circuit
    # this: one spelling of a region matching is not the region.
    canonical = _canonical_matches(term, present)
    if exact or canonical:
        return sorted(set(exact) | canonical, key=lambda v: str(v))

    codes = codes_for(term)
    if not codes:
        return []

    # 2. The column stores codes: match the denoted set directly. A broad term
    #    (an ITL1 name) legitimately denotes many codes; a narrow one (an ITL3
    #    code) denotes itself.
    by_code = [v for v in present if str(v).strip().upper() in codes]
    if by_code:
        return sorted(set(by_code), key=lambda v: str(v))

    # 3. The column stores NAMES: map the denoted codes back to every name they
    #    can be stored under, then intersect with what is present.
    wanted: Set[str] = set()
    for code in codes:
        for name in _names_for(code):
            wanted |= _variants(name)
    wanted |= _variants(_ALIASES.get(key, key)) | _variants(key)
    by_name = [v for v in present
               if _variants(v) & wanted]
    if by_name:
        return sorted(set(by_name), key=lambda v: str(v))

    # 4. The column stores postcodes: match on the district or area.
    if _POSTCODE_RE.match(key.upper()):
        upper = key.upper()
        by_postcode = [v for v in present
                       if str(v).strip().upper().replace(" ", "").startswith(upper)]
        if by_postcode:
            return sorted(set(by_postcode), key=lambda v: str(v))
    return []


def describe(term: str, resolved: Sequence) -> str:
    """Receipt-grade text: what the term was taken to mean, and via what.

    A resolution the reader cannot see is a substitution.
    """
    if not resolved:
        return f"{term} — no matching value in this book"
    shown = ", ".join(str(v) for v in list(resolved)[:6])
    more = f" (+{len(resolved) - 6} more)" if len(resolved) > 6 else ""
    if len(resolved) == 1 and _norm(resolved[0]) == _norm(term):
        return f"{term}"
    return (f"{term} → {shown}{more} (governed UK ITL mapping, "
            f"{_LOOKUP_NAME})")


__all__ = ["resolve", "codes_for", "looks_like_region_term", "describe"]
