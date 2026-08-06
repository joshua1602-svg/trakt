"""simulation.dialects._format — value formatting shared by the dialects.

Everything a lender can plausibly do to the same number or date, expressed as
small explicit functions. The corresponding parsers already exist in production
(``canonical_transform.to_iso_date`` / ``to_percentage`` / ``to_decimal`` /
``to_bool_yn``), so each formatter here targets a real capability rather than an
invented one.
"""

from __future__ import annotations

from typing import Any, Dict

from ..dates import parse_iso

#: ISO 3166-2 style region codes, used by the locale dialect instead of names.
#: The readable label remains recoverable through the collateral postcode, so a
#: coded region is a disclosure difference and never a loss of economic truth.
REGION_CODES: Dict[str, str] = {
    "South East": "UKJ", "London": "UKI", "South West": "UKK",
    "East of England": "UKH", "North West": "UKD", "West Midlands": "UKG",
    "Yorkshire and The Humber": "UKE", "East Midlands": "UKF",
    "Scotland": "UKM", "North East": "UKC", "Wales": "UKL",
}

#: Alternative status vocabularies. Every value here resolves through the
#: production enum-normalisation path or the approved contract; nothing here is
#: a word the platform has never seen.
STATUS_SYNONYMS_LENDER: Dict[str, str] = {
    "performing": "Active",
    "arrears": "In Arrears",
    "default": "Defaulted",
    "enforcement": "In Enforcement",
    "redeemed": "Redeemed",
    "repossessed": "Repossessed",
    "written_off": "Written Off",
}

STATUS_SYNONYMS_LOCALE: Dict[str, str] = {
    "performing": "PERFORMING",
    "arrears": "PAST DUE",
    "default": "NON-PERFORMING",
    "enforcement": "LEGAL",
    "redeemed": "CLOSED",
    "repossessed": "REPOSSESSED",
    "written_off": "WRITTEN-OFF",
}

#: A second lender status vocabulary, used by the ``change_enum_synonyms``
#: schema-evolution case so a mid-history vocabulary switch is exercised.
STATUS_SYNONYMS_ALTERNATE: Dict[str, str] = {
    "performing": "Current",
    "arrears": "Delinquent",
    "default": "Default",
    "enforcement": "Recovery",
    "redeemed": "Settled",
    "repossessed": "Repossession",
    "written_off": "Loss",
}

#: Boolean renderings. Every one is handled by ``canonical_transform.to_bool_yn``.
BOOLEAN_STYLES: Dict[str, Dict[bool, str]] = {
    "yn": {True: "Y", False: "N"},
    "yesno": {True: "Yes", False: "No"},
    "binary": {True: "1", False: "0"},
}


def iso_date(value: Any) -> str:
    return "" if value in (None, "") else str(value)[:10]


def day_first_date(value: Any) -> str:
    """``31/01/2025`` — the day-first presentation the date ladder handles."""
    if value in (None, ""):
        return ""
    d = parse_iso(str(value))
    return f"{d.day:02d}/{d.month:02d}/{d.year}"


def long_date(value: Any) -> str:
    """``31 Jan 2025`` — a mixed presentation an Excel export often carries."""
    if value in (None, ""):
        return ""
    d = parse_iso(str(value))
    months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    return f"{d.day:02d} {months[d.month - 1]} {d.year}"


def decimal_point(value: Any, places: int = 2) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.{places}f}"


def thousands_grouped(value: Any, places: int = 2) -> str:
    """``1,234,567.89`` — a lender export that kept its display formatting."""
    if value in (None, ""):
        return ""
    return f"{float(value):,.{places}f}"


def comma_decimal(value: Any, places: int = 2) -> str:
    """``1.234.567,89`` -- continental grouping and decimal separator.

    NOT used by any dialect. ``canonical_transform.to_decimal`` strips ``,`` as a
    thousands separator unconditionally, so this rendering would be silently
    misread by a factor of one hundred rather than reported as a parse failure.
    It is retained because ``tests/test_simulation_locale_numerics.py`` pins that
    limitation, and any future fix would be validated against this formatter.
    """
    if value in (None, ""):
        return ""
    whole, _, frac = f"{float(value):,.{places}f}".partition(".")
    return whole.replace(",", ".") + ("," + frac if frac else "")


def percent_points(value: Any, places: int = 4) -> str:
    """``43.1000`` — percentage points, the canonical convention."""
    return decimal_point(value, places)


def percent_string(value: Any, places: int = 2) -> str:
    """``43.10%`` — a percent-suffixed string."""
    if value in (None, ""):
        return ""
    return f"{float(value):.{places}f}%"


def percent_fraction(value: Any, places: int = 6) -> str:
    """``0.431000`` — a 0-1 fraction of the same percentage."""
    if value in (None, ""):
        return ""
    return f"{float(value) / 100.0:.{places}f}"


def boolean(value: Any, style: str = "yn") -> str:
    if value in (None, ""):
        return ""
    truthy = str(value).strip().lower() in ("y", "yes", "true", "1")
    return BOOLEAN_STYLES[style][truthy]


def integer(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(int(float(value)))


def text(value: Any) -> str:
    return "" if value is None else str(value)


__all__ = ["BOOLEAN_STYLES", "REGION_CODES", "STATUS_SYNONYMS_ALTERNATE",
           "STATUS_SYNONYMS_LENDER", "STATUS_SYNONYMS_LOCALE", "boolean",
           "comma_decimal", "day_first_date", "decimal_point", "integer",
           "iso_date", "long_date", "percent_fraction", "percent_points",
           "percent_string", "text", "thousands_grouped"]
