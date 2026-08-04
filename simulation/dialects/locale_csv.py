"""simulation.dialects.locale_csv — a continental servicer's CSV export.

What this dialect deliberately exercises:

* a second, entirely different header vocabulary (also resolved through the
  approved onboarding contract, not the global alias library);
* day-first dates (``31/01/2025``), handled by the production date ladder;
* thousands-separated money, handled by ``canonical_transform.to_decimal``;
* percentages as **0-1 fractions**, which the production ``_resolve_ltv``
  reconciles back onto the canonical percentage-point scale — the same value
  expressed a different way, not a different value;
* booleans as ``Yes``/``No`` and ``1``/``0``;
* a different status vocabulary;
* region CODES rather than readable names, with the postcode still carrying the
  region, so geography stays recoverable;
* a reordered schema and a couple of optional columns omitted.

What it deliberately does NOT do: comma DECIMAL separators. ``to_decimal``
strips ``,`` as a thousands separator unconditionally, so ``1234,56`` would
become ``123456`` — a silent hundred-fold error rather than a parse failure.
Writing it would test a capability the pipeline does not have; the limitation is
pinned instead by ``tests/test_simulation_locale_numerics.py`` and reported as a
production finding.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..assets import get_profile
from ..models import DIALECT_LOCALE_CSV, CaseManifest, SchemaEvolution
from . import register
from . import _format as F
from ._plan import build_plan
from .vocabulary import OPTIONAL_OMISSIONS

#: Written as 0-1 fractions; production reconciles them back to points.
_FRACTION_FIELDS = ("current_loan_to_value", "original_loan_to_value")

#: Booleans rendered ``1``/``0`` rather than ``Yes``/``No``.
_BINARY_BOOLEAN_FIELDS = ("resident", "credit_impaired_obligor")


def _value(field: str, raw: Any, formats: Dict[str, str], enum_style: str) -> str:
    if raw in (None, ""):
        return ""
    if field == "account_status":
        table = (F.STATUS_SYNONYMS_ALTERNATE if enum_style == "alternate"
                 else F.STATUS_SYNONYMS_LOCALE)
        return table.get(str(raw), str(raw))
    if field == "collateral_geography":
        return F.REGION_CODES.get(str(raw), str(raw))
    if field in _FRACTION_FIELDS:
        return F.percent_fraction(raw, 6)
    fmt = formats.get(field, "")
    if fmt == "date":
        return F.day_first_date(raw)
    if fmt == "decimal":
        return F.thousands_grouped(raw, 2)
    if fmt == "integer":
        return F.integer(raw)
    if fmt == "Y/N":
        style = "binary" if field in _BINARY_BOOLEAN_FIELDS else "yesno"
        return F.boolean(raw, style)
    return F.text(raw)


def render(rows: List[Dict[str, Any]], case: CaseManifest, *, period_index: int,
           reporting_date: str, out_dir: Path,
           evolution: Sequence[SchemaEvolution] = ()) -> Path:
    from .._registry import field_formats

    profile = get_profile(case.asset_class)
    formats = field_formats(case.asset_class)
    # This dialect omits a couple of optional columns outright: a servicer
    # extract is narrower than the originator's own export.
    omissions = set(OPTIONAL_OMISSIONS.get(case.asset_class, ()))
    fields = [f for f in profile.canonical_fields if f not in omissions]
    plan = build_plan(case, flavour="locale", period_index=period_index,
                      fields=fields, evolution=evolution)
    # A stable reordering distinct from the lender dialect's, so the two do not
    # accidentally agree on column order.
    plan.columns = list(reversed(plan.columns))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (f"{case.portfolio_id}_{reporting_date.replace('-', '')}"
                      f"_locale.csv")
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(plan.headers)
        for row in rows:
            line = [_value(canon, row.get(canon), formats, plan.enum_style)
                    for _header, canon in plan.columns]
            line += [""] * len(plan.extras)
            writer.writerow(line)
    return path


register(DIALECT_LOCALE_CSV, render)

__all__ = ["render"]
