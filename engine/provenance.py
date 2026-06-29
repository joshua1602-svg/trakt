"""engine.provenance — source-portfolio provenance contract (single source of truth).

Trakt onboards a small lender's *direct* originations alongside one or more
*acquired* back books. For securitisation readiness every loan must carry a
clear source-cohort tag so management, the IB, legal counsel and rating agencies
can split direct originations from acquired books.

This module is the ONE place that defines:

  * the canonical provenance field names (``PROVENANCE_FIELDS``);
  * how ``source_portfolio_type`` is derived from a ``source_portfolio_id``
    prefix when not given explicitly (``derive_portfolio_type``);
  * how a run-level provenance record is assembled and validated, failing
    *closed* (never silently "unknown") when mandatory inputs are missing
    (``build_provenance`` / :class:`ProvenanceError`);
  * how every row of a dataframe is stamped (``stamp_dataframe``);
  * the run-level lineage entries that record these fields came from run
    metadata rather than a source column (``lineage_entries``).

It has NO heavy dependencies beyond pandas so it can be imported by the
onboarding agent, the canonical transform, validation, the regime projector and
the MI agent without pulling in the wider pipeline.

Keep field names simple and deterministic. Do NOT overload the existing
``portfolio_type`` canonical field — that means asset class / product type
(equity_release, sme, cre, …), a different concept from source provenance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:  # pandas is available across the pipeline; keep the import soft for tooling.
    import pandas as pd
except Exception:  # pragma: no cover - pandas always present at runtime
    pd = None  # type: ignore


# --------------------------------------------------------------------------- #
# Field contract
# --------------------------------------------------------------------------- #

#: Canonical provenance field names, in stamping / column order.
PROVENANCE_FIELDS: List[str] = [
    "source_portfolio_id",
    "source_portfolio_type",
    "source_portfolio_label",
    "acquisition_date",
    "seller_name",
    "portfolio_cohort",
]

#: Mandatory on every onboarded canonical row.
REQUIRED_PROVENANCE_FIELDS: List[str] = ["source_portfolio_id", "portfolio_cohort"]

#: Allowed values for ``source_portfolio_type``.
PORTFOLIO_TYPE_DIRECT = "direct"
PORTFOLIO_TYPE_ACQUIRED = "acquired"
VALID_PORTFOLIO_TYPES = (PORTFOLIO_TYPE_DIRECT, PORTFOLIO_TYPE_ACQUIRED)

#: id-prefix → type, used for deterministic derivation.
_PREFIX_TO_TYPE = {
    "direct_": PORTFOLIO_TYPE_DIRECT,
    "acquired_": PORTFOLIO_TYPE_ACQUIRED,
}

#: A source_portfolio_id is a lowercase slug, e.g. ``direct_001`` / ``acquired_002``.
_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")


class ProvenanceError(ValueError):
    """Raised when provenance inputs are missing or inconsistent.

    Managed-service runs fail closed on this rather than assigning ``unknown``.
    """


@dataclass(frozen=True)
class Provenance:
    """A resolved, validated run-level provenance record."""

    source_portfolio_id: str
    source_portfolio_type: str
    source_portfolio_label: Optional[str] = None
    acquisition_date: Optional[str] = None
    seller_name: Optional[str] = None
    portfolio_cohort: Optional[str] = None

    def as_row(self) -> Dict[str, Any]:
        """Return the per-row stamp values (cohort defaulted to the id)."""
        return {
            "source_portfolio_id": self.source_portfolio_id,
            "source_portfolio_type": self.source_portfolio_type,
            "source_portfolio_label": self.source_portfolio_label,
            "acquisition_date": self.acquisition_date,
            "seller_name": self.seller_name,
            "portfolio_cohort": self.portfolio_cohort or self.source_portfolio_id,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable record for run summaries / manifests."""
        return dict(self.as_row())


def _clean(value: Any) -> Optional[str]:
    """Trim a scalar to ``None`` when blank."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def derive_portfolio_type(source_portfolio_id: str) -> Optional[str]:
    """Derive ``direct`` / ``acquired`` from an id prefix, else ``None``.

    ``direct_*`` → ``direct``; ``acquired_*`` → ``acquired``. Anything else
    returns ``None`` so the caller can fail closed asking for an explicit type.
    """
    sid = _clean(source_portfolio_id)
    if not sid:
        return None
    low = sid.lower()
    for prefix, ptype in _PREFIX_TO_TYPE.items():
        if low.startswith(prefix):
            return ptype
    return None


def build_provenance(
    source_portfolio_id: Optional[str],
    source_portfolio_type: Optional[str] = None,
    source_portfolio_label: Optional[str] = None,
    acquisition_date: Optional[str] = None,
    seller_name: Optional[str] = None,
    portfolio_cohort: Optional[str] = None,
    *,
    allow_unknown_acquisition_date: bool = False,
) -> Provenance:
    """Assemble + validate a run-level provenance record, failing closed.

    Rules enforced here (mirrored by the canonical validation gate):

      * ``source_portfolio_id`` is mandatory and must be a stable slug;
      * ``source_portfolio_type`` must be ``direct`` or ``acquired`` — derived
        from the id prefix when not given; if it can be neither given nor
        derived, raise (do NOT guess);
      * ``acquisition_date`` is required for ``acquired`` portfolios unless
        ``allow_unknown_acquisition_date`` is set; it must be empty for
        ``direct`` portfolios unless explicitly provided;
      * ``portfolio_cohort`` defaults to ``source_portfolio_id``.
    """
    sid = _clean(source_portfolio_id)
    if not sid:
        raise ProvenanceError(
            "source_portfolio_id is mandatory for onboarding. Provide e.g. "
            "--source-portfolio-id direct_001 (current book) or acquired_001 "
            "(first acquired back book). Trakt will not assign 'unknown'."
        )
    if not _ID_RE.match(sid.lower()):
        raise ProvenanceError(
            f"source_portfolio_id {sid!r} is not a valid stable id. Use a "
            "lowercase slug such as direct_001, acquired_001 or acquired_002."
        )

    ptype = _clean(source_portfolio_type)
    if ptype is None:
        ptype = derive_portfolio_type(sid)
        if ptype is None:
            raise ProvenanceError(
                f"source_portfolio_type could not be derived from "
                f"source_portfolio_id {sid!r} (expected a 'direct_' or "
                f"'acquired_' prefix). Pass --source-portfolio-type explicitly "
                f"as one of {VALID_PORTFOLIO_TYPES}."
            )
    else:
        ptype = ptype.lower()
        if ptype not in VALID_PORTFOLIO_TYPES:
            raise ProvenanceError(
                f"source_portfolio_type {source_portfolio_type!r} is invalid; "
                f"must be one of {VALID_PORTFOLIO_TYPES}."
            )

    acq = _clean(acquisition_date)
    if ptype == PORTFOLIO_TYPE_ACQUIRED and not acq and not allow_unknown_acquisition_date:
        raise ProvenanceError(
            f"acquisition_date is required for acquired portfolio {sid!r}. "
            "Pass --acquisition-date YYYY-MM-DD, or set "
            "--allow-unknown-acquisition-date to onboard with it unknown."
        )
    if ptype == PORTFOLIO_TYPE_DIRECT and acq is not None:
        # A direct/originated book has no acquisition; keep it null unless the
        # operator explicitly meant it. We surface this rather than dropping it.
        raise ProvenanceError(
            f"acquisition_date {acq!r} was provided for direct portfolio "
            f"{sid!r}. Direct books are originated, not acquired; omit "
            "--acquisition-date for direct portfolios."
        )

    return Provenance(
        source_portfolio_id=sid,
        source_portfolio_type=ptype,
        source_portfolio_label=_clean(source_portfolio_label),
        acquisition_date=acq,
        seller_name=_clean(seller_name),
        portfolio_cohort=_clean(portfolio_cohort) or sid,
    )


def stamp_dataframe(df: "pd.DataFrame", provenance: Provenance) -> "pd.DataFrame":
    """Stamp every row of ``df`` with the provenance fields (in place).

    Run-level metadata is authoritative: existing provenance columns are
    overwritten so a tape can never carry a stale or conflicting source tag.
    """
    row = provenance.as_row()
    for field in PROVENANCE_FIELDS:
        df[field] = row.get(field)
    return df


def lineage_entries(provenance: Provenance) -> Dict[str, Dict[str, Any]]:
    """Field-level lineage showing provenance came from run metadata.

    Shaped to slot into the lineage tracker's ``fields`` map: each entry records
    that the source was run-level operator metadata, not a source column.
    """
    row = provenance.as_row()
    entries: Dict[str, Dict[str, Any]] = {}
    for field in PROVENANCE_FIELDS:
        entries[field] = {
            "source": {
                "origin": "run_metadata",
                "raw_columns": [],
                "methods": [{"raw": None, "method": "run_metadata", "confidence": 1.0}],
                "provided_by": "onboarding_operator",
            },
            "value": row.get(field),
        }
    return entries


def add_cli_arguments(parser) -> None:
    """Attach the standard provenance flags to an argparse parser.

    Shared so the onboarding entrypoint and the live pipeline orchestrator use
    identical flag names and help text.
    """
    g = parser.add_argument_group("source-portfolio provenance")
    g.add_argument(
        "--source-portfolio-id", dest="source_portfolio_id", default="",
        help="Stable source-cohort id stamped on every loan, e.g. direct_001 "
             "(current/directly originated book), acquired_001 (first acquired "
             "back book), acquired_002 (second acquired portfolio).",
    )
    g.add_argument(
        "--source-portfolio-type", dest="source_portfolio_type", default="",
        choices=["", PORTFOLIO_TYPE_DIRECT, PORTFOLIO_TYPE_ACQUIRED],
        help="direct | acquired. Derived from the id prefix when omitted.",
    )
    g.add_argument(
        "--source-portfolio-label", dest="source_portfolio_label", default="",
        help="Human-readable label, e.g. \"Direct Book\" or \"Acquired Portfolio 1\".",
    )
    g.add_argument(
        "--acquisition-date", dest="acquisition_date", default="",
        help="Acquisition date (YYYY-MM-DD) for acquired portfolios.",
    )
    g.add_argument(
        "--seller-name", dest="seller_name", default="",
        help="Seller / vendor name for an acquired book (optional).",
    )
    g.add_argument(
        "--allow-unknown-acquisition-date", dest="allow_unknown_acquisition_date",
        action="store_true",
        help="Permit onboarding an acquired portfolio with an unknown "
             "acquisition date (otherwise the run fails closed).",
    )


def provenance_from_args(args, *, required: bool = True) -> Optional[Provenance]:
    """Build a :class:`Provenance` from parsed argparse args.

    Returns ``None`` when no ``source_portfolio_id`` was supplied and
    ``required`` is False (e.g. legacy/back-compat runs). When ``required`` is
    True and the id is missing, raises :class:`ProvenanceError`.
    """
    sid = _clean(getattr(args, "source_portfolio_id", "") or "")
    if not sid:
        if required:
            raise ProvenanceError(
                "source_portfolio_id is mandatory. Pass --source-portfolio-id "
                "(e.g. direct_001 or acquired_001)."
            )
        return None
    return build_provenance(
        source_portfolio_id=sid,
        source_portfolio_type=getattr(args, "source_portfolio_type", "") or None,
        source_portfolio_label=getattr(args, "source_portfolio_label", "") or None,
        acquisition_date=getattr(args, "acquisition_date", "") or None,
        seller_name=getattr(args, "seller_name", "") or None,
        portfolio_cohort=getattr(args, "portfolio_cohort", "") or None,
        allow_unknown_acquisition_date=bool(
            getattr(args, "allow_unknown_acquisition_date", False)
        ),
    )
