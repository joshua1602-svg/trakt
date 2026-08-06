"""simulation.assets — asset-class profiles.

An asset profile owns three things and nothing else:

* how a loan of that class is **opened** (its contract, collateral and pricing);
* how it **rolls forward** one month (interest, amortisation, status
  transitions, revaluation, exits);
* how one loan renders into **canonical field space** at a reporting date.

Everything else — seeding, history assembly, reconciliation, source formatting,
assertions — is asset-agnostic and lives outside this package. In particular an
asset profile knows nothing about source dialects: the split between "what the
economics are" and "how a lender happens to write them down" is the property
that makes the canonical-equivalence tests meaningful.

Equipment finance is NOT an asset class here. It is a subtype of
:mod:`simulation.assets.asset_finance`, expressed through contract type,
collateral category and residual-value configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Tuple

from ..models import CaseManifest, LoanState


@dataclass
class GenContext:
    """Everything an asset profile needs for one month of one case."""

    case: CaseManifest
    period_index: int
    reporting_date: str
    #: Economic knobs resolved from the case's declared intent.
    knobs: Any
    #: Months since the history started (== period_index).
    @property
    def elapsed_months(self) -> int:
        return self.period_index


class AssetProfile(Protocol):
    """The contract every asset family implements."""

    asset_class: str
    subtypes: Tuple[str, ...]
    #: Ordered canonical fields this asset emits into the source rendering.
    canonical_fields: Tuple[str, ...]
    #: Product profile id in ``config/asset/product_profiles.yaml``.
    product_profile_id: str

    def open_loan(self, rng, ctx: GenContext, *, loan_id: str,
                  origination_date: str, cohort: str) -> LoanState: ...

    def roll(self, rng, loan: LoanState, ctx: GenContext) -> None: ...

    def canonical_row(self, loan: LoanState, ctx: GenContext) -> Dict[str, Any]: ...

    def risk_tests(self, case: CaseManifest) -> List[Dict[str, Any]]: ...


_REGISTRY: Dict[str, AssetProfile] = {}


def register(profile: AssetProfile) -> AssetProfile:
    _REGISTRY[profile.asset_class] = profile
    return profile


def get_profile(asset_class: str) -> AssetProfile:
    from . import asset_finance, bridge, equity_release  # noqa: F401 - registration
    try:
        return _REGISTRY[asset_class]
    except KeyError as exc:  # pragma: no cover - closed vocabulary
        raise KeyError(f"unknown asset class {asset_class!r}; "
                       f"known: {sorted(_REGISTRY)}") from exc


def all_profiles() -> Dict[str, AssetProfile]:
    from . import asset_finance, bridge, equity_release  # noqa: F401
    return dict(_REGISTRY)


__all__ = ["AssetProfile", "GenContext", "all_profiles", "get_profile", "register"]
