"""operations_control.occ_agent.derive — execution facts, read off the case.

The synthetic execution needs a handful of facts the pipeline is written in
terms of: which client, which portfolio, which asset class, which delivery
outcome, which regime, which cadence. Every one of them is already answered on
the onboarding case — this module reads them, and defines nothing of its own.

Two boundaries matter:

* **No second product catalogue.** Products are the ones Client Onboarding's
  catalogue declares (``config/regime/onboarding_standing_fields.yaml``), and
  the delivery outcome is derived from what those products say about themselves
  — ``supported_by_assets_declaring`` and ``regime_id`` — rather than from a
  mapping restated here.
* **Nothing delivery-specific.** A reporting period, and which portfolio a
  particular run is for, are properties of a *delivery*, not of a client's
  standing configuration. Client Onboarding rightly does not collect them, so
  they live on the synthetic run and are passed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..configuration.packages import ASSET_MODEL
from ..contracts import (
    BATCH_DATASET_DEFAULT,
    BATCH_FREQUENCY_DEFAULT,
    OUTCOME_MI,
    OUTCOME_MI_ANNEX2,
)
from ..onboarding.case import OnboardingCase
from ..onboarding.catalogue import Catalogue, catalogue


@dataclass
class ExecutionFacts:
    """What a synthetic run needs to know about the client it is running for."""

    client_id: str = ""
    client_name: str = ""
    portfolio_id: str = ""
    portfolio_name: str = ""
    asset_class: str = ""
    dataset: str = BATCH_DATASET_DEFAULT
    cadence: str = BATCH_FREQUENCY_DEFAULT
    jurisdiction: str = ""
    products: List[str] = field(default_factory=list)
    outcome: str = OUTCOME_MI
    regime: str = ""
    #: Why each of the above is what it is, for the operator-facing panel.
    basis: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id, "client_name": self.client_name,
            "portfolio_id": self.portfolio_id,
            "portfolio_name": self.portfolio_name,
            "asset_class": self.asset_class, "dataset": self.dataset,
            "cadence": self.cadence, "jurisdiction": self.jurisdiction,
            "products": list(self.products), "outcome": self.outcome,
            "regime": self.regime, "basis": dict(self.basis),
        }

    @property
    def complete(self) -> bool:
        """Enough to run against. Identity and asset class are the minimum."""
        return bool(self.client_id and self.portfolio_id and self.asset_class)


def product_labels(cat: Optional[Catalogue] = None) -> Dict[str, str]:
    cat = cat or catalogue()
    return {key: str((spec or {}).get("label") or key)
            for key, spec in (cat.regime_products or {}).items()}


def product_label(product_id: str, cat: Optional[Catalogue] = None) -> str:
    return product_labels(cat).get(product_id,
                                   str(product_id).replace("_", " ").title())


def regime_for(products: List[str], asset_class: str = "",
               cat: Optional[Catalogue] = None) -> str:
    """The regime the selected products deliver under, if any.

    Read from the products' own declarations: ``regime_id`` where a product
    names one outright, and ``supported_by_assets_declaring`` where it names the
    regime capability an asset must declare to support it. The asset model is
    then consulted so a product the asset cannot support contributes no regime.
    """
    cat = cat or catalogue()
    supported = tuple((ASSET_MODEL.get(str(asset_class or ""), {})
                       .get("supports_regimes") or ()))
    for product_id in products:
        spec = (cat.regime_products or {}).get(product_id) or {}
        declared = str(spec.get("regime_id") or "")
        needs = str(spec.get("supported_by_assets_declaring") or "")
        candidate = declared or needs
        if not candidate:
            continue
        if needs and asset_class and needs not in supported:
            continue
        return candidate
    return ""


def outcome_for(products: List[str], asset_class: str = "",
                cat: Optional[Catalogue] = None) -> str:
    """The OCC delivery outcome the selected products imply.

    Management information is produced for every delivery, so MI alone is
    ``mi``; a product that requires the Annex 2 regime capability upgrades it.
    This is how a product selection reaches the existing pipeline vocabulary
    without inventing a second one.
    """
    cat = cat or catalogue()
    supported = tuple((ASSET_MODEL.get(str(asset_class or ""), {})
                       .get("supports_regimes") or ()))
    for product_id in products:
        spec = (cat.regime_products or {}).get(product_id) or {}
        needs = str(spec.get("supported_by_assets_declaring") or "")
        if not needs:
            continue
        if asset_class and needs not in supported:
            continue
        if needs == "ESMA_Annex2":
            return OUTCOME_MI_ANNEX2
    return OUTCOME_MI


def facts(case: OnboardingCase, *, portfolio_id: str = "",
          dataset: str = BATCH_DATASET_DEFAULT,
          cat: Optional[Catalogue] = None) -> ExecutionFacts:
    """Read the execution facts off one onboarding case."""
    cat = cat or catalogue()
    portfolios = case.items("portfolios")
    portfolio: Dict[str, Any] = {}
    if portfolio_id:
        portfolio = case.portfolio(portfolio_id) or {}
    elif portfolios:
        portfolio = portfolios[0]

    chosen_id = str(portfolio.get("portfolio_id") or portfolio_id or "")
    asset_class = str(portfolio.get("asset_class") or "")
    products = list(case.products)

    source = next((s for s in case.items("sources")
                   if str(s.get("portfolio_id") or "") == chosen_id
                   and str(s.get("dataset") or "") == dataset), {})
    cadence = str(source.get("cadence") or "") or BATCH_FREQUENCY_DEFAULT

    out = ExecutionFacts(
        client_id=str(case.client_id or ""),
        client_name=str(case.client_name or ""),
        portfolio_id=chosen_id,
        portfolio_name=str(portfolio.get("display_name") or ""),
        asset_class=asset_class,
        dataset=dataset,
        cadence=cadence,
        jurisdiction=str((case.answers.get("client") or {})
                         .get("jurisdiction") or ""),
        products=products,
        outcome=outcome_for(products, asset_class, cat),
        regime=regime_for(products, asset_class, cat),
    )
    out.basis = {
        "client_id": "the onboarding case's client identifier",
        "portfolio_id": ("the portfolio selected for this run" if portfolio_id
                         else "the case's first portfolio"),
        "asset_class": "the portfolio's declared asset class",
        "cadence": ("the expected delivery cadence" if source
                    else "no delivery is declared for this book; the platform "
                         "default applies"),
        "products": "the reporting products the client selected",
        "outcome": "derived from the selected products and the asset model",
        "regime": ("derived from the products' own regime declarations"
                   if out.regime else "no product requires a regime"),
    }
    return out


def portfolio_choices(case: OnboardingCase) -> List[Dict[str, str]]:
    """The portfolios a run could be for, for an operator to choose between."""
    return [{"portfolio_id": str(p.get("portfolio_id") or ""),
             "label": str(p.get("display_name") or p.get("portfolio_id") or ""),
             "asset_class": str(p.get("asset_class") or "")}
            for p in case.items("portfolios") if p.get("portfolio_id")]
