"""operations_control.onboarding.derivation — what Trakt works out for itself.

Three derivations, each reading a table the platform already owns rather than
restating a rule:

  * which reporting products a client is ELIGIBLE for — from the asset support
    model the effective-configuration resolver uses to accept or refuse a regime
    outcome, so onboarding and delivery can never disagree;
  * which source registrations the portfolios imply — one per portfolio and
    dataset, with regulatory scope only where a regime-capable book carries it;
  * where deliveries are expected to arrive — from the production storage layout
    the path parser enforces.

An operator is asked only what genuinely needs a decision.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..configuration.packages import ASSET_MODEL
from ..contracts import REGIME_CAPABLE_DATASETS
from .case import OnboardingCase, source_key
from .catalogue import Catalogue, catalogue

#: The container new deliveries are expected in. The path parser's production
#: layout, not a new convention.
RAW_CONTAINER = "raw-v2"


def product_eligibility(case: OnboardingCase,
                        cat: Optional[Catalogue] = None) -> Dict[str, Any]:
    """Per product: does it apply, is it available, and why.

    ``derived`` marks a product that is not a choice at all — management
    information is prepared for every delivery, because the workflow outcome
    vocabulary has no option without it.
    """
    cat = cat or catalogue()
    chosen = set(case.products)
    portfolios = case.items("portfolios")
    out: Dict[str, Any] = {}

    for key, product in (cat.regime_products or {}).items():
        label = product.get("label", key)
        if product.get("always_applies"):
            out[key] = {"label": label, "applies": True, "eligible": True,
                        "derived": True,
                        "reason": product.get("derived_from", "")}
            continue

        needs_regime = product.get("supported_by_assets_declaring")
        needs_dataset = product.get("requires_dataset")
        reasons: List[str] = []
        eligible = True

        if needs_regime:
            supporting = [p.get("portfolio_id") for p in portfolios
                          if needs_regime in (ASSET_MODEL.get(
                              str(p.get("asset_class") or ""), {})
                              .get("supports_regimes") or [])]
            if supporting:
                reasons.append("Supported by "
                               + ", ".join(str(s) for s in supporting) + ".")
            else:
                eligible = False
                reasons.append("No portfolio's asset class supports this "
                               "regime.")

        if needs_dataset:
            books = [s for s in case.items("sources")
                     if s.get("dataset") == needs_dataset]
            if books:
                reasons.append(f"A {needs_dataset} book is expected.")
            elif portfolios:
                eligible = False
                reasons.append(f"No {needs_dataset} book is expected.")
            else:
                eligible = False
                reasons.append("No portfolio has been added yet.")

        if not portfolios:
            eligible = False
            reasons = ["No portfolio has been added yet."]

        out[key] = {"label": label, "applies": key in chosen,
                    "eligible": eligible, "derived": False,
                    "reason": " ".join(reasons)}
    return out


def derive_sources(case: OnboardingCase) -> List[Dict[str, Any]]:
    """The source registrations the portfolios imply, preserving any answers the
    operator has already given against them.

    One funded registration per portfolio always; a pipeline registration only
    where the operator has said a pipeline book exists. Regulatory scope is set
    from the reporting answers and only ever on a regime-capable dataset.
    """
    existing = {source_key(str(s.get("portfolio_id") or ""),
                           str(s.get("dataset") or "")): s
                for s in case.items("sources")}
    annex2 = "esma_annex2" in case.products
    out: List[Dict[str, Any]] = []

    for p in case.items("portfolios"):
        pid = str(p.get("portfolio_id") or "")
        if not pid:
            continue
        supports = "ESMA_Annex2" in (ASSET_MODEL.get(
            str(p.get("asset_class") or ""), {}).get("supports_regimes") or [])
        for dataset in _datasets_for(p, existing, pid):
            previous = existing.get(source_key(pid, dataset)) or {}
            regime_required = bool(annex2 and supports
                                   and dataset in REGIME_CAPABLE_DATASETS)
            out.append({
                **previous,
                "source_key": source_key(pid, dataset),
                "portfolio_id": pid,
                "dataset": dataset,
                # Carried from the book, so a question about the DELIVERY can be
                # conditioned on how the book was acquired without the condition
                # having to reach across sections. Not collected here: the
                # portfolio owns it, and this is a copy for evaluation.
                "portfolio_type": str(p.get("portfolio_type") or ""),
                "cadence": previous.get("cadence") or "",
                "source_party": previous.get("source_party") or "",
                "regime_required": regime_required,
                "expected_location": blob_prefix(
                    client_id=str((case.answers.get("client") or {})
                                  .get("client_id") or ""),
                    portfolio_type=str(p.get("portfolio_type") or "direct"),
                    dataset=dataset,
                    cadence=str(previous.get("cadence") or "{cadence}"),
                    portfolio_id=pid),
            })
    return out


def _datasets_for(portfolio: Dict[str, Any], existing: Dict[str, Any],
                  pid: str) -> List[str]:
    """A funded book always; a pipeline book once one has been added.

    The pipeline registration is not invented from a portfolio field, because
    whether a client sends pipeline data is a delivery arrangement, not a
    property of the book. The operator adds it on the deliveries step and it is
    preserved here.
    """
    datasets = ["funded"]
    if source_key(pid, "pipeline") in existing:
        datasets.append("pipeline")
    return datasets


def add_pipeline_source(case: OnboardingCase, portfolio_id: str) -> None:
    """Register that a portfolio also sends a pipeline book."""
    if source_key(portfolio_id, "pipeline") in {
            source_key(str(s.get("portfolio_id") or ""),
                       str(s.get("dataset") or ""))
            for s in case.items("sources")}:
        return
    case.items("sources").append({
        "source_key": source_key(portfolio_id, "pipeline"),
        "portfolio_id": portfolio_id, "dataset": "pipeline",
        "cadence": "", "source_party": "", "regime_required": False})


def blob_prefix(*, client_id: str, portfolio_type: str, dataset: str,
                cadence: str, portfolio_id: str,
                container: str = RAW_CONTAINER) -> str:
    """Where deliveries for one registration are expected to arrive.

    The production layout the path parser enforces:
    ``{container}/{client}/{book_type}/{dataset}/{cadence}/{portfolio}/{period}/``.
    Shown so an operator can confirm it; never typed in.
    """
    return (f"{container}/{client_id or '{client}'}/{portfolio_type}/{dataset}/"
            f"{cadence or '{cadence}'}/{portfolio_id}/{{reporting_period}}/")


def apply_derived_defaults(case: OnboardingCase,
                           cat: Optional[Catalogue] = None) -> None:
    """Fill values the catalogue says follow from another answer.

    Only ever fills a BLANK: an operator who has said something explicit is not
    overruled by a default on the next save.
    """
    from .catalogue import evaluate
    cat = cat or catalogue()
    for section in cat.sections:
        if not section.repeatable:
            continue
        for item in case.items(section.key):
            for f in section.fields:
                if not f.derived_default or item.get(f.key) is not None:
                    continue
                item[f.key] = evaluate(f.derived_default, case.answers, item)
