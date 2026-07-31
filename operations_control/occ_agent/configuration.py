"""operations_control.occ_agent.configuration — the client configuration candidate.

The output is a **real** client configuration candidate in the shape the
pipeline already consumes (``config/client/config_client_*.yaml``), resolved
through the **existing** configuration hierarchy — not a second format invented
for this feature:

* the layer merge is done by :class:`operations_control.configuration.resolver.
  EffectiveConfigResolver`, which applies the platform's precedence contract
  (``system < regime < asset < client < portfolio < run_decisions``) using the
  existing merge engine in ``config/system/config_resolver.py``;
* the resolver is constructed against the **synthetic** store, so the immutable
  :class:`EffectiveConfiguration` it persists lands in the synthetic container
  and never in the live one;
* the regulatory completeness check is the existing
  :mod:`operations_control.annex2.preflight`, so a product that needs an LEI
  blocks here for the same reason and with the same wording it would live.

Every proposed value carries provenance (§10). A value that is *material* — one
that changes identity, scope, or a regulatory output — or that was inferred with
low confidence is marked ``requires_human_confirmation``; readiness refuses to
pass while any such value is unconfirmed, so an inference cannot slip through on
confidence alone.

Nothing here writes to production configuration. :meth:`ConfigurationService.
promote_to_production` exists only to name the capability the synthetic policy
refuses, so the boundary is exercised rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from ..configuration import resolver as _resolver
from ..configuration.packages import ASSET_MODEL, ConfigPackageStore
from ..contracts import canonical_json, stable_hash
from ..engine import OpsError
from ..rules import RuleStore
from ..stores import OpsStore
from .case import (
    PROV_AGENT_INFERENCE,
    PROV_ASSET_CONFIG,
    PROV_CLIENT_RESPONSE,
    PROV_DEFAULT,
    PROV_HUMAN_INSTRUCTION,
    PROV_PRODUCT_CONFIG,
    ProposedValue,
    SyntheticCase,
)
from .policy import CAP_PRODUCTION_CONFIG_WRITE, SyntheticPolicy
from .store import SyntheticCaseStore
from .vocabulary import asset_vocabulary, product_vocabulary

#: Configuration keys that are material by definition: they change who the
#: report is about, what it covers, or what a regulator receives. A proposed
#: value for one of these always needs an explicit human confirmation.
MATERIAL_KEYS = frozenset({
    "client.client_id",
    "portfolio.asset_class",
    "portfolio.country",
    "portfolio.base_currency",
    "portfolio.static_reporting_date",
    "default_regime",
    "supported_regimes",
    "defaults.originator_legal_entity_identifier",
    "defaults.originator_name",
    "defaults.originator_establishment_country",
})

#: What each material key affects downstream, in operator language.
_IMPACT: Dict[str, str] = {
    "client.client_id":
        "Identifies the client on every report and in every storage location.",
    "portfolio.asset_class":
        "Selects which canonical fields are required and which regimes are "
        "available.",
    "portfolio.country":
        "Selects geographic reference data and regulatory treatment.",
    "portfolio.base_currency":
        "All amounts are validated and presented in it.",
    "portfolio.static_reporting_date":
        "Sets the period every delivery is filed and reported under.",
    "default_regime":
        "Decides which regulatory output Trakt prepares by default.",
    "supported_regimes":
        "Decides which regulatory outputs the client may request at all.",
    "defaults.originator_legal_entity_identifier":
        "Appears in the regulatory submission and identifies the originator.",
    "defaults.originator_name":
        "Appears in the regulatory submission.",
    "defaults.originator_establishment_country":
        "Appears in the regulatory submission.",
    "pipeline.mi_enabled": "Whether Trakt prepares the MI pack.",
    "pipeline.esma_enabled":
        "Whether Trakt prepares the regulatory submission.",
}


class ConfigurationDraftError(OpsError):
    """The configuration candidate could not be drafted."""

    def __init__(self, detail: str):
        super().__init__(
            "OCC_AGENT_CONFIG_DRAFT_FAILED",
            "Trakt could not prepare a configuration for this case. "
            f"({detail})", http_status=400)


@dataclass
class ConfigurationDraft:
    """A drafted client configuration and everything needed to judge it."""

    candidate: Dict[str, Any] = field(default_factory=dict)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    status: str = ""                       # READY | READY_WITH_WARNINGS | BLOCKED
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    effective_config_id: str = ""
    effective_content_hash: str = ""
    resolved_values: Dict[str, Any] = field(default_factory=dict)
    value_provenance: Dict[str, str] = field(default_factory=dict)
    content_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate, "provenance": self.provenance,
            "status": self.status, "blockers": self.blockers,
            "warnings": self.warnings, "conflicts": self.conflicts,
            "effective_config_id": self.effective_config_id,
            "effective_content_hash": self.effective_content_hash,
            "resolved_values": self.resolved_values,
            "value_provenance": self.value_provenance,
            "content_hash": self.content_hash,
        }

    def unconfirmed_material(self) -> List[Dict[str, Any]]:
        return [p for p in self.provenance
                if ProposedValue.from_dict(p).needs_confirmation()]


class ConfigurationService:
    """Drafts and validates the client configuration candidate."""

    def __init__(self, store: SyntheticCaseStore, policy: SyntheticPolicy, *,
                 resolver_factory: Optional[
                     Callable[[SyntheticCase], Any]] = None):
        self.store = store
        self.policy = policy
        self._resolver_factory = resolver_factory

    # ------------------------------------------------------------------ #
    def draft(self, case: SyntheticCase) -> ConfigurationDraft:
        """Build the candidate, then resolve it through the real hierarchy."""
        req = case.confirmed or case.extracted
        if not req.client_name and not case.client_id:
            raise ConfigurationDraftError("the client is not confirmed yet")
        assets = asset_vocabulary()
        products = product_vocabulary()
        selected = [p for p in
                    (products.by_id(pid) for pid in req.required_products)
                    if p is not None]
        outcome = products.outcome_for(req.required_products)
        asset = case.asset_type or req.asset_type
        if asset and asset not in ASSET_MODEL:
            raise ConfigurationDraftError(
                f"the asset type '{asset}' is not configured in Trakt")

        values: List[ProposedValue] = []

        def propose(key: str, value: Any, source: str, *, evidence: str = "",
                    confidence: Optional[float] = None) -> None:
            material = key in MATERIAL_KEYS
            pv = ProposedValue(
                key=key, value=value, source=source, evidence=evidence,
                confidence=confidence, material=material,
                downstream_impact=_IMPACT.get(key, ""))
            pv.requires_human_confirmation = pv.needs_confirmation()
            pv.validate()
            values.append(pv)

        # -- identity ---------------------------------------------------- #
        propose("client.client_id", case.client_id or req.proposed_client_id,
                PROV_HUMAN_INSTRUCTION if req.provenance.get(
                    "proposed_client_id") == PROV_HUMAN_INSTRUCTION
                else PROV_AGENT_INFERENCE,
                evidence="from the confirmed instruction",
                confidence=req.confidence.get("proposed_client_id"))
        propose("client.display_name", req.client_name,
                PROV_HUMAN_INSTRUCTION, evidence="stated by the operator")
        # Never 'production': a synthetic case's candidate must not claim to be
        # a live environment even as text.
        propose("client.environment", "synthetic_candidate", PROV_DEFAULT,
                evidence="synthetic case")

        # -- portfolio --------------------------------------------------- #
        propose("portfolio.asset_class", asset,
                PROV_HUMAN_INSTRUCTION if req.provenance.get("asset_type") ==
                PROV_HUMAN_INSTRUCTION else PROV_AGENT_INFERENCE,
                evidence=f"recognised as {assets.label(asset)}"
                         if asset else "not stated",
                confidence=req.confidence.get("asset_type"))
        # A value the client answered with beats one derived from the
        # instruction: it is the more specific source, and it is what a
        # correction supplies.
        country = _answer_from_case(case, "portfolio.country") or req.jurisdiction
        propose("portfolio.country", country,
                PROV_CLIENT_RESPONSE
                if _answer_from_case(case, "portfolio.country")
                else (PROV_HUMAN_INSTRUCTION if req.jurisdiction
                      else PROV_AGENT_INFERENCE),
                evidence="stated jurisdiction" if country else "not stated")
        answered_currency = _answer_from_case(case, "portfolio.base_currency")
        currency = answered_currency or _currency_for(country)
        propose("portfolio.base_currency", currency,
                PROV_CLIENT_RESPONSE if answered_currency
                else (PROV_DEFAULT if currency else PROV_AGENT_INFERENCE),
                evidence=("supplied by the client" if answered_currency else
                          f"standard currency for {country}" if currency
                          else "no jurisdiction confirmed"),
                confidence=None if currency else 0.0)
        propose("portfolio.static_reporting_date", req.reporting_date,
                PROV_HUMAN_INSTRUCTION if req.reporting_date
                else PROV_AGENT_INFERENCE,
                evidence="confirmed first reporting date"
                         if req.reporting_date else "not stated")

        # -- regimes, from the ASSET support model ------------------------ #
        supported = list(assets.supports_regimes(asset)) if asset else []
        propose("supported_regimes", supported, PROV_ASSET_CONFIG,
                evidence="regimes this asset type supports")
        regime_products = [p for p in selected if p.requires_regime()]
        default_regime = regime_products[0].regime_id if regime_products else ""
        if default_regime:
            propose("default_regime", default_regime, PROV_PRODUCT_CONFIG,
                    evidence=f"required by "
                             f"{regime_products[0].label}")

        # -- pipeline switches, from the selected products ---------------- #
        propose("pipeline.mi_enabled",
                any(p.outcome == "mi" or not p.requires_regime()
                    for p in selected),
                PROV_PRODUCT_CONFIG, evidence="from the product selection")
        propose("pipeline.esma_enabled", bool(default_regime),
                PROV_PRODUCT_CONFIG, evidence="from the product selection")
        propose("pipeline.loan_engine_enabled", False, PROV_DEFAULT,
                evidence="not requested")

        # -- regulatory defaults the regime requires ---------------------- #
        if default_regime:
            for key, question in (
                    ("defaults.originator_legal_entity_identifier",
                     "the originator's legal entity identifier"),
                    ("defaults.originator_name", "the originator's name"),
                    ("defaults.originator_establishment_country",
                     "the originator's country of establishment")):
                answer = _answer_from_case(case, key)
                propose(key, answer,
                        PROV_CLIENT_RESPONSE if answer
                        else PROV_AGENT_INFERENCE,
                        evidence=("supplied by the client" if answer
                                  else f"still needed: {question}"),
                        confidence=None if answer else 0.0)

        candidate = _to_candidate({v.key: v.value for v in values})
        draft = ConfigurationDraft(
            candidate=candidate,
            provenance=[v.to_dict() for v in values])

        # -- resolve through the REAL hierarchy --------------------------- #
        self._resolve_into(case, draft, asset=asset, outcome=outcome,
                           reporting_period=req.reporting_date)

        draft.content_hash = stable_hash(canonical_json(
            {"candidate": draft.candidate, "provenance": draft.provenance}))
        return draft

    # ------------------------------------------------------------------ #
    def _resolve_into(self, case: SyntheticCase, draft: ConfigurationDraft, *,
                      asset: str, outcome: str, reporting_period: str) -> None:
        """Run the existing resolver over the candidate.

        The candidate is materialised to a temporary YAML file inside the case
        sandbox and passed as the resolver's client-config layer, so the
        resolver reads it exactly as it reads a real client config — including
        the regulatory preflight, which is what makes a missing LEI a blocker
        here for the same reason it is one live.
        """
        if not asset:
            draft.status = _resolver.BLOCKED
            draft.blockers = ["The asset type must be confirmed before "
                              "configuration can be resolved."]
            return
        client_id = str(draft.candidate.get("client", {}).get("client_id") or "")
        portfolio_id = case.portfolio_id or case.confirmed.proposed_portfolio_id
        if not client_id or not portfolio_id:
            draft.status = _resolver.BLOCKED
            draft.blockers = ["The client and portfolio identifiers must be "
                              "confirmed before configuration can be "
                              "resolved."]
            return

        candidate_path = self.store.case_dir(case.tenant, case.case_id) \
            / "client_config_candidate.yaml"
        candidate_path.write_text(
            yaml.safe_dump(draft.candidate, sort_keys=True), encoding="utf-8")

        resolver = self._resolver_for(case, candidate_path)
        try:
            outcome_obj = resolver.resolve(
                client_id=client_id, portfolio_id=portfolio_id,
                asset_type=asset, outcome=outcome,
                reporting_period=reporting_period,
                # workflow_id is deliberately empty: passing one makes the
                # resolver PERSIST an effective configuration under a workflow
                # run. A synthetic case has no live workflow, and persisting
                # under a fabricated one would put a synthetic document in a
                # workflow-shaped location.
                workflow_id="", created_by=case.initiating_user)
        except Exception as exc:  # noqa: BLE001 — surfaced, never swallowed
            raise ConfigurationDraftError(
                f"{type(exc).__name__} while resolving the configuration"
            ) from exc

        draft.status = outcome_obj.status
        draft.blockers = list(outcome_obj.blockers)
        draft.warnings = list(outcome_obj.warnings)
        draft.conflicts = list(outcome_obj.conflicts)
        if outcome_obj.effective is not None:
            eff = outcome_obj.effective
            draft.effective_config_id = eff.effective_config_id
            draft.effective_content_hash = eff.content_hash
            draft.resolved_values = eff.resolved_values
            draft.value_provenance = eff.value_provenance
            # Persist the immutable contract inside the SYNTHETIC container,
            # keyed by case rather than by workflow run.
            resolver.persist(case.tenant, f"case/{case.case_id}", eff,
                             status=outcome_obj.status)

    def _resolver_for(self, case: SyntheticCase, candidate_path: Path):
        if self._resolver_factory is not None:
            return self._resolver_factory(case)
        # An OpsStore over the SYNTHETIC container: the resolver's package
        # seeding and effective-config persistence therefore stay inside the
        # synthetic environment.
        synthetic_ops = OpsStore(self.store.storage,
                                 _SyntheticLayout(self.store.container))
        return _resolver.EffectiveConfigResolver(
            synthetic_ops, RuleStore(synthetic_ops),
            ConfigPackageStore(synthetic_ops),
            client_config_path=candidate_path)

    # ------------------------------------------------------------------ #
    def validate(self, draft: ConfigurationDraft) -> List[str]:
        """Schema validation of the candidate itself.

        The resolver validates the merged result; this validates the SHAPE of
        the candidate, so a malformed document is reported as a configuration
        problem rather than as a resolver crash.
        """
        problems: List[str] = []
        candidate = draft.candidate
        if not isinstance(candidate, dict):
            return ["The configuration candidate is not a document."]
        client = candidate.get("client")
        if not isinstance(client, dict) or not client.get("client_id"):
            problems.append("The configuration has no client identifier.")
        portfolio = candidate.get("portfolio")
        if not isinstance(portfolio, dict):
            problems.append("The configuration has no portfolio section.")
        else:
            for key, label in (("asset_class", "asset type"),
                               ("country", "country"),
                               ("base_currency", "reporting currency"),
                               ("static_reporting_date", "reporting date")):
                if not portfolio.get(key):
                    problems.append(f"The configuration is missing the "
                                    f"{label}.")
        regimes = candidate.get("supported_regimes")
        if regimes is not None and not isinstance(regimes, list):
            problems.append("Supported regimes must be a list.")
        default_regime = candidate.get("default_regime")
        if default_regime and default_regime not in (regimes or []):
            problems.append("The default regulatory regime is not one this "
                            "asset type supports.")
        for pv in (ProposedValue.from_dict(p) for p in draft.provenance):
            try:
                pv.validate()
            except OpsError as exc:
                problems.append(exc.message)
        return problems

    # ------------------------------------------------------------------ #
    def promote_to_production(self, case: SyntheticCase, *,
                              actor: str = "") -> None:
        """The live-promotion seam. Always refused in synthetic mode."""
        self.policy.require(CAP_PRODUCTION_CONFIG_WRITE,
                            detail="client configuration",
                            case_id=case.case_id, tenant=case.tenant,
                            actor=actor)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class _SyntheticLayout:
    """An :class:`~operations_control.stores.OpsLayout` substitute pinned to the
    synthetic container.

    ``OpsLayout`` reads its container from the environment via ``from_env``;
    constructing it directly with the synthetic container keeps the resolver's
    writes inside the synthetic environment without touching the live layout.
    """

    def __init__(self, container: str):
        from ..stores import OpsLayout
        self._inner = OpsLayout(container=container)
        self.container = container

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


#: Standard reporting currency by jurisdiction. A deterministic default, not an
#: inference: where the jurisdiction is unknown the value is left empty so the
#: operator is asked.
_CURRENCY_BY_COUNTRY = {"GB": "GBP", "IE": "EUR", "NL": "EUR", "ES": "EUR",
                        "FR": "EUR", "DE": "EUR", "US": "USD"}


def _currency_for(country: str) -> str:
    return _CURRENCY_BY_COUNTRY.get(str(country or "").upper(), "")


def _answer_from_case(case: SyntheticCase, key: str) -> Any:
    """A configuration value supplied through a client answer, if any.

    Answers are recorded on the case under ``client_answers``, keyed by the
    configuration key they satisfy — which is what lets a supplied value close
    the gap the resolver reported.
    """
    answers = case.client_answers or {}
    if isinstance(answers, dict) and key in answers:
        return answers[key]
    return ""


@dataclass(frozen=True)
class ConfigurationKeyVocabulary:
    """The configuration keys an operator may answer, and how they say them.

    Bounded on purpose: a natural-language instruction can only reach a key the
    platform already asks about, so no arbitrary configuration key can be set
    from a sentence.
    """

    #: (phrase, key), longest phrase first.
    phrases: tuple

    def match(self, label: str) -> str:
        from .vocabulary import normalise
        text = normalise(label)
        for phrase, key in self.phrases:
            if f" {normalise(phrase).strip()} " in text:
                return key
        return ""

    def keys(self) -> List[str]:
        return sorted({key for _, key in self.phrases})


#: How an operator refers to each answerable configuration key.
_KEY_PHRASES: Dict[str, tuple] = {
    "defaults.originator_legal_entity_identifier": (
        "originator legal entity identifier", "originator lei", "lei",
        "legal entity identifier"),
    "defaults.originator_name": ("originator name", "originator"),
    "defaults.originator_establishment_country": (
        "originator establishment country", "country of establishment",
        "establishment country"),
    "portfolio.base_currency": ("reporting currency", "base currency",
                                "currency"),
    "portfolio.country": ("jurisdiction", "country"),
}


def configuration_key_vocabulary() -> ConfigurationKeyVocabulary:
    pairs = [(phrase, key) for key, phrases in _KEY_PHRASES.items()
             for phrase in phrases]
    pairs.sort(key=lambda p: (-len(p[0]), p[0]))
    return ConfigurationKeyVocabulary(phrases=tuple(pairs))


def _to_candidate(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Turn dotted keys into the nested client-config document shape."""
    out: Dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out
