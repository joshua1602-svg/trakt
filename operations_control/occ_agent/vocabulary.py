"""operations_control.occ_agent.vocabulary — what the agent can recognise, from config.

Assets, products, delivery frequencies and source-artefact roles are read from
the configuration the platform already uses, not from lists in this feature:

* **assets** — :data:`operations_control.configuration.packages.ASSET_MODEL`
  (the label and the regimes each asset supports), with recognition synonyms
  from ``config/asset/product_profiles.yaml`` (each profile's ``match`` block:
  ``asset_class``, ``product_type``, ``signal_tokens``);
* **products** — the OCC delivery outcomes (``OUTCOMES``) plus the capability
  vocabulary declared in ``product_profiles.yaml``, so a new capability becomes
  a recognisable product without code change;
* **frequencies** — ``apps.blob_trigger_app.path_parser.VALID_FREQUENCIES``,
  the same list the governed storage path is validated against;
* **artefact roles** — ``config/system/workflow_input_requirements.yaml``
  (``role_labels`` and the per-workflow required/optional roles), which is the
  administrator-governed definition of what an input pack must contain.

Adding an asset, a product or a source role is therefore a configuration change.
Nothing here hard-codes a client, an asset type or a reporting product.

Lookups are cached per configuration path because they are read on every
interpretation; :func:`reset_cache` clears it for tests that vary the config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..contracts import OUTCOME_MI, OUTCOME_MI_ANNEX2
from ..configuration.packages import ASSET_MODEL

REPO = Path(__file__).resolve().parents[2]

PRODUCT_PROFILES_PATH = REPO / "config/asset/product_profiles.yaml"
INPUT_REQUIREMENTS_PATH = REPO / "config/system/workflow_input_requirements.yaml"

#: Confidence attached to a match on a profile's explicit signal token vs a
#: looser asset-class/product-type synonym. Mirrors the onboarding agent's own
#: apply/confirm split (product_profiles.yaml `defaults`).
_SIGNAL_CONFIDENCE = 0.95
_SYNONYM_CONFIDENCE = 0.7


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def normalise(text: str) -> str:
    """Lower-cased, with every run of non-alphanumerics collapsed to a space.

    Both sides of a match go through this, so ``equity-release``,
    ``equity_release`` and ``equity release`` are the same phrase, and a token
    can be compared on word boundaries without worrying about punctuation.
    """
    return " " + re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip() + " "


def _contains(haystack: str, token: str) -> bool:
    """Whole-token containment, so 'erm' does not match inside 'determine'."""
    t = normalise(token).strip()
    if not t:
        return False
    return f" {t} " in normalise(haystack)


# --------------------------------------------------------------------------- #
# Assets
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AssetVocabulary:
    """Asset types the platform is configured for, and how to recognise them."""

    #: asset key -> label (from ASSET_MODEL).
    labels: Dict[str, str]
    #: asset key -> supported regime ids.
    regimes: Dict[str, Tuple[str, ...]]
    #: (token, asset key, confidence), longest token first.
    tokens: Tuple[Tuple[str, str, float], ...]

    def keys(self) -> List[str]:
        return sorted(self.labels)

    def label(self, asset: str) -> str:
        return self.labels.get(asset, str(asset).replace("_", " "))

    def supports_regimes(self, asset: str) -> Tuple[str, ...]:
        return self.regimes.get(asset, ())

    def match(self, lower_text: str) -> Tuple[str, float]:
        """Best asset match in ``lower_text``, or ``("", 0.0)``."""
        for token, asset, confidence in self.tokens:
            if _contains(lower_text, token):
                return asset, confidence
        return "", 0.0


@lru_cache(maxsize=4)
def _asset_vocabulary(profiles_path: str) -> AssetVocabulary:
    labels = {k: str(v.get("label") or k) for k, v in ASSET_MODEL.items()}
    regimes = {k: tuple(v.get("supports_regimes") or ())
               for k, v in ASSET_MODEL.items()}
    tokens: List[Tuple[str, str, float]] = []
    # The asset key itself, spelled out, is always recognised.
    for key in labels:
        tokens.append((key.replace("_", " "), key, _SIGNAL_CONFIDENCE))
        tokens.append((labels[key].lower(), key, _SIGNAL_CONFIDENCE))

    profiles = (_load_yaml(Path(profiles_path)).get("profiles") or {})
    for _pid, profile in profiles.items():
        match = (profile or {}).get("match") or {}
        synonyms = [str(s) for s in (match.get("asset_class") or [])]
        # A profile belongs to whichever configured asset its asset_class list
        # names; unmatched profiles contribute no asset tokens.
        owner = next((k for k in labels
                      if k in {s.lower() for s in synonyms}), "")
        if not owner:
            continue
        for token in match.get("signal_tokens") or []:
            tokens.append((str(token).lower(), owner, _SIGNAL_CONFIDENCE))
        for group, confidence in (("asset_class", _SIGNAL_CONFIDENCE),
                                  ("product_type", _SYNONYM_CONFIDENCE)):
            for token in match.get(group) or []:
                tokens.append((str(token).replace("_", " ").lower(), owner,
                               confidence))
    # Longest first so "lifetime mortgage" beats "mortgage". Where the same
    # token appears in more than one group (a signal token that is also a
    # product-type synonym), the strongest evidence wins.
    unique: Dict[Tuple[str, str], float] = {}
    for t, a, c in tokens:
        if not t:
            continue
        unique[(t, a)] = max(unique.get((t, a), 0.0), c)
    ordered = sorted(((t, a, c) for (t, a), c in unique.items()),
                     key=lambda x: (-len(x[0]), x[0]))
    return AssetVocabulary(labels=labels, regimes=regimes,
                           tokens=tuple(ordered))


def asset_vocabulary(profiles_path: Optional[Path] = None) -> AssetVocabulary:
    return _asset_vocabulary(str(profiles_path or PRODUCT_PROFILES_PATH))


# --------------------------------------------------------------------------- #
# Products
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Product:
    """One Trakt product a client can be onboarded onto."""

    product_id: str
    label: str
    #: The OCC delivery outcome this product implies (``mi`` / ``mi_annex2``).
    outcome: str
    #: The onboarding capability it depends on (product_profiles.yaml).
    capability: str = ""
    #: The regime it delivers under, if any.
    regime_id: str = ""
    #: Recognition tokens.
    tokens: Tuple[str, ...] = ()
    #: Extra onboarding questions this product requires beyond the base set.
    questions: Tuple[str, ...] = ()

    def requires_regime(self) -> bool:
        return bool(self.regime_id)


@dataclass(frozen=True)
class ProductVocabulary:
    products: Tuple[Product, ...]

    def by_id(self, product_id: str) -> Optional[Product]:
        return next((p for p in self.products if p.product_id == product_id),
                    None)

    def ids(self) -> List[str]:
        return [p.product_id for p in self.products]

    def match_all(self, lower_text: str) -> List[str]:
        """Every product named in the text, in catalogue order."""
        hits: List[str] = []
        for product in self.products:
            if any(_contains(lower_text, t) for t in product.tokens):
                hits.append(product.product_id)
        return hits

    def outcome_for(self, product_ids: List[str]) -> str:
        """The OCC delivery outcome the selected products together imply.

        A regulatory product upgrades the outcome; MI alone stays MI. This is
        how a product selection reaches the existing pipeline vocabulary
        without inventing a second one.
        """
        for pid in product_ids:
            product = self.by_id(pid)
            if product and product.outcome == OUTCOME_MI_ANNEX2:
                return OUTCOME_MI_ANNEX2
        return OUTCOME_MI


#: The product catalogue, built from the capability vocabulary. Each entry names
#: the capability it needs, so a capability added to product_profiles.yaml can be
#: surfaced as a product by adding its presentation row here — and a capability
#: REMOVED from configuration disappears from the catalogue automatically.
_CAPABILITY_PRODUCTS: Dict[str, Dict[str, Any]] = {
    "base_mi": {
        "product_id": "portfolio_mi",
        "label": "Portfolio MI",
        "outcome": OUTCOME_MI,
        "tokens": ("portfolio mi", "management information", "monthly mi",
                   "mi reporting", "mi pack", "portfolio reporting"),
        "questions": (
            "Which portfolio groupings should the MI pack report on?",
            "Which reporting currency should the MI pack use?",
        ),
    },
    "pipeline_mi": {
        "product_id": "pipeline_mi",
        "label": "Pipeline MI",
        "outcome": OUTCOME_MI,
        "tokens": ("pipeline mi", "pipeline reporting", "pipeline tape",
                   "application pipeline"),
        "questions": (
            "At which stage does a case enter the reported pipeline?",
            "Which pipeline stages count as committed?",
        ),
    },
    "risk_monitor": {
        "product_id": "risk_monitoring",
        "label": "Risk monitoring",
        "outcome": OUTCOME_MI,
        "tokens": ("risk monitoring", "risk monitor", "risk limits",
                   "covenant monitoring"),
        "questions": (
            "Which risk limits apply to this portfolio?",
            "Who should be notified when a limit is breached?",
        ),
    },
    "risk_migration": {
        "product_id": "risk_migration",
        "label": "Risk migration",
        "outcome": OUTCOME_MI,
        "tokens": ("risk migration", "migration analysis"),
        "questions": ("Which prior period should migration be measured "
                      "against?",),
    },
    "spv_segmentation": {
        "product_id": "investor_reporting",
        "label": "Investor reporting",
        "outcome": OUTCOME_MI,
        "tokens": ("investor reporting", "investor report", "warehouse "
                   "reporting", "spv reporting", "noteholder reporting"),
        "questions": (
            "Which funding vehicles or SPVs should each loan be attributed to?",
            "Which investor reporting date convention applies?",
            "Which counterparties receive the investor report?",
        ),
    },
    "mna_segmentation": {
        "product_id": "portfolio_diligence",
        "label": "Portfolio diligence",
        "outcome": OUTCOME_MI,
        "tokens": ("diligence", "m&a", "acquisition analysis"),
        "questions": ("Which acquisition cohorts should be reported "
                      "separately?",),
    },
    "regulatory_reporting": {
        "product_id": "regulatory_reporting",
        "label": "Regulatory reporting",
        "outcome": OUTCOME_MI_ANNEX2,
        "regime_id": "ESMA_Annex2",
        "tokens": ("regulatory reporting", "regulatory report", "esma",
                   "annex 2", "annex2", "securitisation reporting"),
        "questions": (
            "Which regulator and annex does the client report under?",
            "What is the originator's legal entity identifier?",
            "In which country is the originator established?",
            "What is the securitisation identifier?",
        ),
    },
}


@lru_cache(maxsize=4)
def _product_vocabulary(profiles_path: str) -> ProductVocabulary:
    configured = [str(c) for c in
                  (_load_yaml(Path(profiles_path)).get("capabilities") or [])]
    products: List[Product] = []
    for capability in configured:
        spec = _CAPABILITY_PRODUCTS.get(capability)
        if not spec:
            continue
        products.append(Product(
            product_id=spec["product_id"], label=spec["label"],
            outcome=spec["outcome"], capability=capability,
            regime_id=spec.get("regime_id", ""),
            tokens=tuple(spec.get("tokens") or ()),
            questions=tuple(spec.get("questions") or ())))
    return ProductVocabulary(products=tuple(products))


def product_vocabulary(profiles_path: Optional[Path] = None) -> ProductVocabulary:
    return _product_vocabulary(str(profiles_path or PRODUCT_PROFILES_PATH))


# --------------------------------------------------------------------------- #
# Frequencies
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FrequencyVocabulary:
    values: Tuple[str, ...]
    tokens: Tuple[Tuple[str, str], ...]

    def match(self, lower_text: str) -> str:
        for token, value in self.tokens:
            if _contains(lower_text, token):
                return value
        return ""


@lru_cache(maxsize=1)
def frequency_vocabulary() -> FrequencyVocabulary:
    from apps.blob_trigger_app.path_parser import VALID_FREQUENCIES
    tokens: List[Tuple[str, str]] = []
    for value in VALID_FREQUENCIES:
        tokens.append((value.replace("_", " "), value))
        # The adverb form operators actually type.
        adverb = {"monthly": "each month", "weekly": "each week",
                  "daily": "each day"}.get(value)
        if adverb:
            tokens.append((adverb, value))
    tokens.sort(key=lambda t: (-len(t[0]), t[0]))
    return FrequencyVocabulary(values=tuple(VALID_FREQUENCIES),
                               tokens=tuple(tokens))


# --------------------------------------------------------------------------- #
# Source artefact roles
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ArtefactRole:
    role: str
    label: str
    required_for: Tuple[str, ...] = ()     # OCC outcomes requiring this role
    optional_for: Tuple[str, ...] = ()
    tokens: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ArtefactVocabulary:
    roles: Tuple[ArtefactRole, ...] = ()
    #: Minimum recognition confidence for a role to be satisfied without an
    #: operator confirmation (from workflow_input_requirements.yaml).
    min_confidence: float = 0.4

    def by_role(self, role: str) -> Optional[ArtefactRole]:
        return next((r for r in self.roles if r.role == role), None)

    def label(self, role: str) -> str:
        r = self.by_role(role)
        return r.label if r else str(role).replace("_", " ")

    def required_roles(self, outcome: str) -> List[str]:
        return [r.role for r in self.roles if outcome in r.required_for]

    def optional_roles(self, outcome: str) -> List[str]:
        return [r.role for r in self.roles if outcome in r.optional_for]

    def match_all(self, lower_text: str) -> List[str]:
        hits: List[str] = []
        for role in self.roles:
            if any(_contains(lower_text, t) for t in role.tokens):
                if role.role not in hits:
                    hits.append(role.role)
        return hits


#: Extra recognition tokens per role, for the words an operator uses in prose
#: that the filename classifier would not see. The ROLES themselves come from
#: configuration; only the phrasing lives here.
_ROLE_PROSE_TOKENS: Dict[str, Tuple[str, ...]] = {
    "loan_extract": ("loan tape", "loan extract", "loan book", "loanbook",
                     "portfolio tape", "loan report", "loan level data"),
    "property_extract": ("property tape", "property extract", "valuation tape",
                         "valuation extract", "ivsr", "ivsr actuals",
                         "indexed valuation"),
    "collateral_extract": ("collateral tape", "collateral extract",
                           "security schedule"),
    "cashflow_extract": ("cashflow", "cash flow", "executed cashflows",
                         "cashflow tape", "cash-flow tape"),
    "funder_pi_extract": ("funder principal and interest", "funder p&i",
                          "principal and interest tape", "funder tape"),
    "pipeline_report": ("pipeline tape", "pipeline report", "application tape",
                        "kfi"),
}


@lru_cache(maxsize=4)
def _artefact_vocabulary(requirements_path: str) -> ArtefactVocabulary:
    doc = _load_yaml(Path(requirements_path))
    labels = doc.get("role_labels") or {}
    workflows = doc.get("workflows") or {}
    seen: Dict[str, Dict[str, Any]] = {}
    for outcome, spec in workflows.items():
        for role in (spec or {}).get("required_roles") or []:
            entry = seen.setdefault(role, {"required": [], "optional": []})
            entry["required"].append(outcome)
        for role in (spec or {}).get("optional_roles") or []:
            entry = seen.setdefault(role, {"required": [], "optional": []})
            entry["optional"].append(outcome)
    # Roles that only have prose tokens (e.g. a pipeline tape, which is a
    # dataset rather than a required role) are still recognisable in an
    # instruction, so the agent can record what the client said it will send.
    for role in _ROLE_PROSE_TOKENS:
        seen.setdefault(role, {"required": [], "optional": []})

    roles = tuple(
        ArtefactRole(
            role=role,
            label=str(labels.get(role) or role.replace("_", " ").title()),
            required_for=tuple(entry["required"]),
            optional_for=tuple(entry["optional"]),
            tokens=_ROLE_PROSE_TOKENS.get(role, (role.replace("_", " "),)))
        for role, entry in sorted(seen.items()))
    try:
        min_conf = float(doc.get("min_required_role_confidence", 0.4))
    except (TypeError, ValueError):
        min_conf = 0.4
    return ArtefactVocabulary(roles=roles, min_confidence=min_conf)


def artefact_vocabulary(
        requirements_path: Optional[Path] = None) -> ArtefactVocabulary:
    return _artefact_vocabulary(str(requirements_path
                                    or INPUT_REQUIREMENTS_PATH))


def reset_cache() -> None:
    """Clear the configuration caches (tests that vary configuration)."""
    _asset_vocabulary.cache_clear()
    _product_vocabulary.cache_clear()
    _artefact_vocabulary.cache_clear()
    frequency_vocabulary.cache_clear()
