"""mi_agent_pptx.registry_loader — canonical MI Agent registry access.

This module is the *only* place the PPTX stack reads the MI Agent canonical
registries. It is a thin, read-only loader over the existing registry YAMLs —
it introduces no new business logic and no economic derivations. Everything the
deck needs to know about a field (its human label, format, default aggregation,
weighting field, bucket field), and about a dimension's lens/state eligibility,
is answered from these registries so the deck stays a *by-product* of the MI
Agent semantic layer rather than a parallel source of truth.

Registries consumed (all already present in the repo):

* ``mi_agent/mi_semantics_field_registry.yaml`` — field semantics + metric defs.
* ``config/mi/buckets.yaml``                    — bucket edge definitions.
* ``config/mi/stratification_catalogue.yaml``   — dimension -> field -> bucket -> states.
* ``config/mi/state_library.yaml``              — portfolio-state definitions.
* ``config/routes/mi_route.yaml``               — allowed dimensions / lenses / capabilities.
* ``config/mi/mi_equity_release_uk_applicability.yaml`` — field applicability overlay
  (this is where broker-channel suppressibility is expressed).

No writes, no network, no Streamlit, no LLM. Missing registries degrade to
empty mappings rather than raising, so the deck can still render (with
placeholders) against a partial checkout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Repo root = parent of the ``mi_agent_pptx`` package directory.
REPO_ROOT = Path(__file__).resolve().parents[1]

SEMANTICS_PATH = REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
BUCKETS_PATH = REPO_ROOT / "config" / "mi" / "buckets.yaml"
STRAT_CATALOGUE_PATH = REPO_ROOT / "config" / "mi" / "stratification_catalogue.yaml"
STATE_LIBRARY_PATH = REPO_ROOT / "config" / "mi" / "state_library.yaml"
MI_ROUTE_PATH = REPO_ROOT / "config" / "routes" / "mi_route.yaml"
APPLICABILITY_PATH = (
    REPO_ROOT / "config" / "mi" / "mi_equity_release_uk_applicability.yaml"
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file, returning ``{}`` when absent or empty."""
    try:
        if not path.exists():
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data or {}
    except Exception:  # pragma: no cover - defensive: a malformed registry
        return {}


@dataclass(frozen=True)
class FieldSpec:
    """A resolved view of one semantic-registry field entry."""

    key: str
    canonical_field: str
    role: str = "metric"                 # metric | dimension | date
    business_name: str = ""
    display_name: str = ""
    fmt: str = "string"                  # currency | percent | integer | string | ...
    default_aggregation: Optional[str] = None
    allowed_aggregations: List[str] = field(default_factory=list)
    weight_field: Optional[str] = None
    bucket_field: Optional[str] = None
    mi_tier: str = "core"

    @property
    def label(self) -> str:
        return self.business_name or self.display_name or self.key


class RegistryLoader:
    """Read-only accessor over the MI Agent canonical registries."""

    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = Path(repo_root) if repo_root else REPO_ROOT
        self._semantics = _load_yaml(self._p(SEMANTICS_PATH))
        self._buckets = _load_yaml(self._p(BUCKETS_PATH))
        self._strat = _load_yaml(self._p(STRAT_CATALOGUE_PATH))
        self._states = _load_yaml(self._p(STATE_LIBRARY_PATH))
        self._route = _load_yaml(self._p(MI_ROUTE_PATH))
        self._applicability = _load_yaml(self._p(APPLICABILITY_PATH))

    def _p(self, default_path: Path) -> Path:
        """Rebase a default path onto this loader's repo root."""
        rel = default_path.relative_to(REPO_ROOT)
        return self.repo_root / rel

    # ------------------------------------------------------------------ fields
    @property
    def fields(self) -> Dict[str, Any]:
        return self._semantics.get("fields", {}) or {}

    def field_spec(self, key: str) -> Optional[FieldSpec]:
        """Return the :class:`FieldSpec` for a semantic key, or ``None``."""
        entry = self.fields.get(key)
        if not entry:
            return None
        return FieldSpec(
            key=key,
            canonical_field=entry.get("canonical_field", key),
            role=entry.get("role", "metric"),
            business_name=entry.get("business_name", ""),
            display_name=entry.get("display_name", ""),
            fmt=entry.get("format", "string"),
            default_aggregation=entry.get("default_aggregation"),
            allowed_aggregations=list(entry.get("allowed_aggregations", []) or []),
            weight_field=entry.get("weight_field"),
            bucket_field=entry.get("bucket_field"),
            mi_tier=entry.get("mi_tier", "core"),
        )

    def label_for(self, key: str, fallback: Optional[str] = None) -> str:
        """Human ``business_name`` for a field key, falling back gracefully."""
        spec = self.field_spec(key)
        if spec:
            return spec.label
        return fallback or key.replace("_", " ").title()

    def format_for(self, key: str, fallback: str = "string") -> str:
        spec = self.field_spec(key)
        return spec.fmt if spec else fallback

    def weight_field_for(self, key: str) -> Optional[str]:
        spec = self.field_spec(key)
        return spec.weight_field if spec else None

    def default_aggregation_for(self, key: str) -> Optional[str]:
        spec = self.field_spec(key)
        return spec.default_aggregation if spec else None

    def metric_definitions(self) -> Dict[str, Any]:
        """Named composite metric definitions from the semantics metadata."""
        return (self._semantics.get("metadata", {}) or {}).get(
            "metric_definitions", {}
        ) or {}

    @property
    def default_weight_field(self) -> str:
        return (self._semantics.get("metadata", {}) or {}).get(
            "default_weight_field", "current_outstanding_balance"
        )

    # ----------------------------------------------------------------- buckets
    @property
    def buckets(self) -> Dict[str, Any]:
        return self._buckets.get("buckets", {}) or {}

    def bucket_spec(self, bucket_key: str) -> Optional[Dict[str, Any]]:
        return self.buckets.get(bucket_key)

    # ------------------------------------------------------ stratification/lens
    @property
    def dimensions(self) -> Dict[str, Any]:
        return self._strat.get("dimensions", {}) or {}

    def dimension_spec(self, dimension: str) -> Optional[Dict[str, Any]]:
        return self.dimensions.get(dimension)

    def dimension_applies_to_state(self, dimension: str, state: str) -> bool:
        """Is *dimension* valid for portfolio *state* per the catalogue?

        Unknown dimensions default to ``True`` (permissive) so a deck config
        referencing a field not yet catalogued still renders; a catalogued
        dimension is gated strictly by its ``applies_to_states`` list.
        """
        spec = self.dimension_spec(dimension)
        if not spec:
            return True
        states = spec.get("applies_to_states")
        if not states:
            return True
        return state in states

    def dimension_semantic_field(self, dimension: str) -> Optional[str]:
        spec = self.dimension_spec(dimension)
        return spec.get("semantic_field") if spec else None

    # ------------------------------------------------------------------ states
    @property
    def states(self) -> Dict[str, Any]:
        return self._states.get("states", {}) or {}

    # ------------------------------------------------------------------- route
    def allowed_dimensions(self) -> List[str]:
        return list(self._route.get("allowed_dimensions", []) or [])

    def route_capability(self, name: str) -> Any:
        return self._route.get(name)

    # ----------------------------------------------------- applicability / lens
    def applicability_rule(self, field_name: str) -> Optional[Dict[str, Any]]:
        """Applicability-overlay rule for a canonical field, if declared."""
        for rule in self._applicability.get("rules", []) or []:
            if rule.get("field") == field_name:
                return rule
        return None

    def is_field_suppressible(self, field_name: str) -> bool:
        """True when a field is optional/non-blocking per the overlay.

        Broker channel, for example, is declared ``needs_confirmation`` /
        ``blocking: false`` for the UK equity-release MI pack — i.e. it may be
        absent for acquired/consolidated books and must be suppressed rather
        than block the deck. A field with no rule is treated as *not*
        specifically suppressible (its presence is governed by data coverage).
        """
        rule = self.applicability_rule(field_name)
        if not rule:
            return False
        status = str(rule.get("coverage_status_if_no_source", "")).lower()
        blocking = bool(rule.get("blocking", False))
        suppressible_states = {
            "needs_confirmation",
            "optional_for_mi",
            "not_applicable",
        }
        return (status in suppressible_states) and not blocking


@lru_cache(maxsize=4)
def load_registries(repo_root: Optional[str] = None) -> RegistryLoader:
    """Cached :class:`RegistryLoader` factory (keyed by repo root)."""
    return RegistryLoader(Path(repo_root) if repo_root else None)
