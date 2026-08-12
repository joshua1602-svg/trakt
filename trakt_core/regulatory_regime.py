"""trakt_core.regulatory_regime — which regulatory fields are required, and how hard.

The distinction that makes this useful
--------------------------------------
Regulatory field requirements are not one category. A disclosure template
divides into three, and conflating them turns a blocker into a footnote or a
footnote into a blocker:

* **No fallback permitted.** The field admits no No-Data value at all. If the
  tape does not carry it, no valid submission exists. A hard blocker.
* **ND1-ND4 permitted.** "Not available" may be reported, with a reason. The
  submission is valid; the disclosure is weaker.
* **ND5 permitted.** "Not applicable" may be reported. Often entirely correct
  for the asset class, and frequently not a gap at all.

``nd1_4_allowed`` and ``nd5_allowed`` already live in
``config/regime/annex2_field_universe.yaml``. This module reads them alongside
the ``regime_mapping`` block that 313 canonical fields already carry in
``config/system/fields_registry.yaml``, and joins the two into the one question
a readiness review actually asks: *which required fields would stop a
submission, and does this tape carry them?*

Nothing is invented here
------------------------
No requirement is authored, no field is judged, and no threshold is applied. The
regime says what it requires; this module reads it. That is why it sits in
``trakt_core`` with no pandas and no measurement — the *coverage* measurement
needs a dataframe and lives in the handler, but *what the regime demands* is
configuration and belongs with the other governed configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config_cache

DEFAULT_FIELD_UNIVERSE = "config/regime/annex2_field_universe.yaml"
FIELD_UNIVERSE_ENV = "TRAKT_REGIME_FIELD_UNIVERSE"

DEFAULT_FIELDS_REGISTRY = "config/system/fields_registry.yaml"
FIELDS_REGISTRY_ENV = "TRAKT_FIELDS_REGISTRY"

#: The regime this module currently models. Annex 2 is the loan-level
#: (underlying exposure) template; Annex 12 is deal-level and is a different
#: question that this does not attempt to answer.
DEFAULT_REGIME = "ESMA_Annex2"

# -- how hard a missing field is -------------------------------------------- #
BLOCKING = "blocking"          # no ND fallback: no valid submission without it
DEGRADED = "nd1_4_permitted"   # "not available" may be reported, with a reason
OPTIONAL = "nd5_permitted"     # "not applicable" may be reported
SEVERITIES = (BLOCKING, DEGRADED, OPTIONAL)


@dataclass(frozen=True)
class RegimeField:
    """One field the regime asks for."""

    code: str
    field_name: str = ""
    section: str = ""
    format: str = ""
    nd1_4_allowed: bool = False
    nd5_allowed: bool = False
    priority: str = ""
    #: The canonical field that supplies it, where the registry declares one.
    canonical_field: Optional[str] = None

    @property
    def fallback_severity(self) -> str:
        """How serious an absence is, per the regime's own ND rules."""
        if not self.nd1_4_allowed and not self.nd5_allowed:
            return BLOCKING
        if self.nd1_4_allowed:
            return DEGRADED
        return OPTIONAL

    @property
    def blocking(self) -> bool:
        return self.fallback_severity == BLOCKING

    @property
    def mapped(self) -> bool:
        return bool(self.canonical_field)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "field_name": self.field_name,
            "section": self.section,
            "canonical_field": self.canonical_field,
            "priority": self.priority,
            "nd1_4_allowed": self.nd1_4_allowed,
            "nd5_allowed": self.nd5_allowed,
            "fallback_severity": self.fallback_severity,
        }


@dataclass(frozen=True)
class RegimeModel:
    """The regime's loan-level field requirements, joined to canonical names."""

    regime: str = DEFAULT_REGIME
    fields: Tuple[RegimeField, ...] = ()
    configured: bool = False
    source: str = ""

    def by_code(self, code: str) -> Optional[RegimeField]:
        return next((f for f in self.fields if f.code == code), None)

    @property
    def blocking_fields(self) -> List[RegimeField]:
        return [f for f in self.fields if f.blocking]

    @property
    def mapped_fields(self) -> List[RegimeField]:
        return [f for f in self.fields if f.mapped]

    def describe(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "configured": self.configured,
            "source": self.source,
            "field_count": len(self.fields),
            "blocking_count": len(self.blocking_fields),
            "mapped_to_canonical": len(self.mapped_fields),
            "unmapped": len(self.fields) - len(self.mapped_fields),
        }


def _canonical_index(registry_path: Path, regime: str) -> Dict[str, Tuple[str, str]]:
    """``{regime code: (canonical_field, priority)}`` from the field registry.

    Reversed from ``regime_mapping``, which is authored the other way round
    because a canonical field maps into several annexes at once.
    """
    if not registry_path.exists():
        return {}
    import yaml
    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    fields = raw.get("fields") or raw
    index: Dict[str, Tuple[str, str]] = {}
    for canonical_field, spec in (fields or {}).items():
        mapping = ((spec or {}).get("regime_mapping") or {}).get(regime) or {}
        code = str(mapping.get("code") or "").strip()
        if not code or code in index:
            # First declaration wins. A code claimed twice would make the
            # canonical source of a regulatory field ambiguous, and silently
            # picking the later one would hide that.
            continue
        index[code] = (str(canonical_field), str(mapping.get("priority") or ""))
    return index


def load_regime_model(regime: str = DEFAULT_REGIME,
                      universe_path: Optional[str | Path] = None,
                      registry_path: Optional[str | Path] = None) -> RegimeModel:
    """The regime's field universe joined to canonical field names, cached."""
    universe = Path(universe_path
                    or os.environ.get(FIELD_UNIVERSE_ENV)
                    or DEFAULT_FIELD_UNIVERSE)
    registry = Path(registry_path
                    or os.environ.get(FIELDS_REGISTRY_ENV)
                    or DEFAULT_FIELDS_REGISTRY)

    def _load() -> RegimeModel:
        if not universe.exists():
            return RegimeModel(regime=regime, configured=False)
        import yaml
        raw = yaml.safe_load(universe.read_text(encoding="utf-8")) or {}
        inner = raw.get("annex2_field_universe") or raw
        declared = inner.get("fields") or {}
        canonical = _canonical_index(registry, regime)

        fields: List[RegimeField] = []
        for code, spec in sorted((declared or {}).items()):
            spec = spec or {}
            mapped, priority = canonical.get(str(code), (None, ""))
            fields.append(RegimeField(
                code=str(code),
                field_name=str(spec.get("field_name") or ""),
                section=str(spec.get("section") or ""),
                format=str(spec.get("format") or ""),
                nd1_4_allowed=bool(spec.get("nd1_4_allowed", False)),
                nd5_allowed=bool(spec.get("nd5_allowed", False)),
                priority=priority,
                canonical_field=mapped))

        return RegimeModel(regime=regime, fields=tuple(fields), configured=True,
                           source=str(universe))

    return config_cache.get_or_load(
        f"regime_model:{regime}", universe, _load,
        # The join depends on BOTH files, so the registry's identity has to be
        # part of the key or a registry edit would serve a stale mapping.
        extra_key=(str(registry), *config_cache.file_identity(registry)))
