"""mi_agent_pptx — MI Agent-native investor/funder PPTX pack generator.

A standalone module that generates a standardised 12–15 slide institutional
investor/funder PowerPoint deck as a by-product of a completed MI Agent
pipeline run. It consumes the MI Agent canonical registries and post-pipeline
artifacts as the single source of truth.

Design principles (enforced across the module):

* Registries only — no legacy Streamlit (`streamlit_app_erm.py`) dependency,
  no legacy Streamlit state/filters/chart wrappers.
* Config-driven — slides, metrics, chart specs, field bindings, lens
  eligibility and missing-data behaviour live in registries/configs.
* No economic derivations in the PPTX layer beyond aggregation methods the MI
  Agent semantic registry already declares.
* Missing fields produce branded placeholders + appendix notes, never crashes.

Entry point: ``python -m mi_agent_pptx.cli`` (see :mod:`mi_agent_pptx.cli`).
"""

from __future__ import annotations

__version__ = "1.0.0"

from .artifact_loader import RunArtifacts, load_run_artifacts
from .data_resolver import ResolvedData, resolve_data
from .deck_config import DeckConfig, load_deck_config
from .metric_resolver import MetricResolver, MetricResult
from .registry_loader import RegistryLoader, load_registries

__all__ = [
    "__version__",
    "load_run_artifacts",
    "RunArtifacts",
    "resolve_data",
    "ResolvedData",
    "load_deck_config",
    "DeckConfig",
    "RegistryLoader",
    "load_registries",
    "MetricResolver",
    "MetricResult",
]
