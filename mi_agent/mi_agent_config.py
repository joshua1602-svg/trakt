#!/usr/bin/env python3
"""
mi_agent_config.py

Environment-driven configuration for the MI Agent's optional LLM path.

Cost control: the default model is a cheap/fast Claude model already used
elsewhere in this repo (the onboarding agent uses ``claude-haiku-4-5-20251001``
as its cheap tier). Everything is overridable via environment variables so no
code change is needed to switch models or providers.

Environment variables
---------------------
ENABLE_LLM_MI_AGENT        "true"/"false"  (default false -> deterministic only)
MI_AGENT_LLM_PROVIDER      anthropic | mock | none   (default anthropic)
MI_AGENT_LLM_MODEL         model id        (default claude-haiku-4-5-20251001)
MI_AGENT_MAX_REPAIR_ATTEMPTS  int          (default 2)
ANTHROPIC_API_KEY          required for provider=anthropic

The app must never hard-fail just because no LLM key is configured: when the
LLM is requested but unavailable, ``get_llm_config`` reports ``available=False``
with a human-readable status + warnings, and callers fall back to the
deterministic parser.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Mapping, Optional

# Cheap default model — matches the repo's existing cheap tier
# (agents/onboarding_agent.py). Overridable via MI_AGENT_LLM_MODEL.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_PROVIDER = "anthropic"
# Cost-conscious default: one repair attempt (override with MI_AGENT_MAX_REPAIR_ATTEMPTS).
DEFAULT_MAX_REPAIR_ATTEMPTS = 1
DEFAULT_CATALOG_MODE = "core"      # core | full
DEFAULT_ZERO_COST_FIRST = True

ENV_ENABLE = "ENABLE_LLM_MI_AGENT"
ENV_PROVIDER = "MI_AGENT_LLM_PROVIDER"
ENV_MODEL = "MI_AGENT_LLM_MODEL"
ENV_MAX_REPAIR = "MI_AGENT_MAX_REPAIR_ATTEMPTS"
ENV_CATALOG_MODE = "MI_AGENT_LLM_CATALOG_MODE"
ENV_ZERO_COST_FIRST = "MI_AGENT_ZERO_COST_FIRST"
ENV_API_KEY = "ANTHROPIC_API_KEY"

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


@dataclass
class LLMConfig:
    enabled: bool                       # ENABLE_LLM_MI_AGENT requested
    provider: str                       # anthropic | mock | none
    model: str
    max_repair_attempts: int
    api_key_present: bool
    available: bool                     # LLM can actually be used now
    status: str                         # human-readable summary for the UI
    catalog_mode: str = DEFAULT_CATALOG_MODE        # core | full (cost control)
    zero_cost_first: bool = DEFAULT_ZERO_COST_FIRST  # try deterministic first
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "model": self.model,
            "max_repair_attempts": self.max_repair_attempts,
            "api_key_present": self.api_key_present,
            "available": self.available,
            "status": self.status,
            "catalog_mode": self.catalog_mode,
            "zero_cost_first": self.zero_cost_first,
            "warnings": list(self.warnings),
        }


def _anthropic_importable() -> bool:
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def get_llm_config(env: Optional[Mapping[str, str]] = None) -> LLMConfig:
    """Resolve the MI-Agent LLM configuration from the environment.

    Never raises: an unusable configuration is reported via ``available=False``
    plus warnings so the UI can fall back to the deterministic parser.
    """
    env = env if env is not None else os.environ

    enabled = str(env.get(ENV_ENABLE, "")).strip().lower() in _TRUTHY
    provider = str(env.get(ENV_PROVIDER, DEFAULT_PROVIDER)).strip().lower() or DEFAULT_PROVIDER
    model = str(env.get(ENV_MODEL, DEFAULT_MODEL)).strip() or DEFAULT_MODEL
    try:
        max_repair = int(env.get(ENV_MAX_REPAIR, DEFAULT_MAX_REPAIR_ATTEMPTS))
    except (TypeError, ValueError):
        max_repair = DEFAULT_MAX_REPAIR_ATTEMPTS
    max_repair = max(0, max_repair)
    api_key_present = bool(str(env.get(ENV_API_KEY, "")).strip())

    catalog_mode = str(env.get(ENV_CATALOG_MODE, DEFAULT_CATALOG_MODE)).strip().lower()
    if catalog_mode not in ("core", "full"):
        catalog_mode = DEFAULT_CATALOG_MODE
    zc = str(env.get(ENV_ZERO_COST_FIRST, "")).strip().lower()
    zero_cost_first = DEFAULT_ZERO_COST_FIRST if zc == "" else (zc in _TRUTHY)

    warnings: List[str] = []
    available = False

    if not enabled:
        status = (f"LLM disabled ({ENV_ENABLE} not 'true') — using the "
                  f"deterministic parser.")
    elif provider == "none":
        status = "LLM provider set to 'none' — using the deterministic parser."
    elif provider == "mock":
        available = True
        status = "LLM enabled in 'mock' provider mode (deterministic test stub)."
    elif provider == "anthropic":
        if not api_key_present:
            warnings.append(
                f"{ENV_API_KEY} is not set; the LLM path is unavailable. "
                f"Falling back to the deterministic parser."
            )
            status = "LLM requested but API key missing — deterministic fallback."
        elif not _anthropic_importable():
            warnings.append(
                "The 'anthropic' package is not installed; the LLM path is "
                "unavailable. Falling back to the deterministic parser."
            )
            status = "LLM requested but 'anthropic' not installed — deterministic fallback."
        else:
            available = True
            status = f"LLM enabled — provider=anthropic, model={model}."
    else:
        warnings.append(
            f"Unknown {ENV_PROVIDER}={provider!r}; using the deterministic parser."
        )
        status = f"Unknown LLM provider {provider!r} — deterministic fallback."

    return LLMConfig(
        enabled=enabled, provider=provider, model=model,
        max_repair_attempts=max_repair, api_key_present=api_key_present,
        available=available, status=status, catalog_mode=catalog_mode,
        zero_cost_first=zero_cost_first, warnings=warnings,
    )
