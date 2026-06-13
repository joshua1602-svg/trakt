"""
llm_policy.py
=============

PART 3 / PART 8 — load and resolve the low-cost onboarding LLM policy from
``config/system/onboarding_agent.yaml`` and apply CLI overrides / budget
profiles.

The policy is deliberately conservative: ``enabled`` defaults to ``False`` so
onboarding stays fully deterministic unless the LLM review is explicitly opted
in via config or CLI. All caps (calls, items, prompt chars, output tokens) and
the zero-cost-first / uncertainty-budget controls live here so the reviewer is
just an executor of an already-resolved budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "system" / "onboarding_agent.yaml"


@dataclass
class LLMPolicy:
    enabled: bool = False
    provider: str = "anthropic"
    model: str = "claude-haiku-4-5-20251001"
    zero_cost_first: bool = True
    max_llm_calls_per_run: int = 10
    max_items_per_call: int = 10
    max_total_prompt_chars_per_run: int = 50000
    max_total_output_tokens_per_run: int = 4000
    catalogue_mode: str = "compact"
    include_sample_values: str = "redacted_only"
    max_sample_values_per_field: int = 3
    repair_attempts: int = 1
    # skip_llm_if
    deterministic_confidence_above: float = 0.85
    value_match_rate_above: float = 0.98
    field_out_of_scope_skip: bool = True
    # uncertainty_budget
    unresolved_items_budget: int = 25
    do_not_expand_context: bool = True
    create_user_gap_questions: bool = True
    # low profile extra
    prioritise_core_blocking_only: bool = False
    # provenance of the resolved profile
    profile: str = ""

    @property
    def max_total_prompt_chars(self) -> int:
        return self.max_total_prompt_chars_per_run

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def _load_raw(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else _CONFIG_PATH
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_llm_policy(config_path: Path | None = None) -> LLMPolicy:
    """Load the base ``llm_policy`` block (enabled:false by default)."""
    raw = _load_raw(config_path)
    block = raw.get("llm_policy", {}) or {}
    skip = block.get("skip_llm_if", {}) or {}
    budget = block.get("uncertainty_budget", {}) or {}
    return LLMPolicy(
        enabled=bool(block.get("enabled", False)),
        provider=str(block.get("provider", "anthropic")),
        model=str(block.get("model", "claude-haiku-4-5-20251001")),
        zero_cost_first=bool(block.get("zero_cost_first", True)),
        max_llm_calls_per_run=int(block.get("max_llm_calls_per_run", 10)),
        max_items_per_call=int(block.get("max_items_per_call", 10)),
        max_total_prompt_chars_per_run=int(block.get("max_total_prompt_chars_per_run", 50000)),
        max_total_output_tokens_per_run=int(block.get("max_total_output_tokens_per_run", 4000)),
        catalogue_mode=str(block.get("catalogue_mode", "compact")),
        include_sample_values=str(block.get("include_sample_values", "redacted_only")),
        max_sample_values_per_field=int(block.get("max_sample_values_per_field", 3)),
        repair_attempts=int(block.get("repair_attempts", 1)),
        deterministic_confidence_above=float(skip.get("deterministic_confidence_above", 0.85)),
        value_match_rate_above=float(skip.get("value_match_rate_above", 0.98)),
        field_out_of_scope_skip=bool(skip.get("field_out_of_scope", True)),
        unresolved_items_budget=int(budget.get("if_unresolved_items_above", 25)),
        do_not_expand_context=bool(budget.get("then_do_not_expand_context", True)),
        create_user_gap_questions=bool(budget.get("create_user_gap_questions_instead", True)),
    )


def resolve_llm_policy(
    *,
    config_path: Path | None = None,
    enable_llm_review: Optional[bool] = None,
    budget_profile: str = "",
    max_calls: Optional[int] = None,
    max_items_per_call: Optional[int] = None,
) -> LLMPolicy:
    """Resolve the effective policy from config + budget profile + CLI overrides.

    Precedence (lowest to highest): base config block -> named budget profile ->
    explicit CLI flags (``enable_llm_review``, ``max_calls``,
    ``max_items_per_call``). The default remains LLM-off / deterministic-only.
    """
    raw = _load_raw(config_path)
    policy = load_llm_policy(config_path)

    profile = (budget_profile or "").strip().lower()
    if profile:
        profiles = raw.get("llm_budget_profiles", {}) or {}
        prof = profiles.get(profile, {}) or {}
        policy.profile = profile
        if "enabled" in prof:
            policy.enabled = bool(prof["enabled"])
        if "max_llm_calls_per_run" in prof:
            policy.max_llm_calls_per_run = int(prof["max_llm_calls_per_run"])
        if "max_items_per_call" in prof:
            policy.max_items_per_call = int(prof["max_items_per_call"])
        if "prioritise_core_blocking_only" in prof:
            policy.prioritise_core_blocking_only = bool(prof["prioritise_core_blocking_only"])
        if profile == "off":
            policy.enabled = False

    if enable_llm_review is not None:
        policy.enabled = bool(enable_llm_review)
    if max_calls is not None:
        policy.max_llm_calls_per_run = int(max_calls)
    if max_items_per_call is not None:
        policy.max_items_per_call = int(max_items_per_call)

    return policy
