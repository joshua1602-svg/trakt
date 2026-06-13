"""
mode_policy.py
=============

PART 1/2 — load and apply the onboarding mode policy
(``config/system/onboarding_modes.yaml``).

A :class:`ModePolicy` carries the per-mode required fields, gap-category
severity baseline, blocking categories, readiness label and outputs in scope.
The key behaviour is :meth:`ModePolicy.severity_for`, which re-ranks a detected
gap's severity according to the selected mode — the same detected gap is
prioritised differently per mode without changing detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

VALID_MODES = ("mi_mna", "regulatory_mi", "warehouse_securitisation")

_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "system" / "onboarding_modes.yaml"

_SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "blocking": 4}


def severity_rank(sev: str) -> int:
    return _SEVERITY_RANK.get(sev, 0)


@dataclass
class ModePolicy:
    name: str = "regulatory_mi"
    objective: str = ""
    readiness_status_label: str = "ready_for_regulatory_handoff"
    required_config_fields: List[str] = field(default_factory=list)
    high_priority_fields: List[str] = field(default_factory=list)
    field_groups_required: List[str] = field(default_factory=list)
    file_types_expected: List[str] = field(default_factory=list)
    gap_category_severity: Dict[str, str] = field(default_factory=dict)
    blocking_gap_categories: List[str] = field(default_factory=list)
    recommended_outputs: List[str] = field(default_factory=list)
    optional_outputs: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def severity_for(self, category: str, detected_severity: str) -> str:
        """Re-rank a detected gap severity for this mode.

        Rule: take the mode baseline for the category; if the detector flagged a
        critical (blocking) gap AND the category is allowed to block in this
        mode, escalate to blocking.
        """
        baseline = self.gap_category_severity.get(category, detected_severity)
        if detected_severity == "blocking" and category in self.blocking_gap_categories:
            return "blocking"
        return baseline

    def is_in_scope_config_field(self, field_name: str) -> bool:
        """A config field is in scope if required or recommended for this mode."""
        return field_name in self.required_config_fields


# ---------------------------------------------------------------------------


def _load_raw(policy_path: Path | None = None) -> dict:
    path = Path(policy_path) if policy_path else _POLICY_PATH
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def default_mode(policy_path: Path | None = None) -> str:
    raw = _load_raw(policy_path)
    return raw.get("default_mode", "regulatory_mi")


def load_mode_policy(mode: str, policy_path: Path | None = None) -> ModePolicy:
    """Load the :class:`ModePolicy` for ``mode``; falls back to defaults safely."""
    raw = _load_raw(policy_path)
    modes = raw.get("modes", {}) or {}
    if mode not in modes:
        mode = raw.get("default_mode", "regulatory_mi")
    m = modes.get(mode, {}) or {}
    return ModePolicy(
        name=mode,
        objective=str(m.get("objective", "")).strip(),
        readiness_status_label=m.get("readiness_status_label", "requires_review"),
        required_config_fields=list(m.get("required_config_fields", []) or []),
        high_priority_fields=list(m.get("high_priority_fields", []) or []),
        field_groups_required=list(m.get("field_groups_required", []) or []),
        file_types_expected=list(m.get("file_types_expected", []) or []),
        gap_category_severity=dict(m.get("gap_category_severity", {}) or {}),
        blocking_gap_categories=list(m.get("blocking_gap_categories", []) or []),
        recommended_outputs=list(m.get("recommended_outputs", []) or []),
        optional_outputs=list(m.get("optional_outputs", []) or []),
    )
