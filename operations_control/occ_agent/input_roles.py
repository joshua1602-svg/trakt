"""operations_control.occ_agent.input_roles — the semantic roles of a delivery.

Client Onboarding's field catalogue covers everything about the *client*: who
they are, which portfolios they run, which products they take, how their books
arrive. It says nothing about what a *file* is, because it stops at activation
and never reads one.

This module is that one missing vocabulary, and it is not invented here either:
it is read from ``config/system/workflow_input_requirements.yaml`` — the
administrator-governed declaration of which semantic input roles each workflow
outcome requires, the role labels, and the minimum recognition confidence below
which a file must be confirmed by a human. The same file the live intake route
classifies against.

Adding a source role is therefore a configuration change, here as elsewhere.
Only the *prose* tokens — the words an operator types when they mean "the loan
tape", which a filename classifier never sees — live in this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO = Path(__file__).resolve().parents[2]

INPUT_REQUIREMENTS_PATH = REPO / "config/system/workflow_input_requirements.yaml"


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


def contains(haystack: str, token: str) -> bool:
    """Whole-token containment, so 'erm' does not match inside 'determine'."""
    t = normalise(token).strip()
    if not t:
        return False
    return f" {t} " in normalise(haystack)


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
            if any(contains(lower_text, t) for t in role.tokens):
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
    """Clear the configuration cache (tests that vary configuration)."""
    _artefact_vocabulary.cache_clear()
