"""operations_control.annex2.preflight — regulatory configuration readiness (I3).

A governed check of the static client/reporting attributes the authoritative
Annex 2 route needs BEFORE projection/normalisation — so failures surface as a
concise operator card, not buried inside the XML builder.

Approved values supplied by the operator persist as ``client_rule`` records
through the EXISTING governed rule store (no second configuration store); the
projection stage materialises them as an overlay config file passed to the
projector via its existing ``--config`` flag. Repository configuration files
are never modified.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

LEI_RE = re.compile(r"^[A-Z0-9]{18}[0-9]{2}$")   # ISO 17442 shape
CCY_RE = re.compile(r"^[A-Z]{3}$")
COUNTRY_RE = re.compile(r"^[A-Z]{2}$")


@dataclass
class CheckResult:
    key: str                       # canonical setting key (evidence only)
    label: str                     # operator wording
    status: str                    # ok | needs_review | blocked
    blocking: bool
    found_value: str = ""          # evidence only, may be redacted upstream
    problem: str = ""              # plain-English, shown when not ok

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _deep_find(node: Any, key: str) -> Optional[Any]:
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for v in node.values():
            hit = _deep_find(v, key)
            if hit is not None:
                return hit
    elif isinstance(node, list):
        for v in node:
            hit = _deep_find(v, key)
            if hit is not None:
                return hit
    return None


def run_preflight(*, client_config_path: Path,
                  overrides: Optional[Dict[str, Any]] = None,
                  reporting_period: str = "") -> Dict[str, Any]:
    """Assess regulatory configuration readiness.

    ``overrides`` are approved client_rule settings (setting -> value) from the
    governed rule store; they take precedence over the file.
    """
    cfg: Dict[str, Any] = {}
    if client_config_path and Path(client_config_path).exists():
        cfg = yaml.safe_load(Path(client_config_path).read_text(
            encoding="utf-8")) or {}
    ov = overrides or {}

    def get(key: str) -> str:
        if key in ov and str(ov[key]).strip():
            return str(ov[key]).strip()
        v = _deep_find(cfg, key)
        return "" if v is None else str(v).strip()

    checks: List[CheckResult] = []

    def add(key, label, *, blocking, validator=None, problem_missing,
            problem_invalid=""):
        val = get(key)
        if not val:
            checks.append(CheckResult(key, label, "blocked" if blocking
                                      else "needs_review", blocking,
                                      problem=problem_missing))
        elif validator and not validator(val):
            checks.append(CheckResult(key, label, "blocked" if blocking
                                      else "needs_review", blocking,
                                      found_value=val,
                                      problem=problem_invalid or problem_missing))
        else:
            checks.append(CheckResult(key, label, "ok", False,
                                      found_value=val))

    # These are the ORIGINATOR's details (Annex 2 RREL82/83/84), not the
    # reporting entity's. The two are separate roles that a client may assign
    # to separate legal entities, so naming the wrong one here would send an
    # operator to enter the wrong company's identifier. Keys, validators and
    # blocking behaviour are unchanged; only the wording names the right role.
    add("originator_legal_entity_identifier",
        "Originator identifier (LEI)", blocking=True,
        validator=lambda v: bool(LEI_RE.fullmatch(v)),
        problem_missing="The originator's LEI has not been configured.",
        problem_invalid="The configured originator identifier is not a "
                        "valid 20-character LEI.")
    add("originator_establishment_country",
        "Originator country", blocking=False,
        validator=lambda v: bool(COUNTRY_RE.fullmatch(v)),
        problem_missing="The originator's country has not been confirmed.")
    add("country", "Portfolio jurisdiction", blocking=False,
        validator=lambda v: bool(COUNTRY_RE.fullmatch(v)),
        problem_missing="The portfolio's jurisdiction has not been confirmed.")
    add("base_currency", "Reporting currency", blocking=False,
        validator=lambda v: bool(CCY_RE.fullmatch(v)),
        problem_missing="The reporting currency has not been confirmed.")

    # Reporting date comes from the workflow itself.
    if reporting_period:
        checks.append(CheckResult("reporting_period", "Reporting date", "ok",
                                  False, found_value=reporting_period))
    else:
        checks.append(CheckResult("reporting_period", "Reporting date",
                                  "blocked", True,
                                  problem="No reporting date was provided for "
                                          "this delivery."))

    # Informational: no-data policy presence (non-blocking; the delivery rules
    # carry governed ND defaults regardless).
    nd = _deep_find(cfg, "nd_defaults")
    checks.append(CheckResult(
        "nd_defaults", "No-data policy", "ok" if nd else "needs_review", False,
        found_value=("configured" if nd else ""),
        problem=("" if nd else "No client-specific no-data policy is set; the "
                               "standard regulatory defaults will apply.")))

    blocked = [c for c in checks if c.status == "blocked"]
    review = [c for c in checks if c.status == "needs_review"]
    status = "blocked" if blocked else ("needs_review" if review else "ready")
    n = len(blocked) + len(review)
    summary = ("All regulatory details are in place." if status == "ready" else
               f"{n} regulatory detail{'s' if n != 1 else ''} "
               + ("are needed before Annex 2 XML can be created."
                  if blocked else "should be reviewed."))
    return {"status": status, "summary": summary,
            "checks": [c.to_dict() for c in checks],
            "blocked": [c.to_dict() for c in blocked],
            "needs_review": [c.to_dict() for c in review]}


def materialise_effective_config(*, base_config_path: Path,
                                 overrides: Dict[str, Any],
                                 out_path: Path) -> Path:
    """Write the effective client config = repository config + approved
    client_rule overrides, into run staging. The repository file is read-only;
    the overlay is what the projector receives via its existing --config flag."""
    cfg: Dict[str, Any] = {}
    if base_config_path and Path(base_config_path).exists():
        cfg = yaml.safe_load(Path(base_config_path).read_text(
            encoding="utf-8")) or {}

    def deep_set(node: Dict[str, Any], key: str, value: Any) -> bool:
        """Replace the first existing occurrence of ``key`` anywhere in the
        tree; returns True when replaced."""
        if key in node:
            node[key] = value
            return True
        for v in node.values():
            if isinstance(v, dict) and deep_set(v, key, value):
                return True
        return False

    for key, value in (overrides or {}).items():
        if not deep_set(cfg, key, value):
            cfg[key] = value
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "# GENERATED by operations_control — effective client configuration\n"
        "# (repository config + approved client_rule overrides). Do not edit.\n"
        + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_path
