"""mi_agent.borrowing_base.config — the governed facility configuration loader.

Precedence, highest first, always disclosed on the loaded object:

1. ``config/client/config_client_<CLIENT>.yaml`` → ``funding_facility:`` — the
   block OCC onboarding writes from the operator-approved facility answers.
   The client's own governed configuration wins.
2. ``config/risk/funding_facilities.yaml`` → the platform-level register.
3. Nothing — the client has no facility. Every caller degrades to "no funding
   facility is configured", never to a default facility.

There is no fourth source. A facility term never comes from Python, from an
environment variable, or from a document read at runtime.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .models import (
    ENV_PRODUCTION,
    POPULATION_ELIGIBLE,
    TREATMENT_MONITOR_ONLY,
    EligibilityRule,
    FacilityConfiguration,
)

logger = logging.getLogger("mi_agent.borrowing_base.config")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACILITIES_PATH = REPO_ROOT / "config" / "risk" / "funding_facilities.yaml"
CLIENT_CONFIG_DIR = REPO_ROOT / "config" / "client"

#: Environment override for the platform register (tests, alternative deploys).
FACILITIES_PATH_ENV = "TRAKT_FUNDING_FACILITIES_PATH"
#: Environment override for the client-configuration directory.
CLIENT_CONFIG_DIR_ENV = "TRAKT_CLIENT_CONFIG_DIR"

#: The block name inside a client configuration document.
CLIENT_CONFIG_BLOCK = "funding_facility"


def _yaml(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("facility configuration unreadable at %s: %s", path, exc)
        return {}


def _number(value: Any) -> Optional[float]:
    """A configured amount, or None. A blank / null / unparseable value is
    None — never 0.0, which would read as a real zero commitment or a real
    zero drawing."""
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "").replace("£", "").strip())
    except (TypeError, ValueError):
        return None


def _advance_rate(value: Any) -> Optional[float]:
    """The advance rate as a RATIO.

    Configuration states a ratio (1.03). A value above 2 is read as a
    percentage and divided by 100, because "103" can only mean 103% — no
    facility advances 10,300% — and refusing it outright would fail a
    reasonable operator entry with no upside. Anything at or below 2 is taken
    verbatim: 1.03 is 103%, and 0.85 is 85%.
    """
    number = _number(value)
    if number is None:
        return None
    return number / 100.0 if number > 2.0 else number


def _rules(raw: Any) -> List[EligibilityRule]:
    out: List[EligibilityRule] = []
    for item in raw or []:
        if isinstance(item, dict):
            out.append(EligibilityRule.from_dict(item))
    return out


def _build(entry: Dict[str, Any], *, client_id: str, source: str,
           config_version: str) -> FacilityConfiguration:
    eligibility = entry.get("eligibility") or {}
    concentration = entry.get("concentration") or {}
    return FacilityConfiguration(
        client_id=str(entry.get("client_id") or client_id or "").strip(),
        facility_id=str(entry.get("facility_id") or "").strip(),
        facility_label=str(entry.get("facility_label") or "").strip(),
        facility_type=str(entry.get("facility_type") or "warehouse").strip(),
        currency=str(entry.get("currency") or "GBP").strip().upper(),
        commitment=_number(entry.get("commitment")),
        advance_rate=_advance_rate(entry.get("advance_rate")),
        concentration_denominator_floor=_number(
            entry.get("concentration_denominator_floor")),
        current_drawn_amount=_number(entry.get("current_drawn_amount")),
        current_drawn_amount_as_of=str(
            entry.get("current_drawn_amount_as_of") or "").strip(),
        effective_date=str(entry.get("effective_date") or "").strip(),
        maturity_date=str(entry.get("maturity_date") or "").strip(),
        environment=str(entry.get("environment") or ENV_PRODUCTION).strip(),
        governance=dict(entry.get("governance") or {}),
        financing_portfolio=dict(entry.get("financing_portfolio") or {}),
        eligibility_rule_version=str(eligibility.get("rule_version") or "").strip(),
        eligibility_rules=_rules(eligibility.get("rules")),
        prototype_assume_financing_portfolio_eligible=bool(
            eligibility.get("prototype_assume_financing_portfolio_eligible")),
        concentration_population=str(
            concentration.get("population") or POPULATION_ELIGIBLE).strip(),
        borrowing_base_treatment=str(
            concentration.get("borrowing_base_treatment")
            or TREATMENT_MONITOR_ONLY).strip(),
        config_source=source,
        config_version=str(config_version or "").strip(),
    )


def facilities_path() -> Path:
    return Path(os.environ.get(FACILITIES_PATH_ENV) or DEFAULT_FACILITIES_PATH)


def client_config_dir() -> Path:
    return Path(os.environ.get(CLIENT_CONFIG_DIR_ENV) or CLIENT_CONFIG_DIR)


def _client_config_paths(client_id: str) -> List[Path]:
    """Candidate client-configuration documents, in the platform's own naming.

    Client identifiers appear both as written (``ere_funding_uk``) and under
    the legacy upper-case file naming (``config_client_ERM_UK.yaml``), so the
    directory is scanned rather than a single name guessed.
    """
    directory = client_config_dir()
    if not directory.exists():
        return []
    exact = directory / f"config_client_{client_id}.yaml"
    out = [exact] if exact.exists() else []
    out += sorted(p for p in directory.glob("config_client_*.yaml")
                  if p not in out)
    return out


def load_from_client_config(client_id: str) -> Optional[FacilityConfiguration]:
    """The ``funding_facility:`` block from the client's own configuration."""
    if not client_id:
        return None
    for path in _client_config_paths(client_id):
        doc = _yaml(path)
        block = doc.get(CLIENT_CONFIG_BLOCK)
        if not isinstance(block, dict) or not block:
            continue
        declared = str(((doc.get("client") or {}).get("client_id")
                        or doc.get("client_id") or "")).strip()
        if declared and declared != client_id:
            continue
        if not declared and path.name != f"config_client_{client_id}.yaml":
            # An unattributed block in another client's file is not this
            # client's facility. Skip rather than borrow.
            continue
        return _build(block, client_id=client_id,
                      source=f"client_config:{path.name}",
                      config_version=str(block.get("config_version") or ""))
    return None


def load_from_platform_register(client_id: str
                                ) -> Optional[FacilityConfiguration]:
    """The client's entry in ``config/risk/funding_facilities.yaml``."""
    path = facilities_path()
    doc = _yaml(path)
    version = str(doc.get("config_version") or doc.get("schema_version") or "")
    for entry in doc.get("facilities") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("client_id") or "").strip() != str(client_id).strip():
            continue
        return _build(entry, client_id=client_id,
                      source=f"platform_register:{path.name}",
                      config_version=version)
    return None


def load_facility(client_id: str) -> Optional[FacilityConfiguration]:
    """The governed facility for ``client_id``, or None when none is configured.

    Never raises on a malformed document: an unreadable register behaves as
    "no facility configured", which every caller already handles, rather than
    taking down an unrelated dashboard request.
    """
    client_id = str(client_id or "").strip()
    if not client_id:
        return None
    try:
        facility = load_from_client_config(client_id)
        if facility is not None and facility.facility_id:
            return facility
        return load_from_platform_register(client_id)
    except Exception as exc:  # noqa: BLE001 — configuration must never 500
        logger.warning("facility configuration failed to load for %s: %s",
                       client_id, exc)
        return None


def configuration_generation() -> str:
    """A cheap token that CHANGES when facility configuration changes.

    The prepared canonical frame now carries an eligibility determination, so a
    cache keyed only on the tape would keep serving the determination made
    under the previous configuration until the tape itself changed. Folding
    this token into that key means an approved change to a facility's terms
    reaches the next request rather than the next upload.

    Cheap on purpose: modification times only, never a read or a parse.
    """
    stamps: List[str] = []
    for path in (facilities_path(), client_config_dir()):
        try:
            stamps.append(f"{path}:{Path(path).stat().st_mtime_ns}")
        except OSError:
            stamps.append(f"{path}:absent")
    return "|".join(stamps)


def load_facility_checked(client_id: str
                          ) -> "tuple[Optional[FacilityConfiguration], List[str]]":
    """``(facility, problems)`` — the same load, with validation surfaced."""
    facility = load_facility(client_id)
    return (facility, facility.validate() if facility else [])
