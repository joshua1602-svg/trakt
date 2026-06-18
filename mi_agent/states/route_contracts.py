"""mi_agent.states.route_contracts — lightweight route/state eligibility.

Phase 3 MI state assembler. A small, testable helper that reads the Phase 0B
route config (``config/routes/<route>_route.yaml``) and validates whether a
requested state is allowed for a route. This is NOT a runtime route resolver or
orchestration layer — it is a pure config read + membership check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .models import UNSUPPORTED_STATE_FOR_ROUTE, WARNING, make_issue

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTES_DIR = REPO_ROOT / "config" / "routes"

# Descriptive aliases used by callers/tests -> canonical state in the configs.
STATE_ALIASES: Dict[str, str] = {
    "cohort_by_origination_date": "cohort_by_date",
    "cohort_by_funding_date": "cohort_by_date",
    "cohort_by_acquisition_date": "cohort_by_date",
}


def canonical_state(state_name: str) -> str:
    """Resolve a descriptive alias to its canonical config state name."""
    return STATE_ALIASES.get(state_name, state_name)


def load_route_contract(route: str,
                        routes_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load ``config/routes/<route>_route.yaml`` as a dict.

    Accepts either the bare route id (``mi``) or the file stem
    (``mi_route``)."""
    routes_dir = Path(routes_dir) if routes_dir else DEFAULT_ROUTES_DIR
    stem = route if route.endswith("_route") else f"{route}_route"
    path = routes_dir / f"{stem}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"route contract not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def allowed_states(route: str,
                   routes_dir: Optional[Path] = None) -> List[str]:
    contract = load_route_contract(route, routes_dir=routes_dir)
    return list(contract.get("allowed_states") or [])


def is_state_allowed(state_name: str, route: str,
                     routes_dir: Optional[Path] = None) -> bool:
    return canonical_state(state_name) in allowed_states(route, routes_dir=routes_dir)


def validate_state_for_route(state_name: str, route: str,
                             routes_dir: Optional[Path] = None
                             ) -> Optional[Dict[str, Any]]:
    """Return an ``unsupported_state_for_route`` issue if *state_name* is not in
    *route*'s ``allowed_states``; otherwise ``None``."""
    canonical = canonical_state(state_name)
    allowed = allowed_states(route, routes_dir=routes_dir)
    if canonical in allowed:
        return None
    return make_issue(
        UNSUPPORTED_STATE_FOR_ROUTE, WARNING,
        f"state {state_name!r} (canonical {canonical!r}) is not allowed for "
        f"route {route!r}; allowed: {allowed}",
        field=state_name, route=route, allowed_states=allowed)
