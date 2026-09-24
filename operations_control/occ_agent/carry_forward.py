"""Carry an activated case's settled mappings into the governed rules.

    python -m operations_control.occ_agent.carry_forward ONB-2026-0010
    python -m operations_control.occ_agent.carry_forward ONB-2026-0010 --tenant ERE

For a case activated before set-asides were promoted: the columns an operator
set aside in the Client Onboarding screen are written as governed rules, so the
next workflow run leaves them out instead of asking about them again. Every
settled mapping is restated by the same governed path activation uses, and the
act is audited on the case. Prints what was written; prints no client data.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case_ref")
    parser.add_argument("--tenant", action="append", default=[],
                        help="tenant the practice case is filed under; "
                             "tried in order (default: synthetic)")
    parser.add_argument("--actor", default="operator_ssh")
    args = parser.parse_args(argv)

    from apps.blob_trigger_app.storage import open_storage
    from ..engine import OpsEngine, OpsError
    from ..stores import OpsStore
    from .service import OccAgentService

    storage = open_storage()
    service = OccAgentService(storage, engine=OpsEngine(OpsStore.from_env()))
    agent_case = None
    for tenant in args.tenant or ["synthetic"]:
        if service.store.exists(tenant, args.case_ref):
            agent_case = service.load(tenant, args.case_ref)
            break
    if agent_case is None:
        print(f"{args.case_ref} was not found under tenant(s) "
              f"{args.tenant or ['synthetic']}. Pass --tenant.")
        return 2
    try:
        written = service.carry_mappings_forward(agent_case, actor=args.actor)
    except OpsError as exc:
        print(f"Refused: {exc}")
        return 1
    asides = [w for w in written if w.get("set_aside") == "true"]
    withdrawn = [w for w in written if w.get("withdrawn") == "true"]
    mapped = [w for w in written if w not in asides and w not in withdrawn]
    print(f"{len(mapped)} mapping(s) restated, {len(asides)} column(s) set "
          f"aside, {len(withdrawn)} other rule(s) withdrawn.")
    for w in asides:
        print(f"  set aside: '{w['source_column']}'  "
              f"({w['rule_id']} v{w['version']})")
    for w in withdrawn:
        print(f"  withdrawn: '{w['source_column']}'  "
              f"({w['rule_id']} v{w['version']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
