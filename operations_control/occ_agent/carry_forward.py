"""Carry an activated case's settled mappings into the governed rules.

    python -m operations_control.occ_agent.carry_forward ONB-2026-0010
    python -m operations_control.occ_agent.carry_forward ONB-2026-0010 --tenant ERE

    # Correct one column of one file first (the case is corrected, then
    # carried forward): keep it as a field, or set it aside.
    python -m operations_control.occ_agent.carry_forward ONB-2026-0010 \
        --map "PropertyExtract - Omni 2026_09_01.xlsx::Post Code=postcode" \
        --set-aside "LoanExtract One - OMNI 2026_09_01.xlsx::Post Code"

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
    parser.add_argument("--map", action="append", default=[],
                        metavar="FILE::COLUMN=FIELD",
                        help="this column of this file feeds FIELD")
    parser.add_argument("--set-aside", action="append", default=[],
                        metavar="FILE::COLUMN",
                        help="this column of this file feeds nothing")
    args = parser.parse_args(argv)
    corrections = []
    for spec in args.map:
        where, _, field = spec.rpartition("=")
        source_file, _, column = where.partition("::")
        if not (source_file and column and field):
            parser.error(f"--map needs FILE::COLUMN=FIELD, got {spec!r}")
        corrections.append({"source_file": source_file.strip(),
                            "source_column": column.strip(),
                            "target_field": field.strip()})
    for spec in args.set_aside:
        source_file, _, column = spec.partition("::")
        if not (source_file and column):
            parser.error(f"--set-aside needs FILE::COLUMN, got {spec!r}")
        corrections.append({"source_file": source_file.strip(),
                            "source_column": column.strip(),
                            "target_field": ""})

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
        if corrections:
            written = service.correct_live_mappings(
                agent_case, corrections=corrections, actor=args.actor)
        else:
            written = service.carry_mappings_forward(agent_case,
                                                     actor=args.actor)
    except OpsError as exc:
        print(f"Refused: {exc}")
        return 1
    asides = [w for w in written if w.get("set_aside") == "true"]
    replaced = [w for w in written if w.get("replaced") == "true"]
    withdrawn = [w for w in written if w.get("withdrawn") == "true"
                 and w not in replaced]
    mapped = [w for w in written
              if w not in asides and w not in withdrawn and w not in replaced]
    print(f"{len(mapped)} mapping(s) restated, {len(asides)} column(s) set "
          f"aside, {len(withdrawn)} mapping(s) withdrawn, {len(replaced)} "
          f"old-style set-aside(s) replaced.")
    for w in asides:
        print(f"  set aside: {w.get('source_file') or '*'} | "
              f"'{w['source_column']}'")
    for w in withdrawn:
        print(f"  withdrawn: '{w['source_column']}'  "
              f"({w['rule_id']} v{w['version']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
