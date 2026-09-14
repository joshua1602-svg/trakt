#!/usr/bin/env python3
"""Inventory every blob a client wipe would delete. READ ONLY.

This script opens storage, lists, and writes nothing — not a blob, not an
audit event, not a local file. Run it BEFORE deleting a client, so the
deletion set is something you have read rather than something you assumed.

WHY THIS EXISTS RATHER THAN `az storage blob list`
--------------------------------------------------
Two reasons, and the second is the important one.

The Azure CLI is not installed everywhere this needs to run, while the Azure
SDK already is — it is what the platform itself uses. Reusing
``apps.blob_trigger_app.storage.open_storage`` means this sees exactly what the
platform sees, through the same credentials and the same container resolution.

And a shell loop counting `az ... | wc -l` reports ZERO when the command is
missing, when credentials are wrong, and when the prefix is genuinely empty.
Three different states, one answer, and the dangerous one looks like the safe
one. This script separates them: an unreachable store is an ERROR and a
non-zero exit, never a count of nothing.

WHAT IT LOOKS AT
----------------
Two passes, because they catch different mistakes:

1. KNOWN PREFIXES — the paths the code's own layout authorities build for a
   client (``OpsLayout``, ``apps.blob_trigger_app.layout.Layout``,
   ``operations_control.manual_intake.derive_raw_prefix``). This is the
   deletion set you plan.

2. A FULL SWEEP for the client id anywhere in a blob name, across every
   container. This is the deletion set you did not plan: anything written
   outside those path builders, by a script, a migration or a hand upload.
   Pass 1 tells you the plan is complete; pass 2 tells you whether the plan is
   ENOUGH.

Plus the identity records that survive a prefix delete and then refuse the
identifier back — the ones that turn a clean wipe into a blocked onboarding.

USAGE
-----
::

    export TRAKT_STORAGE_BACKEND=blob
    export TRAKT_BLOB_CONNECTION='<the storage connection string>'
    python scripts/inventory_client_state.py --client ERE

    # keep the full list as pre-wipe evidence
    python scripts/inventory_client_state.py --client ERE --json > wipe_set.json

EXIT CODES
----------
0  inventory completed (whatever it found)
2  the store could not be reached — the result is NOT "nothing to delete"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Containers holding platform state. Deliberately NOT the Azure-managed ones
#: (``azure-webjobs-*``, ``app-package-*``, ``$web``, ``$logs``): those are
#: runtime infrastructure, nothing to do with a client, and listing them would
#: only invite someone to delete one.
CONTAINERS = ("raw-v2", "processed-v2", "operations-control", "trakt-state",
              "operations-control-synthetic")


def _known_prefixes(client: str) -> List[Tuple[str, str]]:
    """``(prefix, name_filter)`` pairs the code's layout authorities build.

    Every prefix here ends at a SEGMENT BOUNDARY, and anything finer is done
    with ``name_filter`` instead. That is not tidiness — the two storage
    backends read a prefix differently:

    * blob (``storage.py:302``) passes it to ``name_starts_with``: a string
      prefix, so ``runs/ERE_`` matches ``runs/ERE_direct_001_....json``;
    * file (``storage.py:202``) resolves it to a directory and ``rglob``s it,
      so ``runs/ERE_`` is a directory that does not exist, and the answer is
      an empty list.

    A partial-segment prefix therefore works against Azure and silently
    reports NOTHING TO DELETE against a local store — the one failure mode
    this script exists to prevent. Listing the segment and filtering by name
    gives the same answer on both.
    """
    return [
        # raw uploads — manual_intake.derive_raw_prefix
        (f"blob://raw-v2/{client}/", ""),
        # canonical outputs — apps.blob_trigger_app.layout.Layout
        (f"blob://processed-v2/accepted/{client}/", ""),
        (f"blob://processed-v2/platform/{client}/", ""),
        (f"blob://processed-v2/regime/{client}/", ""),
        (f"blob://processed-v2/mi/{client}/", ""),
        # governed OCC state — operations_control.stores.OpsLayout
        (f"blob://operations-control/{client}/", ""),
        # run records — pack keys begin with the client id, so this one is a
        # partial segment and must be filtered rather than prefixed.
        ("blob://trakt-state/runs/", f"{client}_"),
    ]


def _identity_records() -> List[str]:
    """Blobs that name a client OUTSIDE its own prefix.

    These survive a prefix delete, and every one of them feeds
    ``OnboardingService._identifiers_in_use()``. Leave them and the platform
    refuses to issue the identifier back: "'X' is already in use by another
    client."
    """
    return [
        "blob://operations-control/_index/clients.json",
        "blob://operations-control/_onboarding/clients.json",
        "blob://trakt-state/registry/source_registry.yaml",
    ]


def inventory(storage, client: str) -> Dict[str, Any]:
    planned: Dict[str, List[str]] = {}
    for prefix, name_filter in _known_prefixes(client):
        found = storage.list(prefix)
        if name_filter:
            found = [u for u in found
                     if u.rsplit("/", 1)[-1].startswith(name_filter)]
        label = prefix + (f"{name_filter}*" if name_filter else "")
        planned[label] = sorted(found)

    # The sweep. Listing a whole container is the point: a blob written outside
    # the layout authorities is exactly what the planned set cannot show.
    needle = client.casefold()
    everywhere: Dict[str, List[str]] = {}
    unreachable: Dict[str, str] = {}
    for container in CONTAINERS:
        try:
            names = storage.list(f"blob://{container}/")
        except Exception as exc:  # noqa: BLE001 — one container is not the sweep
            unreachable[container] = f"{type(exc).__name__}: {exc}"
            continue
        hits = sorted(n for n in names if needle in n.casefold())
        if hits:
            everywhere[container] = hits

    planned_set = {uri for uris in planned.values() for uri in uris}
    swept_set = {uri for uris in everywhere.values() for uri in uris}

    identity: Dict[str, Any] = {}
    for uri in _identity_records():
        try:
            present = storage.exists(uri)
        except Exception:  # noqa: BLE001
            identity[uri] = "UNREADABLE"
            continue
        identity[uri] = "present" if present else "absent"

    return {
        "client": client,
        "planned_prefixes": planned,
        "planned_total": len(planned_set),
        "sweep_by_container": everywhere,
        "sweep_total": len(swept_set),
        # The whole reason for two passes.
        "found_outside_planned_prefixes": sorted(swept_set - planned_set),
        "identity_records": identity,
        "containers_unreachable": unreachable,
    }


def _print_human(report: Dict[str, Any]) -> None:
    client = report["client"]
    print(f"client               : {client}")
    print(f"planned for deletion : {report['planned_total']} blobs")
    print(f"named anywhere       : {report['sweep_total']} blobs")
    print()

    print("PLANNED PREFIXES")
    print("-" * 64)
    for prefix, uris in report["planned_prefixes"].items():
        print(f"{len(uris):>6}  {prefix}")
    print()

    extra = report["found_outside_planned_prefixes"]
    print("OUTSIDE THE PLANNED PREFIXES")
    print("-" * 64)
    if not extra:
        print("  none — the planned set covers every blob naming this client.")
    else:
        print(f"  {len(extra)} blob(s) name {client} but sit outside the paths")
        print("  the layout authorities build. READ THESE BEFORE DELETING:")
        for uri in extra[:50]:
            print(f"    {uri}")
        if len(extra) > 50:
            print(f"    ... and {len(extra) - 50} more (use --json for all)")
    print()

    print("IDENTITY RECORDS (survive a prefix delete; refuse the id back)")
    print("-" * 64)
    for uri, state in report["identity_records"].items():
        mark = "DELETE" if state == "present" else "      "
        print(f"  {mark}  {state:<10} {uri}")

    if report["containers_unreachable"]:
        print()
        print("CONTAINERS NOT LISTED — this inventory is INCOMPLETE")
        print("-" * 64)
        for container, why in report["containers_unreachable"].items():
            print(f"  {container}: {why}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--client", required=True,
                    help="Client identifier to inventory, e.g. ERE")
    ap.add_argument("--json", action="store_true",
                    help="Emit the full machine-readable inventory")
    args = ap.parse_args()

    try:
        from apps.blob_trigger_app.storage import open_storage
        storage = open_storage()
        report = inventory(storage, args.client)
    except Exception as exc:  # noqa: BLE001
        print(f"STORE UNREACHABLE: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("This is NOT 'nothing to delete'. Fix storage configuration and "
              "re-run before deleting anything.", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
