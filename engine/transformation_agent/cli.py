"""
cli.py
======

Subcommand-style CLI for the Trakt Transformation Agent (mirrors the
``<agent> <verb>`` convention)::

    python -m engine.transformation_agent.cli transform \\
      --handoff-manifest onboarding_output/client_001/run_annex2_onboarding_exit_check/output/handoff/24_onboarding_handoff_manifest.json

This is a thin wrapper over :func:`engine.transformation_agent.workflow.main`;
the flag-only entry point ``python -m engine.transformation_agent.workflow`` is
also available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.transformation_agent import workflow as wf  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trakt Transformation Agent CLI.")
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser(
        "transform",
        help="Consume the onboarding handoff package and produce the "
        "validation-ready transformed canonical package.",
        parents=[wf.build_parser()], add_help=False)
    t.set_defaults(_handler="transform")
    return p


def main(argv=None) -> int:
    args, _ = build_parser().parse_known_args(argv)
    if getattr(args, "command", None) == "transform":
        # Delegate to the workflow main, stripping the leading subcommand.
        passthrough = list(argv if argv is not None else sys.argv[1:])
        if passthrough and passthrough[0] == "transform":
            passthrough = passthrough[1:]
        return wf.main(passthrough)
    print("Unknown command", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
