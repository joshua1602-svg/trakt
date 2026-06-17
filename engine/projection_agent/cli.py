"""
cli.py
======

Subcommand-style CLI for the Trakt Projection Agent (mirrors the
``<agent> <verb>`` convention)::

    python -m engine.projection_agent.cli project \\
      --validation-manifest onboarding_output/client_001/run_projection_blocker_diagnostic_fix/output/validation/40_validation_manifest.json

This is a thin wrapper over :func:`engine.projection_agent.workflow.main`; the
flag-only entry point ``python -m engine.projection_agent.workflow`` is also
available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.projection_agent import workflow as wf  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Trakt Projection Agent CLI.")
    sub = p.add_subparsers(dest="command", required=True)
    pr = sub.add_parser(
        "project",
        help="Consume the validation package and produce a projected Annex 2 "
        "target frame (NOT XML) for delivery normalisation.",
        parents=[wf.build_parser()], add_help=False)
    pr.set_defaults(_handler="project")
    return p


def main(argv=None) -> int:
    args, _ = build_parser().parse_known_args(argv)
    if getattr(args, "command", None) == "project":
        passthrough = list(argv if argv is not None else sys.argv[1:])
        if passthrough and passthrough[0] == "project":
            passthrough = passthrough[1:]
        return wf.main(passthrough)
    print("Unknown command", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
