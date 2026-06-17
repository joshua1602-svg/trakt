"""
cli.py
======

Subcommand-style CLI for the Trakt Validation Agent (mirrors the
``<agent> <verb>`` convention)::

    python -m engine.validation_agent.cli validate \\
      --transformation-manifest onboarding_output/client_001/run_tva_exit_check/output/transformation/30_transformation_manifest.json

This is a thin wrapper over :func:`engine.validation_agent.workflow.main`; the
flag-only entry point ``python -m engine.validation_agent.workflow`` is also
available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.validation_agent import workflow as wf  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Trakt Validation Agent CLI.")
    sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser(
        "validate",
        help="Consume the transformation package and produce the validation "
        "readiness package for the Projection Agent.",
        parents=[wf.build_parser()], add_help=False)
    v.set_defaults(_handler="validate")
    return p


def main(argv=None) -> int:
    args, _ = build_parser().parse_known_args(argv)
    if getattr(args, "command", None) == "validate":
        passthrough = list(argv if argv is not None else sys.argv[1:])
        if passthrough and passthrough[0] == "validate":
            passthrough = passthrough[1:]
        return wf.main(passthrough)
    print("Unknown command", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
