"""
cli.py
======

Subcommand-style CLI for the Trakt Delivery/XML Agent (mirrors the
``<agent> <verb>`` convention)::

    python -m engine.delivery_xml_agent.cli deliver \\
      --projection-manifest output/projection/50_projection_manifest.json

This is a thin wrapper over :func:`engine.delivery_xml_agent.workflow.main`; the
flag-only entry point ``python -m engine.delivery_xml_agent.workflow`` is also
available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.delivery_xml_agent import workflow as wf  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Trakt Delivery/XML Agent CLI.")
    sub = p.add_subparsers(dest="command", required=True)
    dl = sub.add_parser(
        "deliver",
        help="Consume the Projection package and produce a delivery package "
        "(NO production XML; refuses XML unless all readiness gates pass).",
        parents=[wf.build_parser()], add_help=False)
    dl.set_defaults(_handler="deliver")
    return p


def main(argv=None) -> int:
    args, _ = build_parser().parse_known_args(argv)
    if getattr(args, "command", None) == "deliver":
        passthrough = list(argv if argv is not None else sys.argv[1:])
        if passthrough and passthrough[0] == "deliver":
            passthrough = passthrough[1:]
        return wf.main(passthrough)
    print("Unknown command", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
