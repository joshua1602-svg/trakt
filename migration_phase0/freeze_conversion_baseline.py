#!/usr/bin/env python3
"""migration_phase0/freeze_conversion_baseline.py — Phase 1B conversion baseline.

READ-ONLY. Captures the EXACT production answer for every question the
`portfolio_summary` recogniser owns, BEFORE the compositional switch, so the
conversion is judged against what shipped rather than against a description of
it.

This is a CONVERSION baseline only. It creates no new interpretation or semantic
baseline, and it re-derives nothing: it records `execute_governed_mi_query`'s
output verbatim, minus the fields that legitimately vary between runs.

    python -m migration_phase0.freeze_conversion_baseline            # write
    python -m migration_phase0.freeze_conversion_baseline --compare  # diff vs the frozen file
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

warnings.simplefilter("ignore")
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "migration_phase0" / "CONVERSION_BASELINE.json"

#: The nine questions the shipped recogniser claims, plus the two that look like
#: it and are NOT claimed. Both halves are frozen: a conversion that starts
#: claiming X1 or X2 has changed route ownership, which is a movement.
CASES: Tuple[Tuple[str, str], ...] = (
    ("A1", "Please provide a portfolio summary"),
    ("A2", "Give me a summary of the portfolio"),
    ("A3", "Can you summarise the book for me?"),
    ("A4", "portfolio summary"),
    ("A5", "summarise the portfolio"),
    ("A6", "overview of the portfolio"),
    ("L1", "Summarise the acquired book"),
    ("L2", "Summarise the direct book"),
    ("L3", "portfolio summary for the acquired book"),
    ("X1", "Summarise the front book"),
    ("X2", "What is the portfolio position for the direct book?"),
)

#: Keys whose value legitimately differs run to run and says nothing about
#: behaviour. Stripped recursively before comparison so a diff is a real diff.
_VOLATILE = {"runId", "run_id", "generatedAt", "generated_at", "timestamp",
             "durationMs", "duration_ms", "elapsed", "queryId", "query_id",
             "requestId", "request_id", "traceId", "trace_id"}


def _env() -> str:
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return cfg.CLIENT_ID


#: A generated KPI id (`kpi_<8 hex>`). Randomised per render on the
#: point-in-time path, so it differs between two identical runs. Stripped by
#: VALUE SHAPE rather than by key name: `id` elsewhere in the payload may be
#: meaningful, and dropping every `id` would hide a real difference.
_KPI_ID_RE = re.compile(r"^kpi_[0-9a-f]{8}$")


def _strip(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip(v) for k, v in sorted(node.items())
                if k not in _VOLATILE
                and not (k == "id" and isinstance(v, str) and _KPI_ID_RE.match(v))}
    if isinstance(node, (list, tuple)):
        return [_strip(v) for v in node]
    if isinstance(node, float):
        # A float that round-trips through JSON must compare equal.
        return round(node, 6)
    return node


def capture(client_id: str) -> Dict[str, Any]:
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    ctx = ExecutionContext.for_internal(client_id)
    out: Dict[str, Any] = {}
    for case_id, question in CASES:
        result = execute_governed_mi_query(MiQueryRequest(question=question), ctx).result or {}
        metadata = result.get("metadata") or {}
        # `route` is published on metadata, not at the top level, and
        # `executionSummary` is top-level. Both were read from the wrong place in
        # the first draft of this instrument; a baseline that records `route:
        # None` for every case cannot detect a change of route ownership, which
        # is one of the movements this conversion must not make.
        summary = (result.get("executionSummary")
                   or metadata.get("executionSummary") or {})
        artifacts = []
        for artifact in (result.get("artifacts") or []):
            artifacts.append({
                "type": artifact.get("type"),
                "title": artifact.get("title"),
                "columns": [c.get("key") for c in (artifact.get("columns") or [])],
                "rowKeys": sorted((artifact.get("rows") or [{}])[0].keys())
                           if artifact.get("rows") else [],
                "rows": artifact.get("rows"),
                "kpis": artifact.get("kpis"),
            })
        out[case_id] = _strip({
            "question": question,
            "ok": result.get("ok"),
            "route": metadata.get("route"),
            "answer": result.get("answer"),
            "error": result.get("error"),
            "payloadKeys": sorted(result.keys()),
            "artifacts": artifacts,
            "reconciliation": result.get("reconciliation"),
            "sourceNotes": result.get("sourceNotes") or result.get("source_notes"),
            "portfolioScope": result.get("portfolioScope"),
            "lensApplied": metadata.get("lensApplied"),
            "engine": metadata.get("engine"),
            "assumptions": result.get("assumptions"),
            "interpreted": result.get("interpreted"),
            "warnings": result.get("warnings"),
            "executionSummary": summary,
            "facets": [(f.get("kind"), f.get("label"), f.get("field"), f.get("status"))
                       for f in (summary.get("facets") or [])],
            "notApplied": summary.get("notApplied"),
            "metadataKeys": sorted(metadata.keys()),
        })
    return out


def _flatten(node: Any, prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            flat.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            flat.update(_flatten(value, f"{prefix}[{i}]"))
    else:
        flat[prefix] = node
    return flat


def compare(client_id: str) -> int:
    if not OUT.exists():
        print(f"no baseline at {OUT.relative_to(_REPO)} — run without --compare first")
        return 2
    before = json.loads(OUT.read_text())["cases"]
    after = capture(client_id)

    total = 0
    print("=" * 78)
    print("portfolio_summary — PRODUCTION PAYLOAD, frozen vs now")
    print("=" * 78)
    for case_id, _question in CASES:
        old_flat = _flatten(before.get(case_id, {}))
        new_flat = _flatten(after.get(case_id, {}))
        diffs: List[str] = []
        for key in sorted(set(old_flat) | set(new_flat)):
            old, new = old_flat.get(key, "<absent>"), new_flat.get(key, "<absent>")
            if isinstance(old, (int, float)) and isinstance(new, (int, float)) \
                    and not isinstance(old, bool) and not isinstance(new, bool):
                if abs(float(old) - float(new)) < 0.005:
                    continue
            if old != new:
                diffs.append(f"{key}: {old!r} -> {new!r}")
        total += len(diffs)
        mark = "IDENTICAL" if not diffs else f"{len(diffs)} DIFFERENCE(S)"
        print(f"  {case_id}  {mark}")
        for d in diffs:
            print(f"       {d}")
    print("=" * 78)
    print(f"total differences across {len(CASES)} cases: {total}")
    return 0 if total == 0 else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m migration_phase0.freeze_conversion_baseline")
    ap.add_argument("--compare", action="store_true",
                    help="diff the live production payload against the frozen one")
    args = ap.parse_args(argv)
    client_id = _env()
    if args.compare:
        return compare(client_id)

    import subprocess
    head = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    payload = {
        "artefact": "portfolio_summary conversion baseline (Phase 1B)",
        "purpose": ("The EXACT production answer for every question the shipped "
                    "recogniser owns, before the compositional switch. A "
                    "conversion baseline only — no new interpretation or "
                    "semantic baseline is created here."),
        "headSha": head,
        "entrypoint": "mi_agent_api.mi_service.execute_governed_mi_query",
        "volatileKeysStripped": sorted(_VOLATILE),
        "cases": capture(client_id),
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(_REPO)}  ({len(CASES)} cases, head {head[:12]})")
    for case_id, question in CASES:
        c = payload["cases"][case_id]
        print(f"  {case_id}  ok={str(c['ok']):5s} route={str(c['route']):18s} "
              f"facets={len(c['facets'])}  {question[:44]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
