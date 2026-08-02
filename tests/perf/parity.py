#!/usr/bin/env python3
"""tests/perf/parity.py — capture MI responses for before/after comparison.

Phase 1A is a serving-layer change: every figure the API returns must be
identical afterwards. This harness captures the full response body of each
principal MI route (and a spread of ``/mi/query`` questions) against the fixed
representative fixture, canonicalises the parts that are legitimately volatile,
and writes one JSON file per capture.

Run it at the base revision and again at HEAD, then ``--compare`` the two. Any
difference is a regression until explicitly justified.

Canonicalisation removes ONLY:
  * absolute filesystem paths (the scratch/mirror directory is per-run);
  * request/correlation ids and wall-clock timestamps;
  * duration fields.
Every business value, label, ordering, rounding, date and source/run reference
is compared exactly.

Usage::

    python tests/perf/parity.py --root FIXTURE --label before
    python tests/perf/parity.py --root FIXTURE --label after --compare before
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.perf.measure import ROUTES, configure_env  # noqa: E402

#: Keys whose values are legitimately volatile between two runs of the same
#: code. Nothing here carries an economic figure.
_VOLATILE_KEYS = {
    "requestId", "request_id", "correlationId", "correlation_id",
    "generatedAt", "generated_at", "createdAt", "created_at", "timestamp",
    "durationMs", "duration_ms", "elapsedMs", "elapsed_ms",
}

#: Absolute paths differ per run (scratch dirs, temp mirrors). The FILENAME is
#: preserved, because that is provenance and must not drift.
_ABS_PATH_RE = re.compile(r"(/[\w.\-]+)+/([\w.\-]+\.(csv|json|xlsx|pptx))")

#: Artifact / KPI surrogate ids are ``uuid4`` per request (see
#: ``mi_agent_api.adapters._new_id``), so they differ between two runs of the
#: SAME revision. Verified by capturing twice at one commit. They are render
#: handles, never business values, so they are normalised — everything else
#: about the artifact (title, type, rows, values, ordering) is still compared
#: exactly.
_SURROGATE_ID_RE = re.compile(r"^(art|kpi|msg)_[0-9a-f]{6,}$")

#: Questions spanning the routed capabilities AND the point-in-time executor.
QUESTIONS = [
    "What is the total funded balance by region?",
    "Show me the LTV distribution",
    "What are the top 10 loans by balance?",
    "What is the weighted average interest rate?",
    "What is the cohort conversion rate?",
    "How has the funded balance changed versus the prior month?",
    "What is the forecast funded balance?",
    "Show me concentration by broker",
    "What is the loan count by product?",
    "Summarise the portfolio",
]


def _canonical(value: Any) -> Any:
    """Recursively strip volatile fields and absolutise-away scratch paths."""
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in sorted(value.items())
                if k not in _VOLATILE_KEYS}
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    if isinstance(value, str):
        if _SURROGATE_ID_RE.match(value):
            return "<surrogate-id>"
        return _ABS_PATH_RE.sub(lambda m: m.group(2), value)
    if isinstance(value, float):
        # Guard against platform float repr drift without hiding a real change:
        # 12 significant decimals is far finer than any figure the MI layer
        # rounds to (2dp for currency, 4dp for ratios).
        return round(value, 12)
    return value


def capture(root: Path) -> Dict[str, Any]:
    configure_env(root)
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app

    out: Dict[str, Any] = {}
    with TestClient(app) as client:
        for name, path, params in ROUTES:
            resp = client.get(path, params=params)
            out[f"GET {name}"] = {
                "status": resp.status_code,
                "body": _canonical(_safe_json(resp)),
            }
        for i, question in enumerate(QUESTIONS):
            resp = client.post("/mi/query", json={
                "question": question,
                "portfolioId": "ERE/2026-07-31",
                "datasetContext": "funded"})
            out[f"POST query[{i}] {question}"] = {
                "status": resp.status_code,
                "body": _canonical(_safe_json(resp)),
            }
        # The Copilot channel must stay in lockstep with React.
        for i, question in enumerate(QUESTIONS[:4]):
            resp = client.post("/v1/copilot/mi/query", json={"question": question})
            out[f"POST copilot[{i}] {question}"] = {
                "status": resp.status_code,
                "body": _canonical(_safe_json(resp)),
            }
    return out


def _safe_json(resp) -> Any:
    try:
        return resp.json()
    except Exception:  # noqa: BLE001
        return {"__non_json__": resp.text[:2000]}


def _diff(before: Dict[str, Any], after: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    for key in sorted(set(before) | set(after)):
        if key not in before:
            problems.append(f"ADDED   {key}")
            continue
        if key not in after:
            problems.append(f"REMOVED {key}")
            continue
        b, a = before[key], after[key]
        if b == a:
            continue
        if b.get("status") != a.get("status"):
            problems.append(
                f"STATUS  {key}: {b.get('status')} -> {a.get('status')}")
        for path, bv, av in _walk(b.get("body"), a.get("body"), ""):
            problems.append(f"VALUE   {key}{path}: {bv!r} -> {av!r}")
    return problems


def _walk(b: Any, a: Any, path: str, limit: int = 12
          ) -> List[Tuple[str, Any, Any]]:
    """Leaf-level differences, capped so one structural change cannot flood."""
    if b == a:
        return []
    if isinstance(b, dict) and isinstance(a, dict):
        out: List[Tuple[str, Any, Any]] = []
        for k in sorted(set(b) | set(a)):
            if len(out) >= limit:
                break
            out.extend(_walk(b.get(k), a.get(k), f"{path}.{k}", limit - len(out)))
        return out
    if isinstance(b, list) and isinstance(a, list):
        if len(b) != len(a):
            return [(f"{path}[len]", len(b), len(a))]
        out = []
        for i, (bi, ai) in enumerate(zip(b, a)):
            if len(out) >= limit:
                break
            out.extend(_walk(bi, ai, f"{path}[{i}]", limit - len(out)))
        return out
    return [(path, b, a)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    captured = capture(Path(args.root))
    (out_dir / f"{args.label}.json").write_text(json.dumps(captured, indent=2,
                                                           sort_keys=True))
    print(f"captured {len(captured)} responses -> {args.label}.json")

    if args.compare:
        prior_path = out_dir / f"{args.compare}.json"
        if not prior_path.exists():
            print(f"no baseline {prior_path}", file=sys.stderr)
            return 2
        problems = _diff(json.loads(prior_path.read_text()), captured)
        if not problems:
            print(f"PARITY OK: {len(captured)} responses identical "
                  f"to '{args.compare}'")
            return 0
        print(f"PARITY DIFFERENCES ({len(problems)}):")
        for p in problems[:200]:
            print("  " + p)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
