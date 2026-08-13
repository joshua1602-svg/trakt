#!/usr/bin/env python3
"""tests/perf/measure.py — measure the principal MI routes, cold and warm.

Drives the REAL FastAPI application against the representative fixture tree from
``generate_fixture.py`` and reports, per route:

  * cold duration (first request in a fresh process — every cache empty);
  * warm duration (a repeat of the same request);
  * storage operations (list / etag / download / read) per request;
  * per-stage timings from :mod:`trakt_core.perf`;
  * repeated-calculation counts (historical model builds, pipeline preps,
    funded preps);
  * response size.

The numbers come from the same instrumentation that runs in production, so a
before/after comparison is measuring the same thing the deployment reports.

Usage::

    python tests/perf/measure.py --root /tmp/mi_perf --label before
    python tests/perf/measure.py --root /tmp/mi_perf --label after --compare before
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: The dashboard's principal read paths, in the order a user meets them.
ROUTES = [
    ("snapshots_index", "/mi/snapshots", {}),
    ("portfolio_context", "/mi/portfolio-context", {}),
    ("funded_snapshot", "/mi/snapshot", {"portfolioId": "ERE/2026-07-31"}),
    ("geo_exposure", "/mi/geo/exposure", {"portfolioId": "ERE/2026-07-31"}),
    ("funded_evolution", "/mi/evolution/funded", {"portfolioId": "ERE/2026-07-31"}),
    ("cohorts", "/mi/cohorts", {"portfolioId": "ERE/2026-07-31"}),
    ("forecast_snapshot", "/mi/forecast/snapshot", {"portfolioId": "ERE/2026-07-31"}),
    ("pipeline_evolution", "/mi/evolution/pipeline", {"portfolioId": "ERE/2026-07-31"}),
    ("funnel_evolution", "/mi/evolution/funnel", {"portfolioId": "ERE/2026-07-31"}),
    ("forecast_evolution", "/mi/evolution/forecast", {"portfolioId": "ERE/2026-07-31"}),
    ("forecast_extrapolation", "/mi/forecast/extrapolation",
     {"portfolioId": "ERE/2026-07-31"}),
    ("risk_limits", "/mi/risk-limits", {"portfolioId": "ERE/2026-07-31"}),
    ("concentration_tests", "/mi/concentration-tests",
     {"portfolioId": "ERE/2026-07-31"}),
]

#: One representative chat question that needs NO historical completion model.
#: The whole point of remediation item 1 is that this must not build one.
MI_QUERY_SIMPLE = "What is the total funded balance by region?"
#: One that genuinely does need it.
MI_QUERY_HISTORICAL = "What is the cohort conversion rate?"


class _PerfSink(logging.Handler):
    """Captures the structured ``perf.request`` record emitted per request."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: List[Dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            payload = msg.split("perf ", 1)[1] if msg.startswith("perf ") else None
            if payload:
                self.records.append(json.loads(payload))
        except Exception:  # noqa: BLE001 - a sink fault must not fail the run
            pass


#: Every environment variable :func:`configure_env` writes or removes.
#:
#: Named here, next to the function that mutates them, so a caller that needs to
#: put the environment back has one list to restore and it cannot fall behind:
#: adding a variable to ``configure_env`` without adding it here is visible in
#: the same diff.
#:
#: This exists because the mutation genuinely leaks. ``MI_AGENT_AUTH_ENABLED``
#: is set to "false" below, and a unittest class that called ``configure_env``
#: in ``setUpClass`` without restoring it disabled authentication for every test
#: that ran after it in the same process — turning a security assertion in an
#: unrelated file into a collection-order coin toss. See
#: ``tests/test_auth_env_order_independence.py``.
CONFIGURED_ENV_KEYS = (
    "TRAKT_STORAGE_BACKEND",
    "TRAKT_LOCAL_BLOB_ROOT",
    "TRAKT_PROCESSED_CONTAINER",
    "MI_AGENT_ONBOARDING_OUTPUT_ROOT",
    "MI_AGENT_PIPELINE_ROOT",
    "MI_AGENT_PLATFORM_URI",
    "MI_AGENT_CLIENT_ID",
    "MI_AGENT_SCRATCH",
    "MI_AGENT_AUTH_ENABLED",
    "TRAKT_RUNTIME_MODE",
    "TRAKT_PERF_INSTRUMENTATION",
    "MI_AGENT_PIPELINE_URI",
    "MI_AGENT_PIPELINE_SOURCE",
)


def snapshot_env() -> Dict[str, Optional[str]]:
    """The current value of every key ``configure_env`` touches.

    ``None`` records "was not set", which must be restored as *absence* rather
    than as an empty string — an empty ``MI_AGENT_AUTH_ENABLED`` enforces auth,
    but an empty ``TRAKT_LOCAL_BLOB_ROOT`` is a path that does not exist.
    """
    return {key: os.environ.get(key) for key in CONFIGURED_ENV_KEYS}


def restore_env(snapshot: Dict[str, Optional[str]]) -> None:
    """Put back exactly what :func:`snapshot_env` recorded."""
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def configure_env(root: Path) -> None:
    """Point the API at the fixture tree through the filesystem blob backend.

    Mutates the process environment. A test that calls this **must** restore it
    afterwards with :func:`snapshot_env` / :func:`restore_env`; the variables it
    writes change the behaviour of unrelated tests later in the same run.
    """
    os.environ["TRAKT_STORAGE_BACKEND"] = "file"
    os.environ["TRAKT_LOCAL_BLOB_ROOT"] = str(root)
    os.environ["TRAKT_PROCESSED_CONTAINER"] = "processed-v2"
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = "blob://processed-v2/platform/ERE"
    os.environ["MI_AGENT_PIPELINE_ROOT"] = "blob://processed-v2/pipeline/ERE"
    os.environ["MI_AGENT_PLATFORM_URI"] = (
        "blob://processed-v2/platform/ERE/latest/platform_canonical_typed.csv")
    os.environ["MI_AGENT_CLIENT_ID"] = "ERE"
    os.environ["MI_AGENT_SCRATCH"] = str(root / "_scratch")
    # Auth off: this harness measures the analytical path, not the guard. The
    # guard's own cost is separately covered by the API test suite.
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    os.environ["TRAKT_RUNTIME_MODE"] = "development"
    os.environ["TRAKT_PERF_INSTRUMENTATION"] = "on"
    os.environ.pop("MI_AGENT_PIPELINE_URI", None)
    os.environ.pop("MI_AGENT_PIPELINE_SOURCE", None)


def _summarise(rec: Dict[str, Any]) -> Dict[str, Any]:
    counters = rec.get("counters") or {}
    storage_ops = {k: v for k, v in counters.items() if k.startswith("storage.")}
    stages = rec.get("stages") or {}
    stage_calls = rec.get("stage_calls") or {}
    return {
        "total_ms": rec.get("total_ms"),
        "status": rec.get("status"),
        "bytes": rec.get("response_bytes"),
        "storage_ops": sum(storage_ops.values()),
        "storage_detail": storage_ops,
        "file_reads": sum(v for k, v in counters.items() if k.startswith("file.read.")),
        "history_builds": stage_calls.get("historical_completion_model_build", 0),
        "pipeline_preps": stage_calls.get("pipeline_prep", 0),
        "funded_preps": stage_calls.get("funded_prep", 0),
        "top_stages": dict(sorted(stages.items(), key=lambda kv: -kv[1])[:6]),
        "caches": rec.get("caches") or {},
    }


def run(root: Path, repeats: int) -> Dict[str, Any]:
    configure_env(root)
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app

    sink = _PerfSink()
    perf_logger = logging.getLogger("mi_agent_api.perf")
    perf_logger.setLevel(logging.INFO)
    perf_logger.addHandler(sink)
    perf_logger.propagate = False

    results: Dict[str, Any] = {}
    with TestClient(app) as client:
        for name, path, params in ROUTES:
            samples = []
            for attempt in range(repeats + 1):
                sink.records.clear()
                resp = client.get(path, params=params)
                rec = sink.records[-1] if sink.records else {}
                summary = _summarise(rec)
                summary["http_status"] = resp.status_code
                samples.append(summary)
            results[name] = {
                "path": path, "params": params,
                "cold": samples[0],
                "warm": samples[-1],
                "warm_median_ms": statistics.median(
                    [s["total_ms"] or 0 for s in samples[1:]]) if repeats else None,
            }

        for label, question in (("mi_query_simple", MI_QUERY_SIMPLE),
                                ("mi_query_historical", MI_QUERY_HISTORICAL)):
            samples = []
            for _ in range(repeats + 1):
                sink.records.clear()
                resp = client.post("/mi/query", json={
                    "question": question,
                    "portfolioId": "ERE/2026-07-31",
                    "datasetContext": "funded"})
                rec = sink.records[-1] if sink.records else {}
                summary = _summarise(rec)
                summary["http_status"] = resp.status_code
                samples.append(summary)
            results[label] = {"path": "/mi/query", "params": {"question": question},
                              "cold": samples[0], "warm": samples[-1]}

    perf_logger.removeHandler(sink)
    return results


def render(results: Dict[str, Any], compare: Optional[Dict[str, Any]] = None) -> str:
    lines = []
    head = ("| Route | Cold ms | Warm ms | Storage ops (cold/warm) | File reads (cold/warm) "
            "| Hist builds | Pipe preps | Funded preps | Bytes |")
    lines.append(head)
    lines.append("|" + "---|" * 9)
    for name, r in results.items():
        c, w = r["cold"], r["warm"]
        lines.append(
            f"| `{name}` | {c['total_ms']} | {w['total_ms']} | "
            f"{c['storage_ops']}/{w['storage_ops']} | {c['file_reads']}/{w['file_reads']} | "
            f"{c['history_builds']}/{w['history_builds']} | "
            f"{c['pipeline_preps']}/{w['pipeline_preps']} | "
            f"{c['funded_preps']}/{w['funded_preps']} | {c['bytes']} |")
    if compare:
        lines.append("")
        lines.append("| Route | Cold before → after | Δ% cold | Warm before → after | Δ% warm "
                     "| Storage warm before → after |")
        lines.append("|" + "---|" * 6)
        for name, r in results.items():
            b = compare.get(name)
            if not b:
                continue
            def pct(before, after):
                if not before:
                    return "n/a"
                return f"{(after - before) / before * 100:+.1f}%"
            lines.append(
                f"| `{name}` | {b['cold']['total_ms']} → {r['cold']['total_ms']} | "
                f"{pct(b['cold']['total_ms'], r['cold']['total_ms'])} | "
                f"{b['warm']['total_ms']} → {r['warm']['total_ms']} | "
                f"{pct(b['warm']['total_ms'], r['warm']['total_ms'])} | "
                f"{b['warm']['storage_ops']} → {r['warm']['storage_ops']} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--label", required=True, help="before | after | any tag")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--compare", default=None, help="label to compare against")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or (Path(args.root).parent / "perf_results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run(Path(args.root), args.repeats)
    (out_dir / f"{args.label}.json").write_text(json.dumps(results, indent=2))

    prior = None
    if args.compare:
        prior_path = out_dir / f"{args.compare}.json"
        if prior_path.exists():
            prior = json.loads(prior_path.read_text())
    table = render(results, prior)
    (out_dir / f"{args.label}.md").write_text(table)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
