#!/usr/bin/env python3
"""Generate a static funded-MI dashboard review from REAL /mi/query envelopes.

Runs the MI Agent API (in-process, via FastAPI TestClient — no server needed)
against a promoted, MI-prepared funded central lender tape for one or more runs,
captures /health + the funded MI query set, and writes a static HTML review pack:

    artifacts/funded_mi_dashboard_review.html

The HTML uses the SAME envelopes the React dashboard consumes (kpi/chart/table/
validation artefacts) — not hand-written values — with a section per run showing:
API health, KPI summary, available stratifications, and missing dimensions with
reason codes. It is a STATIC PREVIEW (clearly marked), not a browser screenshot.

Usage (point at promoted runs under an onboarding output root):

    python mi_agent_api/scripts/generate_dashboard_review.py \
        --output-root onboarding_output/client_001 \
        --client-id client_001 --runs mi_2025_10 mi_2025_11 \
        --out artifacts/funded_mi_dashboard_review.html
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

QUERIES = [
    "portfolio summary",
    "current outstanding balance by ltv bucket",
    "current outstanding balance by ticket bucket",
    "current outstanding balance by interest rate bucket",
    "current outstanding balance by time on book bucket",
    "current outstanding balance by age bucket",
    "current outstanding balance by region",
    "current outstanding balance by original ltv bucket",
    "current outstanding balance by broker channel",
]


def _num(v: Any) -> Optional[float]:
    """Parse one value via the SHARED deterministic parser (no separate regex)."""
    import pandas as pd
    from analytics_lib.numeric import coerce_numeric
    s = coerce_numeric(pd.Series([v])).iloc[0]
    return float(s) if pd.notna(s) else None


def _central_tape(output_root: Path, client_id: str, run_id: str) -> Optional[Path]:
    for c in (output_root / run_id / "output" / "central" / "18_central_lender_tape.csv",
              output_root / client_id / run_id / "output" / "central" / "18_central_lender_tape.csv",
              output_root / "central" / "18_central_lender_tape.csv"):
        if c.exists():
            return c
    hits = sorted(output_root.glob(f"**/{run_id}/**/18_central_lender_tape.csv"))
    return hits[0] if hits else None


def _serve(tape: Path, client_id: str, run_id: str):
    from mi_agent_api import data_source
    for k in ("MI_AGENT_DATA_CSV", "MI_AGENT_ANALYTICS_DATASET", "MI_AGENT_DISABLE_PREP"):
        os.environ.pop(k, None)
    os.environ["MI_AGENT_CENTRAL_TAPE"] = str(tape)
    os.environ["MI_AGENT_CLIENT_ID"] = client_id
    os.environ["MI_AGENT_RUN_ID"] = run_id
    data_source.reset_cache()


def _artifact_table(env: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a compact (dimension -> balance) view + totals from an envelope."""
    arts = env.get("artifacts", [])
    types = [a.get("type") for a in arts]
    rows: List[Dict[str, str]] = []
    total = 0.0
    n = 0
    # KPI summary
    kpi = next((a for a in arts if a.get("type") == "kpi"), None)
    if kpi:
        for k in kpi.get("kpis", []):
            rows.append({"label": str(k.get("label", "")), "value": str(k.get("value", ""))})
    # chart/table -> dimension breakdown
    tab = next((a for a in arts if a.get("type") in ("chart", "table")), None)
    if tab:
        xkey = tab.get("xKey")
        if not xkey and tab.get("columns"):
            xkey = (tab["columns"][0] or {}).get("key")
        series = tab.get("series") or []
        ykey = (tab.get("yKey") or tab.get("valueKey")
                or (series[0].get("key") if series else None))
        for r in (tab.get("rows") or [])[:30]:
            label = str(r.get(xkey, "")) if xkey else ""
            val = r.get(ykey) if ykey else None
            fv = _num(val)
            if fv is not None:
                total += fv
                n += 1
            rows.append({"label": label,
                         "value": (f"{fv:,.0f}" if fv is not None else ("" if val is None else str(val)))})
    return {"types": types, "rows": rows, "group_total": round(total, 2) if n else None,
            "group_count": n}


def _query(client, question: str) -> Dict[str, Any]:
    body = client.post("/mi/query", json={"question": question}).json()
    info = _artifact_table(body)
    val = body.get("validation", {}) or {}
    return {
        "question": question,
        "ok": body.get("ok"),
        "artifacts": info["types"],
        "rowCount": (body.get("metadata", {}) or {}).get("rowCount"),
        "resultType": (body.get("metadata", {}) or {}).get("resultType"),
        "group_total": info["group_total"],
        "group_count": info["group_count"],
        "rows": info["rows"],
        "errors": val.get("errors", []),
        "answer": body.get("answer", ""),
    }


def collect_run(output_root: Path, client_id: str, run_id: str, registry: str) -> Dict[str, Any]:
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app
    from mi_agent_api import data_source, funded_mi_trace

    tape = _central_tape(output_root, client_id, run_id)
    if tape is None:
        return {"run_id": run_id, "error": f"no 18_central_lender_tape.csv under {output_root}"}
    _serve(tape, client_id, run_id)
    client = TestClient(app)
    health = client.get("/health").json()
    queries = [_query(client, q) for q in QUERIES]
    # Per-field trace (raw -> ... -> React), best-effort (needs the project dir).
    project_dir = tape.parents[2]  # .../output/central/tape -> project dir
    trace_rows = []
    try:
        trace_rows = funded_mi_trace.trace(project_dir, tape)
    except Exception as exc:  # pragma: no cover
        trace_rows = [{"error": str(exc)}]
    import pandas as pd
    from analytics_lib.numeric import coerce_numeric
    df = pd.read_csv(tape)
    bal = coerce_numeric(df.get("current_outstanding_balance")).sum()
    return {
        "run_id": run_id, "tape": str(tape), "row_count": int(len(df)),
        "aggregate_balance": round(float(bal), 2),
        "health": health, "queries": queries, "trace": trace_rows,
    }


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

def _kind_banner(kind: str, prepared: bool) -> str:
    label = {"funded_mi_prepared_dataset": "REAL prepared funded MI dataset",
             "funded_central_lender_tape_raw": "RAW central tape (thin, KPI-only)",
             "synthetic_demo": "SYNTHETIC DEMO data",
             "explicit_csv": "explicit CSV"}.get(kind, kind)
    colour = "#1d7a4d" if kind == "funded_mi_prepared_dataset" else (
        "#b46a00" if kind == "funded_central_lender_tape_raw" else "#a11")
    return (f'<div class="banner" style="background:{colour}">Data source: '
            f'<b>{escape(label)}</b> · dataSourceKind=<code>{escape(str(kind))}</code> · '
            f'preparationApplied=<b>{prepared}</b></div>')


def _run_section(run: Dict[str, Any]) -> str:
    if run.get("error"):
        return f'<h2>{escape(run["run_id"])}</h2><p class="err">{escape(run["error"])}</p>'
    h = run["health"]
    kind = h.get("dataSourceKind", "")
    out = [f'<h2>Run {escape(run["run_id"])}</h2>',
           _kind_banner(kind, h.get("preparationApplied", False))]
    out.append(f'<p><b>Funded universe:</b> {run["row_count"]} loans · '
               f'<b>aggregate current_outstanding_balance:</b> £{run["aggregate_balance"]:,.2f}</p>')

    # Health: dimensions available / missing (reason-coded)
    avail = ", ".join(escape(d) for d in h.get("dimensionsAvailable", [])) or "—"
    miss = h.get("missingDimensions", [])
    miss_rows = "".join(
        f"<tr><td><code>{escape(str(m.get('dimension')))}</code></td>"
        f"<td><code>{escape(str(m.get('reason')))}</code></td>"
        f"<td>{escape(str(m.get('detail','')))}</td></tr>" for m in miss)
    out.append(f'<h3>API health — dimensions</h3><p><b>dimensionsAvailable:</b> {avail}</p>')
    out.append('<table><thead><tr><th>missing dimension</th><th>reason</th><th>detail</th></tr>'
               f'</thead><tbody>{miss_rows or "<tr><td colspan=3>none</td></tr>"}</tbody></table>')

    # Query results
    out.append('<h3>/mi/query results (envelopes the React dashboard consumes)</h3>')
    out.append('<table><thead><tr><th>query</th><th>ok</th><th>artefacts</th>'
               '<th>rows</th><th>group Σ balance</th><th>reason / answer</th></tr></thead><tbody>')
    for q in run["queries"]:
        status = "✓" if q["ok"] else "✗"
        arts = ", ".join(str(a) for a in q["artifacts"]) or "—"
        gt = f'£{q["group_total"]:,.0f}' if q["group_total"] is not None else "—"
        reason = (escape("; ".join(q["errors"])) if not q["ok"] and q["errors"]
                  else escape(str(q["answer"]))[:90])
        out.append(f'<tr><td>{escape(q["question"])}</td><td class="c">{status}</td>'
                   f'<td>{escape(arts)}</td><td class="c">{q["group_count"] or q["rowCount"] or "—"}</td>'
                   f'<td class="r">{gt}</td><td>{reason}</td></tr>')
    out.append('</tbody></table>')

    # One example stratification breakdown (first chart query that succeeded)
    strat = next((q for q in run["queries"] if q["ok"] and "chart" in q["artifacts"]
                  and q["question"] != "portfolio summary"), None)
    if strat:
        cells = "".join(f"<tr><td>{escape(r['label'])}</td><td class='r'>{escape(r['value'])}</td></tr>"
                        for r in strat["rows"][:20])
        out.append(f'<h3>Example stratification — {escape(strat["question"])}</h3>'
                   f'<table><thead><tr><th>bucket</th><th>value</th></tr></thead>'
                   f'<tbody>{cells}</tbody></table>')

    # Trace summary
    tr_rows = "".join(
        f"<tr><td><code>{escape(str(t.get('canonical_field')))}</code></td>"
        f"<td>{escape(str(t.get('in_central_tape')))} ({t.get('non_null_count','')})</td>"
        f"<td>{escape(str(t.get('dimension_available')))}</td>"
        f"<td><b>{escape(str(t.get('status')))}</b></td>"
        f"<td><code>{escape(str(t.get('reason')))}</code></td></tr>"
        for t in run["trace"] if "canonical_field" in t)
    out.append('<h3>Per-field trace (raw → mapping → MI contract → tape → prep → React)</h3>')
    out.append('<table><thead><tr><th>field</th><th>in tape (non-null)</th><th>dim avail</th>'
               f'<th>status</th><th>reason</th></tr></thead><tbody>{tr_rows}</tbody></table>')
    return "\n".join(out)


def render(runs: List[Dict[str, Any]], note: str) -> str:
    body = "\n".join(_run_section(r) for r in runs)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<title>Funded MI dashboard review</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,Arial;margin:24px;color:#13233a;background:#f6f8fb}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:30px;border-top:2px solid #dde}} h3{{font-size:13px;color:#33506f}}
.banner{{color:#fff;padding:8px 12px;border-radius:8px;font-size:13px;margin:6px 0}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:12px;margin:6px 0 14px}}
th,td{{border:1px solid #e2e8f0;padding:5px 8px;text-align:left}} th{{background:#eef2f7}}
td.c{{text-align:center}} td.r{{text-align:right;font-variant-numeric:tabular-nums}}
code{{background:#eef;padding:0 3px;border-radius:3px}} .err{{color:#a11}}
.note{{background:#fff7e6;border:1px solid #f0d9a0;padding:10px;border-radius:8px;font-size:13px}}
</style></head><body>
<h1>Funded MI dashboard review — static preview</h1>
<div class="note">{note}<br>Generated {ts}. This is a <b>static preview</b> built from the real
<code>/mi/query</code> envelopes the React dashboard consumes (not a live browser screenshot,
not hand-written values).</div>
{body}
</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--client-id", default="client_001")
    ap.add_argument("--runs", nargs="+", default=["mi_2025_10", "mi_2025_11"])
    ap.add_argument("--registry", default="config/system/fields_registry.yaml")
    ap.add_argument("--out", default="artifacts/funded_mi_dashboard_review.html")
    ap.add_argument("--note", default="Source: promoted funded central lender tape (MI-prepared).")
    args = ap.parse_args()

    root = Path(args.output_root)
    runs = [collect_run(root, args.client_id, r, args.registry) for r in args.runs]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(runs, escape(args.note)), encoding="utf-8")
    for r in runs:
        if r.get("error"):
            print(f"{r['run_id']}: ERROR {r['error']}")
        else:
            k = r["health"].get("dataSourceKind")
            print(f"{r['run_id']}: {r['row_count']} loans £{r['aggregate_balance']:,.0f} "
                  f"kind={k} avail={len(r['health'].get('dimensionsAvailable',[]))} "
                  f"missing={len(r['health'].get('missingDimensions',[]))}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
