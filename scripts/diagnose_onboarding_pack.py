"""Run a delivery's onboarding + central-tape build and report what happened.

WHY THIS EXISTS. A delivery that halts reports one sentence to its operator,
and everything that would explain the sentence — how each file was classified,
what reporting period each was read as, which file became the loan listing,
under which key column — is written into the run's project directory on
scratch storage and then thrown away with it. So a halt can only be diagnosed
by deploying a change and reading the next screen, which is a twenty-minute
loop that yields one bit per cycle.

This runs the SAME two calls the live adapter makes, against a folder of files,
and prints what they recorded.

IT REPORTS STRUCTURE, NOT DATA. Column NAMES are included, because which
columns exist is the question being asked. No cell value, no loan identifier
and no row of any file is read out — so the report can be sent to somebody who
is not entitled to the pack. ``--show-headers no`` drops the column names too,
for a pack whose schema is itself confidential.

    python -m scripts.diagnose_onboarding_pack \\
        --input-dir ./pack --client-name ERE --portfolio-id direct_001 \\
        --reporting-period 2026-08

Nothing is uploaded, promoted or registered: it writes only under --work-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _rule(title: str) -> str:
    return f"\n{'=' * 72}\n{title}\n{'=' * 72}"


def _inventory(project: Path, show_headers: bool) -> List[str]:
    out = [_rule("1. WHAT TRAKT THINKS EACH FILE IS  (01_file_inventory.json)")]
    inv = _load(project / "01_file_inventory.json") or []
    if not inv:
        out.append("  NOTHING. The inventory is empty — no file was read at "
                   "all from the input folder.")
        return out
    for i in inv:
        out.append(f"\n  {i.get('file_name', '?')}")
        out.append(f"    read as      : {i.get('classification', '?')} "
                   f"(confidence {i.get('confidence', '?')})")
        out.append(f"    sheet        : {i.get('sheet_name', '') or '(none)'}")
        out.append(f"    rows x cols  : {i.get('row_count')} x "
                   f"{i.get('column_count')}")
        if i.get("detected_reporting_date"):
            out.append(f"    date on file : {i['detected_reporting_date']}")
        if i.get("notes"):
            out.append(f"    note         : {i['notes']}")
        if show_headers:
            cols = i.get("columns") or i.get("headers") or []
            if cols:
                out.append(f"    columns      : {', '.join(map(str, cols))}")
    return out


def _eligibility(project: Path) -> List[str]:
    out = [_rule("2. WHAT PERIOD EACH FILE WAS READ AS  "
                 "(04c_source_period_eligibility.json)")]
    doc = _load(project / "04c_source_period_eligibility.json")
    # Written as {"rows": [...]}; tolerate a bare list from an older run.
    rows = (doc.get("rows") if isinstance(doc, dict) else doc) or []
    tape = [r for r in rows if isinstance(r, dict)
            and r.get("output_domain") == "central_lender_tape"]
    if not tape:
        out.append("  NOTHING. No period eligibility was recorded.")
        return out
    for r in tape:
        out.append(f"\n  {r.get('source_file', '?')}")
        out.append(f"    role            : {r.get('artefact_role', '')}")
        out.append(f"    read as period  : {r.get('inferred_reporting_period', '')}"
                   f"   (delivery is {r.get('run_reporting_period', '')})")
        out.append(f"    decided by      : {r.get('eligibility_basis', '')} "
                   f"(confidence {r.get('confidence', '')})")
        if r.get("source_period_column"):
            out.append(f"    period column   : {r['source_period_column']}")
        out.append(f"    eligible        : {r.get('is_period_eligible')}")
        out.append(f"    IS LOAN LISTING : {r.get('is_universe_source')}")
        if r.get("reason_excluded"):
            out.append(f"    set aside because: {r['reason_excluded']}")
    return out


def _universe(res: Dict[str, Any]) -> List[str]:
    out = [_rule("3. HOW THE LOAN TAPE WAS BUILT  (18f universe debug)")]
    dbg = ((res or {}).get("lender_summary") or {}).get("universe_debug") or {}
    if not dbg:
        out.append("  NOTHING. No universe debug was recorded.")
        return out
    for key in ("run_reporting_period", "period_gate_active", "universe_basis",
                "selected_universe_source_file", "selected_universe_role",
                "selected_universe_key_column",
                "selected_universe_normalisation_rule",
                "raw_universe_rows", "canonical_universe_rows",
                "duplicate_raw_keys_collapsed"):
        if key in dbg:
            out.append(f"  {key:38s}: {dbg.get(key)}")
    roles = dbg.get("universe_roles") or []
    if roles:
        out.append(f"  {'roles that count as a loan listing':38s}: "
                   f"{', '.join(map(str, roles))}")

    considered = [c for c in (dbg.get("considered_sources") or [])
                  if isinstance(c, dict)]
    out.append("\n  FILES OPENED BY THE BUILD:")
    if not considered:
        out.append("    (none)")
    for c in considered:
        out.append(f"    {c.get('source_file', '?')}")
        out.append(f"      role={c.get('artefact_role', '')} "
                   f"period={c.get('inferred_reporting_period', '')} "
                   f"eligible={c.get('period_eligible')} "
                   f"universe={c.get('is_universe_source')}")
        out.append(f"      key column={c.get('key_column', '') or '(none found)'} "
                   f"keys={c.get('key_count', 0)} rows={c.get('rows_raw', 0)} "
                   f"found={c.get('file_in_inventory')} "
                   f"loaded={c.get('frame_loaded')}")

    excluded = [e for e in (dbg.get("excluded_sources") or [])
                if isinstance(e, dict)]
    out.append("\n  FILES SET ASIDE:")
    if not excluded:
        out.append("    (none)")
    for e in excluded:
        out.append(f"    {e.get('source_file', '?')}: {e.get('reason', '')} "
                   f"(read as {e.get('inferred_reporting_period', '')}, "
                   f"{e.get('row_count', 0)} rows)")
    return out


def _verdict(res: Dict[str, Any]) -> List[str]:
    from engine.onboarding_agent import central_tape_builder as ctb
    out = [_rule("4. THE VERDICT")]
    loans = (res or {}).get("loan_count")
    tape = (res or {}).get("central_lender_tape_path") or ""
    out.append(f"  loans on the tape : {loans}")
    out.append(f"  tape written to   : {tape or '(none)'}")
    if not loans:
        out.append("\n  WHY IT IS EMPTY — what the operator would be told:")
        for s in (ctb.explain_empty_lender_tape(res) or ["(no explanation)"]):
            out.append(f"    * {s}")
    return out


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True,
                   help="Folder holding the delivery's files.")
    p.add_argument("--client-name", required=True,
                   help="The TENANT, e.g. ERE.")
    p.add_argument("--portfolio-id", default="direct_001",
                   help="The source portfolio, e.g. direct_001.")
    p.add_argument("--reporting-period", default="",
                   help="The delivery's period, e.g. 2026-08.")
    p.add_argument("--work-dir", default="",
                   help="Where to write artefacts (default: a temp folder).")
    p.add_argument("--registry", default="config/system/fields_registry.yaml")
    p.add_argument("--aliases-dir", default="config/system")
    p.add_argument("--show-headers", choices=("yes", "no"), default="yes",
                   help="Include column NAMES (never values). Default yes.")
    a = p.parse_args(argv)

    from engine.onboarding_agent import (central_tape_builder, storage_paths,
                                         workflow as _wf)

    in_dir = Path(a.input_dir).resolve()
    files = sorted(f.name for f in in_dir.iterdir()
                   if f.is_file()) if in_dir.is_dir() else []
    work = Path(a.work_dir).resolve() if a.work_dir else Path(
        tempfile.mkdtemp(prefix="trakt_diag_"))
    work.mkdir(parents=True, exist_ok=True)

    report: List[str] = [
        _rule("0. WHAT WAS HANDED TO THE RUN"),
        f"  input folder : {in_dir}",
        f"  files in it  : {len(files)}",
    ]
    report += [f"    - {n}" for n in files] or ["    (none)"]
    report += [f"  tenant       : {a.client_name}",
               f"  portfolio    : {a.portfolio_id}",
               f"  period       : {a.reporting_period or '(not given)'}"]

    # THE RUN ID CARRIES THE PERIOD, and the period gate is the thing under
    # test. `run_period()` reads the period from the run id (or an input folder
    # named for it) and returns "" otherwise — and an empty run period makes
    # the funded-period match permissive, so a diagnostic run named `run` would
    # pass a pack that production sets aside. Named for the period, this gate
    # is the one production applies.
    period = (a.reporting_period or "").strip()
    run_id = f"mi_{period.replace('-', '_')}" if period else "run"
    report.append(f"  run id       : {run_id}")

    res: Dict[str, Any] = {}
    try:
        _wf.run_operator_workflow(
            input_dir=str(in_dir), client_name=a.client_name,
            client_id=a.portfolio_id, run_id=run_id, project_dir=str(work),
            mode="mi_only", registry=a.registry, aliases_dir=a.aliases_dir,
            enable_mapping_review=True,
            reporting_date=a.reporting_period,
            reporting_period=a.reporting_period,
            managed_service=True)
    except Exception as exc:  # noqa: BLE001 — a failure here IS the finding
        report.append(_rule("ONBOARDING RAISED"))
        report.append(f"  {type(exc).__name__}: {exc}")

    try:
        run_paths = storage_paths.resolve_run_paths(
            project_dir=str(work), input_dir=str(in_dir), output_root=None,
            client_id=a.portfolio_id, run_id=run_id,
            storage_backend="local", input_uri="", output_uri="")
        res = central_tape_builder.build_central_tapes(
            str(work), run_paths, a.registry, mode="mi_only") or {}
    except Exception as exc:  # noqa: BLE001
        report.append(_rule("CENTRAL TAPE BUILD RAISED"))
        report.append(f"  {type(exc).__name__}: {exc}")

    report += _inventory(work, a.show_headers == "yes")
    report += _eligibility(work)
    report += _universe(res)
    report += _verdict(res)
    report.append(f"\n  full artefacts under: {work}")

    text = "\n".join(report)
    print(text)
    (work / "DIAGNOSIS.txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
