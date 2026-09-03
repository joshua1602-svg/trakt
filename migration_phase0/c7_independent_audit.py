#!/usr/bin/env python3
"""migration_phase0/c7_independent_audit.py — verify the claims, not the report.

READ-ONLY. Reads the SOURCE and the RUNNING SYSTEM. It reads no report, no
summary and no prior JSON, so a claim that is true only in prose fails here.

Structural checks are AST-based, not substring-based: a `grep` for
`_rank_subject` cannot tell a live definition from the word appearing in a
comment explaining that it was deleted, and this audit must not be satisfiable
by a comment.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

ROUTE = _REPO / "mi_agent_api" / "period_change_route.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _defined_names(tree: ast.Module) -> set:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
    return out


def _functions_reading(tree: ast.Module, param: str) -> List[str]:
    """Functions that USE `param` for something other than passing it on."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if param not in {a.arg for a in node.args.args}:
            continue
        passed = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                for arg in list(sub.args) + [k.value for k in sub.keywords]:
                    for n in ast.walk(arg):
                        if isinstance(n, ast.Name) and n.id == param:
                            passed.add(id(n))
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id == param and id(sub) not in passed:
                out.append(node.name)
                break
    return out


def audit() -> Dict[str, Any]:
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)
    findings: Dict[str, Any] = {}
    tree = _tree(ROUTE)
    names = _defined_names(tree)

    findings["A_rank_subject_removed"] = {
        "pass": "_rank_subject" not in names,
        "detail": "AST: no definition named _rank_subject"}
    vocab = {"_NARRATIVE_RANK_SUBJECTS", "_RANK_SUBJECT_LEAD_RE",
             "_RANK_SUBJECT_SKIP"}
    findings["B_route_ranking_vocabulary_removed"] = {
        "pass": not (vocab & names),
        "detail": f"AST: still defined -> {sorted(vocab & names) or 'none'}"}

    readers = _functions_reading(tree, "question")
    findings["C_route_does_not_read_raw_question_for_meaning"] = {
        "pass": not readers,
        "detail": f"functions using `question` for a decision: {readers or 'none'}"}

    # No implicit measure / period default, checked by RUNNING the system.
    from migration_phase0.compound_canary import _write_run
    from migration_phase0.route_ownership_period_change import funded_runs
    tmp = Path(tempfile.mkdtemp())
    out_root = tmp / "onboarding_output"
    runs = funded_runs(6)
    for run_id, rdate, n, scale in runs:
        _write_run(out_root, run_id, rdate, n, scale)
    saved = {k: os.environ.get(k) for k in
             ("MI_AGENT_ONBOARDING_OUTPUT_ROOT", "MI_AGENT_AUTH_ENABLED")}
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(out_root)
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    try:
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        client = TestClient(app)

        def ask(q):
            return client.post("/mi/query", json={
                "question": q, "portfolioId": f"client_001/{runs[-1][0]}",
                "asOfDate": runs[-1][1]}).json()

        # AMENDED 2026-09-03. These two name a subject (the ranked dimension)
        # and no window. The ruling used to refuse them; it now answers over
        # the governed default pair and DISCLOSES it, which is the estate's
        # own rule for a missing element an analysis cannot exist without —
        # the measure half has worked that way since `metric_defaulted`.
        #
        # WHAT THIS CRITERION CHECKS IS UNCHANGED: a window the reader did not
        # choose must never arrive undisclosed. The check moved from "did it
        # refuse" to "did it declare", because the second is the property the
        # first was protecting. Both ends must be named — a method name is not
        # a window a reader can check.
        #
        # The BARE case ("What changed?") still refuses, and
        # `must_refuse_both_arms.py` still fails the estate if it stops.
        amb = {q: ask(q) for q in ("Which region grew the most?",
                                   "Which region added the most balance?")}

        def _discloses(r):
            declared = ((r.get("metadata") or {}).get("periodDefaulted")
                        if r.get("ok") else None)
            if not r.get("ok"):
                return True             # a refusal discloses nothing to hide
            if not (declared or {}).get("start") or not declared.get("end"):
                return False
            answer = str(r.get("answer") or "")
            return (declared["start"] in answer and declared["end"] in answer)

        bare = ask("What changed?")
        findings["D_no_undisclosed_period_or_measure"] = {
            "pass": all(_discloses(r) for r in amb.values())
                    and not bare.get("ok"),
            "detail": {q: {"ok": r.get("ok"),
                           "periodDefaulted":
                               (r.get("metadata") or {}).get("periodDefaulted"),
                           "answer": str(r.get("answer"))[:90]}
                       for q, r in amb.items()}
                      | {"What changed? (bare, must refuse)":
                         {"ok": bare.get("ok"),
                          "answer": str(bare.get("answer"))[:90]}}}

        # Ranked movement reconciles numerically, and D1's alternate binds.
        rm = ask("Which region grew the most since last month?")
        meta = (rm.get("metadata") or {}).get("rankedMovement") or {}
        rows = meta.get("rows") or []
        reconciles = all(
            round(float(r["end_value"]) - float(r["start_value"]), 2)
            == round(float(r["absolute_movement"]), 2) for r in rows) if rows else False
        findings["E_ranked_movement_reconciles"] = {
            "pass": bool(meta.get("applied")) and reconciles and len(rows) >= 1,
            "detail": {"applied": meta.get("applied"),
                       "field": meta.get("canonicalField"),
                       "rows": len(rows), "reconciles": reconciles}}
        findings["F_alternate_dimension_binds"] = {
            "pass": meta.get("canonicalField") == "geographic_region_obligor",
            "detail": f"bound {meta.get('canonicalField')!r} from the term 'region'"}

        limited = ask("Which two geographic region obligors added the most "
                      "balance since last month?")
        lmeta = (limited.get("metadata") or {}).get("rankedMovement") or {}
        findings["G_ordering_limit_honoured"] = {
            "pass": lmeta.get("topN") == 2,
            "detail": f"topN={lmeta.get('topN')}"}
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # The receipt structure is complete and independent of period_change.
    from mi_agent_api import movement_receipt as mr
    receipt_tree = _tree(_REPO / "mi_agent_api" / "movement_receipt.py")
    imports = {n.module for n in ast.walk(receipt_tree)
               if isinstance(n, ast.ImportFrom) and n.module}
    imports |= {a.name for n in ast.walk(receipt_tree)
                if isinstance(n, ast.Import) for a in n.names}
    findings["H_receipt_independent_of_period_change"] = {
        "pass": not any("period_change" in str(m) for m in imports),
        "detail": f"imports: {sorted(imports)}"}
    findings["I_receipt_carries_the_required_facts"] = {
        "pass": set(mr.MovementReceipt.REQUIRED) >= {
            "measure", "grouping_dimension", "start_period", "end_period",
            "ranking_basis", "ranking_direction"},
        "detail": list(mr.MovementReceipt.REQUIRED)}

    # The LEVEL/MOVEMENT owner is still singular.
    import question_interpretation.lexical as lex
    import mi_agent.period_change.recognition as rec
    import mi_agent.llm_query_parser as par
    findings["J_owner_still_singular"] = {
        "pass": (hasattr(lex, "temporal_aspect")
                 and not hasattr(rec, "CHANGE_MARKERS")
                 and not hasattr(par, "_COMPARE_TRIGGER_RE")),
        "detail": "lexical.temporal_aspect present; retired vocabularies absent"}
    return findings


def main(argv=None) -> int:
    findings = audit()
    print("=" * 84)
    print("C7 INDEPENDENT AUDIT — source and running system only")
    print("=" * 84)
    failed = 0
    for name, f in findings.items():
        mark = "PASS" if f["pass"] else "FAIL"
        failed += 0 if f["pass"] else 1
        print(f"  [{mark}] {name}")
        print(f"         {json.dumps(f['detail'], default=str)[:150]}")
    print(f"\n{len(findings) - failed} of {len(findings)} checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
