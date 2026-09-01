#!/usr/bin/env python3
"""scripts/run_mi_query_multivariate_audit.py — audit only. Changes nothing.

Runs the multivariate pipeline bank through the REAL current `/mi/query` path
and captures, for every question, the evidence needed to attribute a failure to
a LAYER rather than to "recognition" generically:

    parsed interpretation · dataset · filters · grouping · measure ·
    time/comparison · route owner · answer · receipt · verdict

It reuses the existing bank architecture rather than introducing another one:
the same YAML case/formulation schema, the same seeded funded tape from
`run_mi_query_stage_movement_banks`, the same `POST /mi/query` entry, and the
same DELIVER/`must`/`must_not` grading shape. The only addition is the evidence
capture above, which is recorded for the report and changes no verdict.

    python scripts/run_mi_query_multivariate_audit.py --out out/mv
    python scripts/run_mi_query_multivariate_audit.py --out out/mv-lang \\
        --concept-merge --model claude-opus-5

The pipeline root is `tests/fixtures/pipeline_multivariate`, whose expected
answers are proven independently by
`scripts/prove_multivariate_pipeline_fixture.py`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

def _stage_bank_module():
    """The existing bank runner, loaded BY PATH.

    Importing it as `scripts.run_...` would require making `scripts/` a package,
    and adding an `__init__.py` there changes import semantics for every other
    script in the tree — not something an audit may do. This reuses the existing
    fixture builder without changing anything about how it is packaged.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_mi_stage_bank_runner",
        _REPO / "scripts" / "run_mi_query_stage_movement_banks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_STAGE_BANK = _stage_bank_module()
MONTHS = _STAGE_BANK.MONTHS
PORTFOLIO = _STAGE_BANK.PORTFOLIO
write_funded_tape = _STAGE_BANK.write_funded_tape

BANK = (_REPO / "tests" / "fixtures" / "mi_query_stage_movement"
        / "MULTIVARIATE_PIPELINE_BANK.yaml")
PIPELINE_FIXTURE = _REPO / "tests" / "fixtures" / "pipeline_multivariate"
AS_OF = "2026-06-30"

CORRECT = "CORRECT"
WRONG = "WRONG"
DECLINE = "HONEST_DECLINE"
CLARIFIED = "AMBIGUOUS_AND_CORRECTLY_CLARIFIED"


def build_client(concept_merge: bool, model: Optional[str]):
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)
    root = Path(tempfile.mkdtemp(prefix="mv_audit_")) / "onboarding_output"
    for run_id, date, rows in MONTHS:
        write_funded_tape(root, run_id, date, rows)
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(root)
    os.environ["MI_AGENT_PIPELINE_ROOT"] = str(PIPELINE_FIXTURE)
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    os.environ.setdefault("MI_AGENT_LLM_PARSER", "off")
    if concept_merge:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("--concept-merge needs ANTHROPIC_API_KEY")
        os.environ["MI_AGENT_CONCEPT_MERGE"] = "on"
        if model:
            os.environ["MI_AGENT_CONCEPT_MERGE_MODEL"] = model
    else:
        os.environ["MI_AGENT_CONCEPT_MERGE"] = "off"

    from fastapi.testclient import TestClient
    from mi_agent_api import concept_merge_arm as merge_arm
    from mi_agent_api import datasets as ds
    from mi_agent_api.app import app

    cfg = ds._mi_llm_config()
    if cfg.enabled or cfg.available:
        raise SystemExit("AUDIT INVALID - the free-form LLM parser arm is live")
    print("   language layer: %s" % ("concept merge ON, model %s"
                                     % merge_arm.model_name()
                                     if merge_arm.enabled() else "OFF"))
    return TestClient(app)


def bank() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    import yaml

    doc = yaml.safe_load(BANK.read_text())
    cases = doc["cases"]
    by_id = {f["id"]: c for c in cases for f in c["formulations"]}
    return cases, by_id


def evidence(env: Dict[str, Any]) -> Dict[str, Any]:
    """What the agent says it did — read from the envelope, never inferred."""
    meta = env.get("metadata") or {}
    spec = env.get("spec") if isinstance(env.get("spec"), dict) else {}
    summary = env.get("executionSummary") or {}
    guard = env.get("semanticGuard") or {}
    artifacts = env.get("artifacts") or []
    return {
        "route": meta.get("route"),
        "dataset_context": meta.get("datasetContext"),
        "reconciliation_dataset": (env.get("reconciliation") or {}).get("dataset"),
        "spec_filters": spec.get("filters"),
        "spec_dimensions": (list(spec.get("dimensions") or [])
                            + ([spec["dimension"]] if spec.get("dimension") else [])),
        "spec_metric": spec.get("metric"),
        "spec_aggregation": spec.get("aggregation"),
        "spec_temporal_mode": spec.get("temporal_mode"),
        "unavailable_filters": spec.get("unavailable_filters"),
        "population_applied": meta.get("populationApplied"),
        "analytical_intent": (meta.get("analyticalIntent") or {}).get("families"),
        "guard_verdict": guard.get("verdict"),
        "guard_facets": [
            {"kind": f.get("kind"), "field": f.get("field"),
             "label": f.get("label"), "status": f.get("status")}
            for f in (guard.get("facets") or [])],
        "receipt": summary.get("line") or _receipt_from(env.get("answer") or ""),
        "artifact_rows": max([len(a.get("rows") or []) for a in artifacts] or [0]),
        "artifact_sample": [a.get("rows") or [] for a in artifacts][:1],
        "artifact_titles": [a.get("title") for a in artifacts][:4],
        "warnings": [str(w)[:200] for w in (env.get("warnings") or [])][:4],
        "unaccounted": [
            c.get("term") or c.get("value")
            for c in ((meta.get("semanticCoverage") or {}).get("unaccounted") or [])],
    }


def _receipt_from(answer: str) -> str:
    parts = answer.split("\n\nCalculated:")
    return ("Calculated:" + parts[1]).strip() if len(parts) > 1 else ""


def _norm_digits(text: str) -> str:
    """Answer text with thousands separators removed, so 2,960,000 matches 2960000."""
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def grade(case: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, str]:
    """The bank's frozen assertions, applied to the answer text.

    Deliberately the same shape the stage-movement bank uses: a required figure
    and a forbidden one, where the forbidden figure is what the answer would
    carry if a concept had been dropped.
    """
    answer = row["answer"] or ""
    expect = str(case.get("expect") or "DELIVER")

    if not row["ok"]:
        if expect == "DELIVER_OR_CLARIFY":
            return {"grade": CLARIFIED,
                    "why": "the governed period is undefined; refusing is correct"}
        return {"grade": DECLINE, "why": (row["error"] or "")[:220]}

    # THE ANSWER IS THE WHOLE ENVELOPE, not the prose. A grouped answer states
    # its groups in the RECEIPT and its values in the ARTIFACT ROWS — grading
    # the sentence alone marked four correct breakdowns wrong.
    receipt = (row["evidence"].get("receipt") or "")
    rows_text = json.dumps(row["evidence"].get("artifact_sample") or [])
    low = _norm_digits((answer + " " + rows_text).lower())
    receipt_low = receipt.lower()

    must_not = [str(m) for m in (case.get("must_not") or [])]
    hit = [m for m in must_not if _norm_digits(m.lower()) in low]
    if hit:
        return {"grade": WRONG,
                "why": "carries the figure a dropped concept would give: %s" % hit}

    must = [str(m) for m in (case.get("must") or [])]
    if must:
        found = [m for m in must if _norm_digits(m.lower()) in low]
        needed = must if case.get("must_all") else must[:1]
        if (case.get("must_all") and len(found) < len(must)) or not found:
            missing = [m for m in must if m not in found]
            return {"grade": WRONG,
                    "why": "required figure(s) absent: %s" % missing}

    for phrase in (case.get("must_receipt") or []):
        if str(phrase).lower() not in receipt_low:
            return {"grade": WRONG,
                    "why": "the receipt does not state %r (receipt: %s)"
                           % (phrase, receipt[:120])}

    min_rows = case.get("min_rows")
    if min_rows and row["evidence"]["artifact_rows"] < int(min_rows):
        return {"grade": WRONG,
                "why": "expected at least %s grouped rows, got %d"
                       % (min_rows, row["evidence"]["artifact_rows"])}
    return {"grade": CORRECT, "why": ""}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concept-merge", action="store_true")
    ap.add_argument("--model")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    cases, by_id = bank()
    client = build_client(args.concept_merge, args.model)

    rows: List[Dict[str, Any]] = []
    for case in cases:
        for form in case["formulations"]:
            env = client.post("/mi/query", json={
                "question": form["q"], "portfolioId": PORTFOLIO,
                "asOfDate": AS_OF}).json()
            row = {
                "case": case["id"], "id": form["id"], "question": form["q"],
                "construction": case.get("construction"),
                "expected": case.get("expected"),
                "ok": bool(env.get("ok")),
                "answer": (env.get("answer") or "").strip(),
                "error": (env.get("error") or "").strip(),
                "evidence": evidence(env),
            }
            row.update(grade(case, row))
            rows.append(row)
            print(".", end="", flush=True)
    print()

    (args.out / "multivariate.json").write_text(
        json.dumps(rows, indent=1, default=str), encoding="utf-8")

    counts = Counter(r["grade"] for r in rows)
    total = len(rows)
    correct = counts[CORRECT]
    safe = correct + counts[DECLINE] + counts[CLARIFIED]
    print("  %s" % dict(counts))
    print("  correct rate %.1f%%   safe rate %.1f%%"
          % (100.0 * correct / total, 100.0 * safe / total))
    print("  SILENT WRONG ANSWERS: %d" % counts[WRONG])
    for r in rows:
        if r["grade"] == WRONG:
            print("    %s  %s" % (r["id"], r["question"]))
            print("        %s" % r["why"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
