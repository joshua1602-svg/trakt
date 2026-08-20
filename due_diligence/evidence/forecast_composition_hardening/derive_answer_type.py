"""Derive expected_answer_type for all 252 calibration cases, and REPORT every
place the question's wording and the declared metric disagree.

The derivation is: the question's own wording decides, cross-checked against the
declared expected_metric. Where they disagree the case is listed rather than
silently resolved -- a disagreement means either the wording classifier or the
declared metric is wrong, and that is a finding either way.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
sys.path.insert(0, "/home/user/trakt")
import yaml
from mi_agent import answer_type as A
from mi_agent.mi_query_validator import load_mi_semantics

SEM = load_mi_semantics("/home/user/trakt/mi_agent/mi_semantics_field_registry.yaml")
BANK = Path("/home/user/trakt/config/mi/golden_questions/ere_mi_calibration_250.yaml")
doc = yaml.safe_load(BANK.read_text())

rows, disagree = [], []
for c in doc["questions"]:
    q, status = c["question"], c.get("expected_status")
    if status in ("refuse", "clarify"):
        t = A.NONE
    elif c.get("execution") == "parse_only":
        t = A.ANY          # never executed; nothing to type
    else:
        from_q = A.asked(q)
        em = c.get("expected_metric")
        from_m = A.of_measure(em, None, SEM) if em else None
        if em and from_m and not A.satisfies(from_q, from_m):
            disagree.append((c["id"], q, from_q, from_m, em))
            t = from_m     # the declared metric wins; the disagreement is reported
        elif em:
            t = from_m
        else:
            t = from_q
    rows.append((c["id"], t))

print(f"derived {len(rows)} answer types")
import collections
print(dict(collections.Counter(t for _i, t in rows)))
print(f"\nDISAGREEMENTS between question wording and declared metric: {len(disagree)}")
for i, q, fq, fm, em in disagree:
    print(f"  {i:12s} {q[:52]:54s} wording={fq:8s} metric={fm:8s} ({em})")
Path("d/answer_types.yaml").write_text(
    yaml.safe_dump({i: t for i, t in rows}, sort_keys=False))
