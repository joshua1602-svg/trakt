"""Blast-radius sweep for each vocabulary term D2 added or removed.

The "Complete"-as-geography regression was found because ONE acceptance test
happened to cover the phrasing. That is luck, not method. This runs every
question in the estate plus a set of deliberately non-idiomatic probes through
the deterministic parser and asks, for each new term: where does it match, and
is the match the intended reading?
"""
from __future__ import annotations
import json, re, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
sys.path.insert(0, "/home/user/trakt")
S = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(S))
import yaml
from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.mi_query_harness import build_fixture
from mi_agent import llm_query_parser as P

SEM = load_mi_semantics("/home/user/trakt/mi_agent/mi_semantics_field_registry.yaml")
COLS = set(build_fixture().columns)

def corpus():
    seen, out = set(), []
    cal = yaml.safe_load(Path("/home/user/trakt/config/mi/golden_questions/"
                              "ere_mi_calibration_250.yaml").read_text())
    for q in cal["questions"]:
        out.append(("calibration", q["question"]))
    sys.path.insert(0, str(S))
    from regression_bank import BANK
    out += [("30-bank", q) for q in BANK]
    out += [("80-bank", l.strip()) for l in (S / "wide_q.txt").read_text().splitlines() if l.strip()]
    from nl_bank import bank
    out += [("44-bank", e["question"]) for e in bank("alderbridge")]
    from cfo_questions import QUESTIONS
    out += [("cfo9", q) for _i, q in QUESTIONS]
    # Deliberately NON-IDIOMATIC probes: the term in a position where the new
    # reading would be wrong.
    probes = [
        # "regional" as an adjective on something other than the dimension
        "what is the regional average LTV",
        "show me regional differences in balance",
        "is there regional concentration risk",
        # "size" where it is not the measure
        "balance by ticket size",
        "show the size of the pipeline",
        "what size of loans do we write",
        "average size",
        # "best"/"worst" where a measure IS named
        "which broker has the worst arrears",
        "best performing region by balance",
        "which region has the worst LTV",
        # the "to" scope idiom in non-place positions
        "when is it expected to complete",
        "how long to reach 100m",
        "compared to last month",
        "exposure to borrowers over 85",
        "lending to joint borrowers",
        "concentration to the acquired book",
        # temporal non-place guard
        "how has the book changed relative to the prior quarter",
    ]
    out += [("probe", q) for q in probes]
    for src, q in out:
        if q.lower() not in seen:
            seen.add(q.lower())
            yield src, q


TERMS = {
    "regional": r"\bregional\b",
    "size-measure": r"\b(loan|ticket|deal|average) size\b",
    "best/worst": r"\b(best|worst)\b",
    "to-scope": r"\b(exposure|exposures|lending|concentration|allocation)\s+to\b",
    "bare-to": r"\bto\b",
}

hits = {k: [] for k in TERMS}
for src, q in corpus():
    ql = q.lower()
    matched = [k for k, pat in TERMS.items() if re.search(pat, ql)]
    if not matched:
        continue
    try:
        spec, meta = P.parse_with_repair(q, SEM, available_columns=COLS, llm_enabled=False)
    except Exception as exc:
        for k in matched:
            hits[k].append((src, q, f"PARSE ERROR {exc}"))
        continue
    desc = (f"metric={spec.metric} agg={spec.aggregation} dim={spec.dimension} "
            f"filters={spec.filters} note={meta.get('note')}")
    for k in matched:
        hits[k].append((src, q, desc))

for k in ("regional", "size-measure", "best/worst", "to-scope"):
    rows = hits[k]
    print(f"\n{'='*100}\nTERM {k}: {len(rows)} question(s) in the estate contain it")
    for src, q, d in rows:
        print(f"  [{src:11s}] {q[:62]:64s} {d}")

# The bare "to" set is large; report only those that bound a GEOGRAPHY, which is
# the only way the new idiom can misfire.
print(f"\n{'='*100}\nBARE 'to' binding a geography filter (the misfire shape):")
bad = [(s, q, d) for s, q, d in hits["bare-to"]
       if "geographic_region_obligor" in d or "collateral_geography" in d]
for s, q, d in bad:
    print(f"  [{s:11s}] {q[:62]:64s} {d}")
print(f"  total: {len(bad)} of {len(hits['bare-to'])} questions containing 'to'")
