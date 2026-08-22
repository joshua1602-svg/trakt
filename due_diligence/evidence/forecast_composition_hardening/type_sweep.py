"""TYPE-CONFORMANCE SWEEP — does the answer's TYPE match the question's type?

Why this exists
---------------
Two banks certify a case as correct by BYTE-EQUALITY against a frozen baseline.
That proves stability, not correctness: any wrong answer already in the baseline
is graded correct for ever. Two such cases were found by accident while fixing
D2 — "How many loans are in the back book?" answered with a BALANCE, and "show
loan count by age band" with AVERAGE BORROWER AGE. This sweep looks for the rest
mechanically.

It never reads answer prose. The ASKED type comes from a small, explicit
question vocabulary; the RETURNED type comes from the spec's metric and
aggregation joined to the registry's declared format. A disagreement between
the two is a finding, whatever the baseline says.
"""
from __future__ import annotations
import json, re, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
sys.path.insert(0, "/home/user/trakt")
from mi_agent.mi_query_validator import load_mi_semantics

SEM = load_mi_semantics("/home/user/trakt/mi_agent/mi_semantics_field_registry.yaml")
FIELDS = SEM.get("fields") or SEM

COUNT, CURRENCY, RATE, AGE, DATE, CATEGORY, ANY = (
    "COUNT", "CURRENCY", "RATE", "AGE", "DATE", "CATEGORY", "ANY")

#: The question vocabulary, deliberately small and explicit.
_ASK = (
    (DATE,     r"\bwhen\b|\bhow long until\b|\bby when\b|\bwhat date\b"),
    (COUNT,    r"\bhow many\b|\bnumber of\b|\bloan count\b|\bcase count\b|\bcount\b"),
    (RATE,     r"\bltv\b|\bloan to value\b|\binterest rate\b|\bwhat rate\b|"
               r"\bpercentage\b|\bproportion\b|\bshare of\b|\bwhat %|\bcoupon\b"),
    (AGE,      r"\bborrower age\b|\bage of\b|\bhow old\b"),
    (CURRENCY, r"\bbalance\b|\bexposure\b|\baum\b|\bamount\b|\bvalue of\b|"
               r"\bhow much\b|\bloan size\b|£"),
)
#: A question that names a grouping wants a BREAKDOWN; one that does not wants a
#: single figure. "by"/"per"/"across"/"split"/"breakdown"/"distribution".
_GROUPING_RE = re.compile(r"\bby\b|\bper\b|\bacross\b|\bsplit\b|\bbreakdown\b|"
                          r"\bdistribution\b|\bheatmap\b|\bmix\b")
#: A RANKING or COMPARISON question needs a grouping to answer at all: "which
#: region has the largest exposure" must group by region before it can pick one,
#: and "front book versus back book" must group by seasoning. Counting those as
#: unrequested groupings measured the classifier, not the product.
_IMPLIED_GROUPING_RE = re.compile(
    r"\bwhich\b|\bwhere is\b|\bwhere are\b|\bwho\b|\btop\b|\bbottom\b|"
    r"\blargest\b|\bbiggest\b|\bhighest\b|\blowest\b|\bsmallest\b|\bmost\b|"
    r"\bleast\b|\bworst\b|\bbest\b|\bconcentration\b|\bconcentrated\b|"
    r"\bcompare[sd]?\b|\bcomparison\b|\bversus\b|\bvs\b|\bbetter or worse\b|"
    r"\brank(?:ed|ing)?\b|\bnew lending\b|\bfront book\b|\bback book\b")


def asked_type(q: str) -> str:
    text = (q or "").lower()
    for kind, pattern in _ASK:
        if re.search(pattern, text):
            return kind
    return ANY


def asked_breakdown(q: str) -> bool:
    text = (q or "").lower()
    return bool(_GROUPING_RE.search(text) or _IMPLIED_GROUPING_RE.search(text))


def metric_type(metric, aggregation) -> str:
    agg = (aggregation or "").lower()
    if agg in ("count", "count_distinct") and not metric:
        return COUNT
    if agg == "share":
        return RATE
    entry = (FIELDS.get(metric) or {}) if metric else {}
    fmt = str(entry.get("format") or "").lower()
    unit = str(entry.get("unit") or "").lower()
    if fmt in ("currency", "money") or unit in ("gbp", "currency"):
        return CURRENCY
    if fmt in ("percent", "percentage", "rate"):
        return RATE
    if "age" in str(metric or ""):
        return AGE
    if fmt in ("integer", "count") and agg in ("count", "sum"):
        return COUNT
    if fmt:
        return fmt.upper()
    # A measure the FUNDED registry does not carry — pipeline_amount and its
    # siblings live in the pipeline contract, not in the field registry — is
    # typed from its name. Without this the sweep reported every pipeline
    # answer as an unknown type, which measures the classifier's reach rather
    # than the product's behaviour.
    name = str(metric or "").lower()
    if any(t in name for t in ("amount", "balance", "value", "exposure")):
        return CURRENCY
    if any(t in name for t in ("rate", "ltv", "yield", "conversion", "share")):
        return RATE
    if "count" in name or name.endswith("_n"):
        return COUNT
    return ANY if not metric else "OTHER:" + str(metric)


#: Which returned types are acceptable for each asked type. A question that
#: names no measure ("breakdown by region") accepts anything: the balance
#: default is documented behaviour, not a substitution.
_OK = {
    COUNT:    {COUNT},
    CURRENCY: {CURRENCY},
    RATE:     {RATE, CURRENCY},   # "share of balance" legitimately reports both
    AGE:      {AGE, COUNT, CURRENCY},
    DATE:     {DATE, ANY, CURRENCY, COUNT},   # horizon routes answer in prose
    CATEGORY: {CURRENCY, COUNT, RATE},
    ANY:      None,
}


def sweep(rows, label, declared=None):
    findings = []
    for r in rows:
        q = r["q"]
        if r.get("refusal") or r.get("ok") is False:
            continue
        want, got = asked_type(q), metric_type(r.get("metric"), r.get("aggregation"))
        allowed = _OK.get(want)
        if allowed is not None and got not in allowed:
            findings.append((q, "TYPE", f"asked {want}, returned {got} "
                                        f"(metric={r.get('metric')} agg={r.get('aggregation')})"))
        has_dim = bool(r.get("dimension") or r.get("dimensions"))
        if has_dim and not asked_breakdown(q):
            findings.append((q, "SHAPE", f"no grouping requested, grouped by "
                                         f"{r.get('dimension') or r.get('dimensions')}"))
    print(f"\n{'='*96}\n{label}: {len(rows)} cases, {len(findings)} type/shape findings")
    for q, kind, detail in findings:
        print(f"  {kind:6s} {q[:66]:68s} {detail}")
    return findings


if __name__ == "__main__":
    total = []
    for path, label in ((sys.argv[i], sys.argv[i + 1])
                        for i in range(1, len(sys.argv), 2)):
        total += sweep(json.load(open(path)), label)
    print(f"\nTOTAL type/shape findings: {len(total)}")
