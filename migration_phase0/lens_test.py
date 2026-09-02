#!/usr/bin/env python3
"""Does the portfolio lens survive a period comparison? One question, two scopes.

Nine ranking refusals said "no category moved that way" and seven period-change
refusals said the population "could not be applied to both snapshots" — one of
them, CFO27, carrying no user filter at all. That points at the LENS failing on
the HISTORICAL snapshot rather than at a book that did not move: governed run
artefacts carry no source-portfolio provenance, so a comparison scoped to Direct
may have nothing on its opening side.

Asking the same twelve at scope=total, where no lens is applied, separates the
two. Answers at total + refusals at direct = the lens is the cause. Refusals at
both = the book genuinely did not move, and nine more refusals are correct.

Prints ids, outcomes and redacted reasons. No figures, no values.
"""
import os, re, sys, glob, json
os.environ["MI_AGENT_AUTH_ENABLED"] = "false"

if not os.path.isdir("mi_agent_api"):
    for p in glob.glob("/tmp/*/mi_agent_api"):
        sys.path.insert(0, os.path.dirname(p)); break
else:
    sys.path.insert(0, os.getcwd())

_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_FIG = re.compile(r"[£$€]\s?[\d,.]+\s*(?:[kKmMbB]{1,2}|MM)?|\b\d[\d,]{3,}(?:\.\d+)?\b|\b\d+\.\d+\b")
def redact(t):
    if not t: return ""
    keep = {}
    def stash(m):
        k = "\x00%d\x00" % len(keep); keep[k] = m.group(0); return k
    out = _FIG.sub("[figure]", _ISO.sub(stash, str(t)))
    for k, v in keep.items(): out = out.replace(k, v)
    return out.strip()

QUESTIONS = [
    ("Q21A", "Which region added the most balance last month for loans with LTV above 50%?"),
    ("Q21B", "For loans over 50% LTV, which region contributed the most balance growth since last month?"),
    ("CFO50", "Which region added the most balance since last month?"),
    ("CFO51", "Which region lost the most balance since last month?"),
    ("CFO52", "Which two regions added the most balance since last month?"),
    ("CFO53", "Which three regions added the most balance since last month?"),
    ("CFO54", "Which region grew fastest in balance since last month?"),
    ("CFO56", "Which region added the most balance since last month for loans with LTV above 50%?"),
    ("CFO57", "For loans with LTV above 50%, which region added the most balance since last month?"),
    ("CFO27", "How did balance change since last month?"),
    ("CFO77", "Which region grew the most?"),
    ("Q20A", "How did drawdown loans change last month?"),
]

from fastapi.testclient import TestClient
from mi_agent_api.app import app
client = TestClient(app, raise_server_exceptions=False)

def ask(q, scope):
    r = client.post("/mi/query", json={"question": q, "sourcePortfolioLens": scope})
    e = r.json()
    why = ""
    for w in (e.get("warnings") or []):
        if "ranking unavailable" in w or "both snapshots" in w:
            why = w.split(":", 1)[-1].strip()[:38]
    return bool(e.get("ok")), why or redact(e.get("error"))[:38]

print("%-7s | %-28s | %-28s" % ("id", "scope=direct", "scope=total"))
print("-" * 72)
flipped = []
for qid, q in QUESTIONS:
    ok_d, why_d = ask(q, "direct")
    ok_t, why_t = ask(q, "total")
    if ok_t and not ok_d:
        flipped.append(qid)
    print("%-7s | %-28s | %-28s" % (
        qid,
        "ANSWERED" if ok_d else why_d,
        "ANSWERED" if ok_t else why_t))
print()
print("answered at TOTAL but refused at DIRECT: %d of %d  %s"
      % (len(flipped), len(QUESTIONS), ",".join(flipped)))
print()
print("non-zero  => the portfolio lens is the cause, not the book")
print("zero      => the book genuinely did not move; those refusals are correct")
