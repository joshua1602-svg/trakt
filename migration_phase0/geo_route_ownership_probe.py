"""Does the CONTRACT separate specialist geographic exposure from generic ranking?

THE MEASUREMENT BEHIND "STOP — CONTRACT STILL CANNOT SEPARATE ROUTE OWNERSHIP".

Run it against the shipped `/mi/query` path and read the two banks side by side.
It resolves each request's real `QuestionInterpretation` through
`RouteRequest.resolve_interpretation()` — the same object a recogniser would
consult — and prints every governed field that could separate the shapes.

The finding it records: `OperationClaim.type == "ranking"` with a filled
grouping dimension is claimed by BOTH

    "Which region has the largest balance?"            (generic)
    "What is the largest geographic area concentration?"   (specialist)

and every other governed field — the four ordering values, `modifiers`,
`subject.state`, `subject.candidate_concept`, `subject.span` (the whole question
in both), the dimension claims and `residue` — is identical. Re-run this before
concluding the contract has gained a separator.

    MI_AGENT_ONBOARDING_OUTPUT_ROOT etc. are set from SWEEP_ENV (default
    /tmp/cfo_env). Point it at any onboarded book.
"""
import os, sys, warnings, logging
warnings.simplefilter("ignore"); logging.disable(logging.ERROR)
sys.path.insert(0, "/home/user/trakt")
ENV = os.environ.get("SWEEP_ENV", "/tmp/cfo_env")
os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = f"{ENV}/onboarding_output"
os.environ["TRAKT_PORTFOLIO_REGISTRY"] = f"{ENV}/portfolio_registry.yaml"
os.environ["MI_AGENT_PIPELINE_ROOT"] = "/home/user/trakt/tests/fixtures/pipeline_history_5w"
os.environ["MI_AGENT_AUTH_ENABLED"] = "false"

CAPTURED = {}
import mi_agent_api.recogniser_registry as RR
_orig = RR.RecogniserRegistry.candidates
def spy(self, request):
    CAPTURED[request.question] = request
    return _orig(self, request)
RR.RecogniserRegistry.candidates = spy

from fastapi.testclient import TestClient
from mi_agent_api.app import app
C = TestClient(app)

GENERIC = ["Which region has the largest balance?",
           "Which region has the smallest balance?",
           "What are the top three regions by balance?",
           "Which broker channel has the largest balance?"]
SPECIALIST = ["Where is the book concentrated geographically?",
              "Show geographic exposure.",
              "Analyse geographic concentration.",
              "What is the largest geographic area concentration?",
              "Which area has the largest concentration?",
              "Where are we most exposed geographically?"]

def probe(q):
    C.post("/mi/query", json={"question": q, "portfolioId": "client_001/mi_2026_06",
                              "asOfDate": "2026-06-30"})
    req = CAPTURED.get(q)
    qi = req.resolve_interpretation() if req is not None else None
    op = getattr(qi, "operation", None)
    subj = getattr(qi, "subject", None)
    dims = getattr(qi, "dimensions", None) or []
    return {
        "type": getattr(op, "type", None),
        "dir": getattr(op, "ordering_direction", None),
        "basis": getattr(op, "ordering_basis", None),
        "limit": getattr(op, "ordering_limit", None),
        "subject.state": getattr(subj, "state", None),
        "subject.field": getattr(subj, "field_key", None),
        "spec.metric": getattr(req.spec, "metric", None) if req else None,
        "spec.sort_by": getattr(req.spec, "sort_by", None) if req else None,
        "spec.ranking_mode": getattr(req.spec, "ranking_mode", None) if req else None,
        "spec.dimension": getattr(req.spec, "dimension", None) if req else None,
        "grouping_dims": [(getattr(d, "field_key", None), getattr(d, "role", None))
                          for d in dims],
        "residue": list(getattr(qi, "residue", None) or []),
        "subject.candidate": getattr(subj, "candidate_concept", None),
        "subject.has_span": getattr(subj, "has_span", None),
        "subject.span": getattr(subj, "span", None),
        "op.has_span": getattr(op, "has_span", None),
        "op.state": getattr(op, "state", None),
        "subject.raw_text": getattr(subj, "raw_text", None),
        "subject.source": getattr(subj, "source", None),
        "dim.candidates": [(getattr(d, "candidate_concept", None),
                            getattr(d, "raw_text", None)) for d in dims],
        "subject.wording": getattr(subj, "wording", None),
        "op.modifiers": tuple(getattr(op, "modifiers", ()) or ()),
        "op.wording": getattr(op, "wording", None),
    }

for label, bank in (("GENERIC RANKING", GENERIC), ("SPECIALIST GEO", SPECIALIST)):
    print("#" * 78); print(label)
    for q in bank:
        p = probe(q)
        print(f"\n  {q}")
        for k, v in p.items():
            print(f"      {k:20} {v}")
