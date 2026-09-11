import json, os, sys
sys.path.insert(0, os.getcwd())
from mi_agent.llm_query_parser import parse_with_repair
from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.query_plan_adapter import plan_from_spec
from mi_agent_api.recogniser_registry import REGISTRY, RouteRequest
import mi_agent_api.chat_routing  # register
SEM = load_mi_semantics("mi_agent/mi_semantics_field_registry.yaml")
QS = [
 "Summarise the portfolio.",
 "What has changed versus last month?",
 "Give me the stage movement summary.",
 "How many cases went from KFI into Application?",
 "What is our borrowing base?",
 "What is the facility utilisation?",
 "What is the current pipeline position?",
 "How much is in the pipeline by stage?",
 "Show cohort progression by vintage.",
 "Track the 2023 vintage across reporting dates.",
 "Which brokers have the largest exposure?",
 "What is the top 5 regions by balance?",
 "What percentage of the book has LTV above 50%?",
 "Show funded balance evolution by month.",
 "Compare October and November funded balance.",
 "When do we reach 100m?",
 "If conversion improved by 10% when do we reach 50m?",
 "Are we breaching any concentration limits?",
 "What drove the change in funded balance?",
 "Where is the book concentrated?",
 "What is the balance by broker in the acquired book?",
 "What is the total funded balance?",
]
for q in QS:
    spec, meta = parse_with_repair(q, SEM, llm_enabled=False)
    sd = spec.to_dict() if hasattr(spec,'to_dict') else {}
    req = RouteRequest(question=q, spec=spec, spec_dict=sd, semantics=SEM, view="funded",
                       client_id="c", run_id=None, portfolio_id=None)
    claims=[]
    for rec in REGISTRY.ordered():
        try:
            v = rec.recognise(req.for_recognition())
        except Exception as e:
            continue
        if getattr(v,'matched',v): claims.append(rec.name)
    lift = plan_from_spec(spec) is not None
    print(f"{'LIFT' if lift else '----'} | {(claims[0] if claims else '<generic>'):26s} | {q}")
