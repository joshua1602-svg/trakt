"""Read-only probe: which grouped paths disclose at which arity, today."""
import os, sys, warnings, logging, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter('ignore'); logging.disable(logging.WARNING)
os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
from demo_platform import config as cfg
os.environ.update(cfg.mi_env(period_role='current'))
os.environ['MI_AGENT_LLM_PARSER']='off'; os.environ['MI_AGENT_LLM_ENABLED']='0'
from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
from trakt_core.context import ExecutionContext
ctx = ExecutionContext.for_internal("alderbridge")

QS = [
 ("arity 1, avg      ", "What is the average borrower age by region?"),
 ("arity 2, avg      ", "What is the average borrower age by region and LTV band?"),
 ("arity 1, sum      ", "Show me balance by LTV band"),
 ("arity 2, sum      ", "Show me balance by LTV band and ticket size"),
 ("contribution 1 dim", "What is each region's contribution to the portfolio weighted average LTV?"),
 ("contribution 2 dim", "What is each region and LTV band's contribution to the portfolio weighted average LTV?"),
]
for label, q in QS:
    r = (execute_governed_mi_query(MiQueryRequest(question=q), ctx).result or {})
    arts = r.get("artifacts") or []
    cols, nrows = [], 0
    for a in arts:
        rows = a.get("rows") or []
        if rows:
            cols = sorted(rows[0].keys()); nrows = len(rows); break
    md = r.get("metadata") or {}
    warns = r.get("warnings") or md.get("warnings") or []
    thin = [w for w in warns if "thin sample" in str(w)]
    rejected = md.get("rejected_dimensions") or []
    gfk = md.get("group_field_keys") or []
    print(f"{label} | ok={str(r.get('ok')):5s} rows={nrows:4d}")
    print(f"    group_field_keys : {gfk}")
    print(f"    artifact columns : {cols}")
    print(f"    loan_count col   : {'loan_count' in cols}")
    print(f"    thin-sample warn : {bool(thin)}   {thin[:1]}")
    print(f"    rejected dims    : {rejected}")
