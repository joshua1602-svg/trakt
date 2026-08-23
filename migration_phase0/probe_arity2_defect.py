"""How many leaf groups does the arity-2 case hide? Read-only."""
import os, sys, warnings, logging
warnings.simplefilter('ignore'); logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TRAKT_RUNTIME_MODE","development")
from demo_platform import config as cfg
os.environ.update(cfg.mi_env(period_role='current'))
from mi_agent_api.data_source import semantics_path
from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.mi_query_executor import LOW_GROUP_COUNT
from mi_agent_api import evolution as evo
import pandas as pd
frames = evo.funded_frames(os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"],"alderbridge",None)
df = frames[-1]["df"]
print(f"LOW_GROUP_COUNT = {LOW_GROUP_COUNT}   (policy value — NOT changed by this study)")
for dims in (["collateral_geography"], ["collateral_geography","ltv_bucket"],
             ["ltv_bucket","ticket_bucket"]):
    s = df.assign(**{d: df[d].astype(str) for d in dims})
    sizes = s.groupby(dims, sort=False).size()
    thin = int((sizes < LOW_GROUP_COUNT).sum())
    disclosed = (len(dims) == 1)
    print(f"  arity {len(dims)}  {dims}")
    print(f"      {len(sizes):5d} leaf groups, {thin:5d} thin "
          f"({thin/len(sizes)*100:5.1f}%)   disclosed today: {disclosed}")
