"""Reproduce the PRE-Tranche-B prep behaviour in-process, for the delta table.

The only change Tranche B made to the exclusion tier was WHICH stages it
excludes: the old code excluded WITHDRAWN by a hard-coded test and wrote the
literal source "excluded_withdrawn". Restricting _excluded_stages to WITHDRAWN
reproduces that tier exactly, including the source string. Nothing else is
patched, and no production file is edited.
"""
import json, logging, os, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
REPO = Path("/home/user/trakt"); S = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(S)); os.chdir(REPO)
BOOK = sys.argv[1]
if BOOK == "alderbridge":
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    CID, RID = os.environ["MI_AGENT_CLIENT_ID"], os.environ["MI_AGENT_RUN_ID"]
else:
    from tests.analytical import second_book as bk
    os.environ.update(bk.mi_env(S / "kestrelmoor_store"))
    CID, RID = bk.CLIENT_ID, bk.REPORTING_DATES[-1]
os.environ["MI_AGENT_PIPELINE_ROOT"] = str(S / "pipeline_root")
os.environ["MI_AGENT_DATA_CACHE_TTL"] = "0"
logging.disable(logging.WARNING)
from mi_agent_api import pipeline_prep as PP
PP._excluded_stages = lambda: frozenset({"WITHDRAWN"})
from mi_agent_api import app as A
from mi_agent_api import pipeline_contract as P
src = A._resolve_pipeline_source(CID, RID)
hist = A._pipeline_history(src.get("client_id", CID))
df, rep = P.load_prepared_pipeline(src, historical_model=hist)
summ = rep.get("completion_probability_summary") or {}
print(BOOK, "(pre-Tranche-B reproduction)")
print("  basis            :", rep.get("completion_probability_basis"))
print("  by source        :", json.dumps(summ.get("by_source")))
for k in ("gross_amount","excluded_amount","excluded_count","active_gross_amount",
          "weighted_expected_amount","blended_weighted_conversion"):
    if k in summ: print(f"  {k:17s}:", summ[k])
