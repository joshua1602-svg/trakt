"""Capture answer text AND structured findings for the whole bank, for A5."""
import json, logging, os, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
REPO = Path(sys.argv[1]).resolve(); S = Path(__file__).resolve().parent.parent
sys.path[:] = [p for p in sys.path if p not in ("", str(Path.cwd()))]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(S)); os.chdir(REPO)
from demo_platform import config as cfg
env = cfg.mi_env(period_role="current"); os.environ.update(env)
os.environ["MI_AGENT_PIPELINE_ROOT"] = str(S / "pipeline_root")
os.environ["MI_AGENT_DATA_CACHE_TTL"]="0"
os.environ["MI_AGENT_LLM_PARSER"]="off"; os.environ["MI_AGENT_LLM_ENABLED"]="0"
os.environ.pop("ANTHROPIC_API_KEY", None)
logging.disable(logging.WARNING)
from mi_agent_api import data_source; data_source.reset_cache()
from fastapi.testclient import TestClient
from mi_agent_api.app import app
from nl_bank import bank
c = TestClient(app)
PID = f"{env['MI_AGENT_CLIENT_ID']}/{env['MI_AGENT_RUN_ID']}"
out = []
for row in bank("alderbridge"):
    r = c.post("/mi/query", json={"question": row["question"], "portfolioId": PID,
                                  "datasetContext": "funded"}).json()
    m = r.get("metadata") or {}
    out.append({"q": row["question"], "ok": r.get("ok"),
                "answer": (r.get("answer") or "").strip(),
                "findings": m.get("analyticalFindings") or []})
Path(sys.argv[2]).write_text(json.dumps(out, indent=1, default=str))
print(f"wrote {sys.argv[2]} ({len(out)} questions)")
