#!/usr/bin/env python3
"""Generate funded-tape /mi/query envelope fixtures for the React component test.

Runs the real onboarding+promotion pipeline, points the MI Agent API data source
at the promoted 18_central_lender_tape.csv, captures the /mi/query 'portfolio
summary' envelope per run. Generic by client_id/run_id. Usage:

    python frontend/mi-agent-ui/scripts/generate_funded_fixtures.py
"""
import os, json, tempfile, warnings
from pathlib import Path
import pandas as pd
warnings.simplefilter("ignore")
REPO = Path(__file__).resolve().parents[3]
import sys; sys.path.insert(0, str(REPO))
REG = str(REPO/"config/system/fields_registry.yaml")
OUT = REPO/"frontend/mi-agent-ui/src/test/fixtures"
OUT.mkdir(parents=True, exist_ok=True)
OCT_N,OCT=33,127515.15; NOV_N,NOV=73,121958.90

def make_pack(root):
    inp=root/"input"; inp.mkdir(parents=True)
    ids=[760000+i for i in range(NOV_N)]; lng=[s*100+1 for s in ids]; o=[];n=[]
    for i,lid in enumerate(ids):
        n.append({"Loan Policy Number":lid,"Month Run":"November","Loan Interest Rate":3.1+(i%5)*0.05,"Current Outstanding Balance":NOV,"Policy Completion Date":f"20{16+i%7}-0{1+i%9}-15"})
        if i<OCT_N:o.append({"Loan Policy Number":lid,"Month Run":"October","Loan Interest Rate":3.1+(i%5)*0.05,"Current Outstanding Balance":OCT,"Policy Completion Date":f"20{16+i%7}-0{1+i%9}-15"})
    pd.DataFrame(o+n).to_csv(inp/"LoanExtract One.csv",index=False)
    pd.DataFrame({"Account Number":lng,"Latest Property Value":[250000.0+i for i in range(NOV_N)]}).to_csv(inp/"Collateral Extract.csv",index=False)
    pd.DataFrame({"application_id":[f"APP{i}" for i in range(20)],"Account Number":[990000+i for i in range(20)],"product rate":[4.0]*20}).to_csv(inp/"M2L KFI and Pipeline 2025_12_01.csv",index=False)
    return inp

root=Path(tempfile.mkdtemp()); inp=make_pack(root)
from engine.onboarding_agent import workflow as wf, storage_paths, central_tape_builder
from mi_agent_api import data_source
for run_id in ("mi_2025_10","mi_2025_11"):
    proj=root/f"proj_{run_id}"
    wf.run_operator_workflow(input_dir=str(inp),client_name="Client 001",client_id="client_001",run_id=run_id,mode="mi_only",project_dir=str(proj),product_profile="equity_release_lifetime_mortgage")
    rp=storage_paths.resolve_run_paths(project_dir=str(proj),input_dir=str(inp),output_root=None,client_id="client_001",run_id=run_id,storage_backend="local",input_uri="",output_uri="")
    res=central_tape_builder.build_central_tapes(str(proj),rp,REG,mode="mi_only")
    os.environ["MI_AGENT_CENTRAL_TAPE"]=res["central_lender_tape_path"]
    os.environ["MI_AGENT_CLIENT_ID"]="client_001"; os.environ["MI_AGENT_RUN_ID"]=run_id
    data_source.get_dataframe.cache_clear()
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app
    body=TestClient(app).post("/mi/query",json={"question":"portfolio summary","portfolioId":f"client_001/{run_id}","asOfDate":"2025-10-31"}).json()
    (OUT/f"funded_summary_{run_id}.json").write_text(json.dumps(body,indent=2,default=str))
    strat=TestClient(app).post("/mi/query",json={"question":"current outstanding balance by ltv bucket","portfolioId":f"client_001/{run_id}","asOfDate":"2025-10-31"}).json()
    (OUT/f"funded_strat_ltv_{run_id}.json").write_text(json.dumps(strat,indent=2,default=str))
    kpi=next(a for a in body["artifacts"] if a["type"]=="kpi")
    print(run_id, "ok=",body["ok"], "kpis=", [(k["label"],k["value"]) for k in kpi["kpis"]])
import os, json, tempfile, warnings
from pathlib import Path
import pandas as pd
warnings.simplefilter("ignore")
REPO = Path(__file__).resolve().parents[3]
import sys; sys.path.insert(0, str(REPO))
REG = str(REPO/"config/system/fields_registry.yaml")
OUT = REPO/"frontend/mi-agent-ui/src/test/fixtures"
OUT.mkdir(parents=True, exist_ok=True)
OCT_N,OCT=33,127515.15; NOV_N,NOV=73,121958.90

def make_pack(root):
    inp=root/"input"; inp.mkdir(parents=True)
    ids=[760000+i for i in range(NOV_N)]; lng=[s*100+1 for s in ids]; o=[];n=[]
    for i,lid in enumerate(ids):
        n.append({"Loan Policy Number":lid,"Month Run":"November","Loan Interest Rate":3.1+(i%5)*0.05,"Current Outstanding Balance":NOV,"Policy Completion Date":f"20{16+i%7}-0{1+i%9}-15"})
        if i<OCT_N:o.append({"Loan Policy Number":lid,"Month Run":"October","Loan Interest Rate":3.1+(i%5)*0.05,"Current Outstanding Balance":OCT,"Policy Completion Date":f"20{16+i%7}-0{1+i%9}-15"})
    pd.DataFrame(o+n).to_csv(inp/"LoanExtract One.csv",index=False)
    pd.DataFrame({"Account Number":lng,"Latest Property Value":[250000.0+i for i in range(NOV_N)]}).to_csv(inp/"Collateral Extract.csv",index=False)
    pd.DataFrame({"application_id":[f"APP{i}" for i in range(20)],"Account Number":[990000+i for i in range(20)],"product rate":[4.0]*20}).to_csv(inp/"M2L KFI and Pipeline 2025_12_01.csv",index=False)
    return inp

root=Path(tempfile.mkdtemp()); inp=make_pack(root)
from engine.onboarding_agent import workflow as wf, storage_paths, central_tape_builder
from mi_agent_api import data_source
for run_id in ("mi_2025_10","mi_2025_11"):
    proj=root/f"proj_{run_id}"
    wf.run_operator_workflow(input_dir=str(inp),client_name="Client 001",client_id="client_001",run_id=run_id,mode="mi_only",project_dir=str(proj),product_profile="equity_release_lifetime_mortgage")
    rp=storage_paths.resolve_run_paths(project_dir=str(proj),input_dir=str(inp),output_root=None,client_id="client_001",run_id=run_id,storage_backend="local",input_uri="",output_uri="")
    res=central_tape_builder.build_central_tapes(str(proj),rp,REG,mode="mi_only")
    os.environ["MI_AGENT_CENTRAL_TAPE"]=res["central_lender_tape_path"]
    os.environ["MI_AGENT_CLIENT_ID"]="client_001"; os.environ["MI_AGENT_RUN_ID"]=run_id
    data_source.get_dataframe.cache_clear()
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app
    body=TestClient(app).post("/mi/query",json={"question":"portfolio summary","portfolioId":f"client_001/{run_id}","asOfDate":"2025-10-31"}).json()
    (OUT/f"funded_summary_{run_id}.json").write_text(json.dumps(body,indent=2,default=str))
    strat=TestClient(app).post("/mi/query",json={"question":"current outstanding balance by ltv bucket","portfolioId":f"client_001/{run_id}","asOfDate":"2025-10-31"}).json()
    (OUT/f"funded_strat_ltv_{run_id}.json").write_text(json.dumps(strat,indent=2,default=str))
    kpi=next(a for a in body["artifacts"] if a["type"]=="kpi")
    print(run_id, "ok=",body["ok"], "kpis=", [(k["label"],k["value"]) for k in kpi["kpis"]])
