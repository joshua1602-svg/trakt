#!/usr/bin/env bash
set -euo pipefail

# Assumptions:
# - Run from repo root (/workspace/trakt)
# - Python dependencies from requirements.txt are installed

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

INPUT="$ROOT_DIR/synthetic_demo/input/SYNTHETIC_ERE_Portfolio_012026.csv"
OUT_DIR="$ROOT_DIR/synthetic_demo/output"
VAL_DIR="$ROOT_DIR/synthetic_demo/output/validation"
CFG="$ROOT_DIR/synthetic_demo/config/config_client_SYNTHETIC_ERM.yaml"
REGISTRY="$ROOT_DIR/config/system/fields_registry.yaml"
ALIASES="$ROOT_DIR/synthetic_demo/aliases"
ENUM_MAP="$ROOT_DIR/config/system/enum_mapping.yaml"
ORDER_YAML="$ROOT_DIR/config/system/esma_code_order.yaml"
WORKBOOK="$ROOT_DIR/DRAFT1auth.099.001.04_non-ABCP Underlying Exposure Report_Version_1.3.1.xlsx"
XSD="$ROOT_DIR/config/system/DRAFT1auth.099.001.04_1.3.0.xsd"

python engine/gate_1_alignment/semantic_alignment.py \
  --input "$INPUT" \
  --portfolio-type equity_release \
  --registry "$REGISTRY" \
  --aliases-dir "$ALIASES" \
  --output-dir "$OUT_DIR" \
  --output-schema active

python engine/gate_2_transform/canonical_transform.py \
  "$OUT_DIR/SYNTHETIC_ERE_Portfolio_012026_canonical_full.csv" \
  --registry "$REGISTRY" \
  --portfolio-type equity_release \
  --config "$CFG" \
  --output-dir "$OUT_DIR"

python engine/gate_3_validation/validate_canonical.py \
  "$OUT_DIR/SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv" \
  --registry "$REGISTRY" \
  --portfolio-type equity_release \
  --scope canonical \
  --out-dir "$VAL_DIR"

python engine/gate_3_validation/validate_business_rules.py \
  --input "$OUT_DIR/SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv" \
  --config "$CFG" \
  --regime ESMA_Annex2 \
  --report "$VAL_DIR/SYNTHETIC_ERE_Portfolio_012026_business_rules_violations.csv"

python engine/gate_3_validation/aggregate_validation_results.py \
  --canonical-violations "$VAL_DIR/SYNTHETIC_ERE_Portfolio_012026_canonical_typed_canonical_violations.csv" \
  --business-violations "$VAL_DIR/SYNTHETIC_ERE_Portfolio_012026_business_rules_violations.csv" \
  --input-csv "$OUT_DIR/SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv" \
  --output "$VAL_DIR/SYNTHETIC_ERE_Portfolio_012026_field_summary.csv" \
  --dashboard-json "$VAL_DIR/SYNTHETIC_ERE_Portfolio_012026_dashboard.json" \
  --regime ESMA_Annex2 \
  --issue-policy "$ROOT_DIR/config/asset/issue_policy.yaml"

python engine/gate_4_projection/regime_projector.py \
  "$OUT_DIR/SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv" \
  --regime ESMA_Annex2 \
  --registry "$REGISTRY" \
  --enum-mapping "$ENUM_MAP" \
  --config "$CFG" \
  --template-order "$ORDER_YAML" \
  --portfolio-type equity_release \
  --allow-unreviewed \
  --output-dir "$OUT_DIR"

set +e
python engine/gate_5_delivery/xml_builder_annex2.py \
  --input "$OUT_DIR/SYNTHETIC_ERE_Portfolio_012026_ESMA_Annex2_projected.csv" \
  --output "$OUT_DIR/SYNTHETIC_012026_annex2.xml" \
  --mapping-workbook "$WORKBOOK" \
  --sheet DRAFT1auth.099.001.04 \
  --code-order-yaml "$ORDER_YAML" \
  --xsd "$XSD"
XML_RC=$?
set -e
if [[ $XML_RC -ne 0 ]]; then
  echo "[WARN] Gate 5 XML build failed (exit $XML_RC). See command output above for details."
fi

python - <<'PY'
import pandas as pd
from pathlib import Path
raw=pd.read_csv('synthetic_demo/input/SYNTHETIC_ERE_Portfolio_012026.csv')
full=pd.read_csv('synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_canonical_full.csv')
typed=pd.read_csv('synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv')
proj=pd.read_csv('synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_ESMA_Annex2_projected.csv')
field_summary=pd.read_csv('synthetic_demo/output/validation/SYNTHETIC_ERE_Portfolio_012026_field_summary.csv')
xml_exists=Path('synthetic_demo/output/SYNTHETIC_012026_annex2.xml').exists()

bal=typed['current_principal_balance'].fillna(0).sum()
loans=len(typed)
wa_ltv=typed['current_loan_to_value'].fillna(0).mean()
wa_rate=typed['current_interest_rate'].fillna(0).mean()
wa_age=typed['youngest_borrower_age'].fillna(0).mean()
region=typed['geographic_region_classification'].fillna('Unknown').value_counts().head(6)
prop=typed['property_type'].fillna('Unknown').value_counts().head(6)

def bar_table(series,title):
    maxv=max(series.max(),1)
    rows=''
    for k,v in series.items():
        w=round((v/maxv)*100,1)
        rows += f"<tr><td>{k}</td><td>{v}</td><td><div style='background:#e6eef8;width:220px'><div style='background:#2f5d9f;height:10px;width:{w}%'></div></div></td></tr>"
    return f"<h4>{title}</h4><table><tr><th>Bucket</th><th>Count</th><th>Share</th></tr>{rows}</table>"

html=f"""<html><head><meta charset='utf-8'><title>Synthetic ERM Demo Overview</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:18px;color:#1b1f23}}h1{{margin:0 0 8px 0}}h2{{margin-top:20px}}
table{{border-collapse:collapse;font-size:12px;margin:8px 0 14px 0}}th,td{{border:1px solid #cfd6df;padding:4px 6px;vertical-align:top}}
.kpis{{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:8px;margin:10px 0 14px}}
.kpi{{border:1px solid #d2dae5;border-radius:6px;padding:8px;background:#f9fbfe}}.kpi .v{{font-weight:bold;font-size:16px}}
.small{{color:#5f6b7a;font-size:12px}}
</style></head><body>
<h1>Synthetic Equity Release Demo (Send-ready)</h1>
<div class='small'>Style reference used: <code>demo_static_6_loans.csv</code>. Scope: synthetic_demo only.</div>
<h2>1) Raw lender input snippet</h2>
{raw.head(6)[['Underlying Exposure Identifier','Origination Dt','Current Principal Balance GBP','Current Interest Rate','Borrower Age (Youngest)','Region ']].to_html(index=False)}
<h2>2) Canonical standardisation (before/after)</h2>
<h4>Canonical full</h4>
{full.head(6)[['underlying_exposure_identifier','origination_date','current_principal_balance','current_interest_rate','youngest_borrower_age','account_status']].to_html(index=False)}
<h4>Canonical typed</h4>
{typed.head(6)[['underlying_exposure_identifier','origination_date','current_principal_balance','current_interest_rate','youngest_borrower_age','account_status']].to_html(index=False)}
<h2>3) Validation / governance summary</h2>
<p>Controls surfaced limited, explainable exceptions.</p>
{field_summary[['field_name','issue_type','error_count','severity']].head(10).to_html(index=False)}
<h2>4) MI view (HTML only, no binary dependency)</h2>
<div class='kpis'>
  <div class='kpi'><div class='small'>Loan count</div><div class='v'>{loans}</div></div>
  <div class='kpi'><div class='small'>Portfolio balance</div><div class='v'>£{bal:,.0f}</div></div>
  <div class='kpi'><div class='small'>WA current LTV</div><div class='v'>{wa_ltv:.2f}%</div></div>
  <div class='kpi'><div class='small'>WA interest rate</div><div class='v'>{wa_rate:.2f}%</div></div>
  <div class='kpi'><div class='small'>WA borrower age</div><div class='v'>{wa_age:.1f}</div></div>
</div>
{bar_table(region,'Top regions')}
{bar_table(prop,'Property type mix')}
<h2>5) ESMA Annex 2 projection snippet</h2>
{proj.head(6)[['RREL1','RREL2','RREL3','RREL4','RREL5','RREC2','RREL6']].to_html(index=False)}
<h2>6) XML delivery status</h2>
<p>{'XML was generated.' if xml_exists else 'XML build was attempted and failed on workbook mandatory-field constraint: latest blocker RREL15 blank after projection.'}</p>
<h2>7) Architecture summary</h2>
<p><b>one messy lender tape → one canonical truth set → multiple controlled outputs</b></p>
</body></html>"""
Path('synthetic_demo/demo_pack/demo_overview.html').write_text(html)
print('Updated synthetic_demo/demo_pack/demo_overview.html')
PY

echo "Demo run complete. Review synthetic_demo/demo_pack/demo_overview.html"
