#!/usr/bin/env bash
# Full pipeline run for synthetic demo.
# Inserts a pandas-3.0 compatibility pre-processing step between Gate 1 and Gate 2
# without modifying any pipeline scripts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

INPUT="$ROOT/synthetic_demo/input/SYNTHETIC_ERE_Portfolio_012026.csv"
OUT_DIR="$ROOT/synthetic_demo/output"
VAL_DIR="$ROOT/synthetic_demo/output/validation"
CFG="$ROOT/synthetic_demo/config/config_client_SYNTHETIC_ERM.yaml"
REGISTRY="$ROOT/config/system/fields_registry.yaml"
ALIASES="$ROOT/synthetic_demo/aliases"
ENUM_MAP="$ROOT/config/system/enum_mapping.yaml"
ORDER_YAML="$ROOT/config/system/esma_code_order.yaml"
WORKBOOK="$ROOT/DRAFT1auth.099.001.04_non-ABCP Underlying Exposure Report_Version_1.3.1.xlsx"
XSD="$ROOT/config/system/DRAFT1auth.099.001.04_1.3.0.xsd"
DELIVERY_RULES="$ROOT/config/regime/annex2_delivery_rules.yaml"

echo "[Gate 1] Semantic alignment..."
python engine/gate_1_alignment/semantic_alignment.py \
  --input "$INPUT" \
  --portfolio-type equity_release \
  --registry "$REGISTRY" \
  --aliases-dir "$ALIASES" \
  --output-dir "$OUT_DIR" \
  --output-schema active

echo "[Pre-process] Applying pandas-3.0 compatibility fix to canonical_full.csv..."
python synthetic_demo/preprocess_canonical_full.py

echo "[Gate 2] Canonical transform..."
python engine/gate_2_transform/canonical_transform.py \
  "$OUT_DIR/SYNTHETIC_ERE_Portfolio_012026_canonical_full.csv" \
  --registry "$REGISTRY" \
  --portfolio-type equity_release \
  --config "$CFG" \
  --output-dir "$OUT_DIR"

echo "[Gate 3a] Schema validation..."
python engine/gate_3_validation/validate_canonical.py \
  "$OUT_DIR/SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv" \
  --registry "$REGISTRY" \
  --portfolio-type equity_release \
  --scope canonical \
  --out-dir "$VAL_DIR"

echo "[Gate 3b] Business rules..."
python engine/gate_3_validation/validate_business_rules.py \
  --input "$OUT_DIR/SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv" \
  --config "$CFG" \
  --regime ESMA_Annex2 \
  --report "$VAL_DIR/SYNTHETIC_ERE_Portfolio_012026_business_rules_violations.csv"

echo "[Gate 3c] Aggregate validation..."
python engine/gate_3_validation/aggregate_validation_results.py \
  --canonical-violations "$VAL_DIR/SYNTHETIC_ERE_Portfolio_012026_canonical_typed_canonical_violations.csv" \
  --business-violations "$VAL_DIR/SYNTHETIC_ERE_Portfolio_012026_business_rules_violations.csv" \
  --input-csv "$OUT_DIR/SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv" \
  --output "$VAL_DIR/SYNTHETIC_ERE_Portfolio_012026_field_summary.csv" \
  --dashboard-json "$VAL_DIR/SYNTHETIC_ERE_Portfolio_012026_dashboard.json" \
  --regime ESMA_Annex2 \
  --issue-policy "$ROOT/config/asset/issue_policy.yaml"

echo "[Gate 4] ESMA Annex 2 projection..."
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
echo "[Gate 4b] Delivery preflight..."
python engine/gate_4b_delivery/annex2_delivery_normalizer.py \
  --input "$OUT_DIR/SYNTHETIC_ERE_Portfolio_012026_ESMA_Annex2_projected.csv" \
  --rules "$DELIVERY_RULES" \
  --output-dir "$OUT_DIR"
DELIVERY_RC=$?
set -e
if [[ $DELIVERY_RC -ne 0 ]]; then
  echo "[WARN] Gate 4b delivery preflight failed (exit $DELIVERY_RC). XML build will be skipped."
fi

set +e
XML_RC=99
if [[ $DELIVERY_RC -eq 0 ]]; then
  echo "[Gate 5] XML builder..."
  python engine/gate_5_delivery/xml_builder_annex2.py \
    --input "$OUT_DIR/SYNTHETIC_ERE_Portfolio_012026_ESMA_Annex2_delivery_ready.csv" \
    --output "$OUT_DIR/SYNTHETIC_012026_annex2.xml" \
    --mapping-workbook "$WORKBOOK" \
    --sheet DRAFT1auth.099.001.04 \
    --code-order-yaml "$ORDER_YAML" \
    --xsd "$XSD"
  XML_RC=$?
else
  XML_RC=98
fi
set -e

if [[ $XML_RC -eq 98 ]]; then
  echo "[WARN] Gate 5 XML build skipped because Gate 4b preflight failed."
elif [[ $XML_RC -ne 0 ]]; then
  echo "[WARN] Gate 5 XML build failed (exit $XML_RC)."
fi

echo ""
echo "[HTML] Refreshing demo_overview_v2.html..."
python synthetic_demo/build_demo_html.py

echo ""
echo "Pipeline complete."
echo "  HTML: synthetic_demo/demo_overview_v2.html"
echo "  XML:  synthetic_demo/output/SYNTHETIC_012026_annex2.xml"
