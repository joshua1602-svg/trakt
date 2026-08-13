#!/usr/bin/env bash
# Multi-book, multi-period pipeline run for the synthetic demonstration.
#
# Runs the REAL production gates once per book per reporting period, then
# assembles each period's books into a combined platform canonical — the same
# shape a managed-service run produces for a client with several books, and the
# shape `mi_agent_api.datasets` already expects ("a combined tape may carry
# several cut-off dates").
#
# Provenance is stamped at Gate 2 from the CLI flags, so book identity is a
# first-class attribute of the governed model rather than a filter applied
# afterwards. Gate 4/4b/5 and the exception reconciliation then run against the
# CURRENT period's combined canonical, which is what a submission would carry.
#
# Regenerate the source extracts first:
#     python synthetic_demo/build_multibook_input.py
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

IN_DIR="$ROOT/synthetic_demo/multibook_input"
OUT_DIR="$ROOT/synthetic_demo/output/multibook"
VAL_DIR="$OUT_DIR/validation"
CFG="$ROOT/synthetic_demo/config/config_client_SYNTHETIC_MULTIBOOK.yaml"
REGISTRY="$ROOT/config/system/fields_registry.yaml"
ALIASES="$ROOT/synthetic_demo/aliases"
ENUM_MAP="$ROOT/config/system/enum_mapping.yaml"
ORDER_YAML="$ROOT/config/system/esma_code_order.yaml"
DELIVERY_RULES="$ROOT/config/regime/annex2_delivery_rules.yaml"
UNIVERSE="$ROOT/config/regime/annex2_field_universe.yaml"

PERIODS=("2026-05-31" "2026-06-30")
CURRENT_PERIOD="2026-06-30"

# book_id:type:label:acquisition_date (blank for direct provenance)
BOOKS=(
  "alp_origination:direct:ALP Origination Book:"
  "alp_acquired:acquired:ALP Acquired Back Book:2024-09-30"
  "spv1_sponsored:direct:SPV1 Sponsored Securitisation:"
)

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR" "$VAL_DIR"

for PERIOD in "${PERIODS[@]}"; do
  for ENTRY in "${BOOKS[@]}"; do
    IFS=":" read -r BOOK PTYPE LABEL ACQUIRED <<< "$ENTRY"
    STEM="${BOOK}_${PERIOD}"
    PROV_ARGS=(--source-portfolio-id "$BOOK"
               --source-portfolio-type "$PTYPE"
               --source-portfolio-label "$LABEL")
    if [[ -n "$ACQUIRED" ]]; then
      PROV_ARGS+=(--acquisition-date "$ACQUIRED")
    fi

    echo ""
    echo "=== ${LABEL} — ${PERIOD} ==========================================="

    echo "[Gate 1] Semantic alignment..."
    python engine/gate_1_alignment/semantic_alignment.py \
      --input "$IN_DIR/${STEM}.csv" \
      --portfolio-type equity_release \
      --registry "$REGISTRY" \
      --aliases-dir "$ALIASES" \
      --output-dir "$OUT_DIR" \
      --output-schema active

    echo "[Pre-process] pandas-3.0 compatibility..."
    python synthetic_demo/preprocess_canonical_full.py \
      --input "$OUT_DIR/${STEM}_canonical_full.csv"

    echo "[Gate 2] Canonical transform (stamping book provenance)..."
    python engine/gate_2_transform/canonical_transform.py \
      "$OUT_DIR/${STEM}_canonical_full.csv" \
      --registry "$REGISTRY" \
      --portfolio-type equity_release \
      --config "$CFG" \
      --output-dir "$OUT_DIR" \
      "${PROV_ARGS[@]}"
  done
done

echo ""
echo "=== Assembling combined platform canonicals ========================="
python synthetic_demo/assemble_multibook.py \
  --output-dir "$OUT_DIR" \
  --periods "${PERIODS[@]}"

for PERIOD in "${PERIODS[@]}"; do
  STEM="platform_${PERIOD}"
  echo ""
  echo "=== Validating combined canonical — ${PERIOD} ======================"

  echo "[Gate 3a] Schema validation..."
  python engine/gate_3_validation/validate_canonical.py \
    "$OUT_DIR/${STEM}_canonical_typed.csv" \
    --registry "$REGISTRY" \
    --portfolio-type equity_release \
    --scope canonical \
    --out-dir "$VAL_DIR"

  echo "[Gate 3b] Business rules..."
  python engine/gate_3_validation/validate_business_rules.py \
    --input "$OUT_DIR/${STEM}_canonical_typed.csv" \
    --config "$CFG" \
    --regime ESMA_Annex2 \
    --report "$VAL_DIR/${STEM}_business_rules_violations.csv"

  echo "[Gate 3c] Aggregate validation..."
  python engine/gate_3_validation/aggregate_validation_results.py \
    --canonical-violations "$VAL_DIR/${STEM}_canonical_typed_canonical_violations.csv" \
    --business-violations "$VAL_DIR/${STEM}_business_rules_violations.csv" \
    --input-csv "$OUT_DIR/${STEM}_canonical_typed.csv" \
    --output "$VAL_DIR/${STEM}_field_summary.csv" \
    --dashboard-json "$VAL_DIR/${STEM}_dashboard.json" \
    --regime ESMA_Annex2 \
    --issue-policy "$ROOT/config/asset/issue_policy.yaml"
done

STEM="platform_${CURRENT_PERIOD}"
echo ""
echo "=== ESMA Annex 2 — current period (${CURRENT_PERIOD}) ==============="

echo "[Gate 4] Regime projection..."
python engine/gate_4_projection/regime_projector.py \
  "$OUT_DIR/${STEM}_canonical_typed.csv" \
  --regime ESMA_Annex2 \
  --registry "$REGISTRY" \
  --enum-mapping "$ENUM_MAP" \
  --config "$CFG" \
  --template-order "$ORDER_YAML" \
  --portfolio-type equity_release \
  --allow-unreviewed \
  --output-dir "$OUT_DIR"

# The preflight is a hard gate and this dataset carries deliberate exceptions,
# so a non-zero exit here is the expected, demonstrated outcome — not a build
# failure. The reconciliation below is what explains it.
set +e
echo "[Gate 4b] Delivery preflight..."
python engine/gate_4b_delivery/annex2_delivery_normalizer.py \
  --input "$OUT_DIR/${STEM}_ESMA_Annex2_projected.csv" \
  --rules "$DELIVERY_RULES" \
  --output-dir "$OUT_DIR"
DELIVERY_RC=$?
set -e
if [[ $DELIVERY_RC -ne 0 ]]; then
  echo "[EXPECTED] Preflight reports blocking errors — the seeded exceptions."
fi

# Outside the frozen gate surface: reads the artefacts above, writes its own.
echo "[Reconciliation] Exceptions by effect on the submission..."
python trakt_core/regime_exception_reconciliation.py \
  --field-summary "$VAL_DIR/${STEM}_field_summary.csv" \
  --violations "$VAL_DIR/${STEM}_canonical_typed_canonical_violations.csv" \
  --canonical "$OUT_DIR/${STEM}_canonical_typed.csv" \
  --rules "$DELIVERY_RULES" \
  --universe "$UNIVERSE" \
  --enum-mapping "$ENUM_MAP" \
  --business-rules "$VAL_DIR/${STEM}_business_rules_violations.csv" \
  --delivery-issues "$OUT_DIR/${STEM}_ESMA_Annex2_delivery_issues.csv" \
  --projected "$OUT_DIR/${STEM}_ESMA_Annex2_projected.csv" \
  --delivery-report "$OUT_DIR/${STEM}_ESMA_Annex2_delivery_report.json" \
  --output "$OUT_DIR/${STEM}_regime_exception_reconciliation.json"

echo ""
echo "Multi-book pipeline complete. Output: $OUT_DIR"
