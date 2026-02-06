#!/usr/bin/env python3
"""
regime_projector_frozen_v1_0.py (Frozen Spine-aligned)

Purpose (locked contract):
- Read canonical_typed.csv (truth set from canonical_transform)
- Project to regime-specific schema (e.g., ESMA_Annex2)
- Apply enum mappings (canonical → ESMA codes)
- Apply ND defaults (from client config)
- Map field names to ESMA codes (e.g., loan_identifier → LI)
- Enforce priority ordering (Mandatory fields first)
- Output regime-compliant CSV ready for XML generation

Key Design Principles:
1. Canonical truth set is NEVER modified (read-only)
2. Regime projections are VIEWS on canonical data
3. ND codes are inserted ONLY in regime projections
4. Enum mappings are regime-specific
5. Field ordering follows ESMA spec (Mandatory → Optional)

Usage:
    python regime_projector_frozen_v1_0.py \\
      canonical_typed.csv \\
      --regime ESMA_Annex2 \\
      --registry fields_registry_v6_core_canonical_pricing_currency_code.yaml \\
      --enum-mapping enum_mapping.yaml \\
      --config config_ERM_UK.yaml \\
      --output-dir out \\
      --output-prefix ERE_122025
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


# ============================================================
# CONFIGURATION LOADING
# ============================================================

def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file with error handling."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_yaml_optional(path: Path) -> Dict[str, Any]:
    """Load optional YAML file, return empty dict if missing."""
    if not path.exists():
        logging.warning(f"Optional file not found: {path}")
        return {}
    return load_yaml(path)


def load_template_order(regime: str, template_file: Path) -> List[str]:
    """
    Load ESMA template field order from YAML file.
    
    Args:
        regime: Target regime (e.g., 'ESMA_Annex2')
        template_file: Path to esma_code_order.yaml
    
    Returns:
        List of ESMA codes in template order (e.g., ['LI', 'RREL1', 'RREL2', ...])
        Empty list if file not found or regime not defined
    """
    if not template_file.exists():
        logging.warning(f"Template order file not found: {template_file}")
        logging.warning("Will use fallback ordering (Mandatory/Optional alphabetic)")
        return []
    
    try:
        template_config = load_yaml(template_file)
    except Exception as e:
        logging.error(f"Failed to load template order file: {e}")
        return []
    
    if regime not in template_config:
        logging.warning(f"Regime '{regime}' not found in template order file")
        logging.warning("Will use fallback ordering (Mandatory/Optional alphabetic)")
        return []
    
    template_order = template_config[regime]
    
    if not isinstance(template_order, list):
        logging.error(f"Template order for '{regime}' must be a list, got {type(template_order)}")
        return []
    
    logging.info(f"Loaded template order: {len(template_order)} fields for {regime}")
    return template_order


# ============================================================
# FIELD REGISTRY PROCESSING
# ============================================================

def get_regime_fields(
    registry: Dict[str, Any],
    regime: str,
    portfolio_type: str
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Extract fields that have regime_mapping for the target regime.
    
    Returns:
        List of (canonical_field_name, regime_metadata) tuples
        where regime_metadata contains: code, priority, allowed_nd_codes, etc.
    """
    pt = (portfolio_type or "").strip().lower()
    regime_key = str(regime).strip()
    
    fields_list: List[Tuple[str, Dict[str, Any]]] = []
    
    for field_name, field_meta in (registry.get("fields") or {}).items():
        # Check portfolio_type applicability
        fpt = str((field_meta or {}).get("portfolio_type", "")).strip().lower()
        if fpt not in ("common", pt):
            continue
        
        # Check if field has regime_mapping for target regime
        regime_mapping = (field_meta or {}).get("regime_mapping") or {}
        if regime_key not in regime_mapping:
            continue
        
        regime_meta = regime_mapping[regime_key] or {}
        fields_list.append((field_name, regime_meta))
    
    return fields_list


def order_fields_by_template(
    fields_list: List[Tuple[str, Dict[str, Any]]],
    template_order: List[str]
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Order fields by ESMA template order (primary method).
    
    This ensures XML field ordering matches ESMA XSD schema requirements.
    
    Args:
        fields_list: List of (canonical_field_name, regime_metadata) tuples
        template_order: List of ESMA codes in template order from esma_code_order.yaml
    
    Returns:
        Ordered fields list matching ESMA template
        
    Example:
        template_order = ['LI', 'RREL1', 'RREL2', 'RREL3']
        Input fields with codes: ['RREL2', 'LI', 'RREL1']
        Output order: ['LI', 'RREL1', 'RREL2']
    """
    if not template_order:
        # No template available - fallback to priority ordering
        logging.warning("No template order available, using fallback ordering")
        return order_fields_by_priority(fields_list)
    
    # Create lookup: ESMA code -> template position
    order_map = {code: idx for idx, code in enumerate(template_order)}
    
    # Sort fields by template position
    # Fields not in template go to end (position 9999)
    ordered = sorted(
        fields_list,
        key=lambda x: order_map.get(x[1].get('code', ''), 9999)
    )
    
    # Log fields not in template (might be new fields or errors)
    not_in_template = [
        (name, meta.get('code'))
        for name, meta in ordered
        if meta.get('code') not in order_map
    ]
    
    if not_in_template:
        codes = [code for _, code in not_in_template[:10]]
        logging.warning(
            f"{len(not_in_template)} fields not in template order: {codes}"
            f"{' ...' if len(not_in_template) > 10 else ''}"
        )
        logging.warning(
            "These fields will appear at end of output. "
            "Update esma_code_order.yaml if these should be included."
        )
    
    logging.info(f"Fields ordered by template: {len(ordered)} total")
    return ordered


def order_fields_by_priority(
    fields_list: List[Tuple[str, Dict[str, Any]]]
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Order fields by priority (Mandatory first, then Optional), then alphabetically.
    
    NOTE: This is a FALLBACK method used only when esma_code_order.yaml is unavailable.
    For production ESMA compliance, always use order_fields_by_template() instead.
    
    Alphabetic ordering does NOT match ESMA XML schema requirements and may cause
    validation failures when generating XML.
    """
    mandatory = []
    optional = []
    
    for field_name, regime_meta in fields_list:
        priority = str(regime_meta.get("priority", "")).strip().lower()
        if priority == "mandatory":
            mandatory.append((field_name, regime_meta))
        else:
            optional.append((field_name, regime_meta))
    
    # Sort alphabetically within each group
    mandatory.sort(key=lambda x: x[0])
    optional.sort(key=lambda x: x[0])
    
    return mandatory + optional


# ============================================================
# ENUM MAPPING
# ============================================================

def apply_enum_mappings(
    df: pd.DataFrame,
    registry: Dict[str, Any],
    enum_mapping: Dict[str, Any],
    regime: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convert canonical enum values to regime-specific codes using enum_mapping.yaml.
    
    Example:
        canonical: amortisation_type = "ANNUITY"
        ESMA: amortisation_type = "FRXX"
    
    Returns:
        (transformed_df, transformation_report)
    """
    report: Dict[str, Any] = {"transformed_fields": {}, "unmapped_values": {}}
    
    regime_enums = enum_mapping.get(regime, {})
    if not regime_enums:
        logging.warning(f"No enum mappings found for regime '{regime}' in enum_mapping.yaml")
        return df, report
    
    for col in df.columns:
        # Check if field has allowed_values (enum) in registry
        field_meta = registry.get("fields", {}).get(col, {})
        allowed_values = field_meta.get("allowed_values")
        
        if not allowed_values or allowed_values == "null":
            continue
        
        # Check if enum_mapping has mappings for this field
        if col not in regime_enums:
            logging.debug(f"No enum mapping for field '{col}' (allowed_values: {allowed_values})")
            continue
        
        field_enum_map = regime_enums[col]
        
        # Apply mapping
        original = df[col].copy()
        transformed = df[col].astype(str).str.strip().str.upper()
        
        # Map values
        mapped = transformed.map(field_enum_map)
        
        # Track unmapped values
        unmapped_mask = mapped.isna() & transformed.notna() & (transformed != "") & (transformed != "NAN")
        if unmapped_mask.any():
            unmapped_values = original[unmapped_mask].unique().tolist()
            # Convert pandas NA to string for JSON serialization
            unmapped_values = [str(v) if pd.notna(v) else "NULL" for v in unmapped_values]
            report["unmapped_values"][col] = unmapped_values
            logging.warning(
                f"Field '{col}': {len(unmapped_values)} unmapped values: {unmapped_values[:5]}"
            )
        
        # Update dataframe
        df[col] = mapped.where(mapped.notna(), original)  # Keep original if no mapping
        
        # Report
        transformed_count = int((df[col] != original).sum())
        if transformed_count > 0:
            report["transformed_fields"][col] = {
                "rows_transformed": transformed_count,
                "mapping_applied": True,
            }
    
    return df, report


# ============================================================
# ND DEFAULT APPLICATION
# ============================================================

def apply_nd_defaults(
    df: pd.DataFrame,
    fields_list: List[Tuple[str, Dict[str, Any]]],
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply ND defaults from client config for fields that are blank.
    
    Example:
        config:
          defaults:
            nd_defaults:
              maturity_date: ND2
    
    Logic:
        - Only apply if field is blank/null
        - Only apply if field is in client config nd_defaults
        - ND codes are ONLY inserted in regime projections, never in canonical
    
    Returns:
        (transformed_df, nd_application_report)
    """
    report: Dict[str, Any] = {"nd_defaults_applied": {}}
    
    nd_defaults = (config.get("defaults") or {}).get("nd_defaults") or {}
    if not nd_defaults:
        logging.info("No nd_defaults specified in client config")
        return df, report
    
    for field_name, regime_meta in fields_list:
        if field_name not in df.columns:
            continue
        
        if field_name not in nd_defaults:
            continue
        
        nd_code = nd_defaults[field_name]
        
        # Apply ND default only where field is blank/null
        blank_mask = df[field_name].isna() | (df[field_name].astype(str).str.strip() == "")
        blank_count = int(blank_mask.sum())
        
        if blank_count == 0:
            continue
        
        df.loc[blank_mask, field_name] = nd_code
        
        report["nd_defaults_applied"][field_name] = {
            "nd_code": nd_code,
            "rows_filled": blank_count,
        }
        
        logging.info(f"Applied ND default '{nd_code}' to {blank_count} blank rows in '{field_name}'")
    
    return df, report


# ============================================================
# ESMA CODE MAPPING
# ============================================================

def rename_to_esma_codes(
    df: pd.DataFrame,
    fields_list: List[Tuple[str, Dict[str, Any]]]
) -> pd.DataFrame:
    """
    Rename canonical field names to ESMA codes.
    
    Example:
        loan_identifier → LI
        account_status → RREL69
        amortisation_type → CREL87
    """
    rename_map = {}
    
    for field_name, regime_meta in fields_list:
        esma_code = regime_meta.get("code")
        if esma_code:
            rename_map[field_name] = esma_code
    
    df = df.rename(columns=rename_map)
    
    logging.info(f"Renamed {len(rename_map)} fields to ESMA codes")
    return df

def apply_esma_uk_geography_override(
    df: pd.DataFrame,
    config: Dict[str, Any],
    regime: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    report: Dict[str, Any] = {"applied": False, "rows_overridden": 0, "fields": []}

    overrides_cfg = (config.get("regime_overrides") or {}).get(regime) or {}
    uk_geo_cfg = overrides_cfg.get("uk_geography") or {}

    if not bool(uk_geo_cfg.get("enabled", False)):
        return df, report

    # Canonical field that indicates country of exposure/collateral (optional).
    country_field = str(uk_geo_cfg.get("country_field", "")).strip()

    # ---------- Determine UK mask (exactly once) ----------
    uk_mask = None

    # Option A: Row-scoped canonical country field
    if country_field and country_field in df.columns:
        c = df[country_field].astype("string").str.strip().str.upper()
        uk_mask = c.isin({"GB", "UK"})
        report["country_detection"] = {"mode": "row", "country_field": country_field}

    # Option B: Deal/portfolio-scoped fallback
    if uk_mask is None:
        portfolio_country = str((config.get("portfolio") or {}).get("country", "")).strip().upper()
        if portfolio_country in {"GB", "UK"}:
            uk_mask = pd.Series(True, index=df.index)
        else:
            uk_mask = pd.Series(False, index=df.index)
        report["country_detection"] = {"mode": "portfolio", "portfolio_country": portfolio_country}

    if not uk_mask.any():
        report["skipped_reason"] = "No UK rows detected (by canonical country_field or portfolio.country)"
        return df, report
    # ------------------------------------------------------

    # Canonical geography fields (pre-rename) that feed ESMA geography codes.
    target_fields = uk_geo_cfg.get("target_fields") or []
    target_fields = [f for f in target_fields if f in df.columns]

    if not target_fields:
        report["skipped_reason"] = "No target_fields present in df"
        return df, report

    override_value = str(uk_geo_cfg.get("override_value", "GBZZZ")).strip().upper()

    # Apply override only for UK rows
    for f in target_fields:
        df.loc[uk_mask, f] = override_value

    report["applied"] = True
    report["rows_overridden"] = int(uk_mask.sum())
    report["fields"] = target_fields
    report["override_value"] = override_value
    return df, report

# ============================================================
# MAIN PROJECTION LOGIC
# ============================================================

def project_to_regime(
    canonical_df: pd.DataFrame,
    registry: Dict[str, Any],
    enum_mapping: Dict[str, Any],
    config: Dict[str, Any],
    regime: str,
    portfolio_type: str,
    template_order: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main projection function: canonical truth set → regime projection.
    
    Steps:
        1. Select fields that have regime_mapping
        2. Order by ESMA template (or fallback to priority if no template)
        3. Apply enum mappings (canonical → ESMA codes)
        4. Apply ND defaults (from client config)
        5. Rename fields to ESMA codes
        6. Return regime-compliant dataframe
    
    Args:
        canonical_df: Canonical truth set dataframe
        registry: Field registry (fields_registry.yaml)
        enum_mapping: Enum mapping config (enum_mapping.yaml)
        config: Client config (config_*.yaml)
        regime: Target regime (e.g., 'ESMA_Annex2')
        portfolio_type: Asset class (e.g., 'equity_release')
        template_order: ESMA field codes in template order (from esma_code_order.yaml)
    
    Returns:
        (regime_df, projection_report)
    """
    report: Dict[str, Any] = {
        "regime": regime,
        "portfolio_type": portfolio_type,
        "canonical_rows": len(canonical_df),
        "canonical_fields": len(canonical_df.columns),
    }
    
    # Step 1: Get regime fields
    fields_list = get_regime_fields(registry, regime, portfolio_type)
    if not fields_list:
        raise ValueError(f"No fields found for regime '{regime}' and portfolio_type '{portfolio_type}'")
    
    logging.info(f"Found {len(fields_list)} fields for regime '{regime}'")
    
    # Step 2: Order by ESMA template (or fallback to priority if no template)
    fields_list = order_fields_by_template(fields_list, template_order)
    
    mandatory_count = sum(1 for _, meta in fields_list if meta.get("priority", "").lower() == "mandatory")
    optional_count = len(fields_list) - mandatory_count
    
    report["regime_fields"] = len(fields_list)
    report["mandatory_fields"] = mandatory_count
    report["optional_fields"] = optional_count
    
    logging.info(f"Field breakdown: {mandatory_count} Mandatory, {optional_count} Optional")
    
    # Step 3: Project schema (select only regime fields)
    regime_field_names = [field_name for field_name, _ in fields_list]
    
    # Handle missing fields gracefully
    available_fields = [f for f in regime_field_names if f in canonical_df.columns]
    missing_fields = [f for f in regime_field_names if f not in canonical_df.columns]
    
    if missing_fields:
        logging.warning(f"{len(missing_fields)} regime fields missing in canonical dataset: {missing_fields[:10]}")
        report["missing_fields"] = missing_fields
    
    regime_df = canonical_df[available_fields].copy()
    
    # Add missing fields as empty columns efficiently (avoid fragmentation)
    if missing_fields:
        missing_df = pd.DataFrame({field: pd.NA for field in missing_fields}, index=regime_df.index)
        regime_df = pd.concat([regime_df, missing_df], axis=1)
    
    # Reorder to match fields_list
    regime_df = regime_df[[f for f, _ in fields_list if f in regime_df.columns]]
    
    # Step 4: Apply enum mappings
    regime_df, enum_report = apply_enum_mappings(regime_df, registry, enum_mapping, regime)
    report["enum_mapping"] = enum_report
    
    # Step 5: Apply ND defaults
    regime_df, nd_report = apply_nd_defaults(regime_df, fields_list, config)
    report["nd_defaults"] = nd_report
    
    # Step 5b: ESMA UK geography override (UK -> GBZZZ)
    regime_df, uk_geo_report = apply_esma_uk_geography_override(regime_df, config, regime)
    report["uk_geography_override"] = uk_geo_report
    
    # Step 6: Rename to ESMA codes
    regime_df = rename_to_esma_codes(regime_df, fields_list)
    
    report["output_fields"] = len(regime_df.columns)
    report["output_rows"] = len(regime_df)
    
    return regime_df, report


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Regime Projector (Frozen Spine v1.0)")
    ap.add_argument("canonical_csv", help="Input canonical_typed.csv from canonical_transform")
    ap.add_argument("--regime", required=True, help="Target regime (e.g., ESMA_Annex2, ESMA_Annex3)")
    ap.add_argument("--registry", required=True, help="Field registry YAML")
    ap.add_argument("--enum-mapping", required=True, help="Enum mapping YAML (canonical → ESMA codes)")
    ap.add_argument("--config", default=None, help="Client config YAML (for ND defaults)")
    ap.add_argument("--product-defaults", default=None, help="Layer 2 Product Defaults YAML")
    ap.add_argument("--template-order", default="esma_code_order.yaml", 
                   help="ESMA template field order YAML (default: esma_code_order.yaml)")
    ap.add_argument("--portfolio-type", default="equity_release", help="Portfolio type (e.g., equity_release, sme, cre)")
    ap.add_argument("--output-dir", default="out", help="Output directory")
    ap.add_argument("--output-prefix", default=None, help="Override output stem")
    args = ap.parse_args()
    
    # Load inputs
    in_path = Path(args.canonical_csv)
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    
    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = Path(__file__).resolve().parent / registry_path
    
    enum_mapping_path = Path(args.enum_mapping)
    if not enum_mapping_path.is_absolute():
        enum_mapping_path = Path(__file__).resolve().parent / enum_mapping_path
    
    template_order_path = Path(args.template_order)
    if not template_order_path.is_absolute():
        template_order_path = Path(__file__).resolve().parent / template_order_path
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    registry = load_yaml(registry_path)
    enum_mapping = load_yaml(enum_mapping_path)
    # ==============================================================================
    # START: Stacked Config Merge (Product + Client)
    # ==============================================================================
    
    # 1. Load Layer 2 (Product Defaults)
    product_config = {}
    if args.product_defaults:
        p_path = Path(args.product_defaults)
        if not p_path.is_absolute():
            p_path = Path(__file__).resolve().parent / p_path
        product_config = load_yaml_optional(p_path)

    # 2. Load Layer 3 (Client Config)
    client_config = {}
    if args.config:
        c_path = Path(args.config)
        if not c_path.is_absolute():
            c_path = Path(__file__).resolve().parent / c_path
        client_config = load_yaml_optional(c_path)

    # 3. Recursive Deep Merge Function
    def deep_merge(base, overlay):
        """Recursively merges overlay into base."""
        # If either is not a dict, the overlay wins (no merge possible)
        if not isinstance(base, dict) or not isinstance(overlay, dict):
            return overlay
            
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    # 4. Create Final Config (Client overrides Product)
    # We copy product_config to avoid mutating the original if we reused it
    import copy
    config = deep_merge(copy.deepcopy(product_config), client_config)
    
    if args.product_defaults:
        logging.info(f"Merged Product Defaults ({args.product_defaults}) with Client Config")

    # ==============================================================================
    # END: Stacked Config Merge
    # ==============================================================================
    template_order = load_template_order(args.regime, template_order_path)
    
    # Override portfolio_type from config if not specified
    if args.config and "portfolio" in config:
        args.portfolio_type = config["portfolio"].get("asset_class", args.portfolio_type)
    
    # Load canonical dataset
    df_canonical = pd.read_csv(in_path, low_memory=False)
    logging.info(f"Loaded canonical dataset: {len(df_canonical)} rows, {len(df_canonical.columns)} columns")
    
    # Project to regime
    df_regime, report = project_to_regime(
        df_canonical,
        registry,
        enum_mapping,
        config,
        args.regime,
        args.portfolio_type,
        template_order
    )
    
    # Write outputs
    stem = args.output_prefix or in_path.stem.replace("_canonical_typed", "")
    out_csv = out_dir / f"{stem}_{args.regime}_projected.csv"
    out_json = out_dir / f"{stem}_{args.regime}_projection_report.json"
    
    df_regime.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    
    logging.info(f"Wrote: {out_csv}")
    logging.info(f"Wrote: {out_json}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"REGIME PROJECTION COMPLETE: {args.regime}")
    print("=" * 60)
    print(f"Input (canonical):  {len(df_canonical)} rows × {len(df_canonical.columns)} fields")
    print(f"Output (regime):    {len(df_regime)} rows × {len(df_regime.columns)} fields")
    print(f"  • Mandatory:      {report['mandatory_fields']} fields")
    print(f"  • Optional:       {report['optional_fields']} fields")
    
    if report.get("missing_fields"):
        print(f"  ⚠️  Missing:       {len(report['missing_fields'])} fields (will be ND-filled)")
    
    enum_report = report.get("enum_mapping", {})
    if enum_report.get("transformed_fields"):
        print(f"  ✓ Enum mappings:  {len(enum_report['transformed_fields'])} fields transformed")
    
    if enum_report.get("unmapped_values"):
        print(f"  ⚠️  Unmapped enums: {len(enum_report['unmapped_values'])} fields have unmapped values")
    
    nd_report = report.get("nd_defaults", {})
    if nd_report.get("nd_defaults_applied"):
        print(f"  ✓ ND defaults:    {len(nd_report['nd_defaults_applied'])} fields filled")
    
    print("=" * 60)
    print(f"Output file: {out_csv}")
    print(f"Report:      {out_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
