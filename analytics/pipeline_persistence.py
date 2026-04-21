"""Persistence helpers for forward_exposure_latest artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from analytics.blob_storage import is_azure_configured, upload_bytes_to_blob, upload_file_to_blob
from analytics.pipeline_expected_funding import build_expected_funding_dataset, load_expected_funding_config
from analytics.pipeline_prep import PipelinePrepConfig, normalize_pipeline_snapshot
from analytics.pipeline_reconciliation import reconcile_completed_pipeline_to_funded
from analytics.portfolio_semantics import normalize_region_labels, region_codes_from_labels


@dataclass(frozen=True)
class PipelinePersistenceConfig:
    enabled: bool = False
    local_output_dir: str = "out_pipeline"
    blob_subfolder: str = "pipeline/latest"
    write_manifest: bool = True
    upload_to_blob: bool = True
    require_blob_upload: bool = False
    blob_container: str | None = None
    pipeline_input_path: str | None = None
    expected_funding_config_path: str = "config/client/pipeline_expected_funding.yaml"


def _load_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _choose_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _build_funded_forward_rows(funded_df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    exposure_col = _choose_col(funded_df, ["current_principal_balance", "total_balance", "current_outstanding_balance"])
    if not exposure_col:
        raise KeyError("No funded exposure column found for forward exposure persistence.")

    region_col = _choose_col(funded_df, ["geographic_region", "geographic_region_classification", "property_region"])
    product_col = _choose_col(funded_df, ["erm_product_type", "product"])
    broker_col = _choose_col(funded_df, ["broker_channel", "broker"])
    ltv_col = _choose_col(funded_df, ["current_loan_to_value", "current_ltv"])

    out = pd.DataFrame(index=funded_df.index)
    out["exposure_type"] = "FUNDED"
    out["snapshot_date"] = snapshot_date
    out["account_number"] = funded_df.get("loan_policy_number", pd.Series(pd.NA, index=funded_df.index))
    out["loan_identifier"] = funded_df.get("loan_identifier", pd.Series(pd.NA, index=funded_df.index))
    out["loan_policy_number"] = funded_df.get("loan_policy_number", pd.Series(pd.NA, index=funded_df.index))

    region_series = funded_df.get(region_col, pd.Series("", index=funded_df.index)) if region_col else pd.Series("", index=funded_df.index)
    out["property_region"] = normalize_region_labels(region_series)
    out["property_region_code"] = region_codes_from_labels(out["property_region"])

    out["product"] = funded_df.get(product_col, pd.Series(pd.NA, index=funded_df.index)) if product_col else pd.NA
    out["broker"] = funded_df.get(broker_col, pd.Series(pd.NA, index=funded_df.index)) if broker_col else pd.NA
    out["current_ltv"] = pd.to_numeric(funded_df.get(ltv_col, pd.NA), errors="coerce") if ltv_col else pd.NA
    out["exposure_amount"] = pd.to_numeric(funded_df[exposure_col], errors="coerce").fillna(0.0)

    out["pipeline_opportunity_id"] = pd.NA
    out["reconciliation_match_rule"] = pd.NA
    out["model_version"] = pd.NA
    return out


def _build_expected_forward_rows(
    funded_df: pd.DataFrame,
    pipeline_input_path: str | None,
    expected_config_path: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not pipeline_input_path:
        return pd.DataFrame(), {}

    pipeline_path = Path(pipeline_input_path)
    if not pipeline_path.exists():
        return pd.DataFrame(), {}

    raw_pipeline = pd.read_csv(pipeline_path, low_memory=False)
    pipeline_df = normalize_pipeline_snapshot(raw_pipeline, PipelinePrepConfig())
    recon_df = reconcile_completed_pipeline_to_funded(pipeline_df, funded_df)
    model_cfg_raw = _load_yaml(expected_config_path)
    model_cfg = load_expected_funding_config(model_cfg_raw)

    expected_df = build_expected_funding_dataset(pipeline_df, recon_df, model_cfg)
    if expected_df.empty:
        return pd.DataFrame(), {"model_version": model_cfg.model_version}

    out = pd.DataFrame(index=expected_df.index)
    out["exposure_type"] = "EXPECTED"
    out["snapshot_date"] = pd.to_datetime(expected_df.get("snapshot_date"), errors="coerce")
    out["account_number"] = pd.NA
    out["loan_identifier"] = pd.NA
    out["loan_policy_number"] = pd.NA
    out["property_region"] = normalize_region_labels(expected_df.get("property_region", ""))
    out["property_region_code"] = expected_df.get("property_region_code", pd.NA)
    out["product"] = expected_df.get("product", pd.NA)
    out["broker"] = expected_df.get("broker", pd.NA)
    out["current_ltv"] = pd.to_numeric(expected_df.get("current_ltv", pd.NA), errors="coerce")
    out["exposure_amount"] = pd.to_numeric(expected_df.get("expected_funded_amount", pd.NA), errors="coerce").fillna(0.0)
    out["pipeline_opportunity_id"] = expected_df.get("pipeline_opportunity_id", pd.NA)
    out["reconciliation_match_rule"] = expected_df.get("reconciliation_match_rule", pd.NA)
    out["model_version"] = expected_df.get("model_version", model_cfg.model_version)

    meta = {
        "pipeline_input_path": str(pipeline_path),
        "model_version": model_cfg.model_version,
        "expected_funding_config_path": expected_config_path,
    }
    return out, meta


def persist_forward_exposure_latest(
    funded_df: pd.DataFrame,
    funded_source_path: str,
    cfg: PipelinePersistenceConfig,
    run_timestamp_utc: str | None = None,
) -> dict[str, Any]:
    if funded_df is None or funded_df.empty:
        raise ValueError("Cannot persist forward exposure: funded dataframe is empty.")

    run_ts = run_timestamp_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    snapshot_date = pd.Timestamp(datetime.now(timezone.utc).date())

    funded_out = _build_funded_forward_rows(funded_df, snapshot_date=snapshot_date)
    expected_out, expected_meta = _build_expected_forward_rows(
        funded_df,
        pipeline_input_path=cfg.pipeline_input_path,
        expected_config_path=cfg.expected_funding_config_path,
    )

    combined = pd.concat([funded_out, expected_out], ignore_index=True)

    out_dir = Path(cfg.local_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "forward_exposure_latest.csv"
    json_path = out_dir / "forward_exposure_latest.json"
    pointer_path = out_dir / "latest_forward_exposure_path.txt"

    combined.to_csv(csv_path, index=False)
    pointer_path.write_text(str(csv_path.resolve()), encoding="utf-8")

    manifest = {
        "run_timestamp_utc": run_ts,
        "snapshot_date": str(snapshot_date.date()),
        "source_funded_file_path": funded_source_path,
        "source_pipeline_file_path": expected_meta.get("pipeline_input_path"),
        "expected_funding_model_version": expected_meta.get("model_version"),
        "expected_funding_config_path": expected_meta.get("expected_funding_config_path"),
        "total_row_count": int(len(combined)),
        "funded_row_count": int((combined["exposure_type"] == "FUNDED").sum()),
        "expected_row_count": int((combined["exposure_type"] == "EXPECTED").sum()),
        "output_csv_path": str(csv_path.resolve()),
        "blob_destination_path": f"{cfg.blob_subfolder.rstrip('/')}/forward_exposure_latest.csv",
    }

    if cfg.write_manifest:
        json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    uploaded_csv = None
    uploaded_json = None
    if cfg.upload_to_blob:
        if not is_azure_configured():
            msg = "Azure blob is not configured for forward exposure upload."
            if cfg.require_blob_upload:
                raise EnvironmentError(msg)
        else:
            csv_blob = f"{cfg.blob_subfolder.rstrip('/')}/forward_exposure_latest.csv"
            uploaded_csv = upload_file_to_blob(str(csv_path), csv_blob, container=cfg.blob_container)

            if cfg.write_manifest:
                json_blob = f"{cfg.blob_subfolder.rstrip('/')}/forward_exposure_latest.json"
                uploaded_json = upload_bytes_to_blob(
                    json.dumps(manifest, indent=2).encode("utf-8"),
                    json_blob,
                    container=cfg.blob_container,
                )

    return {
        "csv_path": str(csv_path),
        "manifest_path": str(json_path) if cfg.write_manifest else None,
        "pointer_path": str(pointer_path),
        "uploaded_csv_blob": uploaded_csv,
        "uploaded_json_blob": uploaded_json,
        "manifest": manifest,
    }
