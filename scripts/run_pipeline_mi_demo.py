#!/usr/bin/env python3
"""Run a synthetic funded+pipeline MI demo end-to-end."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from analytics.pipeline_expected_funding import build_expected_funding_dataset, load_expected_funding_config
from analytics.pipeline_forward_risk import ForwardRiskSchemaConfig, aggregate_forward_region_exposure
from analytics.pipeline_persistence import PipelinePersistenceConfig, persist_forward_exposure_latest
from analytics.pipeline_prep import PipelinePrepConfig, normalize_pipeline_snapshot
from analytics.pipeline_reconciliation import reconcile_completed_pipeline_to_funded


def _load_yaml(path: str) -> dict:
    import yaml

    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic demo runner for pipeline-to-funded MI flow")
    ap.add_argument(
        "--funded-input",
        default="synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv",
    )
    ap.add_argument(
        "--pipeline-input",
        default="demo/synthetic_pipeline_input.csv",
    )
    ap.add_argument("--expected-config", default="config/client/pipeline_expected_funding.yaml")
    ap.add_argument("--output-dir", default="out_pipeline_demo")
    ap.add_argument("--upload-to-blob", action="store_true", help="Attempt blob upload if Azure is configured")
    args = ap.parse_args()

    funded_path = Path(args.funded_input)
    pipeline_path = Path(args.pipeline_input)
    if not funded_path.exists():
        raise FileNotFoundError(f"Funded input not found: {funded_path}")
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline input not found: {pipeline_path}")

    funded_df = pd.read_csv(funded_path, low_memory=False)
    raw_pipeline_df = pd.read_csv(pipeline_path, low_memory=False)

    pipeline_df = normalize_pipeline_snapshot(raw_pipeline_df, PipelinePrepConfig())
    recon_df = reconcile_completed_pipeline_to_funded(pipeline_df, funded_df)

    expected_cfg_raw = _load_yaml(args.expected_config)
    expected_cfg = load_expected_funding_config(expected_cfg_raw)
    expected_df = build_expected_funding_dataset(pipeline_df, recon_df, expected_cfg)

    forward_schema_raw = expected_cfg_raw.get("forward_risk", {}) if isinstance(expected_cfg_raw, dict) else {}
    configured_region_col = forward_schema_raw.get("funded_region_column")
    configured_exposure_col = forward_schema_raw.get("funded_exposure_column")
    if configured_region_col not in funded_df.columns:
        configured_region_col = None
    if configured_exposure_col not in funded_df.columns:
        configured_exposure_col = None

    forward_region_df = aggregate_forward_region_exposure(
        funded_df=funded_df,
        expected_df=expected_df,
        schema_config=ForwardRiskSchemaConfig(
            funded_region_column=configured_region_col,
            funded_exposure_column=configured_exposure_col,
        ),
    )

    persistence_cfg = PipelinePersistenceConfig(
        enabled=True,
        local_output_dir=args.output_dir,
        upload_to_blob=bool(args.upload_to_blob),
        write_manifest=True,
        blob_subfolder="pipeline/latest",
        blob_container="outbound",
        pipeline_input_path=str(pipeline_path),
        expected_funding_config_path=args.expected_config,
    )

    persisted = persist_forward_exposure_latest(
        funded_df=funded_df,
        funded_source_path=str(funded_path),
        cfg=persistence_cfg,
    )

    print("=== Synthetic Pipeline MI Demo: SUCCESS ===")
    print(f"Funded rows: {len(funded_df):,}")
    print(f"Pipeline rows (normalized): {len(pipeline_df):,}")
    print(f"Reconciliation rows (COMPLETED only): {len(recon_df):,}")
    print(f"Expected funding rows: {len(expected_df):,}")
    print(f"Forward region rows: {len(forward_region_df):,}")
    print(f"Forward exposure CSV: {persisted['csv_path']}")
    if persisted.get("manifest_path"):
        print(f"Forward exposure manifest: {persisted['manifest_path']}")
    print(f"Blob upload attempted: {bool(args.upload_to_blob)}")
    print(f"Blob CSV uploaded: {persisted.get('uploaded_csv_blob')}")


if __name__ == "__main__":
    main()
