"""Pure helpers for pipeline and forward-exposure tabs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

DEFAULT_EXPECTED_CONFIG_RELATIVE = "config/client/pipeline_expected_funding.yaml"


def resolve_expected_funding_config_path(path: str | None) -> Path | None:
    if not path:
        return None

    p = Path(path)
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        repo_root = Path(__file__).resolve().parents[1]
        candidates.append(repo_root / p)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def load_expected_funding_config_yaml(path: str | None) -> tuple[dict, str | None]:
    resolved = resolve_expected_funding_config_path(path)
    if resolved is None:
        raise FileNotFoundError(
            f"Expected funding config file not found: {path}. "
            f"Tried working directory and module-relative resolution."
        )
    with resolved.open("r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}), str(resolved)


def add_pipeline_stratification_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "current_ltv" in out.columns and "ltv_bucket" not in out.columns:
        ltv = pd.to_numeric(out["current_ltv"], errors="coerce")
        bins = [0, 20, 30, 40, 50, 60, 70, 80, 200]
        labels = ["0-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80%+"]
        out["ltv_bucket"] = pd.cut(ltv, bins=bins, labels=labels, include_lowest=True)
        out["ltv_bucket"] = out["ltv_bucket"].cat.add_categories(["Unknown"]).fillna("Unknown")

    if "loan_amount" in out.columns and "ticket_bucket" not in out.columns:
        amt = pd.to_numeric(out["loan_amount"], errors="coerce")
        bins = [0, 50_000, 100_000, 150_000, 200_000, 300_000, 500_000, float("inf")]
        labels = ["<50k", "50-100k", "100-150k", "150-200k", "200-300k", "300-500k", "500k+"]
        out["ticket_bucket"] = pd.cut(amt, bins=bins, labels=labels, include_lowest=True)
        out["ticket_bucket"] = out["ticket_bucket"].cat.add_categories(["Unknown"]).fillna("Unknown")

    if "borrower_age" in out.columns and "age_bucket" not in out.columns:
        age = pd.to_numeric(out["borrower_age"], errors="coerce")
        bins = [0, 55, 60, 65, 70, 75, 80, 85, 120]
        labels = ["<55", "55-60", "60-65", "65-70", "70-75", "75-80", "80-85", "85+"]
        out["age_bucket"] = pd.cut(age, bins=bins, labels=labels, include_lowest=True)
        out["age_bucket"] = out["age_bucket"].cat.add_categories(["Unknown"]).fillna("Unknown")

    return out
