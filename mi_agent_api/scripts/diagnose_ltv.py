"""Diagnose why an LTV (or any bucket) dimension comes back blank for a pack.

Run this against a promoted central lender tape to see, in one shot, exactly
why ``ltv_bucket`` (and friends) materialise to no value — scale mismatch,
out-of-range edges, missing derivation inputs, or a column-name mismatch.

It loads the RAW central lender tape, runs the SAME preparation the API runs
(``prepare_funded_mi_dataset``), and prints:

  * which LTV source columns are present + their raw numeric describe(),
  * the LTV derivation basis the prep recorded (method / inputs / confidence),
  * the prepared ``current_loan_to_value`` describe() (post-normalisation),
  * ``ltv_bucket`` value_counts (incl. blanks) against the configured edges,
  * the full prep report's dimensions_available / missing_dimensions.

Usage (from repo root, with the real pack available locally):

  python -m mi_agent_api.scripts.diagnose_ltv \
      onboarding_output/client_001/mi_2025_10/output/central/18_central_lender_tape.csv

If no path is given it falls back to the data source the API would resolve
(MI_AGENT_* env vars), so you can also just run it after exporting those.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from mi_agent_api.funded_prep import _LTV_INPUTS, prepare_funded_mi_dataset


def _describe(s: pd.Series) -> str:
    n = pd.to_numeric(s, errors="coerce")
    valid = n.dropna()
    if valid.empty:
        return "  (no numeric values)"
    return (
        f"  count={int(valid.count())} min={valid.min():.4g} "
        f"median={valid.median():.4g} mean={valid.mean():.4g} "
        f"max={valid.max():.4g} >1.0={int((valid > 1.0).sum())} "
        f"<=0={int((valid <= 0).sum())}"
    )


def _load(path: str | None) -> tuple[pd.DataFrame, str]:
    if path:
        return pd.read_csv(path), path
    # Fall back to whatever the API would serve.
    from mi_agent_api.data_source import _resolve_central_tape  # type: ignore

    tape = _resolve_central_tape()
    if tape is None:
        raise SystemExit(
            "No path given and no MI_AGENT_* data source resolves. "
            "Pass the path to 18_central_lender_tape.csv."
        )
    return pd.read_csv(tape), str(tape)


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else None
    raw, src = _load(path)
    print(f"\n=== source: {src}  ({len(raw)} rows) ===\n")

    print("--- RAW LTV source columns ---")
    ltv_cols = ["current_loan_to_value", "original_loan_to_value"]
    for tgt in ltv_cols:
        num, den = _LTV_INPUTS[tgt]
        for col in (tgt, num, den):
            present = col in raw.columns
            print(f"{col:32} present={present}")
            if present:
                print(_describe(raw[col]))
    print()

    prepared, report = prepare_funded_mi_dataset(raw)

    print("--- LTV derivation basis (what prep decided) ---")
    for b in report.get("ltv_derivation_basis", []):
        print(f"  {b.get('target')}: method={b.get('method')} "
              f"inputs={b.get('source_fields')} confidence={b.get('confidence')} "
              f"reason={b.get('reason', '-')} detail={b.get('detail', '-')}")
    print()

    print("--- PREPARED LTV columns (post-normalisation) ---")
    for tgt in ltv_cols:
        if tgt in prepared.columns:
            print(f"{tgt}:")
            print(_describe(prepared[tgt]))
    print()

    print("--- ltv_bucket / original_ltv_bucket value_counts ---")
    for b in ("ltv_bucket", "original_ltv_bucket"):
        if b in prepared.columns:
            vc = prepared[b].fillna("(blank)").replace("", "(blank)").value_counts(dropna=False)
            print(f"{b}:")
            for k, v in vc.items():
                print(f"  {k!r:18} {v}")
        else:
            print(f"{b}: column not produced")
    print()

    print("--- prep report summary ---")
    print(f"  dimensions_available: {report.get('dimensions_available')}")
    for m in report.get("missing_dimensions", []):
        print(f"  MISSING {m.get('dimension')}: {m.get('reason')} — {m.get('detail')}")
    if report.get("bucket_errors"):
        print(f"  bucket_errors: {report['bucket_errors']}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
