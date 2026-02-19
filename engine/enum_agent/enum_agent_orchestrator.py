from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Dict, List, Literal

import pandas as pd

from .enum_governance import write_enum_governance_artifact
from .enum_mapping_agent import EnumAliasLearner, LLMEnumMapper, resolve_enums_for_field
from .enum_review import review_cli, review_streamlit

ReviewMode = Literal["cli", "streamlit", "none"]


def run_orchestrator(
    df: pd.DataFrame,
    enum_fields: Dict[str, List[str]],
    namespace: str,
    regime: str,
    output_dir: Path,
    input_file: str = "",
    review_mode: ReviewMode = "cli",
) -> pd.DataFrame:
    session_id = str(uuid.uuid4())
    mapper = LLMEnumMapper()
    all_suggestions = []
    hashes = {}

    # Pass 1: deterministic + LLM suggestions
    for field_name, allowed_values in enum_fields.items():
        mapped, report, candidates, meta = resolve_enums_for_field(
            field_name=field_name,
            series=df[field_name],
            allowed_values=allowed_values,
            namespace=namespace,
            regime=regime,
        )
        df[field_name] = mapped
        hashes[field_name] = meta["allowed_values_hash"]
        if candidates:
            candidates = mapper.suggest(candidates)
        all_suggestions.extend(report)

    # Pass 2: human review + persistence
    if review_mode == "cli":
        review_cli(all_suggestions)
    elif review_mode == "streamlit":
        review_streamlit(all_suggestions)
    # review_mode == "none": skip interactive review (CI / batch runs)

    learner = EnumAliasLearner(Path("config/system/enum_synonyms_confirmed.yaml"))
    persisted = learner.persist_confirmed(all_suggestions)

    # Pass 3: rerun deterministic resolve after learning
    for field_name, allowed_values in enum_fields.items():
        remapped, _, _, _ = resolve_enums_for_field(
            field_name=field_name,
            series=df[field_name],
            allowed_values=allowed_values,
            namespace=namespace,
            regime=regime,
        )
        df[field_name] = remapped

    write_enum_governance_artifact(
        output_dir=output_dir,
        session_id=session_id,
        input_file=input_file,
        namespace=namespace,
        regime=regime,
        model=mapper.model,
        prompt_source="engine/enum_agent/enum_mapping_agent.py::LLMEnumMapper._system_prompt",
        allowed_values_hashes=hashes,
        suggestions=all_suggestions,
        aliases_persisted=persisted,
    )
    return df


def _parse_enum_fields(values: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for item in values:
        field, allowed_csv = item.split("=", 1)
        out[field] = [v.strip() for v in allowed_csv.split(",") if v.strip()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Enum mapping agent orchestrator")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--namespace", default="global")
    ap.add_argument("--regime", default="")
    ap.add_argument(
        "--enum-field",
        action="append",
        default=[],
        help="Repeatable. Format: field=ALLOWED1,ALLOWED2 (e.g. --enum-field gender=M,F,Unknown)",
    )
    ap.add_argument("--output-dir", default="out")
    ap.add_argument(
        "--review-mode",
        choices=["cli", "streamlit", "none"],
        default="cli",
        help="'cli' for interactive terminal review, 'streamlit' for browser UI, 'none' to skip (CI/batch)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    enum_fields = _parse_enum_fields(args.enum_field)
    run_orchestrator(
        df=df,
        enum_fields=enum_fields,
        namespace=args.namespace,
        regime=args.regime,
        output_dir=Path(args.output_dir),
        input_file=args.input_csv,
        review_mode=args.review_mode,
    )


if __name__ == "__main__":
    main()
