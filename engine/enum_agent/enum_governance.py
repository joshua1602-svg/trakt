from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .enum_mapping_agent import EnumSuggestion


def write_enum_governance_artifact(
    output_dir: Path,
    session_id: str,
    input_file: str,
    namespace: str,
    regime: str,
    model: str,
    prompt_source: str,
    allowed_values_hashes: Dict[str, str],
    suggestions: List[EnumSuggestion],
    aliases_persisted: int,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sent = [s for s in suggestions if s.sent_to_llm]
    payload: Dict[str, Any] = {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": input_file,
        "namespace": namespace,
        "regime": regime,
        "field_names": sorted({s.field_name for s in suggestions}),
        "deterministic_summary": {
            "resolved": sum(1 for s in suggestions if s.deterministic_method != "unmapped"),
            "unmapped": sum(1 for s in suggestions if s.deterministic_method == "unmapped"),
        },
        "llm_summary": {
            "sent_to_llm": len(sent),
            "null_suggestions": sum(1 for s in sent if s.suggested_value is None),
            "avg_confidence": (sum(s.confidence for s in sent) / len(sent)) if sent else 0.0,
            "model": model,
            "prompt_source": prompt_source,
            "allowed_values_hashes": allowed_values_hashes,
        },
        "human_summary": {
            "confirmed": sum(1 for s in suggestions if s.status == "confirmed"),
            "rejected": sum(1 for s in suggestions if s.status == "rejected"),
            "remapped": sum(1 for s in suggestions if s.status == "remapped"),
            "skipped": sum(1 for s in suggestions if s.status == "skipped"),
        },
        "aliases_persisted": aliases_persisted,
        "suggestions": [asdict(s) for s in suggestions],
    }
    out = output_dir / f"enum_governance_{session_id}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
