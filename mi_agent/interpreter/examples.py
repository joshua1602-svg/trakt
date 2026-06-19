"""mi_agent.interpreter.examples — golden-example loader + coverage list (Phase 8A).

Loads the deterministic golden question/spec dataset and documents the controlled
question coverage of the baseline interpreter. The dataset itself lives in
``tests/fixtures/mi_interpreter/golden_questions.yaml`` so it can be graded by
both the deterministic baseline and a future LLM interpreter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = (REPO_ROOT / "tests" / "fixtures" / "mi_interpreter"
               / "golden_questions.yaml")

# The controlled question families the deterministic baseline supports.
SUPPORTED_QUESTION_FAMILIES = [
    "current state (funded / pipeline / forecast-funded)",
    "breakdown by portfolio / region / stage",
    "temporal trend (last three months)",
    "temporal compare (vs last month / what changed)",
    "risk grade / IFRS 9 / PD migration",
    "risk deterioration flags",
    "concentration by region / broker",
    "quantile buckets (balance / interest rate / time on book) + configured LTV",
    "ambiguous → clarification (bare stage / portfolio / risk / changes / rate)",
]


def load_golden(path: Path | str | None = None) -> List[Dict[str, Any]]:
    """Load the golden question examples list."""
    path = Path(path) if path else GOLDEN_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("examples") or [])
