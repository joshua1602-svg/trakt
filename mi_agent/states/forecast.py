"""mi_agent.states.forecast — config-driven stage->probability support.

Phase 4. Minimal, deterministic loader for the existing forecast config
(``config/client/pipeline_expected_funding.yaml``). Phase 3 deferred this; here
it is exposed so ``total_forecast_funded`` can fall back to a config
stage->probability mapping when a pipeline row has no row-level probability.

No probabilities are invented: this only reads what the config declares.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORECAST_CONFIG = (REPO_ROOT / "config" / "client"
                           / "pipeline_expected_funding.yaml")


def load_stage_probabilities(path: Optional[Path] = None) -> Dict[str, float]:
    """Return ``{stage_lower: probability}`` from the forecast config.

    Returns ``{}`` if the file or the ``stage_probabilities`` block is absent —
    callers then emit the appropriate missing-probability issues rather than
    inventing values.
    """
    path = Path(path) if path else DEFAULT_FORECAST_CONFIG
    if not path.exists():
        return {}
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = cfg.get("stage_probabilities") or {}
    out: Dict[str, float] = {}
    for stage, prob in raw.items():
        try:
            out[str(stage).strip().lower()] = float(prob)
        except (TypeError, ValueError):
            continue
    return out
