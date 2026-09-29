"""What "scale" means for a portfolio — owner decision D9 (2026-09-29).

A question that names "scale" instead of an amount ("when does the book reach
scale?", "are we at securitisation scale?") is answered against a threshold the
PORTFOLIO carries, never one the model invents and never a fixed ladder:

  - the stage is the client configuration's ``portfolio.stage``, read by
    ``mi_agent.portfolio_metadata.client_portfolio_stage`` — the module that
    already owns reading the client layer;
  - what each stage means is ``config/system/scale_policy.yaml``: an
    established client is at scale once its assets under management (the
    funded balance across all its portfolios) exceed £200MM; a new SPV before
    securitisation has scale at £100MM.

With no stage recorded there is no threshold, and the caller says so rather
than guessing. The figures never reach the model: the vocabulary names the word
``scale``, and the number is resolved here, after interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

#: The one named threshold a milestone target may carry instead of an amount.
NAMED_THRESHOLD = "scale"

POLICY_PATH = Path(__file__).resolve().parents[1] / "config" / "system" / "scale_policy.yaml"
DECISION = "D9"

#: Why no threshold could be given. Stable strings: a refusal names them.
STAGE_NOT_RECORDED = "SCALE_STAGE_NOT_RECORDED"
POLICY_UNAVAILABLE = "SCALE_POLICY_UNAVAILABLE"


@dataclass(frozen=True)
class ScaleThreshold:
    stage: str
    label: str
    threshold: float
    measured_on_label: str

    def to_dict(self) -> Dict[str, Any]:
        return {"named": NAMED_THRESHOLD, "decision": DECISION,
                "stage": self.stage, "stage_label": self.label,
                "threshold": self.threshold,
                "measured_on": self.measured_on_label}


@lru_cache(maxsize=1)
def load_policy() -> Mapping[str, Any]:
    import yaml
    doc = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8")) or {}
    return dict(doc.get("stages") or {})


def resolve(client_id: Optional[str], *,
            document: Optional[Mapping[str, Any]] = None
            ) -> Tuple[Optional[ScaleThreshold], str, str]:
    """``(threshold, reason, detail)``: the portfolio's scale, or why not."""
    from mi_agent.portfolio_metadata import client_portfolio_stage

    stage = client_portfolio_stage(client_id, document=document)
    if stage is None:
        return (None, STAGE_NOT_RECORDED,
                f"scale is specific to the portfolio's stage (D9), and no stage "
                f"is recorded for {client_id or 'this client'}: set "
                f"portfolio.stage in the client configuration to "
                f"pre_securitisation_spv or established")
    try:
        row = load_policy().get(stage) or {}
    except Exception as exc:                                         # noqa: BLE001
        return None, POLICY_UNAVAILABLE, f"{type(exc).__name__}: {exc}"[:200]
    threshold = row.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) \
            or threshold <= 0:
        return (None, POLICY_UNAVAILABLE,
                f"the scale policy gives no positive threshold for {stage!r}")
    return (ScaleThreshold(stage=stage, label=str(row.get("label") or stage),
                           threshold=float(threshold),
                           measured_on_label=str(row.get("measured_on_label") or "")),
            "", "")
