"""apps.blob_trigger_app.target_selection — dataset/frequency → orchestrator target.

Routing rules (Regime/ESMA is NEVER run for pipeline or forecast):

  * funded   : target = "all" when regulatory output is required, else "mi"
  * pipeline : target = "mi"
  * forecast : target = "mi"
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetSelection:
    target: str            # "mi" | "all"  (regime never selected standalone here)
    run_regime: bool       # whether ESMA/Annex 2 is in scope
    reason: str


def select_target(dataset: str, frequency: str, *, regime_required: bool = False) -> TargetSelection:
    ds = (dataset or "").lower()
    if ds in ("pipeline", "forecast"):
        # Pipeline/forecast are MI-only — never Regime/ESMA.
        return TargetSelection(target="mi", run_regime=False,
                               reason=f"{ds} is MI-only (no Regime/ESMA)")
    if ds == "funded":
        if regime_required:
            return TargetSelection(target="all", run_regime=True,
                                   reason="funded + regulatory output required → MI + Regime")
        return TargetSelection(target="mi", run_regime=False,
                               reason="funded, no regulatory output required → MI")
    # Unknown dataset → safe default MI-only.
    return TargetSelection(target="mi", run_regime=False,
                           reason=f"unknown dataset {ds!r} → MI-only default")
