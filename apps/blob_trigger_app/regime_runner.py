"""apps.blob_trigger_app.regime_runner — run the ESMA projector over a central canonical.

Orchestration only: locates a central platform canonical, runs the **existing**
regime projector (``engine.assembler_agent.build_regime_command`` →
``engine/gate_4_projection/regime_projector.py``) over it, and reports the local
output directory. The projector logic is unchanged; ESMA output stays
template-clean and the provenance companion is written by the projector. The
caller (router/persistence) uploads the whole output dir under the regime prefix.

Injectable: the router accepts any callable with this signature, so tests pass a
stub instead of running the real projector.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def default_regime_runner(
    *, central_canonical_path: str, client_id: str, period: str,
    regime: str = "ESMA_Annex2", out_dir: Optional[str] = None,
    allow_unreviewed: bool = False,
) -> Dict[str, Any]:
    """Run the existing regime projector over ``central_canonical_path``.

    Returns ``{output_dir, ok, returncode, stderr_tail}``. Never raises on a
    projector failure — the caller records ``ok`` and persists whatever was
    produced.
    """
    from engine.assembler_agent import build_regime_command

    output_dir = Path(out_dir or "out/_regime") / client_id / period
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_regime_command(central_canonical_path, output_dir, regime,
                               allow_unreviewed=allow_unreviewed)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "output_dir": str(output_dir),
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "command": cmd,
        "stderr_tail": (proc.stderr or "")[-2000:],
    }
