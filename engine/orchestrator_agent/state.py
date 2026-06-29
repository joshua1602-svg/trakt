"""engine.orchestrator_agent.state — resumable run-state for the orchestration.

A single JSON document captures the whole run: every per-portfolio agent step,
its status, readiness flags, output/manifest paths and any blockers, plus the
consolidation + routing steps. It is the resumability checkpoint AND the
end-to-end lineage record (which agent produced which artifact) for diligence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Step / portfolio / run statuses.
STEP_PENDING = "pending"
STEP_RUNNING = "running"
STEP_DONE = "done"
STEP_HALTED = "halted"      # blocking gate — needs a human, run is resumable
STEP_FAILED = "failed"      # hard error
STEP_SKIPPED = "skipped"

#: The per-portfolio agent steps, in order.
PORTFOLIO_STEPS = ("onboard", "transform", "validate", "stamp")


@dataclass
class StepState:
    name: str
    status: str = STEP_PENDING
    output_path: Optional[str] = None
    manifest_path: Optional[str] = None
    readiness: Dict[str, Any] = field(default_factory=dict)
    blockers: List[str] = field(default_factory=list)
    message: str = ""

    @property
    def done(self) -> bool:
        return self.status == STEP_DONE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "status": self.status,
            "output_path": self.output_path, "manifest_path": self.manifest_path,
            "readiness": self.readiness, "blockers": self.blockers,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StepState":
        return cls(
            name=d["name"], status=d.get("status", STEP_PENDING),
            output_path=d.get("output_path"), manifest_path=d.get("manifest_path"),
            readiness=d.get("readiness") or {}, blockers=d.get("blockers") or [],
            message=d.get("message", ""),
        )


@dataclass
class PortfolioState:
    source_portfolio_id: str
    source_portfolio_type: str
    source_portfolio_label: Optional[str] = None
    acquisition_date: Optional[str] = None
    seller_name: Optional[str] = None
    allow_unknown_acquisition_date: bool = False
    input: str = ""
    status: str = STEP_PENDING
    steps: Dict[str, StepState] = field(default_factory=dict)

    def step(self, name: str) -> StepState:
        if name not in self.steps:
            self.steps[name] = StepState(name)
        return self.steps[name]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_portfolio_id": self.source_portfolio_id,
            "source_portfolio_type": self.source_portfolio_type,
            "source_portfolio_label": self.source_portfolio_label,
            "acquisition_date": self.acquisition_date,
            "seller_name": self.seller_name,
            "allow_unknown_acquisition_date": self.allow_unknown_acquisition_date,
            "input": self.input,
            "status": self.status,
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PortfolioState":
        ps = cls(
            source_portfolio_id=d["source_portfolio_id"],
            source_portfolio_type=d["source_portfolio_type"],
            source_portfolio_label=d.get("source_portfolio_label"),
            acquisition_date=d.get("acquisition_date"),
            seller_name=d.get("seller_name"),
            allow_unknown_acquisition_date=bool(d.get("allow_unknown_acquisition_date")),
            input=d.get("input", ""),
            status=d.get("status", STEP_PENDING),
        )
        ps.steps = {k: StepState.from_dict(v) for k, v in (d.get("steps") or {}).items()}
        return ps


@dataclass
class RunState:
    run_id: str
    client_id: str
    target: str
    out_root: str
    created_at: str
    status: str = STEP_RUNNING
    portfolios: List[PortfolioState] = field(default_factory=list)
    assemble: StepState = field(default_factory=lambda: StepState("assemble"))
    route: StepState = field(default_factory=lambda: StepState("route"))
    project: StepState = field(default_factory=lambda: StepState("project"))
    central_canonical_path: Optional[str] = None
    blockers: List[str] = field(default_factory=list)

    # -- persistence -------------------------------------------------------- #
    def state_path(self) -> Path:
        return Path(self.out_root) / self.run_id / "run_state.json"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id, "client_id": self.client_id,
            "target": self.target, "out_root": self.out_root,
            "created_at": self.created_at, "status": self.status,
            "central_canonical_path": self.central_canonical_path,
            "blockers": self.blockers,
            "portfolios": [p.to_dict() for p in self.portfolios],
            "assemble": self.assemble.to_dict(),
            "route": self.route.to_dict(),
            "project": self.project.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunState":
        rs = cls(
            run_id=d["run_id"], client_id=d["client_id"], target=d["target"],
            out_root=d["out_root"], created_at=d["created_at"],
            status=d.get("status", STEP_RUNNING),
            central_canonical_path=d.get("central_canonical_path"),
            blockers=d.get("blockers") or [],
        )
        rs.portfolios = [PortfolioState.from_dict(p) for p in (d.get("portfolios") or [])]
        if d.get("assemble"):
            rs.assemble = StepState.from_dict(d["assemble"])
        if d.get("route"):
            rs.route = StepState.from_dict(d["route"])
        if d.get("project"):
            rs.project = StepState.from_dict(d["project"])
        return rs

    def save(self) -> Path:
        p = self.state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "RunState":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
