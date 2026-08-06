"""simulation.runner — the developer entry point and the run orchestration.

    python -m simulation.runner list
    python -m simulation.runner generate  --case bridge_maturity_wall_v1
    python -m simulation.runner run       --case bridge_maturity_wall_v1
    python -m simulation.runner run-all   --profile smoke
    python -m simulation.runner run-all   --profile standard
    python -m simulation.runner run-all   --profile performance
    python -m simulation.runner reproduce --case bridge_maturity_wall_v1 --seed 41204

Every run writes a structured evidence directory::

    run/<case_id>/
      manifest.json
      generated_sources/            one file per dialect per reporting period
      canonical_outputs/            the production canonical per dialect/period
      integrated_snapshot_evidence/ the governed snapshot store + registrations
      expected_truth/               the independent expected-truth package
      mi_results/  risk_results/  agent_results/  regime_outputs/
      assertion_results.json
      run_summary.json

The runner never calculates anything analytical. It drives production, collects
evidence and delegates every judgement to :mod:`simulation.assertions`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import SIMULATION_VERSION
from . import manifests as M
from . import pipeline as P
from .assertions import canonical as A_canonical
from .assertions import history as A_history
from .assertions import mi as A_mi
from .assertions import platform as A_platform
from .assertions import regime as A_regime
from .assertions import risk as A_risk
from .dialects import render
from .history import ReconciliationError, canonical_frame_rows, generate_history
from .models import (
    DIALECT_CLEAN_CSV,
    EXPECT_FAILURE,
    REGIME_APPLICABLE,
    SCALES,
    STAGES,
    STAGE_AGENT,
    STAGE_ASSEMBLE,
    STAGE_CANONICAL,
    STAGE_EQUIVALENCE,
    STAGE_GENERATE,
    STAGE_HISTORY,
    STAGE_INGEST,
    STAGE_MI,
    STAGE_REGIME,
    STAGE_RENDER,
    STAGE_RISK,
    Assertion,
    CaseManifest,
    CaseResult,
    FailureClass,
    StageResult,
)
from .reference_truth import (
    build_expected_truth,
    combine_truths,
    write_expected_truth,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = REPO_ROOT / "out_simulation"

#: Per-asset-class master client config (isolated from ``config/client/*``).
CLIENT_CONFIGS = {
    "equity_release": "config_client_SIM_EQUITY_RELEASE.yaml",
    "bridge": "config_client_SIM_BRIDGE.yaml",
    "asset_finance": "config_client_SIM_ASSET_FINANCE.yaml",
}

#: The dialect every other dialect is compared against for equivalence.
REFERENCE_DIALECT = DIALECT_CLEAN_CSV


def client_config(case: CaseManifest) -> Path:
    return Path(__file__).resolve().parent / "config" / CLIENT_CONFIGS[case.asset_class]


def _stage(name: str, started: float, *, ok: bool = True,
           assertions: Optional[List[Assertion]] = None, **kw) -> StageResult:
    assertions = assertions or []
    failed = [a for a in assertions if not a.passed]
    result = StageResult(stage=name, ok=ok and not failed,
                         duration_seconds=time.monotonic() - started,
                         assertions=assertions, **kw)
    if failed and result.failure_class is None:
        result.failure_class = failed[0].failure_class or FailureClass.UNCLASSIFIED_EXCEPTION
        result.message = failed[0].detail or failed[0].name
    return result


def run_case(case: CaseManifest, *, run_root: Path, scale: str = "small",
             jobs: int = 4, keep_sources: bool = True,
             measure_memory: bool = False,
             run_label: Optional[str] = None,
             stages: Optional[Sequence[str]] = None,
             deadline: Optional[float] = None,
             progress: bool = False) -> CaseResult:
    """Generate, render, ingest, integrate and verify one case end to end.

    ``run_label`` names the evidence directory. It defaults to the case id and
    is only overridden when one case is run more than once in a single profile
    — the performance profile runs its large case at two scales — so that the
    two runs do not overwrite each other's evidence.

    ``stages`` restricts the run to a subset of :data:`STAGES`. The earlier
    stages a selected stage depends on always run: asking for ``mi`` without
    ``ingest`` is not a smaller measurement, it is a broken one. Stages that
    are excluded are recorded as skipped WITH the reason, so a narrowed run
    never reads as a run that verified everything.

    ``deadline`` is a ``time.monotonic()`` value. It is checked BETWEEN stages,
    never mid-stage: a half-written canonical is not evidence. On expiry the
    case stops cleanly, every completed stage keeps its timing and its
    artefacts, and the remaining stages are recorded as skipped ``timeout``.

    ``progress`` prints each stage as it starts and finishes, with its elapsed
    time, so a long run is observable while it runs rather than only after it.
    """
    label = run_label or case.case_id
    run_dir = Path(run_root) / label
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    result = CaseResult(
        case_id=case.case_id, seed=case.seed, asset_class=case.asset_class,
        simulation_version=case.simulation_version, expectation=case.expectation,
        expected_failure_class=case.expected_failure_class,
        run_dir=str(run_dir), manifest_hash=case.content_hash(),
        run_label=label, scale=scale)
    M.write_manifest(case, run_dir / "manifest.json")
    P.reset_observations()
    if measure_memory:
        tracemalloc.start()

    # A stage that fails ends the run: everything after it would be verifying
    # outputs that were never produced. ``observed_failure_class`` — not
    # ``ok`` — is the signal, because ``ok`` answers a different question
    # ("did the case end the way its manifest declared"), and a guard case
    # whose declared failure has not happened YET is legitimately not ok.
    def _stopped() -> bool:
        return result.observed_failure_class is not None

    selected = _selected_stages(stages)
    result.selected_stages = list(selected)
    tracker = _Progress(label, enabled=progress)

    def _gate(stage: str, ready: bool = True) -> bool:
        """May ``stage`` run? Records WHY NOT whenever it may not.

        Every stage that does not run leaves a skip record with its reason, so
        the evidence distinguishes "verified" from "not selected", "out of
        time" and "the stage before it produced nothing to verify".
        """
        if _stopped():
            return False
        if stage not in selected:
            reason = "not selected for this run"
        elif deadline is not None and time.monotonic() >= deadline:
            result.timed_out = True
            reason = "timeout: the run budget expired before this stage"
        elif not ready:
            reason = "an earlier stage produced no input for this stage"
        else:
            tracker.start(stage)
            return True
        result.stages.append(StageResult(stage=stage, ok=True, skipped=True,
                                         skip_reason=reason))
        return False

    try:
        history = truth = sources = canonicals = None
        assembled = None
        store = None
        if _gate(STAGE_GENERATE):
            history, truth = _stage_generate(case, run_dir, result, scale)
            tracker.done(result)
        if _gate(STAGE_RENDER, history is not None):
            sources = _stage_render(case, run_dir, result, history)
            tracker.done(result)
        if _gate(STAGE_INGEST, bool(sources)):
            canonicals = _stage_ingest(case, run_dir, result, sources, jobs)
            tracker.done(result)
        if _gate(STAGE_CANONICAL, bool(canonicals)):
            _stage_canonical(case, run_dir, result, canonicals, truth)
            tracker.done(result)
        if _gate(STAGE_EQUIVALENCE, bool(canonicals)):
            _stage_equivalence(case, run_dir, result, canonicals, truth)
            tracker.done(result)
        if _gate(STAGE_ASSEMBLE, bool(canonicals)):
            assembled = _stage_assemble(case, run_dir, result, canonicals, truth)
            tracker.done(result)
        if _gate(STAGE_HISTORY, bool(assembled)):
            store = _stage_history(case, run_dir, result, assembled, truth)
            tracker.done(result)
        if _gate(STAGE_MI, store is not None):
            _stage_mi(case, run_dir, result, store, truth)
            tracker.done(result)
        if _gate(STAGE_RISK, store is not None):
            _stage_risk(case, run_dir, result, store, truth)
            tracker.done(result)
        if _gate(STAGE_AGENT, store is not None):
            _stage_agent(case, run_dir, result, store, truth, assembled)
            tracker.done(result)
        if _gate(STAGE_REGIME, bool(assembled)):
            _stage_regime(case, run_dir, result, assembled, truth)
            tracker.done(result)
    except Exception as exc:  # noqa: BLE001 - an escape is itself a finding
        import traceback
        # An unhandled exception is NEVER a valid outcome, so it is recorded in
        # full: a classified failure needs the place it came from, not a
        # one-line summary a reader has to reproduce to understand.
        result.stages.append(StageResult(
            stage="runner", ok=False,
            failure_class=FailureClass.UNCLASSIFIED_EXCEPTION,
            message=f"{type(exc).__name__}: {exc}",
            artefacts={"traceback": traceback.format_exc().splitlines()[-14:]}))

    if not keep_sources:
        shutil.rmtree(run_dir / "generated_sources", ignore_errors=True)
    return _finish(result, measure_memory)


#: Every stage a selected stage needs before it can mean anything. Measuring
#: ``mi`` without ``ingest`` is not a cheaper measurement, it is a broken one.
STAGE_REQUIRES = {
    STAGE_RENDER: (STAGE_GENERATE,),
    STAGE_INGEST: (STAGE_GENERATE, STAGE_RENDER),
    STAGE_CANONICAL: (STAGE_GENERATE, STAGE_RENDER, STAGE_INGEST),
    STAGE_EQUIVALENCE: (STAGE_GENERATE, STAGE_RENDER, STAGE_INGEST),
    STAGE_ASSEMBLE: (STAGE_GENERATE, STAGE_RENDER, STAGE_INGEST),
    STAGE_HISTORY: (STAGE_GENERATE, STAGE_RENDER, STAGE_INGEST, STAGE_ASSEMBLE),
    STAGE_MI: (STAGE_GENERATE, STAGE_RENDER, STAGE_INGEST, STAGE_ASSEMBLE,
               STAGE_HISTORY),
    STAGE_RISK: (STAGE_GENERATE, STAGE_RENDER, STAGE_INGEST, STAGE_ASSEMBLE,
                 STAGE_HISTORY),
    STAGE_AGENT: (STAGE_GENERATE, STAGE_RENDER, STAGE_INGEST, STAGE_ASSEMBLE,
                  STAGE_HISTORY),
    STAGE_REGIME: (STAGE_GENERATE, STAGE_RENDER, STAGE_INGEST, STAGE_ASSEMBLE),
}


def _selected_stages(stages: Optional[Sequence[str]]) -> List[str]:
    """Resolve a stage selection, closing it over its prerequisites."""
    if not stages:
        return list(STAGES)
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; known: {list(STAGES)}")
    wanted = set(stages)
    for stage in list(wanted):
        wanted.update(STAGE_REQUIRES.get(stage, ()))
    return [s for s in STAGES if s in wanted]


class _Progress:
    """Live per-stage progress, printed as it happens rather than at the end."""

    def __init__(self, label: str, *, enabled: bool):
        self.label = label
        self.enabled = enabled
        self.stage: Optional[str] = None
        self.started = 0.0

    def start(self, stage: str) -> None:
        self.stage, self.started = stage, time.monotonic()
        if self.enabled:
            print(f"    {self.label} :: {stage} ...", flush=True)

    def done(self, result: CaseResult) -> None:
        if not self.enabled or self.stage is None:
            return
        elapsed = time.monotonic() - self.started
        last = result.stages[-1] if result.stages else None
        mark = "skip" if (last and last.skipped) else (
            "ok" if (last and last.ok) else "FAIL")
        counts = len(last.assertions) if last else 0
        print(f"    {self.label} :: {self.stage:22s} {mark:5s} "
              f"{elapsed:8.2f}s  {counts} assertions", flush=True)
        self.stage = None


def _finish(result: CaseResult, measure_memory: bool) -> CaseResult:
    if measure_memory and tracemalloc.is_tracing():
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result.peak_memory_mb = round(peak / (1024 * 1024), 2)
    for stage in result.stages:
        result.timings[stage.stage] = stage.duration_seconds
    # OBSERVED, not declared: a module appears here only because something
    # recorded evidence that it ran during THIS case.
    observed = P.observed_modules()
    result.production_modules = sorted(observed)
    result.production_module_evidence = dict(observed)
    run_dir = Path(result.run_dir)
    (run_dir / "assertion_results.json").write_text(
        json.dumps([a.to_dict() for s in result.stages for a in s.assertions],
                   indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "run_summary.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str) + "\n",
        encoding="utf-8")
    return result


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def _stage_generate(case, run_dir: Path, result: CaseResult, scale: str):
    """Generate every funded source this case delivers.

    A single-source case generates exactly one history and its truth IS the
    case truth, unchanged. A multi-source case generates one history per
    source, from that source's own seed stream, and the SPV-level truth is the
    INDEPENDENT SUM of the per-source truths — never a re-run of the generator
    over a merged book, and never anything read back from the platform.
    """
    started = time.monotonic()
    total_loans = _scaled_loan_count(case, scale)
    sources = case.funded_sources()

    histories: Dict[str, Any] = {}
    truths: Dict[str, Any] = {}
    for source in sources:
        source_case = case.source_case(source)
        count = max(1, int(round(total_loans * source.share)))
        try:
            histories[source.portfolio_id] = generate_history(
                source_case, loan_count=count)
        except ReconciliationError as exc:
            result.stages.append(StageResult(
                stage=STAGE_GENERATE, ok=False,
                failure_class=FailureClass.SIMULATION_DEFECT,
                message=f"{source.portfolio_id}: {exc}",
                duration_seconds=time.monotonic() - started))
            return None, None
        truths[source.portfolio_id] = build_expected_truth(
            source_case, histories[source.portfolio_id])

    if case.is_multi_source:
        truth = combine_truths(case, truths)
    else:
        truth = truths[sources[0].portfolio_id]
    truth["scale"] = scale
    truth["opening_loan_count_used"] = total_loans
    write_expected_truth(truth, run_dir / "expected_truth" / "expected_truth.json")

    assertions: List[Assertion] = []
    for pid, history in histories.items():
        tag = f"{pid}: " if case.is_multi_source else ""
        assertions += [
            A_canonical.check(STAGE_GENERATE, f"{tag}history has the declared periods",
                              len(history.periods) == case.months,
                              expected=case.months, actual=len(history.periods),
                              failure_class=FailureClass.SIMULATION_DEFECT),
            A_canonical.check(STAGE_GENERATE, f"{tag}every period reconciles",
                              all(p.reconciles() for p in history.periods),
                              failure_class=FailureClass.SIMULATION_DEFECT),
            A_canonical.check(STAGE_GENERATE, f"{tag}loan identifiers are stable",
                              _identifiers_are_stable(history),
                              detail="A loan must keep its identity across the "
                                     "history or nothing can be reconciled.",
                              failure_class=FailureClass.SIMULATION_DEFECT),
        ]
    if case.is_multi_source:
        assertions += _assert_sources_are_distinct(case, histories)

    result.stages.append(_stage(STAGE_GENERATE, started, assertions=assertions,
                                artefacts={"periods": case.months,
                                           "opening_loans": total_loans,
                                           "sources": list(histories)}))
    return histories, truth


def _assert_sources_are_distinct(case, histories: Dict[str, Any]) -> List[Assertion]:
    """Two funded populations must be genuinely two populations."""
    out: List[Assertion] = []
    ids: Dict[str, set] = {
        pid: {loan.loan_id for period in h.periods for loan in period.loans}
        for pid, h in histories.items()}
    overlap: Dict[str, list] = {}
    seen: Dict[str, str] = {}
    for pid, loan_ids in ids.items():
        for loan_id in loan_ids:
            if loan_id in seen:
                overlap.setdefault(f"{seen[loan_id]}|{pid}", []).append(loan_id)
            seen[loan_id] = pid
    out.append(A_canonical.check(
        STAGE_GENERATE, "loan identifiers are distinct across sources",
        not overlap, actual={k: v[:5] for k, v in overlap.items()},
        detail="Two sources sharing a loan identifier would make the "
               "assembler's composite key the only thing preventing a "
               "collision, which is not what this case is testing.",
        failure_class=FailureClass.SIMULATION_DEFECT))

    dates = {pid: [p.reporting_date for p in h.periods]
             for pid, h in histories.items()}
    aligned = len({tuple(v) for v in dates.values()}) == 1
    out.append(A_canonical.check(
        STAGE_GENERATE, "every source reports on the same month-ends", aligned,
        actual=dates, detail="Assembly of misaligned reporting dates would "
                             "consolidate different points in time.",
        failure_class=FailureClass.SIMULATION_DEFECT))
    return out


def _scaled_loan_count(case: CaseManifest, scale: str) -> int:
    """Scale a manifest's opening loan count without editing the manifest."""
    base = SCALES["small"]
    return max(1, int(round(case.opening_loan_count * SCALES[scale] / base)))


def _identifiers_are_stable(history) -> bool:
    seen: Dict[str, str] = {}
    for period in history.periods:
        for loan in period.loans:
            previous = seen.get(loan.loan_id)
            if previous is not None and previous != loan.origination_date:
                return False
            seen[loan.loan_id] = loan.origination_date
    return True


def _stage_render(case, run_dir: Path, result: CaseResult, histories):
    """Render every declared dialect for every period of every funded source.

    Returns ``{dialect: {portfolio_id: {reporting_date: path}}}``. A separately
    delivered population is a separately delivered FILE — that is what makes
    the assembly step real rather than a partition of one export.
    """
    started = time.monotonic()
    root = run_dir / "generated_sources"
    sources: Dict[str, Dict[str, Dict[str, Path]]] = {
        d: {pid: {} for pid in histories} for d in case.dialects}
    for pid, history in histories.items():
        source_case = next(case.source_case(s) for s in case.funded_sources()
                           if s.portfolio_id == pid)
        for index, snapshot in enumerate(history.periods):
            rows = canonical_frame_rows(source_case, history, index)
            # A declared ``switch_format`` is handled INSIDE the renderer: the
            # dialect keeps its vocabulary and only its container changes, so
            # the equivalence comparison still lines the periods up.
            for dialect in case.dialects:
                path = render(dialect, rows, source_case, period_index=index,
                              reporting_date=snapshot.reporting_date,
                              out_dir=root / dialect / pid / snapshot.reporting_date,
                              evolution=case.schema_evolution)
                sources[dialect][pid][snapshot.reporting_date] = path

    expected = len(case.dialects) * sum(len(h.periods) for h in histories.values())
    produced = sum(len(by_date) for by_pid in sources.values()
                   for by_date in by_pid.values())
    assertions = [A_canonical.check(
        STAGE_RENDER, "every dialect rendered every period of every source",
        produced == expected, expected=expected, actual=produced,
        failure_class=FailureClass.SIMULATION_DEFECT)]
    result.stages.append(_stage(STAGE_RENDER, started, assertions=assertions,
                                artefacts={"files": produced,
                                           "dialects": list(case.dialects),
                                           "sources": list(histories)}))
    return sources


def _stage_ingest(case, run_dir: Path, result: CaseResult,
                  sources: Dict[str, Dict[str, Path]], jobs: int):
    """Run the REAL funded ingestion path for every rendered source."""
    started = time.monotonic()
    config = client_config(case)
    by_pid = {s.portfolio_id: case.source_case(s) for s in case.funded_sources()}
    work: List[tuple] = [(d, pid, date, path)
                         for d, per_pid in sources.items()
                         for pid, by_date in sorted(per_pid.items())
                         for date, path in sorted(by_date.items())]

    def _run_one(item):
        dialect, pid, date, path = item
        base = run_dir / "canonical_outputs" / dialect / pid / date
        # Each source is ingested with ITS OWN provenance — that is what makes
        # direct/acquired real on the tape rather than a label added later.
        return dialect, pid, date, P.run_mi_gates(
            path, by_pid[pid], reporting_date=date, out_dir=base / "out",
            validation_dir=base / "validation", master_config=config)

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        outcomes = list(pool.map(_run_one, work))

    canonicals: Dict[str, Dict[str, Dict[str, Path]]] = {}
    failures: List[Dict[str, Any]] = []
    for dialect, pid, date, gate in outcomes:
        if gate.ok and gate.canonical_path is not None:
            canonicals.setdefault(dialect, {}).setdefault(pid, {})[date] = \
                gate.canonical_path
        else:
            failures.append({"dialect": dialect, "portfolio_id": pid,
                             "reporting_date": date,
                             "returncode": gate.returncode,
                             "canonical_validation_failed":
                                 gate.canonical_validation_failed,
                             "canonical_row_count": gate.canonical_row_count,
                             "gate_lines": gate.gate_lines,
                             "stderr": gate.stderr[-1500:]})

    if failures:
        classification = _classify_ingest_failure(case, failures)
        result.stages.append(StageResult(
            stage=STAGE_INGEST, ok=False, failure_class=classification,
            message=f"{len(failures)} of {len(work)} ingestion runs failed",
            duration_seconds=time.monotonic() - started,
            artefacts={"failures": failures}))
        return canonicals

    result.stages.append(_stage(
        STAGE_INGEST, started,
        artefacts={"runs": len(work),
                   "sources": sorted({pid for _d, pid, _dt, _g in outcomes}),
                   "gates": outcomes[0][3].gate_lines if outcomes else []}))
    return canonicals


def _classify_ingest_failure(case, failures: List[Dict[str, Any]]) -> str:
    """Classify an ingestion failure at the boundary that produced it."""
    blob = "\n".join(str(f.get("stderr", "")) for f in failures).lower()
    # Classify from what the PLATFORM reported, not from what the manifest said
    # it was going to do — otherwise a case could declare its way to a pass.
    if "unsupported source schema" in blob or "no registry fields selected" in blob:
        return FailureClass.UNSUPPORTED_SOURCE_SCHEMA
    if any(f.get("canonical_validation_failed") for f in failures):
        return FailureClass.MISSING_MANDATORY_DATA
    if any(f.get("canonical_row_count") == 0 for f in failures):
        return FailureClass.UNSUPPORTED_SOURCE_SCHEMA
    return FailureClass.PRODUCTION_DEFECT


def _source_truths(case, truth) -> Dict[str, Any]:
    """``{portfolio_id: truth}`` — the per-source truth each canonical is checked against.

    A per-source canonical carries ONE delivery, so it must be checked against
    that delivery's own expected figures. The SPV-level combined truth is what
    the ASSEMBLED view is checked against, in the assembly stage.
    """
    if case.is_multi_source:
        return dict(truth["by_source"])
    return {case.portfolio_id: truth}


def _stage_canonical(case, run_dir: Path, result: CaseResult,
                     canonicals: Dict[str, Dict[str, Dict[str, Path]]], truth):
    started = time.monotonic()
    per_pid = (canonicals.get(REFERENCE_DIALECT)
               or next(iter(canonicals.values()), {}))
    truths = _source_truths(case, truth)
    assertions: List[Assertion] = []
    for pid, by_date in sorted(per_pid.items()):
        source_case = next((case.source_case(s) for s in case.funded_sources()
                            if s.portfolio_id == pid), case)
        for period in truths[pid]["periods"]:
            path = by_date.get(period["reporting_date"])
            if path is None:
                continue
            frame = A_canonical.load_canonical(path)
            assertions += A_canonical.assert_canonical_truth(
                frame, period, source_case)
    result.stages.append(_stage(STAGE_CANONICAL, started, assertions=assertions))


def _stage_equivalence(case, run_dir: Path, result: CaseResult,
                       canonicals: Dict[str, Dict[str, Dict[str, Path]]], truth):
    started = time.monotonic()
    assertions: List[Assertion] = []
    reference = canonicals.get(REFERENCE_DIALECT)
    if reference is None or len(canonicals) < 2:
        result.stages.append(StageResult(
            stage=STAGE_EQUIVALENCE, ok=True, skipped=True,
            skip_reason="fewer than two dialects were rendered for this case",
            duration_seconds=time.monotonic() - started))
        return

    # Equivalence is a property of ONE delivery rendered three ways, so it is
    # compared per source: mixing two deliveries into one comparison would
    # compare different books, not different dialects.
    for pid in sorted(reference):
        for date in sorted(reference[pid]):
            ref_path = reference[pid][date]
            ref_frame = A_canonical.load_canonical(ref_path)
            lineage: Dict[str, Dict[str, Any]] = {}
            for dialect, per_pid in canonicals.items():
                path = per_pid.get(pid, {}).get(date)
                if path is None:
                    continue
                lineage[dialect] = _read_json(path.parent / "field_lineage.json")
                if dialect == REFERENCE_DIALECT:
                    continue
                assertions += A_canonical.assert_dialect_equivalence(
                    ref_frame, REFERENCE_DIALECT,
                    A_canonical.load_canonical(path), dialect, date=date)
            assertions += A_canonical.assert_source_lineage_distinct(lineage, date)
    result.stages.append(_stage(STAGE_EQUIVALENCE, started, assertions=assertions))


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _stage_assemble(case, run_dir: Path, result: CaseResult,
                    canonicals: Dict[str, Dict[str, Dict[str, Path]]], truth):
    """Consolidate each period's per-source canonicals into the SPV view.

    Runs the PRODUCTION Assembler Agent, which runs
    ``engine/platform_assembler.py``. Returns ``{dialect: {date: path}}`` — the
    single canonical every downstream stage then reads, so MI, risk, the Agent
    and the regime all answer from the assembled SPV rather than from one of
    its parts.

    For a single-source case the stage is a genuine no-op and says so: there is
    nothing to consolidate, and running the assembler over one input would
    prove nothing while pretending to.
    """
    started = time.monotonic()
    if not case.is_multi_source:
        flat = {d: dict(next(iter(per_pid.values())))
                for d, per_pid in canonicals.items() if per_pid}
        result.stages.append(StageResult(
            stage=STAGE_ASSEMBLE, ok=True, skipped=True,
            duration_seconds=time.monotonic() - started,
            skip_reason="single-source case: there is nothing to consolidate, "
                        "so engine/platform_assembler.py is correctly not run"))
        return flat

    out_root = run_dir / "platform_assembly"
    assembled: Dict[str, Dict[str, Path]] = {}
    manifests: Dict[str, Any] = {}
    for dialect, per_pid in canonicals.items():
        for date in truth["reporting_dates"]:
            inputs = [per_pid[pid][date] for pid in sorted(per_pid)
                      if date in per_pid.get(pid, {})]
            if len(inputs) != len(per_pid):
                continue
            outcome = P.assemble_spv_canonical(
                inputs, case, reporting_date=date,
                out_dir=out_root / dialect / date)
            assembled.setdefault(dialect, {})[date] = outcome["canonical"]
            if dialect == REFERENCE_DIALECT:
                manifests[date] = outcome["manifest"]

    (out_root / "assembler_manifests.json").write_text(
        json.dumps(manifests, indent=2, default=str) + "\n", encoding="utf-8")

    assertions = A_platform.assert_assembly(
        case, truth, assembled.get(REFERENCE_DIALECT, {}), manifests)
    result.stages.append(_stage(
        STAGE_ASSEMBLE, started, assertions=assertions,
        artefacts={"periods_assembled": len(assembled.get(REFERENCE_DIALECT, {})),
                   "sources": sorted(next(iter(canonicals.values()), {})),
                   "assembler_run_ids": [m.get("assembler_run_id")
                                         for m in manifests.values()]}))
    return assembled


def _stage_history(case, run_dir: Path, result: CaseResult,
                   canonicals: Dict[str, Dict[str, Path]], truth):
    """Register every period through the production snapshot store."""
    started = time.monotonic()
    evidence = run_dir / "integrated_snapshot_evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    store = P.snapshot_store(evidence / "snapshots")
    reference = canonicals.get(REFERENCE_DIALECT) or {}

    registrations: List[Dict[str, Any]] = []
    for period in truth["periods"]:
        date = period["reporting_date"]
        path = reference.get(date)
        if path is None:
            continue
        registrations.append(P.register_period(store, path, case,
                                               reporting_date=date))
        # Publish into the dated platform layout the MI capability resolves.
        P.publish_platform(run_dir, case, reporting_date=date, canonical=path,
                           latest=(date == truth["reporting_dates"][-1]))
    (evidence / "registrations.json").write_text(
        json.dumps(registrations, indent=2, default=str) + "\n", encoding="utf-8")

    assertions = A_history.assert_registration(registrations, truth)
    assertions += A_history.assert_resolution(store, case, truth)
    assertions += A_history.assert_reconciliation(store, case, truth)
    assertions += A_history.assert_no_fallback_dataset(store, case, truth)
    assertions += A_history.assert_restatement(truth, case)
    last = truth["reporting_dates"][-1]
    if reference.get(last) is not None:
        assertions += A_history.assert_duplicate_period_policy(
            store, reference[last], case, last)
    result.stages.append(_stage(STAGE_HISTORY, started, assertions=assertions,
                                artefacts={"registrations": len(registrations)}))
    return store


def _stage_mi(case, run_dir: Path, result: CaseResult, store, truth):
    started = time.monotonic()
    assertions = A_mi.assert_funded_state(store, case, truth)
    assertions += A_mi.assert_period_movement(store, case, truth)
    assertions += A_mi.assert_trend(store, case, truth)
    assertions += A_mi.assert_concentration(store, case, truth)
    assertions += A_mi.assert_cohorts(store, case, truth)
    assertions += A_mi.assert_asset_measure(store, case, truth)
    assertions += A_mi.assert_runtime_boundary(store, case, truth)
    (run_dir / "mi_results").mkdir(parents=True, exist_ok=True)
    (run_dir / "mi_results" / "assertions.json").write_text(
        json.dumps([a.to_dict() for a in assertions], indent=2, default=str) + "\n",
        encoding="utf-8")
    result.stages.append(_stage(STAGE_MI, started, assertions=assertions))


def _stage_risk(case, run_dir: Path, result: CaseResult, store, truth):
    started = time.monotonic()
    evaluation = A_risk.evaluate(store, case, truth)
    (run_dir / "risk_results").mkdir(parents=True, exist_ok=True)
    (run_dir / "risk_results" / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, default=str) + "\n", encoding="utf-8")
    assertions = A_risk.assert_risk_state(evaluation, case)
    assertions += A_risk.assert_movement_disclosed(
        evaluation, periods=len(truth["reporting_dates"]))
    if case.is_multi_source:
        assertions += A_risk.assert_provenance_scoped_evaluation(
            store, case, truth)
    result.stages.append(_stage(
        STAGE_RISK, started, assertions=assertions,
        artefacts={"overall_status": (evaluation.get("summary") or {})
                   .get("overallStatus"),
                   "active_tests": (evaluation.get("summary") or {})
                   .get("activeTests")}))


def _stage_agent(case, run_dir: Path, result: CaseResult, store, truth,
                 canonicals):
    from .assertions import agent as A_agent

    started = time.monotonic()
    out_dir = run_dir / "agent_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    reporting_date = truth["reporting_dates"][-1]
    env = P.mi_environment(run_dir, case, reporting_date=reporting_date)

    assertions: List[Assertion] = []
    transcript: List[Dict[str, Any]] = []
    with A_agent.governed_environment(env):
        for spec in A_agent.question_suite(case.asset_class):
            governed = A_agent.ask(case, spec["question"])
            assertions += A_agent.assert_answer(governed, spec, case, truth,
                                                reporting_date=reporting_date)
            if spec["expects"] != "refusal":
                assertions += A_mi.assert_channel_parity(governed)
            transcript.append({
                "id": spec["id"], "question": spec["question"],
                "status": governed.status,
                "ok": (governed.result or {}).get("ok"),
                "answer": str((governed.result or {}).get("answer"))[:400],
                "evidence": A_agent.numeric_evidence(governed.result or {})[:20],
                "ranked": A_agent.ranked_groups(governed.result or {})[:12],
                "snapshot": {
                    "reporting_date": getattr(governed.snapshot, "reporting_date", None),
                    "dataset_label": getattr(governed.snapshot, "dataset_label", None),
                    "content_hash": getattr(governed.snapshot, "content_hash", None)},
            })
        if case.is_multi_source:
            # The point of a multi-source SPV: total, direct-only and
            # acquired-only must reconcile and must not leak into each other.
            assertions += A_agent.assert_provenance_scopes(case, truth)
    (out_dir / "transcript.json").write_text(
        json.dumps(transcript, indent=2, default=str) + "\n", encoding="utf-8")
    result.stages.append(_stage(STAGE_AGENT, started, assertions=assertions,
                                artefacts={"questions": len(transcript)}))


def _stage_regime(case, run_dir: Path, result: CaseResult, canonicals, truth):
    started = time.monotonic()
    out_dir = run_dir / "regime_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if case.regime.applicability != REGIME_APPLICABLE:
        result.stages.append(StageResult(
            stage=STAGE_REGIME, ok=True, skipped=True,
            skip_reason=f"{case.regime.applicability}: {case.regime.reason}",
            duration_seconds=time.monotonic() - started,
            assertions=A_regime.assert_skipped(case)))
        return

    reference = canonicals.get(REFERENCE_DIALECT) or {}
    date = truth["reporting_dates"][-1]
    source = _source_for(run_dir, case, date)
    if source is None or date not in reference:
        result.stages.append(StageResult(
            stage=STAGE_REGIME, ok=False,
            failure_class=FailureClass.REGULATORY_ARTEFACT_FAILURE,
            message="no canonical source available for the regime run",
            duration_seconds=time.monotonic() - started))
        return

    regime = case.regime.regime_id
    config = client_config(case)
    gate = P.run_regulatory_gates(source, case, regime=regime,
                                  reporting_date=date, out_dir=out_dir / "run",
                                  validation_dir=out_dir / "validation",
                                  master_config=config)
    projected = P.projected_regime_csv(out_dir / "run", regime)
    xml_path = out_dir / "run" / f"{regime.replace('ESMA_', '').lower()}_final.xml"

    assertions: List[Assertion] = []
    if projected is None:
        assertions.append(A_regime.check(
            STAGE_REGIME, f"{regime}: Gate 4 produced a projection", False,
            actual=A_regime._tail(f"{gate.stdout}\n{gate.stderr}"),
            failure_class=FailureClass.REGULATORY_ARTEFACT_FAILURE))
    else:
        assertions += A_regime.assert_projection(projected, reference[date], case,
                                                 truth["periods"][-1])
        P.run_regulatory_gates(source, case, regime=regime,
                               reporting_date=date,
                               out_dir=out_dir / "rerun",
                               validation_dir=out_dir / "rerun_validation",
                               master_config=config)
        reprojected = P.projected_regime_csv(out_dir / "rerun", regime)
        if reprojected is not None:
            assertions += A_regime.assert_reproducible(projected, reprojected, regime)
    assertions += A_regime.assert_artefact(case, gate, xml_path)
    result.stages.append(_stage(STAGE_REGIME, started, assertions=assertions,
                                artefacts={"projected": str(projected),
                                           "xml": str(xml_path)}))


def _source_for(run_dir: Path, case, date: str) -> Optional[Path]:
    """The input the regulatory run starts from for this reporting date.

    A single-source case submits its own delivered file. A multi-source SPV
    submits the ASSEMBLED platform canonical — the regulatory tape for an SPV
    is the whole SPV, not whichever delivery happened to be listed first.
    """
    if case.is_multi_source:
        assembled = (run_dir / "platform_assembly" / REFERENCE_DIALECT / date
                     / P.PLATFORM_CANONICAL_NAME)
        return assembled if assembled.exists() else None
    folder = (run_dir / "generated_sources" / REFERENCE_DIALECT
              / case.portfolio_id / date)
    if not folder.exists():
        return None
    files = sorted(p for p in folder.iterdir() if p.is_file())
    return files[0] if files else None


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #
def run_profile(profile: str, *, run_root: Path, jobs: int = 4,
                cases: Optional[Sequence[str]] = None,
                timeout: Optional[float] = None,
                stages: Optional[Sequence[str]] = None,
                progress: bool = True) -> Dict[str, Any]:
    """Run a profile, optionally under a wall-clock budget.

    ``timeout`` is the budget for the WHOLE profile, in seconds. It is enforced
    between stages, so a run that runs out of time stops cleanly with every
    completed stage's timings and evidence persisted, and reports
    ``PARTIAL_PERFORMANCE`` rather than hanging or discarding what it measured.
    """
    selected = ([M.get_case(c) for c in cases] if cases
                else M.cases_for_profile(profile))
    specs = _profile_specs(profile, selected, stages)
    started = time.monotonic()
    deadline = (started + float(timeout)) if timeout else None
    results: List[CaseResult] = []
    not_run: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}
    for case, spec in zip(selected, specs):
        run_case_manifest = _profile_manifest(profile, case, spec)
        scale = spec["scale"]
        # The same case may legitimately appear twice in one profile at two
        # scales; the second run gets a scale-suffixed evidence directory.
        seen[case.case_id] = seen.get(case.case_id, 0) + 1
        label = (case.case_id if seen[case.case_id] == 1
                 else f"{case.case_id}__{scale}")
        if deadline is not None and time.monotonic() >= deadline:
            # Not started at all: recorded as not run, never as passed.
            not_run.append({"run_label": label, "case_id": case.case_id,
                            "scale": scale, "reason": "timeout"})
            print(f"  [{profile}] {label} NOT RUN (budget expired)", flush=True)
            continue
        print(f"  [{profile}] {case.case_id} ({scale}) "
              f"stages={','.join(spec['stages']) if spec.get('stages') else 'all'}",
              flush=True)
        results.append(run_case(
            run_case_manifest, run_root=run_root, scale=scale, jobs=jobs,
            run_label=label, stages=spec.get("stages"), deadline=deadline,
            progress=progress,
            measure_memory=(profile == M.PROFILE_PERFORMANCE)))
    summary = _summarise(profile, results, time.monotonic() - started, run_root,
                         list(selected))
    summary["not_run"] = not_run
    summary["timeout_seconds"] = timeout
    partial = bool(not_run) or any(r.timed_out for r in results)
    summary["status"] = "PARTIAL_PERFORMANCE" if partial else "COMPLETE"
    if partial:
        print(f"  [{profile}] PARTIAL_PERFORMANCE: budget of {timeout}s expired; "
              f"{len(not_run)} run(s) not started, "
              f"{sum(1 for r in results if r.timed_out)} truncated. "
              f"All completed timings and evidence are persisted.", flush=True)
    Path(run_root).mkdir(parents=True, exist_ok=True)
    (Path(run_root) / f"run_summary_{profile}.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


def _coverage(results: List[CaseResult], cases: List[CaseManifest]) -> Dict[str, Any]:
    """What the run actually exercised, counted from the evidence on disk.

    Reported so a reader can see the SHAPE of the coverage rather than a raw
    file count: how many economic cases, how many reporting periods, how many
    source files, how many canonical outputs and how many assertions of each
    kind.
    """
    run_dirs = [Path(r.run_dir) for r in results if r.run_dir]
    sources = sum(len([p for p in (d / "generated_sources").rglob("*")
                       if p.is_file()]) for d in run_dirs if d.exists())
    canonicals = sum(len(list((d / "canonical_outputs").rglob("*_canonical_typed.csv")))
                     for d in run_dirs if d.exists())
    by_stage: Dict[str, int] = {}
    for r in results:
        for stage in r.stages:
            by_stage[stage.stage] = by_stage.get(stage.stage, 0) + len(stage.assertions)
    return {
        "economic_cases": sum(1 for c in cases if c.expectation != EXPECT_FAILURE),
        "guard_cases": sum(1 for c in cases if c.expectation == EXPECT_FAILURE),
        "asset_classes": sorted({c.asset_class for c in cases}),
        "reporting_periods": sum(c.months for c in cases),
        "dialect_renderings": sum(c.months * len(c.dialects) for c in cases),
        "generated_source_files": sources,
        "canonical_outputs": canonicals,
        "assertions_by_stage": by_stage,
    }


def _profile_manifest(profile: str, case: CaseManifest,
                      spec: Dict[str, Any]) -> CaseManifest:
    """The manifest shape a profile runs a case at."""
    if profile == M.PROFILE_SMOKE:
        # Fast enough for ordinary CI: three periods, one dialect, small scale.
        return replace(case, months=3, dialects=(REFERENCE_DIALECT,))
    if profile == M.PROFILE_PERFORMANCE:
        # One dialect: the profile measures the pathway, not the renderers.
        return replace(case, dialects=(REFERENCE_DIALECT,),
                       months=int(spec.get("months") or case.months))
    return case


def _profile_specs(profile: str, selected: List[CaseManifest],
                   stages: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    """The run spec for each selected case, positionally.

    Only the performance profile varies scale or narrows stages, and it does so
    from the explicit ``PERFORMANCE_RUNS`` schedule rather than by inferring
    anything from a case id. An explicit ``--case`` selection falls back to the
    standard scale so a hand-picked performance run still measures something
    meaningful. An explicit ``--stages`` always wins: that is how the expensive
    regime and Agent measurements are asked for at scale.
    """
    override = tuple(stages) if stages else None
    if profile != M.PROFILE_PERFORMANCE:
        return [{"scale": "small", "stages": override} for _ in selected]
    scheduled = list(M.PERFORMANCE_RUNS)
    if [s["case_id"] for s in scheduled] == [c.case_id for c in selected]:
        return [{"scale": s["scale"], "months": s.get("months"),
                 "stages": override or s.get("stages")} for s in scheduled]
    return [{"scale": "standard", "stages": override} for _ in selected]


def _summarise(profile: str, results: List[CaseResult], elapsed: float,
               run_root: Path, cases: List[CaseManifest]) -> Dict[str, Any]:
    expected_failures = [r for r in results
                         if r.expectation == EXPECT_FAILURE and r.ok]
    unexpected = [r for r in results if not r.ok]
    assertions = {"total": 0, "passed": 0, "failed": 0}
    for r in results:
        counts = r.assertion_counts()
        for key in assertions:
            assertions[key] += counts[key]
    stages_passed: Dict[str, int] = {}
    stages_skipped: Dict[str, int] = {}
    for r in results:
        for s in r.stages:
            target = stages_skipped if s.skipped else (
                stages_passed if s.ok else None)
            if target is not None:
                target[s.stage] = target.get(s.stage, 0) + 1
    coverage = _coverage(results, cases)
    return {
        "profile": profile,
        "simulation_version": SIMULATION_VERSION,
        "cases_run": len(results),
        "cases_ok": sum(1 for r in results if r.ok),
        "expected_failures": [r.case_id for r in expected_failures],
        "unexpected_failures": [
            {"case_id": r.case_id, "seed": r.seed,
             "failure_class": r.observed_failure_class,
             "stages": [s.to_dict() for s in r.failures]} for r in unexpected],
        "coverage": coverage,
        "assertions": assertions,
        "stages_passed": stages_passed,
        "stages_skipped": stages_skipped,
        "elapsed_seconds": round(elapsed, 2),
        # Keyed by RUN LABEL, not case id: one profile may run the same case
        # twice at two scales, and keying by case id would silently drop one of
        # the two measurements.
        "reproducibility": {
            "seeds": {r.run_label: r.seed for r in results},
            "manifest_hashes": {r.run_label: r.manifest_hash for r in results},
            "reproduce": {
                r.run_label: (f"python -m simulation.runner reproduce --case "
                              f"{r.case_id} --seed {r.seed}"
                              + (f" --scale {r.scale}" if r.scale != "small"
                                 else "")) for r in results},
        },
        "production_modules": sorted({m for r in results
                                      for m in r.production_modules}),
        "scales": {r.run_label: r.scale for r in results},
        "stages_selected": {r.run_label: list(r.selected_stages) for r in results},
        "timed_out": [r.run_label for r in results if r.timed_out],
        "skip_reasons": {
            r.run_label: {s.stage: s.skip_reason for s in r.stages if s.skipped}
            for r in results if any(s.skipped for s in r.stages)},
        "timings": {r.run_label: r.timings for r in results},
        "elapsed_seconds_by_run": {r.run_label: round(sum(r.timings.values()), 2)
                                   for r in results},
        "peak_memory_mb": {r.run_label: r.peak_memory_mb for r in results
                           if r.peak_memory_mb is not None},
        "run_root": str(run_root),
        "cases": [r.to_dict() for r in results],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_case_result(result: CaseResult) -> None:
    counts = result.assertion_counts()
    verdict = "OK" if result.ok else "FAILED"
    print(f"\n{result.case_id}: {verdict}  "
          f"({counts['passed']}/{counts['total']} assertions)")
    for stage in result.stages:
        mark = "skip" if stage.skipped else ("ok" if stage.ok else "FAIL")
        detail = stage.skip_reason if stage.skipped else stage.message
        print(f"  {stage.stage:22s} {mark:5s} {stage.duration_seconds:7.2f}s "
              f"{detail[:90]}")
        for failure in stage.failed_assertions[:6]:
            print(f"      - {failure.name}: expected={failure.expected!r} "
                  f"actual={failure.actual!r}")
    print(f"  evidence: {result.run_dir}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m simulation.runner",
        description="Deterministic asset-class hardening framework for Trakt.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list the case catalogue")

    gen = sub.add_parser("generate", help="generate a case's sources only")
    gen.add_argument("--case", required=True)
    gen.add_argument("--scale", choices=sorted(SCALES), default="small")
    gen.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))

    one = sub.add_parser("run", help="run one case end to end")
    one.add_argument("--case", required=True)
    one.add_argument("--scale", choices=sorted(SCALES), default="small")
    one.add_argument("--jobs", type=int, default=4)
    one.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))

    every = sub.add_parser("run-all", help="run a profile")
    every.add_argument("--profile", choices=M.PROFILES, default=M.PROFILE_SMOKE)
    every.add_argument("--jobs", type=int, default=4)
    every.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    every.add_argument("--case", action="append", dest="cases",
                       help="restrict the profile to specific case ids")
    every.add_argument("--timeout", type=float, default=None,
                       help="wall-clock budget for the whole profile, in "
                            "seconds. Enforced between stages: the run stops "
                            "cleanly and reports PARTIAL_PERFORMANCE with every "
                            "completed timing and artefact kept.")
    every.add_argument("--stages", default=None,
                       help="comma-separated stages to run (prerequisites are "
                            f"added automatically). Known: {','.join(STAGES)}. "
                            "This is how the expensive regime and agent stages "
                            "are asked for at large scale.")
    every.add_argument("--quiet", action="store_true",
                       help="suppress live per-stage progress")

    repro = sub.add_parser("reproduce",
                           help="re-run a case at an explicit seed")
    repro.add_argument("--case", required=True)
    repro.add_argument("--seed", type=int, required=True)
    repro.add_argument("--scale", choices=sorted(SCALES), default="small")
    repro.add_argument("--jobs", type=int, default=4)
    repro.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))

    args = parser.parse_args(argv)

    if args.command == "list":
        for case in M.all_cases():
            print(f"{case.case_id:46s} {case.asset_class:15s} seed={case.seed} "
                  f"months={case.months} scale-base={case.opening_loan_count} "
                  f"profiles={','.join(case.profiles)}")
            print(f"    {case.purpose}")
        return 0

    if args.command == "generate":
        case = M.get_case(args.case)
        history = generate_history(case,
                                   loan_count=_scaled_loan_count(case, args.scale))
        truth = build_expected_truth(case, history)
        run_dir = Path(args.run_root) / case.case_id
        (run_dir / "generated_sources").mkdir(parents=True, exist_ok=True)
        M.write_manifest(case, run_dir / "manifest.json")
        write_expected_truth(truth, run_dir / "expected_truth" / "expected_truth.json")
        result = CaseResult(case_id=case.case_id, seed=case.seed,
                            asset_class=case.asset_class,
                            simulation_version=case.simulation_version,
                            expectation=case.expectation, run_dir=str(run_dir))
        _stage_render(case, run_dir, result, history)
        print(f"generated {case.case_id} -> {run_dir}")
        return 0

    if args.command in ("run", "reproduce"):
        case = M.get_case(args.case)
        if args.command == "reproduce":
            case = replace(case, seed=int(args.seed))
        # A non-default scale gets its own evidence directory, so measuring
        # one case at two scales never overwrites the other's evidence.
        label = (case.case_id if args.scale == "small"
                 else f"{case.case_id}__{args.scale}")
        result = run_case(case, run_root=Path(args.run_root), scale=args.scale,
                          jobs=args.jobs, run_label=label)
        _print_case_result(result)
        return 0 if result.ok else 1

    stage_filter = ([s.strip() for s in args.stages.split(",") if s.strip()]
                    if getattr(args, "stages", None) else None)
    summary = run_profile(args.profile, run_root=Path(args.run_root),
                          jobs=args.jobs, cases=getattr(args, "cases", None),
                          timeout=getattr(args, "timeout", None),
                          stages=stage_filter,
                          progress=not getattr(args, "quiet", False))
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"},
                     indent=2, default=str))
    if summary["unexpected_failures"]:
        return 1
    # A partial run is not a pass and must not be reported as one, but it is
    # also not a failure of the platform: exit 2 says "measured, incompletely".
    return 2 if summary["status"] == "PARTIAL_PERFORMANCE" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
