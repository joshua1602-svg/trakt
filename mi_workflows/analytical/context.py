"""mi_workflows.analytical.context — everything an executor may read.

One object, resolved lazily and memoised, so a plan that needs the funded frame
twice and the pipeline once pays for each exactly once and a plan that needs
neither touches no storage at all.

Every resolution here delegates to the service the rest of the platform already
uses. Nothing in this module discovers a dataset, prepares a frame or decides a
reporting date for itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger("mi_workflows.analytical.context")


@dataclass
class AnalyticalContext:
    """The governed execution context for one analytical plan."""

    question: str
    spec: Any
    spec_dict: Dict[str, Any]
    semantics: Mapping[str, Any]
    client_id: str
    run_id: Optional[str] = None
    output_root: Optional[str] = None
    pipeline_root: Optional[str] = None
    portfolio_id: Optional[str] = None
    as_of: Optional[str] = None
    view: str = "funded"
    #: The RESOLVED portfolio lens for this request (``chat_routing._resolve_lens``).
    lens: Any = None
    #: ``f(client_id, run_id) -> DataFrame | None``. Supplied by ``mi_service``;
    #: it applies the governed row population (P1L) and records the evidence the
    #: execution receipt reconciles against.
    frame_resolver: Optional[Callable[[str, Optional[str]], Any]] = None
    #: ``f(client_id, run_id) -> DataFrame | None`` for the governed funded frame
    #: BEFORE the request's row population is applied. This layer plans its own
    #: populations — including both sides of a governed partition — so it must
    #: be able to address the book they partition. Falls back to
    #: ``frame_resolver`` when a caller supplies only that.
    base_frame_resolver: Optional[Callable[[str, Optional[str]], Any]] = None

    #: Populated by executors: warnings the answer must carry.
    warnings: List[str] = field(default_factory=list)
    #: Populated by executors: source notes for the answer's provenance block.
    source_notes: List[Dict[str, Any]] = field(default_factory=list)

    _memo: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    # -- funded ------------------------------------------------------------ #
    def funded_frame(self):
        """The current governed funded frame, population already applied.

        Resolved through ``frame_resolver`` so the population narrowing and its
        evidence are produced by the SAME seam every other route uses — this
        layer never re-interprets ``spec.filters`` for itself.
        """
        if "funded_frame" not in self._memo:
            frame = None
            if self.frame_resolver is not None:
                try:
                    frame = self.frame_resolver(self.client_id, self.run_id)
                except Exception as exc:  # noqa: BLE001 - a frame fault is a gap
                    logger.warning("funded frame resolution failed: %s", exc)
                    frame = None
            self._memo["funded_frame"] = frame
        return self._memo["funded_frame"]

    def base_frame(self):
        """The governed funded book before any row population is applied.

        The frame every population in a plan is measured against. Resolving the
        primary population through :meth:`funded_frame` is still done exactly
        once per request, because that is what produces the shared P1L evidence
        the execution receipt reconciles against — but it is not what the
        populations are computed from.
        """
        if "base_frame" not in self._memo:
            frame = None
            resolver = self.base_frame_resolver or self.frame_resolver
            if resolver is not None:
                try:
                    frame = resolver(self.client_id, self.run_id)
                except Exception as exc:  # noqa: BLE001 - a frame fault is a gap
                    logger.warning("base frame resolution failed: %s", exc)
                    frame = None
            self._memo["base_frame"] = frame
        return self._memo["base_frame"]

    def snapshots(self) -> Tuple[Any, ...]:
        """Every governed funded snapshot for this client and lens, oldest first.

        Delegates to ``period_change_route.build_snapshots`` — the existing
        governed per-period frame service — so a dated frame here is the same
        dated frame the Period Change workflow sees.
        """
        if "snapshots" not in self._memo:
            try:
                from mi_agent_api import period_change_route as pcr
                snaps = pcr.build_snapshots(self.output_root, self.client_id,
                                            to_run_id=self.run_id, lens=self.lens)
            except Exception as exc:  # noqa: BLE001
                logger.warning("snapshot resolution failed: %s", exc)
                snaps = ()
            self._memo["snapshots"] = tuple(snaps)
        return self._memo["snapshots"]

    # -- pipeline ---------------------------------------------------------- #
    def pipeline(self) -> Tuple[Any, Optional[Dict[str, Any]], Dict[str, Any]]:
        """``(frame, report, meta)`` for the current governed pipeline extract.

        The SAME resolution the forecast and concentration views use: the latest
        governed weekly extract, prepared with the historical completion model
        (empirical stage rates where the observed sample clears the sufficiency
        floor, configured fallback below it). ``(None, None, meta)`` with a
        ``reason`` whenever no governed pipeline source exists — never a guess
        and never a zero.
        """
        if "pipeline" not in self._memo:
            self._memo["pipeline"] = self._resolve_pipeline()
        return self._memo["pipeline"]

    def _resolve_pipeline(self):
        meta: Dict[str, Any] = {"available": False}
        try:
            from mi_agent_api import datasets as datasets_mod
            from mi_agent_api import pipeline_contract as pipeline_mod

            source = datasets_mod._resolve_pipeline_source(self.client_id,
                                                           self.run_id)
            if source is None:
                meta["reason"] = ("No governed pipeline source is configured for "
                                  "this book.")
                return None, None, meta
            model = datasets_mod._pipeline_history(self.client_id)
            frame, report = pipeline_mod.load_prepared_pipeline(
                source, historical_model=model)
            meta = {
                "available": True,
                "source": source,
                "basis": (report or {}).get("completion_probability_basis"),
                "asOf": (source.get("pipeline_as_of_date")
                         if isinstance(source, dict) else None),
                "sourceFile": (source.get("source_file")
                               if isinstance(source, dict) else str(source)),
                "weeklyExtractsUsed": (model or {}).get("uniqueWeeklyExtractsUsed"),
            }
            return frame, report, meta
        except Exception as exc:  # noqa: BLE001 - pipeline resolution fails closed
            logger.warning("pipeline resolution failed: %s", exc)
            meta["reason"] = f"The governed pipeline could not be prepared: {exc}"
            return None, None, meta

    def pipeline_snapshot(self) -> Optional[Dict[str, Any]]:
        """The governed pipeline snapshot block, or ``None``."""
        if "pipeline_snapshot" not in self._memo:
            frame, report, meta = self.pipeline()
            snapshot = None
            if frame is not None and report is not None:
                try:
                    from mi_agent_api import pipeline_contract as pipeline_mod
                    snapshot = pipeline_mod.compute_pipeline_snapshot(
                        frame, report, dict(self.semantics),
                        client_id=self.client_id, run_id=str(self.run_id or ""),
                        source=meta.get("source") if isinstance(
                            meta.get("source"), dict) else None)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("pipeline snapshot failed: %s", exc)
                    snapshot = None
            self._memo["pipeline_snapshot"] = snapshot
        return self._memo["pipeline_snapshot"]

    # -- registries -------------------------------------------------------- #
    def business_semantics(self):
        """The Business Semantics Registry (measure aggregation / weighting)."""
        if "bsr" not in self._memo:
            from mi_workflows.semantics import load_business_semantics
            try:
                self._memo["bsr"] = load_business_semantics()
            except Exception as exc:  # noqa: BLE001
                logger.warning("business semantics unavailable: %s", exc)
                self._memo["bsr"] = None
        return self._memo["bsr"]

    def memoised(self, key: str, build):
        """``build()``'s result, computed at most once per request."""
        if key not in self._memo:
            self._memo[key] = build()
        return self._memo[key]

    def warn(self, message: str) -> None:
        if message and message not in self.warnings:
            self.warnings.append(message)

    def note(self, label: str, note: str) -> None:
        entry = {"label": label, "note": note}
        if entry not in self.source_notes:
            self.source_notes.append(entry)
