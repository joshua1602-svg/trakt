"""
engine.annex_delivery_agent.preflight
=====================================

Cheap, blocking checks that run *before* an expensive artefact build.

Two rules govern this module:

1. **Fail before the expensive thing.** An Annex 2 build reads a 9.6 MB mapping
   workbook and can write a 200 MB file. Discovering a missing LEI at that point
   costs three minutes and a confusing traceback.
2. **Say what is wrong in the operator's language.** A preflight failure is
   read by someone who has to *fix* it, so it names the missing thing and the
   action, not the Python object that happened to be ``None``. Raw exception
   text is retained as ``technical_detail`` for engineers, never as the message.

The checks here are annex-neutral. Annex-specific checks are contributed by the
provider through :meth:`~engine.annex_delivery_agent.providers.base.AnnexProvider.preflight`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

SEVERITY_BLOCKING = "blocking"
SEVERITY_WARNING = "warning"


@dataclass(frozen=True)
class PreflightFinding:
    """One preflight observation.

    ``message`` is the operator sentence. It must be a complete, actionable
    statement — "Annex 2 cannot be prepared because …" — not a symbol name.
    """

    check: str
    severity: str
    message: str
    remedy: str = ""
    technical_detail: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return self.severity == SEVERITY_BLOCKING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "remedy": self.remedy,
            "technical_detail": self.technical_detail,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PreflightFinding":
        d = dict(data)
        return cls(check=str(d.get("check", "")), severity=str(d.get("severity", SEVERITY_BLOCKING)),
                   message=str(d.get("message", "")), remedy=str(d.get("remedy", "")),
                   technical_detail=str(d.get("technical_detail", "")),
                   details=dict(d.get("details") or {}))


@dataclass
class PreflightReport:
    """The collected findings for one run."""

    findings: List[PreflightFinding] = field(default_factory=list)

    def add(self, finding: PreflightFinding) -> None:
        self.findings.append(finding)

    def block(self, check: str, message: str, *, remedy: str = "",
              technical_detail: str = "", **details: Any) -> None:
        self.add(PreflightFinding(check=check, severity=SEVERITY_BLOCKING,
                                  message=message, remedy=remedy,
                                  technical_detail=technical_detail, details=details))

    def warn(self, check: str, message: str, *, remedy: str = "", **details: Any) -> None:
        self.add(PreflightFinding(check=check, severity=SEVERITY_WARNING,
                                  message=message, remedy=remedy, details=details))

    def extend(self, findings: Iterable[PreflightFinding]) -> None:
        self.findings.extend(findings)

    @property
    def blockers(self) -> List[PreflightFinding]:
        return [f for f in self.findings if f.blocking]

    @property
    def warnings(self) -> List[PreflightFinding]:
        return [f for f in self.findings if not f.blocking]

    @property
    def passed(self) -> bool:
        return not self.blockers

    def operator_message(self) -> str:
        """One sentence per blocker, in the order they were found."""
        if self.passed:
            return "All preflight checks passed."
        return " ".join(f.message for f in self.blockers)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "blocker_count": len(self.blockers),
            "warning_count": len(self.warnings),
            "operator_message": self.operator_message(),
            "findings": [f.to_dict() for f in self.findings],
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "PreflightReport":
        d = dict(data or {})
        return cls(findings=[PreflightFinding.from_dict(f) for f in (d.get("findings") or ())])


# --------------------------------------------------------------------------- #
# Reusable checks — annex-neutral
# --------------------------------------------------------------------------- #


def require_file(report: PreflightReport, *, check: str, path: Optional[str],
                 what: str, annex_name: str, remedy: str = "") -> Optional[Path]:
    """Blocking check that a required file exists and is readable."""
    if not path:
        report.block(check,
                     f"{annex_name} cannot be prepared because {what} has not been "
                     f"configured.",
                     remedy=remedy or f"Set the location of {what} in the annex configuration.")
        return None
    p = Path(path)
    if not p.is_file():
        report.block(check,
                     f"{annex_name} cannot be prepared because {what} was not found "
                     f"at the configured location.",
                     remedy=remedy or f"Restore {what} or correct its configured path.",
                     technical_detail=f"missing path: {p}",
                     path=str(p))
        return None
    if not os.access(p, os.R_OK):
        report.block(check,
                     f"{annex_name} cannot be prepared because {what} exists but "
                     f"cannot be read by the service.",
                     remedy=remedy or "Grant the service read access to the file.",
                     technical_detail=f"unreadable path: {p}",
                     path=str(p))
        return None
    return p


def require_writable_directory(report: PreflightReport, *, check: str,
                               path: Optional[str], annex_name: str) -> Optional[Path]:
    """Blocking check that the output location exists (or can be made) and is writable."""
    if not path:
        report.block(check,
                     f"{annex_name} cannot be prepared because no output location "
                     f"was supplied for the report.",
                     remedy="Supply an output location for the delivery run.")
        return None
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        report.block(check,
                     f"{annex_name} cannot be prepared because the output location "
                     f"could not be created.",
                     remedy="Choose an output location the service can write to.",
                     technical_detail=f"{type(exc).__name__}: {exc}",
                     path=str(p))
        return None
    if not os.access(p, os.W_OK):
        report.block(check,
                     f"{annex_name} cannot be prepared because the output location "
                     f"is not writable by the service.",
                     remedy="Grant the service write access to the output location.",
                     path=str(p))
        return None
    return p


def require_reporting_date(report: PreflightReport, *, reporting_date: Optional[str],
                           annex_name: str) -> None:
    """Blocking check on the reporting date, including its shape."""
    value = str(reporting_date or "").strip()
    if not value:
        report.block("reporting_date",
                     f"{annex_name} cannot be prepared because no reporting date "
                     f"was supplied for this submission.",
                     remedy="Supply the reporting reference date (YYYY-MM-DD).")
        return
    parts = value.split("-")
    ok = (len(parts) == 3 and len(parts[0]) == 4
          and all(p.isdigit() for p in parts))
    if not ok:
        report.block("reporting_date",
                     f"{annex_name} cannot be prepared because the reporting date "
                     f"'{value}' is not a valid calendar date.",
                     remedy="Supply the reporting reference date as YYYY-MM-DD.",
                     reporting_date=value)


def require_non_empty_input(report: PreflightReport, *, row_count: Optional[int],
                            annex_name: str) -> None:
    """Blocking check that the projected input actually carries exposures."""
    if row_count is None:
        return
    if row_count <= 0:
        report.block("projected_records",
                     f"{annex_name} cannot be prepared because the projected data "
                     f"for this portfolio and period contains no exposure records.",
                     remedy="Re-run the projection stage for this reporting date, "
                            "then prepare the report again.",
                     row_count=row_count)


def translate_exception(exc: BaseException, *, annex_name: str, stage: str) -> PreflightFinding:
    """Turn an unexpected component failure into an operator-readable blocker.

    The exception text is preserved verbatim under ``technical_detail`` — this
    is a *translation*, not a suppression.
    """
    stage_phrase = {
        "normalise": "while preparing the delivery data",
        "build": "while creating the report file",
        "validate": "while checking the report against the official schema",
    }.get(stage, f"during the {stage} stage")
    return PreflightFinding(
        check=f"{stage}_failure",
        severity=SEVERITY_BLOCKING,
        message=(f"{annex_name} could not be prepared: the process stopped "
                 f"{stage_phrase}. No report file has been produced."),
        remedy="Send the technical detail below to the Trakt engineering team; "
               "no data has been published.",
        technical_detail=f"{type(exc).__name__}: {exc}",
        details={"stage": stage},
    )


__all__ = [
    "SEVERITY_BLOCKING", "SEVERITY_WARNING",
    "PreflightFinding", "PreflightReport",
    "require_file", "require_writable_directory", "require_reporting_date",
    "require_non_empty_input", "translate_exception",
]
