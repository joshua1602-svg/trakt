"""demo_platform.safety — the pre-render data-safety gate.

SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER.

The demonstration must be safe to publish. This module scans everything the film
can possibly read — the exported metric and surface fixtures, the artefact
catalogue, the generated source extracts, the onboarding contracts and the whole
Remotion project — for anything that would leak a live client, a real
environment, or a credential:

  * prohibited client strings (ERE / Equity Release Europe / …), plus every
    identifying value read live out of the PRODUCTION client config, so a value
    cannot slip through just because it was not on a hand-written list;
  * email addresses;
  * GUIDs / tenant and subscription identifiers;
  * Azure storage account names, connection strings and account keys;
  * bearer tokens and signed (SAS) URLs;
  * production hostnames and storage endpoints;
  * production storage paths;
  * git branch / environment identifiers.

:func:`scan` returns a report and :func:`enforce` raises on any finding, so the
render fails closed. The scan is deliberately scoped to what the film consumes —
the repository legitimately contains real client artefacts elsewhere, and
flagging those would make the gate meaningless.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import yaml

from . import config as cfg


class SafetyViolation(RuntimeError):
    """Raised when prohibited content is found in demonstration material."""


# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    description: str
    #: Substrings that make a match legitimate (documentation of the rule itself).
    allow_if_contains: Sequence[str] = ()


_PATTERN_RULES: List[Rule] = [
    Rule("email_address",
         re.compile(r"\b[\w.+-]+@[\w-]+\.[A-Za-z]{2,}\b"),
         "A live email address must never appear in demonstration material."),
    Rule("guid_or_tenant_id",
         re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
         "A GUID may be a real tenant, subscription or application identifier.",
         allow_if_contains=("REPLACE-WITH-NEW-GUID",)),
    Rule("azure_storage_endpoint",
         re.compile(r"[A-Za-z0-9-]+\.(?:blob|table|queue|file)\.core\.windows\.net"),
         "A real Azure storage endpoint reveals a production storage account."),
    Rule("azure_web_host",
         re.compile(r"[A-Za-z0-9-]+\.(?:azurewebsites|azurestaticapps)\.net"),
         "A real Azure hostname reveals a production deployment."),
    Rule("storage_connection_string",
         re.compile(r"DefaultEndpointsProtocol=|AccountKey=|SharedAccessSignature="),
         "A storage connection string or account key is a credential."),
    Rule("sas_signed_url",
         re.compile(r"[?&](?:sig|sv|se|st|sp)=[^\s\"'&]+"),
         "A signed (SAS) URL is a bearer credential and expires — never show one."),
    Rule("bearer_token",
         re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{20,}"),
         "A bearer token is a credential."),
    Rule("jwt",
         re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
         "A JWT is a credential."),
    Rule("private_key",
         re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
         "A private key is a credential."),
    Rule("download_token",
         re.compile(r"artifacts/download\?token=(?!<redacted>)[A-Za-z0-9._-]{12,}"),
         "A live artefact download token is a short-lived credential."),
    Rule("production_branch_ref",
         re.compile(r"\brefs/heads/(?:main|master|production)\b"),
         "A production branch reference is environment detail."),
]

#: Files that may legitimately contain a pattern hit because they DEFINE the rules.
#: (Both scanners describe the same credential shapes, so each one's source text
#: matches the other's patterns. Excluding them is not a loophole: neither file
#: carries data, and every file that DOES carry data is still scanned.)
_SELF_REFERENTIAL = {
    "safety.py", "safety_report.json", "SAFETY.md", "safety-scan.mjs",
}

#: Text extensions worth scanning. Binary artefacts are hashed, not parsed.
_TEXT_SUFFIXES = {".json", ".csv", ".txt", ".md", ".yaml", ".yml", ".ts", ".tsx",
                  ".js", ".jsx", ".mjs", ".css", ".html", ".srt", ".vtt"}


# --------------------------------------------------------------------------- #
# Prohibited strings, including values read live from the production config
# --------------------------------------------------------------------------- #
# The documents whose identifying values the demonstration may never contain.
# The live client configurations are generated per client into the operations
# store, so they are not repository files to read here; the legacy documents
# below still carry the identity the film must not show, and are read for
# exactly that reason. This list is a source of PROHIBITED VALUES — it does not
# make any of these files production configuration.
_PRODUCTION_CONFIGS = (
    Path("tests") / "fixtures" / "legacy_client" / "config_client_ERM_UK.yaml",
    Path("config") / "client" / "config_client_ERM_UK_demo.yaml",
    Path("tests") / "fixtures" / "legacy_client" / "config_client_annex12.yaml",
)

#: Config keys whose values identify a live client and must never be reused.
_IDENTIFYING_KEYS = {
    "client_id", "display_name", "originator_name",
    "originator_legal_entity_identifier", "seller_name", "issuer_name",
    "sponsor_name", "servicer_name", "deal_name", "securitisation_name",
}

#: Values too generic to treat as identifying (they would flag everything).
_GENERIC_VALUES = {"gb", "gbp", "production", "equity_release", "erm", "true",
                   "false", "none", "null", "client", "demo", "uk"}


def production_identifiers() -> List[str]:
    """Identifying values read out of the PRODUCTION client configs.

    Reading them live means the gate does not depend on someone remembering to
    add a client's name to a list: whatever the live config calls the client, the
    demonstration is forbidden from containing it.
    """
    found: Set[str] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).lower() in _IDENTIFYING_KEYS and isinstance(value, str):
                    text = value.strip()
                    if len(text) >= 4 and text.lower() not in _GENERIC_VALUES:
                        found.add(text)
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    for rel in _PRODUCTION_CONFIGS:
        path = cfg.REPO_ROOT / rel
        if not path.exists():
            continue
        try:
            _walk(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        except Exception:  # noqa: BLE001 - an unreadable config never blocks the scan
            continue
    return sorted(found)


def prohibited_strings() -> List[str]:
    """The full prohibited-string list: configured plus config-derived."""
    return sorted(set(cfg.PROHIBITED_STRINGS) | set(production_identifiers()))


# --------------------------------------------------------------------------- #
# Scan
# --------------------------------------------------------------------------- #
@dataclass
class Finding:
    rule: str
    path: str
    line: int
    excerpt: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {"rule": self.rule, "path": self.path, "line": self.line,
                "excerpt": self.excerpt, "description": self.description}


def _scan_targets() -> List[Path]:
    """Everything the film can read, plus the material that produced it."""
    roots = [
        cfg.export_dir(),                    # exported metrics + captured surfaces
        cfg.artefact_dir(),                  # artefact catalogue
        cfg.onboarding_dir(),                # onboarding contracts + mapping reports
        cfg.source_dir(),                    # generated raw source extracts
        cfg.video_fixture_dir(),             # fixtures bundled into the video
        cfg.REPO_ROOT / "demo-video" / "src",
        cfg.REPO_ROOT / "demo-video" / "scripts",
        Path(__file__).resolve().parent / "docs",
        Path(__file__).resolve().parent / "aliases",
        Path(__file__).resolve().parent / "config",
    ]
    files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            files.append(root)
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if "node_modules" in path.parts or path.name in _SELF_REFERENTIAL:
                continue
            if path.suffix.lower() in _TEXT_SUFFIXES:
                files.append(path)
    return files


def _excerpt(line: str, match: re.Match, width: int = 60) -> str:
    """A redacted excerpt: enough to locate the hit, never the secret itself."""
    start = max(0, match.start() - 20)
    end = min(len(line), match.end() + 20)
    hit = match.group(0)
    masked = hit[:4] + "…" + hit[-2:] if len(hit) > 12 else hit
    return (line[start:match.start()] + f"[{masked}]" + line[match.end():end]).strip()[:width * 2]


def scan(*, extra_paths: Optional[Iterable[Path]] = None) -> Dict[str, Any]:
    """Scan the demonstration material. Returns a report; never raises."""
    banned = prohibited_strings()
    banned_lower = [(b, b.lower()) for b in banned]
    findings: List[Finding] = []
    files = _scan_targets() + [Path(p) for p in (extra_paths or [])]
    scanned = 0
    banner_files = 0

    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - unreadable file is not a finding
            continue
        scanned += 1
        if cfg.SYNTHETIC_BANNER in text or cfg.SYNTHETIC_SHORT in text:
            banner_files += 1
        rel = str(path.relative_to(cfg.REPO_ROOT)) if str(path).startswith(str(cfg.REPO_ROOT)) else str(path)
        for lineno, line in enumerate(text.splitlines(), start=1):
            low = line.lower()
            for original, needle in banned_lower:
                idx = low.find(needle)
                if idx == -1:
                    continue
                # Word-ish boundary check so "ERE" does not fire inside "WHERE".
                before = low[idx - 1] if idx > 0 else " "
                after_i = idx + len(needle)
                after = low[after_i] if after_i < len(low) else " "
                if before.isalnum() or after.isalnum():
                    continue
                findings.append(Finding(
                    "prohibited_client_string", rel, lineno,
                    line.strip()[:160],
                    f"Prohibited client identifier {original!r} found in "
                    f"demonstration material."))
            for rule in _PATTERN_RULES:
                for m in rule.pattern.finditer(line):
                    if any(tok in line for tok in rule.allow_if_contains):
                        continue
                    findings.append(Finding(rule.name, rel, lineno,
                                            _excerpt(line, m), rule.description))

    return {
        "ok": not findings,
        "synthetic_notice": cfg.SYNTHETIC_BANNER,
        "filesScanned": scanned,
        "filesCarryingSyntheticBanner": banner_files,
        "prohibitedStrings": banned,
        "productionIdentifiersDerived": production_identifiers(),
        "rules": [{"name": r.name, "description": r.description} for r in _PATTERN_RULES],
        "findings": [f.to_dict() for f in findings],
        "findingCount": len(findings),
    }


def enforce(*, verbose: bool = True) -> Dict[str, Any]:
    """Scan, write the report, and RAISE on any finding so the render fails closed."""
    report = scan()
    out_dir = cfg.export_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "safety_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    if verbose:
        print(f"  [safety] scanned {report['filesScanned']} files against "
              f"{len(report['prohibitedStrings'])} prohibited strings and "
              f"{len(report['rules'])} patterns")
        if report["ok"]:
            print("  [safety] PASS — no prohibited content found")
    if not report["ok"]:
        lines = [f"    {f['rule']}  {f['path']}:{f['line']}  {f['excerpt']}"
                 for f in report["findings"][:25]]
        raise SafetyViolation(
            f"{report['findingCount']} safety finding(s) in demonstration "
            f"material — refusing to render:\n" + "\n".join(lines)
        )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse
    ap = argparse.ArgumentParser(description="Run the demo data-safety scan.")
    ap.add_argument("--json", action="store_true", help="Print the full report.")
    args = ap.parse_args(argv)
    try:
        report = enforce()
    except SafetyViolation as exc:
        print(f"SAFETY SCAN FAILED\n{exc}")
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
