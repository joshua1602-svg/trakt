"""operations_control.language — plain-English translation for operators.

Everything the UI renders passes through here. The rules are contractual
(tested in tests/operations_control/test_language.py):

  * no file paths, blob URIs or container names;
  * no Python exception classes or stack traces;
  * no schema/regime codes, manifest numbers or agent module names;
  * one or two short sentences, business language only.

Technical detail belongs in Governed Agent Result provenance fields, which the
UI never renders.
"""

from __future__ import annotations

import re
from typing import Iterable, List

# Substrings that must never appear in operator-facing text. Deliberately
# lowercase; matching is case-insensitive.
FORBIDDEN_FRAGMENTS = (
    "traceback", "exception", ".py", "blob://", "sha256:", "/home/", "/tmp/",
    "manifest", ".yaml", ".json", ".csv", "annex2_", "esma_annex",
    "kwargs", "stderr", "returncode", "_agent", "handoff", "fingerprint",
    "registry", "canonical", "orchestrat", "trakt-state", "processed-v2",
    "raw-v2", "operations-control",
)

_PATH_RE = re.compile(r"(/[\w.\-]+){2,}|[A-Za-z]:\\\S+|blob://\S+")
_CODE_RE = re.compile(r"\b\d{2}[a-z]?_[a-z_]+\b")   # 24_onboarding_..., 28a_...


def is_operator_safe(text: str) -> bool:
    """True when ``text`` contains none of the forbidden technical vocabulary."""
    low = (text or "").lower()
    if any(f.lower() in low for f in FORBIDDEN_FRAGMENTS):
        return False
    if _PATH_RE.search(text or "") or _CODE_RE.search(text or ""):
        return False
    return True


# Known internal blocker phrases -> operator sentences. Matched by substring,
# first hit wins; anything unmatched falls through to the generic sentence.
_TRANSLATIONS = (
    ("ready_for_transformation_validation",
     "Some of the data needs your decisions before Trakt can continue."),
    ("mapping review", "Some columns in the files need your confirmation."),
    ("blocking decision", "A decision is waiting for you before Trakt can continue."),
    ("central lender tape",
     "Trakt could not build the combined loan record from these files. "
     "The files may be missing a loan listing."),
    ("central pipeline tape",
     "Trakt could not find the expected application listing in these files."),
    ("validation has blocking exceptions",
     "Some figures did not pass Trakt's checks and need your review."),
    ("not ready_for_validation",
     "The data could not be fully prepared for checking. It needs your review."),
    ("provenance",
     "Details about where this portfolio came from are incomplete. "
     "Check the portfolio details you entered."),
    ("assembler",
     "Trakt could not combine the portfolios into one report. This needs attention."),
    ("regime projection failed",
     "The regulatory report could not be prepared from this data. This needs attention."),
    ("no handoff", "The first step did not finish cleanly. Try running again."),
    ("missing onboarding handoff",
     "The first step did not finish cleanly. Try running again."),
)

GENERIC_PROBLEM = ("Something did not go as expected on our side. "
                   "Nothing has been lost. You can try running this step again.")


def humanise_blocker(raw: str) -> str:
    """Translate one internal blocker string into an operator sentence."""
    low = (raw or "").lower()
    for needle, sentence in _TRANSLATIONS:
        if needle in low:
            return sentence
    return GENERIC_PROBLEM


def humanise_blockers(raw: Iterable[str]) -> List[str]:
    """Translate + de-duplicate a list of internal blockers."""
    out: List[str] = []
    for r in raw or []:
        s = humanise_blocker(r)
        if s not in out:
            out.append(s)
    return out


# Scope explanations shown next to the scope picker.
SCOPE_EXPLANATIONS = {
    "file": "Applies to this delivery only. Trakt will ask again next time.",
    "portfolio": "Applies to every future delivery of this portfolio.",
    "client": "Applies to all of this client's portfolios, including future ones.",
    "global": "Applies to every client across Trakt. Choose this only for "
              "universal facts.",
}

STAGE_LABELS = {
    "received": "Files received",
    "understanding": "Data understood",
    "mapping": "Mapping",
    "validation": "Validation",
    "projection": "Regulatory report",
    "assembly": "Assembly",
    "publication": "Publication",
}

STATUS_LABELS = {
    "waiting": "Waiting",
    "running": "Running",
    "needs_review": "Needs review",
    "blocked": "Blocked",
    "ready": "Ready",
    "approved": "Approved",
    "rejected": "Rejected",
    "completed": "Completed",
}

WORKFLOW_TYPE_LABELS = {
    "new_client": "New client onboarding",
    "new_portfolio": "New portfolio onboarding",
    "recurring": "Recurring reporting",
    "backfill": "Historical backfill",
}

CLASSIFICATION_SENTENCES = {
    "new_client": "This looks like a completely new client. Trakt will guide "
                  "you through onboarding step by step.",
    "new_portfolio": "We know this client. This looks like a new portfolio, so "
                     "Trakt will reuse the rules you've already approved and "
                     "only ask about what's new.",
    "recurring": "We recognise this delivery. Trakt will apply your approved "
                 "rules automatically and only flag anything new.",
    "backfill": "This looks like an earlier period for a portfolio we already "
                "know. Trakt will apply your approved rules to the historical "
                "delivery.",
}
