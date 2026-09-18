"""operations_control.occ_agent.llm_mapping — the model's second opinion on a
column the deterministic mapper could not settle.

WHAT WAS MISSING. ``engine/gate_1_alignment/llm_mapper_agent.py`` has existed
all along, with its redaction, its canonical-only enforcement and its human
review session. ``config/system/onboarding_agent.yaml`` has carried a full
budget policy for it. And nothing in ``operations_control/occ_agent`` ever
called it: a real client's hundred-column tape arrived, thirty columns matched
nothing, and those thirty were reported as unreadable with no proposal against
any of them. The operator was left to name each canonical field from memory.

THE CONTRACT, WHICH IS THE ENGINE'S OWN. "The LLM only reviews unresolved
ambiguity and never writes final mappings." So:

  * it is shown ONLY what the deterministic tiers could not settle — a column
    the mapper matched at ``exact``, ``normalized`` or ``alias``, or above the
    configured confidence, is never sent, because there is nothing to ask;
  * what comes back is a SUGGESTION attached to a decision a human answers. It
    never enters ``resolved``, never reaches the tape, and cannot make a
    mapping stand without someone confirming it;
  * every suggestion says it came from the model, so an operator approving one
    knows what they are approving. A proposal presented as a deterministic
    match would be the model writing mappings by another route.

THE BUDGET IS THE CONFIGURATION'S, NOT THIS MODULE'S. Calls per run, items per
call, how many sample values may be shown, which model, and the confidence
above which there is nothing to ask all come from ``llm_policy`` and ``llm`` in
``config/system/onboarding_agent.yaml``. Frontier escalation is declared off
there, and this module has no code path that could reach an escalation model
even if it were on.

WHEN IT DOES NOTHING, IT SAYS SO. No key, the policy switched off, the budget
spent, too many unresolved columns to ask about sensibly — each returns a
reason in words, recorded on the run. A model that silently did not run looks
exactly like a model that had nothing to say, and those are not the same thing
for anyone deciding whether to trust the mapping.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

REPO = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO / "config/system/onboarding_agent.yaml"

#: The environment variable the deployment supplies the key in.
API_KEY_ENV = "ANTHROPIC_API_KEY"

#: Recorded on every row and decision the model touched, so "the model
#: suggested this" is never mistaken for "the mapper matched this". The word is
#: the one the operator contract already uses for a recommendation's origin
#: (``operations_control.contracts``: ``llm|deterministic|memory|operator``), so
#: a decision raised here and one raised anywhere else say it the same way.
BASIS = "llm"


@dataclass(frozen=True)
class Policy:
    """The governed budget for one run, read from configuration.

    Nothing here has a default that would be more permissive than the file: an
    unreadable or absent policy leaves the model switched OFF, because a
    missing budget is not an unlimited one.
    """

    enabled: bool = False
    model: str = ""
    max_calls: int = 0
    max_items_per_call: int = 0
    #: A deterministic match at or above this is settled; there is nothing to
    #: ask, and asking would spend budget to be told what is already known.
    settled_above: float = 1.0
    max_samples: int = 0
    #: Above this many unresolved columns the configuration says to stop asking
    #: the model and put the question to a person instead.
    unresolved_ceiling: int = 0
    allow_frontier_escalation: bool = False

    @property
    def max_items(self) -> int:
        return max(0, self.max_calls) * max(0, self.max_items_per_call)

    @classmethod
    def load(cls, path: Path = POLICY_PATH) -> "Policy":
        try:
            doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except Exception:                      # pragma: no cover — config guard
            return cls()
        p = doc.get("llm_policy") or {}
        llm = doc.get("llm") or {}
        skip = p.get("skip_llm_if") or {}
        budget = p.get("uncertainty_budget") or {}
        return cls(
            enabled=bool(p.get("enabled", False)),
            model=str(llm.get("mapping_model") or p.get("model") or ""),
            max_calls=int(p.get("max_llm_calls_per_run", 0) or 0),
            max_items_per_call=int(p.get("max_items_per_call", 0) or 0),
            settled_above=float(skip.get("deterministic_confidence_above", 1.0)),
            max_samples=int(p.get("max_sample_values_per_field", 0) or 0),
            unresolved_ceiling=int(budget.get("if_unresolved_items_above", 0)
                                   or 0),
            allow_frontier_escalation=bool(
                llm.get("allow_frontier_escalation", False)),
        )


@dataclass
class Suggestion:
    """One column, and what the model made of it. Never a mapping."""

    column: str
    field_name: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    alternative: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"column": self.column, "field_name": self.field_name,
                "confidence": round(float(self.confidence), 4),
                "reasoning": self.reasoning, "alternative": self.alternative,
                "basis": BASIS}


@dataclass
class Outcome:
    """What the model was asked, what it answered, and why not where not."""

    suggestions: List[Suggestion] = field(default_factory=list)
    asked: int = 0
    #: Plain words for the run record. Empty when the model ran normally.
    skipped_because: str = ""
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"suggestions": [s.to_dict() for s in self.suggestions],
                "asked": self.asked, "model": self.model,
                "skipped_because": self.skipped_because}

    @property
    def by_column(self) -> Dict[str, Suggestion]:
        return {s.column: s for s in self.suggestions if s.field_name}


#: Tiers that settle a column outright, so there is nothing to ask a model
#: about. The first three are ``execution._TRUSTED_TIERS`` — exact or
#: contract-backed, settled whatever the numeric confidence, and on a first
#: onboarding PROPOSED to a person rather than applied. Either way the mapper
#: has an answer and a model is not needed to supply one. The fourth is an
#: operator's own answer, which ``execution`` handles on a separate branch
#: before the mapper is consulted at all; it is named here because this module
#: reads the REPORT rather than that branch, and a column a person already
#: answered is the last thing that should be sent to a model.
SETTLED_TIERS = frozenset({"exact", "normalized", "alias", "operator_approved"})


def unresolved(rows: Sequence[Dict[str, Any]], policy: Policy) -> List[str]:
    """The columns of the primary tape the deterministic pass did not settle.

    A settled tier or a confidence above the threshold is excluded on the
    configuration's own terms: there is nothing to ask, and asking would spend
    budget to be told what is already known.

    THE PRIMARY TAPE ONLY, WHICH IS A BUDGET DECISION AND NOT A CLAIM ABOUT
    WORTH. Every file's columns are now put to a PERSON, and a mapping approved
    on any of them is promoted to a governed rule. What is confined to the
    primary tape is the MODEL: the sample values it reasons from come from that
    frame, and ``max_llm_calls_per_run`` is a per-run budget that a four-file
    pack would exhaust on the first extract. A column in another file that
    nothing matched is reported unmapped and offered to an operator, which is
    the same remedy a model's suggestion leads to.
    """
    out: List[str] = []
    seen = set()
    for row in rows or []:
        if not row.get("primary"):
            continue
        column = str(row.get("source_column") or "")
        if not column or column in seen:
            continue
        tier = str(row.get("tier") or "")
        if tier in SETTLED_TIERS:
            continue
        if float(row.get("confidence") or 0.0) > policy.settled_above:
            continue
        seen.add(column)
        out.append(column)
    return out


def suggest(columns: Sequence[str], frame: Any, *, policy: Policy,
            registry_path: Path, aliases_dir: Path, asset_type: str,
            api_key: str = "") -> Outcome:
    """Ask the model about the unsettled columns, within the governed budget.

    Never raises. Every refusal and every failure comes back as a reason in
    ``skipped_because``, because a model that quietly did not run and a model
    that had nothing to say look identical from the outside and are not the
    same thing.
    """
    outcome = Outcome(model=policy.model)
    if not policy.enabled:
        outcome.skipped_because = ("the model is switched off for this "
                                   "environment")
        return outcome
    if policy.allow_frontier_escalation:
        # Declared off in configuration. If it were ever turned on, this module
        # still has no path to an escalation model, and saying so is better
        # than appearing to honour a setting it does not implement.
        outcome.skipped_because = ("this step does not escalate beyond the "
                                   "configured mapping model")
        return outcome
    key = api_key or os.environ.get(API_KEY_ENV, "")
    if not key:
        outcome.skipped_because = "no model access is configured here"
        return outcome
    columns = [c for c in columns if c]
    if not columns:
        return outcome
    if policy.unresolved_ceiling and len(columns) > policy.unresolved_ceiling:
        # The configuration's own instruction: past this many unsettled
        # columns, stop asking the model and put the question to a person.
        outcome.skipped_because = (
            f"{len(columns)} columns are unsettled, which is more than this "
            "step asks a model about — they are put to you instead")
        return outcome
    if policy.max_items <= 0:
        outcome.skipped_because = "no budget is allowed for this run"
        return outcome

    asked = list(columns[:policy.max_items])
    try:
        from engine.gate_1_alignment.llm_mapper_agent import LLMFieldMapper
        mapper = LLMFieldMapper(
            registry_path=Path(registry_path),
            portfolio_type=asset_type or "",
            aliases_dir=Path(aliases_dir),
            api_key=key,
            model=policy.model,
            batch_size=max(1, policy.max_items_per_call),
            max_sample_values=max(0, policy.max_samples),
        )
        raw = mapper.suggest_mappings(asked, frame)
    except Exception as exc:                   # noqa: BLE001 — never a crash
        outcome.skipped_because = (f"the model could not be reached "
                                   f"({type(exc).__name__})")
        return outcome

    outcome.asked = len(asked)
    for item in raw:
        outcome.suggestions.append(Suggestion(
            column=str(getattr(item, "raw_header", "")),
            field_name=str(getattr(item, "suggested_field", "") or ""),
            confidence=float(getattr(item, "confidence", 0.0) or 0.0),
            reasoning=str(getattr(item, "reasoning", "") or ""),
            alternative=str(getattr(item, "alternative_field", "") or "")))
    return outcome


def decision(suggestion: Suggestion, *, source_file: str,
             populated: int, rows: int) -> Dict[str, Any]:
    """A suggestion, in the shape the run raises a decision in.

    Deliberately says the model proposed it, in the question itself. An
    operator confirming a mapping is entitled to know whether they are
    confirming a contract-backed match or a proposal — and a suggestion
    presented as a match would be the model writing mappings by another route.
    """
    field_words = suggestion.field_name.replace("_", " ")
    evidence = (f"Trakt's own matching could not place this column. "
                f"{populated} of {rows} records carry a value.")
    if suggestion.reasoning:
        evidence = f"{evidence} The model's reason: {suggestion.reasoning}"
    return {
        "decision_id": f"llm_{_slug(suggestion.column)}",
        "decision_type": "mapping_confirmation",
        "target_field": suggestion.field_name,
        "source_column": suggestion.column,
        "source_file": source_file,
        "status": "pending",
        "blocking": True,
        "recommended_action": "accept_mapping",
        "available_actions": ["accept_mapping", "choose_alternative",
                              "mark_unavailable"],
        "confidence": round(float(suggestion.confidence), 4),
        "basis": BASIS,
        "issue": (f"Trakt could not match '{suggestion.column}' on its own. "
                  f"A model suggests it is {field_words} — it needs your "
                  "confirmation before anything uses it."),
        "evidence_summary": evidence,
        "proposed_mapping": f"{suggestion.column} → {suggestion.field_name}",
        "alternative_field": suggestion.alternative,
    }


def _slug(value: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")[:48]
