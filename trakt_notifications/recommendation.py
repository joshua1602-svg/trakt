#!/usr/bin/env python3
"""The action-language policy.

The capability this exists to make safe: telling a lender that a concentration
limit is likely to be breached, and where the projected exposure is coming
from, without telling them what to do about it. The evidence supports the first
statement and does not support the second.

Three levels, and a message may never exceed the configured one:

  **Level 0 — observation.**   "Broker A in the South West is the largest
                                projected contributor."
  **Level 1 — consideration.** "Consider reviewing completion timing for
                                Broker A's South West pipeline."
  **Level 2 — approved action.** Only reachable where an explicit, named client
                                action rule supplies the wording.

Level 2 is not a level this module can reach on its own. It requires a rule
object naming the test it applies to and carrying the exact sentence to use;
without one, a request for level 2 silently degrades to level 1 rather than
inventing an instruction. That asymmetry is deliberate: escalation must require
evidence, de-escalation must not.

No language model participates. Every sentence below is a template over
governed values.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from . import formatting as fmt
from .config import (
    LEVEL_APPROVED_ACTION, LEVEL_CONSIDERATION, LEVEL_OBSERVATION, LEVEL_RANK,
)

logger = logging.getLogger("trakt_notifications.recommendation")

#: Wording that may never appear in a generated recommendation at any level
#: below an approved action rule. Asserted by test, because the failure mode is
#: a plausible-sounding sentence rather than a crash.
FORBIDDEN_PHRASES = (
    "stop ", "reject", "decline ", "must ", "cease", "suspend", "halt ",
    "do not accept", "refuse",
)


def _contributor_phrase(contributor: Dict[str, Any]) -> Optional[str]:
    """``Broker A in the South West``, or a single dimension where that is all
    the governed attribution supports."""
    broker = contributor.get("broker")
    dimension = contributor.get("dimension")
    if broker and dimension and str(broker) != str(dimension):
        return f"{broker} in the {dimension}"
    return str(broker or dimension or contributor.get("name") or "") or None


def evidence_sentence(contributors: Optional[Dict[str, Any]]) -> Optional[str]:
    """The quantified evidence sentence, at every level.

    Names the primary projected contributor and the probability-weighted amount
    it contributes. Both come from the governed contributor aggregation, which
    reconciles to the expected numerator movement, so the sentence is a
    restatement of the concentration evaluation rather than a claim about it.
    """
    if not contributors or not contributors.get("available"):
        return None
    primary = contributors.get("primaryContributor")
    if not primary:
        return None
    phrase = _contributor_phrase(primary)
    if not phrase:
        return None
    amount = primary.get("expectedContribution")
    if amount is None:
        return f"Primary projected driver: {phrase}."
    return (f"{fmt.money(amount)} of probability-weighted expected completions "
            f"from {phrase} contributes to the projected position.")


def observation(contributors: Optional[Dict[str, Any]]) -> Optional[str]:
    """Level 0 — states who the largest projected contributor is. Nothing more."""
    if not contributors or not contributors.get("available"):
        return None
    primary = contributors.get("primaryContributor")
    phrase = _contributor_phrase(primary or {})
    if not phrase:
        return None
    return f"{phrase} is the largest projected contributor."


def consideration(contributors: Optional[Dict[str, Any]]) -> Optional[str]:
    """Level 1 — invites a review of timing. Never instructs an outcome.

    "Consider reviewing completion timing" is the strongest wording the
    evidence carries: the governed attribution shows where projected
    utilisation is coming from, which supports looking at that flow. It does
    not establish that the flow should be reduced, and the sentence must not
    imply that it does.
    """
    if not contributors or not contributors.get("available"):
        return None
    primary = contributors.get("primaryContributor")
    if not primary:
        return None
    broker = primary.get("broker")
    dimension = primary.get("dimension")
    if broker and dimension and str(broker) != str(dimension):
        target = f"{broker}'s {dimension} pipeline"
    elif broker or dimension:
        target = f"the {broker or dimension} pipeline"
    else:
        return None
    return (f"Consider reviewing completion timing for {target}. "
            f"Slowing or reprioritising this flow would reduce projected "
            f"utilisation.")


def approved_action(rule: Optional[Dict[str, Any]]) -> Optional[str]:
    """Level 2 — the operator-approved sentence, verbatim, or nothing.

    The wording is the client's, not ours. This function only checks that a
    rule exists, names an action and is not attempting to smuggle an
    instruction through a level the deployment has not enabled.
    """
    if not rule:
        return None
    statement = str(rule.get("statement") or "").strip()
    if not statement:
        return None
    return statement


def resolve(contributors: Optional[Dict[str, Any]], *, level: str,
            action_rule: Optional[Dict[str, Any]] = None
            ) -> Dict[str, Optional[str]]:
    """The recommendation lines for one risk item, at the configured ceiling.

    Returns the evidence sentence — which every level carries — plus at most
    one recommendation sentence, and the level actually used. A caller renders
    both; nothing downstream re-decides the level.
    """
    ceiling = LEVEL_RANK.get(level, LEVEL_RANK[LEVEL_CONSIDERATION])
    evidence = evidence_sentence(contributors)

    text: Optional[str] = None
    used = LEVEL_OBSERVATION

    if ceiling >= LEVEL_RANK[LEVEL_APPROVED_ACTION]:
        text = approved_action(action_rule)
        if text:
            used = LEVEL_APPROVED_ACTION
        else:
            # Configured for level 2 with no governing rule: degrade, never
            # improvise an instruction.
            logger.info("notifications: approved-action level requested with no "
                        "client action rule; using consideration wording")
    if text is None and ceiling >= LEVEL_RANK[LEVEL_CONSIDERATION]:
        text = consideration(contributors)
        if text:
            used = LEVEL_CONSIDERATION
    if text is None:
        text = observation(contributors)
        used = LEVEL_OBSERVATION

    if text and used != LEVEL_APPROVED_ACTION:
        lowered = text.lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase in lowered:
                # A template should never produce these; if one ever does, the
                # sentence is dropped rather than sent.
                logger.error("notifications: suppressed a recommendation "
                             "containing operational instruction wording")
                text = None
                break

    return {"evidence": evidence, "recommendation": text, "level": used}


def lines(resolved: Dict[str, Optional[str]]) -> List[str]:
    """The resolved recommendation as ordered, renderable sentences."""
    return [s for s in (resolved.get("evidence"),
                        resolved.get("recommendation")) if s]
