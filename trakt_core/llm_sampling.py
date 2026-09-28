"""Whether a sampling parameter can be sent — asked, not assumed, in one place.

`anthropic` 1.x removed ``temperature``, ``top_p`` and ``top_k`` from
``messages.create()`` altogether, and the current models reject them anyway. A
call that still passes one raises::

    TypeError: Messages.create() got an unexpected keyword argument 'temperature'

before any HTTP request is made.

:mod:`mi_agent.llm_query_parser` worked this out the hard way — an LLM-enabled
acceptance run over 166 questions raised that `TypeError` on every call and was
reported as a clean model comparison having never reached the API — and its
comment says why an allowlist of model names cannot answer it:

    Whether `temperature` can be sent is TWO facts […] DOES THIS SDK EXPOSE THE
    PARAMETER? […] DOES THIS MODEL ACCEPT IT?

THE LESSON WAS LEARNED IN ONE MODULE AND THE OTHER CALLERS NEVER HEARD IT. The
onboarding mapping reviewer, the Gate 1 mapper and the enum agent each went on
passing ``temperature=`` and each crashes on the installed SDK — and because
onboarding records a failed run rather than raising, the crash reached its
operator four stages later as "the delivery appears to hold no files".

So the mechanism lives here, once, and every caller asks it. Nothing in this
module needs editing when a model ships.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, FrozenSet, Iterable

#: Phrases an SDK or the API uses when a sampling kwarg is the problem. Not a
#: model list: a list of ways one specific rejection is worded.
SAMPLING_REJECTION_MARKS = ("temperature", "top_p", "top_k")


def sdk_sampling_parameters(client) -> FrozenSet[str]:
    """Which sampling parameters THIS SDK's ``messages.create`` accepts.

    Asked of the signature rather than assumed from a version string, because a
    version string is one more thing to keep in step with a release. An SDK that
    accepts arbitrary ``**kwargs`` reports them all as acceptable and the
    model's own rejection is then the authority, which is the correct order: the
    SDK cannot know what the API allows.
    """
    try:
        params = inspect.signature(client.messages.create).parameters
    except (TypeError, ValueError, AttributeError):  # an unreadable signature
        return frozenset(SAMPLING_REJECTION_MARKS)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return frozenset(SAMPLING_REJECTION_MARKS)
    return frozenset(n for n in SAMPLING_REJECTION_MARKS if n in params)


def is_sampling_rejection(exc: BaseException) -> bool:
    """Is this failure the sampling parameter being refused?

    A ``TypeError`` from the SDK and a 400 from the API are the same fact told
    two ways, and both must downgrade rather than escape.
    """
    text = str(exc).lower()
    if not any(mark in text for mark in SAMPLING_REJECTION_MARKS):
        return False
    return (isinstance(exc, TypeError) or "400" in text
            or "unexpected keyword" in text or "not supported" in text
            or "deprecated" in text or "unsupported" in text
            or "invalid_request" in text)


def sampling_for(client, model: str, *,
                 rejected: Iterable[str] = ()) -> Dict[str, Any]:
    """The sampling kwargs to send for ``model`` on this client — often none.

    Determinism where the runtime allows it. Where it does not, the caller's
    task is a constrained parse validated downstream, so the model's own
    default sampling is used rather than the call being failed.

    ``rejected`` is the caller's memory of models the API has already refused,
    so one model release costs one retry rather than a source edit.
    """
    if (model or "") in set(rejected or ()):
        return {}
    if "temperature" not in sdk_sampling_parameters(client):
        return {}
    return {"temperature": 0.0}
