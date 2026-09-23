"""trakt_core.fault_report — a recorded failure says WHERE it happened.

Twice in one delivery a run stopped, recorded ``f"{type(exc).__name__}: {exc}"``
and threw the traceback away:

    TypeError: Messages.create() got an unexpected keyword argument 'temperature'
    ValueError: The truth value of a Series is ambiguous. Use a.empty, ...

Both are one line of code away from obvious and were hours away from found,
because the one thing a message like that cannot tell you is which line. The
second took a static scan of every boolean test in the package to locate — a
recorded ``source_table_loader.py:201`` would have taken a minute.

So: an exception recorded for diagnosis carries its location and its traceback.

THIS IS NOT OPERATOR TEXT. ``operations_control.language`` forbids exception
classes, paths and tracebacks in anything the UI renders, and it is right to —
an operator is owed a sentence about their delivery, not a stack. What is
recorded here belongs in the technical record beside it: the run artefacts and
the event log, which the UI never renders and an engineer always reads.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, Optional

#: The tree whose frames are OURS. A failure's deepest frame is usually inside
#: pandas or the SDK; the useful one is the last frame we wrote.
_ROOT = Path(__file__).resolve().parents[1]


def _is_ours(filename: str) -> bool:
    try:
        Path(filename).resolve().relative_to(_ROOT)
    except (ValueError, OSError):
        return False
    # A dependency installed inside the tree is not our code.
    return "site-packages" not in filename and "dist-packages" not in filename


def fault_location(exc: BaseException) -> str:
    """``relative/path.py:line in function`` for the deepest frame we own.

    Falls back to the deepest frame of any origin, because a failure raised
    entirely inside a library is still better placed than not placed at all.
    Never raises: a diagnostic that fails while reporting a failure is worse
    than the gap it was added to close.
    """
    try:
        frames = traceback.extract_tb(exc.__traceback__)
    except Exception:                       # noqa: BLE001 — see docstring
        return ""
    if not frames:
        return ""
    ours = [f for f in frames if _is_ours(f.filename)]
    frame = (ours or frames)[-1]
    try:
        where = str(Path(frame.filename).resolve().relative_to(_ROOT))
    except (ValueError, OSError):
        where = Path(frame.filename).name
    return f"{where}:{frame.lineno} in {frame.name}"


def fault_report(exc: BaseException, *, include_traceback: bool = True) -> Dict[str, Any]:
    """The technical record of one failure: what, where, and the whole stack."""
    report: Dict[str, Any] = {
        "error": f"{type(exc).__name__}: {exc}",
        "error_type": type(exc).__name__,
        "error_location": fault_location(exc),
    }
    if include_traceback:
        try:
            report["error_traceback"] = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__))
        except Exception:                   # noqa: BLE001 — see fault_location
            report["error_traceback"] = ""
    return report


def describe(exc: BaseException) -> str:
    """One technical line: the message, and where it came from.

    ``ValueError: ... (at engine/onboarding_agent/source_table_loader.py:201 in
    redetect_header)``
    """
    where: Optional[str] = fault_location(exc)
    head = f"{type(exc).__name__}: {exc}"
    return f"{head} (at {where})" if where else head
