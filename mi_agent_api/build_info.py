#!/usr/bin/env python3
"""WHICH COMMIT IS SERVING THIS REQUEST — an immutable stamp, not a version.

THE PROBLEM THIS FIXES. The live certification reported

    deployed commit : version=1.0.0

and `1.0.0` is the application's own version string. It is written by hand, it
changes when somebody remembers to change it, and it was identical across every
deploy this year. It establishes nothing: a certification that passes against
`1.0.0` cannot say whether it certified the release it was pointed at, an older
build that never got replaced, or a rollback nobody recorded. Provenance that
cannot distinguish two builds is not provenance.

WHY A FILE AND NOT `git rev-parse`. The deployed artefact has no `.git` — it is
a zip of selected paths. Asking git at runtime therefore returns nothing on the
server and the DEVELOPER'S OWN checkout locally, which is worse than nothing: it
would report a commit that has never been deployed anywhere. The stamp is
written ONCE, by the workflow that builds the artefact, from the SHA that
workflow checked out, and travels inside the zip. It cannot drift from the code
beside it because it is packaged with it.

WHAT IT IS NOT. It is not a secret and not a capability: a commit SHA of a
private repository names a revision to someone who already has the repository
and nothing to anyone who does not, which is the same posture `/health` already
takes with `semantics_path().name`.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

#: Written into the artefact root by `.github/workflows/deploy-mi-api.yml`.
#: Beside the packaged code, so a rollback carries its own stamp back with it.
_STAMP = Path(__file__).resolve().parents[1] / "build_info.json"

#: The environment is consulted FIRST so a container can be stamped without
#: rebuilding the artefact, and so a local run can identify itself. A stamp file
#: is the deployed case; this is the escape hatch, and it is named explicitly
#: rather than sniffed from whatever CI variable happens to be set.
_ENV_COMMIT = "TRAKT_BUILD_COMMIT"
_ENV_REF = "TRAKT_BUILD_REF"
_ENV_TIME = "TRAKT_BUILD_TIME"


@lru_cache(maxsize=1)
def build_info() -> Dict[str, Any]:
    """``{"commit", "ref", "builtAt", "source"}`` — ``commit`` may be None.

    NONE IS AN HONEST ANSWER and the caller must be able to see it. A
    certification that cannot establish the deployed commit has to fail, and it
    can only do that if "unknown" is reported as unknown rather than filled in
    with a version string, a branch name, or the string "unknown" dressed up as
    a value.
    """
    commit = (os.environ.get(_ENV_COMMIT) or "").strip()
    if commit:
        return {"commit": commit,
                "ref": (os.environ.get(_ENV_REF) or "").strip() or None,
                "builtAt": (os.environ.get(_ENV_TIME) or "").strip() or None,
                "source": "environment"}
    try:
        stamped = json.loads(_STAMP.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - an unstamped build must say so
        return {"commit": None, "ref": None, "builtAt": None,
                "source": "unstamped"}
    value = str(stamped.get("commit") or "").strip()
    return {"commit": value or None,
            "ref": stamped.get("ref") or None,
            "builtAt": stamped.get("builtAt") or None,
            "source": "artefact" if value else "unstamped"}
