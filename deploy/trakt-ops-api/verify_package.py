"""Verify an unpacked trakt-ops-api deployment artefact can actually serve.

Run with the CURRENT WORKING DIRECTORY set to the unpacked artefact root, so the
import resolves exactly as it will under gunicorn on App Service:

    unzip -q ops-api.zip -d _pkgcheck
    cd _pkgcheck && python /path/to/deploy/trakt-ops-api/verify_package.py

Why this exists: ``py_compile`` parses a single file and never executes imports,
so it cannot catch a missing runtime package — the artefact's real failure mode
is ``ModuleNotFoundError`` at worker start, which looks like a healthy build in
CI and a dead service in production. Importing the app from the artefact itself
turns that into a build failure.

Exits non-zero with a named reason on any problem.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Routes that must exist for the service to be useful at all: the unauthenticated
#: probe, the principal lookup the React shell needs, and the entry point of the
#: administrator configuration area.
REQUIRED_ROUTES = ("/health", "/ops/me", "/ops/admin/config")

#: Governed configuration read from disk at runtime by RELATIVE path, so it has to
#: be present in the artefact next to the code.
REQUIRED_FILES = (
    "config/system/fields_registry.yaml",
    "config/regime/annex2_delivery_rules.yaml",
    "config/asset/product_profiles.yaml",
)

#: Executed as subprocesses by the Annex 2 delivery stages (not imported), so an
#: import check alone would not notice they are missing.
REQUIRED_SCRIPTS = (
    "engine/gate_4b_delivery/annex2_delivery_normalizer.py",
    "engine/gate_5_delivery/xml_builder_annex2.py",
)

EXPECTED_TITLE = "Trakt Operations Control API"


def fail(message: str) -> None:
    print(f"ARTEFACT CHECK FAILED: {message}")
    sys.exit(1)


def main() -> None:
    root = Path.cwd()

    # Import from the ARTEFACT, not from wherever this script happens to live.
    # Running `python /abs/path/verify_package.py` puts the script's own directory
    # on sys.path — never the working directory — so without this the check would
    # import nothing and report a false failure. gunicorn resolves the app from
    # its working directory, which is what we are reproducing here.
    sys.path.insert(0, str(root))

    for rel in REQUIRED_FILES + REQUIRED_SCRIPTS:
        if not (root / rel).exists():
            fail(f"missing {rel} (needed at runtime, resolved by path)")

    # The frontend must never ship inside the API artefact.
    for forbidden in ("frontend", "node_modules"):
        if (root / forbidden).exists():
            fail(f"{forbidden}/ must not ship in the API artefact")

    try:
        from operations_control.api.app import app
    except Exception as exc:  # noqa: BLE001 — the point is to report, not raise
        fail(f"could not import operations_control.api.app ({type(exc).__name__}: {exc})")
        return

    if app.title != EXPECTED_TITLE:
        fail(f"unexpected application: {app.title!r} (expected {EXPECTED_TITLE!r})")

    paths = {getattr(route, "path", "") for route in app.routes}
    missing = [route for route in REQUIRED_ROUTES if route not in paths]
    if missing:
        fail(f"application is missing route(s): {', '.join(missing)}")

    print(f"artefact OK: {app.title} — {len(paths)} routes, "
          f"{len(REQUIRED_FILES)} governed config files, "
          f"{len(REQUIRED_SCRIPTS)} delivery scripts")


if __name__ == "__main__":
    main()
