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

#: Routes contributed by INCLUDED routers. This FastAPI version keeps an included
#: router as a single object in ``app.routes`` rather than flattening its children,
#: so these are checked against the OpenAPI document, which does flatten. Without
#: this, a router that failed to mount would leave the artefact looking healthy.
REQUIRED_ROUTER_ROUTES = (
    "/ops/onboarding/home",
    "/ops/concentration/{client}/review",
)

#: Governed configuration read from disk at runtime by RELATIVE path, so it has to
#: be present in the artefact next to the code.
REQUIRED_FILES = (
    "config/system/fields_registry.yaml",
    "config/regime/annex2_delivery_rules.yaml",
    "config/asset/product_profiles.yaml",
    # The governed concentration-test metric registry, resolved relative to the
    # artefact root by mi_agent.concentration_tests.library.
    "config/risk/concentration_test_library.yaml",
)

#: Executed as subprocesses by the Annex 2 delivery stages (not imported), so an
#: import check alone would not notice they are missing.
REQUIRED_SCRIPTS = (
    "engine/gate_4b_delivery/annex2_delivery_normalizer.py",
    "engine/gate_5_delivery/xml_builder_annex2.py",
)

#: Imported at runtime but NOT at application import time, so a missing one
#: would not fail the import check above. ``trakt_mail`` is loaded when the OCC
#: Agent mounts, and its absence is deliberately non-fatal there (a pack is
#: recorded rather than sent) — which is exactly why it has to be asserted here
#: instead. A silently mail-less deployment is the failure this catches.
REQUIRED_LAZY_MODULES = ("trakt_mail.adapter", "trakt_mail.presentation",
                         "trakt_mail.inbound", "trakt_mail.ingest")

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

    try:
        served = set(app.openapi().get("paths") or {})
    except Exception as exc:  # noqa: BLE001 — report, do not raise
        fail(f"could not build the OpenAPI document ({type(exc).__name__}: {exc})")
        return
    missing_router = [r for r in REQUIRED_ROUTER_ROUTES if r not in served]
    if missing_router:
        fail("application is missing included-router route(s): "
             + ", ".join(missing_router))

    import importlib
    for module in REQUIRED_LAZY_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 — report, do not raise
            fail(f"could not import {module}, which the OCC Agent loads at "
                 f"runtime ({type(exc).__name__}: {exc})")

    # The PDF renderer is imported ON USE, so importing the module above does
    # not prove reportlab is installed. Render one, from the artefact.
    try:
        from trakt_mail import presentation as _pres
        _pdf = _pres.render_pdf(_pres.parse_pack(
            "# Onboarding — Package check\n\nReference: ONB-0000-0000\n\n"
            "- [required] **A question**\n"), receipt_id="com_pkgcheck")
        if not _pdf.startswith(b"%PDF-"):
            fail("the client checklist renderer did not produce a PDF")
    except Exception as exc:  # noqa: BLE001 — report, do not raise
        fail("the client checklist renderer does not work in this artefact "
             f"({type(exc).__name__}: {exc}); check reportlab is in "
             "requirements.txt")

    print(f"artefact OK: {app.title} — {len(served)} served paths, "
          f"{len(REQUIRED_FILES)} governed config files, "
          f"{len(REQUIRED_SCRIPTS)} delivery scripts")


if __name__ == "__main__":
    main()
