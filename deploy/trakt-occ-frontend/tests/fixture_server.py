"""Fixture frontend + API for exercising smoke_test.sh without Azure.

Starts two HTTP servers on the ports given, mimicking the shapes the smoke test
depends on:

  frontend  /                     the built index.html, referencing a hashed asset
            /assets/index-*.js    a bundle containing the API base URL and the
                                  X-Operator-Token header name
  api       /health               the operations-control health envelope
            /ops/me               401 without a token, 200 with the fixture token
            /ops/dashboard        401 without a token, 200 with it
            OPTIONS *             CORS preflight headers

Usage:
  python3 fixture_server.py <frontend-port> <api-port> <api-base-url> [options]

Options:
    --cors-origin ORIGIN   echo this origin on preflight (default: echo whatever
                           the request sent, i.e. a correctly configured API)
    --cors-none            send no allow-origin header (misconfigured API)
    --cors-wildcard        send '*' (forbidden for this API)
    --no-cors-token-header omit X-Operator-Token from allow-headers
    --omit-api-url         build a bundle WITHOUT the API base URL inlined
    --embed-token          embed the operator token in the bundle (must be caught)
    --unbuilt-index        serve the UNBUILT project index.html (the project
                           directory was uploaded instead of dist/)
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIXTURE_TOKEN = "fixture-operator-token"
ASSET_NAME = "index-Ab12Cd34.js"


def build_bundle(api_url: str, *, omit_api_url: bool, embed_token: bool) -> bytes:
    parts = [
        "// fixture bundle\n",
        'const KEY="trakt_ops_token";\n',
        'const HEADER="X-Operator-Token";\n',
    ]
    if not omit_api_url:
        parts.append(f'const BASE="{api_url}";\n')
    if embed_token:
        parts.append(f'const LEAKED="{FIXTURE_TOKEN}";\n')
    return "".join(parts).encode()


INDEX_HTML = (
    "<!doctype html><html lang=\"en\"><head><title>Trakt Operations</title>"
    f'<script type="module" src="/assets/{ASSET_NAME}"></script>'
    "</head><body><div id=\"root\"></div></body></html>"
).encode()

#: The UNBUILT project index.html, byte-for-byte in the shape that matters: it
#: has the same root element (so a naive probe passes) but points at the dev
#: module entry and references no hashed asset. Uploading the project directory
#: instead of dist/ serves exactly this.
SOURCE_INDEX_HTML = (
    "<!doctype html><html lang=\"en\"><head><title>Trakt Operations</title>"
    "</head><body><div id=\"root\"></div>"
    '<script type="module" src="/src/main.tsx"></script>'
    "</body></html>"
).encode()


def make_frontend_handler(bundle: bytes, *, unbuilt: bool = False):
    index = SOURCE_INDEX_HTML if unbuilt else INDEX_HTML

    class FrontendHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence
            pass

        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self._send(200, index, "text/html")
            elif self.path == f"/assets/{ASSET_NAME}" and not unbuilt:
                self._send(200, bundle, "text/javascript")
            elif self.path == "/health":
                # Excluded from navigationFallback, so it must NOT return the app.
                self._send(404, b"not found", "text/plain")
            else:
                # navigationFallback: any other route serves the app shell.
                self._send(200, index, "text/html")

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return FrontendHandler


def make_api_handler(opts):
    class ApiHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_OPTIONS(self):
            origin = self.headers.get("Origin", "")
            self.send_response(200)
            if opts["cors_wildcard"]:
                self.send_header("Access-Control-Allow-Origin", "*")
            elif not opts["cors_none"]:
                self.send_header("Access-Control-Allow-Origin",
                                 opts["cors_origin"] or origin)
            if not opts["no_cors_token_header"]:
                self.send_header("Access-Control-Allow-Headers",
                                 "X-Operator-Token, Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "*")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"ok": True, "service": "operations-control",
                                 "auth_configured": True, "storage_ok": True})
                return
            if self.path in ("/ops/me", "/ops/dashboard"):
                if self.headers.get("X-Operator-Token") != FIXTURE_TOKEN:
                    self._json(401, {"ok": False, "errorCode": "OPS_UNAUTHENTICATED",
                                     "message": "Please sign in to continue."})
                    return
                if self.path == "/ops/me":
                    self._json(200, {"ok": True, "principal": {
                        "name": "Administrator", "role": "admin",
                        "is_admin": True, "all_clients": True, "clients": []}})
                else:
                    self._json(200, {"ok": True, "tiles": {}, "needs_attention": [],
                                     "recently_published": []})
                return
            self._json(404, {"ok": False, "message": "not found"})

        def _json(self, code, payload):
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ApiHandler


def main() -> None:
    frontend_port = int(sys.argv[1])
    api_port = int(sys.argv[2])
    api_url = sys.argv[3]
    flags = set(sys.argv[4:])

    opts = {
        "cors_none": "--cors-none" in flags,
        "cors_wildcard": "--cors-wildcard" in flags,
        "no_cors_token_header": "--no-cors-token-header" in flags,
        "cors_origin": None,
    }
    bundle = build_bundle(
        api_url,
        omit_api_url="--omit-api-url" in flags,
        embed_token="--embed-token" in flags,
    )

    frontend = ThreadingHTTPServer(
        ("127.0.0.1", frontend_port),
        make_frontend_handler(bundle, unbuilt="--unbuilt-index" in flags))
    api = ThreadingHTTPServer(("127.0.0.1", api_port), make_api_handler(opts))
    threading.Thread(target=frontend.serve_forever, daemon=True).start()
    threading.Thread(target=api.serve_forever, daemon=True).start()
    print(f"fixture frontend on {frontend_port}, api on {api_port}", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    main()
