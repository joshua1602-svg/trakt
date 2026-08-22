/**
 * A SPA bundle is public. Anyone who loads the page can read every byte of it,
 * so a client secret in there is a published credential — which is precisely why
 * this app is a PUBLIC client using Authorization Code + PKCE and holds none.
 *
 * These read the auth SOURCE, through Vite's raw glob rather than node:fs, so
 * the file stays browser-safe and needs no @types/node in the app's typecheck.
 * A secret would have to pass through here to reach a bundle. The compiled
 * output is scanned separately by `scripts/check-bundle-leaks.mjs`, which the
 * deployment workflow runs against `dist/`.
 */

import { describe, expect, it } from "vitest";

// Eager + raw: the source text of every non-test module in this directory.
const sources = import.meta.glob("./*.{ts,tsx}", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

function authSources(): [string, string][] {
  return Object.entries(sources).filter(([name]) => !name.includes(".test."));
}

describe("the auth layer holds no credential", () => {
  it("has sources to inspect", () => {
    // Guards every assertion below from passing on an empty set.
    expect(authSources().length).toBeGreaterThanOrEqual(4);
  });

  it("declares no client secret", () => {
    for (const [name, text] of authSources()) {
      expect(text, `${name} names a client secret`).not.toMatch(/clientSecret|client_secret/);
    }
  });

  it("reads no secret-shaped environment variable", () => {
    // VITE_* is inlined into the bundle at build time, so a VITE_ variable with
    // SECRET/PASSWORD/PRIVATE in its name is a secret published to the world.
    for (const [name, text] of authSources()) {
      expect(text, `${name} reads a secret-shaped VITE_ variable`)
        .not.toMatch(/VITE_[A-Z_]*(SECRET|PASSWORD|PRIVATE)/);
    }
  });

  it("uses no flow that would require one", () => {
    for (const [name, text] of authSources()) {
      // ROPC (username/password) and client credentials need a secret or the
      // user's password; implicit returns tokens in the URL. None belong in a
      // SPA, and none are used here.
      expect(text, `${name} uses ROPC`).not.toMatch(/acquireTokenByUsernamePassword/);
      expect(text, `${name} uses client credentials`)
        .not.toMatch(/acquireTokenByClientCredential|client_credentials/);
      expect(text, `${name} enables an implicit flow`)
        .not.toMatch(/allowImplicitFlow|response_type=\s*["']?token/);
    }
  });

  it("requests tokens through the redirect/silent PKCE paths only", () => {
    const provider = sources["./msalTokenProvider.ts"];
    expect(provider, "msalTokenProvider.ts was not found by the glob").toBeTruthy();
    // msal-browser performs Authorization Code + PKCE for these; the code
    // challenge is what replaces a secret.
    expect(provider).toMatch(/acquireTokenSilent/);
    expect(provider).toMatch(/acquireTokenRedirect|loginRedirect/);
  });
});
