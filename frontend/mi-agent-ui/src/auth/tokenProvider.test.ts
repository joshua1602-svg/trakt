import { afterEach, describe, expect, it, vi } from "vitest";
import {
  InteractionRequiredAuthError,
  type IPublicClientApplication,
} from "@azure/msal-browser";

import { authorizationHeaders, setAccessTokenProvider } from "./tokenProvider";
import { createMsalTokenProvider, resolveActiveAccount } from "./msalTokenProvider";
import { resolveAuthConfig, type AuthEnv } from "./msalConfig";

const SCOPE = "api://f7473442-de15-49f4-b6da-a9634bcbea8f/MI.Access";
const env: AuthEnv = {
  VITE_MI_BEARER_AUTH: "true",
  VITE_AAD_CLIENT_ID: "11111111-2222-3333-4444-555555555555",
  VITE_MI_API_SCOPE: SCOPE,
};
const cfg = resolveAuthConfig(env);

const account = (id: string) =>
  ({ homeAccountId: id, localAccountId: id, environment: "login.microsoftonline.com",
     tenantId: "t", username: "user@example.com" }) as never;

/** A minimal MSAL stand-in; only what the provider actually calls. */
function msalStub(over: Partial<Record<string, unknown>> = {}): IPublicClientApplication {
  let active: unknown = null;
  return {
    getActiveAccount: vi.fn(() => active),
    setActiveAccount: vi.fn((a: unknown) => { active = a; }),
    getAllAccounts: vi.fn(() => []),
    acquireTokenSilent: vi.fn(async () => ({ accessToken: "silent-token" })),
    acquireTokenRedirect: vi.fn(async () => undefined),
    loginRedirect: vi.fn(async () => undefined),
    logoutRedirect: vi.fn(async () => undefined),
    ...over,
  } as unknown as IPublicClientApplication;
}

afterEach(() => setAccessTokenProvider(null));

describe("authorizationHeaders — the single place a credential is attached", () => {
  it("attaches nothing when no provider is registered (flag off)", async () => {
    expect(await authorizationHeaders()).toEqual({});
  });

  it("attaches a bearer token when one is available", async () => {
    setAccessTokenProvider(async () => "abc.def.ghi");
    expect(await authorizationHeaders()).toEqual({ Authorization: "Bearer abc.def.ghi" });
  });

  it("attaches nothing when the user is not signed in", async () => {
    setAccessTokenProvider(async () => null);
    expect(await authorizationHeaders()).toEqual({});
  });

  it("attaches nothing — and does not throw — when acquisition fails", async () => {
    // A rejected acquisition inside a data-loading effect must not become an
    // unhandled rejection; the request goes out bare and the API answers 401.
    setAccessTokenProvider(async () => { throw new Error("network down"); });
    await expect(authorizationHeaders()).resolves.toEqual({});
  });
});

describe("account selection", () => {
  it("keeps an account the user already chose", () => {
    const chosen = account("chosen");
    const msal = msalStub({ getActiveAccount: vi.fn(() => chosen) });
    expect(resolveActiveAccount(msal)).toBe(chosen);
  });

  it("adopts the only known account", () => {
    const only = account("only");
    const msal = msalStub({ getAllAccounts: vi.fn(() => [only]) });
    expect(resolveActiveAccount(msal)).toBe(only);
    expect(msal.setActiveAccount).toHaveBeenCalledWith(only);
  });

  it("refuses to guess between two accounts", () => {
    const msal = msalStub({ getAllAccounts: vi.fn(() => [account("a"), account("b")]) });
    expect(resolveActiveAccount(msal)).toBeNull();
    expect(msal.setActiveAccount).not.toHaveBeenCalled();
  });
});

describe("token acquisition", () => {
  it("acquires silently for the MI API scope and the active account", async () => {
    const acc = account("a");
    const acquireTokenSilent = vi.fn(async () => ({ accessToken: "silent-token" }));
    const msal = msalStub({ getActiveAccount: vi.fn(() => acc), acquireTokenSilent });

    const token = await createMsalTokenProvider(msal, cfg)();

    expect(token).toBe("silent-token");
    expect(acquireTokenSilent).toHaveBeenCalledWith(
      expect.objectContaining({ scopes: [SCOPE], account: acc }));
  });

  it("returns null without calling MSAL when nobody is signed in", async () => {
    const msal = msalStub();
    expect(await createMsalTokenProvider(msal, cfg)()).toBeNull();
    expect(msal.acquireTokenSilent).not.toHaveBeenCalled();
  });

  it("falls back to interaction when consent or a policy requires it", async () => {
    const msal = msalStub({
      getActiveAccount: vi.fn(() => account("a")),
      acquireTokenSilent: vi.fn(async () => {
        throw new InteractionRequiredAuthError("interaction_required");
      }),
    });
    const interactive = vi.fn();

    expect(await createMsalTokenProvider(msal, cfg, interactive)()).toBeNull();
    expect(interactive).toHaveBeenCalledTimes(1);
  });

  it("does NOT redirect for a transient failure", async () => {
    // Offline is not a consent problem. Redirecting would throw the user out of
    // the app every time the network hiccuped.
    const msal = msalStub({
      getActiveAccount: vi.fn(() => account("a")),
      acquireTokenSilent: vi.fn(async () => { throw new Error("Failed to fetch"); }),
    });
    const interactive = vi.fn();

    expect(await createMsalTokenProvider(msal, cfg, interactive)()).toBeNull();
    expect(interactive).not.toHaveBeenCalled();
  });

  it("treats an empty access token as no token", async () => {
    const msal = msalStub({
      getActiveAccount: vi.fn(() => account("a")),
      acquireTokenSilent: vi.fn(async () => ({ accessToken: "" })),
    });
    expect(await createMsalTokenProvider(msal, cfg)()).toBeNull();
    // …so no `Authorization: Bearer ` with an empty value is ever sent.
    setAccessTokenProvider(createMsalTokenProvider(msal, cfg));
    expect(await authorizationHeaders()).toEqual({});
  });

  it("asks MSAL on every call, so an expired token renews itself", async () => {
    const acquireTokenSilent = vi.fn(async () => ({ accessToken: "fresh" }));
    const msal = msalStub({ getActiveAccount: vi.fn(() => account("a")), acquireTokenSilent });
    const provider = createMsalTokenProvider(msal, cfg);

    await provider();
    await provider();
    await provider();

    // No token is cached on our side; MSAL serves from its cache and renews from
    // the refresh token when it has expired. Caching here would defeat that.
    expect(acquireTokenSilent).toHaveBeenCalledTimes(3);
  });
});
