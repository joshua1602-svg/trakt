import { describe, expect, it } from "vitest";
import {
  DEFAULT_AUTHORITY,
  apiTokenRequest,
  buildMsalConfig,
  isBearerAuthEnabled,
  loginRequest,
  resolveAuthConfig,
  type AuthEnv,
} from "./msalConfig";

const CLIENT_ID = "11111111-2222-3333-4444-555555555555";
const SCOPE = "api://f7473442-de15-49f4-b6da-a9634bcbea8f/MI.Access";

const complete: AuthEnv = {
  VITE_MI_BEARER_AUTH: "true",
  VITE_AAD_CLIENT_ID: CLIENT_ID,
  VITE_MI_API_SCOPE: SCOPE,
};

describe("the feature flag", () => {
  it("is off when unset — the Static Web Apps path stays untouched", () => {
    const cfg = resolveAuthConfig({});
    expect(cfg.enabled).toBe(false);
    expect(isBearerAuthEnabled({})).toBe(false);
  });

  it("is off for an explicit false, and for anything that is not a yes", () => {
    for (const value of ["false", "0", "no", "off", "", "maybe"]) {
      expect(resolveAuthConfig({ ...complete, VITE_MI_BEARER_AUTH: value }).enabled).toBe(false);
    }
  });

  it("is on for the usual affirmatives, case-insensitively", () => {
    for (const value of ["true", "1", "yes", "on", "TRUE", " True "]) {
      expect(resolveAuthConfig({ ...complete, VITE_MI_BEARER_AUTH: value }).enabled).toBe(true);
    }
  });
});

describe("configuration", () => {
  it("defaults the authority to /organizations", () => {
    expect(resolveAuthConfig(complete).authority).toBe(DEFAULT_AUTHORITY);
    expect(DEFAULT_AUTHORITY).toBe("https://login.microsoftonline.com/organizations");
  });

  it("never defaults to a tenant-pinned or consumer-inclusive authority", () => {
    // A tenant GUID is the single-tenant pin this migration removes; /common
    // admits personal accounts, which carry no directory for the API to resolve
    // an organisation from.
    expect(DEFAULT_AUTHORITY).not.toMatch(/\/common$/);
    expect(DEFAULT_AUTHORITY).not.toMatch(/[0-9a-f]{8}-[0-9a-f]{4}/i);
  });

  it("honours an explicit authority override", () => {
    const cfg = resolveAuthConfig({ ...complete, VITE_AAD_AUTHORITY: "https://login.microsoftonline.com/common" });
    expect(cfg.authority).toBe("https://login.microsoftonline.com/common");
  });

  it("reports a flag that is on but unconfigured, naming what is missing", () => {
    const cfg = resolveAuthConfig({ VITE_MI_BEARER_AUTH: "true" });
    expect(cfg.misconfigured).toBe(true);
    expect(cfg.missing).toEqual(["VITE_AAD_CLIENT_ID", "VITE_MI_API_SCOPE"]);
    // A build that says it authenticates and cannot must not be treated as one
    // that does: attaching no token would surface as an unexplained 401.
    expect(isBearerAuthEnabled({ VITE_MI_BEARER_AUTH: "true" })).toBe(false);
  });

  it("does not complain about missing values while the flag is off", () => {
    expect(resolveAuthConfig({}).misconfigured).toBe(false);
  });
});

describe("token requests", () => {
  it("asks for the MI API scope alone when acquiring an ACCESS token", () => {
    // openid/profile alongside a resource scope makes Entra issue a token for a
    // different audience, which the API rejects on `aud`.
    expect(apiTokenRequest(resolveAuthConfig(complete)).scopes).toEqual([SCOPE]);
  });

  it("asks for MI.Access on the API's own audience", () => {
    const [scope] = apiTokenRequest(resolveAuthConfig(complete)).scopes!;
    expect(scope).toMatch(/^api:\/\/[0-9a-f-]+\/MI\.Access$/);
  });

  it("collects sign-in and resource consent together at login", () => {
    expect(loginRequest(resolveAuthConfig(complete)).scopes).toEqual(["openid", "profile", SCOPE]);
  });
});

describe("the MSAL configuration", () => {
  it("is a public client: no secret of any kind", () => {
    const raw = JSON.stringify(buildMsalConfig(resolveAuthConfig(complete)));
    expect(raw).not.toMatch(/secret/i);
    expect(raw).not.toMatch(/client_secret|clientSecret/);
  });

  it("keeps the token cache in sessionStorage, not localStorage", () => {
    expect(buildMsalConfig(resolveAuthConfig(complete)).cache?.cacheLocation).toBe("sessionStorage");
  });

  it("carries the client id and authority through", () => {
    const c = buildMsalConfig(resolveAuthConfig(complete));
    expect(c.auth.clientId).toBe(CLIENT_ID);
    expect(c.auth.authority).toBe(DEFAULT_AUTHORITY);
  });
});
