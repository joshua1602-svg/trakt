/**
 * msalConfig — the dashboard's Entra sign-in configuration, and the flag that
 * decides whether it is used at all.
 *
 * Phase 3 of the multitenant migration (docs/react_multitenant_auth_migration.md).
 * The SPA signs the user in against `/organizations` with MSAL and calls the MI
 * API with an access token for `api://<mi-api-app-id>/MI.Access`, which
 * `mi_agent_api/react_auth.py` validates. Nothing here removes the Static Web
 * Apps path: with `VITE_MI_BEARER_AUTH` unset or false, no MSAL object is
 * constructed, no token is requested and no header is attached — the app behaves
 * exactly as it does today.
 *
 * What must NOT appear in this file, or anywhere else in the bundle
 * ----------------------------------------------------------------
 * A client secret. This is a PUBLIC client using Authorization Code + PKCE: the
 * browser holds no credential, and the code challenge is what binds the
 * redirect back to this app. A secret in a SPA bundle is readable by anyone who
 * opens devtools, so `src/auth/noSecrets.test.ts` fails the build if one appears
 * in the compiled output.
 *
 * Configuration is BUILD time, not run time. Vite inlines `VITE_*` into the
 * bundle, so these are set on the Static Web Apps workflow, not on the App
 * Service — a bundle compiled without them contains no path to the feature.
 */

import type { Configuration, PopupRequest, RedirectRequest } from "@azure/msal-browser";

/** The Vite build-time env this module reads. Injectable so it can be tested. */
export interface AuthEnv {
  VITE_MI_BEARER_AUTH?: string;
  VITE_AAD_CLIENT_ID?: string;
  VITE_AAD_AUTHORITY?: string;
  VITE_MI_API_SCOPE?: string;
  PROD?: boolean;
}

/**
 * Work and school accounts in ANY Entra directory, and only those.
 *
 * `/organizations` rather than `/common` (which also admits personal Microsoft
 * accounts, which have no directory and therefore no `tid` for Trakt to resolve
 * an organisation from) and rather than a tenant GUID (which is the single-tenant
 * pin this whole migration exists to remove — it is what makes an external user's
 * sign-in fail with AADSTS50020 no matter how multitenant the App Registration is).
 */
export const DEFAULT_AUTHORITY = "https://login.microsoftonline.com/organizations";

const TRUE = new Set(["1", "true", "yes", "on"]);

function flag(value: string | undefined): boolean {
  return TRUE.has((value ?? "").trim().toLowerCase());
}

export interface AuthConfig {
  /** Whether the bearer path is switched on for this build. */
  enabled: boolean;
  clientId: string;
  authority: string;
  /** The delegated scope requested for the MI API access token. */
  apiScope: string;
  /**
   * True when the flag is on but the build is missing what MSAL needs. The app
   * must not silently fall back to the SWA path in that state: a bundle that
   * says it does bearer auth and then sends no token produces a 401 with no
   * explanation, so this is surfaced instead.
   */
  misconfigured: boolean;
  /** Which required values are absent, for the operator-facing message. */
  missing: string[];
}

/**
 * Resolve the auth configuration from build-time env.
 *
 * Pure and env-injectable, so every branch — flag off, flag on and complete,
 * flag on and half-configured — is unit-testable without a browser or a network.
 */
export function resolveAuthConfig(
  env: AuthEnv = import.meta.env as unknown as AuthEnv,
): AuthConfig {
  const enabled = flag(env.VITE_MI_BEARER_AUTH);
  const clientId = (env.VITE_AAD_CLIENT_ID ?? "").trim();
  const apiScope = (env.VITE_MI_API_SCOPE ?? "").trim();
  const authority = (env.VITE_AAD_AUTHORITY ?? "").trim() || DEFAULT_AUTHORITY;

  const missing: string[] = [];
  if (enabled && !clientId) missing.push("VITE_AAD_CLIENT_ID");
  if (enabled && !apiScope) missing.push("VITE_MI_API_SCOPE");

  return { enabled, clientId, authority, apiScope, missing, misconfigured: missing.length > 0 };
}

/** Whether this build attaches bearer tokens to MI API requests. */
export function isBearerAuthEnabled(
  env: AuthEnv = import.meta.env as unknown as AuthEnv,
): boolean {
  const cfg = resolveAuthConfig(env);
  return cfg.enabled && !cfg.misconfigured;
}

/**
 * The MSAL configuration.
 *
 * `cacheLocation: "sessionStorage"` deliberately: the token cache dies with the
 * tab rather than persisting on a shared machine, and a silent renewal still
 * works within the session because MSAL holds the refresh token there. The
 * redirect URI is the app's own origin — it must match the SPA redirect URI
 * registered on `trakt-react-dashboard`, and registering it as a **SPA**
 * platform (not Web) is what enables the PKCE flow without a secret.
 */
export function buildMsalConfig(cfg: AuthConfig): Configuration {
  return {
    auth: {
      clientId: cfg.clientId,
      authority: cfg.authority,
      // Every tenant's own issuer is valid here — that is the point of a
      // multitenant SPA. MSAL requires the authority host to be known; the
      // public cloud host is built in, so no knownAuthorities entry is needed.
      redirectUri: typeof window !== "undefined" ? window.location.origin : "/",
      postLogoutRedirectUri: typeof window !== "undefined" ? window.location.origin : "/",
      navigateToLoginRequestUrl: true,
    },
    cache: {
      // Tokens die with the tab rather than persisting on a shared machine. A
      // silent renewal still works within the session because MSAL holds the
      // refresh token here.
      cacheLocation: "sessionStorage",
      // The REQUEST state — code verifier, nonce, the `state` value — also goes
      // into a short-lived cookie, and this is not the same thing as storing
      // tokens. That state has to survive a full-page round trip to
      // login.microsoftonline.com and back; when the browser discards it (ITP,
      // strict site-data settings, some in-app browsers), MSAL cannot match the
      // response to the request that started it, so the user signs in
      // successfully at Microsoft and returns with no account — landing back on
      // the sign-in screen with nothing to explain why. The cookie is the
      // documented mitigation, it holds no token, and it is cleared as soon as
      // the redirect is processed.
      storeAuthStateInCookie: true,
    },
    system: {
      // MSAL logs verbosely by default; keep the console usable and never log
      // token material.
      loggerOptions: {
        piiLoggingEnabled: false,
        logLevel: 1, // LogLevel.Error — errors only
        loggerCallback: () => {},
      },
    },
  };
}

/**
 * The scopes for an MI API ACCESS token.
 *
 * Only the API scope goes here, and that is deliberate: mixing `openid`/`profile`
 * with a resource scope makes Entra issue a token for a different audience, and
 * the API rejects it (`aud` is checked against `api://<mi-api-app-id>`). The ID
 * token the SPA gets from sign-in is for the SPA's own session — it is never a
 * credential for the API.
 */
export function apiTokenRequest(cfg: AuthConfig): Pick<RedirectRequest, "scopes"> {
  return { scopes: [cfg.apiScope] };
}

/**
 * The interactive sign-in request. `openid`/`profile` belong HERE — this is the
 * request that establishes who the user is; the resource scope is asked for
 * alongside so consent is collected once rather than twice.
 */
export function loginRequest(cfg: AuthConfig): RedirectRequest | PopupRequest {
  return { scopes: ["openid", "profile", cfg.apiScope] };
}
