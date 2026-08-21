/**
 * tokenProvider — the one seam between MSAL and the API client.
 *
 * `HttpAgentClient` is constructed by `createAgentClient()` (src/api/index.ts),
 * which `AppShell` calls in a `useMemo`. Threading a React context through that
 * would mean rebuilding the client whenever the auth state changed — a new
 * client means a cold result cache and a refetch of every panel. So the token
 * source is registered once at bootstrap and read by the client at REQUEST time
 * instead: the client keeps its identity, and each call gets whatever token is
 * current.
 *
 * A module-level registry rather than a global: it is set in `main.tsx`, read in
 * `createAgentClient()`, and reset in tests through {@link setAccessTokenProvider}.
 *
 * When nothing is registered — which is every build with `VITE_MI_BEARER_AUTH`
 * off — `getAccessTokenProvider()` returns null, the client attaches no
 * Authorization header, and requests go out exactly as they do today.
 */

/**
 * Returns a bearer token for the MI API, or null when the caller is not signed
 * in. Never throws for "not signed in" — that is a normal state the UI renders,
 * not an error. It MAY throw when acquisition genuinely fails (network down,
 * consent revoked mid-session); the client turns that into a normal request
 * failure rather than a crash.
 */
export type AccessTokenProvider = () => Promise<string | null>;

let provider: AccessTokenProvider | null = null;

/** Register the process-wide token source. Pass null to clear it (tests). */
export function setAccessTokenProvider(next: AccessTokenProvider | null): void {
  provider = next;
}

/** The registered token source, or null when this build does not use one. */
export function getAccessTokenProvider(): AccessTokenProvider | null {
  return provider;
}

/**
 * `{ Authorization: "Bearer …" }` when a token is available, `{}` otherwise.
 *
 * Every MI API request goes through this, so there is exactly one place where a
 * credential is attached — and one place to look when asking whether a given
 * request carries one. A provider that throws is treated as "no token": the
 * request then goes out unauthenticated and the API answers 401, which the UI
 * already handles, rather than the failure surfacing as an unhandled rejection
 * inside a data-loading effect.
 */
export async function authorizationHeaders(): Promise<Record<string, string>> {
  const current = provider;
  if (!current) return {};
  try {
    const token = await current();
    return token ? { Authorization: `Bearer ${token}` } : {};
  } catch {
    return {};
  }
}
