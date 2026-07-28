const TOKEN_KEY = "trakt_ops_token";

/** Fired (on window) whenever the API rejects our token, so the app can re-show sign-in. */
export const UNAUTHORIZED_EVENT = "trakt-ops-unauthorized";

export function getToken(): string | null {
  try {
    return window.localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function saveToken(token: string): void {
  try {
    window.localStorage.setItem(TOKEN_KEY, token);
  } catch {
    // Storage unavailable — the session just won't persist.
  }
}

export function clearToken(): void {
  try {
    window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    // ignore
  }
}

export function announceUnauthorized(): void {
  window.dispatchEvent(new Event(UNAUTHORIZED_EVENT));
}
