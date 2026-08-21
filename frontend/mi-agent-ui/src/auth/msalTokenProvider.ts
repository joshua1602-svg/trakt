/**
 * msalTokenProvider — turning an MSAL instance into an {@link AccessTokenProvider}.
 *
 * The acquisition rule, and why it is this way
 * -------------------------------------------
 * `acquireTokenSilent` on EVERY request, not once at sign-in. MSAL serves a
 * cached access token while it is valid and renews it from the refresh token
 * when it is not, so asking each time is how expiry is handled — there is no
 * timer to get wrong, and a token that expires mid-session is renewed on the
 * next call rather than producing one mystifying 401.
 *
 * Silent acquisition fails for exactly one reason worth reacting to:
 * `InteractionRequiredAuthError` — consent was revoked, the session expired, an
 * MFA/conditional-access policy now applies. That is a redirect to Microsoft,
 * not an error message. Everything else (offline, MSAL not initialised) is a
 * transient failure: return null, let the request go out without a credential,
 * and let the API's 401 drive the UI.
 *
 * Redirect, not popup: popups are blocked by default in enough browsers that a
 * popup-only flow reads as "sign-in silently does nothing".
 */

import {
  InteractionRequiredAuthError,
  type AccountInfo,
  type IPublicClientApplication,
} from "@azure/msal-browser";

import { apiTokenRequest, loginRequest, type AuthConfig } from "./msalConfig";
import type { AccessTokenProvider } from "./tokenProvider";

/**
 * The account this session acts as.
 *
 * MSAL can hold several accounts (a user who has signed in with two directories
 * in the same tab). Silent acquisition against an ambiguous set is a coin flip,
 * so the active account is always set explicitly:
 *
 *   * an account already marked active wins — the user's own choice, kept;
 *   * exactly one known account becomes active automatically;
 *   * several accounts and none active resolves to null, and the UI asks the
 *     user which one rather than guessing on their behalf.
 */
export function resolveActiveAccount(msal: IPublicClientApplication): AccountInfo | null {
  const active = msal.getActiveAccount();
  if (active) return active;
  const accounts = msal.getAllAccounts();
  if (accounts.length === 1) {
    msal.setActiveAccount(accounts[0]);
    return accounts[0];
  }
  return null;
}

/**
 * Build the request-time token source.
 *
 * `onInteractionRequired` is injected so a caller can decide what a re-consent
 * looks like; it defaults to a full-page redirect to Microsoft. It is invoked
 * at most once per failure, and the provider returns null for that request —
 * the page is navigating away, so there is nothing useful to return.
 */
export function createMsalTokenProvider(
  msal: IPublicClientApplication,
  cfg: AuthConfig,
  onInteractionRequired?: (msal: IPublicClientApplication, cfg: AuthConfig) => void,
): AccessTokenProvider {
  const interactive =
    onInteractionRequired ??
    ((instance: IPublicClientApplication, config: AuthConfig) => {
      void instance.acquireTokenRedirect(apiTokenRequest(config) as never);
    });

  return async function getAccessToken(): Promise<string | null> {
    const account = resolveActiveAccount(msal);
    // Not signed in (or not yet resolved to one account) is a normal state, not
    // a failure: the UI renders the sign-in gate and no request should carry a
    // half-formed credential.
    if (!account) return null;

    try {
      const result = await msal.acquireTokenSilent({
        ...apiTokenRequest(cfg),
        account,
      });
      // An empty accessToken is not a token. Returning "" would attach
      // `Authorization: Bearer ` and turn a clear 401 into a malformed request.
      return result.accessToken || null;
    } catch (err) {
      if (err instanceof InteractionRequiredAuthError) {
        interactive(msal, cfg);
        return null;
      }
      // Transient. The caller attaches no header and the API answers 401, which
      // is already a state the UI handles.
      return null;
    }
  };
}

/** Start an interactive sign-in. */
export function signIn(msal: IPublicClientApplication, cfg: AuthConfig): Promise<void> {
  return msal.loginRedirect(loginRequest(cfg) as never);
}

/**
 * Sign in while explicitly choosing an account — the escape hatch for a user
 * with two directories, and the answer when {@link resolveActiveAccount} finds
 * more than one account and refuses to guess.
 */
export function signInWithAccountPicker(
  msal: IPublicClientApplication,
  cfg: AuthConfig,
): Promise<void> {
  return msal.loginRedirect({ ...loginRequest(cfg), prompt: "select_account" } as never);
}

/**
 * Sign out of this application.
 *
 * `logoutRedirect` with the active account clears MSAL's cache and ends the
 * Microsoft session for this app. It does NOT touch the Static Web Apps session
 * cookie: while both paths exist, signing out of one is not signing out of the
 * other, and pretending otherwise would leave a user who clicked "sign out"
 * still admitted through the header path.
 */
export function signOut(msal: IPublicClientApplication): Promise<void> {
  const account = msal.getActiveAccount() ?? undefined;
  return msal.logoutRedirect({ account });
}
