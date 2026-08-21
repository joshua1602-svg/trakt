/**
 * bootstrap — bringing MSAL up before the app renders, when the flag is on.
 *
 * Three things must happen in order, and getting the order wrong is the classic
 * source of a login that loops:
 *
 *   1. `initialize()` — required by msal-browser v3+; every other call throws
 *      until it resolves.
 *   2. `handleRedirectPromise()` — consumes the `#code=…` fragment Microsoft
 *      just sent back. Skipping it leaves the fragment in the URL, so the app
 *      renders signed-out, redirects to Microsoft again, and comes back to the
 *      same state. That is a redirect loop, and it is caused by the app, not by
 *      Entra.
 *   3. Set the active account, from the redirect response when there is one, so
 *      the first silent token acquisition has an unambiguous account.
 *
 * Only then is the token provider registered and the app allowed to render.
 *
 * Two kinds of failure, and they are NOT the same
 * -----------------------------------------------
 * This used to wrap all three steps in one try/catch and return no instance on
 * any error, which turned a transient condition into a dead end: an origin that
 * had previously bounced through Easy Auth can carry a stale hash fragment, and
 * a leftover `interaction_in_progress` lock survives in session storage — both
 * make step 2 reject, neither says anything about whether MSAL works. The page
 * then showed "sign-in is not configured" forever, and no amount of reloading
 * fixed it because the fragment came back every time.
 *
 * So the two are separated:
 *
 *   * construction or `initialize()` failing means there is genuinely no usable
 *     instance — the configuration is wrong (a client id that is not a GUID is
 *     the usual cause) and the app must say so;
 *   * `handleRedirectPromise()` failing means THIS redirect could not be
 *     processed. The instance is fine. Log it, clear the fragment so the next
 *     attempt starts clean, and let the user sign in again.
 */

import { PublicClientApplication, type IPublicClientApplication } from "@azure/msal-browser";

import { buildMsalConfig, resolveAuthConfig, type AuthConfig } from "./msalConfig";
import { createMsalTokenProvider, resolveActiveAccount } from "./msalTokenProvider";
import { setAccessTokenProvider } from "./tokenProvider";

export interface Bootstrapped {
  msal: IPublicClientApplication | null;
  config: AuthConfig;
  /**
   * Why there is no instance, when there is none. Surfaced in the UI so the
   * reason is on the page rather than only in the console — a deployed SPA is
   * usually debugged by someone who cannot open devtools on the machine that
   * hit the problem.
   */
  error?: string;
  /**
   * Why THIS sign-in did not take, when MSAL is fine but the user came back
   * signed out. Without it the app returns to the sign-in screen saying nothing,
   * and the user presses the button again — the exact loop reported after the
   * first bearer deployment. The reason is on the page for the same reason as
   * above, and because this is the failure a user (not a developer) hits.
   */
  signInError?: string;
}

function describe(err: unknown): string {
  if (err instanceof Error) {
    // MSAL errors carry an errorCode that is far more actionable than the
    // message ("invalid_client_id", "redirect_uri_mismatch").
    const code = (err as { errorCode?: string }).errorCode;
    return code ? `${code}: ${err.message}` : err.message;
  }
  return String(err);
}

/**
 * Initialise MSAL and register the token provider, or do nothing at all.
 *
 * Returns `msal: null` when the flag is off or the build is missing its
 * configuration — in both cases no token provider is registered, so
 * `HttpAgentClient` attaches no Authorization header and the Static Web Apps
 * path is exactly as it was.
 */
export async function bootstrapAuth(
  config: AuthConfig = resolveAuthConfig(),
): Promise<Bootstrapped> {
  if (!config.enabled || config.misconfigured) return { msal: null, config };

  let msal: IPublicClientApplication;
  try {
    msal = new PublicClientApplication(buildMsalConfig(config));
    await msal.initialize();
  } catch (err) {
    // No instance exists, so there is nothing to sign in with. This is a
    // configuration fault and the page has to say so.
    // eslint-disable-next-line no-console
    console.error("[MI Agent] Microsoft sign-in could not be initialised:", err);
    return { msal: null, config, error: describe(err) };
  }

  // Whether Microsoft actually sent us back with a response to process. Captured
  // BEFORE the promise runs, because handling it clears the fragment: it is the
  // difference between "a normal page load, nobody has signed in yet" and "a
  // sign-in came back and produced no account", which look identical afterwards
  // and need completely different messages.
  const returningFromSignIn =
    typeof window !== "undefined" && /[#&](code|error|id_token)=/.test(window.location.hash);
  let signInError: string | undefined;

  try {
    // Step 2: consume the redirect. Returns null on a normal page load.
    const result = await msal.handleRedirectPromise();
    if (result?.account) {
      msal.setActiveAccount(result.account);
    } else {
      resolveActiveAccount(msal);
      if (returningFromSignIn && !msal.getActiveAccount()) {
        // Came back from Microsoft, no error thrown, and still no account. The
        // usual cause is auth state that did not survive the round trip, so the
        // response could not be matched to the request that started it.
        signInError =
          "Microsoft returned a sign-in response, but no account was established. "
          + "This usually means the browser discarded the sign-in state during the "
          + "redirect (private browsing, or blocked site data).";
        // eslint-disable-next-line no-console
        console.error("[MI Agent] redirect processed but no account resulted");
      }
    }
  } catch (err) {
    // This redirect failed; MSAL did not. Drop the fragment so a reload — or
    // the user pressing "sign in" again — starts from a clean URL instead of
    // reprocessing the same broken response forever.
    // eslint-disable-next-line no-console
    console.error("[MI Agent] this sign-in response could not be processed:", err);
    signInError = describe(err);
    if (typeof window !== "undefined" && window.location.hash) {
      window.history.replaceState(null, "", window.location.pathname + window.location.search);
    }
    resolveActiveAccount(msal);
  }

  setAccessTokenProvider(createMsalTokenProvider(msal, config));
  return { msal, config, signInError };
}
