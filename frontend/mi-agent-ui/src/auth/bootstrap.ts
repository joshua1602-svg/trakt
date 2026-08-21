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
 */

import { PublicClientApplication, type IPublicClientApplication } from "@azure/msal-browser";

import { buildMsalConfig, resolveAuthConfig, type AuthConfig } from "./msalConfig";
import { createMsalTokenProvider, resolveActiveAccount } from "./msalTokenProvider";
import { setAccessTokenProvider } from "./tokenProvider";

export interface Bootstrapped {
  msal: IPublicClientApplication | null;
  config: AuthConfig;
}

/**
 * Initialise MSAL and register the token provider, or do nothing at all.
 *
 * Returns `msal: null` when the flag is off or the build is missing its
 * configuration — in both cases no token provider is registered, so
 * `HttpAgentClient` attaches no Authorization header and the Static Web Apps
 * path is exactly as it was.
 *
 * A failure to initialise is deliberately not fatal to the module: it returns a
 * null instance, `AuthBoundary` renders its "not configured" panel, and the
 * console carries the reason. A thrown error here would leave a blank page with
 * nothing to read.
 */
export async function bootstrapAuth(
  config: AuthConfig = resolveAuthConfig(),
): Promise<Bootstrapped> {
  if (!config.enabled || config.misconfigured) return { msal: null, config };

  try {
    const msal = new PublicClientApplication(buildMsalConfig(config));
    await msal.initialize();

    // Step 2: consume the redirect. Returns null on a normal page load.
    const result = await msal.handleRedirectPromise();
    if (result?.account) {
      msal.setActiveAccount(result.account);
    } else {
      resolveActiveAccount(msal);
    }

    setAccessTokenProvider(createMsalTokenProvider(msal, config));
    return { msal, config };
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error("[MI Agent] Microsoft sign-in could not be initialised:", err);
    return { msal: null, config };
  }
}
