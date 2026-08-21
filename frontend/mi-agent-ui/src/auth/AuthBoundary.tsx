/**
 * AuthBoundary — the sign-in gate, and the flag that keeps it out of the way.
 *
 * With `VITE_MI_BEARER_AUTH` off this component renders its children and
 * NOTHING else: no MsalProvider, no context, no effect, no token. That is the
 * property that makes Phase 3 safe to ship dark — the existing Static Web Apps
 * path is not "disabled", it is untouched, because none of this code runs.
 *
 * With the flag on, the app is gated on a signed-in Microsoft account. Being
 * signed in is NOT the same as having Trakt access: the API still resolves the
 * organisation and the named principal from the verified token, and can refuse
 * a perfectly valid Microsoft user. That refusal surfaces where it belongs — in
 * the app's own error handling for a 401/403 — not here.
 */

import { type ReactNode } from "react";
import { MsalProvider, useIsAuthenticated, useMsal } from "@azure/msal-react";
import type { IPublicClientApplication } from "@azure/msal-browser";

import { resolveAuthConfig, type AuthConfig } from "./msalConfig";
import { signIn, signInWithAccountPicker, signOut } from "./msalTokenProvider";

export interface AuthBoundaryProps {
  children: ReactNode;
  /** The initialised MSAL instance. Absent when the flag is off. */
  msal?: IPublicClientApplication | null;
  /** Injectable for tests; defaults to the build-time configuration. */
  config?: AuthConfig;
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div
      className="flex min-h-screen items-center justify-center bg-navy-950 p-6"
      data-testid="auth-gate"
    >
      <div className="w-full max-w-md rounded-xl border border-[var(--color-line)] bg-navy-900/60 p-8 text-center">
        <h1 className="mb-2 text-lg font-medium text-white">{title}</h1>
        {children}
      </div>
    </div>
  );
}

/** The signed-out screen. */
function SignInGate({ config }: { config: AuthConfig }) {
  const { instance } = useMsal();
  const several = instance.getAllAccounts().length > 1;
  return (
    <Panel title="Trakt MI">
      <p className="mb-6 text-sm text-slate-300">
        Sign in with your organisation&rsquo;s Microsoft account.
      </p>
      <button
        type="button"
        data-testid="sign-in"
        onClick={() => void signIn(instance, config)}
        className="w-full rounded-lg bg-[var(--color-accent,#919dd1)] px-4 py-2 text-sm font-medium text-navy-950"
      >
        Sign in with Microsoft
      </button>
      {/* Only offered when it is actually needed: MSAL holds more than one
          account, so nothing can safely pick for the user. */}
      {several && (
        <button
          type="button"
          data-testid="choose-account"
          onClick={() => void signInWithAccountPicker(instance, config)}
          className="mt-3 w-full rounded-lg border border-[var(--color-line)] px-4 py-2 text-sm text-slate-300"
        >
          Use a different account
        </button>
      )}
    </Panel>
  );
}

/** Flag on but the build lacks a client id or an API scope. */
function MisconfiguredGate({ config }: { config: AuthConfig }) {
  return (
    <Panel title="Sign-in is not configured">
      <p className="text-sm text-slate-300">
        This build has bearer authentication switched on but was compiled without{" "}
        <code className="text-slate-100">{config.missing.join(", ")}</code>. These are
        build-time values — set them on the Static Web Apps workflow and redeploy.
      </p>
    </Panel>
  );
}

function Gate({ children, config }: { children: ReactNode; config: AuthConfig }) {
  const isAuthenticated = useIsAuthenticated();
  return isAuthenticated ? <>{children}</> : <SignInGate config={config} />;
}

export function AuthBoundary({ children, msal, config }: AuthBoundaryProps) {
  const cfg = config ?? resolveAuthConfig();

  // The whole feature, switched off in one branch.
  if (!cfg.enabled) return <>{children}</>;

  if (cfg.misconfigured) return <MisconfiguredGate config={cfg} />;

  // Flag on, configured, but bootstrap did not hand us an instance. Refusing is
  // the honest outcome: rendering the app would send unauthenticated requests
  // from a build that claims to authenticate.
  if (!msal) return <MisconfiguredGate config={{ ...cfg, missing: ["an initialised MSAL instance"] }} />;

  return (
    <MsalProvider instance={msal}>
      <Gate config={cfg}>{children}</Gate>
    </MsalProvider>
  );
}

/** Sign-out action for the header, exported so the shell need not know about MSAL. */
export function useSignOut(): () => void {
  const { instance } = useMsal();
  return () => void signOut(instance);
}
