import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import type { IPublicClientApplication } from "@azure/msal-browser";

import { AuthBoundary } from "./AuthBoundary";
import { resolveAuthConfig, type AuthEnv } from "./msalConfig";

const on: AuthEnv = {
  VITE_MI_BEARER_AUTH: "true",
  VITE_AAD_CLIENT_ID: "11111111-2222-3333-4444-555555555555",
  VITE_MI_API_SCOPE: "api://f7473442-de15-49f4-b6da-a9634bcbea8f/MI.Access",
};

const APP = <div data-testid="the-app">dashboard</div>;

/** MSAL stand-in with a controllable account list. */
function msalStub(accounts: unknown[] = []): IPublicClientApplication {
  const logger: Record<string, unknown> = {
    verbose: vi.fn(), info: vi.fn(), warning: vi.fn(), error: vi.fn(), trace: vi.fn(),
    verbosePii: vi.fn(), infoPii: vi.fn(), warningPii: vi.fn(), errorPii: vi.fn(),
    tracePii: vi.fn(), isPiiLoggingEnabled: vi.fn(() => false),
  };
  logger.clone = vi.fn(() => logger);
  return {
    getActiveAccount: vi.fn(() => accounts[0] ?? null),
    setActiveAccount: vi.fn(),
    getAllAccounts: vi.fn(() => accounts),
    getAccount: vi.fn(() => accounts[0] ?? null),
    loginRedirect: vi.fn(async () => undefined),
    logoutRedirect: vi.fn(async () => undefined),
    acquireTokenSilent: vi.fn(async () => ({ accessToken: "t" })),
    addEventCallback: vi.fn(() => "cb-id"),
    removeEventCallback: vi.fn(),
    enableAccountStorageEvents: vi.fn(),
    disableAccountStorageEvents: vi.fn(),
    initialize: vi.fn(async () => undefined),
    handleRedirectPromise: vi.fn(async () => null),
    // MsalProvider clones the logger and logs through the clone, so `clone`
    // must return a logger too — not undefined.
    getLogger: vi.fn(() => logger),
    setLogger: vi.fn(),
    initializeWrapperLibrary: vi.fn(),
    setNavigationClient: vi.fn(),
    getConfiguration: vi.fn(() => ({ auth: { clientId: "x" } })),
  } as unknown as IPublicClientApplication;
}

describe("the feature-flag fallback", () => {
  it("renders the app directly when the flag is off — no gate, no MSAL", () => {
    render(<AuthBoundary config={resolveAuthConfig({})}>{APP}</AuthBoundary>);

    expect(screen.getByTestId("the-app")).toBeInTheDocument();
    expect(screen.queryByTestId("auth-gate")).not.toBeInTheDocument();
  });

  it("renders the app with the flag off even if an MSAL instance is passed", () => {
    render(
      <AuthBoundary config={resolveAuthConfig({})} msal={msalStub()}>{APP}</AuthBoundary>,
    );
    expect(screen.getByTestId("the-app")).toBeInTheDocument();
  });
});

describe("with the flag on", () => {
  it("gates the app behind a Microsoft sign-in when nobody is signed in", () => {
    render(
      <AuthBoundary config={resolveAuthConfig(on)} msal={msalStub([])}>{APP}</AuthBoundary>,
    );

    expect(screen.queryByTestId("the-app")).not.toBeInTheDocument();
    expect(screen.getByTestId("sign-in")).toHaveTextContent(/sign in with microsoft/i);
  });

  it("renders the app once an account is signed in", async () => {
    const account = { homeAccountId: "a", localAccountId: "a", tenantId: "t",
                      environment: "login.microsoftonline.com", username: "u@example.com" };
    render(
      <AuthBoundary config={resolveAuthConfig(on)} msal={msalStub([account])}>{APP}</AuthBoundary>,
    );

    // Awaited deliberately: msal-react reports "not authenticated" until its
    // startup completes (initialize → handleRedirectPromise). Asserting
    // synchronously would pass for the wrong reason today and hide a real
    // regression tomorrow — and this is the same window main.tsx waits out
    // before its first render, which is what stops the sign-in looping.
    expect(await screen.findByTestId("the-app")).toBeInTheDocument();
  });

  it("offers an account picker only when more than one account is known", () => {
    render(
      <AuthBoundary config={resolveAuthConfig(on)} msal={msalStub([])}>{APP}</AuthBoundary>,
    );
    expect(screen.queryByTestId("choose-account")).not.toBeInTheDocument();
  });

  it("refuses to render the app when the build is missing its configuration", () => {
    // A bundle that claims to authenticate and cannot must not fall through to
    // the app: every request would go out bare and 401 with no explanation.
    render(
      <AuthBoundary config={resolveAuthConfig({ VITE_MI_BEARER_AUTH: "true" })}>{APP}</AuthBoundary>,
    );

    expect(screen.queryByTestId("the-app")).not.toBeInTheDocument();
    expect(screen.getByTestId("auth-gate")).toHaveTextContent("VITE_AAD_CLIENT_ID");
  });

  it("refuses to render the app when MSAL failed to initialise", () => {
    render(
      <AuthBoundary config={resolveAuthConfig(on)} msal={null}>{APP}</AuthBoundary>,
    );

    expect(screen.queryByTestId("the-app")).not.toBeInTheDocument();
    expect(screen.getByTestId("auth-gate")).toBeInTheDocument();
  });

  it("does not blame the build configuration when the configuration is fine", () => {
    // The two failures have different causes and different fixes. Saying
    // "compiled without …" for an initialisation failure sends whoever is on
    // the deployment to check workflow variables that are demonstrably present.
    render(
      <AuthBoundary config={resolveAuthConfig(on)} msal={null}>{APP}</AuthBoundary>,
    );

    const gate = screen.getByTestId("auth-gate");
    expect(gate).toHaveTextContent(/could not start/i);
    expect(gate).not.toHaveTextContent(/compiled without/i);
    expect(gate).not.toHaveTextContent(/VITE_/);
  });

  it("shows the underlying reason so it can be diagnosed without devtools", () => {
    render(
      <AuthBoundary config={resolveAuthConfig(on)} msal={null}
                    initError="invalid_client_id: The client id is not a valid GUID">
        {APP}
      </AuthBoundary>,
    );

    expect(screen.getByTestId("auth-gate")).toHaveTextContent("invalid_client_id");
    expect(screen.getByTestId("retry-init")).toBeInTheDocument();
  });
});
