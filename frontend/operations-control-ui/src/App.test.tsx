import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { copy } from "@/lib/copy";
import App from "./App";

/**
 * Composition tests for the application root.
 *
 * These exist because a merge once left `App.tsx` importing SessionProvider,
 * AdminOnly and every configuration screen while its <Routes> body still had
 * none of them. Nothing in the suite rendered `App`, so the whole feature was
 * unreachable and `Shell`'s useSession() would have thrown at first render —
 * a blank page — with a green test run. Only `noUnusedLocals` caught it, and
 * only because the imports happened to survive; deleting them instead would
 * have produced a clean compile and a broken app.
 *
 * So: render the real root and assert the wiring, not just the types.
 */

function renderApp(route: string) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <App />
    </MemoryRouter>,
  );
}

describe("App composition", () => {
  beforeEach(() => {
    // Mock mode makes createOpsClient() return MockOpsClient and skips sign-in,
    // so these tests exercise routing without a backend or a stored token.
    vi.stubEnv("VITE_OPS_MODE", "mock");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("renders without a missing-provider crash", async () => {
    // Shell calls useSession(), which throws when SessionProvider is not above
    // it. If the provider is ever dropped from App again, this fails loudly.
    renderApp("/");
    expect(await screen.findByText(copy.home.subtitle)).toBeInTheDocument();
  });

  it("offers the administrator entry once the principal is known", async () => {
    renderApp("/");
    expect((await screen.findAllByText(copy.nav.admin)).length).toBeGreaterThan(0);
  });

  it("routes /admin/config to the configuration overview", async () => {
    renderApp("/admin/config");
    expect(await screen.findByText(copy.admin.subtitle)).toBeInTheDocument();
    expect(await screen.findByText("System package")).toBeInTheDocument();
  });

  it("routes each administrator configuration page", async () => {
    for (const [route, heading] of [
      ["/admin/config/system", copy.admin.system.title],
      ["/admin/config/assets", copy.admin.assets.title],
      ["/admin/config/regimes", copy.admin.regimes.title],
      ["/admin/config/history", copy.admin.audit.subtitle],
    ] as const) {
      const view = renderApp(route);
      // findAllByText: some pages legitimately show their title twice (a section
      // heading plus the package workspace heading). The assertion is that the
      // route reached its page at all.
      expect(
        (await screen.findAllByText(heading)).length,
        `${route} did not render its page`,
      ).toBeGreaterThan(0);
      view.unmount();
    }
  });

  it("still routes the operator pages", async () => {
    const view = renderApp("/workflows");
    expect(await screen.findByText(copy.workflows.subtitle)).toBeInTheDocument();
    view.unmount();

    renderApp("/rules");
    expect(await screen.findByText(copy.rules.subtitle)).toBeInTheDocument();
  });
});
