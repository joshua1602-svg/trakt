import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import type { ReactNode } from "react";
import { MockOpsClient } from "@/api/MockOpsClient";
import type { OpsClient } from "@/api/OpsClient";
import { OpsClientProvider } from "@/api/context";
import { SessionProvider } from "@/api/session";
import { ToastProvider } from "@/components/Toast";

/** An instant mock client signed in with the given role. */
export function mockClient(role: "admin" | "operator" = "admin"): MockOpsClient {
  return new MockOpsClient(0, role);
}

/** Renders a screen inside the same providers `App` supplies. */
export function renderAdmin(
  ui: ReactNode,
  client: OpsClient = mockClient(),
  { route = "/admin/config" }: { route?: string } = {},
) {
  return render(
    <OpsClientProvider client={client}>
      <ToastProvider>
        <SessionProvider>
          <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
        </SessionProvider>
      </ToastProvider>
    </OpsClientProvider>,
  );
}
