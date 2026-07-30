import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { MockOpsClient } from "@/api/MockOpsClient";
import { OpsClientProvider } from "@/api/context";
import { ToastProvider } from "@/components/Toast";
import { ReviewDetailScreen } from "./ReviewDetail";

/** Stands in for the workflow page so the redirect target is observable. */
function WorkflowStub() {
  return <p>workflow page</p>;
}

describe("ReviewDetail deep link", () => {
  it("forwards an existing approval link into the delivery's workflow", async () => {
    render(
      <OpsClientProvider client={new MockOpsClient(0)}>
        <ToastProvider>
          <MemoryRouter initialEntries={["/reviews/rev-2001"]}>
            <Routes>
              <Route path="/reviews/:id" element={<ReviewDetailScreen />} />
              <Route path="/workflows/:id" element={<WorkflowStub />} />
            </Routes>
          </MemoryRouter>
        </ToastProvider>
      </OpsClientProvider>,
    );

    // The question is no longer answered on a page of its own: the link lands
    // on the delivery it belongs to, so the operator answers it in context.
    expect(await screen.findByText("workflow page")).toBeInTheDocument();
  });
});
