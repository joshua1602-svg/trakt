import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { MockOpsClient } from "@/api/MockOpsClient";
import { OpsClientProvider } from "@/api/context";
import { ToastProvider } from "@/components/Toast";
import { ReviewDetailScreen } from "./ReviewDetail";

describe("ReviewDetail", () => {
  it("renders the question, options, recommendation and scope explanations", async () => {
    render(
      <OpsClientProvider client={new MockOpsClient(0)}>
        <ToastProvider>
          <MemoryRouter initialEntries={["/reviews/rev-2001"]}>
            <Routes>
              <Route path="/reviews/:id" element={<ReviewDetailScreen />} />
            </Routes>
          </MemoryRouter>
        </ToastProvider>
      </OpsClientProvider>,
    );

    // Question is the headline.
    expect(
      await screen.findByText(/What should it be treated as\?/),
    ).toBeInTheDocument();

    // Options render as radios, with the recommendation pre-selected.
    expect(screen.getAllByText("Market value in local currency").length).toBeGreaterThan(0);
    expect(screen.getByText("Market value in reporting currency")).toBeInTheDocument();
    expect(screen.getByText("Purchase cost")).toBeInTheDocument();
    expect(screen.getByText("Trakt suggests")).toBeInTheDocument();

    // Scope rows with friendly labels and their explanations.
    expect(screen.getByText("This delivery only")).toBeInTheDocument();
    expect(screen.getByText("This portfolio")).toBeInTheDocument();
    expect(screen.getByText("This client")).toBeInTheDocument();
    expect(screen.getByText("All of Trakt")).toBeInTheDocument();
    expect(
      screen.getByText(/remember this answer whenever it prepares reports for this portfolio/),
    ).toBeInTheDocument();
    expect(screen.getByText(/every client, everywhere/)).toBeInTheDocument();

    // Mapping questions offer an amend path.
    expect(screen.getByText("It's something else")).toBeInTheDocument();
  });
});
