import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { MockOpsClient } from "@/api/MockOpsClient";
import { OpsClientProvider } from "@/api/context";
import { ToastProvider } from "@/components/Toast";
import { decisionHref } from "@/lib/ids";
import { ReviewsScreen } from "./Reviews";

describe("review queue", () => {
  it("opens the delivery a question belongs to, not an isolated form", async () => {
    render(
      <OpsClientProvider client={new MockOpsClient(0)}>
        <ToastProvider>
          <MemoryRouter initialEntries={["/reviews"]}>
            <ReviewsScreen />
          </MemoryRouter>
        </ToastProvider>
      </OpsClientProvider>,
    );

    await waitFor(() => expect(document.querySelectorAll("a").length).toBeGreaterThan(0));
    const links = Array.from(document.querySelectorAll("a")).map((a) => a.getAttribute("href"));
    expect(links.length).toBeGreaterThan(0);
    for (const href of links) {
      expect(href).toMatch(/^\/workflows\//);
      // The question is carried through, so the workflow opens on it.
      expect(href).toContain("decision=");
    }
    expect(links.some((href) => href?.startsWith("/reviews/"))).toBe(false);
  });

  it("still answers a question raised before any delivery exists", async () => {
    // A file that has to be identified is held against the input pack, which
    // has no case file yet — so it keeps its own page rather than pointing at
    // a workflow that does not exist.
    expect(decisionHref("dec-1", "batch_7001")).toBe("/reviews/dec-1");
    expect(decisionHref("dec-1", "wf-1001")).toBe(
      "/workflows/wf-1001?decision=dec-1",
    );
    expect(screen.queryByText("Nothing to review right now.")).toBeNull();
  });
});
