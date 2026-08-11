import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CopilotDemo } from "@/components/demo/CopilotDemo";
import { Hero } from "@/components/site/Hero";
import { LeadForm } from "@/components/site/LeadForm";
import { Nav } from "@/components/site/Nav";
import type { DemoMetaResponse } from "@/types/demo";

const META: DemoMetaResponse = {
  scope: {
    client: "Alderbridge Lending Platform",
    clientDescription: "A synthetic UK equity release lender.",
    portfolio: "ALP_Platform_202606",
    portfolioName: "Equity Release Portfolio",
    assetClass: "UK equity release mortgages",
    asOfDate: "2026-06-30",
    asOfDisplay: "30 November 2025",
    loanCount: 36,
    totalBalanceDisplay: "£37,270,061",
    platformBalanceDisplay: "£27,407,089",
    books: [
      { id: "alp_origination", label: "ALP Origination Book", shortLabel: "Origination",
        balanceSheetStatus: "warehoused", statusDetail: "Warehoused, destined for SPV2",
        balanceDisplay: "£15,432,544" },
      { id: "alp_acquired", label: "ALP Acquired Back Book", shortLabel: "Acquired",
        balanceSheetStatus: "warehoused", statusDetail: "Warehoused, destined for SPV2",
        balanceDisplay: "£11,974,544" },
      { id: "spv1_sponsored", label: "SPV1 Sponsored Securitisation", shortLabel: "SPV1",
        balanceSheetStatus: "sold",
        statusDetail: "Sold and derecognised; servicing and reporting retained",
        balanceDisplay: "£9,862,973" },
    ],
    currency: "GBP",
    regulatoryRegime: "ESMA Annex 2",
    synthetic: true,
  },
  suggestedQuestions: [
    { id: "funded_balance", label: "What is the current funded portfolio balance?", category: "portfolio_kpi" },
    { id: "region_exposure", label: "Which regions have the highest exposure?", category: "concentration" },
  ],
  reportActions: [
    { id: "investor_report", label: "Generate the latest investor report.", documentTitle: "Investor & Funder MI Pack" },
  ],
  exampleUnsupported: [
    { id: "temporal_movement", label: "How has the portfolio changed since last month?" },
  ],
  limits: { questionsPerSession: 12, reportsPerSession: 3, maxQuestionLength: 240 },
  provenanceNote: "Produced by the Trakt deterministic MI engine.",
};

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

beforeEach(() => {
  // jsdom implements neither of these; the components call them defensively.
  Element.prototype.scrollIntoView = vi.fn();
  window.matchMedia = vi.fn().mockReturnValue({
    matches: false,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  }) as unknown as typeof window.matchMedia;
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Hero", () => {
  it("renders the proposition, the trust proof points and both CTAs", () => {
    render(<Hero scope={META.scope} />);

    expect(
      screen.getByRole("heading", {
        level: 1,
        name: /one governed view of your lending portfolios\./i,
      }),
    ).toBeInTheDocument();
    // The strongest sentence on the page leads, rather than being buried.
    expect(
      screen.getByText(/reconciled by construction rather than by comparison/i),
    ).toBeInTheDocument();
    for (const proof of [/deterministic engine/i, /traceable lineage/i, /client-isolated/i]) {
      expect(screen.getByText(proof)).toBeInTheDocument();
    }
    expect(screen.getByRole("link", { name: /explore the live demo/i })).toHaveAttribute(
      "href",
      "#example",
    );
    expect(screen.getByRole("link", { name: /book a portfolio walkthrough/i })).toHaveAttribute(
      "href",
      "#book-a-demo",
    );
    // The card leads on the differentiator, with this platform's real figures:
    // three books, and both governed totals rather than one flat number.
    for (const label of ["Origination", "Acquired", "SPV1"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
    expect(screen.getByText("sold")).toBeInTheDocument();
    expect(screen.getByText(/platform total/i)).toBeInTheDocument();
    expect(screen.getByText("£27,407,089")).toBeInTheDocument();
    expect(screen.getByText(/sponsor total/i)).toBeInTheDocument();
    expect(screen.getByText("£37,270,061")).toBeInTheDocument();
  });
});

describe("Nav", () => {
  it("exposes the required destinations", () => {
    render(<Nav />);
    const nav = screen.getByRole("navigation", { name: /primary/i });
    for (const label of ["Platform", "Controls", "Onboarding", "Intelligence"]) {
      expect(within(nav).getByRole("link", { name: label })).toBeInTheDocument();
    }
    expect(within(nav).getByRole("link", { name: /book a demo/i })).toBeInTheDocument();
  });

  it("opens and closes the mobile menu", async () => {
    const user = userEvent.setup();
    render(<Nav />);

    const toggle = screen.getByRole("button", { name: /open menu/i });
    expect(toggle).toHaveAttribute("aria-expanded", "false");

    await user.click(toggle);
    expect(screen.getByRole("button", { name: /close menu/i })).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    expect(document.getElementById("mobile-nav")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /close menu/i }));
    expect(document.getElementById("mobile-nav")).not.toBeInTheDocument();
  });
});

describe("CopilotDemo", () => {
  it("shows the synthetic scope and the suggested questions before any interaction", () => {
    render(<CopilotDemo meta={META} />);

    expect(screen.getByText("Alderbridge Lending Platform")).toBeInTheDocument();
    expect(screen.getByText("Synthetic data")).toBeInTheDocument();
    expect(screen.getByText(/36 exposures/)).toBeInTheDocument();
    expect(screen.getByText(/30 November 2025/)).toBeInTheDocument();

    for (const question of META.suggestedQuestions) {
      expect(screen.getByRole("button", { name: question.label })).toBeInTheDocument();
    }
    expect(
      screen.getByRole("button", { name: /generate the latest investor report/i }),
    ).toBeInTheDocument();
  });

  it("displays a supported answer with its metrics and provenance footer", async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          status: "answered",
          intentId: "funded_balance",
          question: "What is the current funded portfolio balance?",
          answer: "The funded book stands at £5.4MM across 36 exposures.",
          interpreted: "Metric: Balance · Aggregation: Sum",
          artifacts: [
            {
              kind: "kpi",
              title: "Portfolio summary",
              kpis: [{ label: "Current outstanding balance", value: "£5.4MM" }],
              coverage: 100,
            },
          ],
          asOfDate: "2026-06-30",
          asOfDisplay: "30 November 2025",
          portfolioScope: "Alderbridge Lending Platform · ALP_Platform_202606",
          followUps: [{ id: "region_exposure", label: "Which regions have the highest exposure?" }],
          synthetic: true,
          usage: { questionsUsed: 1, questionsRemaining: 11 },
        }),
      ),
    );

    render(<CopilotDemo meta={META} />);
    await user.click(
      screen.getByRole("button", { name: "What is the current funded portfolio balance?" }),
    );

    await waitFor(() =>
      expect(screen.getByText(/the funded book stands at/i)).toBeInTheDocument(),
    );
    expect(screen.getByText("Current outstanding balance")).toBeInTheDocument();
    expect(screen.getByText("£5.4MM")).toBeInTheDocument();
    expect(screen.getByText(/Metric: Balance/)).toBeInTheDocument();
    expect(screen.getByText("As at 30 November 2025")).toBeInTheDocument();
    expect(screen.getByText("Synthetic portfolio")).toBeInTheDocument();
    // The session counter stays silent this early — it is a cap, not a meter.
    expect(screen.queryByText(/questions remaining/i)).not.toBeInTheDocument();
    expect(screen.getByText(/trakt declines what it cannot derive/i))
      .toBeInTheDocument();
  });

  it("shows a clear message for an unsupported question", async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          status: "unsupported",
          intentId: "temporal_movement",
          question: "How has the portfolio changed since last month?",
          answer: "Temporal comparison needs two governed reporting periods.",
          productionNote: "In a client environment, Trakt holds a governed snapshot history.",
          asOfDate: "2026-06-30",
          asOfDisplay: "30 November 2025",
          portfolioScope: "Alderbridge Lending Platform · ALP_Platform_202606",
          followUps: [],
          synthetic: true,
          usage: { questionsUsed: 1, questionsRemaining: 11 },
        }),
      ),
    );

    render(<CopilotDemo meta={META} />);
    await user.click(
      screen.getByRole("button", { name: /how has the portfolio changed since last month/i }),
    );

    await waitFor(() =>
      expect(screen.getByText(/not supported in this demonstration/i)).toBeInTheDocument(),
    );
    expect(screen.getByText(/two governed reporting periods/i)).toBeInTheDocument();
    expect(screen.getByText(/governed snapshot history/i)).toBeInTheDocument();
  });

  it("resets the conversation", async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          status: "answered",
          intentId: "funded_balance",
          question: "q",
          answer: "The funded book stands at £5.4MM.",
          interpreted: null,
          artifacts: [],
          asOfDisplay: "30 November 2025",
          portfolioScope: "scope",
          followUps: [],
          synthetic: true,
          usage: { questionsUsed: 1, questionsRemaining: 11 },
        }),
      ),
    );

    render(<CopilotDemo meta={META} />);
    await user.click(
      screen.getByRole("button", { name: "What is the current funded portfolio balance?" }),
    );
    await waitFor(() => expect(screen.getByText(/the funded book stands at/i)).toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: /reset/i }));
    expect(screen.queryByText(/the funded book stands at/i)).not.toBeInTheDocument();
    // Suggestions survive the reset.
    expect(
      screen.getByRole("button", { name: "What is the current funded portfolio balance?" }),
    ).toBeInTheDocument();
  });

  it("submits a typed question", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        status: "answered",
        intentId: "wa_ltv",
        question: "average ltv",
        answer: "Weighted average current LTV is 43.5%.",
        interpreted: null,
        artifacts: [],
        asOfDisplay: "30 November 2025",
        portfolioScope: "scope",
        followUps: [],
        synthetic: true,
        usage: { questionsUsed: 1, questionsRemaining: 11 },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<CopilotDemo meta={META} />);
    await user.type(screen.getByLabelText(/ask a question/i), "average ltv");
    await user.click(screen.getByRole("button", { name: /ask trakt/i }));

    await waitFor(() => expect(screen.getByText(/weighted average current ltv/i)).toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/demo/query",
      expect.objectContaining({ body: JSON.stringify({ question: "average ltv" }) }),
    );
  });

  it("surfaces a server error without breaking the demo", async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ error: "Too many requests." }, 429)),
    );

    render(<CopilotDemo meta={META} />);
    await user.click(screen.getByRole("button", { name: /which regions have the highest exposure/i }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/too many requests/i));
    // The input is usable again.
    expect(screen.getByLabelText(/ask a question/i)).toBeEnabled();
  });
});

describe("CopilotDemo — accessibility", () => {
  it("announces loading and success through a polite status region", async () => {
    const user = userEvent.setup();
    let resolve: ((value: Response) => void) | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => new Promise<Response>((r) => (resolve = r))),
    );

    render(<CopilotDemo meta={META} />);
    const status = screen.getByRole("status");
    expect(status).toHaveClass("sr-only");

    await user.click(
      screen.getByRole("button", { name: "What is the current funded portfolio balance?" }),
    );
    await waitFor(() => expect(status).toHaveTextContent(/working on your question/i));

    resolve?.(
      jsonResponse({
        status: "answered",
        intentId: "funded_balance",
        question: "q",
        answer: "The funded book stands at £5.4MM.",
        interpreted: null,
        artifacts: [],
        asOfDisplay: "30 November 2025",
        portfolioScope: "scope",
        followUps: [],
        synthetic: true,
        usage: { questionsUsed: 1, questionsRemaining: 11 },
      }),
    );

    await waitFor(() => expect(status).toHaveTextContent(/answer ready/i));
  });

  it("announces a refusal without repeating the visible wording", async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          status: "unsupported",
          intentId: "temporal_movement",
          question: "q",
          answer: "Temporal comparison needs two governed reporting periods.",
          productionNote: "In a production Trakt environment…",
          asOfDisplay: "30 November 2025",
          portfolioScope: "scope",
          followUps: [],
          synthetic: true,
          usage: { questionsUsed: 1, questionsRemaining: 11 },
        }),
      ),
    );

    render(<CopilotDemo meta={META} />);
    await user.click(
      screen.getByRole("button", { name: /how has the portfolio changed since last month/i }),
    );

    const status = screen.getByRole("status");
    await waitFor(() => expect(status).toHaveTextContent(/declined that question/i));
    // The visible heading says something different, so nothing is read twice.
    expect(status).not.toHaveTextContent(/not supported in this demonstration/i);
  });

  it("announces an error so a screen-reader user is not left waiting", async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ error: "Too many requests." }, 429)),
    );

    render(<CopilotDemo meta={META} />);
    await user.click(screen.getByRole("button", { name: /which regions have the highest exposure/i }));

    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent(/too many requests/i),
    );
  });

  it("keeps the transcript out of the live region", () => {
    render(<CopilotDemo meta={META} />);
    // Only the short status region is live; a long answer is not read wholesale.
    const live = document.querySelectorAll("[aria-live]");
    expect(live).toHaveLength(1);
    expect(live[0]).toHaveClass("sr-only");
  });
});

describe("Nav — keyboard behaviour", () => {
  it("moves focus into the menu, traps Tab and restores focus on Escape", async () => {
    const user = userEvent.setup();
    render(<Nav />);

    const toggle = screen.getByRole("button", { name: /open menu/i });
    await user.click(toggle);

    const menu = document.getElementById("mobile-nav");
    expect(menu).toBeInTheDocument();
    // Focus lands on the first item in the menu, not left on the page behind.
    await waitFor(() => expect(menu).toContainElement(document.activeElement as HTMLElement));

    await user.keyboard("{Escape}");
    expect(document.getElementById("mobile-nav")).not.toBeInTheDocument();
    // And returns to the control that opened it.
    expect(screen.getByRole("button", { name: /open menu/i })).toHaveFocus();
  });
});

describe("LeadForm", () => {
  it("labels every field, requires only three, and requires consent", () => {
    render(<LeadForm />);
    for (const label of [/^name/i, /work email/i, /^company/i, /^role/i, /message/i]) {
      expect(screen.getByLabelText(label)).toBeInTheDocument();
    }
    for (const label of [/^name/i, /work email/i, /^company/i]) {
      expect(screen.getByLabelText(label)).toBeRequired();
    }
    for (const label of [/^role/i, /message/i]) {
      expect(screen.getByLabelText(label)).not.toBeRequired();
    }
    expect(screen.getByRole("checkbox")).toBeRequired();
    expect(screen.getByRole("button", { name: /book a tailored demonstration/i })).toBeInTheDocument();
  });

  it("blocks submission and shows field errors when invalid", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(<LeadForm />);
    await user.type(screen.getByLabelText(/work email/i), "not-an-email");
    await user.click(screen.getByRole("button", { name: /book a tailored demonstration/i }));

    expect(await screen.findByText(/please enter your name/i)).toBeInTheDocument();
    expect(screen.getByText(/valid work email address/i)).toBeInTheDocument();
    expect(screen.getByText(/happy for us to contact you/i)).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("confirms a successful submission", async () => {
    const user = userEvent.setup();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ status: "received" }, 201)));

    render(<LeadForm />);
    await user.type(screen.getByLabelText(/^name/i), "Alex Fenn");
    await user.type(screen.getByLabelText(/work email/i), "alex@northbridge.co.uk");
    await user.type(screen.getByLabelText(/^company/i), "Northbridge Credit");
    await user.type(screen.getByLabelText(/^role/i), "Head of Portfolio");
    await user.click(screen.getByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: /book a tailored demonstration/i }));

    expect(await screen.findByRole("status")).toHaveTextContent(/thank you/i);
  });

  it("keeps the honeypot out of the accessibility tree", () => {
    const { container } = render(<LeadForm />);
    const honeypot = container.querySelector('input[name="website"]');
    expect(honeypot).toBeTruthy();
    expect(honeypot?.closest("[aria-hidden='true']")).toBeTruthy();
    expect(honeypot).toHaveAttribute("tabindex", "-1");
  });
});
