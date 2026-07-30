import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { MockOpsClient } from "@/api/MockOpsClient";
import { OpsClientProvider } from "@/api/context";
import type { OpsClient } from "@/api/OpsClient";
import { ToastProvider } from "@/components/Toast";
import { OnboardingClientScreen } from "./ClientEditor";
import { OnboardingHomeScreen } from "./Home";
import { OnboardingWizardScreen } from "./Wizard";

function renderAt(client: OpsClient, path: string) {
  return render(
    <OpsClientProvider client={client}>
      <ToastProvider>
        <MemoryRouter initialEntries={[path]}>
          <Routes>
            <Route path="/onboarding" element={<OnboardingHomeScreen />} />
            <Route path="/onboarding/drafts/:id" element={<OnboardingWizardScreen />} />
            <Route path="/onboarding/clients/:id" element={<OnboardingClientScreen />} />
          </Routes>
        </MemoryRouter>
      </ToastProvider>
    </OpsClientProvider>,
  );
}

describe("Client Onboarding home", () => {
  it("lists a client that predates onboarding and offers to adopt it", async () => {
    renderAt(new MockOpsClient(0), "/onboarding");

    expect(await screen.findByText("ERE")).toBeInTheDocument();
    expect(screen.getByText("Not yet onboarded")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Adopt current configuration" }),
    ).toBeInTheDocument();
    // Adoption is an offer, not a demand — the client still works untouched.
    expect(
      screen.getByText(/Configured before Client Onboarding existed/),
    ).toBeInTheDocument();
  });
});

describe("the wizard", () => {
  it("shows a new client every step, with nothing answered yet", async () => {
    const client = new MockOpsClient(0);
    const draft = await client.startOnboardingDraft({});
    renderAt(client, `/onboarding/drafts/${draft.draft_id}`);

    for (const step of [
      "Client",
      "Portfolios",
      "Reporting",
      "Regime configuration",
      "Source registration",
      "Review",
    ]) {
      expect(await screen.findByRole("button", { name: new RegExp(step) })).toBeInTheDocument();
    }
    expect(await screen.findByText("Who is the client?")).toBeInTheDocument();
  });

  it("fills an adopted client's current values in and says where they came from", async () => {
    const client = new MockOpsClient(0);
    const draft = await client.startOnboardingDraft({ client_id: "ERE", adopt: true });
    renderAt(client, `/onboarding/drafts/${draft.draft_id}`);

    expect(
      await screen.findByDisplayValue("ERE Funding - Equity Release Mortgages"),
    ).toBeInTheDocument();
    expect(screen.getByDisplayValue("213800ABCDE123456701")).toBeInTheDocument();
    expect(
      screen.getByText(/Taken from defaults.originator_legal_entity_identifier/),
    ).toBeInTheDocument();
    // What the legacy files could not answer is named, not guessed.
    expect(screen.getByText(/Still needed: Primary reporting contact/)).toBeInTheDocument();
  });

  it("does not offer management information as a choice", async () => {
    const client = new MockOpsClient(0);
    const draft = await client.startOnboardingDraft({ client_id: "ERE", adopt: true });
    renderAt(client, `/onboarding/drafts/${draft.draft_id}`);

    await userEvent.click(await screen.findByRole("button", { name: /Reporting/ }));
    const mi = await screen.findByText("Management Information");
    const toggle = mi.closest("label")!.querySelector("input")!;
    expect(toggle).toBeDisabled();
    expect(toggle).toBeChecked();
    expect(screen.getByText(/Always prepared/)).toBeInTheDocument();
  });

  it("asks regime questions in business terms and names the reported field", async () => {
    const client = new MockOpsClient(0);
    const draft = await client.startOnboardingDraft({ client_id: "ERE", adopt: true });
    renderAt(client, `/onboarding/drafts/${draft.draft_id}`);

    await userEvent.click(await screen.findByRole("button", { name: /Regime configuration/ }));
    expect(await screen.findByText("Originator LEI")).toBeInTheDocument();
    expect(screen.getByText(/Reported as RREL83/)).toBeInTheDocument();
    // What the regime asks for that Trakt cannot hold is stated, not hidden.
    expect(screen.getByText("Not held here")).toBeInTheDocument();
    expect(screen.getByText("Sponsor")).toBeInTheDocument();
  });

  it("derives source registrations rather than asking for them", async () => {
    const client = new MockOpsClient(0);
    const draft = await client.startOnboardingDraft({ client_id: "ERE", adopt: true });
    renderAt(client, `/onboarding/drafts/${draft.draft_id}`);

    await userEvent.click(await screen.findByRole("button", { name: /Source registration/ }));
    expect(await screen.findByText(/Trakt creates them for you/)).toBeInTheDocument();
    expect(screen.getByText("ERE Direct Originations — Funded")).toBeInTheDocument();
    expect(screen.getByText("ERE Direct Originations — Pipeline")).toBeInTheDocument();
    expect(screen.getByText("BigBank Legacy Book — Funded")).toBeInTheDocument();
  });

  it("shows everything that will be written before anything is written", async () => {
    const client = new MockOpsClient(0);
    const draft = await client.startOnboardingDraft({ client_id: "ERE", adopt: true });
    renderAt(client, `/onboarding/drafts/${draft.draft_id}`);

    await userEvent.click(await screen.findByRole("button", { name: /Review/ }));
    expect(await screen.findByText("What will be written")).toBeInTheDocument();
    expect(screen.getByText("Client configuration")).toBeInTheDocument();
    expect(screen.getByText("Portfolio metadata")).toBeInTheDocument();
    expect(screen.getByText("Source registrations")).toBeInTheDocument();
    expect(screen.getAllByText("Will be created").length).toBeGreaterThan(0);
  });

  it("refuses to approve until the outstanding answers are given", async () => {
    const client = new MockOpsClient(0);
    const draft = await client.startOnboardingDraft({ client_id: "ERE", adopt: true });
    renderAt(client, `/onboarding/drafts/${draft.draft_id}`);

    await userEvent.click(await screen.findByRole("button", { name: /Review/ }));
    const approve = await screen.findByRole("button", {
      name: "Approve and write configuration",
    });
    expect(approve).toBeDisabled();
    expect(screen.getByText(/answers still needed/)).toBeInTheDocument();
  });

  it("approves once the profile is complete, and records the reason", async () => {
    const client = new MockOpsClient(0);
    const draft = await client.startOnboardingDraft({ client_id: "ERE", adopt: true });
    await client.saveOnboardingStep(draft.draft_id, "client", {
      ...draft.profile.client,
      primary_reporting_contact: "Ada Reporter",
      reporting_email: "reporting@ere.example",
      operational_contact: "Ola Ops",
      operational_email: "ops@ere.example",
    });
    renderAt(client, `/onboarding/drafts/${draft.draft_id}`);

    await userEvent.click(await screen.findByRole("button", { name: /Review/ }));
    const approve = await screen.findByRole("button", {
      name: "Approve and write configuration",
    });
    await waitFor(() => expect(approve).toBeDisabled()); // no reason given yet
    await userEvent.type(
      screen.getByPlaceholderText("A short note for the record"),
      "Adopting the existing client",
    );
    await waitFor(() => expect(approve).toBeEnabled());
    await userEvent.click(approve);

    await waitFor(async () => {
      const detail = await client.getOnboardingClient("ERE");
      expect(detail.status).toBe("onboarded");
      expect(detail.history[0].reason).toBe("Adopting the existing client");
    });
  });
});

describe("the existing-client editor", () => {
  async function onboardEre(client: MockOpsClient) {
    const draft = await client.startOnboardingDraft({ client_id: "ERE", adopt: true });
    await client.saveOnboardingStep(draft.draft_id, "client", {
      ...draft.profile.client,
      primary_reporting_contact: "Ada Reporter",
      reporting_email: "reporting@ere.example",
      operational_contact: "Ola Ops",
      operational_email: "ops@ere.example",
    });
    await client.approveOnboardingDraft(draft.draft_id, "Adopting the existing client");
    return client;
  }

  it("shows the configuration in force across every tab", async () => {
    const client = await onboardEre(new MockOpsClient(0));
    renderAt(client, "/onboarding/clients/ERE");

    expect(await screen.findByText("Version 1 in force")).toBeInTheDocument();
    expect(screen.getByText("213800ABCDE123456701")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Portfolios" }));
    expect(await screen.findByText("ERE Direct Originations")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Regimes" }));
    expect(await screen.findByText("Esma annex2")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Source registrations" }));
    expect(await screen.findByText(/What Trakt expects to receive/)).toBeInTheDocument();
  });

  it("keeps an immutable history with who, when, why and what changed", async () => {
    const client = await onboardEre(new MockOpsClient(0));
    const second = await client.startOnboardingDraft({ client_id: "ERE" });
    await client.saveOnboardingStep(second.draft_id, "client", {
      ...second.profile.client,
      reporting_email: "newreporting@ere.example",
    });
    await client.approveOnboardingDraft(second.draft_id, "Reporting contact changed");

    renderAt(client, "/onboarding/clients/ERE");
    await userEvent.click(await screen.findByRole("button", { name: "History" }));

    const versions = await screen.findAllByText(/^Version [12]$/);
    expect(versions).toHaveLength(2);
    expect(screen.getByText("Reporting contact changed")).toBeInTheDocument();
    expect(screen.getByText("In force")).toBeInTheDocument();
    expect(screen.getByText("Replaced")).toBeInTheDocument();
    // Before and after are both shown. The old address appears twice — as
    // version 1's value and as version 2's "before" — which is the point:
    // history is additive, so the earlier version still reads as it did.
    expect(screen.getAllByText("reporting@ere.example").length).toBe(2);
    expect(screen.getByText("newreporting@ere.example")).toBeInTheDocument();
  });

  it("offers adoption for a client that has not been onboarded", async () => {
    renderAt(new MockOpsClient(0), "/onboarding/clients/ERE");
    expect(
      await screen.findByText(/This client has not been through Client Onboarding/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Adopt current configuration" }),
    ).toBeInTheDocument();
  });
});

describe("an ordinary operator", () => {
  it("is refused when approving, by the same rule the server applies", async () => {
    const client = new MockOpsClient(0, "operator");
    const draft = await client.startOnboardingDraft({ client_id: "ERE", adopt: true });
    await expect(
      client.approveOnboardingDraft(draft.draft_id, "trying"),
    ).rejects.toThrow(/administrator/i);
  });
});
