import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it } from "vitest";
import { MockOpsClient } from "@/api/MockOpsClient";
import { OpsClientProvider } from "@/api/context";
import type { OpsClient } from "@/api/OpsClient";
import { ToastProvider } from "@/components/Toast";
import { OnboardingCaseScreen } from "./CaseWizard";
import { OnboardingClientScreen } from "./ClientView";
import { OnboardingHomeScreen } from "./Home";

function renderAt(client: OpsClient, path: string) {
  return render(
    <OpsClientProvider client={client}>
      <ToastProvider>
        <MemoryRouter initialEntries={[path]}>
          <Routes>
            <Route path="/onboarding" element={<OnboardingHomeScreen />} />
            <Route path="/onboarding/cases/:id" element={<OnboardingCaseScreen />} />
            <Route path="/onboarding/clients/:id" element={<OnboardingClientScreen />} />
          </Routes>
        </MemoryRouter>
      </ToastProvider>
    </OpsClientProvider>,
  );
}

const VALID_LEI = "549300ABCDE123456702";

/** Answer a blank case end to end, exactly as an operator would. */
async function completeCase(client: MockOpsClient) {
  const created = await client.startNewClientCase();
  const id = created.case_id;
  await client.saveCaseStep(id, "client", {
    client_id: "NORDIC",
    client_name: "Nordic Lending",
    jurisdiction: "SE",
    reporting_currency: "SEK",
    time_zone: "Europe/Stockholm",
  });
  const withEntities = await client.saveCaseStep(id, "entities", {
    entities: [
      {
        legal_name: "Nordic Lending AB",
        roles: ["originator", "servicer"],
        lei: VALID_LEI,
        country_of_establishment: "SE",
      },
    ],
  });
  const entityId = String(
    ((withEntities.answers.entities ?? []) as Record<string, unknown>[])[0].entity_id,
  );
  await client.saveCaseStep(id, "contacts", {
    reporting_contact_name: "Rae Reporter",
    reporting_contact_email: "reporting@nordic.example",
    operational_contact_name: "Ola Ops",
    operational_contact_email: "ops@nordic.example",
  });
  await client.saveCaseStep(id, "portfolios", {
    portfolios: [
      {
        portfolio_id: "direct_001",
        display_name: "Nordic Direct",
        portfolio_type: "direct",
        asset_class: "equity_release",
        structure: "spv",
        owning_entity: entityId,
        period_convention: "calendar_month_end",
      },
    ],
  });
  await client.saveCaseStep(id, "reporting", { products: ["esma_annex2"] });
  await client.saveCaseStep(id, "sources", {
    sources: [
      {
        source_key: "direct_001/funded",
        portfolio_id: "direct_001",
        dataset: "funded",
        cadence: "monthly",
        source_party: "Nordic core",
        delivery_channel: "sftp",
        file_format: "csv",
      },
    ],
  });
  await client.saveCaseStep(id, "regime", {
    regime: {
      esma_annex2: {
        originator_name: "Nordic Lending AB",
        originator_legal_entity_identifier: VALID_LEI,
        originator_establishment_country: "SE",
      },
    },
  });
  return id;
}

describe("Client Onboarding home", () => {
  it("leads with starting a new client, not with importing one", async () => {
    renderAt(new MockOpsClient(0), "/onboarding");
    const start = await screen.findByRole("button", { name: /Start new client onboarding/ });
    expect(start).toBeInTheDocument();
    // The primary action does not ask which client — there isn't one yet.
    expect(screen.queryByText(/Adopt current configuration/)).not.toBeInTheDocument();
  });

  it("shows the working queues", async () => {
    renderAt(new MockOpsClient(0), "/onboarding");
    expect(await screen.findByText("In progress")).toBeInTheDocument();
    expect(screen.getByText("Waiting on the client")).toBeInTheDocument();
    expect(screen.getByText("Ready for review")).toBeInTheDocument();
    expect(screen.getByText("Active clients")).toBeInTheDocument();
  });

  it("offers legacy migration separately and secondarily", async () => {
    renderAt(new MockOpsClient(0), "/onboarding");
    const heading = await screen.findByText("Bring in an existing client");
    expect(heading).toBeInTheDocument();
    expect(
      screen.getByText(/Clients Trakt already serves whose configuration predates onboarding/),
    ).toBeInTheDocument();
    // It appears after the primary action in the document, not before it.
    const start = screen.getByRole("button", { name: /Start new client onboarding/ });
    expect(start.compareDocumentPosition(heading) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });
});

describe("the blank new-client case", () => {
  it("opens with a generated reference and no client", async () => {
    const client = new MockOpsClient(0);
    const created = await client.startNewClientCase();
    renderAt(client, `/onboarding/cases/${created.case_id}`);
    expect(await screen.findByText(new RegExp(created.case_id))).toBeInTheDocument();
    expect(screen.getByText(/New client · Draft/)).toBeInTheDocument();
  });

  it("renders its questions from the governed catalogue", async () => {
    const client = new MockOpsClient(0);
    const created = await client.startNewClientCase();
    renderAt(client, `/onboarding/cases/${created.case_id}`);
    // Fields nobody hard-coded in a component: they come from the catalogue.
    expect(await screen.findByText("Client name")).toBeInTheDocument();
    expect(screen.getByText("Client identifier")).toBeInTheDocument();
    expect(screen.getByText("Jurisdiction")).toBeInTheDocument();
    // System-generated values are not questions.
    expect(screen.queryByText("Onboarding reference")).not.toBeInTheDocument();
  });

  it("walks every step of the journey", async () => {
    const client = new MockOpsClient(0);
    const created = await client.startNewClientCase();
    renderAt(client, `/onboarding/cases/${created.case_id}`);
    for (const step of [
      "About the client",
      "Legal and reporting entities",
      "Contacts and distribution",
      "Portfolios",
      "Expected deliveries",
      "Reporting requirements",
      "Regulatory information",
      "Review and activate",
    ]) {
      expect(await screen.findByRole("button", { name: new RegExp(step) })).toBeInTheDocument();
    }
  });

  it("lets one entity hold several roles", async () => {
    const client = new MockOpsClient(0);
    const created = await client.startNewClientCase();
    await client.saveCaseStep(created.case_id, "entities", {
      entities: [{ legal_name: "Nordic Lending AB", roles: ["originator"] }],
    });
    renderAt(client, `/onboarding/cases/${created.case_id}`);
    await userEvent.click(
      await screen.findByRole("button", { name: /Legal and reporting entities/ }),
    );
    const card = (await screen.findByText("Nordic Lending AB")).closest("section")!;
    await userEvent.click(within(card).getByRole("button", { name: "Servicer" }));
    await waitFor(async () => {
      const updated = await client.getCase(created.case_id);
      const roles = ((updated.answers.entities ?? []) as Record<string, unknown>[])[0]
        .roles as string[];
      expect(roles).toEqual(["originator", "servicer"]);
    });
  });

  it("does not offer management information as a choice", async () => {
    const client = new MockOpsClient(0);
    const id = await completeCase(client);
    renderAt(client, `/onboarding/cases/${id}`);
    await userEvent.click(await screen.findByRole("button", { name: /Reporting requirements/ }));
    const mi = await screen.findByText("Management Information");
    const toggle = mi.closest("label")!.querySelector("input")!;
    expect(toggle).toBeDisabled();
    expect(toggle).toBeChecked();
  });

  it("derives deliveries and shows where they will arrive", async () => {
    const client = new MockOpsClient(0);
    const id = await completeCase(client);
    renderAt(client, `/onboarding/cases/${id}`);
    await userEvent.click(await screen.findByRole("button", { name: /Expected deliveries/ }));
    expect(await screen.findByText(/Nordic Direct — Funded/)).toBeInTheDocument();
    expect(screen.getByText(/raw-v2\/NORDIC\/direct\/funded\/monthly\/direct_001/)).toBeInTheDocument();
  });

  it("builds the client checklist from what is still outstanding", async () => {
    const client = new MockOpsClient(0);
    const created = await client.startNewClientCase();
    await client.saveCaseStep(created.case_id, "client", { client_id: "NEWCO" });
    renderAt(client, `/onboarding/cases/${created.case_id}`);
    await userEvent.click(await screen.findByRole("button", { name: /Contacts and distribution/ }));
    const panel = (await screen.findByText("What the client still needs to tell us")).closest(
      "section",
    )!;
    expect(within(panel).getByText("Client name")).toBeInTheDocument();
    // The identifier is the operator's decision, so it is never asked of the client.
    expect(within(panel).queryByText("Client identifier")).not.toBeInTheDocument();
  });

  it("raises an information request from the checklist", async () => {
    const client = new MockOpsClient(0);
    const created = await client.startNewClientCase();
    await client.saveCaseStep(created.case_id, "client", { client_id: "NEWCO" });
    renderAt(client, `/onboarding/cases/${created.case_id}`);
    await userEvent.click(await screen.findByRole("button", { name: /Contacts and distribution/ }));
    const panel = (await screen.findByText("What the client still needs to tell us")).closest(
      "section",
    )!;
    await userEvent.click(within(panel).getAllByRole("checkbox")[0]);
    await userEvent.click(within(panel).getByRole("button", { name: /Ask the client for these/ }));
    await waitFor(async () => {
      const updated = await client.getCase(created.case_id);
      expect(updated.information_requests).toHaveLength(1);
      expect(updated.status).toBe("information_requested");
    });
  });

  it("shows everything that will be created before anything is created", async () => {
    const client = new MockOpsClient(0);
    const id = await completeCase(client);
    renderAt(client, `/onboarding/cases/${id}`);
    await userEvent.click(await screen.findByRole("button", { name: /Review and activate/ }));
    expect(await screen.findByText("What will be created")).toBeInTheDocument();
    expect(screen.getByText("Client configuration")).toBeInTheDocument();
    expect(screen.getByText("Portfolio metadata")).toBeInTheDocument();
    expect(screen.getByText("Source registrations")).toBeInTheDocument();
    expect(screen.getByText("Client index")).toBeInTheDocument();
    // And what Trakt itself generated.
    expect(screen.getByText("What Trakt has generated")).toBeInTheDocument();
    // Still nothing active.
    const detail = await client.getOnboardingClient("NORDIC");
    expect(detail.status).toBe("not_onboarded");
  });

  it("refuses approval until the answers are complete", async () => {
    const client = new MockOpsClient(0);
    const created = await client.startNewClientCase();
    renderAt(client, `/onboarding/cases/${created.case_id}`);
    await userEvent.click(await screen.findByRole("button", { name: /Review and activate/ }));
    expect(await screen.findByRole("button", { name: "Approve" })).toBeDisabled();
    expect(screen.getByText(/still needed before this can be approved/)).toBeInTheDocument();
  });

  it("approves, then activates, and only then is the client configured", async () => {
    const client = new MockOpsClient(0);
    const id = await completeCase(client);
    renderAt(client, `/onboarding/cases/${id}`);
    await userEvent.click(await screen.findByRole("button", { name: /Review and activate/ }));
    await userEvent.type(
      await screen.findByPlaceholderText("A short note for the record"),
      "New client onboarded",
    );
    await userEvent.click(screen.getByRole("button", { name: "Approve" }));

    // Approved is not activated: no configuration exists yet.
    await waitFor(async () => {
      expect((await client.getCase(id)).status).toBe("approved");
    });
    expect((await client.getOnboardingClient("NORDIC")).status).toBe("not_onboarded");

    await userEvent.click(await screen.findByRole("button", { name: "Activate client" }));
    await waitFor(async () => {
      const detail = await client.getOnboardingClient("NORDIC");
      expect(detail.status).toBe("active");
      expect(detail.version).toBe(1);
    });
  });
});

describe("legacy migration", () => {
  it("pre-populates the same model and flags what today's rules refuse", async () => {
    const client = new MockOpsClient(0);
    const created = await client.startMigrationCase("LEGACYCO");
    renderAt(client, `/onboarding/cases/${created.case_id}`);
    expect(await screen.findByDisplayValue("Legacy Lending")).toBeInTheDocument();
    expect(screen.getByText(/Trakt has filled in what it already holds/)).toBeInTheDocument();
    // The invalid legacy identifier is raised, not carried silently.
    const raised = (await client.getCase(created.case_id)).open_questions;
    expect(raised.some((q) => q.question.includes("20-character identifier"))).toBe(true);
  });

  it("changes no active configuration before approval", async () => {
    const client = new MockOpsClient(0);
    await client.startMigrationCase("LEGACYCO");
    expect((await client.getOnboardingClient("LEGACYCO")).status).toBe("not_onboarded");
  });
});

describe("amendments", () => {
  async function activated(client: MockOpsClient) {
    const id = await completeCase(client);
    await client.approveCase(id, "New client onboarded");
    await client.activateCase(id);
    return client;
  }

  it("start from the version in force", async () => {
    const client = await activated(new MockOpsClient(0));
    const amendment = await client.startAmendmentCase("NORDIC");
    expect(amendment.kind).toBe("amendment");
    expect(amendment.based_on_version).toBe(1);
    expect((amendment.answers.client as Record<string, string>).client_name).toBe("Nordic Lending");
  });

  it("create a new version and keep the old one readable", async () => {
    const client = await activated(new MockOpsClient(0));
    const amendment = await client.startAmendmentCase("NORDIC");
    await client.saveCaseStep(amendment.case_id, "client", {
      client_name: "Nordic Lending Group",
    });
    await client.approveCase(amendment.case_id, "Rebranded");
    await client.activateCase(amendment.case_id);

    renderAt(client, "/onboarding/clients/NORDIC");
    await userEvent.click(await screen.findByRole("button", { name: "History" }));
    const versions = await screen.findAllByText(/^Version [12]$/);
    expect(versions).toHaveLength(2);
    expect(screen.getByText("Rebranded")).toBeInTheDocument();
    expect(screen.getByText("In force")).toBeInTheDocument();
    expect(screen.getByText("Replaced")).toBeInTheDocument();
  });

  it("are offered from the active client, not as a fresh onboarding", async () => {
    const client = await activated(new MockOpsClient(0));
    renderAt(client, "/onboarding/clients/NORDIC");
    expect(await screen.findByRole("button", { name: /Amend/ })).toBeInTheDocument();
    expect(screen.getByText("Version 1 in force")).toBeInTheDocument();
  });
});

describe("an ordinary operator", () => {
  it("may work a case but not approve it", async () => {
    const client = new MockOpsClient(0, "operator");
    const id = await completeCase(client);
    await expect(client.approveCase(id, "trying")).rejects.toThrow(/administrator/i);
  });
});

describe("at 390px", () => {
  beforeEach(() => {
    Object.defineProperty(window, "innerWidth", { writable: true, value: 390 });
  });

  it("keeps long generated values inside the page", async () => {
    const client = new MockOpsClient(0);
    const id = await completeCase(client);
    renderAt(client, `/onboarding/cases/${id}`);
    await userEvent.click(await screen.findByRole("button", { name: /Review and activate/ }));
    const location = await screen.findByText(/raw-v2\/NORDIC\/direct\/funded/);
    // A long storage path must wrap rather than push the page sideways.
    expect(location.className).toMatch(/break-all/);
  });

  it("lets the step rail scroll instead of overflowing", async () => {
    const client = new MockOpsClient(0);
    const created = await client.startNewClientCase();
    renderAt(client, `/onboarding/cases/${created.case_id}`);
    const rail = (await screen.findByRole("button", { name: /About the client/ })).closest("ol")!;
    expect(rail.className).toMatch(/overflow-x-auto/);
  });
});

describe("what Trakt answers for itself", () => {
  it("fills the currency and time zone from the jurisdiction", async () => {
    const client = new MockOpsClient(0);
    const created = await client.startNewClientCase();
    const answered = await client.saveCaseStep(created.case_id, "client", {
      client_name: "Nordic Lending",
      jurisdiction: "SE",
    });
    const block = answered.answers.client as Record<string, string>;
    expect(block.reporting_currency).toBe("SEK");
    expect(block.time_zone).toBe("Europe/Stockholm");
    expect(block.client_id).toBe("NORDIC");
  });

  it("shows an inferred value with where it came from, still editable", async () => {
    const client = new MockOpsClient(0);
    const created = await client.startNewClientCase();
    await client.saveCaseStep(created.case_id, "client", {
      client_name: "Nordic Lending",
      jurisdiction: "SE",
    });
    renderAt(client, `/onboarding/cases/${created.case_id}`);
    expect(await screen.findByDisplayValue("SEK")).toBeInTheDocument();
    expect(screen.getByText(/Taken from the currency used in SE/)).toBeInTheDocument();
    // Inferred is not the same as fixed: the operator can overrule it.
    const currency = screen.getByDisplayValue("SEK");
    await userEvent.clear(currency);
    await userEvent.type(currency, "EUR");
    await waitFor(async () => {
      const updated = await client.getCase(created.case_id);
      expect((updated.answers.client as Record<string, string>).reporting_currency).toBe("EUR");
    });
  });

  it("hides values that follow from another answer", async () => {
    const client = new MockOpsClient(0);
    const id = await completeCase(client);
    renderAt(client, `/onboarding/cases/${id}`);
    await userEvent.click(await screen.findByRole("button", { name: /Regulatory information/ }));
    // The Annex 2 originator block is the originator entity, restated. It is
    // never offered as a separate question.
    await screen.findByText("ESMA Annex 2");
    expect(screen.queryByText("Originator name")).not.toBeInTheDocument();
    expect(screen.queryByText("Originator LEI")).not.toBeInTheDocument();
    // But it is generated.
    const preview = await client.getCasePreview(id);
    expect(preview.artefacts[0].text).toContain("originator_legal_entity_identifier");
  });

  it("reads the format and asset class off a sample pack", async () => {
    const client = new MockOpsClient(0);
    const created = await client.startNewClientCase();
    await client.saveCaseStep(created.case_id, "client", {
      client_name: "Nordic",
      jurisdiction: "SE",
    });
    await client.saveCaseStep(created.case_id, "portfolios", {
      portfolios: [{ display_name: "Direct", portfolio_type: "direct" }],
    });
    const withSample = await client.registerSample(created.case_id, [
      { name: "LoanExtract.csv", headers: ["loan_id", "no_negative_equity", "roll_up"] },
    ]);
    const portfolio = (withSample.answers.portfolios as Record<string, unknown>[])[0];
    expect(portfolio.asset_class).toBe("equity_release");
    expect(portfolio.portfolio_id).toBe("direct_001");
    const source = (withSample.answers.sources as Record<string, unknown>[])[0];
    expect(source.file_format).toBe("csv");
    expect(source.expected_files).toEqual(["LoanExtract.csv"]);
  });
});
