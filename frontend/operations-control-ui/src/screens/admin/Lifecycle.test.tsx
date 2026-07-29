import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { OpsError } from "@/api/OpsClient";
import type { OpsClient } from "@/api/OpsClient";
import { MockOpsClient } from "@/api/MockOpsClient";
import { copy } from "@/lib/copy";
import { ConfigSystemScreen } from "./ConfigSystem";
import { mockClient, renderAdmin } from "./testUtils";

async function renderSystem(client: MockOpsClient | OpsClient = mockClient()) {
  const view = renderAdmin(<ConfigSystemScreen />, client, {
    route: "/admin/config/system",
  });
  await screen.findByText(copy.admin.system.title);
  return view;
}

/** The draft panel, so assertions do not collide with the active-version card. */
function draftPanel(): HTMLElement {
  const heading = screen.getByText(copy.admin.draft.heading);
  const panel = heading.parentElement?.querySelector("div.rounded-2xl");
  if (!panel) throw new Error("draft panel not rendered");
  return panel as HTMLElement;
}

describe("configuration lifecycle", () => {
  it("shows the empty draft state before a draft exists", async () => {
    await renderSystem();
    expect(screen.getByText(copy.admin.draft.none)).toBeInTheDocument();
  });

  it("creates a draft after an explicit confirmation", async () => {
    const user = userEvent.setup();
    const client = mockClient();
    const spy = vi.spyOn(client, "createConfigDraft");
    await renderSystem(client);

    await user.click(screen.getByRole("button", { name: copy.admin.actions.createDraft }));
    expect(screen.getByText(copy.admin.draft.createPrompt)).toBeInTheDocument();
    expect(spy).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: copy.admin.draft.createButton }));
    expect(spy).toHaveBeenCalledWith("system");

    // The package state is refreshed from the backend, not guessed.
    expect(await screen.findByText(copy.admin.draft.heading)).toBeInTheDocument();
    await waitFor(() => expect(within(draftPanel()).getByText("v2")).toBeInTheDocument());
    expect(within(draftPanel()).getByText(copy.admin.validation.notChecked)).toBeInTheDocument();
  });

  it("refuses to activate a draft that has not been checked", async () => {
    const user = userEvent.setup();
    const client = mockClient();
    await client.createConfigDraft("system");
    const spy = vi.spyOn(client, "activateConfigVersion");
    await renderSystem(client);

    const activate = within(draftPanel()).getByRole("button", {
      name: copy.admin.actions.activate,
    });
    expect(activate).toBeDisabled();
    await user.click(activate);
    expect(spy).not.toHaveBeenCalled();
  });

  it("shows a validation failure with plain English and hides the technical text", async () => {
    const user = userEvent.setup();
    const client = mockClient();
    await client.createConfigDraft("system", {
      edits: { "config/system/run_context_policy.yaml": "not: [valid" },
    });
    await renderSystem(client);

    await user.click(within(draftPanel()).getByRole("button", { name: copy.admin.actions.validate }));

    expect(await screen.findByText(copy.admin.validation.fail)).toBeInTheDocument();
    expect(screen.getByText(/1 configuration check failed/)).toBeInTheDocument();
    expect(screen.getByText(/Run policy could not be read/)).toBeInTheDocument();
    // The raw problem text stays behind the diagnostics toggle.
    expect(screen.queryByText(/ScannerError/)).not.toBeInTheDocument();

    await user.click(
      within(draftPanel()).getByRole("button", { name: copy.admin.actions.showTechnical }),
    );
    expect(screen.getByText(/ScannerError/)).toBeInTheDocument();

    // An invalid draft still cannot be activated.
    expect(
      within(draftPanel()).getByRole("button", { name: copy.admin.actions.activate }),
    ).toBeDisabled();
  });

  it("activates a valid draft only after the confirmation dialog", async () => {
    const user = userEvent.setup();
    const client = mockClient();
    await client.createConfigDraft("system", {
      edits: { "config/system/run_context_policy.yaml": "version: 2\n" },
    });
    await client.validateConfigVersion("system", 2);
    const spy = vi.spyOn(client, "activateConfigVersion");
    await renderSystem(client);

    await user.click(within(draftPanel()).getByRole("button", { name: copy.admin.actions.activate }));

    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText(copy.admin.activate.heading)).toBeInTheDocument();
    expect(within(dialog).getByText("System package")).toBeInTheDocument();
    expect(within(dialog).getByText("v2")).toBeInTheDocument();
    expect(within(dialog).getByText("v1")).toBeInTheDocument();
    expect(within(dialog).getByText(copy.admin.activate.pinnedNote)).toBeInTheDocument();
    expect(within(dialog).getByText(copy.admin.activate.futureNote)).toBeInTheDocument();
    expect(within(dialog).getByText(copy.admin.activate.historyNote)).toBeInTheDocument();
    expect(spy).not.toHaveBeenCalled();

    await user.click(within(dialog).getByRole("button", { name: copy.admin.activate.button }));
    expect(spy).toHaveBeenCalledWith("system", 2);

    // State refreshes: the active version becomes 2 and no draft remains.
    expect(await screen.findByText(copy.admin.draft.none)).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("v2")).toBeInTheDocument());
    expect(await screen.findByText(copy.admin.activate.done)).toBeInTheDocument();
  });

  it("cancels activation without calling the backend", async () => {
    const user = userEvent.setup();
    const client = mockClient();
    await client.createConfigDraft("system");
    await client.validateConfigVersion("system", 2);
    const spy = vi.spyOn(client, "activateConfigVersion");
    await renderSystem(client);

    await user.click(within(draftPanel()).getByRole("button", { name: copy.admin.actions.activate }));
    await user.click(screen.getByRole("button", { name: copy.common.cancel }));
    expect(spy).not.toHaveBeenCalled();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("rolls back only after an explicit confirmation", async () => {
    const user = userEvent.setup();
    const client = mockClient();
    await client.createConfigDraft("system", {
      edits: { "config/system/run_context_policy.yaml": "version: 3\n" },
    });
    await client.validateConfigVersion("system", 2);
    await client.activateConfigVersion("system", 2);
    const spy = vi.spyOn(client, "rollbackConfig");
    await renderSystem(client);

    await user.click(screen.getByRole("button", { name: copy.admin.actions.rollback }));
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText(copy.admin.rollback.heading)).toBeInTheDocument();
    expect(within(dialog).getByText(copy.admin.rollback.help)).toBeInTheDocument();
    expect(spy).not.toHaveBeenCalled();

    await user.click(within(dialog).getByRole("button", { name: copy.admin.rollback.button }));
    expect(spy).toHaveBeenCalledWith("system", 1);
    expect(await screen.findByText(copy.admin.rollback.done)).toBeInTheDocument();
  });

  it("reports a backend failure and leaves the shown state unchanged", async () => {
    const user = userEvent.setup();
    const client = mockClient();
    await client.createConfigDraft("system");
    await client.validateConfigVersion("system", 2);
    vi.spyOn(client, "activateConfigVersion").mockRejectedValue(
      new OpsError(
        "Equity Release reports under ESMA Annex 2, but this version does not carry that regulatory package.",
        "OPS_CONFIG_INCOMPATIBLE",
        [
          {
            kind: "incompatible_dependency",
            message:
              "Equity Release reports under ESMA Annex 2, but this version does not carry that regulatory package.",
          },
        ],
      ),
    );
    await renderSystem(client);

    await user.click(within(draftPanel()).getByRole("button", { name: copy.admin.actions.activate }));
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: copy.admin.activate.button,
      }),
    );

    expect(
      await screen.findByText(/does not carry that regulatory package/),
    ).toBeInTheDocument();
    // The draft is still a draft; nothing was claimed as successful.
    expect(screen.queryByText(copy.admin.activate.done)).not.toBeInTheDocument();
    expect(within(draftPanel()).getByText("v2")).toBeInTheDocument();
  });

  it("shows the activation blockers the backend returned when reopened", async () => {
    const user = userEvent.setup();
    const client = mockClient();
    await client.createConfigDraft("system");
    await client.validateConfigVersion("system", 2);
    vi.spyOn(client, "activateConfigVersion").mockRejectedValue(
      new OpsError("Run policy is missing from this version.", "OPS_CONFIG_INCOMPATIBLE", [
        { kind: "missing_file", message: "Run policy is missing from this version." },
      ]),
    );
    await renderSystem(client);

    await user.click(within(draftPanel()).getByRole("button", { name: copy.admin.actions.activate }));
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: copy.admin.activate.button,
      }),
    );
    await screen.findAllByText("Run policy is missing from this version.");

    await user.click(within(draftPanel()).getByRole("button", { name: copy.admin.actions.activate }));
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText(copy.admin.activate.blockedHeading)).toBeInTheDocument();
    expect(
      within(dialog).getByRole("button", { name: copy.admin.activate.button }),
    ).toBeDisabled();
  });
});
