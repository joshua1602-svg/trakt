import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "@/App";
import { copy } from "@/lib/copy";

/**
 * Taking a standing rule back.
 *
 * The Rules screen was a reader — search, filter, expand to see history — and
 * nothing more. A mapping confirmed in error ("treat 'Curr Bal' as the current
 * balance") went on being applied to every future delivery, and the only way
 * it ever stopped was by chance: a later delivery happening to raise the same
 * question again, answered differently, superseding it. That is not a
 * reversal, it is a hope.
 *
 * Two things this screen has to get right, and both are about not making
 * withdrawal feel casual:
 *
 * - the reason is required, because a rule is read on data nobody has sent
 *   yet and this is the only thing that will explain the change later;
 * - the wording has to say it is not a delete, because an operator hesitating
 *   over the button is usually afraid of losing the record.
 */

function renderApp(route: string) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <App />
    </MemoryRouter>,
  );
}

/** Open the first rule in the list and return its card.
 *  Found by its own handle rather than by a styling class, so the test does
 *  not break the next time the card is restyled. */
async function openFirstRule() {
  await screen.findAllByText(copy.rules.sourceTerm);
  const card = document.querySelector("[data-rule]") as HTMLElement;
  await userEvent.click(within(card).getAllByRole("button")[0]);
  return card;
}

describe("withdrawing a rule", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_OPS_MODE", "mock");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("offers it on a rule that is in force", async () => {
    renderApp("/rules");
    const card = await openFirstRule();
    expect(
      within(card).getByRole("button", { name: copy.rules.retire }),
    ).toBeInTheDocument();
  });

  it("says it is not a delete before asking for anything", async () => {
    const user = userEvent.setup();
    renderApp("/rules");
    const card = await openFirstRule();

    await user.click(within(card).getByRole("button", { name: copy.rules.retire }));
    expect(within(card).getByText(copy.rules.retireHelp)).toBeInTheDocument();
    expect(copy.rules.retireHelp).toMatch(/kept/i);
  });

  it("will not withdraw without a reason", async () => {
    const user = userEvent.setup();
    renderApp("/rules");
    const card = await openFirstRule();

    await user.click(within(card).getByRole("button", { name: copy.rules.retire }));
    expect(
      within(card).getByRole("button", { name: copy.rules.retireConfirm }),
    ).toBeDisabled();
  });

  it("withdraws once a reason is given", async () => {
    const user = userEvent.setup();
    renderApp("/rules");
    const card = await openFirstRule();

    await user.click(within(card).getByRole("button", { name: copy.rules.retire }));
    await user.type(
      within(card).getByLabelText(copy.rules.retireReason),
      "The column was the original balance, not the current one.",
    );
    await user.click(
      within(card).getByRole("button", { name: copy.rules.retireConfirm }),
    );

    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: copy.rules.retireConfirm }),
      ).not.toBeInTheDocument(),
    );
  });

  it("can be backed out of without withdrawing anything", async () => {
    const user = userEvent.setup();
    renderApp("/rules");
    const card = await openFirstRule();

    await user.click(within(card).getByRole("button", { name: copy.rules.retire }));
    await user.click(within(card).getByRole("button", { name: copy.rules.retireCancel }));

    expect(
      within(card).getByRole("button", { name: copy.rules.retire }),
    ).toBeInTheDocument();
    expect(
      within(card).queryByRole("button", { name: copy.rules.retireConfirm }),
    ).not.toBeInTheDocument();
  });
});
