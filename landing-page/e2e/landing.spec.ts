import { expect, test } from "@playwright/test";

/**
 * The visitor journey the page exists to support:
 * load → launch the demo → ask → see an answer with metrics → preview a report
 * → read the capability stack → submit the lead form.
 */

test.describe("Trakt landing page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("loads with the proposition, navigation and both CTAs", async ({ page }) => {
    await expect(page).toHaveTitle(/Trakt \| Governed Portfolio Intelligence/);
    await expect(
      page.getByRole("heading", { level: 1, name: /portfolio intelligence\. wherever you work\./i }),
    ).toBeVisible();
    await expect(
      page.getByText(/turns fragmented portfolio data into trusted answers/i),
    ).toBeVisible();
    await expect(
      page.getByText(/Microsoft 365 Copilot, Teams, the Trakt workspace or automated reporting/i),
    ).toBeVisible();
    await expect(page.getByRole("link", { name: /explore the live demo/i })).toBeVisible();
    await expect(
      page.getByRole("link", { name: /book a portfolio walkthrough/i }),
    ).toBeVisible();
  });

  test("the hero CTA takes the visitor to the demo", async ({ page }) => {
    await page.getByRole("link", { name: /explore the live demo/i }).click();
    await expect(page).toHaveURL(/#live-demo$/);
    await expect(page.locator("#live-demo")).toBeVisible();
    await expect(page.getByText("Synthetic Demo Lender").first()).toBeVisible();
  });

  test("a suggested question returns an answer with metrics and provenance", async ({ page }) => {
    const demo = page.locator("#live-demo");
    await demo.scrollIntoViewIfNeeded();

    await demo.getByRole("button", { name: "Which regions have the highest exposure?" }).click();

    await expect(demo.getByText(/is the largest regional exposure at/i)).toBeVisible();
    // Values the deterministic engine produced for this synthetic portfolio.
    await expect(demo.getByRole("heading", { name: "Current balance by region" })).toBeVisible();
    await expect(demo.getByText("East Midlands").first()).toBeVisible();
    await expect(demo.getByText(/As at 30 November 2025/)).toBeVisible();
    await expect(demo.getByText("Synthetic portfolio", { exact: true })).toBeVisible();
    await expect(demo.getByText(/11 of 12 questions remaining/)).toBeVisible();
  });

  test("a typed unsupported question is declined clearly, not guessed", async ({ page }) => {
    const demo = page.locator("#live-demo");
    await demo.scrollIntoViewIfNeeded();

    await demo.getByLabel(/ask a question/i).fill("How has the portfolio changed since last month?");
    await demo.getByRole("button", { name: /ask trakt/i }).click();

    await expect(demo.getByText(/not supported in this demonstration/i)).toBeVisible();
    await expect(demo.getByText(/two governed reporting periods/i)).toBeVisible();
  });

  test("the conversation can be reset", async ({ page }) => {
    const demo = page.locator("#live-demo");
    await demo.scrollIntoViewIfNeeded();

    await demo.getByRole("button", { name: "What is the current funded portfolio balance?" }).click();
    await expect(demo.getByText(/the funded book stands at/i)).toBeVisible();

    await demo.getByRole("button", { name: /^reset$/i }).click();
    await expect(demo.getByText(/the funded book stands at/i)).toHaveCount(0);
  });

  test("a report preview renders pages without offering a download", async ({ page }) => {
    const demo = page.locator("#live-demo");
    await demo.scrollIntoViewIfNeeded();

    await demo.getByRole("button", { name: /generate the latest investor report/i }).click();

    await expect(demo.getByText("Investor & Funder MI Pack", { exact: true })).toBeVisible();
    await expect(demo.getByRole("heading", { name: "Executive summary", level: 4 })).toBeVisible();
    await expect(demo.getByText(/preview only, no document is produced/i)).toBeVisible();

    // Page through the preview.
    await demo.getByRole("navigation", { name: /report pages/i }).getByRole("button", { name: "4" }).click();
    await expect(demo.getByText("Geographic exposure")).toBeVisible();

    // Nothing on the page offers a file.
    await expect(page.locator("a[download]")).toHaveCount(0);
    await expect(page.locator('a[href*=".pptx"], a[href*=".csv"], a[href*="blob.core"]')).toHaveCount(0);
  });

  test("all eight capability areas are shown", async ({ page }) => {
    const capabilities = page.locator("#capabilities");
    await capabilities.scrollIntoViewIfNeeded();

    for (const name of [
      "Portfolio Integration",
      "Portfolio Analytics",
      "Management Reporting",
      "Investor Reporting",
      "Regulatory Reporting",
      "Governance and Audit",
      "Portfolio Monitoring",
      "Omnichannel Intelligence",
    ]) {
      await expect(capabilities.getByRole("heading", { name, level: 3 })).toBeVisible();
    }
  });

  test("the four delivery channels and the operating model are explained", async ({ page }) => {
    for (const name of ["Microsoft 365 Copilot", "Teams", "Trakt Workspace", "Automated Delivery"]) {
      await expect(page.locator("#channels").getByRole("heading", { name, level: 3 })).toBeVisible();
    }

    const model = page.locator("#how-it-works");
    await expect(model.getByText("Source data and documents")).toBeVisible();
    await expect(model.getByText("Trakt governed data layer")).toBeVisible();
    await expect(model.getByText("Analytics and business rules")).toBeVisible();
    await expect(
      model.getByText(/calculates the answer once, governs it centrally/i),
    ).toBeVisible();
  });

  test("the lead form validates, then accepts a complete submission", async ({ page }) => {
    const cta = page.locator("#book-a-demo");
    await cta.scrollIntoViewIfNeeded();

    await cta.getByRole("button", { name: /book a tailored demonstration/i }).click();
    await expect(cta.getByText(/please enter your name/i)).toBeVisible();

    await cta.getByLabel(/^name/i).fill("Alex Fenn");
    await cta.getByLabel(/work email/i).fill("alex@northbridge-credit.co.uk");
    await cta.getByLabel(/^company/i).fill("Northbridge Credit");
    await cta.getByLabel(/^role/i).fill("Head of Portfolio");
    await cta.getByRole("checkbox").check();

    // The minimum form-fill time is enforced server-side.
    await page.waitForTimeout(3000);
    await cta.getByRole("button", { name: /book a tailored demonstration/i }).click();

    await expect(cta.getByRole("status")).toContainText(/thank you/i);
  });

  test("the demo scope and synthetic-data notices are stated", async ({ page }) => {
    await expect(
      page.getByText(/This demonstration shows Trakt's conversational interface/),
    ).toBeVisible();
    await expect(
      page.getByText(/The demonstration uses a wholly synthetic portfolio/).first(),
    ).toBeVisible();
  });

  test("has no horizontal overflow and one h1", async ({ page }) => {
    await expect(page.locator("h1")).toHaveCount(1);
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(0);
  });

  test("is keyboard navigable from the skip link", async ({ page }) => {
    await page.keyboard.press("Tab");
    await expect(page.getByRole("link", { name: /skip to content/i })).toBeFocused();
  });
});

test.describe("mobile navigation", () => {
  test.skip(({ isMobile }) => !isMobile, "mobile viewport only");

  test("the menu opens and links through", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: /open menu/i }).click();

    const menu = page.locator("#mobile-nav");
    await expect(menu).toBeVisible();
    await menu.getByRole("link", { name: "Capabilities" }).click();

    await expect(page).toHaveURL(/#capabilities$/);
    await expect(menu).toBeHidden();
  });
});
