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
      page.getByRole("heading", {
        level: 1,
        name: /one governed portfolio dataset\. every book, every report\./i,
      }),
    ).toBeVisible();
    await expect(
      page.getByText(/turns loan tapes, servicing extracts, valuations and funding data/i),
    ).toBeVisible();
    await expect(
      page.getByText(/reconciled by construction rather than by comparison/i),
    ).toBeVisible();
    await expect(page.getByRole("link", { name: /explore the live demo/i })).toBeVisible();
    await expect(
      page.getByRole("link", { name: /book a portfolio walkthrough/i }),
    ).toBeVisible();
  });

  test("the hero CTA takes the visitor to the demo", async ({ page }) => {
    await page.getByRole("link", { name: /explore the live demo/i }).click();
    await expect(page).toHaveURL(/#example$/);
    await expect(page.locator("#example")).toBeVisible();
    await expect(page.getByText("Alderbridge Lending Platform").first()).toBeVisible();
  });

  test("a suggested question returns an answer with metrics and provenance", async ({ page }) => {
    const demo = page.locator("#example");
    await demo.scrollIntoViewIfNeeded();

    await demo.getByRole("button", { name: "Show the funded balance by book." }).click();

    // The differentiator: three books, and BOTH governed totals — the platform
    // on balance sheet, and the sponsor scope including the sold SPV.
    await expect(demo.getByText(/across three governed books/i)).toBeVisible();
    await expect(
      demo.getByRole("heading", { name: "Funded balance by governed book" }),
    ).toBeVisible();
    await expect(demo.getByText("SPV1 Sponsored Securitisation").first()).toBeVisible();
    await expect(demo.getByText("Platform total (warehoused)")).toBeVisible();
    await expect(demo.getByText("Sponsor total (including SPV1)")).toBeVisible();
    await expect(demo.getByText(/As at 30 June 2026/)).toBeVisible();
    await expect(demo.getByText("Synthetic portfolio", { exact: true })).toBeVisible();
    // The cap is real but silent this early: it is not a meter.
    await expect(demo.getByText(/questions remaining/)).toHaveCount(0);
  });

  test("a typed unsupported question is declined clearly, not guessed", async ({ page }) => {
    const demo = page.locator("#example");
    await demo.scrollIntoViewIfNeeded();

    await demo.getByLabel(/ask a question/i).fill("Show me individual loan records with borrower details.");
    await demo.getByRole("button", { name: /ask trakt/i }).click();

    await expect(demo.getByText(/not supported in this demonstration/i)).toBeVisible();
    // Refuses on governance grounds, not a data gap — the stronger claim, and
    // the one an institutional reader is actually worried about.
    await expect(demo.getByText(/will not return exposure-level records/i)).toBeVisible();
    await expect(demo.getByText(/role-based access/i)).toBeVisible();
  });

  test("the conversation can be reset", async ({ page }) => {
    const demo = page.locator("#example");
    await demo.scrollIntoViewIfNeeded();

    await demo.getByRole("button", { name: "Show the funded balance by book." }).click();
    await expect(demo.getByText(/across three governed books/i)).toBeVisible();

    await demo.getByRole("button", { name: /^reset$/i }).click();
    await expect(demo.getByText(/across three governed books/i)).toHaveCount(0);
  });

  test("a report preview renders pages without offering a download", async ({ page }) => {
    const demo = page.locator("#example");
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

  test("the ingestion claim leads, and the tiles below are peers", async ({ page }) => {
    const capabilities = page.locator("#capabilities");
    await capabilities.scrollIntoViewIfNeeded();

    // Lifted out of the grid: it is the reason the others are possible, and
    // comparative emphasis inside the grid fails on a phone.
    await expect(capabilities).toContainText(/each reportable on its own and in aggregate/i);
    await expect(
      capabilities.getByRole("heading", { name: "Capabilities on one governed layer" }),
    ).toBeVisible();

    const names = [
      "Portfolio analytics and monitoring",
      "Management reporting",
      "Investor reporting",
      "Regulatory reporting",
      "Governance and audit",
    ];
    for (const name of names) {
      await expect(capabilities.getByRole("heading", { name, level: 3 })).toBeVisible();
    }
    await expect(capabilities.getByRole("heading", { level: 3 })).toHaveCount(names.length);

    // "Illustrative capabilities" hedged every tile; it is gone.
    await expect(capabilities.getByText(/illustrative/i)).toHaveCount(0);
  });

  test("delivery modes separate what ships from what is planned", async ({ page }) => {
    const delivery = page.locator("#delivery");
    await delivery.scrollIntoViewIfNeeded();

    await expect(delivery.getByText("Available today")).toBeVisible();
    for (const name of [
      "Managed service",
      "Trakt Agent workspace",
      "Microsoft 365 Copilot and Teams",
    ]) {
      await expect(delivery.getByRole("heading", { name, level: 3 })).toBeVisible();
    }

    // Roadmap is labelled as roadmap. Neither item is a shipped route today:
    // both are reserved channels in trakt_core/context.py.
    await expect(delivery.getByText("Roadmap")).toBeVisible();
    for (const name of ["Enterprise agent deployment", "Agent-to-agent integration"]) {
      await expect(delivery.getByRole("heading", { name, level: 3 })).toBeVisible();
    }

    // No timer trigger or cron exists in the repository, so nothing claims
    // "scheduled" delivery as a platform capability.
    await expect(delivery).not.toContainText(/scheduled/i);
    await expect(delivery).toContainText(/calculated once and distributed/i);

    const model = page.locator("#how-it-works");
    await expect(model.getByText("Source data and documents")).toBeVisible();
    await expect(model.getByText("Trakt governed data layer")).toBeVisible();
    await expect(model.getByText("Analytics and business rules")).toBeVisible();
  });

  test("the lead form validates, then accepts a complete submission", async ({ page }) => {
    const cta = page.locator("#book-a-demo");
    await cta.scrollIntoViewIfNeeded();

    await cta.getByRole("button", { name: /book a tailored demonstration/i }).click();
    await expect(cta.getByText(/please enter your name/i)).toBeVisible();

    await cta.getByLabel(/^name/i).fill("Alex Fenn");
    await cta.getByLabel(/work email/i).fill("alex@northbridge-credit.co.uk");
    await cta.getByLabel(/^company/i).fill("Northbridge Credit");
    // Role is optional — a complete submission does not need it.
    await expect(cta.getByLabel(/^role/i)).not.toHaveAttribute("required", /.*/);
    await cta.getByRole("checkbox").check();

    // The minimum form-fill time is enforced server-side.
    await page.waitForTimeout(3000);
    await cta.getByRole("button", { name: /book a tailored demonstration/i }).click();

    await expect(cta.getByRole("status")).toContainText(/thank you/i);
  });

  test("the synthetic-data disclaimer is stated exactly once, in the example", async ({ page }) => {
    const disclaimer = page.getByText(
      /The portfolios are wholly synthetic, and the page accepts no uploads/,
    );
    await expect(disclaimer).toHaveCount(1);
    await expect(page.locator("#example").getByText(
      /The portfolios are wholly synthetic, and the page accepts no uploads/,
    )).toBeVisible();

    // Repeated reassurance reads as anxiety: the old wording appeared five times.
    await expect(page.getByText(/uses a wholly synthetic portfolio/)).toHaveCount(0);

    // Refusal is a differentiator, so it is stated rather than footnoted.
    await expect(
      page.locator("#example").getByText(/Trakt declines what it cannot derive/),
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

  test("reports the pinned portfolio identity from both probes", async ({ request }) => {
    const health = await request.get("/api/health");
    expect(health.status()).toBe(200);
    const healthBody = await health.json();
    expect(healthBody.status).toBe("ok");
    expect(healthBody.demoPack).toBe("ready");

    const ready = await request.get("/api/ready");
    const readyBody = await ready.json();
    expect(readyBody.components.demoPack).toBe("ready");

    // Neither probe leaks configuration.
    const serialised = JSON.stringify(healthBody) + JSON.stringify(readyBody);
    expect(serialised).not.toMatch(/@|\.csv|\/home\/|blob\.core/);
  });

  test("publishes which demo pack it is serving, and it is the one on the page", async ({
    page,
    request,
  }) => {
    // The contract the post-deploy gate asserts against. Health and readiness
    // both return 200 for a healthy server running an older bundle; this is
    // what says *which dataset* is behind the figures.
    const response = await request.get("/api/demo-identity");
    expect(response.status()).toBe(200);
    expect(response.headers()["cache-control"]).toContain("no-store");

    const identity = await response.json();
    expect(identity.sourceFingerprint).toMatch(/^[0-9a-f]{64}$/);
    expect(identity.synthetic).toBe(true);

    const marker = page.locator('meta[name="trakt:pack"]');
    await expect(marker).toHaveAttribute(
      "content",
      [identity.clientId, identity.portfolioId, identity.reportingDate].join("/"),
    );

    await expect(
      page.locator("#product").getByText(identity.totalBalanceDisplay).first(),
    ).toBeVisible();
  });

  test("the hero preview and the demo metadata agree on the portfolio", async ({
    page,
    request,
  }) => {
    const meta = await (await request.get("/api/demo/meta")).json();
    const { totalBalanceDisplay, loanCount, asOfDisplay, client } = meta.scope;

    // The figure in the hero is the same figure the demo answers with.
    await expect(page.locator("#product").getByText(totalBalanceDisplay).first()).toBeVisible();
    await expect(page.locator("#example")).toContainText(client);
    await expect(page.locator("#example")).toContainText(`${loanCount} exposures`);
    await expect(page.locator("#example")).toContainText(asOfDisplay);

    // en-GB formatting throughout: pounds, thousands separators, no dollars.
    expect(totalBalanceDisplay).toMatch(/^£[\d,]+$/);
    expect(asOfDisplay).toMatch(/^\d{1,2} [A-Z][a-z]+ \d{4}$/);
    await expect(page.locator("body")).not.toContainText("$");
  });

  test("an analytics beacon carries an event id and no question text", async ({ page }) => {
    const posted: { url: string; body: string }[] = [];
    page.on("request", (req) => {
      if (req.url().includes("/api/analytics") && req.method() === "POST") {
        posted.push({ url: req.url(), body: req.postData() ?? "" });
      }
    });

    const demo = page.locator("#example");
    await demo.scrollIntoViewIfNeeded();
    await demo.getByLabel(/ask a question/i).fill("what is the weighted average ltv");
    await demo.getByRole("button", { name: /ask trakt/i }).click();
    await expect(demo.getByText(/weighted average current ltv is/i).first()).toBeVisible();

    // The default provider is "none", so nothing should be sent at all. If a
    // deployment enables the collector, the payload must still carry no text.
    for (const { body } of posted) {
      expect(body).not.toContain("weighted average ltv");
      expect(body).not.toContain("what is the");
      expect(body).toMatch(/"event"\s*:/);
    }
  });

  test("the lead form submits and the response carries only a reference", async ({ page }) => {
    const cta = page.locator("#book-a-demo");
    await cta.scrollIntoViewIfNeeded();
    await cta.getByLabel(/^name/i).fill("Jordan Vale");
    await cta.getByLabel(/work email/i).fill("jordan@ridgeline-credit.co.uk");
    await cta.getByLabel(/^company/i).fill("Ridgeline Credit");
    await cta.getByLabel(/^role/i).fill("Finance Director");
    await cta.getByRole("checkbox").check();
    await page.waitForTimeout(3000);

    // Await the response explicitly rather than racing an event handler.
    const [response] = await Promise.all([
      page.waitForResponse((res) => res.url().includes("/api/leads") && res.request().method() === "POST"),
      cta.getByRole("button", { name: /book a tailored demonstration/i }).click(),
    ]);

    await expect(cta.getByRole("status")).toContainText(/thank you/i);
    const responseBody = await response.text();
    const body = JSON.parse(responseBody);
    expect(Object.keys(body).sort()).toEqual(["reference", "status"]);
    // Attribution is attached server-side and never returned.
    expect(responseBody).not.toContain("utm_");
  });

  test("campaign attribution is captured without reaching the page", async ({ page }) => {
    await page.goto("/?utm_source=linkedin&utm_campaign=q3-erm&persona=coo");

    // Capture runs in an effect, so wait for hydration rather than assuming it.
    await expect
      .poll(() => page.evaluate(() => window.sessionStorage.getItem("trakt_attribution")))
      .toContain("linkedin");

    const stored = await page.evaluate(() =>
      window.sessionStorage.getItem("trakt_attribution"),
    );
    expect(stored).toContain("q3-erm");

    // It is never rendered.
    await expect(page.locator("body")).not.toContainText("q3-erm");
    await expect(page.locator("body")).not.toContainText("utm_source");
  });

  test("logs no console errors", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });
    page.on("pageerror", (error) => errors.push(String(error)));

    await page.goto("/");
    const demo = page.locator("#example");
    await demo.scrollIntoViewIfNeeded();
    await demo.getByRole("button", { name: "Show the funded balance by book." }).click();
    await expect(demo.getByText(/across three governed books/i)).toBeVisible();

    expect(errors).toEqual([]);
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
