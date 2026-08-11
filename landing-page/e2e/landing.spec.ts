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
        name: /one governed view of your lending portfolios\./i,
      }),
    ).toBeVisible();
    await expect(
      page.getByText(/connects loan data, documents and funding requirements/i),
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

  test("the platform section explains the layer and its outputs", async ({ page }) => {
    const platform = page.locator("#platform");
    await platform.scrollIntoViewIfNeeded();

    await expect(
      platform.getByRole("heading", { name: /build the portfolio once\. use it everywhere\./i }),
    ).toBeVisible();
    await expect(platform.getByText("Data and documents", { exact: true })).toBeVisible();
    await expect(
      platform.getByText("One governed portfolio layer", { exact: true }),
    ).toBeVisible();
    // Reporting appears as two chips among six peer outputs — an output of the
    // layer, no longer the page's identity.
    for (const output of [
      "Portfolio MI",
      "Forecasting",
      "Risk & covenant controls",
      "Investor reporting",
      "Regulatory reporting",
      "AI & Copilot interaction",
    ]) {
      await expect(platform.getByText(output, { exact: true })).toBeVisible();
    }
  });

  test("the controls section carries the forward-risk claim and stays honest", async ({
    page,
  }) => {
    const controls = page.locator("#controls");
    await controls.scrollIntoViewIfNeeded();

    await expect(
      controls.getByRole("heading", {
        name: /turn portfolio requirements into live controls\./i,
      }),
    ).toBeVisible();
    // The differentiator: three evaluation bases, not a single status.
    await expect(controls.getByText(/what the portfolio is moving toward/i)).toBeVisible();
    // Activation is a human decision; the path is visible.
    await expect(controls.getByText("Reviewed", { exact: true })).toBeVisible();

    // The demo loop mounts once the section approaches the viewport: muted,
    // looping, inline, and served with its poster so nothing shifts.
    const video = controls.locator("video");
    await expect(video).toBeVisible();
    await expect(video).toHaveAttribute("poster", /controls-demo-poster/);
    await expect(video).toHaveAttribute("loop", /.*/);
    await expect(video).toHaveAttribute("playsinline", /.*/);
    // The illustrative provenance is DOM text, not only pixels in the video.
    await expect(controls.getByText(/figures illustrative/i)).toBeVisible();
  });

  test("reduced-motion visitors get the static control preview, not the video", async ({
    page,
  }) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto("/");
    const controls = page.locator("#controls");
    await controls.scrollIntoViewIfNeeded();

    await expect(controls.locator("video")).toHaveCount(0);
    // Real DOM text — the same end state the loop resolves to.
    await expect(controls.getByText("Funded book", { exact: true })).toBeVisible();
    await expect(controls.getByText("Expected forecast", { exact: true })).toBeVisible();
    await expect(
      controls.getByText("Including full pipeline", { exact: true }),
    ).toBeVisible();
    await expect(controls.getByText(/projected breach horizon: nov 2026/i)).toBeVisible();
    await expect(controls.getByText("Illustrative", { exact: true })).toBeVisible();
  });

  test("onboarding is a governed sequence ending in a live portfolio", async ({ page }) => {
    const onboarding = page.locator("#onboarding");
    await onboarding.scrollIntoViewIfNeeded();

    await expect(
      onboarding.getByRole("heading", {
        name: /from source files to a live portfolio — under governance\./i,
      }),
    ).toBeVisible();
    for (const step of [
      "Source data and documents",
      "Assisted interpretation",
      "Governed configuration",
      "Live portfolio",
    ]) {
      await expect(onboarding.getByRole("heading", { name: step, level: 3 })).toBeVisible();
    }
    // The outcome claim is repeatability, not a speed guarantee.
    await expect(onboarding.getByText(/repeatable process as additional portfolios/i)).toBeVisible();
    await expect(onboarding).not.toContainText(/instant/i);
  });

  test("the lens section shows one truth across governed books", async ({ page }) => {
    const lenses = page.locator("#lenses");
    await lenses.scrollIntoViewIfNeeded();

    await expect(
      lenses.getByRole("heading", { name: /one portfolio truth\. every relevant lens\./i }),
    ).toBeVisible();
    await expect(lenses).toContainText(/each reportable on its own and in aggregate/i);
    await expect(lenses.getByText("Consolidated platform", { exact: true })).toBeVisible();
    await expect(lenses.getByText("SPV1 Sponsored Securitisation")).toBeVisible();
    await expect(lenses.getByText("sold", { exact: true })).toBeVisible();
  });

  test("governance separates what ships from what is planned", async ({ page }) => {
    const governance = page.locator("#governance");
    await governance.scrollIntoViewIfNeeded();

    await expect(
      governance.getByRole("heading", {
        name: /deterministic underneath\. governed throughout\./i,
      }),
    ).toBeVisible();
    for (const name of [
      "Deterministic calculation",
      "Reviewed configuration",
      "Traceable outputs",
      "Client separation",
    ]) {
      await expect(governance.getByRole("heading", { name, level: 3 })).toBeVisible();
    }
    // Asset extensibility is an architecture claim, never a coverage claim.
    await expect(governance.getByText(/asset-specific configuration/i)).toBeVisible();
    await expect(governance).not.toContainText(/every asset class/i);
    // The agentic direction is a quiet design-intent sentence, not a roadmap
    // block — and it is worded as direction, never as live capability.
    await expect(
      governance.getByText(/toward increasingly agentic operation/i),
    ).toBeVisible();
    await expect(governance).not.toContainText(/roadmap/i);
    await expect(governance).not.toContainText(/autonomous/i);
  });

  test("reporting is a band of outputs, not the identity", async ({ page }) => {
    const reporting = page.locator("#reporting");
    await reporting.scrollIntoViewIfNeeded();

    await expect(
      reporting.getByRole("heading", {
        name: /the portfolio truth that runs the business also reports it\./i,
      }),
    ).toBeVisible();
    for (const output of [
      "Management reporting",
      "Investor & funding-partner packs",
      "Regulatory submissions",
    ]) {
      await expect(reporting.getByText(output, { exact: true })).toBeVisible();
    }
    // Regime names stay off the homepage; they anchor an asset class.
    await expect(reporting).not.toContainText(/annex/i);
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

  test("the intelligence channels read at a glance, each with its glyph", async ({ page }) => {
    const intelligence = page.locator("#intelligence");
    await intelligence.scrollIntoViewIfNeeded();

    for (const channel of ["Trakt workspace", "Microsoft Teams", "Microsoft 365 Copilot"]) {
      await expect(intelligence.getByText(channel, { exact: true })).toBeVisible();
    }
    // Three channel chips, each carrying a small neutral glyph — labels do the
    // naming, no imitation product logos.
    await expect(intelligence.locator("span:has(> svg)")).toHaveCount(3);
    await expect(intelligence.getByText("Available today")).toBeVisible();
  });

  test("the synthetic-data disclaimer is stated exactly once, in the example", async ({ page }) => {
    const disclaimer = page.getByText(
      /The portfolios are wholly synthetic, and the page accepts no uploads/,
    );
    await expect(disclaimer).toHaveCount(1);
    await expect(page.locator("#intelligence").getByText(
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
    await menu.getByRole("link", { name: "Platform" }).click();

    await expect(page).toHaveURL(/#platform$/);
    await expect(menu).toBeHidden();
  });
});
