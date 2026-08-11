import { expect, test, type Locator, type Page } from "@playwright/test";

/**
 * The visitor journey the page exists to support:
 * load → start the query demo → see a governed answer → read the platform →
 * watch the controls demo → read governance → submit the lead form.
 *
 * Both demos are user-started: nothing plays until the visitor presses play,
 * and nothing loops silently from the middle.
 */

/**
 * Start the query demo. Starting runs the scripted opening question
 * ("Show the funded balance by book."), so the balance-by-book answer is the
 * signal that the demo is live.
 */
async function startQueryDemo(page: Page): Promise<Locator> {
  const demo = page.locator("#example");
  await demo.scrollIntoViewIfNeeded();
  await demo.getByRole("button", { name: /watch query demo/i }).click();
  await expect(demo.getByText(/both totals are correct/i)).toBeVisible();
  return demo;
}

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
      page.getByText(/connect loan data, documents and funding requirements once/i),
    ).toBeVisible();
    await expect(
      page.getByRole("link", { name: /explore the live demo/i }).first(),
    ).toBeVisible();
    await expect(
      page.getByRole("link", { name: /book a portfolio walkthrough/i }),
    ).toBeVisible();
  });

  test("the hero CTA lands on the query demo, which waits to be started", async ({
    page,
  }) => {
    await page
      .getByRole("link", { name: /explore the live demo/i })
      .first()
      .click();
    await expect(page).toHaveURL(/#query-demo$/);
    await expect(page.locator("#query-demo")).toBeVisible();
    // Nothing has run yet: the poster and the start affordance are showing.
    await expect(
      page.locator("#example").getByRole("button", { name: /watch query demo/i }),
    ).toBeVisible();
    await expect(page.getByText("Alderbridge Lending Platform").first()).toBeVisible();
  });

  test("starting the query demo runs the scripted question to a governed answer", async ({
    page,
  }) => {
    const demo = await startQueryDemo(page);

    // The differentiator: three books, and BOTH governed totals — the platform
    // on balance sheet, and the sponsor scope including the sold SPV.
    await expect(
      demo.getByRole("heading", { name: "Funded balance by governed book" }),
    ).toBeVisible();
    await expect(demo.getByText("SPV1 Sponsored Securitisation").first()).toBeVisible();
    await expect(demo.getByText("Platform total (warehoused)")).toBeVisible();
    await expect(demo.getByText("Sponsor total (including SPV1)")).toBeVisible();
    // The SPV1 story lives in a tooltip on its own row, not the answer prose.
    await expect(demo.locator('[title*="risk retention"]')).toBeVisible();
    // The as-at date appears once, in the portfolio header — not per answer.
    await expect(demo.getByText(/as at 30 June 2026/)).toHaveCount(1);
    // No internal identifiers reach the page.
    await expect(demo).not.toContainText("ALP_Platform_202606");
    await expect(demo).not.toContainText(/balance coverage/i);
    // The cap is real but silent this early: it is not a meter.
    await expect(demo.getByText(/questions remaining/)).toHaveCount(0);
  });

  test("a typed unsupported question is declined clearly, not guessed", async ({ page }) => {
    const demo = await startQueryDemo(page);

    await demo
      .getByLabel(/ask a question/i)
      .fill("Show me individual loan records with borrower details.");
    await demo.getByRole("button", { name: /ask trakt/i }).click();

    await expect(demo.getByText(/not supported in this demonstration/i)).toBeVisible();
    // Refuses on governance grounds, not a data gap — the stronger claim, and
    // the one an institutional reader is actually worried about.
    await expect(demo.getByText(/will not return exposure-level records/i)).toBeVisible();
    await expect(demo.getByText(/role-based access/i)).toBeVisible();
  });

  test("the conversation can be reset, and the scripted opener does not replay", async ({
    page,
  }) => {
    const demo = await startQueryDemo(page);

    await demo.getByRole("button", { name: /^reset$/i }).click();
    await expect(demo.getByText(/both totals are correct/i)).toHaveCount(0);
    // Suggestions survive the reset, ready for the visitor's own questions.
    await expect(
      demo.getByRole("button", { name: "Show the funded balance by book." }),
    ).toBeVisible();
  });

  test("a report preview renders pages without offering a download", async ({ page }) => {
    const demo = await startQueryDemo(page);

    await demo.getByRole("button", { name: /generate the latest investor report/i }).click();

    await expect(demo.getByText("Investor & Funder MI Pack", { exact: true })).toBeVisible();
    await expect(demo.getByRole("heading", { name: "Executive summary", level: 4 })).toBeVisible();
    await expect(demo.getByText(/preview only, no document is produced/i)).toBeVisible();

    // Page through the preview.
    await demo
      .getByRole("navigation", { name: /report pages/i })
      .getByRole("button", { name: "4" })
      .click();
    await expect(demo.getByText("Geographic exposure")).toBeVisible();

    // Nothing on the page offers a file.
    await expect(page.locator("a[download]")).toHaveCount(0);
    await expect(
      page.locator('a[href*=".pptx"], a[href*=".csv"], a[href*="blob.core"]'),
    ).toHaveCount(0);
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
    for (const output of [
      "Portfolio MI",
      "Forecasting",
      "Risk & covenant controls",
      "Investor reporting",
      "Regulatory reporting",
      "AI & Copilot",
    ]) {
      await expect(platform.getByText(output, { exact: true })).toBeVisible();
    }
  });

  test("the controls demo waits for the visitor, then plays under their control", async ({
    page,
  }) => {
    const controls = page.locator("#controls");
    await controls.scrollIntoViewIfNeeded();

    await expect(
      controls.getByRole("heading", {
        name: /turn portfolio requirements into live controls\./i,
      }),
    ).toBeVisible();
    await expect(controls.getByText(/what the portfolio is moving toward/i)).toBeVisible();
    await expect(
      controls.getByRole("heading", {
        name: /see a portfolio requirement become a live control\./i,
      }),
    ).toBeVisible();

    // Idle: poster showing, nothing playing, and the element can neither
    // autoplay nor loop silently from the middle.
    const video = controls.locator("video");
    await expect(video).toHaveAttribute("poster", /controls-demo-poster/);
    await expect(video).not.toHaveAttribute("autoplay", /.*/);
    await expect(video).not.toHaveAttribute("loop", /.*/);
    expect(await video.evaluate((v: HTMLVideoElement) => v.paused)).toBe(true);
    const play = controls.getByRole("button", { name: /watch controls demo/i });
    await expect(play).toBeVisible();
    await expect(controls.getByText(/~18 sec/)).toBeVisible();

    // Play starts from frame zero.
    await play.click();
    await expect
      .poll(async () => video.evaluate((v: HTMLVideoElement) => !v.paused && v.currentTime > 0))
      .toBe(true);

    // Pause and resume are the visitor's.
    await controls.getByRole("button", { name: /pause demo/i }).click();
    expect(await video.evaluate((v: HTMLVideoElement) => v.paused)).toBe(true);
    await controls.getByRole("button", { name: /resume demo/i }).click();
    await expect
      .poll(async () => video.evaluate((v: HTMLVideoElement) => !v.paused))
      .toBe(true);

    // Restart returns to the beginning.
    await controls.getByRole("button", { name: /restart demo/i }).click();
    expect(await video.evaluate((v: HTMLVideoElement) => v.currentTime)).toBeLessThan(2);

    // Completion offers "Watch again" rather than looping.
    await video.evaluate((v: HTMLVideoElement) => {
      v.currentTime = v.duration - 0.2;
    });
    await expect(controls.getByRole("button", { name: /watch again/i })).toBeVisible();

    // The illustrative provenance is DOM text, not only pixels in the video.
    await expect(controls.getByText(/from documented requirement to live monitoring/i)).toBeVisible();
    await expect(controls.getByText(/figures illustrative/i)).toBeVisible();
  });

  test("reduced-motion visitors see no motion until they ask for it", async ({ page }) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto("/");
    const controls = page.locator("#controls");
    await controls.scrollIntoViewIfNeeded();

    // Poster and play affordance only — nothing runs on its own.
    const video = controls.locator("video");
    expect(await video.evaluate((v: HTMLVideoElement) => v.paused)).toBe(true);
    await expect(controls.getByRole("button", { name: /watch controls demo/i })).toBeVisible();

    // The query demo likewise waits.
    await expect(
      page.locator("#example").getByRole("button", { name: /watch query demo/i }),
    ).toBeVisible();
  });

  test("onboarding detail stays off the homepage", async ({ page }) => {
    // No onboarding section or accordion exists on the page — the homepage
    // makes no onboarding claims, so it can overstate none. Detail belongs
    // on a product page.
    await expect(page.getByText("How onboarding works")).toHaveCount(0);
    await expect(page.locator("#onboarding")).toHaveCount(0);
    await expect(page.locator("main")).not.toContainText(/instant/i);
  });

  test("the lens switcher recomputes the same rows under each lens", async ({ page }) => {
    const lenses = page.locator("#lenses");
    await lenses.scrollIntoViewIfNeeded();

    await expect(
      lenses.getByRole("heading", { name: /one portfolio truth\. every relevant lens\./i }),
    ).toBeVisible();
    await expect(lenses).toContainText(/individually reportable and consolidated/i);

    // Default lens: the consolidated sponsor scope, summed from the very rows
    // on screen — and it reconciles to the governed total to the penny.
    await expect(lenses.getByText("£37,270,061")).toBeVisible();
    await expect(lenses.getByText("SPV1 Sponsored Securitisation")).toBeVisible();
    await expect(lenses.getByText("sold", { exact: true })).toBeVisible();

    // Narrowing the lens recomputes the total over the same rows: the value
    // appears twice — once on the book row, once as the recomputed total.
    await lenses.getByRole("button", { name: "Acquired" }).click();
    await expect(lenses.getByText("£11,974,544")).toHaveCount(2);
    await lenses.getByRole("button", { name: "Origination" }).click();
    await expect(lenses.getByText("£15,432,544")).toHaveCount(2);
    await lenses.getByRole("button", { name: "Sponsor total" }).click();
    await expect(lenses.getByText("£37,270,061")).toBeVisible();
  });

  test("the delivery accordion opens one panel at a time, under user control", async ({
    page,
    isMobile,
  }) => {
    const delivery = page.locator("#delivery");
    await delivery.scrollIntoViewIfNeeded();

    await expect(
      delivery.getByRole("heading", { name: /every mode reads the same governed layer\./i }),
    ).toBeVisible();

    if (isMobile) {
      // Below 768px: a plain vertical stack — every panel visible, no
      // horizontal behaviour, no expand/collapse controls.
      for (const name of [
        "Managed service",
        "Trakt Agent",
        "Copilot",
        "Enterprise agent",
        "Agent-to-agent",
      ]) {
        await expect(delivery.getByRole("heading", { name, level: 3 })).toBeVisible();
      }
      await expect(delivery.getByRole("button")).toHaveCount(0);
      return;
    }

    // The horizontal accordion is the visible surface on desktop; the mobile
    // stack coexists in the DOM behind a display toggle, so assertions scope
    // to the accordion group.
    const accordion = delivery.getByRole("group", { name: "Delivery modes" });

    // Panel 1 open on load, and exactly one panel is ever expanded.
    await expect(accordion.getByText(/recurring reporting, regulatory output/i)).toBeVisible();

    // Selecting another spine moves the expansion — never all-collapsed.
    await accordion.getByRole("button", { name: "Copilot" }).click();
    await expect(accordion.getByText(/inside the tools your teams already use/i)).toBeVisible();
    await expect(accordion.getByText(/recurring reporting, regulatory output/i)).toHaveCount(0);

    // Roadmap panels expand too, and stay labelled as roadmap in grey.
    await accordion.getByRole("button", { name: "Agent-to-agent" }).click();
    await expect(accordion.getByText(/consulting the governed layer directly/i)).toBeVisible();
    await expect(accordion.getByText("Roadmap", { exact: true })).toBeVisible();

    // Arrow keys move between panels.
    await accordion.getByRole("button", { name: "Enterprise agent" }).focus();
    await page.keyboard.press("ArrowLeft");
    const focused = await page.evaluate(
      () =>
        document.activeElement?.getAttribute("aria-label") ??
        document.activeElement?.textContent ??
        "",
    );
    expect(focused).toMatch(/copilot/i);
  });

  test("governance makes four claims once, with the reconciliation proof", async ({ page }) => {
    const governance = page.locator("#governance");
    await governance.scrollIntoViewIfNeeded();

    await expect(
      governance.getByRole("heading", {
        name: /deterministic underneath\. governed throughout\./i,
      }),
    ).toBeVisible();
    // The hero's old proof line now leads the trust section.
    await expect(
      governance.getByText(/reconciled by construction rather than by comparison/i),
    ).toBeVisible();
    for (const name of ["Deterministic", "Traceable", "Controlled", "Isolated"]) {
      await expect(governance.getByRole("heading", { name, level: 3 })).toBeVisible();
    }
    await expect(governance.getByText(/microsoft entra id/i)).toBeVisible();
    // Asset extensibility is an architecture claim, never a coverage claim,
    // and the agentic direction is design intent, never live capability.
    await expect(governance.getByText(/asset-specific configuration/i)).toBeVisible();
    await expect(governance).not.toContainText(/every asset class/i);
    await expect(governance.getByText(/toward increasingly agentic operation/i)).toBeVisible();
    await expect(governance).not.toContainText(/roadmap/i);
    await expect(governance).not.toContainText(/autonomous/i);

    // Governance flows straight into the closing CTA — nothing sits between.
    const nextSectionId = await page
      .locator("#governance")
      .evaluate((section) => section.nextElementSibling?.id ?? "");
    expect(nextSectionId).toBe("book-a-demo");
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

  test("the intelligence section covers distribution without repeating the demo", async ({
    page,
  }) => {
    const intelligence = page.locator("#intelligence");
    await intelligence.scrollIntoViewIfNeeded();

    for (const channel of ["Trakt workspace", "Microsoft Teams", "Microsoft 365 Copilot"]) {
      await expect(intelligence.getByText(channel, { exact: true })).toBeVisible();
    }
    // Three channel chips, each carrying a small neutral glyph — labels do the
    // naming, no imitation product logos.
    await expect(intelligence.locator("span:has(> svg)")).toHaveCount(3);
    await expect(intelligence.getByText("Available today")).toBeVisible();
    // The query demo lives in section 2 — no duplicate here.
    await expect(intelligence.locator("#example")).toHaveCount(0);
    await expect(intelligence.locator("video")).toHaveCount(0);
  });

  test("the synthetic disclosure is the amber pill, stated exactly once", async ({
    page,
  }) => {
    // Before the demo starts: one pill, on the poster's portfolio header.
    await expect(page.getByText("Synthetic data", { exact: true })).toHaveCount(1);
    await expect(page.getByText(/wholly synthetic/i)).toHaveCount(0);
    await expect(page.getByText("Synthetic portfolio", { exact: true })).toHaveCount(0);

    // After it starts: still exactly one, now on the live demo's header.
    const demo = await startQueryDemo(page);
    await expect(page.getByText("Synthetic data", { exact: true })).toHaveCount(1);

    // Refusal is promoted out of small print: a titled block with the two
    // example prompts as first-class buttons.
    await expect(
      demo.getByRole("heading", { name: /trakt declines what it cannot derive/i }),
    ).toBeVisible();
    await expect(
      demo.getByRole("button", { name: /show me individual loan records/i }),
    ).toBeVisible();
    await expect(
      demo.getByRole("button", { name: /summarise the current pipeline/i }),
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

  test("the hero preview and the demo poster agree on the portfolio", async ({
    page,
    request,
  }) => {
    const meta = await (await request.get("/api/demo/meta")).json();
    const { totalBalanceDisplay, loanCount, asOfDisplay, client } = meta.scope;

    // The figure in the hero is the same figure the demo answers with — and
    // the poster carries the scope before the demo is even started.
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

    const demo = await startQueryDemo(page);
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
      page.waitForResponse(
        (res) => res.url().includes("/api/leads") && res.request().method() === "POST",
      ),
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
    await startQueryDemo(page);

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
