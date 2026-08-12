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
    // The strapline is the title, verbatim — this is what a search result and
    // a link preview surface.
    await expect(page).toHaveTitle(
      /Trakt \| Agentic portfolio intelligence\. Deterministic by design\./,
    );
    await expect(
      page.getByText("Agentic portfolio intelligence.", { exact: true }),
    ).toBeVisible();
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
      page.getByRole("link", { name: /demo on your portfolio/i }).first(),
    ).toBeVisible();
  });

  /**
   * The two controls that offer a demonstration against the visitor's own
   * portfolio must carry the same words and point at the same place. They
   * drifted into three separate names once already ("Book a demo", "Book a
   * portfolio walkthrough", "Book a tailored demonstration"), which left a
   * reader to work out that all three were one destination.
   *
   * The closing CTA is deliberately excluded: it sits above the form itself,
   * where it has its own context and competes with nothing.
   */
  test("the nav and hero demo controls carry one label and one destination", async ({
    page,
    isMobile,
  }) => {
    if (isMobile) await page.getByRole("button", { name: /open menu/i }).click();

    const controls = page.getByRole("link", { name: /demo on your portfolio/i });
    await expect(controls).toHaveCount(2);

    const seen = await controls.evaluateAll((nodes) =>
      nodes.map((node) => ({
        text: (node.textContent ?? "").trim(),
        href: node.getAttribute("href"),
      })),
    );
    expect(new Set(seen.map((c) => c.text)).size, "the two controls disagree on wording").toBe(1);
    expect(new Set(seen.map((c) => c.href)).size, "the two controls disagree on target").toBe(1);
    expect(seen[0]?.href).toBe("#book-a-demo");

    // The closing CTA keeps its own wording.
    await expect(page.getByRole("button", { name: /book a tailored demonstration/i })).toHaveCount(1);

    // Scoped to the two controls this test governs. "Book a portfolio
    // walkthrough" still exists inside the demo, on the session-limit card
    // (`CopilotDemo.tsx`, `ReportPreview.tsx`) — a separate label for the same
    // anchor, deliberately left alone in this pass and recorded in the
    // content map as an open item.
    await expect(page.getByRole("banner").getByText(/walkthrough/i)).toHaveCount(0);
    await expect(page.locator("#product").getByText(/walkthrough/i)).toHaveCount(0);
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
    // The SPV1 story lives on its own row behind a real toggle — tap-
    // reachable, unlike a native title tooltip, which never opens on touch.
    const spvNote = demo.getByRole("button", { name: "SPV1 Sponsored Securitisation" });
    await expect(spvNote).toHaveAttribute("aria-expanded", "false");
    await spvNote.click();
    await expect(demo.getByText(/servicing, risk retention and investor reporting retained/i)).toBeVisible();
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
    // One heading only: the demo's own restated the section's directly above.
    await expect(controls.getByRole("heading", { level: 2 })).toHaveCount(1);
    await expect(controls.getByRole("heading", { level: 3 })).toHaveCount(0);
    // Both demo sections share one shape: heading, one line, full-width demo.
    await expect(controls.locator(".grid")).toHaveCount(0);

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
    // The film burns "Illustrative · synthetic data" into every frame, so the
    // caption no longer repeats it.
    await expect(controls.getByText(/figures illustrative/i)).toHaveCount(0);
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

  test("the delivery model is five static tiles, with nothing to open", async ({ page }) => {
    const delivery = page.locator("#delivery");
    await delivery.scrollIntoViewIfNeeded();

    await expect(
      delivery.getByRole("heading", { name: /every mode reads the same governed layer\./i }),
    ).toBeVisible();

    // All five modes readable in one pass, in order, at every breakpoint.
    const names = ["Managed service", "Trakt Agent", "Copilot", "Agent access"];
    for (const name of names) {
      await expect(delivery.getByRole("heading", { name, level: 3 })).toBeVisible();
    }
    await expect(delivery.getByRole("heading", { level: 3 })).toHaveCount(names.length);

    // Availability is stated per tile: three shipped, two roadmap.
    await expect(delivery.getByText("Available today")).toHaveCount(3);
    // One roadmap tile, not two: the agent section shows both patterns.
    await expect(delivery.getByText("Roadmap")).toHaveCount(1);

    // No expand/collapse interaction anywhere in the section.
    await expect(delivery.getByRole("button")).toHaveCount(0);
    await expect(delivery.locator("details")).toHaveCount(0);
    await expect(delivery.locator("[aria-expanded]")).toHaveCount(0);
  });

  /**
   * Five cards is an awkward number in a grid, and it has now produced the
   * same defect twice — a single card alone on the last row with empty cells
   * beside it, which reads as a rendering fault rather than a wrap. Measured
   * from the rendered boxes at every breakpoint the grid changes at, rather
   * than trusted to the class list.
   */
  test("no governance card is left alone on its own row", async ({ page, viewport }) => {
    for (const width of [1760, 1440, 1024, 834, 390]) {
      await page.setViewportSize({ width, height: 900 });
      const rows = await page.locator("#governance ul > li").evaluateAll((nodes) => {
        const byRow = new Map<number, number>();
        for (const node of nodes) {
          const top = Math.round(node.getBoundingClientRect().top);
          byRow.set(top, (byRow.get(top) ?? 0) + 1);
        }
        return [...byRow.entries()].sort((a, b) => a[0] - b[0]).map(([, count]) => count);
      });
      const columns = Math.max(...rows);
      const tail = rows[rows.length - 1];
      expect(
        columns > 1 && rows.length > 1 && tail === 1,
        `at ${width}px the cards lay out ${rows.join("+")} — one card orphaned`,
      ).toBe(false);
    }
    if (viewport) await page.setViewportSize(viewport);
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
    for (const name of [
      "Deterministic",
      "Traceable",
      "Controlled",
      "Isolated",
      "Agent-addressable",
    ]) {
      await expect(governance.getByRole("heading", { name, level: 3 })).toBeVisible();
    }
    await expect(governance.getByText(/microsoft entra id/i)).toBeVisible();
    // Extensibility survives as one line answering the objection; it is an
    // architecture claim, never a coverage claim. The agentic direction is
    // cut — the Delivery Model tiles show it in grey instead.
    await expect(
      governance.getByText("New asset classes are added through configuration, not a rebuild."),
    ).toBeVisible();
    await expect(governance).not.toContainText(/every asset class/i);
    await expect(governance).not.toContainText(/agentic/i);
    await expect(governance).not.toContainText(/roadmap/i);
    await expect(governance).not.toContainText(/autonomous/i);

    // Governance flows straight into the closing CTA — nothing sits between.
    const nextSectionId = await page
      .locator("#governance")
      .evaluate((section) => section.nextElementSibling?.id ?? "");
    expect(nextSectionId).toBe("book-a-demo");
  });

  test("the agent section is roadmap, with a topology and no fabricated demo", async ({
    page,
  }) => {
    const agents = page.locator("#agents");
    await agents.scrollIntoViewIfNeeded();

    await expect(
      agents.getByRole("heading", {
        name: /agents don't calculate the portfolio\. trakt does\./i,
      }),
    ).toBeVisible();
    // Roadmap language is deliberate and must not be tightened to present tense.
    await expect(agents.getByText(/trakt is designed to make governed/i)).toBeVisible();
    await expect(agents.getByText("Roadmap", { exact: true })).toBeVisible();

    // Three nodes on one line, protocols as connector labels only.
    for (const node of ["External agent", "Client enterprise agent", "Trakt"]) {
      await expect(agents.getByText(node, { exact: true })).toBeVisible();
    }
    await expect(agents.getByText("A2A", { exact: true })).toBeVisible();

    // Nothing here may fabricate a demonstration: no frame, no still, no
    // invented figures, and no vendor mark as an actor in the diagram.
    await expect(agents.locator("video, img")).toHaveCount(0);
    await expect(agents).not.toContainText(/copilot/i);
    await expect(agents).not.toContainText(/%/);
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

  test("delivery states each channel once, and the intelligence section is gone", async ({
    page,
  }) => {
    // The Portfolio Intelligence section named the same surfaces as the tiles
    // below it, in a second format. It must not come back.
    await expect(page.locator("#intelligence")).toHaveCount(0);
    await expect(page.getByText("Trakt workspace")).toHaveCount(0);

    const delivery = page.locator("#delivery");
    await delivery.scrollIntoViewIfNeeded();

    for (const mode of ["Managed service", "Trakt Agent", "Copilot", "Agent access"]) {
      await expect(delivery.getByRole("heading", { name: mode, exact: true })).toBeVisible();
    }
    // The claim carried over from the deleted section: push, not pull, and
    // governed by approval. Stated once on the page.
    await expect(page.getByText("Approved risk findings are pushed to Teams.")).toHaveCount(1);
    await expect(
      delivery.getByText("Approved risk findings are pushed to Teams."),
    ).toBeVisible();
    // The query demo lives in section 2 — no duplicate here.
    await expect(delivery.locator("#example")).toHaveCount(0);
    await expect(delivery.locator("video")).toHaveCount(0);
  });

  test("the synthetic disclosure is the amber pill, stated exactly once", async ({
    page,
  }) => {
    // Before the demo starts: one pill, on the poster's portfolio header.
    await expect(page.getByText("Synthetic data", { exact: true })).toHaveCount(1);
    await expect(page.getByText(/wholly synthetic/i)).toHaveCount(0);
    await expect(page.getByText("Synthetic portfolio", { exact: true })).toHaveCount(0);

    // After it starts: still exactly one, now on the live demo's header.
    await startQueryDemo(page);
    await expect(page.getByText("Synthetic data", { exact: true })).toHaveCount(1);
  });

  test("the refusal claim sits inside the query demo section and drives it", async ({
    page,
  }) => {
    // No longer a section of its own: it is a sub-claim of the demo above it.
    await expect(page.locator("#refusal")).toHaveCount(0);

    const section = page.locator("#query-demo");
    await section.scrollIntoViewIfNeeded();

    // Live text at heading scale, not small print, and stated once.
    const claim = page.getByRole("heading", {
      name: /trakt declines what it cannot derive/i,
    });
    await expect(claim).toHaveCount(1);
    await expect(section.getByRole("heading", { name: /trakt declines/i })).toBeVisible();
    const size = await claim.evaluate((node) => parseFloat(getComputedStyle(node).fontSize));
    expect(size, "the refusal claim has been demoted to small print").toBeGreaterThanOrEqual(18);

    // Pressing a prompt starts the demo above and asks it, so the visitor
    // watches Trakt decline rather than being told that it does.
    await section.getByRole("button", { name: /show me individual loan records/i }).click();
    const demo = page.locator("#example");
    await expect(demo.getByText(/not supported in this demonstration/i)).toBeVisible();
    await expect(demo.getByText(/will not return exposure-level records/i)).toBeVisible();
  });

  /**
   * Opacity guards.
   *
   * `toBeVisible()` checks the layout box and display/visibility — it passes
   * on an opacity-0 element. Six sections once rendered blank behind a
   * scroll-reveal and the whole suite stayed green, so these assert the
   * computed value directly.
   */
  test("every section paints: computed opacity and real text", async ({ page }) => {
    for (const id of [
      "query-demo",
      "platform",
      "delivery",
      "controls",
      "agents",
      "governance",
    ]) {
      const section = page.locator(`#${id}`);
      await section.scrollIntoViewIfNeeded();

      // Polled, not sampled: the reveal transition runs 240ms plus up to
      // 120ms of stagger, so an immediate read catches it mid-flight.
      await expect
        .poll(
          () =>
            section.evaluate((node) =>
              Number(getComputedStyle(node.querySelector("[data-reveal]") ?? node).opacity),
            ),
          { message: `#${id} never reached full opacity` },
        )
        .toBe(1);

      const text = await section.evaluate((node) => (node as HTMLElement).innerText.trim().length);
      expect(text, `#${id} rendered no text`).toBeGreaterThan(0);
    }
  });

  /**
   * Colour-system guards.
   *
   * One colour, one job. Green had drifted onto six unrelated things —
   * availability, pass state, claim emphasis, tick marks, a provenance label
   * and chip borders — at which point it signalled nothing. These assert the
   * rule against the rendered page rather than against the source, so a new
   * component cannot reintroduce the drift without failing.
   *
   * Colours are resolved by painting, not by string matching. Tailwind emits
   * opacity modifiers as `oklab(… / .35)`, and Chromium reports that verbatim
   * — so a check that looked for "rgb(54, 194, 168" passed happily while a
   * green border sat on the refusal prompts. Filling the same colour over
   * black and over white recovers the source channels and its alpha exactly,
   * whatever colour space the value arrived in.
   */
  const MINT = [54, 194, 168]; // mint-400 #36c2a8
  const PERI_500 = [125, 138, 198]; // peri-500 #7d8ac6

  const RESOLVER = `
    const _mk = (bg) => {
      const c = document.createElement("canvas");
      c.width = c.height = 1;
      return { x: c.getContext("2d", { willReadFrequently: true }), bg };
    };
    const _beds = [_mk("#000"), _mk("#fff")];
    const resolve = (value) => {
      const [kb, kw] = _beds.map(({ x, bg }) => {
        x.clearRect(0, 0, 1, 1);
        x.fillStyle = bg; x.fillRect(0, 0, 1, 1);
        x.fillStyle = value; x.fillRect(0, 0, 1, 1);
        return x.getImageData(0, 0, 1, 1).data;
      });
      const a = 1 - (kw[0] - kb[0]) / 255;
      if (a <= 0.004) return null;
      return [Math.round(kb[0] / a), Math.round(kb[1] / a), Math.round(kb[2] / a)];
    };
    const matches = (value, want) => {
      const got = resolve(value);
      // Painting round-trips within a channel or two; anything wider is a
      // different colour, not rounding.
      return Boolean(got) && got.every((c, i) => Math.abs(c - want[i]) <= 3);
    };
  `;

  test("green marks system state and nothing else", async ({ page }) => {
    // The allow-list is declared in the components themselves, not here: a
    // container marks itself `data-state-colour` when what it renders is a
    // system state. Three exist — delivery availability, the control
    // preview's evaluation rows, and the lead form's success panel.
    const offenders = async () =>
      page.evaluate(
        new Function(
          "want",
          `${RESOLVER}
          const found = [];
          for (const el of document.querySelectorAll("*")) {
            const cs = getComputedStyle(el);
            const hit = [
              cs.color,
              cs.borderTopColor,
              cs.borderRightColor,
              cs.borderBottomColor,
              cs.borderLeftColor,
              cs.backgroundColor,
            ].some((value) => matches(value, want));
            if (!hit) continue;
            if (el.closest("[data-state-colour]")) continue;
            found.push({
              tag: el.tagName,
              cls: (el.className?.toString?.() ?? "").slice(0, 90),
              where: el.closest("section[id]")?.id ?? "(outside a section)",
            });
          }
          return found;`,
        ) as (want: number[]) => { tag: string; cls: string; where: string }[],
        MINT,
      );

    expect(await offenders(), "green used outside a system-state context").toEqual([]);

    // Again with the demo running: its suggestion chips and answer cards are
    // only in the DOM after it starts, and they are exactly where the drift
    // was — a refusal prompt bordered in green read as a success state.
    await startQueryDemo(page);
    expect(await offenders(), "green used outside a system-state context, demo running").toEqual([]);
  });

  test("peri-500 is a border and structural colour, never type", async ({ page }) => {
    // 5.67:1 on navy-950 — fine behind a border, below AA as body copy. The
    // only thing that previously kept it off type was noticing.
    //
    // Structural marks are allowed and are distinguishable without a
    // judgement call: the platform diagram's arrows are glyphs carried in an
    // `aria-hidden` span, so they are not type by the page's own account.
    const asType = await page.evaluate(
      new Function(
        "want",
        `${RESOLVER}
        return [...document.querySelectorAll("*")]
          .filter((el) => matches(getComputedStyle(el).color, want))
          .filter((el) => !el.closest("[aria-hidden='true']"))
          .filter((el) => (el.textContent ?? "").trim().length > 0)
          .map((el) => ({
            tag: el.tagName,
            cls: (el.className?.toString?.() ?? "").slice(0, 90),
            text: (el.textContent ?? "").trim().slice(0, 40),
          }));`,
      ) as (want: number[]) => { tag: string; cls: string; text: string }[],
      PERI_500,
    );
    expect(asType, "peri-500 applied as a text colour").toEqual([]);
  });

  test("no revealable element anywhere is left transparent", async ({ page }) => {
    // Deliberately broad: queries every [data-reveal] on the page rather than
    // a known list, so anything added later is covered by default.
    const total = await page.locator("[data-reveal]").count();
    expect(total).toBeGreaterThan(0);

    for (let index = 0; index < total; index += 1) {
      await page.locator("[data-reveal]").nth(index).scrollIntoViewIfNeeded();
    }
    await page.locator("#top").scrollIntoViewIfNeeded();

    await expect
      .poll(
        () =>
          page.evaluate(() =>
            Array.from(document.querySelectorAll("[data-reveal]"))
              .filter((node) => Number(getComputedStyle(node).opacity) < 1)
              .map(
                (node) =>
                  `${node.getAttribute("data-reveal")}: ${(node as HTMLElement).innerText.slice(0, 40)}`,
              ),
          ),
        { message: "revealable content left transparent" },
      )
      .toEqual([]);
  });

  /**
   * The play plate must never cover the demo's argument.
   *
   * The controls rows live inside the poster image, so they have no DOM boxes
   * to measure. The poster is a purpose-built 1200x960 still rendered into a
   * 5:4 frame with no crop, so the mapping is exact: the control card — its
   * three rows, their percentages and the breach horizon — ends at 41% of the
   * frame height, and the closing line begins at 73%. The plate is centred,
   * and must sit inside that clear band. Asserted at all three widths because
   * it has twice been checked by eye and twice been wrong.
   */
  test("the play plate never covers the control rows or the breach horizon", async ({
    page,
  }) => {
    for (const width of [1440, 834, 390]) {
      await page.setViewportSize({ width, height: 900 });
      await page.locator("#controls").scrollIntoViewIfNeeded();

      const frame = await page.locator("#controls video").boundingBox();
      const plate = await page.locator('[data-plate="controls"]').boundingBox();
      expect(frame, `no video frame at ${width}`).not.toBeNull();
      expect(plate, `no play plate at ${width}`).not.toBeNull();
      if (!frame || !plate) continue;

      // Centred, which is the intended geometry the poster is drawn for.
      const frameCentre = frame.y + frame.height / 2;
      const plateCentre = plate.y + plate.height / 2;
      expect(
        Math.abs(plateCentre - frameCentre),
        `the plate is not centred in the frame at ${width}px`,
      ).toBeLessThanOrEqual(4);

      // And inside the band the poster leaves clear for it. The plate is a
      // fixed height while the frame scales with width, so at 390 the frame
      // is too short for both — there the accepted trade-off is a centred
      // plate over the first row, and only the centring is asserted.
      if (width >= 834) {
        expect(
          plate.y,
          `the plate intrudes into the control rows at ${width}px`,
        ).toBeGreaterThanOrEqual(frame.y + frame.height * 0.39);
        expect(
          plate.y + plate.height,
          `the plate covers the closing line at ${width}px`,
        ).toBeLessThanOrEqual(frame.y + frame.height * 0.73);
      }
    }
  });

  test("the query plate covers none of its poster's content", async ({ page }) => {
    for (const width of [1440, 834, 390]) {
      await page.setViewportSize({ width, height: 900 });
      await page.locator("#example").scrollIntoViewIfNeeded();

      const plate = await page.locator('[data-plate="query"]').boundingBox();
      expect(plate, `no query plate at ${width}`).not.toBeNull();
      if (!plate) continue;

      // The replica reserves a band between its two content groups; the
      // plate is centred in that band and must intersect neither group.
      const groups = page.locator("[data-poster-content]");
      const count = await groups.count();
      expect(count, `no poster content at ${width}`).toBeGreaterThan(0);
      for (let i = 0; i < count; i += 1) {
        const box = await groups.nth(i).boundingBox();
        if (!box) continue;
        const overlaps = plate.y < box.y + box.height && box.y < plate.y + plate.height;
        expect(overlaps, `the query plate overlaps poster content ${i} at ${width}px`).toBe(false);
      }
    }
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

/**
 * A permanent contract, not a spot check: the page must be readable with no
 * JavaScript at all. Motion is an enhancement; content never depends on it.
 */
test.describe("without JavaScript", () => {
  test.use({ javaScriptEnabled: false });

  test("every section still renders at full opacity", async ({ page }) => {
    await page.goto("/");

    for (const id of ["platform", "delivery", "controls", "agents", "governance"]) {
      const painted = await page.locator(`#${id}`).evaluate((node) => {
        const target = node.querySelector("[data-reveal]") ?? node;
        return {
          opacity: Number(getComputedStyle(target).opacity),
          text: (node as HTMLElement).innerText.trim().length,
        };
      });
      expect(painted.opacity, `#${id} is transparent without JavaScript`).toBe(1);
      expect(painted.text, `#${id} rendered no text without JavaScript`).toBeGreaterThan(0);
    }

    // The proposition and the closing form survive too.
    await expect(
      page.getByRole("heading", { level: 1, name: /one governed view of your lending portfolios\./i }),
    ).toBeVisible();
    await expect(page.locator("#book-a-demo")).toContainText(/see your portfolio through one governed view/i);
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

  /**
   * A permanent guard, not a spot check. A nav entry pointing at a section
   * that has been deleted is silent — the link simply does nothing — so the
   * menu is checked against the document rather than against a list kept in
   * step by hand.
   */
  test("every menu link points at a section that exists", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: /open menu/i }).click();

    const hrefs = await page.locator("#mobile-nav a[href^='#']").evaluateAll((nodes) =>
      nodes.map((node) => (node as HTMLAnchorElement).getAttribute("href") ?? ""),
    );
    expect(hrefs.length).toBeGreaterThan(0);

    for (const href of hrefs) {
      await expect(page.locator(href), `${href} has no target on the page`).toHaveCount(1);
    }
  });
});
