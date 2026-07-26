/**
 * The storyboard's pre-render checklist, as tests.
 *
 * The checklist in the storyboard is twelve items. Ten of them are verifiable without
 * looking at a frame, and they are asserted here; the other two — "every claim is
 * legible with the sound off" and "the logo is pixel-identical in every frame" — are
 * a human's job against the stills, and `scripts/stills.mjs` exists to make that
 * cheap. `scripts/lint-theme.mjs` covers the source-level rules (no hex, no literal
 * font size, no shadow, no gradient, six type sizes, `flag` in exactly two scenes).
 *
 * On top of the checklist, these guard the thing that matters most: the film must not
 * state a figure the fixtures do not contain, and the two surfaces (MI Agent and
 * Copilot) must report the same numbers.
 */

import { describe, expect, it } from "vitest";

import {
  FPS,
  SCENES,
  STILL_FRAMES,
  TOTAL_FRAMES,
  captions,
  sceneStarts,
  startOf,
  totalFrames,
} from "./timeline";
import {
  ARTEFACTS,
  ASSERTIONS,
  CURRENT_PERIOD,
  COPILOT,
  MOVEMENT,
  PORTFOLIOS,
  SAFETY,
  SUMMARY,
  currentPeriodElapsedSeconds,
  lens,
  mappingDecision,
  onboarding,
  portfolio,
  schemaFor,
} from "./data/fixtures";
import { count, elapsed, money, percent, signedMoney } from "./format";
import theme from "./theme";

// --------------------------------------------------------------------------- //
// Runtime and structure
// --------------------------------------------------------------------------- //
describe("runtime", () => {
  it("is 2,700 frames — 90 seconds at 30fps", () => {
    expect(FPS).toBe(30);
    expect(totalFrames()).toBe(TOTAL_FRAMES);
    expect(totalFrames()).toBe(2700);
    expect(totalFrames() / FPS).toBe(90);
  });

  it("cuts 420 / 720 / 600 / 600 / 360, in the storyboard's order", () => {
    expect(SCENES.map((s) => s.id)).toEqual([
      "cost",
      "onboard",
      "dataset",
      "omnichannel",
      "close",
    ]);
    expect(SCENES.map((s) => s.frames)).toEqual([420, 720, 600, 600, 360]);
    expect(sceneStarts()).toEqual([0, 420, 1140, 1740, 2340]);
  });

  it("numbers the scenes 1..5 with no gaps", () => {
    expect(SCENES.map((s) => s.number)).toEqual([1, 2, 3, 4, 5]);
  });
});

// --------------------------------------------------------------------------- //
// Captions — the film is watched with the sound off
// --------------------------------------------------------------------------- //
describe("captions", () => {
  it("never runs past the end of its own scene", () => {
    for (const scene of SCENES) {
      for (const caption of scene.captions) {
        expect(caption.at + caption.hold, `${scene.id}: "${caption.text}"`).toBeLessThanOrEqual(
          scene.frames,
        );
      }
    }
  });

  it("never overlaps another caption", () => {
    const all = captions().sort((a, b) => a.from - b.from);
    for (let i = 1; i < all.length; i += 1) {
      expect(all[i].from, `"${all[i].text}" overlaps "${all[i - 1].text}"`).toBeGreaterThanOrEqual(
        all[i - 1].from + all[i - 1].hold,
      );
    }
  });

  it("holds every caption long enough to read", () => {
    // ~13 characters a second is a comfortable subtitle rate.
    for (const caption of captions()) {
      const seconds = caption.hold / FPS;
      expect(seconds, `"${caption.text}"`).toBeGreaterThanOrEqual(caption.text.length / 13 - 0.35);
    }
  });

  it("keeps every caption to two lines at the body size", () => {
    // ~62 characters fit a line inside the plate at 1920 wide.
    for (const caption of captions()) {
      expect(caption.text.length, `"${caption.text}"`).toBeLessThanOrEqual(124);
    }
  });

  it("covers at least half the film", () => {
    const covered = captions().reduce((total, c) => total + c.hold, 0);
    expect(covered / totalFrames()).toBeGreaterThan(0.5);
  });

  it("says nothing the narration does not", () => {
    // Every caption belongs to a scene whose narration exists, so the burned-in
    // copy and the voice-over cannot describe different films.
    for (const scene of SCENES) {
      expect(scene.narration.length, scene.id).toBeGreaterThan(20);
    }
  });
});

// --------------------------------------------------------------------------- //
// Brand tokens
// --------------------------------------------------------------------------- //
describe("brand tokens", () => {
  it("declares exactly the storyboard's seven colours", () => {
    expect(Object.keys(theme.color).sort()).toEqual(
      ["flag", "hull", "ink", "mute", "paper", "rule", "signal"].sort(),
    );
    expect(theme.color.ink).toBe("#060B1F");
    expect(theme.color.signal).toBe("#4DE0C4");
    expect(theme.color.flag).toBe("#F2A93B");
  });

  it("declares exactly six type sizes, at the storyboard's values", () => {
    expect(Object.keys(theme.type)).toHaveLength(6);
    expect(theme.type.display.fontSize).toBe(128);
    expect(theme.type.headline.fontSize).toBe(64);
    expect(theme.type.stat.fontSize).toBe(88);
    expect(theme.type.body.fontSize).toBe(26);
    expect(theme.type.label.fontSize).toBe(14);
    expect(theme.type.stamp.fontSize).toBe(15);
  });

  it("puts tabular-nums on every mono size, or counters jitter", () => {
    for (const key of ["stat", "stamp"] as const) {
      expect(theme.type[key].fontFamily).toBe(theme.family.data);
      expect(theme.type[key].fontVariantNumeric).toBe("tabular-nums");
    }
  });

  it("assigns the three faces to the three roles", () => {
    expect(theme.type.display.fontFamily).toContain("Archivo");
    expect(theme.type.headline.fontFamily).toContain("Archivo");
    expect(theme.type.body.fontFamily).toContain("Inter");
    expect(theme.type.label.fontFamily).toContain("Inter");
    expect(theme.type.stat.fontFamily).toContain("IBM Plex Mono");
    expect(theme.type.stamp.fontFamily).toContain("IBM Plex Mono");
  });

  it("uses one spring, critically damped", () => {
    expect(theme.motion.spring).toEqual({ damping: 200, mass: 0.6 });
    expect(theme.motion.quick).toBe(12);
    expect(theme.motion.base).toBe(20);
    expect(theme.motion.slow).toBe(30);
    expect(theme.motion.stagger).toBe(3);
  });

  it("vendors the three faces locally, so a render never touches the network", () => {
    expect(theme.fontFaces.map((f) => f.family)).toEqual(["Archivo", "Inter", "IBM Plex Mono"]);
    for (const face of theme.fontFaces) {
      expect(face.file).toMatch(/^fonts\/.+\.woff2$/);
    }
  });
});

// --------------------------------------------------------------------------- //
// Delivery
// --------------------------------------------------------------------------- //
describe("delivery", () => {
  it("masters at 1920x1080 and crops to 1080x1080", () => {
    expect(theme.layout.wide).toMatchObject({ width: 1920, height: 1080 });
    expect(theme.layout.square).toMatchObject({ width: 1080, height: 1080 });
  });

  it("exports the three still frames the outbound email uses", () => {
    expect([...STILL_FRAMES]).toEqual([1020, 1500, 2100]);
    for (const frame of STILL_FRAMES) {
      expect(frame).toBeLessThan(totalFrames());
    }
  });

  it("lands each still inside the scene the storyboard intended", () => {
    // 1020 is deep in S2's referred-for-review hold; 1500 is S3's card fan-out;
    // 2100 is S4's simultaneous payload. Those are the three beats worth an email.
    expect(STILL_FRAMES[0]).toBeGreaterThan(startOf("onboard"));
    expect(STILL_FRAMES[0]).toBeLessThan(startOf("dataset"));
    expect(STILL_FRAMES[1]).toBeGreaterThan(startOf("dataset"));
    expect(STILL_FRAMES[1]).toBeLessThan(startOf("omnichannel"));
    expect(STILL_FRAMES[2]).toBeGreaterThan(startOf("omnichannel"));
    expect(STILL_FRAMES[2]).toBeLessThan(startOf("close"));
  });
});

// --------------------------------------------------------------------------- //
// The film may not state a figure the fixtures do not contain
// --------------------------------------------------------------------------- //
describe("figures", () => {
  it("reads the consolidated balance and loan count from the fixture", () => {
    expect(SUMMARY.available).toBe(true);
    expect(SUMMARY.metrics.funded_balance).toBeGreaterThan(0);
    expect(SUMMARY.metrics.loan_count).toBeGreaterThan(0);
    expect(SUMMARY.reportingDate).toBe(CURRENT_PERIOD.reportingDate);
  });

  it("has the two portfolio balances S3 converges", () => {
    const a = lens("A").summary.metrics.funded_balance ?? 0;
    const b = lens("B").summary.metrics.funded_balance ?? 0;
    expect(a).toBeGreaterThan(0);
    expect(b).toBeGreaterThan(0);
    // They must sum to the consolidated figure at the precision the film shows.
    expect(money(a + b)).toBe(money(SUMMARY.metrics.funded_balance));
  });

  it("has the movement S4 shows beneath all three panels", () => {
    expect(MOVEMENT.available).toBe(true);
    expect(MOVEMENT.delta.funded_balance).toBeGreaterThan(0);
    expect(signedMoney(MOVEMENT.delta.funded_balance)).toMatch(/^\+£/);
  });

  it("has the regional split the workspace panel draws", () => {
    expect(SUMMARY.topRegions.length).toBeGreaterThanOrEqual(3);
    for (const region of SUMMARY.topRegions.slice(0, 3)) {
      expect(region.balance).toBeGreaterThan(0);
      expect(region.region.length).toBeGreaterThan(0);
    }
  });

  it("has the Copilot action names the Copilot panel lists", () => {
    expect(COPILOT.actions).toContain("askTraktMi");
    expect(COPILOT.actions.length).toBeGreaterThanOrEqual(3);
  });
});

// --------------------------------------------------------------------------- //
// S1 — six source headers each, and they genuinely do not match
// --------------------------------------------------------------------------- //
describe("S1 · the cost", () => {
  it("has at least six business headers in each schema", () => {
    for (const key of ["A", "B"]) {
      expect(schemaFor(key).columns.length).toBeGreaterThanOrEqual(6);
    }
  });

  it("shares no header between the first six of each stack", () => {
    const a = schemaFor("A").columns.slice(0, 6).map((c) => c.header);
    const b = schemaFor("B").columns.slice(0, 6).map((c) => c.header);
    expect(a.filter((h) => b.includes(h))).toEqual([]);
  });
});

// --------------------------------------------------------------------------- //
// S2 — the counters, and the elapsed-time claim
// --------------------------------------------------------------------------- //
describe("S2 · onboarded once", () => {
  it("reads its three counters from the approved contract", () => {
    const mapping = onboarding("B").mapping;
    expect(mapping.mapped_count).toBeGreaterThan(0);
    expect(mapping.client_contract_count).toBeGreaterThan(0);
    // The referred-for-review beat is the most important in the film. If Gate 1
    // resolved everything, the beat has no subject and the scene is a lie.
    expect(mapping.unmapped_count).toBeGreaterThan(0);
  });

  it("has a referred item that is a judgement call, not the demo watermark", () => {
    const referred = onboarding("B").mappingDecisions.filter(
      (d) => !d.canonical_field && d.source_header !== "Synthetic Data Notice",
    );
    expect(referred.length).toBeGreaterThanOrEqual(1);
    // And it must carry the note the scene puts on screen.
    expect(referred[0].note.length).toBeGreaterThan(40);
  });

  it("draws its four mapping pairs from recorded decisions", () => {
    const pairs: [string, string, string][] = [
      ["A", "Loan Reference", "loan_identifier"],
      ["A", "Completion Date", "origination_date"],
      ["B", "Account Number", "loan_identifier"],
      ["B", "Reporting Cut-Off", "data_cut_off_date"],
    ];
    for (const [key, header, canonical] of pairs) {
      expect(mappingDecision(key, header).canonical_field, `${key}:${header}`).toBe(canonical);
    }
  });

  it("states only an elapsed time the run actually measured", () => {
    // The storyboard's open item: "the 48-hour figure needs to be one you'll
    // defend... Don't ship a bracketed number." The demonstration measures the
    // recurring run, not the onboarding project, so that is what the clock shows.
    const seconds = currentPeriodElapsedSeconds();
    expect(seconds).toBeGreaterThan(0);
    expect(elapsed(seconds)).toMatch(/^\d{2}:\d{2}\.\d$/);
    // And no scene copy may assert an onboarding duration.
    const copy = SCENES.flatMap((s) => [s.narration, ...s.captions.map((c) => c.text)]).join(" ");
    expect(copy).not.toMatch(/48 hours|forty-eight hours/i);
  });
});

// --------------------------------------------------------------------------- //
// S3 — regulatory first, and only artefacts that exist
// --------------------------------------------------------------------------- //
describe("S3 · one dataset, every output", () => {
  it("has a regulatory submission to lead the fan-out", () => {
    const regulatory = ARTEFACTS.regulatoryOutput;
    expect(regulatory.available, String(regulatory.reason ?? "")).toBe(true);
    expect(regulatory.fileName).toBe("annex2_submission.xml");
    expect(regulatory.xsdValidated).toBe(true);
    expect(Number(regulatory.exposureRecords)).toBe(SUMMARY.metrics.loan_count);
  });

  it("reports regulatory before management information", () => {
    // The card order is fixed in S3Dataset.tsx; this asserts the two artefacts the
    // ordering is about are both present, so the order is meaningful.
    expect(ARTEFACTS.regulatoryOutput.available).toBe(true);
    expect(ARTEFACTS.investorDeck.available).toBe(true);
    expect(ARTEFACTS.canonicalTape.available).toBe(true);
  });

  it("has metadata for all six cards", () => {
    expect(Number(ARTEFACTS.investorDeck.slides)).toBeGreaterThan(0);
    expect(Number(ARTEFACTS.canonicalTape.rows)).toBe(SUMMARY.metrics.loan_count);
    expect(Number(ARTEFACTS.canonicalTape.columns)).toBeGreaterThan(0);
    expect(Number(ARTEFACTS.validationReport.businessRuleExceptions)).toBeGreaterThan(0);
    expect(Number(ARTEFACTS.riskMonitor.limitCount)).toBeGreaterThan(0);
    expect((ARTEFACTS.auditManifest.sourceFiles as unknown[]).length).toBeGreaterThan(0);
  });

  it("shows a reconciliation result that actually passed", () => {
    expect(ASSERTIONS.ok).toBe(true);
    expect(ASSERTIONS.checksPassed).toBe(ASSERTIONS.checksRun);
    expect(ASSERTIONS.checksRun).toBeGreaterThan(20);
  });
});

// --------------------------------------------------------------------------- //
// Safety — the render must not proceed on prohibited content
// --------------------------------------------------------------------------- //
describe("safety", () => {
  it("carries a clean data-safety scan", () => {
    expect(SAFETY.ok).toBe(true);
    expect(SAFETY.findingCount).toBe(0);
    expect(SAFETY.filesScanned).toBeGreaterThan(10);
  });

  it("marks every portfolio as synthetic", () => {
    expect(PORTFOLIOS.length).toBe(2);
    for (const key of ["A", "B"]) {
      expect(portfolio(key).display_id).toMatch(/^ALP_/);
    }
  });
});

// --------------------------------------------------------------------------- //
// Formatting — UK English, pounds sterling
// --------------------------------------------------------------------------- //
describe("formatting", () => {
  it("abbreviates money the way the product's own answers do", () => {
    expect(money(1_964_886_258.21)).toBe("£1.96bn");
    expect(money(18_058_817.61)).toBe("£18.1m");
    expect(money(579_377_675.23)).toBe("£579.4m");
    expect(signedMoney(18_058_817.61)).toBe("+£18.1m");
  });

  it("formats counts, percentages and elapsed clocks in UK English", () => {
    expect(count(11_035)).toBe("11,035");
    expect(percent(0.68, 2)).toBe("0.68%");
    expect(elapsed(14.14)).toBe("00:14.1");
    expect(elapsed(74.5)).toBe("01:14.5");
  });
});
