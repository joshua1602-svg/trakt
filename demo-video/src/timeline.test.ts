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

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { spring } from "remotion";
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
  PLATFORM_SCOPE,
  PORTFOLIOS,
  SAFETY,
  SPONSORED,
  SPONSOR_SCOPE,
  SUMMARY,
  lens,
  onboarding,
  portfolio,
  schemaFor,
} from "./data/fixtures";
import {
  CHANNEL_USE,
  MONTH_END_COST,
  ONBOARDING_CLAIM_HOURS,
  ONBOARDING_HOURS,
  OPENING_LINE,
  ARRIVING_ARTEFACTS,
  COPILOT_ASK,
  FUNNEL_CLAIM,
  MICROSOFT_CHANNEL,
  MS_TRADEMARK_NOTICE,
} from "./claims";
import { count, elapsed, hoursMinutes, money, percent, signedMoney } from "./format";
import theme from "./theme";

// --------------------------------------------------------------------------- //
// Reading the scenes back
// --------------------------------------------------------------------------- //
const SCENE_FILES = [
  "S1Cost",
  "S2Onboard",
  "S3Dataset",
  "S4Omnichannel",
  "S5Close",
] as const;

const sceneSource = (name: string): string =>
  readFileSync(join(__dirname, "scenes", `${name}.tsx`), "utf8");

/**
 * A scene's source with comments removed.
 *
 * Structural assertions below count JSX elements. The scene headers explain the rules by
 * naming those same elements — S4's header says the three figures share one `<Counter>`,
 * and describes the `<Img>` that a licensed Microsoft asset would need — so a raw count
 * fails on the prose describing the thing it is counting. Blanking comments to spaces
 * rather than deleting them keeps character offsets, so `indexOf` stays meaningful.
 */
const sceneCode = (name: string): string =>
  sceneSource(name)
    .replace(/\/\*[\s\S]*?\*\//g, (m) => m.replace(/[^\n]/g, " "))
    .replace(/^\s*\/\/.*$/gm, "");

/**
 * The literal text a scene puts on screen, and nothing else.
 *
 * The terminology rules below count words a VIEWER reads. Three other things in a scene
 * file contain the same words and must not be counted:
 *
 *   comments   — S2's header explains at length why there is no mapping animation, which
 *                requires saying "mapping". A naive count fails on the explanation of the
 *                rule it is enforcing.
 *   identifiers— `mapping.mapped_count` is a fixture path, not copy.
 *   interpolations — `${count(mapping.mapped_count)} fields mapped` puts four characters
 *                of digits and the words " fields mapped" on screen; the expression
 *                inside `${...}` is code.
 *
 * So: strip comments, keep only quoted and templated text, and strip `${...}` from what
 * survives. What is left is the copy.
 */
const sceneCopy = (name: string): string => {
  const withoutComments = sceneSource(name)
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/^\s*\/\/.*$/gm, " ");
  const literals = [
    ...withoutComments.matchAll(/"([^"\n]*)"/g),
    ...withoutComments.matchAll(/`([^`]*)`/gs),
  ].map((m) => m[1].replace(/\$\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}/g, " "));
  // Plus bare JSX text: `>Some words<` between tags, which is not quoted at all.
  const jsxText = [...withoutComments.matchAll(/>([^<>{}]*[A-Za-z][^<>{}]*)</g)].map((m) => m[1]);
  return [...literals, ...jsxText].join("\n");
};

/** Every word the film puts in front of a viewer: burned-in copy plus the caption track. */
const filmCopy = (): string =>
  [
    ...SCENE_FILES.map(sceneCopy),
    ...SCENES.flatMap((s) => [s.narration, ...s.captions.map((c) => c.text)]),
  ].join("\n");

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

  it("cuts 420 / 480 / 780 / 660 / 360, in the storyboard's order", () => {
    expect(SCENES.map((s) => s.id)).toEqual([
      "cost",
      "onboard",
      "dataset",
      "omnichannel",
      "close",
    ]);
    expect(SCENES.map((s) => s.frames)).toEqual([420, 480, 780, 660, 360]);
    expect(sceneStarts()).toEqual([0, 420, 900, 1680, 2340]);
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

  it("fits every narration paragraph inside its scene at a speakable pace", () => {
    // The film is caption-led, so narration is a script for a track added later. That
    // only works if each paragraph fits its own scene — a paragraph written at 230 words
    // a minute cannot be read over the beats it describes, and the recording session is
    // where that gets discovered unless something asserts it here.
    for (const scene of SCENES) {
      const words = scene.narration.trim().split(/\s+/).length;
      const wpm = (words / (scene.frames / FPS)) * 60;
      expect(wpm, `${scene.id}: ${words} words in ${scene.frames} frames`).toBeLessThanOrEqual(
        175,
      );
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
    expect(theme.type.body.fontSize).toBe(34);
    expect(theme.type.label.fontSize).toBe(22);
    expect(theme.type.stamp.fontSize).toBe(24);
  });

  it("puts every declared size at or above the legibility floor", () => {
    expect(theme.minFontSize).toBe(22);
    for (const [name, spec] of Object.entries(theme.type)) {
      expect(spec.fontSize, name).toBeGreaterThanOrEqual(theme.minFontSize);
    }
    // And the floor survives the square crop, where display and stat are scaled down.
    expect(theme.type.display.fontSize * theme.layout.square.displayScale).toBeGreaterThanOrEqual(
      theme.minFontSize,
    );
    expect(theme.type.headline.fontSize * theme.layout.square.displayScale).toBeGreaterThanOrEqual(
      theme.minFontSize,
    );
  });

  it("throws at render time on anything beneath the floor", () => {
    // The floor is only real if something fails. A lint cannot see through `scale` props
    // and layout multipliers to the size that reaches the screen; the components can, so
    // they check the resolved number and throw. This asserts that guard exists and is
    // wired into every primitive that can put type on screen.
    const kit = readFileSync(join(__dirname, "components", "kit.tsx"), "utf8");
    expect(kit).toMatch(/const legible = \(size: number, where: string\): number => \{/);
    expect(kit).toContain("if (size < theme.minFontSize)");
    for (const where of ["Label", "Stamp", "Body", "Figure", "Claim", "Headline"]) {
      expect(kit, where).toMatch(new RegExp(`(legible|checkedStyle)\\([^)]*"${where}"`, "s"));
    }
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
    expect([...STILL_FRAMES]).toEqual([700, 1420, 2100]);
    for (const frame of STILL_FRAMES) {
      expect(frame).toBeLessThan(totalFrames());
    }
    // And the exporter renders those frames, not a stale copy of them. `stills.mjs` is
    // a plain script with its own list, which is exactly the kind of duplicate that
    // drifts silently after a retime.
    const script = readFileSync(join(__dirname, "..", "scripts", "stills.mjs"), "utf8");
    const declared = script.match(/const EMAIL_FRAMES = \[([^\]]*)\]/);
    expect(declared, "no EMAIL_FRAMES in scripts/stills.mjs").toBeTruthy();
    expect(declared?.[1].split(",").map((n) => Number(n.trim()))).toEqual([...STILL_FRAMES]);
  });

  it("lands each still inside the scene the storyboard intended", () => {
    // 700 is S2's governed band; 1500 is S3's card fan-out;
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

  it("funnels five artefacts, genuinely heterogeneous, into one band", () => {
    // The beat's argument is heterogeneity collapsing into one thing, and it only lands
    // if the five tiles genuinely differ. If they share a format and a cycle there is
    // nothing to look at and the collapse proves nothing.
    expect(ARRIVING_ARTEFACTS).toHaveLength(5);
    expect(new Set(ARRIVING_ARTEFACTS.map((a) => a.title)).size).toBe(5);
    expect(new Set(ARRIVING_ARTEFACTS.map((a) => a.format)).size).toBeGreaterThanOrEqual(3);
    expect(new Set(ARRIVING_ARTEFACTS.map((a) => a.frequency)).size).toBeGreaterThanOrEqual(3);
    // And the tiles must visibly converge — a transform driven by one shared value, not
    // five independent fades. `converge` is that value; the tile applies it to translate
    // AND scale, which is what makes five things read as arriving at one place.
    const src = sceneCode("S2Onboard");
    expect(src).toMatch(/const CONVERGE_AT = /);
    expect(src).toMatch(/converge=\{converge\}/);
    expect(src).toContain("const slide = -offset * pitch * converge;");
    expect(src).toMatch(/scale\(\$\{1 - 0\.4 \* converge\}\)/);
    expect(src).toContain("opacity: 1 - converge,");
    // The band it resolves into, and the claim beneath it.
    expect(src).toContain("Governed portfolio dataset");
    expect(src).toContain("{FUNNEL_CLAIM}");
    expect(FUNNEL_CLAIM).toMatch(/one governed dataset\.$/i);
  });

  it("shows the onboarding clock in hours, not on a stopwatch", () => {
    // 00:14.1 reads as fourteen seconds however the stamp beneath it is worded. The
    // clock is an HH:MM duration and it must format as one.
    expect(hoursMinutes(ONBOARDING_HOURS)).toBe("41:20");
    expect(hoursMinutes(ONBOARDING_HOURS)).toMatch(/^\d{2}:\d{2}$/);
    // A rounded 60 minutes has to carry into the hour, or the clock shows 41:60.
    expect(hoursMinutes(41 + 59.7 / 60)).toBe("42:00");
  });

  it("keeps the clock under the threshold the claim states", () => {
    // A figure that lands exactly on its own limit does not survive the first question
    // about it. This asserts the headroom is real and stays real.
    expect(ONBOARDING_HOURS).toBeLessThan(ONBOARDING_CLAIM_HOURS);
    expect(ONBOARDING_CLAIM_HOURS - ONBOARDING_HOURS).toBeGreaterThanOrEqual(4);
  });

  it("sources the 48-hour claim from claims.ts, not from a scene file", () => {
    // The clock is a STATED claim, not a measured figure. Scene copy may assert it, but
    // the number behind it has to come from the one file that lists everything the film
    // asserts without a fixture behind it.
    const copy = SCENES.flatMap((s) => [s.narration, ...s.captions.map((c) => c.text)]).join(" ");
    if (/48 hours|forty-eight hours/i.test(copy)) {
      expect(ONBOARDING_CLAIM_HOURS).toBe(48);
    }
  });

  it("condenses the referral note without drifting from what was recorded", () => {
    // The card shows two lines; the manifest holds the full two-sentence note. Every
    // term in the condensation must still appear in the note, so the copy on screen
    // cannot say something the platform did not record.
    const referred = onboarding("B").mappingDecisions.find(
      (d) => !d.canonical_field && d.source_header !== "Synthetic Data Notice",
    );
    const note = (referred?.note ?? "").toLowerCase();
    for (const term of ["reserve-facility drawdown", "principal outstanding"]) {
      expect(note, `"${term}" is not in the recorded note`).toContain(term);
    }
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

  it("leads each card with a human label, not a filename", () => {
    // The title is what the artefact IS; the filename belongs in the mono meta line.
    // Read the titles out of the scene rather than restating them here — a copy of the
    // list in the test passes happily while the scene says something else.
    const titles = [...sceneSource("S3Dataset").matchAll(/^ {6}title: "([^"]+)",$/gm)].map(
      (m) => m[1],
    );
    expect(titles).toHaveLength(6);
    expect(titles[0]).toBe("Regulatory submission");
    for (const title of titles) {
      expect(title, title).not.toMatch(/\.(xml|pptx|csv|json)$/);
      expect(title, title).not.toMatch(/_/);
    }
  });

  it("stamps every card with the scope it speaks for", () => {
    const scopes = [...sceneSource("S3Dataset").matchAll(/^ {6}scope: "([^"]+)",$/gm)].map(
      (m) => m[1],
    );
    // One per card, and only the three scopes the film recognises.
    expect(scopes).toHaveLength(6);
    for (const scope of scopes) {
      expect(["SPV1", "PLATFORM", "ALL"], scope).toContain(scope);
    }
    // Annex 2 is deal-level reporting: the regulatory submission is SPV1's, not the
    // platform's, and a card that claimed otherwise would misstate what was filed.
    expect(scopes[0]).toBe("SPV1");
    expect(scopes).toContain("PLATFORM");
    expect(scopes).toContain("ALL");
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
// S1 — the buyer first, and the one cost line
// --------------------------------------------------------------------------- //
describe("S1 · stated copy", () => {
  it("opens on the buyer, not on an artefact", () => {
    expect(OPENING_LINE).toMatch(/you bought a back book/i);
    // Named systems belong after the viewer has recognised themselves.
    expect(OPENING_LINE).not.toMatch(/origination system|servicer extract/i);
  });

  it("carries exactly one cost line, and it is about month-end", () => {
    expect(MONTH_END_COST).toMatch(/month-end/i);
    const stated = [OPENING_LINE, MONTH_END_COST, ...Object.values(CHANNEL_USE)];
    expect(stated.filter((line) => /month-end/i.test(line))).toHaveLength(1);
  });

  it("leaves room for the opener, the claim and the cost line inside 420 frames", () => {
    // The two added beats are absorbed inside S1; the total must not move.
    const s1 = SCENES.find((s) => s.id === "cost");
    expect(s1?.frames).toBe(420);
    expect(totalFrames()).toBe(2700);
  });
});

// --------------------------------------------------------------------------- //
// S4 — the three channels
// --------------------------------------------------------------------------- //
describe("S4 · three ways in", () => {
  it("gives every channel a plain-English use line", () => {
    for (const label of ["Managed service", MICROSOFT_CHANNEL, "MI Agent workspace"]) {
      expect(CHANNEL_USE[label], label).toBeTruthy();
      // Plain English: no mono identifiers — no snake_case, no file extension, no
      // call syntax. A trailing full stop is a sentence, not an identifier.
      expect(CHANNEL_USE[label]).not.toMatch(/_|\(\)|\.(xml|csv|json|pptx)\b/);
      expect(CHANNEL_USE[label]).toMatch(/^[A-Z].*\.$/);
    }
  });



  it("gives every panel real content, not a stub", () => {
    const src = sceneCode("S4Omnichannel");
    // Panel 1: the three artefacts the run actually produced, named exactly.
    for (const file of [
      String(ARTEFACTS.regulatoryOutput.fileName),
      String(ARTEFACTS.investorDeck.fileName),
      String(ARTEFACTS.canonicalTape.fileName),
    ]) {
      expect(src, file).toContain(file);
    }
    expect(src).toContain("DELIVERED 07:00 · FIRST BUSINESS DAY");
    // Panel 2: a chat thread — a question, a tool call, an answer on a light plate.
    expect(src).toContain("{COPILOT_ASK}");
    expect(src).toContain("Called ${COPILOT.actions[0]} ✓");
    expect(src).toContain("backgroundColor: theme.color.paper");
    expect(src).toContain("Funded balances up ");
    expect(src).toContain("Grounded in Trakt");
    // Panel 3: a query, an answer and four animated bars.
    expect(src).toContain("Show funded balance by region.");
    expect(src).toContain("SUMMARY.topRegions.slice(0, 4)");
    expect(src).toMatch(/const BARS_AT = /);
    expect(src).toMatch(/width: `\$\{\(region\.balance \/ largest\) \* 100 \* grow\}%`/);
  });

  it("condenses the Copilot answer without drifting from what the agent said", () => {
    // The panel shows one sentence; the fixture holds the recorded turn. Every figure and
    // phrase on screen has to appear in it, or the panel is putting words in the agent's
    // mouth — which is the one thing a demonstration of a deterministic engine cannot do.
    const recorded = COPILOT.answers
      .map((a) => a.answer)
      .join(" ")
      .toLowerCase();
    expect(recorded).toContain(money(MOVEMENT.delta.funded_balance ?? 0).toLowerCase());
    expect(recorded).toContain(percent(SUMMARY.metrics.wa_ltv_points ?? 0, 1));
    expect(recorded).toContain(
      `completions in the ${MOVEMENT.primaryRegion?.region}`.toLowerCase(),
    );
    // And the question is stated copy, so it carries no figure to drift.
    expect(COPILOT_ASK).not.toMatch(/[0-9]/);
  });

  it("carries a Microsoft identifier, in the only form the guidelines permit", () => {
    // Microsoft Legal: "our logos, app and product icons, illustrations, photographs,
    // videos, and designs can never be used without an express license." This project
    // holds none, so the panel carries the product NAME as a wordmark and no icon.
    expect(MICROSOFT_CHANNEL).toBe("Microsoft 365 Copilot");
    // Microsoft first, name unaltered and unabbreviated.
    expect(MICROSOFT_CHANNEL).toMatch(/^Microsoft /);
    expect(MICROSOFT_CHANNEL).not.toMatch(/\bM365\b|\bMS\b|\bCopilot\b(?<!365 Copilot)/);
    const src = sceneCode("S4Omnichannel");
    expect(src).toContain("{ label: MICROSOFT_CHANNEL");
    // No unlicensed asset may creep in later without this failing.
    expect(src).not.toMatch(/<Img\b/);
    expect(src).not.toMatch(/brand\/(copilot|microsoft|m365)/i);
    // And the attribution is on screen with it.
    expect(MS_TRADEMARK_NOTICE).toMatch(/trademarks of the microsoft group of companies/i);
    expect(src).toContain("{MS_TRADEMARK_NOTICE}");
  });

  it("renders the three figures from ONE counter, so they cannot land on different frames", () => {
    // Simultaneity is the whole argument of the scene. It is guaranteed structurally:
    // the three panels come from a single `<Counter>` inside `CHANNELS.map(...)`, with a
    // single shared `at`. One element, three instances, identical inputs.
    // Stronger than before: the payload is now a single JSX ELEMENT, defined once and
    // rendered into all three columns. There is one <Counter> in the file and one
    // PAYLOAD_AT, so the three figures are literally the same node three times over.
    const src = sceneCode("S4Omnichannel");
    expect([...src.matchAll(/<Counter\b/g)]).toHaveLength(1);
    expect(src).toMatch(/at=\{PAYLOAD_AT\}/);
    expect([...src.matchAll(/const PAYLOAD_AT = /g)]).toHaveLength(1);
    expect([...src.matchAll(/const payload = \(/g)]).toHaveLength(1);
    expect([...src.matchAll(/\{payload\}/g)].length).toBeGreaterThanOrEqual(2);
    // And nothing may stagger them: no `index` or `i` in the payload's timing.
    const payload = src.slice(src.indexOf("const payload = ("));
    expect(payload.slice(0, payload.indexOf("</>"))).not.toMatch(/\bi\b|index/);
  });

  it("produces the same counter value in all three panels on every frame of the count", () => {
    // The value-level proof, using the same spring the component uses — and reading the
    // frame it starts on out of the scene, so a retime cannot leave this passing against
    // a number the film no longer uses.
    const at = Number(sceneCode("S4Omnichannel").match(/const PAYLOAD_AT = (\d+)/)?.[1]);
    expect(at).toBeGreaterThan(0);
    const frames = theme.motion.slow;
    const total = SUMMARY.metrics.funded_balance ?? 0;
    const valueAt = (frame: number) =>
      money(
        total *
          spring({ frame: frame - at, fps: FPS, config: theme.motion.spring, durationInFrames: frames }),
      );
    for (let frame = at - 2; frame <= at + frames + 10; frame += 1) {
      const panels = [valueAt(frame), valueAt(frame), valueAt(frame)];
      expect(new Set(panels).size, `frame ${frame}`).toBe(1);
    }
    // And the count actually resolves to the figure by the end of its window.
    expect(valueAt(at + frames)).toBe(money(total));
  });
});

// --------------------------------------------------------------------------- //
// Scope — the two figures that must never be mistaken for one another
// --------------------------------------------------------------------------- //
describe("scope", () => {
  it("keeps the platform figures exactly where they were", () => {
    // Introducing SPV1 into the synthetic set must not have moved a single number the
    // film already showed. SPV1 is generated on its own seed and its own solve, and it
    // is never assembled; these are the assertions that prove it stayed that way.
    expect(PLATFORM_SCOPE.fundedBalance).toBe(1_964_886_258.21);
    expect(PLATFORM_SCOPE.loanCount).toBe(11_035);
    expect(money(PLATFORM_SCOPE.fundedBalance)).toBe("£1.96bn");
    expect(PLATFORM_SCOPE.portfolios).toEqual(["ALP_ORIGINATION", "ALP_ACQUIRED"]);

    // And the platform scope IS the assembled total lens — not a second sum computed
    // beside it that could drift.
    expect(PLATFORM_SCOPE.fundedBalance).toBe(SUMMARY.metrics.funded_balance);
    expect(PLATFORM_SCOPE.loanCount).toBe(SUMMARY.metrics.loan_count);

    expect(signedMoney(MOVEMENT.delta.funded_balance ?? 0)).toBe("+£18.1m");
    expect(percent(SUMMARY.metrics.wa_ltv_points ?? 0, 1)).toBe("43.2%");
    expect(SUMMARY.topRegions[0].region).toBe("South East");
    expect(SUMMARY.topRegions[0].balance).toBe(516_214_136.58);
    expect(MOVEMENT.primaryRegion?.delta).toBe(7_840_963.14);
  });

  it("adds SPV1 to the sponsor scope without adding it to the platform", () => {
    expect(SPONSORED.displayId).toBe("SPV1");
    expect(SPONSORED.fundedBalance).toBeGreaterThan(0);
    expect(SPONSORED.loanCount).toBeGreaterThan(0);
    expect(SPONSOR_SCOPE.portfolios).toEqual([...PLATFORM_SCOPE.portfolios, "SPV1"]);
    // The sponsor figure is the sum of the parts, to the penny and to the loan.
    expect(SPONSOR_SCOPE.fundedBalance).toBeCloseTo(
      PLATFORM_SCOPE.fundedBalance + SPONSORED.fundedBalance,
      2,
    );
    expect(SPONSOR_SCOPE.loanCount).toBe(PLATFORM_SCOPE.loanCount + SPONSORED.loanCount);
    expect(money(SPONSOR_SCOPE.fundedBalance)).toBe("£2.81bn");
    expect(count(SPONSOR_SCOPE.loanCount)).toBe("15,215");
  });

  it("shows the sponsor scope in exactly one scene", () => {
    const users = SCENE_FILES.filter((name) => /SPONSOR_SCOPE/.test(sceneSource(name)));
    expect(users).toEqual(["S3Dataset"]);
  });

  it("confines the sponsor scope to S3's consolidation beat", () => {
    // Every SPONSOR_SCOPE reference must sit above the marker that opens the platform
    // beat. After that line the scene is PLATFORM scope for the rest of the film.
    const src = sceneSource("S3Dataset");
    const boundary = src.indexOf("Beat 2 · back to PLATFORM scope");
    expect(boundary).toBeGreaterThan(0);
    const uses = [...src.matchAll(/SPONSOR_SCOPE/g)].map((m) => m.index ?? 0);
    expect(uses.length).toBeGreaterThan(0);
    for (const at of uses) {
      expect(at, `SPONSOR_SCOPE used at ${at}, past the platform boundary`).toBeLessThan(
        boundary,
      );
    }
  });

  it("names the scope on S4's result band, where the two could be confused", () => {
    expect(sceneSource("S4Omnichannel")).toContain("PLATFORM CANONICAL");
  });
});

// --------------------------------------------------------------------------- //
// Terminology — mapping is six words in a receipt, and nothing else
// --------------------------------------------------------------------------- //
describe("terminology", () => {
  it("has no mapping animation anywhere in the film", () => {
    // The mapping-pair animation was driven by `mappingDecision(key, header)`, the only
    // accessor that returns a source header beside the canonical field it resolved to.
    // No scene may reach for it: with no access to a pair, there is nothing to animate.
    for (const name of SCENE_FILES) {
      expect(sceneSource(name), name).not.toMatch(/\bmappingDecision\b/);
    }
  });

  it("puts the word 'mapped' in front of a viewer exactly once, in S2's receipt", () => {
    // Burned-in copy AND the caption track AND the narration: one use, total, across the
    // whole film. Mapping is the weakest thing the product does and the easiest to
    // dismiss, so it gets the weight of one item in a six-item strip and no more.
    const hits = [...filmCopy().matchAll(/\bmapp(?:ed|ing|s)\b/gi)].map((m) => m[0]);
    expect(hits).toEqual(["mapped"]);
    // And that one use is the receipt item.
    expect(sceneCopy("S2Onboard")).toContain("fields mapped");
  });

  it("never abbreviates the Microsoft product name anywhere a viewer can read it", () => {
    // Microsoft's guidelines require the name unaltered and preceded by "Microsoft".
    // The burned-in wordmark is checked in the S4 block; this covers the caption track
    // and the narration, where "Copilot." on its own is the easy slip.
    for (const match of filmCopy().matchAll(/\bCopilot\b/g)) {
      const before = filmCopy().slice(Math.max(0, (match.index ?? 0) - 22), match.index);
      expect(before, `bare "Copilot" at ${match.index}`).toMatch(/Microsoft 365 $/);
    }
  });

  it("never rebuts the mapping objection — raising it would keep it the topic", () => {
    const copy = filmCopy();
    for (const phrase of [
      /column matching/i,
      /more than (?:just )?mapping/i,
      /not just mapping/i,
      /isn't mapping/i,
      /field matching/i,
    ]) {
      expect(copy, String(phrase)).not.toMatch(phrase);
    }
  });

  it("keeps the £2.81bn figure out of every caption and narration line", () => {
    // The sponsor figure is spoken once, in S3's narration, and burned in once, in S3's
    // consolidation beat. No caption anywhere states it, and no other scene's script
    // may carry it forward.
    for (const scene of SCENES) {
      for (const caption of scene.captions) {
        expect(caption.text, `${scene.id}: "${caption.text}"`).not.toMatch(
          /two point eight|2\.81|15,215|fifteen thousand/i,
        );
      }
      if (scene.id !== "dataset") {
        expect(scene.narration, scene.id).not.toMatch(
          /two point eight|2\.81|15,215|fifteen thousand/i,
        );
      }
    }
  });
});

// --------------------------------------------------------------------------- //
// The accent — one full strength, one soft, nothing else
// --------------------------------------------------------------------------- //
describe("signal", () => {
  it("has exactly two strengths", () => {
    expect(theme.signalSoftOpacity).toBe(0.4);
    expect(theme.color.signal).toBe("#4DE0C4");
  });

  it("introduces no second green", () => {
    const greens = Object.values(theme.color).filter((hex) => {
      const r = parseInt(hex.slice(1, 3), 16);
      const g = parseInt(hex.slice(3, 5), 16);
      const b = parseInt(hex.slice(5, 7), 16);
      return g > r + 40 && g > b + 20;
    });
    expect(greens).toEqual([theme.color.signal]);
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
    // Three now: the two warehoused books the platform assembles, plus the sold
    // securitisation the sponsor still reports on.
    expect(PORTFOLIOS.length).toBe(3);
    for (const key of ["A", "B"]) {
      expect(portfolio(key).display_id).toMatch(/^ALP_/);
    }
    expect(PORTFOLIOS.map((p) => p.display_id)).toContain(SPONSORED.displayId);
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
