/**
 * Tests for the storyboard and the fixtures the film renders from.
 *
 * These are the guards that stop the film making a claim the data does not
 * support, or drifting outside its brief: the run time, the scene order, the
 * caption timing, and — most importantly — that every figure the scenes read is
 * present in the fixtures and consistent across the two surfaces.
 */

import { describe, expect, it } from "vitest";
import {
  SCENES,
  TEASER_SCENE_IDS,
  TRANSITION_SECONDS,
  sceneFrames,
  sceneStarts,
  totalFrames,
  totalSeconds,
} from "./timeline";
import { FPS } from "./design/motion";
import {
  ARTEFACTS,
  ASSERTIONS,
  CLIENT,
  COPILOT,
  CURRENT_PERIOD,
  DASHBOARD,
  MOVEMENT,
  MI_TURNS,
  PERIODS,
  PORTFOLIOS,
  PRIMARY_REGION,
  PRIOR_PERIOD,
  SAFETY,
  SCHEMAS,
  SERIES,
  SUMMARY,
  contribution,
  lens,
  portfolio,
  schemaFor,
} from "./data/fixtures";
import { money, noBreakSigns, percent, signedMoney } from "./design/format";
import { SCENE_COMPONENTS } from "./Film";

// --------------------------------------------------------------------------- //
// Storyboard
// --------------------------------------------------------------------------- //
describe("storyboard", () => {
  it("runs for between 2.5 and 3.5 minutes", () => {
    const seconds = totalSeconds();
    expect(seconds).toBeGreaterThanOrEqual(150);
    expect(seconds).toBeLessThanOrEqual(210);
  });

  it("has the eight scenes of the storyboard, in order", () => {
    expect(SCENES.map((s) => s.number)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(SCENES.map((s) => s.id)).toEqual([
      "problem",
      "onboard",
      "orchestrate",
      "mi-summary",
      "mi-movement",
      "copilot",
      "outputs",
      "closing",
    ]);
  });

  it("registers a component for every scene", () => {
    for (const scene of SCENES) {
      expect(SCENE_COMPONENTS[scene.id], scene.id).toBeTypeOf("function");
    }
  });

  it("keeps every scene within its storyboard duration band", () => {
    // The brief's approximate durations, with a tolerance either side.
    const bands: Record<string, [number, number]> = {
      problem: [10, 16],
      onboard: [20, 27],
      orchestrate: [25, 32],
      "mi-summary": [35, 45],
      "mi-movement": [25, 36],
      copilot: [30, 40],
      outputs: [20, 26],
      closing: [10, 16],
    };
    for (const scene of SCENES) {
      const [lo, hi] = bands[scene.id];
      expect(scene.seconds, scene.id).toBeGreaterThanOrEqual(lo);
      expect(scene.seconds, scene.id).toBeLessThanOrEqual(hi);
    }
  });

  it("sequences the scenes with no gap and no overrun", () => {
    const starts = sceneStarts();
    expect(starts[0]).toBe(0);
    const overlap = Math.round(TRANSITION_SECONDS * FPS);
    for (let i = 1; i < SCENES.length; i += 1) {
      const previousEnd = starts[i - 1] + sceneFrames(SCENES[i - 1]);
      expect(starts[i]).toBe(previousEnd - overlap);
    }
    expect(totalFrames()).toBe(
      starts[SCENES.length - 1] + sceneFrames(SCENES[SCENES.length - 1]),
    );
  });

  it("keeps every caption inside its own scene", () => {
    for (const scene of SCENES) {
      for (const caption of scene.captions) {
        expect(caption.at, `${scene.id}: caption starts before the scene`)
          .toBeGreaterThanOrEqual(0);
        expect(
          caption.at + caption.hold,
          `${scene.id}: caption "${caption.text.slice(0, 30)}…" outlives the scene`,
        ).toBeLessThanOrEqual(scene.seconds);
      }
    }
  });

  it("never overlaps two captions in the same scene", () => {
    for (const scene of SCENES) {
      const sorted = [...scene.captions].sort((a, b) => a.at - b.at);
      for (let i = 1; i < sorted.length; i += 1) {
        expect(sorted[i].at, scene.id).toBeGreaterThanOrEqual(
          sorted[i - 1].at + sorted[i - 1].hold,
        );
      }
    }
  });

  it("holds every caption long enough to read", () => {
    // ~13 characters a second is a comfortable reading pace.
    for (const scene of SCENES) {
      for (const caption of scene.captions) {
        const needed = caption.text.length / 13;
        expect(caption.hold, `${scene.id}: "${caption.text.slice(0, 40)}…"`)
          .toBeGreaterThanOrEqual(needed * 0.85);
      }
    }
  });

  it("gives every scene narration that fits at a normal speaking pace", () => {
    for (const scene of SCENES) {
      const words = scene.narration.trim().split(/\s+/).length;
      const wpm = (words / scene.seconds) * 60;
      expect(wpm, `${scene.id}: ${Math.round(wpm)} words per minute`)
        .toBeLessThanOrEqual(175);
    }
  });

  it("builds the teaser from real scenes", () => {
    for (const id of TEASER_SCENE_IDS) {
      expect(SCENES.some((s) => s.id === id), id).toBe(true);
    }
  });

  it("uses UK English throughout the copy", () => {
    const americanisms =
      /\b(summarize|organize|analyze|behavior|color|center|licen[sc]e plate|utilize)\b/i;
    for (const scene of SCENES) {
      expect(scene.narration, scene.id).not.toMatch(americanisms);
      for (const caption of scene.captions) {
        expect(caption.text, scene.id).not.toMatch(americanisms);
      }
    }
  });

  it("makes no pricing or deployment claim", () => {
    const forbidden =
      /\b(price|pricing|per seat|per month|£\s?\d+\s?(?:pcm|per month)|customers already|live clients|deployed across)\b/i;
    for (const scene of SCENES) {
      expect(scene.narration, scene.id).not.toMatch(forbidden);
      for (const caption of scene.captions) {
        expect(caption.text, scene.id).not.toMatch(forbidden);
      }
    }
  });
});

// --------------------------------------------------------------------------- //
// Fixtures — the film must not be able to show a figure that is not there
// --------------------------------------------------------------------------- //
describe("fixtures", () => {
  it("carries the synthetic-data notice", () => {
    expect(ASSERTIONS.ok).toBe(true);
    expect(SAFETY.ok).toBe(true);
    expect(SAFETY.findingCount).toBe(0);
  });

  it("passed every reconciliation check", () => {
    expect(ASSERTIONS.checksPassed).toBe(ASSERTIONS.checksRun);
    expect(ASSERTIONS.checksRun).toBeGreaterThanOrEqual(30);
  });

  it("describes a fictional client and two portfolios", () => {
    expect(CLIENT.displayName).toContain("Alderbridge");
    expect(PORTFOLIOS).toHaveLength(2);
    expect(portfolio("A").source_portfolio_type).toBe("direct");
    expect(portfolio("B").source_portfolio_type).toBe("acquired");
  });

  it("has three reporting periods with a current and a prior", () => {
    expect(PERIODS).toHaveLength(3);
    expect(SERIES).toHaveLength(3);
    expect(CURRENT_PERIOD.reportingDate > PRIOR_PERIOD.reportingDate).toBe(true);
  });

  it("has two genuinely different source schemas", () => {
    const a = new Set(schemaFor("A").headers);
    const b = new Set(schemaFor("B").headers);
    const shared = [...a].filter((h) => b.has(h) && h !== "Synthetic Data Notice");
    expect(shared).toEqual([]);
    expect(Object.keys(SCHEMAS)).toHaveLength(2);
  });

  it("has both MI Agent turns with an answer and supporting artifacts", () => {
    expect(MI_TURNS).toHaveLength(2);
    for (const turn of MI_TURNS) {
      expect(turn.answer.length).toBeGreaterThan(80);
      expect(turn.artifacts.length).toBeGreaterThan(0);
    }
  });

  it("has both Copilot answers and the three real actions", () => {
    expect(COPILOT.answers).toHaveLength(2);
    expect(COPILOT.actions).toEqual([
      "askTraktMi",
      "getLatestInvestorDeck",
      "getLatestCanonicalTape",
    ]);
  });

  it("never carries a live download token", () => {
    const url = COPILOT.artefactAction?.downloadUrl ?? "";
    expect(url).toContain("<redacted>");
    expect(url).not.toMatch(/[?&]sig=/);
  });

  it("has the dashboard state the MI Agent scenes render", () => {
    expect(DASHBOARD.fundedSnapshot.kpis.length).toBeGreaterThan(4);
    expect(DASHBOARD.sourcePortfolios.lenses.length).toBeGreaterThanOrEqual(3);
  });

  it("has enough regions for the charts the scenes draw", () => {
    expect(SUMMARY.topRegions.length).toBeGreaterThanOrEqual(7);
    expect(MOVEMENT.regionContributions.length).toBeGreaterThanOrEqual(9);
  });
});

// --------------------------------------------------------------------------- //
// The numbers the film states
// --------------------------------------------------------------------------- //
describe("the narrative the film states", () => {
  it("reconciles the movement to current less prior", () => {
    const delta =
      (MOVEMENT.current.funded_balance ?? 0) - (MOVEMENT.prior.funded_balance ?? 0);
    expect(Math.abs(delta - (MOVEMENT.delta.funded_balance ?? 0))).toBeLessThan(0.05);
  });

  it("reconciles the two portfolio contributions to the consolidated movement", () => {
    const total = contribution("A").delta + contribution("B").delta;
    expect(Math.abs(total - (MOVEMENT.delta.funded_balance ?? 0))).toBeLessThan(0.05);
  });

  it("reconciles the regional contributions to the consolidated movement", () => {
    const total = MOVEMENT.regionContributions.reduce((sum, c) => sum + c.delta, 0);
    expect(Math.abs(total - (MOVEMENT.delta.funded_balance ?? 0))).toBeLessThan(5000);
  });

  it("names the South East as the primary contributor", () => {
    expect(MOVEMENT.primaryRegion?.region).toBe(PRIMARY_REGION);
    expect(MOVEMENT.primaryRegionIsDominant).toBe(true);
  });

  it("only says 'driven by completions' when that is evidenced", () => {
    const answer = MI_TURNS[1].answer;
    if (answer.includes("driven by completions")) {
      expect(MOVEMENT.completionsEvidenced).toBe(true);
      expect(MOVEMENT.completionsShareOfPrimaryRegion ?? 0).toBeGreaterThanOrEqual(0.5);
    }
  });

  it("reconciles the per-portfolio lenses to the consolidated total", () => {
    const perLens =
      (lens("A").summary.metrics.funded_balance ?? 0) +
      (lens("B").summary.metrics.funded_balance ?? 0);
    expect(Math.abs(perLens - (SUMMARY.metrics.funded_balance ?? 0))).toBeLessThan(0.05);
  });

  it("shows the same figures on both surfaces", () => {
    const monies = (text: string) => text.match(/£[\d,.]+(?:bn|m|k)?/g) ?? [];
    const percents = (text: string) => text.match(/[\d.]+%/g) ?? [];
    for (let i = 0; i < MI_TURNS.length; i += 1) {
      const mi = MI_TURNS[i].answer;
      const cop = COPILOT.answers[i].answer;
      expect(monies(cop), `question ${i + 1} money`).toEqual(monies(mi));
      expect(percents(cop), `question ${i + 1} percent`).toEqual(percents(mi));
    }
  });

  it("formats the headline figures the way the answers state them", () => {
    // The scene tiles and the answer text must agree character for character.
    expect(MI_TURNS[0].answer).toContain(money(SUMMARY.metrics.funded_balance));
    expect(MI_TURNS[0].answer).toContain(percent(SUMMARY.metrics.wa_ltv_points));
    expect(MI_TURNS[1].answer).toContain(
      signedMoney(MOVEMENT.delta.funded_balance).replace("+", ""),
    );
  });

  it("only lists artefacts the run actually produced", () => {
    for (const [key, artefact] of Object.entries(ARTEFACTS)) {
      if (typeof artefact !== "object" || artefact === null) continue;
      if (!("available" in artefact)) continue;
      if (!artefact.available) {
        expect(artefact.reason, `${key} is unavailable with no stated reason`).toBeTruthy();
      }
    }
    expect(ARTEFACTS.canonicalTape.available).toBe(true);
    expect(ARTEFACTS.investorDeck.available).toBe(true);
    expect(ARTEFACTS.auditManifest.available).toBe(true);
  });
});

// --------------------------------------------------------------------------- //
// Formatting
// --------------------------------------------------------------------------- //
describe("formatting", () => {
  it("formats sterling the way the product does", () => {
    expect(money(1_964_886_258.21)).toBe("£1.96bn");
    expect(money(18_058_817.61)).toBe("£18.1m");
    expect(money(516_200_000)).toBe("£516.2m");
    expect(money(178_059)).toBe("£178k");
    expect(money(null)).toBe("—");
  });

  it("uses British number grouping", () => {
    expect(money(1_234_567_890)).toBe("£1.23bn");
    expect(percent(43.156, 1)).toBe("43.2%");
  });

  it("keeps a signed money token from breaking across a line", () => {
    const joined = noBreakSigns("contributed approximately +£11.5m and");
    expect(joined).toContain("+⁠£11.5m");
    // The digits and the sign are untouched, so the figure comparison still holds.
    expect(joined.replace(/⁠/g, "")).toBe(
      "contributed approximately +£11.5m and",
    );
  });

  it("marks a negative movement with a true minus sign", () => {
    expect(signedMoney(-18_000_000)).toBe("−£18.0m");
    expect(signedMoney(18_000_000)).toBe("+£18.0m");
  });
});
