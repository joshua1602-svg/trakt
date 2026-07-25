/**
 * The storyboard: one place that controls every scene's length, order and copy.
 *
 * Adjust a duration here and the composition, the captions, the voice-over script
 * and the subtitle file all move together — they are all generated from this
 * table, so they cannot drift apart.
 *
 * Target run time 2:30–3:30. The current total is asserted by a unit test.
 */

import { s } from "./design/motion";

export interface Caption {
  /** Seconds from the start of the SCENE. */
  at: number;
  /** Seconds the caption stays up. */
  hold: number;
  text: string;
  /** A secondary line, rendered smaller beneath. */
  sub?: string;
}

export interface SceneSpec {
  id: string;
  /** Scene number as the storyboard names it. */
  number: number;
  title: string;
  /** Duration in seconds. */
  seconds: number;
  /** On-screen captions, scene-relative. */
  captions: Caption[];
  /** The voice-over paragraph for this scene (one per scene, UK English). */
  narration: string;
}

export const SCENES: SceneSpec[] = [
  {
    id: "problem",
    number: 1,
    title: "The reporting problem",
    seconds: 15,
    captions: [
      {
        at: 1.0,
        hold: 5.4,
        text: "Specialist lenders receive recurring portfolio data in different formats.",
      },
      {
        at: 6.8,
        hold: 7.8,
        text:
          "Management information, investor reporting and regulatory outputs must " +
          "still reconcile to one version of the truth.",
      },
    ],
    narration:
      "Specialist lenders receive recurring portfolio data in different formats — " +
      "one extract from their own origination system, another from the servicer of " +
      "an acquired book. Management information, investor reporting and regulatory " +
      "outputs must still reconcile to one version of the truth.",
  },
  {
    id: "onboard",
    number: 2,
    title: "Onboard once",
    seconds: 24,
    captions: [
      {
        at: 0.8,
        hold: 6.4,
        text: "Trakt's Onboarding Agent establishes a governed data contract for each portfolio.",
      },
      {
        at: 15.4,
        hold: 7.2,
        text: "Different source schemas. One canonical operating model.",
        sub: "Configured once at onboarding, not repeated every reporting period.",
      },
    ],
    narration:
      "Trakt's Onboarding Agent establishes a governed data contract for each " +
      "portfolio. It profiles the incoming source, maps every field to the Trakt " +
      "canonical model, applies the client's own transformations, refers " +
      "low-confidence decisions for review, and sets the validation and reporting " +
      "rules. Different source schemas. One canonical operating model. This is " +
      "configured once, at onboarding.",
  },
  {
    id: "orchestrate",
    number: 3,
    title: "Orchestrate continuously",
    seconds: 29,
    captions: [
      {
        at: 0.8,
        hold: 5.4,
        text: "Once onboarded, every reporting cycle is processed automatically.",
      },
      {
        at: 20.0,
        hold: 7.6,
        text: "The Assembler consolidates every portfolio into one platform view.",
      },
    ],
    narration:
      "Once onboarded, every reporting cycle is processed automatically. A new " +
      "month-end extract arrives, the Orchestration Agent identifies the portfolio " +
      "and the reporting date, applies the approved contract, transforms the data " +
      "into the canonical model, and runs canonical and business-rule validation. " +
      "The Assembler then consolidates every portfolio into one platform view, and " +
      "the latest approved outputs become available through the Trakt APIs.",
  },
  {
    id: "mi-summary",
    number: 4,
    title: "Consolidated MI Agent",
    seconds: 40,
    captions: [
      {
        at: 1.0,
        hold: 5.0,
        text: "The Trakt MI Agent answers from the governed consolidated dataset.",
      },
    ],
    narration:
      "The Trakt MI Agent answers from that governed consolidated dataset. The " +
      "portfolio selector covers all portfolios or either book on its own. Ask it " +
      "to summarise the portfolio, and the answer is computed deterministically " +
      "from the same figures the dashboard shows — loan count, funded balance, " +
      "weighted-average LTV and interest rate, borrower age, the largest regional " +
      "exposures, and the data cut-off date.",
  },
  {
    id: "mi-movement",
    number: 5,
    title: "Follow-up comparison",
    seconds: 32,
    captions: [
      {
        at: 1.0,
        hold: 5.2,
        text: "The same conversation, the same governed metrics — now across periods.",
      },
    ],
    narration:
      "The same conversation continues across reporting periods. Every figure in " +
      "the answer reconciles: the movement equals current less prior, the regional " +
      "contributions sum exactly to the total, and the two portfolios account for " +
      "the whole of the change.",
  },
  {
    id: "copilot",
    number: 6,
    title: "Microsoft 365 Copilot",
    seconds: 35,
    captions: [
      {
        at: 1.0,
        hold: 5.4,
        text: "Trakt is available as an agent inside Microsoft 365 Copilot.",
      },
      {
        at: 28.4,
        hold: 6.4,
        text: "Copilot is the experience. Trakt is the execution engine.",
      },
    ],
    narration:
      "Trakt is also available as an agent inside Microsoft 365 Copilot. Copilot " +
      "does not compute portfolio figures: it calls controlled Trakt actions, and " +
      "the answers are the same governed values the MI Agent returned. It can also " +
      "retrieve the latest approved artefacts — the investor deck and the canonical " +
      "tape — as short-lived, secure download links. Copilot is the experience. " +
      "Trakt is the execution engine.",
  },
  {
    id: "outputs",
    number: 7,
    title: "Outputs",
    seconds: 23,
    captions: [
      { at: 1.0, hold: 6.0, text: "One governed dataset. Multiple controlled outputs." },
    ],
    narration:
      "One governed dataset. Multiple controlled outputs — the consolidated " +
      "canonical tape, the validation report, the MI dashboard, the investor pack, " +
      "the concentration risk monitor, and a full audit manifest with " +
      "field-level provenance and content hashes for every file.",
  },
  {
    id: "closing",
    number: 8,
    title: "Closing",
    seconds: 14,
    captions: [
      {
        at: 0.8,
        hold: 5.4,
        text: "From recurring portfolio data to governed portfolio intelligence.",
      },
    ],
    narration:
      "From recurring portfolio data to governed portfolio intelligence. Trakt " +
      "combines onboarding, orchestration, management information, investor " +
      "reporting and controlled AI interaction in one managed operating layer.",
  },
];

/** Cross-fade between consecutive scenes, in seconds. */
export const TRANSITION_SECONDS = 0.55;

/** Frames per scene, derived from the table. */
export const sceneFrames = (scene: SceneSpec): number => s(scene.seconds);

/** Start frame of each scene, accounting for the overlap of the cross-fades. */
export const sceneStarts = (): number[] => {
  const overlap = s(TRANSITION_SECONDS);
  const starts: number[] = [];
  let cursor = 0;
  SCENES.forEach((scene, i) => {
    starts.push(cursor);
    cursor += sceneFrames(scene) - (i < SCENES.length - 1 ? overlap : 0);
  });
  return starts;
};

/** Total composition length in frames. */
export const totalFrames = (): number => {
  const starts = sceneStarts();
  const last = SCENES.length - 1;
  return starts[last] + sceneFrames(SCENES[last]);
};

/** Total composition length in seconds. */
export const totalSeconds = (): number => totalFrames() / 30;

/** The teaser cut: the scenes that carry the proposition without the detail. */
export const TEASER_SCENE_IDS = ["problem", "mi-movement", "closing"] as const;
