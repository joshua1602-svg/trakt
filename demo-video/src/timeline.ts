/**
 * The storyboard, as data.
 *
 * One place controls every scene's length and order, every burned-in caption and
 * every narration line. The composition, the captions track and the generated
 * voice-over script all read this table, so they cannot drift apart.
 *
 * Frames, not seconds, because the storyboard is frame-precise: 2,700 frames at
 * 30fps, cut 420 / 480 / 780 / 660 / 360. A unit test asserts the total and the
 * per-scene boundaries.
 *
 * On-screen COPY lives here. On-screen FIGURES do not — those are read from the
 * fixtures at render time (see src/data/fixtures.ts). Nothing in this file states
 * a number the pipeline produced.
 */

export const FPS = 30;
export const TOTAL_FRAMES = 2700;

export interface Caption {
  /** Frames from the start of the SCENE. */
  at: number;
  /** Frames the caption stays up. */
  hold: number;
  /** Max two lines, and it must carry the argument with the sound off. */
  text: string;
}

export interface SceneSpec {
  id: string;
  /** Scene number as the storyboard names it. */
  number: number;
  title: string;
  frames: number;
  captions: Caption[];
  /** The voice-over paragraph for this scene (UK English). */
  narration: string;
}

export const SCENES: SceneSpec[] = [
  {
    id: "cost",
    number: 1,
    title: "The cost",
    frames: 420,
    // No caption over a display claim, and none over the opening line — that line is
    // already burned-in body copy doing the caption's job. A caption underneath either
    // one just competes with it.
    captions: [
      {
        at: 84,
        hold: 150,
        text: "Every month-end, someone rebuilds it by hand in a spreadsheet.",
      },
    ],
    narration:
      "You bought a back book. It didn't come with your data model. It arrived as a " +
      "servicer extract that looks nothing like your origination system, and every " +
      "month-end someone rebuilds it by hand in a spreadsheet. Five days of month-end, " +
      "every month, for every portfolio.",
  },
  {
    id: "onboard",
    number: 2,
    title: "Disparate cuts in, one governed portfolio out",
    frames: 480,
    // The arrivals beat (120–270) is deliberately UNcaptioned. Five artefacts from five
    // owners on three cycles is an argument that works by being looked at; a caption
    // underneath it would explain a picture that does not need explaining, and every
    // sentence spent on it is a sentence spent near the word "mapping".
    captions: [
      { at: 6, hold: 108, text: "Onboarded once, in under forty-eight hours." },
      {
        at: 276,
        hold: 144,
        text: "Everything the run produced — including what it would not guess.",
      },
    ],
    narration:
      "Trakt onboards a portfolio once, in under forty-eight hours. And it is never one " +
      "clean file: five artefacts, from five owners, on three reporting cycles. What " +
      "comes back is a receipt — every field accounted for, and the two decisions the " +
      "platform referred to a human, because it knows what it doesn't know. One " +
      "governed portfolio, loan level to sponsor level.",
  },
  {
    id: "dataset",
    number: 3,
    title: "Three portfolios, one sponsor view, every output",
    frames: 780,
    // Nothing is captioned over the sponsor beat (198–300) or the closing claim
    // (696–780): both are burned-in display copy already doing the caption's job.
    captions: [
      { at: 18, hold: 168, text: "Three portfolios. One sold, two held, all governed the same way." },
      { at: 306, hold: 138, text: "The two warehoused books, consolidated." },
      {
        at: 450,
        hold: 222,
        text: "Your regulatory submission, your investor pack, your management information.",
      },
    ],
    narration:
      "Three portfolios. One sold into a securitisation, two held — all governed the " +
      "same way. Together, two point eight one billion across fifteen thousand two " +
      "hundred and fifteen loans. No single system gives you that number today. Narrow " +
      "to the warehoused book and it's one point nine six billion across eleven thousand " +
      "and thirty-five loans — one dataset behind your regulatory submission, your " +
      "investor pack and your management information. Every output reconciles.",
  },
  {
    id: "omnichannel",
    number: 4,
    title: "Three ways in",
    frames: 660,
    captions: [
      { at: 6, hold: 120, text: "Consume it however you already work." },
      {
        at: 132,
        hold: 180,
        text: "A managed service. Copilot. Or the workspace, when you need to drill in.",
      },
      { at: 318, hold: 168, text: "Same engine, same answer, every time." },
    ],
    narration:
      "Consume it however you already work. Run it as a managed service and never " +
      "log in. Ask it from Copilot, in the tools you have. Or open the workspace " +
      "when you need to drill in. Same platform canonical, same answer, every time.",
  },
  {
    id: "close",
    number: 5,
    title: "Close",
    frames: 360,
    // The close is entirely burned-in display copy — the capability line and the ask
    // are the on-screen text, so a caption plate under them would only compete.
    captions: [],
    narration: "Trakt. A data operating system for specialist lenders.",
  },
];

/** Scene start frames, in order. */
export const sceneStarts = (): number[] => {
  let at = 0;
  return SCENES.map((scene) => {
    const start = at;
    at += scene.frames;
    return start;
  });
};

/** The absolute frame a scene starts on. */
export const startOf = (id: string): number => {
  const index = SCENES.findIndex((s) => s.id === id);
  if (index < 0) throw new Error(`Unknown scene ${id}`);
  return sceneStarts()[index];
};

export const totalFrames = (): number =>
  SCENES.reduce((total, scene) => total + scene.frames, 0);

export interface AbsoluteCaption extends Caption {
  scene: string;
  /** Absolute frame the caption appears on. */
  from: number;
}

/** Every caption, resolved to absolute frames. */
export const captions = (): AbsoluteCaption[] => {
  const starts = sceneStarts();
  return SCENES.flatMap((scene, i) =>
    scene.captions.map((c) => ({ ...c, scene: scene.id, from: starts[i] + c.at })),
  );
};

/**
 * The still frames the outbound email body uses — one per scene that carries an
 * argument on its own, taken at the beat's midpoint rather than its entrance:
 *
 *   800  S2, the receipt strip with the referred-for-review card beneath it
 *   1500 S3, the six governed outputs fanned out under the platform figure
 *   2100 S4, the same balance landed simultaneously in all three panels
 */
export const STILL_FRAMES = [800, 1500, 2100] as const;

export const timecode = (frame: number): string => {
  const total = frame / FPS;
  const mm = Math.floor(total / 60);
  const ss = Math.floor(total % 60);
  return `${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
};
